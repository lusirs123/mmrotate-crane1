"""Two fixed-budget temporary M copies; no official model changes or TEST stage."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT/'crane_project/configs/crane_midpoint_leavevideo_diag_v1.json'
VERSION = 'symeood_r4_midpoint_leavevideo_diag_v1'
L5_SHA = 'b272c50dfe6329414803d5f22c6e1e84d9c3db37b6bb2e31f1c7ed140f662c3e'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def pin_files(paths):
    # Python3.8 retains a relative __file__ when launched by relative script path.
    return {str(Path(q).resolve().relative_to(ROOT)): sha(Path(q).resolve()) for q in paths}


def status(out, stage, **extra):
    value = dict(stage=stage, pid=os.getpid(), TEST=False,
                 time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(), **extra)
    temporary = out/'stage_status.tmp'
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(out/'stage_status.json')
    print(json.dumps(value, allow_nan=False), flush=True)


def checked_config():
    spec = read(CONFIG)
    assert spec['protocol'] == VERSION and spec['physical_gpu'] == 3
    assert spec['arms'] == ['real_seq01', 'real_seq13']
    assert spec['updates_per_arm'] == 17980 and spec['checkpoint_interval'] == 4495
    assert spec['training'] == dict(lr=.001, weight_decay=0., batch_size=8, seed=1703, clip_norm=10.)
    return spec


def schedule(engine, records, omitted, updates):
    """Keep the original ordering/eligibility and batching implementation intact."""
    remaining = [r for r in records if r['sequence'] != omitted]
    excluded = [r for r in records if r['sequence'] == omitted]
    if not excluded or {r['scale'] for r in excluded} != {1., .5}:
        raise ValueError('Both held-out video scales must exist')
    batches = []
    virtual = []
    epoch = 0
    while len(batches) < updates:
        epoch += 1
        full = engine.epoch_batches(remaining, epoch)
        take = min(len(full), updates-len(batches))
        virtual.append(dict(epoch=epoch, full_steps=len(full), used_steps=take))
        batches.extend(full[:take])
    exposure = Counter()
    h = hashlib.sha256()
    for indices in batches:
        rows = [remaining[i] for i in indices]
        if (len(rows) != 8 or Counter(r['domain'] for r in rows) != dict(real=4, sim=4)
                or any(r['role'] != 'train' or not r['eligible'] or r['sequence'] == omitted for r in rows)):
            raise ValueError('TRAIN eligibility/domain/video schedule leak')
        keys = [(r['image'], r['scale']) for r in rows]
        h.update((json.dumps(keys, separators=(',', ':'))+'\n').encode())
        exposure.update(r['sequence']+'/'+str(r['scale']) for r in rows)
    return remaining, batches, dict(updates=len(batches), slots=sum(exposure.values()),
        ordered_view_sha256=h.hexdigest(), video_scale_exposure=dict(sorted(exposure.items())),
        virtual_epochs=virtual, excluded_views=len(excluded),
        excluded_eligible=sum(r['eligible'] for r in excluded),
        remaining_eligible_by_domain=dict(Counter(r['domain'] for r in remaining if r['eligible'])))


def good(metric):
    return metric['output'] and max(metric['long_edge_relative_error'], metric['short_edge_relative_error']) <= .1


def cache_digest(records):
    h = hashlib.sha256()
    for r in records:
        h.update(json.dumps([r[k] for k in ('image', 'scale', 'sequence', 'role', 'eligible')]).encode())
        for k in ('roi', 'support', 'boxes_original', 'boxes_model', 'scale_xy', 'gt_original'):
            h.update(r[k].numpy().tobytes())
    return h.hexdigest()


def signed(row):
    """Canonical log(GT/M), positive means enlargement is needed."""
    if row['midpoint'] is None:
        return None
    g = sorted(row['gt'][2:4], reverse=True)
    m = sorted(row['midpoint'][2:4], reverse=True)
    return [math.log(g[i]/m[i]) for i in range(2)]


def w1(first, second, np):
    if not first or not second:
        raise ValueError('Empty residual distribution')
    q = np.linspace(0., 1., 1001)
    a, b = np.asarray(first), np.asarray(second)
    return [float(np.abs(np.quantile(a[:, i], q)-np.quantile(b[:, i], q)).mean()) for i in range(2)]


def compare_replay(current, saved, np):
    maximum = 0.
    if len(current) != len(saved):
        raise ValueError('M20 replay frame count differs')
    for r in current:
        ref = saved[r['image'], r['scale']]
        for key in ('gt', 'b'):
            if r[key] != ref[key]:
                raise ValueError('Cache/M20 reference identity differs: '+key)
        a, b = r['midpoint'], ref['midpoint']
        if (a is None) != (b is None):
            raise ValueError('M20 replay output presence differs')
        if a is not None:
            maximum = max(maximum, max(abs(x-y) for x, y in zip(a, b)))
            if a[5] != b[5] or not np.allclose(a[:5], b[:5], atol=1e-4, rtol=1e-6):
                raise ValueError('M20 full-cache replay differs')
    return maximum


def reachability(engine, record, row):
    """Existing nominal polytope plus original native-dtype witness, CPU only."""
    from crane_project.tools import diagnose_port_geometry_midpoint_edge_feasibility_v1 as f
    from crane_project.utils import port_geometry_midpoint_edge_feasibility_v1 as geometry
    if row['midpoint'] is None:
        return None
    torch, original = engine.torch, engine.m
    b, bm, xy = [record[k] for k in ('boxes_original', 'boxes_model', 'scale_xy')]
    m = b.new_tensor([row['midpoint']])
    _, _, _, sides, rotation = original.frame(b, bm, xy)
    runtime_frame = dict(context_sides_model=sides[0].tolist(), rotation_model=rotation[0].tolist())
    frame = geometry.make_frame(row['b'], bm[0].tolist(), xy[0].tolist(), runtime_frame)
    diag = geometry.diagnose_frame(row['b'], row['midpoint'], row['gt'], bm[0].tolist(), xy[0].tolist(),
        runtime_frame=runtime_frame, source_guard_checks=f.native_source_guards(torch, original, b, m),
        verify_witness=lambda pair, goal: f.tensor_witness(torch, original, b, bm, xy, m, pair, goal, frame))
    return diag['delivered_size_reachable']


def summarize(engine, rows):
    values = [r['metrics']['midpoint'] for r in rows]
    outputs = [v for v in values if v['output']]
    n, present = len(rows), len(outputs)
    needs = [signed(r) for r in rows if r['midpoint'] is not None]
    result = dict(views=n, outputs=present, joint10=sum(good(v) for v in values),
        long_MAE=sum(v['long_edge_relative_error'] for v in outputs)/present if present else None,
        short_MAE=sum(v['short_edge_relative_error'] for v in outputs)/present if present else None,
        mean_RIoU=sum(v['riou'] for v in values)/n if n else None,
        center_hits=sum(v['center_hit'] for v in values),
        center_MAE_px=sum(v['center_error_px'] for v in outputs)/present if present else None,
        pure_angle_MAE_deg=sum(v['angle_error_deg'] for v in outputs)/present if present else None,
        fallbacks=sum(r.get('accepted') is False for r in rows))
    for i, edge in enumerate(('long', 'short')):
        field = edge+'_edge_relative_error'
        result[edge+'_needed_log_mean'] = sum(v[i] for v in needs)/present if present else None
        result[edge+'_need_counts'] = dict(accurate_le3pct=sum(v[field] <= .03 for v in outputs),
            enlarge=sum(v[field] > .03 and d[i] > 0 for v, d in zip(outputs, needs)),
            shrink=sum(v[field] > .03 and d[i] < 0 for v, d in zip(outputs, needs)))
    if rows:
        offline = [dict(domain=r['domain'], seq_id=r['sequence'].split('_', 1)[1], frame_id=r['frame_id'],
            gt_box=engine.g.ready.canonical(r['gt']),
            pred_box=engine.g.ready.canonical(r['midpoint'][:5]) if r['midpoint'] is not None else None,
            score=r['midpoint'][5] if r['midpoint'] is not None else 0., plc_rope=None, image=r['image']) for r in rows]
        # Historical mode='test' selects the evaluator's temporal formula, not the data split.
        metric = engine.old.CraneOfflineEvaluator(mode='test').evaluate_records(offline)
        result['metric_protocol_v2'] = {k: v for k, v in metric.items() if isinstance(v, (int, float))}
    return result


def paired(first, second):
    out = dict(common_outputs=0, repaired_joint10=0, damaged_joint10=0)
    for edge in ('long', 'short'):
        out[edge+'_accurate_le3pct'] = dict(n=0, damaged=0, before_MAE=None, after_MAE=None)
    sums = {e: [0., 0.] for e in ('long', 'short')}
    for a, b in zip(first, second):
        if (a['image'], a['scale'], a['b'], a['gt']) != (b['image'], b['scale'], b['b'], b['gt']):
            raise ValueError('Paired comparison identity differs')
        u, v = a['metrics']['midpoint'], b['metrics']['midpoint']
        if u['output'] != v['output']:
            raise ValueError('Temporary M changed output existence')
        out['repaired_joint10'] += not good(u) and good(v)
        out['damaged_joint10'] += good(u) and not good(v)
        if not u['output']:
            continue
        out['common_outputs'] += 1
        for e in ('long', 'short'):
            field = e+'_edge_relative_error'
            if u[field] <= .03:
                cell = out[e+'_accurate_le3pct']
                cell['n'] += 1
                cell['damaged'] += v[field] > u[field]+1e-6
                sums[e][0] += u[field]
                sums[e][1] += v[field]
    for e in sums:
        cell = out[e+'_accurate_le3pct']
        if cell['n']:
            cell['before_MAE'], cell['after_MAE'] = [x/cell['n'] for x in sums[e]]
    return out


def grouped(engine, reference, candidate, omitted):
    result = {}
    for split, predicate in [('train_s1', lambda r: r['sequence'] not in engine.g.ready.VAL_COUNTS and r['scale'] == 1.),
                             ('train_s05', lambda r: r['sequence'] not in engine.g.ready.VAL_COUNTS and r['scale'] == .5),
                             ('val', lambda r: r['sequence'] in engine.g.ready.VAL_COUNTS)]:
        indices = [i for i, r in enumerate(reference) if predicate(r)]
        groups = dict(overall=indices)
        groups.update({d: [i for i in indices if reference[i]['domain'] == d] for d in ('real', 'sim')})
        groups.update({s: [i for i in indices if reference[i]['sequence'] == s]
                       for s in sorted({reference[i]['sequence'] for i in indices})})
        if split != 'val':
            groups['remaining_TRAIN'] = [i for i in indices if reference[i]['sequence'] != omitted]
        result[split] = {}
        for name, ii in groups.items():
            if not ii:
                continue
            a, b = [reference[i] for i in ii], [candidate[i] for i in ii]
            base = [dict(r, midpoint=r['b'], metrics=dict(midpoint=r['metrics']['b']), accepted=None) for r in a]
            result[split][name] = dict(R4=summarize(engine, base), M20=summarize(engine, a),
                temporary_M=summarize(engine, b), versus_M20=paired(a, b), versus_R4=paired(base, b))
    return result


def screen(engine, records, reference, candidate, omitted, spec):
    indices = [i for i, r in enumerate(reference) if r['sequence'] == omitted and r['midpoint'] is not None]
    native = [i for i in indices if reference[i]['scale'] == 1.]
    target = [signed(r) for r in reference if r['sequence'] == spec['reference_val_video'][omitted] and r['midpoint'] is not None]
    before = w1([signed(reference[i]) for i in native], target, engine.np)
    after = w1([signed(candidate[i]) for i in native], target, engine.np)
    newly_bad = [i for i in indices if good(reference[i]['metrics']['midpoint']) and not good(candidate[i]['metrics']['midpoint'])]
    reach = [reachability(engine, records[i], candidate[i]) for i in newly_bad]
    all_bad = [i for i in indices if not good(candidate[i]['metrics']['midpoint'])]
    all_reach = [reachability(engine, records[i], candidate[i]) for i in all_bad]
    cfg = spec['screen']
    tol = cfg['signed_residual_W1_tolerance']
    checks = dict(native_each_marginal_not_worse=all(b <= a+tol for a, b in zip(before, after)),
        native_mean_strictly_closer=sum(after)/2 < sum(before)/2-tol,
        enough_new_bad_views=len(newly_bad) >= cfg['minimum_new_bad_views'],
        newly_bad_size10_reachable=bool(reach) and sum(v is True for v in reach)/len(reach) >= cfg['minimum_new_bad_reachable_fraction'])
    return dict(reference_VAL_video=spec['reference_val_video'][omitted],
        M20_native_residual_W1=before, temporary_native_residual_W1=after,
        newly_bad_views=len(newly_bad), newly_bad_verified_reachable=sum(v is True for v in reach),
        newly_bad_unresolved=sum(v is None for v in reach),
        all_bad_views=len(all_bad), all_bad_verified_reachable=sum(v is True for v in all_reach),
        all_bad_unresolved=sum(v is None for v in all_reach), checks=checks,
        screen_passed=all(checks.values()), statistical_significance=False,
        interpretation='Descriptive two-video M-layer diagnostic. Same-family geometry/video differences remain; not a causal estimate or new LVSR performance.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--repair-startup', action='store_true',
        help='Only preserve and restart the known zero-update relative-path startup failure')
    args = parser.parse_args()
    spec = checked_config()
    if not args.run:
        print(json.dumps(spec, indent=2))
        return
    out = ROOT/spec['output']
    if args.repair_startup:
        stopped = read(out/'stop.json')
        names = {q.name for q in out.iterdir()}
        if (stopped['status'] != 'STOPPED_NOT_COMPLETE'
                or 'does not start with' not in stopped['error']
                or names != {'stop.json', 'stage_status.json', 'startup_001_runner.py'}):
            raise ValueError('Repair permitted only for the recorded pre-cache zero-update path error')
        for name in ('stop.json', 'stage_status.json'):
            target = out/('startup_001_'+name)
            if target.exists():
                raise ValueError('Do not overwrite startup failure evidence')
            (out/name).rename(target)
    else:
        out.mkdir(exist_ok=False)
    started = time.monotonic()
    status(out, 'starting')
    def timeout(*unused):
        raise RuntimeError('Fixed10800-second total budget exceeded; no automatic retry')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, spec['wall_seconds'])
    try:
        run(out, spec, started)
    except BaseException as exc:
        save(out/'stop.json', dict(status='STOPPED_NOT_COMPLETE', error=repr(exc),
            elapsed_seconds=time.monotonic()-started, TEST=False, automatic_retry=False))
        status(out, 'stopped', error=repr(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def run(out, spec, started):
    from crane_project.tools import train_symeood_r4_midpoint_v1 as adapted
    from crane_project.utils import symeood_r4_midpoint_protocol_v1 as p
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as lp
    resource = adapted.resource_check(spec['physical_gpu'], before=True)
    actual_uuid = subprocess.check_output(['nvidia-smi', '-i', str(spec['physical_gpu']),
        '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if actual_uuid != spec['gpu_uuid']:
        raise ValueError('Physical GPU UUID differs')
    engine = adapted.make_engine(spec['physical_gpu'])
    runtime = engine.runtime(0)
    torch = engine.torch
    torch.set_num_threads(1)
    for logger in ('crane_project.tools.eval_crane_offline', 'crane_offline_evaluator'):
        logging.getLogger(logger).disabled = True
    identity = p.checked_sources()
    lidentity = lp.checked_sources()
    if lidentity['midpoint_training_identity'] != identity:
        raise ValueError('M/LVSR parent identity differs')
    parent = lp.midpoint_directory()
    cp = parent/'train/head_epoch_20.pth'
    l5_candidates = [lp.paths()['train']/'head_epoch_05.pth',
        lp.BASE/'readbacks/lvsr_completed/lvsr/attempt_001/train/head_epoch_05.pth']
    l5 = next((q for q in l5_candidates if q.exists()), None)
    if l5 is None:
        raise ValueError('Original L5 checkpoint not found; do not guess another version')
    prior = parent/'train/completion.json'
    reference_file = ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/attempt_001/L5_all.rows.jsonl.gz'
    pinned = [ROOT/p.R4_CHECKPOINT, cp, l5,
        parent/'roi_cache/cache_manifest.json', parent/'train/selection.json', prior,
        parent/'verification/completion.json', reference_file, CONFIG, Path(__file__),
        ROOT/'tests/test_midpoint_leavevideo_diag_v1.py',
        ROOT/'crane_project/utils/port_geometry_midpoint_edge_feasibility_v1.py',
        ROOT/'crane_project/tools/diagnose_port_geometry_midpoint_edge_feasibility_v1.py']
    pins = pin_files(pinned)
    if pins[str(cp.relative_to(ROOT))] != lp.MIDPOINT_SHA:
        raise ValueError('Official M20 SHA differs')
    if pins[str((ROOT/p.R4_CHECKPOINT).relative_to(ROOT))] != p.R4_SHA:
        raise ValueError('Official R4 SHA differs')
    if pins[str(l5.relative_to(ROOT))] != L5_SHA:
        raise ValueError('Official L5 SHA differs')
    train, val, cache = engine.load_cache(parent/'roi_cache', identity)
    if cache['runtime'] != runtime:
        raise ValueError('Historical M/cache runtime differs')
    cache = dict(cache, manifest_sha256=sha(parent/'roi_cache/cache_manifest.json'))
    p.verification_binding(read(parent/'verification/completion.json'), identity, cache['manifest_sha256'])
    records = train+val
    digest = cache_digest(records)
    original_steps = len(engine.epoch_batches(train, 1))
    if original_steps*20 != spec['updates_per_arm']:
        raise ValueError('Budget is not original M20 optimizer count')
    for key, value in spec['training'].items():
        if engine.SETTINGS[key] != value:
            raise ValueError('Original M training setting differs: '+key)
    schedules = {s: schedule(engine, train, s, spec['updates_per_arm']) for s in spec['arms']}
    save(out/'protocol.json', dict(spec=spec, source_pins=pins, identity=identity,
        original_settings=engine.SETTINGS, original_steps_per_epoch=original_steps,
        original_M20_updates=original_steps*20, runtime=runtime, resource_before=resource,
        schedules={s: value[2] for s, value in schedules.items()}, TEST=False,
        official_updates=dict(R4=0, M20=0, L5=0)))
    payload = torch.load(str(cp), map_location='cpu')
    if (payload['identity'] != identity or payload['epoch'] != 20 or payload['updates'] != spec['updates_per_arm']
            or payload['protocol'] != p.VERSION):
        raise ValueError('Original M20 checkpoint metadata differs')
    engine.seed_all()
    reference_head = engine.new_head().to('cuda:0')
    reference_head.load_state_dict(payload['head_state'], strict=True)
    if engine.g.state_digest(reference_head) != payload['head_digest']:
        raise ValueError('Original M20 state digest differs')
    with gzip.open(reference_file, 'rt') as f:
        saved = {(r['image'], r['scale']): r for r in map(json.loads, f)}
    reference = engine.evaluate(reference_head, records, 'cuda:0')
    replay_diff = compare_replay(reference, saved, engine.np)
    del reference_head, saved, payload
    torch.cuda.empty_cache()
    status(out, 'engineering', replay_max_absolute_difference=replay_diff)
    smoke = engine.smoke(train, val, cache, identity, 'cuda:0', out)
    engine.seed_all()
    initial = engine.new_head()
    initial_state = engine.g.state_digest(initial)
    if initial_state != read(prior)['initial_state']:
        raise ValueError('Fresh seed1703 initialization differs from original M training')
    del initial
    save(out/'engineering.json', dict(smoke=smoke, reference_M20_full_cache_views=len(reference),
        reference_replay_max_absolute_difference=replay_diff, original_initial_state=initial_state,
        schedule_exclusion_and_budget_passed=True, original_zero_update_verification_bound=True,
        detector_loaded=False, reference_weights_unchanged=True, TEST=False))
    all_results = {}
    for omitted in spec['arms']:
        arm_started = time.monotonic()
        remaining, batches, planned = schedules[omitted]
        engine.seed_all()
        head = engine.new_head().to('cuda:0')
        optimizer = engine.optimizer_for(head)
        if engine.g.state_digest(head) != initial_state:
            raise ValueError('Arm fresh initialization differs')
        checkpoints = []
        exposure = Counter()
        ordered = hashlib.sha256()
        interval_losses = []
        for step, indices in enumerate(batches, 1):
            metric = engine.update(head, optimizer, remaining, indices, 'cuda:0')
            rr = [remaining[i] for i in indices]
            exposure.update(r['sequence']+'/'+str(r['scale']) for r in rr)
            ordered.update((json.dumps([(r['image'], r['scale']) for r in rr], separators=(',', ':'))+'\n').encode())
            interval_losses.append(metric['loss'])
            if step % 250 == 0 or step == len(batches):
                resource = engine.resource_check(0)
                status(out, 'training', omitted=omitted, updates=step, target=len(batches),
                    elapsed_seconds=time.monotonic()-started, loss_mean=sum(interval_losses)/len(interval_losses),
                    resource=resource)
                interval_losses = []
            if step % spec['checkpoint_interval'] == 0:
                checkpoint = engine.save_checkpoint(out/(omitted+'_step_%05d.pth'%step),
                    head, optimizer, 0, step, identity,
                    dict(engine.checkpoint_context(cache), diagnostic_protocol=VERSION,
                         omitted_video=omitted, official_model=False, schedule=planned))
                checkpoints.append(checkpoint)
        if (ordered.hexdigest() != planned['ordered_view_sha256'] or dict(sorted(exposure.items())) != planned['video_scale_exposure']
                or sum(exposure.values()) != 8*spec['updates_per_arm']):
            raise ValueError('Actual schedule/exposure differs from predeclared schedule')
        status(out, 'terminal_evaluation', omitted=omitted, updates=len(batches))
        candidate = engine.evaluate(head, records, 'cuda:0')
        # Terminal snapshot replays exactly; no second candidate or checkpoint choice.
        reloaded = torch.load(str(out/checkpoints[-1]['path']), map_location='cpu')
        if not engine.tree_equal(head.state_dict(), reloaded['head_state']):
            raise ValueError('Terminal checkpoint/head mismatch')
        with gzip.open(str(out/(omitted+'_terminal.rows.jsonl.gz')), 'xt') as f:
            for row in candidate:
                f.write(json.dumps(row, allow_nan=False)+'\n')
        result = dict(omitted=omitted, updates=len(batches), checkpoints=checkpoints,
            actual_ordered_view_sha256=ordered.hexdigest(), actual_video_scale_exposure=dict(sorted(exposure.items())),
            comparison=grouped(engine, reference, candidate, omitted),
            diagnostic_screen=screen(engine, records, reference, candidate, omitted, spec),
            elapsed_seconds=time.monotonic()-arm_started, TEST=False, selection='fixed_terminal17980',
            official_adoption=False)
        save(out/(omitted+'_summary.json'), result)
        all_results[omitted] = result
        del head, optimizer, candidate, reloaded
        torch.cuda.empty_cache()
        status(out, 'arm_complete', omitted=omitted, screen_passed=result['diagnostic_screen']['screen_passed'])
    if cache_digest(records) != digest or p.checked_sources() != identity or lp.checked_sources() != lidentity:
        raise ValueError('Frozen original cache/source closure changed')
    for name, value in pins.items():
        if sha(ROOT/name) != value:
            raise ValueError('Pinned input changed during run: '+name)
    passed = all(v['diagnostic_screen']['screen_passed'] for v in all_results.values())
    save(out/'completion.json', dict(status='TWO_FIXED_BUDGET_M_DIAGNOSTICS_COMPLETE',
        formal_diagnostic_updates=2*spec['updates_per_arm'], discarded_engineering_updates=smoke['actual_optimizer_steps'],
        elapsed_seconds=time.monotonic()-started, all_pins_and_cache_unchanged=True,
        both_diagnostic_screens_passed=passed, automatic_LVSR_training=False, automatic_adoption=False,
        recommendation='Review a concrete LVSR input-generation proposal before any next training.' if passed else
            'Insufficient evidence for this input-adaptation route; stop this direction and retain L5.',
        TEST=False, official_updates=dict(R4=0, M20=0, L5=0), resource_after=engine.resource_check(0)))
    core = ['protocol.json', 'engineering.json', 'completion.json']+[s+'_summary.json' for s in spec['arms']]
    save(out/'core_manifest.json', {name: sha(out/name) for name in core})
    status(out, 'complete', both_diagnostic_screens_passed=passed, elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    main()
