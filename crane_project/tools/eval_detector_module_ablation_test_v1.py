#!/usr/bin/env python3
"""Four VAL-selected module arms on fixed TEST, two independent GPU lanes.

Reuses the original native capture, transforms and metrics. No fitting,
selection, full TRAIN access, cache generation, or depth evaluation.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_detector_module_ablation_v1 as training

CONTRACT = ROOT/'crane_project/data_contracts/detector_module_ablation_test_v1.json'


def checked():
    lock = training.read(CONTRACT)
    if lock['protocol'] != 'detector_module_ablation_test_v1':
        raise ValueError('Unknown fixed TEST protocol')
    if training.sha(training.CONTRACT) != lock['training_contract_sha256']:
        raise ValueError('Original training recipe changed')
    training.checked_contract()
    for name, digest in {**lock['sources'], **lock['artifacts']}.items():
        if training.sha(ROOT/name) != digest:
            raise ValueError('Frozen TEST source/artifact differs: '+name)
    for arm, spec in lock['heads'].items():
        selection = training.read(ROOT/spec['selection'])
        if selection['selected_checkpoint'] != spec['selected_checkpoint'] or selection['identity'] != spec['identity']:
            raise ValueError('Original VAL selection changed: '+arm)
    return lock


def validate_payload(payload, spec):
    cp = spec['selected_checkpoint']
    if (payload['protocol'] != 'detector_module_ablation_v1' or
            payload['epoch'] != cp['epoch'] or payload['updates'] != cp['updates'] or
            payload['identity'] != spec['identity'] or payload['parent_checkpoint'] is not None):
        raise ValueError('Requires exact fresh-trained VAL-selected head')


def saved_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def compare_boxes(actual, expected, arms, np, score_atol=0., score_rtol=0.):
    maximum = {}
    for arm in arms:
        a, b = actual[arm], expected[arm]
        if (a is None) != (b is None):
            raise ValueError('Presence mismatch: '+arm)
        if a is not None:
            if (not np.isclose(a[5], b[5], atol=score_atol, rtol=score_rtol) or
                    not np.allclose(a[:5], b[:5], atol=1e-4, rtol=1e-6)):
                raise ValueError('Frozen prediction mismatch: '+json.dumps(dict(
                    arm=arm, actual=a, expected=b,
                    difference=[x-y for x, y in zip(a, b)])))
            maximum[arm] = max(abs(x-y) for x, y in zip(a[:5], b[:5]))
        else:
            maximum[arm] = 0.
    return maximum


def cached_VAL_gate(lock, family, heads, f, torch):
    """Reuse a completed gate only for the identical heads and inference sources."""
    reuse = lock['groups'][family].get('reuse_cached_VAL_gate')
    if not reuse:
        return replay_cached_VAL(lock, family, heads, f, torch)
    for name, digest in reuse['files'].items():
        if training.sha(ROOT/name) != digest:
            raise ValueError('Completed VAL gate identity changed: '+name)
    old = training.read(ROOT/reuse['frozen_contract'])
    if (old['heads'] != lock['heads'] or old['training_contract_sha256'] != lock['training_contract_sha256'] or
            old['artifacts'] != lock['artifacts']):
        raise ValueError('Completed VAL gate model/artifacts changed')
    allowed = {str(Path(__file__).relative_to(ROOT)), 'tests/test_detector_module_ablation_test_v1.py'}
    if {k: v for k, v in old['sources'].items() if k not in allowed} != {
            k: v for k, v in lock['sources'].items() if k not in allowed}:
        raise ValueError('Completed VAL gate inference implementation changed')
    report = training.read(ROOT/reuse['receipt'])
    if report.get('passed') is not True or report.get('frames') != 887:
        raise ValueError('Requires completed all887 cached VAL gate')
    return dict(report, reused_from=reuse['receipt'], detector_forwards_this_attempt=0,
        cached_head_replay_frames_this_attempt=0)


def references(lock, family):
    if family == 'eood':
        single = {r['image']: r for r in saved_rows(ROOT/lock['val_rows']['eood_l'])}
        combined = {r['image']: r for r in saved_rows(ROOT/lock['val_rows']['eood_ml'])}
        if single.keys() != combined.keys():
            raise ValueError('E VAL identities differ')
        result = {}
        for name, a in single.items():
            b = combined[name]
            if a['b'] != b['b'] or a['gt'] != b['gt']:
                raise ValueError('E frozen reference differs')
            result[name] = dict(a, eood_m=b['midpoint'], eood_l=a['edge_residual'], eood_ml=b['edge_residual'])
    else:
        result = {r['image']: dict(r, **{'110_l': r['edge_residual']})
            for r in saved_rows(ROOT/lock['val_rows']['110_l'])}
    if len(result) != 887:
        raise ValueError('All887 VAL references required')
    return result


def load_heads(lock, family, f, torch):
    from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
    heads = {}
    for arm in lock['groups'][family]['arms']:
        spec = lock['heads'][arm]
        path = ROOT/spec['checkpoint']
        if training.sha(path) != spec['selected_checkpoint']['sha256']:
            raise ValueError('Selected head weight SHA differs: '+arm)
        payload = torch.load(str(path), map_location='cpu')
        validate_payload(payload, spec)
        h = f.new_head() if spec['kind'] == 'midpoint' else PairSpecificSizeHead()
        h.load_state_dict(payload['head_state'], strict=True)
        heads[arm] = h.cuda().eval().requires_grad_(False)
    return heads


def forward_heads(tensors, heads, family):
    """Online head boundary: detached visual tensors and reference boxes only."""
    x = [tensors[k].cuda() for k in ('roi', 'support', 'boxes_original', 'boxes_model', 'scale_xy')]
    b = x[2]
    values = {'b': b}
    if family == 'eood':
        m = heads['eood_m'](*x)['boxes_original']
        values.update(eood_m=m, eood_l=heads['eood_l'](*x, b)['boxes_original'],
            eood_ml=heads['eood_ml'](*x, m)['boxes_original'])
    else:
        values['110_l'] = heads['110_l'](*x, b)['boxes_original']
    result = {}
    for arm, value in values.items():
        if len(value) != len(b) or not value[:, 5].equal(b[:, 5]):
            raise ValueError('Head changed score/output count: '+arm)
        if arm not in ('b', 'eood_m'):
            ref = values['eood_m'] if arm == 'eood_ml' else b
            if not value[:, [0, 1, 4, 5]].equal(ref[:, [0, 1, 4, 5]]):
                raise ValueError('Size head changed reference center/angle/score')
        result[arm] = value[0].detach().cpu().tolist() if len(value) else None
    return result


def replay_cached_VAL(lock, family, heads, f, torch):
    spec = lock['heads'][lock['groups'][family]['arms'][0]]
    folder = ROOT/lock['groups'][family]['cache']
    manifest = training.read(folder/'cache_manifest.json')
    if training.sha(folder/'cache_manifest.json') != spec['identity']['cache_manifest_sha256']:
        raise ValueError('VAL cache parent changed')
    path = folder/'val_s1.pt'
    if training.sha(path) != spec['identity']['shards']['val_s1.pt']:
        raise ValueError('VAL cache shard changed')
    payload = torch.load(str(path), map_location='cpu')
    if payload['detector_state'] != spec['identity']['detector_state'] or payload['identity'] != manifest['identity']:
        raise ValueError('Cached detector identity differs')
    rows, refs = payload['records'], references(lock, family)
    if len(rows) != 887 or {r['image'] for r in rows} != set(refs):
        raise ValueError('Full VAL frame set differs')
    arms = ['b']+lock['groups'][family]['arms']
    maxima = {a: 0. for a in arms}
    with torch.no_grad():
        for row in rows:
            ref = refs[row['image']]
            if row['role'] != 'val' or row['scale'] != 1. or row['gt_original'][0].tolist() != ref['gt']:
                raise ValueError('Cached VAL metadata/GT differs')
            pred = forward_heads(row, heads, family)
            errors = compare_boxes(pred, ref, arms, f.np)
            for a, value in errors.items():
                maxima[a] = max(maxima[a], value)
    return dict(passed=True, frames=887, maxima=maxima, detector_forwards=0, head_updates=0)


def backup_heads(lock, family):
    result = []
    for arm in lock['groups'][family]['arms']:
        spec = lock['heads'][arm]
        source = ROOT/spec['checkpoint']
        copies = []
        for raw in lock['backup_directories']:
            folder = Path(raw)
            if not folder.is_absolute():
                folder = ROOT/folder
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/(arm+'_'+source.name)
            if not path.exists():
                with source.open('rb') as src, path.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
            if training.sha(path) != spec['selected_checkpoint']['sha256']:
                raise ValueError('Backup differs: '+arm)
            copies.append(dict(path=str(path), device=path.stat().st_dev, sha256=training.sha(path)))
        result.append(dict(arm=arm, copies=copies, independent_filesystems=len({p['device'] for p in copies}) == 2))
    return result


def summarize(predictions, family):
    from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
    result, riou_cache = {}, {}
    arms = ('eood_l', 'eood_ml') if family == 'eood' else ('110_l',)
    for arm in arms:
        ref = 'eood_m' if arm == 'eood_ml' else 'b'
        adapted = [dict(r, midpoint=r[ref], edge_residual=r[arm]) for r in predictions]
        summary = metrics.summary(adapted, riou_cache)
        for group, values in summary.items():
            target = result.setdefault(group, {})
            target.setdefault('eood' if family == 'eood' else '110', values['b'])
            if ref == 'eood_m':
                target['eood_m'] = values['midpoint']
            target[arm] = values['edge_residual']
            target[arm+'_paired_vs_reference'] = dict(reference=ref, **values['paired'])
    return dict(split='test', metrics=result, AP='task Top1 VOC07 eleven-point',
        TEST_used_for_selection=False, geometry='native float32 boxes without TXT rounding')


def run(family):
    lock = checked()
    group = lock['groups'][family]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(group['physical_gpu']) or os.environ.get('WORLD_SIZE', '1') != '1':
        raise ValueError('Use the fixed physical GPU and single process')
    out = ROOT/group.get('output_directory', lock['output_directory']+'/'+family)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    def status(stage, **extra):
        tmp = out/'status.tmp'
        tmp.write_text(json.dumps(dict(stage=stage, pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), **extra), indent=2))
        tmp.replace(out/'status.json')
    try:
        f, torch = training.runtime_imports()
        torch.set_num_threads(1)
        runtime = f.runtime(0)
        if runtime != training.read(training.CONTRACT)['runtime_reference']:
            raise ValueError('Frozen environment differs')
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes < lock['resources']['minimum_free_mib']*2**20:
            raise ValueError('Insufficient free GPU memory for fixed TEST')
        torch.cuda.set_per_process_memory_fraction(lock['resources']['allocator_cap_mib']*2**20/total_bytes, 0)
        f.seed_all()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        heads = load_heads(lock, family, f, torch)
        states = lambda: {a: f.g.state_digest(h) for a, h in heads.items()}
        before_heads = states()
        training.write(out/'frozen_contract.json', lock)
        status('full_cached_VAL_head_replay')
        replay = cached_VAL_gate(lock, family, heads, f, torch)
        training.write(out/'full_cached_VAL_gate.json', replay)
        cfg, detector = training.build_frontend(training.read(training.CONTRACT), family, f, torch)
        before_detector = f.g.state_digest(detector)
        if before_detector != lock['heads'][group['arms'][0]]['identity']['detector_state']:
            raise ValueError('Strictly loaded detector differs from cached state')
        from mmdet.datasets.pipelines import Compose
        from PIL import Image
        from crane_project.tools.eval_port_size_proposal_downstream_v1 import native_view
        pipeline = Compose(deepcopy(cfg.data.val.pipeline))
        if cfg.data.test.pipeline != cfg.data.val.pipeline:
            raise ValueError('TEST/VAL transforms differ')
        refs = references(lock, family)
        manifest = training.read(f.g.ready.DATA/'manifest.json')
        indexed = {r['id']: r for r in manifest['records'] if r['split'] == 'val'}
        arms = ['b']+group['arms']
        status('eight_online_VAL_views')
        with torch.no_grad():
            maxima = {a: 0. for a in arms}
            for name in lock['val_names']:
                src = indexed[name]
                image_path = f.g.ready.DATA/src['images']['path']
                ann_path = f.g.ready.DATA/src['annfiles']['path']
                if training.sha(image_path) != src['images']['sha256'] or training.sha(ann_path) != src['annfiles']['sha256']:
                    raise ValueError('Online VAL image/annotation changed')
                with Image.open(image_path) as im:
                    size = list(im.size)
                source = dict(image=name, image_path=str(image_path), image_size=size)
                image, metas = native_view(pipeline, source, f.old)
                pred = forward_heads(f.capture_cache(detector, image, metas), heads, family)
                errors = compare_boxes(pred, refs[name], arms, f.np)
                for a, value in errors.items():
                    maxima[a] = max(maxima[a], value)
        training.write(out/'online_VAL_gate.json', dict(passed=True, frames=8, maxima=maxima))
        training.write(out/'selected_head_backups.json', backup_heads(lock, family))
        # TEST data are first opened only after both gates and backup succeed.
        sources, test_identity = f.old.fixed_test_inputs()
        saved_baselines = {r['image']: r for r in saved_rows(ROOT/lock['existing_TEST_rows'])}
        if len(sources) != 1440 or set(saved_baselines) != {s['image'] for s in sources}:
            raise ValueError('Fixed TEST frame identity differs')
        predictions, baseline_max = [], 0.
        baseline_score_max, baseline_score_nonidentical = 0., 0
        status('TEST', done=0, total=1440)
        with torch.no_grad(), (out/'test_rows.jsonl').open('x') as stream:
            for i, source in enumerate(sources):
                source = dict(source, image_path=str(f.g.ready.DATA/'test/images'/(source['image']+'.jpg')))
                image, metas = native_view(pipeline, source, f.old)
                pred = forward_heads(f.capture_cache(detector, image, metas), heads, family)
                baseline = saved_baselines[source['image']]['eood_aug' if family == 'eood' else 'b']
                tolerance = lock.get('historical_TEST_score_tolerance', {})
                try:
                    errors = compare_boxes({'b': pred['b']}, {'b': baseline}, ['b'], f.np,
                        score_atol=tolerance.get('atol', 0.), score_rtol=tolerance.get('rtol', 0.))
                except ValueError as exc:
                    raise ValueError(source['image']+': '+str(exc)) from exc
                if pred['b'] is not None:
                    score_difference = abs(pred['b'][5]-baseline[5])
                    baseline_score_max = max(baseline_score_max, score_difference)
                    baseline_score_nonidentical += int(score_difference != 0.)
                baseline_max = max(baseline_max, errors['b'])
                pred.update(**{k: source[k] for k in ('image', 'sequence', 'domain', 'frame_id')},
                    scale=1., split='test', shard='test_s1', gt=f.np.asarray(source['gt'], dtype=f.np.float32).tolist())
                predictions.append(pred)
                stream.write(json.dumps(pred, allow_nan=False)+'\n')
                if (i+1) % 100 == 0 or i+1 == 1440:
                    stream.flush()
                    status('TEST', done=i+1, total=1440)
                    print(json.dumps(dict(family=family, TEST_done=i+1, total=1440)), flush=True)
        status('summarizing_saved_TEST')
        training.write(out/'test_summary.json', summarize(predictions, family))
        if states() != before_heads or f.g.state_digest(detector) != before_detector:
            raise ValueError('Frozen state changed')
        checked()
        report = dict(status='FIXED_VAL_SELECTED_TEST_COMPLETE', family=family, arms=group['arms'], frames=1440,
            physical_gpu=group['physical_gpu'], runtime=runtime, training_updates=0, selection=False, depth=False,
            full_cached_VAL_frames=887, online_VAL_frames=8, TEST_frames=1440,
            detector_feature_extractions=1448, native_head_calls=4344,
            baseline_maximum_box_difference=baseline_max, data_identity=test_identity,
            baseline_maximum_score_difference=baseline_score_max,
            baseline_nonidentical_score_frames=baseline_score_nonidentical,
            historical_TEST_score_tolerance=lock.get('historical_TEST_score_tolerance', {}),
            before_heads=before_heads, after_heads=states(), frozen_detector_state=before_detector,
            TEST_repeatedly_exposed=True, elapsed_seconds=time.monotonic()-start, contract_sha256=training.sha(CONTRACT))
        training.write(out/'completion.json', report)
        status('complete', frames=1440, updates=0)
        training.write(out/'artifacts.json', dict(files={p.name: training.sha(p) for p in out.iterdir() if p.is_file()}))
    except BaseException as exc:
        training.write(out/'failure.json', dict(error=repr(exc), training_updates=0, selection=False, depth=False))
        status('stopped_on_error', error=repr(exc))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=('eood', '110'), required=True)
    p.add_argument('--execute', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.execute:
        run(args.family)
    else:
        checked()
        print(json.dumps(dict(static_check_passed=True, TEST_started=False)))
