"""One direct-size control; original V implementation/results are reused."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT/'crane_project/configs/crane_lvsr_r4_vd_control_v1.json'
SOURCES = ['crane_project/configs/crane_lvsr_r4_vd_control_v1.json',
    'crane_project/utils/port_size_pair_direct_control_v1.py',
    'crane_project/tools/train_lvsr_vd_control_v1.py', 'tests/test_lvsr_vd_control_v1.py',
    'work_dirs/lvsr_diagnostic_20261112_v1/diagnose.py',
    'crane_project/tools/train_lvsr_separate_activation_v1.py',
    'crane_project/utils/port_size_structure_v1.py']


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    return {n: sha(ROOT/n) for n in SOURCES}


def checked_config():
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    c = read(CONFIG)
    for key in ('epochs', 'batch_size', 'seed', 'lr', 'weight_decay', 'betas', 'eps', 'clip_norm'):
        if c[key] != p.SETTINGS[key]:
            raise ValueError('Original recipe differs: '+key)
    if c['physical_gpu'] != 3 or c['parameters'] != dict(V=2480, D=2450) or c['TEST']:
        raise ValueError('Wrong control scope')
    return c


def pick(events, p):
    # Deliberately no eligibility filtering for either representation.
    return min(events, key=lambda e: p.selection_key(e['metrics'], e['epoch']))


def status(out, stage, **kw):
    row = dict(stage=stage, pid=os.getpid(), TEST=False,
        time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(), **kw)
    tmp = out/'status.tmp'
    tmp.write_text(json.dumps(row, allow_nan=False))
    tmp.replace(out/'status.json')
    print(json.dumps(row, allow_nan=False), flush=True)


def frame_hash(present, order):
    return hashlib.sha256(json.dumps([(present[i]['image'], present[i]['scale']) for i in order],
        separators=(',', ':')).encode()).hexdigest()


def all_groups(rows, common):
    return {name: common['grouped']([r for r in rows if predicate(r)]) for name, predicate in
        [('train_s1', lambda r: r['split'] != 'val' and r['scale'] == 1.),
         ('train_s05', lambda r: r['split'] != 'val' and r['scale'] == .5),
         ('val', lambda r: r['split'] == 'val')]}


def new_pair(torch, seed):
    from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
    from crane_project.utils.port_size_pair_direct_control_v1 import PairDirectSizeControlHead
    seed(torch)
    v = PairSpecificSizeHead()
    d = PairDirectSizeControlHead(v)  # Deep-copy actual initialized tensors.
    if any(not torch.equal(a, b) for a, b in zip(v.stem.state_dict().values(), d.stem.state_dict().values())):
        raise ValueError('Copied V/D stems differ')
    return v, d


def engineering(engine, base, original, p, train, val, source, out, seed, equal, rng_get, rng_set):
    torch = engine.torch
    present = [r for r in train if len(r['midpoint_original'])]
    v, d = new_pair(torch, seed)
    initial_v, initial_d = engine.g.state_digest(v), engine.g.state_digest(d)
    prior = read(source/'verification/completion.json')
    if prior['initial_state'] != initial_v or not prior['discarded']:
        raise ValueError('Original V fresh initialization/verification differs')
    initial_d_state = deepcopy(d.state_dict())
    # Original epoch1 replay certifies the reused V initialization/Adam/order.
    v = v.cuda()
    opt = torch.optim.Adam(v.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    order = base.schedule_indices(len(present), 1)
    for start in range(0, len(order), 8):
        base.update(v, opt, base.batch([present[i] for i in order[start:start+8]], torch), 'fixed_b', torch)
        if start % 800 == 0:
            engine.resource_check(0)
            status(out, 'V_epoch1_replay', step=start//8+1, target=639)
    saved = torch.load(str(source/'train/head_epoch_01.pth'), map_location='cpu')
    if (saved['updates'] != 639 or not equal(v.state_dict(), saved['head_state'], torch)
            or not equal(opt.state_dict(), saved['optimizer_state'], torch)):
        raise ValueError('Original V epoch1 head/Adam replay differs; do not retrain V automatically')
    if original.summarize(base.evaluate(v, val, torch)) != read(source/'train/epoch_01_VAL.json')['metrics']:
        raise ValueError('Reused V epoch1 VAL replay differs')
    del v, opt, saved
    d = d.cuda()
    neutral = base.evaluate(d, train+val, torch)
    if any(r['midpoint'] != r['edge_residual'] for r in neutral):
        raise ValueError('D6003-view neutral output differs from M')
    # Evaluate actual empty outputs too (base evaluator skips their forwards).
    empty = next(r for r in train if not len(r['midpoint_original']))
    with torch.no_grad():
        x = [empty[k].cuda() for k in ('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original')]
        if len(d(*x)['boxes_original']):
            raise ValueError('D empty forward created output')
    baseline = original.summarize([r for r in neutral if r['split'] == 'val'])
    chosen = [r for domain in ('real', 'sim') for scale in (.5, 1.)
              for r in [q for q in present if q['domain'] == domain and q['scale'] == scale][:4]]
    opt = torch.optim.Adam(d.parameters(), lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=0.)
    logs = [base.update(d, opt, base.batch(chosen[i:i+8], torch), 'fixed_b', torch) for i in (0, 8)]
    if logs[0]['terminal_grad_norm'] <= 0 or logs[1]['stem_grad_norm'] <= 0:
        raise ValueError('Direct terminal/stem cannot learn')
    snapshot = dict(head=deepcopy(d.state_dict()), optimizer=deepcopy(opt.state_dict()), rng=rng_get(torch))
    engine.atomic_save(out/'discarded_D_smoke.pth', snapshot)
    loaded = torch.load(str(out/'discarded_D_smoke.pth'), map_location='cpu')
    if not equal(snapshot, loaded, torch):
        raise ValueError('D head/Adam/RNG save differs')
    _, clone = new_pair(torch, seed)
    clone = clone.cuda()
    clone.load_state_dict(loaded['head'], strict=True)
    other = torch.optim.Adam(clone.parameters(), lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=0.)
    other.load_state_dict(loaded['optimizer'])
    if base.evaluate(d, chosen, torch) != base.evaluate(clone, chosen, torch):
        raise ValueError('D reloaded inference differs')
    rng_set(snapshot['rng'], torch)
    a = base.update(d, opt, base.batch(chosen[:8], torch), 'fixed_b', torch)
    rng_set(loaded['rng'], torch)
    b = base.update(clone, other, base.batch(chosen[:8], torch), 'fixed_b', torch)
    if a != b or not equal(d.state_dict(), clone.state_dict(), torch) or not equal(opt.state_dict(), other.state_dict(), torch):
        raise ValueError('D next optimizer step differs after restore')
    proof = dict(passed=True, original_V_epoch1_head_Adam_and_VAL_exact=True,
        V_replay_updates_discarded=639, D_smoke_updates_discarded=4, copied_stems_equal=True,
        initial_V=initial_v, initial_D=initial_d, neutral_views=6003, empty_forward_passed=True,
        D_restore_and_next_step_exact=True, TEST=False)
    save(out/'engineering.json', proof)
    save(out/'baseline_VAL.json', baseline)
    return initial_d_state, proof, baseline


def timing(v, d, val, base, torch):
    subset = []
    for seq in sorted({r['sequence'] for r in val}):
        rr = sorted([r for r in val if r['sequence'] == seq], key=lambda r:r['frame_id'])
        subset.extend(rr[round(i*(len(rr)-1)/9)] for i in range(10))
    def one(h, r):
        x = [r[k].cuda() for k in ('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original')]
        o = h(*x)
        return o['boxes_original'].cpu().tolist()
    result = dict(V=[], D=[])
    v.eval(); d.eval()
    with torch.no_grad():
        for h in (v, d):
            for i in range(20):
                one(h, subset[i % len(subset)])
        for block in range(3):
            for name, h in ([('V',v),('D',d)] if block % 2 == 0 else [('D',d),('V',v)]):
                torch.cuda.synchronize(); tick = time.perf_counter()
                for r in subset:
                    one(h, r)
                torch.cuda.synchronize()
                result[name].append(1000*(time.perf_counter()-tick)/len(subset))
    return dict(scope='CPU cached ROI/M inputs -> H2D/head/CPU OBB, batch1; excludes detector/ROI extraction/M',
        frames=30, warmup=20, repeats=3, mean_ms={k:statistics.mean(a) for k,a in result.items()},
        blocks_ms=result, full_chain_latency=False)


def run(c, out):
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    from crane_project.tools import train_symeood_r4_lvsr_pair_v1 as original
    from crane_project.tools import train_lvsr_separate_activation_v1 as gates
    from crane_project.tools.run_port_size_native_train_v1 import seed, rng_get, rng_set, equal
    hashes = source_hashes()
    parent = p.checked_sources()
    source = p.paths()['attempt']
    engine, base, V, vector, runtime = original.runtime(3)
    torch = engine.torch
    uuid = subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=uuid','--format=csv,noheader'], text=True).strip()
    if uuid != c['gpu_uuid']:
        raise ValueError('GPU3 UUID differs')
    train, val, inputs = original.load_inputs(p.paths(), parent, engine)
    common = runpy.run_path(str(ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/diagnose.py'))
    before = common['memory_digest'](train+val)
    present = [r for r in train if len(r['midpoint_original'])]
    if len(present) != 5112 or len(val) != 887:
        raise ValueError('Wrong frozen input counts')
    paths = [ROOT/p.parent.R4_CHECKPOINT, p.midpoint_directory()/'train/head_epoch_20.pth',
        source/'inputs/train.pt', source/'inputs/val.pt', source/'inputs/completion.json',
        source/'verification/completion.json', source/'train/selection.json', source/'train/completion.json']
    vevents = []
    old_selection = read(source/'train/selection.json')
    if sha(source/'train/selection.json') != '94bfe79882ae625863639c64d09df07209cf03b4b2d11600b84649166eda7887':
        raise ValueError('Original sealed V selection receipt differs')
    for epoch in range(1, 25):
        paths.extend([source/'train'/('head_epoch_%02d.pth'%epoch), source/'train'/('epoch_%02d_VAL.json'%epoch)])
        event = read(paths[-1])
        payload = torch.load(str(paths[-2]), map_location='cpu')
        if (payload['identity'] != parent or payload['updates'] != epoch*639 or payload['parameters'] != 2480
                or payload['detector_sha256'] != p.parent.R4_SHA or payload['midpoint_sha256'] != p.MIDPOINT_SHA
                or event != old_selection['epochs'][epoch-1] or sha(paths[-2]) != event['checkpoint']['sha256']):
            raise ValueError('Historical V24 identity/update budget differs')
        vevents.append(event)
    pins = {str(q.relative_to(ROOT)):p.sha(q) for q in paths}
    if pins[str((source/'train/head_epoch_05.pth').relative_to(ROOT))] != 'b272c50dfe6329414803d5f22c6e1e84d9c3db37b6bb2e31f1c7ed140f662c3e':
        raise ValueError('Historical L5 checkpoint differs')
    if read(source/'train/selection.json')['selected_epoch'] is not None or pick(vevents,p)['epoch'] != 5:
        raise ValueError('Preserve original V descriptive ep5 and null selection')
    plans = {}
    for epoch in range(1, 25):
        indices = base.schedule_indices(5112, epoch)
        independent = list(range(5112));random.Random(1703+epoch).shuffle(independent)
        if indices != independent or sorted(indices) != list(range(5112)):
            raise ValueError('Original schedule differs')
        plans[str(epoch)] = dict(views=5112, frame_order_SHA=frame_hash(present, indices))
    identity = dict(settings=c, sources=hashes, parent=parent)
    save(out/'protocol.json', dict(identity=identity, frozen_pins=pins, runtime=runtime,
        planned_sampling=plans, video_scale_counts=dict(Counter(r['sequence']+'|'+str(r['scale']) for r in present)),
        historical_V_reused=True, formal_V_training=False, TEST=False))
    status(out, 'engineering')
    initial, proof, baseline = engineering(engine,base,original,p,train,val,source,out,seed,equal,rng_get,rng_set)
    _, h = new_pair(torch, seed)
    h.load_state_dict(initial, strict=True);h = h.cuda()
    if engine.g.state_digest(h) != proof['initial_D']:
        raise ValueError('Smoke state reused instead of fresh D')
    opt = torch.optim.Adam(h.parameters(), lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=0.)
    events = []
    updates = 0
    for epoch in range(1, 25):
        tick = time.monotonic();order = base.schedule_indices(5112, epoch)
        loss_sum = 0.; exposure = Counter(); actual_order = []
        for start in range(0, 5112, 8):
            rr = [present[i] for i in order[start:start+8]]
            stats = base.update(h,opt,base.batch(rr,torch),'fixed_b',torch);updates += 1
            exposure.update(r['sequence']+'|'+str(r['scale']) for r in rr)
            actual_order.extend(order[start:start+8]);loss_sum += stats['loss']
            if any(int(opt.state[q]['step']) != updates for q in h.parameters()):
                raise ValueError('D Adam update counter differs')
            if start % 800 == 0:
                engine.resource_check(0);status(out,'D_training',epoch=epoch,step=start//8+1,updates=updates)
        if (updates != epoch*639 or frame_hash(present,actual_order) != plans[str(epoch)]['frame_order_SHA']
                or dict(exposure) != dict(Counter(r['sequence']+'|'+str(r['scale']) for r in present))):
            raise ValueError('Actual D order/exposure differs from V recipe')
        vr = base.evaluate(h,val,torch);metrics = original.summarize(vr)
        for domain in metrics:
            if not metrics[domain]['frozen_identity'] or any(metrics[domain][m] != baseline[domain][m] for m in ('b','midpoint')):
                raise ValueError('Frozen baseline/identity changed')
        cp = out/('D_epoch_%02d.pth'%epoch)
        payload = dict(protocol=c['protocol'],identity=identity,epoch=epoch,updates=updates,parameters=2450,
            head_state={k:v.detach().cpu().clone() for k,v in h.state_dict().items()},optimizer_state=deepcopy(opt.state_dict()),
            rng=rng_get(torch),parent_checkpoint=None,head_type='PairDirectSizeControlHead')
        engine.atomic_save(cp,payload)
        if not equal(payload,torch.load(str(cp),map_location='cpu'),torch):
            raise ValueError('D checkpoint reload differs')
        event = dict(epoch=epoch,updates=updates,metrics=metrics,selection_key=p.selection_key(metrics,epoch),
            original_chain_gate=p.assess(metrics),increment_vs_M=gates.increment_gate(metrics),
            versus_L5=gates.assess(metrics,vevents[4]['metrics']),
            checkpoint=dict(path=cp.name,sha256=p.sha(cp),epoch=epoch),loss_mean=loss_sum/639,
            sampling=dict(frame_order_SHA=frame_hash(present,actual_order),video_scale_counts=dict(exposure)),
            seconds=time.monotonic()-tick)
        save(out/('D_epoch_%02d_VAL.json'%epoch),event)
        original.write_rows(out/('D_epoch_%02d_VAL.rows.jsonl.gz'%epoch),vr)
        events.append(event)
        status(out,'D_epoch_complete',epoch=epoch,updates=updates,metrics=metrics['overall']['edge_residual'])
    chosen = pick(events,p)
    selected = chosen['epoch'] if chosen['original_chain_gate']['pass_all'] else None
    save(out/'selection.json',dict(representative_epoch=chosen['epoch'],representative_checkpoint=chosen['checkpoint'],
        selected_epoch=selected,ranking='all24 original ordering; gates not prefilter',TEST=False,automatic_adoption=False))
    from crane_project.utils.port_size_pair_direct_control_v1 import PairDirectSizeControlHead
    replay = PairDirectSizeControlHead().cuda()
    replay.load_state_dict(torch.load(str(out/chosen['checkpoint']['path']),map_location='cpu')['head_state'],strict=True)
    dr = base.evaluate(replay,train+val,torch)
    if original.summarize([r for r in dr if r['split']=='val']) != chosen['metrics']:
        raise ValueError('D representative replay differs')
    original.write_rows(out/'D_representative_TRAIN_VAL.rows.jsonl.gz',dr)
    v = V().cuda();v.load_state_dict(torch.load(str(source/'train/head_epoch_05.pth'),map_location='cpu')['head_state'],strict=True)
    vr = base.evaluate(v,train+val,torch)
    if original.summarize([r for r in vr if r['split']=='val']) != vevents[4]['metrics']:
        raise ValueError('V ep5 replay differs')
    original.write_rows(out/'V_reference_TRAIN_VAL.rows.jsonl.gz',vr)
    details = dict(V=all_groups(vr,common),D=all_groups(dr,common))
    paired_rows = []
    for a,b in zip(vr,dr):
        if any(a[k]!=b[k] for k in ('image','scale','b','midpoint','gt')):
            raise ValueError('V/D paired identity differs')
        paired_rows.append(dict(b,midpoint=a['edge_residual'],loss_edges_canonical=None,delta_roi=None))
    # M fields replaced only in this offline size-difference report, not model inputs.
    details['D_versus_V_size_only'] = all_groups(paired_rows,common)
    save(out/'core_detail.json',details)
    time_value = timing(v,replay,val,base,torch);save(out/'core_timing.json',time_value)
    save(out/'core_summary.json',dict(settings=c,reference_V=vevents[4],representative_D=chosen,
        V_all24=vevents,D_all24=events,V_selected_epoch=None,D_selected_epoch=selected,
        TEST=False,automatic_adoption=False))
    with (out/'core_epochs.csv').open('x',newline='') as f:
        writer = csv.writer(f);writer.writerow(['arm','epoch','long_MAE_pct','short_MAE_pct','joint10','RIoU','AP75'])
        for arm,ee in [('V',vevents),('D',events)]:
            for e in ee:
                m=e['metrics']['overall']['edge_residual']
                writer.writerow([arm,e['epoch'],100*m['long_MAE'],100*m['short_MAE'],m['joint10'],m['mean_RIoU'],m['AP75']])
    if (common['memory_digest'](train+val)!=before or p.checked_sources()!=parent or source_hashes()!=hashes
            or any(p.sha(ROOT/n)!=h for n,h in pins.items())):
        raise ValueError('Frozen inputs/reference/source changed')
    save(out/'completion.json',dict(status='DIRECT_CONTROL_TRAIN_VAL_COMPLETE',D_epochs=24,D_updates=updates,
        historical_V_reused=True,V_replay_updates_discarded=639,D_engineering_updates_discarded=4,
        reference_representative=5,D_representative=chosen['epoch'],D_checkpoint=chosen['checkpoint'],
        source_and_inputs_unchanged=True,actual_exposure_and_order_verified=True,TEST=False,automatic_adoption=False))
    core=['protocol.json','engineering.json','baseline_VAL.json','selection.json','core_detail.json',
          'core_timing.json','core_summary.json','core_epochs.csv','completion.json']
    save(out/'core_manifest.json',{n:p.sha(out/n) for n in core})
    status(out,'complete',D_representative=chosen['epoch'],updates=updates)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    c=checked_config()
    if not args.run:
        print(json.dumps(dict(settings=c,sources=source_hashes()),indent=2));return
    out=ROOT/c['output_root'];out.mkdir(exist_ok=False)
    def stop(*unused):
        raise RuntimeError('Fixed7200-second experiment budget exceeded')
    signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,c['wall_seconds'])
    try:
        run(c,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error=repr(exc),TEST=False,retry=False));status(out,'failed',error=repr(exc));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    main()
