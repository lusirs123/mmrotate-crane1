#!/usr/bin/env python3
"""One native R4 run, unchanged training and VAL selection; no TEST stage."""
import argparse
import collections
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'crane_project/tools'))
REFERENCE = ROOT / 'work_dirs/symeood_110_retrain_seed0_20261026_v1'
EXTRA = (
    'mmrotate/models/dense_heads/sym_eood_reg_tower_head_v1.py',
    'mmrotate/models/dense_heads/sym_eood_tower_head_v1.py',
    'crane_project/tools/analyze_symeood_110_reg_tower_v1.py',
    'crane_project/utils/symeood_110_optimization_targets_v1.py',
    'crane_project/utils/symeood_tower_trace_v1.py',
    'crane_project/tools/run_symeood_110_reg_tower_v1.py',
    'crane_project/tools/check_symeood_110_reg_tower_v1.py',
    'tests/test_symeood_110_reg_tower_v1.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def assert_model_delta(old, new):
    a, b = deepcopy(old), deepcopy(new)
    assert a['bbox_head']['type'] == 'SymEOODHead'
    assert b['bbox_head']['type'] == 'SymEOODRegTowerHeadV1'
    b['bbox_head']['type'] = a['bbox_head']['type']
    assert a == b, 'Model differs beyond main tower head type'
    assert b['bbox_head']['stacked_convs'] == 4


def assert_config_delta(old, new):
    """Permit structural registration and run paths, not training changes."""
    assert_model_delta(old['model'], new['model'])
    for key in set(old) | set(new):
        if key not in ('model','custom_imports','work_dir','load_from','custom_hooks'):
            assert old.get(key) == new.get(key), ('Unexpected config change', key)
    expected = deepcopy(old['custom_imports'])
    expected['imports'] += ['mmrotate.models.dense_heads.sym_eood_reg_tower_head_v1',
                            'crane_project.utils.symeood_tower_trace_v1']
    assert new['custom_imports'] == expected
    hooks = deepcopy(new['custom_hooks'])
    assert len(hooks) == len(old['custom_hooks']) == 1
    assert hooks[0]['type'] == 'SymEOODTowerTraceV1'
    assert hooks[0].pop('reference_work') == str(REFERENCE/'110')
    hooks[0]['type'] = old['custom_hooks'][0]['type']
    hooks[0]['runtime_lock'] = old['custom_hooks'][0]['runtime_lock']
    assert hooks == old['custom_hooks'], 'Observation hook changed beyond reference identity'
    assert old['load_from'] is None and old['resume_from'] is None
    assert new['resume_from'] is None


def mapped_launch_environment(reference_launch, physical_gpus):
    # Runtime snapshots include cv2's import-time LD_LIBRARY_PATH prepend.
    # Launch from the original pre-import environment, never that snapshot.
    env = deepcopy(reference_launch)
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, physical_gpus))
    return env


def prepare(base, physical_gpus):
    import torch
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from crane_project.utils.symeood_repro_trace_v1 import tensor_digest
    from ckpt_sweep import SELECTION_CONFIG
    old = read(REFERENCE / 'protocol.json')
    assert old['epochs'] == 24 and old['candidate_epochs'] == [16,18,20,22,24]
    assert old['selection'] == SELECTION_CONFIG
    cfgpath = REFERENCE / 'configs/Coff_seed0.py'
    assert sha(cfgpath) == old['config_sha256']
    for name, digest in old['sources'].items():
        assert sha(ROOT / name) == digest, ('Reference source drift', name)
    for row in old['data_inventory']:
        for key in ('image', 'annotation'):
            assert sha(ROOT / row[key]) == row[key + '_sha256']
    cfg = Config.fromfile(str(cfgpath))
    previous = deepcopy(cfg.to_dict())
    assert cfg.load_from is None and cfg.resume_from is None
    assert cfg.data.samples_per_gpu == 2 and cfg.data.workers_per_gpu == 2
    base.mkdir(parents=True, exist_ok=False)
    cfg.model.bbox_head.type = 'SymEOODRegTowerHeadV1'
    assert_model_delta(previous['model'], cfg.to_dict()['model'])
    cfg.custom_imports.imports += [
        'mmrotate.models.dense_heads.sym_eood_reg_tower_head_v1',
        'crane_project.utils.symeood_tower_trace_v1']
    import_modules_from_strings(**cfg.custom_imports)
    mc = deepcopy(cfg.model)
    mc.backbone.init_cfg = None
    model = build_detector(mc)
    original_init = REFERENCE / '110/complete_initial_model.pth'
    initial = torch.load(str(original_init), map_location='cpu')['state_dict']
    state = model.state_dict()
    extra = set(state)-set(initial)
    assert len(extra) == 8 and not (set(initial)-set(state))
    assert all(k.startswith('bbox_head.reg_convs.') for k in extra)
    tower_initial = ROOT/'work_dirs/symeood_110_tower_control_20261029_v2/initial_T4.pth'
    tower_protocol = read(tower_initial.parent/'protocol.json')
    assert sha(tower_initial) == tower_protocol['initial_sha256']
    tower_state = torch.load(str(tower_initial), map_location='cpu')['state_dict']
    for key in state:
        source = initial if key in initial else tower_state
        assert state[key].shape == source[key].shape
        state[key] = source[key].clone()
    model.load_state_dict(state, strict=True)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in initial.items())
    added = sum(state[k].numel() for k in extra)
    assert added == 2360320
    assert sum(p.numel() for p in model.parameters()) == 33921112
    init_path = base / 'initial_R4.pth'
    torch.save(dict(state_dict=state, meta=dict(purpose='training-before-update initialization',
        reference_initial_sha256=sha(original_init), regression_initial_sha256=sha(tower_initial))), str(init_path))
    initial_digest = tensor_digest(model.state_dict().items())
    cfg.work_dir = str(base / 'R4')
    cfg.load_from = str(init_path)
    cfg.custom_hooks[0].update(type='SymEOODTowerTraceV1',
        runtime_lock=str(base / 'runtime_lock.json'), reference_work=str(REFERENCE / '110'))
    cfg.dump(str(base / 'config.py'))
    # All non-structural training/data/inference settings must be identical.
    assert_config_delta(previous, cfg.to_dict())
    sources = dict(old['sources'])
    sources.update({p: sha(ROOT/p) for p in EXTRA})
    lock = read(REFERENCE / 'runtime_identity_gpus01.json')
    assert len(physical_gpus) == 2 and len(set(physical_gpus)) == 2
    lock['physical_gpus'] = list(physical_gpus)
    lock['runtime']['environment']['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, physical_gpus))
    lock['source_sha256'].update({str(ROOT/p): h for p,h in sources.items()})
    save(base / 'runtime_lock.json', lock)
    save(base / 'protocol.json', dict(protocol='symeood_110_r4_single_structure_v1',
        formal_runs=1, epochs=24, seed=0, world_size=2, per_rank_batch=2,
        physical_gpus=list(physical_gpus),
        config=str(base/'config.py'), config_sha256=sha(base/'config.py'),
        initial_checkpoint=str(init_path), initial_sha256=sha(init_path),
        initial_state_sha256=initial_digest, common_initial_exact=True,
        reference_initial_sha256=sha(original_init), added_parameters=added,
        sources=sources, reference_protocol_sha256=sha(REFERENCE/'protocol.json'),
        runtime_lock_sha256=sha(base/'runtime_lock.json'),
        launch_environment=mapped_launch_environment(old['launch_environment'],physical_gpus), selection=SELECTION_CONFIG,
        memory_policy=dict(allocator_limit_mib_per_process=6144,
            enforced_in='SymEOODRegTowerHeadV1 constructor, training/gate/VAL inference',
            driver_overhead_outside_allocator=True, OOM='stop, no retry or batch change'),
        task_target_module='crane_project/utils/symeood_110_optimization_targets_v1.py',
        task_target_sha256=sha(ROOT/'crane_project/utils/symeood_110_optimization_targets_v1.py'),
        candidate_epochs=[16,18,20,22,24], TEST=False, downstream=False,
        automatic_adoption=False, retry=False, initialization='Shared tensors copied from pretraining-before-update original110; regression tower copied exactly from T4 pre-update initialization; no classification tower; not trained detector weights.',
        decision='Report original selection and separate prespecified task-goal eligible selection; no TEST or automatic adoption.'))
    save(base / 'prepared.json', dict(status='PREPARED', formal_training=False,
        added_state_keys=sorted(extra), parameter_count=33921112,
        reference_git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    print('PREPARED', base, flush=True)


def env_for(c):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    for k,v in c['launch_environment'].items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return env


def integrity(base):
    c = read(base/'protocol.json')
    for name,digest in c['sources'].items():
        assert sha(ROOT/name) == digest, ('Source drift', name)
    assert sha(c['config']) == c['config_sha256']
    assert sha(c['initial_checkpoint']) == c['initial_sha256']
    assert sha(base/'runtime_lock.json') == c['runtime_lock_sha256']
    return c


def native(c, work, gate=False):
    return [sys.executable, '-m', 'torch.distributed.launch', '--nproc_per_node=2',
            '--master_port=29687', 'crane_project/tools/check_symeood_110_reg_tower_v1.py' if gate else 'tools/train.py',
            c['config'], '--launcher','pytorch','--seed','0','--work-dir',str(work)]


def execute(base, stage, command):
    c = integrity(base)
    status = dict(stage=stage, state='running', command=command, pid=os.getpid())
    (base/'status.json').write_text(json.dumps(status, indent=2))
    with (base/(stage+'.log')).open('x') as f:
        subprocess.run(command, cwd=str(ROOT), env=env_for(c), stdout=f, stderr=subprocess.STDOUT, check=True)


def gate(base):
    c = integrity(base)
    assert not (base/'gate_work').exists(), 'Never reuse a disposable gate'
    execute(base,'gate',native(c,base/'gate_work',True))
    rr = [read(base/'gate_work'/('gate_rank%d.json'%rank)) for rank in (0,1)]
    assert all(x['passed'] and x['optimizer_updates']==0 for x in rr)
    save(base/'gate_acceptance.json',dict(passed=True,rows=rr,discarded=True))
    print('GATE_PASS_DISCARDED',flush=True)


def write_csv(path, rows):
    with path.open('x') as f:
        w = csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        w.writeheader(); w.writerows(rows)


def target_readback_command(base):
    return [sys.executable, 'crane_project/tools/analyze_symeood_110_reg_tower_v1.py',
            '--base', str(base), '--out', str(base / 'target_VAL')]


def analyze(base):
    import pickle
    import numpy as np
    from analyze_symeood_ablation_first4_v1 import aggregate, txtbox, iou, angle
    from analyze_symeood_c_pair_v1 import frames
    c = integrity(base)
    out = base/'analysis_VAL'; out.mkdir(exist_ok=False)
    sweeps = dict(S=read(REFERENCE/'110/ckpt_sweep/sweep_results.json'),
                  R4=read(base/'R4/ckpt_sweep/sweep_results.json'))
    assert all(s['selection_config']==c['selection'] for s in sweeps.values())
    metrics=[]; allrows=[]
    for side,sw in sweeps.items():
        for ep,cp in sw['all_checkpoints'].items():
            rr=frames(cp,'val',side,ep)
            allrows.extend(rr)
            for domain in ('all','real','sim'):
                ss=[r for r in rr if domain=='all' or r['domain']==domain]
                g=aggregate(ss)
                # Preserve historical TXT metrics for selection diagnostics;
                # separately expose AP on native unrounded saved boxes.
                native_rows=[dict(r,riou_export=r['riou_native']) for r in ss]
                from analyze_symeood_ablation_first4_v1 import ap11
                g.update(AP50_native=ap11(native_rows,.5),AP75_native=ap11(native_rows,.75))
                valid=[r for r in ss if r['output']]
                g.update(center_MAE_px=float(np.mean([r['center_error'] for r in valid])),
                    Recall_pct=100*sum(r['riou_native']>=.5 for r in ss)/len(ss),
                    pure_angle_RMSE_deg=math.sqrt(float(np.mean([r['angle_error_deg']**2 for r in valid]))) if domain=='sim' else None)
                metrics.append(dict(side=side,epoch=ep,domain=domain,
                    selected=ep==sw['selected_checkpoint'],**g))
    write_csv(out/'all_five_VAL_metrics.csv',metrics)
    write_csv(out/'per_frame_VAL.csv',allrows)
    save(out/'selection_identity.json',{k:v['selected_checkpoint'] for k,v in sweeps.items()})
    save(out/'completion.json',dict(status='COMPLETE',TEST=False,formal_runs=1,
        selections={k:v['selected_checkpoint'] for k,v in sweeps.items()},rows=len(allrows),
        note='AP uses existing VOC07 CPU reference; main metrics include raw-box/native RIoU separately from TXT-rounded historical selection. No reselect.'))


def bundle(base):
    paths=[p for p in base.rglob('*') if p.is_file() and p.suffix not in ('.pth','.gz')]
    manifest={str(p.relative_to(base)):sha(p) for p in paths}
    save(base/'transfer_manifest.json',manifest)
    with tarfile.open(base/'analysis_bundle.tar.gz','x:gz') as tar:
        for p in paths+[base/'transfer_manifest.json']:
            tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
    save(base/'archive_receipt.json',dict(sha256=sha(base/'analysis_bundle.tar.gz'),
        includes_weights=False,retain='All raw weights, logs, predictions and per-frame analyses retained; archive temporary after verified extraction.'))


def run(base):
    c=integrity(base)
    assert read(base/'gate_acceptance.json')['passed']
    save(base/'run_claim.json',dict(pid=os.getpid(),started_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    assert not (base/'R4').exists(), 'Never resume or overwrite a formal run'
    try:
        execute(base,'train',native(c,base/'R4'))
        logs=[json.loads(l) for p in (base/'R4').glob('*.log.json') for l in p.read_text().splitlines()]
        assert {r['epoch'] for r in logs if r.get('mode')=='train'}==set(range(1,25))
        assert all(math.isfinite(v) for r in logs if r.get('mode') in ('train','val') for v in r.values() if isinstance(v,(float,int)))
        for rank in (0,1):
            first=read_line(base/'R4'/('repro_initial_rank%d.jsonl'%rank))
            assert first['initial_state_sha256']==c['initial_state_sha256']
            newfile=base/'R4'/('repro_inputs_rank%d.jsonl'%rank)
            oldfile=REFERENCE/'110'/newfile.name
            matches=0
            with newfile.open() as nf,oldfile.open() as of:
                for n,o in zip(nf,of):
                    a,b=json.loads(n),json.loads(o)
                    assert (a['epoch'],a['iteration'],a['combined_sha256'])==(b['epoch'],b['iteration'],b['combined_sha256'])
                    matches+=1
            assert matches==15360
            assert sum(1 for _ in newfile.open())==15360
            rr=[json.loads(l) for l in (base/'R4'/('repro_conservation_rank%d.jsonl'%rank)).read_text().splitlines()]
            assert len(rr)==24 and all(x['passed'] for x in rr)
        save(base/'training_verification.json',dict(passed=True,updates=15360,rank_input_records_each=15360,
            original_input_stream_exact=True,shared_initial_exact=True))
        execute(base,'VAL_sweep',[sys.executable,'crane_project/tools/ckpt_sweep.py',
            '--config',c['config'],'--work-dir',str(base/'R4'),'--epochs','16','18','20','22','24',
            '--gpus',*map(str,c['physical_gpus']),'--center-thresh','15','--mcml-limit','5'])
        analyze(base)
        execute(base,'target_VAL_readback',target_readback_command(base))
        sw=read(base/'R4/ckpt_sweep/sweep_results.json');selected=sw['selected_checkpoint']
        backup=Path('/home/omnisky/symeood_model_backups')/base.name
        backup.mkdir(parents=True,exist_ok=False)
        source=base/'R4'/(selected+'.pth');target=backup/'selected_R4.pth'
        shutil.copyfile(source,target);assert sha(source)==sha(target)
        save(base/'backup_receipt.json',dict(selected=selected,path=str(target),sha256=sha(target)))
        (base/'status.json').write_text(json.dumps(dict(state='complete',stage='VAL_analysis',selected=selected,TEST=False),indent=2))
        bundle(base)
        print('R4_TRAIN_VAL_COMPLETE',selected,flush=True)
    except BaseException as e:
        (base/'status.json').write_text(json.dumps(dict(state='stopped_on_error',error=repr(e),retry=False),indent=2))
        raise


def read_line(path):
    rows=Path(path).read_text().splitlines();assert len(rows)==1
    return json.loads(rows[0])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['prepare','gate','run','analyze','bundle'],required=True)
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--physical-gpus',type=int,nargs=2,help='Required for prepare; frozen into protocol.')
    a=ap.parse_args()
    if a.stage=='prepare':
        if a.physical_gpus is None:ap.error('prepare requires --physical-gpus')
        prepare(a.base.resolve(),a.physical_gpus)
    else:
        if a.physical_gpus is not None:ap.error('GPU mapping is already frozen in protocol')
        globals()[a.stage](a.base.resolve())
