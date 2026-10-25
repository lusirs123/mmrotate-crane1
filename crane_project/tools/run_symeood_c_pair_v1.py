"""Exactly two native24epoch seed0 runs:110 then111, original VAL and TEST.

Requires prior native4step gate. No offload, USG loss, probes, auto-resume,
repeat-until-good, promotion, deletion or overwriting existing run folders.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'crane_project/tools'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()


def paired_environment(parent):
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for key,value in parent.items():
        if value is None:env.pop(key,None)
        else:env[key]=value
    assert env['CUDA_VISIBLE_DEVICES']=='2,3'
    return env


def freeze(base,passed_gate,launch_evidence):
    from mmcv import Config
    from prepare_symeood_b111_canonical_v1 import B_CONFIG,diff
    base.mkdir(parents=True,exist_ok=True)
    gate=passed_gate.resolve();accept=json.loads((gate/'acceptance.json').read_text());assert accept['passed']
    lock=json.loads((gate/'native_runtime_lock.json').read_text());assert lock['state']=='NATIVE_GPU_ACCEPTED'
    propagation=json.loads(launch_evidence.read_text());assert propagation['passed'] and propagation['child']==lock['runtime']['environment']
    env=paired_environment(propagation['parent'])
    for p,h in lock['source_sha256'].items():assert sha(ROOT/p)==h,p
    cpu=ROOT/'work_dirs/symeood_b111_canonical_20261025_v2'
    inventory=json.loads((cpu/'TRAIN_VAL_inventory.json').read_text());assert len(inventory)==3445
    for r in inventory:assert sha(ROOT/r['image'])==r['image_sha256'] and sha(ROOT/r['annotation'])==r['annotation_sha256']
    for g in ('110','111'):assert not (base/g).exists(),('run exists',g)
    configs={};(base/'configs').mkdir(exist_ok=False)
    for g in ('110','111'):
        cfg=Config.fromfile(str(B_CONFIG));cfg.work_dir=str(base/g)
        if g=='110':cfg.model.bbox_head.loss_cls.use_geometry_weight=False
        cfg.custom_imports.imports=list(cfg.custom_imports.imports)+['crane_project.utils.symeood_repro_trace_v1']
        cfg.custom_hooks=[dict(type='SymEOODReproTraceV1',priority='LOWEST',runtime_lock=str(gate/'native_runtime_lock.json'),engineering=False)]
        assert cfg.runner.max_epochs==24 and cfg.load_from is None and cfg.resume_from is None
        assert cfg.data.samples_per_gpu==2 and cfg.data.workers_per_gpu==2
        p=base/'configs'/('C'+('off' if g=='110' else 'on')+'_seed0.py');cfg.dump(str(p));configs[g]=str(p)
    a,b=(Config.fromfile(configs[g]).to_dict() for g in ('110','111'))
    b['model']['bbox_head']['loss_cls']['use_geometry_weight']=True
    assert set(diff(a,b))=={'.work_dir','.model.bbox_head.loss_cls.use_geometry_weight'}
    sources=dict(lock['source_sha256'])
    extras=[Path(__file__).resolve(),ROOT/'crane_project/tools/ckpt_sweep.py',ROOT/'crane_project/tools/eval_crane_offline.py',ROOT/'crane_project/tools/analyze_symeood_c_pair_v1.py']
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in extras})
    sourcepaths=list((ROOT/'configs').rglob('*.py'))
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in sourcepaths})
    from ckpt_sweep import SELECTION_CONFIG
    contract=dict(state='FROZEN',protocol='symeood_C110_C111_native_pair_seed0_v1',date='2026-10-25',
        groups=['110','111'],run_order=['110','111'],seed=0,runs_each=1,epochs_each=24,world_size=2,per_rank_batch=2,
        physical_gpus=[2,3],optimizer=a['optimizer'],lr=a['lr_config'],clip=a['optimizer_config'],
        base_config=str(B_CONFIG),configurations=configs,config_sha256={g:sha(p) for g,p in configs.items()},
        native_gate_path=str(gate),launch_environment=propagation['parent'],launch_environment_evidence=str(launch_evidence.resolve()),
        native_gate_sha256=sha(gate/'acceptance.json'),runtime_lock_sha256=sha(gate/'native_runtime_lock.json'),
        sources=sources,data_inventory=inventory,pretrained=lock['pretrained'],
        common_initialization='fresh independent CPU initialization seed0; exact full parameter/buffer and RNG identity required; no historical detector weights',
        input_pairing='same sampler,worker seeds,batch2,augmentations; compare everyrank/epoch input hash, preserve records',
        training='actual tools/train.py native DDP FP32; intermediates stayGPU; complete runs sequential; no offload or recompute',
        additions='identity/state/input logging only, native deterministic4step invariance accepted; no USG counterfactual or gradient probe',
        candidate_epochs=[16,18,20,22,24],online_VAL_epochs=list(range(2,25,2)),selection=dict(SELECTION_CONFIG),
        center_thresh_px=15,ep24_role='auxiliary fixed budget, not second main selector',TEST='each original VAL selected version only; same evaluator, no TEST-driven reselection',depth=False,
        comparisons=dict(primary=['fullframe center correct coverage','RIoU>=.5 hit count/output coverage','MCML max/mean and failureevents'],
            required_costs=['Real/Sim and pervideo RIoU,DFR,ACI','Sim A-RMSE','samevalidpairsDFR/ACI','angle penalty decomposition'],size='auxiliary'),
        adoption='No automatic model replacement or posthoc utility score. Report all tradeoffs. No numeric acceptable-cost gate supplied; final adoption remains review decision.',
        inference_limit='one paired rerun at same seed0; not independent-seed stability or significance; retain old and new results',
        stop='exactly2complete runs then stop; numerical/identity error stops queue; no bad-score retries or extension; no automaticresume',
        downstream='110cannot inherit historicalB midpoint/LVSR results; no new downstream run authorized here',time=time.strftime('%Y-%m-%d %H:%M:%S'))
    (base/'protocol.json').write_text(json.dumps(contract,indent=2));return contract,env


def run(base):
    base=base.resolve();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2,3'
    contract=json.loads((base/'protocol.json').read_text());assert contract['state']=='FROZEN'
    assert contract['groups']==['110','111'] and contract['seed']==0 and contract['epochs_each']==24
    env=paired_environment(contract['launch_environment'])
    with (base/'queue_claim.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S')),f)
    statusfile=base/'status.json'
    def status(**v):
        p=statusfile.with_suffix('.tmp');p.write_text(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),**v),indent=2));p.replace(statusfile)
    def integrity():
        for p,h in contract['sources'].items():assert sha(ROOT/p)==h,('source drift',p)
        for g,p in contract['configurations'].items():assert sha(p)==contract['config_sha256'][g]
        assert sha(Path(contract['native_gate_path'])/'native_runtime_lock.json')==contract['runtime_lock_sha256']
    def command(group,stage,cmd):
        integrity();status(state='running',group=group,stage=stage,command=cmd)
        with (base/(group+'_'+stage+'.log')).open('x') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        reference_inputs={};reference_initial={}
        for group in ('110','111'):
            cfg=contract['configurations'][group];work=base/group
            cmd=[sys.executable,'-m','torch.distributed.launch','--nproc_per_node=2','--master_port=29680',
                 'tools/train.py',cfg,'--launcher','pytorch','--seed','0','--work-dir',str(work)]
            command(group,'train',cmd);assert (work/'epoch_24.pth').is_file()
            rows=[json.loads(s) for p in work.glob('*.log.json') for s in p.read_text().splitlines()]
            assert {r['epoch'] for r in rows if r.get('mode')=='train'}==set(range(1,25))
            assert {r['epoch'] for r in rows if r.get('mode')=='val'}==set(range(2,25,2))
            for rank in (0,1):
                initial=json.loads((work/('repro_initial_rank%d.jsonl'%rank)).read_text().splitlines()[0])['initial_state_sha256']
                if group=='110':reference_initial[rank]=initial
                else:assert initial==reference_initial[rank],('initial identity',group,rank)
                rr=[json.loads(s) for s in (work/('repro_inputs_rank%d.jsonl'%rank)).read_text().splitlines()]
                assert len(rr)==15360 and {r['iteration'] for r in rr}==set(range(15360))
                for r in rr:
                    key=rank,r['iteration']
                    if group=='110':reference_inputs[key]=r['combined_sha256']
                    else:assert r['combined_sha256']==reference_inputs[key],('input identity',key)
                conserved=[json.loads(s) for s in (work/('repro_conservation_rank%d.jsonl'%rank)).read_text().splitlines()]
                assert len(conserved)==24 and all(r['passed'] for r in conserved)
            assert len(set(reference_initial.values()))==1
        for group in ('110','111'):
            cfg=contract['configurations'][group];work=base/group
            args=[sys.executable,'crane_project/tools/ckpt_sweep.py','--config',cfg,'--work-dir',str(work),
                  '--gpu','3','--center-thresh','15','--mcml-limit','5']
            command(group,'VAL_select',args+['--epochs','16','18','20','22','24'])
            sweep=work/'ckpt_sweep/sweep_results.json';s=json.loads(sweep.read_text());assert len(s['all_checkpoints'])==5
            command(group,'TEST',args+['--final-test-from',str(sweep)])
        command('pair','analysis',[sys.executable,'crane_project/tools/analyze_symeood_c_pair_v1.py','--base',str(base)])
        integrity();status(state='complete',groups=['110','111'],formal_runs=2,epochs=48,depth=False,analysis='analysis_v1',automatic_adoption=False)
    except Exception as exc:
        status(state='stopped_on_error',error=repr(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--passed-gate',type=Path);p.add_argument('--launch-evidence',type=Path);a=p.parse_args()
    if a.prepare_only:
        assert a.passed_gate and a.launch_evidence
        c,_=freeze(a.base.resolve(),a.passed_gate,a.launch_evidence);print('FROZEN_TWO_RUNS_NO_FORMAL_START',c['groups'],c['seed'])
    else:run(a.base)
