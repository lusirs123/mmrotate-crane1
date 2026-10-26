"""Prepare/gate/execute exactly the frozen MAL then S_MAL experiment.

All model invocations are blocking subprocesses: no overlapping models.
Never automatically resumes or retries training. No deletion of result files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()


def dump(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def environment(gpus):
    # Reuse accepted native environment; device selection is explicitly new.
    old=json.loads((ROOT/'work_dirs/symeood_c_pair_seed0_20261025_v2/protocol.json').read_text())
    env=dict(os.environ,PYTHONPATH=str(ROOT))
    for k,v in old['launch_environment'].items():
        if v is None:env.pop(k,None)
        else:env[k]=v
    env['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))
    return env


def prepare(base,gpus):
    from mmcv import Config
    from crane_project.tools.ckpt_sweep import SELECTION_CONFIG
    base.mkdir(parents=True,exist_ok=False)
    (base/'configs').mkdir()
    design=ROOT/'crane_project/data_contracts/symeood_mal_strength_v1_20261026.json'
    contract=json.loads(design.read_text())
    configs={}
    for group,name in [('MAL','mal'),('S_MAL','smal')]:
        cfg=Config.fromfile(str(ROOT/('crane_project/configs/crane_symeood_%s_v1.py'%name)))
        assert cfg.runner.max_epochs==24 and cfg.load_from is None and cfg.resume_from is None
        assert cfg.optimizer.lr==.0025 and cfg.optimizer_config.grad_clip.max_norm==10
        assert cfg.data.samples_per_gpu==cfg.data.workers_per_gpu==2
        cfg.work_dir=str(base/group)
        path=base/'configs'/(group+'.py');cfg.dump(str(path));configs[group]=str(path)
    # Existing canonical inventory is TRAIN+VAL only. No TEST data selection.
    inventory=json.loads((ROOT/'work_dirs/symeood_b111_canonical_20261025_v2/TRAIN_VAL_inventory.json').read_text())
    assert len(inventory)==3445
    for r in inventory:
        assert sha(ROOT/r['image'])==r['image_sha256']
        assert sha(ROOT/r['annotation'])==r['annotation_sha256']
    paths=list((ROOT/'mmrotate').rglob('*.py'))
    paths+=list((ROOT/'configs').rglob('*.py'))
    paths+=list((ROOT/'crane_project/configs').glob('crane_symeood_k1*.py'))
    paths+=list((ROOT/'crane_project/configs').glob('crane_symeood_*mal*v1*.py'))
    paths+=list((ROOT/'crane_project').glob('**/*symeood_mal*v1*.py'))
    paths+=[ROOT/'crane_project/utils/symeood_usg_v1.py',ROOT/'crane_project/utils/symeood_repro_trace_v1.py',ROOT/'tools/train.py',ROOT/'tools/test.py',
        ROOT/'crane_project/tools/native_symeood_b111_entry_check_v1.py',
        ROOT/'crane_project/tools/audit_symeood_cross_entry_short_v1.py',
        ROOT/'crane_project/tools/ckpt_sweep.py',ROOT/'crane_project/tools/eval_crane_offline.py',
        ROOT/'crane_project/tools/analyze_symeood_c_pair_v1.py',design]
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    probes={p:sha(ROOT/p) for p in contract['probe_batches']}
    pretrained=Path('/home/omnisky/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth')
    assert sha(pretrained)=='0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a'
    contract.update(configurations=configs,config_sha256={k:sha(p) for k,p in configs.items()},
        sources=sources,data_inventory=inventory,probe_sha256=probes,selection=dict(SELECTION_CONFIG),
        physical_gpus=gpus,pretrained=dict(path=str(pretrained),sha256=sha(pretrained)),
        design_sha256=sha(design),created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    dump(base/'protocol.json',contract)
    print('PREPARED_NOT_TRAINING',base,flush=True)


def checked(base,contract,stage,cmd):
    for p,h in contract['sources'].items():assert sha(ROOT/p)==h,('source drift',p)
    for g,p in contract['configurations'].items():assert sha(p)==contract['config_sha256'][g]
    status=base/'status.json'
    status.write_text(json.dumps(dict(state='running',stage=stage,command=cmd,
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2))
    with (base/(stage+'.log')).open('x') as log:
        subprocess.run(cmd,cwd=ROOT,env=environment(contract['physical_gpus']),
                       stdout=log,stderr=subprocess.STDOUT,check=True)


def native_command(config,dest,resume=None):
    cmd=[sys.executable,'-m','torch.distributed.launch','--nproc_per_node=2','--master_port=29706',
        'crane_project/tools/native_symeood_b111_entry_check_v1.py','--execute-native-check',str(config),
        '--launcher','pytorch','--seed','0','--no-validate','--deterministic','--work-dir',str(dest),
        '--cfg-options','runner.max_epochs=2']
    if resume:cmd+=['--resume-from',str(resume)]
    return cmd


def gate(base,c):
    from mmcv import Config
    dest=base/'engineering';dest.mkdir(exist_ok=False)
    cases={}
    for name,source,objective,hook in [
        ('native_U',ROOT/'crane_project/configs/crane_symeood_k1_ablation_110_v1.py',None,False),
        ('adapted_U',Path(c['configurations']['MAL']),'U',False),
        ('MAL_plain',Path(c['configurations']['MAL']),'MAL',False),
        ('MAL_hook',Path(c['configurations']['MAL']),'MAL',True),
        ('S_MAL_hook',Path(c['configurations']['S_MAL']),'S_MAL',True)]:
        cfg=Config.fromfile(str(source))
        if objective:cfg.model.bbox_head.loss_cls.objective=objective
        if not hook:cfg.custom_hooks=[]
        cfg.checkpoint_config.interval=1
        p=dest/(name+'.py');cfg.dump(str(p))
        checked(base,c,'gate_'+name,native_command(p,dest/name))
        cases[name]=dest/name
    checks=[]
    keys=['input','rng_before','rng_after','predictions','labels','assignment','loss_parts',
          'logits_gradients','parameter_gradients_before_clip','optimizer_tensor_state',
          'state_sha256','parameters_sha256','buffers_sha256','counters']
    for a,b in [('native_U','adapted_U'),('MAL_plain','MAL_hook')]:
        for rank in (0,1):
            aa=[json.loads(s) for s in (cases[a]/('native_steps_rank%d.jsonl'%rank)).read_text().splitlines()]
            bb=[json.loads(s) for s in (cases[b]/('native_steps_rank%d.jsonl'%rank)).read_text().splitlines()]
            assert len(aa)==len(bb)==4
            for x,y in zip(aa,bb):
                for k in keys:assert x[k]==y[k],('gate divergence',a,b,rank,x['iteration'],k)
            checks.append(dict(reference=a,peer=b,rank=rank,steps=4,fields=keys,passed=True))
    # Verify real epoch-boundary recovery, not only state serialization.
    for group in ('MAL_hook','S_MAL_hook'):
        resume=cases[group]/'recovery_epoch_1.pth'
        checked(base,c,'gate_'+group+'_recovery',native_command(dest/(group+'.py'),dest/(group+'_recovery'),resume))
        for rank in (0,1):
            full=[json.loads(s) for s in (cases[group]/('native_steps_rank%d.jsonl'%rank)).read_text().splitlines()][2:]
            restored=[json.loads(s) for s in (dest/(group+'_recovery')/('native_steps_rank%d.jsonl'%rank)).read_text().splitlines()]
            assert len(full)==len(restored)==2
            for x,y in zip(full,restored):
                for k in keys:assert x[k]==y[k],('recovery divergence',group,rank,k)
            checks.append(dict(recovery=group,rank=rank,steps=2,passed=True))
    # Deterministic TRAIN engineering and formal nondeterministic runtime differ
    # only in the already historical deterministic flag; both explicitly bound.
    dump(base/'engineering_acceptance.json',dict(passed=True,checks=checks,
        no_cpu_offload=True,no_recomputation=True,formal_training=False,
        cases={k:str(v) for k,v in cases.items()}))
    print('NATIVE_GATE_PASSED',flush=True)


def run(base,c):
    assert json.loads((base/'engineering_acceptance.json').read_text())['passed']
    dump(base/'queue_claim.json',dict(pid=os.getpid(),epochs_cap=48,order=c['groups']))
    reference_initial={};reference_inputs={}
    for group in c['groups']:
        work=base/group;assert not work.exists(),('Never overwrite/retry',group)
        checked(base,c,group+'_train',[sys.executable,'-m','torch.distributed.launch','--nproc_per_node=2',
            '--master_port=29708','tools/train.py',c['configurations'][group],
            '--launcher','pytorch','--seed','0','--work-dir',str(work)])
        assert (work/'epoch_24.pth').is_file() and (work/'recovery_epoch_24.pth').is_file()
        logs=[json.loads(s) for p in work.glob('*.log.json') for s in p.read_text().splitlines()]
        assert {r['epoch'] for r in logs if r.get('mode')=='val'}==set(c['online_VAL_epochs'])
        for rank in (0,1):
            init=json.loads((work/('initial_state_rank%d.jsonl'%rank)).read_text().splitlines()[0])['full_state_sha256']
            if group=='MAL':reference_initial[rank]=init
            else:assert reference_initial[rank]==init,('initial mismatch',rank)
            rows=[json.loads(s) for s in (work/('inputs_rank%d.jsonl'%rank)).read_text().splitlines()]
            assert len(rows)==24
            for r in rows:
                assert r['batches']==640
                k=rank,r['epoch']
                if group=='MAL':reference_inputs[k]=r['sha256']
                else:assert reference_inputs[k]==r['sha256'],('input mismatch',k)
            probe=[json.loads(s) for s in (work/('probe_conservation_rank%d.jsonl'%rank)).read_text().splitlines()]
            assert len(probe)==3 and all(r['passed'] for r in probe)
        assert len(set(reference_initial.values()))==1
    # Both train fully before either group is selected/tested. Serial inference.
    for group in c['groups']:
        checked(base,c,group+'_VAL_select',[sys.executable,'crane_project/tools/ckpt_sweep.py',
            '--config',c['configurations'][group],'--work-dir',str(base/group),
            '--gpu',str(c['physical_gpus'][-1]),'--epochs','16','18','20','22','24',
            '--center-thresh','15','--mcml-limit','5'])
    selected={g:sha(base/g/'ckpt_sweep/sweep_results.json') for g in c['groups']}
    dump(base/'VAL_selection_frozen.json',selected)
    for group in c['groups']:
        sweep=base/group/'ckpt_sweep/sweep_results.json';assert sha(sweep)==selected[group]
        checked(base,c,group+'_TEST',[sys.executable,'crane_project/tools/ckpt_sweep.py',
            '--config',c['configurations'][group],'--work-dir',str(base/group),
            '--gpu',str(c['physical_gpus'][-1]),'--center-thresh','15','--mcml-limit','5',
            '--final-test-from',str(sweep)])
    checked(base,c,'analysis',[sys.executable,'crane_project/tools/analyze_symeood_mal_pair_v1.py','--base',str(base),'--require-weights'])
    (base/'status.json').write_text(json.dumps(dict(state='complete',epochs=48,groups=c['groups'],depth=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','gate','run'])
    p.add_argument('--base',type=Path,required=True);p.add_argument('--gpus',type=int,nargs=2)
    a=p.parse_args();base=a.base.resolve();os.chdir(ROOT)
    try:
        if a.stage=='prepare':
            assert a.gpus and len(set(a.gpus))==2
            prepare(base,a.gpus)
        else:
            c=json.loads((base/'protocol.json').read_text())
            gate(base,c) if a.stage=='gate' else run(base,c)
    except Exception as exc:
        if base.exists():
            (base/'status.json').write_text(json.dumps(dict(state='stopped_on_error',error=repr(exc)),indent=2))
        raise
