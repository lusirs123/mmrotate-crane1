"""Native4step acceptance then freeze one environment for the110/111 pair.

Serial whole experiments, original GPU tensor residency, actual train.py.
No formal24epoch run is launched by this tool. Failure preserves all evidence.
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


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path):return [json.loads(s) for s in path.read_text().splitlines()]


def generate(out):
    from mmcv import Config
    from prepare_symeood_b111_canonical_v1 import B_CONFIG,ALIAS_CONFIG,diff
    out.mkdir(parents=True,exist_ok=False)
    assert diff(Config.fromfile(str(B_CONFIG)).to_dict(),Config.fromfile(str(ALIAS_CONFIG)).to_dict())==['.work_dir']
    configurations={}
    for label,base,trace,off in [('B',B_CONFIG,False,False),('111',ALIAS_CONFIG,False,False),('B_trace',B_CONFIG,True,False),('110',B_CONFIG,False,True),('110_trace',B_CONFIG,True,True)]:
        cfg=Config.fromfile(str(base));cfg.work_dir=str(out/label)
        if off:cfg.model.bbox_head.loss_cls.use_geometry_weight=False
        if trace:
            cfg.custom_imports.imports=list(cfg.custom_imports.imports)+['crane_project.utils.symeood_repro_trace_v1']
            cfg.custom_hooks=[dict(type='SymEOODReproTraceV1',priority='LOWEST',engineering=True)]
        p=out/(label+'.py');cfg.dump(str(p));configurations[label]=str(p)
    (out/'configurations.json').write_text(json.dumps(configurations,indent=2))
    return configurations


def main(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2,3'
    cfgs=generate(out);env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def status(**v):(out/'status.json').write_text(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),**v),indent=2))
    try:
        for i,(label,cfg) in enumerate(cfgs.items()):
            status(state='native_gate_running',case=label)
            cmd=[sys.executable,'-m','torch.distributed.launch','--nproc_per_node=2','--master_port='+str(29670+i),
                 'crane_project/tools/native_symeood_b111_entry_check_v1.py',cfg,'--execute-native-check',
                 '--launcher','pytorch','--seed','0','--deterministic','--no-validate','--work-dir',str(out/label),
                 '--cfg-options','runner.max_epochs=2']
            with (out/(label+'.log')).open('x') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        initial={};steps={};runtimes=[];checks=[]
        for rank in (0,1):
            for label in cfgs:
                work=out/label
                initial[label,rank]=load_rows(work/('native_initial_rank%d.jsonl'%rank))[0]
                steps[label,rank]=load_rows(work/('native_steps_rank%d.jsonl'%rank));assert len(steps[label,rank])==4
                assert all(r['nonzero_gradient_tensors']>0 for r in steps[label,rank])
                runtimes+=load_rows(work/('native_runtime_rank%d.jsonl'%rank))
            assert all(initial[label,rank]==initial['B',rank] for label in cfgs),('initial differs',rank)
            for a,b in [('B','111'),('B','B_trace'),('110','110_trace')]:
                equal=steps[a,rank]==steps[b,rank]
                # Memory peaks and allocator reservation are not mathematical
                # state and can differ when logging/saving happens.
                def functional(rr):return [{k:v for k,v in r.items() if not k.startswith('peak_')} for r in rr]
                equal=functional(steps[a,rank])==functional(steps[b,rank]);assert equal,(a,b,rank)
                checks.append(dict(reference=a,peer=b,rank=rank,steps=4,all_functional_fields_bitwise_equal=equal))
            for label in ('B_trace','110_trace'):
                rr=load_rows(out/label/('repro_conservation_rank%d.jsonl'%rank));assert len(rr)==2 and all(r['passed'] for r in rr)
        runtime=runtimes[0]
        for r in runtimes[1:]:
            for key in ('python','torch','mmcv','mmdet','torch_cuda','cudnn','flags','environment'):assert r[key]==runtime[key],('runtime differs',key)
        libraries={}
        for r in runtimes:
            for p,h in r['libraries'].items():assert p not in libraries or libraries[p]==h;libraries[p]=h
        runtime['libraries']=libraries
        from mmcv.runner.checkpoint import get_torchvision_models
        import torch,urllib.parse
        url=get_torchvision_models()['resnet50'];pretrained=Path(torch.hub.get_dir())/'checkpoints'/Path(urllib.parse.urlparse(url).path).name
        sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'mmrotate').rglob('*.py')}
        extras=list((ROOT/'crane_project/configs').glob('*.py'))+[ROOT/'tools/train.py',ROOT/'tools/test.py',ROOT/'tools/dist_train.sh',ROOT/'crane_project/utils/symeood_repro_trace_v1.py',ROOT/'crane_project/tools/native_symeood_b111_entry_check_v1.py',Path(__file__).resolve()]
        sources.update({str(p.relative_to(ROOT)):sha(p) for p in extras})
        lock=dict(state='NATIVE_GPU_ACCEPTED',runtime=runtime,
            formal_flags=dict(runtime['flags'],cudnn_deterministic=False,cudnn_benchmark=False),
            source_sha256=sources,pretrained=dict(path=str(pretrained),sha256=sha(pretrained)),
            limits='Four deterministic disposable updates/case, not24epoch repeatability; same production native GPU tensors, no offload/recompute/mixedprecision',
            native_checks=checks,physical_gpus=[2,3],formal_training_launched=False)
        (out/'native_runtime_lock.json').write_text(json.dumps(lock,indent=2))
        receipt=dict(passed=True,cases=5,checks=checks,hook_epoch_conservation_events=8,
            initial_parameters_sha256=initial['B',0]['parameters'],initial_state_sha256=initial['B',0]['state_sha256'],
            initial_parameters_equal_across_ranks=initial['B',0]['parameters']==initial['B',1]['parameters'],
            peak_allocated_mib=max(r['peak_allocated_mib'] for rr in steps.values() for r in rr),
            peak_reserved_mib=max(r['peak_reserved_mib'] for rr in steps.values() for r in rr),GPU_offload=False,formal_training=False)
        assert receipt['initial_parameters_equal_across_ranks']
        (out/'acceptance.json').write_text(json.dumps(receipt,indent=2));status(state='native_gate_complete',passed=True)
        print('NATIVE_GATE_PASSED',receipt,flush=True)
    except Exception as exc:
        status(state='stopped_on_error',error=repr(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out.resolve())
