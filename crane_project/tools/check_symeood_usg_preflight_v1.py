"""Complete native/DDP U/S/G engineering gate. No formal training here."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'work_dirs/symeood_usg_v1'


def nested_equal(a,b):
    import torch
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(nested_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    import torch
    os.chdir(ROOT)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='2,3';env['PYTHONPATH']=str(ROOT)
    result=[]
    for reg in ('iou','kld'):
        for obj in ('u','s','g'):
            paths={kind:BASE/('preflight_v3_%s_%s_%s'%(reg,obj,kind)) for kind in ('continuous','resumed')}
            for kind,dest in paths.items():
                if dest.exists():
                    raise RuntimeError('Refusing existing preflight directory: '+str(dest))
                cmd=[sys.executable,'-m','torch.distributed.launch','--nproc_per_node=2','--master_port=29642',
                     'crane_project/tools/preflight_symeood_usg_v1.py',
                     'crane_project/configs/crane_symeood_usg_%s_%s_v1.py'%(reg,obj),
                     '--launcher','pytorch','--seed','0','--deterministic','--no-validate','--work-dir',str(dest),
                     '--cfg-options','runner.max_epochs=2']
                if kind=='resumed':cmd+=['--resume-from',str(paths['continuous']/'recovery_epoch_1.pth')]
                with (BASE/(dest.name+'.log')).open('x') as log:
                    subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            a=torch.load(paths['continuous']/'recovery_epoch_2.pth',map_location='cpu')
            b=torch.load(paths['resumed']/'recovery_epoch_2.pth',map_location='cpu')
            model_equal=nested_equal(a['state_dict'],b['state_dict'])
            optimizer_equal=nested_equal(a['optimizer'],b['optimizer'])
            assert model_equal and optimizer_equal, ('Recovery trajectory differs',reg,obj)
            for rank in (0,1):
                trace=[];inputs=[]
                for dest in paths.values():
                    rs=[json.loads(s) for s in (dest/('preflight_trace_rank%d.jsonl'%rank)).read_text().splitlines()]
                    trace.append([r for r in rs if r['epoch']==2])
                    rs=[json.loads(s) for s in (dest/('inputs_rank%d.jsonl'%rank)).read_text().splitlines()]
                    inputs.append([r for r in rs if r['epoch']==2])
                assert trace[0]==trace[1], ('Next epoch loss/updates differs',reg,obj,rank)
                assert inputs[0]==inputs[1], ('Next epoch data differs',reg,obj,rank)
                x=a['meta']['usg_recovery']['ranks'][rank];y=b['meta']['usg_recovery']['ranks'][rank]
                assert x['assigner_calls']==y['assigner_calls']==8
                assert x['loss_calls']==y['loss_calls']==20
            result.append(dict(regression=reg,objective=obj,model_bitwise=True,optimizer_bitwise=True,
                               next_epoch_input_and_trace_bitwise=True,world_size=2,steps=4))
            del a,b
            print('DDP_RECOVERY_PASS',reg,obj,flush=True)
    receipt=dict(passed=True,cases=result,physical_gpus=[2,3],
                 deterministic_engineering_only=True,
                 formal_bitwise_resume_claim=False,
                 scope='two mini-epochs, actual train.py/DDP/SGD/probes; discarded engineering updates; no VAL/TEST')
    with (BASE/'ddp_preflight.json').open('x') as f:json.dump(receipt,f,indent=2)


if __name__=='__main__':main()
