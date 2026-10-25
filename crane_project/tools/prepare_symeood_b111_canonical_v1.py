"""CPU/Gloo-only preparation of current B/111 identity and canonical entry.

No detector forward/backward, optimizer update, VAL/TEST or GPU execution.
CPU preparation never authorizes formal training or native GPU acceptance.
"""
import argparse
import copy
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import random
import sys
import urllib.parse

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
B_CONFIG=ROOT/'crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py'
ALIAS_CONFIG=ROOT/'crane_project/configs/crane_symeood_k1_ablation_111_v1.py'


def diff(a,b,p=''):
    if isinstance(a,dict) and isinstance(b,dict):
        result=[]
        for k in sorted(set(a)|set(b)):
            if k not in a or k not in b:result.append(p+'.'+k)
            else:result.extend(diff(a[k],b[k],p+'.'+k))
        return result
    return [] if a==b else [p]


def cpu_rng():
    import numpy as np,torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())


def cpu_rng_digest():
    import numpy as np,torch
    n=np.random.get_state();h=hashlib.sha256(json.dumps([random.getstate(),[n[0],n[1].tolist(),n[2],n[3],n[4]]]).encode())
    h.update(torch.get_rng_state().numpy().tobytes());return h.hexdigest()


def main(out,rank):
    import torch,torch.distributed as dist
    from mmcv import Config
    from mmcv.runner.checkpoint import get_torchvision_models
    from mmdet.apis import set_random_seed
    from mmdet.models import build_detector
    from mmdet.datasets import build_dataset,build_dataloader
    from crane_project.utils.symeood_repro_trace_v1 import sha,tensor_digest,input_record,runtime
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    configs=[Config.fromfile(str(p)) for p in (B_CONFIG,ALIAS_CONFIG)]
    assert diff(configs[0].to_dict(),configs[1].to_dict())==['.work_dir']
    url=get_torchvision_models()['resnet50'];pretrained=Path(torch.hub.get_dir())/'checkpoints'/Path(urllib.parse.urlparse(url).path).name
    assert pretrained.is_file(),'No automatic pretrained download permitted'
    roles=[('B',configs[0]),('111',configs[1])]
    trace=copy.deepcopy(configs[0]);trace.custom_hooks=[dict(type='SymEOODReproTraceV1',priority='LOWEST',engineering=True)]
    trace.custom_imports.imports.append('crane_project.utils.symeood_repro_trace_v1')
    roles.append(('B_trace',trace))
    dataset=build_dataset(configs[0].data.train);assert len(dataset)==2558
    records=[];input_rows=[];firsts=[]
    for role,cfg in roles:
        set_random_seed(0,deterministic=True)
        model=build_detector(copy.deepcopy(cfg.model),train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
        ctor=tensor_digest(model.state_dict().items());ctor_rng=cpu_rng_digest();calls=[];originals={}
        for op in ('normal_','uniform_','constant_','kaiming_normal_','kaiming_uniform_','xavier_normal_','xavier_uniform_'):
            fn=getattr(torch.nn.init,op);originals[op]=fn
            def observe(t,*args,_op=op,_fn=fn,**kwargs):
                calls.append(dict(op=_op,shape=list(t.shape),rng=cpu_rng_digest()));return _fn(t,*args,**kwargs)
            setattr(torch.nn.init,op,observe)
        try:model.init_weights()
        finally:
            for op,fn in originals.items():setattr(torch.nn.init,op,fn)
        post_rng=cpu_rng();post_digest=cpu_rng_digest()
        state_sha=tensor_digest(model.state_dict().items())
        classes={name:dict(module=type(v).__module__,name=type(v).__qualname__,file=inspect.getfile(type(v)),sha256=sha(inspect.getfile(type(v)))) for name,v in [('detector',model),('head',model.bbox_head),('assigner',model.bbox_head.assigner),('classification',model.bbox_head.loss_cls),('regression',model.bbox_head.loss_bbox),('auxiliary',model.aux_heads[0]) ]}
        row=dict(role=role,rank=rank,constructor_state_sha256=ctor,constructor_rng_sha256=ctor_rng,
            complete_state_sha256=state_sha,parameter_sha256=tensor_digest(model.named_parameters()),buffer_sha256=tensor_digest(model.named_buffers()),
            post_initialization_rng_sha256=post_digest,init_calls=calls,classes=classes,
            parameter_names=[n for n,_ in model.named_parameters()],module_types=[(n,type(m).__module__,type(m).__qualname__) for n,m in model.named_modules()],
            all_state_devices=sorted(set(str(v.device) for v in model.state_dict().values())))
        records.append(row)
        if role=='B' and rank==0:
            torch.save(dict(state_dict=model.state_dict(),cpu_rng=post_rng,seed=0,classes=classes,pretrained_sha256=sha(pretrained)),out/'complete_initial_B_seed0.pth')
        assert post_digest==cpu_rng_digest(),'Saving/hashing changed CPU RNG'
        loader=build_dataloader(dataset,samples_per_gpu=2,workers_per_gpu=2,dist=True,shuffle=True,seed=0,
            runner_type='EpochBasedRunner',persistent_workers=False)
        for epoch in (0,1):
            loader.sampler.set_epoch(epoch);it=iter(loader)
            try:
                for inner in (0,1):
                    data=next(it);record=input_record(data)
                    input_rows.append(dict(role=role,rank=rank,sampler_epoch=epoch,step=epoch*2+inner,**record))
                    if role=='B' and epoch==0 and inner==0:
                        path=out/('first_augmented_batch_rank%d.pt'%rank)
                        torch.save({k:v.data[0] for k,v in data.items()},path)
                        firsts.append(dict(rank=rank,path=str(path),sha256=sha(path),input_sha256=record['combined_sha256']))
            finally:it._shutdown_workers()
        del it,loader,model;gc.collect()
    reference=records[0]
    for r in records[1:]:
        for key in ('constructor_state_sha256','constructor_rng_sha256','complete_state_sha256','parameter_sha256','buffer_sha256','post_initialization_rng_sha256','init_calls','classes','parameter_names','module_types','all_state_devices'):
            assert r[key]==reference[key],(r['role'],key)
    for step in range(4):
        rr=[r for r in input_rows if r['step']==step];assert len(rr)==3 and len({r['combined_sha256'] for r in rr})==1
    assert all(r['all_state_devices']==['cpu'] for r in records) and not torch.cuda.is_initialized()
    (out/('cpu_identity_rank%d.json'%rank)).write_text(json.dumps(dict(passed=True,records=records,first_augmented_batches=firsts),indent=2))
    (out/('cpu_inputs_rank%d.json'%rank)).write_text(json.dumps(input_rows,indent=2))
    if rank==0:
        old=json.loads((ROOT/'work_dirs/symeood_usg_v1/protocol.json').read_text())
        inventory=old['data_inventory'];assert len(inventory)==3445 and all(r['split'] in ('train','val') for r in inventory)
        for r in inventory:
            assert sha(ROOT/r['image'])==r['image_sha256'] and sha(ROOT/r['annotation'])==r['annotation_sha256']
        (out/'TRAIN_VAL_inventory.json').write_text(json.dumps(inventory,indent=2))
        finalize(out)
    dist.barrier()
    print('CPU_PREPARATION_PASS',rank,'roles3','inputs12','GPU_initialized',torch.cuda.is_initialized(),flush=True)


def finalize(out):
    """Finish metadata from preserved passing CPU receipts; no rerun needed."""
    import torch,mmcv
    from mmcv import Config
    from mmcv.runner.checkpoint import get_torchvision_models
    from crane_project.utils.symeood_repro_trace_v1 import sha,runtime
    configs=[Config.fromfile(str(p)) for p in (B_CONFIG,ALIAS_CONFIG)]
    assert diff(configs[0].to_dict(),configs[1].to_dict())==['.work_dir']
    for rank in (0,1):
        passed=json.loads((out/('cpu_identity_rank%d.json'%rank)).read_text());assert passed['passed']
        for rec in passed['records']:
            assert rec['all_state_devices']==['cpu']
            assert all(sha(v['file'])==v['sha256'] for v in rec['classes'].values())
    url=get_torchvision_models()['resnet50'];pretrained=Path(torch.hub.get_dir())/'checkpoints'/Path(urllib.parse.urlparse(url).path).name
    # Engineering flags differ deliberately from future historical-style runs.
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    env=runtime();candidate={}
    for folder in (Path(torch.__file__).parent/'lib',Path(mmcv.__file__).parent):
        for p in folder.glob('*.so*'):
            if any(k in p.name for k in ('cuda','cudnn','cublas','nccl','nvrtc','torch','c10','_ext')):candidate[str(p.resolve())]=sha(p)
    sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'mmrotate').rglob('*.py')}
    for p in [B_CONFIG,ALIAS_CONFIG,ROOT/'tools/train.py',ROOT/'tools/dist_train.sh',ROOT/'crane_project/utils/symeood_repro_trace_v1.py',Path(__file__).resolve()]:sources[str(p.relative_to(ROOT))]=sha(p)
    lock=dict(state='CPU_PREPARED_GPU_ACCEPTANCE_PENDING',runtime=env,
        formal_flags=dict(env['flags'],cudnn_deterministic=False,cudnn_benchmark=False),
        intended_physical_gpus=[2,3],candidate_library_sha256=candidate,source_sha256=sources,
        pretrained=dict(url=url,path=str(pretrained),sha256=sha(pretrained)),
        canonical_config=str(B_CONFIG),alias111_config=str(ALIAS_CONFIG),only_base_config_difference=['work_dir'],
        dataset_inventory_sha256=sha(out/'TRAIN_VAL_inventory.json'),complete_initial_file_sha256=sha(out/'complete_initial_B_seed0.pth'),
        formal_budget=dict(epochs=24,optimizer=configs[0].optimizer,lr=configs[0].lr_config,per_rank_batch=2,world_size=2,clip=configs[0].optimizer_config),
        formal_schedule_frozen=False,repetition_seeds=None,formal_training_launched=False,
        engineering_seed=0,engineering_cpu_deterministic=True,cpu_Gloo_for_data_sampling_only=True,
        pending=['Native train.py4step forward/backward/optimizer checks','Observed GPU-loaded libraries and ranks fingerprint','Repetition count/seeds and acceptance rules before formal schedule'],
        historical_environment_recovered=False,environment_selection_reason='Current installed environment chosen once without testing performance; no install/LD path edits')
    path=out/'runtime_lock_CPU_PENDING.json';assert not path.exists();path.write_text(json.dumps(lock,indent=2,default=str))
    (out/'CPU_complete.json').write_text(json.dumps(dict(passed=True,identity_roles_per_rank=3,ranks=2,input_batches_per_role_rank=4,
        source_class_files_still_match=True,GPU_initialized=torch.cuda.is_initialized(),native_GPU_checks_run=False,formal_training=False,
        finalize_only_reused_saved_passes=True),indent=2))
    assert not torch.cuda.is_initialized()
    print('CPU_METADATA_FINALIZED',len(candidate),'candidate libs','GPU_initialized',torch.cuda.is_initialized(),flush=True)


if __name__=='__main__':
    import torch,torch.distributed as dist
    from mmcv import Config
    from mmrotate.utils import setup_multi_processes
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--local_rank',type=int,default=0);p.add_argument('--finalize-only',action='store_true')
    a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    setup_multi_processes(Config.fromfile(str(B_CONFIG)))
    if a.finalize_only:finalize(a.out.resolve());sys.exit(0)
    dist.init_process_group('gloo',init_method='env://')
    out=a.out.resolve()
    if dist.get_rank()==0:out.mkdir(parents=True,exist_ok=False)
    dist.barrier();main(out,dist.get_rank());dist.destroy_process_group()
