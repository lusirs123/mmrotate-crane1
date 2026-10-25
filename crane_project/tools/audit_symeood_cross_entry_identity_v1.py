"""Read-only historical identities plus fresh CPU initialization audit.

No optimizer, no validation/test inference. Saves common native engineering
initial states for the separately bounded short-path experiment.
"""
import copy
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
PAIRS=[('100','iou','U'),('101','iou','G'),('110','kld','U'),('111','kld','G')]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(out):
    import torch,numpy as np
    from mmcv import Config
    from mmdet.models import build_detector
    from mmdet.apis import set_random_seed
    from mmcv.runner.checkpoint import get_torchvision_models
    from crane_project.utils.symeood_usg_v1 import tensor_digest
    out.mkdir(exist_ok=False,parents=True)
    oldman=[]
    for part in ('first4','last4'):
        oldman.append(json.loads((ROOT/'work_dirs/symeood_ablation_v1'/('launch_'+part+'_20261023')/'manifest.json').read_text()))
    usg=json.loads((ROOT/'work_dirs/symeood_usg_v1/protocol.json').read_text())
    common=[];changed=[]
    for name,h in oldman[0]['sha256'].items():
        if name not in usg['sources']:continue
        assert sha(ROOT/name)==usg['sources'][name],name
        (common if h==usg['sources'][name] else changed).append(name)
    records=[]
    for old,reg,obj in PAIRS+[('none','iou','S'),('none','kld','S')]:
        paths={}
        if old!='none':paths['old']=ROOT/'work_dirs/symeood_ablation_v1'/old/('crane_symeood_k1_ablation_'+old+'_v1.py')
        paths['new']=ROOT/'work_dirs/symeood_usg_v1'/('s0_'+reg+'_'+obj.lower())/('crane_symeood_usg_'+reg+'_'+obj.lower()+'_v1.py')
        for role,path in paths.items():
            cfg=Config.fromfile(str(path));set_random_seed(0,deterministic=True)
            model=build_detector(copy.deepcopy(cfg.model))
            ctor=tensor_digest(model.named_parameters());ctor_rng=sha_bytes(torch.get_rng_state().numpy().tobytes())
            names={id(t):n for n,t in model.named_parameters()};trace=[];originals={}
            for op in ('normal_','uniform_','constant_','kaiming_normal_','kaiming_uniform_','xavier_normal_','xavier_uniform_'):
                original=getattr(torch.nn.init,op);originals[op]=original
                def wrap(t,*args,_op=op,_fn=original,**kwargs):
                    trace.append(dict(op=_op,name=names.get(id(t),'unnamed'),shape=list(t.shape),
                        rng_before=sha_bytes(torch.get_rng_state().numpy().tobytes())))
                    return _fn(t,*args,**kwargs)
                setattr(torch.nn.init,op,wrap)
            try:model.init_weights()
            finally:
                for op,fn in originals.items():setattr(torch.nn.init,op,fn)
            head=model.bbox_head
            attrs=lambda x:{k:v for k,v in vars(x).items() if isinstance(v,(str,bool,int,float,type(None))) and not k.startswith('_')}
            classes={k:dict(module=type(x).__module__,qualname=type(x).__qualname__,file=inspect.getfile(type(x)),source_sha256=sha(inspect.getfile(type(x)))) for k,x in [('model',model),('head',head),('classification',head.loss_cls),('regression',head.loss_bbox),('assigner',head.assigner)]}
            name=(old if role=='old' else reg+'_'+obj.lower())
            record=dict(name=name,old_group=old,role=role,regression=reg,objective=obj,config=str(path),
                snapshot_sha256=sha(path),resolved_config=cfg.to_dict(),constructor_parameters_sha256=ctor,
                constructor_cpu_rng_sha256=ctor_rng,parameters_sha256=tensor_digest(model.named_parameters()),
                buffers_sha256=tensor_digest(model.named_buffers()),cpu_rng_sha256=sha_bytes(torch.get_rng_state().numpy().tobytes()),
                parameter_names=[n for n,_ in model.named_parameters()],classes=classes,
                assigner=attrs(head.assigner),classification=attrs(head.loss_cls),regression_loss=attrs(head.loss_bbox),
                init_trace=trace)
            records.append(record)
            if role=='old':torch.save(dict(state_dict=model.state_dict(),cpu_rng=torch.get_rng_state()),out/('initial_'+old+'.pth'))
            del model,head;gc.collect()
    result=dict(records=records,common_frozen_sources=common,changed_frozen_sources=changed,
                current_torchvision_resnet50_url=get_torchvision_models()['resnet50'],
                pretrained_cache=usg['pretrained_backbone_cache'],
                note='Fresh initialization and present config equivalence do not replace missing historical initialization/input traces')
    with (out/'identity.json').open('x') as f:json.dump(result,f,indent=2,default=str)
    for r in records:print(r['name'],r['role'],r['constructor_parameters_sha256'],r['parameters_sha256'],r['cpu_rng_sha256'],flush=True)


def sha_bytes(b):return hashlib.sha256(b).hexdigest()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out.resolve())
