"""Read existing gradients and probe frozen E+B stages; no optimizer or training."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
REFERENCE='work_dirs/eood_AB_diagnostic_20261031_v5/run'
CONFIG='work_dirs/eood_B_regression_20261031_v3/candidate_config.py'
CASES=(('ep4_MaxIoU',4,3,'fef3ebe49b0b8498c26c2d59cd34e3a887be574097d23cfd3e1a1048ca218a2b'),
       ('ep4_POLA',4,4,'fef3ebe49b0b8498c26c2d59cd34e3a887be574097d23cfd3e1a1048ca218a2b'),
       ('ep16_POLA',16,15,'9d0c38c925203ffe7cea08ea7a6f74107a6c8e20eeb6f65d823996baaf966a9d'),
       ('ep24_POLA',24,23,'f3c25c3e15742497fb692ec3edfb8a5e5c53af68bf0e1b8ee9934b5dcecf9d21'))
ALPHAS=(0.,.25,1.)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def mixture(norm_iou,norm_kld,cosine,alpha):
    if norm_iou<=0 or norm_kld<0 or not -1.000001<=cosine<=1.000001 or alpha<0:
        raise ValueError('Invalid norm, cosine or mixture weight')
    cosine=max(-1.,min(1.,cosine))
    ratio=norm_kld/norm_iou
    factor=math.sqrt(max(0.,1+2*alpha*ratio*cosine+(alpha*ratio)**2))
    if factor<=0:raise ValueError('Cancelling regression gradient')
    return dict(norm_over_DIoU=factor,cosine_to_DIoU=(1+alpha*ratio*cosine)/factor,
                geometry_norm_over_DIoU=alpha*ratio)


def describe(values):
    assert values and all(math.isfinite(v) for v in values)
    return dict(n=len(values),median=statistics.median(values),mean=statistics.mean(values),
                minimum=min(values),maximum=max(values))


def offline(out):
    base=ROOT/REFERENCE
    summary=json.loads((base/'summary.json').read_text())
    assert summary['passed'] and summary['training_updates']==0
    rows=[json.loads(x) for x in (base/'result/network_gradients.jsonl').read_text().splitlines()]
    assert len(rows)==192
    groups=[]
    for arm in ('E','EA','EB'):
        for scale in (1.,.5):
            group=[r for r in rows if r['arm']==arm and r['scale']==scale]
            for alpha in ALPHAS:
                vals=[];aux=[]
                for r in group:
                    g=r['fpn_gradient_l2'];n0=g['counterfactual_DIoU'];n1=g['counterfactual_SymKLD']
                    value=mixture(n0,n1,r['counterfactual_fpn_cosine'],alpha)
                    vals.append(value);aux.append(g['auxiliary']/(n0*value['norm_over_DIoU']))
                groups.append(dict(arm=arm,scale=scale,alpha=alpha,
                    stats={k:describe([v[k] for v in vals]) for k in vals[0]},
                    auxiliary_over_mixed_reg=describe(aux)))
    result=dict(passed=True,training_updates=0,model_forwards=0,TEST=False,VAL=False,
        formula='L=2*(linear_rotated_DIoU + alpha*min(sqrt(1+d_sym)-1,10))',
        alpha_role='0 identity, .25 historical B2 coefficient, 1 unit-coefficient diagnostic; not validation selection.',
        note='Norms reconstructed exactly up to saved FP rounding from ||g0||, ||g1|| and cosine; not a new model performance result.',
        groups=groups,input_sha256=sha(base/'result/network_gradients.jsonl'),source_sha256=sha(__file__))
    save(out/'existing_gradient_audit.json',result)


def gpu_probe(out,gpu):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpu)
    assert os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import collate,scatter
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from mmrotate.datasets import build_dataset
    from mmrotate.models.losses.rotated_iou_loss import RotatedIoULoss
    from mmrotate.models.losses.eood_sym_kld_positive_loss_v1 import EOODSymKLDPositiveLossV1
    from mmrotate.models.losses.sym_iou_hybrid_loss_v1 import SymIoUHybridLossV1
    from mmrotate.models.losses.sym_kld_calculator import sym_kld
    from crane_project.tools.diagnose_eood_AB_frozen_v2 import state_sha,telemetry,enforce_resources
    from crane_project.tools.preflight_port_center_size_v1 import fixed_train_specs
    torch.set_num_threads(1);enforce_resources(telemetry(gpu),0,before=True)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(6144*2**20/torch.cuda.get_device_properties(0).total_memory,0)
    def forbidden(*a,**kw):raise AssertionError('Optimizer construction forbidden')
    torch.optim.Optimizer.__init__=forbidden
    iou=RotatedIoULoss(mode='linear',diou=True,loss_weight=2.)
    kld=EOODSymKLDPositiveLossV1(loss_weight=2.,strict_finite=True)
    hybrids={a:SymIoUHybridLossV1(geometry_weight=a) for a in ALPHAS}
    # Test the existing hybrid with the actual dense interface contract.
    p=torch.tensor([[22.,18.,16.,8.,.17],[48.,54.,20.,9.,-.12],[90.,80.,4.,3.,0.]],device='cuda',requires_grad=True)
    t=torch.tensor([[20.,20.,15.,7.,.11],[50.,52.,19.,10.,-.18],[0.,0.,0.,0.,0.]],device='cuda')
    w=torch.tensor([[1.]*5,[.5]*5,[0.]*5],device='cuda')
    checks=[]
    for alpha,loss in hybrids.items():
        value=loss(p,t,w,avg_factor=2.)
        expected=iou(p[:2],t[:2],w[:2],avg_factor=2.)+alpha*kld(p[:2],t[:2],w[:2],avg_factor=2.)
        g=torch.autograd.grad(value,p,retain_graph=True)[0]
        ge=torch.autograd.grad(expected,p,retain_graph=True)[0]
        assert torch.isfinite(g).all() and torch.allclose(value,expected,atol=1e-6,rtol=1e-5)
        assert torch.allclose(g,ge,atol=1e-6,rtol=1e-5) and g[2].abs().sum()==0
        checks.append(dict(alpha=alpha,dense_value_and_gradient_match_positive_only=True,negative_gradient_zero=True))
    save(out/'hybrid_interface_checks.json',dict(passed=True,checks=checks,parameters=sum(p.numel() for p in hybrids[.25].parameters())))
    # Local shape near the optimum: these are synthetic geometry checks, not data metrics.
    synthetic=[]
    target=torch.tensor([[128.,128.,80.,40.,.2]],device='cuda')
    for kind in ('center','size','angle'):
        for delta in (.001,.01,.1):
            pred=target.clone()
            if kind=='center':pred[0,0]+=delta*80
            elif kind=='size':pred[0,2]*=math.exp(delta)
            else:pred[0,4]+=delta
            pred.requires_grad_(True)
            vals={name:loss(pred,target,avg_factor=1.) for name,loss in (('DIoU',iou),('SymKLD',kld))}
            norms={}
            for name,v in vals.items():
                grad=torch.autograd.grad(v,pred,retain_graph=True)[0]
                assert torch.isfinite(grad).all()
                norms[name]=float((grad*torch.tensor([[80.,40.,float(pred[0,2]),float(pred[0,3]),1.]],device='cuda')).norm())
            synthetic.append(dict(kind=kind,normalized_error=delta,losses={k:float(v.detach()) for k,v in vals.items()},gradient_norms=norms))
    save(out/'synthetic_local_shape.json',dict(passed=True,rows=synthetic))
    cfg=Config.fromfile(str(ROOT/CONFIG));import_modules_from_strings(**cfg.custom_imports)
    old=json.loads((ROOT/REFERENCE/'result/completion.json').read_text())
    assert old['arms']['EB']['config_sha256']==sha(ROOT/CONFIG)
    real=np.linspace(0,1809,32).round().astype(int).tolist()
    sim=(1810+np.linspace(0,747,32).round().astype(int)).tolist()
    pairs=[(real[i],sim[i]) for i in (0,8,16,24)]
    inventory=json.loads((ROOT/'work_dirs/symeood_110_retrain_seed0_20261026_v1/protocol.json').read_text())['data_inventory']
    inventory={str((ROOT/r['image']).resolve()):r for r in inventory if r['split']=='train'}
    inputs={};receipts=[];count=0
    for label,epoch,phase,digest in CASES:
        checkpoint=ROOT/('work_dirs/eood_B_regression_20261031_v3/train/epoch_%d.pth'%epoch)
        assert sha(checkpoint)==digest
        mc=deepcopy(cfg.model);mc.backbone.init_cfg=None
        model=build_detector(mc);load_checkpoint(model,str(checkpoint),map_location='cpu',strict=True)
        model.cuda().train();model.set_epoch(phase);torch.cuda.reset_peak_memory_stats()
        before=state_sha(model);main=model.bbox_head.predictors[0]
        for scale in (1.,.5):
            dataset=build_dataset(fixed_train_specs(cfg,scale))
            for pair in pairs:
                enforce_resources(telemetry(gpu),0)
                batch=scatter(collate([dataset[i] for i in pair],samples_per_gpu=2),[0])[0]
                for bi,ix in enumerate(pair):
                    info=inventory[str(Path(batch['img_metas'][bi]['filename']).resolve())]
                    for kind in ('image','annotation'):assert sha(ROOT/info[kind])==info[kind+'_sha256']
                ih=hashlib.sha256(batch['img'].cpu().numpy().tobytes()+b''.join(g.cpu().numpy().tobytes() for g in batch['gt_bboxes'])).hexdigest()
                key=str((pair,scale))
                if label==CASES[0][0]:inputs[key]=ih
                else:assert inputs[key]==ih
                buffers={k:v.detach().clone() for k,v in model.named_buffers()};captured=[]
                original=main.loss_iou.forward
                def observe(pred,target,weight=None,avg_factor=None,**kwargs):
                    positive=(weight[:,0] if weight.ndim==2 else weight)>0
                    if positive.any():captured.append((pred[positive],target[positive],avg_factor))
                    return original(pred,target,weight,avg_factor,**kwargs)
                main.loss_iou.forward=observe
                try:
                    model.zero_grad(set_to_none=True)
                    native=model(return_loss=True,**batch)
                    scalar={k:sum(v) if isinstance(v,(tuple,list)) else v for k,v in native.items() if 'loss' in k}
                    assert all(torch.isfinite(v).all() for v in scalar.values()) and captured
                    normalizers={float(x[2]) for x in captured};assert len(normalizers)==1
                    normalizer=normalizers.pop()
                    pred=torch.cat([x[0] for x in captured]);target=torch.cat([x[1] for x in captured])
                    v0=iou(pred,target,avg_factor=normalizer);v1=kld(pred,target,avg_factor=normalizer)
                    assert torch.allclose(v1,scalar['loss_0_loss_iou'],atol=1e-6,rtol=1e-5)
                    params=[p for p in model.neck.parameters() if p.requires_grad]
                    def vector(loss):
                        gg=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                        assert all(g is None or torch.isfinite(g).all() for g in gg)
                        return torch.cat([g.detach().reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1) for g,p in zip(gg,params)])
                    g0=vector(v0);g1=vector(v1)
                    aux=sum(v for k,v in scalar.items() if k.startswith('loss_1_'))
                    ga=vector(aux);gc=vector(scalar['loss_0_loss_cls'])
                    mixed=[]
                    for alpha in ALPHAS:
                        actual=hybrids[alpha](pred,target,avg_factor=normalizer)
                        gm=vector(actual);expected=g0+alpha*g1
                        assert torch.allclose(actual,v0+alpha*v1,atol=1e-6,rtol=1e-5)
                        assert torch.allclose(gm,expected,atol=2e-5,rtol=2e-4)
                        n0=float(g0.norm());n1=float(g1.norm());nm=float(gm.norm())
                        cosine=float(torch.nn.functional.cosine_similarity(g0,g1,dim=0))
                        predicted=mixture(n0,n1,cosine,alpha)
                        assert abs(nm/n0-predicted['norm_over_DIoU'])<2e-4
                        mixed.append(dict(alpha=alpha,norm_over_DIoU=nm/n0,
                            cosine_to_DIoU=float(torch.nn.functional.cosine_similarity(g0,gm,dim=0)),
                            auxiliary_over_reg=float(ga.norm())/nm,direct_gradient_matches_linear_sum=True))
                    d=sym_kld(pred.detach(),target.detach())
                    row=dict(case=label,checkpoint_epoch=epoch,replay_epoch_index=phase,
                        assignment='MaxIoU' if phase<4 else 'POLA',pair=list(pair),scale=scale,input_sha256=ih,
                        positives=len(pred),normalizer=normalizer,raw_kld_median=float(d.median()),
                        saturated_positive_count=int((d>120).sum()),loss_DIoU=float(v0.detach()),loss_SymKLD=float(v1.detach()),
                        FPN_KLD_over_DIoU=float(g1.norm()/g0.norm()),
                        FPN_KLD_DIoU_cosine=float(torch.nn.functional.cosine_similarity(g0,g1,dim=0)),
                        auxiliary_over_native_reg=float(ga.norm()/g1.norm()),
                        classification_over_native_reg=float(gc.norm()/g1.norm()),mixed=mixed,optimizer_updates=0)
                    del g0,g1,ga,gc,gm,expected
                    sum(scalar.values()).backward()
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                    with (out/'stage_rows.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                finally:
                    main.loss_iou.forward=original;model.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        for k,v in model.named_buffers():v.copy_(buffers[k])
                count+=1;print(label,scale,pair,'native backward, zero updates',flush=True)
                del batch,native,scalar,captured,pred,target,v0,v1,actual,aux,params,buffers,row
        after=state_sha(model);assert before==after
        receipt=dict(case=label,checkpoint_sha256=digest,before_state=before,after_state=after,
            state_unchanged=True,batches=8,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        receipts.append(receipt);save(out/(label+'_receipt.json'),receipt)
        del model;torch.cuda.empty_cache()
    assert count==32
    save(out/'completion.json',dict(passed=True,batches=count,views=64,training_updates=0,
        optimizer_constructed=False,TEST=False,VAL=False,receipts=receipts,
        checkpoint_note='Frozen snapshots replayed at stated assignment phases; no reconstruction of historical training batches.',
        config_sha256=sha(ROOT/CONFIG),source_sha256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=('offline','gpu'),required=True)
    p.add_argument('--gpu',type=int,choices=(0,1),default=1)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    if a.stage=='offline':offline(a.out)
    else:gpu_probe(a.out,a.gpu)
