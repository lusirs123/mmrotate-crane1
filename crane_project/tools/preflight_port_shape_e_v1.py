#!/usr/bin/env python3
"""Bounded E0/E-H mechanism + frozen B TRAIN probe; no optimizer, VAL or TEST.

E-H is measured at UNIT weight. A gradient-matching coefficient is descriptive,
not a frozen experiment or an automatically approved training configuration.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.diagnose_port_center_size_d_v1 import (
    sha, gradient_vector, grad_norm, grad_cos, parameter_digest,
    checkpoint_contract, ordered_data_infos, positive_geometry)
from crane_project.tools.preflight_port_center_size_v1 import (
    CONTROL, check_configs, fixed_train_specs, set_assignment_phase, sum_loss)


@contextmanager
def capture_main_positives(head):
    """Capture the actual KLD graph nodes, never sibling D slices."""
    captured = []
    original = head.loss_bbox.forward
    def observe(pred, target, weight=None, avg_factor=None, **kwargs):
        value = original(pred, target, weight=weight, avg_factor=avg_factor, **kwargs)
        captured.append((pred, target, weight, avg_factor, value))
        return value
    head.loss_bbox.forward = observe
    try:
        yield captured
    finally:
        head.loss_bbox.forward = original


def shape_losses(captured):
    from mmrotate.models.losses.covariance_shape_loss import CovarianceShapeLoss
    e0 = CovarianceShapeLoss(mode='symkl', loss_weight=.25)
    eh = CovarianceShapeLoss(mode='hellinger', loss_weight=1.)
    if not captured:
        raise RuntimeError('No main loss nodes captured')
    return {name:sum(loss(p, t, weight=w, avg_factor=n) for p,t,w,n,_ in captured)
            for name,loss in [('e0',e0),('eh_unit',eh)]}


def inspect_gradients(terms, groups, captured):
    """Same B positives/normalizer for all terms; mixed-domain parameter gradients."""
    import torch
    params = [p for _,ps in groups for p in ps]
    vectors = {name:gradient_vector(loss,params) for name,loss in terms.items()}
    report, start = {}, 0
    for name,ps in groups:
        sl = slice(start,start+len(ps)); start += len(ps)
        norms = {k:grad_norm(v[sl]) for k,v in vectors.items()}
        base = norms['symkld']
        report[name] = dict(norms=norms,
            ratios_to_symkld={k:norms[k]/base if base else None for k in ('e0','eh_unit')},
            cosines_to_symkld={k:grad_cos(vectors[k][sl],vectors['symkld'][sl])
                               for k in ('e0','eh_unit')},
            e0_matching_eh_coefficient=norms['e0']/norms['eh_unit'] if norms['eh_unit'] else None)
    direct = {}
    nodes = [p for p,t,w,n,v in captured]
    for name in ('symkld','e0','eh_unit'):
        gs = gradient_vector(terms[name],nodes)
        if any(g is None and len(p)>0 for g,p in zip(gs,nodes)):
            raise RuntimeError('Missing actual positive-node dependency: '+name)
        if name != 'symkld' and any(g is not None and torch.count_nonzero(g[:,:2]) for g in gs):
            raise RuntimeError('Unexpected extra centre gradient')
        direct[name] = dict(node_scope='actual_main_symkld_positive_tensors',
            nonempty_levels=sum(len(p)>0 for p in nodes),
            connected_nonempty_levels=sum(g is not None and len(p)>0 for g,p in zip(gs,nodes)),
            coordinate_norms={key:grad_norm([g[:,j:j+1] if g is not None else None for g in gs])
                              for j,key in enumerate(('x','y','w','h','angle'))})
    # GT detach is an implementation contract checked separately in unit tests.
    if report['classification_conv']['norms']['e0'] or report['classification_conv']['norms']['eh_unit']:
        raise RuntimeError('Unexpected direct classification-convolution gradient')
    return dict(parameters=report, direct=direct)


def reference_riou(first, second):
    """Independent float64 convex clipping for MECHANISM checks, not an evaluator change.

    OpenCV's float32 intersection can fail on shared/near-collinear edges.
    Local coordinates reduce cancellation; CCW half-plane clipping handles containment.
    """
    import numpy as np
    first,second = (np.asarray(x,dtype=np.float64).reshape(-1)[:5] for x in (first,second))
    if any(len(x)!=5 or not np.isfinite(x).all() or (x[2:4]<=0).any() for x in (first,second)):
        raise ValueError('Expected finite positive OBBs')
    origin = (first[:2]+second[:2])*.5
    def vertices(x):
        c,s = math.cos(x[4]),math.sin(x[4])
        local = np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])*x[2:4]*.5
        return local@np.array([[c,s],[-s,c]])+x[:2]-origin
    def cross(a,b):return a[0]*b[1]-a[1]*b[0]
    poly,clip = list(vertices(first)),vertices(second)
    tolerance = 1e-12*max(1.,float(max(first[2:4].max(),second[2:4].max()))**2)
    for i,a in enumerate(clip):
        edge = clip[(i+1)%4]-a
        old,poly = poly,[]
        if not old:break
        start = old[-1];ds = cross(edge,start-a)
        for end in old:
            de = cross(edge,end-a)
            inside_s,inside_e = ds>=-tolerance,de>=-tolerance
            if inside_s!=inside_e:
                denominator = ds-de
                if denominator!=0:poly.append(start+(end-start)*(ds/denominator))
            if inside_e:poly.append(end)
            start,ds = end,de
    area = (abs(sum(cross(poly[i],poly[(i+1)%len(poly)]) for i in range(len(poly))))*.5
            if len(poly)>=3 else 0.)
    a1,a2 = first[2]*first[3],second[2]*second[3]
    area = min(max(area,0.),a1,a2)
    return float(area/(a1+a2-area))


def geometry_check():
    """Analytic cases only: no dataset, predictions or checkpoint required."""
    import torch
    from mmrotate.models.losses.covariance_shape_loss import CovarianceShapeLoss
    rows = []
    for aspect in (1.,2.3,3.):
        target = torch.tensor([[120.,100.,40.*aspect,40.,.2]],dtype=torch.double)
        for case in ('match','scale_small','scale_large','long_small','short_small',
                     'angle_2deg','angle_5deg','joint','large_error','centre_only'):
            pred = target.clone()
            if case == 'scale_small':pred[:,2:4] *= .95
            if case == 'scale_large':pred[:,2:4] /= .95
            if case == 'long_small':pred[:,2] *= .95
            if case == 'short_small':pred[:,3] *= .95
            if case.startswith('angle_'):pred[:,4] += math.radians(2 if case=='angle_2deg' else 5)
            if case == 'joint':pred[:,2:4] *= .95; pred[:,4] += math.radians(3)
            if case == 'large_error':pred[:,2] *= .2; pred[:,3] *= 3.; pred[:,4] += .6
            if case == 'centre_only':pred[:,:2] += 50.
            pred.requires_grad_()
            riou = reference_riou(pred.detach().numpy()[0],target.numpy()[0])
            for mode,weight in [('symkl',.25),('hellinger',1.)]:
                loss = CovarianceShapeLoss(mode=mode,loss_weight=weight)(pred,target)
                g = torch.autograd.grad(loss,pred)[0]
                if not torch.isfinite(loss) or not torch.isfinite(g).all():
                    raise RuntimeError('Nonfinite geometry case')
                if torch.count_nonzero(g[:,:2]):raise RuntimeError('Extra centre gradient')
                # One tiny step in dimensionless log-edge / radian coordinates.
                # This tests a direction, not a detector or a training schedule.
                direction = torch.stack([g[0,2]*pred[0,2],g[0,3]*pred[0,3],g[0,4]]).detach()
                updated = pred.detach().clone()
                step = 1e-5
                if direction.norm() > 1e-12:
                    direction /= direction.norm()
                    updated[0,2:4] *= torch.exp(-step*direction[:2])
                    updated[0,4] -= step*direction[2]
                new_loss = CovarianceShapeLoss(mode=mode,loss_weight=weight)(updated,target)
                new_riou = reference_riou(updated.numpy()[0],target.numpy()[0])
                rows.append(dict(aspect=aspect,case=case,candidate='e0' if mode=='symkl' else 'eh_unit',
                    loss=float(loss),gradient=g.detach().tolist()[0],riou=riou,
                    dimensionless_descent_loss_delta=float(new_loss-loss.detach()),
                    dimensionless_descent_riou_delta=new_riou-riou))
    return dict(rows=rows,eps=1e-6,internal_precision='float64',
        riou_method='independent_float64_convex_clipping_synthetic_only',
        note='Synthetic directions; not proof of metric alignment everywhere or generalization. '
             'Square angle blindness and centre-only blindness are expected for shape-only Gaussians.')


def frozen_b_identity(sweep):
    selection_path = Path(sweep)/'sweep_results.json'
    s = json.loads(selection_path.read_text())
    if (s.get('evidence_role')!='source_val_checkpoint_selection'
            or s['selected_checkpoint']!='epoch_24' or s['config_sha256']!=sha(CONTROL)):
        raise ValueError('Expected frozen B epoch24 selection/config')
    record = s['all_checkpoints']['epoch_24']
    checkpoint = Path(record['checkpoint']).resolve()
    if Path(s['selected_path']).resolve()!=checkpoint or sha(checkpoint)!=record['checkpoint_sha256']:
        raise ValueError('Frozen B checkpoint identity differs')
    return dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                selection_sha256=sha(selection_path),selected_epoch='epoch_24')


def probe_batch(model, batch, phase, scale, cfg):
    import torch
    head = model.bbox_head
    model.zero_grad(set_to_none=True)
    set_assignment_phase(head,phase)
    if any(m.get('flip',False) for m in batch['img_metas']):
        raise ValueError('Unexpected flip')
    with capture_main_positives(head) as captured:
        losses = model(return_loss=True,**batch)
        bbox,cls = sum_loss(losses['loss_bbox']),sum_loss(losses['loss_cls'])
        if not torch.allclose(sum(v for p,t,w,n,v in captured),bbox):
            raise RuntimeError('Captured actual KLD differs from emitted main loss')
        extras = shape_losses(captured)
        terms = dict(symkld=bbox,classification=cls,**extras)
        if not all(torch.isfinite(v).all() for v in terms.values()):
            raise RuntimeError('Nonfinite candidate loss')
        groups = [('regression_conv',[head.retina_reg.weight,head.retina_reg.bias]),
                  ('classification_conv',[head.retina_cls.weight,head.retina_cls.bias]),
                  ('full_fpn',[p for p in model.neck.parameters() if p.requires_grad]),
                  ('backbone_parameter_probe',[model.backbone.layer4[-1].conv3.weight])]
        gradients = inspect_gradients(terms,groups,captured)
        for name in ('e0','eh_unit'):
            for group in ('regression_conv','full_fpn'):
                if gradients['parameters'][group]['norms'][name]<=0:
                    raise RuntimeError('No candidate '+group+' signal: '+name)
        positive_count = sum(len(p) for p,t,w,n,v in captured)
        if not positive_count:raise RuntimeError('No positive sample')
        stats = positive_geometry(torch.cat([p for p,t,w,n,v in captured]),
                                  torch.cat([t for p,t,w,n,v in captured]))
        from mmrotate.models.losses.covariance_shape_loss import covariance_shape_terms
        _,shape_guards = covariance_shape_terms(torch.cat([p for p,t,w,n,v in captured]),
            torch.cat([t for p,t,w,n,v in captured]),return_guards=True)
        total = sum(sum_loss(v) for k,v in losses.items() if 'loss' in k)
        clipping = {}
        for name,extra in [('actual_b',bbox*0.),('b_plus_e0',extras['e0']),
                           ('b_plus_eh_unit_stress',extras['eh_unit'])]:
            model.zero_grad(set_to_none=True)
            (total+extra).backward(retain_graph=True)
            params = [p for p in model.parameters() if p.requires_grad]
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise RuntimeError('Nonfinite complete counterfactual gradient')
            clip = cfg.optimizer_config.grad_clip
            before = float(torch.nn.utils.clip_grad_norm_(params,clip.max_norm,norm_type=clip.norm_type))
            clipping[name] = dict(before=before,after=grad_norm([p.grad for p in params]),
                multiplier=min(1.,float(clip.max_norm)/(before+1e-6)))
        row = dict(phase=phase,scale=scale,images=[Path(m['filename']).stem for m in batch['img_metas']],
            image_sha256=[sha(m['filename']) for m in batch['img_metas']],
            input_shapes=[list(m['img_shape']) for m in batch['img_metas']],
            losses={k:float(v.detach()) for k,v in terms.items()},gradients=gradients,
            positives=stats,shape_guards=shape_guards,
            normalizers=[float(n) for p,t,w,n,v in captured],clipping=clipping)
    model.zero_grad(set_to_none=True)
    return row


def runtime_probe(cfg, identity, gpu, progress):
    import numpy as np
    import torch
    from mmcv.parallel import collate,scatter
    from mmcv.runner import load_checkpoint
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    from crane_project.tools.ckpt_sweep import annotation_set_sha256
    torch.cuda.set_device(gpu)
    random.seed(1701);np.random.seed(1701);torch.manual_seed(1701);torch.cuda.manual_seed_all(1701)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    datasets = {s:build_dataset(fixed_train_specs(cfg,s)) for s in (1.,.5)}
    for ds in datasets.values():
        if [len(p) for p in ds.datasets]!=[1810,748]:raise ValueError('TRAIN split differs')
    infos = ordered_data_infos(datasets[1.])
    if [i['filename'] for i in infos]!=[i['filename'] for i in ordered_data_infos(datasets[.5])]:
        raise ValueError('TRAIN views differ in ordering')
    pairs = [(0,1810),(905,2184)]  # Reuse the previous bounded TRAIN preflight's four images.
    for real,sim in pairs:
        if not Path(infos[real]['filename']).stem.startswith('real_') or not Path(infos[sim]['filename']).stem.startswith('sim_'):
            raise ValueError('TRAIN domain index mismatch')
    model = build_detector(deepcopy(cfg.model))
    loaded = load_checkpoint(model,identity['checkpoint'],map_location='cpu',strict=True)
    meta = loaded.get('meta',{})
    contract = checkpoint_contract(meta,cfg)
    if contract['status']!='MATCH' or meta.get('epoch')!=24 or meta.get('seed')!=0:
        raise ValueError('Saved B checkpoint config/epoch/seed differs or missing')
    saved_metadata = dict(epoch=meta.get('epoch'),iter=meta.get('iter'),seed=meta.get('seed'))
    del loaded
    model.cuda(gpu).train()
    before = parameter_digest(model)
    torch.cuda.reset_peak_memory_stats(gpu)
    rows = []
    for scale in (1.,.5):
        for phase in ('warmup_o2m','late_o2o'):
            for pair in pairs:
                batch = scatter(collate([datasets[scale][i] for i in pair],samples_per_gpu=2),[gpu])[0]
                row = probe_batch(model,batch,phase,scale,cfg)
                row['source_indices'] = list(pair)
                rows.append(row)
                with Path(progress).open('a') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                print('TRAIN',phase,'scale',scale,'images',row['images'],'losses',row['losses'],flush=True)
                del batch
    after = parameter_digest(model)
    if before!=after:raise RuntimeError('Probe changed model parameters')
    ratios = [r['gradients']['parameters']['full_fpn']['e0_matching_eh_coefficient'] for r in rows]
    valid = [v for v in ratios if v is not None and math.isfinite(v) and v>0]
    return dict(rows=rows,checkpoint_contract=contract,checkpoint_metadata=saved_metadata,
        parameter_sha256_before=before,
        parameter_sha256_after=after,parameters_unchanged=True,
        coefficient_reference=dict(rule='median per-batch full-FPN norm(E0)/norm(unit E-H)',
            valid_batches=len(valid),proposed_lambda=float(np.median(valid)) if valid else None,
            range=[min(valid),max(valid)] if valid else None,
            note='Descriptive matching reference, NOT a frozen coefficient or accuracy claim. '
                 'Review per-phase/per-scale ratios and numerical guards before selecting one experiment.'),
        source_annotations_sha256={s:annotation_set_sha256(str(ROOT/cfg.data.train[0].data_root/s/'annfiles'))
                                   for s in ('train','train_sim')},
        peak_allocated_mib=torch.cuda.max_memory_allocated(gpu)/2**20,
        gpu_name=torch.cuda.get_device_name(gpu),torch_version=torch.__version__,
        optimizer_steps=0,training_epochs=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--geometry-only',action='store_true')
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--reference-sweep',default='work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/val_sweep_port_v1')
    ap.add_argument('--out-json',required=True)
    args = ap.parse_args();os.chdir(ROOT)
    out = Path(args.out_json).resolve()
    artifact,progress = out.with_suffix('.artifacts.json'),out.with_suffix('.progress.jsonl')
    if any(p.exists() for p in (out,artifact,progress)):
        raise FileExistsError('Preserve previous results; choose a new output name')
    cfg,_ = check_configs()
    report = dict(protocol='port_shape_e_preflight_v1',evidence_role='source_train_only_candidate_check',
        sources={p:sha(ROOT/p) for p in ('crane_project/tools/preflight_port_shape_e_v1.py',
            'crane_project/tools/diagnose_port_center_size_d_v1.py',
            'mmrotate/models/losses/covariance_shape_loss.py','mmrotate/models/losses/sym_kld_calculator.py',
            'mmrotate/models/losses/sym_kld_loss.py','mmrotate/models/dense_heads/sym_eood_head.py',
            'crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py')},
        candidate_settings=dict(e0_weight=.25,eh_probe_weight=1.,eps=1e-6,
            internal_precision='float64',eh_smoothing='sqrt(-expm1(-T)+eps)-sqrt(eps)'),
        formal_training_configuration='NOT_FROZEN',geometry=geometry_check())
    out.parent.mkdir(parents=True,exist_ok=True)
    if args.geometry_only:
        report['status']='GEOMETRY_ONLY_RUNTIME_UNVERIFIED'
    else:
        report['reference']=frozen_b_identity(Path(args.reference_sweep).resolve())
        with artifact.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
        progress.touch(exist_ok=False)
        try:
            report['train']=runtime_probe(cfg,report['reference'],args.gpu,progress)
            report['status']='TRAIN_PROBE_COMPLETE_REVIEW_REQUIRED'
        except Exception as exc:
            report['status']='CHECK_FAILED'
            report['error']=type(exc).__name__+': '+str(exc)
            with out.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
            raise
    with out.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print('Saved',out,'status',report['status'])


if __name__=='__main__':main()
