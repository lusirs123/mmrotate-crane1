#!/usr/bin/env python3
"""Fixed E-H integration + original seed0 initialization, four TRAIN batches.

No saved B checkpoint, VAL/TEST inference, optimizer or coefficient search.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.preflight_port_center_size_v1 import (
    CONTROL, check_configs as check_b_d_configs, fixed_train_specs,
    set_assignment_phase, sum_loss)
from crane_project.tools.preflight_port_shape_e_v1 import capture_main_positives
from crane_project.tools.diagnose_port_center_size_d_v1 import (
    sha, parameter_digest, gradient_vector, grad_norm, grad_cos, ordered_data_infos)

EXPERIMENT = ROOT/'crane_project/configs/crane_symeood_k1_port_day2night_shape_e_h_v1.py'
EXPECTED = dict(type='CovarianceShapeLoss',mode='hellinger',loss_weight=.05,
                eps=1e-6,reduction='mean')
SOURCES = ('crane_project/tools/preflight_port_shape_e_h_v1.py',
    'crane_project/tools/preflight_port_shape_e_v1.py',
    'crane_project/tools/preflight_port_center_size_v1.py',
    'crane_project/tools/diagnose_port_center_size_d_v1.py',
    'mmrotate/models/losses/covariance_shape_loss.py',
    'mmrotate/models/losses/__init__.py',
    'mmrotate/models/dense_heads/sym_eood_head.py',
    'mmrotate/models/detectors/sym_eood_detector.py',
    'mmrotate/models/losses/sym_kld_loss.py',
    'mmrotate/models/losses/sym_kld_calculator.py',
    'mmrotate/core/bbox/coder/delta_xywha_rbbox_coder.py',
    'mmrotate/datasets/pipelines/port_train_augment.py',
    'tools/train.py',
    'tools/dist_train.sh',
    'crane_project/configs/crane_symeood_k1.py',
    'crane_project/configs/crane_symeood_k1_port_day2night_v1.py',
    'crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py',
    'crane_project/configs/crane_symeood_k1_port_day2night_shape_e_h_v1.py')


def check_configs():
    from mmcv import Config
    b,_ = check_b_d_configs()
    e = Config.fromfile(str(EXPERIMENT))
    baseline,actual = deepcopy(b.to_dict()),deepcopy(e.to_dict())
    if actual['model']['bbox_head'].pop('shape_compensation')!=EXPECTED:
        raise ValueError('Fixed E-H formula/coefficient differs')
    actual['work_dir'] = baseline['work_dir']
    if actual!=baseline:
        raise ValueError('E-H differs from B beyond shape loss and work_dir')
    if e.model.bbox_head.get('center_size_compensation') is not None:
        raise ValueError('E-H must not include D')
    if e.model.backbone.init_cfg!=dict(type='Pretrained',checkpoint='torchvision://resnet50'):
        raise ValueError('Expected original ImageNet initialization')
    return b,e


@contextmanager
def observe_integration(head):
    """Capture actual extra nodes and upstream decoder guards; restore on failure."""
    extras,levels = [],[]
    original_extra,original_single = head.shape_compensation.forward,head.loss_single
    def extra_forward(pred,target,weight=None,avg_factor=None,**kwargs):
        value=original_extra(pred,target,weight=weight,avg_factor=avg_factor,**kwargs)
        extras.append((pred,target,weight,avg_factor,value))
        return value
    def observed_single(self,*args,**kwargs):
        import torch
        cls,reg,anchors,labels=args[:4]
        with torch.no_grad():
            flat=reg.permute(0,2,3,1).reshape(-1,5)
            pre=self.bbox_coder.decode(anchors.reshape(-1,5),flat)
            mask=(labels.reshape(-1)>=0)&(labels.reshape(-1)<self.num_classes)
            p=pre[mask]
            maximum=kwargs.get('decode_max_size')
            max_wh=float(maximum) if maximum is not None else 2048.
            levels.append(dict(n=int(mask.sum()),
                delta_wh_clip_coordinates=int((flat[mask,2:4].abs()>abs(math.log(16/1000))).sum()),
                decoded_edge_clamp_boxes=int(((p[:,2:4]<1)|(p[:,2:4]>max_wh)).any(-1).sum()),
                decoded_center_clamp_boxes=(int(((p[:,:2]<0)|(p[:,:2]>float(maximum)-1)).any(-1).sum())
                                            if maximum is not None else 0)))
        return original_single(*args,**kwargs)
    head.shape_compensation.forward=extra_forward
    head.loss_single=MethodType(observed_single,head)
    try:
        yield extras,levels
    finally:
        head.shape_compensation.forward=original_extra
        head.loss_single=original_single


def integration_batch(model,batch,scale,cfg):
    import torch
    from mmrotate.models.losses.covariance_shape_loss import CovarianceShapeLoss, covariance_shape_terms
    head=model.bbox_head
    if any(m.get('flip',False) for m in batch['img_metas']):
        raise ValueError('Unexpected flip')
    model.zero_grad(set_to_none=True)
    set_assignment_phase(head,'warmup_o2m')
    with capture_main_positives(head) as main, observe_integration(head) as (extra_nodes,levels):
        losses=model(return_loss=True,**batch)
    bbox,extra=sum_loss(losses['loss_bbox']),sum_loss(losses['loss_shape_compensation'])
    if not main or len(main)!=len(extra_nodes) or len(main)!=len(levels):
        raise RuntimeError('Main/extra capture differs')
    if not torch.allclose(sum(x[4] for x in main),bbox):
        raise RuntimeError('Main KLD capture differs')
    direct_fn=CovarianceShapeLoss(**{k:v for k,v in EXPECTED.items() if k!='type'})
    independent=[]
    for a,b in zip(main,extra_nodes):
        if a[0] is not b[0] or a[1] is not b[1] or a[3]!=b[3] or not torch.equal(a[2],b[2]):
            raise RuntimeError('Extra does not reuse actual main positive tensors')
        independent.append(direct_fn(a[0],a[1],weight=a[2],avg_factor=a[3]))
    if not torch.allclose(sum(independent),extra,rtol=1e-6,atol=1e-8):
        raise RuntimeError('Integrated extra differs from independent formula')
    if not torch.allclose(sum(x[4] for x in extra_nodes),extra):
        raise RuntimeError('Extra capture differs')
    count=sum(len(x[0]) for x in main)
    if count<=0 or int(losses['shape_positive_count'])!=count:
        raise RuntimeError('Positive count differs')
    base_terms={k:sum_loss(v) for k,v in losses.items() if 'loss' in k and k!='loss_shape_compensation'}
    base_total=sum(base_terms.values())
    total=base_total+extra
    if any(not torch.isfinite(v).all() for v in [extra,bbox,total,*base_terms.values()]):
        raise RuntimeError('Nonfinite integrated loss')
    nodes=[x[0] for x in main]
    direct=gradient_vector(extra,nodes)
    if any(g is None and len(p)>0 for g,p in zip(direct,nodes)):
        raise RuntimeError('Missing actual positive graph dependency')
    if any(g is not None and torch.count_nonzero(g[:,:2]) for g in direct):
        raise RuntimeError('Unexpected direct centre gradient')
    coordinate_norms={k:grad_norm([g[:,j:j+1] for g in direct if g is not None])
                      for j,k in enumerate(('x','y','w','h','angle'))}
    groups={'regression_conv':[head.retina_reg.weight,head.retina_reg.bias],
            'classification_conv':[head.retina_cls.weight,head.retina_cls.bias],
            'full_fpn':[p for p in model.neck.parameters() if p.requires_grad]}
    gradients={}
    for name,params in groups.items():
        gs={k:gradient_vector(v,params) for k,v in [('main_symkld',bbox),('shape',extra),('complete_b',base_total)]}
        ns={k:grad_norm(v) for k,v in gs.items()}
        gradients[name]=dict(norms=ns,
            shape_to_main=ns['shape']/ns['main_symkld'] if ns['main_symkld'] else None,
            shape_to_complete_b=ns['shape']/ns['complete_b'] if ns['complete_b'] else None,
            cos_shape_main=grad_cos(gs['shape'],gs['main_symkld']),
            cos_shape_complete_b=grad_cos(gs['shape'],gs['complete_b']))
    if gradients['classification_conv']['norms']['shape']!=0:
        raise RuntimeError('Unexpected direct classification gradient')
    if any(gradients[g]['norms']['shape']<=0 for g in ('regression_conv','full_fpn')):
        raise RuntimeError('No shape signal')
    clipping={}
    params=[p for p in model.parameters() if p.requires_grad]
    for name,value in [('complete_b',base_total),('complete_e_h',total)]:
        model.zero_grad(set_to_none=True)
        value.backward(retain_graph=True)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            raise RuntimeError('Nonfinite combined gradient')
        clip=cfg.optimizer_config.grad_clip
        before=float(torch.nn.utils.clip_grad_norm_(params,clip.max_norm,norm_type=clip.norm_type))
        clipping[name]=dict(before=before,after=grad_norm([p.grad for p in params]),
            multiplier=min(1.,float(clip.max_norm)/(before+1e-6)))
    _,guards=covariance_shape_terms(torch.cat(nodes),torch.cat([x[1] for x in main]),return_guards=True)
    # Same initialized detector with extra disabled is exactly B's model config.
    # Reset scheduling counters; original B/E common losses must remain equal.
    saved=head.shape_compensation
    try:
        head.shape_compensation=None
        set_assignment_phase(head,'warmup_o2m')
        # Keep the same grad-enabled forward mode as E to avoid comparing
        # different backend forward paths. Discard B's graph immediately.
        b_losses=model(return_loss=True,**batch)
    finally:
        head.shape_compensation=saved
    baseline={k:sum_loss(v).detach() for k,v in b_losses.items() if 'loss' in k}
    del b_losses
    if baseline.keys()!=base_terms.keys() or any(
            not torch.allclose(baseline[k],v,rtol=1e-5,atol=1e-7) for k,v in base_terms.items()):
        raise RuntimeError('Original B losses changed with E-H enabled')
    row=dict(scale=scale,phase='warmup_o2m_actual_seed0_initialization',
        images=[Path(m['filename']).stem for m in batch['img_metas']],
        image_sha256=[sha(m['filename']) for m in batch['img_metas']],
        input_shapes=[list(m['img_shape']) for m in batch['img_metas']],
        positives=count,normalizers=[float(x[3]) for x in main],decoder_guards=levels,
        shape_guards=guards,direct_coordinate_norms=coordinate_norms,
        actual_node_reuse=True,base_losses_unchanged=True,
        losses={k:float(v.detach()) for k,v in base_terms.items()},
        loss_shape=float(extra.detach()),loss_complete_e=float(total.detach()),
        gradients=gradients,clipping=clipping)
    model.zero_grad(set_to_none=True)
    return row


def initialize_matched_models(b,e):
    """Match tools/train.py: seed0 -> build_detector -> init_weights, twice."""
    from mmdet.apis import set_random_seed
    from mmrotate.models import build_detector
    set_random_seed(0,deterministic=False)
    baseline=build_detector(deepcopy(b.model));baseline.init_weights()
    digest=parameter_digest(baseline)
    buffers={k:v.detach().clone() for k,v in baseline.named_buffers()}
    keys=list(baseline.state_dict())
    del baseline
    set_random_seed(0,deterministic=False)
    model=build_detector(deepcopy(e.model));model.init_weights()
    if parameter_digest(model)!=digest or list(model.state_dict())!=keys:
        raise RuntimeError('B/E initialization parameters or state keys differ')
    import torch
    current=dict(model.named_buffers())
    if current.keys()!=buffers.keys() or any(not torch.equal(current[k],v) for k,v in buffers.items()):
        raise RuntimeError('B/E initialization buffers differ')
    return model,digest


def runtime_probe(b,e,gpu,progress):
    import torch
    from mmcv.parallel import collate,scatter
    from mmrotate.datasets import build_dataset
    from crane_project.tools.ckpt_sweep import annotation_set_sha256
    torch.cuda.set_device(gpu)
    model,initial_digest=initialize_matched_models(b,e)
    model.cuda(gpu).train()
    datasets={s:build_dataset(fixed_train_specs(e,s)) for s in (1.,.5)}
    for ds in datasets.values():
        if [len(x) for x in ds.datasets]!=[1810,748]:raise ValueError('TRAIN split differs')
    infos=ordered_data_infos(datasets[1.])
    if [x['filename'] for x in infos]!=[x['filename'] for x in ordered_data_infos(datasets[.5])]:
        raise ValueError('TRAIN ordering differs')
    torch.cuda.reset_peak_memory_stats(gpu)
    rows=[]
    for scale in (1.,.5):
        for pair in ((0,1810),(905,2184)):
            if not (Path(infos[pair[0]]['filename']).stem.startswith('real_') and
                    Path(infos[pair[1]]['filename']).stem.startswith('sim_')):
                raise ValueError('TRAIN image domain differs')
            batch=scatter(collate([datasets[scale][i] for i in pair],samples_per_gpu=2),[gpu])[0]
            row=integration_batch(model,batch,scale,e);row['source_indices']=list(pair)
            rows.append(row)
            with Path(progress).open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            print('INIT TRAIN',scale,row['images'],'shape',row['loss_shape'],
                  'FPN shape/main',row['gradients']['full_fpn']['shape_to_main'],
                  'clip',row['clipping'],flush=True)
            del batch
    after=parameter_digest(model)
    if after!=initial_digest:raise RuntimeError('Probe changed model parameters')
    return dict(rows=rows,seed=0,b_e_initialization_equal=True,parameters_unchanged=True,
        parameter_sha256_before=initial_digest,parameter_sha256_after=after,
        initialization='tools/train.py seed0, build_detector, init_weights; no B checkpoint',
        source_annotations_sha256={s:annotation_set_sha256(str(ROOT/e.data.train[0].data_root/s/'annfiles'))
                                   for s in ('train','train_sim')},
        optimizer_steps=0,training_epochs=0,
        gpu_name=torch.cuda.get_device_name(gpu),torch_version=torch.__version__,
        peak_allocated_mib=torch.cuda.max_memory_allocated(gpu)/2**20)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config-only',action='store_true')
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--out-json',required=True)
    args=ap.parse_args();os.chdir(ROOT)
    out=Path(args.out_json).resolve()
    artifacts,progress=out.with_suffix('.artifacts.json'),out.with_suffix('.progress.jsonl')
    if any(p.exists() for p in (out,artifacts,progress)):raise FileExistsError('Use a new output name')
    b,e=check_configs()
    report=dict(protocol='port_shape_e_h_v1_initialization_check',
        evidence_role='source_train_only_fixed_formula_integration_check',
        formula_settings=EXPECTED,formula='sqrt(-expm1(-T)+1e-6)-sqrt(1e-6)',
        internal_precision='float64',formal_formula_and_coefficient='FROZEN',
        sources={p:sha(ROOT/p) for p in SOURCES},seed=0,
        limitations=['Four TRAIN images only; no optimizer, VAL, TEST or accuracy evidence.',
                     'Matching current B initialization does not reconstruct an archived historical initial state.',
                     'Training requires review of numerical guards, strength and clipping after this check.'])
    out.parent.mkdir(parents=True,exist_ok=True)
    if args.config_only:report['status']='CONFIG_ONLY_INITIALIZATION_UNVERIFIED'
    else:
        with artifacts.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
        progress.touch(exist_ok=False)
        try:
            report['train']=runtime_probe(b,e,args.gpu,progress)
            report['status']='INITIALIZATION_CHECK_COMPLETE_REVIEW_REQUIRED'
        except Exception as exc:
            report['status']='CHECK_FAILED';report['error']=type(exc).__name__+': '+str(exc)
            with out.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
            raise
    with out.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print('Saved',out,'status',report['status'])


if __name__=='__main__':main()
