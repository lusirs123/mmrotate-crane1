"""Diagnostic computations and artifact handling; no claimed training performance."""
import json
from copy import deepcopy
from types import MethodType

import pytest
import torch

from crane_project.tools.diagnose_port_center_size_d_v1 import (
    d_parts, grad_cos, gradient_report, positive_geometry, read_logs, runs,
    val_analysis, config_differences, checkpoint_contract)


def test_decomposition_matches_actual_loss_and_gradient():
    from mmrotate.models.losses.center_size_compensation import CenterSizeCompensationLoss
    p=torch.tensor([[3.,4.,45.,19.,.3],[4.,5.,22.,33.,-.4]],dtype=torch.double,requires_grad=True)
    t=torch.tensor([[2.,2.,40.,20.,.2],[3.,7.,30.,20.,.1]],dtype=torch.double,requires_grad=True)
    w=torch.tensor([1.,.5],dtype=torch.double)
    c,s=d_parts(p,t,w,7.)
    actual=CenterSizeCompensationLoss()(p,t,weight=w,avg_factor=7.)
    assert torch.allclose(c+s,actual)
    gc=torch.autograd.grad(c+s,p,retain_graph=True)[0]
    ga=torch.autograd.grad(actual,p,retain_graph=True)[0]
    assert torch.allclose(gc,ga)
    assert torch.autograd.grad(c+s,t,allow_unused=True)[0] is None
    assert torch.count_nonzero(gc[:,4])==0


def test_empty_gradients_and_zero_cosine_are_defined():
    p=torch.empty(0,5,requires_grad=True)
    c,s=d_parts(p,p,torch.empty(0),2.)
    assert float(c+s)==0.
    (c+s).backward()
    assert p.grad.shape==(0,5)
    assert grad_cos((torch.zeros(3),),(torch.ones(3),)) is None
    assert grad_cos((torch.tensor([1.,0.]),),(torch.tensor([-2.,0.]),))==-1.


def test_raw_kld_cap_and_geometry_counts():
    t=torch.tensor([[10.,20.,40.,20.,.2],[10.,20.,40.,20.,.2]])
    p=t.clone(); p[1,0]+=1000.
    report=positive_geometry(p,t)
    assert report['n']==2 and report['symkld_loss_cap_count']==1
    assert report['center_smoothl1_linear_coordinate_count']==1
    assert report['size_smoothl1_linear_coordinate_count']==0
    assert report['determinant_guard_count']==0


def test_logs_preserve_nonfinite_and_multiple_runs(tmp_path):
    (tmp_path/'a.log.json').write_text('\n'.join([
        json.dumps(dict(mode='train',epoch=1,loss_center_size_compensation=.2,grad_norm=12.)),
        json.dumps(dict(mode='val',epoch=1,loss_center_size_compensation=999.)),
        json.dumps(dict(mode='train',epoch=1,loss_center_size_compensation=float('nan'))),
        'bad json']))
    report=read_logs(tmp_path)
    fields=report['logs'][0]['epochs']['1']
    assert fields['loss_center_size_compensation']['n']==1
    assert fields['loss_center_size_compensation']['nonfinite_count']==1
    assert fields['grad_norm']['above_clip10_count']==1
    assert report['logs'][0]['malformed_lines']==[4]
    assert read_logs(tmp_path/'missing')['missing']


def row(seq,i,pred):
    from crane_project.tools.audit_port_train_val_geometry_v1 import decompose
    gt=[30.,30.,40.,20.,.1]
    return dict(image=seq+'_'+str(i).zfill(5),sequence=seq,frame_id=i,
                image_sha256='synthetic-fixture-'+seq+'-'+str(i),
                domain=seq.split('_')[0],split='val',gt=gt,pred=pred,
                input_gt_short_px=20.,gt_aspect=2.,gt_angle_deg=5.73,
                metrics=decompose(gt,None if pred is None else pred[:5]))


def test_val_pairing_does_not_hide_missing_outputs_or_cross_gaps():
    good=[30.,30.,40.,20.,.1,.9]
    wrong=[200.,200.,40.,20.,.1,.8]
    b=[row('real_seq07',1,wrong),row('real_seq07',2,good),row('real_seq07',4,None),
       row('sim_seq10',1,good)]
    d=[row('real_seq07',1,good),row('real_seq07',2,None),row('real_seq07',4,None),
       row('sim_seq10',1,good)]
    report=val_analysis(b,d)
    assert report['paired']['real_ordinary_shared']['n']==0
    assert report['summary']['output_churn']['control_b_only_output']==1
    assert report['summary']['compensated_d']['output_frames']==2
    failures=report['temporal']['d']['real_seq07']['no_output']
    assert [r['length'] for r in failures]==[1,1]
    assert report['historical_severe_b_frames']==['real_seq07_00001']


def test_saved_config_comparison_reports_actual_difference():
    from mmcv import Config
    expected=Config(dict(model=dict(type='x'),data=dict(samples_per_gpu=2)))
    assert checkpoint_contract({},expected)['status'].startswith('MISSING')
    same='model=dict(type="x")\ndata=dict(samples_per_gpu=2)\n'
    assert checkpoint_contract(dict(config=same),expected)['status']=='MATCH'
    changed=checkpoint_contract(dict(config=same.replace('=2','=4')),expected)
    assert changed['status']=='DIFFERENCES_REVIEW_REQUIRED'
    assert changed['differences'][0]['path']=='data.samples_per_gpu'
    assert config_differences({'x':[1,2]},{'x':[1,3]})[0]['path']=='x[1]'


@pytest.mark.parametrize('phase',['warmup_o2m','late_o2o'])
def test_actual_head_capture_decomposition_and_gradient_report(phase):
    from mmrotate.models import build_head
    from crane_project.tools.preflight_port_center_size_v1 import check_configs,set_assignment_phase
    _,cfg=check_configs()
    spec=deepcopy(cfg.model.bbox_head)
    spec.train_cfg,spec.test_cfg=cfg.model.train_cfg,cfg.model.test_cfg
    head=build_head(spec).train()
    set_assignment_phase(head,phase)
    def cpu_anchors(self,sizes,metas,device='cpu'):
        anchors=self.anchor_generator.grid_priors(sizes,device='cpu')
        flags=[self.anchor_generator.valid_flags(sizes,m['pad_shape'],device='cpu') for m in metas]
        return [anchors for _ in metas],flags
    head.get_anchors=MethodType(cpu_anchors,head)
    captured=[]
    original=head.center_size_compensation.forward
    def capture(p,t,weight=None,avg_factor=None,**kwargs):
        value=original(p,t,weight=weight,avg_factor=avg_factor,**kwargs)
        captured.append((p,t,weight,avg_factor,value))
        return value
    head.center_size_compensation.forward=capture
    torch.manual_seed(1701)
    feats=[torch.randn(2,256,n,n,requires_grad=True) for n in [8,4,2,1,1]]
    gt=[torch.tensor([[30.,30.,32.,16.,.2]]) for _ in range(2)]
    meta=[dict(img_shape=(64,64,3),pad_shape=(128,128,3),scale_factor=[1.,1.,1.,1.]) for _ in range(2)]
    losses=head.loss(*head(feats),gt,[torch.zeros(1,dtype=torch.long) for _ in gt],meta)
    parts=[d_parts(p,t,w,n) for p,t,w,n,v in captured]
    center=sum(c for c,s in parts);size=sum(s for c,s in parts)
    assert torch.allclose(center+size,sum(losses['loss_center_size_compensation']))
    bbox=sum(losses['loss_bbox']);cls=sum(losses['loss_cls'])
    report=gradient_report(dict(symkld=bbox,classification=cls,other_losses=bbox*0.,
        d_center=center,d_size=size),[
        ('regression_conv',[head.retina_reg.weight,head.retina_reg.bias]),
        ('classification_conv',[head.retina_cls.weight]),('feature_probe',[feats[0]])])
    assert report['regression_conv']['norms']['d_sum']>0
    assert report['classification_conv']['norms']['d_sum']==0
    assert report['feature_probe']['norms']['base_total']>0
    assert report['regression_conv']['cosines']['d_sum_vs_symkld'] is not None
    (bbox+cls+center+size).backward()
    assert all(torch.isfinite(f.grad).all() for f in feats if f.grad is not None)


def test_runtime_orchestration_with_actual_head_and_cpu_io_fixture(tmp_path,monkeypatch):
    """Exercise CUDA orchestration on CPU with tiny shared features and fake I/O.

    The head/loss/assignment/autograd are real. This does not validate CUDA,
    ResNet activations, server artifact identities or detector accuracy.
    """
    import torch.nn as nn
    from mmcv.parallel import collate,scatter
    import mmcv.parallel
    import mmrotate.datasets
    import mmrotate.models
    from mmrotate.models import build_head
    from mmrotate.models.dense_heads.sym_eood_head import SymEOODHead
    from crane_project.tools.preflight_port_center_size_v1 import check_configs
    from crane_project.tools.diagnose_port_center_size_d_v1 import runtime_probe,sha
    b,cfg=check_configs()
    class Conv(nn.Module):
        def __init__(self):
            super().__init__(); self.conv=nn.Conv2d(256,256,1)
        def forward(self,x): return self.conv(x)
    class Block(nn.Module):
        def __init__(self):
            super().__init__(); self.conv3=nn.Conv2d(3,256,1)
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            spec=deepcopy(cfg.model.bbox_head)
            spec.train_cfg,spec.test_cfg=cfg.model.train_cfg,cfg.model.test_cfg
            self.bbox_head=build_head(spec)
            self.neck=nn.Module();self.neck.fpn_convs=nn.ModuleList([Conv()])
            self.backbone=nn.Module();self.backbone.layer4=nn.ModuleList([Block()])
        def forward(self,img,img_metas,gt_bboxes,gt_labels,return_loss=True):
            import torch.nn.functional as F
            x=self.neck.fpn_convs[0](self.backbone.layer4[0].conv3(img))
            feats=[F.adaptive_avg_pool2d(x,(n,n)) for n in (8,4,2,1,1)]
            result=self.bbox_head.loss(*self.bbox_head(feats),gt_bboxes,gt_labels,img_metas)
            result['fixture_aux_loss']=x.square().mean()*.01
            return result
    class Dataset:
        datasets=[range(1810),range(748)]
        data_infos=[dict(filename=('real_seq01_' if i<1810 else 'sim_seq08_')+str(i)+'.png')
                    for i in range(2558)]
        def __getitem__(self,i):
            path=tmp_path/self.data_infos[i]['filename']
            if not path.exists():path.write_bytes(b'synthetic image identity fixture')
            return dict(img=torch.ones(3,16,16),gt_bboxes=torch.tensor([[30.,30.,32.,16.,.2]]),
                        gt_labels=torch.zeros(1,dtype=torch.long),
                        img_metas=dict(filename=str(path),img_shape=(64,64,3),pad_shape=(128,128,3),
                                       scale_factor=[1.,1.,1.,1.],flip=False))
    def batch(items,samples_per_gpu=2):
        return dict(img=torch.stack([i['img'] for i in items]),
            img_metas=[i['img_metas'] for i in items],gt_bboxes=[i['gt_bboxes'] for i in items],
            gt_labels=[i['gt_labels'] for i in items])
    def cpu_anchors(self,sizes,metas,device='cpu'):
        anchors=self.anchor_generator.grid_priors(sizes,device='cpu')
        flags=[self.anchor_generator.valid_flags(sizes,m['pad_shape'],device='cpu') for m in metas]
        return [anchors for _ in metas],flags
    checkpoint=tmp_path/'fixture.pth';torch.save(dict(state_dict=Tiny().state_dict()),str(checkpoint))
    before=sha(checkpoint)
    monkeypatch.setattr(torch.cuda,'set_device',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *a:'CPU fixture; CUDA unverified')
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *a:0)
    monkeypatch.setattr(nn.Module,'cuda',lambda self,*a:self)
    monkeypatch.setattr(mmrotate.models,'build_detector',lambda *a:Tiny())
    monkeypatch.setattr(mmrotate.datasets,'build_dataset',lambda *a:Dataset())
    monkeypatch.setattr(mmcv.parallel,'collate',batch)
    monkeypatch.setattr(mmcv.parallel,'scatter',lambda batch,gpus:[batch])
    monkeypatch.setattr(SymEOODHead,'get_anchors',cpu_anchors)
    ids={a:dict(checkpoint=str(checkpoint)) for a in ('b','d')}
    progress=tmp_path/'progress.jsonl'
    report=runtime_probe(cfg,ids,0,1,progress,control_cfg=b)
    assert len(report['rows'])==12 and report['optimizer_steps']==0
    assert len(report['summary'])==12
    assert all(r['positive_count']>0 for r in report['summary'])
    assert len(progress.read_text().splitlines())==12
    assert sha(checkpoint)==before
    assert not report['problems']
    assert all(r['parameters_unchanged'] for r in report['checkpoint_metadata'].values())
    assert all(r['domain_direct_gradient']['real']['positive_count']>0 for r in report['rows'])
    assert all(r['domain_direct_gradient']['sim']['positive_count']>0 for r in report['rows'])
    assert all(r['actual_objective_includes_d']==(r['arm']=='d') for r in report['rows'])
