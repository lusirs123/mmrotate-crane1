"""Meaningful candidate/graph contracts; CPU fixtures do not establish GPU accuracy."""
import math
from copy import deepcopy
from types import MethodType

import pytest
import torch

from mmrotate.models.losses.covariance_shape_loss import (
    CovarianceShapeLoss, covariance_shape_terms)
from crane_project.tools.preflight_port_shape_e_v1 import (
    capture_main_positives, shape_losses, geometry_check, probe_batch, frozen_b_identity)


@pytest.mark.parametrize('mode',['symkl','hellinger'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_geometry_equivalence_scale_and_designed_gradient_scope(mode,dtype):
    p=torch.tensor([[3.,4.,46.,19.,.32],[4.,5.,21.,35.,-.4]],dtype=dtype,requires_grad=True)
    t=torch.tensor([[2.,2.,40.,20.,.2],[3.,7.,30.,20.,.1]],dtype=dtype,requires_grad=True)
    loss=CovarianceShapeLoss(mode=mode,loss_weight=1.)
    value=loss(p,t)
    swapped=p.detach().clone();swapped[:,2:4]=p.detach()[:,[3,2]];swapped[:,4]+=math.pi/2
    period=p.detach().clone();period[:,4]+=math.pi
    for changed in [swapped,period]:assert torch.allclose(loss(changed,t),value,atol=2e-6,rtol=2e-5)
    factor=torch.tensor([.5,.5,.5,.5,1.],dtype=dtype)
    assert torch.allclose(loss(p*factor,t*factor),value,atol=1e-7,rtol=1e-6)
    grad=torch.autograd.grad(value,p,retain_graph=True)[0]
    assert torch.count_nonzero(grad[:,:2])==0
    assert torch.count_nonzero(grad[:,2:4])>0 and torch.count_nonzero(grad[:,4])>0
    assert torch.autograd.grad(value,t,allow_unused=True)[0] is None


@pytest.mark.parametrize('mode',['symkl','hellinger'])
def test_empty_identity_square_and_weight_normalizer(mode):
    loss=CovarianceShapeLoss(mode=mode,loss_weight=1.)
    empty=torch.empty(0,5,requires_grad=True)
    loss(empty,empty).backward();assert empty.grad.shape==(0,5)
    assert loss(empty,empty,reduction_override='none').shape==(0,)
    target=torch.tensor([[10.,20.,40.,20.,.2],[5.,5.,20.,20.,.1]],dtype=torch.double)
    match=target.clone().requires_grad_()
    z=loss(match,target);g=torch.autograd.grad(z,match)[0]
    assert abs(float(z))<1e-10 and torch.isfinite(g).all()
    square=target[1:].clone();square[:,4]+=.3;square.requires_grad_()
    sq=loss(square,target[1:]);sg=torch.autograd.grad(sq,square)[0]
    assert abs(float(sq))<1e-10 and abs(float(sg[0,4]))<1e-10
    pred=target.clone();pred[:,2]*=.95;pred.requires_grad_()
    values=loss(pred,target,reduction_override='none')
    weights=torch.tensor([1.,.5],dtype=torch.double)
    expected=(values*weights).sum()/7.
    assert torch.allclose(loss(pred,target,weight=weights,avg_factor=7.),expected)
    assert torch.allclose(loss(pred,target,weight=weights[:,None].expand(-1,5),avg_factor=7.),expected)


@pytest.mark.parametrize('mode',['symkl','hellinger'])
def test_autograd_matches_finite_differences_and_extreme_inputs_are_finite(mode):
    p=torch.tensor([[0.,0.,70.,21.,.37]],dtype=torch.double,requires_grad=True)
    t=torch.tensor([[100.,-200.,60.,20.,.1]],dtype=torch.double)
    loss=CovarianceShapeLoss(mode=mode,loss_weight=1.)
    assert torch.autograd.gradcheck(lambda x:loss(x,t),(p,),eps=1e-5,atol=1e-5,rtol=1e-4)
    extreme=torch.tensor([[0.,0.,2048.,1.,.785],[0.,0.,.2,.1,.4]],requires_grad=True)
    q=torch.tensor([[1.,2.,2048.,1.,-.785],[1.,2.,4.,2.,.3]])
    value=loss(extreme,q);value.backward()
    assert torch.isfinite(value) and torch.isfinite(extreme.grad).all()
    with pytest.raises(ValueError):loss(torch.tensor([[0.,0.,-1.,2.,0.]]),q[:1])


def test_small_error_expansions_and_synthetic_descent_are_descriptive():
    t=torch.tensor([[0.,0.,60.,20.,.2]],dtype=torch.double)
    p=t.clone();u=.001;v=-.002;p[:,2]*=math.exp(u);p[:,3]*=math.exp(v)
    terms=covariance_shape_terms(p,t)
    assert math.isclose(float(terms['symkl']),math.cosh(2*u)+math.cosh(2*v)-2,abs_tol=1e-12)
    assert math.isclose(float(terms['bhattacharyya']),.5*(math.log(math.cosh(u))+math.log(math.cosh(v))),abs_tol=1e-12)
    result=geometry_check();assert len(result['rows'])==60
    assert all(r['gradient'][0]==r['gradient'][1]==0 for r in result['rows'])
    ordinary=[r for r in result['rows'] if r['case'] not in ['match','centre_only'] and r['aspect']>1.]
    assert all(r['dimensionless_descent_loss_delta']<=1e-9 for r in ordinary)
    assert all(r['riou']<1 for r in result['rows'] if r['case']=='centre_only')


def test_capture_uses_actual_graph_node_and_restores_after_exception():
    from types import SimpleNamespace
    from mmrotate.models.losses.sym_kld_loss import SymKLDLoss
    head=SimpleNamespace(loss_bbox=SymKLDLoss(loss_weight=2.))
    original=head.loss_bbox.forward
    parent=torch.tensor([[0.,0.,40.,20.,.3]],requires_grad=True)
    target=torch.tensor([[2.,1.,42.,19.,.1]])
    main=parent[torch.tensor([True])];sibling=parent[torch.tensor([True])]
    with capture_main_positives(head) as captured:
        bbox=head.loss_bbox(main,target,weight=torch.ones(1),avg_factor=3.)
        assert captured[0][0] is main
        assert torch.autograd.grad(bbox,sibling,retain_graph=True,allow_unused=True)[0] is None
        extras=shape_losses(captured)
        for value in [bbox,*extras.values()]:
            assert torch.autograd.grad(value,main,retain_graph=True,allow_unused=True)[0] is not None
    assert head.loss_bbox.forward==original
    with pytest.raises(RuntimeError):
        with capture_main_positives(head):raise RuntimeError('fixture')
    assert head.loss_bbox.forward==original


def test_frozen_identity_refuses_wrong_epoch_and_changed_checkpoint(tmp_path):
    import json
    from crane_project.tools.diagnose_port_center_size_d_v1 import sha
    from crane_project.tools.preflight_port_center_size_v1 import CONTROL
    cp=tmp_path/'epoch_24.pth';cp.write_bytes(b'fixture')
    s=dict(evidence_role='source_val_checkpoint_selection',selected_checkpoint='epoch_24',
           config_sha256=sha(CONTROL),selected_path=str(cp),all_checkpoints={
           'epoch_24':dict(checkpoint=str(cp),checkpoint_sha256=sha(cp))})
    path=tmp_path/'sweep_results.json';path.write_text(json.dumps(s))
    assert frozen_b_identity(tmp_path)['selected_epoch']=='epoch_24'
    cp.write_bytes(b'changed')
    with pytest.raises(ValueError):frozen_b_identity(tmp_path)
    s['selected_checkpoint']='epoch_22';path.write_text(json.dumps(s))
    with pytest.raises(ValueError):frozen_b_identity(tmp_path)


@pytest.mark.parametrize('phase',['warmup_o2m','late_o2o'])
def test_actual_head_shared_feature_probe_and_parameter_immutability(tmp_path,phase,monkeypatch):
    import torch.nn as nn
    import torch.nn.functional as F
    from mmrotate.models import build_head
    from crane_project.tools.preflight_port_center_size_v1 import check_configs
    from crane_project.tools.diagnose_port_center_size_d_v1 import parameter_digest
    cfg,_=check_configs()
    class Conv(nn.Module):
        def __init__(self):super().__init__();self.conv3=nn.Conv2d(3,256,1)
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            spec=deepcopy(cfg.model.bbox_head);spec.train_cfg=cfg.model.train_cfg;spec.test_cfg=cfg.model.test_cfg
            self.bbox_head=build_head(spec)
            self.backbone=nn.Module();self.backbone.layer4=nn.ModuleList([Conv()])
            self.neck=nn.Conv2d(256,256,1)
            def cpu_anchors(head,sizes,metas,device='cpu'):
                anchors=head.anchor_generator.grid_priors(sizes,device='cpu')
                flags=[head.anchor_generator.valid_flags(sizes,m['pad_shape'],device='cpu') for m in metas]
                return [anchors for _ in metas],flags
            self.bbox_head.get_anchors=MethodType(cpu_anchors,self.bbox_head)
        def forward(self,img,img_metas,gt_bboxes,gt_labels,return_loss=True):
            x=self.neck(self.backbone.layer4[0].conv3(img))
            fs=[F.adaptive_avg_pool2d(x,(n,n)) for n in (8,4,2,1,1)]
            r=self.bbox_head.loss(*self.bbox_head(fs),gt_bboxes,gt_labels,img_metas)
            r['fixture_aux_loss']=x.square().mean()*.01
            return r
    torch.manual_seed(1701);model=Tiny().train()
    metas=[]
    for domain in ['real','sim']:
        path=tmp_path/(domain+'_seq01_00001.png');path.write_bytes(b'fixture')
        metas.append(dict(filename=str(path),img_shape=(64,64,3),pad_shape=(128,128,3),
            scale_factor=[1.,1.,1.,1.],flip=False))
    batch=dict(img=torch.ones(2,3,16,16),img_metas=metas,
        gt_bboxes=[torch.tensor([[30.,30.,32.,16.,.2]]) for _ in metas],
        gt_labels=[torch.zeros(1,dtype=torch.long) for _ in metas])
    before=parameter_digest(model)
    row=probe_batch(model,batch,phase,1.,cfg)
    assert parameter_digest(model)==before
    assert row['positives']['n']>0
    for name in ['symkld','e0','eh_unit']:
        d=row['gradients']['direct'][name]
        assert d['connected_nonempty_levels']==d['nonempty_levels']>0
        assert row['gradients']['parameters']['regression_conv']['norms'][name]>0
    for name in ['e0','eh_unit']:
        assert row['gradients']['direct'][name]['coordinate_norms']['x']==0
        assert row['gradients']['direct'][name]['coordinate_norms']['angle']>0
    assert row['gradients']['parameters']['full_fpn']['e0_matching_eh_coefficient']>0
    assert all(p.grad is None for p in model.parameters())
    if phase=='warmup_o2m':
        # End-to-end CLI fixture: real head/autograd, fake images/backbone/CUDA/I/O.
        # The production ConcatDataset wrapper is intentionally retained.
        import json
        import sys
        import mmcv.parallel
        import mmrotate.datasets
        import mmrotate.models
        import crane_project.tools.ckpt_sweep as sweep_module
        from mmdet.datasets.dataset_wrappers import ConcatDataset
        from crane_project.tools.preflight_port_shape_e_v1 import main
        from crane_project.tools.preflight_port_center_size_v1 import CONTROL
        from crane_project.tools.diagnose_port_center_size_d_v1 import sha
        class Leaf:
            CLASSES=('grab',)
            def __init__(self,domain,count):
                self.domain=domain
                self.data_infos=[dict(filename=domain+'_seq01_'+str(i)+'.png') for i in range(count)]
            def __len__(self):return len(self.data_infos)
            def __getitem__(self,i):
                path=tmp_path/self.data_infos[i]['filename']
                if not path.exists():path.write_bytes(b'CPU image fixture')
                return dict(img=torch.ones(3,16,16),gt_bboxes=torch.tensor([[30.,30.,32.,16.,.2]]),
                    gt_labels=torch.zeros(1,dtype=torch.long),img_metas=dict(filename=str(path),
                    img_shape=(64,64,3),pad_shape=(128,128,3),scale_factor=[1.,1.,1.,1.],flip=False))
        dataset=ConcatDataset([Leaf('real',1810),Leaf('sim',748)])
        def collate(items,samples_per_gpu=2):
            return dict(img=torch.stack([r['img'] for r in items]),
                **{k:[r[k] for r in items] for k in ('gt_bboxes','gt_labels','img_metas')})
        monkeypatch.setattr(mmrotate.datasets,'build_dataset',lambda *a:dataset)
        monkeypatch.setattr(mmrotate.models,'build_detector',lambda *a:Tiny())
        monkeypatch.setattr(mmcv.parallel,'collate',collate)
        monkeypatch.setattr(mmcv.parallel,'scatter',lambda data,devices:[data])
        monkeypatch.setattr(nn.Module,'cuda',lambda self,*a:self)
        for name in ['set_device','manual_seed_all','reset_peak_memory_stats']:
            monkeypatch.setattr(torch.cuda,name,lambda *a:None)
        monkeypatch.setattr(torch.cuda,'get_device_name',lambda *a:'CPU fixture; CUDA unverified')
        monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *a:0)
        monkeypatch.setattr(sweep_module,'annotation_set_sha256',lambda *a:'CPU annotations fixture')
        checkpoint=tmp_path/'epoch_24.pth'
        torch.save(dict(state_dict=model.state_dict(),meta=dict(config=cfg.pretty_text,epoch=24,seed=0)),checkpoint)
        selection=dict(evidence_role='source_val_checkpoint_selection',selected_checkpoint='epoch_24',
            config_sha256=sha(CONTROL),selected_path=str(checkpoint),all_checkpoints={
                'epoch_24':dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))})
        (tmp_path/'sweep_results.json').write_text(json.dumps(selection))
        output=tmp_path/'shape.json'
        monkeypatch.setattr(sys,'argv',['preflight','--reference-sweep',str(tmp_path),'--out-json',str(output)])
        main()
        result=json.loads(output.read_text())
        assert result['status']=='TRAIN_PROBE_COMPLETE_REVIEW_REQUIRED'
        assert result['formal_training_configuration']=='NOT_FROZEN'
        assert len(result['train']['rows'])==8
        assert result['train']['parameters_unchanged'] and result['train']['optimizer_steps']==0
        assert len(output.with_suffix('.progress.jsonl').read_text().splitlines())==8
        assert output.with_suffix('.artifacts.json').is_file()
        assert result['train']['coefficient_reference']['valid_batches']==8
        with pytest.raises(FileExistsError):main()


def test_independent_riou_shared_edges_analytic_overlap_and_symmetry():
    from crane_project.tools.preflight_port_shape_e_v1 import reference_riou
    import numpy as np
    for angle in [0.,.2,math.pi/4,1.5]:
        gt=np.array([120.,100.,92.,40.,angle])
        for edge in [2,3]:
            p=gt.copy();p[edge]*=.95
            assert math.isclose(reference_riou(p,gt),.95,abs_tol=1e-10)
            assert math.isclose(reference_riou(gt,p),.95,abs_tol=1e-10)
            shifted=p.copy();shifted[:2]+=1e5
            shifted_gt=gt.copy();shifted_gt[:2]+=1e5
            assert math.isclose(reference_riou(shifted,shifted_gt),.95,abs_tol=1e-9)
        assert math.isclose(reference_riou(gt,gt),1.,abs_tol=1e-10)
    assert math.isclose(reference_riou([0,0,2,2,0],[1,0,2,2,0]),1/3,abs_tol=1e-10)
    assert reference_riou([0,0,2,2,0],[10,0,2,2,0])==0
    assert math.isclose(reference_riou([0,0,2,2,0],[0,0,2,2,math.pi/4]),1/math.sqrt(2),abs_tol=1e-10)
