"""Fixed E-H isolation and actual head/gradient contracts; CUDA uses a CPU fixture."""
from copy import deepcopy
from types import MethodType

import pytest
import torch

from crane_project.tools.preflight_port_shape_e_h_v1 import (
    check_configs, integration_batch, initialize_matched_models, EXPECTED)
from crane_project.tools.preflight_port_center_size_v1 import set_assignment_phase
from crane_project.tools.preflight_port_shape_e_v1 import capture_main_positives
from crane_project.tools.diagnose_port_center_size_d_v1 import parameter_digest
from mmrotate.models.losses.covariance_shape_loss import CovarianceShapeLoss


def cpu_head(cfg):
    from mmrotate.models import build_head
    spec=deepcopy(cfg.model.bbox_head)
    spec.train_cfg,spec.test_cfg=cfg.model.train_cfg,cfg.model.test_cfg
    head=build_head(spec)
    def anchors(self,sizes,metas,device='cpu'):
        grids=self.anchor_generator.grid_priors(sizes,device='cpu')
        flags=[self.anchor_generator.valid_flags(sizes,m['pad_shape'],device='cpu') for m in metas]
        return [grids for _ in metas],flags
    head.get_anchors=MethodType(anchors,head)
    return head


def test_config_state_and_same_seed_constructor_identity():
    from mmrotate.models import build_detector
    b,e=check_configs()
    assert e.model.bbox_head.shape_compensation==EXPECTED
    assert e.load_from is None and e.resume_from is None and e.runner.max_epochs==24
    torch.manual_seed(0);baseline=build_detector(deepcopy(b.model))
    keys=list(baseline.state_dict());digest=parameter_digest(baseline)
    del baseline
    torch.manual_seed(0);model=build_detector(deepcopy(e.model))
    assert list(model.state_dict())==keys and parameter_digest(model)==digest
    assert model.bbox_head.center_size_compensation is None
    assert isinstance(model.bbox_head.shape_compensation,CovarianceShapeLoss)


@pytest.mark.parametrize('phase',['warmup_o2m','late_o2o'])
def test_real_head_original_losses_inference_actual_nodes_and_complete_gradient(phase):
    b,e=check_configs()
    torch.manual_seed(0);heads=[cpu_head(b),cpu_head(e)]
    heads[1].load_state_dict(heads[0].state_dict(),strict=True)
    feats=[torch.randn(1,256,n,n,requires_grad=True) for n in (8,4,2,1,1)]
    gt=[torch.tensor([[30.,30.,32.,16.,.2]])]
    metas=[dict(img_shape=(64,64,3),pad_shape=(128,128,3),scale_factor=[1.,1.,1.,1.])]
    rows,predictions=[],[]
    for head in heads:
        head.train();set_assignment_phase(head,phase)
        outs=head(feats)
        grids=head.anchor_generator.grid_priors([f.shape[-2:] for f in feats],device='cpu')
        predictions.append(head._get_bboxes_single([x[0].detach() for x in outs[0]],
            [x[0].detach() for x in outs[1]],grids,(64,64,3),[1.,1.,1.,1.],head.test_cfg,rescale=True))
        with capture_main_positives(head) as nodes:
            rows.append(head.loss(*outs,gt,[torch.zeros(1,dtype=torch.long)],metas))
        if head is heads[1]:actual=nodes
    assert all(torch.equal(a,b) for a,b in zip(*predictions))
    assert set(rows[1])-set(rows[0])=={'loss_shape_compensation','shape_positive_count'}
    assert all(torch.equal(a,b) for k in ('loss_cls','loss_bbox') for a,b in zip(rows[0][k],rows[1][k]))
    extra=sum(rows[1]['loss_shape_compensation'])
    independent=sum(heads[1].shape_compensation(p,t,weight=w,avg_factor=n) for p,t,w,n,_ in actual)
    assert torch.equal(extra,independent) and extra>0
    grads=torch.autograd.grad(extra,[x[0] for x in actual],retain_graph=True,allow_unused=True)
    for g,(p,*_) in zip(grads,actual):
        assert g is not None
        assert torch.count_nonzero(g[:,:2])==0
    assert sum(float(g[:,2:4].abs().sum()) for g in grads)>0
    assert sum(float(g[:,4].abs().sum()) for g in grads)>0
    cls=torch.autograd.grad(extra,heads[1].retina_cls.weight,retain_graph=True,allow_unused=True)[0]
    assert cls is None
    sum(sum(v) for k,v in rows[1].items() if 'loss' in k).backward()
    assert all(torch.isfinite(f.grad).all() for f in feats if f.grad is not None)


def test_empty_positive_level_and_mixed_d_e_rejected():
    from mmrotate.models import build_head
    _,e=check_configs();head=cpu_head(e)
    cls=torch.zeros(1,3,1,1,requires_grad=True)
    reg=torch.zeros(1,15,1,1,requires_grad=True)
    anchors=torch.tensor([[16.,16.,32.,16.,0.]]).repeat(3,1)
    result=head.loss_single(cls,reg,anchors,torch.ones(3,dtype=torch.long),torch.ones(3),
        torch.zeros(3,5),torch.zeros(3,5),num_total_samples=1)
    assert len(result)==7 and float(result[6])==0
    result[6].backward();assert torch.isfinite(reg.grad).all() and torch.count_nonzero(reg.grad)==0
    spec=deepcopy(e.model.bbox_head)
    spec.center_size_compensation=dict(type='CenterSizeCompensationLoss',loss_weight=.25)
    with pytest.raises(ValueError,match='separate experiments'):build_head(spec)


class Tiny(torch.nn.Module):
    """Real SymEOODHead; tiny shared features replace ResNet/FPN for CPU I/O."""
    def __init__(self,cfg):
        import torch.nn as nn
        super().__init__()
        self.bbox_head=cpu_head(cfg)
        self.neck=nn.Conv2d(3,256,1)
    def init_weights(self):
        self.bbox_head.init_weights()
        torch.nn.init.normal_(self.neck.weight,std=.01)
    def forward(self,img,img_metas,gt_bboxes,gt_labels,return_loss=True):
        import torch.nn.functional as F
        x=self.neck(img)
        feats=[F.adaptive_avg_pool2d(x,(n,n)) for n in (8,4,2,1,1)]
        result=self.bbox_head.loss(*self.bbox_head(feats),gt_bboxes,gt_labels,img_metas)
        result['aux0_loss_fixture']=x.square().mean()*.01
        return result


def test_integrated_probe_cli_and_original_initialization_with_cpu_io(tmp_path,monkeypatch):
    import json
    import sys
    import mmcv.parallel
    import mmrotate.models
    import mmrotate.datasets
    from mmdet.datasets.dataset_wrappers import ConcatDataset
    import crane_project.tools.ckpt_sweep as sweep
    from crane_project.tools.preflight_port_shape_e_h_v1 import main
    b,e=check_configs()
    monkeypatch.setattr(mmrotate.models,'build_detector',lambda cfg:Tiny(b if not cfg.bbox_head.get('shape_compensation') else e))
    model,digest=initialize_matched_models(b,e)
    assert parameter_digest(model)==digest
    class Leaf:
        CLASSES=('grab',)
        def __init__(self,domain,count):
            self.domain=domain
            self.data_infos=[dict(filename=domain+'_seq01_'+str(i)+'.jpg') for i in range(count)]
        def __len__(self):return len(self.data_infos)
        def __getitem__(self,i):
            path=tmp_path/self.data_infos[i]['filename']
            if not path.exists():path.write_bytes(b'CPU fixture')
            return dict(img=torch.ones(3,16,16),gt_bboxes=torch.tensor([[30.,30.,32.,16.,.2]]),
                gt_labels=torch.zeros(1,dtype=torch.long),img_metas=dict(filename=str(path),
                img_shape=(64,64,3),pad_shape=(128,128,3),scale_factor=[1.,1.,1.,1.],flip=False))
    ds=ConcatDataset([Leaf('real',1810),Leaf('sim',748)])
    def collate(items,samples_per_gpu=2):
        return dict(img=torch.stack([x['img'] for x in items]),
            **{k:[x[k] for x in items] for k in ('gt_bboxes','gt_labels','img_metas')})
    monkeypatch.setattr(mmrotate.datasets,'build_dataset',lambda *a:ds)
    monkeypatch.setattr(mmcv.parallel,'collate',collate)
    monkeypatch.setattr(mmcv.parallel,'scatter',lambda data,devices:[data])
    monkeypatch.setattr(torch.nn.Module,'cuda',lambda self,*a:self)
    for name in ('set_device','manual_seed_all','reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda,name,lambda *a:None)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *a:'CPU fixture; CUDA unverified')
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *a:0)
    monkeypatch.setattr(sweep,'annotation_set_sha256',lambda *a:'CPU fixture')
    out=tmp_path/'init.json'
    monkeypatch.setattr(sys,'argv',['preflight','--out-json',str(out)])
    main();report=json.loads(out.read_text())
    assert report['status']=='INITIALIZATION_CHECK_COMPLETE_REVIEW_REQUIRED'
    assert report['formal_formula_and_coefficient']=='FROZEN'
    assert report['train']['seed']==0 and report['train']['optimizer_steps']==0
    assert report['train']['b_e_initialization_equal'] and report['train']['parameters_unchanged']
    assert len(report['train']['rows'])==4
    assert len(out.with_suffix('.progress.jsonl').read_text().splitlines())==4
    for row in report['train']['rows']:
        assert row['actual_node_reuse'] and row['base_losses_unchanged']
        assert 'aux0_loss_fixture' in row['losses']
        assert row['direct_coordinate_norms']['x']==row['direct_coordinate_norms']['y']==0
        assert row['gradients']['full_fpn']['norms']['shape']>0
        assert row['clipping']['complete_e_h']['before']>0
    with pytest.raises(FileExistsError):main()


def test_observer_restores_on_exception():
    from crane_project.tools.preflight_port_shape_e_h_v1 import observe_integration
    _,e=check_configs();head=cpu_head(e)
    original=head.loss_single;extra=head.shape_compensation.forward
    with pytest.raises(RuntimeError):
        with observe_integration(head):raise RuntimeError('fixture')
    assert head.loss_single==original and head.shape_compensation.forward==extra


def test_val_denominators_and_explicit_e_labels():
    from crane_project.tools.compare_port_shape_e_h_val_v1 import coverage, rename_e
    rows=[dict(metrics=dict(output=True,center_hit=True)),
          dict(metrics=dict(output=True,center_hit=False)),
          dict(metrics=dict(output=False,center_hit=False))]
    result=coverage(rows)
    assert result['output_center_hit_pct']==50.
    assert result['output_coverage_pct']==pytest.approx(200/3)
    assert result['all_frame_center_correct_coverage_pct']==pytest.approx(100/3)
    assert rename_e(dict(d=dict(d_minus_b=1),compensated_d_strata=2))==dict(e_h=dict(e_h_minus_b=1),shape_e_h_strata=2)


def test_val_comparison_checks_existing_artifacts_without_inference(tmp_path,monkeypatch):
    import json
    import sys
    import crane_project.tools.compare_port_shape_e_h_val_v1 as comparison
    from crane_project.tools.audit_port_train_val_geometry_v1 import decompose
    epochs=[16,18,20,22,24]
    base=dict(candidate_epochs=epochs,selected_checkpoint='epoch_24',selection_config=dict(mcml_limit=5),
        metric_protocol_version='fixture',center_thresh_px=15.,selection_info=dict(selection='fallback_fixture'),
        all_checkpoints={'epoch_'+str(x):{} for x in epochs})
    for name in ('b','e'):
        folder=tmp_path/name;folder.mkdir();(folder/'sweep_results.json').write_text(json.dumps(base))
    gt=[0.,0.,40.,20.,.2]
    def row(i,output,center_ok):
        pred=gt+[.9] if output else None
        if output and not center_ok:pred=[20.,0.,40.,20.,.2,.9]
        return dict(image='real_seq07_'+str(i),image_sha256='fixture',gt=gt,pred=pred,
            domain='real',sequence='real_seq07',frame_id=i,split='val',input_gt_short_px=20.,
            gt_aspect=2.,gt_angle_deg=11.,metrics=decompose(gt,pred[:5] if pred else None))
    rows=[row(0,True,True),row(1,True,False),row(2,False,False)]
    rows.append(dict(rows[0],image='sim_seq10_0',domain='sim',sequence='sim_seq10'))
    monkeypatch.setattr(comparison,'load_frozen_val',lambda *args:(deepcopy(rows),dict(selected_epoch='epoch_24')))
    out=tmp_path/'compare.json'
    monkeypatch.setattr(sys,'argv',['compare','--b-sweep',str(tmp_path/'b'),'--e-sweep',str(tmp_path/'e'),'--out-json',str(out)])
    comparison.main();result=json.loads(out.read_text())
    assert result['selection_info']['selection']=='fallback_fixture'
    assert result['pre_registered_conditions']['coverage']['e_h']['real']['output_center_hit_pct']==50.
    assert not result['pre_registered_conditions']['all_conditions_met']
    assert 'e_h' in result['geometry']['rows'] and 'd' not in result['geometry']['rows']
    assert result['riou_crosscheck']['e_h']['output_pairs']==3
    with pytest.raises(FileExistsError):comparison.main()
