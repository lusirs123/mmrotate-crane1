"""Compensation math, real head integration and exact B/D isolation (CPU)."""
from copy import deepcopy
import math
from pathlib import Path

import pytest
import torch

from mmrotate.models.losses.center_size_compensation import CenterSizeCompensationLoss
from crane_project.tools.preflight_port_center_size_v1 import (
    check_configs, fixed_train_specs, set_assignment_phase)


def test_exact_equivalent_boxes_and_angle_not_supervised():
    fn = CenterSizeCompensationLoss()
    target = torch.tensor([[10., 20., 40., 20., .4]])
    equivalent = torch.tensor([[10., 20., 20., 40., .4 + math.pi / 2]], requires_grad=True)
    loss = fn(equivalent, target)
    assert float(loss) == 0.
    loss.backward()
    assert torch.count_nonzero(equivalent.grad) == 0
    # Different angles alone are intentionally left to SymKLD.
    other = target.clone(); other[:,4] += .8
    assert float(fn(other, target)) == 0.


def test_closed_form_center_error_and_uniform_scale_invariance():
    fn = CenterSizeCompensationLoss()
    target = torch.tensor([[10., 20., 40., 20., .4]], dtype=torch.double)
    pred = target.clone(); pred[0,0] += 2.
    # x residual=.1, y=0: center mean SmoothL1=.025, weighted=.00625.
    assert float(fn(pred, target)) == pytest.approx(.00625)
    pred[0,2:4] *= 1.2
    for scale in [.5, 2.]:
        a, b = pred.clone(), target.clone()
        a[:,:4] *= scale; b[:,:4] *= scale
        assert torch.allclose(fn(a,b),fn(pred,target))


def test_gradients_only_prediction_center_size_and_gradient_check():
    fn = CenterSizeCompensationLoss()
    target = torch.tensor([[10.,20.,40.,20.,.4]], dtype=torch.double, requires_grad=True)
    pred = torch.tensor([[12.,21.,44.,18.,.9]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: fn(x, target), (pred,))
    gp, gt = torch.autograd.grad(fn(pred,target), (pred,target), allow_unused=True)
    assert torch.count_nonzero(gp[:,:4]) == 4
    assert torch.count_nonzero(gp[:,4]) == 0
    assert gt is None


def test_empty_and_weighted_global_positive_normalization():
    fn = CenterSizeCompensationLoss()
    empty = torch.empty((0,5),requires_grad=True)
    fn(empty,empty).backward()
    assert empty.grad.shape == (0,5)
    gt = torch.tensor([[10.,20.,40.,20.,0.],[0.,0.,20.,10.,0.]])
    pred = gt.clone(); pred[:,0] += 2.; pred.requires_grad_()
    weights = torch.tensor([1.,0.])
    loss = fn(pred,gt,weight=weights,avg_factor=2.)
    assert float(loss) == pytest.approx(.00625/2)
    loss.backward()
    assert torch.count_nonzero(pred.grad[1]) == 0


@pytest.mark.parametrize('settings',[dict(loss_weight=0),dict(beta=0),dict(eps=-1),dict(loss_weight=float('nan'))])
def test_invalid_hyperparameters_rejected(settings):
    with pytest.raises(ValueError):
        CenterSizeCompensationLoss(**settings)


def test_actual_config_only_loss_diff_and_state_dict_identity():
    from mmrotate.models import build_detector
    b, d = check_configs()
    models = [build_detector(deepcopy(cfg.model)) for cfg in [b,d]]
    assert models[0].bbox_head.center_size_compensation is None
    assert isinstance(models[1].bbox_head.center_size_compensation, CenterSizeCompensationLoss)
    assert models[0].state_dict().keys() == models[1].state_dict().keys()
    models[1].load_state_dict(models[0].state_dict(),strict=True)
    for scale in [1.,.5]:
        specs = fixed_train_specs(d,scale)
        assert all(s['flip_ratio']==[0.,0.,0.] for e in specs for s in e.pipeline if s['type']=='RRandomFlip')
        assert all(s['scale_range']==(scale,scale) for e in specs for s in e.pipeline if s['type']=='PortIsotropicShrink')
    assert d.data.train[0].pipeline[3]['scale_range']==(.5,1.)


def test_head_training_integration_keeps_original_losses_and_gradient_scope():
    from mmrotate.models import build_head
    from types import MethodType
    b, d = check_configs()
    heads = []
    for cfg in [b,d]:
        spec = deepcopy(cfg.model.bbox_head)
        spec.train_cfg, spec.test_cfg = cfg.model.train_cfg, cfg.model.test_cfg
        head = build_head(spec)
        head.train()
        set_assignment_phase(head,'late_o2o')
        # The legacy get_anchors passes device positionally as grid_priors'
        # dtype argument; on this CPU-only host explicitly use the CPU device.
        # Target assignment, decoding and loss integration remain real code.
        def cpu_anchors(self, sizes, metas, device='cpu'):
            anchors = self.anchor_generator.grid_priors(sizes, device='cpu')
            flags = [self.anchor_generator.valid_flags(sizes, meta['pad_shape'], device='cpu')
                     for meta in metas]
            return [anchors for _ in metas], flags
        head.get_anchors = MethodType(cpu_anchors,head)
        heads.append(head)
    heads[1].load_state_dict(heads[0].state_dict(),strict=True)
    torch.manual_seed(1701)
    feats = [torch.randn(1,256,n,n,requires_grad=True) for n in [8,4,2,1,1]]
    gt = [torch.tensor([[30.,30.,32.,16.,.2]])]
    meta = [dict(img_shape=(64,64,3),pad_shape=(128,128,3),scale_factor=[1.,1.,1.,1.])]
    values = []
    predictions = []
    for head in heads:
        outs = head(feats)
        sizes = [feat.shape[-2:] for feat in feats]
        anchors = head.anchor_generator.grid_priors(sizes,device='cpu')
        predictions.append(head._get_bboxes_single(
            [v[0].detach() for v in outs[0]], [v[0].detach() for v in outs[1]],
            anchors, (64,64,3), [1.,1.,1.,1.], head.test_cfg, rescale=True))
        values.append(head.loss(*outs, gt, [torch.zeros(1,dtype=torch.long)], meta))
    assert torch.equal(predictions[0][0],predictions[1][0])
    assert torch.equal(predictions[0][1],predictions[1][1])
    assert 'loss_center_size_compensation' not in values[0]
    for key in ['loss_cls','loss_bbox']:
        for a,c in zip(values[0][key],values[1][key]):
            assert torch.allclose(a,c)
    extra = sum(values[1]['loss_center_size_compensation'])
    assert extra > 0 and values[1]['center_size_positive_count'] == 1
    parameters = [heads[1].retina_reg.weight, heads[1].retina_cls.weight] + feats
    grads = torch.autograd.grad(extra,parameters,retain_graph=True,allow_unused=True)
    assert grads[0].abs().sum() > 0 and torch.count_nonzero(grads[0][4::5]) == 0
    assert grads[1] is None
    assert sum(g.abs().sum() for g in grads[2:] if g is not None) > 0
    # Full summed train objective must remain differentiable (including empty levels).
    sum(sum(v) for k,v in values[1].items() if 'loss' in k).backward()
    assert all(torch.isfinite(f.grad).all() for f in feats if f.grad is not None)


def test_no_positive_head_level_is_safe_and_zero():
    from mmrotate.models import build_head
    _,d = check_configs()
    spec = deepcopy(d.model.bbox_head)
    spec.train_cfg, spec.test_cfg = d.model.train_cfg,d.model.test_cfg
    head = build_head(spec)
    cls = torch.zeros(1,3,1,1,requires_grad=True)
    reg = torch.zeros(1,15,1,1,requires_grad=True)
    anchors = torch.tensor([[16.,16.,32.,16.,0.]]).repeat(3,1)
    result = head.loss_single(cls,reg,anchors,torch.ones(3,dtype=torch.long),
        torch.ones(3),torch.zeros(3,5),torch.zeros(3,5),num_total_samples=1)
    assert len(result)==7 and float(result[6])==0
    result[6].backward()
    assert torch.isfinite(reg.grad).all() and torch.count_nonzero(reg.grad)==0


def test_val_comparison_labels_and_frozen_artifact_loader(tmp_path,monkeypatch):
    """Exercise the complete VAL PKL/TXT loader with real dataset/GT metadata.

    Predictions are synthetic GT boxes to validate artifact plumbing, not to
    claim detector performance. No server weights or GPU are needed.
    """
    import json
    import pickle
    import numpy as np
    from crane_project.tools import compare_port_center_size_val_v1 as compare
    from crane_project.tools.ckpt_sweep import annotation_set_sha256,pkl_to_dota
    from mmrotate.datasets import build_dataset
    b,_ = check_configs()
    spec = deepcopy(b.data.val); spec.test_mode=True
    dataset = build_dataset(spec)
    predictions = [[np.column_stack((dataset.get_ann_info(i)['bboxes'],[.9]))]
                   for i in range(len(dataset))]
    images = [Path(info['filename']).stem for info in dataset.data_infos]
    pred_dir = tmp_path/'preds';pred_dir.mkdir()
    pkl = pred_dir/'results.pkl'
    with pkl.open('wb') as stream:pickle.dump(predictions,stream)
    pkl_to_dota(str(pkl),images,str(pred_dir))
    checkpoint = tmp_path/'epoch_24.pth';checkpoint.write_bytes(b'fixture')
    selection=dict(evidence_role='source_val_checkpoint_selection',selected_checkpoint='epoch_24',
        config_sha256=compare.sha(compare.CONTROL),selected_path=str(checkpoint),
        source_val_annotations_sha256=annotation_set_sha256(str(compare.ROOT/
            'crane_project/data/crane_grab_port_day2night_v1/val/annfiles')),
        all_checkpoints=dict(epoch_24=dict(checkpoint=str(checkpoint),
            checkpoint_sha256=compare.sha(checkpoint),results_pkl=str(pkl),
            results_pkl_sha256=compare.sha(pkl))))
    (tmp_path/'sweep_results.json').write_text(json.dumps(selection))
    # Provenance validator already has its own tests; this fixture intentionally
    # does not claim a real checkpoint produced the synthetic predictions.
    import crane_project.tools.ckpt_sweep as sweep
    monkeypatch.setattr(sweep,'check_or_record_prediction',lambda *args:None)
    rows,_ = compare.load_val(b,compare.CONTROL,tmp_path)
    assert len(rows)==887 and all(r['metrics']['riou']>.999 for r in rows)
    labels = compare.name_arms(compare.paired_report(rows,rows,True))
    assert labels['common_outputs']['compensated_d']['frames']==887
    assert labels['output_churn']['both_output']==887
    assert 'eood' not in labels and 'symeood' not in labels
