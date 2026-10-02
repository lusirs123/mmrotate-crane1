"""TRAIN qualification, denominator and cache-integrity regressions (CPU only)."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest
import torch

from crane_project.utils import port_structure_reliability_v1 as old
from crane_project.utils import port_reliability_train_policy_v1 as policy
from crane_project.tools import check_port_reliability_train_support_v1 as tool


def gt():
    return torch.tensor([[40., 30., 23.96, 20., .0]])


def qualification(domain='real', value=None):
    value = gt() if value is None else value
    b = value.numpy()[0]
    v = .5*b[2]*np.array([math.cos(b[4]), math.sin(b[4])])
    axis = np.stack((b[:2]-v, b[:2]+v))
    if domain == 'real':
        return policy.direction_qualification(b, domain, axis, 1.2)
    return policy.direction_qualification(b, domain)


def source(name='real_seq05_00000', domain='real', value=None):
    value = gt() if value is None else value
    return dict(image=name, domain=domain, sequence=name.rsplit('_', 1)[0],
                gt_original=value.numpy()[0].tolist(), direction_qualification=qualification(domain, value))


def test_valid_native_axis_bypasses_conversion_ratio_boundary_and_targets_unchanged():
    g = gt(); probes, _ = old.component_probes_original(g)
    previous, old_mask, errors = old.quality_targets_original(probes, g)
    target, mask, current = policy.train_quality_targets_original(probes, g, qualification())
    assert old_mask[:, 2].sum() == 0
    assert mask[:, 2].sum() == len(probes)
    assert torch.equal(previous, target) and torch.equal(errors, current)
    assert torch.equal(mask[:, :2], old_mask[:, :2])
    assert qualification()['evaluation_angle_eligible'] is False


def test_new_angle_loss_has_gradient_with_detached_labels_only():
    g = gt().requires_grad_(); probes, _ = old.component_probes_original(g)
    q = torch.full((13, 3), .5, requires_grad=True)
    target, mask, errors = policy.train_quality_targets_original(probes, g, qualification())
    loss, _ = old.quality_loss(q, target, mask, 0)
    loss.backward()
    assert q.grad[:, 2].norm() > 0 and torch.isfinite(q.grad).all()
    assert g.grad is None and not target.requires_grad and not errors.requires_grad


def test_existing_masked_mean_normalization_is_preserved_and_effect_is_explicit():
    g = gt(); probes, _ = old.component_probes_original(g)
    target, old_mask, _ = old.quality_targets_original(probes, g)
    _, new_mask, _ = policy.train_quality_targets_original(probes, g, qualification())
    q0 = torch.full((13, 3), .5, requires_grad=True)
    q1 = q0.detach().clone().requires_grad_()
    old.quality_loss(q0, target, old_mask, 0)[0].backward()
    old.quality_loss(q1, target, new_mask, 0)[0].backward()
    # Adding the third eligible component changes the shared mean denominator.
    assert torch.allclose(q1.grad[:, :2], q0.grad[:, :2]*2/3)
    assert q0.grad[:, 2].norm() == 0 and q1.grad[:, 2].norm() > 0


@pytest.mark.parametrize('bad', [None, [[0., 0.], [0., 0.]], [[np.nan, 0.], [2., 0.]],
                                  [[40., 30.], [65., 30.]]])
def test_missing_invalid_or_inconsistent_real_axis_fails_closed(bad):
    with pytest.raises(ValueError):
        policy.direction_qualification(gt()[0], 'real', bad, 1.2)


def test_native_endpoint_order_is_unsigned_and_not_visibility_supervision():
    q = qualification()
    b = gt()[0].numpy(); axis = tool.ready.box_axis(b)[::-1].copy()
    reverse = policy.direction_qualification(b, 'real', axis, 1.2)
    assert q == reverse
    assert 'visibility' not in q and 'axis_endpoints' not in q


def test_sim_webots_obb_requires_no_native_axis_and_retains_geometric_mask():
    short = gt()
    assert qualification('sim', short)['train_angle_eligible'] is False
    long = short.clone(); long[:, 2] = 60
    q = qualification('sim', long)
    assert q['train_angle_eligible'] is True
    assert q['direction_gt_source'] == 'webots_generated_obb_gt'
    assert 'proxy_no_native_axis' in q['structure_source']
    with pytest.raises(ValueError, match='Webots'):
        policy.direction_qualification(long[0], 'sim', [[10, 30], [70, 30]], 3.)


def test_qualification_cannot_be_used_with_a_different_original_gt():
    wrong = gt().clone(); wrong[:, 0] += 1
    with pytest.raises(ValueError, match='different original GT'):
        policy.train_quality_targets_original(wrong, wrong, qualification())


def test_transformed_view_uses_original_gt_qualification_and_keeps_pixel_targets():
    g = gt(); b = g.clone(); b[:, 0] += 15
    meta = dict(scale_factor=[.5, .5, .5, .5], ori_shape=(100, 200, 3),
                img_shape=(50, 100, 3), pad_shape=(64, 128, 3), flip=True,
                flip_direction='horizontal')
    restored = old.map_boxes(old.map_boxes(b, meta), meta, inverse=True)
    target, mask, _ = policy.train_quality_targets_original(restored, g, qualification())
    assert target[0, 0] == pytest.approx(math.exp(-1)) and mask[0, 2] == 1


def test_quality_pi_period_and_swapped_edges_preserved():
    g = gt(); swap = g[:, [0, 1, 3, 2, 4]].clone(); swap[:, 4] += math.pi/2
    target, _, errors = policy.train_quality_targets_original(swap, g, qualification())
    assert torch.allclose(target, torch.ones_like(target), atol=1e-6)
    assert errors[0, 2] < 1e-4


def test_missing_outputs_and_train_eval_angle_denominators_stay_separate():
    inputs = [source(), source('real_seq05_00001'), source('real_seq05_00002')]
    p = list(inputs[0]['gt_original']) + [.8]
    bad = p.copy(); bad[0] += 15; bad[4] += math.radians(5)
    s = tool.summarize(tool.support_rows(inputs, [p, bad, None]))
    assert s['output_coverage'] == pytest.approx(2/3)
    assert s['center_hit_rate_on_outputs'] == .5
    assert s['all_frame_center_correct_coverage'] == pytest.approx(1/3)
    assert s['center_bad_outputs'] == 1
    assert s['train_angle']['assessed_output_frames'] == 2
    assert s['train_angle']['bad_outputs'] == 1
    assert s['train_angle']['eligible_no_output_frames'] == 1
    assert s['evaluation_angle']['assessed_output_frames'] == 0
    assert s['evaluation_angle']['errors_deg']['rmse'] is None
    assert s['evaluation_angle']['unassessed_output_frames'] == 2


def test_empty_outputs_do_not_become_successful_angles_or_zero_rmse():
    s = tool.summarize(tool.support_rows([source()], [None]))
    assert s['center_hit_rate_on_outputs'] is None
    assert s['all_frame_center_correct_coverage'] == 0
    assert s['train_angle']['errors_deg']['rmse'] is None


@pytest.mark.parametrize('pred', [[1, 2, -3, 4, 0, .9], [1, 2, 3, 4, 0, .05],
                                  [1, 2, 3, 4, np.nan, .9], [1, 2, 3, 4, 0, 1.01]])
def test_cache_accepts_only_valid_frozen_b_top1_outputs(pred):
    with pytest.raises(ValueError):
        tool.checked_prediction(pred)


def cache_fixture(tmp_path):
    inputs = [source(), source('real_seq05_00001')]
    identity = dict(protocol=tool.VERSION, frozen_b=dict(checkpoint_sha256='frozen'))
    raw = tmp_path/'predictions.jsonl'
    rows = [dict(index=i, image=s['image'], input_sha256=tool.fingerprint(s),
                 pred_original=s['gt_original']+[.8] if i == 0 else None) for i, s in enumerate(inputs)]
    raw.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    complete = dict(identity_sha256=tool.fingerprint(identity), frames=2,
                    prediction_file_sha256=tool.ready.sha(raw), b_state_before={'params':'abc'},
                    b_state_after={'params':'abc'}, detector_optimizer_steps=0, checkpoint_sha256='frozen')
    (tmp_path/'identity.json').write_text(json.dumps(identity))
    (tmp_path/'complete.json').write_text(json.dumps(complete))
    return inputs, identity, complete, rows


def test_complete_verified_cache_reuse_is_cpu_only(tmp_path, monkeypatch):
    inputs, identity, _, _ = cache_fixture(tmp_path)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda *a: pytest.fail('CPU cache reuse touched CUDA'))
    preds, _ = tool.read_cache(tmp_path, identity, inputs)
    assert len(preds) == 2 and preds[1] is None


@pytest.mark.parametrize('change', ['prediction', 'source', 'identity', 'order', 'state', 'checkpoint'])
def test_cache_tamper_changed_gt_or_model_is_rejected(tmp_path, change):
    inputs, identity, complete, rows = cache_fixture(tmp_path)
    if change == 'prediction':
        rows[0]['pred_original'][0] += 1
    elif change == 'source':
        inputs[0]['gt_original'][0] += 1
    elif change == 'identity':
        identity['protocol'] += '_changed'
    elif change == 'order':
        rows.reverse()
    elif change == 'state':
        complete['b_state_after'] = {'params':'different'}
    elif change == 'checkpoint':
        complete['checkpoint_sha256'] = 'other'
    if change in ('prediction', 'order'):
        (tmp_path/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        if change == 'order':
            complete['prediction_file_sha256'] = tool.ready.sha(tmp_path/'predictions.jsonl')
    (tmp_path/'complete.json').write_text(json.dumps(complete))
    with pytest.raises(ValueError):
        tool.read_cache(tmp_path, identity, inputs)


def test_inference_uses_only_train_paths_and_unchanged_keep_ratio_pipeline():
    cfg = tool.prior.check_cfg()
    original = deepcopy(cfg.data)
    specs, pipeline = tool.inference_specs(cfg)
    assert cfg.data == original
    assert pipeline == cfg.data.val.pipeline
    assert [s['ann_file'] for s in specs] == ['train/annfiles/', 'train_sim/annfiles/']
    assert all(s['test_mode'] for s in specs)
    cfg.data.train[0]['ann_file'] = 'test/annfiles/'
    with pytest.raises(ValueError, match='allowlisted TRAIN'):
        tool.inference_specs(cfg)


def test_collect_loop_freezes_model_uses_native_rescale_and_seals_reusable_cache(tmp_path, monkeypatch):
    """Exercise the writer/loop on a tiny CPU fixture, not B/GPU performance."""
    from types import SimpleNamespace
    import mmcv.parallel
    import mmcv.runner
    import mmrotate.datasets
    import mmrotate.models

    inputs = [source(), source('real_seq05_00001')]
    for value in inputs:
        value['image_size'] = [200, 100]
    checkpoint = tmp_path/'fixture.pth'; checkpoint.write_bytes(b'fixture')
    checksum = tool.ready.sha(checkpoint)
    monkeypatch.setitem(tool.ready.FROZEN_B, 'checkpoint_sha256', checksum)
    identity = dict(protocol=tool.VERSION, frozen_b=dict(checkpoint_sha256=checksum))

    class Part:
        def __init__(self, r):
            self.row = r; self.data_infos = [dict(filename=r['image']+'.jpg')]
        def __len__(self):
            return 1
        def get_ann_info(self, index):
            return dict(bboxes=np.asarray([self.row['gt_original']], dtype=np.float32))
        def __getitem__(self, index):
            return dict(img=[torch.zeros(1, 3, 100, 200)], img_metas=[[dict(
                filename=self.row['image']+'.jpg', scale_factor=[1.]*4,
                ori_shape=(100, 200, 3), img_shape=(100, 200, 3), pad_shape=(100, 200, 3), flip=False)]])

    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.bn = torch.nn.BatchNorm1d(2); self.calls = 0
        def cuda(self, index):
            return self
        def forward(self, return_loss, rescale, img, img_metas):
            assert return_loss is False and rescale is True and len(img) == 1
            assert not torch.is_grad_enabled()
            tool.assert_detector_frozen(self)
            self.bn(torch.ones(1, 2))
            self.calls += 1
            p = np.asarray([inputs[0]['gt_original']+[.8]], dtype=np.float32)
            return [[p if self.calls == 1 else np.zeros((0, 6), dtype=np.float32)]]

    detector = Detector()
    monkeypatch.setattr(mmrotate.models, 'build_detector', lambda cfg: detector)
    monkeypatch.setattr(mmrotate.datasets, 'build_dataset', lambda spec: Part(inputs[spec]))
    monkeypatch.setattr(mmcv.runner, 'load_checkpoint', lambda *a, **k: dict(meta=dict(epoch=24)))
    monkeypatch.setattr(mmcv.parallel, 'collate', lambda items, **k: items[0])
    monkeypatch.setattr(mmcv.parallel, 'scatter', lambda batch, gpu: [batch])
    for name in ('set_device', 'manual_seed_all', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *a: 'CPU_fixture_no_GPU_measurement')
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *a: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda *a: 0)
    cfg = SimpleNamespace(model={}, custom_imports=dict(imports=[], allow_failed_imports=False))
    predictions, complete = tool.collect(tmp_path/'cache', identity, inputs, cfg, [0, 1], checkpoint, 0)
    assert detector.calls == 2 and predictions[1] is None
    assert complete['b_state_before'] == complete['b_state_after']
    assert tool.summarize(tool.support_rows(inputs, predictions))['output_coverage'] == .5
    reused, _ = tool.read_cache(tmp_path/'cache', identity, inputs)
    assert reused == predictions
    with pytest.raises(FileExistsError):
        tool.collect(tmp_path/'cache', identity, inputs, cfg, [0, 1], checkpoint, 0)
