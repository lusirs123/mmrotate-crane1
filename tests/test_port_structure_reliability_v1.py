"""Coordinate/label/gradient and isolated preflight logic, CPU torch 1.8.

The tiny detector below checks the runner contract, not real B performance.
Actual B/checkpoint/CUDA integration is exclusively the server preflight.
"""
from copy import deepcopy
import inspect
import math

import numpy as np
import pytest
import torch

from crane_project.utils import port_structure_reliability_v1 as core
from crane_project.tools import preflight_port_structure_reliability_v1 as runner


def meta(sx=.5, sy=.5, flip=False, direction='horizontal'):
    return dict(scale_factor=np.array([sx, sy, sx, sy], dtype=np.float32),
                ori_shape=(100, 200, 3), img_shape=(50, 100, 3),
                pad_shape=(64, 128, 3), flip=flip, flip_direction=direction)


@pytest.mark.parametrize('direction', ['horizontal', 'vertical', 'diagonal'])
@pytest.mark.parametrize('size_mode', ['detector', 'annotation'])
def test_roundtrip_reflection_and_rounded_keep_ratio(direction, size_mode):
    m = meta(.5, .501, True, direction)
    b = torch.tensor([[50., 40., 60., 20., .3]], dtype=torch.float64)
    p = torch.tensor([[20., 30.], [75., 45.]], dtype=torch.float64)
    mapped = core.map_boxes(b, m, size_mode=size_mode)
    restored = core.map_boxes(mapped, m, inverse=True, size_mode=size_mode)
    assert torch.allclose(b[:, :4], restored[:, :4], atol=1e-12)
    assert core.quality_targets_original(restored, b)[2][0, 2] < 1e-12
    assert torch.allclose(core.map_points(core.map_points(p, m), m, inverse=True), p, atol=1e-12)
    if direction in ('horizontal', 'diagonal'):
        assert mapped[0, 0] == pytest.approx(100-1-25)


def test_detector_and_annotation_size_conventions_are_distinguished():
    m = meta(.5, .501)
    b = torch.tensor([[60., 40., 60., 20., .3]], dtype=torch.float64)
    d, a = core.map_boxes(b, m), core.map_boxes(b, m, size_mode='annotation')
    assert d[0, 2] == 30 and d[0, 3] == pytest.approx(20*float(m['scale_factor'][1]))
    assert a[0, 2]/b[0, 2] == pytest.approx(a[0, 3]/b[0, 3])
    assert not torch.allclose(d, a)
    with pytest.raises(ValueError, match='Anisotropic'):
        core.map_boxes(b, meta(.5, .7))
    with pytest.raises(ValueError):
        core.map_boxes(b, dict(m, flip=True, flip_direction='other'))


def test_quality_equivalences_and_independent_component_probe_labels():
    gt = torch.tensor([[40., 30., 60., 20., .25]], dtype=torch.float64)
    exchanged = gt[:, [0, 1, 3, 2, 4]].clone(); exchanged[:, 4] += math.pi/2
    periodic = gt.clone(); periodic[:, 4] += math.pi
    q, mask, e = core.quality_targets_original(torch.cat((gt, exchanged, periodic)), gt)
    assert torch.allclose(q, torch.ones_like(q), atol=1e-12)
    assert torch.allclose(e, torch.zeros_like(e), atol=1e-12)
    assert bool((mask == 1).all())
    probes, names = core.component_probes_original(gt)
    targets, _, errors = core.quality_targets_original(probes, gt)
    assert len(probes) == len(names) == core.SETTINGS['probes_per_image'] == 13
    for name, target, error in zip(names, targets, errors):
        if name.startswith('center'):
            assert error[0] == 15 and target[0] == pytest.approx(math.exp(-1))
            assert error[1:].abs().max() < 1e-12
        elif 'size' in name:
            assert error[0] == 0 and error[1] == pytest.approx(.15)
            assert error[2] < 1e-12
        elif name.startswith('angle'):
            assert error[:2].abs().max() == 0 and error[2] == pytest.approx(5.)
    square = gt.clone(); square[:, 2] = 21
    assert core.quality_targets_original(gt, square)[1][:, 2].sum() == 0


def test_quality_targets_are_original_pixel_based_and_detached():
    gt = torch.tensor([[40., 30., 60., 20., .25]], requires_grad=True)
    b = gt.detach().clone(); b[:, 0] += 15; b.requires_grad_(True)
    q, mask, errors = core.quality_targets_original(b, gt)
    assert q[0, 0] == pytest.approx(math.exp(-1))
    assert not q.requires_grad and not errors.requires_grad and not mask.requires_grad
    m = meta(.5, .5)
    restored = core.map_boxes(core.map_boxes(b, m), m, inverse=True)
    assert torch.allclose(core.quality_targets_original(restored, gt)[0], q)


def test_unsigned_axis_and_padding_mask_have_effective_small_target_support():
    m = meta()
    axis = torch.tensor([[20., 20.], [24., 21.]])
    t, valid = core.response_targets(axis, (8, 16), m)
    reverse, _ = core.response_targets(axis.flip(0), (8, 16), m)
    assert torch.allclose(t, reverse, atol=1e-6)
    assert (t*valid).sum() > 0 and int(valid.sum()) == 7*13
    logits = torch.zeros_like(t, requires_grad=True)
    loss, _ = core.balanced_response_loss(logits, t, valid)
    loss.backward()
    assert float(loss) == pytest.approx(math.log(2))
    assert (logits.grad*(1-valid)).abs().max() == 0
    assert logits.grad.abs().sum() > 0
    changed = logits.detach() + (1-valid)*1000
    assert float(core.balanced_response_loss(changed, t, valid)[0]) == pytest.approx(float(loss))
    with pytest.raises(ValueError):
        core.response_targets(torch.zeros(2, 2), (8, 16), m)


def test_feature_sampling_uses_anchor_origin_and_square_context_without_stretch():
    feature = torch.arange(256., dtype=torch.float64).reshape(1, 1, 16, 16)
    points = core.feature_points((16, 16), feature)
    assert points[0, 0].tolist() == [0, 0] and points[3, 4].tolist() == [32, 24]
    b = torch.tensor([[48., 48., 32., 8., .3]], dtype=torch.float64)
    roi, support = core.sample_square(feature, b, torch.ones(1, 1, 16, 16, dtype=torch.float64))
    assert roi[0, 0, 4, 4] == pytest.approx(float(feature[0, 0, 6, 6]))
    assert support.min() == 1
    # The square sample spans 64px in BOTH image axes despite a 4:1 box.
    assert roi[0, 0, 4, 8]-roi[0, 0, 4, 0] == pytest.approx(8)
    assert roi[0, 0, 8, 4]-roi[0, 0, 0, 4] == pytest.approx(16*8)


def test_genuine_loss_weight_does_not_depend_on_thirteen_probe_count():
    pred = torch.tensor([[0., 0., 0.]]+[[1., 1., 1.]]*13, requires_grad=True)
    target = torch.ones_like(pred); mask = torch.ones_like(pred)
    value, parts = core.quality_loss(pred, target, mask, 1)
    assert value == .25 and parts['genuine'] == .5 and parts['probe'] == 0
    value.backward()
    assert pred.grad[0].abs().sum() > 0
    # Missing output has no fabricated genuine-quality supervision.
    value, parts = core.quality_loss(pred[1:], target[1:], mask[1:], 0)
    assert value == 0 and parts['genuine'] == 0


def test_branch_detaches_detector_input_and_box_score_and_updates_only_branch():
    torch.manual_seed(1701)
    p3 = torch.randn(1, 256, 8, 16, requires_grad=True)
    boxes = torch.tensor([[40., 30., 40., 15., .25]], requires_grad=True)
    scores = torch.tensor([.8], requires_grad=True)
    model = core.StructureComponentReliability()
    before = runner.state_digest(model)
    online = model(p3, boxes, scores, meta())
    t, valid = core.response_targets(torch.tensor([[20., 30.], [60., 30.]]), (8, 16), meta())
    loss = core.balanced_response_loss(online['response_logits'], t, valid)[0] + online['qualities'].square().mean()
    loss.backward()
    assert p3.grad is boxes.grad is scores.grad is None
    assert all(v > 0 for v in runner.norms(model).values())
    torch.optim.SGD(model.parameters(), lr=.001).step()
    assert runner.state_digest(model) != before
    empty = model(p3, boxes.detach()[:0], scores.detach()[:0], meta())
    assert empty['qualities'].shape == (0, 3)
    assert not any(isinstance(m, (torch.nn.BatchNorm2d, torch.nn.Dropout)) for m in model.modules())
    assert list(inspect.signature(model.forward).parameters) == ['p3', 'boxes_model', 'scores', 'meta']


def test_roi_control_has_matched_descriptor_and_quality_capacity():
    a = core.StructureComponentReliability(); b = core.StructureComponentReliability(use_structure=False)
    assert runner.state_digest(a)['parameter_count'] == runner.state_digest(b)['parameter_count'] == 52293
    b.load_state_dict(a.state_dict())
    feature = torch.randn(1, 256, 8, 16)
    boxes, scores = torch.tensor([[40., 30., 40., 15., .25]]), torch.tensor([.8])
    assert not torch.equal(a(feature, boxes, scores, meta())['qualities'], b(feature, boxes, scores, meta())['qualities'])


class TinyDetector(torch.nn.Module):
    """Contract fixture with real frozen BN/parameters, no MMRotate GPU ops."""
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 256, 1)
        self.bn = torch.nn.BatchNorm2d(256)
    def extract_feat(self, image):
        return [self.bn(self.conv(torch.nn.functional.avg_pool2d(image, 8)))]
    def simple_test_from_features(self, feats, metas, rescale=False):
        raw = np.array([[120., 100., 60., 20., .2, .8]], dtype=np.float32)
        if rescale:
            raw[:, :4] /= metas[0]['scale_factor']
        return [[raw]]


def test_runner_one_discarded_step_preserves_detector_and_raw_outputs(monkeypatch):
    monkeypatch.setattr(runner, 'seed_all', lambda: torch.manual_seed(1701))
    monkeypatch.setattr(runner, 'measured', lambda fn, gpu: (fn(), {'cpu_fixture_only': True}))
    monkeypatch.setattr(core.StructureComponentReliability, 'cuda', lambda self, gpu: self)
    detector = core.freeze_detector(TinyDetector())
    initial = runner.state_digest(detector)
    m = dict(meta(1., 1.), ori_shape=(512, 1024, 3), img_shape=(512, 1024, 3),
             pad_shape=(1024, 1024, 3), filename='/train/real_seq01_00000.jpg')
    gt = torch.tensor([[120., 100., 60., 20., .2]])
    axis = torch.tensor([[90., 100.], [150., 100.]])
    batch = dict(img=torch.randn(1, 3, 1024, 1024), img_metas=[m], gt_bboxes=[gt])
    sample = dict(image='real_seq01_00000', gt=gt[0].tolist(), native_axis=axis.tolist(),
                  obb_derived_axis=axis.tolist(), source='native_axis', split='train',
                  image_sha256='fixture', annotation_sha256='fixture')
    row = runner.probe_view(detector, initial, batch, sample, 1., False, 0)
    assert row['branch_optimizer_steps'] == 1 and row['branch_checkpoint_saved'] is False
    assert row['b_parameters_buffers_unchanged'] and row['b_raw_prediction_exactly_unchanged']
    assert row['genuine_count'] == 1 and row['probes'] == 13
    assert all(v > 0 for v in row['supervised_task_final_layer_gradient_norms'].values())
    assert runner.state_digest(detector) == initial
    core.assert_detector_frozen(detector)
    detector.bn.train()
    with pytest.raises(ValueError, match='eval'):
        core.assert_detector_frozen(detector)


def test_fixed_view_pipeline_is_copy_only_and_never_reads_val_or_test():
    cfg = runner.check_cfg()
    before = deepcopy(cfg.to_dict())
    view = runner.fixed_specs(cfg, .5, True)
    assert cfg.to_dict() == before
    assert len(runner.VIEWS) == 14 and len({n for n, _, _ in runner.VIEWS}) == 12
    for entry in view:
        assert entry.ann_file in ('train/annfiles/', 'train_sim/annfiles/')
        assert next(s for s in entry.pipeline if s['type'] == 'RRandomFlip')['flip_ratio'] == [1., 0., 0.]
        assert next(s for s in entry.pipeline if s['type'] == 'PortIsotropicShrink')['scale_range'] == (.5, .5)
    assert all(n.rsplit('_', 1)[0] in runner.ready.TRAIN_SPLITS for n, _, _ in runner.VIEWS)


def test_existing_evidence_is_not_overwritten_and_invalid_prediction_is_rejected(tmp_path):
    p = tmp_path/'existing.json'; p.write_text('keep')
    with pytest.raises(FileExistsError): runner.write_new(p, {'replace': True})
    assert p.read_text() == 'keep'
    assert runner.flatten_prediction([[np.empty((0, 6))]]).shape == (0, 6)
    with pytest.raises(ValueError): runner.flatten_prediction([[np.ones((2, 6))]])
    bad = torch.tensor([[0., 0., -1., 4., 0.]])
    with pytest.raises(ValueError): core.canonical_boxes(bad)
