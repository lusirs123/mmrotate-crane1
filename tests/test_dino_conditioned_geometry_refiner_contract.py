"""CPU-only contract tests for the shared five-delta refiner."""

import importlib.util
import math
import pathlib
import sys
import types

import pytest
import torch
import torch.nn as nn

from crane_project.utils.dual_tower_geometry_refiner_checkpoint import (
    compose_dual_tower_state)


class _Registry:
    def register_module(self, *args, **kwargs):
        del args, kwargs
        return lambda cls: cls


class _Coder:
    def encode(self, proposals, gt):
        px, py, pw, ph, pa = proposals.unbind(-1)
        gx, gy, gw, gh, ga = gt.unbind(-1)
        dx = (torch.cos(pa) * (gx - px) + torch.sin(pa) * (gy - py)) / pw
        dy = (-torch.sin(pa) * (gx - px) + torch.cos(pa) * (gy - py)) / ph
        return torch.stack(
            [dx, dy, torch.log(gw / pw), torch.log(gh / ph), ga - pa],
            dim=-1)

    def decode(self, proposals, deltas, max_shape=None):
        del max_shape
        px, py, pw, ph, pa = proposals.unbind(-1)
        dx, dy, dw, dh, da = deltas.unbind(-1)
        gx = px + dx * pw * torch.cos(pa) - dy * ph * torch.sin(pa)
        gy = py + dx * pw * torch.sin(pa) + dy * ph * torch.cos(pa)
        return torch.stack(
            [gx, gy, pw * torch.exp(dw), ph * torch.exp(dh), pa + da],
            dim=-1)


class _RoIExtractor(nn.Module):
    def __init__(self, out_channels=1, out_size=1):
        super().__init__()
        self.out_channels = out_channels
        self.out_size = out_size

    def forward(self, features, rois):
        return features[0].new_ones(
            (rois.shape[0], self.out_channels,
             self.out_size, self.out_size))


def _rbbox2roi(boxes):
    rows = []
    for index, item in enumerate(boxes):
        rows.append(torch.cat([
            item.new_full((item.shape[0], 1), index), item], dim=1))
    return torch.cat(rows, dim=0)


def _load_module():
    names = (
        'mmrotate', 'mmrotate.core', 'mmrotate.models',
        'mmrotate.models.builder', 'mmrotate.models.losses',
        'mmrotate.models.losses.sym_kld_calculator',
        'mmrotate.models.roi_heads')
    modules = {name: types.ModuleType(name) for name in names}
    for module in modules.values():
        module.__path__ = []
    modules['mmrotate.core'].build_bbox_coder = lambda cfg: _Coder()
    modules['mmrotate.core'].norm_angle = lambda value, version: value
    modules['mmrotate.core'].rbbox2roi = _rbbox2roi
    modules['mmrotate.models.builder'].ROTATED_HEADS = _Registry()
    modules['mmrotate.models.builder'].build_roi_extractor = lambda cfg: (
        _RoIExtractor(cfg['out_channels'], cfg['roi_layer']['out_size']))
    modules['mmrotate.models.losses.sym_kld_calculator'].sym_kld = (
        lambda pred, target: ((pred - target) ** 2).sum(dim=-1))
    old = {name: sys.modules.get(name) for name in names}
    sys.modules.update(modules)
    try:
        root = pathlib.Path(__file__).resolve().parents[1]
        path = root / ('mmrotate/models/roi_heads/'
                       'dino_conditioned_geometry_refiner.py')
        name = ('mmrotate.models.roi_heads.'
                'dino_conditioned_geometry_refiner')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in old.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


MODULE = _load_module()
Refiner = MODULE.DinoConditionedGeometryRefiner
DualRefiner = MODULE.DinoConditionedDualTowerGeometryRefiner
CausalRefiner = MODULE.DinoConditionedCausalHistoryRefiner
K1PhaseRefiner = MODULE.K1AnchoredCausalPhaseGeometryRefiner
K1RetentiveRefiner = MODULE.K1RetentiveCausalPhaseGeometryRefiner


def _refiner(**kwargs):
    return Refiner(
        in_channels=1, roi_output_size=1, fc_channels=4, num_fcs=1,
        **kwargs)


def _dual_refiner(**kwargs):
    return DualRefiner(
        in_channels=1, roi_output_size=1, fc_channels=4, num_fcs=1,
        **kwargs)


def _causal_refiner(**kwargs):
    return CausalRefiner(
        in_channels=1, roi_output_size=1, fc_channels=4, num_fcs=1,
        history_horizon=2, **kwargs)


def _k1_phase_refiner(**kwargs):
    return K1PhaseRefiner(
        in_channels=1, roi_output_size=1, fc_channels=4, num_fcs=1,
        history_horizon=2, **kwargs)


def _k1_retentive_refiner(**kwargs):
    return K1RetentiveRefiner(
        in_channels=1, roi_output_size=1, fc_channels=4, num_fcs=1,
        history_horizon=2, **kwargs)


def test_zero_initialization_is_exact_canonical_dino_identity():
    model = _refiner()
    proposal = torch.tensor([[10., 12., 8., 4., 0.2]])
    features = [torch.zeros((1, 1, 4, 4))]
    deltas = model(features, [proposal])
    decoded = model.decode_and_normalize([proposal], deltas)[0]
    assert torch.equal(deltas, torch.zeros_like(deltas))
    assert torch.equal(decoded, proposal)


def test_size_only_mask_blocks_center_and_angle_outputs_and_gradients():
    model = _refiner(
        refine_center=False, refine_size=True, refine_angle=False)
    raw = torch.tensor(
        [[1., 2., 3., 4., 5.]], requires_grad=True)
    masked = model.mask_deltas(raw)
    assert masked.tolist() == [[0., 0., 3., 4., 0.]]
    masked.sum().backward()
    assert raw.grad.tolist() == [[0., 0., 1., 1., 0.]]


def test_edge_swap_canonicalizes_equivalent_le90_box():
    box = torch.tensor([[5., 6., 4., 8., -0.4]])
    canonical = MODULE.canonicalize_le90(box)
    assert canonical[0, 2:4].tolist() == pytest.approx([8., 4.])
    expected = ((-0.4 + math.pi / 2 + math.pi / 2) % math.pi
                - math.pi / 2)
    assert canonical[0, 4].item() == pytest.approx(expected)


@pytest.mark.parametrize('direction', ['horizontal', 'vertical', 'diagonal'])
def test_model_coordinate_mapping_round_trip_for_every_flip(direction):
    box = torch.tensor([[30., 20., 12., 6., 0.3]])
    meta = dict(
        scale_factor=[2., 2., 2., 2.],
        img_shape=(100, 160, 3),
        flip=True,
        flip_direction=direction)
    mapped = MODULE.map_original_obb_to_model(box, meta)
    restored = MODULE.map_model_obb_to_original(mapped, meta)
    assert restored == pytest.approx(box, abs=1e-5)


def test_angle_loss_is_pi_periodic():
    model = _refiner()
    predicted = torch.zeros((1, 5))
    target = torch.zeros((1, 5))
    target[0, 4] = math.pi
    losses = model.loss(predicted, target)
    assert losses['refiner_angle_objective'].item() == pytest.approx(0.0)
    assert losses['loss_geometry_refiner'].item() == pytest.approx(0.0)


def test_dual_tower_zero_initialization_is_exact_dino_identity():
    model = _dual_refiner()
    proposal = torch.tensor([[10., 12., 8., 4., 0.2]])
    features = [torch.zeros((1, 1, 4, 4))]
    deltas = model(features, [proposal])
    decoded = model.decode_and_normalize([proposal], deltas)[0]
    assert torch.equal(deltas, torch.zeros_like(deltas))
    assert torch.equal(decoded, proposal)
    contract = model.component_contract()
    assert contract['architecture'] == 'dual_tower_size_pose_v2'
    assert contract['trainable_fc_towers_shared'] is False


def test_dual_tower_rejects_inactive_component_configuration():
    with pytest.raises(ValueError, match='requires center, size, and angle'):
        _dual_refiner(refine_angle=False)


def test_dual_tower_allows_fully_frozen_evaluation_only_runtime():
    model = _dual_refiner(
        train_size_tower=False,
        train_pose_tower=False,
        train_roi_extractor=False,
        evaluation_only=True)
    assert all(not parameter.requires_grad
               for parameter in model.parameters())
    assert model.component_contract()['evaluation_only'] is True


def test_dual_tower_rejects_fully_frozen_training_mode():
    with pytest.raises(ValueError, match='branch must be trainable'):
        _dual_refiner(
            train_size_tower=False,
            train_pose_tower=False,
            train_roi_extractor=False)


def test_causal_history_no_valid_input_is_exact_current_only_output():
    model = _causal_refiner()
    current_features = [torch.zeros((1, 1, 4, 4))]
    history_features = [torch.ones((1, 2, 1, 4, 4))]
    proposals = [torch.tensor([[10., 12., 8., 4., 0.2]])]
    history_boxes = torch.tensor([[
        [9., 12., 8., 4., 0.2],
        [8., 12., 8., 4., 0.2]]])
    current = model(current_features, proposals)
    causal = model(
        current_features, proposals,
        history_features=history_features,
        history_proposals=history_boxes,
        history_valid_mask=torch.zeros((1, 2), dtype=torch.bool),
        history_ages=torch.tensor([[1, 2]]))
    assert torch.equal(causal, current)
    assert torch.equal(causal, torch.zeros_like(causal))


def test_causal_history_residual_is_bounded_and_masked():
    model = _causal_refiner(zero_init_output=False, history_gate_bias=20.0)
    with torch.no_grad():
        model.delta_head.weight.zero_()
        model.delta_head.bias.zero_()
        model.history_delta_head.weight.zero_()
        model.history_delta_head.bias.fill_(100.0)
    current_features = [torch.zeros((1, 1, 4, 4))]
    history_features = [torch.ones((1, 2, 1, 4, 4))]
    proposals = [torch.tensor([[10., 12., 8., 4., 0.2]])]
    history_boxes = torch.tensor([[
        [9., 12., 8., 4., 0.2],
        [8., 12., 8., 4., 0.2]]])
    output = model(
        current_features, proposals,
        history_features=history_features,
        history_proposals=history_boxes,
        history_valid_mask=torch.tensor([[True, False]]),
        history_ages=torch.tensor([[1, 2]]))
    bounds = model.history_residual_bounds
    assert torch.all(output.abs() <= bounds.reshape(1, 5) + 1e-6)
    assert torch.all(output > 0.0)


def test_causal_history_contract_forbids_identity_routing_and_state():
    contract = _causal_refiner().component_contract()
    assert contract['architecture'] == (
        'current_anchored_causal_history_refiner_v1')
    assert contract['strictly_causal'] is True
    assert contract['current_frame_anchored'] is True
    assert contract['rejectable_history_gate'] is True
    assert contract['exact_current_only_when_no_history'] is True
    assert contract['domain_routing'] is False
    assert contract['sequence_frame_routing'] is False
    assert contract['temporal_state'] is False


def test_k1_phase_zero_initialization_preserves_k1_not_dino_geometry():
    model = _k1_phase_refiner()
    k1 = torch.tensor([[10., 12., 8., 4., 0.2]])
    dino = torch.tensor([[15., 9., 12., 7., -0.3]])
    features = [torch.zeros((1, 1, 4, 4))]
    phase = model(
        features, [k1], conditioning_proposal_list=[dino])
    decoded = model.decode_and_normalize([k1], phase)[0]
    assert phase.shape == (1, 6)
    assert torch.equal(phase, torch.zeros_like(phase))
    assert torch.equal(decoded, k1)


def test_k1_phase_no_history_is_exact_current_only():
    model = _k1_phase_refiner()
    features = [torch.zeros((1, 1, 4, 4))]
    history_features = [torch.ones((1, 2, 1, 4, 4))]
    k1 = [torch.tensor([[10., 12., 8., 4., 0.2]])]
    dino = [torch.tensor([[11., 12., 9., 4., 0.1]])]
    history_boxes = torch.tensor([[[9., 12., 8., 4., 0.2],
                                   [8., 12., 8., 4., 0.2]]])
    current = model(features, k1, conditioning_proposal_list=dino)
    causal = model(
        features, k1, conditioning_proposal_list=dino,
        history_features=history_features,
        history_proposals=history_boxes,
        history_valid_mask=torch.zeros((1, 2), dtype=torch.bool),
        history_ages=torch.tensor([[1, 2]]))
    assert torch.equal(causal, current)


def test_k1_component_current_only_removes_history_correction():
    model = _k1_phase_refiner(
        zero_init_output=False, history_gate_bias=20.0,
        inference_component_mode='current_only')
    with torch.no_grad():
        model.delta_head.weight.zero_()
        model.delta_head.bias.zero_()
        model.history_delta_head.weight.zero_()
        model.history_delta_head.bias.fill_(100.0)
    features = [torch.zeros((1, 1, 4, 4))]
    history_features = [torch.ones((1, 2, 1, 4, 4))]
    k1 = [torch.tensor([[10., 12., 8., 4., 0.2]])]
    dino = [torch.tensor([[11., 12., 9., 4., 0.1]])]
    output = model(
        features, k1, conditioning_proposal_list=dino,
        history_features=history_features,
        history_proposals=torch.tensor([[[9., 12., 8., 4., 0.2],
                                         [8., 12., 8., 4., 0.2]]]),
        history_valid_mask=torch.tensor([[True, True]]),
        history_ages=torch.tensor([[1, 2]]))
    assert torch.equal(output, torch.zeros_like(output))


def test_k1_component_center_only_masks_size_and_angle_exactly():
    model = _k1_phase_refiner(
        zero_init_output=False, inference_component_mode='center_only')
    raw = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.1]])
    five = model._phase_to_five(raw)
    masked = model._apply_inference_component_mode(five)
    assert masked[:, :2] == pytest.approx(five[:, :2])
    assert torch.equal(masked[:, 2:], torch.zeros_like(masked[:, 2:]))


def test_k1_component_identity_is_exact_anchor_and_mode_is_validated():
    model = _k1_phase_refiner(
        zero_init_output=False, inference_component_mode='k1_identity')
    k1 = torch.tensor([[10., 12., 8., 4., 0.2]])
    output = model(
        [torch.zeros((1, 1, 4, 4))], [k1],
        conditioning_proposal_list=[torch.tensor(
            [[15., 8., 12., 7., -0.3]])])
    decoded = model.decode_and_normalize([k1], output)[0]
    assert torch.equal(output, torch.zeros_like(output))
    assert torch.equal(decoded, k1)
    assert model.component_contract()['inference_component_mode'] == (
        'k1_identity')
    with pytest.raises(ValueError, match='inference_component_mode'):
        _k1_phase_refiner(inference_component_mode='unsupported')


def test_k1_phase_target_uses_double_angle_periodicity():
    model = _k1_phase_refiner()
    proposal = [torch.tensor([[10., 12., 8., 4., 0.2]])]
    gt = [torch.tensor([[10., 12., 8., 4., 0.2 + math.pi]])]
    target = model.encode_targets(proposal, gt)
    assert target[0, 4].item() == pytest.approx(0.0, abs=1e-5)
    assert target[0, 5].item() == pytest.approx(0.0, abs=1e-5)


def test_k1_phase_contract_is_unified_bounded_and_route_free():
    contract = _k1_phase_refiner().component_contract()
    assert contract['architecture'] == 'k1_anchored_causal_phase_refiner_v2'
    assert contract['current_k1_geometry_anchor'] is True
    assert contract['native_dino_anchor_fallback'] is True
    assert contract['continuous_double_angle_phase'] is True
    assert contract['bounded_current_residual'] is True
    assert contract['same_forward_all_domains'] is True
    assert contract['domain_routing'] is False
    assert contract['sequence_frame_routing'] is False
    assert contract['temporal_state'] is False


def test_k1_phase_zero_identity_has_finite_nonzero_output_head_gradient():
    model = _k1_phase_refiner(decoded_geometry_loss_weight=0.1)
    k1 = [torch.tensor([[10., 12., 8., 4., 0.2]])]
    dino = [torch.tensor([[11., 11., 10., 5., 0.1]])]
    gt = [torch.tensor([[10.5, 12.2, 8.8, 4.4, 0.3]])]
    features = [torch.zeros((1, 1, 4, 4))]
    predicted = model(
        features, k1, conditioning_proposal_list=dino)
    targets = model.encode_targets(k1, gt)
    loss = model.loss(
        predicted, targets, proposal_list=k1,
        gt_box_list=gt)['loss_geometry_refiner']
    loss.backward()
    gradient = model.delta_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() > 0


def test_dual_tower_size_finetune_freezes_pose_and_roi_parameters():
    model = _dual_refiner(
        train_size_tower=True,
        train_pose_tower=False,
        train_roi_extractor=False)
    assert all(parameter.requires_grad for parameter in
               model.size_fcs.parameters())
    assert all(parameter.requires_grad for parameter in
               model.size_head.parameters())
    assert all(not parameter.requires_grad for parameter in
               model.pose_fcs.parameters())
    assert all(not parameter.requires_grad for parameter in
               model.pose_head.parameters())
    contract = model.component_contract()
    assert contract['train_size_tower'] is True
    assert contract['train_pose_tower'] is False
    assert contract['train_roi_extractor'] is False


def test_decoded_geometry_and_temporal_size_losses_reach_only_size_tower():
    torch.manual_seed(9)
    model = _dual_refiner(
        zero_init_output=False,
        train_size_tower=True,
        train_pose_tower=False,
        decoded_geometry_loss_weight=0.25,
        temporal_size_loss_weight=0.20)
    features = [torch.zeros((2, 1, 4, 4))]
    proposals = [
        torch.tensor([[10., 12., 8., 4., 0.2]]),
        torch.tensor([[11., 12., 8., 4., 0.2]])]
    gt = [
        torch.tensor([[10., 12., 10., 5., 0.2]]),
        torch.tensor([[11., 12., 11., 5.5, 0.2]])]
    predicted = model(features, proposals)
    targets = model.encode_targets(proposals, gt)
    losses = model.loss(
        predicted, targets,
        proposal_list=proposals,
        gt_box_list=gt,
        temporal_pair_indices=[(0, 1)])
    losses['loss_geometry_refiner'].backward()
    assert 'refiner_decoded_geometry_objective' in losses
    assert 'refiner_temporal_size_objective' in losses
    assert losses['refiner_temporal_pair_count'].item() == 1
    assert all(parameter.grad is not None for parameter in
               model.size_fcs.parameters())
    assert all(parameter.grad is not None for parameter in
               model.size_head.parameters())
    assert all(parameter.grad is None for parameter in
               model.pose_fcs.parameters())
    assert all(parameter.grad is None for parameter in
               model.pose_head.parameters())


def test_temporal_size_loss_uses_error_change_not_object_motion():
    predicted = torch.tensor([
        [0., 0., 0.20, -0.10, 0.],
        [0., 0., 0.35, 0.05, 0.]])
    target = torch.tensor([
        [0., 0., 0.10, -0.20, 0.],
        [0., 0., 0.25, -0.05, 0.]])
    value = Refiner._temporal_size_error_loss(
        predicted, target, [(0, 1)])
    assert value.item() == pytest.approx(0.0, abs=1e-8)


def test_k1_retentive_v3_activates_pair_error_and_continuous_retention():
    model = _k1_retentive_refiner(
        temporal_size_loss_weight=0.20,
        retention_loss_weight=0.25)
    proposals = [
        torch.tensor([[10., 12., 8., 4., 0.2]]),
        torch.tensor([[11., 12., 8., 4., 0.2]])]
    gt = [
        torch.tensor([[10., 12., 8.5, 4.2, 0.2]]),
        torch.tensor([[11., 12., 8.6, 4.3, 0.2]])]
    targets = model.encode_targets(proposals, gt)
    predicted = torch.zeros_like(targets, requires_grad=True)
    losses = model.loss(
        predicted, targets, proposal_list=proposals, gt_box_list=gt,
        temporal_pair_indices=[(0, 1)])
    assert 'refiner_temporal_size_objective' in losses
    assert 'refiner_continuous_retention_objective' in losses
    assert losses['refiner_temporal_pair_count'].item() == 1
    losses['loss_geometry_refiner'].backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    contract = model.component_contract()
    assert contract['architecture'] == 'k1_retentive_causal_phase_refiner_v3'
    assert contract['continuous_k1_retention'] is True
    assert contract['source_adjacent_pair_error_consistency'] is True
    assert contract['inference_sequence_input'] is False


def _parent_checkpoint(model, center, size, angle):
    contract = dict(
        source_train_frames=2781, source_val_frames=738,
        target_data_read=False, source_gate_passed=False,
        domain_routing=False, sequence_frame_routing=False,
        temporal_state=False, representation='five_delta_xywha',
        angle_range='le90', edge_swap=True, proj_xy=True,
        refine_center=center, refine_size=size, refine_angle=angle)
    return dict(
        state_dict={
            'geometry_refiner.' + key: value.detach().clone()
            for key, value in model.state_dict().items()},
        meta={'geometry_refiner_checkpoint_contract': contract})


def test_dual_tower_forward_exactly_recomposes_parent_components():
    torch.manual_seed(17)
    size = _refiner(
        refine_center=False, refine_size=True, refine_angle=False,
        zero_init_output=False)
    full = _refiner(zero_init_output=False)
    dual = _dual_refiner(zero_init_output=False)
    state, _, _ = compose_dual_tower_state(
        _parent_checkpoint(size, False, True, False),
        _parent_checkpoint(full, True, True, True))
    dual.load_state_dict(state, strict=True)
    proposal = torch.tensor([[10., 12., 8., 4., 0.2]])
    features = [torch.zeros((1, 1, 4, 4))]
    size_delta = size(features, [proposal])
    full_delta = full(features, [proposal])
    wanted = torch.cat([
        full_delta[:, :2], size_delta[:, 2:4], full_delta[:, 4:5]],
        dim=1)
    actual = dual(features, [proposal])
    assert torch.equal(actual, wanted)
