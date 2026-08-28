import torch

from crane_project.utils.dual_tower_geometry_refiner_checkpoint import (
    compose_dual_tower_state)


def _contract(center, size, angle):
    return dict(
        source_train_frames=2781,
        source_val_frames=738,
        target_data_read=False,
        source_gate_passed=False,
        domain_routing=False,
        sequence_frame_routing=False,
        temporal_state=False,
        representation='five_delta_xywha',
        angle_range='le90', edge_swap=True, proj_xy=True,
        refine_center=center, refine_size=size, refine_angle=angle)


def _checkpoint(seed, contract):
    generator = torch.Generator().manual_seed(seed)
    state = {
        'geometry_refiner.shared_fcs.0.weight': torch.randn(
            4, 3, generator=generator),
        'geometry_refiner.shared_fcs.0.bias': torch.randn(
            4, generator=generator),
        'geometry_refiner.delta_head.weight': torch.randn(
            5, 4, generator=generator),
        'geometry_refiner.delta_head.bias': torch.randn(
            5, generator=generator)}
    return dict(
        state_dict=state,
        meta={'geometry_refiner_checkpoint_contract': contract})


def test_dual_tower_state_uses_size_rows_and_full_pose_rows():
    size = _checkpoint(3, _contract(False, True, False))
    full = _checkpoint(7, _contract(True, True, True))
    dual, _, _ = compose_dual_tower_state(size, full)
    size_state = size['state_dict']
    full_state = full['state_dict']
    assert torch.equal(
        dual['size_fcs.0.weight'],
        size_state['geometry_refiner.shared_fcs.0.weight'])
    assert torch.equal(
        dual['pose_fcs.0.weight'],
        full_state['geometry_refiner.shared_fcs.0.weight'])
    assert torch.equal(
        dual['size_head.weight'],
        size_state['geometry_refiner.delta_head.weight'][2:4])
    assert torch.equal(
        dual['size_head.bias'],
        size_state['geometry_refiner.delta_head.bias'][2:4])
    rows = torch.tensor([0, 1, 4])
    assert torch.equal(
        dual['pose_head.weight'],
        full_state['geometry_refiner.delta_head.weight'].index_select(0, rows))
    assert torch.equal(
        dual['pose_head.bias'],
        full_state['geometry_refiner.delta_head.bias'].index_select(0, rows))
