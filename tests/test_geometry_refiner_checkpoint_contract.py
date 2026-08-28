import pathlib

import pytest
import torch
import torch.nn as nn

from crane_project.utils.geometry_refiner_checkpoint import (
    CONTRACT_KEY, load_source_gated_geometry_refiner_checkpoint,
    validate_source_gated_geometry_refiner_contract)


def _contract(**overrides):
    values = dict(
        source_train_frames=2781,
        source_val_frames=738,
        target_data_read=False,
        source_gate_passed=True,
        domain_routing=False,
        sequence_frame_routing=False,
        temporal_state=False,
        representation='five_delta_xywha',
        angle_range='le90', edge_swap=True, proj_xy=True,
        refine_center=True, refine_size=True, refine_angle=True)
    values.update(overrides)
    return values


def test_checkpoint_contract_rejects_ungated_or_target_read_artifact():
    with pytest.raises(RuntimeError, match='source_gate_passed'):
        validate_source_gated_geometry_refiner_contract(
            _contract(source_gate_passed=False))
    with pytest.raises(RuntimeError, match='target_data_read'):
        validate_source_gated_geometry_refiner_contract(
            _contract(target_data_read=True))


def test_runtime_loads_only_matching_shared_refiner_state(tmp_path):
    source = nn.Linear(2, 1)
    target = nn.Linear(2, 1)
    state = {
        'geometry_refiner.' + key: value.detach().clone()
        for key, value in source.state_dict().items()}
    path = pathlib.Path(tmp_path) / 'refiner.pth'
    torch.save(dict(
        state_dict=state,
        meta={CONTRACT_KEY: _contract()}), path)
    loaded = load_source_gated_geometry_refiner_checkpoint(
        target, path,
        expected_contract=dict(
            refine_center=True, refine_size=True, refine_angle=True))
    assert loaded['source_gate_passed'] is True
    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)


def test_runtime_rejects_component_contract_mismatch(tmp_path):
    module = nn.Linear(2, 1)
    path = pathlib.Path(tmp_path) / 'size_only.pth'
    torch.save(dict(
        state_dict={
            'geometry_refiner.' + key: value
            for key, value in module.state_dict().items()},
        meta={CONTRACT_KEY: _contract(
            refine_center=False, refine_angle=False)}), path)
    with pytest.raises(RuntimeError, match='refine_center'):
        load_source_gated_geometry_refiner_checkpoint(
            module, path,
            expected_contract=dict(
                refine_center=True, refine_size=True, refine_angle=True))
