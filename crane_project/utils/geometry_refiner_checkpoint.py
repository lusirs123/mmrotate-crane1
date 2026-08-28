"""Checkpoint contract for the source-only geometry refiner."""

import os

import torch


CONTRACT_KEY = 'geometry_refiner_checkpoint_contract'


def _checkpoint_contract(checkpoint):
    contract = checkpoint.get(CONTRACT_KEY)
    if contract is None:
        contract = dict(checkpoint.get('meta') or {}).get(CONTRACT_KEY)
    if not isinstance(contract, dict):
        raise RuntimeError(
            'Geometry-refiner checkpoint has no explicit contract')
    return dict(contract)


def validate_source_gated_geometry_refiner_contract(contract, expected=None):
    required = dict(
        source_train_frames=2781,
        source_val_frames=738,
        target_data_read=False,
        source_gate_passed=True,
        domain_routing=False,
        sequence_frame_routing=False,
        temporal_state=False,
        representation='five_delta_xywha',
        angle_range='le90',
        edge_swap=True,
        proj_xy=True)
    required.update(dict(expected or {}))
    failures = []
    for key, value in required.items():
        if contract.get(key) != value:
            failures.append(
                '{} expected={!r} got={!r}'.format(
                    key, value, contract.get(key)))
    if failures:
        raise RuntimeError(
            'Geometry-refiner checkpoint contract failed: '
            + '; '.join(failures))
    return dict(contract)


def _refiner_state_dict(checkpoint):
    state = dict(checkpoint.get('state_dict') or checkpoint)
    prefixes = ('module.geometry_refiner.', 'geometry_refiner.')
    selected = {}
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                selected[key[len(prefix):]] = value
                break
    if selected:
        return selected
    # A deliberately packaged refiner-only checkpoint may already use local
    # module keys.  Never treat metadata dictionaries as tensors.
    selected = {
        key: value for key, value in state.items()
        if torch.is_tensor(value)
    }
    if not selected:
        raise RuntimeError('Geometry-refiner checkpoint has no tensor state')
    return selected


def load_source_gated_geometry_refiner_checkpoint(
        module, checkpoint_path, expected_contract=None):
    path = os.path.abspath(os.fspath(checkpoint_path))
    if not os.path.isfile(path):
        raise RuntimeError(
            'Geometry-refiner checkpoint does not exist: ' + path)
    checkpoint = torch.load(path, map_location='cpu')
    contract = validate_source_gated_geometry_refiner_contract(
        _checkpoint_contract(checkpoint), expected_contract)
    incompatible = module.load_state_dict(
        _refiner_state_dict(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            'Geometry-refiner checkpoint state mismatch: missing={} '
            'unexpected={}'.format(
                incompatible.missing_keys, incompatible.unexpected_keys))
    return contract
