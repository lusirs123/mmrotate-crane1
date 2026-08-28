"""Build a source-only dual-tower refiner from two audited V1 checkpoints."""

import hashlib
import os

import torch


CONTRACT_KEY = 'geometry_refiner_checkpoint_contract'


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _contract(checkpoint):
    value = checkpoint.get(CONTRACT_KEY)
    if value is None:
        value = dict(checkpoint.get('meta') or {}).get(CONTRACT_KEY)
    if not isinstance(value, dict):
        raise RuntimeError('Parent checkpoint has no refiner contract')
    return dict(value)


def _local_state(checkpoint):
    state = dict(checkpoint.get('state_dict') or checkpoint)
    prefixes = ('module.geometry_refiner.', 'geometry_refiner.')
    selected = {}
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                selected[key[len(prefix):]] = value
                break
    if not selected:
        raise RuntimeError('Parent checkpoint has no geometry_refiner state')
    return selected


def _validate_parent(contract, expected_components, name):
    required = dict(
        source_train_frames=2781,
        source_val_frames=738,
        target_data_read=False,
        domain_routing=False,
        sequence_frame_routing=False,
        temporal_state=False,
        representation='five_delta_xywha',
        angle_range='le90',
        edge_swap=True,
        proj_xy=True)
    required.update(expected_components)
    failures = [
        '{}={!r} expected {!r}'.format(key, contract.get(key), expected)
        for key, expected in required.items()
        if contract.get(key) != expected]
    if failures:
        raise RuntimeError(
            '{} parent contract failed: {}'.format(
                name, '; '.join(failures)))


def compose_dual_tower_state(size_checkpoint, full_checkpoint):
    """Return local dual-tower state plus validated parent contracts."""
    size_contract = _contract(size_checkpoint)
    full_contract = _contract(full_checkpoint)
    _validate_parent(size_contract, dict(
        refine_center=False, refine_size=True, refine_angle=False), 'size')
    _validate_parent(full_contract, dict(
        refine_center=True, refine_size=True, refine_angle=True), 'full')
    size_state = _local_state(size_checkpoint)
    full_state = _local_state(full_checkpoint)
    allowed_prefixes = ('roi_extractor.', 'shared_fcs.', 'delta_head.')
    for name, state in (('size', size_state), ('full', full_state)):
        unexpected = sorted(
            key for key in state
            if not key.startswith(allowed_prefixes))
        if unexpected:
            raise RuntimeError(
                '{} parent has unexpected refiner keys: {}'.format(
                    name, unexpected))
    required = ('delta_head.weight', 'delta_head.bias')
    if any(key not in size_state or key not in full_state
           for key in required):
        raise RuntimeError('Parent checkpoint lacks five-delta output state')
    if (size_state['delta_head.weight'].shape[0] != 5
            or full_state['delta_head.weight'].shape[0] != 5
            or size_state['delta_head.bias'].shape[0] != 5
            or full_state['delta_head.bias'].shape[0] != 5):
        raise RuntimeError('Parent delta head is not five-dimensional')
    size_fc = {
        key[len('shared_fcs.'):]: value
        for key, value in size_state.items()
        if key.startswith('shared_fcs.')}
    full_fc = {
        key[len('shared_fcs.'):]: value
        for key, value in full_state.items()
        if key.startswith('shared_fcs.')}
    if not size_fc or set(size_fc) != set(full_fc):
        raise RuntimeError('Parent FC tower structures do not match')
    dual = {}
    for key, value in size_fc.items():
        dual['size_fcs.' + key] = value.detach().clone()
    for key, value in full_fc.items():
        dual['pose_fcs.' + key] = value.detach().clone()
    dual['size_head.weight'] = (
        size_state['delta_head.weight'][2:4].detach().clone())
    dual['size_head.bias'] = (
        size_state['delta_head.bias'][2:4].detach().clone())
    pose_rows = torch.tensor([0, 1, 4], dtype=torch.long)
    dual['pose_head.weight'] = full_state['delta_head.weight'].index_select(
        0, pose_rows).detach().clone()
    dual['pose_head.bias'] = full_state['delta_head.bias'].index_select(
        0, pose_rows).detach().clone()
    size_roi = {
        key: value for key, value in size_state.items()
        if key.startswith('roi_extractor.')}
    full_roi = {
        key: value for key, value in full_state.items()
        if key.startswith('roi_extractor.')}
    if set(size_roi) != set(full_roi):
        raise RuntimeError('Parent ROI extractor structures do not match')
    for key in size_roi:
        if not torch.equal(size_roi[key], full_roi[key]):
            raise RuntimeError('Parent ROI extractor tensors differ')
        dual[key] = size_roi[key].detach().clone()
    return dual, size_contract, full_contract


def load_parent_checkpoint(path):
    absolute = os.path.abspath(os.fspath(path))
    if not os.path.isfile(absolute):
        raise RuntimeError('Parent checkpoint does not exist: ' + absolute)
    return torch.load(absolute, map_location='cpu')
