import hashlib
import json

import torch

from crane_project.tools.symeood_dino_dual_tower_v21_promote import (
    PROMOTION_PROTOCOL, promote)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_epoch7_promotion_verifies_gate_and_exports_only_refiner(tmp_path):
    checkpoint = tmp_path / 'epoch_7.pth'
    results = tmp_path / 'source_val_epoch7_results.pkl'
    gate_path = tmp_path / 'source_val_epoch7_gate.json'
    output = tmp_path / 'promoted.pth'
    report_path = tmp_path / 'promotion.json'
    results.write_bytes(b'locked-source-val-result')
    contract = dict(
        protocol='source_only_dual_tower_size_refinement_v21',
        architecture='dual_tower_size_pose_v2',
        source_train_frames=2781,
        source_val_frames=738,
        target_data_read=False,
        fixed_test_read=False,
        source_gate_passed=False,
        domain_routing=False,
        sequence_frame_routing=False,
        temporal_state=False,
        source_adjacent_pair_supervision=True,
        inference_sequence_input=False,
        train_size_tower=True,
        train_pose_tower=False,
        train_roi_extractor=False,
        representation='five_delta_xywha',
        angle_range='le90', edge_swap=True, proj_xy=True,
        refine_center=True, refine_size=True, refine_angle=True)
    torch.save(dict(
        state_dict={
            'geometry_refiner.size_head.weight': torch.ones(2, 3),
            'geometry_refiner.pose_head.weight': torch.ones(3, 3),
            'baseline.forbidden': torch.ones(1)},
        meta={'geometry_refiner_checkpoint_contract': contract}), checkpoint)
    gate = dict(
        protocol='dual_tower_v21_relaxed_composite_source_gate_v1',
        evidence_boundary='source_val_only',
        target_data_read=False,
        fixed_test_read=False,
        passed=True,
        eligible_for_checkpoint_promotion=True,
        eligible_for_fixed_test=False,
        decision='ALLOW_DUAL_TOWER_V21_CHECKPOINT_PROMOTION',
        input=dict(
            candidate_checkpoint=str(checkpoint),
            candidate_checkpoint_sha256=_sha256(checkpoint),
            candidate_results=str(results),
            candidate_results_sha256=_sha256(results)),
        relaxed_composite_gate=dict(passed=True))
    gate_path.write_text(json.dumps(gate))

    report = promote(
        checkpoint, results, gate_path, _sha256(checkpoint),
        _sha256(gate_path), 7,
        output, report_path)
    assert report['passed'] is True
    assert report['eligible_for_one_fixed_test'] is True
    assert report['decision'] == 'ALLOW_ONE_DUAL_TOWER_V21_FIXED_TEST'
    promoted = torch.load(output, map_location='cpu')
    state = promoted['state_dict']
    assert sorted(state) == [
        'geometry_refiner.pose_head.weight',
        'geometry_refiner.size_head.weight']
    promoted_contract = promoted['meta'][
        'geometry_refiner_checkpoint_contract']
    assert promoted_contract['protocol'] == PROMOTION_PROTOCOL
    assert promoted_contract['source_gate_passed'] is True
    assert promoted_contract['selected_source_epoch'] == 7
    assert promoted_contract['target_data_read'] is False
    assert promoted_contract['fixed_test_read'] is False


def test_promotion_rejects_wrong_epoch(tmp_path):
    checkpoint = tmp_path / 'epoch_6.pth'
    checkpoint.write_bytes(b'wrong')
    results = tmp_path / 'results.pkl'
    results.write_bytes(b'result')
    gate = tmp_path / 'gate.json'
    gate.write_text('{}')
    try:
        promote(
            checkpoint, results, gate, _sha256(checkpoint),
            _sha256(gate), 7,
            tmp_path / 'out.pth', tmp_path / 'out.json')
    except RuntimeError as error:
        assert 'locked epoch' in str(error)
    else:
        raise AssertionError('wrong epoch was promoted')
