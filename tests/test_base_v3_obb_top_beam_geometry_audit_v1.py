import copy

import pytest

from crane_project.tools import base_v3_obb_top_beam_geometry_audit_v1 as audit


def _manifest():
    return dict(
        protocol=audit.MANIFEST_PROTOCOL,
        target_definition='grab_top_beam_obb',
        target_geometry_note='rigid top beam',
        online_input_authorized=False,
        rules=[
            dict(domain='real', sequence='seq03', frame_start_inclusive=1,
                 frame_end_inclusive=129, operation_group='transport_seq03'),
            dict(domain='real', sequence='seq03', frame_start_inclusive=130,
                 frame_end_inclusive=180,
                 operation_group='grabbing_seq03_130_180'),
            dict(domain='real', sequence='seq03', frame_start_inclusive=181,
                 frame_end_inclusive=200,
                 operation_group='transport_seq03')])


def _row(frame, error_marker=0.0):
    box = [float(frame), 10.0, 20.0, 10.0, 0.0]
    return dict(
        frame_key='real_seq03_{:05d}'.format(frame), domain='real',
        sequence='seq03', frame=frame, anchor_source='k1', anchor_score=.8,
        base_v3_box=copy.deepcopy(box), k1_box=copy.deepcopy(box),
        dino_box=None, gt_box=copy.deepcopy(box),
        arbitrary_offline_error=error_marker)


def test_seq03_phase_boundaries_are_exact_and_other_is_transport():
    rows = audit.attach_operation_groups(
        [_row(frame) for frame in (129, 130, 180, 181)], _manifest())
    assert [row['operation_group'] for row in rows] == [
        'transport_seq03', 'grabbing_seq03_130_180',
        'grabbing_seq03_130_180', 'transport_seq03']


def test_overlapping_phase_rules_are_rejected():
    manifest = _manifest()
    manifest['rules'].append(dict(
        domain='real', sequence='seq03', frame_start_inclusive=129,
        frame_end_inclusive=130, operation_group='overlap'))
    with pytest.raises(ValueError, match='matches 2 operation rules'):
        audit.attach_operation_groups([_row(129)], manifest)


def test_geometry_error_separates_long_short_center_and_angle():
    result = audit.geometry_error(
        [3.0, 4.0, 22.0, 8.0, 0.1],
        [0.0, 0.0, 20.0, 10.0, 0.0])
    assert result['center_error_px'] == pytest.approx(5.0)
    assert result['long_side_relative_error'] == pytest.approx(.1)
    assert result['short_side_relative_error'] == pytest.approx(.2)
    assert result['max_side_relative_error'] == pytest.approx(.2)
    assert result['dominant_scale_error_component'] == 'short_side'
    assert result['angle_error_deg'] == pytest.approx(5.72957795)


def test_uniform_sample_is_independent_of_error_fields_and_keeps_references():
    rows = audit.attach_operation_groups(
        [_row(frame) for frame in range(130, 181)], _manifest())
    plan = [dict(
        sample_group='grab', domain='real', sequence='seq03',
        operation_group='grabbing_seq03_130_180',
        frame_start_inclusive=130, frame_end_inclusive=180,
        selection='uniform_ordered_index_plus_required_frames',
        sample_count=3, required_frames=[177])]
    first = audit.select_fixed_sample(rows, plan)
    changed = copy.deepcopy(rows)
    for index, row in enumerate(changed):
        row['arbitrary_offline_error'] = 1000.0-index
    second = audit.select_fixed_sample(changed, plan)
    assert [row['frame'] for row in first] == [130, 155, 180, 177]
    assert [row['frame_key'] for row in first] == [
        row['frame_key'] for row in second]


def test_contract_rejects_error_based_selection_authorization():
    contract = dict(
        protocol=audit.CONTRACT_PROTOCOL, fixed_test_read=True,
        fixed_test_previously_exposed=True, phase_labels_used_online=False,
        sample_selection_uses_model_errors=True,
        parameter_tuning_authorized=False)
    with pytest.raises(ValueError, match='sample_selection_uses_model_errors'):
        audit.validate_contract(contract)
