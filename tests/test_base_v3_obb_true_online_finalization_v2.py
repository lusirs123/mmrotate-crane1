import copy

import pytest

from crane_project.tools import base_v3_obb_true_online_finalization_v2 as final


def _row(frame,dino=True):
    box=[float(frame),2.0,10.0,5.0,0.1]
    observation={component:dict(
        value=([float(frame),2.0] if component=='center' else
               [10.0,5.0] if component=='scale' else 0.1),
        valid=True,state='measurement',source='base_v3_measurement',
        age_since_measurement_frames=0,measurement_available=True,
        measurement_accepted=True,risk=None,gate='test',
        reason='accepted_measurement') for component in final.COMPONENTS}
    return dict(
        frame_key='real_seq02_{:05d}'.format(frame),domain='real',
        sequence='seq02',frame=frame,timestamp_seconds=None,image_size=[20,10],
        detector_components=dict(
            anchor_source='k1',anchor_score=0.9,
            base_v3_box=copy.deepcopy(box),k1_box=copy.deepcopy(box),
            dino_box=copy.deepcopy(box) if dino else None),
        observations={method:copy.deepcopy(observation) for method in final.METHODS},
        online_gt_fields_consumed=[])


def test_geometry_audit_separates_presence_and_propagated_difference():
    current=[dict(r['detector_components'],frame_key=r['frame_key'])
             for r in (_row(1),_row(2))]
    historical=copy.deepcopy(current)
    historical[0]['dino_box']=None
    historical[1]['base_v3_box'][0]+=0.5
    tolerances=dict(center_px=0.001,side_px=0.001,angle_deg=0.05)
    dino=final.compare_geometry_lane(current,historical,'dino_box',tolerances)
    base=final.compare_geometry_lane(current,historical,'base_v3_box',tolerances)
    assert dino['presence_mismatch_count']==1
    assert dino['mismatch_frame_keys']==['real_seq02_00001']
    assert base['geometry_mismatch_count']==1
    assert base['maximum_difference']['center_px']==pytest.approx(0.5)


def test_geometry_audit_rejects_length_mismatch():
    current=[dict(_row(1)['detector_components'],
                  frame_key='real_seq02_00001')]
    with pytest.raises(RuntimeError,match='different lengths'):
        final.compare_geometry_lane(
            current,[], 'base_v3_box',
            dict(center_px=0.001,side_px=0.001,angle_deg=0.05))


def test_evaluation_alignment_rejects_threshold_drift():
    evaluation=dict(center_error_threshold_px=5.0,
                    scale_relative_error_threshold=0.1,
                    angle_error_threshold_deg=3.0)
    final.validate_evaluation_alignment(
        {'evaluation':copy.deepcopy(evaluation)},
        {'evaluation':copy.deepcopy(evaluation)})
    changed=copy.deepcopy(evaluation)
    changed['scale_relative_error_threshold']=0.2
    with pytest.raises(RuntimeError,match='scale_relative_error_threshold'):
        final.validate_evaluation_alignment(
            {'evaluation':evaluation},{'evaluation':changed})


def test_current_rows_attach_gt_only_after_online_output():
    online=[_row(1)]
    historical=[dict(frame_key=online[0]['frame_key'],
                     gt_box=[1.0,2.0,10.0,5.0,0.1])]
    rows=final._current_rows(online,historical)
    assert rows[0]['gt_box']==historical[0]['gt_box']
    assert 'gt_box' not in online[0]


def test_csv_export_contains_online_state_but_no_gt():
    rows=list(final._csv_rows([_row(1)]))
    assert len(rows)==len(final.METHODS)*len(final.COMPONENTS)
    assert all('gt_box' not in row for row in rows)
    assert {row['state'] for row in rows}=={'measurement'}


def test_diagnostic_records_keep_anchor_source_for_joint_groups():
    online=[_row(1)]
    historical=[dict(frame_key=online[0]['frame_key'],
                     gt_box=[1.0,2.0,10.0,5.0,0.1])]
    rows=final._current_rows(online,historical)
    diagnostic=final._diagnostic_records(rows,online)
    assert diagnostic[0]['anchor_source']=='k1'
    joint=final.diagnostic.joint_availability(
        diagnostic,{'evaluation':dict(
            center_error_threshold_px=5.0,
            scale_relative_error_threshold=0.1,
            angle_error_threshold_deg=3.0)})
    assert joint['raw']['anchor:k1']['frame_count']==1


def test_contract_rejects_parameter_tuning():
    contract=dict(
        protocol=final.CONTRACT_PROTOCOL,fixed_test_read=True,
        fixed_test_previously_exposed=True,threshold_fitting_performed=False,
        parameter_tuning_authorized=False,expected_inputs=dict(
            true_online_protocol=final.TRUE_ONLINE_PROTOCOL,
            paper_report_protocol=final.PAPER_REPORT_PROTOCOL,
            v51_eval_contract_protocol=final.V51_EVAL_CONTRACT_PROTOCOL,
            attribution_protocol=final.ATTRIBUTION_PROTOCOL),
        methods=list(final.METHODS),components=list(final.COMPONENTS),
        comparison_tolerances=dict(center_px=0.001,side_px=0.001,angle_deg=0.05))
    final.validate_contract(contract)
    contract['parameter_tuning_authorized']=True
    with pytest.raises(ValueError,match='parameter_tuning_authorized'):
        final.validate_contract(contract)
