import copy

import pytest

from crane_project.tools import base_v3_obb_true_online_paper_metrics_v3 as final


def _contract():
    return dict(
        protocol=final.CONTRACT_PROTOCOL,
        fixed_test_read=True, fixed_test_previously_exposed=True,
        threshold_fitting_performed=False, parameter_tuning_authorized=False,
        expected_inputs=dict(
            true_online_finalization_protocol=final.V2_PROTOCOL,
            true_online_pipeline_protocol=final.ONLINE_PROTOCOL),
        methods=list(final.METHODS), components=list(final.COMPONENTS),
        matched_coverage_targets=[1.0, 0.5])


def test_contract_rejects_parameter_tuning():
    contract=_contract()
    final.validate_contract(contract)
    contract['parameter_tuning_authorized']=True
    with pytest.raises(ValueError,match='parameter_tuning_authorized'):
        final.validate_contract(contract)


def test_build_records_uses_online_geometry_and_offline_metrics():
    observation={component:dict(valid=True,state='measurement')
                 for component in final.COMPONENTS}
    online=[dict(
        frame_key='real_seq02_00001',domain='real',sequence='seq02',frame=1,
        detector_components=dict(anchor_source='k1',anchor_score=0.8),
        observations={method:copy.deepcopy(observation)
                      for method in final.METHODS})]
    errors={method:{component:0.0 for component in final.COMPONENTS}
            for method in final.METHODS}
    v2=[dict(frame_key='real_seq02_00001',offline_errors=errors,
             offline_riou={method:1.0 for method in final.METHODS})]
    records=final.build_diagnostic_records(online,v2)
    assert records[0]['anchor_source']=='k1'
    assert records[0]['anchor_score']==pytest.approx(0.8)
    assert records[0]['offline_riou']['raw']==pytest.approx(1.0)
    records[0]['offline_errors']['raw']['center']=2.0
    assert v2[0]['offline_errors']['raw']['center']==0.0


def test_threshold_validation_rejects_drift():
    contract={'evaluation':dict(
        center_error_threshold_px=5.0,
        scale_relative_error_threshold=0.1,
        angle_error_threshold_deg=3.0,
        tail_quantiles=[0.5,0.9],riou_hit_threshold=0.5)}
    v2={'evaluation_thresholds':copy.deepcopy(contract['evaluation'])}
    final._validate_thresholds(v2,contract)
    v2['evaluation_thresholds']['angle_error_threshold_deg']=4.0
    with pytest.raises(RuntimeError,match='angle_error_threshold_deg'):
        final._validate_thresholds(v2,contract)


def test_validate_inputs_rejects_frame_set_mismatch_before_reporting():
    contract=_contract()
    contract.update(
        pipeline=['rgb_image'],
        base_v3_output_score_semantics='constant_refiner_output_not_confidence',
        fixed_test_scope=dict(frame_count=1,domain_counts={'real':1},
                              sequence_counts={'seq02':1}),
        evaluation=dict(center_error_threshold_px=5.0,
                        scale_relative_error_threshold=0.1,
                        angle_error_threshold_deg=3.0,
                        tail_quantiles=[0.5],riou_hit_threshold=0.5))
    v2=dict(
        protocol=final.V2_PROTOCOL,
        claim_status='POST_EXPOSURE_TRUE_ONLINE_FINALIZATION_V2',
        fixed_test_read=True,fixed_test_previously_exposed=True,
        threshold_fitting_performed=False,parameter_tuning_authorized=False,
        evaluation_thresholds=copy.deepcopy(contract['evaluation']),
        records=[{'frame_key':'real_seq02_00002'}])
    observations={method:{component:{} for component in final.COMPONENTS}
                  for method in final.METHODS}
    online=dict(
        protocol=final.ONLINE_PROTOCOL,
        execution_mode='true_online_model_inference',pipeline=['rgb_image'],
        leakage_controls={'annotations_or_gt_read':False},
        records=[dict(
            frame_key='real_seq02_00001',domain='real',sequence='seq02',frame=1,
            timestamp_seconds=None,online_gt_fields_consumed=[],
            detector_components=dict(
                anchor_score=0.8,
                base_v3_output_score_semantics=
                    'constant_refiner_output_not_confidence'),
            observations=observations)])
    with pytest.raises(RuntimeError,match='frame sets differ'):
        final.validate_inputs(v2,online,contract)
