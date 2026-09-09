import copy

import pytest

from crane_project.tools import (
    base_v3_obb_hybrid_fixed_test_diagnostic_v51 as diagnostic)


def _contract():
    return dict(
        protocol=diagnostic.CONTRACT_PROTOCOL,
        evidence_boundary='diagnostic',
        fixed_test_read=True, fixed_test_previously_exposed=True,
        parameter_tuning_authorized=False,
        expected_input=dict(protocol=diagnostic.INPUT_PROTOCOL,
                            sha256='a'*64),
        fixed_test_scope=dict(frame_count=4, domain_counts={'real': 4}),
        matched_coverage_targets=[1.0, 0.5],
        evaluation=dict(center_error_threshold_px=5.0,
                        scale_relative_error_threshold=0.1,
                        angle_error_threshold_deg=3.0),
        claim_limit='diagnostic only')


def _observation(state, value, risk=None):
    return dict(state=state, value=copy.deepcopy(value),
                valid=state != 'unavailable', risk=risk)


def _row(frame, score, scale_risk, errors, states):
    values = dict(center=[1.0, 2.0], scale=[10.0, 5.0], angle=0.1)
    observations = {}
    offline = {}
    for method in ('raw', 'score_rejection_only', 'v51_hybrid'):
        observations[method] = {}
        offline[method] = {}
        for component in diagnostic.COMPONENTS:
            state = states.get(method, {}).get(component, 'measurement')
            risk = scale_risk if method == 'v51_hybrid' and component == 'scale' else None
            observations[method][component] = _observation(
                state, None if state == 'unavailable' else values[component], risk)
            offline[method][component] = (
                None if state == 'unavailable' else errors[method][component])
    return dict(
        frame_key='real_seq_{:05d}'.format(frame), domain='real',
        sequence='seq', frame=frame, anchor_source='k1', anchor_score=score,
        online_observations=observations, offline_errors=offline)


def _report():
    raw_errors = [
        dict(center=1.0, scale=0.01, angle=1.0),
        dict(center=1.0, scale=0.20, angle=5.0),
        dict(center=1.0, scale=0.02, angle=1.0),
        dict(center=1.0, scale=0.30, angle=5.0)]
    rows = []
    for index, raw in enumerate(raw_errors, 1):
        errors = dict(raw=raw, score_rejection_only=copy.deepcopy(raw),
                      v51_hybrid=copy.deepcopy(raw))
        states = {}
        if index == 2:
            states = {'score_rejection_only': {
                component: 'unavailable' for component in diagnostic.COMPONENTS},
                'v51_hybrid': {'center': 'unavailable',
                               'scale': 'measurement', 'angle': 'prediction'}}
            errors['score_rejection_only'] = {
                component: None for component in diagnostic.COMPONENTS}
            errors['v51_hybrid']['center'] = None
            errors['v51_hybrid']['angle'] = 2.0
        if index == 3:
            states = {'v51_hybrid': {'center': 'unavailable',
                                     'scale': 'measurement',
                                     'angle': 'unavailable'}}
            errors['v51_hybrid']['center'] = None
            errors['v51_hybrid']['angle'] = None
        rows.append(_row(index, 1.0-index*.1,
                         [0.1, 0.9, 0.2, 0.8][index-1], errors, states))
    return dict(
        protocol=diagnostic.INPUT_PROTOCOL,
        fixed_test_read=True, fixed_test_previously_exposed=True,
        frozen_calibration=True, threshold_fitting_performed=False,
        policy_selection_performed=False,
        eligible_for_parameter_tuning_from_this_report=False,
        method_inventory=['raw', 'score_rejection_only', 'v51_hybrid'],
        records=rows)


def test_audit_adds_matched_coverage_angle_pairs_and_joint_availability():
    result = diagnostic.run(_report(), _contract())

    scale = result['scale_matched_coverage']['all']['points'][1]
    assert scale['anchor_score']['retained_count'] == 2
    assert scale['learned_scale_risk']['retained_count'] == 2
    assert scale['learned_scale_risk']['mean_error'] < (
        scale['anchor_score']['mean_error'])
    angle = result['angle_hold_pair_audit']['all']
    assert angle['prediction_count'] == 1
    assert angle['correctness_transitions'] == {
        'raw_bad_to_hold_correct': 1}
    assert angle['improved_count'] == 1
    joint = result['joint_component_availability']['v51_hybrid']['all']
    assert joint['all_components_valid_count'] == 2
    assert joint['partial_valid_count'] == 2
    assert joint['longest_incomplete_obb_run'] == 2


def test_contract_cannot_authorize_test_tuning():
    contract = _contract()
    contract['parameter_tuning_authorized'] = True
    with pytest.raises(ValueError, match='must not authorize'):
        diagnostic.validate_contract(contract)
