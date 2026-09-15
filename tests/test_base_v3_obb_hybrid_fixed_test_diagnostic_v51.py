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


def test_matched_coverage_equal_metrics_are_reported_as_ties_not_wins():
    report = _report()
    result = diagnostic.scale_matched_coverage(
        report['records'], _contract())['all']
    assert result['comparison_count'] == 2
    assert result['matched_coverage_tie_count'] >= 1
    assert result['tie_policy'] == 'strict_metric_improvement_required'
    for point in result['points']:
        comparison = point['learned_minus_score']
        if comparison['tie_mean_and_failure']:
            assert comparison['wins_mean_and_failure'] is False


def test_repository_contract_uses_contract_protocol_and_not_report_protocol():
    import json
    from pathlib import Path
    contract_path = Path(__file__).parents[1] / (
        'crane_project/data_contracts/'
        'base_v3_obb_hybrid_fixed_test_diagnostic_v51.json')
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    diagnostic.validate_contract(contract)
    assert contract['protocol'] == diagnostic.CONTRACT_PROTOCOL
    assert diagnostic.PROTOCOL == 'base_v3_obb_hybrid_fixed_test_diagnostic_v51_v31'
    assert contract['protocol'] != diagnostic.PROTOCOL


@pytest.mark.parametrize('mean,failure,tied,won', [
    (0., 0., True, False), (1e-12, -1e-12, True, False),
    (0.5e-12, -0.5e-12, True, False),
    (-1.0001e-12, -1.0001e-12, False, True),
    (-2e-12, 0., False, False), (0., -2e-12, False, False),
    (2e-12, -2e-12, False, False)])
def test_absolute_comparison_boundaries(mean, failure, tied, won):
    flags = diagnostic.comparison_flags(mean, failure)
    assert flags['tie_mean_and_failure'] is tied
    assert flags['wins_mean_and_failure'] is won


def test_full_coverage_all_groups_and_permutations():
    rows = _report()['records']
    for i, row in enumerate(rows):
        row['domain'] = ('real', 'sim')[i % 2]
        row['sequence'] = 'seq' + str(i % 2)
        row['anchor_source'] = ('k1', 'dino_fallback')[i % 2]
        row['offline_errors']['raw']['scale'] = [1e10, 0.1, 0.2, 0.3][i]
    contract = _contract()
    contract['matched_coverage_targets'] = [1.0]
    for permutation in (rows, list(reversed(rows)), rows[1:] + rows[:1]):
        groups = diagnostic.scale_matched_coverage(permutation, contract)
        for group in groups.values():
            assert group['matched_coverage_tie_count'] == 1
            assert group['risk_wins_both_count'] == 0


def test_strict_wins_exact_count_and_raw_deltas_unchanged():
    values = [(-2e-12, -3e-12), (-2e-12, 0.), (1e-12, -1e-12)]
    scale = {'all': {'points': [dict(learned_minus_score=dict(
        mean_error=m, failure_rate=f)) for m, f in values]}}
    result = diagnostic.revise_scale_comparisons(scale)['all']
    assert result['risk_wins_both_count'] == 1
    assert result['matched_coverage_tie_count'] == 1
    assert [(p['learned_minus_score']['mean_error'],
             p['learned_minus_score']['failure_rate']) for p in result['points']] == values


def test_actual_contract_rejects_output_protocol():
    import json
    from pathlib import Path
    path = Path(__file__).parents[1] / 'crane_project/data_contracts/base_v3_obb_hybrid_fixed_test_diagnostic_v51.json'
    contract = json.loads(path.read_text())
    contract['protocol'] = diagnostic.PROTOCOL
    with pytest.raises(ValueError, match='Unexpected'):
        diagnostic.validate_contract(contract)


def test_diagnostic_cli_temporary_files(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys
    from pathlib import Path
    report = tmp_path / 'input.json'
    report.write_text(json.dumps(_report()))
    contract = _contract()
    contract['expected_input']['sha256'] = hashlib.sha256(report.read_bytes()).hexdigest()
    contract_path = tmp_path / 'contract.json'
    contract_path.write_text(json.dumps(contract))
    output = tmp_path / 'out.json'
    command = [sys.executable, '-m', diagnostic.__name__,
               '--fixed-test-v51-report', str(report), '--contract', str(contract_path),
               '--out-json', str(output)]
    subprocess.run(command, cwd=Path(__file__).parents[1], check=True, capture_output=True)
    result = json.loads(output.read_text())
    assert result['statistical_revision'] == diagnostic.STATISTICAL_REVISION
    assert result['comparison_tolerance'] == 1e-12
    assert result['input']['sha256'] == contract['expected_input']['sha256']
    assert result['scale_matched_coverage']['all']['matched_coverage_tie_count'] == 1
