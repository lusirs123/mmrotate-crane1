#!/usr/bin/env python3
"""Audit the exposed V5.2 source report without refitting any policy.

V5.2.1 preserves the original V5.2 artifact and corrects its interpretation.
It quantifies how many rejected scale observations a one-frame hold fills,
how many of those additions are actually correct, and whether their error is
better than the current raw observation.  It also compares every predeclared
V5.2 ranker with V5.1 and anchor confidence at matched coverage by domain.
"""

import argparse
import json
from pathlib import Path

from crane_project.tools import (
    base_v3_obb_split_scale_policy_development_v52 as v52)
from crane_project.tools.base_v3_obb_reliability_baseline import (
    _identity, _write_exact)


PROTOCOL = 'base_v3_obb_split_scale_policy_diagnostic_v521'
CONTRACT_PROTOCOL = 'base_v3_obb_split_scale_policy_diagnostic_contract_v521'
TARGETS = ('long_side', 'short_side', 'scale')


def _require_identity(identity, expected, role):
    if identity['sha256'] != expected:
        raise RuntimeError('{} identity mismatch: expected {}, got {}'.format(
            role, expected, identity['sha256']))


def validate_contract(contract):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected V5.2.1 diagnostic contract')
    if contract.get('fixed_test_read') is not False:
        raise ValueError('V5.2.1 must not authorize fixed TEST')
    if contract.get('target_data_read') is not False:
        raise ValueError('V5.2.1 must remain source-only')
    if contract.get('refitting_authorized') is not False:
        raise ValueError('V5.2.1 must not refit models or thresholds')
    if contract.get('parameter_tuning_authorized') is not False:
        raise ValueError('V5.2.1 must not tune parameters')
    expected = contract.get('expected_inputs', {})
    if expected.get('v52_report_protocol') != v52.PROTOCOL:
        raise ValueError('V5.2.1 must bind the V5.2 source report')
    gates = contract.get('scale_hold_gate', {})
    if float(gates.get('minimum_correct_prediction_fraction', -1)) != 0.5:
        raise ValueError('Scale-hold correctness gate changed')
    if float(gates.get('maximum_mean_error_delta_vs_raw', 1)) != 0.0:
        raise ValueError('Scale-hold error-delta gate changed')
    if gates.get('require_improved_not_fewer_than_degraded') is not True:
        raise ValueError('Scale-hold paired comparison gate changed')
    if contract.get('automatic_ablation_promotion') is not False:
        raise ValueError('V5.2.1 must not promote an ablation automatically')


def validate_v52_report(report, contract):
    if report.get('protocol') != v52.PROTOCOL:
        raise ValueError('Unexpected V5.2 report protocol')
    if report.get('fixed_test_read') is not False:
        raise ValueError('V5.2 report violates fixed-TEST boundary')
    if report.get('target_data_read') is not False:
        raise ValueError('V5.2 report violates source-only boundary')
    if report.get('decision') != 'STOP_V52_PRIMARY_AND_REPORT_ABLATIONS_ONLY':
        raise ValueError('V5.2.1 expects the stopped primary candidate')
    if report.get('primary_split_risk_passed') is not False:
        raise ValueError('V5.2.1 expects the failed V5.2 primary gate')
    if report.get('v51_reproduction_check', {}).get('passed') is not True:
        raise ValueError('V5.2 did not reproduce the V5.1 reference')
    if report.get('automatic_ablation_promotion') is not False:
        raise ValueError('V5.2 report permits automatic ablation promotion')
    if report.get('image_quality_ablation_enabled') is not True:
        raise ValueError('V5.2.1 expects the image-quality ablation')
    expected_split = {
        'calibration': int(contract['expected_split_counts']['calibration']),
        'evaluation': int(contract['expected_split_counts']['evaluation'])}
    if report.get('split_counts') != expected_split:
        raise ValueError('V5.2 source split counts changed')
    expected_methods = set(contract['expected_methods'])
    if set(report.get('method_inventory', [])) != expected_methods:
        raise ValueError('V5.2 method inventory changed')


def _count_from_coverage(value, frame_count, role):
    raw = float(value)*int(frame_count)
    count = int(round(raw))
    if abs(raw-count) > 1e-8:
        raise RuntimeError('{} does not map to an integer count'.format(role))
    return count


def hold_cost_summary(rejection, hold):
    """Recover exact addition counts from aggregate scale-state metrics."""
    frame_count = int(rejection['frame_count'])
    if int(hold['frame_count']) != frame_count:
        raise RuntimeError('Hold/rejection frame counts differ')
    prediction_count = int(hold['prediction_count'])
    if int(rejection['prediction_count']) != 0:
        raise RuntimeError('Scale rejection reference contains predictions')
    if int(hold['measurement_count']) != int(rejection['measurement_count']):
        raise RuntimeError('Scale hold changed accepted measurements')
    if int(hold['unavailable_count'])+prediction_count != int(
            rejection['unavailable_count']):
        raise RuntimeError('Scale hold is not a pure rejected-frame fill')
    correct_rejection = _count_from_coverage(
        rejection['correct_output_coverage'], frame_count,
        'rejection correct-output coverage')
    correct_hold = _count_from_coverage(
        hold['correct_output_coverage'], frame_count,
        'hold correct-output coverage')
    correct_predictions = correct_hold-correct_rejection
    if not 0 <= correct_predictions <= prediction_count:
        raise RuntimeError('Invalid correct scale-prediction count')
    bad_rejection = _count_from_coverage(
        rejection['bad_available_output_rate']*(
            frame_count-int(rejection['unavailable_count'])),
        1, 'rejection bad-output rate')
    bad_hold = _count_from_coverage(
        hold['bad_available_output_rate']*(
            frame_count-int(hold['unavailable_count'])),
        1, 'hold bad-output rate')
    wrong_predictions = prediction_count-correct_predictions
    if bad_hold-bad_rejection != wrong_predictions:
        raise RuntimeError('Bad-output accounting does not close')
    return {
        'frame_count': frame_count,
        'measurement_count': int(rejection['measurement_count']),
        'filled_prediction_count': prediction_count,
        'correct_prediction_count': correct_predictions,
        'wrong_prediction_count': wrong_predictions,
        'correct_prediction_fraction': (
            None if prediction_count == 0 else
            correct_predictions/prediction_count),
        'improved_prediction_count': int(hold['prediction_improved_count']),
        'degraded_prediction_count': int(hold['prediction_degraded_count']),
        'mean_prediction_error': hold['mean_prediction_error'],
        'mean_prediction_error_delta_vs_raw': (
            hold['mean_prediction_error_delta_vs_raw']),
        'output_coverage_gain': (
            float(hold['output_coverage'])-
            float(rejection['output_coverage'])),
        'correct_output_coverage_gain': (
            float(hold['correct_output_coverage'])-
            float(rejection['correct_output_coverage'])),
        'new_bad_output_count': bad_hold-bad_rejection,
        'longest_unavailable_run_before': int(
            rejection['longest_unavailable_run']),
        'longest_unavailable_run_after': int(
            hold['longest_unavailable_run']),
    }


def strict_hold_verdict(summary_by_group, contract):
    gates = contract['scale_hold_gate']
    primary = summary_by_group['all']
    correct_fraction = primary['correct_prediction_fraction']
    error_delta = primary['mean_prediction_error_delta_vs_raw']
    checks = {
        'produces_at_least_one_prediction': (
            primary['filled_prediction_count'] > 0),
        'correct_prediction_fraction_at_least_half': (
            correct_fraction is not None and correct_fraction >= float(
                gates['minimum_correct_prediction_fraction'])),
        'mean_prediction_error_not_worse_than_current_raw': (
            error_delta is not None and error_delta <= float(
                gates['maximum_mean_error_delta_vs_raw'])),
        'improved_not_fewer_than_degraded': (
            primary['improved_prediction_count'] >=
            primary['degraded_prediction_count']),
    }
    return {'checks': checks, 'passes_all': all(checks.values())}


def _curve_outcome(challenger, reference):
    if len(challenger) != len(reference):
        raise RuntimeError('Matched-coverage curve lengths differ')
    wins, losses, tradeoffs = [], [], []
    for item, other in zip(challenger, reference):
        if item['target_coverage'] != other['target_coverage']:
            raise RuntimeError('Matched-coverage targets differ')
        if item['retained_count'] != other['retained_count']:
            raise RuntimeError('Matched-coverage retained counts differ')
        no_worse = (item['mean_error'] <= other['mean_error'] and
                    item['failure_rate'] <= other['failure_rate'])
        better = (item['mean_error'] < other['mean_error'] or
                  item['failure_rate'] < other['failure_rate'])
        no_better = (item['mean_error'] >= other['mean_error'] and
                     item['failure_rate'] >= other['failure_rate'])
        worse = (item['mean_error'] > other['mean_error'] or
                 item['failure_rate'] > other['failure_rate'])
        target = float(item['target_coverage'])
        if no_worse and better:
            wins.append(target)
        elif no_better and worse:
            losses.append(target)
        else:
            tradeoffs.append(target)
    return {'win_count': len(wins), 'loss_count': len(losses),
            'wins': wins, 'losses': losses,
            'ties_or_tradeoffs': tradeoffs}


def matched_coverage_audit(report):
    output = {}
    for group, targets in report['matched_coverage_by_group'].items():
        output[group] = {}
        for candidate in report['fitted_split_scale_candidates']:
            output[group][candidate] = {}
            for target in TARGETS:
                curves = targets[target]
                output[group][candidate][target] = {
                    'vs_v51_single_scale_risk': _curve_outcome(
                        curves[candidate], curves['v51_single_scale_risk']),
                    'vs_anchor_score': _curve_outcome(
                        curves[candidate], curves['anchor_score'])}
    return output


def image_quality_hypothesis_verdict(audit, contract):
    name = 'split_image_quality_ablation'
    minimum = int(contract['image_quality_hypothesis_gate'][
        'minimum_win_count'])
    all_vs_v51 = {
        target: audit['all'][name][target]['vs_v51_single_scale_risk'][
            'win_count'] for target in TARGETS}
    real_scale_vs_score = audit['real'][name]['scale']['vs_anchor_score'][
        'win_count']
    sim_scale_vs_score = audit['sim'][name]['scale']['vs_anchor_score'][
        'win_count']
    checks = {
        'all_targets_win_at_least_twice_vs_v51': all(
            count >= minimum for count in all_vs_v51.values()),
        'real_scale_wins_at_least_twice_vs_anchor_score': (
            real_scale_vs_score >= minimum),
    }
    return {
        'checks': checks,
        'passes_hypothesis_gate': all(checks.values()),
        'all_target_win_counts_vs_v51': all_vs_v51,
        'real_scale_win_count_vs_anchor_score': real_scale_vs_score,
        'sim_scale_win_count_vs_anchor_score': sim_scale_vs_score,
        'eligible_for_automatic_promotion': False,
    }


def run(report, contract):
    validate_contract(contract)
    validate_v52_report(report, contract)
    methods = report['method_reports']
    candidate_names = list(report['fitted_split_scale_candidates'])
    hold_costs = {}
    hold_verdicts = {}
    for candidate in candidate_names:
        rejection = methods[candidate+'_rejection_only']
        hold = methods[candidate+'_one_frame_scale_hold']
        hold_costs[candidate] = {
            group: hold_cost_summary(
                rejection[group]['scale'], hold[group]['scale'])
            for group in contract['report_groups']}
        hold_verdicts[candidate] = strict_hold_verdict(
            hold_costs[candidate], contract)
    coverage = matched_coverage_audit(report)
    image_hypothesis = image_quality_hypothesis_verdict(coverage, contract)
    primary = report['primary_candidate']
    primary_hold_passed = hold_verdicts[primary]['passes_all']
    return {
        'protocol': PROTOCOL,
        'evidence_boundary': contract['evidence_boundary'],
        'claim_status': 'POST_EXPOSURE_SOURCE_VAL_V52_DIAGNOSTIC_ONLY',
        'fixed_test_read': False,
        'target_data_read': False,
        'refitting_performed': False,
        'threshold_fitting_performed': False,
        'parameter_tuning_performed': False,
        'operation_or_phase_labels_used': False,
        'v52_primary_candidate_passed': False,
        'v52_primary_decision_confirmed': (
            'STOP_V52_PRIMARY_AND_REPORT_ABLATIONS_ONLY'),
        'scale_hold_cost_by_candidate_and_group': hold_costs,
        'strict_scale_hold_verdict_by_candidate': hold_verdicts,
        'primary_scale_hold_passed': primary_hold_passed,
        'scale_hold_decision': (
            'KEEP_ONE_FRAME_SCALE_HOLD_AS_VALID_OBSERVATION'
            if primary_hold_passed else
            'DISABLE_V52_SCALE_HOLD_AS_VALID_OBSERVATION'),
        'matched_coverage_audit': coverage,
        'image_quality_hypothesis_verdict': image_hypothesis,
        'next_development_candidate': (
            'IMAGE_QUALITY_ABLATION_HYPOTHESIS_ONLY'
            if image_hypothesis['passes_hypothesis_gate'] else
            'NO_V52_ABLATION_SUPPORTED_FOR_FURTHER_DEVELOPMENT'),
        'automatic_ablation_promotion': False,
        'v51_status': 'KEEP_AS_REFERENCE',
        'eligible_for_fixed_test_reuse_as_final_validation': False,
        'eligible_for_unknown_sequence_claim': False,
        'claim_limit': contract['claim_limit'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v52-report', required=True)
    parser.add_argument('--v52-contract', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    identities = {name: _identity(path) for name, path in (
        ('v52_report', args.v52_report),
        ('v52_contract', args.v52_contract),
        ('contract', args.contract))}
    report = json.loads(Path(args.v52_report).read_text())
    contract = json.loads(Path(args.contract).read_text())
    validate_contract(contract)
    _require_identity(
        identities['v52_report'],
        contract['expected_inputs']['v52_report_sha256'], 'v52_report')
    _require_identity(
        identities['v52_contract'],
        contract['expected_inputs']['v52_contract_sha256'], 'v52_contract')
    payload = run(report, contract)
    payload['inputs'] = identities
    output = _write_exact(args.out_json, payload)
    primary = report['primary_candidate']
    primary_cost = payload['scale_hold_cost_by_candidate_and_group'][
        primary]['all']
    print(json.dumps({
        'output': output,
        'v52_primary_decision': payload['v52_primary_decision_confirmed'],
        'primary_scale_hold_cost': primary_cost,
        'primary_scale_hold_passed': payload['primary_scale_hold_passed'],
        'scale_hold_decision': payload['scale_hold_decision'],
        'next_development_candidate': payload['next_development_candidate'],
        'claim_status': payload['claim_status'],
        'fixed_test_read': False,
    }, indent=2))


if __name__ == '__main__':
    main()
