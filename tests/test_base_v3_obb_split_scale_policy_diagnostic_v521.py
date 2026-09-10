import copy

import pytest

from crane_project.tools import (
    base_v3_obb_split_scale_policy_diagnostic_v521 as v521)


def _scale_report(frame_count=10, measurements=8, predictions=0,
                  unavailable=2, correct=6, bad=2, improved=0, degraded=0,
                  prediction_error=None, prediction_delta=None):
    available = frame_count-unavailable
    return {
        'frame_count': frame_count,
        'measurement_count': measurements,
        'prediction_count': predictions,
        'unavailable_count': unavailable,
        'output_coverage': available/frame_count,
        'correct_output_coverage': correct/frame_count,
        'bad_available_output_rate': bad/available,
        'prediction_improved_count': improved,
        'prediction_degraded_count': degraded,
        'mean_prediction_error': prediction_error,
        'mean_prediction_error_delta_vs_raw': prediction_delta,
        'longest_unavailable_run': unavailable,
    }


def _contract():
    return {
        'scale_hold_gate': {
            'minimum_correct_prediction_fraction': 0.5,
            'maximum_mean_error_delta_vs_raw': 0.0,
            'require_improved_not_fewer_than_degraded': True}}


def test_hold_cost_counts_correct_and_wrong_filled_observations():
    rejection = _scale_report()
    hold = _scale_report(
        predictions=2, unavailable=0, correct=7, bad=3,
        improved=1, degraded=1, prediction_error=0.2,
        prediction_delta=0.01)
    result = v521.hold_cost_summary(rejection, hold)
    assert result['filled_prediction_count'] == 2
    assert result['correct_prediction_count'] == 1
    assert result['wrong_prediction_count'] == 1
    assert result['new_bad_output_count'] == 1
    assert result['correct_prediction_fraction'] == 0.5


def test_strict_hold_gate_rejects_low_correctness_and_positive_error_delta():
    summary = {
        'filled_prediction_count': 19,
        'correct_prediction_count': 2,
        'wrong_prediction_count': 17,
        'correct_prediction_fraction': 2/19,
        'improved_prediction_count': 11,
        'degraded_prediction_count': 8,
        'mean_prediction_error_delta_vs_raw': 0.002}
    verdict = v521.strict_hold_verdict(
        {'all': summary}, _contract())
    assert verdict['checks']['improved_not_fewer_than_degraded'] is True
    assert verdict['checks'][
        'correct_prediction_fraction_at_least_half'] is False
    assert verdict['checks'][
        'mean_prediction_error_not_worse_than_current_raw'] is False
    assert verdict['passes_all'] is False


def test_curve_comparison_requires_identical_retained_count():
    first = [{'target_coverage': 0.9, 'retained_count': 9,
              'mean_error': 0.1, 'failure_rate': 0.2}]
    second = copy.deepcopy(first)
    second[0]['retained_count'] = 8
    with pytest.raises(RuntimeError, match='retained counts'):
        v521._curve_outcome(first, second)


def test_curve_comparison_records_win_loss_and_tradeoff():
    reference = [
        {'target_coverage': 0.9, 'retained_count': 9,
         'mean_error': 0.2, 'failure_rate': 0.3},
        {'target_coverage': 0.8, 'retained_count': 8,
         'mean_error': 0.1, 'failure_rate': 0.2},
        {'target_coverage': 0.7, 'retained_count': 7,
         'mean_error': 0.1, 'failure_rate': 0.2}]
    challenger = [
        {'target_coverage': 0.9, 'retained_count': 9,
         'mean_error': 0.1, 'failure_rate': 0.2},
        {'target_coverage': 0.8, 'retained_count': 8,
         'mean_error': 0.2, 'failure_rate': 0.3},
        {'target_coverage': 0.7, 'retained_count': 7,
         'mean_error': 0.08, 'failure_rate': 0.25}]
    result = v521._curve_outcome(challenger, reference)
    assert result['wins'] == [0.9]
    assert result['losses'] == [0.8]
    assert result['ties_or_tradeoffs'] == [0.7]


def test_contract_rejects_refitting_authorization():
    with pytest.raises(ValueError, match='refit'):
        v521.validate_contract({
            'protocol': v521.CONTRACT_PROTOCOL,
            'fixed_test_read': False, 'target_data_read': False,
            'refitting_authorized': True,
            'parameter_tuning_authorized': False})
