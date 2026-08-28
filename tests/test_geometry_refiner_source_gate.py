from crane_project.utils.geometry_refiner_source_gate import (
    relaxed_composite_gate)


REFERENCE = {
    'real/R_center(%)': 99.12,
    'real/mean_RIoU': 0.7518,
    'real/DFR(%/frame)': 4.6181,
    'real/ACI': 0.9253,
    'real/TDR_w10(%)': 100.0,
    'real/MCML_max(frames)': 0,
    'sim/A-RMSE(deg)': 13.2351,
    'sim/R_center(%)': 99.61,
    'sim/mean_RIoU': 0.8023,
    'sim/DFR(%/frame)': 4.8309,
    'sim/ACI': 0.9075,
    'sim/TDR_w10(%)': 100.0,
    'sim/MCML_max(frames)': 2,
}


def test_relaxed_gate_allows_average_gain_with_small_individual_decline():
    candidate = dict(REFERENCE)
    candidate.update({
        'real/mean_RIoU': 0.775,
        'sim/mean_RIoU': 0.835,
        'real/DFR(%/frame)': 4.15,
        'sim/DFR(%/frame)': 4.35,
        'real/ACI': 0.920,
        'sim/ACI': 0.900,
    })
    result = relaxed_composite_gate(candidate, REFERENCE)
    assert result['composite_mean_relative_gain'] > 0.005
    assert result['passed'] is True


def test_relaxed_gate_rejects_severe_single_metric_regression():
    candidate = dict(REFERENCE)
    candidate.update({
        'real/mean_RIoU': 0.80,
        'sim/mean_RIoU': 0.86,
        'sim/DFR(%/frame)': 6.0,
    })
    result = relaxed_composite_gate(candidate, REFERENCE)
    assert result['checks']['sim_dfr_increase_le_0p75pp'] is False
    assert result['passed'] is False


def test_relaxed_gate_still_requires_positive_composite_gain():
    result = relaxed_composite_gate(dict(REFERENCE), REFERENCE)
    assert result['composite_mean_relative_gain'] == 0.0
    assert result['passed'] is False
