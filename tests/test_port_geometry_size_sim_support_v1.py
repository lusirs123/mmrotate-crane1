"""Diagnostic accounting checks; these examples are not experiment results."""
import math
import pytest
from crane_project.tools import analyze_port_geometry_size_sim_support_v1 as tool


def row(p=(100., 50.), q=(102., 49.), gt=(100., 50.)):
    box = lambda xy: [0., 0., *xy, 0., .9] if xy else None
    return dict(image='sim_1', domain='sim', sequence='sim_seq01', split='train', scale=1.,
                gt=[0., 0., *gt, 0.], b=box(p), midpoint=box(p), edge_residual=box(q),
                target_log=[.1, -.2], delta_roi=[.05, -.1])


def test_update_direction_is_not_error_direction():
    # Short shrink helps an oversized edge while long grows correctly.
    d = tool.frame_delta(row((90., 55.), (95., 52.)), 'midpoint', 'edge_residual')
    assert d['update_case'] == 'up/down'
    assert d['edge_case'] == 'up/up'
    s = tool.pair_summary([row((90., 55.), (95., 52.))], 'midpoint', 'edge_residual')
    assert s['long_gain_short_down'] == 1 and s['long_gain_short_down_harm'] == 0


def test_missing_outputs_do_not_become_size_corrections():
    rs = [row(None, (100., 50.)), row((100., 50.), None), row()]
    s = tool.pair_summary(rs, 'midpoint', 'edge_residual')
    assert s['frames'] == 3 and s['common_outputs'] == 1
    assert s['new_output'] == s['lost_output'] == 1
    assert sum(s['edge_cases'].values()) == 1


def test_canonical_axis_swap_and_three_percent_groups():
    r = row((50., 100.), (50., 100.))
    assert tool.errors(r, 'midpoint')['e'] == [0., 0.]
    assert tool.metrics.group(r['gt'], r['midpoint']) == 'near_correct'
    r = row((96., 48.), (100., 50.))
    assert tool.metrics.group(r['gt'], r['midpoint']) == 'need_enlarge'
    assert tool.errors(r, 'midpoint')['correct']  # size10 is a different set


def test_fixed_membership_and_bias_telescoping():
    r = row((105., 52.), (100., 50.))
    r['b'] = [0., 0., 110., 55., 0., .9]
    ds = [tool.frame_delta(r, a, b) for a, b in tool.PAIRS]
    assert len({d['m_group'] for d in ds}) == 1
    for axis in ('long', 'short'):
        k = axis+'_signed_error_after'; j = axis+'_signed_error_before'
        assert math.isclose((ds[0][k]-ds[0][j])+(ds[1][k]-ds[1][j]), ds[2][k]-ds[2][j], abs_tol=1e-15)


def test_loss_is_snapshot_and_mean_over_two_axes():
    assert tool.snapshot_loss(row(), True) == pytest.approx((.05+.15)/2)
    assert tool.snapshot_loss(row()) == pytest.approx((.0125+.05)/2)
    assert tool.snapshot_loss(row(None, None)) is None


def test_train_fitted_strata_do_not_use_gt_or_val():
    rs = [row((90., 45.)), row((110., 55.))]
    fitted = tool.fit_strata(rs)
    r = row((100., 50.)); label = tool.stratum(r, fitted)
    r['gt'][2:4] = [900., 2.]
    assert tool.stratum(r, fitted) == label
    assert fitted['fit_n'] == 2


def test_empty_summary_denominators():
    s = tool.cell_summary([])
    assert s['methods']['midpoint']['outputs'] == 0
    assert s['methods']['midpoint']['edges']['short']['mae'] is None
    assert s['pairs']['midpoint->edge_residual']['mean_delta']['short_error_delta'] is None


@pytest.mark.parametrize('counts', [dict(undersized=38, oversized=130, near3=580),
                                   dict(undersized=1, oversized=99, near3=1)])
def test_proposed_axis_weights_conserve_exposure_and_protect_near(counts):
    w = tool.axis_balance_weights(counts)
    assert w['near3'] == 1. and max(w.values()) <= 4.
    assert sum(counts[g]*w[g] for g in counts) == pytest.approx(sum(counts.values()))
    assert tool.axis_balance_weights(dict(undersized=0, oversized=2, near3=1)) is None
