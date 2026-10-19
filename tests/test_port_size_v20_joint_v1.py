"""S036 paired-unit, threshold and response conservation; no model needed."""
from copy import deepcopy
import math
import pytest
from crane_project.tools import analyze_port_size_v20_joint_v1 as a


def record(m=(100, 40), v=(101, 45), l=(103, 46), gt=(100, 50), image='one'):
    box = lambda z: [0., 0., *z, 0., .9] if z is not None else None
    base = dict(image=image, sequence='sim_seq', domain='sim', frame_id=1,
                scale=1., split='val', b=box(m), midpoint=box(m), gt=box(gt)[:5])
    x, y = deepcopy(base), deepcopy(base)
    for r, prediction in ((x, v), (y, l)):
        r['edge_residual'] = box(prediction)
        r['metrics'] = {k: {'riou': .8 if r[k] is not None else None}
                        for k in ('b', 'midpoint', 'edge_residual')}
    return a.bind({(1., image): x}, {(1., image): y}, 'val')[0]


def test_threshold_loss_keeps_other_edge_improvement_and_excess():
    r = record(m=(108, 54), v=(113, 51))
    assert a.transition(r, 'm', 'v20') == 'right_to_wrong'
    s = a.summary([r], 'm', 'v20')
    assert s['new_failure_other_edge_improved'] == 1
    assert s['after_excess_pp_max'] == pytest.approx(3.)
    assert s['after_both_edges_gt10'] == 0
    assert s['short_pp_edges_gain_sum'] == pytest.approx(6.)
    assert s['long_pp_edges_harm_sum'] == pytest.approx(5.)


def test_scale_improvement_can_coincide_with_ratio_damage():
    r = record(m=(90, 45), v=(100, 45), l=(100, 45))
    x, y = r['g']['m'], r['g']['v20']
    assert abs(y['a']) < abs(x['a'])
    assert abs(y['r']) > abs(x['r'])
    z = a.summary([r], 'm', 'v20')
    assert z['a_log_gain_sum'] > 0 and z['r_log_harm_sum'] > 0


def test_cohort_partition_uses_fixed_m_not_v_l_pair_benefit():
    rows = [record(v=(100, 45), l=(100, 46), image='retained'),
            record(v=(100, 45), l=(100, 38), image='lost'),
            record(v=(100, 38), l=(100, 45), image='new'),
            record(v=(100, 38), l=(100, 39), image='neither')]
    assert [a.cohort(r) for r in rows] == list(a.COHORTS)
    rr = a.short_responses(rows)
    lost = rr[1]
    assert lost['reversed_direction'] and lost['L_short_kind'] == 'wrong_direction'
    assert lost['M_short_group'] == 'need_enlarge'


def test_lost_can_be_overshoot_or_exact_no_change_not_just_small_update():
    overshoot = record(v=(100, 45), l=(100, 65))
    unchanged = record(v=(100, 45), l=(100, 40))
    assert a.cohort(overshoot) == a.cohort(unchanged) == 'lost'
    r = a.short_responses([overshoot, unchanged])
    assert r[0]['L_short_kind'] == 'correct_overshoot_harm'
    assert r[0]['same_direction'] and not r[0]['smaller_absolute_update']
    assert r[1]['L_short_kind'] == 'unchanged'


def test_gains_harms_and_transition_denominators_conserve():
    r = record(m=(100, 50), v=(105, 45), l=(100, 50))
    z = a.summary([r], 'm', 'v20')
    assert z['long_pp_edges_harm_sum'] == pytest.approx(5)
    assert z['short_pp_edges_harm_sum'] == pytest.approx(10)
    assert sum(z[k] for k in ('right_to_right', 'right_to_wrong', 'wrong_to_right',
                              'wrong_to_wrong', 'new_output', 'lost_output', 'both_missing')) == z['frames']
    assert a.summary([], 'm', 'v20')['before_RIoU_full'] is None


def test_missing_is_separate_from_size_error():
    r = record()
    for method in r['g']:
        r['g'][method] = None
    r['member'] = 'missing'
    z = a.summary([r], 'm', 'v20')
    assert z['both_missing'] == 1 and z['paired_outputs'] == 0
    assert z['right_to_wrong'] == z['wrong_to_right'] == 0
    assert z['after_RIoU_full'] == 0
    assert not a.short_responses([r])


def test_new_output_does_not_become_size_correction():
    r = record()
    r['g']['b'] = None
    z = a.summary([r], 'b', 'v20')
    assert z['new_output'] == 1 and z['wrong_to_right'] == 0
    assert z['paired_outputs'] == 0


def test_unsaved_train_riou_is_unavailable_not_zero_quality():
    r = record()
    for g in r['g'].values():
        g['riou'] = None
    s = a.summary([r], 'm', 'v20')
    assert s['before_RIoU_full'] is s['after_RIoU_full'] is None
    assert s['RIoU_paired_evaluable_outputs'] == 0
    assert s['paired_outputs'] == 1


def test_axis_canonicalization_and_nonfinite_values():
    g = a.geometry([0, 0, 100, 50, 0], [0, 0, 50, 100, -math.pi/2], .9)
    assert g['e'] == [0, 0] and g['a'] == g['r'] == 0
    with pytest.raises(ValueError):
        a.geometry([0, 0, 100, 50, 0], [0, 0, 0, 10, 0], .9)
    with pytest.raises(ValueError):
        a.geometry([0, 0, 100, 50, 0], [0, 0, 100, 50, 0], float('nan'))


def test_scales_not_merged_and_joint_cells_partition_present():
    rows = [record(image='one'), record(image='two')]
    for r, scale in zip(rows, (1., .5)):
        r['split'] = 'train'
        r['scale'] = scale
    transitions, joint, _, cohorts = a.analyze(rows)
    for scale in (.5, 1.):
        sub = [r for r in joint if r['split'] == 'train' and r['scale'] == scale
               and r['group'] == 'sim' and r['pair'] == 'm->v20']
        assert sum(r['paired_outputs'] for r in sub) == 1
        sub = [r for r in cohorts if r['scale'] == scale and r['group'] == 'sim'
               and r['M_member'] == r['M_short_group'] == 'all']
        assert sum(r['frames'] for r in sub) == 1
    assert {r['scale'] for r in transitions} == {.5, 1.}
