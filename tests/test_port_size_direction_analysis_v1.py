from collections import Counter
import pytest
from crane_project.tools import analyze_port_size_direction_evidence_v1 as a


def test_signed_amounts_remain_separate_and_tolerance_is_in_pp():
    c=Counter()
    for delta in (-2., 3., -0.00001):a.add_effect(c,'x',delta)
    assert c['x_gain_sum']==pytest.approx(2.00001)
    assert c['x_harm_sum']==3.
    assert c['x_improved']==c['x_harmed']==1


def test_boolean_none_is_not_false_and_frozen_strata_keep_scale():
    assert a.boolean('True') and not a.boolean('False')
    with pytest.raises(ValueError):a.boolean('')
    r=dict(scale='0.5',domain='sim',sequence='sim_seq08')
    assert list(a.strata(r))==[('0.5','overall'),('0.5','sim'),('0.5','sim_seq08')]


def test_counter_summary_retains_exact_pair_units():
    cells={('0.5','sim'):Counter(paired_count=3,only_normal=1,only_control=2)}
    rows=a.finish(cells,('scale','group'))
    assert rows==[dict(scale='0.5',group='sim',paired_count=3,only_normal=1,only_control=2)]


def test_direction_rate_denominator_excludes_near_and_empty_is_missing():
    rows=a.finish({('sim',):Counter(count=10,direction_evaluable=4,direction_correct=3),
                   ('empty',):Counter(count=2,direction_evaluable=0,direction_correct=0)},('group',))
    by={r['group']:r for r in rows}
    assert by['sim']['direction_correct_rate']==.75
    assert by['empty']['direction_correct_rate'] is None
