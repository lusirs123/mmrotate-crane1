"""Only new size qualification/selection logic; no model or optimizer updates."""
from copy import deepcopy
from crane_project.tools import train_symeood_110_lvsr_pair_v1 as m


def fixture():
    summary, control, near, damage, old_damage = {}, {}, {}, {}, {}
    for d in ('real', 'sim'):
        temporal = dict(mcml_max=2, mcml_mean=1., tdr_w10=dict(numerator=5, denominator=5),
            aci=.9, aci_adjacent_pairs=6, dfr_percent_per_frame=1.)
        metric = dict(sizes=dict(edges={e:dict(mae=.06) for e in ('long','short')},
            joint10=200, a=dict(abs_mae=.03), r=dict(abs_mae=.03)), mean_riou_full_frame=.85,
            temporal={d:temporal})
        candidate = deepcopy(metric)
        candidate['sizes'].update(edges={e:dict(mae=.04) for e in ('long','short')}, joint10=220,
            a=dict(abs_mae=.02), r=dict(abs_mae=.02))
        candidate['mean_riou_full_frame'] = .9
        candidate['temporal'][d]['dfr_percent_per_frame'] = 99.
        original = deepcopy(candidate)
        original['sizes'].update(edges={e:dict(mae=.05) for e in ('long','short')},joint10=210)
        summary[d] = dict(midpoint=metric, edge_residual=candidate,
            paired=dict(frozen_identity=True, new_output=0, lost_output=0))
        control[d] = dict(edge_residual=original)
        near[d] = dict(near={e:dict(after_mae=.01, before_mae=.01) for e in ('long','short')})
        damage[d] = {e+'_near_positive_harm_mean_pct':.02 for e in ('long','short')}
        old_damage[d] = {e+'_near_positive_harm_mean_pct':.04 for e in ('long','short')}
        for e in ('long','short'):
            damage[d][e+'_large10_mae'] = .10
            old_damage[d][e+'_large10_mae'] = .11
        damage[d]['long_near_material_harmed_0p1pp'] = 1
        old_damage[d]['long_near_material_harmed_0p1pp'] = 2
    summary['overall'] = dict(midpoint=dict(mean_riou_full_frame=.85),
        edge_residual=dict(mean_riou_full_frame=.9,sizes=dict(joint10=440)))
    control['overall'] = dict(edge_residual=dict(sizes=dict(joint10=420)))
    return summary, near, damage, control, old_damage


def test_new_gate_reports_DFR_without_changing_old_failure():
    args = fixture()
    gate = m.eligibility(*args)
    assert gate['pass_all']
    assert not gate['old_protocol_reference_only']['pass_all']
    assert not gate['old_protocol_reference_only']['checks']['real/dfr']
    assert not any(k.endswith('/dfr') for k in gate['checks'])
    args[0]['real']['paired']['frozen_identity'] = False
    assert not m.eligibility(*args)['pass_all']


def test_units_strict_joint_accurate_edge_and_empty_difficult_subset():
    args = fixture()
    args[2]['sim']['long_near_positive_harm_mean_pct'] = .04
    assert 'sim/near_long_harm_strict_original' in m.eligibility(*args)['failed']
    args = fixture()
    args[0]['overall']['edge_residual']['sizes']['joint10'] = 420
    assert 'original/overall_joint_strict' in m.eligibility(*args)['failed']
    args = fixture()
    args[2]['sim']['short_large10_mae'] = None
    assert 'sim/short_large10_vs_original' in m.eligibility(*args)['failed']
    args = fixture()
    # pct field is divided by 100; .1pp is .001 in relative-fraction units.
    args[2]['real']['short_near_positive_harm_mean_pct'] = .140001
    assert 'real/short_near_harm_vs_original' in m.eligibility(*args)['failed']


def test_joint_priority_qualified_selection_and_fixed_primary24():
    s = fixture()[0]
    a = dict(epoch=1, eligibility=dict(pass_all=False), selection_key=m.selection_key(s,1))
    t = deepcopy(s)
    t['overall']['edge_residual']['sizes']['joint10'] -= 1
    b = dict(epoch=2, eligibility=dict(pass_all=True), selection_key=m.selection_key(t,2))
    chosen = m.choose([a,b])
    assert chosen['selected']==chosen['representative']==2 and chosen['fixed_primary']==24
    b['eligibility']['pass_all']=False
    chosen = m.choose([a,b])
    assert chosen['selected'] is None and chosen['representative']==1
