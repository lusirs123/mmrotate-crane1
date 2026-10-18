from copy import deepcopy
import pytest
from crane_project.tools import analyze_port_size_correction_behavior_v1 as a


@pytest.mark.parametrize('before,after,kind,effect', [
    (80, 70, 'wrong_direction', 'harmed'),
    (120, 130, 'wrong_direction', 'harmed'),
    (80, 130, 'correct_overshoot_harm', 'harmed'),
    (120, 70, 'correct_overshoot_harm', 'harmed'),
    (80, 110, 'correct_crossed_beneficial', 'improved'),
    (120, 90, 'correct_crossed_beneficial', 'improved'),
    (80, 85, 'correct_not_crossed_beneficial', 'improved'),
    (80, 100, 'correct_hit_gt', 'improved'),
    (80, 120, 'correct_equal_error', 'tied'),
    (80, 80, 'unchanged', 'tied'),
    (100, 101, 'm_exact_moved', 'harmed'),
    (102, 99, 'correct_crossed_beneficial', 'improved'),
])
def test_two_directions_crossing_and_actual_relative_error(before, after, kind, effect):
    x = a.classify(before, after, 100)
    assert x['kind'] == kind and x['effect'] == effect
    assert x['abs_error_delta'] == pytest.approx(abs(after/100-1)-abs(before/100-1))
    if kind == 'correct_not_crossed_beneficial':
        assert x['beneficial_under_remaining_gt3'] and x['beneficial_under_remaining_gt10']


def test_harm_magnitude_is_not_signed_bias_cancellation():
    x, y = a.classify(80, 70, 100), a.classify(80, 90, 100)
    v = a.summary([x, y])
    assert v['mean_error_delta_pp'] == pytest.approx(0)
    assert v['harm_sum_pp_edges'] == pytest.approx(10)
    assert v['gain_sum_pp_edges'] == pytest.approx(10)
    assert v['harmed'] == v['improved'] == 1
    assert a.summary([])['mean_harm_pp'] is None


def fixture(image='one', missing=False):
    m = None if missing else [0, 0, 100, 40, 0, .9]
    r = dict(image=image, sequence='sim1', frame_id=1, domain='sim',
             scale=1., split='train_sim', gt=[0, 0, 100, 50, 0], b=m, midpoint=m)
    left, right = deepcopy(r), deepcopy(r)
    # Exactly accurate long edge moved by both arms; wrong short edge improves.
    left['edge_residual'] = None if missing else [0, 0, 102, 45, 0, .9]
    right['edge_residual'] = None if missing else [0, 0, 103, 46, 0, .9]
    return {(1., image): r}, {(1., image): left}, {(1., image): right}


def test_paired_identity_missing_and_cross_edge_taxonomy():
    b, c, n = fixture(); bm, cm, nm = fixture('missing', True)
    rs = a.bind({**b, **bm}, {**c, **cm}, {**n, **nm}, 'train')
    assert len(rs) == 2 and sum(r['present'] for r in rs) == 1
    row = next(r for r in rs if r['present'])
    assert row['edges']['cached_p3'][0]['cross_edge_near_harmed']
    assert row['edges']['cached_p3'][0]['kind'] == 'm_exact_moved'
    assert not row['edges']['cached_p3'][1]['near']
    behavior, overlap = a.analyze(rs)
    assert a.validate_decomposition(behavior, overlap)['category_count_and_harm_amount_conservation']
    cell = next(x for x in overlap if x['group'] == 'sim' and x['edge'] == 'long'
                and x['stratum'] == 'near' and x['category'] == 'both_harmed')
    assert cell['count'] == cell['denominator_edges'] == 1
    assert cell['kind_pairs'] == {'m_exact_moved|m_exact_moved': 1}
    assert all(x['frames'] == 2 and x['missing'] == 1 for x in behavior if x['group'] == 'sim')
    bad = deepcopy(n); bad[1., 'one']['edge_residual'][0] = 1
    with pytest.raises(ValueError): a.bind(b, c, bad, 'train')
    with pytest.raises(ValueError): a.bind(b, c, {}, 'train')


def test_same_frame_exclusive_damage_is_not_common_failure():
    b, c, n = fixture()
    n[1., 'one']['edge_residual'][2] = 100
    _, overlap = a.analyze(a.bind(b, c, n, 'train'))
    cells = [x for x in overlap if x['group'] == 'overall' and x['edge'] == 'long' and x['stratum'] == 'all']
    assert sum(x['count'] for x in cells) == 1
    assert next(x for x in cells if x['category'] == 'cached_only_harmed')['count'] == 1
    assert next(x for x in cells if x['category'] == 'both_harmed')['count'] == 0


def test_omitted_or_corrupt_failure_categories_stop():
    b, c, n = fixture()
    behavior, overlap = a.analyze(a.bind(b, c, n, 'train'))
    bad = deepcopy(behavior)
    next(x for x in bad if x['kind'] == 'm_exact_moved')['count'] += 1
    with pytest.raises(ValueError): a.validate_decomposition(bad, overlap)
    bad_overlap = deepcopy(overlap)
    bad_overlap[0]['count'] += 1
    with pytest.raises(ValueError): a.validate_decomposition(behavior, bad_overlap)


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_lengths_stop(value):
    with pytest.raises(ValueError): a.classify(value, 100, 100)
