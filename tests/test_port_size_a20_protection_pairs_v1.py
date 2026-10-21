from copy import deepcopy
import pytest
from crane_project.tools.analyze_port_size_a20_protection_pairs_v1 import summarize_pair
from crane_project.utils.port_size_a20_protection_train_v1 import reference_table


def case():
    return dict(image='x',scale=1.,domain='sim',sequence='sim_seq',gt=[0,0,100,50,0],
        midpoint=[0,0,90,50,0,.8],edge_residual=[0,0,95,50,0,.8],pre_guard=[0,0,95,50,0,.8])


def cell(result,edge='long',mode='delivered'):
    return next(r for r in result if r['group']=='sim' and r['edge']==edge and r['mode']==mode)


def test_weaker_response_and_lost_useful_not_conflated_with_direction_reversal():
    a=case();a.update(split='val');refs=reference_table([a]);b=deepcopy(a);b['edge_residual']=[0,0,92,50,0,.8]
    r=cell(summarize_pair([a],[b],refs))
    assert r['same_direction_weaker']==1 and r['reversed_vs_left']==0 and r['retention_right']==pytest.approx(.4)
    b['edge_residual']=[0,0,88,50,0,.8];r=cell(summarize_pair([a],[b],refs))
    assert r['reversed_vs_left']==1 and r['left_useful_lost']==1 and r['retention_right']==0


def test_neutral_and_preguard_separate():
    a=case();a.update(split='val');refs=reference_table([a]);b=deepcopy(a);b['edge_residual']=a['midpoint'];b['pre_guard']=[0,0,88,50,0,.8]
    r=summarize_pair([a],[b],refs)
    assert cell(r)['returned_to_neutral']==1 and cell(r,mode='pre_guard')['reversed_vs_left']==1


def test_identity_mismatch_rejected():
    a=case();a.update(split='val');b=deepcopy(a);b['image']='other'
    with pytest.raises(ValueError):summarize_pair([a],[b],reference_table([a]))
