"""S037 engineering and evaluation semantics; no real-visual effectiveness claim."""
from copy import deepcopy
import inspect
import pytest
import torch
from crane_project.utils import port_size_direction_evidence_v1 as d
from crane_project.tools import audit_port_size_direction_evidence_v1 as audit

torch.set_num_threads(1)


@pytest.mark.parametrize('swap,aniso', [(False,False),(True,True)])
def test_natural_nonzero_head_exact_and_frozen(swap,aniso):
    torch.manual_seed(1703);x=audit.fixture(torch,swap,aniso)
    h=d.v.LocalVectorSizeHead().eval()
    with torch.no_grad():h.terminal.weight.normal_(0,.01);h.terminal.bias.normal_(0,.01)
    before=deepcopy(h.state_dict())
    with torch.no_grad():
        a=h(*x);b=d.readout(h,*x,x[-1])
    for k in ('boxes_original','candidate_original','vector_valid'):
        assert torch.equal(a[k],b[k])
    assert all(torch.equal(v,before[k]) for k,v in h.state_dict().items())


def test_gt_free_single_edge_proposals_and_channel_dependency():
    x=audit.fixture(torch);q,safe=d.proposals(x[-1]);assert safe.all()
    assert 'gt' not in inspect.signature(d.proposals).parameters
    assert 'gt' not in inspect.signature(d.readout).parameters
    assert torch.equal(q[:,0],x[-1])
    assert torch.equal(q[:,1,3],x[-1][:,3])
    assert q[0,1,2]>x[-1][0,2] and q[0,2,2]<x[-1][0,2]
    h=d.v.LocalVectorSizeHead();seen=[]
    hook=h.stem[0].register_forward_pre_hook(lambda _,args:seen.append(args[0].detach().clone()))
    with torch.no_grad():
        d.readout(h,*x,q[:,0]);d.readout(h,*x,q[:,1])
    hook.remove()
    assert torch.equal(seen[0][:,:259],seen[1][:,:259])
    assert not torch.equal(seen[0][:,259:],seen[1][:,259:])


def test_q_source_guard_not_falsely_accepted_as_midpoint_and_fallback_separate():
    x=audit.fixture(torch);h=d.v.LocalVectorSizeHead();q=x[-1].clone();q[:,2]*=1.6
    with pytest.raises(ValueError):h(*x[:-1],q)
    with torch.no_grad():o=d.readout(h,*x,q)
    assert torch.equal(o['preguard_candidate'],q)  # neutral head, before guard
    assert not o['accepted'].any()
    assert torch.equal(o['boxes_original'],x[-1])


def test_axis_crossing_is_unavailable_not_silently_sorted():
    x=list(audit.fixture(torch));x[2][:,2:4]=torch.tensor([[100.,99.9]])
    x[3]=x[2][:,:5].clone();x[5]=x[2].clone()
    q,safe=d.proposals(x[-1]);assert not safe[0,2] and not safe[0,3]
    with pytest.raises(ValueError,match='axis crossing'):
        d.readout(d.v.LocalVectorSizeHead(),*x,q[:,2])


def test_q_cannot_change_center_or_score():
    x=audit.fixture(torch);q=x[-1].clone();q[:,0]+=1.
    with pytest.raises(ValueError,match='center/angle/score'):
        d.readout(d.v.LocalVectorSizeHead(),*x,q)


def test_donor_deterministic_same_video_scale_no_gt_or_self():
    records=[dict(image=str(i),sequence=v,scale=s) for v,s,ids in [('a',1.,range(4)),('a',.5,range(2)),('b',1.,range(1))] for i in ids]
    donors=d.donor_indices(records)
    for i,j in donors.items():
        if j is None:assert records[i]['sequence']=='b'
        else:
            assert i!=j and records[i]['sequence']==records[j]['sequence'] and records[i]['scale']==records[j]['scale']
    assert donors==d.donor_indices(records)


def test_near_not_required_to_hold_and_direction_not_equal_benefit():
    x=d.edge_evaluation(101.,100.,100.)
    assert x['required_group']=='near' and x['direction_correct'] is None and x['improved'] and not x['hold']
    x=d.edge_evaluation(80.,130.,100.)
    assert x['direction_correct'] and x['harmed']  # correct direction, overshoot
    x=d.edge_evaluation(80.,70.,100.)
    assert not x['direction_correct'] and x['harmed']


def response(visual,slot,q,p):
    return dict(image='one',sequence='sim1',domain='sim',scale=1.,visual=visual,slot=slot,
                stage='preguard',edge='long',predicted_edge=p,fallback=False,
                **d.edge_evaluation(q,p,100.))


def test_paired_visual_and_true_bidirectional_denominators():
    rows=[]
    for visual in d.VISUALS:
        for slot,q in [('long_minus',90.),('long_plus',110.)]:
            rows.append(response(visual,slot,q,100. if visual=='normal' else q))
    pairs,opposite=audit.paired_tables(rows)
    assert len(pairs)==4 and all(r['only_normal_direction_correct'] for r in pairs)
    normal=next(r for r in opposite if r['visual']=='normal')
    assert normal['crossed_GT'] and normal['both_nonnear'] and normal['both_directions_correct'] and normal['both_errors_improved']
    assert normal['final_size_secant']==0. and normal['correction_secant']==-1.
    assert all(not r['both_errors_improved'] for r in opposite if r['visual']!='normal')


def test_plus_minus_relative_m_does_not_imply_straddling_gt():
    rows=[response('normal','long_minus',70.,75.),response('normal','long_plus',80.,85.)]
    _,pairs=audit.paired_tables(rows)
    assert not pairs[0]['crossed_GT'] and not pairs[0]['both_directions_correct']


def test_empty_control_pairs_dont_imply_normal_wins():
    pairs,opposite=audit.paired_tables([response('normal','long_minus',90.,100.)])
    assert pairs==opposite==[]
