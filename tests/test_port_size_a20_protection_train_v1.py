from copy import deepcopy
import io
import json
from pathlib import Path
import pytest
import torch
from crane_project.utils import port_size_a20_protection_train_v1 as t
from crane_project.utils import port_size_qalign_v1 as old
from crane_project.tools import run_port_size_a20_protection_v1 as run
from test_port_size_native_p3_v1 import fixture


def row(m=90.,a=95.,split='train'):
    return dict(image='image',scale=1.,split=split,domain='sim',sequence='sim_seq',frame_id=0,
        gt=[0,0,100,50,0],midpoint=[0,0,m,50,0,.8],edge_residual=[0,0,a,50,0,.8],pre_guard=[0,0,a,50,0,.8],size_delivery='accepted')


def test_float64_membership_not_recomputed_after_float32_rounding():
    r=row(m=103.00000001,a=104.)
    table=t.reference_table([r]);assert table['image',1.]['near'][0] is False
    packed=t.pack_references([r],table,'cpu',torch,training=True)
    assert packed['active'].dtype==torch.bool and packed['reference'].dtype==torch.float64
    assert not bool(packed['near'][0,0])


def test_val_cannot_enter_training_reference_batch():
    r=row(split='val');table=t.reference_table([r])
    with pytest.raises(ValueError):t.pack_references([r],table,'cpu',torch,training=True)


def test_two_real_head_C_updates_use_original_loss_and_adam_exactly():
    x=list(fixture()[:6]);gt=x[5][:,:5].clone();gt[:,2]*=.97;gt[:,3]*=1.02;x.append(gt)
    torch.manual_seed(1703);a=old.AlignedSizeHead('fixed_b');b=deepcopy(a)
    oa=torch.optim.Adam(a.parameters(),lr=.0001);ob=torch.optim.Adam(b.parameters(),lr=.0001)
    refs=dict(active=torch.tensor([[True,True]]),reference=torch.zeros(1,2,dtype=torch.float64))
    for _ in range(2):
        oa.zero_grad();v=a(*x[:-1]);loss=old.size_loss(v,x[-1],x[-2],x[3]);loss.backward();torch.nn.utils.clip_grad_norm_(a.parameters(),10.);oa.step()
        result,_=run.update(b,ob,x,refs,'C',torch)
        assert float(loss)==result['total']
        assert run.equal(a.state_dict(),b.state_dict(),torch) and run.equal(oa.state_dict(),ob.state_dict(),torch)


@pytest.mark.parametrize('arm',['R','P'])
def test_head_optimizer_reload_next_update(arm):
    x=list(fixture()[:6]);gt=x[5][:,:5].clone();gt[:,2]*=.97;gt[:,3]*=1.02;x.append(gt)
    a=old.AlignedSizeHead('fixed_b');oa=torch.optim.Adam(a.parameters(),lr=.0001)
    refs=dict(active=torch.tensor([[True,True]]),reference=torch.zeros(1,2,dtype=torch.float64))
    run.update(a,oa,x,refs,arm,torch)
    stream=io.BytesIO();torch.save(dict(head=a.state_dict(),opt=oa.state_dict()),stream);stream.seek(0);got=torch.load(stream)
    b=old.AlignedSizeHead('fixed_b');b.load_state_dict(got['head']);ob=torch.optim.Adam(b.parameters(),lr=.0001);ob.load_state_dict(got['opt'])
    run.update(a,oa,x,refs,arm,torch);run.update(b,ob,x,refs,arm,torch)
    assert run.equal(a.state_dict(),b.state_dict(),torch) and run.equal(oa.state_dict(),ob.state_dict(),torch)


def test_protection_report_keeps_fallback_harm_and_retention_separate():
    r=row();table=t.reference_table([r]);changed=deepcopy(r)
    changed['pre_guard']=[0,0,85,52,0,.8];changed['edge_residual']=r['midpoint'];changed['size_delivery']='fallback'
    report=t.protection_report([changed],table)['sim'];e=report['edges']['long']
    assert report['fallback']==1 and e['delivered']['useful']['fraction']==0
    assert e['pre_guard']['useful_direction_reversed']==1 and e['delivered']['useful_direction_reversed']==0
    assert report['edges']['short']['pre_guard']['near_harmed']==1 and report['edges']['short']['delivered']['near_harmed']==0


def eligible_fixture():
    s={};diag={};a20={}
    def cell(edges,a=.03,riou=.9):
        return dict(sizes=dict(edges={e:dict(mae=x) for e,x in zip(('long','short'),edges)},a=dict(abs_mae=a),r=dict(abs_mae=a)),
            mean_riou_full_frame=riou)
    for domain in ('real','sim'):
        m=cell([.08,.08] if domain=='real' else [.05,.05]);a=cell([.07,.07] if domain=='real' else [.04,.04],.02,.91)
        c=cell([.07,.07] if domain=='real' else [.039,.039],.01,.92)
        temp=dict(dfr_percent_per_frame=1.,mcml_max=4,mcml_mean=2.,tdr_w10=dict(numerator=10,denominator=10),aci=.95,aci_adjacent_pairs=100)
        m['temporal']={domain:deepcopy(temp)};c['temporal']={domain:deepcopy(temp)}
        s[domain]=dict(midpoint=m,edge_residual=c,paired=dict(frozen_identity=True,new_output=0,lost_output=0));a20[domain]=dict(edge_residual=a)
        diag[domain]=dict(near={e:dict(before_mae=.01,after_mae=.01) for e in ('long','short')})
    s['overall']=dict(midpoint=cell([0,0],riou=.9),edge_residual=cell([0,0],riou=.92));a20['overall']=dict(edge_residual=cell([0,0],riou=.91))
    return s,diag,a20


def test_expanded_gates_and_strict_protection_not_relabelled():
    s,d,a=eligible_fixture();assert t.eligibility(s,d,a)['pass_all']
    s['sim']['edge_residual']['temporal']['sim']['dfr_percent_per_frame']=1.00001
    assert 'sim/dfr' in t.eligibility(s,d,a)['failed']
    d['real']['near']['long']['after_mae']=.0111
    assert 'real/near_long' in t.eligibility(s,d,a)['failed']
    assert json.loads(Path('crane_project/tools/port_size_a20_protection_train_v1_protocol.json').read_text())['val_evaluations']==18


def test_invalid_candidate_stops_instead_of_filtering():
    x=list(fixture());h=old.AlignedSizeHead('fixed_b');v=h(*x[:6]);v['candidate_valid'][0]=False
    refs=dict(active=torch.tensor([[True,True]]),reference=torch.zeros(1,2,dtype=torch.float64))
    with pytest.raises(ValueError):t.losses(v,x[5][:,:5],x[5],x[3],refs,'P',torch)
