"""S049 changed frozen versions/roles, original contracts remain unchanged."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from crane_project.utils import port_size_direct_continuation_downstream_v1 as d
from crane_project.tools import eval_port_size_direct_continuation_downstream_v1 as e
from crane_project.tools.analyze_port_size_direct_continuation_downstream_v1 import verify_depth,verify_detection
from test_port_size_a20_protection_downstream_v1 import depth_fixture


def fixture():
    p,t,m,c=depth_fixture()
    for row in p:
        row.pop('p5')
        for a in ('d02','dc6','dr6','dc1','dr1'):row[a]=list(row['a20'])
    return p,t,m,c


def test_all_ten_actual_versions_main_and_aux_fixed_before_evaluation():
    assert len(d.ARMS)==10 and len(d.HEADS)==8
    p=d.protocol();assert set(p['main_versions'])=={'b','midpoint','d02','c6','r6','dc6','dr6'}
    assert set(p['auxiliary_versions'])=={'a20','dc1','dr1'}
    assert p['limits']['training_updates']==0 and not p['limits']['selection'] and not p['limits']['formula_refit']
    assert not p['limits']['fixed_dev'] and not p['limits']['train03']
    data=fixture();snap=deepcopy(data);rr,s=d.depth_report(*data)
    assert data==snap and len(s['paired'])==45 and s['all_versions_common_valid']['count']==2
    assert s['paired']['c6->dc6']['tied']==2 and s['paired']['r6->dr6']['tied']==2
    verify_depth(rr,s,dict(camera=data[2]['camera'],physical_reference=data[2]['obb_reference_geometry']),data[3])


@pytest.mark.parametrize('arm',['dc6','dr6','dc1','dr1'])
def test_exact_continuation_metadata_source_epoch_updates(arm):
    spec=d.HEADS[arm];p=dict(protocol='port_size_direct_continuation_v1',arm=spec['arm'],epoch=spec['epoch'],continuation_updates=spec['epoch']*639,source_D02_sha256=d.PINS['d02'],input_identity=dict(frontend=dict(b_sha256=d.PINS['b'],m_sha256=d.PINS['midpoint'])))
    d.head_metadata(p,arm)
    for k,v in [('protocol','port_size_a20_protection_train_v1'),('source_D02_sha256',d.PINS['a20']),('continuation_updates',0),('epoch',3),('arm','P')]:
        q=deepcopy(p);q[k]=v
        with pytest.raises(ValueError):d.head_metadata(q,arm)
    q=deepcopy(p);q['input_identity']['frontend']['m_sha256']='bad'
    with pytest.raises(ValueError):d.head_metadata(q,arm)


@pytest.mark.parametrize('arm',list(d.HEADS))
def test_head_family_original_state_shape_and_no_trainable_weights_in_eval(arm):
    h=d.make_head(arm).eval().requires_grad_(False)
    assert sum(p.numel() for p in h.parameters())==(2746 if arm in d.DIRECT else 2800)
    assert not h.training and not any(p.requires_grad for p in h.parameters())


def val_fixture():
    b=[0.,0.,100.,50.,0.,.8]
    r=dict(delivery={a:'accepted' for a in d.HEADS},failed_checks={a:[] for a in d.HEADS},representation_valid={a:True for a in d.HEADS},pre_guard={a:list(b) for a in d.HEADS},**{a:list(b) for a in d.ARMS})
    return [dict(deepcopy(r),image=str(i)) for i in range(887)]


@pytest.mark.parametrize('arm',['dc6','dr6','dc1','dr1'])
def test_full_val_scores_delivery_preguard_and_missing_cannot_be_skipped(arm):
    r=val_fixture();ref=deepcopy(r);proof=d.compare_boxes(r,ref)
    with pytest.raises(ValueError):d.require_val_proof(proof)
    proof['intermediate_frames']=887;d.require_val_proof(proof)
    for field,value in [('delivery','fallback'),('representation_valid',False),('failed_checks',['axis'])]:
        q=deepcopy(r);q[0][field][arm]=value
        with pytest.raises(ValueError):d.compare_boxes(q,ref)
    q=deepcopy(r);q[0]['pre_guard'][arm][2]+=1
    with pytest.raises(ValueError):d.compare_boxes(q,ref)
    r[0][arm][5]+=.0000001
    with pytest.raises(ValueError):d.compare_boxes(r,ref)
    with pytest.raises(ValueError):d.compare_boxes(ref[:1],ref[:1])


def test_numeric_validity_and_signed_propagation_not_additive_mae():
    p,t,m,c=fixture();p[0]['dc6'][2]=1e100
    _,s=d.depth_report(p,t,m,c)
    assert s['paired']['c6->dc6']['common']==1 and s['paired']['c6->dc6']['lost_numeric_frame_ids']==['frame_00000']
    assert s['all_versions_common_valid']['count']==1
    p,t,m,c=fixture();p[0]['dr6'][3]=49.;p[1]['dr6'][3]=51.
    rr,s=d.depth_report(p,t,m,c);met=s['methods']['dr6']['depth_metrics']
    assert met['mae_m']>abs(met['bias_m'])
    assert sum(s['propagation']['dr6']['signed_error_component_mean_m'].values())==pytest.approx(met['bias_m'])
    rr[0]['dr6']['depth']['z_m']+=.01
    with pytest.raises(ValueError):verify_depth(rr,s,dict(camera=m['camera'],physical_reference=m['obb_reference_geometry']),c)


def test_workflow_transitions_start_and_comparator_near_delivery_complete():
    rr=[]
    for domain in ('real','sim'):
        for i in range(3):
            b=[0.,0.,100.,50.,0.,.8]
            r=dict(image=domain+'_seq_fixture_'+str(i),sequence=domain+'_seq_fixture',frame_id=i,domain=domain,scale=1.,split='test',gt=b[:5],**{a:list(b) for a in d.ARMS})
            r['d02'][2]=111.;r['dc6'][2]=101.
            r.update(pre_guard={a:list(r[a]) for a in d.HEADS},delivery={a:'accepted' for a in d.HEADS},failed_checks={a:[] for a in d.HEADS},representation_valid={a:True for a in d.HEADS})
            if i==2:
                for a in d.ARMS:r[a]=None
                r['pre_guard']={a:None for a in d.HEADS}
            rr.append(r)
    s=d.detection_report(rr,False)
    assert len(s['workflow_transitions']['overall'])==45
    assert s['workflow_transitions']['sim']['d02->dc6']['wrong_to_right']==2
    assert s['diagnostics']['real']['d02']['near_and_joint']['near']['long']['positive_harm_sum']==pytest.approx(.22)
    assert s['diagnostics']['real']['d02']['pre_guard_sizes']['joint10']==0
    with pytest.raises(ValueError):verify_detection(rr,s)


def test_no_run_has_no_side_effect_or_epoch_override(tmp_path):
    with pytest.raises(ValueError):e.run(SimpleNamespace(run=False,out_dir=tmp_path/'unused'))
    assert not (tmp_path/'unused').exists()
    text=e.parser().format_help();assert '--direct-run' in text and '--epoch' not in text
    assert d.protocol()['resources']['physical_gpu_choices']==[2,3]


def test_historical_native_guard_only_size_heads_and_deep_rows_exact(tmp_path):
    import json,hashlib
    folder=tmp_path/'history';(folder/'val').mkdir(parents=True)
    b=[0.,0.,100.,50.,0.,.8];rr=dict(image='frame',image_sha256='same',native_meta=dict(scale_xy=[1.,1.]),gt=b[:5],sequence='real_seq_fixture',domain='real',**{a:list(b) for a in d.ARMS},pre_guard={a:list(b) for a in d.HEADS},delivery={a:'accepted' for a in d.HEADS},failed_checks={a:[] for a in d.HEADS},representation_valid={a:True for a in d.HEADS})
    p=folder/'val/evaluation_rows.jsonl';p.write_text(json.dumps(rr)+'\n')
    (folder/'artifacts.json').write_text(json.dumps(dict(files={'val/evaluation_rows.jsonl':hashlib.sha256(p.read_bytes()).hexdigest()})))
    assert e.historical_replay([rr],folder,'val')['output_reuse'] is False
    changed=deepcopy(rr);changed['delivery']['c6']='fallback'
    with pytest.raises(ValueError,match='guard'):e.historical_replay([changed],folder,'val')
    changed=deepcopy(rr);changed['d02'][2]+=1
    with pytest.raises(ValueError,match='output'):e.historical_replay([changed],folder,'val')
