"""S047 fixed V/D downstream contracts; synthetic CPU only."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from crane_project.utils import port_size_structure_downstream_v1 as d
from crane_project.utils import port_size_structure_v1 as m
from crane_project.tools import eval_port_size_structure_downstream_v1 as e
from crane_project.tools.analyze_port_size_structure_downstream_v1 import verify_depth,verify_detection
from test_port_size_a20_protection_downstream_v1 import depth_fixture


def fixture():
    p,t,manifest,c=depth_fixture()
    for row in p:
        row.pop('p5');row['v20']=list(row['a20']);row['d02']=list(row['a20'])
    return p,t,manifest,c


def test_seven_actual_versions_and_21_depth_pairs():
    data=fixture();snapshot=deepcopy(data);rows,s=d.depth_report(*data)
    assert data==snapshot and len(s['paired'])==21 and s['all_versions_common_valid']['count']==2
    assert set(s['methods']['d02']['depth_metrics'])>={'max_abs_error_m','abs_rel','mae_m'}
    assert s['paired']['v20->d02']['tied']==2
    verify_depth(rows,s,dict(camera=data[2]['camera'],physical_reference=data[2]['obb_reference_geometry']),data[3])
    rows[0]['d02']['depth']['z_m']+=.01
    with pytest.raises(ValueError):verify_depth(rows,s,dict(camera=data[2]['camera'],physical_reference=data[2]['obb_reference_geometry']),data[3])


def test_common_valid_and_frame_pairing_no_filter():
    p,t,m,c=fixture();p[0]['d02'][2]=1e100
    _,s=d.depth_report(p,t,m,c)
    assert s['methods']['d02']['output_frame_count']==2 and s['methods']['d02']['numeric_depth_count']==1
    assert s['paired']['v20->d02']['common']==1 and s['paired']['v20->d02']['lost_numeric_frame_ids']==['frame_00000']
    assert s['all_versions_common_valid']['count']==1


def test_signed_bias_components_not_mae():
    p,t,m,c=fixture();p[0]['d02'][3]=49.;p[1]['d02'][3]=51.
    _,s=d.depth_report(p,t,m,c)
    assert s['paired']['v20->d02']['worsened']==2
    metrics=s['methods']['d02']['depth_metrics'];assert metrics['mae_m']>abs(metrics['bias_m'])
    prop=s['propagation']['d02'];assert sum(prop['signed_error_component_mean_m'].values())==pytest.approx(metrics['bias_m'])


def test_state_implementation_equivalence_requires_all_parameters():
    a=m.head('V_structure');v=deepcopy(a);heads=dict(v20=v,a20=a)
    assert e.equivalence(heads,torch)['effective_state_exact']
    with torch.no_grad():v.terminal.bias[0]+=.0001
    assert not e.equivalence(heads,torch)['effective_state_exact']
    heads['v20']=m.head('D_direct');assert not e.equivalence(heads,torch)['effective_state_exact']


@pytest.mark.parametrize('arm',['v20','d02'])
def test_fixed_metadata_rejects_epoch_or_frontend_changes(arm):
    spec=d.HEADS[arm];payload=dict(protocol='port_size_structure_v1',arm=spec['arm'],epoch=spec['epoch'],updates=spec['epoch']*639,input_identity=dict(b_sha256=d.PINS['b'],m_sha256=d.PINS['midpoint']))
    d.head_metadata(payload,arm)
    for k,v in [('epoch',24),('updates',0),('arm','historical')]:
        q=deepcopy(payload);q[k]=v
        with pytest.raises(ValueError):d.head_metadata(q,arm)
    q=deepcopy(payload);q['input_identity']['m_sha256']='bad'
    with pytest.raises(ValueError):d.head_metadata(q,arm)


def val_fixture():
    box=[0.,0.,100.,50.,0.,.8]
    row=dict(delivery={a:'accepted' for a in d.HEADS},failed_checks={a:[] for a in d.HEADS},representation_valid={a:True for a in d.HEADS},pre_guard={a:list(box) for a in d.HEADS},**{a:list(box) for a in d.ARMS})
    return [dict(deepcopy(row),image=str(i)) for i in range(887)]


@pytest.mark.parametrize('arm',list(d.HEADS))
def test_val_full_frames_score_guard_validity_and_preguard(arm):
    r=val_fixture();ref=deepcopy(r);proof=d.compare_boxes(r,ref)
    with pytest.raises(ValueError):d.require_val_proof(proof)
    proof['intermediate_frames']=887;d.require_val_proof(proof)
    for field,value in [('delivery','fallback'),('representation_valid',False),('failed_checks',['axis'])]:
        q=deepcopy(r);q[0][field][arm]=value
        with pytest.raises(ValueError):d.compare_boxes(q,ref)
    r[0]['pre_guard'][arm][2]+=1
    with pytest.raises(ValueError):d.compare_boxes(r,ref)
    r=deepcopy(ref);r[0][arm][5]+=.0000001
    with pytest.raises(ValueError):d.compare_boxes(r,ref)
    with pytest.raises(ValueError):d.compare_boxes(r[:1],ref[:1])


def test_diagnostics_include_near_positive_harm_preguard_and_v_d_transitions():
    records=[]
    for dom in ('real','sim'):
        for i in range(3):
            box=[0.,0.,100.,50.,0.,.8]
            r=dict(image=dom+'_seq_fixture_'+str(i),sequence=dom+'_seq_fixture',frame_id=i,domain=dom,scale=1.,split='test',gt=box[:5],**{a:list(box) for a in d.ARMS})
            r['v20'][2]=111.;r['d02'][2]=101.
            r.update(pre_guard={a:list(r[a]) for a in d.HEADS},delivery={a:'accepted' for a in d.HEADS},failed_checks={a:[] for a in d.HEADS})
            if i==2:
                for a in d.ARMS:r[a]=None
                r['pre_guard']={a:None for a in d.HEADS}
            records.append(r)
    report=d.detection_report(records,False)
    for arm in d.ARMS:assert report['metrics']['real'][arm]['components']['outputs']==2
    assert report['structure_transitions']['real']['v20->d02']['wrong_to_right']==2
    assert report['diagnostics']['real']['v20']['near_and_joint']['near']['long']['positive_harm_sum']==pytest.approx(.22)
    assert report['diagnostics']['real']['v20']['pre_guard_sizes']['joint10']==0
    with pytest.raises(ValueError):verify_detection(records,report)


def test_no_run_does_not_create_result_and_protocol_gpu3_only(tmp_path):
    with pytest.raises(ValueError):e.run(SimpleNamespace(run=False,out_dir=tmp_path/'unused'))
    assert not (tmp_path/'unused').exists()
    text=e.parser().format_help();assert '--epoch' not in text and '--structure-run' in text
    p=d.protocol();assert p['resources']['physical_gpu_choices']==[3] and p['limits']['selection'] is False
    assert not p['limits']['train03'] and not p['limits']['fixed_dev'] and 'p5' not in d.HEADS
