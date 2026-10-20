"""CPU contracts for frozen six-version reports; no dataset inference."""
from copy import deepcopy
import math
from types import SimpleNamespace
import pytest
import torch
from crane_project.utils import port_size_a20_protection_downstream_v1 as d
from crane_project.utils import port_size_qalign_downstream_v1 as previous
from crane_project.tools import eval_port_size_a20_protection_downstream_v1 as e
from crane_project.tools.analyze_port_size_a20_protection_downstream_v1 import verify_depth,verify_detection
from test_port_size_native_p3_v1 import fixture


def depth_fixture(count=2):
    box=[500.,500.,100.,50.,0.,.8]
    manifest=dict(sequence_id=d.SEQUENCE,split='unknown_test',camera=dict(intrinsics=dict(fx=1000.,fy=1000.)),
        obb_reference_geometry=dict(long_edge_mean_m=2.,short_edge_mean_m=1.))
    calibration=dict(parameters=dict(c=1.,beta=2.,b_m=0.,q_signed_min=-.01,q_signed_max=.01))
    predictions=[dict(frame_id='frame_%05d'%i,**{a:list(box) for a in d.ARMS}) for i in range(count)]
    truths=[dict(camera_geometry=dict(z_cg_opt_m=20.),pivot_relative=dict(theta_total_deg=1.),
        obb_geometry=dict(cx_px=500.,cy_px=500.,w_px=100.,h_px=50.,gamma_deg=0.)) for i in range(count)]
    return predictions,truths,manifest,calibration


def test_six_versions_frozen_formula_and_all_fifteen_pairs():
    data=depth_fixture();before=deepcopy(data);rows,s=d.depth_report(*data)
    assert data==before and len(s['paired'])==15
    assert s['all_versions_common_valid']['count']==2
    for a in d.ARMS:assert s['methods'][a]['depth_metrics']['mae_m']==0
    assert 'c6->r6' in s['paired'] and s['paired']['c6->r6']['tied']==2
    p,t,m,c=deepcopy(data)
    for x in p:x['q20']=list(x['a20'])
    _,old=previous.depth_report(p,t,m,c)
    for a in ['b','midpoint','a20']:assert old['methods'][a]==s['methods'][a]


def test_coverage_numeric_validity_no_q_filter_or_size_filter():
    p,t,m,c=depth_fixture();p[0]['r6'][2]=1e100;p[1]['p5'][2]=1e100
    _,s=d.depth_report(p,t,m,c)
    assert s['methods']['r6']['output_frame_count']==2 and s['methods']['r6']['numeric_depth_count']==1
    assert s['all_versions_common_valid']['count']==0 and s['paired']['c6->r6']['common']==1
    assert s['methods']['c6']['numeric_depth_count']==2


def test_pairing_directions_and_signed_propagation_not_mae():
    p,t,m,c=depth_fixture();p[0]['r6'][3]=49.;p[1]['r6'][3]=51.
    rows,s=d.depth_report(p,t,m,c);pair=s['paired']['c6->r6']
    assert pair['worsened']==2 and pair['frame_ids']['worsened']==['frame_00000','frame_00001']
    assert s['methods']['r6']['depth_metrics']['mae_m']>abs(s['methods']['r6']['depth_metrics']['bias_m'])
    verify_depth(rows,s,dict(camera=m['camera'],physical_reference=m['obb_reference_geometry']),c)
    rows[0]['r6']['depth']['z_m']+=.01
    with pytest.raises(ValueError):verify_depth(rows,s,dict(camera=m['camera'],physical_reference=m['obb_reference_geometry']),c)


def test_missing_order_identity_and_full_truth():
    p,t,m,c=depth_fixture();p[0]['p5']=None
    with pytest.raises(ValueError):d.depth_report(p,t,m,c)
    for arm in d.ARMS:p[0][arm]=None
    _,s=d.depth_report(p,t,m,c);assert s['methods']['p5']['numeric_depth_count']==1
    p,t,m,c=depth_fixture();p[0]['frame_id']='frame_00001'
    with pytest.raises(ValueError):d.depth_report(p,t,m,c)
    with pytest.raises(ValueError):d.depth_report(p,t[:1],m,c)


@pytest.mark.parametrize('swap,aniso',[(False,False),(True,True)])
def test_intermediate_coordinates_and_common_w_reused(swap,aniso):
    x=list(fixture(swap,aniso)[:6]);assert all(v==0 for v in d.replay_inputs(x,deepcopy(x)).values())
    changed=deepcopy(x);changed[1][0,0,4,4]=0
    with pytest.raises(ValueError):d.replay_inputs(x,changed)


def val_fixture():
    row=dict(image='x',delivery={a:'accepted' for a in d.HEADS},failed_checks={a:[] for a in d.HEADS},
        vector_valid={a:True for a in d.HEADS},**{a:[0.,0.,100.,50.,0.,.8] for a in d.ARMS})
    return [dict(deepcopy(row),image=str(i)) for i in range(887)]


@pytest.mark.parametrize('public',['a20','c6','r6','p5'])
def test_full_val_gate_frozen_six_scores_and_guard_decisions(public):
    rows=val_fixture();ref=deepcopy(rows);proof=d.compare_boxes(rows,ref)
    with pytest.raises(ValueError):d.require_val_proof(proof)
    proof['intermediate_frames']=887;d.require_val_proof(proof)
    rows[0]['delivery'][public]='fallback'
    with pytest.raises(ValueError):d.compare_boxes(rows,ref)
    rows=deepcopy(ref);rows[0][public][5]+=.00000001
    with pytest.raises(ValueError):d.compare_boxes(rows,ref)
    with pytest.raises(ValueError):d.compare_boxes(rows[:1],ref[:1])


@pytest.mark.parametrize('public',['c6','r6','p5'])
def test_only_original_continuation_epoch_and_frontend_allowed(public):
    spec=d.HEADS[public];p=dict(protocol='port_size_a20_protection_train_v1',arm=spec['arm'],epoch=spec['epoch'],
        continuation_updates=639*spec['epoch'],source_A20_sha256=d.PINS['a20'],
        input_identity=dict(frontend=dict(b_sha256=d.PINS['b'],m_sha256=d.PINS['midpoint'])))
    d.head_metadata(p,public)
    for k,v in [('epoch',2),('continuation_updates',0),('source_A20_sha256','bad'),('arm','other')]:
        bad=deepcopy(p);bad[k]=v
        with pytest.raises(ValueError):d.head_metadata(bad,public)


def test_six_stream_detection_denominators_qualification_and_diagnostics():
    records=[]
    for i in range(3):
        m=[0.,0.,89.,50.,0.,.8]
        row=dict(image='real_seq_fixture_'+str(i),frame_id=i,sequence='real_seq_fixture',domain='real',scale=1.,split='test',
            gt=[0.,0.,100.,50.,0.],b=list(m),midpoint=list(m),**{a:[0.,0.,95.,51.,0.,.8] for a in d.HEADS})
        row['p5']=[0.,0.,85.,52.,0.,.8]
        if i==2:
            for a in d.ARMS:row[a]=None
        records.append(row)
    records.append(dict(deepcopy(records[0]),image='sim_seq_fixture_0',frame_id=0,sequence='sim_seq_fixture',domain='sim'))
    report=d.detection_report(records,verify_native=False)
    for arm in d.ARMS:
        comp=report['metrics']['real'][arm]['components'];assert comp['outputs']==2 and comp['frames']==3 and comp['missing']==1 and comp['angle_eligible_outputs']==2
    assert report['diagnostics']['real']['p5']['cross_edge']['both_harmed']==2
    assert report['metrics']['real']['r6_paired_vs_M']['wrong_to_right']==2
    # The CPU verifier requires the native AP flag; a synthetic report cannot masquerade as a real run.
    with pytest.raises(ValueError):verify_detection(records,report)


def test_no_run_guard_and_no_epoch_selection_cli(tmp_path):
    with pytest.raises(ValueError):e.run(SimpleNamespace(run=False,out_dir=tmp_path/'unused'))
    assert not (tmp_path/'unused').exists()
    text=e.parser().format_help();assert '--epoch' not in text and '--protection-run' in text
    p=d.protocol();assert p['limits']['selection'] is False and p['limits']['formula_refit'] is False
    assert p['versions']==list(d.ARMS) and p['limits']['fixed_dev'] is False and p['limits']['train03'] is False
    assert len(d.HEADS)==4 and 'q20' not in d.ARMS


def test_latest_gpu1_request_does_not_change_legacy_gpu_choices(monkeypatch,tmp_path):
    class NoThread:
        def __init__(self,**kwargs):pass
        def start(self):pass
    monkeypatch.setattr(d,'smi',lambda fields,kind='gpu':[['1','GPU-second','4096'],['2','GPU-third','6000']])
    monkeypatch.setattr(d.threading,'Thread',NoThread)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    state,stop,thread=d.setup_gpu(tmp_path)
    assert state['physical_gpu']==1 and state['user_limit_mib']==7000
    assert d.protocol()['resources']['physical_gpu_choices']==[1]
    assert previous.protocol()['resources']['physical_gpu_choices']==[2,3]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2')
    with pytest.raises(ValueError):d.setup_gpu(tmp_path)


def test_memory_stop_and_insufficient_free_fail_closed(monkeypatch,tmp_path):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    monkeypatch.setattr(d,'smi',lambda fields,kind='gpu':[['1','GPU-second','3000']])
    with pytest.raises(RuntimeError):d.setup_gpu(tmp_path)
    e.check_memory(dict(process_mib=6799))
    with pytest.raises(RuntimeError):e.check_memory(dict(process_mib=6800))
