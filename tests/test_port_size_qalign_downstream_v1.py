"""Synthetic CPU checks only; no image/cache/dataset inference or server access."""
from copy import deepcopy
import math
from types import SimpleNamespace
import pytest
import torch
from crane_project.utils import port_size_qalign_downstream_v1 as d
from crane_project.tools import eval_port_size_qalign_downstream_v1 as e
from crane_project.tools.analyze_port_size_qalign_downstream_v1 import assert_numbers,verify_depth,detection_row
from test_port_size_native_p3_v1 import fixture


def depth_fixture(count=1):
    box=[500.,500.,100.,50.,0.,.8]
    manifest=dict(sequence_id=d.SEQUENCE,split='unknown_test',camera=dict(intrinsics=dict(fx=1000.,fy=1000.)),
        obb_reference_geometry=dict(long_edge_mean_m=2.,short_edge_mean_m=1.))
    calibration=dict(parameters=dict(c=1.,beta=2.,b_m=0.,q_signed_min=-.01,q_signed_max=.01))
    predictions=[dict(frame_id='frame_%05d'%i,**{a:list(box) for a in d.ARMS}) for i in range(count)]
    truths=[dict(camera_geometry=dict(z_cg_opt_m=20.),pivot_relative=dict(theta_total_deg=1.),
        obb_geometry=dict(cx_px=500.,cy_px=500.,w_px=100.,h_px=50.,gamma_deg=0.)) for i in range(count)]
    return predictions,truths,manifest,calibration


def test_depth_frozen_formula_zero_error_and_no_mutation():
    data=depth_fixture();before=deepcopy(data);rows,s=d.depth_report(*data)
    assert data==before
    for a in d.ARMS:
        assert s['methods'][a]['depth_metrics']['mae_m']==0 and s['methods'][a]['numeric_depth_count']==1
        assert s['size_geometry'][a]['a']['mae']==0


def test_no_q_filter_numeric_coverage_and_common_pairing():
    p,t,m,c=depth_fixture(2)
    p[0]['a20'][2]=1e100;p[1]['q20'][2]=1e100
    rows,s=d.depth_report(p,t,m,c)
    assert s['methods']['a20']['output_frame_count']==2
    assert s['methods']['a20']['numeric_depth_count']==1 and s['methods']['a20']['numeric_depth_coverage']==.5
    assert s['all_four_common_valid']['count']==0
    assert s['paired']['midpoint->a20']['common']==1 and s['paired']['midpoint->a20']['lost_numeric']==1
    p,t,m,c=depth_fixture();p[0]['a20'][2]=70.
    _,s=d.depth_report(p,t,m,c)
    assert s['methods']['a20']['q_out_of_fit_support_count']==1 and s['methods']['a20']['numeric_depth_count']==1


def test_signed_propagation_not_additive_mae_and_paired_frame_ids():
    p,t,m,c=depth_fixture(2);p[0]['a20'][3]=45.;p[1]['a20'][3]=55.
    rows,s=d.depth_report(p,t,m,c);value=s['propagation']['a20']
    assert value['max_signed_identity_residual_m']<1e-12
    assert sum(value['signed_error_component_mean_m'].values())==pytest.approx(s['methods']['a20']['depth_metrics']['bias_m'])
    assert s['methods']['a20']['depth_metrics']['mae_m']>abs(s['methods']['a20']['depth_metrics']['bias_m'])
    assert s['paired']['midpoint->a20']['worsened']==2
    assert s['paired']['midpoint->a20']['frame_ids']['worsened']==['frame_00000','frame_00001']


def test_missing_identity_order_and_truth_lengths():
    p,t,m,c=depth_fixture();p[0]['a20']=None
    with pytest.raises(ValueError):d.depth_report(p,t,m,c)
    for a in d.ARMS:p[0][a]=None
    _,s=d.depth_report(p,t,m,c);assert s['methods']['a20']['numeric_depth_count']==0
    p,t,m,c=depth_fixture();p[0]['frame_id']='frame_00001'
    with pytest.raises(ValueError):d.depth_report(p,t,m,c)
    with pytest.raises(ValueError):d.depth_report([],t,m,c)


@pytest.mark.parametrize('swap,aniso',[(False,False),(True,True)])
def test_native_intermediate_replay_and_changed_support_rejected(swap,aniso):
    x=list(fixture(swap,aniso)[:6]);r=d.replay_inputs(x,deepcopy(x));assert all(v==0 for v in r.values())
    changed=deepcopy(x);changed[1][0,0,4,4]=0
    with pytest.raises(ValueError):d.replay_inputs(x,changed)
    changed=deepcopy(x);changed[4][0,0]*=.99
    with pytest.raises(ValueError):d.replay_inputs(x,changed)


def test_full_val_gate_and_delivery_checks_cannot_be_bypassed():
    row=dict(image='x',delivery=dict(a20='accepted',q20='accepted'),failed_checks=dict(a20=[],q20=[]),vector_valid=dict(a20=True,q20=True),
        **{a:[0.,0.,100.,50.,0.,.8] for a in d.ARMS})
    rows=[dict(deepcopy(row),image=str(i)) for i in range(887)];ref=deepcopy(rows)
    p=d.compare_boxes(rows,ref)
    with pytest.raises(ValueError):d.require_val_proof(p)
    p['intermediate_frames']=887;d.require_val_proof(p)
    rows[0]['delivery']['a20']='fallback'
    with pytest.raises(ValueError):d.compare_boxes(rows,ref)
    with pytest.raises(ValueError):d.compare_boxes(rows[:1],ref[:1])


def test_ep20_only_and_frontend_identity():
    p=dict(protocol='port_size_qalign_v1',arm='fixed_b',epoch=20,updates=12780,input_identity=dict(b_sha256=d.PINS['b'],m_sha256=d.PINS['midpoint']))
    d.head_metadata(p,'a20');p['epoch']=2
    with pytest.raises(ValueError):d.head_metadata(p,'a20')
    assert d.PINS['a20'].startswith('545e1845') and d.PINS['q20'].startswith('dac187b4')


def test_neutral_square_delivery_precedence_matches_s039():
    out=dict(boxes_original=torch.ones(1,6),neutral_bypass=[True],square_bypass=[True],accepted=[False])
    assert d.delivery_label(out)=='neutral'
    out['neutral_bypass']=[False];assert d.delivery_label(out)=='square'
    out['boxes_original']=torch.empty(0,6);assert d.delivery_label(out)=='no_output'


def test_code_only_guard_does_not_create_result_directory(tmp_path):
    with pytest.raises(ValueError):e.run(SimpleNamespace(run=False,out_dir=tmp_path/'unused'))
    assert not (tmp_path/'unused').exists()
    help=e.parser().format_help()
    assert '--run' in help and '--qalign-run' in help and '--epoch' not in help
    p=d.protocol();assert p['limits']['fixed_dev'] is False and p['limits']['selection'] is False


def test_independent_numeric_check_does_not_relax_size_tolerance():
    assert_numbers(dict(mean_riou_full_frame=.8),dict(mean_riou_full_frame=.80000001))
    with pytest.raises(ValueError):assert_numbers(dict(mae=.08),dict(mae=.08000001))


def test_detection_geometry_missing_and_other_edge_harm():
    rows=[]
    for i in range(3):
        gt=[0.,0.,100.,50.,0.];m=[0.,0.,90.,50.,0.,.8]
        rows.append(dict(image='real_seq_fixture_'+str(i),frame_id=i,sequence='real_seq_fixture',domain='real',scale=1.,split='test',gt=gt,b=list(m),midpoint=list(m),a20=[0.,0.,95.,51.,0.,.8],q20=[0.,0.,85.,52.,0.,.8]))
    for a in d.ARMS:rows[2][a]=None
    rows.append(dict(deepcopy(rows[0]),image='sim_seq_fixture_0',frame_id=0,sequence='sim_seq_fixture',domain='sim'))
    report=d.detection_report(rows,verify_native=False)
    comp=report['metrics']['real']['a20']['components']
    assert comp['frames']==3 and comp['outputs']==2 and comp['missing']==1
    assert comp['center_correct']==2 and comp['size_correct']==2 and comp['angle_eligible_outputs']==2
    a=report['diagnostics']['real']['a20']['cross_edge'];q=report['diagnostics']['real']['q20']['cross_edge']
    assert a['edges']['long']['corrected_and_other_harmed']==2 and q['both_harmed']==2


def test_saved_depth_independent_formula_and_metric_replay():
    p,t,m,c=depth_fixture(2);p[0]['a20'][3]=49.;p[1]['q20'][3]=52.
    rows,s=d.depth_report(p,t,m,c)
    identity=dict(camera=m['camera'],physical_reference=m['obb_reference_geometry'])
    verify_depth(rows,s,identity,c)
    bad=deepcopy(rows);bad[0]['a20']['depth']['z_m']+=.01
    with pytest.raises(ValueError):verify_depth(bad,s,identity,c)


def test_zero_output_export_does_not_hide_missing_frames():
    cell=dict(components=dict(frames=3,outputs=0,missing=3,center_correct=0,size_correct=0,angle_correct=0,angle_eligible_outputs=0,angle_eligible_frames=3),
        sizes=dict(edges={e:dict(mae=None,p95=None,bias=None) for e in ('long','short')},**{e:dict(abs_mae=None,bias=None,abs_p95=None) for e in ('a','r')}),
        mean_riou_full_frame=0.,native_ap11={str(t):dict(ap_percent=0.) for t in (.5,.75)},temporal={})
    row=detection_row('real','a20',cell,'test')
    assert row['missing']==3 and row['coverage']==0 and row['size_rate'] is None and row['size_full']==0 and row['long_mae_pct'] is None
