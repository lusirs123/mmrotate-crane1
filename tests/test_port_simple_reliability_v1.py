"""Meaningful CPU checks for three flags, fitting separation, IO and B invariance."""
from argparse import Namespace
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.tools import run_port_simple_reliability_v1 as entry


def protocol():
    return json.loads(entry.PROTOCOL.read_text())


def rows(count=16):
    result = []
    for i in range(count):
        gt = [100., 80., 40., 20., .1]
        pred = gt+[.15+.7*i/max(1, count-1)]
        if i % 2:
            pred[2] *= .75; pred[4] += math.radians(6)
        seq = 'real_seq01' if i < count//2 else 'sim_seq08'
        result.append(dict(image=seq+'_%05d' % i, sequence=seq, domain=seq.split('_')[0],
            split='train' if seq.startswith('real') else 'train_sim', frame_id=i,
            image_size=[200, 160], gt=gt, pred=pred, train_angle_eligible=True,
            angle_axis_well_defined=True))
    return result


def val_rows():
    values = rows(8)
    for i,r in enumerate(values):
        r.update(image='real_seq07_%05d' % i, sequence='real_seq07', domain='real', split='val')
    values[-1]['pred'] = None
    return values


def policy():
    return simple.create_policy(rows(), val_rows(), protocol())


def test_online_signature_has_no_GT_history_domain_or_feature_tensor():
    assert list(inspect.signature(simple.SimpleComponentReliability.decide).parameters) == [
        'self', 'pred_original', 'image_size', 'method']
    runtime = simple.SimpleComponentReliability(policy())
    p = [100., 80., 40., 20., .1, .7]
    before = deepcopy(p)
    out = runtime.decide(p, [200, 160])
    assert p == before and out['raw_b_output'] == p
    assert all(type(out[c+'_accepted']) is bool for c in simple.COMPONENTS)


def test_size_angle_rejection_never_deletes_center_or_changes_raw_w_h_angle_score():
    q = policy()
    q['cutoffs']['simple']['size']['risk_le'] = 0.
    q['cutoffs']['simple']['angle']['risk_le'] = 0.
    p = [101., 82., 20., 40., -1., .8]
    out = simple.SimpleComponentReliability(q).decide(p, [200, 160])
    assert out['center_accepted'] and not out['size_accepted'] and not out['angle_accepted']
    assert out['raw_b_output'] == p


def test_component_flags_are_independent_and_no_output_is_three_false():
    q = policy(); q['cutoffs']['simple']['size']['risk_le'] = 0.
    q['cutoffs']['simple']['angle']['risk_le'] = 1.
    runtime = simple.SimpleComponentReliability(q)
    out = runtime.decide(rows()[0]['pred'], [200, 160])
    assert [out[c+'_accepted'] for c in simple.COMPONENTS] == [True, False, True]
    missing = runtime.decide(None, [200, 160])
    assert missing['raw_b_output'] is None
    assert not any(missing[c+'_accepted'] for c in simple.COMPONENTS)


@pytest.mark.parametrize('bad', [[0,0,-1,2,0,.8], [0,0,1,2,0,.05], [0,0,1,2,0,1.1],
                                [0,0,1,2,np.nan,.8], [[0,0,1,2,0,.8]]])
def test_invalid_output_not_silently_converted_to_a_missing_or_good_frame(bad):
    with pytest.raises(ValueError):
        simple.prediction(bad)


def test_features_and_risk_invariant_to_isotropic_units_and_equivalent_width_swap():
    p = np.array([100., 80., 40., 20., .1, .7])
    small = p.copy(); small[:4] *= .5
    swap = p.copy(); swap[2],swap[3] = p[3],p[2]; swap[4] += math.pi/2
    x = simple.descriptor(p, [200,160])
    assert np.allclose(x, simple.descriptor(small, [100,80]), atol=1e-15)
    assert np.allclose(x, simple.descriptor(swap, [200,160]), atol=1e-15)
    q = policy()
    assert float(simple.linear_risk(q['models']['angle'], x)) == pytest.approx(
        float(simple.linear_risk(q['models']['angle'], simple.descriptor(swap, [200,160]))))
    assert simple.geometry_errors(p[:5], swap)['angle_deg'] < 1e-12


def test_loss_gradient_and_hessian_match_independent_finite_differences():
    rng = np.random.RandomState(17)
    x = np.column_stack((np.ones(8), rng.randn(8,3)))
    y = np.array([0,1]*4); w = rng.randn(4)*.2; sw = np.ones(8)/8
    loss,g,h = simple.logistic_objective(w,x,y,sw,.1)
    numeric_g=[]; numeric_h=[]
    for i in range(4):
        d=np.zeros(4);d[i]=1e-6
        a=simple.logistic_objective(w+d,x,y,sw,.1)
        b=simple.logistic_objective(w-d,x,y,sw,.1)
        numeric_g.append((a[0]-b[0])/2e-6)
        numeric_h.append((a[1]-b[1])/2e-6)
    assert np.allclose(g,numeric_g,atol=1e-9)
    assert np.allclose(h,np.array(numeric_h).T,atol=1e-9)
    assert math.isfinite(loss)


def test_fit_converges_with_genuine_two_class_support_and_rejects_single_class():
    x = np.array([simple.descriptor(r['pred'],r['image_size']) for r in rows()])
    fit = simple.fit_linear_risk(x, np.arange(len(x))%2, protocol()['fitting'])
    assert fit['converged'] and fit['final_objective'] < fit['initial_objective']
    assert fit['max_abs_gradient'] <= 1e-8
    with pytest.raises(ValueError, match='Both genuine'):
        simple.fit_linear_risk(x,np.zeros(len(x)),protocol()['fitting'])


def test_VAL_GT_does_not_affect_models_normalization_or_operating_cutoffs():
    train=rows(); val=val_rows()
    a=simple.create_policy(train,val,protocol())
    changed=deepcopy(val)
    for r in changed:
        r['gt']=[0,0,1,1,1.5];r['angle_axis_well_defined']=False
    b=simple.create_policy(train,changed,protocol())
    assert a==b
    assert a['parameter_count']==8


def test_threshold_keeps_whole_ties_missing_is_not_accepted():
    c=simple.coverage_cutoff([.1,.2,.2,.2,.9],6,.5)
    assert c['capped_count']==3 and c['actual_accepted']==4
    c=simple.coverage_cutoff([.1,.2],10,.95)
    assert c['actual_accepted']==2 and not c['target_reachable']


def test_center_denominators_angle_unassessed_and_joint_OBB_are_separate():
    v=val_rows()[:3];v[1]['pred'][0]+=15;v[1]['angle_axis_well_defined']=False;v[2]['pred']=None
    records,s=simple.evaluate(v,policy());s=s['all']
    assert s['output_coverage']==pytest.approx(2/3)
    assert s['center_hit_rate_on_outputs']==.5
    assert s['all_frame_center_correct_coverage']==pytest.approx(1/3)
    assert s['components']['center']['simple']['accepted_frames']==2
    assert s['components']['angle']['raw']['unassessed_accepted']==1
    assert s['components']['angle']['raw']['eligible_frames']==2
    assert s['complete_obb']['raw']['joint_assessed']==1
    assert s['complete_obb']['raw']['joint_unassessed']==1
    assert not records[2]['methods']['simple']['center_accepted']


def test_sequence_and_number_gaps_reset_unavailable_run():
    v=rows(4)
    for r in v:r['pred']=None
    v[0].update(sequence='real_seq01',frame_id=0)
    v[1].update(sequence='real_seq01',frame_id=10)
    v[2].update(sequence='real_seq02',frame_id=11)
    v[3].update(sequence='real_seq02',frame_id=12)
    _,s=simple.evaluate(v,policy())
    assert s['all']['components']['center']['simple']['longest_flag_unavailable_all_frames']==2
    assert s['all']['center_hit_rate_on_outputs'] is None


def test_atomic_json_no_overwrite_and_nonfinite_payload_leaves_no_artifact(tmp_path):
    p=tmp_path/'result.json';entry.write_new(p,{'ok':True})
    with pytest.raises(FileExistsError):entry.write_new(p,{'ok':False})
    assert json.loads(p.read_text())=={'ok':True}
    with pytest.raises(ValueError):entry.write_new(tmp_path/'bad.json',{'bad':float('nan')})
    assert not (tmp_path/'bad.json').exists()
    assert sorted(p.name for p in tmp_path.iterdir())==['result.json']


def test_complete_CPU_fit_save_reload_and_policy_integrity(tmp_path):
    p=protocol();sources={'synthetic':True};proof={'artifact_sha256':p['reviewed_artifacts']}
    args=Namespace(out_dir=tmp_path)
    entry.fit(args,rows(),val_rows(),proof,p,sources)
    q=entry.load_policy(tmp_path/'policy.json',sources,p)
    assert q['models']['angle']['bad_rows']==8
    complete=json.loads((tmp_path/'completion.json').read_text())
    assert complete['save_reload_decisions_exact'] and complete['cached_b_inputs_unchanged']
    (tmp_path/'val_decisions.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='changed'):
        entry.load_policy(tmp_path/'policy.json',sources,p)


def test_wrong_calibration_completion_or_current_sources_rejected(tmp_path):
    p=protocol();sources={'synthetic':True};proof={'artifact_sha256':p['reviewed_artifacts']}
    entry.fit(Namespace(out_dir=tmp_path),rows(),val_rows(),proof,p,sources)
    with pytest.raises(ValueError,match='completed'):
        entry.load_policy(tmp_path/'policy.json',{'other_source':True},p)
    entry.write_new(tmp_path/'failure.json',{'status':'FAILED'})
    with pytest.raises(ValueError,match='completed'):
        entry.load_policy(tmp_path/'policy.json',sources,p)


def test_keep_ratio_meta_accepts_resize_rounding_and_rejects_stretch_or_flip():
    meta=dict(scale_factor=[.5,.5,.5,.5],ori_shape=(200,400,3),img_shape=(100,200,3),pad_shape=(1024,1024,3),flip=False)
    entry.checked_meta(meta)
    for changes in ({'scale_factor':[.5,.4,.5,.4]}, {'flip':True}, {'img_shape':(1100,200,3)}):
        with pytest.raises(ValueError):entry.checked_meta(dict(meta,**changes))


def test_current_source_contract_and_exact_reviewed_cache_check_is_CPU_only(monkeypatch):
    p,sources=entry.checked_sources()
    args=Namespace(train_cache=Path('work_dirs/port_reliability_train_support_v1_server_20261002'),
        input_snapshot=Path('work_dirs/port_reliability_train_support_v1_server_20261002/train_input_snapshot.json'),
        val_dir=Path('work_dirs/port_reliability_branches_v1_server_review_20261003'))
    # The real reviewed compact files are used only for integrity, never fitting.
    if not args.input_snapshot.exists():pytest.skip('Returned numeric evidence unavailable on this host')
    train,val,proof=entry.checked_inputs(args,p)
    assert len(train)==2558 and len(val)==887 and proof['train_angle_eligible']==2558
    assert sum(r['pred'] is not None for r in val)==886
    assert not proof['old_quality_scores_used'] and proof['detector_inferences']==0
    changed=deepcopy(p);changed['reviewed_artifacts']['train_predictions']='0'*64
    with pytest.raises(ValueError,match='reviewed genuine'):
        entry.checked_inputs(args,changed)


def test_online_infer_emits_flags_keeps_B_state_and_never_consumes_GT(tmp_path,monkeypatch):
    import torch
    from torch import nn
    detector=nn.Linear(1,1).eval().requires_grad_(False)
    monkeypatch.setattr(entry,'build_online',lambda *a:(detector,object()))
    original=[100.,80.,40.,20.,.1,.7]
    monkeypatch.setattr(entry,'infer_one',lambda *a:(original.copy(),[200,160]))
    for name in ('reset_peak_memory_stats','synchronize'):monkeypatch.setattr(torch.cuda,name,lambda *a:None)
    for name in ('max_memory_allocated','max_memory_reserved'):monkeypatch.setattr(torch.cuda,name,lambda *a:0)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *a:'CPU fixture')
    image_dir=tmp_path/'images';image_dir.mkdir();(image_dir/'current.jpg').write_bytes(b'fixture')
    output=tmp_path/'out';output.mkdir();q=tmp_path/'policy.json';q.write_text(json.dumps(policy()))
    args=Namespace(mode='infer',out_dir=output,image_dir=image_dir,b_checkpoint=Path('unused'),gpu=0,policy=q)
    before=entry.state_digest(detector)
    entry.online(args,policy(),protocol())
    row=json.loads((output/'decisions.jsonl').read_text())
    assert row['raw_b_output']==original and 'gt' not in row and 'domain' not in row
    assert entry.state_digest(detector)==before
    assert json.loads((output/'completion.json').read_text())['frames']==1


def test_six_view_smoke_reports_cache_and_flag_differences_without_substitution(tmp_path,monkeypatch):
    import torch
    from torch import nn
    detector=nn.Linear(1,1).eval().requires_grad_(False)
    monkeypatch.setattr(entry,'build_online',lambda *a:(detector,object()))
    original=[100.,80.,40.,20.,.1,.7]
    monkeypatch.setattr(entry,'infer_one',lambda *a:(original.copy(),[200,160]))
    for name in ('reset_peak_memory_stats','synchronize'):monkeypatch.setattr(torch.cuda,name,lambda *a:None)
    for name in ('max_memory_allocated','max_memory_reserved'):monkeypatch.setattr(torch.cuda,name,lambda *a:0)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *a:'CPU fixture')
    monkeypatch.setattr(entry,'DATA',tmp_path/'data')
    folder=entry.DATA/'val/images';folder.mkdir(parents=True)
    values=[]
    for seq in protocol()['counts']['val']:
        for i in (0,10):
            p=folder/(seq+'_%05d.jpg'%i);p.write_bytes(b'fixture')
            values.append(dict(image=p.stem,sequence=seq,frame_id=i,image_size=[200,160],
                image_sha256=entry.sha(p),pred=None))
    q=tmp_path/'policy.json';q.write_text(json.dumps(policy()))
    out=tmp_path/'out';out.mkdir()
    args=Namespace(mode='smoke',out_dir=out,b_checkpoint=Path('unused'),gpu=0,policy=q)
    entry.online(args,policy(),protocol(),values)
    result=json.loads((out/'completion.json').read_text())
    assert result['frames']==6 and result['historical_cache_parity_mismatches']==6
    assert result['historical_cache_flag_differences']==6
    saved=[json.loads(line) for line in (out/'decisions.jsonl').read_text().splitlines()]
    assert all(r['raw_b_output']==original for r in saved)
