"""GT axis transport, paired loss/gradients and bounded TRAIN-only execution."""
from copy import deepcopy
from collections import Counter
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from crane_project.utils import port_geometry_midpoint_v1 as m
from crane_project.utils import port_geometry_midpoint_motion_v1 as loss
from crane_project.tools import preflight_port_geometry_midpoint_motion_v1 as t


def rectangles(edges,angles=None):
    angles=angles or [0.]*len(edges)
    return torch.tensor([[64.,48.,w,h,a] for (w,h),a in zip(edges,angles)])


def test_gt_change_target_not_motion_suppression_and_constant_bias_nullspace():
    target=m.box_midpoints(rectangles([(32.,16.),(40.,20.)]))
    predicted=target.clone().requires_grad_()
    exact,d=loss.motion_loss(predicted,dict(original=target))
    assert exact==0 and d['gt_log_increment'].abs().sum()>0
    frozen=target.clone();frozen[1]=frozen[0]
    wrong,_=loss.motion_loss(frozen,dict(original=target))
    assert wrong>0
    biased=64.+(target-64.)*1.2
    bias_loss,_=loss.motion_loss(biased,dict(original=target))
    assert bias_loss<1e-10
    # Independent translation/resize of each frame is removed from log P/G.
    scaled=target.clone();scaled[1]=64.+(scaled[1]-64.)*1.03
    value,_=loss.motion_loss(scaled,dict(original=target))
    changed,_=loss.motion_loss(scaled*torch.tensor([.7,1.3])[:,None,None]+30.,
        dict(original=target*torch.tensor([.7,1.3])[:,None,None]+30.))
    assert torch.allclose(value,changed,atol=1e-7,rtol=0.)


@pytest.mark.parametrize('roll',[0,1,2,3])
def test_independent_cyclic_axis_order_is_transported_for_prediction_and_gt(roll):
    targets=m.box_midpoints(rectangles([(32.,16.),(34.,18.)],[.2,.22]))
    predicted=m.box_midpoints(rectangles([(31.,17.),(36.,17.)],[.21,.23]))
    original,detail=loss.motion_loss(predicted,dict(original=targets))
    # Re-index both matched next frames, preserving CCW pair geometry.
    shifted_t=targets.clone();shifted_p=predicted.clone()
    shifted_t[1]=torch.roll(shifted_t[1],-roll,0)
    shifted_p[1]=torch.roll(shifted_p[1],-roll,0)
    other,d=loss.motion_loss(shifted_p,dict(original=shifted_t))
    assert bool(d['swap'][0])==bool(roll%2)
    assert torch.allclose(original,other,atol=1e-7,rtol=0.)
    assert torch.allclose(detail['log_increment_residual'],d['log_increment_residual'],atol=1e-6,rtol=0.)


@pytest.mark.parametrize('angle',[.2,math.pi/2-1e-5,-math.pi/2+1e-5])
def test_raw_width_exchange_periodic_wrap_and_square_edge_crossing(angle):
    gt=rectangles([(32.,31.),(31.,32.)],[angle,angle+.01])
    target=m.box_midpoints(gt)
    equivalent=gt.clone();equivalent[:,[2,3]]=gt[:,[3,2]];equivalent[:,4]+=math.pi/2
    alternate=m.box_midpoints(equivalent)
    predicted=target.clone();predicted[1,0,0]+=.1
    a,detail=loss.motion_loss(predicted,dict(original=target))
    b,_=loss.motion_loss(predicted,dict(original=alternate))
    assert torch.allclose(a,b,atol=1e-6,rtol=0.)
    assert detail['swap'].shape==(1,)
    assert float(detail['transported_angle_deg'][0])<1.
    exact,_=loss.motion_loss(target,dict(original=target))
    assert exact<1e-10


def test_gt_45_degree_tie_is_disclosed_not_dropped_or_prediction_selected():
    gt=m.box_midpoints(rectangles([(32.,16.),(33.,17.)],[0.,math.pi/4]))
    detail=loss.axis_correspondence(gt)
    assert bool(detail['ambiguous'][0]) and not bool(detail['swap'][0])
    predicted=gt.clone();predicted[1,0,0]+=2.
    value,changed=loss.motion_loss(predicted,dict(original=gt))
    assert value>0 and bool(changed['ambiguous'][0])
    assert torch.equal(detail['swap'],changed['swap'])


@pytest.mark.parametrize('bad',['odd','empty','nan','collapsed','negative'])
def test_invalid_pair_geometry_aborts_without_clamp_or_drop(bad):
    target=m.box_midpoints(rectangles([(32.,16.),(34.,18.)]))
    predicted=target.clone()
    if bad=='odd':target=target[:1];predicted=predicted[:1]
    elif bad=='empty':target=target[:0];predicted=predicted[:0]
    elif bad=='nan':predicted[0,0,0]=float('nan')
    elif bad=='collapsed':target[:]=0.
    else:predicted[:,[1,3]]=predicted[:,[3,1]]
    with pytest.raises(ValueError):loss.motion_loss(predicted,dict(original=target))


def model_pair():
    b=rectangles([(32.,16.),(34.,18.)],[.2,.22])
    b=torch.cat((b,torch.full((2,1),.8)),1)
    xy=torch.tensor([[.5,.5001],[.5,.5001]])
    bm=b[:,:5].clone();bm[:,:4]*=xy[:,[0,1,0,1]]
    gt=b[:,:5].clone();gt[0,2:4]*=torch.tensor([1.03,.95]);gt[1,2:4]*=torch.tensor([.96,1.06])
    return b,bm,xy,gt


def test_continuous_motion_gradients_two_frames_stem_and_detached_labels_inputs():
    torch.set_num_threads(1);torch.manual_seed(1703)
    b,bm,xy,gt=model_pair()
    for x in (b,bm,xy,gt):x.requires_grad_()
    roi=torch.randn(2,256,9,9,requires_grad=True);support=torch.ones(2,1,9,9,requires_grad=True)
    head=m.SpatialMidpointHead();optimizer=torch.optim.Adam(head.parameters(),lr=.001)
    for step in range(2):
        optimizer.zero_grad();out=head(roi,support,b,bm,xy);target=m.target_points(gt,b,bm,xy)
        value,_=loss.motion_loss(out['points_original'],target)
        if step==0:assert torch.equal(out['boxes_original'],b)
        point_gradient=torch.autograd.grad(value,out['points_original'],retain_graph=True)[0]
        assert all(float(x.norm())>0 for x in point_gradient)
        assert torch.allclose(point_gradient.sum(1),torch.zeros(2,2),atol=1e-7,rtol=0.)
        value.backward()
        assert head.output.weight.grad.norm()>0
        stem=sum(float(p.grad.square().sum()) for p in head.stem.parameters())
        assert stem==0 if step==0 else stem>0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        optimizer.step()
    assert all(v.grad is None for v in (b,bm,xy,gt,roi,support))


def metadata_records():
    return [dict(image=seq+'_'+str(i).zfill(5),sequence=seq,domain=seq.split('_')[0],
        frame_id=i,role='train',scale=scale,eligible=True)
        for scale in (1.,.5) for seq in ('real_seq01','real_seq05','real_seq06','real_seq12','real_seq13','sim_seq08')
        for i in range(220)]


def synthetic_views():
    metadata,_=t.base.select_blocks(metadata_records());generator=torch.Generator().manual_seed(92)
    result=[]
    for r in metadata:
        b=rectangles([(32.,16.)],[.2]);b=torch.cat((b,torch.tensor([[.8]])),1)
        xy=torch.tensor([[r['scale'],r['scale']+.0001]])
        bm=b[:,:5].clone();bm[:,:4]*=xy[:,[0,1,0,1]]
        gt=b[:,:5].clone();gt[:,:2]+=torch.tensor([.5,-.3])
        gt[:,2:4]*=torch.tensor([1.04+.01*(r['frame_id']%3),.96-.01*(r['frame_id']%3)])
        result.append(dict(r,role='train',boxes_original=b,boxes_model=bm,scale_xy=xy,gt_original=gt,
            roi=torch.randn(1,256,9,9,generator=generator)*.1,support=torch.ones(1,1,9,9)))
    return result


def test_pair_catalog_same_scale_sequence_role_adjacent_and_missing_pair_breaks():
    records,blocks=t.base.select_blocks(synthetic_views())
    victim=next(r for r in records if r['image']=='real_seq13_00000' and r['scale']==.5)
    victim['eligible']=False;victim['boxes_original']=victim['boxes_original'][:0]
    catalog=t.pair_catalog(records)
    assert len(catalog['pairs'])==224
    assert sum(p['eligible_for_gradient'] for p in catalog['pairs'])==167
    assert sum(p['role']=='probe' for p in catalog['pairs'])==56
    assert catalog['support']['fit/real/0.5']['missing_output_pairs']==1
    for p in catalog['pairs']:
        a,b=(records[i] for i in p['indices'])
        assert (a['sequence'],a['scale'],a['role'])==(b['sequence'],b['scale'],b['role'])
        assert b['frame_id']==a['frame_id']+1
    batches=t.schedule(catalog)
    assert len(batches)==200 and batches==t.schedule(catalog)
    assert all(Counter(catalog['pairs'][i]['domain'] for i in batch)=={'real':2,'sim':2} for batch in batches)
    assert all(catalog['pairs'][i]['eligible_for_gradient'] for batch in batches for i in batch)
    assert {i for batch in batches for i in batch}=={i for i,p in enumerate(catalog['pairs']) if p['eligible_for_gradient']}
    probe=next(i for i,p in enumerate(catalog['pairs']) if p['role']=='probe')
    with pytest.raises(ValueError):t.batch(records,catalog,[probe],torch.device('cpu'),torch)


@pytest.mark.parametrize('bad',['val','test','scale','duplicate'])
def test_pair_catalog_rejects_wrong_role_scale_or_duplicates(bad):
    records,_=t.base.select_blocks(synthetic_views())
    if bad in ('val','test'):records[0]['role']=bad
    elif bad=='scale':records[0]['scale']=2.
    else:records.append(deepcopy(records[0]))
    with pytest.raises(ValueError):t.pair_catalog(records)


def test_probe_missing_output_wrong_center_three_fractions_and_no_gap_bridge():
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as f
    torch.set_num_threads(1)
    records,_=t.base.select_blocks(synthetic_views())
    probe=[r for r in records if (r['role'],r['domain'],r['scale'])==('probe','real',1.)]
    for field in ('roi','support','boxes_original','boxes_model','scale_xy'):
        probe[3][field]=probe[3][field][:0]
    probe[3]['eligible']=False
    probe[5]['gt_original'][:,:2]+=30.;probe[5]['eligible']=False
    catalog=t.pair_catalog(records)
    result=t.base.evaluate(m.SpatialMidpointHead(),records,torch.device('cpu'),f)
    group=result['groups']['probe/real/1.0']
    assert group['output_coverage']==dict(numerator=15,denominator=16,pct=93.75)
    assert group['midpoint']['conditional_center_correct']['numerator']==14
    assert group['midpoint']['conditional_center_correct']['denominator']==15
    assert group['midpoint']['all_frame_center_correct']['numerator']==14
    assert group['midpoint']['all_frame_center_correct']['denominator']==16
    assert group['temporal_pairs']==12
    assert catalog['support']['probe/real/1.0']['missing_output_pairs']==2
    assert not any(p['eligible_for_gradient'] for p in catalog['pairs'] if p['role']=='probe')


def fake_contract(tmp_path):
    parent=t.read_json(t.base.PARENT)
    identity=dict(sources=parent['sources'],sources_sha256=t.sha(t.base.PARENT),
        protocol_sha256=t.sha(t.ROOT/'crane_project/tools/port_geometry_midpoint_formal_v1_protocol.json'),
        frozen_b=t.read_json(t.ROOT/'crane_project/tools/port_geometry_midpoint_formal_v1_protocol.json')['frozen_b'])
    train=tmp_path/'training';cache=tmp_path/'cache';train.mkdir();cache.mkdir()
    document=dict(status='COMPLETE_FROZEN_B_TRAIN_VAL_CACHE',identity=identity,
        record_counts=dict(train_s1=2558,train_s05=2558,val_s1=887),files={'train_s1.pt':'a','train_s05.pt':'b','val_s1.pt':'c'},
        feature_extractions=6003,native_head_calls=18009,detector_updates=0,test_access=False,
        runtime={'synthetic':True},detector_state={'synthetic':True})
    t.write_json(cache/'cache_manifest.json',document)
    complete=dict(identity=identity,status='FORMAL_MIDPOINT_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED',
        epochs_completed=24,detector_updates=0,test_access=False,cache_manifest_sha256=t.sha(cache/'cache_manifest.json'))
    t.write_json(train/'completion.json',complete)
    t.write_json(train/'artifacts.json',dict(files={'completion.json':t.sha(train/'completion.json')}))
    return train,cache,document


@pytest.mark.parametrize('relative',[True,False])
def test_real_python38_cli_static_no_tensor_gpu_and_existing_directory_preserved(tmp_path,relative):
    train,cache,_=fake_contract(tmp_path);script=Path(t.__file__).resolve()
    target=str(script.relative_to(t.ROOT)) if relative else str(script);out=tmp_path/'static'
    cmd=[sys.executable,'-B',target,'--check-only','--training-dir',str(train),'--cache-dir',str(cache),'--out-dir',str(out)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    proc=subprocess.run(cmd,cwd=str(t.ROOT),env=env,capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    report=t.read_json(out/'completion.json')
    assert report['status']=='STATIC_MOTION_CONTRACT_PASS_NO_TENSOR_LOAD_NO_GPU_NO_UPDATES'
    assert report['head_updates_total']==report['cache_tensor_loads']==0 and report['gpu_devices_used']==[]
    before={p.name:p.read_bytes() for p in out.iterdir()}
    again=subprocess.run(cmd,cwd=str(t.ROOT),env=env,capture_output=True,text=True)
    assert again.returncode!=0 and 'Existing results are preserved' in again.stderr
    assert before=={p.name:p.read_bytes() for p in out.iterdir()}


def test_static_entry_no_runtime_and_source_tampering_fails(tmp_path,monkeypatch):
    train,cache,_=fake_contract(tmp_path)
    def forbidden(*a,**k):raise AssertionError('Runtime loaded in static mode')
    monkeypatch.setattr(t,'runtime_modules',forbidden)
    args=t.parser().parse_args(['--check-only','--training-dir',str(train),'--cache-dir',str(cache),'--out-dir',str(tmp_path/'out')])
    report=t.run(args);assert report['cache_tensor_loads']==0
    bad=deepcopy(t.read_json(t.SOURCES));bad['sources'][str(Path(t.__file__).resolve().relative_to(t.ROOT))]='bad'
    path=tmp_path/'bad.json';t.write_json(path,bad);monkeypatch.setattr(t,'SOURCES',path)
    with pytest.raises(ValueError,match='Source SHA'):t.checked_contract(train,cache)


def test_paired_two_step_entry_axis_audit_formula_coverage_and_failure_preservation(tmp_path,monkeypatch):
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as f
    torch.set_num_threads(1)
    train,cache,document=fake_contract(tmp_path);views=synthetic_views()
    monkeypatch.setitem(t.SETTINGS,'steps_per_arm',2)
    monkeypatch.setattr(t,'checked_contract',lambda *a:(t.protocol_document(),document,{'synthetic':True}))
    monkeypatch.setattr(t.base,'execution_device',lambda *a:(torch.device('cpu'),{'synthetic':True}))
    monkeypatch.setattr(t.base,'load_train_cache',lambda *a:views)
    args=t.parser().parse_args(['--training-dir',str(train),'--cache-dir',str(cache),'--out-dir',str(tmp_path/'run')])
    report=t.run(args)
    assert report['head_updates_total']==4 and report['eligible_fit_pair_count']==168 and report['pair_count']==224
    assert not report['formal_training_approved'] and not report['val_access'] and not report['test_access']
    assert not list((tmp_path/'run').glob('*.pth'))
    assert len(t.read_json(tmp_path/'run/axis_correspondence.json'))==224
    arms=[t.read_json(tmp_path/'run'/(a+'.json'))['result'] for a in t.ARMS]
    assert arms[0]['initial']==arms[1]['initial'] and arms[0]['initial_state']==arms[1]['initial_state']
    assert arms[0]['final_state']!=arms[1]['final_state']
    for arm,result in zip(t.ARMS,arms):
        for log in result['logs']:
            expected=log['point_loss']+(log['motion_weighted_loss'] if arm=='point_motion' else 0)
            assert log['total_loss']==pytest.approx(expected,abs=1e-8)
            assert log['component_gradients']['motion_unit_norm']>0
        group=result['final']['groups']['probe/real/1.0']
        assert group['output_coverage']['denominator']==16
        assert group['midpoint']['conditional_center_correct']['denominator']==16
        assert group['midpoint']['all_frame_center_correct']['denominator']==16
        assert group['temporal_pairs']==14
    verdict=t.review(arms[0]['final'],arms[0]['final'])
    assert not verdict['diagnostic_conditions_met']
    altered=deepcopy(arms[0]['final'])
    for key,g in altered['groups'].items():
        if key.startswith('probe/'):
            for field in ('short','diagonal'):
                g['midpoint']['temporal'][field]['log_increment_error']['rmse']*=.9
    assert t.review(arms[0]['final'],altered)['checks']['real/strict_short_and_diagonal_motion_gain']
    changed=deepcopy(altered);changed['groups']['probe/real/1.0']['midpoint']['static']['short']['relative_error']['mae']+=.01
    assert not t.review(arms[0]['final'],changed)['checks']['probe/real/1.0']['short/static_mae']
    changed=deepcopy(arms[0]['final']);changed['groups']['probe/sim/1.0']['midpoint']['dfr_pct_per_frame']=0.
    assert not t.review(arms[0]['final'],changed)['diagnostic_conditions_met']
    original=t.fit_arm
    def fail_second(*args):
        if args[4]=='point_motion':raise ValueError('synthetic second arm failure')
        return original(*args)
    monkeypatch.setattr(t,'fit_arm',fail_second);args.out_dir=str(tmp_path/'failed')
    with pytest.raises(ValueError,match='second arm'):t.run(args)
    failed=t.read_json(tmp_path/'failed/completion.json')
    assert failed['head_updates_total']==2 and failed['status']=='FAILED_MOTION_PREFLIGHT_REVIEW_REQUIRED'
    assert (tmp_path/'failed/point_only.json').exists() and (tmp_path/'failed/artifacts.json').exists()


def test_nonfinite_after_executed_update_preserves_exact_failure_budget(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    train,cache,document=fake_contract(tmp_path);views=synthetic_views()
    monkeypatch.setitem(t.SETTINGS,'steps_per_arm',2)
    monkeypatch.setattr(t,'checked_contract',lambda *a:(t.protocol_document(),document,{'synthetic':True}))
    monkeypatch.setattr(t.base,'execution_device',lambda *a:(torch.device('cpu'),{'synthetic':True}))
    monkeypatch.setattr(t.base,'load_train_cache',lambda *a:views)
    original=torch.optim.Adam.step
    def broken_update(optimizer,*args,**kwargs):
        original(optimizer,*args,**kwargs)
        with torch.no_grad():optimizer.param_groups[0]['params'][0].fill_(float('nan'))
    monkeypatch.setattr(torch.optim.Adam,'step',broken_update)
    args=t.parser().parse_args(['--training-dir',str(train),'--cache-dir',str(cache),'--out-dir',str(tmp_path/'failed')])
    with pytest.raises(ValueError,match='Nonfinite updated'):t.run(args)
    failed=t.read_json(tmp_path/'failed/completion.json')
    assert failed['head_updates_total']==1 and failed['status']=='FAILED_MOTION_PREFLIGHT_REVIEW_REQUIRED'
    assert (tmp_path/'failed/artifacts.json').exists()


def test_valid_zero_motion_gradient_is_reported_without_numerical_failure(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    train,cache,document=fake_contract(tmp_path);views=synthetic_views()
    monkeypatch.setitem(t.SETTINGS,'steps_per_arm',2)
    monkeypatch.setattr(t,'checked_contract',lambda *a:(t.protocol_document(),document,{'synthetic':True}))
    monkeypatch.setattr(t.base,'execution_device',lambda *a:(torch.device('cpu'),{'synthetic':True}))
    monkeypatch.setattr(t.base,'load_train_cache',lambda *a:views)
    original=loss.motion_loss
    def zero_motion(*args):
        value,detail=original(*args)
        return value*0.,dict(detail,parts=detail['parts']*0.)
    monkeypatch.setattr(loss,'motion_loss',zero_motion)
    args=t.parser().parse_args(['--training-dir',str(train),'--cache-dir',str(cache),'--out-dir',str(tmp_path/'zero')])
    report=t.run(args)
    assert report['head_updates_total']==4 and report['status']=='TRAIN_MOTION_SHORT_FIT_COMPLETE_REVIEW_REQUIRED'
    assert not report['review']['checks']['measured_motion_gradient_signal_both_arms']
    assert not report['review']['diagnostic_conditions_met'] and not report['formal_training_approved']
