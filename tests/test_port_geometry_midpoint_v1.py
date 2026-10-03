"""Geometry/gradient/isolation/entry regressions, using synthetic CPU data only."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest
import torch

from crane_project.tools import preflight_port_geometry_midpoint_v1 as t
m=t.m


def inputs(swap=False,angle=.2,scales=(1.,1.)):
    b=torch.tensor([[64.,48.,32.,16.,angle,.8]])
    if swap:
        b[:,[2,3]]=b[:,[3,2]]
    xy=b.new_tensor([scales]); bm=b[:,:5].clone(); bm[:,:4]*=xy[:,[0,1,0,1]]
    return b,bm,xy


@pytest.mark.parametrize('swap',[False,True])
@pytest.mark.parametrize('angle',[.2,math.pi/2-1e-6,-math.pi/2+1e-6])
def test_zero_initial_points_box_score_bitwise_identity_and_empty(swap,angle):
    b,bm,xy=inputs(swap,angle)
    head=m.SpatialMidpointHead()
    out=head(torch.randn(1,256,9,9),torch.ones(1,1,9,9),b,bm,xy)
    assert torch.equal(out['boxes_original'],b)
    assert torch.equal(out['points_original'],m.box_midpoints(b[:,:5]))
    assert torch.equal(out['point_delta_roi'],torch.zeros(1,4,2)) and out['accepted'].all()
    assert torch.allclose(out['probabilities'].sum(dim=(2,3)),torch.ones(1,4),atol=1e-6,rtol=0.)
    assert sum(p.numel() for p in head.parameters())==m.PARAMETER_COUNT
    empty=head(torch.empty(0,256,9,9),torch.empty(0,1,9,9),b[:0],bm[:0],xy[:0])
    assert empty['boxes_original'].shape==(0,6) and empty['points_roi'].shape==(0,4,2)


@pytest.mark.parametrize('square',[False,True])
def test_gt_width_exchange_period_and_square_matching_reconstruct_same_geometry(square):
    b,bm,xy=inputs()
    gt=b[:,:5].clone(); gt[:,:2]+=.4; gt[:,2:4]*=1.03; gt[:,4]+=.04
    if square:
        gt[:,2:4]=24.
    baseline=m.target_points(gt,b,bm,xy)
    for swap,period in ((True,0),(False,math.pi),(True,-math.pi)):
        alt=gt.clone(); alt[:,4]+=period
        if swap:
            alt[:,[2,3]]=gt[:,[3,2]]; alt[:,4]+=math.pi/2
        other=m.target_points(alt,b,bm,xy)
        assert torch.allclose(other['original'],baseline['original'],atol=1e-5,rtol=0.)
        assert torch.allclose(other['roi'],baseline['roi'],atol=1e-6,rtol=0.)


def test_zero_maps_are_centered_on_each_b_midpoint_not_roi_center_or_gt():
    b,bm,xy=inputs(); head=m.SpatialMidpointHead()
    out=head(torch.zeros(1,256,9,9),torch.ones(1,1,9,9),b,bm,xy)
    index=out['probabilities'].reshape(4,81).argmax(-1)
    coords=torch.stack((index%9,index//9),-1).float()/8-.5
    assert (coords-out['anchor_roi'][0]).abs().max()<=1/16+1e-6
    assert len(set(index.tolist()))==4
    assert torch.equal(out['points_roi'],out['anchor_roi'])


@pytest.mark.parametrize('scale',[1.,.5])
def test_point_mapping_odd_resize_preserves_xy_and_raw_w_h_before_axes(scale):
    geom=t.view_geometry([1031,721],scale); sx,sy=geom['scale_factor'][:2]
    b,bm,xy=inputs(True,scales=(sx,sy))
    points=m.box_midpoints(b[:,:5]); local=m.original_to_roi(points,b,bm,xy)
    moved=points+points.new_tensor([.7,-.4])
    delta=m.original_to_roi(moved,b,bm,xy)-local
    restored=points+m.roi_delta_to_original(delta,b,bm,xy)
    assert torch.allclose(restored,moved,atol=1e-5,rtol=0.)
    # Anisotropic rounding is applied to actual xy points; GT's native box
    # annotation scaling remains a separate unchanged pipeline convention.
    cb=t.g.canonical_boxes(bm); side=(cb[:,2:4]*1.5).clamp(min=16.)
    model_delta=(moved-points)*xy[:,None]
    c,s=cb[:,4].cos(),cb[:,4].sin()
    expected=torch.stack((model_delta[...,0]*c[:,None]+model_delta[...,1]*s[:,None],
        -model_delta[...,0]*s[:,None]+model_delta[...,1]*c[:,None]),-1)/side[:,None]
    assert torch.allclose(delta,expected,atol=1e-6,rtol=0.)
    wrong=bm.clone(); wrong[:,[2,3]]=wrong[:,[3,2]]
    with pytest.raises(ValueError,match='restoration'):
        m.frame(b,wrong,xy)


@pytest.mark.parametrize('swap',[False,True])
def test_exact_rectangle_midpoints_decode_center_edges_angle_with_raw_association(swap):
    b,bm,xy=inputs(swap)
    desired=b.clone(); desired[:,:2]+=torch.tensor([.7,-.4]); desired[:,2:4]*=torch.tensor([1.04,.96]); desired[:,4]+=.03
    target=m.target_points(desired[:,:5],b,bm,xy)
    out=m.decode_points(target['original'],b,target['roi'],torch.tensor([False]))
    assert out['accepted'].all()
    assert torch.allclose(out['boxes_original'],desired,atol=1e-5,rtol=0.)
    assert out['rectangle_projection_rms_px'].max()<1e-5
    assert torch.equal(out['boxes_original'][:,5],b[:,5])


@pytest.mark.parametrize('reason',['center_bound','edge_bound','angle_bound','roi_inside','positive','direction_valid','finite'])
def test_each_failure_returns_whole_b_without_gt_or_score_change(reason):
    b,bm,xy=inputs(); points=m.box_midpoints(b[:,:5]); local=m.original_to_roi(points,b,bm,xy)
    if reason=='center_bound':
        points+=torch.tensor([5.,0.])
    elif reason=='edge_bound':
        points=b[:,None,:2]+(points-b[:,None,:2])*1.30
    elif reason=='angle_bound':
        rotated=b[:,:5].clone(); rotated[:,4]+=.19; points=m.box_midpoints(rotated)
    elif reason=='roi_inside':
        local[:,0,0]=.6
    elif reason=='positive':
        points[:,1]=points[:,3]
    elif reason=='direction_valid':
        points[:]=b[:,None,:2]
    else:
        points[:,0,0]=float('nan')
    out=m.decode_points(points,b,local,torch.tensor([False]))
    assert not out['checks'][reason].all() and not out['accepted'].any()
    assert torch.equal(out['boxes_original'],b)
    if reason in ('positive','direction_valid','finite'):
        assert not out['candidate_valid'].any()


def test_continuous_expectation_and_effective_four_point_then_stem_gradient_no_input_graph():
    torch.manual_seed(1703); head=m.SpatialMidpointHead()
    b,bm,xy=inputs(); b.requires_grad_(); bm.requires_grad_(); xy.requires_grad_()
    roi=torch.randn(2,256,9,9,requires_grad=True); support=torch.ones(2,1,9,9,requires_grad=True)
    b=b.repeat(2,1); bm=bm.repeat(2,1); xy=xy.repeat(2,1)
    b.retain_grad(); bm.retain_grad(); xy.retain_grad()
    gt=b[:,:5].detach().clone(); gt[:,:2]+=torch.tensor([.5,-.3]); gt[:,2:4]*=1.04; gt[:,4]+=.03; gt.requires_grad_()
    opt=torch.optim.Adam(head.parameters(),lr=.001)
    for step in range(2):
        opt.zero_grad(); out=head(roi,support,b,bm,xy)
        target=m.target_points(gt,b,bm,xy); loss,parts=m.point_loss(out['points_roi'],target)
        if step==0:
            norms=[torch.autograd.grad(p,head.output.weight,retain_graph=True)[0].norm() for p in parts]
            assert all(x>0 for x in norms)
        loss.backward()
        stem=sum(float(p.grad.square().sum()) for p in head.stem.parameters())
        assert stem==0 if step==0 else stem>0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        opt.step()
    assert all(x.grad is None for x in (roi,support,b,bm,xy,gt))
    # Intermediate continuous positions are not restricted to integer ROI cells.
    assert not torch.allclose(out['points_roi']*8,(out['points_roi']*8).round())


def synthetic():
    samples=[dict(image='%s_seq01_%05d'%(domain,j+(0 if role=='fit' else 10)),
        domain=domain,role=role,sequence=domain+'_seq01',frame_id=j,
        image_size=[1024,1024]) for role in ('fit','probe') for domain in ('real','sim') for j in range(3)]
    records=[]
    rng=torch.Generator().manual_seed(211)
    for scale in (1.,.5):
        for i,sample in enumerate(samples):
            b,bm,xy=inputs(scales=(scale,scale)); b[:,0]+=i; bm[:,0]+=i*scale
            gt=b[:,:5].clone(); gt[:,:2]+=torch.tensor([.4,-.2]); gt[:,2:4]*=1.03; gt[:,4]+=.02
            sample['gt']=gt[0].tolist()
            local={arm:dict(roi=torch.randn(1,256,9,9,generator=rng)*.1,
                support=torch.ones(1,1,9,9)) for arm in t.g.ARMS}
            records.append(dict(sample,scale=scale,eligible=True,boxes_original=b,
                boxes_model=bm,gt_original=gt,local=local,
                parsed_gt_original=gt[0].tolist(),reference_gt_absolute_delta=[0.]*5))
    return records,samples


def test_donor_changes_aligned_roi_only_two_scales_paired_and_fit_only_schedule():
    records,samples=synthetic(); mapping=t.a.donor_map(samples)
    prepared,assignment=t.prepare_records(records,samples,mapping)
    lookup={(r['image'],r['scale']):r for r in records}
    for old,new,who in zip(records,prepared,assignment):
        donor=lookup[(mapping[old['image']],old['scale'])]
        for k in old:
            if k!='local': assert new[k] is old[k]
        for k in ('roi','support'):
            assert new['local']['matched'][k] is old['local']['aligned'][k]
            assert new['local']['shuffled'][k] is donor['local']['aligned'][k]
        assert who['role']==donor['role'] and who['domain']==donor['domain']
    batches=t.g.schedule(prepared)
    assert batches==t.g.schedule(prepared)
    assert all(len(b)==8 and sum(prepared[i]['domain']=='real' for i in b)==4 and
               all(prepared[i]['role']=='fit' for i in b) for b in batches)
    bad=deepcopy(mapping); bad[samples[0]['image']]=samples[0]['image']
    with pytest.raises(ValueError,match='mapping'):
        t.prepare_records(records,samples,bad)
    records[0]['gt_original'][:,0]+=200
    with pytest.raises(ValueError,match='outside'):
        t.prepare_records(records,samples,mapping)


def test_denominators_include_missing_and_incorrect_outputs():
    b,_,_=inputs(); gt=b[0,:5].numpy(); rows=[]
    for j,pred in enumerate((None,gt.copy(),gt.copy())):
        if j==2: pred[0]+=50
        rows.append(dict(role='probe',domain='real',scale=1.,image=str(j),sequence='real_seq01',
            frame_id=j,metrics=t.g.decompose(gt,pred),point_error_px=[0.]*4,
            point_error_roi=[0.]*4,point_entropy=[0.]*4,accepted=True,candidate_valid=True))
    group=t.summarize_rows(rows)['probe/real/1.0']
    assert group['output_coverage_fraction']==dict(numerator=2,denominator=3)
    assert group['conditional_center_correct_fraction']==dict(numerator=1,denominator=2)
    assert group['all_frame_center_correct_fraction']==dict(numerator=1,denominator=3)
    assert group['output_center_hit_pct']==50 and group['longest_riou_failure_run'] is None


def test_in_roi_but_unreachable_calibrated_target_is_not_silently_clamped(monkeypatch):
    records,samples=synthetic(); old=m.target_points
    def impossible(*args):
        value=old(*args)
        # Geometrically inside the ROI, but outside the calibrated expectation
        # range for the positive-long B anchor near +1/3.
        value['roi'][:,0,0]=-.5
        return value
    monkeypatch.setattr(m,'target_points',impossible)
    with pytest.raises(ValueError,match='expectation unreachable'):
        t.prepare_records(records,samples,t.a.donor_map(samples))


def test_review_requires_image_increment_and_joint_geometry_not_loss_alone():
    records,samples=synthetic(); records,_=t.prepare_records(records,samples,t.a.donor_map(samples))
    initial,_=t.evaluate(m.SpatialMidpointHead(),records,'matched',torch.device('cpu'))
    improved_loss=deepcopy(initial)
    improved_loss['point_loss']={k:0. for k in initial['point_loss']}
    assert not t.review_conditions(initial,improved_loss,initial)['diagnostic_continuation_conditions_met']
    good=deepcopy(initial)
    for row in good['rows']:
        row['point_error_px']=[v*.5 for v in row['point_error_px']]
        row['point_error_roi']=[v*.5 for v in row['point_error_roi']]
    for key,group in good['groups'].items():
        for units in ('px','roi'):
            for stat in ('mean','rmse'):
                group['point_error_'+units][stat]*=.5
    good['groups']['probe/real/1.0']['center_error_px']['mean']-=.01
    good['groups']['probe/real/1.0']['all_frame_mean_riou']+=.001
    good['groups']['probe/sim/1.0']['angle_error_deg']['rmse']-=.01
    good['groups']['probe/sim/1.0']['all_frame_mean_riou']+=.001
    assert t.review_conditions(initial,good,initial)['diagnostic_continuation_conditions_met']
    assert not t.review_conditions(initial,good,good)['diagnostic_continuation_conditions_met']
    good['groups']['probe/real/0.5']['short_edge_relative_error']['mean']+=.001
    assert not t.review_conditions(initial,good,initial)['diagnostic_continuation_conditions_met']


def test_static_entry_never_loads_cache_initializes_gpu_or_updates(monkeypatch,tmp_path):
    _,samples=synthetic(); protocol=t.protocol_document(samples)
    monkeypatch.setattr(t,'checked_contract',lambda _: (protocol,samples,{},{}))
    def forbidden(*_,**__): raise AssertionError('Cache/GPU/fit forbidden')
    monkeypatch.setattr(torch,'load',forbidden)
    monkeypatch.setattr(torch.cuda,'init',forbidden)
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    monkeypatch.setattr(t.g,'checked_cache',forbidden)
    monkeypatch.setattr(t,'fit_arm',forbidden)
    report=t.run(t.parser().parse_args(['--check-only','--out-dir',str(tmp_path/'static')]))
    assert report['head_updates_total']==report['cache_loads']==report['detector_forward_calls']==0
    assert report['gpu_devices_used']==[] and not report['formal_training_approved']
    assert not (tmp_path/'static'/'cache_manifest.json').exists()
    artifacts=json.loads((tmp_path/'static'/'artifacts.json').read_text())
    assert artifacts['protocol']==t.VERSION
    assert all(t.g.ready.sha(tmp_path/'static'/p)==sha for p,sha in artifacts['files'].items())


@pytest.mark.parametrize('fail_second',[False,True])
def test_full_entry_cpu_simulation_final_artifacts_budget_and_failure(monkeypatch,tmp_path,fail_second):
    records,samples=synthetic(); cache_dir=tmp_path/'cache'; cache_dir.mkdir()
    t.g.write_new(cache_dir/'cache_manifest.json',dict(synthetic=True))
    monkeypatch.setattr(t.a,'CACHE_MANIFEST_SHA',t.g.ready.sha(cache_dir/'cache_manifest.json'))
    protocol=t.protocol_document(samples); mapping=protocol['donor_mapping']
    prepared,_=t.prepare_records(records,samples,mapping)
    baseline=dict(minibatches=t.g.schedule(prepared)[:2],
                  arms=dict(ordinary=dict(initial=dict(rows=[dict(pred=r['boxes_original'][0].tolist()) for r in records]))))
    monkeypatch.setitem(m.SETTINGS,'steps_per_arm',2)
    monkeypatch.setattr(t,'checked_contract',lambda _: (protocol,samples,dict(prior_identity=dict(g_identity={})),baseline))
    monkeypatch.setattr(t.a,'checked_runtime',lambda _: dict(evidence='CPU_SIMULATION'))
    monkeypatch.setattr(t.g,'checked_cache',lambda *_:(dict(records=records,detector_state_before={}),dict(synthetic=True)))
    monkeypatch.setattr(t.g,'validate_records',lambda *_:None)
    monkeypatch.setattr(t.a,'check_cache_reference',lambda *_:None)
    original_schedule=t.g.schedule
    monkeypatch.setattr(t.g,'schedule',lambda r:original_schedule(r)[:2])
    monkeypatch.setattr(t.g,'measured',lambda op,_:(op(),dict(evidence='CPU_SIMULATION')))
    original_fit=t.fit_arm
    def fit(rec,bat,init,condition,_,progress):
        if fail_second and condition=='shuffled': raise ValueError('synthetic second-arm failure')
        return original_fit(rec,bat,init,condition,torch.device('cpu'),progress)
    monkeypatch.setattr(t,'fit_arm',fit)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'set_device',lambda _:None)
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda _:None)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _:'CPU simulation')
    out=tmp_path/'full'; args=t.parser().parse_args(['--cache-dir',str(cache_dir),'--out-dir',str(out)])
    if fail_second:
        with pytest.raises(ValueError,match='second-arm'):
            t.run(args)
    else:
        report=t.run(args)
        assert report['arms']['matched']['initial']==report['arms']['shuffled']['initial']
        assert report['arms']['matched']['initial_state']==report['arms']['shuffled']['initial_state']
        assert report['review']['formal_training_approved'] is False
        assert len(json.loads((out/'shuffled.spatial.json').read_text())['views'])==len(records)
    report=json.loads((out/'completion.json').read_text())
    assert report['head_updates_total']==(2 if fail_second else 4)
    assert report['status']==('FAILED_REVIEW_REQUIRED' if fail_second else 'TRAIN_MIDPOINT_CHECK_COMPLETE_REVIEW_REQUIRED')
    assert report['detector_updates']==report['detector_forward_calls']==report['feature_extractions']==0
    artifacts=json.loads((out/'artifacts.json').read_text())
    assert artifacts['protocol']==t.VERSION
    assert all(t.g.ready.sha(out/p)==sha for p,sha in artifacts['files'].items())
    events=[json.loads(line) for line in (out/'progress.jsonl').read_text().splitlines()]
    assert sum(e['stage']=='update' for e in events)==report['head_updates_total']


def test_fixed_sources_and_protocol_no_self_donor_and_no_probe_training():
    sources=t.checked_sources()
    samples=json.loads(t.g.FIXTURE.read_text())['samples']
    assert json.loads(t.PROTOCOL.read_text())==t.protocol_document(samples)
    assert len(sources)==len(json.loads(t.MANIFEST.read_text())['sources'])
    assert m.SETTINGS['steps_per_arm']==200 and m.SETTINGS['seed']==1703


def test_altered_source_is_rejected_without_refreshing_manifest(monkeypatch):
    old=t.g.ready.sha
    monkeypatch.setattr(t.g.ready,'sha',lambda p:'changed' if str(p).endswith('utils/port_geometry_midpoint_v1.py') else old(p))
    with pytest.raises(ValueError,match='source SHA'):
        t.checked_sources()
