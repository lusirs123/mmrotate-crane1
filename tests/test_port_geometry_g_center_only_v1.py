"""CPU regressions for task isolation, paired evidence and fixed scope."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from crane_project.tools import preflight_port_geometry_g_center_only_v1 as o
from test_port_geometry_g_center_v1 import records_with_center_targets


def reference():
    o.g.seed_all(); wide = o.g.LocalGeometryRefiner()
    baseline = dict(arms=dict(ordinary=dict(initial_state=o.g.state_digest(wide))))
    state,evidence = o.c.checked_initialization(baseline['arms']['ordinary'])
    joint = dict(joint_center=dict(initial_state=evidence['joint_initial_state']))
    return state,baseline,joint


def evaluation():
    records,samples = records_with_center_targets(); state,baseline,joint = reference()
    head = o.CenterOnlyLocalGeometryRefiner(); head.load_state_dict(state)
    value = o.evaluate(head,records,torch.device('cpu'))
    return records,samples,state,value


def test_same_joint_state_and_zero_b_outputs_with_frozen_bypassed_shape(monkeypatch):
    state,baseline,joint = reference()
    checked,ev = o.checked_initialization(baseline,joint)
    assert ev['shared_joint_state_exact'] and ev['initial_state']==joint['joint_center']['initial_state']
    assert all(torch.equal(state[k],v) for k,v in checked.items())
    head = o.CenterOnlyLocalGeometryRefiner(); head.load_state_dict(state)
    assert sum(p.numel() for p in head.parameters())==184037
    assert sum(p.numel() for p in head.parameters() if p.requires_grad)==183842
    records,_ = records_with_center_targets()
    roi,support,b,bm,_ = o.g.batch_tensors(records,[0],'ordinary',torch.device('cpu'))
    assert torch.equal(head(roi,support,b,bm)['boxes_original'],b)
    # Poison the unused layer and prohibit its execution: no hidden shape path.
    with torch.no_grad(): head.output.bias.fill_(100.)
    monkeypatch.setattr(head.output,'forward',lambda *_: (_ for _ in ()).throw(AssertionError('shape output executed')))
    assert torch.equal(head(roi,support,b,bm)['boxes_original'],b)
    joint['joint_center']['initial_state']['parameters']='wrong'
    with pytest.raises(ValueError,match='initialization/trainable partition'):
        o.checked_initialization(baseline,joint)


def test_center_loss_matches_joint_coefficient_and_ignores_gt_shape_and_edge_inflation():
    b = torch.tensor([[30.,40.,40.,20.,.2,.8]],requires_grad=True)
    gt = torch.tensor([[31.,38.,40.,20.,.2]],requires_grad=True)
    refined = b.detach().clone().requires_grad_()
    loss,parts = o.regression_loss(refined,gt,b)
    _,joint_parts = o.c.regression_loss(refined,gt,b)
    assert torch.equal(parts,joint_parts[3:]) and torch.equal(loss,joint_parts[3:].sum()/3.)
    assert not torch.equal(loss,parts.mean())
    changed_gt = gt.detach().clone(); changed_gt[:,2:]=torch.tensor([2.,90.,-1.])
    inflated = refined.detach().clone(); inflated[:,2:4]*=1.2
    other,other_parts = o.regression_loss(inflated,changed_gt,b)
    assert torch.equal(loss,other) and torch.equal(parts,other_parts)
    loss.backward()
    assert torch.count_nonzero(refined.grad[:,:2])==2
    assert torch.count_nonzero(refined.grad[:,2:])==0
    assert b.grad is None and gt.grad is None
    with pytest.raises(ValueError,match='nonempty paired'):
        o.regression_loss(refined[:0],gt[:0],b[:0])


def test_center_gradients_exact_and_two_updates_reach_stem_without_shape_or_input_gradients():
    records,_ = records_with_center_targets(); state,_,_ = reference()
    batches = o.g.schedule(records)[:2]
    check = o.gradient_check(records,batches[0],state,torch.device('cpu'))
    assert check['initial_shared_center_gradients_exact'] and check['both_center_tasks_effective']
    before = [r['boxes_original'].clone() for r in records]; progress=[]
    result = o.fit_center_only(records,batches,state,torch.device('cpu'),progress.append)
    assert result['logs'][0]['stem_grad_norm']==0
    assert result['stem_gradient_after_zero_output_step_effective']
    assert result['frozen_shape_state_unchanged'] and result['finite_gradients']
    assert len(result['logs'])==2 and sum(v['stage']=='update' for v in progress)==2
    for row,b in zip(result['final']['rows'],before):
        assert row['pred'][2:]==b[0,2:].tolist()
    assert any(row['pred'][:2]!=b[0,:2].tolist() for row,b in zip(result['final']['rows'],before))
    assert all(torch.equal(b,r['boxes_original']) for b,r in zip(before,records))
    roi,support,b,bm,gt = [v.detach().clone().requires_grad_() for v in o.g.batch_tensors(records,[0],'ordinary',torch.device('cpu'))]
    head=o.CenterOnlyLocalGeometryRefiner(); head.load_state_dict(state)
    loss,_=o.regression_loss(head(roi,support,b,bm)['boxes_original'],gt,b); loss.backward()
    assert all(v.grad is None for v in (roi,support,b,bm,gt))
    assert all(p.grad is None for p in head.output.parameters())
    assert head.center_output.weight.grad.norm()>0


@pytest.mark.parametrize('xy',[(8.,-8.),(1e20,-1e20)])
def test_radial_decode_finite_bound_scale_and_raw_shape_identity(xy):
    records,_=records_with_center_targets(); state,_,_=reference()
    head=o.CenterOnlyLocalGeometryRefiner(); head.load_state_dict(state)
    with torch.no_grad(): head.center_output.bias.copy_(torch.tensor(xy))
    roi,support,b,bm,_=o.g.batch_tensors(records,[0],'ordinary',torch.device('cpu'))
    output=head(roi,support,b,bm); o.validate_prediction(output,b)
    assert torch.isfinite(output['boxes_original']).all()
    assert .29<output['center_residual'].norm().item()<=.3000001
    scaled=b.clone(); scaled[:,:4]*=.5
    target=output['boxes_original'].clone(); target[:,:4]*=.5
    assert torch.allclose(head(roi,support,scaled,bm)['boxes_original'],target,atol=1e-6,rtol=0)
    assert head(roi[:0],support[:0],b[:0],bm[:0])['boxes_original'].shape==(0,6)
    swapped=b.clone(); swapped[:,2:4]=b[:,[3,2]]
    new=head(roi,support,swapped,bm)['boxes_original']
    assert torch.equal(new[:,2:],swapped[:,2:]) and torch.equal(new[:,:2],output['boxes_original'][:,:2])
    bad={k:v.detach().clone() for k,v in output.items()}; bad['boxes_original'][:,4]+=.1
    with pytest.raises(ValueError,match='raw B size/angle'):
        o.validate_prediction(bad,b)


def test_saved_hybrid_pairs_centers_with_b_raw_shape_and_does_not_mutate_reports():
    _,_,_,base=evaluation(); learned=deepcopy(base)
    for row in learned['rows']:
        row['pred'][0]+=.25; row['pred'][2]*=1.1; row['pred'][4]+=.02
        row['metrics']=o.g.decompose(row['gt_original'],row['pred'][:5])
    baseline=dict(arms=dict(ordinary=dict(initial=base)))
    joint=dict(joint_center=dict(final=learned)); unchanged=deepcopy((baseline,joint))
    hybrid=o.saved_joint_center_b_shape(baseline,joint)
    assert (baseline,joint)==unchanged and hybrid['additional_updates']==0
    for old,new,ref in zip(base['rows'],learned['rows'],hybrid['rows']):
        assert ref['pred'][:2]==new['pred'][:2] and ref['pred'][2:]==old['pred'][2:]
    joint['joint_center']['final']['rows'][0]['scale']=.25
    with pytest.raises(ValueError,match='Paired view/GT'):
        o.saved_joint_center_b_shape(baseline,joint)


def test_three_denominators_direction_counts_and_center_failure_gate():
    _,_,_,base=evaluation(); current=deepcopy(base)
    subset=[r for r in base['rows'] if r['role']=='probe' and r['domain']=='real' and r['scale']==1.]
    for value in (base,current):
        value['rows'][value['rows'].index(next(r for r in value['rows'] if r['image']==subset[0]['image'] and r['scale']==1.))]['pred']=None
        for row in value['rows']:
            row['metrics']=o.g.decompose(row['gt_original'],row['pred'][:5] if row['pred'] else None)
    change=next(r for r in current['rows'] if r['image']==subset[1]['image'] and r['scale']==1.)
    change['pred'][0]+=30; change['metrics']=o.g.decompose(change['gt_original'],change['pred'][:5])
    summary=o.summarize_rows(current['rows'])['probe/real/1.0']
    assert summary['output_coverage_fraction']==dict(numerator=3,denominator=4)
    assert summary['conditional_center_correct_fraction']==dict(numerator=2,denominator=3)
    assert summary['all_frame_center_correct_fraction']==dict(numerator=2,denominator=4)
    motion=o.motion_report(base,current); group=motion['groups']['probe/real/1.0']
    assert group['output_views']==3 and group['paired_counts']['center_worse']==1
    assert group['sign_counts']['actual_shift_px']['x']==dict(negative=0,positive=1,zero=2)
    review=o.review(motion['comparison'],motion['comparison'])
    assert not review['all_probe_non_regression_checks_met'] and not review['formal_training_approved']


def test_review_requires_real_gain_vs_b_and_joint_and_all_groups_including_real_fit():
    _,_,_,base=evaluation(); same=o.c.paired_changes(base,base)
    report=o.review(same,same)
    assert report['all_probe_non_regression_checks_met'] and not report['diagnostic_continuation_conditions_met']
    improve=deepcopy(same)
    improve['groups']['probe/real/1.0']['delta'].update({'center_error_px/mean':-.01,'all_frame_mean_riou':.001})
    assert o.review(improve,improve)['diagnostic_continuation_conditions_met']
    assert not o.review(improve,same)['diagnostic_continuation_conditions_met']
    tiny=deepcopy(same)
    tiny['groups']['probe/real/1.0']['delta'].update({'center_error_px/mean':-1e-7,'all_frame_mean_riou':1e-6})
    assert not o.review(tiny,tiny)['diagnostic_continuation_conditions_met']
    improve['groups']['fit/real/1.0']['delta']['center_error_px/rmse']=.01
    assert not o.review(improve,improve)['real_fit_center_non_regression_met']
    del improve['groups']['probe/sim/0.5']
    with pytest.raises(ValueError,match='all8'): o.review(improve,same)


def test_unreviewed_joint_report_rejected_before_it_can_define_reference(monkeypatch,tmp_path):
    samples=json.loads(o.g.FIXTURE.read_text())['samples']
    monkeypatch.setattr(o.c,'checked_contract',lambda _:(None,samples,{},{}))
    changed=tmp_path/'joint.json'; changed.write_text('{}\n')
    with pytest.raises(ValueError,match='exact reviewed joint completion'):
        o.checked_contract(Path('/unused-baseline'),changed)


def test_sources_fail_before_data_and_static_entry_never_loads_cache_gpu_or_head(monkeypatch,tmp_path):
    manifest=json.loads(o.MANIFEST.read_text())
    manifest['sources']['crane_project/utils/port_geometry_refine_g_center_only_v1.py']='0'*64
    bad=tmp_path/'bad.json'; bad.write_text(json.dumps(manifest)); monkeypatch.setattr(o,'MANIFEST',bad)
    with pytest.raises(ValueError,match='source SHA differs'):
        o.checked_contract(Path('/missing'),Path('/missing'))
    _,samples,_,base=evaluation(); protocol=o.protocol_document(samples)
    baseline=dict(arms=dict(ordinary=dict(initial=base))); hybrid=deepcopy(base)
    monkeypatch.setattr(o,'checked_contract',lambda *_:(protocol,samples,{},baseline,{},hybrid))
    def forbidden(*_): raise AssertionError('Static attempted cache/GPU/head/update')
    monkeypatch.setattr(o.g,'checked_cache',forbidden); monkeypatch.setattr(o,'checked_initialization',forbidden)
    monkeypatch.setattr(o.torch.cuda,'is_available',forbidden); monkeypatch.setattr(o,'fit_center_only',forbidden)
    out=tmp_path/'static'; monkeypatch.setattr(o.sys,'argv',['only','--check-only','--out-dir',str(out)])
    o.main(); report=json.loads((out/'completion.json').read_text())
    assert report['head_updates_total']==report['gpu_devices_used']==report['detector_forward_calls']==0
    assert report['status']=='STATIC_CENTER_ONLY_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES'
    artifacts=json.loads((out/'artifacts.json').read_text())
    assert all(o.g.ready.sha(out/p)==sha for p,sha in artifacts['files'].items())
    with pytest.raises(FileExistsError): o.main()


@pytest.mark.parametrize('failed_review',[False,True])
def test_full_entry_two_cpu_updates_and_failure_preserves_completed_head(monkeypatch,tmp_path,failed_review):
    records,samples=records_with_center_targets(); state,baseline,joint=reference()
    monkeypatch.setattr(o,'STEPS',2); monkeypatch.setitem(o.g.SETTINGS,'steps_per_arm',2)
    head=o.CenterOnlyLocalGeometryRefiner(); head.load_state_dict(state)
    b=o.evaluate(head,records,torch.device('cpu'))
    baseline['arms']['ordinary']['initial']=b; baseline['minibatches']=o.g.schedule(records)
    joint.update(minibatches=baseline['minibatches'])
    joint['joint_center']['initial']=dict(eligible_center_loss=deepcopy(b['eligible_center_loss']))
    if failed_review: joint['joint_center']['initial']['eligible_center_loss']['fit/real/1.0']+=.01
    protocol=o.protocol_document(samples); cache=dict(identity={},status='COMPLETE',record_count=len(records))
    baseline['cache']=cache
    monkeypatch.setattr(o,'checked_contract',lambda *_:(protocol,samples,dict(prior_identity=dict(prior_identity=dict(g_identity={}))),baseline,joint,deepcopy(b)))
    monkeypatch.setattr(o.c.a,'checked_runtime',lambda _:dict(evidence='CPU simulation only'))
    monkeypatch.setattr(o.g,'checked_cache',lambda *_:(dict(records=records),cache))
    monkeypatch.setattr(o.g,'validate_records',lambda *_:None)
    monkeypatch.setattr(o.g,'measured',lambda fn,_:(fn(),dict(evidence='CPU simulation only')))
    fit,gradient=o.fit_center_only,o.gradient_check
    monkeypatch.setattr(o,'fit_center_only',lambda r,b,i,_d,p:fit(r,b,i,torch.device('cpu'),p))
    monkeypatch.setattr(o,'gradient_check',lambda r,b,i,_d:gradient(r,b,i,torch.device('cpu')))
    for k,v in [('is_available',lambda:True),('device_count',lambda:1),('set_device',lambda _:None),('get_device_name',lambda _:'CPU simulation only')]:
        monkeypatch.setattr(o.torch.cuda,k,v)
    cache_dir=tmp_path/'cache'; cache_dir.mkdir(); o.g.write_new(cache_dir/'cache_manifest.json',cache)
    monkeypatch.setattr(o.c.a,'CACHE_MANIFEST_SHA',o.g.ready.sha(cache_dir/'cache_manifest.json'))
    out=tmp_path/'run'; monkeypatch.setattr(o.sys,'argv',['only','--cache-dir',str(cache_dir),'--out-dir',str(out)])
    if failed_review:
        with pytest.raises(ValueError,match='Initial center loss'): o.main()
    else: o.main()
    report=json.loads((out/'completion.json').read_text())
    assert report['head_updates_total']==2 and report['additional_joint_updates']==0
    assert report['detector_forward_calls']==report['detector_updates']==0
    assert (out/'center_only.json').exists() and not any(p.suffix in ('.pth','.pt') for p in out.rglob('*'))
    assert report['status']==('FAILED_REVIEW_REQUIRED' if failed_review else 'TRAIN_CENTER_ONLY_CHECK_COMPLETE_REVIEW_REQUIRED')
    if not failed_review:
        assert len(report['center_only']['logs'])==2
        assert not report['probe_review']['formal_training_approved']
    artifacts=json.loads((out/'artifacts.json').read_text())
    assert all(o.g.ready.sha(out/p)==sha for p,sha in artifacts['files'].items())


def test_failed_serialization_preserves_scope_budget(monkeypatch,tmp_path):
    protocol=o.protocol_document(json.loads(o.g.FIXTURE.read_text())['samples'])
    baseline=dict(arms=dict(ordinary=dict(initial=dict(rows=[]))))
    monkeypatch.setattr(o,'checked_contract',lambda *_:(protocol,[],dict(bad=float('nan')),baseline,{},{}))
    out=tmp_path/'failed'; monkeypatch.setattr(o.sys,'argv',['only','--check-only','--out-dir',str(out)])
    with pytest.raises(ValueError,match='JSON compliant'): o.main()
    report=json.loads((out/'completion.json').read_text())
    assert report['status']=='FAILED_REVIEW_REQUIRED' and report['head_updates_total']==0
    assert report['additional_joint_updates']==0 and not report['full_report_available']
