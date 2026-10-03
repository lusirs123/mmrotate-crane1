"""Necessary CPU tests for bounded center correction and evidence boundaries."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch

from crane_project.tools import preflight_port_geometry_g_center_v1 as c
from crane_project.utils.port_geometry_refine_g_center_v1 import apply_joint_residual
from test_port_geometry_g_capacity_v1 import synthetic


def records_with_center_targets():
    records,samples=synthetic()
    for i,r in enumerate(records):
        gt=r['gt_original'].clone(); gt[:,:2]+=torch.tensor([.6,-.4])
        r['gt_original']=gt; r['gt']=gt[0].tolist()
        r['parsed_gt_original']=gt[0].tolist()
    return records,samples


def initial():
    c.g.seed_all(); wide=c.g.LocalGeometryRefiner()
    return c.checked_initialization(dict(initial_state=c.g.state_digest(wide)))


def test_shared_initialization_exact_and_initial_five_outputs_are_b():
    state,evidence=initial()
    c.g.seed_all(); wide=c.g.LocalGeometryRefiner()
    head=c.CenterLocalGeometryRefiner(); head.load_state_dict(state)
    assert sum(p.numel() for p in head.parameters())==184037
    assert evidence['all_shared_tensors_exact'] and evidence['added_center_units_zero']
    for k,v in wide.state_dict().items():
        assert torch.equal(state[k],v) and state[k].data_ptr()!=v.data_ptr()
    records,_=records_with_center_targets()
    output=head(*c.g.batch_tensors(records,[0],'ordinary',torch.device('cpu'))[:4])
    assert torch.equal(output['boxes_original'],records[0]['boxes_original'])
    assert torch.equal(output['center_residual'],torch.zeros(1,2))


@pytest.mark.parametrize('bad',['keys','shape','nonfinite','learned','digest'])
def test_invalid_initialization_cannot_be_reused(bad):
    c.g.seed_all(); wide=c.g.LocalGeometryRefiner(); state=deepcopy(wide.state_dict())
    if bad=='keys': del state['hidden.0.bias']
    elif bad=='shape': state['hidden.0.bias']=state['hidden.0.bias'][:16]
    elif bad=='nonfinite': state['stem.0.weight'][0,0,0,0]=float('nan')
    elif bad=='learned': state['output.bias'][0]=.001
    else:
        digest=c.g.state_digest(wide); digest['parameters']='wrong'
        with pytest.raises(ValueError,match='initialization differs'):
            c.checked_initialization(dict(initial_state=digest))
        return
    with pytest.raises(ValueError): c.initial_state_from_reference(state)


def test_radial_bound_is_not_per_axis_and_is_scale_equivariant():
    state,_=initial(); head=c.CenterLocalGeometryRefiner(); head.load_state_dict(state)
    with torch.no_grad():
        head.center_output.bias.copy_(torch.tensor([8.,-8.]))
    records,_=records_with_center_targets()
    roi,support,b,bm,_=c.g.batch_tensors(records,[0],'ordinary',torch.device('cpu'))
    out=head(roi,support,b,bm); c.validate_prediction(out,b)
    shift=(out['boxes_original'][:,:2]-b[:,:2]).norm().item()
    assert .29<shift/min(b[0,2:4]).item()<.30
    assert torch.equal(out['boxes_original'][:,2:],b[:,2:])
    scaled=b.clone(); scaled[:,:4]*=.5
    # Hold model ROI/descriptor fixed: original coordinates alone change units.
    scaled_out=head(roi,support,scaled,bm)['boxes_original']
    expected=out['boxes_original'].clone(); expected[:,:4]*=.5
    assert torch.allclose(scaled_out,expected,atol=1e-6,rtol=0)
    with pytest.raises(ValueError,match='radial bound'):
        apply_joint_residual(b,torch.zeros(1,3),torch.tensor([[.25,.25]]))


def test_xy_reflection_and_width_exchange_keep_physical_meaning():
    b=torch.tensor([[30.,40.,40.,20.,.2,.8]])
    shape=torch.tensor([[.03,-.04,.02]]); xy=torch.tensor([[.1,-.15]])
    result,_=apply_joint_residual(b,shape,xy)
    meta=dict(ori_shape=(128,128,3),img_shape=(64,64,3),pad_shape=(64,64,3),
              scale_factor=[.5]*4,flip=True,flip_direction='horizontal')
    mapped=c.g.map_boxes(result[:,:5],meta)
    restored=c.g.map_boxes(mapped,meta,inverse=True)
    assert torch.allclose(restored,result[:,:5],atol=2e-6,rtol=0)
    swapped=b.clone(); swapped[:,2:4]=b[:,[3,2]]; swapped[:,4]-=math.pi/2
    other,_=apply_joint_residual(swapped,shape,xy)
    assert torch.allclose(c.g.canonical_boxes(result[:,:5]),c.g.canonical_boxes(other[:,:5]),atol=1e-6,rtol=0)
    empty,_=apply_joint_residual(b[:0],shape[:0],xy[:0])
    assert empty.shape==(0,6)


def test_large_center_activations_remain_finite_and_do_not_collapse_to_zero():
    state,_=initial(); head=c.CenterLocalGeometryRefiner(); head.load_state_dict(state)
    records,_=records_with_center_targets()
    roi,support,b,bm,_=c.g.batch_tensors(records,[0],'ordinary',torch.device('cpu'))
    with torch.no_grad(): head.center_output.bias.copy_(torch.tensor([1e20,-1e20]))
    output=head(roi,support,b,bm); c.validate_prediction(output,b)
    assert torch.isfinite(output['boxes_original']).all()
    assert output['center_residual'].norm().item()==pytest.approx(.3,abs=1e-7)
    empty=head(roi[:0],support[:0],b[:0],bm[:0])
    assert empty['boxes_original'].shape==(0,6)


def test_center_loss_cannot_be_reduced_by_inflating_refined_edges():
    b=torch.tensor([[30.,40.,40.,20.,.2,.8]],requires_grad=True)
    gt=torch.tensor([[31.,38.,40.,20.,.2]],requires_grad=True)
    refined=b.detach().clone().requires_grad_()
    total,parts=c.regression_loss(refined,gt,b)
    shape,_=c.g.regression_loss(refined,gt)
    assert torch.equal(total,shape+parts[3:].sum()/3)
    bigger=refined.detach().clone(); bigger[:,2:4]*=1.2
    _,other=c.regression_loss(bigger,gt,b)
    assert torch.equal(parts[3:],other[3:])
    gx=torch.autograd.grad(parts[3:].sum(),refined,retain_graph=True)[0]
    assert gx[0,0]>0 or gx[0,0]<0
    assert torch.equal(gx[:,2:],torch.zeros_like(gx[:,2:]))
    total.backward()
    assert b.grad is None and gt.grad is None


def test_actual_gradient_contract_and_two_updates_reach_shared_stem():
    records,_=records_with_center_targets(); state,evidence=initial()
    batches=c.g.schedule(records)[:2]
    check=c.gradient_check(records,batches[0],state,torch.device('cpu'))
    assert check['original_shape_gradients_exact'] and check['both_center_tasks_effective']
    before=[r['boxes_original'].clone() for r in records]; progress=[]
    result=c.fit_joint(records,batches,state,torch.device('cpu'),progress.append)
    assert result['initial_state']==evidence['joint_initial_state']
    assert result['initial_all_task_gradients_effective'] and result['stem_gradient_after_zero_output_step_effective']
    assert result['logs'][0]['stem_grad_norm']==0
    assert len(result['logs'])==len(batches)==2
    assert all(v['grad_norm_after_clip']<=10.0001 for v in result['logs'])
    assert all(torch.equal(b,r['boxes_original']) for b,r in zip(before,records))
    assert any(row['pred'][:2]!=b[0,:2].tolist() for row,b in zip(result['final']['rows'],before))
    roi=torch.randn(1,256,9,9,requires_grad=True); support=torch.ones(1,1,9,9,requires_grad=True)
    b=before[0].requires_grad_(); bm=records[0]['boxes_model'].clone().requires_grad_()
    gt=records[0]['gt_original'].clone().requires_grad_()
    head=c.CenterLocalGeometryRefiner(); head.load_state_dict(state)
    loss,_=c.regression_loss(head(roi,support,b,bm)['boxes_original'],gt,b); loss.backward()
    assert all(v.grad is None for v in (roi,support,b,bm,gt))
    assert head.center_output.weight.grad.norm()>0 and head.output.weight.grad.norm()>0


def test_fit_support_does_not_use_probe_targets_or_choose_bound():
    records,_=records_with_center_targets(); state,_=initial()
    head=c.CenterLocalGeometryRefiner(); head.load_state_dict(state)
    rows=c.evaluate(head,records,torch.device('cpu'))['rows']
    original=c.fit_center_support(rows)
    changed=deepcopy(rows)
    for row in changed:
        if row['role']=='probe': row['gt_original'][:2]=[1e8,1e8]
    assert original==c.fit_center_support(changed)
    assert c.CENTER_SETTINGS['max_center_over_b_short']==.30


def test_center_changes_are_reported_with_correct_denominators_and_fixed_scores():
    records,_=records_with_center_targets(); state,_=initial()
    head=c.CenterLocalGeometryRefiner(); head.load_state_dict(state)
    reference=c.evaluate(head,records,torch.device('cpu')); current=deepcopy(reference)
    # Include an unchanged absent output: it belongs only to full-frame denominator.
    missing=next(r for r in reference['rows'] if r['role']=='probe' and r['domain']=='real' and r['scale']==1.)
    for value in (reference,current):
        row=next(r for r in value['rows'] if r['image']==missing['image'] and r['scale']==1.)
        row['pred']=None; row['metrics']=c.g.decompose(row['gt_original'],None)
    change=next(r for r in current['rows'] if r['role']=='probe' and r['domain']=='real' and r['scale']==1. and r['pred'] is not None)
    change['pred'][0]+=30
    change['metrics']=c.g.decompose(change['gt_original'],change['pred'][:5])
    comparison=c.paired_changes(reference,current); group=comparison['groups']['probe/real/1.0']
    assert group['delta']['output_coverage_pct']==0
    assert group['delta']['output_center_hit_pct']==pytest.approx(-100/3)
    assert group['delta']['all_frame_center_hit_pct']==-25
    assert group['events']['new_center_failures']==[change['image']]
    assert not c.probe_review(comparison)['all_probe_non_regression_checks_met']
    assert not c.probe_review(comparison)['formal_training_approved']
    current['rows'][0]['pred'][5]+=.01
    with pytest.raises(ValueError,match='presence/score'):
        c.paired_changes(reference,current)


def test_sources_fail_before_data_and_static_cli_avoids_cache_gpu_heads(monkeypatch,tmp_path):
    manifest=json.loads(c.MANIFEST.read_text()); manifest['sources']['crane_project/utils/port_geometry_refine_g_center_v1.py']='0'*64
    bad=tmp_path/'sources.json'; bad.write_text(json.dumps(manifest))
    with monkeypatch.context() as patch:
        patch.setattr(c,'MANIFEST',bad)
        with pytest.raises(ValueError,match='source SHA differs'):
            c.checked_contract(Path('/missing.json'))
    samples=json.loads(c.g.FIXTURE.read_text())['samples']; protocol=c.protocol_document(samples)
    monkeypatch.setattr(c,'checked_contract',lambda _:(protocol,samples,{},dict(arms=dict(ordinary=dict(initial=dict(rows=[]))))))
    def forbidden(*_): raise AssertionError('Static must not use cache/GPU/head')
    monkeypatch.setattr(c.g,'checked_cache',forbidden)
    monkeypatch.setattr(c,'checked_initialization',forbidden)
    monkeypatch.setattr(c.torch.cuda,'is_available',forbidden)
    out=tmp_path/'static'
    monkeypatch.setattr(c.sys,'argv',['center','--check-only','--out-dir',str(out)])
    c.main()
    report=json.loads((out/'completion.json').read_text())
    assert report['head_updates_total']==report['detector_forward_calls']==report['gpu_devices_used']==0
    artifacts=json.loads((out/'artifacts.json').read_text())
    assert all(c.g.ready.sha(out/p)==h for p,h in artifacts['files'].items())
    with pytest.raises(FileExistsError): c.main()


@pytest.mark.parametrize('failed_comparison',[False,True])
def test_full_entry_two_cpu_updates_and_backup_on_failed_review(monkeypatch,tmp_path,failed_comparison):
    records,samples=records_with_center_targets()
    monkeypatch.setattr(c,'STEPS',2); monkeypatch.setitem(c.g.SETTINGS,'steps_per_arm',2)
    c.g.seed_all(); wide=c.g.LocalGeometryRefiner(); initial_wide=deepcopy(wide.state_dict())
    batches=c.g.schedule(records)
    reference=c.g.fit_arm(records,batches,initial_wide,'ordinary',torch.device('cpu'),lambda _:None)
    if failed_comparison: reference['initial']['eligible_loss']['probe/real/1.0']+=.001
    cache=dict(identity={},status='COMPLETE',record_count=len(records))
    baseline=dict(arms=dict(ordinary=reference),cache=cache,minibatches=batches)
    protocol=c.protocol_document(samples)
    monkeypatch.setattr(c,'checked_contract',lambda _:(protocol,samples,dict(prior_identity=dict(g_identity={})),baseline))
    monkeypatch.setattr(c.a,'checked_runtime',lambda _:dict(evidence='CPU simulation only'))
    monkeypatch.setattr(c.g,'checked_cache',lambda *_:(dict(records=records),cache))
    monkeypatch.setattr(c.g,'validate_records',lambda *_:None)
    monkeypatch.setattr(c.g,'measured',lambda fn,_:(fn(),dict(evidence='CPU simulation only')))
    fit,gradient=c.fit_joint,c.gradient_check
    monkeypatch.setattr(c,'fit_joint',lambda r,b,i,_d,p:fit(r,b,i,torch.device('cpu'),p))
    monkeypatch.setattr(c,'gradient_check',lambda r,b,i,_d:gradient(r,b,i,torch.device('cpu')))
    for k,v in [('is_available',lambda:True),('device_count',lambda:1),('set_device',lambda _:None),('get_device_name',lambda _:'CPU simulation only')]:
        monkeypatch.setattr(c.torch.cuda,k,v)
    cache_dir=tmp_path/'cache'; cache_dir.mkdir(); c.g.write_new(cache_dir/'cache_manifest.json',cache)
    monkeypatch.setattr(c.a,'CACHE_MANIFEST_SHA',c.g.ready.sha(cache_dir/'cache_manifest.json'))
    out=tmp_path/'run'
    monkeypatch.setattr(c.sys,'argv',['center','--cache-dir',str(cache_dir),'--out-dir',str(out)])
    if failed_comparison:
        with pytest.raises(ValueError,match='initial shape loss'): c.main()
    else: c.main()
    report=json.loads((out/'completion.json').read_text())
    assert report['head_updates_total']==2 and report['additional_reference_updates']==0
    assert report['detector_forward_calls']==report['detector_updates']==0
    assert (out/'joint_center.json').exists() and not any(p.suffix=='.pth' for p in out.rglob('*'))
    assert report['status']==('FAILED_REVIEW_REQUIRED' if failed_comparison else 'TRAIN_CENTER_CHECK_COMPLETE_REVIEW_REQUIRED')
    if not failed_comparison:
        assert len(report['joint_center']['logs'])==2
        assert not report['probe_review']['formal_training_approved']
    artifacts=json.loads((out/'artifacts.json').read_text())
    assert all(c.g.ready.sha(out/p)==h for p,h in artifacts['files'].items())


def test_failed_serialization_preserves_scope_budget(monkeypatch,tmp_path):
    protocol=c.protocol_document(json.loads(c.g.FIXTURE.read_text())['samples'])
    monkeypatch.setattr(c,'checked_contract',lambda _:(protocol,[],dict(bad=float('nan')),dict(arms=dict(ordinary=dict(initial=dict(rows=[]))))))
    out=tmp_path/'failed'; monkeypatch.setattr(c.sys,'argv',['center','--check-only','--out-dir',str(out)])
    with pytest.raises(ValueError,match='JSON compliant'): c.main()
    report=json.loads((out/'completion.json').read_text())
    assert report['status']=='FAILED_REVIEW_REQUIRED' and report['head_updates_total']==0
    assert not report['full_report_available']
