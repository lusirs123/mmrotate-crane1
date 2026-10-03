"""Necessary regressions for the single FC-width intervention; CPU only."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest
import torch

from crane_project.tools import preflight_port_geometry_g_capacity_v1 as c


def synthetic():
    samples=[dict(image=d+'_seq01_'+str(j)+'_'+role,domain=d,role=role,
                  sequence=d+'_seq01',frame_id=j)
             for role in ('fit','probe') for d in ('real','sim') for j in range(4)]
    records=[]
    for scale in (1.,.5):
        for i,sample in enumerate(samples):
            b=torch.tensor([[32.+i,24.,32.,16.,.17,.8]])
            bm=b[:,:5].clone(); bm[:,:4]*=scale
            gt=b[:,:5].clone(); gt[:,2:4]*=1.04; gt[:,4]-=.03
            sample['gt']=gt[0].tolist()
            local={arm:dict(roi=torch.full((1,256,9,9),.02+i*.003+scale*.001),
                            support=torch.ones(1,1,9,9)) for arm in c.g.ARMS}
            records.append(dict(sample,scale=scale,eligible=True,boxes_original=b,
                boxes_model=bm,gt_original=gt,local=local,parsed_gt_original=gt[0].tolist(),
                reference_gt_absolute_delta=[0.]*5,gt_input_short_cells=float(gt[:,2:4].min()*scale/8),
                target_residual=[.0392207,.0392207,-.03]))
    return records,samples


def reference_initial():
    c.g.seed_all()
    head = c.g.LocalGeometryRefiner()
    return head.state_dict(), c.g.state_digest(head)


def test_only_capacity_changes_and_common_initial_units_are_identical():
    reference, digest = reference_initial()
    state, evidence = c.checked_initialization(dict(initial_state=digest))
    head = c.CompactLocalGeometryRefiner(); head.load_state_dict(state)
    assert head.forward.__func__ is c.g.LocalGeometryRefiner.forward
    assert evidence['common_stem_exact'] and evidence['output_zero']
    assert sum(p.numel() for p in head.parameters()) == 59107
    assert c.g.SETTINGS['hidden_fc'] == 64 and c.HIDDEN_FC == 16
    for key in reference:
        expected = reference[key]
        if key.startswith('hidden.0.'):
            expected = expected[:16]
        elif key == 'output.weight':
            expected = expected[:,:16]
        assert torch.equal(state[key],expected)
        assert state[key].data_ptr() != reference[key].data_ptr()
    state2, evidence2 = c.checked_initialization(dict(initial_state=digest))
    assert evidence2 == evidence and all(torch.equal(state2[k],v) for k,v in state.items())


@pytest.mark.parametrize('bad', ['keys','width','output','nonfinite','digest'])
def test_incorrect_or_learned_initialization_is_rejected(bad):
    state, digest = reference_initial()
    state = deepcopy(state)
    if bad == 'keys':
        del state['stem.0.weight']
    elif bad == 'width':
        state['hidden.0.bias'] = state['hidden.0.bias'][:32]
    elif bad == 'output':
        state['output.weight'][0,0]=.01
    elif bad == 'nonfinite':
        state['stem.0.weight'][0,0,0,0]=float('nan')
    else:
        digest['parameters']='different'
        with pytest.raises(ValueError,match='initialization differs'):
            c.checked_initialization(dict(initial_state=digest))
        return
    with pytest.raises(ValueError):
        c.compact_initial_state(state)


def test_fit_body_matches_original_except_for_head_factory():
    # Protect optimizer, loss, coordinate and gradient semantics against drift.
    class Normalize(ast.NodeTransformer):
        def visit_Attribute(self,node):
            node=self.generic_visit(node)
            if isinstance(node.value,ast.Name) and node.value.id=='g':
                return ast.copy_location(ast.Name(id=node.attr,ctx=node.ctx),node)
            return node
        def visit_Name(self,node):
            if node.id=='CompactLocalGeometryRefiner':
                node.id='LocalGeometryRefiner'
            return node
    def body(function):
        tree=Normalize().visit(ast.parse(inspect.getsource(function)))
        tree.body[0].name='fit'
        return ast.dump(tree,include_attributes=False)
    assert body(c.fit_compact_arm)==body(c.g.fit_arm)


def test_actual_two_step_compact_fit_preserves_b_and_has_effective_gradients(tmp_path):
    records,_=synthetic()
    _,digest=reference_initial()
    initial,evidence=c.checked_initialization(dict(initial_state=digest))
    before=[r['boxes_original'].clone() for r in records]
    progress=[]
    result=c.fit_compact_arm(records,c.g.schedule(records)[:2],initial,'ordinary',torch.device('cpu'),progress.append)
    assert result['initial_state']==evidence['fc16_initial_state']
    assert result['initial_all_task_gradients_effective'] and result['stem_gradient_after_zero_output_step_effective']
    assert result['logs'][0]['stem_grad_norm']==0 and result['logs'][1]['stem_grad_norm']>0
    assert len([v for v in progress if v['stage']=='update'])==2
    for b,row in zip(before,result['final']['rows']):
        assert [row['pred'][i] for i in (0,1,5)]==b[0,[0,1,5]].tolist()
    assert all(torch.equal(r['boxes_original'],b) for r,b in zip(records,before))
    path=tmp_path/'result.json'; c.g.write_new(path,result)
    assert json.loads(path.read_text())==result


def test_compact_online_inputs_and_offline_gt_receive_no_gradients():
    _,digest=reference_initial()
    state,_=c.checked_initialization(dict(initial_state=digest))
    head=c.CompactLocalGeometryRefiner(); head.load_state_dict(state)
    roi=torch.randn(1,256,9,9,requires_grad=True)
    support=torch.ones(1,1,9,9,requires_grad=True)
    b=torch.tensor([[32.,24.,32.,16.,.17,.8]],requires_grad=True)
    bm=b[:,:5].detach().clone().requires_grad_()
    gt=b[:,:5].detach().clone(); gt[:,2:4]*=1.04; gt[:,4]-=.03; gt.requires_grad_()
    output=head(roi,support,b,bm)['boxes_original']
    assert torch.equal(output,b)
    loss,_=c.g.regression_loss(output,gt); loss.backward()
    assert all(x.grad is None for x in (roi,support,b,bm,gt))
    assert head.output.weight.grad.norm()>0


def test_source_change_fails_before_train_data_or_gpu(monkeypatch,tmp_path):
    manifest=json.loads(c.MANIFEST.read_text())
    manifest['sources']['crane_project/utils/port_geometry_refine_g_capacity_v1.py']='0'*64
    path=tmp_path/'sources.json'; path.write_text(json.dumps(manifest))
    monkeypatch.setattr(c,'MANIFEST',path)
    with pytest.raises(ValueError,match='source SHA differs'):
        c.checked_contract(Path('/missing.json'))


def test_static_cli_no_cache_or_gpu_and_no_overwrite(monkeypatch,tmp_path):
    samples=json.loads(c.g.FIXTURE.read_text())['samples']
    protocol=c.protocol_document(samples)
    monkeypatch.setattr(c,'checked_contract',lambda _:(protocol,samples,{},{}))
    def forbidden(*_):
        raise AssertionError('Static mode must not load cache, create heads or use CUDA')
    monkeypatch.setattr(c.g,'checked_cache',forbidden)
    monkeypatch.setattr(c,'checked_initialization',forbidden)
    monkeypatch.setattr(c.torch.cuda,'is_available',forbidden)
    out=tmp_path/'static'
    monkeypatch.setattr(c.sys,'argv',['capacity','--check-only','--out-dir',str(out)])
    c.main()
    report=json.loads((out/'completion.json').read_text())
    assert report['status']=='STATIC_CAPACITY_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES'
    assert report['head_updates_total']==report['additional_reference_updates']==report['detector_forward_calls']==0
    assert report['gpu_devices_used']==0
    artifacts=json.loads((out/'artifacts.json').read_text())
    assert artifacts['protocol']==c.VERSION
    assert all(c.g.ready.sha(out/p)==h for p,h in artifacts['files'].items())
    with pytest.raises(FileExistsError):
        c.main()


def test_serialization_failure_leaves_valid_failed_report(monkeypatch,tmp_path):
    protocol=c.protocol_document(json.loads(c.g.FIXTURE.read_text())['samples'])
    monkeypatch.setattr(c,'checked_contract',lambda _:(protocol,[],dict(bad=float('nan')),{}))
    out=tmp_path/'failed'
    monkeypatch.setattr(c.sys,'argv',['capacity','--check-only','--out-dir',str(out)])
    with pytest.raises(ValueError,match='JSON compliant'):
        c.main()
    report=json.loads((out/'completion.json').read_text())
    assert report['protocol']==c.VERSION and report['status']=='FAILED_REVIEW_REQUIRED'
    assert not report['full_report_available'] and report['head_updates_total']==0


@pytest.mark.parametrize('changed_start',[False,True])
def test_complete_entry_cpu_simulation_and_saved_arm_on_failed_comparison(monkeypatch,tmp_path,changed_start):
    # Real two CPU updates, with CUDA orchestration and scope budget mocked.
    # This does not claim a real cached TRAIN/GPU run.
    records,samples=synthetic()
    monkeypatch.setattr(c,'STEPS_PER_ARM',2)
    monkeypatch.setitem(c.g.SETTINGS,'steps_per_arm',2)
    wide_initial,digest=reference_initial()
    batches=c.g.schedule(records)
    reference=c.g.fit_arm(records,batches,wide_initial,'ordinary',torch.device('cpu'),lambda _:None)
    if changed_start:
        reference['initial']['eligible_loss']['probe/real/1.0']+=.001
    cache=dict(identity={},status='COMPLETE',record_count=len(records))
    baseline=dict(arms=dict(ordinary=reference),cache=cache,minibatches=batches)
    protocol=c.protocol_document(samples)
    monkeypatch.setattr(c,'checked_contract',lambda _:(protocol,samples,dict(prior_identity=dict(g_identity={})),baseline))
    monkeypatch.setattr(c.a,'checked_runtime',lambda _:dict(evidence='CPU simulation only'))
    monkeypatch.setattr(c.g,'checked_cache',lambda *_:(dict(records=records),cache))
    monkeypatch.setattr(c.g,'validate_records',lambda *_:None)
    monkeypatch.setattr(c.g,'measured',lambda operation,_:(operation(),dict(evidence='CPU simulation only')))
    original_fit=c.fit_compact_arm
    monkeypatch.setattr(c,'fit_compact_arm',lambda rec,bat,ini,arm,_dev,prog:
                        original_fit(rec,bat,ini,arm,torch.device('cpu'),prog))
    for key,value in [('is_available',lambda:True),('device_count',lambda:1),('set_device',lambda _:None),
                      ('manual_seed_all',lambda _:None),('get_device_name',lambda _:'CPU simulation only')]:
        monkeypatch.setattr(c.torch.cuda,key,value)
    cache_dir=tmp_path/'cache'; cache_dir.mkdir()
    c.g.write_new(cache_dir/'cache_manifest.json',cache)
    monkeypatch.setattr(c.a,'CACHE_MANIFEST_SHA',c.g.ready.sha(cache_dir/'cache_manifest.json'))
    out=tmp_path/'run'
    monkeypatch.setattr(c.sys,'argv',['capacity','--cache-dir',str(cache_dir),'--out-dir',str(out)])
    if changed_start:
        with pytest.raises(ValueError,match='initial B'):
            c.main()
    else:
        c.main()
    report=json.loads((out/'completion.json').read_text())
    assert report['head_updates_total']==2 and report['additional_reference_updates']==0
    assert report['detector_updates']==report['detector_forward_calls']==0
    assert report['status']==('FAILED_REVIEW_REQUIRED' if changed_start else 'TRAIN_CAPACITY_CHECK_COMPLETE_REVIEW_REQUIRED')
    assert (out/'compact_fc16.json').exists()
    assert not any(p.suffix=='.pth' for p in out.iterdir())
    assert not (out/'previews').exists()
    artifact=json.loads((out/'artifacts.json').read_text())
    assert all(c.g.ready.sha(out/p)==h for p,h in artifact['files'].items())
