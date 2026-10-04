"""Original sigma sensitivity: real CPU math/IO and synthetic static contracts.

Synthetic data below are fixtures, never evidence of server performance.
"""
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from crane_project.tools import train_port_geometry_midpoint_sigma_v1 as t

f,m = t.f,t.model
spec=importlib.util.spec_from_file_location('formal_fixtures',
    t.ROOT/'tests/test_port_geometry_midpoint_formal_v1.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.mark.parametrize('sigma',[.5,1.,1.5])
@pytest.mark.parametrize('swapped',[False,True])
def test_zero_identity_with_raw_axes_odd_resize_empty_and_effective_gradients(sigma,swapped):
    row=helpers.row('real_seq01_00000',scale=.5)
    b=row['boxes_original'];b[:,4]=math.pi/2-1e-6
    if swapped:b[:,[2,3]]=b[:,[3,2]]
    xy=b.new_tensor([[.497,.499]]);bm=b[:,:5].clone();bm[:,:4]*=xy[:,[0,1,0,1]]
    roi=row['roi'].requires_grad_();support=row['support'].requires_grad_()
    head=m.SigmaMidpointHead(sigma);out=head(roi,support,b,bm,xy)
    assert torch.equal(out['boxes_original'],b) and out['accepted'].all()
    assert torch.equal(out['point_delta_roi'],torch.zeros_like(out['point_delta_roi']))
    assert f.g.state_digest(head)['parameter_count']==17696
    assert head(roi[:0],support[:0],b[:0],bm[:0],xy[:0])['boxes_original'].shape==(0,6)
    gt=b[:,:5].clone();gt[:,:2]+=.4;gt[:,2:4]*=1.03;gt[:,4]+=.015
    opt=f.optimizer_for(head)
    for step in range(2):
        opt.zero_grad();out=head(roi,support,b,bm,xy)
        loss,_=f.m.point_loss(out['points_roi'],f.m.target_points(gt,b,bm,xy));loss.backward()
        assert torch.isfinite(loss) and head.output.weight.grad.norm()>0
        stem=sum(float(p.grad.square().sum()) for p in head.stem.parameters())
        assert stem==0 if step==0 else stem>0
        opt.step()
    assert roi.grad is None and support.grad is None


def test_sigma1_exact_default_forward_gradients_and_two_update_trajectory():
    before=deepcopy(f.m.SETTINGS);row=helpers.row('sim_seq08_00000')
    f.seed_all();old=f.m.SpatialMidpointHead()
    f.seed_all();new=m.SigmaMidpointHead(1.)
    assert f.g.state_digest(old)==f.g.state_digest(new)
    args=[row[k] for k in ('roi','support','boxes_original','boxes_model','scale_xy')]
    opt_a=f.optimizer_for(old);opt_b=f.optimizer_for(new)
    for _ in range(2):
        assert f.tree_equal(old(*args),new(*args))
        a=f.update(old,opt_a,[row],[0],'cpu');b=f.update(new,opt_b,[row],[0],'cpu')
        assert a==b and f.g.state_digest(old)==f.g.state_digest(new)
        assert f.tree_equal(opt_a.state_dict(),opt_b.state_dict())
    assert f.m.SETTINGS==before


def test_only_prior_width_changes_and_neutral_is_recomputed():
    ref=torch.tensor([[[.33,0.],[0.,.33],[-.33,0.],[0.,-.33]]])
    values=[m.point_prior(ref,s) for s in m.SIGMAS]
    assert torch.equal(values[0][0],values[2][0])
    assert not torch.equal(values[0][1],values[2][1])
    assert not torch.equal(values[0][2],values[2][2])
    for grid,prior,neutral in values:
        assert torch.equal(neutral,torch.matmul(prior.softmax(-1),grid))
    for invalid in (True,0.,2.,float('nan'),'1'):
        with pytest.raises(ValueError):m.SigmaMidpointHead(invalid)


@pytest.mark.parametrize('after_step',[False,True])
def test_failed_update_accounting_preserves_completed_step_and_restores_optimizer(monkeypatch,after_step):
    head=m.SigmaMidpointHead(.5);optimizer=f.optimizer_for(head)
    original=optimizer.step;events=[]
    def fail(*args):
        if after_step:optimizer.step()
        raise ValueError('synthetic failure')
    monkeypatch.setattr(f,'update',fail)
    with pytest.raises(ValueError,match='synthetic failure'):
        t.tracked_update(head,optimizer,[],[],'cpu',events.append,dict(sigma_cells=.5))
    assert len(events)==int(after_step) and optimizer.step==original
    if events:assert events[0]['stage']=='optimizer_step'


def test_sigma_checkpoint_binds_parameter_sources_and_exact_reload_no_overwrite(tmp_path):
    train,val=helpers.fixture();proof={'synthetic':True};cache={'manifest_sha256':'synthetic'}
    head=m.SigmaMidpointHead(.5);opt=f.optimizer_for(head)
    f.update(head,opt,train,f.epoch_batches(train,1)[0],'cpu')
    p=tmp_path/'head.pth';checkpoint=t.save_checkpoint(p,head,opt,1,1,proof,cache)
    new,payload=t.load_head(p,.5,proof,cache,'cpu')
    assert f.g.state_digest(new)==f.g.state_digest(head) and checkpoint['save_reload_exact']
    assert f.evaluate(head,val,'cpu')==f.evaluate(new,val,'cpu')
    for sigma,identity in ((1.5,proof),(.5,{'wrong':True})):
        with pytest.raises(ValueError,match='parameter/source/cache'):t.load_head(p,sigma,identity,cache,'cpu')
    before=p.read_bytes()
    with pytest.raises(FileExistsError):t.save_checkpoint(p,head,opt,2,2,proof,cache)
    assert p.read_bytes()==before


@pytest.mark.parametrize('mismatch',[False,True])
def test_existing_default_checkpoint_replay_requires_exact_saved_val(tmp_path,mismatch):
    train,val=helpers.fixture();f.seed_all();head=f.m.SpatialMidpointHead()
    optimizer=f.optimizer_for(head);f.update(head,optimizer,train,f.epoch_batches(train,1)[0],'cpu')
    parent={'synthetic':True};cache={'manifest_sha256':'synthetic'}
    checkpoint=f.save_checkpoint(tmp_path/'head_epoch_23.pth',head,optimizer,23,1,parent,f.checkpoint_context(cache))
    rows=f.evaluate(head,val,'cpu');value=f.summaries(rows)
    baseline=dict(selection={'selected_checkpoint':checkpoint},rows=deepcopy(rows),value=value)
    if mismatch:baseline['rows'][0]['accepted']=not rows[0]['accepted']
    if mismatch:
        with pytest.raises(ValueError,match='full-VAL replay differs'):
            t.replay_default(tmp_path,val,baseline,parent,cache,'cpu')
    else:
        assert t.replay_default(tmp_path,val,baseline,parent,cache,'cpu')['exact_cached_val_replay']


@pytest.mark.parametrize('sigma',[.5,1.5])
def test_discarded_smoke_reloads_inference_rng_optimizer_and_counts_four_updates(tmp_path,sigma):
    train,val=helpers.fixture();f.seed_all();head=f.m.SpatialMidpointHead()
    events=[];proof={'synthetic':True};cache={'manifest_sha256':'synthetic'}
    result=t.smoke_condition(sigma,train,val,deepcopy(head.state_dict()),proof,cache,'cpu',tmp_path,events.append)
    assert len(events)==4 and result['updates_executed']==4 and result['discarded']
    assert result['inference_reload_exact'] and result['optimizer_continuation_exact']
    assert all(z['sigma_cells']==sigma for z in events)


def row_document(seq,i):
    b=[64.,48.,32.,16.,.2,.8];gt=b[:5]
    return dict(image=seq+'_'+str(i).zfill(5),sequence=seq,frame_id=i,domain=seq.split('_')[0],
        scale=1.,gt=gt,b=b,midpoint=b,candidate=b,candidate_valid=True,accepted=True,failed_checks=[],
        metrics={name:f.g.decompose(gt,b[:5]) for name in ('b','midpoint','candidate')})


def contract_fixture(tmp_path):
    """Synthetic indexed24-epoch logs/VAL plus opaque unopened selected weight."""
    parent=f.checked_sources();baseline=tmp_path/'baseline';cache_dir=tmp_path/'cache'
    baseline.mkdir();cache_dir.mkdir()
    cache=dict(protocol=f.VERSION,identity=parent,status='COMPLETE_FROZEN_B_TRAIN_VAL_CACHE',
        record_counts={'train_s1':2558,'train_s05':2558,'val_s1':887},
        files={n:'synthetic' for n in ('train_s1.pt','train_s05.pt','val_s1.pt')},
        feature_extractions=6003,native_head_calls=18009,detector_updates=0,test_access=False,
        tensor_bytes=1,runtime={'synthetic':True})
    t.write(cache_dir/'cache_manifest.json',cache);cache['manifest_sha256']=t.sha(cache_dir/'cache_manifest.json')
    rows=[row_document(seq,i) for seq,n in f.g.ready.VAL_COUNTS.items() for i in range(n)]
    value=f.summaries(rows);entries={};events=[]
    for epoch in range(1,25):
        name='head_epoch_%02d.pth'%epoch
        (baseline/name).write_bytes(b'synthetic unopened checkpoint')
        checkpoint=dict(path=name,sha256=t.sha(baseline/name),epoch=epoch,updates=epoch,
            save_reload_exact=True,head_digest={'synthetic':True})
        metrics=value['groups']['overall']['midpoint']['metric_protocol_v2']
        entries[name]=dict(checkpoint=checkpoint,metrics=metrics)
        t.write(baseline/('val_epoch_%02d.json'%epoch),value)
        events.extend([dict(stage='train',epoch=epoch,step=1,epoch_steps=1,updates=epoch,
            loss=.01,grad_norm_before_clip=.01,grad_norm_after_clip=.01,stem_grad_norm=.01,output_grad_norm=.01),
            dict(stage='epoch_complete',epoch=epoch,updates=epoch,checkpoint=checkpoint,selection_metrics=metrics)])
    chosen,_,info=f.select_best_checkpoint(entries,f.SELECTION_CONFIG)
    selected=entries[chosen]['checkpoint']
    selection=dict(protocol=f.VERSION,identity=parent,split='val',test_access=False,selection_on_test=False,
        selection_config=f.SELECTION_CONFIG,all_checkpoints=entries,selected_checkpoint=selected,
        selection_info=info,cache_manifest_sha256=cache['manifest_sha256'])
    complete=dict(protocol=f.VERSION,status='FORMAL_MIDPOINT_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED',
        identity=parent,epochs_completed=24,updates=24,detector_updates=0,test_access=False,
        runtime=cache['runtime'],selection=selection,cache_manifest_sha256=cache['manifest_sha256'])
    for name,doc in [('selection.json',selection),('completion.json',complete),
            ('selected_val_compare.json',value),('val_baseline.json',value)]:t.write(baseline/name,doc)
    (baseline/('val_epoch_%02d.rows.jsonl'%selected['epoch'])).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (baseline/'progress.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
    t.write(baseline/'artifacts.json',dict(protocol=f.VERSION,files={p.name:t.sha(p) for p in baseline.iterdir() if p.is_file()}))
    return baseline,cache_dir,parent,cache


def reindex(path):
    p=path/'artifacts.json';p.unlink()
    t.write(p,dict(protocol=f.VERSION,files={z.name:t.sha(z) for z in path.iterdir() if z.is_file()}))


@pytest.mark.parametrize('kind',['valid','sha','epoch_metrics','budget','runtime','selection','extra_update'])
def test_baseline_reuse_audit_and_rejection(tmp_path,kind):
    baseline,_,parent,cache=contract_fixture(tmp_path)
    if kind=='sha':(baseline/'val_epoch_02.json').write_text('{}')
    elif kind in ('budget','runtime'):
        c=t.read(baseline/'completion.json')
        c['updates' if kind=='budget' else 'runtime']=25 if kind=='budget' else {}
        (baseline/'completion.json').write_text(json.dumps(c));reindex(baseline)
    elif kind=='epoch_metrics':
        v=t.read(baseline/'val_epoch_02.json');v['groups']['overall']['midpoint']['metric_protocol_v2']['sim/ACI']=.2
        (baseline/'val_epoch_02.json').write_text(json.dumps(v));reindex(baseline)
    elif kind=='selection':
        v=t.read(baseline/'selection.json');v['selected_checkpoint']['epoch']=7
        (baseline/'selection.json').write_text(json.dumps(v));reindex(baseline)
    elif kind=='extra_update':
        with (baseline/'progress.jsonl').open('a') as stream:stream.write(json.dumps(dict(stage='train'))+'\n')
        reindex(baseline)
    if kind=='valid':assert t.audit_baseline(baseline,parent,cache)['epoch_steps']==[1]*24
    else:
        with pytest.raises((ValueError,KeyError)):t.audit_baseline(baseline,parent,cache)


@pytest.mark.parametrize('relative',[True,False])
def test_real_static_cli_no_tensor_gpu_and_no_overwrite(tmp_path,relative):
    baseline,cache,_,_=contract_fixture(tmp_path)
    script=Path(t.__file__).resolve();script=str(script.relative_to(t.ROOT)) if relative else str(script)
    cmd=[sys.executable,'-B',script,'--stage','check','--baseline-dir',str(baseline),
        '--cache-dir',str(cache),'--out-dir',str(tmp_path/'check')]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    p=subprocess.run(cmd,cwd=str(t.ROOT),env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    complete=t.read(tmp_path/'check/completion.json')
    assert complete['head_updates_total']==complete['cache_tensor_loads']==0 and not complete['test_access']
    assert complete['status']=='STATIC_SIGMA_CONTRACT_PASS_NO_TENSOR_LOAD_NO_GPU_NO_UPDATES'
    before=(tmp_path/'check/completion.json').read_bytes()
    p=subprocess.run(cmd,cwd=str(t.ROOT),env=env,capture_output=True,text=True)
    assert p.returncode!=0 and 'Output exists' in p.stderr
    assert (tmp_path/'check/completion.json').read_bytes()==before


def test_static_function_no_load_or_cuda_and_output_cannot_be_input(tmp_path,monkeypatch):
    baseline,cache,_,_=contract_fixture(tmp_path)
    def forbidden(*a,**k):raise AssertionError('Forbidden tensor/GPU/dataset access')
    monkeypatch.setattr(t.torch,'load',forbidden);monkeypatch.setattr(f,'runtime',forbidden)
    args=t.parser().parse_args(['--stage','check','--baseline-dir',str(baseline),'--cache-dir',str(cache),
        '--out-dir',str(tmp_path/'check')])
    assert t.run(args)['head_updates_total']==0
    args.out_dir=str(cache/'nested')
    with pytest.raises(ValueError,match='separate'):t.run(args)
    assert 'test' not in t.parser()._option_string_actions['--stage'].choices


def test_full24_epoch_cpu_training_keeps_selection_parameter_and_main_metric_denominators(tmp_path):
    train,val=helpers.fixture();f.seed_all();head=f.m.SpatialMidpointHead()
    initial=deepcopy(head.state_dict());initial_state=f.g.state_digest(head)
    schedules=[len(f.epoch_batches(train,e)) for e in range(1,25)]
    schedule=dict(epoch_steps=schedules,initial_state=initial_state)
    baseline={'value':f.summaries(f.evaluate(head,val,'cpu'))};cache={'manifest_sha256':'synthetic'}
    events=[];value=t.train_condition(.5,train,val,initial,{'synthetic':True},cache,
        baseline,schedule,'cpu',tmp_path,events.append)
    assert value['updates']==sum(schedules) and len(value['selection']['all_checkpoints'])==24
    assert value['selection']['sigma_cells']==.5 and value['selection']['selection_config']==f.SELECTION_CONFIG
    assert len([z for z in events if z['stage']=='train'])==value['updates']
    assert (tmp_path/'val_epoch_24.rows.jsonl').is_file()
    selected=value['selection']['selected_checkpoint'];head,payload=t.load_head(tmp_path/selected['path'],.5,{'synthetic':True},cache,'cpu')
    assert f.summaries(f.evaluate(head,val,'cpu'))==value['value']
    vals={s:deepcopy(value['value']) for s in m.SIGMAS}
    g=vals[1.]['groups']['real']['midpoint'];g['frames']=12;g['output_frames']=11
    g['output_coverage_pct']=100*11/12;g['output_center_hit_pct']=100*10/11;g['all_frame_center_hit_pct']=100*10/12
    g['conditional_center_correct_fraction']['numerator']=10
    selected_epochs={s:1 for s in m.SIGMAS}
    row=next(z for z in t.report_rows(vals,selected_epochs) if z['sigma_cells']==1. and z['domain']=='real')
    assert row['output_frame_r_center_pct']!=row['full_frame_center_correct_pct']
    assert len(t.report_rows(vals,selected_epochs))==8 and row['a_rmse_deg'] is None and row['scale']==1.
    b=next(z for z in t.report_rows(vals,selected_epochs) if z['method']=='b')
    assert b['sigma_cells'] is None and b['selected_epoch']==24
    assert len([z for z in events if z['stage']=='optimizer_step'])==value['updates']


def test_source_manifest_parent_immutable_and_only_sigma_variable():
    identity=t.checked_sources();assert len(identity['sources'])==74
    assert t.protocol_document()['formal_settings']==f.SETTINGS
    assert t.protocol_document()['scope']['new_epochs']==48
    assert not t.protocol_document()['scope']['feature_extractions']
    assert not t.protocol_document()['scope']['selection_on_test']
    assert 'size_weight' not in t.protocol_document()['formal_settings']
