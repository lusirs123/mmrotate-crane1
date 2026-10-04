"""Fixed sigma identity, cached-to-online VAL gate, and native frozen capture.

All weights/frames here are synthetic CPU fixtures, not performance evidence.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from crane_project.tools import eval_port_geometry_midpoint_sigma15_v1 as s
from test_port_geometry_midpoint_v1_test import FakeDetector


def write(path,value):
    path.write_text(json.dumps(value,allow_nan=False))


def index(directory,protocol):
    write(directory/'artifacts.json',dict(protocol=protocol,
        files={p.name:s.t.sha(p) for p in directory.iterdir() if p.is_file() and p.name!='artifacts.json'}))


def fixture_rows():
    rows = []
    for sequence,count in s.g.ready.VAL_COUNTS.items():
        for frame in range(count):
            source = dict(image='%s_%05d'%(sequence,frame),sequence=sequence,frame_id=frame,
                domain=sequence.split('_')[0],gt=[64.,48.,32.,16.,.2])
            box = np.asarray(source['gt']+[.8],dtype=np.float32).tolist()
            rows.append(s.e.prediction_row(source,dict(b=box,midpoint=box,candidate=box,
                candidate_valid=True,accepted=True,failed_checks=[])))
    return rows


@pytest.fixture
def training(tmp_path,monkeypatch):
    root = tmp_path/'training'; arm = root/'sigma_1p5'; arm.mkdir(parents=True)
    identity = s.checked_sources(); rows = fixture_rows(); value = s.f.summaries(rows)
    runtime = {'synthetic_cpu_fixture':True}
    proof = dict(identity=identity['training_identity'],cache_manifest_sha256=s.CACHE_SHA,
        baseline_files={'synthetic':True},baseline_selected={'synthetic':True})
    entries = {}; epochs = {}
    for epoch in range(1,25):
        name = 'head_epoch_%02d.pth'%epoch
        # Static audit must never try to deserialize these text fixture weights.
        (arm/name).write_bytes(('synthetic unopened epoch%02d'%epoch).encode())
        ckpt = deepcopy(s.FIXED); ckpt.update(path=name,epoch=epoch,updates=epoch*902,
            sha256=s.t.sha(arm/name))
        summary = deepcopy(value)
        if epoch != 3:
            summary['groups']['overall']['midpoint']['metric_protocol_v2']['sim/A-RMSE(deg)'] = 10.
        if epoch == 3:
            ckpt = deepcopy(s.FIXED)
        entries[name] = dict(checkpoint=ckpt,
            metrics=summary['groups']['overall']['midpoint']['metric_protocol_v2'])
        epochs[name] = summary
        write(arm/('val_epoch_%02d.json'%epoch),summary)
    real_sha = s.t.sha
    def fixture_sha(path):
        if Path(path).resolve() == (arm/s.FIXED['path']).resolve():
            # Explicitly mock the missing real server weight bytes, not other files.
            return s.FIXED['sha256']
        return real_sha(path)
    monkeypatch.setattr(s.t,'sha',fixture_sha)
    chosen,_,info = s.f.select_best_checkpoint(entries,s.f.SELECTION_CONFIG)
    assert chosen == s.FIXED['path']
    selection = dict(protocol=s.t.VERSION,sigma_cells=1.5,proof=proof,split='val',
        cache_manifest_sha256=s.CACHE_SHA,selected_checkpoint=deepcopy(s.FIXED),
        selection_config=s.f.SELECTION_CONFIG,selection_info=info,all_checkpoints=entries,
        test_access=False,selection_on_test=False,automatic_promotion=False)
    schedule = dict(epoch_steps=[902]*24,schedule_sha256='synthetic')
    complete = dict(protocol=s.t.VERSION,status='SIGMA_FULL_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED',
        sigma_cells=1.5,epochs_completed=24,updates=21648,proof=proof,selection=selection,
        detector_updates=0,test_access=False,schedule=schedule)
    grid = dict(protocol=s.t.VERSION,stage='train',status='SIGMA_FULL_GRID_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED',
        proof=proof,head_updates_total=43296,reused_updates=21648,represented_updates=64944,
        detector_updates=0,detector_forward_calls=0,feature_extractions=0,test_access=False,
        selection_on_test=False,selected_sigma=None,schedule=schedule,runtime=runtime,
        default_replay=dict(exact_cached_val_replay=True),
        conditions={key:dict(updates=21648,epochs=24,selected_checkpoint=deepcopy(s.FIXED))
            for key in ('sigma_0p5','sigma_1p5')})
    write(arm/'selection.json',selection); write(arm/'completion.json',complete)
    write(arm/'selected_val_compare.json',value)
    (arm/'val_epoch_03.rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    write(root/'completion.json',grid); write(root/'protocol.json',s.t.protocol_document())
    write(root/'sources.json',s.t.read(s.t.SOURCES))
    write(root/'val_sigma_compare.json',dict(protocol=s.t.VERSION,proof=proof,split='val',test_access=False,
        selected_epochs={'sigma_1p5':3},all_epoch_values={'sigma_1p5':epochs},
        groups={'sigma_1p5':value['groups']}))
    index(root,s.t.VERSION); index(arm,s.t.VERSION)
    return arm,identity,selection,rows,value


def val_fixture(directory,identity,proof,rows,value):
    directory.mkdir()
    state = dict(b={'synthetic_frozen_b':True},head=deepcopy(s.FIXED['head_digest']))
    report = dict(protocol=s.VERSION,split='val',status='FROZEN_SIGMA15_VAL_COMPLETE_REVIEW_REQUIRED',
        identity=identity,selection=proof,head_updates=0,detector_updates=0,test_access=False,
        selection_on_test=False,runtime=proof['training_runtime'],frames=887,
        feature_extractions=887,native_head_calls=2661,state_before=state,state_after=deepcopy(state),
        val_replay=dict(frames=887,selected_epoch_box_decision_replay_pass=True))
    write(directory/'completion.json',report); write(directory/'protocol.json',s.protocol_document())
    write(directory/'sources.json',s.t.read(s.SOURCES)); write(directory/'val_compare.json',value)
    online_rows=[dict(r,original_b_raw_exact_before_after=True) for r in rows]
    (directory/'val_rows.jsonl').write_text('\n'.join(json.dumps(r) for r in online_rows)+'\n')
    (directory/'progress.jsonl').write_text('synthetic CPU fixture\n')
    index(directory,s.VERSION)
    return directory/'completion.json'


def test_sources_fixed_candidate_and_original_contracts_unchanged():
    proof = s.checked_sources()
    assert len(proof['sources']) == 82
    assert len(s.t.checked_sources()['sources']) == 74
    assert len(s.e.checked_sources()['sources']) == 73
    assert s.protocol_document()['fixed_checkpoint']['epoch'] == 3
    assert s.protocol_document()['scope']['selection_on_test'] is False
    assert '--sigma' not in s.parser()._option_string_actions
    assert '--epoch' not in s.parser()._option_string_actions


def test_all24_choice_and_complete_budget_verified_without_tensor_load(training,monkeypatch):
    arm,identity,selection,_,_ = training
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Static tensor load'))
    actual,proof = s.checked_selection(arm/'selection.json',identity)
    assert actual == selection and proof['selected_checkpoint'] == s.FIXED


@pytest.mark.parametrize('kind',['sigma','epoch','budget','source','metrics','rows','index'])
def test_training_metadata_or_artifact_mismatch_rejected(training,kind):
    arm,identity,_,_,_ = training
    path = arm/'selection.json'; value = s.t.read(path)
    if kind=='sigma': value['sigma_cells']=1.
    elif kind=='epoch': value['selected_checkpoint']['epoch']=23
    elif kind=='source': value['proof']['identity']={}
    elif kind=='budget':
        path=arm/'completion.json';value=s.t.read(path);value['updates']=2706
    elif kind=='metrics':
        path=arm/'val_epoch_04.json';value=s.t.read(path)
        value['groups']['overall']['midpoint']['metric_protocol_v2']['sim/ACI']=0.
    elif kind=='rows':
        (arm/'val_epoch_03.rows.jsonl').write_text('{}\n')
    elif kind=='index':
        path=arm/'artifacts.json';value=s.t.read(path);value['files']['head_epoch_03.pth']='wrong'
    if kind!='rows':write(path,value)
    with pytest.raises((ValueError,KeyError)):
        s.checked_selection(arm/'selection.json',identity)


def test_static_val_never_touches_gpu_test_images_or_tensor_cache(training,tmp_path,monkeypatch):
    arm,_,_,_,_ = training
    def forbidden(*a,**k):pytest.fail('Static evaluation crossed runtime/data boundary')
    for obj,key in ((s.f,'runtime'),(s.old,'fixed_test_inputs'),(s.e,'val_inputs'),(torch,'load')):
        monkeypatch.setattr(obj,key,forbidden)
    args=s.parser().parse_args(['--split','val','--check-only','--selection',str(arm/'selection.json'),
        '--out-dir',str(tmp_path/'static')])
    s.run(args); report=s.t.read(tmp_path/'static/completion.json')
    assert report['status']=='STATIC_SIGMA15_EVAL_PASS_NO_DATA_GPU_UPDATES'
    assert report['test_access'] is False and report['head_updates']==0
    with pytest.raises(FileExistsError,match='preserve'):
        s.run(args)


@pytest.mark.parametrize('static',[True,False])
def test_test_requires_val_gate_before_runtime_or_test_access(training,tmp_path,monkeypatch,static):
    arm,_,_,_,_ = training
    def forbidden(*a,**k):pytest.fail('TEST/runtime accessed before VAL gate')
    monkeypatch.setattr(s.f,'runtime',forbidden);monkeypatch.setattr(s.old,'fixed_test_inputs',forbidden)
    args=s.parser().parse_args(['--split','test','--selection',str(arm/'selection.json'),
        '--out-dir',str(tmp_path/'missing_gate')]+(['--check-only'] if static else []))
    with pytest.raises(ValueError,match='requires --val-report'):
        s.run(args)
    report=s.t.read(tmp_path/'missing_gate/failure.json')
    assert report['test_access'] is False and report['feature_extractions']==0
    assert not (tmp_path/'missing_gate/completion.json').exists()


def test_full_val_gate_then_static_test_without_actual_test_inputs(training,tmp_path,monkeypatch):
    arm,identity,_,rows,value=training
    _,proof=s.checked_selection(arm/'selection.json',identity)
    val=val_fixture(tmp_path/'val',identity,proof,rows,value)
    assert s.checked_val_report(val,identity,proof,arm/'selection.json')['pass_online_val']
    monkeypatch.setattr(s.old,'fixed_test_inputs',lambda:pytest.fail('Static TEST input access'))
    args=s.parser().parse_args(['--split','test','--check-only','--selection',str(arm/'selection.json'),
        '--val-report',str(val),'--out-dir',str(tmp_path/'test_static')])
    s.run(args)
    assert s.t.read(tmp_path/'test_static/completion.json')['test_access'] is False


@pytest.mark.parametrize('kind',['state','calls','sigma','row_score','summary','byte_tamper'])
def test_stale_or_inconsistent_val_gate_rejected_even_if_reindexed(training,tmp_path,kind):
    arm,identity,_,rows,value=training
    _,proof=s.checked_selection(arm/'selection.json',identity)
    path=val_fixture(tmp_path/'val',identity,proof,rows,value);value=s.t.read(path)
    if kind=='state':value['state_after']['head']={}
    elif kind=='calls':value['native_head_calls']=887
    elif kind=='sigma':value['selection']['selected_checkpoint']['sigma_cells']=1.
    elif kind=='row_score':
        rows[0]['midpoint'][5]=.1
        online_rows=[dict(r,original_b_raw_exact_before_after=True) for r in rows]
        (path.parent/'val_rows.jsonl').write_text('\n'.join(json.dumps(r) for r in online_rows)+'\n')
    elif kind=='summary':
        summary=s.t.read(path.parent/'val_compare.json');summary['split']='test'
        write(path.parent/'val_compare.json',summary)
    else:value['frames']=1
    write(path,value)
    if kind!='byte_tamper':index(path.parent,s.VERSION)
    with pytest.raises((ValueError,KeyError)):
        s.checked_val_report(path,identity,proof,arm/'selection.json')


@pytest.mark.parametrize('empty',[False,True])
@pytest.mark.parametrize('swap',[False,True])
def test_sigma15_native_capture_scale_restoration_score_count_and_frozen_states(empty,swap):
    b=[] if empty else [[64.,48.,16. if swap else 32.,32. if swap else 16.,.2,.8]]
    scales=np.array([.6201,.6204,.6201,.6204],dtype=np.float32)
    meta=dict(ori_shape=(721,1031,3),img_shape=(448,640,3),pad_shape=(1024,1024,3),
        scale_factor=scales,flip=False,filename='no_gt_online.jpg')
    detector=FakeDetector(b,scales[:2]);head=s.t.model.SigmaMidpointHead(1.5).eval().requires_grad_(False)
    before=dict(b=s.g.state_digest(detector),head=s.g.state_digest(head));events=[]
    result=s.old.capture(detector,head,torch.zeros(1,3,1024,1024),[meta],events.append)
    assert before==dict(b=s.g.state_digest(detector),head=s.g.state_digest(head))
    assert sum(v['stage']=='test_feature_extracted' for v in events)==1
    assert sum(v['stage']=='test_native_head_called' for v in events)==3
    assert result['b']==result['midpoint']
    assert result['b'] is None if empty else result['accepted']


def test_strict_sigma_loader_uses_explicit_head_not_default_loader(tmp_path,monkeypatch):
    detector=FakeDetector([[64.,48.,32.,16.,.2,.8]],[.62,.62])
    torch.manual_seed(1703);head=s.t.model.SigmaMidpointHead(1.5)
    with torch.no_grad():head.output.weight[0,0,0,0]=.1
    fixed=deepcopy(s.FIXED);fixed['head_digest']=s.g.state_digest(head)
    monkeypatch.setattr(s,'FIXED',fixed)
    proof=dict(training_proof={'synthetic':True},training_runtime={'synthetic':True})
    context=dict(manifest_sha256=s.CACHE_SHA,runtime=proof['training_runtime'],
        detector_state=s.g.state_digest(detector),data_identity={'synthetic':True})
    payload=dict(protocol=s.t.VERSION,proof=proof['training_proof'],sigma_cells=1.5,epoch=3,updates=2706,
        head_digest=fixed['head_digest'],frozen_b=s.g.ready.FROZEN_B,context=context,head_state=head.state_dict())
    path=tmp_path/fixed['path'];torch.save(payload,path)
    fixed['sha256']=s.t.sha(path)
    monkeypatch.setattr(s.g,'check_cfg',lambda:{'synthetic':True})
    monkeypatch.setattr(s.f,'build_detector',lambda *a:detector)
    monkeypatch.setattr(s.f,'load_selected_pipeline',lambda *a:pytest.fail('Default sigma1 loader'))
    _,loaded,_,restored=s.load_pipeline(tmp_path/'selection.json',proof,0)
    assert loaded.prior_sigma_cells==1.5 and restored==context
    assert s.g.state_digest(loaded)==fixed['head_digest']
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    # A valid state dict alone cannot identify sigma: reject wrong explicit metadata.
    payload['sigma_cells']=1.;torch.save(payload,path);fixed['sha256']=s.t.sha(path)
    with pytest.raises(ValueError,match='payload/context'):
        s.load_pipeline(tmp_path/'selection.json',proof,0)


def test_actual_val_entry_full_count_capture_summary_and_followup_gate(training,tmp_path,monkeypatch):
    arm,identity,selection,rows,_=training
    _,proof=s.checked_selection(arm/'selection.json',identity)
    detector=FakeDetector([[64.,48.,32.,16.,.2,.8]],[.6201,.6204])
    head=s.t.model.SigmaMidpointHead(1.5).eval().requires_grad_(False)
    # CPU synthetic run bypasses only server environment/checkpoint/data loading.
    # Executes the actual native capture, 887-row loop, summary, replay and gate.
    fixed=deepcopy(s.FIXED);fixed['head_digest']=s.g.state_digest(head)
    monkeypatch.setattr(s,'FIXED',fixed)
    proof['selected_checkpoint']=fixed
    # checked_sources is separately covered against the real immutable manifest.
    monkeypatch.setattr(s,'checked_sources',lambda:identity)
    monkeypatch.setattr(s,'checked_selection',lambda *a:(selection,proof))
    runtime={'synthetic_cpu_fixture':True};data=dict(frames=887)
    monkeypatch.setattr(s.f,'runtime',lambda gpu:runtime)
    monkeypatch.setattr(s,'load_pipeline',lambda *a:(detector,head,{},dict(data_identity={'val':data})))
    monkeypatch.setattr(s.e,'val_inputs',lambda:(rows,data))
    monkeypatch.setattr(s.f,'dataset_for',lambda *a:None)
    monkeypatch.setattr(s.f,'infos',lambda *a:[dict(filename=r['image']+'.jpg') for r in rows])
    meta=dict(ori_shape=(721,1031,3),img_shape=(448,640,3),pad_shape=(1024,1024,3),
        scale_factor=np.asarray([.6201,.6204,.6201,.6204],dtype=np.float32),flip=False)
    monkeypatch.setattr(s.old,'load_view',lambda *a:(torch.zeros(1,3,8,8),[meta]))
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *a:0)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda *a:0)
    monkeypatch.setattr(s.old,'fixed_test_inputs',lambda:pytest.fail('VAL accessed TEST'))
    args=s.parser().parse_args(['--split','val','--selection',str(arm/'selection.json'),
        '--out-dir',str(tmp_path/'online_val')])
    s.run(args)
    report=s.t.read(tmp_path/'online_val/completion.json')
    assert report['status']=='FROZEN_SIGMA15_VAL_COMPLETE_REVIEW_REQUIRED'
    assert report['feature_extractions']==887 and report['native_head_calls']==2661
    assert report['state_before']==report['state_after'] and not report['test_access']
    assert s.checked_val_report(tmp_path/'online_val/completion.json',identity,
        report['selection'],arm/'selection.json')['pass_online_val']
