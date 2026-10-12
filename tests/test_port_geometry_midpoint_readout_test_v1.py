"""CPU synthetic safety tests; no real TRAIN/VAL/TEST performance claims."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from crane_project.tools import eval_port_geometry_midpoint_readout_test_v1 as x
from test_port_geometry_midpoint_v1_test import FakeDetector


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.mark.parametrize('empty',[False,True])
@pytest.mark.parametrize('swap',[False,True])
def test_two_heads_share_one_extraction_three_native_calls_and_exact_b(empty,swap):
    b=[] if empty else [[64.,48.,16. if swap else 32.,32. if swap else 16.,.2,.8]]
    scales=np.array([.6201,.6204,.6201,.6204],dtype=np.float32)
    meta=dict(ori_shape=(721,1031,3),img_shape=(448,640,3),pad_shape=(1024,1024,3),
        scale_factor=scales,flip=False,filename='synthetic.jpg')
    detector=FakeDetector(b,scales[:2])
    heads={'uniform':x.r.model.UniformMidpointHead(),'gaussian':x.r.t.model.SigmaMidpointHead(1.5)}
    for h in heads.values():h.eval().requires_grad_(False)
    before=x.states(detector,heads);events=[]
    rows=x.capture(detector,heads,torch.zeros(1,3,1024,1024),[meta],events.append)
    assert events.count('feature_extracted')==1 and events.count('native_head_called')==3
    assert rows['uniform']['b']==rows['gaussian']['b']
    assert all(v['midpoint']==v['b'] for v in rows.values())
    assert x.states(detector,heads)==before
    assert 'gt' not in inspect.signature(x.capture).parameters
    assert 'source' not in inspect.signature(x.capture).parameters


def test_mutated_native_b_fails_without_returning_results():
    b=[[64.,48.,32.,16.,.2,.8]];scales=np.array([.62,.62,.62,.62],np.float32)
    meta=dict(ori_shape=(721,1031,3),img_shape=(448,640,3),pad_shape=(1024,1024,3),scale_factor=scales,flip=False)
    heads={'uniform':x.r.model.UniformMidpointHead(),'gaussian':x.r.t.model.SigmaMidpointHead(1.5)}
    for h in heads.values():h.eval().requires_grad_(False)
    with pytest.raises(ValueError,match='raw output'):
        x.capture(FakeDetector(b,scales[:2],mutate=True),heads,torch.zeros(1,3,1024,1024),[meta],lambda a:None)


def test_fixed_weights_no_epoch_sigma_override_and_no_training():
    p=x.parser()
    assert x.FIXED['epoch']==4 and x.e.FIXED['epoch']==3
    assert x.FIXED['updates']==3608
    for flag in ['--epoch','--sigma','--threshold','--lr']:
        with pytest.raises(SystemExit):p.parse_args(['--stage','test','--out-dir','synthetic',flag,'2'])
    assert 'optimizer' not in inspect.getsource(x.run)
    assert x.protocol_document()['scope']['selection_on_test'] is False
    assert x.protocol_document()['scope']['head_updates']==0


def test_source_closure_and_historical_detector_boundary_are_explicit():
    identity=x.checked_sources()
    assert set(x.r.DETECTOR_ONLY)<=set(identity['sources'])
    assert len(identity['sources'])==91
    assert 'claimed unused here' in x.protocol_document()['source_boundary']


def test_gate_required_before_test_access_and_output_creation(tmp_path,monkeypatch):
    args=x.parser().parse_args(['--stage','test','--out-dir',str(tmp_path/'result')])
    monkeypatch.setattr(x,'checked_sources',lambda:{})
    monkeypatch.setattr(x,'checked_training',lambda a,b:({},{},{},{}))
    def forbidden(*a,**k):raise AssertionError('TEST accessed before VAL gate')
    monkeypatch.setattr(x,'historical_test',forbidden)
    monkeypatch.setattr(x.f.old,'fixed_test_inputs',forbidden)
    monkeypatch.setattr(x.f,'runtime',forbidden)
    with pytest.raises(ValueError,match='requires successful paired'):
        x.run(args)
    assert not (tmp_path/'result').exists()


def test_static_has_no_runtime_tensor_dataset_or_test_access(tmp_path,monkeypatch):
    args=x.parser().parse_args(['--stage','check','--out-dir',str(tmp_path/'result')])
    monkeypatch.setattr(x,'checked_sources',lambda:{})
    monkeypatch.setattr(x,'checked_training',lambda a,b:({},{},{},{}))
    def forbidden(*a,**k):raise AssertionError('Static touched data/GPU/tensor')
    for module,name in [(x.f,'runtime'),(x.torch,'load'),(x,'historical_test'),(x,'load_pipeline'),(x.f.old,'fixed_test_inputs')]:
        monkeypatch.setattr(module,name,forbidden)
    result=x.run(args)
    assert result['status']=='STATIC_READOUT_EVAL_PASS_NO_TEST_DATA_TENSOR_GPU_UPDATES'
    assert result['test_access'] is False and result['head_updates']==0


def gate_fixture(tmp_path):
    identity={'source':'synthetic'};gate={'runtime':{'cudnn':8801}}
    report=dict(protocol=x.VERSION,stage='val',status=x.VAL_STATUS,identity=identity,gate=gate,
        frames=887,head_updates=0,detector_updates=0,test_access=False,selection_on_test=False,automatic_promotion=False,
        feature_extractions=887,native_head_calls=2661,state_before={'frozen':True},state_after={'frozen':True},
        runtime=gate['runtime'],val_replay=dict(uniform=True,gaussian=True))
    x.r.t.write(tmp_path/'completion.json',report);x.r.t.write(tmp_path/'protocol.json',x.protocol_document())
    x.r.t.write(tmp_path/'sources.json',x.r.t.read(x.SOURCES))
    for name in ['uniform_rows.jsonl','gaussian_rows.jsonl','uniform_summary.json','gaussian_summary.json','compare.json']:
        (tmp_path/name).write_text('{}\n')
    x.publish(tmp_path)
    return identity,gate,report


@pytest.mark.parametrize('field,value',[
    ('test_access',True),('frames',886),('head_updates',1),('native_head_calls',2660),
    ('val_replay',dict(uniform=False,gaussian=True)),('state_after',{'mutated':True})])
def test_failed_or_tampered_gate_rejected_even_if_reindexed(tmp_path,field,value):
    identity,gate,report=gate_fixture(tmp_path)
    report[field]=value
    (tmp_path/'completion.json').write_text(json.dumps(report))
    (tmp_path/'artifacts.json').unlink();x.publish(tmp_path)
    with pytest.raises(ValueError,match='successful paired'):
        x.checked_val_report(tmp_path/'completion.json',identity,gate)


def test_success_gate_indexed_and_corrupt_row_bytes_rejected(tmp_path):
    identity,gate,_=gate_fixture(tmp_path)
    assert x.checked_val_report(tmp_path/'completion.json',identity,gate)['pass_online_val']
    (tmp_path/'uniform_rows.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='SHA differs'):
        x.checked_val_report(tmp_path/'completion.json',identity,gate)


def test_overwrite_and_nested_outputs_rejected_before_source_access(tmp_path,monkeypatch):
    args=x.parser().parse_args(['--stage','check','--training-dir',str(tmp_path/'train'),
        '--out-dir',str(tmp_path/'train'/'bad')])
    def forbidden():raise AssertionError('Should reject output first')
    monkeypatch.setattr(x,'checked_sources',forbidden)
    with pytest.raises(ValueError,match='disjoint'):x.run(args)
    args.out_dir=str(tmp_path/'existing');Path(args.out_dir).mkdir()
    with pytest.raises(FileExistsError):x.run(args)


def completed_uniform_fixture(tmp_path,monkeypatch):
    base=x.ROOT/'docs/detection/midpoint_readout_fixed_test_evidence_20261012/train'
    if not base.is_dir():
        base=x.ROOT/x.DEFAULT_TRAIN  # Same fixed receipt, server original; no directory search.
    # Reuse real receipt structure, but no real weights, data cache or optimization.
    for name in ['completion.json','selection.json','selected_val_compare.json','val_readout_compare.json','val_epoch_04.rows.jsonl']:
        (tmp_path/name).write_bytes((base/name).read_bytes())
    table=x.r.t.read(tmp_path/'val_readout_compare.json')
    for epoch in range(1,25):
        x.r.t.write(tmp_path/('val_epoch_%02d.json'%epoch),table['all_epoch_values']['head_epoch_%02d.pth'%epoch])
    (tmp_path/x.FIXED['path']).write_bytes(b'synthetic selected weight')
    complete=x.r.t.read(tmp_path/'completion.json')
    fake_gaussian={'value':dict(split='val',metric_mode='full_video_protocol_v2',groups=table['groups']['gaussian_prior_sigma1p5'])}
    fake_cache={'runtime':complete['runtime'],'manifest_sha256':complete['proof']['cache_manifest_sha256']}
    monkeypatch.setattr(x.r,'checked_contract',lambda *a:(fake_cache, {}, fake_gaussian, complete['proof']))
    realsha=x.r.t.sha
    def digest(path):
        if Path(path)==tmp_path/x.FIXED['path']:return x.FIXED['sha256']
        return realsha(path)
    monkeypatch.setattr(x.r.t,'sha',digest)
    # Standalone synthetic index, unrelated real source artifact left unchanged.
    files={p.name:digest(p) for p in tmp_path.iterdir() if p.is_file()}
    for name,entry in complete['selection']['all_checkpoints'].items():
        files[name]=entry['checkpoint']['sha256']
    x.r.t.write(tmp_path/'artifacts.json',dict(protocol=x.r.VERSION,files=files))
    args=x.parser().parse_args(['--stage','check','--training-dir',str(tmp_path),'--out-dir',str(tmp_path.parent/'output')])
    return args,{'training_identity':complete['proof']['identity']}


def test_completed24_selection_audit_accepts_real_receipt_shape_without_tensor(tmp_path,monkeypatch):
    args,identity=completed_uniform_fixture(tmp_path,monkeypatch)
    def forbidden(*a,**k):raise AssertionError('Static loaded tensor')
    monkeypatch.setattr(x.torch,'load',forbidden)
    _,_,gate,selected=x.checked_training(args,identity)
    assert gate['uniform_checkpoint']==x.FIXED
    assert len(selected['rows'])==887


@pytest.mark.parametrize('field,value',[('epochs_completed',23),('head_updates_total',21647),('test_access',True)])
def test_incomplete_training_or_test_selected_training_rejected(tmp_path,monkeypatch,field,value):
    args,identity=completed_uniform_fixture(tmp_path,monkeypatch)
    p=tmp_path/'completion.json';d=x.r.t.read(p);d[field]=value;p.write_text(json.dumps(d))
    index=x.r.t.read(tmp_path/'artifacts.json');index['files'][p.name]=x.r.t.sha(p)
    (tmp_path/'artifacts.json').write_text(json.dumps(index))
    with pytest.raises(ValueError):x.checked_training(args,identity)
