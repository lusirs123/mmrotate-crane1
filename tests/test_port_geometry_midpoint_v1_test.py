"""Exact reconstruction, TEST isolation, missing frames and metric contracts."""
from copy import deepcopy
import inspect
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from crane_project.tools import eval_port_geometry_midpoint_v1_test as e
from test_port_geometry_midpoint_v1 import synthetic


def test_original_fit_implementation_is_unchanged_and_replay_body_is_exact():
    e.checked_replay_body()
    assert 'gt' not in inspect.signature(e.capture).parameters
    assert 'source' not in inspect.signature(e.capture).parameters
    assert e.protocol_document()['scope']['selection_on_test'] is False
    assert e.protocol_document()['original_probe_gate_passed'] is False


def test_replay_retains_exact_same_final_head_and_all_results(monkeypatch):
    monkeypatch.setitem(e.m.SETTINGS,'steps_per_arm',2)
    records,samples=synthetic()
    prepared,_=e.t.prepare_records(records,samples,e.a.donor_map(samples))
    batches=e.g.schedule(prepared)[:2]
    torch.manual_seed(1703)
    initial=deepcopy(e.m.SpatialMidpointHead().state_dict())
    old,maps=e.t.fit_arm(prepared,batches,initial,'matched','cpu',lambda x:None)
    new,new_maps,head=e.replay_fit(prepared,batches,initial,'matched','cpu',lambda x:None)
    assert new==old and new_maps==maps
    e.verify_replay(new,{'arms':{'matched':old}},head)
    wrong=deepcopy(old); wrong['final']['rows'][0]['pred'][0]+=.01
    with pytest.raises(ValueError,match='stop before TEST'):
        e.verify_replay(new,{'arms':{'matched':wrong}},head)
    with torch.no_grad():
        head.output.weight[0,0,0,0]+=.01
    with pytest.raises(ValueError,match='stop before TEST'):
        e.verify_replay(new,{'arms':{'matched':old}},head)


def test_diagnostic_head_save_reload_uses_file_stream_and_refuses_overwrite(tmp_path,monkeypatch):
    head=e.m.SpatialMidpointHead()
    old=torch.save; streams=[]
    def saving(value,destination,**kwargs):
        assert hasattr(destination,'write')
        streams.append(destination)
        return old(value,destination,**kwargs)
    monkeypatch.setattr(torch,'save',saving)
    path=tmp_path/'fixed_matched_head.pth'
    result=e.save_head(path,head,{'source':'synthetic'})
    assert result['save_reload_exact'] and result['diagnostic_only'] and len(streams)==1
    before=path.read_bytes()
    with pytest.raises(FileExistsError):
        e.save_head(path,head,{'source':'different'})
    assert path.read_bytes()==before
    assert [p.name for p in tmp_path.iterdir()]==['fixed_matched_head.pth']


def test_native_library_identity_cannot_be_silently_changed():
    runtime={k:'synthetic' for k in ('torch','cuda','cudnn','mmcv','mmdet','mmrotate','opencv','gpu')}
    e.verify_native_runtime(runtime,dict(runtime))
    wrong=dict(runtime,mmcv='different')
    with pytest.raises(ValueError,match='libraries/GPU differ'):
        e.verify_native_runtime(runtime,wrong)


class FakeDetector(torch.nn.Module):
    def __init__(self,b,scales,mutate=False):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.zeros(1),requires_grad=False)
        self.b=np.asarray(b,dtype=np.float32).reshape(-1,6)
        self.scales=np.asarray(scales,dtype=np.float32)
        self.mutate=mutate; self.raw_calls=0
        self.eval()

    def extract_feat(self,image):
        return [torch.zeros(1,256,128,128)]

    def simple_test_from_features(self,features,metas,rescale):
        value=self.b.copy()
        if not rescale:
            value[:,:4]*=self.scales[[0,1,0,1]]
            self.raw_calls+=1
            if self.mutate and self.raw_calls>1 and len(value):
                value[0,0]+=.01
        return [[value]]


@pytest.mark.parametrize('empty',[False,True])
@pytest.mark.parametrize('swap',[False,True])
def test_online_original_coordinates_scores_empty_frames_and_exact_call_budget(empty,swap):
    b=[] if empty else [[64.,48.,16. if swap else 32.,32. if swap else 16.,.2,.8]]
    scales=np.array([.6201,.6204,.6201,.6204],dtype=np.float32)
    meta=dict(ori_shape=(721,1031,3),img_shape=(448,640,3),pad_shape=(1024,1024,3),
        scale_factor=scales,flip=False,filename='not_a_gt_input.jpg')
    detector=FakeDetector(b,scales[:2])
    head=e.m.SpatialMidpointHead().eval().requires_grad_(False)
    before=e.g.state_digest(head); events=[]
    result=e.capture(detector,head,torch.zeros(1,3,1024,1024),[meta],events.append)
    assert e.g.state_digest(head)==before
    assert sum(v['stage']=='test_feature_extracted' for v in events)==1
    assert sum(v['stage']=='test_native_head_called' for v in events)==3
    assert result['b']==result['midpoint']
    assert result['b'] is None if empty else result['accepted']


def test_online_rejects_nonfrozen_head_and_b_mutation():
    b=[[64.,48.,32.,16.,.2,.8]]
    meta=dict(ori_shape=(1024,1024,3),img_shape=(1024,1024,3),pad_shape=(1024,1024,3),
        scale_factor=np.ones(4,dtype=np.float32),flip=False)
    detector=FakeDetector(b,[1.,1.],mutate=True)
    head=e.m.SpatialMidpointHead()
    image=torch.zeros(1,3,1024,1024)
    with pytest.raises(ValueError,match='must be frozen'):
        e.capture(detector,head,image,[meta])
    head.eval().requires_grad_(False)
    with pytest.raises(ValueError,match='changed B raw'):
        e.capture(detector,head,image,[meta])


def test_online_bound_fallback_keeps_entire_b_and_reports_raw_candidate():
    class AngleBoundHead(e.m.SpatialMidpointHead):
        def forward(self,roi,support,b,bm,scales):
            candidate=b[:,:5].clone(); candidate[:,4]+=.3
            points=e.m.box_midpoints(candidate)
            local=e.m.original_to_roi(points,b,bm,scales)
            value=e.m.decode_points(points,b,local,torch.tensor([False]))
            value['points_original']=points
            return value
    b=[[64.,48.,32.,16.,.2,.8]]
    meta=dict(ori_shape=(1024,1024,3),img_shape=(1024,1024,3),pad_shape=(1024,1024,3),
        scale_factor=np.ones(4,dtype=np.float32),flip=False)
    head=AngleBoundHead().eval().requires_grad_(False)
    result=e.capture(FakeDetector(b,[1.,1.]),head,torch.zeros(1,3,1024,1024),[meta])
    assert result['accepted'] is False and result['candidate_valid']
    assert 'angle_bound' in result['failed_checks']
    assert result['b']==result['midpoint'] and result['candidate']!=result['b']


def record(frame,pred,sequence='real_seq03'):
    gt=[32.,32.,20.,10.,.2]
    return dict(image=sequence+'_%05d'%frame,sequence=sequence,domain=sequence.split('_')[0],
        frame_id=frame,gt=gt,pred=pred,metrics=e.g.decompose(gt,pred[:5] if pred is not None else None))


def test_center_three_denominators_missing_penalty_pure_angle_and_gap_breaks():
    good=[32.,32.,20.,10.,.2,.8]
    wrong=[52.,32.,20.,10.,.2,.8]
    rows=[record(0,good),record(1,None),record(2,wrong),record(4,None),
          record(5,None),record(6,None,sequence='sim_seq09')]
    value=e.summary(rows)
    assert value['output_coverage_fraction']==dict(numerator=2,denominator=6)
    assert value['conditional_center_correct_fraction']==dict(numerator=1,denominator=2)
    assert value['all_frame_center_correct_fraction']==dict(numerator=1,denominator=6)
    assert value['angle_error_deg']['n']==2 and value['angle_error_deg']['rmse']==0
    assert value['protocol_angle']['n']==6 and value['center_penalty_count']==1
    assert value['no_output_penalty_count']==4
    assert value['longest_failure_run_frames']['center']==2
    assert value['longest_failure_run_frames']['no_output']==2


def test_all_missing_conditional_rate_is_undefined_not_zero_accuracy():
    value=e.summary([record(0,None)])
    assert value['conditional_center_correct_fraction']==dict(numerator=0,denominator=0)
    assert value['output_center_hit_pct'] is None
    assert value['angle_error_deg']['n']==0
    assert value['all_frame_center_hit_pct']==0


def test_protocol_angle_and_temporal_metrics_use_equivalent_long_axis_geometry():
    rows=[record(0,[32.,32.,20.,10.,.2,.8],sequence='sim_seq09'),
          record(1,[32.,32.,10.,20.,.2+math.pi/2,.8],sequence='sim_seq09')]
    raw=deepcopy(rows)
    value=e.summary(rows)
    assert rows==raw  # Normalize an offline metric view only.
    assert value['angle_error_deg']['rmse']<1e-10
    assert value['metric_protocol_v2']['sim/A-RMSE(deg)']==0.
    assert value['metric_protocol_v2']['sim/ACI']==1.


def test_static_does_not_open_test_cache_cuda_or_replay(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'checked_contract',lambda *args:([],{}, {}, {}, {'source':'mock'}))
    source=tmp_path/'sources.json'; source.write_text('{}')
    monkeypatch.setattr(e,'MANIFEST',source)
    def forbidden(*args,**kwargs):
        raise AssertionError('Static must not access TEST/cache/GPU/updates')
    for name in ('fixed_test_inputs','replay_fit','build_detector','build_test_dataset'):
        monkeypatch.setattr(e,name,forbidden)
    monkeypatch.setattr(e.g,'checked_cache',forbidden)
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    monkeypatch.setenv('WORLD_SIZE','1')
    out=tmp_path/'static'
    args=SimpleNamespace(out_dir=str(out),baseline_report='missing',train_report='missing',check_only=True)
    report=e.run(args)
    assert report['test_access'] is False and report['cache_loads']==0 and report['replay_head_updates']==0
    assert report['status']=='STATIC_MIDPOINT_TEST_CONTRACT_COMPLETE_NO_TEST_CACHE_GPU_UPDATES'
    with pytest.raises(FileExistsError):
        e.run(args)


def test_failed_reconstruction_is_saved_and_cannot_open_test(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'checked_contract',lambda *args: (_ for _ in ()).throw(ValueError('SHA differs')))
    monkeypatch.setenv('WORLD_SIZE','1')
    out=tmp_path/'failed'
    args=SimpleNamespace(out_dir=str(out),baseline_report='missing',train_report='missing',check_only=False)
    with pytest.raises(ValueError,match='SHA differs'):
        e.run(args)
    report=json.loads((out/'completion.json').read_text())
    assert report['status']=='FAILED_REVIEW_REQUIRED' and report['test_access'] is False
    artifacts=json.loads((out/'artifacts.json').read_text())
    assert artifacts['files']['completion.json']==e.g.ready.sha(out/'completion.json')


def test_failed_actual_replay_verification_blocks_head_export_and_test(tmp_path,monkeypatch):
    records,samples=synthetic()
    prepared,_=e.t.prepare_records(records,samples,e.a.donor_map(samples))
    runtime={k:'synthetic' for k in ('python','torch','cuda','cudnn','numpy','opencv','gpu')}
    reviewed=dict(runtime=runtime,minibatches=e.g.schedule(prepared))
    identity=dict(prior_identity=dict(g_identity={}))
    monkeypatch.setattr(e,'checked_contract',lambda *args:(samples,identity,{},reviewed,{}))
    source=tmp_path/'sources.json'; source.write_text('{}')
    monkeypatch.setattr(e,'MANIFEST',source)
    monkeypatch.setattr(e.a,'checked_runtime',lambda *args:dict(runtime))
    monkeypatch.setattr(e.g.ready,'sha',lambda p:e.a.CACHE_MANIFEST_SHA)
    monkeypatch.setattr(e.g,'checked_cache',lambda *args:(dict(records=records,detector_state_before={},extraction_runtime={}),{}))
    monkeypatch.setattr(e,'verify_native_runtime',lambda *args:None)
    monkeypatch.setattr(e.g,'validate_records',lambda *args:None)
    monkeypatch.setattr(e.a,'check_cache_reference',lambda *args:None)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'set_device',lambda *args:None)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *args:'synthetic')
    monkeypatch.setattr(e.g,'seed_all',lambda:None)
    monkeypatch.setattr(e.g,'measured',lambda fn,gpu:(fn(),{}))
    monkeypatch.setattr(e,'replay_fit',lambda *args:({},[],e.m.SpatialMidpointHead()))
    def mismatch(*args):
        raise ValueError('TRAIN replay differs')
    monkeypatch.setattr(e,'verify_replay',mismatch)
    def forbidden(*args):
        raise AssertionError('Must not export/read TEST after failed replay')
    monkeypatch.setattr(e,'fixed_test_inputs',forbidden)
    monkeypatch.setattr(e,'save_head',forbidden)
    monkeypatch.setenv('WORLD_SIZE','1')
    out=tmp_path/'failed_replay'
    args=SimpleNamespace(out_dir=str(out),baseline_report='mock',train_report='mock',check_only=False,
        gpu=0,cache_dir=str(tmp_path/'cache'))
    with pytest.raises(ValueError,match='TRAIN replay differs'):
        e.run(args)
    report=json.loads((out/'completion.json').read_text())
    assert report['status']=='FAILED_REVIEW_REQUIRED' and report['test_access'] is False
    assert not (out/'fixed_matched_head.pth').exists()
