"""Full-data stage isolation, epoch coverage, checkpoint IO and VAL selection."""
from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from crane_project.tools import train_port_geometry_midpoint_formal_v1 as f
from test_port_geometry_midpoint_v1_test import FakeDetector


def row(name, role='train', scale=1., missing=False, mismatch=False):
    sequence, number=name.rsplit('_',1)
    b=torch.tensor([[64.,48.,32.,16.,.2,.8]])
    if missing:
        b=b[:0]
    xy=torch.tensor([[scale,scale]]).expand(len(b),-1)
    bm=b[:,:5].clone(); bm[:,:4]*=scale
    gt=torch.tensor([[64.7,47.6,32.9,15.5,.22]])
    if mismatch:
        gt[:,:2]+=30.
    return dict(image=name,sequence=sequence,domain=sequence.split('_')[0],
        frame_id=int(number),role=role,split=('val' if role=='val' else
            'train_sim' if sequence.startswith('sim_') else 'train'),scale=scale,
        eligible=role=='train' and not missing and not mismatch,
        boxes_original=b,boxes_model=bm,scale_xy=xy,gt_original=gt,
        roi=torch.randn(len(b),256,9,9),support=torch.ones(len(b),1,9,9))


def fixture():
    torch.manual_seed(25)
    train=[row(seq+'_'+str(i).zfill(5),scale=scale)
        for scale in (1.,.5) for seq,n in (('real_seq01',3),('sim_seq08',2)) for i in range(n)]
    val=[row(seq+'_'+str(i).zfill(5),'val') for seq in ('real_seq07','real_seq14','sim_seq10') for i in range(12)]
    return train,val


def test_protocol_separates_formal_head_training_from_old_probe_and_test():
    f.checked_sources()
    value=f.protocol_document()
    assert value['scope']['formal_head_training'] and not value['scope']['full_detector_finetuning']
    assert not value['scope']['original_probe_gate_passed'] and not value['scope']['test_access']
    assert value['scope']['train_views']==5116 and value['scope']['cache_feature_extractions']==6003
    assert value['selection_config']==f.SELECTION_CONFIG and value['settings']['epochs']==24
    assert 'gt' not in inspect.signature(f.capture_cache).parameters
    assert 'sequence' not in inspect.signature(f.capture_cache).parameters
    assert 'test' not in f.parser()._option_string_actions['--stage'].choices
    f.old.checked_replay_body()


@pytest.mark.parametrize('missing',[False,True])
def test_actual_cache_capture_uses_gt_free_native_geometry_and_preserves_missing(missing):
    b=[] if missing else [[64.,48.,32.,16.,.2,.8]]
    detector=FakeDetector(b,[.5,.5]); image=torch.zeros(1,3,1024,1024)
    meta=dict(ori_shape=(1024,1024,3),img_shape=(512,512,3),pad_shape=(1024,1024,3),
        scale_factor=np.array([.5,.5,.5,.5]),flip=False)
    before=f.g.state_digest(detector)
    result=f.capture_cache(detector,image,[meta])
    n=0 if missing else 1
    assert result['roi'].shape==(n,256,9,9) and result['boxes_original'].shape==(n,6)
    assert f.g.state_digest(detector)==before and not result['roi'].requires_grad
    if n:
        assert torch.equal(result['boxes_original'],torch.tensor(b))
    detector=FakeDetector(b,[.5,.5],mutate=True)
    if n:
        with pytest.raises(ValueError,match='changed frozen B output'):
            f.capture_cache(detector,image,[meta])


def test_balanced_epochs_visit_all_eligible_train_views_and_exclude_val():
    train,val=fixture()
    train.append(row('real_seq01_00003',missing=True))
    train.append(row('real_seq01_00004',mismatch=True))
    records=train+val
    first=f.epoch_batches(records,1)
    assert first==f.epoch_batches(records,1)
    assert first!=f.epoch_batches(records,2)
    visited={i for batch in first for i in batch}
    assert visited=={i for i,r in enumerate(records) if r['role']=='train' and r['eligible']}
    assert all(len(batch)==8 and sum(records[i]['domain']=='real' for i in batch)==4 for batch in first)
    with pytest.raises(ValueError,match='eligible TRAIN'):
        f.batch(records,[len(train)],'cpu')
    with pytest.raises(ValueError,match='eligible TRAIN'):
        f.batch(records,[len(train)-1],'cpu')


def test_full_cache_validation_rejects_view_role_and_eligibility_leaks(monkeypatch):
    train,val=fixture()
    monkeypatch.setattr(f.g.ready,'TRAIN_COUNTS',{'real_seq01':3,'sim_seq08':2})
    monkeypatch.setattr(f.g.ready,'VAL_COUNTS',{'real_seq07':12,'real_seq14':12,'sim_seq10':12})
    f.validate_records(train,val)
    wrong=deepcopy(train); wrong[0]['eligible']=False
    with pytest.raises(ValueError,match='eligibility'):
        f.validate_records(wrong,val)
    wrong=deepcopy(train); wrong[0]['role']='val'
    with pytest.raises(ValueError,match='role'):
        f.validate_records(wrong,val)
    with pytest.raises(ValueError,match='Duplicate'):
        f.validate_records(train+[train[0]],val)
    wrong=deepcopy(train); wrong[0]['roi'].requires_grad_(True)
    with pytest.raises(ValueError,match='detached CPU'):
        f.validate_records(wrong,val)


def test_atomic_checkpoint_uses_stream_and_preserves_existing_file(tmp_path,monkeypatch):
    old_save=torch.save; streams=[]
    def checked_save(payload,destination,**kw):
        assert hasattr(destination,'write')
        streams.append(True)
        return old_save(payload,destination,**kw)
    monkeypatch.setattr(torch,'save',checked_save)
    path=tmp_path/'head.pth'; head=f.m.SpatialMidpointHead(); optimizer=f.optimizer_for(head)
    proof=f.save_checkpoint(path,head,optimizer,1,0,{'synthetic':True})
    assert proof['save_reload_exact'] and streams
    before=path.read_bytes()
    with pytest.raises(FileExistsError):
        f.save_checkpoint(path,head,optimizer,2,0,{})
    assert path.read_bytes()==before and len(list(tmp_path.iterdir()))==1


def test_checkpoint_detects_optimizer_and_rng_corruption(tmp_path,monkeypatch):
    original=torch.load
    def corrupted(*args,**kw):
        value=original(*args,**kw)
        value['optimizer_state']['param_groups'][0]['lr']*=2
        return value
    monkeypatch.setattr(torch,'load',corrupted)
    head=f.m.SpatialMidpointHead()
    with pytest.raises(ValueError,match='reload differs'):
        f.save_checkpoint(tmp_path/'bad.pth',head,f.optimizer_for(head),1,0,{})


def test_discarded_smoke_checks_exact_optimizer_continuation(tmp_path):
    train,val=fixture()
    result=f.smoke(train,val,{'manifest_sha256':'synthetic'}, {'synthetic':True},'cpu',tmp_path)
    assert result['status']==f.SMOKE_STATUS and result['discarded']
    assert result['optimizer_continuation_exact'] and result['inference_reload_exact']
    assert result['updates_before_save']==2 and result['continuation_updates_each']==1
    assert result['logs'][0]['output_grad_norm']>0 and result['logs'][1]['stem_grad_norm']>0
    assert len(f.smoke_views(val))==6


def test_missing_frames_preserved_and_three_center_denominators_separate():
    _,val=fixture(); val[0]=row(val[0]['image'],'val',missing=True)
    head=f.m.SpatialMidpointHead(); before=f.g.state_digest(head)
    rows=f.evaluate(head,val,'cpu'); result=f.summaries(rows)
    assert f.g.state_digest(head)==before and rows[0]['b'] is None and rows[0]['midpoint'] is None
    total=result['groups']['overall']['midpoint']
    assert total['output_coverage_fraction']=={'numerator':35,'denominator':36}
    assert total['conditional_center_correct_fraction']=={'numerator':35,'denominator':35}
    assert total['all_frame_center_correct_fraction']=={'numerator':35,'denominator':36}
    assert result['split']=='val'
    assert f.comparison_checks(result)['automatic_promotion'] is False
    changed=deepcopy(result)
    changed['groups']['real']['midpoint']['all_frame_center_correct_fraction']['numerator']+=1
    assert f.comparison_checks(changed)['checks']['real/all_frame_center_preserved']


def test_formal_two_epoch_integration_saves_all_and_selects_only_val(tmp_path,monkeypatch):
    train,val=fixture(); monkeypatch.setitem(f.SETTINGS,'epochs',2)
    monkeypatch.setattr(f.g.ready,'TRAIN_COUNTS',{'real_seq01':3,'sim_seq08':2})
    monkeypatch.setattr(f.g.ready,'VAL_COUNTS',{'real_seq07':12,'real_seq14':12,'sim_seq10':12})
    logs=[]
    result=f.train_formal(train,val,{'manifest_sha256':'synthetic'},{'synthetic':True},'cpu',tmp_path,logs.append)
    assert result['epochs_completed']==2 and result['updates']==4
    assert result['detector_updates']==0 and not result['test_access']
    selection=json.loads((tmp_path/'selection.json').read_text())
    assert selection['split']=='val' and not selection['selection_on_test'] and not selection['automatic_promotion']
    assert len(selection['all_checkpoints'])==2
    for epoch in (1,2):
        ckpt=torch.load(str(tmp_path/('head_epoch_%02d.pth'%epoch)),map_location='cpu')
        assert ckpt['epoch']==epoch and ckpt['updates']==2*epoch
        assert len((tmp_path/('val_epoch_%02d.rows.jsonl'%epoch)).read_text().splitlines())==36
    assert len([r for r in logs if r['stage']=='epoch_complete'])==2


def test_selected_pipeline_rejects_checkpoint_or_selection_changes_before_detector(tmp_path,monkeypatch):
    head=f.m.SpatialMidpointHead(); identity={'synthetic':True}
    proof=f.save_checkpoint(tmp_path/'head_epoch_01.pth',head,f.optimizer_for(head),1,2,identity,
        {'manifest_sha256':'cache','detector_state':{}})
    selection=dict(protocol=f.VERSION,split='val',identity=identity,selected_checkpoint=proof,
        selection_config=f.SELECTION_CONFIG,selection_on_test=False,cache_manifest_sha256='cache')
    selection_path=tmp_path/'selection.json'
    monkeypatch.setattr(f,'checked_sources',lambda:identity)
    def forbidden(*args,**kw):
        raise AssertionError('Must reject before building detector')
    monkeypatch.setattr(f,'build_detector',forbidden)
    selection['selection_on_test']=True
    selection_path.write_text(json.dumps(selection))
    with pytest.raises(ValueError,match='full-VAL formal selection'):
        f.load_selected_pipeline(selection_path,0)
    selection['selection_on_test']=False; selection['selected_checkpoint']['sha256']='wrong'
    selection_path.write_text(json.dumps(selection))
    with pytest.raises(ValueError,match='checkpoint SHA'):
        f.load_selected_pipeline(selection_path,0)


def test_static_stage_never_opens_cache_data_checkpoint_or_cuda(tmp_path,monkeypatch):
    args=f.parser().parse_args(['--stage','check','--out-dir',str(tmp_path/'static')])
    monkeypatch.setattr(f.g,'check_cfg',lambda:None)
    def forbidden(*args,**kw):
        raise AssertionError('Static stage accessed data/cache/GPU')
    for name in ('checked_data','runtime','create_cache','load_cache','build_detector'):
        monkeypatch.setattr(f,name,forbidden)
    monkeypatch.setattr(torch,'load',forbidden)
    f.run(args)
    report=json.loads((tmp_path/'static/completion.json').read_text())
    assert report['status']=='STATIC_FORMAL_MIDPOINT_CONTRACT_PASS_NO_DATA_GPU_UPDATES'
