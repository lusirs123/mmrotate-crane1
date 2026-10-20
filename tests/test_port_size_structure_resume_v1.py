"""Regression for the epoch2 exclusive-write crash and SHA-bound recovery."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from crane_project.utils import port_size_structure_resume_v1 as io


def dump(path,value):path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(value)+'\n')


def test_mutable_status_can_complete_multiple_epochs_arms_and_final(tmp_path):
    path=tmp_path/'progress.json'
    for epoch in range(1,25):
        io.status_write(path,dict(epoch=epoch));assert json.loads(path.read_text())['epoch']==epoch
    path=tmp_path/'training.json'
    for stage in ['V_complete','D_complete','COMPLETE']:
        io.status_write(path,dict(stage=stage));assert json.loads(path.read_text())['stage']==stage
    assert len(list(tmp_path.iterdir()))==2


def test_invalid_json_does_not_damage_previous_status(tmp_path):
    p=tmp_path/'progress.json';io.status_write(p,dict(epoch=2));before=p.read_bytes()
    with pytest.raises(ValueError):io.status_write(p,dict(loss=float('nan')))
    assert p.read_bytes()==before and len(list(tmp_path.iterdir()))==1
    for name in ['head_epoch_02.pth','epoch_02_val.json','source_identity.json']:
        with pytest.raises(ValueError):io.status_write(tmp_path/name,dict(x=1))
        assert not (tmp_path/name).exists()


def fixture(tmp_path):
    old=tmp_path/'failed';old.mkdir();out=tmp_path/'recovered';out.mkdir();protocol=dict(protocol='port_size_structure_v1',epochs=24)
    sources={**io.ALLOWED_OLD,'model.py':'model_unchanged_sha'}
    dump(old/'finished.json',dict(status='FAILED',error="FileExistsError(17, 'File exists')"))
    dump(old/'protocol.json',protocol);dump(old/'source_identity.json',dict(sources=sources))
    for n in io.BASE:dump(old/n,dict(status='PASS',name=n))
    dump(old/'input_identity.json',dict(cache='same'))
    for epoch in (1,2):
        d=old/'V_structure';d.mkdir(exist_ok=True);w=d/('head_epoch_%02d.pth'%epoch);w.write_bytes(b'fake-checkpoint-'+str(epoch).encode())
        dump(d/('epoch_%02d_val.json'%epoch),dict(epoch=epoch,checkpoint_sha256=io.sha(w)))
        for name in ['epoch_%02d_val.rows.jsonl.gz','epoch_%02d_train.json','epoch_%02d_train.rows.jsonl.gz']:dump(d/(name%epoch),dict(epoch=epoch))
    dump(old/'V_structure/progress.json',dict(last_complete_epoch=1))
    dump(old/'artifacts.json',dict(files={str(p.relative_to(old)):io.sha(p) for p in old.rglob('*') if p.is_file()}))
    proof=dict(sources={k:('repaired' if k in io.ALLOWED_OLD else h) for k,h in sources.items()})
    return old,out,protocol,proof


def test_complete_epoch2_recovered_despite_stale_progress_and_old_bytes_preserved(tmp_path):
    old,out,protocol,proof=fixture(tmp_path);snapshot={str(p.relative_to(old)):p.read_bytes() for p in old.rglob('*') if p.is_file()}
    r=io.prepare(old,out,protocol,proof)
    assert r['completed_epochs']=={'V_structure':2,'D_direct':0} and r['completed_updates']==1278
    assert len(r['events']['V_structure'])==2 and not (out/'V_structure/progress.json').exists()
    assert (out/'V_structure/head_epoch_02.pth').read_bytes()==(old/'V_structure/head_epoch_02.pth').read_bytes()
    assert snapshot=={str(p.relative_to(old)):p.read_bytes() for p in old.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):io.prepare(old,out,protocol,proof)


def test_model_change_protocol_change_or_unindexed_damage_stops(tmp_path):
    old,out,protocol,proof=fixture(tmp_path)
    changed=deepcopy(proof);changed['sources']['model.py']='changed'
    with pytest.raises(ValueError):io.prepare(old,out,protocol,changed)
    with pytest.raises(ValueError):io.prepare(old,out,dict(protocol,epochs=25),proof)
    (old/'V_structure/head_epoch_02.pth').write_bytes(b'corrupted')
    with pytest.raises(ValueError):io.prepare(old,out,protocol,proof)


def test_partial_epoch_not_silently_retrained(tmp_path):
    old,out,protocol,proof=fixture(tmp_path);w=old/'V_structure/head_epoch_03.pth';w.write_bytes(b'partial')
    i=json.loads((old/'artifacts.json').read_text());i['files'][str(w.relative_to(old))]=io.sha(w);dump(old/'artifacts.json',i)
    with pytest.raises(ValueError):io.prepare(old,out,protocol,proof)


def test_replay_reproduces_val_enrichment_without_dropping_any_fields():
    # Same order as the original writer: forward -> summary -> persist.
    raw=[dict(image='frame',edge_residual=[0.,0.,100.,50.,0.,.8],pre_guard=[0.,0.,100.,50.,0.,.8])]
    def summary(rows,cache):
        assert cache=={}
        for row in rows:row['metrics']=dict(derived_size=row['edge_residual'][2])
    sealed=deepcopy(raw);summary(sealed,{})
    assert io.verify_replay(deepcopy(raw),sealed,'val',summary)['derived_metrics_recomputed']
    assert io.verify_replay(deepcopy(raw),raw,'train',lambda *_:pytest.fail('TRAIN must not be enriched'))['rows_exact']
    for field in ('edge_residual','pre_guard'):
        changed=deepcopy(raw);changed[0][field][2]+=.00001
        with pytest.raises(ValueError,match=field):io.verify_replay(changed,sealed,'val',summary)
    damaged=deepcopy(sealed);damaged[0]['metrics']['derived_size']+=1
    with pytest.raises(ValueError,match='metrics'):io.verify_replay(deepcopy(raw),damaged,'val',summary)
    damaged=deepcopy(sealed);damaged[0]['unexpected']=True
    with pytest.raises(ValueError,match='unexpected'):io.verify_replay(deepcopy(raw),damaged,'val',summary)


def test_replay_count_and_split_mismatch_rejected():
    with pytest.raises(ValueError,match='count'):io.verify_replay([], [dict(image='x')],'train',None)
    with pytest.raises(ValueError,match='split'):io.verify_replay([],[],'test',None)
