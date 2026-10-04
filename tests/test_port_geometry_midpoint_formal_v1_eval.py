"""Frozen formal checkpoint identity, split isolation and replay boundaries."""
from copy import deepcopy
import inspect
import json

import numpy as np
import pytest

from crane_project.tools import eval_port_geometry_midpoint_formal_v1 as e


def selection_fixture(path):
    identity=e.checked_sources(); entries={}
    metrics={'real/TDR_w10(%)':100.,'sim/TDR_w10(%)':100.,
        'real/R_center(%)':99.,'sim/R_center(%)':100.,'sim/ACI':.95,
        'real/MCML_max(frames)':4,'sim/MCML_max(frames)':0,'sim/A-RMSE(deg)':2.}
    for epoch in range(1,25):
        name='head_epoch_%02d.pth'%epoch
        (path/name).write_bytes(b'synthetic unopened weight')
        proof=dict(path=name,sha256=e.g.ready.sha(path/name),epoch=epoch,updates=epoch,
            head_digest={'synthetic':True},save_reload_exact=True)
        entries[name]=dict(checkpoint=proof,metrics=deepcopy(metrics))
    chosen,_,info=e.f.select_best_checkpoint(entries,e.f.SELECTION_CONFIG)
    value=dict(protocol=e.f.VERSION,identity=identity['training_identity'],split='val',
        selected_checkpoint=entries[chosen]['checkpoint'],all_checkpoints=entries,
        selection_info=info,selection_config=e.f.SELECTION_CONFIG,
        cache_manifest_sha256='synthetic',test_access=False,selection_on_test=False)
    (path/'selection.json').write_text(json.dumps(value))
    completion=dict(status='FORMAL_MIDPOINT_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED',
        epochs_completed=24,detector_updates=0,test_access=False,selection=value)
    (path/'completion.json').write_text(json.dumps(completion))
    (path/'selected_val_compare.json').write_text('{}')
    (path/'artifacts.json').write_text(json.dumps(dict(files={p.name:e.g.ready.sha(p) for p in path.iterdir()})))
    return identity,value


def test_new_sources_keep_training_contract_unchanged_and_no_replay_calls():
    proof=e.checked_sources()
    assert len(proof['sources'])==73
    assert e.protocol_document()['scope']['head_updates']==0
    assert e.protocol_document()['scope']['selection_on_test'] is False
    text=inspect.getsource(e.run)
    assert 'old.replay_fit(' not in text and 'fit_arm(' not in text
    assert '--epoch' not in e.parser()._option_string_actions


def test_completed_selection_recomputed_with_old_rule(tmp_path):
    identity,selection=selection_fixture(tmp_path)
    value,proof=e.checked_selection(tmp_path/'selection.json',identity)
    assert value==selection and proof['selected_checkpoint']['epoch']==1


@pytest.mark.parametrize('kind',['weight','selection','completion'])
def test_selection_tampering_or_incomplete_run_rejected_before_gpu(tmp_path,kind):
    identity,value=selection_fixture(tmp_path)
    if kind=='weight':
        (tmp_path/value['selected_checkpoint']['path']).write_bytes(b'changed')
    elif kind=='selection':
        value['selected_checkpoint']=value['all_checkpoints']['head_epoch_02.pth']['checkpoint']
        (tmp_path/'selection.json').write_text(json.dumps(value))
    else:
        (tmp_path/'completion.json').write_text(json.dumps(dict(status='FAILED')))
    with pytest.raises((ValueError,KeyError)):
        e.checked_selection(tmp_path/'selection.json',identity)


@pytest.mark.parametrize('split',['val','test'])
def test_static_check_never_loads_weights_or_opens_datasets(tmp_path,monkeypatch,split):
    training=tmp_path/'train'; training.mkdir(); selection_fixture(training)
    def forbidden(*args,**kw):
        raise AssertionError('Static evaluation accessed data/GPU/torch.load')
    monkeypatch.setattr(e.f,'runtime',forbidden)
    monkeypatch.setattr(e.f,'load_selected_pipeline',forbidden)
    monkeypatch.setattr(e.old,'fixed_test_inputs',forbidden)
    monkeypatch.setattr(e,'val_inputs',forbidden)
    monkeypatch.setattr(e.torch,'load',forbidden)
    args=e.parser().parse_args(['--check-only','--split',split,'--selection',str(training/'selection.json'),
        '--out-dir',str(tmp_path/'static')])
    e.run(args)
    report=json.loads((tmp_path/'static/completion.json').read_text())
    assert report['status']=='STATIC_FORMAL_MIDPOINT_EVAL_PASS_NO_DATA_GPU_UPDATES'
    assert report['test_access'] is False and report['head_updates']==0


def predicted(name='real_seq07_00000',missing=False):
    source=dict(image=name,sequence=name.rsplit('_',1)[0],frame_id=0,domain=name.split('_')[0],
        gt=[64.,48.,32.,16.,.2])
    box=None if missing else source['gt']+[.8]
    output=dict(b=box,midpoint=box,candidate=box,candidate_valid=None if missing else True,
        accepted=None if missing else True,failed_checks=[],original_b_raw_exact_before_after=True)
    return e.prediction_row(source,output)


def test_fresh_val_replay_missing_decisions_scores_and_gt_checked(tmp_path):
    rows=[predicted(),predicted('sim_seq10_00000',True)]
    path=tmp_path/'rows.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    assert e.verify_val_rows(rows,path)['frames']==2
    for key in ('accepted','gt'):
        wrong=deepcopy(rows)
        wrong[0][key]=False if key=='accepted' else [0]*5
        with pytest.raises(ValueError,match='metadata/decision'):
            e.verify_val_rows(wrong,path)
    wrong=deepcopy(rows); wrong[0]['midpoint'][5]=.2
    with pytest.raises(ValueError,match='box differs'):
        e.verify_val_rows(wrong,path)
    wrong=deepcopy(rows); wrong[0]['midpoint'][0]+=.1
    with pytest.raises(ValueError,match='box differs'):
        e.verify_val_rows(wrong,path)


def test_gt_is_float32_like_formal_cached_evaluation_and_missing_kept():
    value=predicted(missing=True)
    assert value['metrics']['midpoint']['output'] is False
    assert value['metrics']['midpoint']['protocol_angle_error_deg']==90.
    assert value['gt']==np.asarray([64.,48.,32.,16.,.2],dtype=np.float32).tolist()
