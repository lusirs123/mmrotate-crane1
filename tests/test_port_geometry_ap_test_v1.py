"""Synthetic explicit-TEST input/gate/report checks; never model performance."""
from copy import deepcopy
from types import SimpleNamespace
import json
import sys

import pytest

from crane_project.tools import eval_port_geometry_ap_test_v1 as t


def rows():
    result = []
    for seq, n in sorted(t.COUNTS.items()):
        start = int(seq == 'real_seq03')
        for i in range(start, n + start):
            b = [20., 30., 10., 5., 0., .9]
            result.append(dict(image='%s_%05d' % (seq, i), sequence=seq, frame_id=i,
                domain=seq.split('_')[0], scale=1., gt=b[:5], b=b[:], midpoint=b[:],
                accepted=True, original_b_raw_exact_before_after=True))
    return result


def val_report():
    entry = dict(ap=1., cpu_reference_ap=1., native_verification=dict(pass_check=True, ap=1.))
    return dict(status='VAL_AP_NATIVE_VERIFIED', protocol=t.ap.VERSION, split='val', frames=887,
        test_access=False, selection_changed=False,
        metric=dict(integration='VOC2007 11points', iou_thresholds=[.5,.75], native_backend='mmrotate'),
        provenance=dict(b_checkpoint_sha256=t.ap.B_SHA, head_checkpoint_sha256=t.ap.HEAD_SHA,
            annotation_set_sha256=t.ap.ANN_SHA, inputs={'val_rows.jsonl':t.ap.ROWS_SHA}),
        groups={d:{m:{label:deepcopy(entry) for label in ['AP50','AP75']}
                   for m in ['b','midpoint']} for d in ['real','sim']})


def test_complete_test_and_missing_semantics_coverage():
    r=rows();r[0]['b']=r[0]['midpoint']=None;r[0]['accepted']=None
    t.validate_rows(r)
    c=t.ap.coverage([x for x in r if x['domain']=='real'],'b')
    assert c['total_frames']==868 and c['output_frames']==867
    assert c['output_center_hit_percent']==100 and c['full_frame_center_correct_percent']<100


@pytest.mark.parametrize('change',['missing','duplicate','val','order','scale','score','fallback','accepted','nan'])
def test_bad_test_rows_rejected(change):
    r=rows()
    if change=='missing':r.pop()
    if change=='duplicate':r[-1]=deepcopy(r[-2])
    if change=='val':r[0]['image']='real_seq07_00001'
    if change=='order':r.reverse()
    if change=='scale':r[0]['scale']=.5
    if change=='score':r[0]['midpoint'][5]=.8
    if change=='fallback':r[0]['accepted']=False;r[0]['midpoint'][0]+=1
    if change=='accepted':r[0]['accepted']=None
    if change=='nan':r[0]['gt'][0]=float('nan')
    with pytest.raises(ValueError):t.validate_rows(r)


def test_val_gate_precedes_any_test_access(tmp_path,monkeypatch):
    accesses=[]
    def fail(path):accesses.append(path);raise ValueError('synthetic gate failed')
    monkeypatch.setattr(t,'checked_val_gate',fail)
    with pytest.raises(ValueError,match='gate failed'):
        t.checked_inputs(tmp_path/'missing-test',tmp_path/'missing-anns',tmp_path/'val')
    assert accesses==[tmp_path/'val']


@pytest.mark.parametrize('change',[None,'sha','cpu-status','weight','split','native-fail','ap'])
def test_gate_requires_fixed_native_val(change,tmp_path,monkeypatch):
    r=val_report()
    if change=='cpu-status':r['status']='VAL_AP_CPU_COMPLETE_NATIVE_CHECK_PENDING'
    if change=='weight':r['provenance']['head_checkpoint_sha256']='wrong'
    if change=='split':r['split']='test'
    if change=='native-fail':r['groups']['real']['b']['AP50']['native_verification']['pass_check']=False
    if change=='ap':r['groups']['sim']['midpoint']['AP75']['ap']=float('inf')
    p=tmp_path/'val.json';p.write_text(json.dumps(r))
    monkeypatch.setattr(t,'VAL_AP_SHA','wrong' if change=='sha' else t.ap.sha(p))
    if change is None:assert t.checked_val_gate(p)['split']=='val'
    else:
        with pytest.raises(ValueError):t.checked_val_gate(p)


def test_saved_test_artifact_mismatch_before_rows_read(tmp_path,monkeypatch):
    monkeypatch.setattr(t,'checked_val_gate',lambda *args:val_report())
    for name in t.REQUIRED:(tmp_path/name).write_text('{}')
    t.ap.write_json(tmp_path/'artifacts.json',dict(files={n:'wrong' for n in t.REQUIRED}))
    with pytest.raises(ValueError,match='TEST artifact SHA'):
        t.checked_inputs(tmp_path,tmp_path,tmp_path/'val')


def test_report_access_flags_and_same_core_without_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(t,'checked_inputs',lambda *args:(rows(),{}))
    args=SimpleNamespace(eval_dir=tmp_path,ann_dir=tmp_path,val_ap_report=tmp_path,
                         out_dir=tmp_path/'cpu',backend='cpu',check_only=False)
    t.run(args);r=t.ap.read_json(args.out_dir/'ap_report.json')
    assert r['split']=='test' and r['frames']==1440 and r['test_access'] is True
    assert r['selection_on_test'] is False and r['selection_changed'] is False
    assert r['inference_calls']==0 and r['status']=='TEST_AP_CPU_COMPLETE_NATIVE_CHECK_PENDING'
    assert r['groups']['real']['b']['AP50']['ap']==1.
    before=(args.out_dir/'ap_report.json').read_bytes()
    with pytest.raises(FileExistsError):t.run(args)
    assert (args.out_dir/'ap_report.json').read_bytes()==before


def test_failed_native_does_not_write_output(tmp_path,monkeypatch):
    monkeypatch.setattr(t,'checked_inputs',lambda *args:(rows(),{}))
    def fail(*args):raise RuntimeError('synthetic native failure')
    monkeypatch.setattr(t.ap,'verify_native',fail)
    args=SimpleNamespace(eval_dir=tmp_path,ann_dir=tmp_path,val_ap_report=tmp_path,
                         out_dir=tmp_path/'native',backend='mmrotate',check_only=False)
    with pytest.raises(RuntimeError,match='native failure'):t.run(args)
    assert not args.out_dir.exists()


def test_invalid_val_report_leaves_no_output(tmp_path):
    args=SimpleNamespace(eval_dir=tmp_path,ann_dir=tmp_path,val_ap_report=tmp_path/'absent',
                         out_dir=tmp_path/'out',backend='cpu',check_only=True)
    with pytest.raises(FileNotFoundError):t.run(args)
    assert not args.out_dir.exists()


def test_original_val_cli_still_has_no_test_route():
    with pytest.raises(SystemExit):t.ap.parser().parse_args(['--split','test','--out-dir','synthetic'])


def test_ranked_ap_can_increase_while_final_tp_count_decreases():
    # Same scores/frames: earlier TP precision can improve even with lower recall.
    gt=[20.,30.,10.,5.,0.];a=[];b=[]
    for i in range(11):
        score=.99-i*.01
        a.append([[20.+(30. if i<2 else 0.),30.,10.,5.,0.,score]])
        b.append([[20.+(0. if i<8 else 30.),30.,10.,5.,0.,score]])
    first=t.ap.evaluate_images(a,[[gt] for _ in a],.5)
    second=t.ap.evaluate_images(b,[[gt] for _ in b],.5)
    assert second['tp']==8 and first['tp']==9 and second['ap']>first['ap']
