"""Bounded CPU tests of evidence roles, denominators, geometry and source binding."""
from copy import deepcopy
import json
import math
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import pytest

from crane_project.tools import check_port_reliability_readiness_v1 as check
from crane_project.tools import snapshot_port_legacy_train_axes_v1 as snapshot


def row(name='real_seq07_00000', center=0., size=1., angle=0., score=.9, output=True, axis=True):
    gt = [30.,20.,40.,20.,0.]
    pred = [30.+center,20.,40.*size,20.*size,angle,score] if output else None
    return dict(image=name, pred=pred, gt=gt, sequence=name.rsplit('_',1)[0], domain='real',
                angle_axis_well_defined=axis,
                **(dict(errors=check.geometry_errors(gt,pred[:5])) if output else {}))


def test_missing_outputs_stay_in_denominator_and_rejection_costs_are_component_specific():
    rows = [row(center=2., size=1.2), row('real_seq07_00001', center=20., angle=.2),
            row('real_seq07_00002', output=False)]
    raw = check.risk_stats(rows[:2], rows)
    assert raw['output_coverage'] == pytest.approx(2/3)
    assert raw['output_center_hit_rate'] == .5
    assert raw['all_frame_center_correct_coverage'] == pytest.approx(1/3)
    accepted = check.risk_stats(rows[:1], rows)
    assert accepted['center_px']['incorrect_accepted'] == 0
    assert accepted['size_max_relative']['incorrect_accepted'] == 1
    assert accepted['angle_deg']['correct_outputs_rejected'] == 0
    assert accepted['size_max_relative']['correct_outputs_rejected'] == 1
    assert accepted['center_px']['all_frame_correct_acceptance'] == pytest.approx(1/3)


def test_ties_cannot_be_cut_by_gt_or_image_order_and_full_coverage_can_be_unreachable():
    rows = [row(center=20.), row('real_seq07_00001',center=2.), row('real_seq07_00002',output=False)]
    a = check.score_curve(rows, (.5, 2/3, 1.))
    assert a == check.score_curve(list(reversed(rows)), (.5,2/3,1.))
    assert len(a['points']) == 1 and a['points'][0]['tie_group_size'] == 2
    assert a['fixed_coverage_grid'][0]['point']['stats']['accepted_frames'] == 0
    assert a['fixed_coverage_grid'][0]['point']['stats']['accepted_center_hit_rate'] is None
    assert a['fixed_coverage_grid'][1]['point']['stats']['accepted_frames'] == 2
    assert a['fixed_coverage_grid'][2]['status'] == 'UNREACHABLE_MISSING_OUTPUT'
    assert check.score_curve([])['fixed_coverage_grid'][0]['status'] == 'NO_FRAMES'


def test_near_square_angle_is_unassessed_rather_than_correct_or_bad():
    rows = [row(axis=False,angle=.2)]
    r = check.risk_stats(rows,rows)
    assert r['angle_deg']['assessable_accepted'] == 0
    assert r['angle_deg']['unassessed_accepted'] == 1
    assert r['angle_deg']['conditional_error_rate'] is None
    assert r['angle_deg']['all_frame_correct_acceptance'] == 0


def test_component_geometry_handles_width_exchange_and_pi_periodicity():
    g = [10.,20.,60.,30.,.3]
    equivalent = [10.,20.,30.,60.,.3+math.pi/2]
    e = check.geometry_errors(g,equivalent)
    assert all(abs(v) < 1e-12 for v in e.values())
    assert check.corner_difference(check.box_polygon(g), check.box_polygon(equivalent)) < 1e-12
    assert check.geometry_errors(g,[10.,20.,60.,30.,.3+math.pi])['angle_deg'] < 1e-12
    with pytest.raises(ValueError): check.canonical([0.,0.,-2.,3.,0.])
    with pytest.raises(ValueError): check.canonical([0.,float('nan'),2.,3.,0.])


def test_native_endpoint_order_is_unordered_and_ratio_is_sequence_specific():
    axis = np.array([[0.,0.],[27.,0.]])
    b = [13.5,0.,27.,10.,0.]
    a = check.axis_consistency(axis,b,2.7)
    assert a['conversion_consistent']
    assert check.axis_consistency(axis[::-1],b,2.7)['errors'] == a['errors']
    assert not check.axis_consistency(axis,b,2.1)['conversion_consistent']
    with pytest.raises(ValueError): check.axis_consistency(axis,b,float('nan'))


def axis_json(path, name='real_seq01_00000', image_path='seq01_00000.jpg'):
    d = dict(imageWidth=100,imageHeight=60,imagePath=image_path,
             shapes=[dict(label='axis',shape_type='line',points=[[10.,20.],[64.,20.]],flags={})])
    path.write_text(json.dumps(d)); return d


def test_legacy_alias_is_limited_and_resolution_or_bad_axis_rejected(tmp_path):
    p = tmp_path/'real_seq01_00000.json'; d = axis_json(p)
    check.read_axis(p,p.stem,(100,60),legacy=True)
    with pytest.raises(ValueError,match='identity'): check.read_axis(p,p.stem,(100,60))
    with pytest.raises(ValueError,match='resolution'): check.read_axis(p,p.stem,(99,60),legacy=True)
    d['imagePath'] = 'seq14_00000.jpg'; p.write_text(json.dumps(d))
    with pytest.raises(ValueError): check.read_axis(p,p.stem,(100,60),legacy=True)
    d = axis_json(p); d['shapes'][0]['points'][1] = d['shapes'][0]['points'][0]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='points'): check.read_axis(p,p.stem,(100,60),legacy=True)


def test_difficulty_is_retained_and_bowtie_polygon_rejected(tmp_path):
    p = tmp_path/'label.txt'; p.write_text('10 10 50 10 50 30 10 30 grab 1\n')
    poly,b = check.polygon_box(p)
    assert b[:4] == pytest.approx([30,20,40,20])
    p.write_text('10 10 50 30 50 10 10 30 grab 0\n')
    with pytest.raises(ValueError,match='nonconvex'): check.polygon_box(p)


def cache_fixture(tmp_path):
    name='real_seq07_00000'; a=tmp_path/(name+'.txt')
    a.write_text('10 10 50 10 50 30 10 30 grab 0\n')
    im=tmp_path/(name+'.jpg'); Image.new('RGB',(100,60)).save(im)
    r=row(name,center=2.); e=r['errors']
    r.update(split='val',frame_id=0,image_sha256=check.sha(im),metrics=dict(output=True,center_hit=True,
        angle_axis_well_defined=True, center_error_px=e['center_px'], long_edge_relative_error=e['long_relative'],
        short_edge_relative_error=e['short_relative'], long_edge_signed_log_ratio=e['long_signed_log'],
        short_edge_signed_log_ratio=e['short_signed_log'], angle_error_deg=e['angle_deg']))
    return r,a,im


@pytest.mark.parametrize('mutation,match',[
    (lambda r:r.update(split='test'),'role'),
    (lambda r:r.update(frame_id=1),'identity'),
    (lambda r:r['metrics'].update(center_error_px=1.),'metric'),
    (lambda r:r.update(pred=None),'presence'),
    (lambda r:r.update(image_sha256='wrong'),'image identity'),
    (lambda r:r['gt'].__setitem__(0,35.),'GT differs')])
def test_cache_role_identity_and_metric_corruption_are_rejected(tmp_path,mutation,match):
    r,a,_=cache_fixture(tmp_path)
    validated=check.validate_cached_rows([r],[a],tmp_path,dict(real_seq07=1))
    assert validated[0]['errors']['center_px'] == 2.
    mutation(r)
    with pytest.raises(ValueError,match=match): check.validate_cached_rows([r],[a],tmp_path,dict(real_seq07=1))


def test_duplicate_frame_and_missing_native_cache_cannot_silently_pass(tmp_path):
    r,a,_=cache_fixture(tmp_path)
    with pytest.raises(ValueError,match='Duplicate'):
        check.validate_cached_rows([r,r],[a],tmp_path,dict(real_seq07=2))
    identity=dict(checkpoint=str(tmp_path/'missing.pth'),pkl=str(tmp_path/'sweep/epoch_24/preds/results.pkl'))
    result=check.native_cache_checks(identity,[r])
    assert result['pkl']['status']=='UNAVAILABLE_REPORT_DERIVED'
    with pytest.raises(FileNotFoundError):check.native_cache_checks(identity,[r],require_native=True)


def test_native_pkl_rows_verified_after_hash_and_mismatches_fail(tmp_path,monkeypatch):
    r,_,_=cache_fixture(tmp_path)
    p=tmp_path/'sweep/epoch_24/preds/results.pkl';p.parent.mkdir(parents=True)
    p.write_bytes(pickle.dumps([[np.array([r['pred']])]]))
    frozen=dict(check.FROZEN_B,pkl_sha256=check.sha(p));monkeypatch.setattr(check,'FROZEN_B',frozen)
    identity=dict(pkl=str(p),checkpoint=str(tmp_path/'missing.pth'))
    verified=check.native_cache_checks(identity,[r])['pkl']
    assert verified['row_correspondence']=='ALL_ROWS_VERIFIED' and verified['verified_frames']==1
    r['pred'][0]+=1.
    with pytest.raises(ValueError,match='correspondence'):check.native_cache_checks(identity,[r])
    frozen['pkl_sha256']='wrong'
    with pytest.raises(ValueError,match='hash'):check.native_cache_checks(identity,[r])


def test_spearman_ties_and_constant_features():
    assert check.rank_association([1,1,2,3],[3,3,2,1]) == pytest.approx(-1.)
    assert check.rank_association([1,1,1],[1,2,3]) is None


def test_snapshot_is_train_only_preserves_json_bytes_and_refuses_overwrite(tmp_path,monkeypatch):
    data=tmp_path/'data';source=tmp_path/'raw';source.mkdir()
    (data/'train/images').mkdir(parents=True);(data/'train/annfiles').mkdir()
    name='real_seq01_00000';p=source/(name+'.json');axis_json(p)
    # This file would fail parsing if the snapshot ever opened a non-TRAIN role.
    (source/'real_seq04_00000.json').write_text('forbidden TEST fixture')
    Image.new('RGB',(100,60)).save(data/'train/images'/(name+'.jpg'))
    (data/'train/annfiles'/(name+'.txt')).write_text('10 10 64 10 64 30 10 30 grab 0\n')
    monkeypatch.setattr(snapshot,'LEGACY_TRAIN',('real_seq01',))
    monkeypatch.setattr(snapshot,'TRAIN_COUNTS',dict(real_seq01=1))
    m=snapshot.snapshot(source,data)
    q=data/'provenance/axis_legacy_train_v1/real_seq01/axis_json'/p.name
    assert q.read_bytes()==p.read_bytes() and len(m['rows'])==1
    with pytest.raises(FileExistsError):snapshot.snapshot(source,data)


def test_existing_output_refused_before_source_access(tmp_path,monkeypatch):
    monkeypatch.setattr(check,'check_cache',lambda *a:pytest.fail('Cannot touch input on overwrite'))
    with pytest.raises(FileExistsError):check.main(['--out-dir',str(tmp_path)])


def test_runtime_import_does_not_load_model_or_gpu_modules():
    import subprocess,sys
    p=subprocess.run([sys.executable,'-c',
        'import sys; import crane_project.tools.check_port_reliability_readiness_v1; '
        'assert not any(k in sys.modules for k in ["torch","mmcv","mmrotate","mmdet"])'],
        cwd=str(check.ROOT),capture_output=True,text=True)
    assert p.returncode==0,p.stderr
