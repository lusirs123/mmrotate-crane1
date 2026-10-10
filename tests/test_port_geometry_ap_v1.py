"""Synthetic AP correctness/contract tests, never measured model performance."""
from copy import deepcopy
import json
import math
import sys
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest

from crane_project.tools import eval_port_geometry_ap_v1 as a


GT = [20., 30., 10., 5., 0.]


def pred(score=.9, shift=0.):
    return [20. + shift, 30., 10., 5., 0., score]


def rows():
    result = []
    for seq, n in sorted(a.COUNTS.items()):
        start = int(seq == 'real_seq07')
        for i in range(start, n + start):
            result.append(dict(image='%s_%05d' % (seq, i), sequence=seq, frame_id=i,
                               domain=seq.split('_')[0], scale=1., gt=GT[:],
                               b=pred(), midpoint=pred(), accepted=True,
                               original_b_raw_exact_before_after=True))
    return result


def test_exact_and_empty_outputs_and_bad_iou():
    exact = a.evaluate_images([[pred()]], [[GT]], .75)
    assert exact['ap'] == 1. and exact['tp'] == 1 and exact['fp'] == 0
    empty = a.evaluate_images([[]], [[GT]], .5)
    assert empty['ap'] == 0. and empty['recall'] == [] and empty['num_gts'] == 1
    fail = a.evaluate_images([[pred(shift=30.)]], [[GT]], .5)
    assert fail['ap'] == 0. and fail['tp'] == 0 and fail['fp'] == 1


def test_thresholds_not_mean_iou_and_missing_gt_denominator():
    # shift=2: exact axis-aligned IoU=(8*5)/(2*50-40)=2/3.
    assert a.evaluate_images([[pred(shift=2.)]], [[GT]], .5)['ap'] == 1.
    assert a.evaluate_images([[pred(shift=2.)]], [[GT]], .75)['ap'] == 0.
    # Recall=0.5 -> six eligible levels {0,.1,.2,.3,.4,.5} -> AP=6/11.
    missed = a.evaluate_images([[pred()], []], [[GT], [GT]], .5)
    assert missed['num_gts'] == 2 and missed['tp'] == 1
    assert missed['ap'] == pytest.approx(6 / 11, abs=1e-7)


def test_duplicate_and_score_order_and_wrong_image_matching():
    duplicate = a.evaluate_images([[pred(.9), pred(.8)]], [[GT]], .5)
    assert duplicate['image_matches'] == [dict(tp=[1, 0], fp=[0, 1])]
    assert duplicate['ap'] == 1.
    high_fp = a.evaluate_images([[pred(.9, 30)], [pred(.8)]], [[GT], [GT]], .5)
    low_fp = a.evaluate_images([[pred(.7, 30)], [pred(.8)]], [[GT], [GT]], .5)
    assert high_fp['ap'] == pytest.approx(3 / 11, abs=1e-7)
    assert low_fp['ap'] == pytest.approx(6 / 11, abs=1e-7)
    # A detection cannot match an otherwise identical GT in another image.
    foreign = a.evaluate_images([[pred()], []], [[], [GT]], .5)
    assert foreign['tp'] == 0 and foreign['fp'] == 1


def test_multi_gt_and_strict_iou_boundary():
    other = [60., 30., 10., 5., 0.]
    p = other + [.7]
    assert a.evaluate_images([[pred(), p]], [[GT, other]], .75)['tp'] == 2
    # Native comparison is >=; containment gives IoU exactly 0.5.
    small = [20., 30., 5., 5., 0., .9]
    assert a.evaluate_images([[small]], [[GT]], .5)['tp'] == 1
    with pytest.raises(ValueError, match='without GT'):
        a.evaluate_images([[]], [[]], .5)
    with pytest.raises(ValueError, match='Image count'):
        a.evaluate_images([], [[GT]], .5)


def test_rotated_iou_not_aabb_and_swapped_axes():
    vertical = [20., 30., 5., 10., 0.]
    assert a.compute_riou(GT, vertical) == pytest.approx(1 / 3)
    equivalent = [20., 30., 5., 10., -math.pi / 2]
    assert a.compute_riou(GT, equivalent) == pytest.approx(1.)
    theta = -.6
    rotated = GT[:4] + [theta]
    assert a.compute_riou(rotated, rotated) == pytest.approx(1., abs=1e-6)


def test_original_gt_polygon_enclosure_and_minimum_area():
    poly = np.array([[15., 27.5], [25., 27.5], [25., 32.5], [15., 32.5]])
    a.validate_saved_gt(GT, poly, np.asarray(GT))
    with pytest.raises(ValueError, match='minimum rectangle'):
        a.validate_saved_gt(GT[:2] + [11., 5., 0.], poly, np.asarray(GT))
    with pytest.raises(ValueError, match='minimum rectangle'):
        a.validate_saved_gt([21., 30.] + GT[2:], poly, np.asarray(GT))


def test_full_contract_and_coverage_denominators():
    r = rows(); r[0]['b'] = r[0]['midpoint'] = None; r[0]['accepted'] = None
    a.validate_rows(r)
    real = [x for x in r if x['domain'] == 'real']
    c = a.coverage(real, 'b')
    assert c['output_center_hit_percent'] == 100.
    assert c['output_frames'] == c['center_correct_frames'] == 374
    assert c['output_coverage_percent'] == c['full_frame_center_correct_percent']
    # center threshold is strict <15, output does not count as a miss.
    real[1]['b'] = pred(shift=15)
    c = a.coverage(real, 'b')
    assert c['center_correct_frames'] == 373 and c['output_frames'] == 374


@pytest.mark.parametrize('kind', ['drop', 'duplicate', 'score', 'scale', 'fallback', 'test', 'order', 'nan'])
def test_contract_fails_closed(kind):
    r = rows()
    if kind == 'drop': r.pop()
    if kind == 'duplicate': r[-1] = deepcopy(r[-2])
    if kind == 'score': r[0]['midpoint'][5] -= .01
    if kind == 'scale': r[0]['scale'] = .5
    if kind == 'fallback': r[0]['accepted'] = False; r[0]['midpoint'][0] += 1
    if kind == 'test': r[0]['image'] = 'real_seq03_00000'
    if kind == 'order': r.reverse()
    if kind == 'nan': r[0]['gt'][0] = float('nan')
    with pytest.raises(ValueError): a.validate_rows(r)


@pytest.mark.parametrize('value', [GT[:4], GT[:2] + [0., 5., 0.],
                                   GT[:4] + [math.pi / 2], [True] + GT[1:],
                                   GT[:4] + [float('inf')]])
def test_invalid_box(value):
    with pytest.raises(ValueError): a.box(value)


def test_nonfinite_json_and_artifact_mismatch(tmp_path):
    p = tmp_path / 'x.json'; p.write_text('{"x": NaN}')
    with pytest.raises(ValueError, match='Non-finite'): a.read_json(p)
    for name in a.REQUIRED: (tmp_path / name).write_text('{}')
    a.write_json(tmp_path / 'artifacts.json', dict(files={n: 'wrong' for n in a.REQUIRED}))
    with pytest.raises(ValueError, match='artifact SHA'):
        a.checked_inputs(tmp_path, tmp_path)


def test_failed_run_leaves_no_directory_and_existing_output_preserved(tmp_path, monkeypatch):
    out = tmp_path / 'out'
    args = SimpleNamespace(eval_dir=tmp_path, ann_dir=tmp_path, out_dir=out, backend='cpu', check_only=True)
    with pytest.raises(FileNotFoundError): a.run(args)
    assert not out.exists()
    monkeypatch.setattr(a, 'checked_inputs', lambda *args: (rows(), {}))
    out.mkdir(); marker = out / 'keep'; marker.write_text('do not overwrite')
    with pytest.raises(FileExistsError): a.run(args)
    assert marker.read_text() == 'do not overwrite'


def test_cpu_cli_and_native_option_do_not_silently_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(a, 'checked_inputs', lambda *args: (rows(), {}))
    out = tmp_path / 'cpu'
    args = SimpleNamespace(eval_dir=tmp_path, ann_dir=tmp_path, out_dir=out, backend='cpu', check_only=False)
    a.run(args)
    report = a.read_json(out / 'ap_report.json')
    assert report['groups']['real']['b']['AP50']['ap'] == 1.
    assert report['test_access'] is False
    assert report['status'] == 'VAL_AP_CPU_COMPLETE_NATIVE_CHECK_PENDING'
    def fail(*args): raise RuntimeError('synthetic native failure')
    monkeypatch.setattr(a, 'verify_native', fail)
    args.backend = 'mmrotate'; args.out_dir = tmp_path / 'native'
    with pytest.raises(RuntimeError, match='native failure'): a.run(args)
    assert not args.out_dir.exists()


@pytest.mark.parametrize('corruption', [None, 'ap', 'recall', 'shape'])
def test_native_adapter_contract_without_claiming_real_mmcv(corruption, monkeypatch):
    ds, gs = [[pred()]], [[GT]]
    ref = a.evaluate_images(ds, gs, .75)
    def stub(detections, annotations, **kwargs):
        assert detections[0][0].shape == (1, 6)
        assert annotations[0]['labels'].tolist() == [0]
        assert annotations[0]['bboxes_ignore'].shape == (0, 5)
        assert kwargs == dict(iou_thr=.75, use_07_metric=True, logger='silent', nproc=1)
        result = dict(num_gts=1, num_dets=1, recall=np.array([1.]), precision=np.array([1.]))
        if corruption == 'recall': result['recall'] = np.array([.5])
        if corruption == 'shape': result['precision'] = np.array([[1.]])
        return (.8 if corruption == 'ap' else 1.), [result]
    mod = ModuleType('mmrotate.core.evaluation.eval_map'); mod.eval_rbbox_map = stub
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    if corruption is None:
        assert a.verify_native(ds, gs, .75, ref)['pass_check'] is True
    else:
        with pytest.raises(ValueError, match='Native/reference'):
            a.verify_native(ds, gs, .75, ref)
