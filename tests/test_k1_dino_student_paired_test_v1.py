import hashlib
import json
import pickle

import numpy as np
import pytest

from crane_project.tools.audit_k1_dino_student_paired_test_v1 import (
    build_report, directory_sha256, markdown)
from crane_project.tools.eval_crane_offline import METRIC_PROTOCOL_VERSION


def _write_arm(root, label, names, boxes, gt_sha):
    pred_dir = root / label / 'Task1_grab'
    pred_dir.mkdir(parents=True)
    results = []
    for name, box in zip(names, boxes):
        if box is None:
            results.append([np.zeros((0, 6), dtype=np.float32)])
            (pred_dir / name).write_text('')
        else:
            x, y = box
            results.append([np.array([[x, y, 10, 10, 0, 0.9]],
                                     dtype=np.float32)])
            (pred_dir / name).write_text(
                '{} {} {} {} {} {} {} {} grab 0\n'.format(
                    x - 5, y - 5, x + 5, y - 5,
                    x + 5, y + 5, x - 5, y + 5))
    pkl_path = root / label / 'results.pkl'
    with pkl_path.open('wb') as stream:
        pickle.dump(results, stream)
    report_path = root / label / 'final_test_metrics_v2.json'
    report_path.write_text(json.dumps(dict(
        protocol='crane_ckpt_sweep_final_test_v2',
        metric_protocol_version=METRIC_PROTOCOL_VERSION,
        frame_count=len(names), center_thresh_px=15.0,
        results_pkl_sha256=hashlib.sha256(pkl_path.read_bytes()).hexdigest(),
        gt_annotations_sha256=gt_sha)))
    return report_path, pkl_path, pred_dir


def test_pairing_separates_recovered_frames_from_common_frame_geometry(
        tmp_path):
    gt_dir = tmp_path / 'gt'
    gt_dir.mkdir()
    names = ['real_seq02_00001.txt', 'real_seq02_00002.txt',
             'real_seq02_00003.txt', 'real_seq02_00004.txt',
             'sim_seq09_00001.txt']
    for name in names:
        (gt_dir / name).write_text('0 0 10 0 10 10 0 10 grab 0\n')
    gt_sha = directory_sha256(sorted(gt_dir.glob('*.txt')))
    baseline = _write_arm(tmp_path, 'baseline', names,
                          [(5, 5), (5, 5), None, None, (5, 5)], gt_sha)
    student = _write_arm(tmp_path, 'student', names,
                         [(5, 5), None, (5, 5), None, (6, 5)], gt_sha)
    report = build_report(gt_dir, baseline, student, expected_frame_count=5)
    real = report['groups']['real']
    paired = real['paired']
    assert real['methods']['baseline']['output_frame_count'] == 2
    assert real['methods']['student']['output_frame_count'] == 2
    assert real['methods']['baseline']['center_hit_given_output'] == 1
    assert real['methods']['student']['center_hit_all_gt_frames'] == 0.5
    assert paired['both_output_count'] == 1
    assert paired['baseline_only_output_count'] == 1
    assert paired['student_only_output_count'] == 1
    assert paired['neither_output_count'] == 1
    assert paired['baseline_only_center_hit_count'] == 1
    assert paired['student_only_center_hit_count'] == 1
    assert paired['common_output']['riou_tie_count'] == 1
    sim = report['groups']['sim']['paired']['common_output']
    assert sim['baseline_riou_better_count'] == 1
    assert sim['student_mean_riou'] < sim['baseline_mean_riou']
    assert len(report['frames']) == 5
    rendered = markdown(report)
    assert '共同输出' in rendered
    assert '输出覆盖率' in rendered
    assert '输出后中心命中率' in rendered
    assert '全帧中心命中率' in rendered


def test_pairing_rejects_mismatched_identity_and_geometry(tmp_path):
    gt_dir = tmp_path / 'gt'
    gt_dir.mkdir()
    name = 'real_seq02_00001.txt'
    (gt_dir / name).write_text('0 0 10 0 10 10 0 10 grab 0\n')
    gt_sha = directory_sha256([gt_dir / name])
    baseline = _write_arm(tmp_path, 'baseline', [name], [(5, 5)], gt_sha)
    student = _write_arm(tmp_path, 'student', [name], [(5, 5)], gt_sha)
    (student[2] / name).write_text('100 100 110 100 110 110 100 110 grab 0\n')
    with pytest.raises(ValueError, match='geometry mismatch|center mismatch'):
        build_report(gt_dir, baseline, student, expected_frame_count=1)
    (student[2] / name).write_text('0 0 10 0 10 10 0 10 grab 0\n')
    identity = json.loads(student[0].read_text())
    identity['results_pkl_sha256'] = '0' * 64
    student[0].write_text(json.dumps(identity))
    with pytest.raises(ValueError, match='identity mismatch'):
        build_report(gt_dir, baseline, student, expected_frame_count=1)
