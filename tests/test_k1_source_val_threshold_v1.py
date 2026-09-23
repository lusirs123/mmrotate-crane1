import hashlib
import json
import pickle

import numpy as np
import pytest

from crane_project.tools.audit_k1_source_val_threshold_v1 import build_report


def test_fixed_score_gate_audit_counts_correct_and_wrong_rescues(tmp_path):
    gt_dir = tmp_path / 'gt'
    gt_dir.mkdir()
    names = (['real_seq07_%05d.txt' % i for i in range(226)] +
             ['sim_seq10_%05d.txt' % i for i in range(512)])
    normal = np.array([[5., 5., 10., 10., 0., 0.9]], dtype=np.float32)
    empty = np.zeros((0, 6), dtype=np.float32)
    baseline, permissive = [], []
    for index, name in enumerate(names):
        (gt_dir / name).write_text('0 0 10 0 10 10 0 10 grab 0\n')
        if index == 0:
            baseline.append([empty])
            permissive.append([np.array(
                [[5., 5., 10., 10., 0., 0.04]], dtype=np.float32)])
        elif index == 1:
            baseline.append([empty])
            permissive.append([np.array(
                [[500., 500., 10., 10., 0., 0.03]], dtype=np.float32)])
        else:
            baseline.append([normal])
            permissive.append([normal])
    base_path = tmp_path / 'base.pkl'
    open_path = tmp_path / 'open.pkl'
    with base_path.open('wb') as stream:
        pickle.dump(baseline, stream)
    with open_path.open('wb') as stream:
        pickle.dump(permissive, stream)
    identity_path = tmp_path / 'identity.json'
    identity_path.write_text(json.dumps(dict(
        result_count=738,
        results_sha256=hashlib.sha256(open_path.read_bytes()).hexdigest(),
        runtime_dataset_order=[dict(frame_key=name[:-4])
                               for name in names])))
    report = build_report(gt_dir, base_path, open_path, identity_path)
    real = report['groups']['real']
    assert real['baseline_missing_count'] == 2
    assert real['rescued_output_count'] == 2
    assert real['rescued_center_hit_count'] == 1
    assert real['rescued_riou_hit_count'] == 1
    assert report['fixed_test_read'] is False

    permissive[2] = [np.array(
        [[6., 5., 10., 10., 0., 0.9]], dtype=np.float32)]
    with open_path.open('wb') as stream:
        pickle.dump(permissive, stream)
    with pytest.raises(ValueError, match='changed an existing box'):
        build_report(gt_dir, base_path, open_path)
    with pytest.raises(ValueError, match='identity'):
        build_report(gt_dir, base_path, open_path, identity_path)
