"""CPU regression for additive cached-output scoring; fixtures are not evidence."""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from crane_project.tools import eval_port_reliability_branches_v1_cached_val as repair
from crane_project.utils.port_reliability_train_policy_v1 import POLICY


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def meta(sx=1., sy=1.):
    return dict(scale_factor=[sx, sy, sx, sy], ori_shape=(1024, 1024, 3),
        img_shape=(int(1024*sy), int(1024*sx), 3), pad_shape=(1024, 1024, 3), flip=False)


def test_differences_expose_fields_without_relaxing_original_check():
    expected = [[32., 30., 24., 12., .1, .7]]
    changed = deepcopy(expected)
    changed[0][4] += .0002
    changed[0][5] += .000001
    delta = repair.prediction_difference(changed, expected)
    assert not delta['coordinates_close'] and not delta['all_six_close']
    assert delta['score_close'] and not delta['score_exact']
    assert delta['abs_delta_by_field']['angle_rad'] == pytest.approx(.0002)
    assert delta['geometry_difference_relative_to_cached_prediction']['center_px'] == 0
    mismatch = repair.prediction_difference(None, expected)
    assert not mismatch['same_shape'] and mismatch['abs_delta_by_field'] is None
    assert repair.prediction_difference(None, None)['all_six_close']
    assert (repair.ATOL, repair.RTOL) == (1e-4, 1e-6)


def test_cache_inputs_preserve_raw_size_association_scores_and_missing_rows():
    # w<h must stay associated with sx/sy until detector scaling completes.
    pred = [40., 32., 12., 24., -.3, .71]
    image = torch.zeros(1, 3, 32, 32)
    m = meta(.5, .5001)
    boxes, scores, delta = repair.cached_model_inputs(pred, image, m)
    assert boxes[0, 2] == pytest.approx(6.)
    assert boxes[0, 3] == pytest.approx(24*.5001)
    assert scores[0] == pytest.approx(.71) and delta < 1e-4
    empty, score_empty, delta_empty = repair.cached_model_inputs(None, image, m)
    assert empty.shape == (0, 5) and score_empty.shape == (0,) and delta_empty == 0
    with pytest.raises(ValueError):
        repair.cached_model_inputs([40., 32., -1., 24., 0., .71], image, m)


def test_diagnostic_separates_cache_wrapper_and_exact_score_predicates():
    class Fake:
        def extract_feat(self, image):
            return [torch.zeros(1, 256, 8, 8)]

        def simple_test_from_features(self, features, metas, rescale):
            return [[np.array([[32., 30., 24., 12., .1, .7 if rescale else .6]])]]

    source = dict(image=repair.DIAGNOSE_IMAGE, image_sha256='fixture',
                  pred=[32., 30., 24., 12., .1, .7])
    result = repair.diagnostic(Fake(), torch.zeros(1, 3, 8, 8), [meta()], source)
    assert result['failed_original_predicates'] == ['raw_vs_native_score_exactness']
    assert result['native_vs_cache']['all_six_close']
    assert result['wrapper_vs_native']['coordinates_close']
    assert result['repeat_raw_exact'] and result['repeat_native_exact']


def bundle_fixture(tmp_path, frozen):
    training_sources, protocol, repair_sources = repair.checked_sources()
    arms = repair.branch.make_arms()
    contract = dict(role='formal_train', sources=training_sources, protocol=protocol,
        direction_policy=POLICY, frozen_b=repair.ready.FROZEN_B, frozen_b_state=frozen,
        numeric_input_snapshot_sha256=repair.train.SNAPSHOT_SHA,
        train_input_manifest_sha256='2094bb4cde19dd97909bcd02b36db19c09c6d5c348c0b246e9b159a68c1b4022',
        architecture=repair.branch.architecture(arms))
    path = tmp_path/'epoch_08.pth'
    repair.branch.save_checkpoint(path, arms, repair.branch.make_optimizers(arms),
        contract, 8, 20464, frozen, 'synthetic_fixture_not_real_training')
    log = tmp_path/'train_steps.jsonl'
    log.write_text('synthetic CPU test only\n')
    repair.branch.write_new(tmp_path/'completion.json', dict(
        status='PAIRED_FORMAL_TRAIN_COMPLETE_FIXED_EPOCH8', sources=training_sources, protocol=protocol,
        checkpoint_sha256=repair.branch.sha(path), optimizer_steps_per_arm=20464,
        save_reload_quality_exact=True, b_raw_unchanged=True, detector_optimizer_steps=0,
        b_state=frozen, train_log_sha256=repair.branch.sha(log)))
    return path, training_sources, protocol, repair_sources


def test_original_training_contract_is_still_required(tmp_path):
    path, sources, protocol, _ = bundle_fixture(tmp_path, dict(fixture=True))
    assert repair.base.fixed_bundle(path, sources, protocol)['epoch'] == 8
    forged = dict(sources, manifest_sha256='wrong')
    with pytest.raises(ValueError, match='another reviewed'):
        repair.base.fixed_bundle(path, forged, protocol)


def test_cached_scoring_finishes_without_skipping_mismatch_or_missing_frame(tmp_path, monkeypatch):
    """Full orchestration on two fixtures, strict original checkpoint loader."""
    class FakeDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
            self.eval()

        def extract_feat(self, image):
            return [torch.zeros(1, 256, 128, 128)]

        def simple_test_from_features(self, features, metas, rescale):
            # Deliberately differs from the locked B cache in the failing frame.
            return [[np.array([[33., 30., 24., 12., .1, .7]], dtype=np.float32)]]

    detector = FakeDetector()
    frozen = repair.prior.state_digest(detector)
    path, sources, protocol, repair_sources = bundle_fixture(tmp_path, frozen)
    gt = [32., 30., 24., 12., .1]
    rows = [dict(image='real_seq07_00001', sequence='real_seq07', domain='real', frame_id=1,
                 image_sha256='fixture_real', gt=gt, pred=None, angle_axis_well_defined=True),
            dict(image=repair.DIAGNOSE_IMAGE, sequence='sim_seq10', domain='sim', frame_id=220,
                 image_sha256='fixture_sim', gt=gt, pred=gt+[.7], angle_axis_well_defined=True,
                 errors=repair.ready.geometry_errors(gt, gt))]
    identity = dict(synthetic=True)
    monkeypatch.setattr(repair.train, 'build_runtime', lambda *args: (detector, dict(synthetic_cpu=True)))
    monkeypatch.setattr(repair, 'val_dataset', lambda *args: object())
    monkeypatch.setattr(repair, 'val_view', lambda *args: (torch.zeros(1, 3, 32, 32), [meta()]))
    monkeypatch.setattr(repair.base, 'val_cache', lambda *args, **kwargs: (rows, identity))
    original_make = repair.branch.make_arms
    monkeypatch.setattr(repair.branch, 'make_arms', lambda *args, **kwargs: original_make('cpu'))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *args: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *args: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda *args: 0)
    from argparse import Namespace
    args = Namespace(mode='score-cache', branch_checkpoint=path, b_checkpoint=Path('fixture'),
        b_report=Path('fixture'), gpu=0, out_dir=tmp_path/'evaluation')
    checkpoint_sha = repair.branch.sha(path)
    repair.run(args, rows, identity, sources, protocol, repair_sources, None)
    result = json.loads((args.out_dir/'val_compare.json').read_text())
    saved = [json.loads(line) for line in (args.out_dir/'val_qualities.jsonl').read_text().splitlines()]
    assert len(saved) == 2 and saved[0]['pred'] is None
    assert saved[1]['pred'] == rows[1]['pred']  # mismatch not substituted/skipped
    assert saved[0]['genuine_online_qualities'] == {name: [] for name in repair.branch.ARMS}
    assert result['strata']['domain:real']['output_coverage'] == 0
    assert result['strata']['domain:sim']['all_frame_center_correct_coverage'] == 1
    assert result['single_frame_runtime_diagnosis']['failed_original_predicates'] == ['runtime_native_vs_locked_cache']
    assert result['fresh_runtime_B_reproduces_all_887_cached_predictions'] is None
    assert repair.branch.sha(path) == checkpoint_sha
    assert detector.weight.grad is None and not result['test_read']
