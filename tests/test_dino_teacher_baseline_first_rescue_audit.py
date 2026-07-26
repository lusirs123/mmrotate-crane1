import argparse

import numpy as np
import pytest

from crane_project.tools import dino_teacher_baseline_first_rescue_audit as audit


def _metrics(hit=False, deployed=False, count=0, score=None, riou=0.0):
    return dict(
        detection_count=count, top1_hit=hit,
        deployment_top1_hit=deployed, top1_score=score, top1_riou=riou)


def _combined_row(frame, baseline, strict, ranked, override=None):
    override = ranked if override is None else override
    return dict(
        seq='seq', frame=frame, baseline_active=False,
        dino_top1_metrics=ranked,
        policies=dict(
            baseline=dict(source='baseline', preserved=True,
                          metrics=baseline),
            strict=dict(source=('dino_rescue' if strict['detection_count']
                                else 'silence'), preserved=True,
                        metrics=strict),
            ranked=dict(source=('dino_rescue' if ranked['detection_count']
                                else 'silence'), preserved=True,
                        metrics=ranked),
            confident_override=dict(
                source=('dino_rescue' if override['detection_count']
                        else 'silence'), preserved=True,
                metrics=override),
            scoped_override=dict(
                source=('dino_rescue' if override['detection_count']
                        else 'silence'), scope_enabled=True,
                preserved=True, metrics=override)))


def test_baseline_first_never_overwrites_existing_detection():
    baseline = np.asarray([[1, 2, 3, 4, 0.1, 0.06]], dtype=np.float32)
    dino = np.asarray([[9, 8, 7, 6, 0.2, 0.99]], dtype=np.float32)
    selected, source = audit.choose_baseline_first(baseline, dino, 0.05)
    assert source == 'baseline'
    assert np.array_equal(selected, baseline)
    assert selected is not baseline


def test_strict_rescue_rejects_low_score_but_ranked_reports_upper_bound():
    baseline = np.zeros((0, 6), dtype=np.float32)
    dino = np.asarray([[1, 2, 3, 4, 0.1, 0.02]], dtype=np.float32)
    strict, strict_source = audit.choose_baseline_first(
        baseline, dino, audit.DINO_DEPLOYMENT_SCORE_THR)
    ranked, ranked_source = audit.choose_baseline_first(baseline, dino, None)
    assert strict.shape == (0, 6)
    assert strict_source == 'silence'
    assert ranked_source == 'dino_rescue'
    assert np.array_equal(ranked, dino)


def test_confident_override_uses_dino_only_above_fixed_threshold():
    baseline = np.asarray([[1, 2, 3, 4, 0.1, 0.8]], dtype=np.float32)
    confident_dino = np.asarray(
        [[9, 8, 7, 6, 0.2, 0.06]], dtype=np.float32)
    low_dino = confident_dino.copy()
    low_dino[0, 5] = 0.02
    selected, source = audit.choose_ranked_confident_override(
        baseline, confident_dino)
    assert source == 'dino_override'
    assert np.array_equal(selected, confident_dino)
    selected, source = audit.choose_ranked_confident_override(
        baseline, low_dino)
    assert source == 'baseline'
    assert np.array_equal(selected, baseline)


def test_confident_override_keeps_low_score_ranked_rescue_on_silence():
    baseline = np.zeros((0, 6), dtype=np.float32)
    low_dino = np.asarray([[1, 2, 3, 4, 0.1, 0.02]], dtype=np.float32)
    selected, source = audit.choose_ranked_confident_override(
        baseline, low_dino)
    assert source == 'dino_rescue'
    assert np.array_equal(selected, low_dino)


def test_scoped_override_is_exact_baseline_fallback_when_disabled():
    baseline = np.asarray([[1, 2, 3, 4, 0.1, 0.8]], dtype=np.float32)
    dino = np.asarray([[9, 8, 7, 6, 0.2, 0.99]], dtype=np.float32)
    selected, source = audit.choose_scoped_confident_override(
        baseline, dino, dino_enabled=False)
    assert source == 'baseline_scope_disabled'
    assert np.array_equal(selected, baseline)


def test_scoped_override_uses_dino_only_when_enabled():
    baseline = np.asarray([[1, 2, 3, 4, 0.1, 0.8]], dtype=np.float32)
    dino = np.asarray([[9, 8, 7, 6, 0.2, 0.06]], dtype=np.float32)
    selected, source = audit.choose_scoped_confident_override(
        baseline, dino, dino_enabled=True)
    assert source == 'dino_override'
    assert np.array_equal(selected, dino)


def test_scope_manifest_covers_records_and_rejects_target_derived_source(
        tmp_path):
    path = tmp_path / 'scope.json'
    path.write_text(
        '{"scope_source":"camera_mode","entries":['
        '{"split":"test","seq":"seq","start":1,"end":2,'
        '"dino_enabled":true}]}', encoding='utf-8')
    records = [
        dict(split='test', seq='seq', frame=1),
        dict(split='test', seq='seq', frame=2)]
    manifest = audit.load_scope_manifest(str(path), records)
    assert manifest['source'] == 'camera_mode'
    assert manifest['values'][('test', 'seq', 1)] is True
    path.write_text(
        '{"scope_source":"target_labels","entries":[]}',
        encoding='utf-8')
    with pytest.raises(ValueError, match='target labels'):
        audit.load_scope_manifest(str(path), records)


def test_combine_rows_preserves_active_baseline_and_rescues_only_silence(
        monkeypatch):
    monkeypatch.setattr(
        audit.labeller, 'parse_original_gt',
        lambda _path: np.zeros((0, 5), dtype=np.float32))

    def fake_metrics(detections, _gt, _riou, deployment_score_thr):
        count = int(detections.shape[0])
        score = float(detections[0, 5]) if count else None
        return dict(
            detection_count=count, top1_hit=False,
            deployment_top1_hit=bool(
                count and score >= deployment_score_thr),
            top1_score=score, top1_riou=0.0)

    monkeypatch.setattr(audit.labeller, 'ranked_detection_metrics', fake_metrics)
    records = [
        dict(split='test', seq='real_seq02', frame=137, annotation='a'),
        dict(split='test', seq='real_seq02', frame=138, annotation='b')]
    baseline_detection = [1, 2, 3, 4, 0.1, 0.06]
    baseline_rows = [
        dict(seq='real_seq02', frame=137,
             detections=[baseline_detection]),
        dict(seq='real_seq02', frame=138, detections=[])]
    dino_rows = [
        dict(seq='real_seq02', frame=137,
             detections=[[9, 9, 9, 9, 0.2, 0.99]]),
        dict(seq='real_seq02', frame=138,
             detections=[[5, 6, 7, 8, 0.3, 0.02]])]
    rows = audit.combine_rows(baseline_rows, dino_rows, records)
    assert rows[0]['policies']['strict']['source'] == 'baseline'
    assert np.array_equal(
        np.asarray(rows[0]['policies']['strict']['detections'],
                   dtype=np.float32),
        np.asarray([baseline_detection], dtype=np.float32))
    assert rows[1]['policies']['strict']['source'] == 'silence'
    assert rows[1]['policies']['ranked']['source'] == 'dino_rescue'
    assert rows[0]['policies']['confident_override']['source'] == (
        'dino_override')
    assert rows[1]['policies']['confident_override']['source'] == (
        'dino_rescue')
    assert rows[0]['policies']['scoped_override']['source'] == (
        'dino_override')


def test_normalize_baseline_result_requires_one_image_and_one_class():
    empty = audit.normalize_baseline_result([[np.zeros((0, 6))]])
    assert empty.shape == (0, 6)
    with pytest.raises(RuntimeError, match='one-image'):
        audit.normalize_baseline_result([])
    with pytest.raises(RuntimeError, match='one-class'):
        audit.normalize_baseline_result([[np.zeros((0, 6)),
                                          np.zeros((0, 6))]])


def test_summary_and_decision_keep_ranked_result_diagnostic_only():
    rows = [
        _combined_row(
            1, _metrics(), _metrics(),
            _metrics(True, False, 1, 0.02)),
        _combined_row(
            2, _metrics(), _metrics(),
            _metrics(True, True, 1, 0.06)),
    ]
    summary = audit.summarize_combination(rows)
    assert summary['strict']['top1_hits'] == 0
    assert summary['ranked']['top1_hits'] == 2
    assert summary['ranked']['deployment_top1_hits'] == 1
    assert summary['routing_diagnostics'][
        'baseline_silent_dino_available_count'] == 2
    assert summary['routing_diagnostics'][
        'baseline_silent_dino_above_threshold_count'] == 1

    source = {
        name: dict(top1_hits=2, top1_mcml=0,
                   baseline_preservation_failures=0, mean_top1_riou=0.8)
        for name in ('baseline', 'strict', 'ranked', 'confident_override')}
    source['routing_diagnostics'] = dict(
        baseline_correct_overridden_to_incorrect_count=0)
    target = {
        'baseline': dict(top1_hits=0, top1_mcml=33,
                         baseline_preservation_failures=0,
                         mean_top1_riou=0.0),
        'strict': dict(top1_hits=15, top1_mcml=10,
                       baseline_preservation_failures=0,
                       mean_top1_riou=0.3),
        'ranked': dict(top1_hits=32, top1_mcml=1,
                       baseline_preservation_failures=0,
                       mean_top1_riou=0.6),
        'confident_override': dict(
            top1_hits=15, top1_mcml=10,
            baseline_preservation_failures=1, mean_top1_riou=0.3)}
    assert audit.make_decision(source, target) == (
        'RANKED_RESCUE_UPPER_BOUND_ONLY')


def test_decision_authorizes_only_strict_policy():
    source = {
        name: dict(top1_hits=10, top1_mcml=0,
                   baseline_preservation_failures=0, mean_top1_riou=0.8)
        for name in ('baseline', 'strict', 'ranked', 'confident_override')}
    source['routing_diagnostics'] = dict(
        baseline_correct_overridden_to_incorrect_count=0)
    target = {
        'baseline': dict(top1_hits=0, top1_mcml=33,
                         baseline_preservation_failures=0,
                         mean_top1_riou=0.0),
        'strict': dict(top1_hits=26, top1_mcml=5,
                       baseline_preservation_failures=0,
                       mean_top1_riou=0.5),
        'ranked': dict(top1_hits=32, top1_mcml=1,
                       baseline_preservation_failures=0,
                       mean_top1_riou=0.6),
        'confident_override': dict(
            top1_hits=25, top1_mcml=6,
            baseline_preservation_failures=1, mean_top1_riou=0.5)}
    assert audit.make_decision(source, target) == (
        'STRICT_BASELINE_FIRST_DINO_RESCUE_PASSES')


def test_decision_accepts_confident_override_only_after_source_non_regression():
    source = {
        'baseline': dict(top1_hits=44, top1_mcml=1,
                         baseline_preservation_failures=0,
                         mean_top1_riou=0.75),
        'strict': dict(top1_hits=45, top1_mcml=0,
                       baseline_preservation_failures=0,
                       mean_top1_riou=0.76),
        'ranked': dict(top1_hits=45, top1_mcml=0,
                       baseline_preservation_failures=0,
                       mean_top1_riou=0.76),
        'confident_override': dict(
            top1_hits=45, top1_mcml=0,
            baseline_preservation_failures=12, mean_top1_riou=0.76),
        'routing_diagnostics': dict(
            baseline_correct_overridden_to_incorrect_count=0)}
    target = {
        'baseline': dict(top1_hits=0, top1_mcml=33,
                         baseline_preservation_failures=0,
                         mean_top1_riou=0.0),
        'strict': dict(top1_hits=14, top1_mcml=12,
                       baseline_preservation_failures=0,
                       mean_top1_riou=0.3),
        'ranked': dict(top1_hits=30, top1_mcml=2,
                       baseline_preservation_failures=0,
                       mean_top1_riou=0.6),
        'confident_override': dict(
            top1_hits=31, top1_mcml=1,
            baseline_preservation_failures=1, mean_top1_riou=0.62)}
    assert audit.make_decision(source, target) == (
        'CONFIDENT_DINO_OVERRIDE_CANDIDATE_PASSES')
    source['confident_override']['top1_hits'] = 43
    assert audit.make_decision(source, target) == (
        'INVALID_SOURCE_CONFIDENT_OVERRIDE_REGRESSION')


def test_source_gate_rejects_hidden_correct_baseline_regression():
    summary = {
        'baseline': dict(top1_hits=44, top1_mcml=1, mean_top1_riou=0.75),
        'confident_override': dict(
            top1_hits=44, top1_mcml=1, mean_top1_riou=0.75),
        'routing_diagnostics': dict(
            baseline_correct_overridden_to_incorrect_count=1)}
    assert not audit.confident_override_non_regression_holds(summary)


def test_validate_fixed_protocol_and_sequential_gpu_sharing(tmp_path):
    files = []
    for name in ('cfg.py', 'baseline.pth', 'labeller.pth', 'dino.pth'):
        path = tmp_path / name
        path.write_bytes(b'x')
        files.append(str(path))
    args = argparse.Namespace(
        seed=0, source_split='val', source_seq='real_seq07',
        source_val_modulus=5, target_split='test', target_seq='real_seq02',
        target_start=137, target_end=169, dino_gpus=[1, 2], head_gpu=0,
        baseline_gpu=0, patch_size=14, rpn_feat_channels=256,
        roi_fc_channels=1024, roi_samples=256, proposal_count=2000,
        max_detections=2000, dino_height=600, dino_max_long_side=1333,
        baseline_config=files[0], baseline_checkpoint=files[1],
        labeller_checkpoint=files[2], dinov2_checkpoint=files[3])
    args.scope_manifest = None
    audit.validate_args(args)
    args.target_start = 136
    with pytest.raises(ValueError, match='137..169'):
        audit.validate_args(args)
