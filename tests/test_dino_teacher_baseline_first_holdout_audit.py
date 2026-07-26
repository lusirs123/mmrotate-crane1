import argparse

import pytest

from crane_project.tools import dino_teacher_baseline_first_holdout_audit as audit


def _args(tmp_path, **overrides):
    paths = {}
    for name in ('config.py', 'baseline.pth', 'labeller.pth', 'dino.pth'):
        path = tmp_path / name
        path.write_bytes(b'x')
        paths[name] = str(path)
    values = dict(
        confirm_fixed_holdout=True, seed=0,
        source_split='val', source_seq='real_seq07',
        source_val_modulus=5, holdout_split='test',
        holdout_seqs=['real_seq03', 'sim_seq09'],
        dino_gpus=[1, 2], head_gpu=0, baseline_gpu=0,
        patch_size=14, rpn_feat_channels=256, roi_fc_channels=1024,
        roi_samples=256, proposal_count=2000, max_detections=2000,
        dino_height=600, dino_max_long_side=1333,
        baseline_config=paths['config.py'],
        baseline_checkpoint=paths['baseline.pth'],
        labeller_checkpoint=paths['labeller.pth'],
        dinov2_checkpoint=paths['dino.pth'],
        out_json=str(tmp_path / 'result.json'))
    values.update(overrides)
    return argparse.Namespace(**values)


def _summary(baseline_hits=10, override_hits=10,
             baseline_mcml=1, override_mcml=1,
             baseline_riou=0.7, override_riou=0.7,
             harmful_overrides=0):
    base = dict(top1_hits=baseline_hits, top1_mcml=baseline_mcml,
                mean_top1_riou=baseline_riou,
                baseline_preservation_failures=0)
    return dict(
        baseline=base,
        strict=dict(top1_hits=baseline_hits, top1_mcml=baseline_mcml,
                    mean_top1_riou=baseline_riou,
                    baseline_preservation_failures=0),
        ranked=dict(top1_hits=baseline_hits, top1_mcml=baseline_mcml,
                    mean_top1_riou=baseline_riou,
                    baseline_preservation_failures=0),
        confident_override=dict(
            top1_hits=override_hits, top1_mcml=override_mcml,
            mean_top1_riou=override_riou,
            baseline_preservation_failures=3),
        routing_diagnostics=dict(
            baseline_correct_overridden_to_incorrect_count=harmful_overrides))


def test_validate_requires_complete_one_shot_protocol(tmp_path):
    audit.validate_args(_args(tmp_path))
    with pytest.raises(ValueError, match='together'):
        audit.validate_args(_args(
            tmp_path, holdout_seqs=['real_seq03']))
    with pytest.raises(ValueError, match='confirm'):
        audit.validate_args(_args(
            tmp_path, confirm_fixed_holdout=False))


def test_validate_refuses_to_overwrite_result(tmp_path):
    args = _args(tmp_path)
    with open(args.out_json, 'w', encoding='utf-8') as handle:
        handle.write('{}')
    with pytest.raises(ValueError, match='overwrite'):
        audit.validate_args(args)


def test_policy_gate_accepts_non_regressing_override():
    assert audit.fixed_policy_non_regression_holds(_summary(
        baseline_hits=10, override_hits=11,
        baseline_mcml=2, override_mcml=1,
        baseline_riou=0.7, override_riou=0.75))


def test_policy_gate_rejects_harmful_override_even_if_totals_match():
    assert not audit.fixed_policy_non_regression_holds(_summary(
        harmful_overrides=1))


def test_policy_gate_rejects_mean_riou_regression():
    assert not audit.fixed_policy_non_regression_holds(_summary(
        baseline_riou=0.7, override_riou=0.69))


def test_decision_requires_source_and_every_holdout_to_pass():
    source = _summary()
    holdouts = {'real_seq03': _summary(), 'sim_seq09': _summary()}
    assert audit.make_decision(source, holdouts) == (
        'CONFIDENT_OVERRIDE_PASSES_FIXED_HOLDOUTS')
    holdouts['sim_seq09'] = _summary(harmful_overrides=1)
    assert audit.make_decision(source, holdouts) == (
        'CONFIDENT_OVERRIDE_FAILS_FIXED_HOLDOUTS:sim_seq09')
    assert audit.make_decision(
        _summary(harmful_overrides=1), holdouts) == (
            'INVALID_SOURCE_ROUTING_REGRESSION')
