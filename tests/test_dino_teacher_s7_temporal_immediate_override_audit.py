import json

from crane_project.tools import (
    dino_teacher_s7_temporal_immediate_override_audit as audit)
from crane_project.tools import (
    dino_teacher_s7_temporal_source_attribution_audit as attribution)


def test_locked_immediate_override_replaces_only_audit_mode(tmp_path):
    candidate = tmp_path / 'candidate.pth'
    candidate.write_bytes(b'checkpoint')
    dino = tmp_path / 'dino.pth'
    dino.write_bytes(b'dino')
    source = tmp_path / 'train_result.json'
    source.write_text(json.dumps({}))
    args = type('Args', (), dict(
        seed=0, dinov2_model='dinov2_vitl14',
        source_result_json=str(source), eval_only_checkpoint=str(candidate),
        dinov2_repo=str(tmp_path), dinov2_checkpoint=str(dino),
        dino_gpus=[1, 2], head_gpu=0, legacy_sdpa_query_chunk=512,
        out_json=str(tmp_path / 'out' / 'result.json'), data_root='data',
        feature_cache_dir=str(tmp_path / 'cache')))()

    original = attribution.build_locked_labeller_argv
    attribution.build_locked_labeller_argv = lambda value: [
        'labeller.py', '--eval-only-checkpoint', value.eval_only_checkpoint,
        '--source-temporal-attribution-audit', '--skip-target-eval']
    try:
        argv = audit.build_locked_immediate_argv(args)
    finally:
        attribution.build_locked_labeller_argv = original

    assert '--source-temporal-immediate-override-audit' in argv
    assert '--source-temporal-attribution-audit' not in argv
    assert '--eval-only-checkpoint' in argv
    assert '--skip-target-eval' in argv
