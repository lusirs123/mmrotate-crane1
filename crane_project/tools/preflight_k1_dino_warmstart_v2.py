"""CPU-only identity and initial-output check for matched K1 warm starts."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


PROTOCOL = 'k1_dino_warmstart_preflight_v2'
EXPECTED_BASELINE_SHA256 = (
    '3ab0885159294beb820da1445c38045a342fd4956c3d094eeaaadf78deb745c2')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_identity(selection, final_report, checkpoint_sha,
                             checkpoint_path):
    """Reject a historical or different K1 checkpoint before any training."""
    selected = selection.get('selected_checkpoint')
    entries = selection.get('all_checkpoints') or {}
    entry = entries.get(selected) or {}
    if (selection.get('metric_protocol_version') != 2
            or selection.get('evidence_role') !=
            'source_val_checkpoint_selection'
            or selected != 'epoch_20'
            or checkpoint_sha != EXPECTED_BASELINE_SHA256
            or entry.get('checkpoint_sha256') != checkpoint_sha
            or Path(selection.get('selected_path', '')).resolve() !=
            checkpoint_path.resolve()
            or Path(entry.get('checkpoint', '')).resolve() !=
            checkpoint_path.resolve()):
        raise ValueError('Source-VAL-selected K1 checkpoint identity mismatch')
    if (final_report.get('protocol') != 'crane_ckpt_sweep_final_test_v2'
            or final_report.get('metric_protocol_version') != 2
            or final_report.get('checkpoint_sha256') != checkpoint_sha
            or Path(final_report.get('checkpoint', '')).resolve() !=
            checkpoint_path.resolve()):
        raise ValueError('K1 final report checkpoint identity mismatch')
    return selected


def validate_configs(control, distill, checkpoint_path):
    """Check that source data and the fine-tuning budget are identical."""
    shared = ('data', 'data_root', 'optimizer', 'optimizer_config', 'lr_config',
              'runner', 'checkpoint_config', 'evaluation', 'train_pipeline',
              'test_pipeline', 'load_from', 'resume_from')
    for name in shared:
        if control.get(name) != distill.get(name):
            raise ValueError('Warm-start arms differ in ' + name)
    if (Path(control.load_from).resolve() != checkpoint_path.resolve()
            or control.resume_from is not None):
        raise ValueError('Warm-start load/resume path mismatch')
    control_model = dict(control.model)
    distill_model = dict(distill.model)
    control_loss = control_model.pop('semantic_distillation', None)
    distill_loss = distill_model.pop('semantic_distillation', None)
    if (control_loss is not None or not distill_loss
            or not distill_loss.get('enabled')
            or control_model != distill_model
            or not control_model['bbox_head'].get(
                'use_semantic_cls_adapter')):
        raise ValueError('Warm-start model difference is not only DINO loss')
    if (control.runner.max_epochs != 4
            or control.optimizer.lr != 0.00025
            or control.data.samples_per_gpu != 2):
        raise ValueError('Warm-start fixed schedule mismatch')
    pipeline = control.train_pipeline
    if sum(step.get('type') == 'LoadDinoFeatureFromCache'
           for step in pipeline) != 1:
        raise ValueError('Both arms must load the same frozen DINO cache')
    train_sets = control.data.train
    if [(item.ann_file, item.img_prefix) for item in train_sets] != [
            ('train/annfiles/', 'train/images/'),
            ('train_sim/annfiles/', 'train/images/')]:
        raise ValueError('Warm-start training must use source train only')
    return distill_loss


def validate_cache_reports(paths, cfg):
    """Use existing CPU cache receipts without rereading every feature."""
    expected = (('train:train', 2033), ('train_sim:train', 748))
    cache_step = next(step for step in cfg.train_pipeline
                      if step.get('type') == 'LoadDinoFeatureFromCache')
    cache_dir = Path(cache_step['cache_dir']).resolve()
    data_root = Path(cfg.data_root).resolve()
    receipts = []
    for path, (dataset, count) in zip(paths, expected):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if (payload.get('protocol') != 'dino_feature_cache_preflight_v1'
                or payload.get('datasets') != [dataset]
                or payload.get('complete') is not True
                or payload.get('image_count') != count
                or payload.get('valid_count') != count
                or payload.get('missing_or_ambiguous_count') != 0
                or payload.get('cache_load_error_count') != 0
                or payload.get('expected_channels') != 1024
                or payload.get('expected_model') != 'dinov2_vitl14'
                or Path(payload.get('data_root', '')).resolve() != data_root
                or Path(payload.get('cache_dir', '')).resolve() != cache_dir):
            raise ValueError('Incomplete or mismatched DINO cache receipt: '
                             + str(path))
        receipts.append(dict(path=str(Path(path).resolve()),
                             sha256=sha256(path), dataset=dataset,
                             valid_count=count))
    return receipts


def _load_source_state(path):
    try:
        payload = torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location='cpu')
    state = payload.get('state_dict') if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError('K1 checkpoint has no model state_dict')
    if any(key.startswith('module.') for key in state):
        raise ValueError('Unexpected module-prefixed K1 checkpoint')
    if any(key.startswith('bbox_head.semantic_cls_adapter.') for key in state):
        raise ValueError('Expected ordinary K1 checkpoint without adapter')
    return state


def smoke_check(control, distill, checkpoint_path):
    """Build both CPU models and compare their pre-training head outputs."""
    from mmcv.runner import load_checkpoint
    from mmrotate.models import build_detector

    source_state = _load_source_state(checkpoint_path)
    observed = []
    for cfg in (control, distill):
        model = build_detector(
            cfg.model, train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        # train.py calls model.init_weights() before loading the checkpoint;
        # this adapter is the only newly introduced inference parameter.
        adapter = model.bbox_head.semantic_cls_adapter.weight
        torch.nn.init.zeros_(adapter)
        target_state = model.state_dict()
        missing = [key for key, value in source_state.items()
                   if key not in target_state or
                   target_state[key].shape != value.shape]
        if missing:
            raise ValueError('K1 weights incompatible with student: '
                             + repr(missing[:5]))
        extra = [key for key in target_state if key not in source_state]
        permitted = ('bbox_head.semantic_cls_adapter.',
                     'semantic_distillation.')
        if any(not key.startswith(permitted) for key in extra):
            raise ValueError('Unexpected new student parameters: '
                             + repr(extra[:5]))
        load_checkpoint(model, str(checkpoint_path), map_location='cpu',
                        strict=False)
        loaded = model.state_dict()
        if any(not torch.equal(loaded[key], value)
               for key, value in source_state.items()):
            raise ValueError('K1 weights changed during warm start')
        if torch.count_nonzero(adapter).item() != 0:
            raise ValueError('Classification adapter is not zero at start')
        model.eval()
        with torch.no_grad():
            # Head-only test needs no image, dataset, GPU, or backbone pass.
            torch.manual_seed(1701)
            feature = torch.randn(1, 256, 8, 8)
            cls_score, bbox_pred = model.bbox_head.forward_single(feature)
            observed.append((cls_score.cpu(), bbox_pred.cpu()))
        del adapter, target_state, loaded, cls_score, bbox_pred, feature, model
    if not all(torch.equal(a, b) for a, b in zip(*observed)):
        raise ValueError('Initial control/distill head outputs differ')
    return dict(shared_checkpoint_parameter_count=len(source_state),
                initial_cls_and_bbox_outputs_equal=True,
                initial_adapter_zero=True)


def write_exact(path, report):
    path = Path(path)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding='utf-8') != encoded:
        raise RuntimeError('Refusing to overwrite different output: '
                           + str(path))
    if not path.exists():
        path.write_text(encoded, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-selection', required=True)
    parser.add_argument('--baseline-final-report', required=True)
    parser.add_argument('--control-config', required=True)
    parser.add_argument('--distill-config', required=True)
    parser.add_argument('--train-cache-report', required=True)
    parser.add_argument('--train-sim-cache-report', required=True)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    from mmcv import Config

    control = Config.fromfile(args.control_config)
    distill = Config.fromfile(args.distill_config)
    checkpoint_path = Path(control.load_from).resolve()
    checkpoint_sha = sha256(checkpoint_path)
    selection = json.loads(Path(args.source_selection).read_text(
        encoding='utf-8'))
    final_report = json.loads(Path(args.baseline_final_report).read_text(
        encoding='utf-8'))
    selected = validate_source_identity(
        selection, final_report, checkpoint_sha, checkpoint_path)
    loss = validate_configs(control, distill, checkpoint_path)
    cache_receipts = validate_cache_reports(
        (args.train_cache_report, args.train_sim_cache_report), control)
    smoke = smoke_check(control, distill, checkpoint_path)
    report = dict(
        protocol=PROTOCOL, decision='MATCHED_WARMSTART_READY',
        evidence_role='source_selected_initialization_only',
        fixed_test_predictions_read=False,
        selected_checkpoint=selected,
        source_selection_sha256=sha256(args.source_selection),
        baseline_final_report_sha256=sha256(args.baseline_final_report),
        baseline_checkpoint_sha256=checkpoint_sha,
        control_config_sha256=sha256(args.control_config),
        distill_config_sha256=sha256(args.distill_config),
        distillation_loss_weight=loss['loss_weight'],
        cache_receipts=cache_receipts,
        smoke_check=smoke)
    write_exact(args.out_json, report)
    print('K1_DINO_WARMSTART_PREFLIGHT_OK')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
