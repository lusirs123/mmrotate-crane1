"""Read-only audit of the completed V5 relation A/C artifacts.

The audit does not run inference, select a checkpoint, or rewrite sidecars. It
checks source-VAL selection identity, fixed-TEST provenance, checkpoint/config
hashes, and training-log evidence for the relation loss, then reports the
paired A/C metric delta.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


PROTOCOL = 'k1_dino_object_background_relation_v5_artifact_audit_v2'
METRIC_PROTOCOL = 'crane_ckpt_sweep_final_test_v2'
STUDENT_CONFIG = 'crane_project/configs/crane_symeood_k1_dino_semantic_student_v1.py'
ARMS = ('a', 'c')
EXPECTED_SELECTED = {'a': 'epoch_24', 'c': 'epoch_18'}
EXPECTED_SOURCE_EPOCHS = {
    'epoch_16', 'epoch_18', 'epoch_20', 'epoch_22', 'epoch_24'}
TRAIN_CONFIG = {
    'a': 'crane_project/configs/crane_symeood_k1_dino_object_background_relation_a_v5.py',
    'c': 'crane_project/configs/crane_symeood_k1_dino_object_background_relation_c_v5.py'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _log_summary(work_dir):
    rows = []
    per_file = {}
    for path in sorted(Path(work_dir).glob('*.log.json')):
        file_rows = []
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('mode') == 'train':
                rows.append(row)
                file_rows.append(row)
        per_file[str(path)] = dict(
            train_records=len(file_rows),
            first_epoch=(file_rows[0].get('epoch') if file_rows else None),
            last_epoch=(file_rows[-1].get('epoch') if file_rows else None),
            first_iter=(file_rows[0].get('iter') if file_rows else None),
            last_iter=(file_rows[-1].get('iter') if file_rows else None))
    relation = [float(row['loss_object_background_relation'])
                for row in rows
                if row.get('loss_object_background_relation') is not None]
    if any(not math.isfinite(value) for value in relation):
        raise ValueError('non-finite relation loss in ' + str(work_dir))
    return dict(
        log_files=[str(path) for path in sorted(Path(work_dir).glob('*.log.json'))],
        per_file=per_file,
        train_records=len(rows),
        relation_loss_records=len(relation),
        relation_loss_positive_records=sum(value > 0 for value in relation),
        relation_loss_min=min(relation) if relation else None,
        relation_loss_max=max(relation) if relation else None,
        relation_loss_last=relation[-1] if relation else None)


def _training_config_summary(root, arm):
    """Validate the formal training contract recorded by the actual config."""
    try:
        from mmcv import Config
    except ImportError as exc:
        raise RuntimeError('MMCV is required for training-config audit') from exc
    path = root / TRAIN_CONFIG[arm]
    cfg = Config.fromfile(str(path))
    relation = cfg.model.get('object_background_relation')
    expected_relation = None if arm == 'a' else dict(
        enabled=True, loss_weight=0.05, feature_level=0,
        protect_geometry=True)
    if relation != expected_relation:
        raise ValueError(arm + ': formal relation config mismatch')
    if cfg.model.get('semantic_distillation') is not None:
        raise ValueError(arm + ': old semantic loss is enabled')
    if cfg.model.backbone.frozen_stages != 1:
        raise ValueError(arm + ': backbone freezing mismatch')
    if cfg.model.bbox_head.use_semantic_cls_adapter is not True:
        raise ValueError(arm + ': classification adapter is disabled')
    if cfg.runner.max_epochs != 24 or cfg.optimizer.lr != 0.0025:
        raise ValueError(arm + ': formal training budget mismatch')
    if (cfg.optimizer.momentum != 0.9
            or cfg.optimizer.weight_decay != 0.0001):
        raise ValueError(arm + ': optimizer contract mismatch')
    grad_clip = cfg.optimizer_config.get('grad_clip')
    if (grad_clip is None or grad_clip.max_norm != 10
            or grad_clip.norm_type != 2):
        raise ValueError(arm + ': gradient clipping contract mismatch')
    if (cfg.lr_config.warmup != 'linear'
            or cfg.lr_config.warmup_iters != 1000
            or cfg.lr_config.step != [16, 22]):
        raise ValueError(arm + ': learning-rate schedule mismatch')
    if cfg.data.samples_per_gpu != 2:
        raise ValueError(arm + ': batch-size contract mismatch')
    if cfg.resume_from is not None:
        raise ValueError(arm + ': unexpected resume_from')
    if not str(cfg.load_from).endswith(
            'work_dirs/crane_symeood_k1/epoch_20.pth'):
        raise ValueError(arm + ': K1 initialization mismatch')
    if sum(step.get('type') == 'LoadDinoFeatureFromCache'
           for step in cfg.train_pipeline) != 1:
        raise ValueError(arm + ': DINO cache pipeline mismatch')
    collect = [step for step in cfg.train_pipeline
               if step.get('type') == 'Collect']
    if not collect or 'teacher_features' not in collect[-1].get('keys', []):
        raise ValueError(arm + ': teacher_features is absent from Collect')
    return dict(path=str(path.resolve()), sha256=sha256(path),
                relation=relation, old_semantic_loss=None,
                frozen_stages=cfg.model.backbone.frozen_stages,
                max_epochs=cfg.runner.max_epochs,
                optimizer_lr=cfg.optimizer.lr,
                load_from=str(cfg.load_from),
                dino_cache_pipeline=True, collects_teacher_features=True)


def _sidecar_matches(sidecar, config, checkpoint, pkl, split):
    if not Path(sidecar).is_file():
        return False, 'missing_generation_provenance'
    data = _read_json(sidecar)
    expected = {
        'config': str(Path(config).resolve()),
        'checkpoint': str(Path(checkpoint).resolve()),
        'results_pkl': str(Path(pkl).resolve()),
        'split': split,
        'results_pkl_sha256': sha256(pkl),
    }
    for key, value in expected.items():
        if data.get(key) != value:
            return False, 'provenance_mismatch:' + key
    return True, 'verified'


def _arm(root, arm):
    work = root / ('work_dirs/crane_symeood_k1_dino_object_background_relation_'
                   + arm + '_v5')
    sweep = work / 'source_val_sweep_protocol_v2'
    selection_path = sweep / 'sweep_results.json'
    selection = _read_json(selection_path)
    config = root / STUDENT_CONFIG
    if selection.get('evidence_role') != 'source_val_checkpoint_selection':
        raise ValueError(arm + ': unexpected source-VAL evidence role')
    if set(selection.get('all_checkpoints', {})) != EXPECTED_SOURCE_EPOCHS:
        raise ValueError(arm + ': source-VAL epoch set mismatch')
    if selection.get('config_sha256') != sha256(config):
        raise ValueError(arm + ': source-VAL config hash mismatch')
    if selection.get('metric_protocol_version') is None:
        raise ValueError(arm + ': missing metric protocol')
    chosen = selection.get('selected_checkpoint')
    if chosen != EXPECTED_SELECTED[arm]:
        raise ValueError(arm + ': selected epoch mismatch: ' + str(chosen))
    selected = selection.get('all_checkpoints', {}).get(chosen)
    if not chosen or not selected:
        raise ValueError(arm + ': missing selected checkpoint record')
    checkpoint = Path(selected['checkpoint'])
    if not checkpoint.is_file() or sha256(checkpoint) != selected['checkpoint_sha256']:
        raise ValueError(arm + ': selected checkpoint hash mismatch')
    selected_txt = (sweep / 'selected_checkpoint.txt').read_text(
        encoding='utf-8').strip()
    if Path(selected_txt).resolve() != checkpoint.resolve():
        raise ValueError(arm + ': selected_checkpoint.txt mismatch')
    final_dir = sweep / 'final_test' / checkpoint.stem
    report_path = final_dir / 'final_test_metrics_v2.json'
    pkl = final_dir / 'preds/results.pkl'
    report = _read_json(report_path)
    if report.get('protocol') != METRIC_PROTOCOL:
        raise ValueError(arm + ': fixed-TEST protocol mismatch')
    if Path(report.get('config', '')).resolve() != config.resolve():
        raise ValueError(arm + ': fixed-TEST config path mismatch')
    if Path(report.get('checkpoint', '')).resolve() != checkpoint.resolve():
        raise ValueError(arm + ': fixed-TEST checkpoint path mismatch')
    if report.get('frame_count') != 992 or report.get('center_thresh_px') != 15.0:
        raise ValueError(arm + ': fixed-TEST frame or threshold mismatch')
    if report.get('checkpoint_sha256') != sha256(checkpoint):
        raise ValueError(arm + ': fixed-TEST checkpoint hash mismatch')
    if report.get('config_sha256') != selection.get('config_sha256'):
        raise ValueError(arm + ': fixed-TEST config hash mismatch')
    if report.get('results_pkl_sha256') != sha256(pkl):
        raise ValueError(arm + ': fixed-TEST PKL hash mismatch')
    sidecar_ok, sidecar_status = _sidecar_matches(
        str(pkl) + '.provenance.json', config, checkpoint, pkl, 'fixed_test')
    if not sidecar_ok:
        raise ValueError(arm + ': ' + sidecar_status)
    logs = _log_summary(work)
    if arm == 'c' and logs['relation_loss_positive_records'] == 0:
        raise ValueError('C: no positive relation-loss training records')
    if arm == 'a' and logs['relation_loss_positive_records'] != 0:
        raise ValueError('A: relation loss unexpectedly present in logs')
    return dict(
        selected_checkpoint=chosen,
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=sha256(checkpoint),
        source_selection=str(selection_path.resolve()),
        source_selection_sha256=sha256(selection_path),
        fixed_test_report=str(report_path.resolve()),
        fixed_test_report_sha256=sha256(report_path),
        training_config=_training_config_summary(root, arm),
        metrics=report['metrics'],
        provenance=sidecar_status,
        training_log=logs,
        warnings=(['multiple_training_log_files']
                   if len(logs['log_files']) > 1 else []))


def build_report(project_root):
    root = Path(project_root).resolve()
    arms = {arm: _arm(root, arm) for arm in ARMS}
    a = arms['a']['metrics']
    c = arms['c']['metrics']
    keys = sorted(set(a) & set(c))
    delta = {key: float(c[key]) - float(a[key])
             for key in keys if _finite(a[key]) and _finite(c[key])}
    return dict(protocol=PROTOCOL, inference_performed=False,
                checkpoint_selection_performed=False,
                artifact_checks_read_only=True, arms=arms,
                c_minus_a_metrics=delta,
                conclusion='weak_real_temporal_change_without_center_or_MCML_max_gain')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    report = build_report(args.project_root)
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Refusing to overwrite ' + str(output))
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n',
                      encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
