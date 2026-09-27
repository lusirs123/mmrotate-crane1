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
import sys
import pickle
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROTOCOL = 'k1_dino_object_background_relation_v5_artifact_audit_v3'
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
            except json.JSONDecodeError as exc:
                raise ValueError('Malformed training log: ' + str(path)) from exc
            if row.get('mode') == 'train':
                rows.append(row)
                file_rows.append(row)
        per_file[str(path)] = dict(
            sha256=sha256(path),
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
    if cfg.optimizer.type != 'SGD':
        raise ValueError('Optimizer must be SGD')
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
            or cfg.lr_config.step != [16, 22]
            or cfg.lr_config.policy != 'step'
            or cfg.lr_config.warmup_ratio != 0.001):
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
                optimizer=dict(cfg.optimizer),
                lr_config=dict(cfg.lr_config),
                gradient_clip=dict(grad_clip),
                samples_per_gpu=cfg.data.samples_per_gpu,
                config_evidence='current repository config; historical binding checked separately',
                resolved_config_sha256=hashlib.sha256(cfg.pretty_text.encode()).hexdigest(),
                load_from=str(cfg.load_from),
                dino_cache_pipeline=True, collects_teacher_features=True)


def _sidecar_matches(sidecar, config, checkpoint, pkl, split):
    if not Path(sidecar).is_file():
        return False, 'missing_generation_provenance'
    data = _read_json(sidecar)
    expected = {
        'config': str(Path(config).resolve()),
        'config_sha256': sha256(config),
        'checkpoint': str(Path(checkpoint).resolve()),
        'checkpoint_sha256': sha256(checkpoint),
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
    from crane_project.tools.eval_crane_offline import METRIC_PROTOCOL_VERSION
    if selection.get('metric_protocol_version') != METRIC_PROTOCOL_VERSION:
        raise ValueError('Metric version mismatch')
    if selection.get('center_thresh_px') != 15.0:
        raise ValueError('Source center threshold mismatch')
    if Path(selection['selected_path']).resolve() != (work / (EXPECTED_SELECTED[arm] + '.pth')).resolve():
        raise ValueError('Selected path mismatch')
    validate_annotations(root, selection, 'val')
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
    source_artifacts = {}
    for epoch in sorted(EXPECTED_SOURCE_EPOCHS):
        item = selection['all_checkpoints'][epoch]
        source_checkpoint = Path(item['checkpoint'])
        source_pkl = Path(item['results_pkl'])
        if (not source_checkpoint.is_file()
                or sha256(source_checkpoint) != item['checkpoint_sha256']):
            raise ValueError(arm + ': source checkpoint hash mismatch: ' + epoch)
        if (not source_pkl.is_file()
                or sha256(source_pkl) != item['results_pkl_sha256']):
            raise ValueError(arm + ': source PKL hash mismatch: ' + epoch)
        ok, status = _sidecar_matches(
            str(source_pkl) + '.provenance.json', config,
            source_checkpoint, source_pkl, 'source_val')
        if not ok:
            raise ValueError(arm + ': source provenance ' + epoch + ': ' + status)
        if source_checkpoint.resolve() != (work / (epoch + '.pth')).resolve():
            raise ValueError('Cross-arm checkpoint path: ' + epoch)
        validate_prediction(root, config, source_checkpoint, source_pkl, 'val')
        source_artifacts[epoch] = dict(
            checkpoint_sha256=item['checkpoint_sha256'],
            results_pkl_sha256=item['results_pkl_sha256'],
            provenance=status)
    final_dir = sweep / 'final_test' / checkpoint.stem
    report_path = final_dir / 'final_test_metrics_v2.json'
    pkl = final_dir / 'preds/results.pkl'
    report = _read_json(report_path)
    validate_annotations(root, report, 'test')
    if report.get('metric_protocol_version') != METRIC_PROTOCOL_VERSION:
        raise ValueError('TEST metric version mismatch')
    if report.get('evidence_role') != 'fixed_test_after_source_val_selection':
        raise ValueError('TEST role mismatch')
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
    validate_prediction(root, config, checkpoint, pkl, 'test')
    logs = _log_summary(work)
    if arm == 'c' and logs['relation_loss_positive_records'] == 0:
        raise ValueError('C: no positive relation-loss training records')
    if not logs['train_records']:
        raise ValueError(arm + ': missing training records')
    if arm == 'a' and logs['relation_loss_records'] != 0:
        raise ValueError('A: relation loss unexpectedly present in logs')
    return dict(
        selected_checkpoint=chosen,
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=sha256(checkpoint),
        source_selection=str(selection_path.resolve()),
        source_selection_sha256=sha256(selection_path),
        source_artifacts=source_artifacts,
        fixed_test_report=str(report_path.resolve()),
        fixed_test_report_sha256=sha256(report_path),
        training_config=_training_config_summary(root, arm),
        metrics=report['metrics'],
        provenance=sidecar_status,
        training_log=logs,
        warnings=(['multiple_training_log_files']
                   if len(logs['log_files']) > 1 else []))



def validate_annotations(root, record, split):
    from crane_project.tools.ckpt_sweep import annotation_set_sha256
    directory = root / ('crane_project/data/crane_grab/' + split + '/annfiles')
    paths = [p for p in directory.glob('*.txt') if not p.name.startswith('._')]
    if len(paths) != (738 if split == 'val' else 992):
        raise ValueError('Annotation frame count mismatch: ' + split)
    key = 'source_val_annotations_sha256' if split == 'val' else 'gt_annotations_sha256'
    if record.get(key) != annotation_set_sha256(str(directory)):
        raise ValueError('Annotation hash mismatch: ' + split)


def validate_prediction(root, config, checkpoint, pkl, split):
    from crane_project.tools.ckpt_sweep import (
        check_or_record_prediction, dota_export_sha256, get_val_img_ids)
    directory = root / ('crane_project/data/crane_grab/' + split + '/annfiles')
    ids = get_val_img_ids(str(directory))
    check_or_record_prediction(str(config), str(checkpoint), str(pkl),
                               str(directory), 'source_val' if split == 'val' else 'fixed_test')
    with pkl.open('rb') as stream:
        if len(pickle.load(stream)) != len(ids):
            raise ValueError('Prediction frame count mismatch')
    dota = pkl.parent / 'Task1_grab'
    if {p.stem for p in dota.glob('*.txt') if not p.name.startswith('._')} != set(ids):
        raise ValueError('DOTA frame set mismatch')
    expected = dict(protocol='crane_dota_export_provenance_v1',
                    results_pkl_sha256=sha256(pkl), frame_ids=ids,
                    dota_sha256=dota_export_sha256(str(dota), ids))
    if _read_json(str(dota) + '.provenance.json') != expected:
        raise ValueError('DOTA provenance mismatch')


def failure_runs(frames, method, field):
    runs, current = [], []
    previous = None
    for row in sorted(frames, key=lambda x: (x['domain'], x['sequence'], x['frame'])):
        key = (row['domain'], row['sequence'])
        contiguous = previous is not None and key == previous[:2] and row['frame'] == previous[2] + 1
        if current and (not contiguous or row[method][field]):
            runs.append(dict(start=current[0], end=current[-1], length=len(current)))
            current = []
        if not row[method][field]:
            current.append(row['frame_key'])
        previous = (*key, row['frame'])
    if current:
        runs.append(dict(start=current[0], end=current[-1], length=len(current)))
    return runs


def paired_test(root, arms):
    from crane_project.tools.audit_k1_dino_student_paired_test_v1 import build_report as paired
    from crane_project.tools.eval_crane_offline import CraneOfflineEvaluator
    specs = []
    gt = root / 'crane_project/data/crane_grab/test/annfiles'
    for arm in ARMS:
        report = Path(arms[arm]['fixed_test_report'])
        pred = report.parent / 'preds'
        specs.append((report, pred / 'results.pkl', pred / 'Task1_grab'))
        evaluator = CraneOfflineEvaluator(mode='test', center_thresh_px=15.0)
        evaluator.extract_from_dirs(str(gt), str(pred / 'Task1_grab'))
        actual = evaluator.compute_metrics()
        if actual != arms[arm]['metrics']:
            raise ValueError(arm + ': recomputed TEST metrics differ from saved report')
    report = paired(gt, *specs)
    report['method_mapping'] = dict(baseline='A', student='C')
    report['failure_intervals'] = {
        arm: {field: failure_runs(report['frames'], method, field)
              for field in ('output', 'center_hit', 'riou_hit')}
        for arm, method in (('A', 'baseline'), ('C', 'student'))}
    report['center_metric_note'] = 'Use center_hit_given_output for requested center accuracy; legacy R_center uses all GT frames. MCML uses RIoU<0.5 failures, not only absent outputs.'
    return report


def checkpoint_evidence(root, arms):
    import torch
    from mmcv import Config
    from crane_project.tools.preflight_k1_dino_warmstart_v2 import EXPECTED_BASELINE_SHA256
    k1 = root / 'work_dirs/crane_symeood_k1/epoch_20.pth'
    if sha256(k1) != EXPECTED_BASELINE_SHA256:
        raise ValueError('K1 baseline hash mismatch')
    result = {}
    for arm in ARMS:
        try:
            payload = torch.load(arms[arm]['checkpoint'], map_location='cpu', weights_only=False)
        except TypeError:
            payload = torch.load(arms[arm]['checkpoint'], map_location='cpu')
        state = {k.removeprefix('module.') if hasattr(k, 'removeprefix') else (k[7:] if k.startswith('module.') else k): v for k, v in payload['state_dict'].items()}
        key = 'bbox_head.semantic_cls_adapter.weight'
        if key not in state or not torch.isfinite(state[key]).all():
            raise ValueError('Missing or nonfinite adapter weights')
        meta = payload.get('meta', {})
        expected_epoch = int(EXPECTED_SELECTED[arm].split('_')[1])
        if meta.get('epoch') != expected_epoch:
            raise ValueError('Checkpoint internal epoch mismatch')
        current = Config.fromfile(str(root / TRAIN_CONFIG[arm]))
        text = meta.get('config')
        differences = []
        if text:
            historical = Config.fromstring(text, '.py')
            for field in ('model', 'data', 'optimizer', 'optimizer_config', 'lr_config', 'runner', 'load_from', 'resume_from'):
                if historical.get(field) != current.get(field):
                    differences.append(field)
        result[arm] = dict(
            epoch=meta.get('epoch'), iteration=meta.get('iter'),
            checkpoint_config_available=bool(text),
            checkpoint_config_differences=differences,
            adapter_nonzero_count=int(torch.count_nonzero(state[key]).item()),
            adapter_l2=float(state[key].float().norm().item()),
            interpretation='Adapter initialized at zero in code; changed weights do not isolate relation-loss contribution from detection loss.',
            training_run_binding='unverified: checkpoint metadata does not necessarily identify a unique log file')
    result['K1_sha256'] = sha256(k1)
    return result


def build_report(project_root):
    root = Path(project_root).resolve()
    arms = {arm: _arm(root, arm) for arm in ARMS}
    a = arms['a']['metrics']
    c = arms['c']['metrics']
    if set(a) != set(c):
        raise ValueError('A/C metric keys differ')
    if not all(_finite(v) for v in list(a.values()) + list(c.values())):
        raise ValueError('Nonfinite saved metric')
    keys = sorted(set(a) & set(c))
    delta = {key: float(c[key]) - float(a[key])
             for key in keys if _finite(a[key]) and _finite(c[key])}
    return dict(protocol=PROTOCOL, inference_performed=False,
                checkpoint_selection_performed=False,
                artifact_checks_read_only=True, arms=arms,
                evidence_role='post_exposure_fixed_test_diagnosis_only',
                audit_script_sha256=sha256(__file__),
                limitations=['No historical gradient trace: positive loss and changed adapter do not prove causal benefit.', 'A log-to-checkpoint run binding may remain unverified.', 'Current configuration cannot alone establish historical runtime settings.'],
                c_minus_a_metrics=delta,
                paired_test=paired_test(root, arms),
                checkpoint_evidence=checkpoint_evidence(root, arms),
                interpretation='Metric differences are descriptive; causal benefit and gradient flow are not established by this audit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    output = Path(args.out_json)
    if output.exists():
        raise FileExistsError('Refusing to overwrite ' + str(output))
    report = build_report(args.project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Refusing to overwrite ' + str(output))
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
