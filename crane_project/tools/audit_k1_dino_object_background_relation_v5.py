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


PROTOCOL = 'k1_dino_object_background_relation_v5_artifact_audit_v1'
METRIC_PROTOCOL = 'crane_ckpt_sweep_final_test_v2'
STUDENT_CONFIG = 'crane_project/configs/crane_symeood_k1_dino_semantic_student_v1.py'
ARMS = ('a', 'c')
EXPECTED_SELECTED = {'a': 'epoch_24', 'c': 'epoch_18'}
EXPECTED_SOURCE_EPOCHS = {
    'epoch_16', 'epoch_18', 'epoch_20', 'epoch_22', 'epoch_24'}


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
    for path in sorted(Path(work_dir).glob('*.log.json')):
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('mode') == 'train':
                rows.append(row)
    relation = [float(row['loss_object_background_relation'])
                for row in rows
                if row.get('loss_object_background_relation') is not None]
    if any(not math.isfinite(value) for value in relation):
        raise ValueError('non-finite relation loss in ' + str(work_dir))
    return dict(
        log_files=[str(path) for path in sorted(Path(work_dir).glob('*.log.json'))],
        train_records=len(rows),
        relation_loss_records=len(relation),
        relation_loss_positive_records=sum(value > 0 for value in relation),
        relation_loss_min=min(relation) if relation else None,
        relation_loss_max=max(relation) if relation else None,
        relation_loss_last=relation[-1] if relation else None)


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
        metrics=report['metrics'],
        provenance=sidecar_status,
        training_log=logs)


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
