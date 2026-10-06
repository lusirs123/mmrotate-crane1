#!/usr/bin/env python3
"""Inventory old Train-03 receipts and fixed weights; no GPU, unpickle or inference."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT/'docs/webots/webots_native_s14_train03_recovery_v1.json'
VERSION = 'webots_native_s14_train03_recovery_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def files_under(root):
    """Inspect result names only; skip other depth splits and directory symlinks."""
    root = Path(root)
    if not root.is_dir():
        return
    for directory, children, files in os.walk(root, followlinks=False):
        children[:] = sorted(n for n in children if not any(
            word in n.lower() for word in ('unknown', 'fixed_dev', 'fixeddev'))
            and not (Path(directory)/n).is_symlink())
        for name in sorted(files):
            path = Path(directory)/name
            if not path.is_symlink():
                yield path


def inspect_manifest(path, expected):
    """A matching old receipt is still missing an original image-byte binding."""
    path = Path(path)
    if path.stat().st_size > 1024*1024:
        return None
    try:
        receipt = json.loads(path.read_text())
    except (ValueError, UnicodeError):
        return None
    if not isinstance(receipt, dict) or receipt.get('sequence_id') != expected['sequence_id']:
        return None
    checks = {
        'checkpoint': receipt.get('checkpoint_sha256') == expected['head_sha256'],
        'model': receipt.get('model_type') == 'FrozenDinoNativeS14Detector',
        'all981': receipt.get('discovered_frame_count') == 981 and receipt.get('processed_frame_count') == 981,
        'no_limit': receipt.get('limit') is None,
        'no_selection': receipt.get('model_selection_performed') is False,
        'no_threshold_search': receipt.get('threshold_tuning_performed') is False,
        'no_metric_truth': receipt.get('metric_truth_read') is False,
        'unchanged_output': receipt.get('detector_outputs_modified') is False,
    }
    contract = receipt.get('formal_detection_contract', {})
    checks['component_contract'] = isinstance(contract, dict) and (
        contract.get('component') == 'frozen_dino_native_s14_alpha05'
        and all(contract.get(k) is False for k in ('symeood_enabled', 'brightaug_enabled',
            's7_enabled', 'target_scope', 'sequence_identity_routing', 'temporal_takeover', 'box_stabilizer'))
        and contract.get('all_frames') is True)
    prediction = Path(str(path)[:-len('.manifest.json')])
    checks['prediction_exists'] = prediction.is_file() and not prediction.is_symlink()
    checks['prediction_size_bounded'] = checks['prediction_exists'] and prediction.stat().st_size <= 32*1024*1024
    info = dict(manifest_path=str(path.resolve()), manifest_sha256=sha(path),
        receipt=receipt, checks=checks, prediction_path=str(prediction.resolve()),
        original_image_bytes_verified=False, evaluation_reuse_approved=False)
    if checks['prediction_size_bounded']:
        info['prediction_sha256'] = sha(prediction)
        try:
            rows = [json.loads(line) for line in prediction.read_text().splitlines() if line.strip()]
            if len(rows) != 981:
                raise ValueError('Not the full Train-03 stream')
            detected = 0
            for i, row in enumerate(rows):
                if (not isinstance(row, dict) or row.get('frame_id') != 'frame_%05d'%i or row.get('image_name') != 'frame_%05d.jpg'%i
                        or not isinstance(row.get('detected'), bool)):
                    raise ValueError('Frame identity/order/presence differs')
                box = row.get('top1')
                if row['detected']:
                    values = [box[k] for k in ('cx_px', 'cy_px', 'width_px', 'height_px', 'angle_rad', 'score')]
                    if (box.get('class_id') != 0 or not all(math.isfinite(v) for v in values)
                            or min(values[2:4]) <= 0 or row.get('num_detections') != 1):
                        raise ValueError('Original top1 interface differs')
                    detected += 1
                elif box is not None or row.get('num_detections') != 0:
                    raise ValueError('Missing-output representation differs')
            checks['prediction_rows'] = True
            checks['detected_count'] = detected == receipt.get('detected_frame_count')
            info['row_count'] = len(rows)
        except (ValueError, TypeError, KeyError, UnicodeError):
            checks['prediction_rows'] = False
    info['metadata_and_rows_consistent'] = all(checks.values())
    info['status'] = ('CANDIDATE_REQUIRES_ORIGINAL_INPUT_PROVENANCE_REVIEW'
        if info['metadata_and_rows_consistent'] else 'REJECTED_IDENTITY_OR_FRAME_STREAM')
    return info


def inspect_report(path, expected):
    """Locate the original Raw-opt report, without recomputing/filtering metrics."""
    if path.stat().st_size > 16*1024*1024:
        return None
    try:
        data = json.loads(path.read_text())
    except (ValueError, UnicodeError):
        return None
    if (not isinstance(data, dict) or data.get('coordinate_contract') != 'raw_opt_v1'
            or data.get('sequence_id') != expected['sequence_id']):
        return None
    current = data.get('current', {})
    receipt = current.get('prediction_manifest', {}) if isinstance(current, dict) else {}
    if not isinstance(receipt, dict) or receipt.get('checkpoint_sha256') != expected['head_sha256']:
        return None
    return dict(path=str(path.resolve()), sha256=sha(path), bytes=path.stat().st_size,
        prediction_sha256=current.get('prediction_sha256'),
        calibration_sha256=data.get('calibration_sha256'),
        original_report_recovered=True, evaluation_reuse_approved=False)


def run(args):
    project = Path(args.project_root).resolve()
    expected = json.loads(IDENTITY.read_text())
    if expected['protocol'] != VERSION:
        raise ValueError('Recovery identity contract differs')
    out = Path(args.out_dir).resolve()
    if out.exists() or out.with_suffix('.tar.gz').exists():
        raise FileExistsError('Use a new recovery directory; never overwrite')
    source_checks = {name:dict(expected_sha256=digest,
        actual_sha256=sha(project/name) if (project/name).is_file() else None)
        for name,digest in expected['sources'].items()}
    for check in source_checks.values():
        check['passed'] = check['actual_sha256'] == check['expected_sha256']
    paths = list(files_under(project/'work_dirs')) + list(files_under(project/'pretrained'))
    weights = {}
    for role, spec in expected['weights'].items():
        weights[role] = [dict(path=str(p.resolve()), sha256=sha(p), bytes=p.stat().st_size)
            for p in paths if p.name == spec['filename']]
        for entry in weights[role]:
            entry['matches_original_sha'] = entry['sha256'] == spec['sha256']
    candidates = []
    reports = []
    archives = [dict(path=str(p.resolve()), bytes=p.stat().st_size, members_inspected=False)
        for p in paths if (p.name.endswith(('.tar.gz', '.tgz', '.zip')) and any(
            word in p.name.lower() for word in ('depth', 'webots', 'train03', 'train_03')))]
    index = project/'work_dirs/port_results/INDEX.md'
    index_receipt = (dict(path=str(index.resolve()), sha256=sha(index))
        if index.is_file() and not index.is_symlink() else None)
    for path in paths:
        if path.name.endswith('.jsonl.manifest.json'):
            info = inspect_manifest(path, expected['prediction_contract'])
            if info is not None:
                candidates.append(info)
        elif path.suffix == '.json' and any(word in str(path.relative_to(project)).lower()
                for word in ('depth', 'webots', 'train03', 'train_03', 'native_s14')):
            info = inspect_report(path, expected['prediction_contract'])
            if info is not None:
                reports.append(info)
    report = dict(protocol=VERSION, status='OLD_NATIVE_S14_RECOVERY_INVENTORY_COMPLETE_REVIEW_REQUIRED',
        identity_contract_sha256=sha(IDENTITY), project_root=str(project),
        source_checks=source_checks, weights=weights, prediction_candidates=candidates,
        old_raw_opt_reports=reports,
        archive_candidates=archives, port_results_index=index_receipt,
        inference_performed=False, weights_unpickled=False, metric_truth_read=False,
        parameter_updates=0, historical_results_modified=False,
        original_prediction_recovered_and_approved=False,
        note='Inventory completion is not provenance approval or a depth performance result.')
    out.mkdir(parents=True, exist_ok=False)
    (out/'recovery.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'identity.json').write_bytes(IDENTITY.read_bytes())
    # Return only consistent candidates, byte-exact, for human provenance review.
    # No weights, arbitrary sibling reports, annotations or unknown data enter the tar.
    package = out.with_suffix('.tar.gz')
    with package.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as tar:
        tar.add(out/'recovery.json', arcname=out.name+'/recovery.json', recursive=False)
        tar.add(out/'identity.json', arcname=out.name+'/identity.json', recursive=False)
        if index_receipt is not None:
            if sha(index) != index_receipt['sha256']:
                raise ValueError('Port results index changed during packaging')
            tar.add(index, arcname=out.name+'/port_results_INDEX.md', recursive=False)
        for i, info in enumerate(candidates):
            if info['metadata_and_rows_consistent']:
                for key in ('manifest_path', 'prediction_path'):
                    path = Path(info[key])
                    expected_sha = info['manifest_sha256'] if key == 'manifest_path' else info['prediction_sha256']
                    if sha(path) != expected_sha:
                        raise ValueError('Original candidate bytes changed during packaging')
                    tar.add(path, arcname=out.name+'/candidate_%03d/'%i+path.name, recursive=False)
        for i, info in enumerate(reports):
            path = Path(info['path'])
            if sha(path) != info['sha256']:
                raise ValueError('Original report bytes changed during packaging')
            tar.add(path, arcname=out.name+'/report_%03d/'%i+path.name, recursive=False)
    print(report['status'])
    print('fixed_sources_match:', all(v['passed'] for v in source_checks.values()))
    print('weight_matches:', {k:sum(e['matches_original_sha'] for e in v) for k,v in weights.items()})
    print('Train03_manifest_candidates:', len(candidates))
    print('old_Raw_opt_reports:', len(reports))
    print(sha(package), package)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=ROOT)
    p.add_argument('--out-dir', type=Path, required=True)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
