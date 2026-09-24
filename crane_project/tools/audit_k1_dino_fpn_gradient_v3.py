"""Read-only audit of the completed A/B/C source-VAL and fixed-TEST runs.

Run on the server with existing artifacts. No inference, selection, or
checkpoint changes are performed. An absent historical generation sidecar is
reported as unattested, never reconstructed from a result report.
"""

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crane_project.tools.audit_k1_dino_student_paired_test_v1 import (
    _load_arm, _metric, directory_sha256)
from crane_project.tools.ckpt_sweep import (
    check_or_record_prediction, prediction_provenance_path, sha256_file)
from crane_project.tools.eval_crane_offline import (
    METRIC_PROTOCOL_VERSION, parse_dota_txt)


PROTOCOL = 'k1_dino_fpn_gradient_existing_artifact_audit_v1'
ARMS = 'abc'
SWEEP = 'source_val_sweep_student_protocol_v2'
STUDENT_CONFIG = ('crane_project/configs/'
                  'crane_symeood_k1_dino_semantic_student_v1.py')


def _load_checkpoint(path):
    try:
        payload = torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location='cpu')
    state = payload['state_dict']
    if not isinstance(state, dict):
        raise ValueError('Checkpoint has no state_dict: ' + str(path))
    return {key[7:] if key.startswith('module.') else key: value.detach().cpu()
            for key, value in state.items()}


def _group(key):
    if key.startswith('backbone.'):
        return 'backbone'
    if key.startswith('neck.'):
        return 'fpn'
    if key.startswith('bbox_head.semantic_cls_adapter.'):
        return 'classification_adapter'
    if key.startswith('bbox_head.'):
        return 'detection_head'
    if key.startswith('aux_heads.'):
        return 'auxiliary_head'
    if key.startswith('semantic_distillation.'):
        return 'training_projection'
    return 'other'


def compare_states(left, right):
    result = {}
    for key in sorted(set(left) | set(right)):
        group = _group(key)
        row = result.setdefault(group, dict(shared_tensors=0,
                                             different_tensors=0,
                                             max_abs_delta=0.0,
                                             l2_delta=0.0,
                                             missing_left=0, missing_right=0))
        if key not in left:
            row['missing_left'] += 1
            continue
        if key not in right:
            row['missing_right'] += 1
            continue
        a, b = left[key], right[key]
        if a.shape != b.shape:
            raise ValueError('Tensor shape mismatch: ' + key)
        row['shared_tensors'] += 1
        if not torch.equal(a, b):
            row['different_tensors'] += 1
            delta = a.float() - b.float()
            row['max_abs_delta'] = max(row['max_abs_delta'],
                                       float(delta.abs().max().item()))
            row['l2_delta'] += float((delta * delta).sum().item())
    for row in result.values():
        row['l2_delta'] = math.sqrt(row['l2_delta'])
    return result


def _training_log_summary(work_dir):
    files = sorted(work_dir.glob('*.log.json'))
    summary = dict(files=[str(path) for path in files],
                   train_records=0, semantic_loss_records=0,
                   semantic_loss_min=None, semantic_loss_max=None,
                   semantic_loss_last=None)
    for path in files:
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get('mode') != 'train':
                continue
            summary['train_records'] += 1
            loss = entry.get('loss_semantic_distill')
            if loss is None:
                continue
            loss = float(loss)
            if not math.isfinite(loss):
                raise ValueError('Nonfinite semantic loss in ' + str(path))
            summary['semantic_loss_records'] += 1
            summary['semantic_loss_min'] = (
                loss if summary['semantic_loss_min'] is None else
                min(summary['semantic_loss_min'], loss))
            summary['semantic_loss_max'] = (
                loss if summary['semantic_loss_max'] is None else
                max(summary['semantic_loss_max'], loss))
            summary['semantic_loss_last'] = loss
    return summary


def _cache_status(config, checkpoint, pkl_path, gt_dir, split):
    if not Path(prediction_provenance_path(str(pkl_path))).is_file():
        return 'historical_generation_unattested'
    check_or_record_prediction(
        str(config), str(checkpoint), str(pkl_path), str(gt_dir),
        split)
    return 'generation_provenance_verified'


def _historical_preflight(root, k1_sha):
    path = root / 'work_dirs/crane_symeood_k1_dino_fpn_gradient_v3_preflight.json'
    if not path.is_file():
        return dict(status='missing')
    report = json.loads(path.read_text(encoding='utf-8'))
    if (report.get('decision') != 'MATCHED_FPN_GRADIENT_READY'
            or report.get('k1_checkpoint_sha256') != k1_sha):
        raise ValueError('Historical preflight identity mismatch')
    rows = {row['arm']: row for row in report['arms']}
    if set(rows) != {'A', 'B', 'C'}:
        raise ValueError('Historical preflight has incomplete arms')
    for arm in ARMS:
        name = arm.upper()
        config_path = (root / 'crane_project/configs' /
                       ('crane_symeood_k1_dino_fpn_gradient_' + arm + '_v3.py'))
        if (report['config_sha256'].get(name) != sha256_file(str(config_path))
                or rows[name]['distillation_gradient_to_fpn'] != (arm == 'c')
                or rows[name]['distillation_gradient_to_adapter'] !=
                (arm != 'a')):
            raise ValueError(name + ' historical preflight/config mismatch')
    return dict(status='verified_historical_gradient_route', path=str(path),
                note='This preflight predates the foreground mask correction')


def _load_one(root, arm, gt_paths, gt_sha, config):
    work = root / 'work_dirs' / ('crane_symeood_k1_dino_fpn_gradient_'
                                + arm + '_v3')
    sweep_dir = work / SWEEP
    selection_path = sweep_dir / 'sweep_results.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if (selection.get('metric_protocol_version') != METRIC_PROTOCOL_VERSION
            or selection.get('evidence_role') !=
            'source_val_checkpoint_selection'
            or selection.get('config_sha256') != sha256_file(str(config))):
        raise ValueError(arm + ' source-VAL selection identity mismatch')
    val_gt_dir = root / 'crane_project/data/crane_grab/val/annfiles'
    val_gt_paths = sorted(path for path in val_gt_dir.glob('*.txt')
                          if not path.name.startswith('._'))
    if (len(val_gt_paths) != 738 or
            selection.get('source_val_annotations_sha256') !=
            directory_sha256(val_gt_paths)):
        raise ValueError(arm + ' source-VAL annotation identity mismatch')
    chosen = selection['selected_checkpoint']
    selected = selection['all_checkpoints'][chosen]
    checkpoint = work / (chosen + '.pth')
    if (str(checkpoint.resolve()) != str(Path(selected['checkpoint']).resolve())
            or str(checkpoint.resolve()) != str(
                Path(selection['selected_path']).resolve())
            or sha256_file(str(checkpoint)) != selected['checkpoint_sha256']):
        raise ValueError(arm + ' selected checkpoint identity mismatch')
    selected_txt = (sweep_dir / 'selected_checkpoint.txt').read_text(
        encoding='utf-8').strip()
    if str(Path(selected_txt).resolve()) != str(checkpoint.resolve()):
        raise ValueError(arm + ' selected_checkpoint.txt mismatch')
    epochs = {f'epoch_{index}' for index in range(1, 5)}
    if not epochs.issubset(selection['all_checkpoints']):
        raise ValueError(arm + ' incomplete four-epoch source-VAL sweep')
    for epoch in epochs:
        item = selection['all_checkpoints'][epoch]
        if (sha256_file(item['checkpoint']) != item['checkpoint_sha256']
                or sha256_file(item['results_pkl']) !=
                item['results_pkl_sha256']):
            raise ValueError(arm + ' source-VAL artifact hash mismatch: '
                             + epoch)
        with open(item['results_pkl'], 'rb') as stream:
            if len(pickle.load(stream)) != len(val_gt_paths):
                raise ValueError(arm + ' source-VAL frame count mismatch: '
                                 + epoch)
    val_cache_status = _cache_status(
        config, checkpoint, Path(selected['results_pkl']), val_gt_dir,
        'source_val')
    final_dir = sweep_dir / 'final_test' / chosen
    report_path = final_dir / 'final_test_metrics_v2.json'
    pkl_path = final_dir / 'preds' / 'results.pkl'
    pred_dir = final_dir / 'preds' / 'Task1_grab'
    loaded = _load_arm(arm, (report_path, pkl_path, pred_dir),
                       gt_paths, gt_sha, len(gt_paths))
    final = loaded['report']
    if (final.get('checkpoint_sha256') != selected['checkpoint_sha256']
            or Path(final['checkpoint']).resolve() != checkpoint.resolve()
            or final.get('config_sha256') != selection['config_sha256']):
        raise ValueError(arm + ' fixed-TEST checkpoint/config mismatch')
    with pkl_path.open('rb') as stream:
        predictions = pickle.load(stream)
    if len(predictions) != len(gt_paths):
        raise ValueError(arm + ' fixed-TEST frame count mismatch')
    boxes = []
    for path, result in zip(gt_paths, predictions):
        raw = np.asarray(result[0])
        boxes.append(raw[0, :6].astype(float).tolist() if len(raw) else None)
    return dict(
        selection=dict(selected_checkpoint=chosen,
                       selection_strategy=selection.get('selection_info', {}),
                       selected_source_val_metrics=selected['metrics']),
        fixed_test_metrics=final['metrics'],
        fixed_test_prediction_sha256=sha256_file(str(pkl_path)),
        source_val_prediction_cache_status=val_cache_status,
        prediction_cache_status=_cache_status(
            config, checkpoint, pkl_path, root / 'crane_project/data/'
            'crane_grab/test/annfiles', 'fixed_test'),
        training_log=_training_log_summary(work),
        checkpoint=str(checkpoint), boxes=boxes,
        evaluated_boxes=loaded['boxes'])


def _pair(left, right, gt, indices):
    summary = dict(both_output=0, left_only_output=0,
                   right_only_output=0, neither_output=0,
                   exact_equal_prediction_frames=0,
                   differing_center_hit_frames=0,
                   left_only_center_hits=0, right_only_center_hits=0,
                   max_abs_box_delta_common_output=0.0,
                   max_abs_score_delta_common_output=0.0)
    for index in indices:
        a, b = left['boxes'][index], right['boxes'][index]
        ea, eb = (left['evaluated_boxes'][index],
                  right['evaluated_boxes'][index])
        target = gt[index]
        ah = _metric(ea, target)['center_hit']
        bh = _metric(eb, target)['center_hit']
        summary['differing_center_hit_frames'] += int(ah != bh)
        summary['left_only_center_hits'] += int(ah and not bh)
        summary['right_only_center_hits'] += int(bh and not ah)
        if a is None and b is None:
            summary['neither_output'] += 1
            summary['exact_equal_prediction_frames'] += 1
        elif a is None:
            summary['right_only_output'] += 1
        elif b is None:
            summary['left_only_output'] += 1
        else:
            summary['both_output'] += 1
            diff = np.abs(np.asarray(a) - np.asarray(b))
            summary['exact_equal_prediction_frames'] += int(not diff.any())
            summary['max_abs_box_delta_common_output'] = max(
                summary['max_abs_box_delta_common_output'],
                float(np.max(diff[:5])))
            summary['max_abs_score_delta_common_output'] = max(
                summary['max_abs_score_delta_common_output'], float(diff[5]))
    return summary


def build_report(root):
    root = Path(root).resolve()
    gt_dir = root / 'crane_project/data/crane_grab/test/annfiles'
    gt_paths = sorted(path for path in gt_dir.glob('*.txt')
                      if not path.name.startswith('._'))
    if len(gt_paths) != 992:
        raise ValueError('Expected 992 fixed-TEST annotations')
    gt_sha = directory_sha256(gt_paths)
    config = root / STUDENT_CONFIG
    arms = {arm: _load_one(root, arm, gt_paths, gt_sha, config)
            for arm in ARMS}
    k1_path = root / 'work_dirs/crane_symeood_k1/epoch_20.pth'
    preflight = _historical_preflight(root, sha256_file(str(k1_path)))
    k1 = _load_checkpoint(k1_path)
    states = {arm: _load_checkpoint(data['checkpoint'])
              for arm, data in arms.items()}
    for arm, state in states.items():
        has_projection = any(key.startswith('semantic_distillation.')
                             for key in state)
        has_adapter = any(key.startswith('bbox_head.semantic_cls_adapter.')
                          for key in state)
        if has_projection != (arm != 'a') or not has_adapter:
            raise ValueError(arm + ' checkpoint architecture mismatch')
    weight_comparisons = {
        arm.upper() + '_vs_K1': compare_states(k1, states[arm])
        for arm in ARMS}
    for arm in ARMS:
        if weight_comparisons[arm.upper() + '_vs_K1']['backbone'][
                'different_tensors']:
            raise ValueError(arm + ' frozen backbone differs from K1')
    for left, right in (('a', 'b'), ('a', 'c'), ('b', 'c')):
        weight_comparisons[left.upper() + '_vs_' + right.upper()] = (
            compare_states(states[left], states[right]))
    gt = []
    for path in gt_paths:
        boxes = parse_dota_txt(str(path))
        if len(boxes) != 1:
            raise ValueError('Expected one GT OBB: ' + path.name)
        gt.append(boxes[0])
    pairs = {}
    groups = dict(all=list(range(len(gt))))
    for domain in ('real', 'sim'):
        groups[domain] = [index for index, path in enumerate(gt_paths)
                          if path.name.startswith(domain + '_')]
    for left, right in (('a', 'b'), ('a', 'c'), ('b', 'c')):
        pairs[left.upper() + '_vs_' + right.upper()] = {
            group: _pair(arms[left], arms[right], gt, indices)
            for group, indices in groups.items()}
    for arm in ARMS:
        del arms[arm]['boxes']
        del arms[arm]['evaluated_boxes']
    return dict(protocol=PROTOCOL,
                evidence_role='post_exposure_fixed_test_diagnosis_only',
                inference_performed=False, checkpoint_selection_performed=False,
                fixed_test_gt_sha256=gt_sha,
                historical_preflight=preflight, arms=arms,
                weight_comparisons=weight_comparisons,
                paired_predictions=pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--out-json', help='Optional new report path')
    args = parser.parse_args()
    report = build_report(args.project_root)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            stream.write(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
