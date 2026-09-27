"""Read-only V5 A/C source-VAL dense-candidate flow diagnosis.

The deployed SymEOOD head has no RPN or NMS. It applies a score threshold per
level, concatenates decoded candidates, then keeps the highest-scoring box.
This tool traces those actual stages and verifies its top-1 against the saved
source-VAL PKL before drawing any inference. It never changes thresholds or
checkpoints and never reads fixed TEST artifacts.
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ARMS = {'a': 'epoch_24', 'c': 'epoch_18'}
CONFIG = ROOT / 'crane_project/configs/crane_symeood_k1_dino_semantic_student_v1.py'
THRESHOLD_RIOU = 0.5


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_arm(root, arm):
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmrotate.models import build_detector
    from crane_project.tools.ckpt_sweep import (
        annotation_set_sha256, check_or_record_prediction)
    work = root / ('work_dirs/crane_symeood_k1_dino_object_background_relation_'
                   + arm + '_v5')
    sweep = work / 'source_val_sweep_protocol_v2'
    selection = json.loads((sweep / 'sweep_results.json').read_text())
    epoch = ARMS[arm]
    if (selection['selected_checkpoint'] != epoch
            or selection['config_sha256'] != sha256(CONFIG)):
        raise ValueError(arm + ': source-VAL selection/config mismatch')
    if (selection.get('evidence_role') != 'source_val_checkpoint_selection'
            or selection.get('center_thresh_px') != 15.0):
        raise ValueError(arm + ': source-VAL protocol mismatch')
    record = selection['all_checkpoints'][epoch]
    checkpoint = work / (epoch + '.pth')
    pkl = Path(record['results_pkl'])
    if (checkpoint.resolve() != Path(record['checkpoint']).resolve()
            or sha256(checkpoint) != record['checkpoint_sha256']
            or sha256(pkl) != record['results_pkl_sha256']):
        raise ValueError(arm + ': selected artifact identity mismatch')
    ann_dir = root / 'crane_project/data/crane_grab/val/annfiles'
    if selection.get('source_val_annotations_sha256') != annotation_set_sha256(str(ann_dir)):
        raise ValueError(arm + ': source-VAL annotations changed')
    check_or_record_prediction(str(CONFIG), str(checkpoint), str(pkl),
                               str(ann_dir), 'source_val')
    cfg = Config.fromfile(str(CONFIG))
    if (cfg.model.get('reg_quality_head') is not None
            or cfg.model.get('pqa_head') is not None
            or cfg.model.get('aux_detach_cls_head') is not None
            or cfg.model.get('platform_context_head') is not None
            or cfg.model.get('platform_context_injector') is not None):
        raise ValueError('This trace supports only the plain SymEOOD main-head path')
    test_cfg = cfg.model.bbox_head.test_cfg
    if test_cfg.score_thr != 0.05 or test_cfg.max_per_img != 1:
        raise ValueError('Unexpected source-VAL postprocessing contract')
    model = build_detector(cfg.model)
    load_checkpoint(model, str(checkpoint), map_location='cpu', strict=True)
    model.cuda().eval()
    with pkl.open('rb') as stream:
        predictions = pickle.load(stream)
    return model, cfg, predictions, dict(checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=sha256(checkpoint), pkl=str(pkl.resolve()),
        pkl_sha256=sha256(pkl), selection_sha256=sha256(sweep / 'sweep_results.json'))


def best_iou(boxes, gt):
    if boxes.numel() == 0:
        return torch.empty((0,), device=boxes.device)
    from mmcv.ops import box_iou_rotated
    return box_iou_rotated(boxes.float(), gt.float()).reshape(-1)


def stage_summary(boxes, scores, gt, score_threshold):
    """Diagnose decoded candidates in resized image coordinates."""
    ious = best_iou(boxes, gt)
    valid = scores > score_threshold
    good = ious >= THRESHOLD_RIOU
    retained = good & valid
    if valid.any():
        top_within_valid = int(torch.argmax(scores[valid]).item())
        selected_index = int(torch.nonzero(valid).reshape(-1)[top_within_valid])
    else:
        selected_index = None
    best_good_score = float(scores[good].max()) if good.any() else None
    best_good_rank = (int((scores > best_good_score).sum()) + 1
                      if best_good_score is not None else None)
    stage = ('no_geometric_candidate' if not good.any() else
             'score_threshold' if not retained.any() else
             'top1_ranking' if selected_index is not None and not good[selected_index] else
             'success')
    return dict(stage=stage, candidate_count=int(scores.numel()),
        geometric_candidate_count=int(good.sum()),
        above_threshold_count=int(valid.sum()),
        geometric_above_threshold_count=int(retained.sum()),
        best_riou=float(ious.max()) if ious.numel() else None,
        best_geometric_score=best_good_score,
        best_geometric_score_rank=best_good_rank,
        top1_index=selected_index,
        top1_score=float(scores[selected_index]) if selected_index is not None else None,
        top1_riou=float(ious[selected_index]) if selected_index is not None else None)


def trace_one(model, image, meta, gt):
    from mmrotate.core import rotated_anchor_center_inside_flags
    head = model.bbox_head
    cfg = head.test_cfg
    with torch.no_grad():
        features = model.extract_feat(image)
        cls_levels, box_levels = head(features)
        anchors = head.anchor_generator.grid_priors(
            [level.shape[-2:] for level in cls_levels], device=image.device)
        all_boxes, all_scores = [], []
        padding_removed = 0
        for cls, reg, prior in zip(cls_levels, box_levels, anchors):
            logits = cls[0].permute(1, 2, 0).reshape(-1, head.cls_out_channels)
            scores = logits.sigmoid() if head.use_sigmoid_cls else logits.softmax(-1)
            if scores.shape[1] != 1:
                raise ValueError('Expected one-class grab head')
            reg = reg[0].permute(1, 2, 0).reshape(-1, 5)
            if head.filter_padding_anchors:
                mask = rotated_anchor_center_inside_flags(prior, meta['img_shape'])
                padding_removed += int((~mask).sum())
                prior, reg, scores = prior[mask], reg[mask], scores[mask]
            if len(prior):
                all_boxes.append(head.bbox_coder.decode(prior, reg, max_shape=meta['img_shape']))
                all_scores.append(scores[:, 0])
        boxes = torch.cat(all_boxes) if all_boxes else image.new_zeros((0, 5))
        scores = torch.cat(all_scores) if all_scores else image.new_zeros((0,))
        summary = stage_summary(boxes, scores, gt, float(cfg.score_thr))
        summary['padding_anchors_removed'] = padding_removed
        if summary['top1_index'] is None:
            output = None
        else:
            i = summary['top1_index']
            output = torch.cat((boxes[i], scores[i:i+1])).detach().cpu().numpy()
    return summary, output


def validate_saved_output(trace, saved, meta, frame):
    raw = np.asarray(saved[0])
    if raw.ndim != 2 or raw.shape[1] != 6 or len(raw) > 1:
        raise ValueError(frame + ': invalid saved single-class prediction')
    if (trace is None) != (len(raw) == 0):
        raise ValueError(frame + ': trace/saved output presence differs')
    if trace is None:
        return
    scale = np.asarray(meta['scale_factor']).reshape(-1)
    expected = trace.copy()
    expected[:4] /= scale[:4]
    if not np.allclose(expected, raw[0], rtol=1e-4, atol=2e-2):
        raise ValueError(frame + ': trace top-1 differs from saved source-VAL prediction')


def summarize(rows, arm):
    from collections import Counter
    return dict(arm=arm, frames=len(rows), stages=dict(Counter(
        row[arm]['stage'] for row in rows)), output_frames=sum(
        row[arm]['top1_index'] is not None for row in rows))


def domain_summary(rows, domain):
    selected = [row for row in rows if row['domain'] == domain]
    return {arm: summarize(selected, arm) for arm in ARMS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = Path(args.out_json)
    if output.exists():
        raise FileExistsError(output)
    if root != ROOT:
        raise ValueError('Run from the same checkout as this script')
    from crane_project.tools import mcml_diag
    from crane_project.tools.eval_crane_offline import parse_seq_frame
    from crane_project.tools.ctx_entry_probe import gt_to_tensor
    annotations = sorted((root / 'crane_project/data/crane_grab/val/annfiles').glob('*.txt'))
    annotations = [p for p in annotations if not p.name.startswith('._')]
    if len(annotations) != 738:
        raise ValueError('Expected 738 source-VAL frames')
    if {parse_seq_frame(p.name)[0] + '_' + parse_seq_frame(p.name)[1]
            for p in annotations} != {'real_seq07', 'sim_seq10'}:
        raise ValueError('Unexpected source-VAL sequences')
    torch.cuda.set_device(args.gpu)
    models = {arm: load_arm(root, arm) for arm in ARMS}
    if any(len(item[2]) != len(annotations) for item in models.values()):
        raise ValueError('Saved source-VAL PKL frame count mismatch')
    compose, scale, flip = mcml_diag.build_test_transforms(models['a'][1])
    rows = []
    for index, annotation in enumerate(annotations):
        image_path = next((annotation.parent.parent / 'images' / (annotation.stem + ext)
                           for ext in ('.jpg', '.png', '.bmp', '.tif')
                           if (annotation.parent.parent / 'images' / (annotation.stem + ext)).is_file()), None)
        if image_path is None:
            raise FileNotFoundError(annotation.stem)
        gts = mcml_diag.parse_dota_ann(str(annotation))
        if len(gts) != 1:
            raise ValueError(annotation.stem + ': expected one GT')
        image, meta, _ = mcml_diag.preprocess_image(str(image_path), compose, scale, flip)
        image = image.cuda(args.gpu)
        resized_gt = mcml_diag.scale_obb_to_img(gts[0], meta)
        gt = gt_to_tensor(resized_gt, image.device)
        row = dict(frame=annotation.stem, domain=parse_seq_frame(annotation.name)[0])
        for arm, (model, _, predictions, _) in models.items():
            trace, final = trace_one(model, image, meta, gt)
            validate_saved_output(final, predictions[index], meta, annotation.stem + '/' + arm)
            row[arm] = trace
        rows.append(row)
        if (index + 1) % 50 == 0:
            print('verified source-VAL frames: %d/738' % (index + 1), flush=True)
    summaries = {arm: summarize(rows, arm) for arm in ARMS}
    domains = {domain: domain_summary(rows, domain) for domain in ('real', 'sim')}
    churn = {kind: [r['frame'] for r in rows if
        (r['a']['top1_index'] is None) != (r['c']['top1_index'] is None) and
        ((r['c']['top1_index'] is not None) == (kind == 'c_only_output'))]
        for kind in ('a_only_output', 'c_only_output')}
    churn_details = {kind: [dict(frame=r['frame'], domain=r['domain'],
                                  a_stage=r['a']['stage'], c_stage=r['c']['stage'],
                                  a_best_riou=r['a']['best_riou'],
                                  c_best_riou=r['c']['best_riou'])
                            for r in rows if r['frame'] in set(churn[kind])]
                     for kind in churn}
    usable_churn = {
        'a_only_riou_hit': [r['frame'] for r in rows
                            if r['a']['stage'] == 'success'
                            and r['c']['stage'] != 'success'],
        'c_only_riou_hit': [r['frame'] for r in rows
                            if r['c']['stage'] == 'success'
                            and r['a']['stage'] != 'success']}
    report = dict(protocol='v5_source_val_candidate_flow_v1', evidence_role='source_val_only',
        inference_trace_only=True, threshold_fixed=0.05, iou_diagnostic_threshold=0.5,
        nms_executed=False, rpn_exists=False, arms={a: m[3] for a, m in models.items()},
        summaries=summaries, domains=domains, output_churn=churn,
        output_churn_details=churn_details,
        usable_riou_churn=usable_churn, frames=rows,
        stage_transitions={a_stage: {c_stage: sum(
            r['a']['stage'] == a_stage and r['c']['stage'] == c_stage
            for r in rows) for c_stage in ('no_geometric_candidate', 'score_threshold',
                                            'top1_ranking', 'success')}
            for a_stage in ('no_geometric_candidate', 'score_threshold',
                            'top1_ranking', 'success')},
        limitations=['The geometric-candidate rate describes decoded dense anchors, not a separate RPN.',
                     'No RPN, class-free preclassification top-K, or NMS runs in the K1/V5 inference path; configured nms_pre is unused.',
                     'Source VAL labels are used only for diagnosis, not threshold selection.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(protocol=report['protocol'], summaries=summaries,
                          output_churn_counts={k: len(v) for k, v in churn.items()}),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
