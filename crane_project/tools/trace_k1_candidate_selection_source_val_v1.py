"""Trace selected K1 A/B checkpoints on source VAL after checkpoint selection.

This read-only report never selects checkpoints or changes the score threshold.
It checks each replayed top-1 against its saved source-VAL prediction before
reporting candidate scores. Historical V5 C failure frames are observations,
not targets for tuning this experiment.
"""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import torch

from crane_project.tools.diagnose_v5_source_val_candidate_flow import (
    best_iou, require_top1_test_cfg, sha256, stage_summary,
    validate_saved_output)


ROOT = Path(__file__).resolve().parents[2]
ARMS = ('a', 'b')
HISTORICAL_FRAMES = ('sim_seq10_00211', 'sim_seq10_00212',
                     'sim_seq10_00213')


def load_arm(root, arm, annotation_dir):
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmrotate.models import build_detector
    from crane_project.tools.ckpt_sweep import (
        annotation_set_sha256, check_or_record_prediction)

    config = root / ('crane_project/configs/'
                     'crane_symeood_k1_candidate_selection_' + arm + '_v1.py')
    work = root / ('work_dirs/crane_symeood_k1_candidate_selection_' + arm + '_v1')
    selection_file = work / 'source_val_sweep_protocol_v2/sweep_results.json'
    selection = json.loads(selection_file.read_text())
    if (selection.get('evidence_role') != 'source_val_checkpoint_selection'
            or selection.get('center_thresh_px') != 15.0
            or selection.get('config_sha256') != sha256(config)
            or selection.get('source_val_annotations_sha256') !=
            annotation_set_sha256(str(annotation_dir))):
        raise ValueError(arm + ': source-VAL selection identity mismatch')
    chosen = selection['selected_checkpoint']
    if chosen not in {'epoch_16', 'epoch_18', 'epoch_20',
                      'epoch_22', 'epoch_24'}:
        raise ValueError(arm + ': selected checkpoint outside fixed sweep')
    record = selection['all_checkpoints'][chosen]
    checkpoint = work / (chosen + '.pth')
    pkl = Path(record['results_pkl'])
    if (Path(record['checkpoint']).resolve() != checkpoint.resolve()
            or Path(selection['selected_path']).resolve() != checkpoint.resolve()
            or sha256(checkpoint) != record['checkpoint_sha256']
            or sha256(pkl) != record['results_pkl_sha256']):
        raise ValueError(arm + ': checkpoint or prediction checksum mismatch')
    check_or_record_prediction(str(config), str(checkpoint), str(pkl),
                               str(annotation_dir), 'source_val')
    cfg = Config.fromfile(str(config))
    require_top1_test_cfg(cfg.model)
    selection_cfg = cfg.model.bbox_head.get('candidate_selection')
    if (bool(selection_cfg) != (arm == 'b')
            or cfg.model.get('reg_quality_head') is not None
            or cfg.model.get('pqa_head') is not None):
        raise ValueError(arm + ': unexpected model branch in source trace')
    model = build_detector(cfg.model)
    require_top1_test_cfg(dict(test_cfg=model.bbox_head.test_cfg))
    load_checkpoint(model, str(checkpoint), map_location='cpu', strict=True)
    model.cuda().eval()
    with pkl.open('rb') as stream:
        predictions = pickle.load(stream)
    identity = dict(config=str(config), config_sha256=sha256(config),
                    selection=str(selection_file),
                    selection_sha256=sha256(selection_file),
                    checkpoint=str(checkpoint),
                    checkpoint_sha256=sha256(checkpoint),
                    results_pkl=str(pkl), results_pkl_sha256=sha256(pkl))
    return model, cfg, predictions, identity


def trace_one(model, image, meta, gt):
    from mmrotate.core import rotated_anchor_center_inside_flags

    head = model.bbox_head
    with torch.no_grad():
        features = model.extract_feat(image)
        cls_levels, reg_levels = head(features)
        priors = head.anchor_generator.grid_priors(
            [level.shape[-2:] for level in cls_levels], device=image.device)
        all_boxes, all_scores = [], []
        for cls, reg, anchors in zip(cls_levels, reg_levels, priors):
            logits = cls[0].permute(1, 2, 0).reshape(-1, head.cls_out_channels)
            if logits.shape[1] != 1 or not head.use_sigmoid_cls:
                raise ValueError('Expected single-class sigmoid K1 head')
            scores = logits[:, 0].sigmoid()
            deltas = reg[0].permute(1, 2, 0).reshape(-1, 5)
            if head.filter_padding_anchors:
                keep = rotated_anchor_center_inside_flags(
                    anchors, meta['img_shape'])
                anchors, deltas, scores = (anchors[keep], deltas[keep],
                                           scores[keep])
            if len(anchors):
                all_boxes.append(head.bbox_coder.decode(
                    anchors, deltas, max_shape=meta['img_shape']))
                all_scores.append(scores)
        boxes = torch.cat(all_boxes) if all_boxes else image.new_zeros((0, 5))
        scores = torch.cat(all_scores) if all_scores else image.new_zeros((0,))
        summary = stage_summary(boxes, scores, gt, float(head.test_cfg.score_thr))
        if boxes.numel():
            ious = best_iou(boxes, gt)
            good = ious >= 0.5
            bad = ~good
            best_bad = float(scores[bad].max()) if bool(bad.any()) else None
        else:
            best_bad = None
        summary['best_bad_score'] = best_bad
        good_score = summary['best_geometric_score']
        summary['bad_minus_good_score'] = (
            best_bad - good_score if best_bad is not None
            and good_score is not None else None)
        selected = summary['top1_index']
        final = (None if selected is None else
                 torch.cat((boxes[selected], scores[selected:selected + 1]))
                 .cpu().numpy())
    return summary, final


def summarize_rows(rows):
    """Aggregate decision-aligned best scores, with explicit denominator."""
    summary = {}
    for arm in ARMS:
        stages = Counter(row[arm]['stage'] for row in rows)
        score_fields = {}
        for field in ('best_geometric_score', 'best_bad_score',
                      'bad_minus_good_score'):
            values = [row[arm].get(field) for row in rows]
            valid = [value for value in values if value is not None]
            score_fields[field] = dict(count=len(valid),
                                       mean=sum(valid) / len(valid)
                                       if valid else None)
        summary[arm] = dict(frames=len(rows), stages=dict(stages),
                            output_frames=sum(row[arm]['top1_index'] is not None
                                              for row in rows),
                            scores=score_fields)

    def select(predicate):
        return [row['frame'] for row in rows if predicate(row)]

    churn = dict(
        a_only_output=select(lambda r: r['a']['top1_index'] is not None
                             and r['b']['top1_index'] is None),
        b_only_output=select(lambda r: r['b']['top1_index'] is not None
                             and r['a']['top1_index'] is None),
        a_only_success=select(lambda r: r['a']['stage'] == 'success'
                              and r['b']['stage'] != 'success'),
        b_only_success=select(lambda r: r['b']['stage'] == 'success'
                              and r['a']['stage'] != 'success'))
    paired = {}
    for field in ('best_geometric_score', 'best_bad_score',
                  'bad_minus_good_score'):
        deltas = [row['b'].get(field) - row['a'].get(field)
                  for row in rows if row['a'].get(field) is not None
                  and row['b'].get(field) is not None]
        paired[field] = dict(count=len(deltas),
                             mean_b_minus_a=sum(deltas) / len(deltas)
                             if deltas else None)
    transitions = {a: dict(Counter(
        row['b']['stage'] for row in rows if row['a']['stage'] == a))
        for a in ('no_geometric_candidate', 'score_threshold',
                  'top1_ranking', 'success')}
    historical = {frame: next((dict(a=row['a'], b=row['b'])
                               for row in rows if row['frame'] == frame), None)
                  for frame in HISTORICAL_FRAMES}
    return dict(summaries=summary, churn=churn,
                churn_counts={name: len(value) for name, value in churn.items()},
                paired_score_change=paired, stage_transitions=transitions,
                historical_v5_c_frames=historical)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    output = Path(args.out_json)
    if output.exists():
        raise FileExistsError(output)
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from the project root')

    from crane_project.tools import mcml_diag
    from crane_project.tools.eval_crane_offline import parse_seq_frame
    from crane_project.tools.ctx_entry_probe import gt_to_tensor

    annotations = sorted((ROOT / 'crane_project/data/crane_grab/val/annfiles').glob('*.txt'))
    annotations = [p for p in annotations if not p.name.startswith('._')]
    if len(annotations) != 738 or {
            '_'.join(parse_seq_frame(p.name)[:2]) for p in annotations
    } != {'real_seq07', 'sim_seq10'}:
        raise ValueError('Expected fixed real_seq07 + sim_seq10 source VAL')
    torch.cuda.set_device(args.gpu)
    models = {arm: load_arm(ROOT, arm, annotations[0].parent) for arm in ARMS}
    if any(len(item[2]) != len(annotations) for item in models.values()):
        raise ValueError('Saved prediction frame count mismatch')
    compose, scale, flip = mcml_diag.build_test_transforms(models['a'][1])
    rows = []
    for index, annotation in enumerate(annotations):
        image_path = next((annotation.parent.parent / 'images' / (annotation.stem + ext)
                           for ext in ('.jpg', '.png', '.bmp', '.tif')
                           if (annotation.parent.parent / 'images' /
                               (annotation.stem + ext)).is_file()), None)
        if image_path is None:
            raise FileNotFoundError(annotation.stem)
        gts = mcml_diag.parse_dota_ann(str(annotation))
        if len(gts) != 1:
            raise ValueError(annotation.stem + ': expected one GT')
        image, meta, _ = mcml_diag.preprocess_image(
            str(image_path), compose, scale, flip)
        image = image.cuda(args.gpu)
        resized_gt = mcml_diag.scale_obb_to_img(gts[0], meta)
        gt = gt_to_tensor(resized_gt, image.device)
        row = dict(frame=annotation.stem,
                   domain=parse_seq_frame(annotation.name)[0])
        for arm, (model, _, predictions, _) in models.items():
            trace, final = trace_one(model, image, meta, gt)
            validate_saved_output(
                final, predictions[index], meta, annotation.stem + '/' + arm)
            row[arm] = trace
        rows.append(row)
        if (index + 1) % 50 == 0:
            print('verified source-VAL frames: %d/738' % (index + 1), flush=True)
    aggregates = summarize_rows(rows)
    if any(value is None for value in aggregates['historical_v5_c_frames'].values()):
        raise ValueError('Historical source-VAL observation frame absent')
    domain_summaries = {}
    for domain in ('real', 'sim'):
        subset = summarize_rows([row for row in rows if row['domain'] == domain])
        domain_summaries[domain] = dict(
            summaries=subset['summaries'],
            churn_counts=subset['churn_counts'],
            paired_score_change=subset['paired_score_change'])
    report = dict(protocol='k1_candidate_selection_source_val_trace_v1',
                  evidence_role='source_val_diagnosis_after_selection',
                  threshold_fixed=0.05, riou_threshold=0.5,
                  model_identity={arm: item[3] for arm, item in models.items()},
                  score_statistic='per-frame highest good/bad score; means exclude missing values',
                  limitations=['V5 C historical frames are observations, not A/B targets.',
                               'These scores are not checkpoint or threshold selection inputs.'],
                  domain_summaries=domain_summaries,
                  **aggregates, frames=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(protocol=report['protocol'],
                          summaries=report['summaries'],
                          domain_summaries=report['domain_summaries'],
                          churn_counts=report['churn_counts'],
                          paired_score_change=report['paired_score_change'],
                          historical_v5_c_frames=report['historical_v5_c_frames']),
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
