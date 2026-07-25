#!/usr/bin/env python3
"""Source-calibrated temporal selection over frozen-DINO top-K proposals.

The detector, DINOv2 backbone, and source-trained ROI head are frozen. Source
annotations identify decoded positive proposals used to calibrate frame-to-frame
OBB motion statistics. Target candidates are generated and a canonical
iteratively segmented Viterbi path is fixed without target annotations; target
labels are read only afterward to report RIoU, top-1 recall, oracle@K, and MCML.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import dino_teacher_frozen_region_audit as audit  # noqa: E402
from crane_project.tools import dino_teacher_source_roi_head_probe as roi_probe  # noqa: E402
from crane_project.tools import frozen_p3_feature_alignment_audit as alignment  # noqa: E402
from crane_project.tools import frozen_p3_objectness_transfer_probe as transfer  # noqa: E402
from crane_project.tools import p3_p4_neighborhood_rescue_audit as neighborhood  # noqa: E402
from crane_project.tools import temporal_reanchor_probe as temporal  # noqa: E402


AUDIT_NAME = 'DINO Top-K Source-Calibrated Temporal Beam Audit V2'
PROTOCOL_VERSION = 2
SOURCE_SPLIT = neighborhood.SOURCE_SPLIT
TARGET_SPLIT = neighborhood.TARGET_SPLIT
TARGET_SEQ = neighborhood.TARGET_SEQ


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument('--config', required=True)
    parser.add_argument('--detector-checkpoint', required=True)
    parser.add_argument('--roi-head-checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-seq', default=neighborhood.SOURCE_SEQ)
    parser.add_argument('--dinov2-repo', required=True)
    parser.add_argument('--dinov2-checkpoint', required=True)
    parser.add_argument('--dinov2-model', default=audit.CANONICAL_MODEL)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dino-gpus', type=int, nargs='+', required=True)
    parser.add_argument('--legacy-sdpa-query-chunk', type=int, default=512)
    parser.add_argument('--dino-height', type=int, default=600)
    parser.add_argument('--dino-max-long-side', type=int, default=1333)
    parser.add_argument('--patch-size', type=int, default=14)
    parser.add_argument('--pool-resolution', type=int, default=7)
    parser.add_argument('--min-roi-in-bounds', type=float, default=0.9)
    parser.add_argument('--detector-candidate-limit', type=int, default=10000)
    parser.add_argument('--beam-size', type=int, default=100)
    parser.add_argument('--boundary-top-m', type=int, default=10)
    parser.add_argument('--roi-chunk-size', type=int, default=16)
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--target-min-wins', type=int, default=26)
    parser.add_argument('--max-mcml', type=int, default=5)
    parser.add_argument('--target-start', type=int,
                        default=neighborhood.TARGET_START)
    parser.add_argument('--target-end', type=int,
                        default=neighborhood.TARGET_END)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def validate_args(args):
    if args.seed != 0:
        raise ValueError('The protocol requires --seed 0')
    if args.detector_candidate_limit < args.beam_size:
        raise ValueError('Detector candidate limit must cover the beam')
    if args.beam_size < 2 or args.boundary_top_m < 1:
        raise ValueError('Beam and boundary sizes must be positive')
    if args.boundary_top_m > args.beam_size:
        raise ValueError('--boundary-top-m cannot exceed --beam-size')
    if args.roi_chunk_size < 1:
        raise ValueError('--roi-chunk-size must be positive')
    if not 0.0 < args.min_roi_in_bounds <= 1.0:
        raise ValueError('--min-roi-in-bounds must be in (0, 1]')


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_roi_head(path: str, device: torch.device):
    payload = torch.load(path, map_location='cpu')
    required = ('state_dict', 'channels', 'pool_resolution', 'hidden_dim')
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(
            'ROI-head checkpoint lacks {}'.format(', '.join(missing)))
    if payload.get('source_only') is not True:
        raise RuntimeError('ROI-head checkpoint is not marked source-only')
    head = roi_probe.TwoFCObjectnessHead(
        int(payload['channels']), int(payload['pool_resolution']),
        int(payload['hidden_dim']))
    head.load_state_dict(payload['state_dict'], strict=True)
    head.to(device)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head, payload


def wrap_half_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + math.pi / 2.0) % math.pi - math.pi / 2.0


def transition_vector(previous: Sequence[float], current: Sequence[float]):
    previous = np.asarray(previous, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    size = math.sqrt(max(float(previous[2] * previous[3]), 1e-6))
    return np.asarray([
        (current[0] - previous[0]) / size,
        (current[1] - previous[1]) / size,
        math.log(max(float(current[2]), 1e-6)
                 / max(float(previous[2]), 1e-6)),
        math.log(max(float(current[3]), 1e-6)
                 / max(float(previous[3]), 1e-6)),
        float(wrap_half_pi(np.asarray([current[4] - previous[4]]))[0]),
    ], dtype=np.float64)


def parse_grab_boxes(annotation: str) -> List[np.ndarray]:
    diag = transfer.entry_probe.get_diag()
    boxes = []
    for gt in diag.parse_dota_ann(annotation):
        if gt.get('cls') != 'grab':
            continue
        boxes.append(np.asarray([
            gt['cx'], gt['cy'], gt['w'], gt['h'],
            math.radians(gt['angle'])], dtype=np.float64))
    return boxes


def select_source_decoded_positive(scores: torch.Tensor,
                                   ious: torch.Tensor,
                                   layout: Sequence[Dict],
                                   riou_thr: float):
    ordered = roi_probe.candidate_order(
        scores, ious, layout, level=None, min_iou=riou_thr)
    return None if not ordered else int(ordered[0])


def collect_source_decoded_transitions(
        detector, transforms, img_scale, flip, args):
    from mmcv.ops import box_iou_rotated

    records = [
        record for record in transfer.discover_labeled_records(
            args.data_root, SOURCE_SPLIT, 0)
        if record['seq'] == args.source_seq]
    records.sort(key=lambda record: int(record['frame']))
    transitions = []
    rows = []
    previous_frame = None
    previous_box = None
    for record_index, record in enumerate(records):
        img_tensor, detector_meta, _stats = (
            transfer.entry_probe.get_diag().preprocess_image(
                record['image'], transforms, img_scale, flip))
        if img_tensor is None:
            raise RuntimeError(
                'Source preprocessing failed: {}'.format(record['image']))
        img_tensor = img_tensor.cuda('cuda:{}'.format(args.gpu))
        with torch.no_grad():
            detector_features = detector.extract_feat(img_tensor)
            _head, boxes, scores, layout, _decode = (
                transfer.forward_main_candidates(
                    detector, detector_features,
                    detector_meta['img_shape']))
            gt_boxes = transfer.scaled_gt_tensors(
                record, detector_meta, boxes.device)
            if gt_boxes.numel() == 0:
                max_ious = torch.zeros_like(scores)
            else:
                max_ious = box_iou_rotated(
                    boxes[:, :5].float(), gt_boxes.float()).max(dim=1).values
            selected_index = select_source_decoded_positive(
                scores, max_ious, layout, args.riou_thr)
            boxes_original = temporal.decoded_boxes_to_ori(
                boxes[:, :5].detach().cpu(), detector_meta)
            scores_cpu = scores.detach().cpu()
            ious_cpu = max_ious.detach().cpu()
        del img_tensor, detector_features, boxes, scores, gt_boxes, max_ious
        frame = int(record['frame'])
        if selected_index is None:
            rows.append(dict(
                frame=frame, matched_positive=False,
                reason='no_decoded_candidate_at_riou_threshold'))
            previous_frame = None
            previous_box = None
            if ((record_index + 1) % 25 == 0
                    or record_index + 1 == len(records)):
                print('[source-transition] {}/{} images matched={} transitions={}'.format(
                    record_index + 1, len(records),
                    sum(item['matched_positive'] for item in rows),
                    len(transitions)))
            continue
        location = layout[selected_index]
        current_box = boxes_original[selected_index].numpy().astype(
            np.float64)
        row = dict(
            frame=frame, matched_positive=True,
            candidate_index=int(selected_index),
            level=int(location['level']),
            anchor_id=int(location['anchor_id']),
            main_cls_score=float(scores_cpu[selected_index].item()),
            riou=float(ious_cpu[selected_index].item()),
            obb_original=[float(value) for value in current_box.tolist()],
            contributes_transition=bool(
                previous_frame is not None and frame == previous_frame + 1))
        if previous_frame is not None and frame == previous_frame + 1:
            transitions.append(transition_vector(previous_box, current_box))
        rows.append(row)
        previous_frame = frame
        previous_box = current_box
        if (record_index + 1) % 25 == 0 or record_index + 1 == len(records):
            print('[source-transition] {}/{} images matched={} transitions={}'.format(
                record_index + 1, len(records),
                sum(item['matched_positive'] for item in rows),
                len(transitions)))
    if len(transitions) < 10:
        raise RuntimeError('Too few consecutive decoded source transitions')
    matched_rows = [row for row in rows if row['matched_positive']]
    level_counts = {}
    anchor_counts = {}
    for row in matched_rows:
        level_key = str(row['level'])
        anchor_key = str(row['anchor_id'])
        level_counts[level_key] = level_counts.get(level_key, 0) + 1
        anchor_counts[anchor_key] = anchor_counts.get(anchor_key, 0) + 1
    collection = dict(
        source_image_count=len(records),
        matched_positive_count=len(matched_rows),
        transition_count=len(transitions),
        selected_level_counts=level_counts,
        selected_anchor_counts=anchor_counts,
        skipped_frames=[int(row['frame']) for row in rows
                        if not row['matched_positive']])
    return np.stack(transitions, axis=0), rows, collection


def robust_transition_model(transitions: np.ndarray,
                            beam_size: int) -> Dict:
    center = np.median(transitions, axis=0)
    mad = np.median(np.abs(transitions - center), axis=0) * 1.4826
    std = transitions.std(axis=0)
    scale = np.maximum.reduce([
        mad, std * 0.25, np.full_like(mad, 1e-3)])
    z = np.clip((transitions - center) / scale, -5.0, 5.0)
    costs = np.square(z).sum(axis=1)
    q90 = float(np.quantile(costs, 0.90))
    q99 = float(np.quantile(costs, 0.99))
    transition_weight = math.log(float(beam_size)) / max(q90, 1e-6)
    return dict(
        count=int(transitions.shape[0]),
        center=center, scale=scale, source_costs=costs,
        q90_cost=q90, q99_cost=q99,
        transition_weight=float(transition_weight))


def transition_cost(previous: Sequence[float], current: Sequence[float],
                    model: Dict) -> float:
    vector = transition_vector(previous, current)
    z = np.clip(
        (vector - model['center']) / model['scale'], -5.0, 5.0)
    return float(np.square(z).sum())


def transition_cost_matrix(previous: Sequence[Dict], current: Sequence[Dict],
                           model: Dict) -> np.ndarray:
    matrix = np.empty((len(previous), len(current)), dtype=np.float64)
    for previous_index, previous_item in enumerate(previous):
        for current_index, current_item in enumerate(current):
            matrix[previous_index, current_index] = transition_cost(
                previous_item['obb_original'],
                current_item['obb_original'], model)
    return matrix


def automatic_segments(frames: Sequence[Dict], model: Dict,
                       boundary_top_m: int) -> List[Tuple[int, int]]:
    if not frames:
        return []
    boundaries = [0]
    for frame_index in range(1, len(frames)):
        previous = frames[frame_index - 1]['candidates'][:boundary_top_m]
        current = frames[frame_index]['candidates'][:boundary_top_m]
        if not previous or not current:
            boundaries.append(frame_index)
            continue
        matrix = transition_cost_matrix(previous, current, model)
        if float(matrix.min()) > float(model['q99_cost']):
            boundaries.append(frame_index)
    boundaries.append(len(frames))
    return [(boundaries[index], boundaries[index + 1])
            for index in range(len(boundaries) - 1)]


def rank_emissions(count: int, beam_size: int) -> np.ndarray:
    ranks = np.arange(1, count + 1, dtype=np.float64)
    return np.log((float(beam_size) + 1.0) / ranks)


def viterbi_segment(frames: Sequence[Dict], model: Dict,
                    beam_size: int) -> List[int]:
    if not frames:
        return []
    scores = rank_emissions(len(frames[0]['candidates']), beam_size)
    backpointers = []
    for frame_index in range(1, len(frames)):
        previous = frames[frame_index - 1]['candidates']
        current = frames[frame_index]['candidates']
        costs = transition_cost_matrix(previous, current, model)
        normalized_previous = scores - float(scores.max())
        transitions = (normalized_previous[:, None]
                       - float(model['transition_weight']) * costs)
        predecessors = transitions.argmax(axis=0)
        scores = (rank_emissions(len(current), beam_size)
                  + transitions[predecessors,
                                np.arange(len(current))])
        backpointers.append(predecessors)
    selected = [int(np.argmax(scores))]
    for predecessors in reversed(backpointers):
        selected.append(int(predecessors[selected[-1]]))
    selected.reverse()
    return selected


def segmented_viterbi(frames: Sequence[Dict], model: Dict,
                      beam_size: int, boundary_top_m: int):
    segments = automatic_segments(frames, model, boundary_top_m)
    selected = [None] * len(frames)
    segment_rows = []
    for start, end in segments:
        local = viterbi_segment(frames[start:end], model, beam_size)
        for offset, candidate_index in enumerate(local):
            selected[start + offset] = int(candidate_index)
        segment_rows.append(dict(
            start_index=int(start), end_index=int(end - 1),
            start_frame=int(frames[start]['frame']),
            end_frame=int(frames[end - 1]['frame']),
            length=int(end - start)))
    return selected, segment_rows


def segments_from_boundaries(boundaries: Sequence[int],
                             frame_count: int) -> List[Tuple[int, int]]:
    normalized = sorted(set(
        [0, int(frame_count)] + [int(value) for value in boundaries
                                 if 0 < int(value) < int(frame_count)]))
    return [(normalized[index], normalized[index + 1])
            for index in range(len(normalized) - 1)]


def selected_path_transition_rows(frames: Sequence[Dict],
                                  selected: Sequence[int],
                                  model: Dict,
                                  boundaries: Sequence[int]) -> List[Dict]:
    boundary_set = set(int(value) for value in boundaries)
    rows = []
    for frame_index, (frame, candidate_index) in enumerate(
            zip(frames, selected)):
        if frame_index == 0:
            rows.append(dict(
                frame=int(frame['frame']), previous_frame=None,
                boundary_before=True, cost=None,
                above_q90=False, above_q99=False))
            continue
        cost = transition_cost(
            frames[frame_index - 1]['candidates'][selected[frame_index - 1]][
                'obb_original'],
            frame['candidates'][candidate_index]['obb_original'], model)
        rows.append(dict(
            frame=int(frame['frame']),
            previous_frame=int(frames[frame_index - 1]['frame']),
            boundary_before=bool(frame_index in boundary_set),
            cost=float(cost),
            above_q90=bool(cost > float(model['q90_cost'])),
            above_q99=bool(cost > float(model['q99_cost']))))
    return rows


def refined_segmented_viterbi(frames: Sequence[Dict], model: Dict,
                              beam_size: int, boundary_top_m: int):
    preliminary = automatic_segments(frames, model, boundary_top_m)
    boundaries = {start for start, _end in preliminary if start > 0}
    iterations = []
    selected = [None] * len(frames)
    for iteration in range(max(1, len(frames))):
        segments = segments_from_boundaries(boundaries, len(frames))
        for start, end in segments:
            local = viterbi_segment(frames[start:end], model, beam_size)
            for offset, candidate_index in enumerate(local):
                selected[start + offset] = int(candidate_index)
        transition_rows = selected_path_transition_rows(
            frames, selected, model, boundaries)
        new_boundaries = {
            index for index, row in enumerate(transition_rows)
            if index > 0 and not row['boundary_before'] and row['above_q99']}
        iterations.append(dict(
            iteration=int(iteration + 1),
            segment_count=len(segments),
            added_boundaries=sorted(int(value) for value in new_boundaries),
            added_boundary_frames=[int(frames[value]['frame'])
                                   for value in sorted(new_boundaries)]))
        if not new_boundaries:
            break
        boundaries.update(new_boundaries)
    else:
        raise RuntimeError('Selected-path boundary refinement did not converge')

    segments = segments_from_boundaries(boundaries, len(frames))
    segment_rows = []
    segment_ids = [None] * len(frames)
    preliminary_boundaries = {start for start, _end in preliminary
                              if start > 0}
    for segment_id, (start, end) in enumerate(segments):
        for index in range(start, end):
            segment_ids[index] = int(segment_id)
        segment_rows.append(dict(
            segment_id=int(segment_id),
            start_index=int(start), end_index=int(end - 1),
            start_frame=int(frames[start]['frame']),
            end_frame=int(frames[end - 1]['frame']),
            length=int(end - start),
            boundary_origin=('sequence_start' if start == 0 else
                             'preliminary_top_m_q99' if
                             start in preliminary_boundaries else
                             'selected_path_q99')))
    transition_rows = selected_path_transition_rows(
        frames, selected, model, boundaries)
    refinement = dict(
        converged=True,
        iteration_count=len(iterations),
        preliminary_segment_count=len(preliminary),
        final_segment_count=len(segments),
        boundary_indices=sorted(int(value) for value in boundaries),
        boundary_frames=[int(frames[value]['frame'])
                         for value in sorted(boundaries)],
        iterations=iterations)
    return selected, segment_rows, segment_ids, transition_rows, refinement


def detector_level0_order(scores: torch.Tensor,
                          layout: Sequence[Dict]) -> List[int]:
    indices = [index for index, location in enumerate(layout)
               if int(location['level']) == 0]
    indices.sort(key=lambda index: float(scores[index].item()), reverse=True)
    return indices


def collect_unlabelled_target_candidates(
        detector, dino, head, transforms, img_scale, flip, args,
        dino_device: torch.device, head_device: torch.device):
    diag = transfer.entry_probe.get_diag()
    frames = []
    for frame_id in range(args.target_start, args.target_end + 1):
        img_path, ann_path = diag.find_files(
            args.data_root, TARGET_SPLIT, TARGET_SEQ, frame_id)
        if img_path is None or ann_path is None:
            raise RuntimeError('Missing target-dev frame {}'.format(frame_id))
        img_tensor, detector_meta, _stats = diag.preprocess_image(
            img_path, transforms, img_scale, flip)
        if img_tensor is None:
            raise RuntimeError('Target preprocessing failed')
        img_tensor = img_tensor.cuda('cuda:{}'.format(args.gpu))
        with torch.no_grad():
            detector_features = detector.extract_feat(img_tensor)
            _main_head, boxes, scores, layout, _decode = (
                transfer.forward_main_candidates(
                    detector, detector_features, detector_meta['img_shape']))
            boxes_cpu = boxes.detach().cpu()
            scores_cpu = scores.detach().cpu()
        del img_tensor, detector_features, boxes, scores
        detector_order = detector_level0_order(scores_cpu, layout)
        detector_order = detector_order[:args.detector_candidate_limit]
        dino_feature, dino_meta = audit._prepare_image_features(
            dino, img_path, args.dino_height, args.patch_size,
            args.dino_max_long_side, dino_device)
        selections = roi_probe.valid_candidate_selections(
            detector_order, boxes_cpu, detector_meta, dino_meta, dino_feature,
            args.patch_size, args.pool_resolution, args.min_roi_in_bounds,
            args.detector_candidate_limit)
        logits = roi_probe.score_target_candidates(
            head, dino_feature, selections, args, head_device)
        boxes_original = temporal.decoded_boxes_to_ori(
            boxes_cpu[:, :5].clone(), detector_meta)
        ranked = []
        for selection, logit in zip(selections, logits):
            index = int(selection['index'])
            location = layout[index]
            ranked.append(dict(
                candidate_index=index,
                detector_rank=int(selection['detector_rank']),
                objectness_logit=float(logit),
                main_cls_score=float(scores_cpu[index].item()),
                level=int(location['level']),
                anchor_id=int(location['anchor_id']),
                obb_original=[float(value)
                              for value in boxes_original[index].tolist()]))
        ranked.sort(key=lambda item: item['objectness_logit'], reverse=True)
        for rank, item in enumerate(ranked, start=1):
            item['dino_rank'] = int(rank)
        frames.append(dict(
            frame=int(frame_id), image=img_path, annotation=ann_path,
            detector_level0_count=len(detector_level0_order(scores_cpu, layout)),
            detector_pre_topk_count=len(detector_order),
            valid_dino_roi_count=len(ranked),
            candidates=ranked[:args.beam_size],
            _all_ranked_candidates=ranked))
        print('[target-beam] frame={} valid={} beam={}'.format(
            frame_id, len(ranked), min(len(ranked), args.beam_size)))
        del dino_feature
    if any(not frame['candidates'] for frame in frames):
        raise RuntimeError('At least one target frame has an empty DINO beam')
    return frames


def gt_tensor_original(annotation: str, device) -> torch.Tensor:
    boxes = parse_grab_boxes(annotation)
    if not boxes:
        return torch.empty((0, 5), dtype=torch.float32, device=device)
    return torch.tensor(np.stack(boxes), dtype=torch.float32, device=device)


def longest_miss(rows: Sequence[Dict], key: str) -> int:
    longest = 0
    current = 0
    previous = None
    for row in rows:
        frame = int(row['frame'])
        if previous is None or frame != previous + 1:
            current = 0
        if row[key]:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
        previous = frame
    return int(longest)


def evaluate_after_selection(frames: Sequence[Dict], selected: Sequence[int],
                             args):
    from mmcv.ops import box_iou_rotated

    rows = []
    for frame, selected_index in zip(frames, selected):
        gt = gt_tensor_original(frame['annotation'], torch.device('cpu'))
        all_candidates = frame['_all_ranked_candidates']
        all_boxes = torch.tensor(
            [item['obb_original'] for item in all_candidates],
            dtype=torch.float32)
        ious = (torch.zeros(len(all_candidates), dtype=torch.float32)
                if gt.numel() == 0 else box_iou_rotated(
                    all_boxes.float(), gt.float()).max(dim=1).values)
        selected_candidate = frame['candidates'][int(selected_index)]
        selected_riou = float(ious[
            int(selected_candidate['dino_rank']) - 1].item())
        top1_riou = float(ious[0].item())
        best_usable_rank = None
        for index, riou in enumerate(ious.tolist(), start=1):
            if float(riou) >= args.riou_thr:
                best_usable_rank = int(index)
                break
        oracle_hits = {
            str(k): bool(best_usable_rank is not None
                         and best_usable_rank <= k)
            for k in (1, 5, 10, 20, args.beam_size)}
        rows.append(dict(
            frame=int(frame['frame']),
            selected_candidate=selected_candidate,
            selected_riou=selected_riou,
            selected_hit=bool(selected_riou >= args.riou_thr),
            dino_top1_riou=top1_riou,
            dino_top1_hit=bool(top1_riou >= args.riou_thr),
            best_usable_rank=best_usable_rank,
            geometry_eligible=bool(best_usable_rank is not None),
            oracle_hits=oracle_hits))
    return rows


def summarize(rows: Sequence[Dict], args) -> Dict:
    selected_hits = int(sum(row['selected_hit'] for row in rows))
    top1_hits = int(sum(row['dino_top1_hit'] for row in rows))
    oracle = {}
    for k in (1, 5, 10, 20, args.beam_size):
        key = str(k)
        oracle[key] = dict(
            hits=int(sum(row['oracle_hits'][key] for row in rows)),
            mcml=longest_miss([
                dict(frame=row['frame'], hit=row['oracle_hits'][key])
                for row in rows], 'hit'))
    return dict(
        frame_count=len(rows),
        geometry_eligible_count=int(sum(
            row['geometry_eligible'] for row in rows)),
        geometry_misses=[int(row['frame']) for row in rows
                         if not row['geometry_eligible']],
        dino_top1=dict(hits=top1_hits,
                       mcml=longest_miss(rows, 'dino_top1_hit')),
        temporal_selected=dict(
            hits=selected_hits,
            mcml=longest_miss(rows, 'selected_hit')),
        oracle_at_k=oracle)


def make_decision(summary: Dict, args) -> str:
    expected = (
        summary['frame_count'] == 33
        and summary['geometry_eligible_count'] == 31
        and summary['geometry_misses'] == [164, 167])
    if not expected:
        return 'AUDIT_INVALID'
    selected = summary['temporal_selected']
    if (selected['hits'] >= args.target_min_wins
            and selected['mcml'] <= args.max_mcml):
        return 'TEMPORAL_BEAM_RESTORES_ORDERING'
    if selected['mcml'] <= args.max_mcml:
        return 'TEMPORAL_BEAM_REDUCES_MCML_ONLY'
    return 'TEMPORAL_BEAM_INSUFFICIENT'


def serializable_transition_model(model: Dict) -> Dict:
    return dict(
        count=int(model['count']),
        center=[float(value) for value in model['center'].tolist()],
        scale=[float(value) for value in model['scale'].tolist()],
        q90_cost=float(model['q90_cost']),
        q99_cost=float(model['q99_cost']),
        transition_weight=float(model['transition_weight']))


def main():
    args = parse_args()
    validate_args(args)
    roi_probe.set_seed(args.seed)
    head_device = torch.device('cuda:{}'.format(args.gpu))
    dino_devices = [torch.device('cuda:{}'.format(gpu))
                    for gpu in args.dino_gpus]
    dino_device = dino_devices[0]

    detector, cfg = transfer.entry_probe.load_model(
        args.config, args.detector_checkpoint, args.gpu)
    transfer.freeze_detector(detector)
    detector_versions = alignment.module_parameter_versions(detector)
    diag = transfer.entry_probe.get_diag()
    transforms, img_scale, flip = diag.build_test_transforms(cfg)
    source_transitions, source_transition_rows, source_collection = (
        collect_source_decoded_transitions(
            detector, transforms, img_scale, flip, args))
    transition_model = robust_transition_model(
        source_transitions, args.beam_size)

    dino, loaded_patch_size = audit.load_frozen_dinov2(
        args.dinov2_repo, args.dinov2_checkpoint,
        args.dinov2_model, dino_devices,
        args.legacy_sdpa_query_chunk)
    if int(loaded_patch_size) != int(args.patch_size):
        raise RuntimeError('Unexpected DINO patch size')
    dino_versions = alignment.module_parameter_versions(dino)
    head, head_checkpoint = load_frozen_roi_head(
        args.roi_head_checkpoint, head_device)
    if int(head_checkpoint['pool_resolution']) != int(args.pool_resolution):
        raise RuntimeError('ROI-head pool resolution mismatch')
    head_versions = alignment.module_parameter_versions(head)

    frames = collect_unlabelled_target_candidates(
        detector, dino, head, transforms, img_scale, flip, args,
        dino_device, head_device)
    (selected, segments, segment_ids, transition_rows,
     refinement) = refined_segmented_viterbi(
        frames, transition_model, args.beam_size, args.boundary_top_m)

    # Target annotations are first consumed after the complete path is fixed.
    rows = evaluate_after_selection(frames, selected, args)
    summary = summarize(rows, args)
    decision = make_decision(summary, args)

    detector_unchanged = (
        detector_versions == alignment.module_parameter_versions(detector))
    dino_unchanged = (
        dino_versions == alignment.module_parameter_versions(dino))
    head_unchanged = (
        head_versions == alignment.module_parameter_versions(head))
    if not detector_unchanged or not dino_unchanged or not head_unchanged:
        raise RuntimeError('Frozen parameter invariant failed')

    frame_output = []
    for frame, row, segment_id, transition_row in zip(
            frames, rows, segment_ids, transition_rows):
        frame_output.append(dict(
            frame=int(frame['frame']),
            segment_id=int(segment_id),
            selected_transition=transition_row,
            detector_level0_count=int(frame['detector_level0_count']),
            detector_pre_topk_count=int(frame['detector_pre_topk_count']),
            valid_dino_roi_count=int(frame['valid_dino_roi_count']),
            beam_candidates=frame['candidates'],
            evaluation=row))
    payload = dict(
        audit=AUDIT_NAME, protocol_version=PROTOCOL_VERSION,
        config=os.path.abspath(args.config),
        detector_checkpoint=os.path.abspath(args.detector_checkpoint),
        dinov2_checkpoint=os.path.abspath(args.dinov2_checkpoint),
        roi_head_checkpoint=os.path.abspath(args.roi_head_checkpoint),
        roi_head_sha256=file_sha256(args.roi_head_checkpoint),
        protocol=dict(
            source_seq=args.source_seq,
            source_calibration=(
                'consecutive_source_GT_matched_decoded_positive_proposals'),
            target_candidate_selection_uses_labels=False,
            target_segmentation_uses_labels=False,
            target_labels_first_used='after_complete_path_fixed',
            detector_candidate_limit=int(args.detector_candidate_limit),
            beam_size=int(args.beam_size),
            boundary_top_m=int(args.boundary_top_m),
            emission='log((beam_size+1)/dino_rank)',
            transition='source_robust_standardized_center_size_angle_cost',
            transition_weight='log(beam_size)/source_q90_cost',
            boundary=(
                'preliminary_top_m_q99_then_iterative_selected_path_q99')),
        isolation=dict(
            creates_optimizer=False, performs_backward=False,
            writes_checkpoint=False,
            detector_frozen=True,
            detector_parameters_unchanged=detector_unchanged,
            dinov2_frozen=True, dinov2_parameters_unchanged=dino_unchanged,
            roi_head_frozen=True, roi_head_parameters_unchanged=head_unchanged,
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            target_labels_used_for_candidate_selection=False,
            target_labels_used_for_segmentation=False,
            target_labels_used_for_evaluation_only=True),
        source_transition_model=serializable_transition_model(
            transition_model),
        source_transition_collection=source_collection,
        source_transition_rows=source_transition_rows,
        segments=segments,
        segmentation_refinement=refinement,
        target_dev=dict(summary=summary, rows=frame_output),
        decision=decision)
    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(out_dir, exist_ok=True)
    replacements = roi_probe.write_json_atomic(args.out_json, payload)
    print('[temporal-beam] {} hits={}/{} mcml={}'.format(
        decision, summary['temporal_selected']['hits'],
        summary['frame_count'], summary['temporal_selected']['mcml']))
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
