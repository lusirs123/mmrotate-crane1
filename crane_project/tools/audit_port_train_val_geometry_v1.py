#!/usr/bin/env python3
"""Fixed EOOD-B/SymEOOD-B TRAIN/VAL geometry diagnosis. Never opens TEST.

VAL: reuse selected checkpoint PKL and verify provenance + DOTA correspondence.
TRAIN: deterministic sequence-stratified sample, clean and fixed 0.5 shrink,
using VAL inference pipeline, no optimizer or training augmentation randomness.
All sizes/centres are original image coordinates. This is not depth accuracy.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.eval_crane_offline import compute_riou, parse_dota_txt


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def canonical(box):
    box = np.asarray(box, dtype=float).copy()
    if box.shape != (5,) or not np.isfinite(box).all() or np.any(box[2:4] <= 0):
        raise ValueError('Expected finite positive Nx5 OBB')
    if box[2] < box[3]:
        box[2], box[3] = box[3], box[2]
        box[4] += math.pi / 2
    box[4] = (box[4] + math.pi / 2) % math.pi - math.pi / 2
    return box


def angle_error(a, b):
    return float(abs((a - b + math.pi / 2) % math.pi - math.pi / 2) * 180 / math.pi)


def decompose(gt, pred):
    gt = canonical(gt)
    if pred is None:
        return dict(output=False, riou=0., center_hit=False,
                    protocol_angle_error_deg=90., angle_penalty_reason='no_output')
    raw = np.asarray(pred, dtype=float)
    pred = canonical(raw)
    center = float(np.linalg.norm(pred[:2] - gt[:2]))
    angle = angle_error(pred[4], gt[4])
    riou = compute_riou(pred, gt)
    errors = dict(output=True, riou=riou, center_hit=center < 15.,
        center_error_px=center, center_error_over_gt_short=center / gt[3],
        long_edge_relative_error=float(abs(pred[2] / gt[2] - 1)),
        short_edge_relative_error=float(abs(pred[3] / gt[3] - 1)),
        long_edge_signed_log_ratio=float(np.log(pred[2] / gt[2])),
        short_edge_signed_log_ratio=float(np.log(pred[3] / gt[3])),
        angle_error_deg=angle,
        raw_representation_angle_error_deg=angle_error(raw[4], gt[4]),
        protocol_angle_error_deg=angle if center < 10 else 90.,
        angle_penalty_reason='center_ge_10px' if center >= 10 else None,
        angle_axis_well_defined=bool(gt[2] / gt[3] >= 1.2))
    # Descriptive substitutions; correlated contributions are not additive causes.
    for name, fields in [('center', [0, 1]), ('size', [2, 3]), ('angle', [4])]:
        corrected = pred.copy()
        corrected[fields] = gt[fields]
        errors['riou_gain_if_gt_' + name] = float(compute_riou(corrected, gt) - riou)
    return errors


def describe(values):
    if not values:
        return dict(n=0, mean=None, median=None, p90=None, rmse=None)
    v = np.asarray(values, dtype=float)
    return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)),
                p90=float(np.percentile(v, 90)), rmse=float(np.sqrt(np.mean(v*v))))


def longest_failure(rows):
    run = longest = 0
    previous = None
    for row in sorted(rows, key=lambda x: (x['sequence'], x['frame_id'])):
        key = (row['sequence'], row['frame_id'])
        if previous is None or key[0] != previous[0] or key[1] != previous[1] + 1:
            run = 0
        run = run + 1 if row['metrics']['riou'] < .5 else 0
        longest = max(longest, run)
        previous = key
    return longest


def summarize(rows, continuous=False):
    metrics = [r['metrics'] for r in rows]
    out = [m for m in metrics if m['output']]
    report = dict(frames=len(rows), output_frames=len(out),
        output_coverage_pct=100*len(out)/len(rows) if rows else None,
        output_center_hit_pct=100*sum(m['center_hit'] for m in out)/len(out) if out else None,
        all_frame_center_hit_pct=100*sum(m['center_hit'] for m in metrics)/len(rows) if rows else None,
        all_frame_mean_riou=float(np.mean([m['riou'] for m in metrics])) if rows else None,
        riou_correct_frames=sum(m['riou'] >= .5 for m in metrics),
        protocol_angle=describe([m['protocol_angle_error_deg'] for m in metrics]),
        no_output_penalty_count=sum(m.get('angle_penalty_reason') == 'no_output' for m in metrics),
        center_penalty_count=sum(m.get('angle_penalty_reason') == 'center_ge_10px' for m in metrics))
    total_sq = sum(m['protocol_angle_error_deg']**2 for m in metrics)
    penalty_sq = 90**2 * (report['no_output_penalty_count'] + report['center_penalty_count'])
    report['protocol_angle_squared_error_penalty_fraction'] = penalty_sq/total_sq if total_sq else 0.
    for field in ['riou', 'center_error_px', 'center_error_over_gt_short',
        'long_edge_relative_error', 'short_edge_relative_error',
        'long_edge_signed_log_ratio', 'short_edge_signed_log_ratio',
        'angle_error_deg', 'raw_representation_angle_error_deg',
        'riou_gain_if_gt_center', 'riou_gain_if_gt_size', 'riou_gain_if_gt_angle']:
        report[field] = describe([m[field] for m in out])
    report['angle_center_valid'] = describe([m['angle_error_deg'] for m in out if m['center_error_px'] < 10])
    report['angle_nonsquare'] = describe([m['angle_error_deg'] for m in out if m['angle_axis_well_defined']])
    report['longest_riou_failure_run'] = longest_failure(rows) if continuous else None
    report['top_protocol_angle_errors'] = sorted(
        [dict(image=r['image'], error=r['metrics']['protocol_angle_error_deg'],
              reason=r['metrics'].get('angle_penalty_reason')) for r in rows],
        key=lambda x: x['error'], reverse=True)[:20]
    return report


def bin_name(value, boundaries):
    for boundary in boundaries:
        if value < boundary:
            return 'lt_' + str(boundary)
    return 'ge_' + str(boundaries[-1])


def strata(rows):
    groups = defaultdict(list)
    for r in rows:
        groups['domain:' + r['domain']].append(r)
        groups['sequence:' + r['sequence']].append(r)
        groups['input_short:' + bin_name(r['input_gt_short_px'], [16, 24, 32, 48])].append(r)
        groups['aspect:' + bin_name(r['gt_aspect'], [1.2, 2., 3.])].append(r)
        groups['abs_angle:' + bin_name(abs(r['gt_angle_deg']), [30, 60])].append(r)
    return {k: summarize(v, continuous=k.startswith('sequence:') and all(r['split']=='val' for r in v))
            for k, v in sorted(groups.items())}


def paired_report(first, second, continuous=False):
    a, b = {r['image']: r for r in first}, {r['image']: r for r in second}
    if a.keys() != b.keys():
        raise ValueError('Paired image identities differ')
    for key in a:
        if a[key]['image_sha256'] != b[key]['image_sha256'] or a[key]['gt'] != b[key]['gt']:
            raise ValueError('Paired GT/image mismatch: ' + key)
    shared = [key for key in a if a[key]['metrics']['output'] and b[key]['metrics']['output']]
    counts = defaultdict(int)
    for key in a:
        ma, mb = a[key]['metrics'], b[key]['metrics']
        counts['both_output' if ma['output'] and mb['output'] else
               'eood_only_output' if ma['output'] else
               'symeood_only_output' if mb['output'] else 'neither_output'] += 1
    return dict(output_churn=dict(counts),
        eood=summarize(first, continuous), symeood=summarize(second, continuous),
        common_outputs=dict(frames=len(shared),
            eood=summarize([a[k] for k in shared]), symeood=summarize([b[k] for k in shared]),
            eood_strata=strata([a[k] for k in shared]),
            symeood_strata=strata([b[k] for k in shared])),
        eood_strata=strata(first), symeood_strata=strata(second))


def inference_contract(model_cfg):
    """Resolve the actual inference head without changing the frozen config."""
    head = model_cfg['bbox_head']
    if model_cfg.get('type') == 'Eood' and head.get('type') == 'EoodHead':
        predictors = head.get('predictors', [])
        if not predictors or predictors[0].get('type') != 'RotatedEoodHead':
            raise ValueError('Expected EOOD predictor 0 = RotatedEoodHead')
        # EoodHead.get_bboxes delegates to predictors[0]; outer test_cfg is None.
        test_cfg = predictors[0].get('test_cfg')
        location = 'model.bbox_head.predictors[0].test_cfg'
        expected_max = 2000
        postprocess = 'per_level_pre_topk_then_rotated_nms'
    elif model_cfg.get('type') == 'SymEOOD' and head.get('type') == 'SymEOODHead':
        test_cfg = model_cfg.get('test_cfg')
        if test_cfg is None:
            test_cfg = head.get('test_cfg')
        location = 'model.test_cfg_or_bbox_head.test_cfg'
        expected_max = 1
        postprocess = 'padding_filter_then_score_threshold_then_top1_without_nms'
    else:
        raise ValueError('Unsupported inference head for this fixed comparison')
    if (test_cfg is None or test_cfg.get('score_thr') != .05
            or test_cfg.get('max_per_img') != expected_max
            or test_cfg.get('nms_pre') != 2000
            or test_cfg.get('nms', {}).get('iou_thr') != .1):
        raise ValueError('Unexpected frozen inference contract at ' + location)
    return dict(config_location=location, score_thr=.05,
                max_per_img=expected_max, postprocess=postprocess,
                evaluation_selection='first_returned_box_highest_score')


def prediction_array(prediction, max_predictions=1):
    if not isinstance(prediction, (list, tuple)) or len(prediction) != 1:
        raise ValueError('Expected single-class prediction')
    a = np.asarray(prediction[0])
    if (a.ndim != 2 or a.shape[1] != 6 or len(a) > max_predictions
            or not np.isfinite(a).all()):
        raise ValueError('Expected finite Nx6 predictions within the head output limit')
    if len(a) and (np.any(a[:,2:4] <= 0) or np.any(a[:,5] <= .05)
                   or np.any(a[:,5] > 1) or np.any(np.diff(a[:,5]) > 0)):
        raise ValueError('Expected positive OBBs sorted by score above fixed threshold .05')
    return a


def raw_box(prediction, max_predictions=1):
    # ckpt_sweep exports rows in order; evaluator uses the FIRST TXT row.
    a = prediction_array(prediction, max_predictions)
    return a[0].tolist() if len(a) else None


def validate_export(prediction, exported, max_predictions):
    a = prediction_array(prediction, max_predictions)
    if len(exported) != len(a) or any(
            compute_riou(box[:5], text_box) < .99
            for box, text_box in zip(a, exported)):
        raise ValueError('VAL PKL/TXT order or export mismatch')


def make_row(dataset, index, prediction, split, input_scale=1.):
    info = dataset.data_infos[index]
    image = Path(dataset.img_prefix) / info['filename']
    from PIL import Image
    with Image.open(image) as im:
        width, height = im.size
    gt_boxes = dataset.get_ann_info(index)['bboxes']
    if gt_boxes.shape != (1, 5):
        raise ValueError('Expected one positive GT: ' + str(image))
    gt = canonical(gt_boxes[0])
    stem = image.stem
    domain, sequence, frame = stem.rsplit('_', 2)
    base_scale = 1024/max(width,height)
    resized_w, resized_h = int(width*base_scale+.5), int(height*base_scale+.5)
    box_scale = math.sqrt((resized_w/width)*(resized_h/height))*input_scale
    return dict(image=stem, image_sha256=sha(image), split=split,
        domain=domain, sequence=domain+'_'+sequence, frame_id=int(frame),
        gt=gt.tolist(), pred=prediction,
        input_gt_short_px=float(gt[3]*box_scale),
        gt_aspect=float(gt[2]/gt[3]), gt_angle_deg=float(np.degrees(gt[4])),
        metrics=decompose(gt, None if prediction is None else prediction[:5]))


def pick_indices(dataset, per_domain, seed):
    rng = np.random.RandomState(seed)
    domains = defaultdict(lambda: defaultdict(list))
    for i, info in enumerate(dataset.data_infos):
        stem = Path(info['filename']).stem
        domain, seq, _ = stem.rsplit('_', 2)
        domains[domain][seq].append(i)
    result = []
    if set(domains) != {'real','sim'}:
        raise ValueError('Expected real + sim TRAIN')
    for domain in sorted(domains):
        groups = domains[domain]
        if per_domain < len(groups) or sum(map(len, groups.values())) < per_domain:
            raise ValueError('Sample budget cannot cover sequences')
        shuffled = {key:list(rng.permutation(values)) for key,values in sorted(groups.items())}
        selected = []
        while len(selected) < per_domain:
            for key in sorted(shuffled):
                if shuffled[key] and len(selected) < per_domain:
                    selected.append(int(shuffled[key].pop()))
        result.extend(selected)
    return sorted(result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', type=int, default=3)
    ap.add_argument('--train-per-domain', type=int, default=32)
    ap.add_argument('--seed', type=int, default=1701)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--cache-dir', default='work_dirs/port_train_val_geometry_v1_cache')
    for arm in ['eood','symeood']:
        ap.add_argument('--'+arm+'-config', default='crane_project/configs/crane_'+arm+'_k1_port_day2night_aug_b_v1.py')
        ap.add_argument('--'+arm+'-sweep', default='work_dirs/crane_'+arm+'_k1_port_day2night_aug_b_v1/val_sweep_port_v1')
    args = ap.parse_args()
    import os
    os.chdir(ROOT)
    out = Path(args.out_json).resolve()
    if out.exists():
        raise FileExistsError(out)
    from mmcv import Config
    from mmrotate.datasets import build_dataset
    from crane_project.tools.ckpt_sweep import annotation_set_sha256, check_or_record_prediction
    configs, identities, val_rows, train_specs = {}, {}, {}, {}
    for arm in ['eood','symeood']:
        config_path = Path(getattr(args,arm+'_config')).resolve()
        sweep = Path(getattr(args,arm+'_sweep')).resolve()
        selection = json.loads((sweep/'sweep_results.json').read_text())
        selected = selection['selected_checkpoint']
        record = selection['all_checkpoints'][selected]
        checkpoint, pkl = Path(record['checkpoint']), Path(record['results_pkl'])
        if (selection.get('evidence_role') != 'source_val_checkpoint_selection'
                or selection['config_sha256'] != sha(config_path)
                or Path(selection['selected_path']).resolve() != checkpoint.resolve()
                or record['checkpoint_sha256'] != sha(checkpoint)
                or record['results_pkl_sha256'] != sha(pkl)):
            raise ValueError('Selected artifact identity mismatch: '+arm)
        cfg = Config.fromfile(str(config_path))
        contract = inference_contract(cfg.model)
        multi = cfg.data.val.pipeline[1]
        if multi['type'] != 'MultiScaleFlipAug' or tuple(multi['img_scale']) != (1024,1024) or multi.get('flip',False):
            raise ValueError('Expected single-scale unflipped 1024 VAL pipeline')
        if [t['type'] for t in multi['transforms']] != ['RResize','Normalize','Pad','DefaultFormatBundle','Collect']:
            raise ValueError('Unexpected VAL transform order')
        configs[arm] = cfg
        dc = deepcopy(cfg.data.val)
        dc['test_mode'] = True
        ann = Path(dc.get('data_root','')) / dc['ann_file']
        if ann.resolve() != (ROOT/'crane_project/data/crane_grab_port_day2night_v1/val/annfiles').resolve():
            raise ValueError('Only the frozen source VAL directory is permitted')
        if selection['source_val_annotations_sha256'] != annotation_set_sha256(str(ann)):
            raise ValueError('VAL annotation identity changed')
        check_or_record_prediction(str(config_path),str(checkpoint),str(pkl),str(ann),'source_val')
        dataset = build_dataset(dc)
        if len(dataset) != 887:
            raise ValueError('Expected frozen 887 VAL frames')
        with pkl.open('rb') as stream:
            predictions = pickle.load(stream)
        if len(predictions) != len(dataset):
            raise ValueError('VAL prediction count mismatch')
        rows = []
        for i, prediction in enumerate(predictions):
            box = raw_box(prediction, contract['max_per_img'])
            row = make_row(dataset,i,box,'val')
            txt = pkl.parent/'Task1_grab'/(row['image']+'.txt')
            if not txt.exists():
                raise FileNotFoundError(txt)
            exported = parse_dota_txt(str(txt))
            validate_export(prediction, exported, contract['max_per_img'])
            rows.append(row)
        if {r['sequence'] for r in rows} != {'real_seq07','real_seq14','sim_seq10'}:
            raise ValueError('Unexpected source VAL sequence identities')
        val_rows[arm] = rows
        train = []
        for position,entry in enumerate(cfg.data.train):
            expected_split = ['train','train_sim'][position] if position < 2 else None
            expected_ann = ROOT/'crane_project/data/crane_grab_port_day2night_v1'/str(expected_split)/'annfiles'
            if (Path(entry.get('data_root',''))/entry['ann_file']).resolve() != expected_ann.resolve():
                raise ValueError('Only frozen TRAIN directories are permitted')
            spec = deepcopy(entry)
            spec['pipeline'] = deepcopy(cfg.data.val.pipeline)
            spec['test_mode'] = True
            train.append(spec)
        if configs.get('eood') is not None and arm=='symeood' and train != train_specs['eood']:
            raise ValueError('TRAIN eval specs differ between arms')
        if len(train) != 2:
            raise ValueError('Expected two TRAIN domains')
        train_specs[arm] = train
        identities[arm] = dict(config=str(config_path),config_sha256=sha(config_path),
            checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
            val_pkl=str(pkl),val_pkl_sha256=sha(pkl),selection_sha256=sha(sweep/'sweep_results.json'),
            inference_contract=contract)
    # ConcatDataset order is deliberately flattened so identity/sample matching is explicit.
    from mmdet.datasets.dataset_wrappers import ConcatDataset
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmrotate.models import build_detector
    import torch
    cached_rows = {}
    for arm in ['eood','symeood']:
        parts = [build_dataset(spec) for spec in train_specs[arm]]
        dataset = ConcatDataset(parts)
        if len(dataset) != 2558:
            raise ValueError('Expected frozen TRAIN real1810 + sim748')
        # ConcatDataset has no flattened data_infos; a read-only facade suffices.
        class Flat:
            data_infos = [info for part in parts for info in part.data_infos]
        flat = Flat()
        indices = pick_indices(flat,args.train_per_domain,args.seed)
        records = []
        for global_index in indices:
            offset = 0
            for part in parts:
                if global_index < offset + len(part):
                    i = global_index-offset
                    row = make_row(part,i,None,'train_clean')
                    records.append((global_index, row))
                    break
                offset += len(part)
        train_ann_hashes = [annotation_set_sha256(str(Path(s.get('data_root',''))/s['ann_file'])) for s in train_specs[arm]]
        identity = dict(arm=arm, selected=identities[arm],script_sha256=sha(__file__),
            runtime_sources={str(p.relative_to(ROOT)):sha(p) for folder in ['mmrotate/models','mmrotate/core/bbox','mmrotate/core/post_processing','mmrotate/datasets/pipelines'] for p in sorted((ROOT/folder).rglob('*.py'))},
            augmentation_sha256=sha(ROOT/'mmrotate/datasets/pipelines/port_train_augment.py'),
            sampling=dict(per_domain=args.train_per_domain,seed=args.seed,indices=indices),
            annotation_hashes=train_ann_hashes,
            sample_inputs=[dict(index=i,image=r['image'],sha256=r['image_sha256'],gt=r['gt']) for i,r in records])
        cache = Path(args.cache_dir)/(arm+'_train_predictions.json')
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved['identity'] != identity:
                raise ValueError('TRAIN cache identity changed; use a new cache-dir')
            predictions = saved['predictions']
            if set(predictions) != {'clean','shrink_half'} or any(len(v)!=len(records) for v in predictions.values()):
                raise ValueError('Malformed TRAIN cache')
        else:
            torch.cuda.set_device(args.gpu)
            model_cfg = deepcopy(configs[arm].model)
            model_cfg.pretrained = None
            model_cfg.train_cfg = None  # Match tools/test.py construction.
            model = build_detector(model_cfg)
            if configs[arm].get('fp16') is not None:
                from mmcv.runner import wrap_fp16_model
                wrap_fp16_model(model)
            load_checkpoint(model,identities[arm]['checkpoint'],map_location='cpu',strict=True)
            model.cuda(args.gpu).eval()
            predictions = {}
            for condition,scale in [('clean',1.),('shrink_half',.5)]:
                specs = deepcopy(train_specs[arm])
                if scale != 1:
                    for spec in specs:
                        transforms = spec['pipeline'][1]['transforms']
                        position = next(i for i,t in enumerate(transforms) if t['type']=='RResize')+1
                        transforms.insert(position,dict(type='PortIsotropicShrink',prob=1.,scale_range=(scale,scale)))
                current = ConcatDataset([build_dataset(spec) for spec in specs])
                values = []
                for index,_ in records:
                    batch = scatter(collate([current[index]],samples_per_gpu=1),[args.gpu])[0]
                    meta = batch['img_metas'][0][0]
                    expected_name = next(r['image'] for j,r in records if j==index)
                    if len(batch['img']) != 1 or Path(meta['filename']).stem != expected_name or meta.get('flip',False):
                        raise ValueError('TRAIN preprocessing identity changed')
                    with torch.no_grad():
                        result = model(return_loss=False,rescale=True,**batch)[0]
                    values.append(raw_box(result, identities[arm]['inference_contract']['max_per_img']))
                predictions[condition] = values
                print(arm,condition,'inferred',len(values),'TRAIN samples',flush=True)
            write_new(cache,dict(identity=identity,predictions=predictions))
            del model
            torch.cuda.empty_cache()
        cached_rows[arm] = {}
        for condition,scale in [('clean',1.),('shrink_half',.5)]:
            rows = []
            for (_,original), pred in zip(records,predictions[condition]):
                if pred is not None:
                    raw_box([np.asarray([pred])])
                r = deepcopy(original)
                r['pred'] = pred
                r['split'] = 'train_'+condition
                r['input_gt_short_px'] *= scale
                r['metrics'] = decompose(r['gt'],None if pred is None else pred[:5])
                rows.append(r)
            cached_rows[arm][condition] = rows
    report = dict(protocol='port_train_val_geometry_v1',evidence_role='train_val_only_design_diagnosis',
        identities=identities,sampling=dict(seed=args.seed,per_domain=args.train_per_domain),
        val=paired_report(val_rows['eood'],val_rows['symeood'],True),
        train={condition:paired_report(cached_rows['eood'][condition],cached_rows['symeood'][condition])
               for condition in ['clean','shrink_half']},
        train_shrink_sensitivity={arm:dict(
            clean=summarize(cached_rows[arm]['clean']),
            shrink_half=summarize(cached_rows[arm]['shrink_half']),
            common_outputs_clean=summarize([a for a,b in zip(cached_rows[arm]['clean'],cached_rows[arm]['shrink_half']) if a['metrics']['output'] and b['metrics']['output']]),
            common_outputs_shrink=summarize([b for a,b in zip(cached_rows[arm]['clean'],cached_rows[arm]['shrink_half']) if a['metrics']['output'] and b['metrics']['output']]))
            for arm in ['eood','symeood']},
        rows=dict(val=val_rows,train=cached_rows),
        limitations=['Selected VAL models may differ in training seed trajectory; this is not a loss ablation.',
            'Conditional geometry is reported on shared outputs; missing outputs remain in coverage and protocol angles.',
            'TRAIN sample is bounded; shrink_half is a controlled synthetic view, not all historical augmentations.',
            'Near-square GT directions are separately reported; geometric equivalence is not a directed mechanical-axis truth.',
            'Counterfactual IoU gains are descriptive and nonadditive, not a causal loss attribution.',
            'No TEST reading, threshold search, checkpoint reselection or depth estimation occurs.'])
    write_new(out,report)
    brief = dict(protocol=report['protocol'], val_common_outputs=report['val']['common_outputs']['frames'])
    for arm in ['eood','symeood']:
        sim = report['val'][arm+'_strata']['domain:sim']
        brief[arm+'_sim_val'] = dict(frames=sim['frames'], outputs=sim['output_frames'],
            mean_riou=sim['all_frame_mean_riou'],
            protocol_angle_rmse=sim['protocol_angle']['rmse'],
            pure_angle_rmse=sim['angle_error_deg']['rmse'],
            angle_center_valid_rmse=sim['angle_center_valid']['rmse'],
            no_output_penalties=sim['no_output_penalty_count'],
            center_penalties=sim['center_penalty_count'],
            penalty_fraction_of_squared_error=sim['protocol_angle_squared_error_penalty_fraction'])
    print(json.dumps(brief,ensure_ascii=False,indent=2))
    print('Detailed report:',out)


if __name__ == '__main__':
    main()
