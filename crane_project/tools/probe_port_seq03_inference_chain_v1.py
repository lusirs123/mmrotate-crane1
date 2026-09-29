"""Fixed TEST diagnosis: export integrity, 103 single-image replays and dense candidates.
No optimization, threshold search, checkpoint selection, or artifact overwrites.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FRAMES = (98, 99, *range(100, 200), 200)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def compare_arrays(a, b, atol=0.02):
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    return dict(match=bool(a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=1e-4)),
                first_shape=list(a.shape), second_shape=list(b.shape),
                max_abs_diff=float(np.max(np.abs(a-b))) if a.shape == b.shape and a.size else None)


def audit_export(dataset, predictions, pred_dir):
    import numpy as np
    from crane_project.tools.eval_crane_offline import parse_dota_txt, compute_riou
    if len(dataset) != len(predictions):
        raise ValueError('Dataset/PKL length mismatch')
    rows = []
    for i, info in enumerate(dataset.data_infos):
        stem = Path(info['filename']).stem
        if not stem.startswith('real_seq03_'):
            continue
        raw = np.asarray(predictions[i][0])
        if raw.ndim != 2 or raw.shape[1] != 6 or len(raw) > 1 or not np.isfinite(raw).all():
            raise ValueError('Invalid top-1 prediction: ' + stem)
        txt = pred_dir / (stem + '.txt')
        boxes = parse_dota_txt(str(txt))
        overlap = compute_riou(raw[0, :5], boxes[0]) if len(raw) == len(boxes) == 1 else None
        rows.append(dict(frame=stem, dataset_index=i, pkl_output_count=len(raw),
                         txt_exists=txt.exists(), txt_output_count=len(boxes),
                         export_riou=overlap,
                         match=bool(txt.exists() and len(raw) == len(boxes)
                                    and (not len(raw) or overlap >= 0.99))))
    if [r['frame'] for r in rows] != [f'real_seq03_{i:05d}' for i in range(1,201)]:
        raise ValueError('Expected ordered seq03 frames 1..200')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', type=int, default=3)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--final-test-dir', required=True,
                    help='Existing final_test/epoch_N directory containing metrics and preds')
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()
    out = Path(args.out_json).resolve()
    os.chdir(ROOT)
    if out.exists():
        raise FileExistsError(out)
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmcv.runner import load_checkpoint
    from mmcv.parallel import collate, scatter
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    from crane_project.tools.diagnose_v5_source_val_candidate_flow import (
        require_top1_test_cfg, trace_one, best_iou)
    from crane_project.tools.ckpt_sweep import annotation_set_sha256

    cfg_path = (ROOT / args.config).resolve()
    ckpt = (ROOT / args.checkpoint).resolve()
    final = (ROOT / args.final_test_dir).resolve()
    pkl = final/'preds/results.pkl'
    report = json.loads((final/'final_test_metrics_v2.json').read_text())
    for field, path in [('config_sha256',cfg_path),('checkpoint_sha256',ckpt),('results_pkl_sha256',pkl)]:
        if report[field] != sha(path):
            raise ValueError('Artifact identity mismatch: '+field)
    cfg = Config.fromfile(str(cfg_path))
    import_modules_from_strings(**cfg.custom_imports)
    require_top1_test_cfg(cfg.model)
    ann_dir = ROOT/cfg.data.test.data_root/cfg.data.test.ann_file
    if report['gt_annotations_sha256'] != annotation_set_sha256(str(ann_dir)):
        raise ValueError('GT identity mismatch')
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    if len(dataset) != 1440:
        raise ValueError('Expected 1440 TEST frames')
    with open(pkl,'rb') as f:
        predictions = pickle.load(f)
    exports = audit_export(dataset, predictions, final/'preds/Task1_grab')
    torch.cuda.set_device(args.gpu)
    model = build_detector(cfg.model)
    load_checkpoint(model, str(ckpt), map_location='cpu', strict=True)
    model.cuda(args.gpu).eval()
    rows = []
    for fid in FRAMES:
        record = exports[fid-1]
        i = record['dataset_index']
        data = scatter(collate([dataset[i]], samples_per_gpu=1), [args.gpu])[0]
        if len(data['img']) != 1:
            raise ValueError('Only single-scale unflipped TEST supported')
        image, meta = data['img'][0], data['img_metas'][0][0]
        if Path(meta['filename']).stem != record['frame']:
            raise ValueError('Preprocessing frame identity mismatch')
        gt = torch.as_tensor(dataset.get_ann_info(i)['bboxes'],device=image.device).clone()
        if gt.shape != (1,5) or meta.get('flip',False):
            raise ValueError('Expected one GT and no flip')
        gt[:,:4] *= gt.new_tensor(meta['scale_factor'])
        with torch.no_grad():
            actual = model(return_loss=False, rescale=True, **data)[0][0]
            summary, traced = trace_one(model,image,meta,gt)
            # Capture raw score maximum and score of the best geometry candidate.
            head = model.bbox_head
            if head.filter_padding_anchors:
                raise ValueError('Plain K1 padding-filter setting changed')
            cls, reg = head(model.extract_feat(image))
            priors = head.anchor_generator.grid_priors([x.shape[-2:] for x in cls],device=image.device)
            scores = torch.cat([x[0].permute(1,2,0).reshape(-1).sigmoid() for x in cls])
            boxes = torch.cat([head.bbox_coder.decode(a,r[0].permute(1,2,0).reshape(-1,5),max_shape=meta['img_shape']) for a,r in zip(priors,reg)])
            ious = best_iou(boxes,gt)
            summary['max_candidate_score'] = float(scores.max())
            summary['highest_score_candidate_riou'] = float(ious[scores.argmax()])
            summary['best_riou_candidate_score'] = float(scores[ious.argmax()])
        traced_array = np.empty((0,6),dtype=np.float32)
        if traced is not None:
            traced_array = traced.copy()[None,:]
            traced_array[:,:4] /= np.asarray(meta['scale_factor'])
        rows.append(dict(frame=record['frame'],
            image_sha256=sha(Path(meta['filename'])),
            single_vs_saved=compare_arrays(actual,predictions[i][0]),
            trace_vs_single=compare_arrays(traced_array,actual),
            single_output=np.asarray(actual).tolist(), saved_output=np.asarray(predictions[i][0]).tolist(),
            candidates=summary))
        print(record['frame'],rows[-1]['single_vs_saved'],summary['stage'],flush=True)
    usable = all(r['match'] for r in exports) and all(r['single_vs_saved']['match'] and r['trace_vs_single']['match'] for r in rows)
    result = dict(protocol='port_seq03_fixed_chain_v1',evidence_role='exposed_test_diagnosis_only',
        checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),config=str(cfg_path),
        final_test_dir=str(final),config_sha256=sha(cfg_path),pkl_sha256=sha(pkl),
        status='CHAIN_CONSISTENT' if usable else 'CHAIN_MISMATCH_REVIEW_BEFORE_ATTRIBUTION',
        score_threshold=0.05,iou_diagnostic_threshold=0.5,frames=list(FRAMES),
        export_rows=exports,replay_rows=rows,
        stage_counts_failure_block=dict(Counter(r['candidates']['stage'] for r in rows
            if 100 <= int(r['frame'].split('_')[-1]) <= 199)),
        stage_counts_controls=dict(Counter(r['candidates']['stage'] for r in rows
            if int(r['frame'].split('_')[-1]) in (98,99,200))),
        limitations=['Historical PKL order is inferred from the same dataset loader and checked against TXT and 103 replays (100..199 plus 98,99,200).',
                      'No causal training diagnosis or threshold recommendation follows from these TEST probes.'])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(result['status'], str(out))


if __name__ == '__main__':
    main()
