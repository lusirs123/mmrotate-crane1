"""Read all five fixed SGAR VAL predictions; no model forward or reselection."""
import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils.eood_sgar_protocol_v1 import describe_change, read, save, sha, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import numpy as np
    from crane_project.tools.eval_crane_offline import compute_riou
    from crane_project.tools.eval_port_geometry_ap_v1 import ap11, validate_saved_gt
    from crane_project.tools.analyze_symeood_ablation_first4_v1 import txtbox
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import (
        canonical, row_metrics, method_summary)

    base = args.base.resolve()
    c = verify(base, launch=True)
    sweep_path = base / 'train/ckpt_sweep/sweep_results.json'
    sweep = read(sweep_path)
    assert sweep['selection_config'] == c['checkpoint_selection']['selection_config']
    assert sweep['candidate_epochs'] == c['candidate_epochs']
    assert sweep['config_sha256'] == c['config_sha256']
    assert sweep['source_val_annotations_sha256'] == c['val_annotations_sha256']
    selected = sweep['selected_checkpoint']
    assert selected is None or selected in {'epoch_%d' % e for e in c['candidate_epochs']}
    reference = ROOT / c['references']['VAL_GT_rows']
    assert sha(reference) == c['reference_sha256']['VAL_GT_rows']
    with gzip.open(reference, 'rt') as stream:
        refs = [json.loads(line) for line in stream]
    names = [r['image'] for r in refs]
    assert len(refs) == 887 and names == sorted(names) and len(set(names)) == 887
    for row in refs:
        path = ROOT / 'crane_project/data/crane_grab_port_day2night_v1/val/annfiles' / (row['image'] + '.txt')
        poly = np.asarray([float(v) for v in path.read_text().split()[:8]]).reshape(4, 2)
        validate_saved_gt(row['gt'], poly, txtbox(path))
    sources = {str(reference): sha(reference), str(sweep_path): sha(sweep_path)}
    frames, metrics = [], []
    for epoch in c['candidate_epochs']:
        name = 'epoch_%d' % epoch
        cp = sweep['all_checkpoints'][name]
        checkpoint = Path(cp['checkpoint'])
        assert checkpoint.resolve() == (base / 'train' / (name + '.pth')).resolve()
        assert sha(checkpoint) == cp['checkpoint_sha256']
        prediction = Path(cp['results_pkl'])
        assert sha(prediction) == cp['results_pkl_sha256']
        sources[str(prediction)] = sha(prediction)
        with prediction.open('rb') as stream:
            raw = pickle.load(stream)
        assert len(raw) == 887
        rows = []
        for reference_row, result in zip(refs, raw):
            assert len(result) == 1
            boxes = np.asarray(result[0])
            assert boxes.ndim == 2 and boxes.shape[1] == 6 and np.isfinite(boxes).all()
            pred = boxes[boxes[:, 5].argmax()].tolist() if len(boxes) else None
            riou = compute_riou(np.asarray(canonical(pred)), np.asarray(canonical(reference_row['gt']))) if pred else 0.
            row = {k: reference_row[k] for k in ('image', 'sequence', 'domain', 'frame_id', 'gt')}
            row.update(epoch=name, selected=name == selected, prediction=pred,
                       score=pred[5] if pred else 0.)
            row.update(row_metrics(reference_row['gt'], pred, riou))
            rows.append(row)
        frames.extend(rows)
        for domain in ('all', 'real', 'sim'):
            group = [r for r in rows if domain == 'all' or r['domain'] == domain]
            assert group
            valid = [r for r in group if r['output']]
            # Reuse strict per-frame definitions and sequence-boundary-aware MCML.
            summary_rows = [dict(r, metrics={'candidate': r}) for r in group]
            geometry = method_summary(summary_rows, 'candidate')
            count = len(valid)
            value = dict(
                side='EOOD_plus_SGAR', epoch=name, selected=name == selected,
                fixed_epoch24=epoch == 24, domain=domain,
                frames=len(group), outputs=count,
                TP50=sum(r['riou'] >= .5 for r in group),
                TP75=sum(r['riou'] >= .75 for r in group),
                Recall_pct=100 * sum(r['riou'] >= .5 for r in group)/len(group),
                mean_RIoU=sum(r['riou'] for r in group)/len(group),
                center_MAE_px=sum(r['center_error_px'] for r in valid)/count if count else None,
                center_RMSE_px=math.sqrt(sum(r['center_error_px']**2 for r in valid)/count) if count else None,
                Sim_angle_RMSE_deg=math.sqrt(sum(r['angle_error_deg']**2 for r in valid)/count)
                    if domain == 'sim' and count else None,
                long_MAE_pct=100 * sum(r['long_relative_error'] for r in valid)/count if count else None,
                short_MAE_pct=100 * sum(r['short_relative_error'] for r in valid)/count if count else None,
                joint_size10=sum(r['joint_size10'] for r in group),
                joint_size10_full_pct=100 * sum(r['joint_size10'] for r in group)/len(group),
                MCML=geometry['longest_failure_run']['riou'])
            order = np.argsort(-np.asarray([r['score'] for r in valid], dtype=np.float32))
            for threshold in (.5, .75):
                if not count:
                    value['AP' + str(round(threshold * 100))] = 0.
                    continue
                tp = np.asarray([r['riou'] >= threshold for r in valid], dtype=np.float32)[order].cumsum()
                fp = np.asarray([r['riou'] < threshold for r in valid], dtype=np.float32)[order].cumsum()
                precision = tp / np.maximum(tp + fp, np.finfo(np.float32).eps)
                value['AP' + str(round(threshold * 100))] = 100 * ap11(tp.astype(np.float64)/len(group), precision)
            metrics.append(value)
    chosen = next((r for r in metrics if r['selected'] and r['domain'] == 'all'), None)
    final = next(r for r in metrics if r['fixed_epoch24'] and r['domain'] == 'all')
    comparison = (describe_change(chosen, c['baseline']['baseline_VAL'])
                  if chosen is not None and chosen['outputs'] else None)
    out = base / 'VAL_analysis'
    out.mkdir(exist_ok=False)
    with (out / 'metrics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    with (out / 'per_frame.jsonl').open('x') as stream:
        for row in frames:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    save(out / 'summary.json', dict(
        passed=True, integrity_only=True, model_forwards=0, optimizer_updates=0,
        TEST=False, selection_changed=False, selected_checkpoint=selected,
        selected=chosen, fixed_epoch24=final,
        baseline=c['baseline']['baseline_VAL'], descriptive_comparison=comparison,
        comparison_missing_reason='No selected checkpoint or no selected outputs' if comparison is None else None,
        metrics=metrics, sources=sources, source_sha256=sha(__file__),
        strict_initialization_pair=False, automatic_adoption=False,
        metric_scope=dict(AP='Top1 OBB VOC07 eleven-point, not native complete-output AP',
            full_frame=['Recall', 'mean_RIoU', 'joint_size10_full_pct'],
            output_conditional=['center_MAE_px', 'center_RMSE_px', 'long_MAE_pct', 'short_MAE_pct'],
            angle='Sim only; not a physical swing angle', MCML='Consecutive RIoU<0.5 GT frames, sequence boundaries preserved')))
    print('EOOD_SGAR_FIXED_VAL_READBACK_COMPLETE', selected, flush=True)


if __name__ == '__main__':
    main()
