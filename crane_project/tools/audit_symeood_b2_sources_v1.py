"""Uniform CPU readback of existing A/B, EOOD and tower controls; zero inference."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def saved_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def run(out, history_identities=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This audit is CPU-only')
    import numpy as np
    from crane_project.tools.eval_crane_offline import compute_riou
    from crane_project.tools.eval_port_geometry_ap_v1 import ap11, validate_saved_gt
    from crane_project.tools.analyze_symeood_ablation_first4_v1 import txtbox
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import canonical, row_metrics

    out.mkdir(exist_ok=False, parents=True)
    base = ROOT / 'work_dirs/symeood_110_reg_tower_20261030_v1'
    val_s = ROOT / 'work_dirs/symeood_110_retrain_seed0_20261026_v1/downstream/lvsr/fixed_b/epoch_01_val.rows.jsonl.gz'
    val_e = ROOT / 'work_dirs/detector_module_ablation_v1_20261028/eood_l/epoch_17_val.rows.jsonl.gz'
    test = ROOT / 'work_dirs/smallpaper_readiness_20261028_v1/saved_predictions_unified.rows.jsonl'
    refs = dict(val=saved_rows(val_s), test=saved_rows(test))
    e_val = saved_rows(val_e)
    assert [(r['image'], r['gt']) for r in refs['val']] == [(r['image'], r['gt']) for r in e_val]
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (val_s, val_e, test)}
    models = {'E': dict(val=[r['b'] for r in e_val], test=[r['eood_aug'] for r in refs['test']])}
    identities = {}
    for split, rr in refs.items():
        assert len(rr) == (887 if split == 'val' else 1440)
        assert [r['image'] for r in rr] == sorted(r['image'] for r in rr)
        anns = ROOT / ('crane_project/data/crane_grab_port_day2night_v1/' + split + '/annfiles')
        for row in rr:
            path = anns / (row['image'] + '.txt')
            poly = np.asarray([float(x) for x in path.read_text().split()[:8]]).reshape(4, 2)
            validate_saved_gt(row['gt'], poly, txtbox(path))
    identity_path = history_identities or ROOT / 'work_dirs/symeood_ablation_v1/analysis_all8_20261024/selected_identity_all8.json'
    assert sha(identity_path) == 'c709b70e1644248857cab95b58a072c8d587701f402e68f5774ae166a7c4b137'
    history_identity = read(identity_path)
    sources[str(identity_path)] = sha(identity_path)
    specs = [
        ('A_only', ROOT / 'work_dirs/symeood_ablation_v1/100', history_identity['100']),
        ('B_only', ROOT / 'work_dirs/symeood_ablation_v1/010', history_identity['010']),
        ('AB_historical', ROOT / 'work_dirs/symeood_ablation_v1/110', history_identity['110']),
        ('S', ROOT / 'work_dirs/symeood_110_retrain_seed0_20261026_v1/110', None),
        ('T4', ROOT / 'work_dirs/symeood_110_tower_control_20261029_v2/T4', None),
        ('R4', base / 'R4', None),
    ]
    for side, work, historical in specs:
        sweep_path = work / 'ckpt_sweep/sweep_results.json'
        sweep = read(sweep_path)
        ep = sweep['selected_checkpoint']
        cp = sweep['all_checkpoints'][ep]
        if historical:
            assert historical['selected'] == ep and historical['checkpoint_sha256'] == cp['checkpoint_sha256']
        sources[str(sweep_path.relative_to(ROOT))] = sha(sweep_path)
        identities[side] = dict(epoch=ep, checkpoint_sha256=cp['checkpoint_sha256'])
        models[side] = {}
        for split in ('val', 'test'):
            if side == 'T4' and split == 'test':
                continue  # T4 has no authorized/completed TEST; do not generate one.
            if split == 'val':
                meta = cp
            elif side == 'R4':
                meta = dict(results_pkl=str(base / 'fixed_TEST_20261030_v1/preds/results.pkl'),
                    results_pkl_sha256=read(base / 'fixed_TEST_20261030_v1/completion.json')['results_pkl_sha256'])
            else:
                meta = read(work / ('ckpt_sweep/final_test/' + ep + '/final_test_metrics_v2.json'))
                assert meta['checkpoint_sha256'] == cp['checkpoint_sha256']
            path = Path(meta['results_pkl'])
            assert sha(path) == meta['results_pkl_sha256']
            sources[str(path.relative_to(ROOT))] = sha(path)
            with path.open('rb') as stream:
                raw = pickle.load(stream)
            assert len(raw) == len(refs[split]) and all(len(row[0]) <= 1 for row in raw)
            models[side][split] = [row[0][0].tolist() if len(row[0]) else None for row in raw]
            if historical and split == 'test':
                assert sha(path) == historical['pkl_sha256']

    metrics, frames = [], []
    for side, splits in models.items():
        for split, predictions in splits.items():
            rows = []
            for ref, pred in zip(refs[split], predictions):
                row = {k: ref[k] for k in ('image', 'sequence', 'domain', 'frame_id', 'gt')}
                riou = compute_riou(np.asarray(canonical(pred)), np.asarray(canonical(ref['gt']))) if pred else 0.
                row.update(side=side, split=split, prediction=pred, score=pred[5] if pred else 0.)
                row.update(row_metrics(ref['gt'], pred, riou))
                rows.append(row)
            frames.extend(rows)
            for domain in ('all', 'real', 'sim'):
                group = [r for r in rows if domain == 'all' or r['domain'] == domain]
                valid = [r for r in group if r['output']]
                value = dict(side=side, split=split, domain=domain, frames=len(group), outputs=len(valid),
                    TP50=sum(r['riou'] >= .5 for r in group), TP75=sum(r['riou'] >= .75 for r in group),
                    Recall_pct=100 * sum(r['riou'] >= .5 for r in group) / len(group),
                    center_MAE_px=sum(r['center_error_px'] for r in valid) / len(valid),
                    mean_RIoU=sum(r['riou'] for r in group) / len(group),
                    long_MAE_pct=100 * sum(r['long_relative_error'] for r in valid) / len(valid),
                    short_MAE_pct=100 * sum(r['short_relative_error'] for r in valid) / len(valid),
                    Sim_angle_RMSE_deg=(math.sqrt(sum(r['angle_error_deg']**2 for r in valid) / len(valid)) if domain == 'sim' else None))
                for threshold in (.5, .75):
                    # One GT and at most one detection per frame: this is
                    # exactly the existing evaluator's final PR arithmetic.
                    order = np.argsort(-np.asarray([r['score'] for r in valid], dtype=np.float32))
                    tp = np.asarray([r['riou'] >= threshold for r in valid], dtype=np.float32)[order].cumsum()
                    fp = np.asarray([r['riou'] < threshold for r in valid], dtype=np.float32)[order].cumsum()
                    precision = tp / np.maximum(tp + fp, np.finfo(np.float32).eps)
                    value['AP' + str(round(threshold * 100))] = 100 * ap11(tp.astype(np.float64) / len(group), precision)
                metrics.append(value)
            print('AUDITED', side, split, flush=True)

    previous_val = read(base / 'target_VAL/summary.json')
    previous_test = read(base / 'fixed_TEST_20261030_v1/analysis/summary.json')
    checks = []
    for row in metrics:
        if row['side'] not in ('E', 'S', 'R4'):
            continue
        previous = previous_val if row['split'] == 'val' else previous_test
        old = next(r for r in previous['metrics'] if r['side'] == row['side'] and r['domain'] == row['domain']
                   and (row['split'] == 'test' or row['side'] == 'E' or r['selected']))
        for key in ('AP75', 'Recall_pct', 'center_MAE_px', 'mean_RIoU'):
            delta = abs(row[key] - old[key])
            assert delta < 1e-5, (row['side'], row['split'], row['domain'], key, delta)
            checks.append(delta)
    with (out / 'metrics.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    with (out / 'per_frame.jsonl').open('x') as stream:
        for row in frames:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    result = dict(passed=True, model_forwards=0, optimizer_updates=0, new_TEST=False,
        existing_exposed_TEST_readback=True, selection_changed=False,
        sources=sources, identities=identities, replay_checks=len(checks), maximum_replay_delta=max(checks),
        metrics=metrics, source_sha256=sha(__file__))
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('UNIFORM_B2_SOURCE_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--history-identities', type=Path)
    args = parser.parse_args()
    run(args.out.resolve(), args.history_identities)
