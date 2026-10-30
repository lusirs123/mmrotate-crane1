"""CPU readback of the completed fixed T4 VAL control; never infers or selects anew."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def run(base, out):
    import numpy as np
    from crane_project.tools.eval_crane_offline import compute_riou
    from crane_project.tools.eval_port_geometry_ap_v1 import evaluate_images, validate_saved_gt
    from crane_project.tools.analyze_symeood_ablation_first4_v1 import txtbox
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import canonical, row_metrics

    out.mkdir(exist_ok=False)
    reference = ROOT / 'work_dirs/symeood_110_retrain_seed0_20261026_v1'
    names = {
        'S': reference / 'downstream/lvsr/fixed_b/epoch_01_val.rows.jsonl.gz',
        'E': ROOT / 'work_dirs/detector_module_ablation_v1_20261028/eood_l/epoch_17_val.rows.jsonl.gz'}
    refs = {}
    for side, path in names.items():
        with gzip.open(path, 'rt') as stream:
            refs[side] = [json.loads(line) for line in stream]
        assert len(refs[side]) == 887 and all(r['split'] == 'val' for r in refs[side])
    assert [r['image'] for r in refs['S']] == sorted(r['image'] for r in refs['S'])
    assert [(r['image'], r['gt']) for r in refs['S']] == [(r['image'], r['gt']) for r in refs['E']]
    anns = ROOT / 'crane_project/data/crane_grab_port_day2night_v1/val/annfiles'
    annotation_digest = hashlib.sha256()
    for row in refs['S']:
        path = anns / (row['image'] + '.txt')
        poly = np.asarray([float(x) for x in path.read_text().split()[:8]]).reshape(4, 2)
        validate_saved_gt(row['gt'], poly, txtbox(path))
        annotation_digest.update(path.name.encode())
        annotation_digest.update(b'\0')
        annotation_digest.update(path.read_bytes())
    sweeps = {'S': load(reference / '110/ckpt_sweep/sweep_results.json'),
              'T4': load(base / 'T4/ckpt_sweep/sweep_results.json')}
    assert annotation_digest.hexdigest() == sweeps['T4']['source_val_annotations_sha256']
    selections = {k: v['selected_checkpoint'] for k, v in sweeps.items()}
    assert selections == {'S': 'epoch_20', 'T4': 'epoch_18'}
    selector_checks = {}
    for side, sweep in sweeps.items():
        c = sweep['selection_config']
        scored = []
        for ep, cp in sweep['all_checkpoints'].items():
            m = cp['metrics']
            rc = sum(c['weight_' + d] * m[d + '/R_center(%)'] / 100 for d in ('sim', 'real'))
            tdr = sum(c['weight_' + d] * m[d + '/TDR_w10(%)'] / 100 for d in ('sim', 'real'))
            mc = max(m[d + '/MCML_max(frames)'] for d in ('sim', 'real'))
            armse = m['sim/A-RMSE(deg)']
            score = c['soft_w_tdr'] * tdr + c['soft_w_r_center'] * rc + c['soft_w_aci'] * m['sim/ACI'] + c['soft_w_angle'] * (1 - min(armse, 90) / 90)
            scored.append(dict(epoch=ep, rc=rc, mc=mc, armse=armse, score=score))
        maximum = max(r['rc'] for r in scored)
        feasible = [r for r in scored if r['mc'] <= c['mcml_limit'] and r['rc'] >= maximum - c['r_center_delta']]
        if not feasible:
            feasible = [r for r in scored if r['mc'] <= c['mcml_limit']]
        chosen = (max(feasible, key=lambda r: r['score']) if feasible else min(scored, key=lambda r: (r['mc'], -r['rc'], r['armse'])))['epoch']
        assert chosen == selections[side]
        selector_checks[side] = dict(reproduced=chosen, candidates=scored)

    metrics, frame_rows, cache, prediction_shas = [], [], {}, {}
    reference_prediction_comparison = None

    def summarize(rows):
        valid = [r for r in rows if r['output']]
        ordered = sorted(valid, key=lambda r: -r['score'])
        value = dict(frames=len(rows), outputs=len(valid),
            center_MAE_px=sum(r['center_error_px'] for r in valid) / len(valid),
            mean_RIoU=sum(r['riou'] for r in rows) / len(rows),
            Recall_pct=100 * sum(r['riou'] >= .5 for r in rows) / len(rows),
            TP75=sum(r['riou'] >= .75 for r in rows),
            center_correct=sum(r['center_hit'] for r in rows),
            joint10=sum(r['joint_size10'] for r in rows),
            joint10_pct=100 * sum(r['joint_size10'] for r in rows) / len(rows),
            FP75_top100=sum(r['riou'] < .75 for r in ordered[:100]),
            score_mean=sum(r['score'] for r in valid) / len(valid),
            long_MAE_pct=100 * sum(r['long_relative_error'] for r in valid) / len(valid),
            short_MAE_pct=100 * sum(r['short_relative_error'] for r in valid) / len(valid),
            pure_angle_RMSE_deg=math.sqrt(sum(r['angle_error_deg']**2 for r in valid) / len(valid)),
            center_P95_px=float(np.percentile([r['center_error_px'] for r in valid], 95)),
            center_max_px=max(r['center_error_px'] for r in valid))
        dets = [[canonical(r['prediction']) + [r['score']]] if r['output'] else [] for r in rows]
        gt = [[canonical(r['gt'])] for r in rows]
        for threshold in (.5, .75):
            ap = evaluate_images(dets, gt, threshold)
            value['AP' + str(round(threshold * 100))] = ap['ap_percent']
            assert ap['tp'] == sum(r['riou'] >= threshold for r in rows)
        # Native-box failure runs respect sequence boundaries and frame gaps.
        run = maximum = 0
        previous = None
        for r in sorted(rows, key=lambda r: (r['sequence'], r['frame_id'])):
            if previous is None or r['sequence'] != previous['sequence'] or r['frame_id'] != previous['frame_id'] + 1:
                run = 0
            run = run + 1 if r['riou'] < .5 else 0
            maximum = max(maximum, run)
            previous = r
        value['MCML_max_native'] = maximum
        return value

    def record(side, ep, predictions):
        assert len(predictions) == 887
        rows = []
        for ref, p in zip(refs['S'], predictions):
            value = dict(image=ref['image'], sequence=ref['sequence'], domain=ref['domain'],
                frame_id=ref['frame_id'], gt=ref['gt'], prediction=p,
                score=p[5] if p is not None else 0., side=side, epoch=ep)
            riou = compute_riou(np.asarray(canonical(p)), np.asarray(canonical(ref['gt']))) if p is not None else 0.
            value.update(row_metrics(ref['gt'], p, riou))
            rows.append(value)
        cache[side, ep] = rows
        frame_rows.extend(rows)
        for domain in ('all', 'real', 'sim', 'real_seq07', 'real_seq14', 'sim_seq10'):
            subset = [r for r in rows if domain == 'all' or domain in (r['domain'], r['sequence'])]
            metrics.append(dict(side=side, epoch=ep, domain=domain,
                selected=ep == selections.get(side), **summarize(subset)))

    for side, sweep in sweeps.items():
        for ep, cp in sweep['all_checkpoints'].items():
            path = Path(cp['results_pkl'])
            assert sha(path) == cp['results_pkl_sha256']
            prediction_shas[side + '/' + ep] = sha(path)
            with path.open('rb') as stream:
                raw = pickle.load(stream)
            assert len(raw) == 887 and all(len(r[0]) <= 1 for r in raw)
            predictions = [r[0][0].tolist() if len(r[0]) else None for r in raw]
            if side == 'S' and ep == selections['S']:
                assert cp['checkpoint_sha256'] == '7f35dacc308943949a8fd2e6aaad312d14651564903ec87c8c8dcec0552ec054'
                assert all((p is None) == (r['b'] is None) for p, r in zip(predictions, refs['S']))
                errors = [np.abs(np.asarray(p) - r['b']) for p, r in zip(predictions, refs['S']) if p is not None]
                maximum = np.max(errors, axis=0)
                assert np.all(maximum < [.001, .001, .001, .001, 2e-6, 2e-6])
                reference_prediction_comparison = dict(
                    checkpoint_identical=True, output_presence_exact_all887=True,
                    boxes_byte_exact=predictions == [r['b'] for r in refs['S']],
                    maximum_absolute_xywha_score_difference=maximum.tolist(),
                    note='Separate native sweep and downstream capture forwards have finite precision differences; source sweep boxes remain the S comparison inputs.')
            record(side, ep, predictions)
    record('E', 'epoch_24', [r['b'] for r in refs['E']])

    paired = []
    contexts = [('selected', ('S', selections['S']), ('T4', selections['T4'])),
        ('fixed_ep24', ('S', 'epoch_24'), ('T4', 'epoch_24')),
        ('E_vs_T4_selected', ('E', 'epoch_24'), ('T4', selections['T4']))]
    for context, first, second in contexts:
        aa, bb = cache[first], cache[second]
        assert [r['image'] for r in aa] == [r['image'] for r in bb]
        for domain in ('all', 'real', 'sim', 'real_seq07', 'real_seq14'):
            ab = [(a, b) for a, b in zip(aa, bb) if domain == 'all' or domain in (a['domain'], a['sequence'])]
            common = [(a, b) for a, b in ab if a['output'] and b['output']]
            value = dict(context=context, domain=domain, frames=len(ab), common_outputs=len(common),
                new_output=sum(not a['output'] and b['output'] for a, b in ab),
                lost_output=sum(a['output'] and not b['output'] for a, b in ab),
                recovered75=sum(a['riou'] < .75 <= b['riou'] for a, b in ab),
                lost75=sum(b['riou'] < .75 <= a['riou'] for a, b in ab),
                recovered50=sum(a['riou'] < .5 <= b['riou'] for a, b in ab),
                lost50=sum(b['riou'] < .5 <= a['riou'] for a, b in ab),
                recovered_size10=sum(not a['joint_size10'] and b['joint_size10'] for a, b in ab),
                lost_size10=sum(a['joint_size10'] and not b['joint_size10'] for a, b in ab))
            for key in ('center_error_px', 'riou', 'long_relative_error', 'short_relative_error'):
                value[key] = {label: sum(pair[i][key] for pair in common) / len(common) for i, label in enumerate(('before', 'after'))}
            paired.append(value)

    # Cross-check the pre-existing report without rewriting it. Its GT uses
    # OpenCV reparse and its AP sums in float64, so retain documented tolerance.
    original = list(csv.DictReader((base / 'analysis_VAL/all_five_VAL_metrics.csv').open()))
    differences = []
    for r in original:
        m = next(x for x in metrics if (x['side'], x['epoch'], x['domain']) == (r['side'], r['epoch'], r['domain']))
        for old, new in [('AP50_native', 'AP50'), ('AP75_native', 'AP75'), ('mean_RIoU_native', 'mean_RIoU'), ('center_MAE_px', 'center_MAE_px'), ('long_MAE_pct', 'long_MAE_pct'), ('short_MAE_pct', 'short_MAE_pct')]:
            delta = abs(float(r[old]) - m[new])
            assert delta < .0002, (r['side'], r['epoch'], r['domain'], old, delta)
            differences.append(dict(side=r['side'], epoch=r['epoch'], domain=r['domain'], metric=new, delta=delta))
    result = dict(passed=True, TEST=False, inference=False, updates=0,
        evaluation='Native saved Top1 predictions, sealed original GT, existing VOC07 float32 AP reference; no new selection.',
        selections=selections, selector_checks=selector_checks,
        old_S_reference_comparison=reference_prediction_comparison, annotation_sha256=annotation_digest.hexdigest(),
        inputs={str(p.relative_to(ROOT)): sha(p) for p in names.values()},
        prediction_shas=prediction_shas, original_report_checks=len(differences),
        maximum_report_difference=max(r['delta'] for r in differences), metrics=metrics, paired=paired,
        source_sha256=sha(__file__))
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    for name, rows in [('metrics.csv', metrics), ('per_frame.jsonl', frame_rows)]:
        with (out / name).open('x') as stream:
            if name.endswith('.csv'):
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            else:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(dict(passed=True, selected=selections, metric_rows=len(metrics), paired_rows=len(paired))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.base.resolve(), args.out.resolve())
