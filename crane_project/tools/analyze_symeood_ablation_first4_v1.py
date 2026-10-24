"""CPU audit of frozen first-four results; never loads model weights.

Uses archived PKL/TXT/GT only. Independently aggregates temporal metrics from
exported polygons and supplements exact-output geometry and 11-point AP.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import re

import cv2
import numpy as np

GROUPS = ('000', '100', '010', '001')


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def obb(poly):
    (x, y), (w, h), a = cv2.minAreaRect(np.asarray(poly, np.float32).reshape(4, 2))
    if w < h:
        w, h, a = h, w, a + 90
    return np.array([x, y, w, h, math.radians((a + 90) % 180 - 90)])


def corners(b):
    return cv2.boxPoints(((float(b[0]), float(b[1])),
                         (float(b[2]), float(b[3])), math.degrees(float(b[4]))))


def iou(a, b):
    if a is None:
        return 0.
    area, _ = cv2.intersectConvexConvex(corners(a), corners(b))
    area = min(max(float(area), 0.), float(a[2] * a[3]), float(b[2] * b[3]))
    return area / max(float(a[2] * a[3] + b[2] * b[3]) - area, 1e-12)


def angle(a, b):
    return abs((float(a - b) + math.pi / 2) % math.pi - math.pi / 2)


def txtbox(p):
    if not p.is_file() or not p.read_text().strip():
        return None
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 1, p
    return obb([float(x) for x in lines[0].split()[:8]])


def ap11(rows, thr):
    valid = sorted((r for r in rows if r['output']), key=lambda r: -r['score'])
    tp = np.array([r['riou_native'] >= thr for r in valid], dtype=float)
    if not len(tp):
        return 0.
    recall = np.cumsum(tp) / len(rows)
    precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    return 100 * sum(max(precision[recall >= t], default=0.)
                     for t in np.arange(0., 1.1, .1)) / 11


def aggregate(rows):
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r['domain'], r['seq'])].append(r)
    dfr, aci, tdr, mcml, mrf = [], [], [], [], []
    for seqrows in buckets.values():
        seqrows.sort(key=lambda r: r['fid'])
        segments = []
        for r in seqrows:
            if not segments or r['fid'] != segments[-1][-1]['fid'] + 1:
                segments.append([])
            segments[-1].append(r)
        for segment in segments:
            hits = [r['riou_export'] >= .5 for r in segment]
            run, maximum = 0, 0
            for r, hit in zip(segment, hits):
                if hit:
                    if run:
                        mrf.append(run)
                    run = 0
                else:
                    run += 1
                    maximum = max(maximum, run)
            mcml.append(maximum)
            tdr.extend(int(any(hits[i:i+10])) for i in range(max(0, len(hits)-9)))
            for p, q in zip(segment, segment[1:]):
                if p['output'] and q['output']:
                    dfr.append(abs(q['diag_export'] - p['diag_export']) / p['diag_export'])
                    aci.append(max(0., 1 - angle(q['angle_export'], p['angle_export']) /
                                   (math.radians(35) + 1e-9)))
    valid = [r for r in rows if r['output']]
    eligible = [r for r in valid if r['angle_eligible']]
    out = dict(frames=len(rows), outputs=len(valid), missing=len(rows)-len(valid),
               coverage_pct=100*len(valid)/len(rows),
               mean_RIoU=float(np.mean([r['riou_export'] for r in rows])),
               mean_RIoU_native=float(np.mean([r['riou_native'] for r in rows])),
               center_correct=sum(r['center_error'] < 15 for r in valid),
               size_correct=sum(max(r['long_error'], r['short_error']) <= .1 for r in valid),
               angle_eligible=len(eligible),
               angle_correct=sum(r['angle_error_deg'] <= 3 for r in eligible),
               AP50_independent=ap11(rows, .5), AP75_independent=ap11(rows, .75),
               DFR=100*float(np.mean(dfr)) if dfr else None, DFR_pairs=len(dfr),
               ACI=float(np.mean(aci)) if aci else None,
               TDR_hits=sum(tdr), TDR_windows=len(tdr),
               TDR_pct=100*float(np.mean(tdr)) if tdr else None,
               MCML_max=max(mcml), MCML_mean=float(np.mean(mcml)), MCML_segments=len(mcml),
               MRF=float(np.mean(mrf)) if mrf else None,
               gated_angle_RMSE_deg=math.sqrt(float(np.mean([r['gated_angle_sq'] for r in rows]))))
    for key in ('long', 'short'):
        errors = [r[key+'_error'] for r in valid]
        bias = [r[key+'_bias'] for r in valid]
        out[key+'_MAE_pct'] = 100*float(np.mean(errors)) if errors else None
        out[key+'_P95_pct'] = 100*float(np.percentile(errors, 95)) if errors else None
        out[key+'_bias_pct'] = 100*float(np.mean(bias)) if bias else None
    out['meanabs_a'] = float(np.mean([abs(r['a']) for r in valid])) if valid else None
    out['meanabs_r'] = float(np.mean([abs(r['r']) for r in valid])) if valid else None
    return out


def run(args):
    root = args.snapshot.resolve()
    original_root = '/media/omnisky/personal_files/ljj/symEOOD/'
    base = root/'work_dirs/symeood_ablation_v1'
    out = args.out.resolve()
    out.mkdir(exist_ok=False, parents=True)
    summary, frames, comparisons = {}, [], []
    for group in args.groups:
        sweep = json.loads((base/group/'ckpt_sweep/sweep_results.json').read_text())
        logrows = [json.loads(l) for p in (base/group).glob('*.log.json')
                   for l in p.read_text().splitlines()]
        assert {r['epoch'] for r in logrows if r.get('mode') == 'train'} == set(range(1, 25))
        assert {r['epoch'] for r in logrows if r.get('mode') == 'val'} == set(range(2, 25, 2))
        summary[group] = dict(selected=sweep['selected_checkpoint'],
                              selection=sweep['selection_info'], online_VAL_count=12,
                              all_online_VAL=[r for r in logrows if r.get('mode') == 'val'],
                              candidates={})
        for name, checkpoint in sweep['all_checkpoints'].items():
            summary[group]['candidates'][name] = checkpoint['derived_selection_metrics']
        # Independently apply the frozen selector to saved full-VAL metrics.
        c = sweep['selection_config']
        enriched = []
        for name, cp in sweep['all_checkpoints'].items():
            m = cp['metrics']
            rc = (c['weight_real']*m['real/R_center(%)'] + c['weight_sim']*m['sim/R_center(%)'])/100
            td = (c['weight_real']*m['real/TDR_w10(%)'] + c['weight_sim']*m['sim/TDR_w10(%)'])/100
            mc = max(m['real/MCML_max(frames)'], m['sim/MCML_max(frames)'])
            ar = m['sim/A-RMSE(deg)']
            score = c['soft_w_tdr']*td + c['soft_w_r_center']*rc + c['soft_w_aci']*m['sim/ACI'] + c['soft_w_angle']*(1-min(ar, 90)/90)
            enriched.append(dict(name=name, rc=rc, mc=mc, ar=ar, score=score))
        best_rc = max(v['rc'] for v in enriched)
        feasible = [v for v in enriched if v['mc'] <= c['mcml_limit'] and v['rc'] >= best_rc-c['r_center_delta']]
        if not feasible:
            feasible = [v for v in enriched if v['mc'] <= c['mcml_limit']]
        chosen = (max(feasible, key=lambda v:v['score']) if feasible else
                  min(enriched, key=lambda v:(v['mc'], -v['rc'], v['ar'])))['name']
        assert chosen == sweep['selected_checkpoint'], (group, chosen)
        summary[group]['selection_reproduced'] = True
        cases = [(n, 'val', cp['metrics'], cp['results_pkl']) for n, cp in sweep['all_checkpoints'].items()]
        report = list((base/group/'ckpt_sweep/final_test').glob('*/final_test_metrics_v2.json'))
        assert len(report) == 1
        report = json.loads(report[0].read_text())
        cases.append((sweep['selected_checkpoint'], 'test', report['metrics'], report['results_pkl']))
        for epoch, split, official, pklpath in cases:
            assert pklpath.startswith(original_root)
            p = root/pklpath[len(original_root):]
            with p.open('rb') as f:
                predictions = pickle.load(f)
            gtroot = root/'crane_project/data/crane_grab_port_day2night_v1'/split/'annfiles'
            gtfiles = sorted(gtroot.glob('*.txt'))
            assert len(predictions) == len(gtfiles) == (887 if split == 'val' else 1440)
            rows = []
            for gtfile, prediction in zip(gtfiles, predictions):
                boxes = prediction[0]
                assert boxes.shape[0] <= 1
                gt = txtbox(gtfile)
                assert gt is not None
                match = re.match(r'(real|sim)_(.+)_(\d+)', gtfile.stem)
                domain, seq, fid = match.groups()
                pred = None if len(boxes) == 0 else np.array(boxes[0][:5], float)
                export = txtbox(p.parent/'Task1_grab'/gtfile.name)
                assert (pred is None) == (export is None)
                center = float(np.linalg.norm(export[:2]-gt[:2])) if export is not None else float('inf')
                row = dict(group=group, epoch=epoch, split=split, frame=gtfile.stem,
                           domain=domain, seq=seq, fid=int(fid), output=pred is not None,
                           score=float(boxes[0][-1]) if len(boxes) else 0.,
                           riou_export=iou(export, gt), riou_native=iou(pred, gt),
                           center_export=center,
                           gated_angle_sq=(math.degrees(angle(export[4], gt[4])) if center < 10 else 90.)**2)
                if export is not None:
                    row.update(diag_export=float(np.hypot(export[2], export[3])), angle_export=float(export[4]))
                if pred is not None:
                    assert np.isfinite(pred).all() and (pred[2:4] > 0).all()
                    sizes = sorted(pred[2:4], reverse=True)
                    lp, sp = sizes
                    lg, sg = gt[2:4]
                    theta = pred[4] + (math.pi/2 if pred[2] < pred[3] else 0)
                    el, es = math.log(lp/lg), math.log(sp/sg)
                    row.update(long_error=abs(lp-lg)/lg, short_error=abs(sp-sg)/sg,
                               long_bias=(lp-lg)/lg, short_bias=(sp-sg)/sg,
                               center_error=float(np.linalg.norm(pred[:2]-gt[:2])),
                               angle_error_deg=math.degrees(angle(theta, gt[4])),
                               angle_eligible=lg/sg >= 1.2, a=(el+es)/2, r=el-es)
                rows.append(row)
            aggregates = {domain:aggregate([r for r in rows if r['domain'] == domain]) for domain in ('real','sim')}
            aggregates['all'] = aggregate(rows)
            for domain in ('real','sim'):
                m = aggregates[domain]
                compare = {'R_center(%)':100*sum(r['center_export'] < 15 for r in rows if r['domain'] == domain)/m['frames'],
                           'mean_RIoU':m['mean_RIoU'], 'DFR(%/frame)':m['DFR'], 'ACI':m['ACI'],
                           'TDR_w10(%)':m['TDR_pct'], 'MCML_max(frames)':m['MCML_max'],
                           'MCML_mean(frames)':m['MCML_mean'], 'MRF(frames)':m['MRF']}
                if domain == 'sim':
                    compare['A-RMSE(deg)'] = m['gated_angle_RMSE_deg']
                for key, value in compare.items():
                    k = domain+'/'+key
                    if k not in official:
                        continue
                    tolerance = .0051 if key in ('R_center(%)','TDR_w10(%)','MCML_mean(frames)','MRF(frames)') else .0002
                    delta = abs(value-official[k])
                    comparisons.append(dict(group=group,epoch=epoch,split=split,metric=k,
                                            independent=value,reported=official[k],absdiff=delta,passed=delta <= tolerance))
            summary[group][split if split == 'test' else epoch] = aggregates
            frames.extend(rows)
    (out/'analysis.json').write_text(json.dumps(summary, indent=2, default=json_default))
    (out/'verification.json').write_text(json.dumps(dict(comparisons=comparisons,
            passed=all(v['passed'] for v in comparisons), comparisons_count=len(comparisons),
            note='Independent convex-polygon intersection; tolerances respect report rounding. AP is independent CPU 11-point supplemental, not native CUDA AP.'), indent=2, default=json_default))
    fields = sorted(set(k for r in frames for k in r))
    with (out/'per_frame.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields);w.writeheader();w.writerows(frames)
    print(json.dumps(dict(groups=len(summary), frames=len(frames),comparisons=len(comparisons),
                          failed=[v for v in comparisons if not v['passed']]), indent=2, default=json_default))
    if not all(v['passed'] for v in comparisons):
        raise RuntimeError('Independent metric verification failed; see verification.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--groups', nargs='+', default=list(GROUPS),
                   choices=('000', '100', '010', '001', '110', '101', '011', '111'))
    run(p.parse_args())
