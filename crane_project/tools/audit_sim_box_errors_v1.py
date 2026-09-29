"""Read-only paired sim TEST geometry diagnosis; no inference or parameter search."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from crane_project.tools.eval_crane_offline import (
    angle_diff, compute_riou, parse_dota_txt)


def digest(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(b'\0')
        h.update(p.read_bytes())
        h.update(b'\0')
    return h.hexdigest()


def errors(pred, gt):
    center = float(np.linalg.norm(pred[:2] - gt[:2]))
    angle = float(abs(np.degrees(angle_diff(pred[4], gt[4]))))
    result = dict(center_px=center, center_over_gt_short=center / float(gt[3]),
                  long_signed_pct=float(100 * (pred[2] / gt[2] - 1)),
                  short_signed_pct=float(100 * (pred[3] / gt[3] - 1)),
                  angle_abs_deg=angle, riou=compute_riou(pred, gt),
                  gt_aspect_ratio=float(gt[2] / gt[3]))
    # Sensitivity only: each replacement is independent; gains are not additive.
    for key, dims in [('center', [0, 1]), ('size', [2, 3]), ('angle', [4])]:
        box = pred.copy()
        box[dims] = gt[dims]
        result['riou_replace_' + key] = compute_riou(box, gt)
    return result


def summarize(rows):
    if not rows:
        return dict(n=0)
    result = dict(n=len(rows))
    for key in rows[0]:
        a = np.array([r[key] for r in rows])
        result[key] = dict(mean=float(a.mean()), median=float(np.median(a)),
                           p90=float(np.percentile(a, 90)))
    for key in ['long_signed_pct', 'short_signed_pct']:
        result[key]['mean_absolute'] = float(np.mean([abs(r[key]) for r in rows]))
    result['angle_rmse_deg_all_outputs'] = float(np.sqrt(np.mean([
        r['angle_abs_deg'] ** 2 for r in rows])))
    for key in ['center', 'size', 'angle']:
        result['mean_riou_gain_replace_' + key] = float(np.mean([
            r['riou_replace_' + key] - r['riou'] for r in rows]))
    return result


def audit(gt_dir, eood_dir, sym_dir):
    files = sorted(gt_dir.glob('sim_seq09_*.txt'))
    if len(files) != 572:
        raise ValueError('Expected 572 sim_seq09 GT frames, found ' + str(len(files)))
    dirs = dict(eood=eood_dir, symeood=sym_dir)
    rows = []
    for path in files:
        gt = parse_dota_txt(str(path))
        if len(gt) != 1:
            raise ValueError('Expected one GT: ' + str(path))
        row = dict(frame=path.stem, gt=gt[0].tolist())
        for arm, directory in dirs.items():
            target = directory / path.name
            # Empty file is an explicit no-output frame; absent file is not.
            if not target.is_file():
                raise FileNotFoundError(target)
            boxes = parse_dota_txt(str(target))
            if len(boxes) > 1:
                raise ValueError('Expected top-1: ' + str(target))
            for box in [gt[0]] + boxes:
                if not np.isfinite(box).all() or np.any(box[2:4] <= 0):
                    raise ValueError('Invalid box: ' + str(target))
            row[arm] = errors(boxes[0], gt[0]) if boxes else None
        rows.append(row)
    paired = [r for r in rows if all(r[a] is not None for a in dirs)]
    ranked = sorted(paired, key=lambda r: r['eood']['riou'] - r['symeood']['riou'], reverse=True)
    return dict(protocol='sim_box_error_decomposition_v1',
        evidence_role='exposed_test_read_only_diagnosis', frames=len(rows),
        paths=dict(gt=str(gt_dir), **{a: str(p) for a, p in dirs.items()}),
        sha256=dict(gt=digest(files), **{a: digest([p/f.name for f in files]) for a,p in dirs.items()}),
        output_counts={a: sum(r[a] is not None for r in rows) for a in dirs},
        common_output_frames=len(paired),
        all_output_metrics={a: summarize([r[a] for r in rows if r[a] is not None]) for a in dirs},
        paired_metrics={a: summarize([r[a] for r in paired]) for a in dirs},
        eood_better_riou_frames=sum(r['eood']['riou'] > r['symeood']['riou'] for r in paired),
        symeood_better_riou_frames=sum(r['eood']['riou'] < r['symeood']['riou'] for r in paired),
        largest_eood_advantage=[dict(frame=r['frame'], riou_gap=r['eood']['riou']-r['symeood']['riou']) for r in ranked[:20]],
        rows=rows, limitations=[
            'Angle is the long-axis orientation modulo 180 degrees; near-square boxes have ambiguous axes.',
            'Angle RMSE uses all output frames, without the evaluator center-validity gate.',
            'GT component replacements measure geometry sensitivity, not training causality; gains are not additive.',
            'Input TXT hashes identify supplied exports, not checkpoint provenance. Use fixed VAL-selected final_test paths.',
            'No threshold selection or training changes are justified solely by this exposed TEST diagnosis.'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gt-dir', required=True)
    ap.add_argument('--eood-pred-dir', required=True)
    ap.add_argument('--symeood-pred-dir', required=True)
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()
    out = Path(args.out_json).resolve()
    if out.exists():
        raise FileExistsError(out)
    result = audit(Path(args.gt_dir).resolve(), Path(args.eood_pred_dir).resolve(),
                   Path(args.symeood_pred_dir).resolve())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ['protocol', 'output_counts',
        'common_output_frames', 'paired_metrics']}, ensure_ascii=False, indent=2))
    print('Detailed report:', out)


if __name__ == '__main__':
    main()
