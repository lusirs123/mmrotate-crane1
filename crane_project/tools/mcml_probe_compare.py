#!/usr/bin/env python3
"""
Compare mcml_diag logs across multiple inference-time probe variants.

Example:
  PYTHONPATH=. python3 crane_project/tools/mcml_probe_compare.py \
      --variant raw=work_dirs/probe/raw \
      --variant norm=work_dirs/probe/linear-clahe-gray \
      --variant adabn=work_dirs/probe/linear-clahe-gray_adabn
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

STATUS_RANK = {'DEAD-global': 0, 'DEAD-local': 1, 'EDGE': 2, 'OK': 3}

def parse_args():
    parser = argparse.ArgumentParser(description='Compare mcml_diag probe logs')
    parser.add_argument(
        '--variant', action='append', required=True,
        help='Variant spec in the form name=log_dir; may be repeated')
    parser.add_argument('--exclude-seq', default='seq02')
    parser.add_argument('--exclude-start', type=int, default=133)
    parser.add_argument('--exclude-end', type=int, default=171)
    parser.add_argument('--save-dir', default=None)
    return parser.parse_args()

def longest_run(flags):
    best = cur = 0
    for x in flags:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best

def parse_variant_spec(spec):
    if '=' not in spec:
        raise SystemExit(f'invalid --variant spec: {spec}')
    name, log_dir = spec.split('=', 1)
    return name.strip(), log_dir.strip()

def log_paths(log_dir):
    paths = []
    for ext in ('*.log', '*.txt'):
        paths.extend(glob.glob(os.path.join(log_dir, '**', ext), recursive=True))
    return sorted(paths)


def parse_logs(log_dir):
    frame_pat = re.compile(
        r'\[(?P<fname>[^\]]+)\]\s+'
        r'(?P<mark>✗✗|✗|△|✓)\s+'
        r'.*?brightness=\s*(?P<bright>[0-9.]+)\s+'
        r'gt=\s*(?P<gt>[0-9.]+)\s+'
        r'roi=\s*(?P<roi>[0-9.]+)\s+'
        r'global=\s*(?P<global>[0-9.]+)@P(?P<level>\d+)'
    )
    top1_pat = re.compile(
        r'#1\s+score=(?P<score>[0-9.]+).*?'
        r'c_dist=(?P<dist>[0-9.]+)px\s+'
        r'RIoU=(?P<riou>[0-9.]+)'
    )
    mark2status = {'✗✗': 'DEAD-global', '✗': 'DEAD-local', '△': 'EDGE', '✓': 'OK'}
    fname_pat = re.compile(r'(?P<seq>.+_seq\d+)_(?P<fid>\d{5})$')

    paths = log_paths(log_dir)
    if not paths:
        raise SystemExit(f'[ERROR] no .log/.txt files found under {log_dir}')

    rows = {}
    cur = None
    for path in paths:
        with open(path, errors='ignore') as f:
            for line in f:
                m = frame_pat.search(line)
                if m:
                    fname = m.group('fname')
                    cur = fname
                    sm = fname_pat.match(fname)
                    rows[fname] = {
                        'fname': fname,
                        'seq': sm.group('seq') if sm else 'unknown',
                        'fid': int(sm.group('fid')) if sm else -1,
                        'status': mark2status[m.group('mark')],
                        'brightness': float(m.group('bright')),
                        'gt': float(m.group('gt')),
                        'roi': float(m.group('roi')),
                        'global': float(m.group('global')),
                        'top1_score': None,
                        'top1_riou': None,
                    }
                    continue
                if cur is not None and cur in rows:
                    t = top1_pat.search(line)
                    if t and rows[cur]['top1_score'] is None:
                        rows[cur]['top1_score'] = float(t.group('score'))
                        rows[cur]['top1_riou'] = float(t.group('riou'))

    if not rows:
        raise SystemExit(f'[ERROR] no frame lines matched in {log_dir}')
    return rows


def is_excluded(row, seq_name, start, end):
    return row['seq'].endswith(seq_name) and start <= row['fid'] <= end


def valid_det(row, score_thr=0.05, riou_thr=0.5):
    return (
        row['top1_score'] is not None
        and row['top1_riou'] is not None
        and row['top1_score'] >= score_thr
        and row['top1_riou'] >= riou_thr
    )


def compare(base_rows, other_rows, exclude_seq, exclude_start, exclude_end):
    common = sorted(set(base_rows) & set(other_rows),
                    key=lambda x: (base_rows[x]['seq'], base_rows[x]['fid']))
    common = [f for f in common if not is_excluded(base_rows[f], exclude_seq, exclude_start, exclude_end)]

    rows = []
    for fname in common:
        b = base_rows[fname]
        d = other_rows[fname]
        b_valid = valid_det(b)
        d_valid = valid_det(d)
        rows.append({
            'fname': fname,
            'seq': b['seq'],
            'fid': b['fid'],
            'base_status': b['status'],
            'other_status': d['status'],
            'status_delta': STATUS_RANK[d['status']] - STATUS_RANK[b['status']],
            'base_global': b['global'],
            'other_global': d['global'],
            'd_global': d['global'] - b['global'],
            'base_gt': b['gt'],
            'other_gt': d['gt'],
            'd_gt': d['gt'] - b['gt'],
            'base_top1_score': b['top1_score'],
            'other_top1_score': d['top1_score'],
            'base_top1_riou': b['top1_riou'],
            'other_top1_riou': d['top1_riou'],
            'base_valid_det': b_valid,
            'other_valid_det': d_valid,
            'effective_revive': b['global'] < 0.02 and d_valid,
            'harmful_drop': b['status'] == 'OK' and d['status'] != 'OK',
            'valid_revive': (not b_valid) and d_valid,
            'valid_drop': b_valid and (not d_valid),
        })
    return rows


def summarize(rows):
    effective = [r for r in rows if r['effective_revive']]
    harmful = [r for r in rows if r['harmful_drop']]
    valid_revive = [r for r in rows if r['valid_revive']]
    valid_drop = [r for r in rows if r['valid_drop']]
    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r['seq']].append(r)

    seq_summary = []
    for seq, items in sorted(by_seq.items()):
        items = sorted(items, key=lambda x: x['fid'])
        seq_summary.append({
            'seq': seq,
            'n': len(items),
            'effective_revive': sum(r['effective_revive'] for r in items),
            'harmful_drop': sum(r['harmful_drop'] for r in items),
            'net_effective': sum(r['effective_revive'] for r in items) - sum(r['harmful_drop'] for r in items),
            'valid_revive': sum(r['valid_revive'] for r in items),
            'valid_drop': sum(r['valid_drop'] for r in items),
            'net_valid': sum(r['valid_revive'] for r in items) - sum(r['valid_drop'] for r in items),
            'base_miss_run': longest_run([not r['base_valid_det'] for r in items]),
            'other_miss_run': longest_run([not r['other_valid_det'] for r in items]),
        })
    return effective, harmful, valid_revive, valid_drop, seq_summary


def show(title, items, base_label, other_label, limit=40):
    print('\n' + '=' * 100)
    print(f'{title} count={len(items)}')
    print('=' * 100)
    for r in items[:limit]:
        print(
            f'{r["fname"]:24s} {r["base_status"]:11s}->{r["other_status"]:11s} '
            f'g {r["base_global"]:.4f}->{r["other_global"]:.4f} '
            f'gt {r["base_gt"]:.4f}->{r["other_gt"]:.4f} '
            f'score {r["base_top1_score"]}->{r["other_top1_score"]} '
            f'RIoU {r["base_top1_riou"]}->{r["other_top1_riou"]}'
        )


def main():
    args = parse_args()
    variants = [parse_variant_spec(s) for s in args.variant]
    if len(variants) < 2:
        raise SystemExit('need at least two variants')

    save_dir = args.save_dir or os.path.join(os.path.dirname(variants[0][1]), 'mcml_probe_compare')
    os.makedirs(save_dir, exist_ok=True)

    parsed = {name: parse_logs(log_dir) for name, log_dir in variants}
    base_name, _ = variants[0]
    base_rows = parsed[base_name]

    for name, _ in variants[1:]:
        rows = compare(base_rows, parsed[name], args.exclude_seq, args.exclude_start, args.exclude_end)
        effective, harmful, valid_revive, valid_drop, seq_summary = summarize(rows)

        print('\n' + '#' * 100)
        print(f'BASE={base_name}  OTHER={name}')
        print('#' * 100)
        print(f'total_frames     = {len(rows)}')
        print(f'effective_revive = {len(effective)}')
        print(f'harmful_drop     = {len(harmful)}')
        print(f'net_effective    = {len(effective) - len(harmful)}')
        print(f'valid_revive     = {len(valid_revive)}')
        print(f'valid_drop       = {len(valid_drop)}')
        print(f'net_valid        = {len(valid_revive) - len(valid_drop)}')

        print('\nBY SEQUENCE')
        for s in seq_summary:
            print(
                f'{s["seq"]:16s} n={s["n"]:3d}  '
                f'effective={s["effective_revive"]:3d} harmful={s["harmful_drop"]:3d} '
                f'net={s["net_effective"]:+3d}  '
                f'valid_revive={s["valid_revive"]:3d} valid_drop={s["valid_drop"]:3d} '
                f'net_valid={s["net_valid"]:+3d}  '
                f'miss_run {s["base_miss_run"]:3d}->{s["other_miss_run"]:3d}'
            )

        eff_sorted = sorted(effective, key=lambda r: (r['other_top1_riou'] or -1, r['other_top1_score'] or -1), reverse=True)
        harm_sorted = sorted(harmful, key=lambda r: (r['base_global'] - r['other_global']), reverse=True)
        valrev_sorted = sorted(valid_revive, key=lambda r: (r['other_top1_riou'] or -1, r['other_top1_score'] or -1), reverse=True)
        valdrop_sorted = sorted(valid_drop, key=lambda r: ((r['base_top1_riou'] or 0) - (r['other_top1_riou'] or 0)), reverse=True)

        show('EFFECTIVE_REVIVE', eff_sorted, base_name, name)
        show('HARMFUL_DROP', harm_sorted, base_name, name)
        show('VALID_REVIVE', valrev_sorted, base_name, name)
        show('VALID_DROP', valdrop_sorted, base_name, name)

        csv_path = os.path.join(save_dir, f'{base_name}_vs_{name}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nSaved: {csv_path}')


if __name__ == '__main__':
    main()
