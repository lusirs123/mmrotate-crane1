#!/usr/bin/env python3
"""S032: bounded direction/magnitude decomposition of saved S031 A21/N18.

CPU, standard library only. No model, proposal generation, coefficient sweep,
TEST/depth reads or inference-time oracle. Classifications are offline GT-based.
"""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.analyze_port_size_joint_v1 import (
    canonical, csv_write, mean, quantile, read, sha, write)

VERSION = 'port_size_correction_behavior_v1'
EPS = 1e-12
ARMS = {'cached_p3': 21, 'native_p3': 18}
KINDS = ('unchanged', 'm_exact_moved', 'wrong_direction',
         'correct_overshoot_harm', 'correct_crossed_beneficial',
         'correct_not_crossed_beneficial', 'correct_hit_gt', 'correct_equal_error')
HARM_KINDS = ('wrong_direction', 'correct_overshoot_harm', 'm_exact_moved')


def classify(before, after, gt):
    """Signed errors and updates in GT-normalized linear length units.

    Direction: update toward or away from GT. Benefit: actual change in
    absolute relative error. Overshooting GT can still be beneficial.
    """
    if not all(math.isfinite(x) and x > 0 for x in (before, after, gt)):
        raise ValueError('Finite positive edge lengths required')
    x, y = before/gt-1., after/gt-1.
    delta = (after-before)/gt
    change = abs(y)-abs(x)
    effect = 'harmed' if change > EPS else 'improved' if change < -EPS else 'tied'
    crossed = x*y < 0
    if abs(delta) <= EPS:
        kind = 'unchanged'
    elif abs(x) <= EPS:
        kind = 'm_exact_moved'
    elif (delta > 0) == (x > 0):
        kind = 'wrong_direction'
    elif effect == 'harmed':
        if not crossed:
            raise ValueError('Toward-GT harmful update must overshoot')
        kind = 'correct_overshoot_harm'
    elif effect == 'tied':
        kind = 'correct_equal_error'
    elif abs(y) <= EPS:
        kind = 'correct_hit_gt'
    elif crossed:
        kind = 'correct_crossed_beneficial'
    else:
        kind = 'correct_not_crossed_beneficial'
    return dict(kind=kind, effect=effect, before_error=x, after_error=y,
                update_gt_normalized=delta, abs_error_delta=change,
                crossed_gt=crossed, near=abs(x) <= .03,
                beneficial_under_remaining_gt3=(kind == 'correct_not_crossed_beneficial' and abs(y) > .03),
                beneficial_under_remaining_gt10=(kind == 'correct_not_crossed_beneficial' and abs(y) > .1))


def load_rows(path):
    with gzip.open(path, 'rt') as f:
        values = [json.loads(s) for s in f]
    indexed = {}
    for r in values:
        if r['split'] not in ('train', 'train_sim', 'val'):
            raise ValueError('Only saved TRAIN/VAL allowed')
        if r['split'] == 'train_sim' and r['domain'] != 'sim':
            raise ValueError('Original Sim TRAIN identity')
        key = (r['scale'], r['image'])
        if key in indexed:
            raise ValueError('Duplicate view')
        indexed[key] = r
    return indexed


def bind(base, left, right, split):
    if base.keys() != left.keys() or base.keys() != right.keys():
        raise ValueError('Paired view identity differs')
    result = []
    for key, m in sorted(base.items()):
        for r in (left[key], right[key]):
            for k in ('image', 'sequence', 'frame_id', 'domain', 'scale', 'split', 'gt', 'b', 'midpoint'):
                if r[k] != m[k]:
                    raise ValueError('Paired metadata differs: '+k)
            p = r['edge_residual']; reference = m['midpoint']
            if (p is None) != (reference is None):
                raise ValueError('New/lost output')
            if p is not None:
                if any(p[j] != reference[j] for j in (0, 1, 4, 5)) or (p[2] < p[3]) != (reference[2] < reference[3]):
                    raise ValueError('Frozen geometry/score/axis identity differs')
        row = {k: m[k] for k in ('image', 'sequence', 'domain', 'scale', 'frame_id')}
        row.update(split=split, present=m['midpoint'] is not None, edges={})
        if row['present']:
            gt, mid = canonical(m['gt']), canonical(m['midpoint'])
            for arm, raw in [('cached_p3', left[key]), ('native_p3', right[key])]:
                p = canonical(raw['edge_residual'])
                edges = [classify(mid[j], p[j], gt[j]) for j in (2, 3)]
                for j, e in enumerate(edges):
                    other = edges[1-j]
                    e['cross_edge_near_harmed'] = (e['near'] and e['effect'] == 'harmed'
                        and not other['near'] and other['effect'] == 'improved')
                row['edges'][arm] = edges
        result.append(row)
    return result


def summary(events):
    n = len(events)
    changes = [e['abs_error_delta'] for e in events]
    harmful = [x for x in changes if x > EPS]
    gains = [-x for x in changes if x < -EPS]
    return dict(count=n, improved=sum(e['effect'] == 'improved' for e in events),
        harmed=len(harmful), tied=sum(e['effect'] == 'tied' for e in events),
        before_mae_pct=100*mean([abs(e['before_error']) for e in events]) if n else None,
        after_mae_pct=100*mean([abs(e['after_error']) for e in events]) if n else None,
        mean_error_delta_pp=100*mean(changes) if n else None,
        harm_sum_pp_edges=100*math.fsum(harmful), gain_sum_pp_edges=100*math.fsum(gains),
        mean_harm_pp=100*mean(harmful) if harmful else None,
        harm_p95_pp=100*quantile(harmful) if harmful else None,
        after_abs_p95_pct=100*quantile([abs(e['after_error']) for e in events]) if n else None,
        crossed_gt=sum(e['crossed_gt'] for e in events),
        beneficial_under_remaining_gt3=sum(e['beneficial_under_remaining_gt3'] for e in events),
        beneficial_under_remaining_gt10=sum(e['beneficial_under_remaining_gt10'] for e in events))


def grouped(records):
    for split in ('train', 'val'):
        for scale in sorted({r['scale'] for r in records if r['split'] == split}):
            rs = [r for r in records if r['split'] == split and r['scale'] == scale]
            groups = dict(overall=rs)
            groups.update({d: [r for r in rs if r['domain'] == d] for d in ('real', 'sim')})
            groups.update({v: [r for r in rs if r['sequence'] == v] for v in sorted({r['sequence'] for r in rs})})
            for group, rows in groups.items():
                yield dict(split=split, scale=scale, group=group, frames=len(rows),
                           outputs=sum(r['present'] for r in rows), missing=sum(not r['present'] for r in rows)), rows


def analyze(records):
    behavior, overlap = [], []
    for prefix, rows in grouped(records):
        for edge, js in [('long', [0]), ('short', [1]), ('both_edges', [0, 1])]:
            paired = [(r['edges']['cached_p3'][j], r['edges']['native_p3'][j])
                      for r in rows if r['present'] for j in js]
            for stratum in ('all', 'near', 'nonnear', 'cross_edge_near_harmed'):
                for arm, index in [('cached_p3', 0), ('native_p3', 1)]:
                    events = [p[index] for p in paired if stratum == 'all'
                              or (stratum == 'near' and p[index]['near'])
                              or (stratum == 'nonnear' and not p[index]['near'])
                              or (stratum == 'cross_edge_near_harmed' and p[index]['cross_edge_near_harmed'])]
                    stem = dict(prefix, edge=edge, stratum=stratum, arm=arm, denominator_edges=len(events))
                    behavior.append(dict(stem, kind='all', **summary(events)))
                    for kind in KINDS:
                        sub = [e for e in events if e['kind'] == kind]
                        behavior.append(dict(stem, kind=kind, **summary(sub)))
                    assert sum(e['kind'] in KINDS for e in events) == len(events)
                # Common failures use the same fixed-M edge identities, not
                # independently filtered cross-edge subsets in the two arms.
                if stratum == 'cross_edge_near_harmed':
                    continue
                pairs = [p for p in paired if stratum == 'all' or p[0]['near'] == (stratum == 'near')]
                for label in ('both_harmed', 'cached_only_harmed', 'native_only_harmed', 'neither_harmed'):
                    def membership(pair):
                        a, n = [p['effect'] == 'harmed' for p in pair]
                        return 'both_harmed' if a and n else 'cached_only_harmed' if a else 'native_only_harmed' if n else 'neither_harmed'
                    sub = [p for p in pairs if membership(p) == label]
                    types = Counter(a['kind']+'|'+n['kind'] for a, n in sub)
                    overlap.append(dict(prefix, edge=edge, stratum=stratum, category=label,
                        denominator_edges=len(pairs), count=len(sub),
                        cached_harm_sum_pp_edges=summary([p[0] for p in sub])['harm_sum_pp_edges'],
                        native_harm_sum_pp_edges=summary([p[1] for p in sub])['harm_sum_pp_edges'],
                        kind_pairs=dict(types)))
    return behavior, overlap


def validate_decomposition(behavior, overlap):
    """Counts, harmful amounts and same-edge intersections must reconcile."""
    grouped_cells = {}
    for r in behavior:
        key = tuple(r[k] for k in ('split', 'scale', 'group', 'edge', 'stratum', 'arm'))
        grouped_cells.setdefault(key, {})[r['kind']] = r
    for key, cells in grouped_cells.items():
        total = cells['all']
        if sum(cells[k]['count'] for k in KINDS) != total['count']:
            raise ValueError('Behavior categories do not partition edges')
        if sum(cells[k]['harmed'] for k in HARM_KINDS) != total['harmed']:
            raise ValueError('Unclassified harmful edge')
        if not math.isclose(math.fsum(cells[k]['harm_sum_pp_edges'] for k in HARM_KINDS),
                            total['harm_sum_pp_edges'], abs_tol=1e-8, rel_tol=1e-12):
            raise ValueError('Harm amount not conserved')
    paired = {}
    for r in overlap:
        key = tuple(r[k] for k in ('split', 'scale', 'group', 'edge', 'stratum'))
        paired.setdefault(key, {})[r['category']] = r
    for key, cells in paired.items():
        if sum(r['count'] for r in cells.values()) != cells['both_harmed']['denominator_edges']:
            raise ValueError('Paired categories do not partition edges')
        for arm, only in [('cached_p3', 'cached_only_harmed'), ('native_p3', 'native_only_harmed')]:
            total = grouped_cells[key+(arm,)]['all']
            if total['harmed'] != cells['both_harmed']['count']+cells[only]['count']:
                raise ValueError('Same-edge harmful intersection differs')
    return dict(category_count_and_harm_amount_conservation=True,
                paired_failure_count_conservation=True)


def run(source, out):
    source, out = source.resolve(), out.resolve()
    # A result-only sibling, never an existing directory or a child of the
    # immutable original run.
    if out.exists() or out.parent != source.parent or out == source:
        raise ValueError('Fresh sibling result directory required')
    report = read(source/'training.json')
    if report['protocol'] != 'port_size_native_train_v1' or report['status'] != 'COMPLETE' or report['formal_head_updates'] != 30672:
        raise ValueError('Not completed S031')
    index = read(source/'artifacts.json')['files']; inputs = {}
    def checked(name):
        p = source/name
        if name not in index or sha(p) != index[name]:
            raise ValueError('Input artifact drift: '+name)
        inputs[name] = index[name]
        return p
    checked('training.json'); records = []
    for split in ('train', 'val'):
        base = load_rows(checked('baseline_'+split+'.rows.jsonl.gz'))
        arms = [load_rows(checked(arm+'/epoch_%02d_'%ep+split+'.rows.jsonl.gz')) for arm, ep in ARMS.items()]
        part = bind(base, *arms, split)
        expected = 5116 if split == 'train' else 887
        if len(part) != expected or sum(r['present'] for r in part) != (5112 if split == 'train' else 886):
            raise ValueError('Full saved split coverage differs')
        if split == 'train' and any(sum(r['scale'] == sc for r in part) != 2558 for sc in (.5, 1.)):
            raise ValueError('TRAIN scale identity')
        if split == 'val' and any(r['scale'] != 1. for r in part):
            raise ValueError('Natural VAL scale')
        records.extend(part)
    behavior, overlap = analyze(records)
    validation = validate_decomposition(behavior, overlap)
    out.mkdir(parents=True, exist_ok=False)
    csv_write(out/'behavior_summary.csv', behavior)
    csv_write(out/'paired_failures.csv', [dict(r, kind_pairs=json.dumps(r['kind_pairs'], sort_keys=True)) for r in overlap])
    with gzip.open(out/'paired_edge_events.jsonl.gz', 'xt') as f:
        for r in records:
            f.write(json.dumps(r, allow_nan=False)+'\n')
    source_files = [Path(__file__).resolve(), ROOT/'crane_project/tools/analyze_port_size_joint_v1.py', ROOT/'tests/test_port_size_correction_behavior_v1.py']
    value = dict(protocol=VERSION, status='COMPLETE_SAVED_TRAIN_VAL_ONLY',
        fixed_epochs=ARMS, selection_changed=False, inputs=inputs,
        input_artifacts_sha256=sha(source/'artifacts.json'),
        code_sha256={str(p.relative_to(ROOT)): sha(p) for p in source_files},
        definitions=dict(error='x=M/GT-1; y=P/GT-1; update=(P-M)/GT; harm=abs(y)-abs(x)>1e-12',
            direction='update has same sign as x => away; opposite => toward; exact M separately',
            overshoot='toward, crosses GT AND actual absolute relative error increases; crossing alone is not harm',
            near='fixed M abs(x)<=0.03; nonnear>0.03; neither is an inference label',
            under='toward, not crossing, actual error decreases; remaining>3%/10% are existing diagnostic thresholds only',
            amounts='harm_sum_pp_edges sums positive increments; divide by group edge count for per-edge MAE cost. Scales not pooled as independent samples.',
            cross_edge='this fixed-M near edge harmed while other fixed-M nonnear edge improves'),
        coverage=[dict(p) for p, _ in grouped(records)],
        behavior=behavior, overlap=overlap, validation=validation, model_calls=0, gpu_calls=0,
        test_reads=0, depth_reads=0, new_training=0, coefficient_search=False,
        limits='Offline behavior cannot identify training cause or supply inference-time direction/amplitude evidence. No new candidate or method chosen by this tool.')
    write(out/'analysis.json', value)
    write(out/'artifacts.json', dict(files={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(status=value['status'], views=len(records), behavior_cells=len(behavior), overlap_cells=len(overlap), source_files=len(inputs))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    run(args.run, args.out)
