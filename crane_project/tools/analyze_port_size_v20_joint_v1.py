#!/usr/bin/env python3
"""S036 fixed V20 paired audit of saved TRAIN/VAL only; no model or search.

10% correctness and fixed-M 3% membership are inherited. Excess is continuous:
no additional severity bins, re-selection, inference, TEST or depth inputs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.analyze_port_size_joint_v1 import (
    canonical, csv_write, mean, quantile, read, sha, state, write)
from crane_project.tools.analyze_port_size_correction_behavior_v1 import classify

VERSION = 'port_size_v20_joint_v1'
EPS = 1e-12
PINS = dict(
    v20='0a44b4254e5e1b1c929b672421dce40b0acd5ed75671f2767a08134c8aa8ca24',
    l20='b6a37d455a3c923233c698faeab069997be929c3a7c857305f1bd21aea3b3527')
INDEX_PINS = dict(
    v20='3fe2dfb1c607eb3755d1ebc2053b56f5aeaf7c0dc95069804f4ad15248077327',
    l20='6b6c1d76703e6b4e5d77dea8ae607a50a788019e086a5495e0ddf9aa0d05b7d3',
    prior='7ac314034b53ffeaf64991749ab42bda2bad5e24218c07caba0c01ba48a93a6a')
MEMBERS = ('both_near', 'long_near_short_error', 'short_near_long_error', 'both_error')
COHORTS = ('retained', 'lost', 'new', 'neither')
PAIRS = (('b', 'v20'), ('m', 'v20'), ('b', 'l20'), ('m', 'l20'), ('v20', 'l20'))


def geometry(gt, box, riou):
    if box is None:
        if riou is not None:
            raise ValueError('Missing box has saved RIoU')
        return None
    g, p = canonical(gt), canonical(box)
    e = [p[j]/g[j]-1 for j in (2, 3)]
    z = [math.log(p[j]/g[j]) for j in (2, 3)]
    if riou is not None and (not math.isfinite(riou) or not 0 <= riou <= 1):
        raise ValueError('Invalid saved RIoU')
    return dict(e=e, a=sum(z)/2, r=z[0]-z[1], riou=riou,
                correct=max(map(abs, e)) <= .1,
                excess=max(0., max(map(abs, e))-.1))


def member(g):
    if g is None:
        return 'missing'
    x, y = [abs(v) <= .03 for v in g['e']]
    return MEMBERS[0] if x and y else MEMBERS[1] if x else MEMBERS[2] if y else MEMBERS[3]


def edge_group(g, edge=1):
    x = g['e'][edge]
    return 'near' if abs(x) <= .03 else 'need_enlarge' if x < 0 else 'need_shrink'


def cohort(row):
    m, v, l = [row['g'][k] for k in ('m', 'v20', 'l20')]
    a = abs(v['e'][1]) < abs(m['e'][1])-EPS
    b = abs(l['e'][1]) < abs(m['e'][1])-EPS
    return 'retained' if a and b else 'lost' if a else 'new' if b else 'neither'


def load(path):
    with gzip.open(path, 'rt') as f:
        rows = [__import__('json').loads(x) for x in f]
    index = {}
    for r in rows:
        if r['split'] not in ('train', 'train_sim', 'val'):
            raise ValueError('Only saved TRAIN/VAL accepted')
        if r['split'] == 'train_sim' and r['domain'] != 'sim':
            raise ValueError('Sim TRAIN source identity')
        key = (r['scale'], r['image'])
        if key in index:
            raise ValueError('Duplicate frame/view')
        index[key] = r
    return index


def bind(left, right, split):
    if left.keys() != right.keys():
        raise ValueError('Paired view identity mismatch')
    records = []
    for key in sorted(left):
        a, b = left[key], right[key]
        for k in ('gt', 'b', 'midpoint', 'domain', 'sequence', 'image', 'scale', 'split', 'frame_id'):
            if a[k] != b[k]:
                raise ValueError('Paired metadata mismatch '+k)
        if ('val' if a['split'] == 'val' else 'train') != split:
            raise ValueError('Split role mismatch')
        for r in (a, b):
            m, p = r['midpoint'], r['edge_residual']
            if (m is None) != (p is None):
                raise ValueError('V/L changes M output identity')
            if m is not None and (any(m[j] != p[j] for j in (0, 1, 4, 5)) or
                                  (m[2] < m[3]) != (p[2] < p[3])):
                raise ValueError('Frozen geometry/score/axis mismatch')
        row = dict(split=split, source_split=a['split'], scale=a['scale'], image=a['image'],
                   domain=a['domain'], sequence=a['sequence'], frame_id=a['frame_id'], g={})
        for k, r, source in (('b', a, 'b'), ('m', a, 'midpoint'),
                             ('v20', a, 'edge_residual'), ('l20', b, 'edge_residual')):
            box = r[source]
            riou = r['metrics'][source]['riou'] if box is not None and 'metrics' in r else None
            row['g'][k] = geometry(r['gt'], box, riou)
        row['member'] = member(row['g']['m'])
        records.append(row)
    return records


def transition(row, before, after):
    x, y = row['g'][before], row['g'][after]
    if x is None or y is None:
        return 'both_missing' if x is None and y is None else 'new_output' if x is None else 'lost_output'
    return ('right' if x['correct'] else 'wrong')+'_to_'+('right' if y['correct'] else 'wrong')


def magnitude(values, prefix, out, factor=1.):
    """Separate positive gains/harms before any averaging; never bias cancellation."""
    gain = [-v for v in values if v < -EPS]
    harm = [v for v in values if v > EPS]
    out.update({prefix+'_improved': len(gain), prefix+'_harmed': len(harm),
                prefix+'_tied': len(values)-len(gain)-len(harm),
                prefix+'_gain_sum': factor*math.fsum(gain),
                prefix+'_harm_sum': factor*math.fsum(harm),
                prefix+'_net_mean': factor*mean(values) if values else None})


def summary(rows, before, after):
    pairs = [(r['g'][before], r['g'][after]) for r in rows
             if r['g'][before] is not None and r['g'][after] is not None]
    n = len(rows)
    out = dict(frames=n, paired_outputs=len(pairs),
               before_outputs=sum(r['g'][before] is not None for r in rows),
               after_outputs=sum(r['g'][after] is not None for r in rows))
    for label in ('right_to_right', 'right_to_wrong', 'wrong_to_right', 'wrong_to_wrong',
                  'new_output', 'lost_output', 'both_missing'):
        out[label] = sum(transition(r, before, after) == label for r in rows)
    for which, method in (('before', before), ('after', after)):
        gs = [r['g'][method] for r in rows if r['g'][method] is not None]
        out[which+'_RIoU_evaluable_outputs'] = sum(g['riou'] is not None for g in gs)
        out[which+'_RIoU_full'] = (math.fsum(g['riou'] for g in gs)/n
            if n and all(g['riou'] is not None for g in gs) else None)
        out[which+'_correct'] = sum(g['correct'] for g in gs)
        out[which+'_both_undersized'] = sum(max(g['e']) < 0 for g in gs)
        out[which+'_both_oversized'] = sum(min(g['e']) > 0 for g in gs)
        out[which+'_both_edges_gt10'] = sum(min(map(abs, g['e'])) > .1 for g in gs)
        for j, edge in enumerate(('long', 'short')):
            es = [g['e'][j] for g in gs]
            out[which+'_'+edge+'_MAE_pct'] = 100*mean(list(map(abs, es))) if es else None
            out[which+'_'+edge+'_bias_pct'] = 100*mean(es) if es else None
            out[which+'_'+edge+'_P95_pct'] = 100*quantile(list(map(abs, es))) if es else None
        for k in ('a', 'r'):
            out[which+'_'+k+'_meanabs'] = mean([abs(g[k]) for g in gs])
        xs = [g['excess'] for g in gs]
        out[which+'_excess_pp_median'] = 100*median(xs) if xs else None
        out[which+'_excess_pp_P95'] = 100*quantile(xs) if xs else None
        out[which+'_excess_pp_max'] = 100*max(xs) if xs else None
    for j, edge in enumerate(('long', 'short')):
        magnitude([abs(y['e'][j])-abs(x['e'][j]) for x, y in pairs], edge+'_pp_edges', out, 100)
    for k in ('a', 'r'):
        magnitude([abs(y[k])-abs(x[k]) for x, y in pairs], k+'_log', out)
    riou_pairs = [(x, y) for x, y in pairs if x['riou'] is not None and y['riou'] is not None]
    out['RIoU_paired_evaluable_outputs'] = len(riou_pairs)
    magnitude([x['riou']-y['riou'] for x, y in riou_pairs], 'RIoU', out)
    out['new_failure_other_edge_improved'] = sum(
        x['correct'] and not y['correct'] and any(
            abs(y['e'][j]) > .1 and abs(y['e'][1-j]) < abs(x['e'][1-j])-EPS for j in (0, 1))
        for x, y in pairs)
    out['error_edge_improved_near_other_harmed'] = sum(any(
        abs(r['g']['m']['e'][j]) > .03 and
        abs(y['e'][j]) < abs(x['e'][j])-EPS and abs(r['g']['m']['e'][1-j]) <= .03 and
        abs(y['e'][1-j]) > abs(x['e'][1-j])+EPS for j in (0, 1))
        for r in rows if r['g'][before] is not None and r['g'][after] is not None
        for x, y in [(r['g'][before], r['g'][after])])
    return out


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n//2] if n % 2 else (s[n//2-1]+s[n//2])/2


def groups(records):
    for split in ('train', 'val'):
        for scale in sorted({r['scale'] for r in records if r['split'] == split}):
            base = [r for r in records if r['split'] == split and r['scale'] == scale]
            for group in ['overall', 'real', 'sim']+sorted({r['sequence'] for r in base}):
                yield dict(split=split, scale=scale, group=group), [
                    r for r in base if group in ('overall', r['domain'], r['sequence'])]


def frame_details(records):
    for r in records:
        for before, after in PAIRS:
            stem = {k: r[k] for k in ('split', 'scale', 'image', 'domain', 'sequence', 'frame_id')}
            stem.update(pair=before+'->'+after, M_member=r['member'], transition=transition(r, before, after))
            x, y = r['g'][before], r['g'][after]
            for label, g in (('before', x), ('after', y)):
                if g is None:
                    continue
                stem.update({label+'_long_error_pct': 100*g['e'][0], label+'_short_error_pct': 100*g['e'][1],
                             label+'_a': g['a'], label+'_r': g['r'], label+'_RIoU': g['riou'],
                             label+'_max_excess_pp': 100*g['excess']})
                for j, edge in enumerate(('long', 'short')):
                    stem[label+'_'+edge+'_excess_pp'] = 100*max(0., abs(g['e'][j])-.1)
            if x is not None and y is not None:
                for j, edge in enumerate(('long', 'short')):
                    stem[edge+'_change'] = state(abs(y['e'][j])-abs(x['e'][j]))
                for k in ('a', 'r'):
                    stem[k+'_change'] = state(abs(y[k])-abs(x[k]))
            yield stem


def short_responses(records):
    """View-level paired response, GT only offline; never pair TRAIN with VAL IDs."""
    result = []
    for r in records:
        if r['g']['m'] is None:
            continue
        m, v, l = [r['g'][k] for k in ('m', 'v20', 'l20')]
        dv, dl = v['e'][1]-m['e'][1], l['e'][1]-m['e'][1]
        sign = lambda x: 1 if x > EPS else -1 if x < -EPS else 0
        out = {k: r[k] for k in ('split', 'scale', 'image', 'domain', 'sequence')}
        out.update(cohort=cohort(r), M_member=r['member'], M_short_group=edge_group(m),
                   V_short_kind=classify(1+m['e'][1], 1+v['e'][1], 1.)['kind'],
                   L_short_kind=classify(1+m['e'][1], 1+l['e'][1], 1.)['kind'],
                   same_direction=sign(dv) == sign(dl) and sign(dv) != 0,
                   reversed_direction=sign(dv)*sign(dl) == -1,
                   smaller_absolute_update=abs(dl) < abs(dv)-EPS,
                   V_short_update_pct_GT=100*dv, L_short_update_pct_GT=100*dl,
                   M_short_error_pct=100*m['e'][1], V_short_error_pct=100*v['e'][1],
                   L_short_error_pct=100*l['e'][1])
        result.append(out)
    return result


def analyze(records):
    transitions, joint, responses = [], [], short_responses(records)
    cohorts = []
    for stem, rows in groups(records):
        present = [r for r in rows if r['g']['m'] is not None]
        for before, after in PAIRS:
            for label in ('all', 'right_to_right', 'right_to_wrong', 'wrong_to_right', 'wrong_to_wrong',
                          'new_output', 'lost_output', 'both_missing'):
                rs = rows if label == 'all' else [r for r in rows if transition(r, before, after) == label]
                transitions.append(dict(stem, pair=before+'->'+after, transition=label, **summary(rs, before, after)))
        for after in ('v20', 'l20'):
            cells = {}
            for r in present:
                x, y = r['g']['m'], r['g'][after]
                k = (state(abs(y['a'])-abs(x['a'])), state(abs(y['r'])-abs(x['r'])),
                     state(abs(y['e'][1])-abs(x['e'][1])), r['member'])
                cells.setdefault(k, []).append(r)
            assert sum(len(v) for v in cells.values()) == len(present)
            for (a, b, c, d), rs in sorted(cells.items()):
                joint.append(dict(stem, pair='m->'+after, a_change=a, r_change=b, short_change=c,
                                  M_member=d, **summary(rs, 'm', after)))
        # Fixed short groups and near membership are crossed, never chosen from VAL.
        for label in COHORTS:
            base = [r for r in present if cohort(r) == label]
            for sg in ('all', 'near', 'need_enlarge', 'need_shrink'):
                for mg in ('all',)+MEMBERS:
                    rs = [r for r in base if (sg == 'all' or edge_group(r['g']['m']) == sg)
                          and (mg == 'all' or r['member'] == mg)]
                    ids = {(r['scale'], r['image']) for r in rs}
                    actual = [r for r in responses if r['split'] == stem['split'] and (r['scale'], r['image']) in ids]
                    cell = dict(stem, cohort=label, M_short_group=sg, M_member=mg, frames=len(rs))
                    for prefix, method in (('V', 'v20'), ('L', 'l20')):
                        cell.update({prefix+'_'+k: v for k, v in summary(rs, 'm', method).items()})
                    cell.update(same_direction=sum(r['same_direction'] for r in actual),
                                reversed_direction=sum(r['reversed_direction'] for r in actual),
                                same_direction_smaller=sum(r['same_direction'] and r['smaller_absolute_update'] for r in actual),
                                L_wrong_direction=sum(r['L_short_kind'] == 'wrong_direction' for r in actual),
                                L_correct_overshoot_harm=sum(r['L_short_kind'] == 'correct_overshoot_harm' for r in actual))
                    cohorts.append(cell)
    return transitions, joint, responses, cohorts


def run(control, local, prior, out):
    if out.exists() or out.parent != local.parent:
        raise ValueError('Require fresh sibling result directory')
    inputs = {}
    for name, directory in (('v20', control), ('l20', local), ('prior', prior)):
        if sha(directory/'artifacts.json') != INDEX_PINS[name]:
            raise ValueError('Pinned result index differs '+name)
        inputs[str((directory/'artifacts.json').relative_to(ROOT))] = INDEX_PINS[name]
    source_paths = []
    for name, directory, prefix in (('v20', control, 'size_only/'), ('l20', local, '')):
        index = read(directory/'artifacts.json')['files']
        for n in (prefix+'head_epoch_20.pth', prefix+'epoch_20_train.rows.jsonl.gz', prefix+'epoch_20_val.rows.jsonl.gz'):
            digest = sha(directory/n)
            if digest != index[n] or (n.endswith('.pth') and digest != PINS[name]):
                raise ValueError('Pinned input differs '+n)
            inputs[str((directory/n).relative_to(ROOT))] = digest
        source_paths.append(directory/prefix)
    for n, digest in read(prior/'artifacts.json')['files'].items():
        if sha(prior/n) != digest:
            raise ValueError('Prior verified analysis differs '+n)
    records = []
    for split, expected, outputs in (('train', 5116, 5112), ('val', 887, 886)):
        a, b = [load(p/('epoch_20_'+split+'.rows.jsonl.gz')) for p in source_paths]
        rs = bind(a, b, split)
        if len(rs) != expected or sum(r['g']['m'] is not None for r in rs) != outputs:
            raise ValueError('Full split/view count mismatch')
        records.extend(rs)
    transitions, joint, responses, cohorts = analyze(records)
    from csv import DictReader
    with (prior/'beneficial_retention.csv').open() as f:
        old = list(DictReader(f))
    for stem, rows in groups(records):
        values = Counter(cohort(r) for r in rows if r['g']['m'] is not None)
        ref = next(x for x in old if x['split'] == stem['split'] and float(x['scale']) == stem['scale']
                   and x['group'] == stem['group'] and x['edge'] == 'short' and x['stratum'] == 'all')
        if values['retained'] != int(ref['candidate_retained']) or values['retained']+values['lost'] != int(ref['control_beneficial']):
            raise ValueError('Existing beneficial retention changed')
    frozen = Counter(r['cohort'] for r in responses if r['split'] == 'val' and r['domain'] == 'sim')
    if frozen != dict(retained=251, lost=127, new=70, neither=64):
        raise ValueError('Fixed four-cohort inventory differs')
    # Verify the expected threshold transitions against already-verified S035.
    old_summary = read(prior/'analysis.json')['selected']
    for arm, method in (('size_only', 'v20'), ('size_plus_local', 'l20')):
        for group, cell in old_summary[arm]['metrics'].items():
            z = next(x for x in transitions if x['split'] == 'val' and x['group'] == group
                     and x['pair'] == 'm->'+method and x['transition'] == 'all')
            if (z['wrong_to_right'], z['right_to_wrong']) != (cell['paired']['wrong_to_right'], cell['paired']['right_to_wrong']):
                raise ValueError('Existing transition count changed')
    out.mkdir()
    for n, rows in (('frame_pairs.csv', list(frame_details(records))), ('threshold_transitions.csv', transitions),
                    ('joint_scale_ratio.csv', joint), ('short_response_frames.csv', responses), ('short_cohorts.csv', cohorts)):
        csv_write(out/n, rows)
    result = dict(protocol=VERSION, completed_utc=datetime.now(timezone.utc).isoformat(),
        inputs=inputs, fixed_checkpoints=PINS, code_sha256=sha(Path(__file__)),
        dependencies={str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT/'crane_project/tools/analyze_port_size_joint_v1.py',
            ROOT/'crane_project/tools/analyze_port_size_correction_behavior_v1.py')},
        counts=dict(train_views=5116, train_outputs=5112, val_frames=887, val_outputs=886,
                    fixed_sim_val_short_cohorts=dict(frozen)),
        checks=dict(metadata_and_delivery_identity=True, transition_counts_reused=True,
                    cohort_counts_reused=True, joint_partition_conserved=True),
        contract=dict(correct='both abs-relative<=0.1', near='fixed M abs-relative<=0.03',
            epsilon=EPS, scale='a=(loglong+logshort)/2', ratio='r=loglong-logshort',
            severity='continuous max excess above0.1; median/P95/max; no new severity threshold',
            direction='GT offline only; actual final delivered canonical lengths',
            units='edge gain/harm sums pp*edges; a/r sums log-units*frames; RIoU sums units*frames',
            cohorts='V/L benefit strictly relative to fixed M short error; not paired TRAIN/VAL identities'),
        boundary='FixedV20/L20 only. No training, inference, GT-based online rule, selector, TEST/depth/formula reads or GPU. B->V RIoU includes center/angle differences. TRAIN RIoU not saved: unavailable, not recomputed; inspect evaluable counts, not zero empty-sample sums.')
    write(out/'analysis.json', result)
    write(out/'artifacts.json', dict(files={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(result['counts'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, default=ROOT/'work_dirs/port_size_local_vectors_v1/train_20261018_v1')
    p.add_argument('--local', type=Path, default=ROOT/'work_dirs/port_size_local_aux_v1/train_20261018_v1')
    p.add_argument('--prior', type=Path, default=ROOT/'work_dirs/port_size_local_aux_v1/analysis_20261019_v1')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run(a.control.resolve(), a.local.resolve(), a.prior.resolve(), a.out.resolve())
