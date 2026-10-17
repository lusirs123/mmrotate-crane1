#!/usr/bin/env python3
"""Post-S019 saved-output diagnosis. No fitting, inference, VAL re-selection or TEST.

All strata are fixed before the diagnostic output: M error groups (existing 3%
definition), and a 2x2 M-size/aspect grid with medians fitted on standard Sim
TRAIN only. The grid is descriptive and never becomes an inference rule.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
from crane_project.utils.port_geometry_size_direction_balance_v1 import capped_weights

VERSION = 'port_geometry_size_sim_support_v1'
GROUPS = ('need_enlarge', 'need_shrink', 'near_correct', 'mixed', 'missing')
METHODS = metrics.METHODS
PAIRS = (('b', 'midpoint'), ('midpoint', 'edge_residual'), ('b', 'edge_residual'))
EPS = 1e-12  # roundoff tie only; not an adoption tolerance


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def load_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def mean(values):
    return sum(values)/len(values) if values else None


def errors(row, method):
    if row[method] is None:
        return None
    gt, pred = (metrics.old.canonical(row[key]) for key in ('gt', method))
    e = [pred[j]/gt[j]-1 for j in (2, 3)]
    z = [math.log1p(x) for x in e]
    return dict(e=e, a=sum(z)/2, r=z[0]-z[1], correct=max(map(abs, e)) <= .1)


def sign(x):
    return 'up' if x > EPS else 'down' if x < -EPS else 'tie'


def frame_delta(row, before, after):
    left, right = errors(row, before), errors(row, after)
    value = dict(image=row['image'], domain=row['domain'], sequence=row['sequence'],
                 split=row['split'], scale=row['scale'], before=before, after=after,
                 m_group=metrics.group(row['gt'], row['midpoint']),
                 output_before=left is not None, output_after=right is not None)
    if left is None or right is None:
        return value
    p, q = (metrics.old.canonical(row[key]) for key in (before, after))
    update = [math.log(q[j]/p[j]) for j in (2, 3)]
    de = [abs(y)-abs(x) for x, y in zip(left['e'], right['e'])]
    value.update(long_update_log=update[0], short_update_log=update[1],
                 long_error_delta=de[0], short_error_delta=de[1],
                 long_signed_error_before=left['e'][0], long_signed_error_after=right['e'][0],
                 short_signed_error_before=left['e'][1], short_signed_error_after=right['e'][1],
                 a_abs_delta=abs(right['a'])-abs(left['a']),
                 r_abs_delta=abs(right['r'])-abs(left['r']),
                 edge_case=sign(-de[0])+'/'+sign(-de[1]),
                 update_case=sign(update[0])+'/'+sign(update[1]),
                 ar_case=sign(abs(left['a'])-abs(right['a']))+'/'+sign(abs(left['r'])-abs(right['r'])),
                 wrong_to_right=int(not left['correct'] and right['correct']),
                 right_to_wrong=int(left['correct'] and not right['correct']))
    if 'metrics' in row:
        value['riou_delta'] = row['metrics'][after]['riou']-row['metrics'][before]['riou']
    return value


def pair_summary(rows, before, after):
    deltas = [frame_delta(r, before, after) for r in rows]
    present = [v for v in deltas if v['output_before'] and v['output_after']]
    short_down = [v for v in present if v['short_update_log'] < -EPS]
    long_gain_short_down = [v for v in short_down if v['long_error_delta'] < -EPS]
    return dict(frames=len(rows), common_outputs=len(present),
        new_output=sum(not v['output_before'] and v['output_after'] for v in deltas),
        lost_output=sum(v['output_before'] and not v['output_after'] for v in deltas),
        edge_cases=dict(Counter(v['edge_case'] for v in present)),
        update_cases=dict(Counter(v['update_case'] for v in present)),
        ar_cases=dict(Counter(v['ar_case'] for v in present)),
        wrong_to_right=sum(v['wrong_to_right'] for v in present),
        right_to_wrong=sum(v['right_to_wrong'] for v in present),
        short_down=len(short_down), short_down_harm=sum(v['short_error_delta'] > EPS for v in short_down),
        long_gain_short_down=len(long_gain_short_down),
        long_gain_short_down_harm=sum(v['short_error_delta'] > EPS for v in long_gain_short_down),
        mean_delta={k:mean([v[k] for v in present if k in v]) for k in
                    ('long_error_delta', 'short_error_delta', 'a_abs_delta', 'r_abs_delta', 'riou_delta')})


def snapshot_loss(row, neutral=False):
    """Recomputed decoded SmoothL1 snapshot; not online losses or gradients."""
    if row['midpoint'] is None:
        return None
    target = row['target_log']
    prediction = [0., 0.] if neutral else row['delta_roi']
    if len(target) != 2 or len(prediction) != 2:
        raise ValueError('Missing saved decoded target/output')
    delta = [abs(x-y) for x, y in zip(prediction, target)]
    return mean([.5*x*x/.1 if x < .1 else x-.05 for x in delta])


def features(row):
    p, b = (metrics.old.canonical(row[key]) for key in ('midpoint', 'b'))
    return [math.log(math.sqrt(p[2]*p[3])), math.log(p[2]/p[3]),
            math.log(p[2]/b[2]), math.log(p[3]/b[3])]


def fit_strata(train):
    # Original-image units only. The saved rows lack actual model scale_xy.
    rs = [r for r in train if r['domain'] == 'sim' and r['scale'] == 1. and r['midpoint'] is not None]
    xs = [features(r) for r in rs]
    return dict(fit='standard Sim TRAIN M; no VAL/GT fitted cutpoints', fit_n=len(rs),
        names=['log_geomean_side_original_px', 'log_aspect', 'log_M_over_B_long', 'log_M_over_B_short'],
        medians=[metrics.old.percentile([x[j] for x in xs], .5) for j in range(2)],
        ranges=[[min(x[j] for x in xs), max(x[j] for x in xs)] for j in range(4)])


def stratum(row, fitted):
    x = features(row)
    return '/'.join(('high' if x[j] > fitted['medians'][j] else 'low') for j in range(2))


def axis_balance_weights(counts):
    """TRAIN-only proposed arithmetic, NOT fitted/applied to any predictions.

    Hold near3 at 1. Equalize the two remaining total contributions subject to
    cap4 while retaining per-axis/cell mean one. Missing direction is unassessed.
    """
    if set(counts) != {'undersized', 'oversized', 'near3'} or any(v < 0 for v in counts.values()):
        raise ValueError('Expected nonnegative complete axis counts')
    if not counts['undersized'] or not counts['oversized']:
        return None
    n = sum(counts[g] for g in ('undersized', 'oversized'))
    low, high = 0., float(n)
    for _ in range(100):
        mid = (low+high)/2
        if sum(min(4*counts[g], mid) for g in ('undersized', 'oversized')) < n:
            low = mid
        else:
            high = mid
    weights = dict(near3=1., **{g:min(4., high/counts[g]) for g in ('undersized', 'oversized')})
    if not math.isclose(sum(counts[g]*weights[g] for g in counts), sum(counts.values()), abs_tol=1e-9):
        raise ValueError('Proposed weights lost mean-one accounting')
    return weights


def cell_summary(rows, loss=False):
    result = dict(frames=len(rows), groups=dict(Counter(metrics.group(r['gt'], r['midpoint']) for r in rows)),
        methods={m:metrics.size_summary(rows, m) for m in METHODS},
        pairs={a+'->'+b:pair_summary(rows, a, b) for a, b in PAIRS})
    if loss:
        actual = [r for r in rows if r['midpoint'] is not None]
        counts = Counter(metrics.group(r['gt'], r['midpoint']) for r in actual)
        weights = capped_weights(dict(counts)) if set(counts) == set(GROUPS[:-1]) else None
        total0 = sum(snapshot_loss(r, True) for r in actual)
        total7 = sum(snapshot_loss(r) for r in actual)
        result['loss_snapshot'] = dict(neutral_mean=total0/len(actual) if actual else None,
            selected_mean=total7/len(actual) if actual else None, groups={},
            old_formula_train_only_weights=weights,
            note='Mean-one cap4 group balance on COMPLETE cell counts; arithmetic only. Not old fit200 exposure, an optimizer replay, or proposed VAL weights.')
        weighted0 = sum(snapshot_loss(r, True)*weights[metrics.group(r['gt'], r['midpoint'])] for r in actual) if weights else None
        weighted7 = sum(snapshot_loss(r)*weights[metrics.group(r['gt'], r['midpoint'])] for r in actual) if weights else None
        for group in GROUPS[:-1]:
            subset = [r for r in actual if metrics.group(r['gt'], r['midpoint']) == group]
            l0 = sum(snapshot_loss(r, True) for r in subset)
            l7 = sum(snapshot_loss(r) for r in subset)
            result['loss_snapshot']['groups'][group] = dict(count=len(subset),
                neutral_sum=l0, selected_sum=l7, neutral_share=l0/total0 if total0 else None,
                selected_share=l7/total7 if total7 else None,
                weighted_neutral_share=l0*weights[group]/weighted0 if weighted0 else None,
                weighted_selected_share=l7*weights[group]/weighted7 if weighted7 else None,
                weighted_exposures=len(subset)*weights[group] if weights else None)
    return result


def verify_rows(rows, split, reference=None):
    expected = 5116 if split == 'train' else 887
    if len(rows) != expected:
        raise ValueError('Incomplete saved '+split)
    keyed = {}
    for r in rows:
        key = (r['image'], r['scale'])
        allowed_split = ('train', 'train_sim') if split == 'train' else ('val',)
        if key in keyed or r['split'] not in allowed_split or r['domain'] not in ('sim', 'real'):
            raise ValueError('Duplicate or invalid row identity')
        if r['scale'] not in ((1., .5) if split == 'train' else (1.,)):
            raise ValueError('Unexpected scale')
        keyed[key] = r
        for method in METHODS:
            if r[method] is not None:
                errors(r, method)
        if reference is not None:
            old = reference[key]
            for field in ('gt', 'b', 'midpoint', 'domain', 'sequence', 'frame_id', 'target_log', 'shard'):
                if r[field] != old[field]:
                    raise ValueError('Baseline drift: '+str(key)+'/'+field)
        m, c = r['midpoint'], r['edge_residual']
        if (m is None) != (c is None) or (m is not None and [m[j] for j in (0, 1, 4, 5)] != [c[j] for j in (0, 1, 4, 5)]):
            raise ValueError('Frozen M output/center/raw angle/score changed')
    return keyed


def verify_summary(rows, summary, method='edge_residual'):
    for domain in ('real', 'sim'):
        cell = [r for r in rows if r['domain'] == domain]
        computed = metrics.size_summary(cell, method)
        expected = summary[domain][method]['sizes']
        for edge in ('long', 'short'):
            for k in ('mae', 'p95', 'bias'):
                if abs(computed['edges'][edge][k]-expected['edges'][edge][k]) > 1e-12:
                    raise ValueError('Saved summary disagreement')
        if computed['joint10'] != expected['joint10'] or computed['outputs'] != expected['outputs']:
            raise ValueError('Saved denominator disagreement')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader(); writer.writerows(rows)


def run(run_dir, output):
    run_dir, output = run_dir.resolve(), output.resolve()
    if output.exists():
        raise ValueError('Refuse to overwrite existing analysis: '+str(output))
    index = read(run_dir/'artifacts.json')['files']
    used = {}

    def checked(name, rows=False):
        path = run_dir/name
        if name not in index or sha(path) != index[name]:
            raise ValueError('Missing/mismatched indexed evidence: '+name)
        used[name] = index[name]
        return load_rows(path) if rows else read(path)

    completion = checked('completion.json')
    if completion['candidate']['selected']['epoch'] != 7 or completion['formal_head_updates'] != 15336:
        raise ValueError('Not the frozen S019/I07 run')
    source_identity = checked('source_identity.json')
    proof = checked('input_runtime_proof.json')
    if proof['git_head'] != '57427341162a9434c8d65419d5b1175ecdd6052b':
        raise ValueError('Unexpected S019 source commit')
    for name in ('port_geometry_size_distribution_metrics_v1.py',
                 'port_geometry_midpoint_edge_residual_metrics_v1.py',
                 'port_geometry_midpoint_edge_feasibility_v1.py'):
        relative = 'crane_project/utils/'+name
        if sha(ROOT/relative) != source_identity['parent']['sources'][relative]:
            raise ValueError('Historical reporting source drift: '+relative)
    head_name = 'candidate/head_epoch_07.pth'
    if sha(run_dir/head_name) != index[head_name] or index[head_name] != 'affe4795b73cae325f878b15c2d81f942461bd965bae0b3c3ca6a7a944eb0464':
        raise ValueError('Selected I07 weight identity changed')
    used[head_name] = index[head_name]  # bytes only, no weight deserialization
    baseline = {s:checked('baseline_'+s+'.rows.jsonl.gz', True) for s in ('train', 'val')}
    refs = {s:verify_rows(baseline[s], s) for s in baseline}
    selected = {s:checked('candidate/'+s+'_epoch_07.rows.jsonl.gz', True) for s in ('train', 'val')}
    for s in selected:
        verify_rows(selected[s], s, refs[s])
    val_summary = checked('candidate/val_epoch_07.summary.json')
    for m in METHODS:
        verify_summary(selected['val'], val_summary, m)
    schedule = checked('schedule.json')
    identities = [(x['image'], x['scale']) for x in schedule['train_view_identity']]
    present = {k for k, r in refs['train'].items() if r['midpoint'] is not None}
    if len(identities) != 5112 or len(set(identities)) != 5112 or set(identities) != present:
        raise ValueError('Training schedule/view identity mismatch')
    exposure = Counter()
    if len(schedule['batches']) != 24:
        raise ValueError('Missing epochs')
    for epoch_batches in schedule['batches']:
        flat = [i for batch in epoch_batches for i in batch]
        if len(epoch_batches) != 639 or any(len(b) != 8 for b in epoch_batches) or sorted(flat) != list(range(5112)):
            raise ValueError('Not one complete TRAIN exposure per epoch')
        for i in flat:
            r = refs['train'][identities[i]]
            exposure[r['domain']+'/'+str(r['scale'])+'/'+metrics.group(r['gt'], r['midpoint'])] += 1
    fitted = fit_strata(selected['train'])
    cells, strata, memberships, deltas = {}, {}, {}, []
    for split, rs in selected.items():
        for domain in ('real', 'sim'):
            for scale in ((1., .5) if split == 'train' else (1.,)):
                cell = [r for r in rs if r['domain'] == domain and r['scale'] == scale]
                name = split+'/'+domain+'/'+str(scale)
                cells[name] = cell_summary(cell, loss=split == 'train')
                cells[name]['fixed_m_groups'] = {g:cell_summary([r for r in cell if metrics.group(r['gt'], r['midpoint']) == g]) for g in GROUPS}
                # One-edge support is not the same as the both-edges group count.
                present_rows = [r for r in cell if r['midpoint'] is not None]
                cells[name]['axis_support'] = {}
                for j, edge in enumerate(('long', 'short')):
                    def edge_group(r):
                        e = errors(r, 'midpoint')['e'][j]
                        return 'near3' if abs(e) <= .03 else 'undersized' if e < 0 else 'oversized'
                    cells[name]['axis_support'][edge] = dict(
                        negative=sum(errors(r, 'midpoint')['e'][j] < 0 for r in present_rows),
                        positive=sum(errors(r, 'midpoint')['e'][j] > 0 for r in present_rows),
                        groups={g:cell_summary([r for r in present_rows if edge_group(r) == g])
                                for g in ('undersized', 'oversized', 'near3')})
                    if split == 'train':
                        counts = {g:cells[name]['axis_support'][edge]['groups'][g]['frames']
                                  for g in ('undersized', 'oversized', 'near3')}
                        cells[name]['axis_support'][edge]['proposed_train_only_weights_not_applied'] = axis_balance_weights(counts)
                cells[name]['videos'] = dict(Counter(r['sequence'] for r in cell))
                if domain != 'sim':
                    continue
                observed = [r for r in cell if r['midpoint'] is not None]
                strata[name] = {}
                for label in ('low/low', 'low/high', 'high/low', 'high/high'):
                    sub = [r for r in observed if stratum(r, fitted) == label]
                    strata[name][label] = cell_summary(sub)
                    memberships[name+'/'+label] = [r['image'] for r in sub]
                xs = [features(r) for r in observed]
                cells[name]['geometry_range_support'] = dict(
                    train_fitted_ranges=fitted['ranges'],
                    outside_each_train_marginal=[sum(x[j] < lo or x[j] > hi for x in xs) for j, (lo, hi) in enumerate(fitted['ranges'])],
                    outside_any_train_marginal=sum(any(x[j] < lo or x[j] > hi for j, (lo, hi) in enumerate(fitted['ranges'])) for x in xs),
                    note='Marginal geometric support only; no visual/ROI information or proof of joint support.')
                for r in cell:
                    for before, after in PAIRS:
                        d = frame_delta(r, before, after)
                        d['stratum'] = stratum(r, fitted) if r['midpoint'] is not None else 'missing'
                        deltas.append(d)
    epochs = []
    for epoch in range(1, 25):
        for split in ('train', 'val'):
            rs = checked('candidate/'+split+'_epoch_%02d.rows.jsonl.gz'%epoch, True)
            verify_rows(rs, split, refs[split])
            if split == 'val':
                verify_summary(rs, checked('candidate/val_epoch_%02d.summary.json'%epoch))
            for scale in ((1., .5) if split == 'train' else (1.,)):
                cell = [r for r in rs if r['domain'] == 'sim' and r['scale'] == scale]
                x = dict(epoch=epoch, split=split, scale=scale, frames=len(cell))
                for m in ('midpoint', 'edge_residual'):
                    summary = metrics.size_summary(cell, m)
                    for e in ('long', 'short'):
                        x[m+'_'+e+'_mae'] = summary['edges'][e]['mae']
                    x[m+'_a'] = summary['a']['abs_mae']; x[m+'_r'] = summary['r']['abs_mae']
                for g in GROUPS[:-1]:
                    sub = [r for r in cell if metrics.group(r['gt'], r['midpoint']) == g]
                    a, b = (metrics.size_summary(sub, m) for m in ('midpoint', 'edge_residual'))
                    x[g+'_n'] = len(sub)
                    x[g+'_a_delta'] = b['a']['abs_mae']-a['a']['abs_mae'] if sub else None
                    x[g+'_short_mae_delta'] = b['edges']['short']['mae']-a['edges']['short']['mae'] if sub else None
                epochs.append(x)
    # Snapshot train losses are never substituted for recorded training gradients.
    result = dict(version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
        fixed_identity=dict(candidate='I07', head_sha256='affe4795b73cae325f878b15c2d81f942461bd965bae0b3c3ca6a7a944eb0464',
                            s019_source_commit='57427341162a9434c8d65419d5b1175ecdd6052b', selection_changed=False),
        strata_definition=fitted, cells=cells, strata=strata, exposure_all24=dict(exposure),
        epoch_pattern_counts={split+'/'+str(scale):dict(
            all_four_a_worse=sum(all(x[g+'_a_delta'] > EPS for g in GROUPS[:-1]) for x in epochs if x['split']==split and x['scale']==scale),
            all_four_a_better=sum(all(x[g+'_a_delta'] < -EPS for g in GROUPS[:-1]) for x in epochs if x['split']==split and x['scale']==scale),
            near_a_worse=sum(x['near_correct_a_delta'] > EPS for x in epochs if x['split']==split and x['scale']==scale))
            for split, scale in (('train', 1.), ('train', .5), ('val', 1.))},
        selected_detection=val_summary,
        definitions=dict(case_up='error reduction for edge/ar cases; numerical edge increase for update cases',
            near='Fixed M both abs relative <=3%; distinct from size10', eps=EPS,
            stratum='low<=TRAIN median, high>TRAIN median; original-image size, no actual ROI scale inferred',
            loss='float64 recomputation of saved decoded SmoothL1 beta0.1 snapshots, averaged over axes; no gradients or optimizer replay'),
        limits=dict(training_updates=0, model_loads=0, feature_extractions=0, server_connections=0,
                    test_access=False, depth_access=False, val_reselection=False,
                    independent_sim_train_videos=1, independent_sim_val_videos=1,
                    visual_separability_tested=False, causal_root_determined=False),
        source=dict(run_directory=str(run_dir), artifacts_sha256=sha(run_dir/'artifacts.json'),
            checked_files=used, script_sha256=sha(Path(__file__)),
            reporting_source_sha256=sha(Path(metrics.__file__)),
            old_balance_source_sha256=sha(ROOT/'crane_project/utils/port_geometry_size_direction_balance_v1.py')))
    output.mkdir(parents=True)
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (output/'analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    (output/'stratum_membership.json').write_text(json.dumps(memberships, indent=2)+'\n')
    write_csv(output/'sim_frame_deltas.csv', deltas)
    write_csv(output/'all24_sim_patterns.csv', epochs)
    receipt = dict(version=VERSION, files={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
                   source_files_verified=len(used), limits=result['limits'])
    (output/'artifacts.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'work_dirs/port_geometry_size_condition_v1/condition_20261017_v1_run')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run_dir, args.out)
    print(json.dumps(dict(status='SAVED_OUTPUT_DIAGNOSIS_COMPLETE', checked_files=len(result['source']['checked_files']),
                          epoch_patterns=result['epoch_pattern_counts'], output=str(args.out)), indent=2))


if __name__ == '__main__':
    main()
