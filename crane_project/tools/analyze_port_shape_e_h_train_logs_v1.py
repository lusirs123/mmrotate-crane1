#!/usr/bin/env python3
"""Read-only fixed B/E-H MMCV JSONL training-log review; standard library only.

No weights, GPU, inference, optimizer, VAL/TEST metrics or coefficient search.
Logged grad_norm is an averaged PRE-clip norm, not a per-step clipping trace.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
NAMES = {
    'b': 'crane_symeood_k1_port_day2night_aug_b_v1',
    'e_h': 'crane_symeood_k1_port_day2night_shape_e_h_v1',
}
SHAPE = dict(type='CovarianceShapeLoss', mode='hellinger', loss_weight=.05,
             eps=1e-6, reduction='mean')
CORE = ('loss_cls', 'loss_bbox', 'loss', 'grad_norm')
WINDOWS = {'epoch_1': (1, 1), 'epochs_2_4': (2, 4),
           'epochs_5_16': (5, 16), 'late_17_24': (17, 24),
           'late_17_22': (17, 22), 'late_23_24': (23, 24)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def config_value(node):
    """Accept literals and dict(...) only; never execute logged config text."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == 'dict' and not node.args:
            if any(k.arg is None for k in node.keywords):
                raise ValueError('dict expansion is unsupported')
            return {k.arg: config_value(k.value) for k in node.keywords}
    if isinstance(node, ast.Dict):
        return {config_value(k): config_value(v)
                for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple)):
        return [config_value(v) for v in node.elts]
    return ast.literal_eval(node)


def metadata_review(meta, arm):
    issues = []
    seed = meta.get('seed')
    if seed is None:
        issues.append('metadata_seed_missing')
    elif seed != 0:
        raise ValueError('Expected seed0 for ' + arm)
    exp = meta.get('exp_name')
    if not exp:
        issues.append('metadata_exp_name_missing')
    elif Path(exp).stem != NAMES[arm]:
        raise ValueError('Wrong experiment metadata: ' + str(exp))
    text = meta.get('config')
    resolved = {}
    if not isinstance(text, str):
        issues.append('resolved_training_config_missing')
    else:
        try:
            for node in ast.parse(text).body:
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id in
                        ('model', 'runner', 'optimizer_config', 'work_dir')):
                    resolved[node.targets[0].id] = config_value(node.value)
        except (ValueError, TypeError, SyntaxError) as error:
            issues.append('resolved_config_parse_unverified: ' + str(error))
            resolved = {}
    required = ('model', 'runner', 'optimizer_config', 'work_dir')
    if not all(k in resolved for k in required):
        issues.append('resolved_config_required_fields_missing')
    if resolved:
        head = resolved.get('model', {}).get('bbox_head', {})
        if not head or 'runner' not in resolved or 'optimizer_config' not in resolved:
            if 'resolved_config_required_fields_missing' not in issues:
                issues.append('resolved_config_required_fields_missing')
        else:
            if head.get('center_size_compensation') is not None:
                raise ValueError('D compensation is not this experiment')
            if head.get('shape_compensation') != (SHAPE if arm == 'e_h' else None):
                raise ValueError('Unexpected shape design in training metadata')
            if head.get('loss_bbox', {}).get('type') != 'SymKLDLoss':
                raise ValueError('Main bbox loss is not SymKLDLoss')
            if head['loss_bbox'].get('loss_weight') != 2.:
                raise ValueError('Expected original KLD weight2')
            if resolved['runner'].get('max_epochs') != 24:
                raise ValueError('Expected original 24 epoch budget')
            if resolved['optimizer_config'].get('grad_clip') != dict(max_norm=10, norm_type=2):
                raise ValueError('Expected original norm2/clip10')
        if resolved.get('work_dir') and Path(resolved['work_dir']).name != NAMES[arm]:
            raise ValueError('Wrong work_dir in training metadata')
    return dict(seed=seed, exp_name=exp, config_text_sha256=(
        hashlib.sha256(text.encode()).hexdigest() if isinstance(text, str) else None),
        checked_config_summary=resolved,
        identity_review_required=bool(issues), issues=issues)


def read_logs(paths, arm):
    rows, sources, seen, events = [], [], set(), []
    for path in paths:
        path = Path(path).resolve()
        meta, skipped = {}, Counter()
        local = []
        previous = None
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError('{}:{} invalid JSON: {}'.format(path, line_no, error))
            if not isinstance(raw, dict):
                raise ValueError('{}:{} expected JSON object'.format(path, line_no))
            if raw.get('mode') != 'train':
                if 'config' in raw or 'exp_name' in raw or 'seed' in raw:
                    meta.update(raw)
                skipped[str(raw.get('mode', 'metadata'))] += 1
                continue
            epoch, iteration = raw.get('epoch'), raw.get('iter')
            if (not isinstance(epoch, int) or isinstance(epoch, bool) or not 1 <= epoch <= 24
                    or not isinstance(iteration, int) or isinstance(iteration, bool) or iteration <= 0):
                raise ValueError('{}:{} invalid epoch/iteration'.format(path, line_no))
            key = (epoch, iteration)
            if previous is not None and key <= previous:
                raise ValueError('Non-increasing training rows in ' + str(path))
            if key in seen:
                raise ValueError('Overlapping logs at {}: preserve run boundaries'.format(key))
            previous = key
            seen.add(key)
            values = {}
            for name, value in raw.items():
                if name in ('lr', 'grad_norm', 'shape_positive_count') or 'loss' in name:
                    values[name] = float(value) if finite(value) else None
                    if not finite(value):
                        events.append(dict(source=str(path), line=line_no,
                                           epoch=epoch, iter=iteration,
                                           field=name, value=repr(value)))
            rows.append(dict(epoch=epoch, iter=iteration, source=str(path),
                             line=line_no, metrics=values))
            local.append(key)
        if not local:
            raise ValueError('No TRAIN records in ' + str(path))
        sources.append(dict(path=str(path), sha256=sha(path), train_records=len(local),
                            first=list(local[0]), last=list(local[-1]),
                            skipped_modes=dict(skipped), metadata=metadata_review(meta, arm)))
    rows.sort(key=lambda row: (row['epoch'], row['iter']))
    return rows, sources, events


def stats(values):
    good = sorted(value for value in values if finite(value))
    if not good:
        return dict(n=0, missing_or_nonfinite=len(values))
    position = (len(good) - 1) * .9
    k = int(position)
    p90 = good[k] + (good[min(k + 1, len(good) - 1)] - good[k]) * (position - k)
    return dict(n=len(good), missing_or_nonfinite=len(values)-len(good),
                mean=statistics.mean(good), median=statistics.median(good),
                p90=p90, min=good[0], max=good[-1])


def add_derived(rows):
    loss_keys = sorted({k for row in rows for k in row['metrics'] if 'loss' in k and k != 'loss'})
    cls_aux = [k for k in loss_keys if k != 'loss_cls' and k.endswith('_cls')]
    bbox_aux = [k for k in loss_keys if k != 'loss_bbox' and k.endswith('_bbox')]
    for row in rows:
        m = row['metrics']
        for name, keys in [('aux_classification_sum', cls_aux), ('aux_bbox_sum', bbox_aux)]:
            vals = [m.get(k) for k in keys]
            m[name] = sum(vals) if vals and all(finite(v) for v in vals) else None
        main, shape = m.get('loss_bbox'), m.get('loss_shape_compensation')
        m['logged_weighted_shape_to_main_kld_ratio'] = (
            shape/main if finite(shape) and finite(main) and main > 0 else None)
        vals = [m.get(k) for k in loss_keys]
        total = m.get('loss')
        m['logged_total_minus_component_sum'] = (
            total-sum(vals) if finite(total) and all(finite(v) for v in vals) else None)
    return dict(component_loss_keys=loss_keys, auxiliary_classification_keys=cls_aux,
                auxiliary_bbox_keys=bbox_aux)


def summarize(rows, arm):
    keys = sorted({k for row in rows for k in row['metrics']})
    metrics = {key: stats([row['metrics'].get(key) for row in rows]) for key in keys}
    required = list(CORE) + (['loss_shape_compensation'] if arm == 'e_h' else [])
    missing = {k: sum(not finite(row['metrics'].get(k)) for row in rows) for k in required}
    norms = [row['metrics']['grad_norm'] for row in rows if finite(row['metrics'].get('grad_norm'))]
    return dict(logged_records=len(rows), metrics=metrics, required_field_missing_or_nonfinite=missing,
                logged_windows_mean_preclip_norm_gt10=sum(v > 10 for v in norms),
                logged_windows_with_finite_grad_norm=len(norms),
                logged_windows_mean_preclip_norm_gt10_fraction=(
                    sum(v > 10 for v in norms)/len(norms) if norms else None),
                interpretation='Window-mean exceedance, NOT per-step clipping frequency or mean clip multiplier.')


def analyze(paths, arm):
    rows, sources, events = read_logs(paths, arm)
    fields = add_derived(rows)
    epochs = sorted({r['epoch'] for r in rows})
    per_epoch = {str(ep): summarize([r for r in rows if r['epoch'] == ep], arm) for ep in epochs}
    windows = {name: summarize([r for r in rows if first <= r['epoch'] <= last], arm)
               for name, (first, last) in WINDOWS.items()}
    missing_epochs = sorted(set(range(1, 25)) - set(epochs))
    required_gaps = summarize(rows, arm)['required_field_missing_or_nonfinite']
    zeros = {k: sum(r['metrics'].get(k) == 0 for r in rows)
             for k in ('loss_cls', 'loss_bbox', 'loss_shape_compensation')}
    return dict(sources=sources, **fields, epoch_coverage=epochs,
                missing_epochs=missing_epochs, nonfinite_or_nonnumeric_events=events,
                logged_zero_counts=zeros, per_epoch=per_epoch, windows=windows,
                first_records=rows[:5], last_records=rows[-5:], trajectory=rows,
                data_review_required=(bool(events or missing_epochs)
                    or any(required_gaps.values())
                    or any(s['metadata']['identity_review_required'] for s in sources)))


def discover(explicit, directory, required):
    if explicit:
        return [Path(p) for p in explicit]
    candidates = sorted(Path(directory).glob('*.log.json'))
    if len(candidates) > 1:
        raise ValueError('Multiple logs; specify --e-log or --b-log explicitly. '
                         'No latest-file selection or automatic merging:\n' +
                         '\n'.join(str(p) for p in candidates))
    if not candidates and required:
        raise FileNotFoundError('No .log.json found in ' + str(directory))
    return candidates


def comparison(arms):
    if 'b' not in arms:
        return dict(available=False, note='B log absent; no historical baseline trajectory inferred.')
    result = {}
    for name in WINDOWS:
        b, e = (arms[a]['windows'][name]['metrics'] for a in ('b', 'e_h'))
        pairs = {}
        for key in ('loss_cls', 'loss_bbox', 'aux_classification_sum', 'grad_norm'):
            bm, em = b.get(key, {}).get('median'), e.get(key, {}).get('median')
            pairs[key] = dict(b_median=bm, e_h_median=em,
                             e_h_to_b_median_ratio=em/bm if finite(em) and finite(bm) and bm > 0 else None)
        result[name] = pairs
    return dict(available=True, fixed_epoch_windows=result,
                note='Descriptive log-window comparison; batches and learned parameters differ.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e-work-dir', default=str(ROOT/'work_dirs'/NAMES['e_h']))
    parser.add_argument('--b-work-dir', default=str(ROOT/'work_dirs'/NAMES['b']))
    parser.add_argument('--e-log', nargs='+', help='Explicit log(s) from one E-H run; overlaps rejected')
    parser.add_argument('--b-log', nargs='+', help='Explicit log(s) from one B run; overlaps rejected')
    parser.add_argument('--no-b', action='store_true', help='Explicit E-only review')
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    out = Path(args.out_json).resolve()
    if out.exists():
        raise FileExistsError('Preserve existing report: ' + str(out))
    if args.no_b and args.b_log:
        parser.error('--no-b conflicts with --b-log')
    ep = discover(args.e_log, args.e_work_dir, True)
    bp = [] if args.no_b else discover(args.b_log, args.b_work_dir, False)
    arms = {'e_h': analyze(ep, 'e_h')}
    if bp:
        arms['b'] = analyze(bp, 'b')
    report = dict(protocol='port_shape_e_h_v1_train_log_review',
                  evidence_role='train_logs_only_no_weight_selection',
                  status='LOG_REVIEW_COMPLETE_RESULTS_REQUIRE_INTERPRETATION',
                  tool_sha256=sha(Path(__file__)), arms=arms, comparison=comparison(arms),
                  limitations=[
                      'Logged losses are weighted scalar window means; not gradient strength or size/angle decomposition.',
                      'grad_norm is the averaged pre-clip norm; exact per-step clip frequency/multiplier cannot be recovered.',
                      'MMCV rounding and log averaging can hide short spikes or tiny losses; rows are not independent replicates.',
                      'Missing epochs/fields are disclosed, never filled with zero; epoch coverage alone does not prove completion.',
                      'Original loss guards can replace nonfinite main losses with zero; finite logs do not prove guards never fired.',
                      'No checkpoint/VAL/TEST/data loading, training changes, coefficient search or root-cause verdict.',
                  ])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    for arm, data in arms.items():
        print(arm, 'epochs', data['epoch_coverage'], 'review_required', data['data_review_required'])
        print('window          records   cls_med   kld_med   shape_med  grad_med  grad_p90  shape/kld_med')
        for name, window in data['windows'].items():
            def value(key, stat='median'):
                v = window['metrics'].get(key, {}).get(stat)
                return '{:.6g}'.format(v) if finite(v) else 'NA'
            print(name, window['logged_records'], value('loss_cls'), value('loss_bbox'),
                  value('loss_shape_compensation'), value('grad_norm'), value('grad_norm', 'p90'),
                  value('logged_weighted_shape_to_main_kld_ratio'))
    print('B comparison available:', bool(bp))
    print('Saved', out)


if __name__ == '__main__':
    main()
