#!/usr/bin/env python3
"""Experiment C: frozen size-error severity, no fitting or online policy export."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_reliability_video_transfer_v1 as metrics
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.tools import review_port_reliability_video_transfer_v1 as independent
VERSION = 'port_reliability_severity_v1'
SEVERITIES = ('mild', 'moderate', 'severe')
TYPES = ('long_only', 'short_only', 'both')
SETTINGS = dict(severity_upper_bounds=[.10, .15, .25], correct_retention=.95,
    methods=['simple', 'score'], error='max(abs(pred_axis-GT_axis)/GT_axis)',
    quantile_method='linear', diagnostic_only=True, fits=0, threshold_updates=0,
    TEST_opened=False, sparse_error_warning_below=10)
EXPECTED = {'TRAIN': dict(FA=13, FR=1037, ED=58, CR=1450, MISSING=0),
            'VAL': dict(FA=190, FR=10, ED=33, CR=653, MISSING=1)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  indent=2, allow_nan=False)+'\n')


def describe(row):
    """GT is used only here and in offline evaluation; no candidate score learned."""
    if row['pred'] is None:
        return dict(severity='missing', error_type='missing', sign='missing',
                    d_long=None, d_short=None, error=None, state='MISSING')
    p = sorted(row['pred'][2:4], reverse=True)
    g = sorted(row['gt'][2:4], reverse=True)
    if any(not math.isfinite(v) or v <= 0 for v in p+g):
        raise ValueError('Nonpositive/nonfinite size')
    d = [(a-b)/b for a,b in zip(p,g)]
    e = max(abs(v) for v in d)
    severity = 'correct' if e <= .1 else 'mild' if e <= .15 else 'moderate' if e <= .25 else 'severe'
    bad_l, bad_s = abs(d[0]) > .1, abs(d[1]) > .1
    kind = 'both' if bad_l and bad_s else 'long_only' if bad_l else 'short_only' if bad_s else 'correct'
    signs = ['small' if value < 0 else 'large' for value in d]
    sign = '_'.join(signs) if kind == 'both' else signs[0 if bad_l else 1] if kind != 'correct' else 'correct'
    accepted = row['original_simple_decision']['size_accepted']
    state = ('FA' if e > .1 else 'CR') if accepted else ('ED' if e > .1 else 'FR')
    return dict(severity=severity, error_type=kind, sign=sign,
                d_long=d[0], d_short=d[1], error=e, state=state)


def stats(values):
    if not values:
        return dict(n=0, median=None, p95=None, maximum=None)
    return dict(n=len(values), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), maximum=float(max(values)))


def breakdown(derived, accepted):
    """Mutually exclusive error partitions; missing/correct frames excluded."""
    errors = [r for r in derived if r['severity'] in SEVERITIES]
    total_fa = sum(r['image'] in accepted for r in errors)
    def part(rows):
        fa = [r for r in rows if r['image'] in accepted]
        n = len(rows)
        return dict(errors=n, FA=len(fa), ED=n-len(fa),
            detection_rate=(n-len(fa))/n if n else None,
            share_all_FA=len(fa)/total_fa if total_fa else None,
            FA_error=stats([r['error'] for r in fa]),
            FA_long_bias=float(np.mean([r['d_long'] for r in fa])) if fa else None,
            FA_short_bias=float(np.mean([r['d_short'] for r in fa])) if fa else None)
    return dict(all=part(errors),
        severity={s: part([r for r in errors if r['severity']==s]) for s in SEVERITIES},
        error_type={t: part([r for r in errors if r['error_type']==t]) for t in TYPES},
        severity_type={s: {t: part([r for r in errors if r['severity']==s and r['error_type']==t]) for t in TYPES} for s in SEVERITIES},
        type_sign={t: {sign: part([r for r in errors if r['error_type']==t and r['sign']==sign]) for sign in (('small_small','large_large','small_large','large_small') if t=='both' else ('small','large'))} for t in TYPES})


def aucs(rows, derived):
    by_id = {r['image']:r for r in derived}
    good = [r for r in rows if by_id[r['image']]['severity']=='correct']
    result = {}
    for label in ('all',)+SEVERITIES+TYPES:
        bad = [r for r in rows if by_id[r['image']]['severity'] in SEVERITIES and
               (label=='all' or by_id[r['image']]['severity']==label or by_id[r['image']]['error_type']==label)]
        values = {}
        for method in SETTINGS['methods']:
            # Rank implementation, independently checked by scalar pair counts below.
            eligible = good+bad
            values[method] = metrics.fusion.level.metrics.auc(eligible, method)
        result[label] = dict(correct=len(good), wrong=len(bad), AUROC=values,
                            sparse_wrong=len(bad)<10, other_errors_excluded=True)
    return result


def chosen_for_point(rows, method, point, count=None):
    ordered = sorted((r for r in rows if r['pred'] is not None), key=lambda r:(r['risks'][method],r['image']))
    return {r['image'] for r in (ordered[:count] if count is not None else
            [r for r in ordered if point['risk_le'] is not None and r['risks'][method]<=point['risk_le']])}


def analyze(rows):
    original = deepcopy(rows)
    derived = [dict(image=r['image'], sequence=r['sequence'], domain=r['domain'], **describe(r)) for r in rows]
    groups = {}
    lookup = {r['image']:r for r in derived}
    for name, part in metrics.fusion.level.metrics.grouped(rows).items():
        dd = [lookup[r['image']] for r in part]
        accepted = {r['image'] for r in part if r['original_simple_decision']['size_accepted']}
        delivery = metrics.counts(part, accepted)
        targets = dict(same_CR=delivery['states']['CR'], correct95=math.ceil(.95*(delivery['states']['CR']+delivery['states']['FR'])), same_count=delivery['accepted'])
        points = {}
        for target, n in targets.items():
            points[target] = {}
            for method in SETTINGS['methods']:
                point = metrics.point(part, method, **({'count':n} if target=='same_count' else {'correct':n}))
                chosen = chosen_for_point(part, method, point, n if target=='same_count' else None)
                points[target][method] = dict(point=point, required=n,
                    errors=breakdown(dd,chosen), accepted_images=sorted(chosen))
        outputs = [r for r in part if r['pred'] is not None]
        hits = sum(math.hypot(r['pred'][0]-r['gt'][0],r['pred'][1]-r['gt'][1])<15 for r in outputs)
        groups[name] = dict(formal=delivery, errors=breakdown(dd,accepted), ranking=aucs(part,dd), offline_points=points,
            center=dict(correct_outputs=hits, outputs=len(outputs), frames=len(part),
                hit_rate_outputs=hits/len(outputs) if outputs else None,
                output_coverage=len(outputs)/len(part), full_frame_correct_coverage=hits/len(part)),
            original_flag_counts={k:sum(r['original_simple_decision'][k] for r in part) for k in ('center_accepted','size_accepted','angle_accepted')})
    if rows != original:
        raise ValueError('Source modified')
    return dict(frames=len(rows), groups=groups), derived


def review(rows, result, derived):
    """Independent scalar errors, pair AUC and selected-set/state reconciliation."""
    checks = 0
    for row, actual in zip(rows,derived):
        if row['image'] != actual['image']:
            raise ValueError('Frame identity differs')
        if row['pred'] is None:
            expected = ('missing','missing','MISSING')
        else:
            pw,ph = row['pred'][2:4]; gw,gh = row['gt'][2:4]
            dl = (max(pw,ph)-max(gw,gh))/max(gw,gh)
            ds = (min(pw,ph)-min(gw,gh))/min(gw,gh)
            e = max(abs(dl),abs(ds))
            level = ('correct','mild','moderate','severe')[sum(e>v for v in (.1,.15,.25))]
            kind = 'both' if abs(dl)>.1 and abs(ds)>.1 else 'long_only' if abs(dl)>.1 else 'short_only' if abs(ds)>.1 else 'correct'
            state = ('FA' if e>.1 else 'CR') if row['original_simple_decision']['size_accepted'] else ('ED' if e>.1 else 'FR')
            expected = (level,kind,state)
            if (dl,ds,e)!=(actual['d_long'],actual['d_short'],actual['error']):
                raise ValueError('Scalar error differs')
        if expected != (actual['severity'],actual['error_type'],actual['state']):
            raise ValueError('Scalar partition differs')
        checks += 1
    index = {r['image']:r for r in derived}
    for name, part in metrics.fusion.level.metrics.grouped(rows).items():
        group = result['groups'][name]
        kept = {r['image'] for r in part if r['original_simple_decision']['size_accepted']}
        independent.check_summary(group['formal'],part,kept)
        states = Counter(index[r['image']]['state'] for r in part)
        if any(states[s]!=group['formal']['states'][s] for s in metrics.STATES):
            raise ValueError('Partition states differ')
        for section in ('severity','error_type'):
            if sum(v['FA'] for v in group['errors'][section].values())!=states['FA'] or sum(v['ED'] for v in group['errors'][section].values())!=states['ED']:
                raise ValueError('Error counts fail to reconcile')
        for level, data in group['ranking'].items():
            good = [r for r in part if index[r['image']]['severity']=='correct']
            bad = [r for r in part if index[r['image']]['severity'] in SEVERITIES and (level=='all' or index[r['image']]['severity']==level or index[r['image']]['error_type']==level)]
            for method, value in data['AUROC'].items():
                expected_auc = sum((b['risks'][method]>g['risks'][method])+.5*(b['risks'][method]==g['risks'][method]) for b in bad for g in good)/(len(bad)*len(good)) if bad and good else None
                if (value is None)!=(expected_auc is None) or value is not None and abs(value-expected_auc)>1e-12:
                    raise ValueError('Pairwise AUC differs')
                checks += 1
        for target, methods in group['offline_points'].items():
            for method, record in methods.items():
                kwargs = {'count':record['required']} if target=='same_count' else {'correct':record['required']}
                independent.check_point(record['point'],part,method,**kwargs)
                chosen = set(record['accepted_images'])
                independent.check_summary(record['point'],part,chosen)
                for severity, values in record['errors']['severity'].items():
                    selected = [r for r in part if index[r['image']]['severity']==severity]
                    fa = sum(r['image'] in chosen for r in selected)
                    if (len(selected),fa,len(selected)-fa)!=(values['errors'],values['FA'],values['ED']):
                        raise ValueError('Offline error partition differs')
                checks += 1
    replay, dd = analyze(rows)
    if replay!=result or dd!=derived:
        raise ValueError('Summary replay differs')
    return dict(passed=True, rows=len(rows), independent_checks=checks, full_summary_replay=True,
                statistical_or_causal_guarantee=False)


def load(allow_absent=False):
    manifest = json.loads((ROOT/('crane_project/tools/'+VERSION+'_sources.json')).read_text())
    if manifest['settings'] != SETTINGS:
        raise ValueError('Settings changed')
    pins = dict(manifest['sources'], **manifest['input_pins'], **manifest['frontend_pins'])
    absent = []
    for name,digest in pins.items():
        if name in manifest['frontend_pins'] and allow_absent and not (ROOT/name).exists():
            absent.append(name)
        elif sha(ROOT/name)!=digest:
            raise ValueError('Source/input changed: '+name)
    wrapped = json.loads((ROOT/manifest['policy']).read_text()); policy = wrapped['simple_policy']
    api = simple.SimpleComponentReliability(policy)
    splits = {}
    for split, filename in manifest['rows'].items():
        with gzip.open(ROOT/filename,'rt') as stream:
            rows = [json.loads(line) for line in stream]
        n = 2558 if split=='TRAIN' else 887
        if len(rows)!=n or len({r['image'] for r in rows})!=n:
            raise ValueError('Incomplete or duplicate frames')
        for row in rows:
            legal = ('train','train_sim') if split=='TRAIN' else ('val',)
            if row['split'] not in legal or row['reliability_role'] not in legal:
                raise ValueError('Wrong data role')
            decision = api.decide(row['pred'],row['image_size'])
            decision['final_box_original'] = decision.pop('raw_b_output')
            saved = row['original_simple_decision']
            if any(decision[k]!=saved[k] for k in decision if k!='risks') or any((v is None)!=(saved['risks'][k] is None) or v is not None and abs(v-saved['risks'][k])>1e-12 for k,v in decision['risks'].items()):
                raise ValueError('Three-component API differs')
            if row['pred'] is not None:
                if row['pred']!=saved['final_box_original']:
                    raise ValueError('Delivered box identity differs')
                if abs(row['risks']['full_simple']-saved['risks']['size'])>1e-12 or abs(row['risks']['score_only']-(1-row['pred'][5]))>1e-12:
                    raise ValueError('Cached score differs')
            row['risks']['simple'] = row['risks']['full_simple']
            row['risks']['score'] = row['risks']['score_only']
        actual = metrics.counts(rows,{r['image'] for r in rows if r['original_simple_decision']['size_accepted']})['states']
        if actual != EXPECTED[split]:
            raise ValueError('Formal counts differ: '+str(actual))
        splits[split] = rows
    return manifest,pins,absent,splits


def run(args):
    manifest,pins,absent,splits = load(args.allow_absent_frontend)
    if args.check_only:
        print('Frozen TRAIN/VAL, three flags, policy, cache and sources verified; TEST unopened')
        return
    out = args.out
    if out is None or out.is_symlink() or out.name!='result' or out.resolve().parent.parent!=(ROOT/'work_dirs'/VERSION).resolve():
        raise ValueError('Use unified version/run/result')
    if args.review_only:
        report = json.loads((out/'report.json').read_text())
        if report['input_pins']!=pins or report['settings']!=SETTINGS:
            raise ValueError('Report identity differs')
    else:
        if absent:
            raise ValueError('Server execution requires formal frontend SHA verification')
        out.mkdir(parents=True,exist_ok=False)
        write(out/'frozen_settings.json',SETTINGS)
        report = dict(protocol=VERSION, settings=SETTINGS, input_pins=pins, results={},
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,universal_newlines=True).strip(),
            fits=0, detector_inferences=0, GPU_used=False, TEST_opened=False,
            GT_offline_only=True, policy_changed=False, flags_changed=False,
            historical_VAL_TEST_exposure_disclosed=True, candidate_adopted=False)
    receipts = {}
    for split, rows in splits.items():
        if args.review_only:
            with gzip.open(out/(split+'_errors.jsonl.gz'),'rt') as stream:
                derived = [json.loads(line) for line in stream]
            result = report['results'][split]
        else:
            result,derived = analyze(rows)
            report['results'][split] = result
            with gzip.open(out/(split+'_errors.jsonl.gz'),'wt') as stream:
                for row in derived:
                    stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        receipts[split] = review(rows,result,derived)
    for name,digest in pins.items():
        if name not in absent and sha(ROOT/name)!=digest:
            raise ValueError('Source/input changed during run')
    if not args.review_only:
        write(out/'report.json',report)
        with (out/'severity.csv').open('w') as stream:
            writer = csv.writer(stream);writer.writerow(['split','group','severity','errors','FA','ED','detection_rate','FA_median','FA_p95','FA_max'])
            for split,result in report['results'].items():
                for group,data in result['groups'].items():
                    for severity,d in data['errors']['severity'].items():
                        writer.writerow([split,group,severity,d['errors'],d['FA'],d['ED'],d['detection_rate'],d['FA_error']['median'],d['FA_error']['p95'],d['FA_error']['maximum']])
    write(out/args.receipt,dict(passed=True,splits=receipts,absent_frontend=absent,
        report_sha256=sha(out/'report.json'), derived_sha256={s:sha(out/(s+'_errors.jsonl.gz')) for s in splits}))
    for split,result in report['results'].items():
        print(split,json.dumps(result['groups']['all']['errors']['severity'],ensure_ascii=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path)
    parser.add_argument('--allow-absent-frontend',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--review-only',action='store_true')
    parser.add_argument('--receipt',default='server_review.json')
    run(parser.parse_args())
