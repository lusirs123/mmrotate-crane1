#!/usr/bin/env python3
"""Independent feature/state/count/pair-order audit of the cached TRAIN report."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_reliability_sparse_candidate_v1 as runner


def close(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, abs_tol=2e-13, rel_tol=2e-13)
    return a == b


def review(result):
    p, pins = runner.load()
    report = json.loads((result / 'report.json').read_text())
    saved = json.loads((result / 'diagnostic_rows.json').read_text())
    if report['input_pins'] != pins or report['cutoffs'] != p['cutoffs'] or report['rows'] != 2558 or len(saved) != 2558:
        raise ValueError('Contract/row coverage differs')
    for key in ('fits', 'calibration_calls', 'detector_inferences', 'detector_updates', 'quality_updates', 'VAL_opened', 'TEST_opened', 'formal_policy_changed', 'candidate_adopted'):
        if report[key]:
            raise ValueError('Scope exceeded: ' + key)
    candidates = runner.read(ROOT / p['candidate_result'] / 'candidates_TRAIN.jsonl.gz')
    reference = {r['image']: r for r in runner.read(ROOT / p['reference_result'] / 'scored_TRAIN.jsonl.gz')}
    checks = 0
    for row, diagnostic in zip(candidates, saved):
        old = reference[row['image']]
        if row['image'] != diagnostic['image'] or row['pred'] != old['pred'] or row['gt'] != old['gt']:
            raise ValueError('Join/geometry differs')
        pred = row['pred']
        neighbors = row['candidate_evidence']['neighbors']
        if pred is None:
            y = None
        else:
            a, b = sorted(pred[2:4]), sorted(row['gt'][2:4])
            y = max(abs(a[0] - b[0]) / b[0], abs(a[1] - b[1]) / b[1]) > .1
        vals = {}
        for i in (1, 2):
            a = sorted(pred[2:4], reverse=True) if pred else None
            b = sorted(neighbors[i - 1]['box'][2:4], reverse=True) if len(neighbors) >= i else None
            for j, axis in enumerate(('L', 'S')):
                vals['neighbor%d_log_%s' % (i, axis)] = math.log(b[j] / a[j]) if b else None
        history = row['temporal']['available']
        risk = old['risks']['crossfit_fusion'] if history else old['risks']['partial_auc4']
        decisions = dict(fallback_decision=pred is not None and risk <= p['cutoffs']['supported' if history else 'fallback'],
            partial_auc4=pred is not None and old['risks']['partial_auc4'] <= p['cutoffs']['fallback'],
            full_simple=pred is not None and old['original_simple_decision']['size_accepted'])
        states = {m: 'MISSING' if y is None else ('CR' if y is False else 'FA') if accept else ('FR' if y is False else 'ED') for m, accept in decisions.items()}
        agreement = []
        for axis in ('L', 'S'):
            a, b = vals['neighbor1_log_' + axis], vals['neighbor2_log_' + axis]
            agreement.append('missing' if pred is None else 'none' if a is None else 'single' if b is None else 'both_positive' if min(a, b) > 0 else 'both_negative' if max(a, b) < 0 else 'mixed_or_zero')
        expected = dict(image=row['image'], sequence=row['sequence'], domain=row['domain'],
            history_available=history, temporal_primary=row['temporal']['primary'], neighbors=len(neighbors),
            features=vals, agreement=agreement, wrong=y, states=states,
            risks=dict(fallback_decision=risk, partial_auc4=old['risks']['partial_auc4'], full_simple=old['risks']['full_simple']),
            neighbor_source_indices=[n['index'] for n in neighbors[:2]])
        if not close(expected, diagnostic) or old['fallback_decision'] != decisions['fallback_decision']:
            raise ValueError('Independent row/state/feature/decision differs')
        checks += 1
    groups = dict(all=saved)
    for key in ('domain', 'sequence'):
        for name in {r[key] for r in saved}:
            groups[key + ':' + name] = [r for r in saved if r[key] == name]
    if set(report['result']['groups']) != set(groups):
        raise ValueError('Group coverage differs')
    masks = dict(all=lambda n, h: True, no_neighbor=lambda n, h: n == 0,
        one=lambda n, h: n == 1, two=lambda n, h: n == 2, three_plus=lambda n, h: n >= 3,
        any=lambda n, h: n >= 1, any_history=lambda n, h: n >= 1 and h)
    for group, part in groups.items():
        for method, counts in report['result']['groups'][group].items():
            for mask, entry in counts.items():
                c = Counter(r['states'][method] for r in part if masks[mask](r['neighbors'], r['history_available']))
                if entry != {s: c[s] for s in runner.core.STATES}:
                    raise ValueError('Independent coverage/state count differs')
                checks += 1
        for method in ('fallback_decision', 'partial_auc4'):
            signal = report['result']['signals'][method][group]
            accepted = [r for r in part if r['neighbors'] and r['states'][method] in ('FA', 'CR')]
            bounds = np.quantile([r['risks'][method] for r in accepted], [.25, .5, .75]).tolist() if accepted else [None] * 3
            if not close(bounds, signal['risk_quartiles']):
                raise ValueError('Risk strata changed')
            for feature, stats in signal['features'].items():
                valid = [r for r in accepted if r['features'][feature] is not None]
                arrays = {s: np.array([r['features'][feature] for r in valid if r['states'][method] == s]) for s in ('FA', 'CR')}
                for state, a in arrays.items():
                    qs = np.quantile(a, [.1, .25, .5, .75, .9]).tolist() if len(a) else [None] * 5
                    expected = dict(n=len(a), mean=float(a.mean()) if len(a) else None, **dict(zip(('q10', 'q25', 'median', 'q75', 'q90'), qs)))
                    if not close(expected, stats[state]):
                        raise ValueError('Distribution differs')
                a, b = arrays['FA'], arrays['CR']
                wins = float(np.sum((a[:, None] > b[None, :]) + .5 * (a[:, None] == b[None, :])))
                n = len(a) * len(b)
                pooled = dict(wrong=len(a), good=len(b), pairs=n, wins=wins,
                    AUC_higher=wins / n if n else None, AUC_lower=1 - wins / n if n else None)
                if not close(pooled, stats['pooled']):
                    raise ValueError('Independent AUC differs')
                strata = []
                for seq in sorted({r['sequence'] for r in valid}):
                    for k in range(4):
                        subset = [r for r in valid if r['sequence'] == seq and sum(r['risks'][method] > t for t in bounds) == k]
                        if not subset:
                            continue
                        a = np.array([r['features'][feature] for r in subset if r['states'][method] == 'FA'])
                        b = np.array([r['features'][feature] for r in subset if r['states'][method] == 'CR'])
                        n = len(a) * len(b)
                        w = float(np.sum((a[:, None] > b[None, :]) + .5 * (a[:, None] == b[None, :])))
                        strata.append(dict(sequence=seq, risk_quartile=k, wrong=len(a), good=len(b), pairs=n, wins=w,
                            AUC_higher=w / n if n else None, AUC_lower=1 - w / n if n else None))
                n = sum(s['pairs'] for s in strata)
                w = sum(s['wins'] for s in strata)
                if not close(strata, stats['same_video_risk_strata']) or stats['conditional_pairs'] != n or not close(stats['conditional_AUC_higher'], w / n if n else None) or not close(stats['conditional_AUC_lower'], 1 - w / n if n else None):
                    raise ValueError('Conditional pair ordering differs')
                checks += 1
            for i, axis in enumerate(('L', 'S')):
                for name, counts in signal['agreement'][axis].items():
                    c = Counter(r['states'][method] for r in part if r['agreement'][i] == name)
                    if counts != {s: c[s] for s in runner.core.STATES}:
                        raise ValueError('Sign agreement count differs')
                    checks += 1
    expected_fa = [r for r in saved if r['states']['fallback_decision'] == 'FA']
    if report['result']['remaining_FA'] != expected_fa or report['result']['status'] != 'DESCRIPTIVE_TRAIN_COMPLETE_NO_TRAINING_AUTHORIZED':
        raise ValueError('FA listing/status differs')
    return dict(passed=True, rows=len(saved), independent_checks=checks, report_sha256=runner.sha(result / 'report.json'),
        diagnostic_rows_sha256=runner.sha(result / 'diagnostic_rows.json'), no_fit=True, no_VAL_TEST=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    runner.write(args.out, review(args.result))
    print('Independent sparse-candidate TRAIN review passed')
