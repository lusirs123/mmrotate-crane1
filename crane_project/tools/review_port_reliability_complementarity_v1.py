#!/usr/bin/env python3
"""Independent scalar reconstruction; no models, fitting, features or TEST."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import audit_port_reliability_complementarity_v1 as run


def wrong(row):
    if row['pred'] is None:
        return None
    a, b = sorted(row['pred'][2:4]), sorted(row['gt'][2:4])
    return any(abs(x-y)/y > .1 for x, y in zip(a, b))


def name(row, keep):
    b = wrong(row)
    if b is None:
        assert not keep
        return 'MISSING'
    return ('FA' if b else 'CR') if keep else ('ED' if b else 'FR')


def count(rows, accepted):
    counts = Counter(dict(FA=0, FR=0, ED=0, CR=0, MISSING=0))
    longest = consecutive = 0; previous = None
    for row in sorted(rows, key=lambda r: (r['sequence'],r['frame_id'],r['image'])):
        s = name(row, row['image'] in accepted); counts[s] += 1
        if previous is None or previous != (row['sequence'],row['frame_id']-1):
            consecutive = 0
        consecutive = consecutive+1 if s == 'FR' else 0
        longest = max(longest, consecutive); previous = row['sequence'],row['frame_id']
    return dict(counts), longest


def verify_summary(rows, kept, value):
    states, longest = count(rows, kept)
    assert value['states'] == states
    assert value['runs']['correct_rejection']['longest'] == longest
    assert value['frames'] == len(rows)
    outputs = len(rows)-states['MISSING']; n = states['FA']+states['CR']; good = states['CR']+states['FR']
    assert value['outputs'] == outputs and value['accepted_outputs'] == n
    ratio = lambda x, y: x/y if y else None
    for key, expected in dict(output_coverage=ratio(outputs,len(rows)),
        acceptance_coverage_all_frames=ratio(n,len(rows)),correct_coverage_all_frames=ratio(states['CR'],len(rows)),
        good_retention=ratio(states['CR'],good),accepted_error_fraction=ratio(states['FA'],n)).items():
        assert value[key] == expected


def pair_auc(rows, method):
    bad = [r['risks'][method] for r in rows if wrong(r) is True]
    good = [r['risks'][method] for r in rows if wrong(r) is False]
    if not bad or not good:
        return None
    return sum(1. if a > b else .5 if a == b else 0. for a in bad for b in good)/(len(bad)*len(good))


def audit(directory):
    directory = Path(directory)
    report = json.loads((directory/'report.json').read_text())
    receipt = json.loads((directory/'completion.json').read_text())
    contract, manifest = run.checked()
    assert report['contract'] == contract and report['sources'] == manifest and report['input_pins'] == contract['input_pins']
    for file, pin in receipt['artifacts'].items():
        assert run.sha(directory/file) == pin
    commit = report['git_commit']
    for file, pin in manifest['sources'].items():
        assert run.hashlib.sha256(subprocess.check_output(['git','show',commit+':'+file],cwd=str(ROOT))).hexdigest() == pin
    assert not report['TEST_read'] and not report['GT_online'] and not report['original_policy_changed']
    assert report['training_updates_added'] == report['detector_inferences_added'] == report['thresholds_fitted'] == report['feature_forwards_added'] == 0
    assert report['selected_method'] is None and not report['candidate_adopted'] and not report['new_data_roles']
    parts, cutoffs, _ = run.load(contract)
    assert cutoffs == report['cutoffs']
    groups_checked = summaries_checked = auc_checked = 0
    for role, original in parts.items():
        rows = run.jsonl(directory/('joined_'+role+'.jsonl.gz'))
        assert rows == original
        for reference in ('formal_simple','simple_retention95'):
            unions = {}
            for method in contract['methods']:
                results = report['results'][role]['references'][reference]['methods'][method]
                for group_name, group in run.core.grouped(rows).items():
                    value = results[group_name]; groups_checked += 1
                    baseline = {r['image'] for r in group if r['pred'] is not None and r['risks']['full_simple'] <= cutoffs[reference]}
                    candidate = {r['image'] for r in group if r['pred'] is not None and r['risks'][method] <= cutoffs[method]}
                    overlay = baseline & candidate
                    for kept, key in ((baseline,'baseline'),(candidate,'candidate'),(overlay,'reject_only_AND')):
                        verify_summary(group, kept, value[key]); summaries_checked += 1
                    table = {a:dict.fromkeys(('FA','FR','ED','CR','MISSING'),0) for a in ('FA','FR','ED','CR','MISSING')}
                    changed = dict(rescued_FA=[],new_FR=[],restored_CR=[],new_FA=[])
                    for r in group:
                        s, t = name(r,r['image'] in baseline),name(r,r['image'] in candidate)
                        table[s][t] += 1
                        for key, pair in dict(rescued_FA=('FA','ED'),new_FR=('CR','FR'),restored_CR=('FR','CR'),new_FA=('ED','FA')).items():
                            if (s,t) == pair: changed[key].append(r['image'])
                    assert value['state_transition'] == table
                    assert value['change_images'] == {k:sorted(v) for k,v in changed.items()}
                    assert value['changes'] == {k:len(v) for k,v in changed.items()}
                    unions.setdefault(group_name,dict(rescued_FA=set(),restored_CR=set()))
                    for key in unions[group_name]: unions[group_name][key].update(changed[key])
                    oracle = value['GT_oracle_upper_bound']
                    assert oracle['remaining_FA'] == value['baseline']['states']['FA']-len(changed['rescued_FA'])
                    assert oracle['remaining_FR'] == value['baseline']['states']['FR']-len(changed['restored_CR'])
                    assert not oracle['executable_policy'] and oracle['GT_selects_changes']
                    conditional = [r for r in group if r['image'] in baseline]
                    diag = value['conditional_on_simple_accept']
                    assert diag['frames'] == len(conditional) and diag['bad'] == sum(wrong(r) is True for r in conditional)
                    assert diag['good'] == sum(wrong(r) is False for r in conditional)
                    for score in (method,'full_simple','score_only'):
                        expected = pair_auc(conditional,score); actual = diag['error_AUROC'][score]
                        assert actual is expected if expected is None else abs(actual-expected)<1e-12
                        auc_checked += 1
                    for kept, key in ((candidate,'candidate_comparison'),(overlay,'AND_comparison')):
                        states, _ = count(group,kept); target = states['CR']
                        n = states['CR']+states['FA']
                        for control in ('full_simple','score_only'):
                            ordered = sorted((r for r in group if r['pred'] is not None),key=lambda r:(r['risks'][control],r['image']))
                            comparison = value[key][control]
                            verify_summary(group,{r['image'] for r in ordered[:n]},comparison['same_count']); summaries_checked += 1
                            if n:
                                boundary = ordered[n-1]['risks'][control]
                                below = [r for r in ordered if r['risks'][control]<boundary]
                                tie = [r for r in ordered if r['risks'][control]==boundary]
                                slots = n-len(below); bw = sum(wrong(r) for r in below)
                                assert comparison['same_count_tie_bounds'] == dict(bad_min=bw+max(0,slots-sum(not wrong(r) for r in tie)),
                                    bad_max=bw+min(slots,sum(wrong(r) for r in tie)),partial_boundary_tie=slots<len(tie),tie_count=len(tie),selection='image_order_not_GT')
                            good = 0; threshold = None
                            if target:
                                for row in ordered:
                                    good += int(not wrong(row))
                                    if good == target:
                                        threshold = row['risks'][control]; break
                            matched = {r['image'] for r in ordered if threshold is not None and r['risks'][control]<=threshold}
                            verify_summary(group,matched,comparison['same_CR']); summaries_checked += 1
                            assert comparison['same_CR_offline_risk_le'] == threshold
                            assert comparison['same_CR_exact'] == (count(group,matched)[0]['CR'] == target)
                            assert comparison['offline_GT_comparison_only']
            for group_name, union in unions.items():
                saved = report['results'][role]['references'][reference]['GT_oracle_union'][group_name]
                assert saved['rescued_FA'] == len(union['rescued_FA']) and saved['restored_CR'] == len(union['restored_CR'])
                assert saved['rescued_images'] == sorted(union['rescued_FA']) and saved['restored_images'] == sorted(union['restored_CR'])
                assert not saved['executable_policy'] and saved['methods_not_selected']
        for group_name, group in run.core.grouped(rows).items():
            out = [r for r in group if r['pred'] is not None]
            hits = sum((r['pred'][0]-r['gt'][0])**2+(r['pred'][1]-r['gt'][1])**2 < 225 for r in out)
            expected = dict(frames=len(group),outputs=len(out),hits=hits,hit_rate_on_outputs=hits/len(out) if out else None,
                            output_coverage=len(out)/len(group),correct_coverage_all_frames=hits/len(group))
            assert report['results'][role]['center'][group_name] == expected
    # Gate is a necessary, descriptive condition, never retroactive model adoption.
    for method in contract['methods']:
        recalculated = run.core.gate(report['results']['VAL']['references']['simple_retention95']['methods'][method])
        assert recalculated == report['evidence_gates'][method]
    expected_status = 'COMPLEMENTARITY_SIGNAL_REQUIRES_NEW_DESIGN' if any(g['passed'] for g in report['evidence_gates'].values()) else 'NO_STABLE_FROZEN_OVERLAY_KEEP_SIMPLE'
    assert report['status'] == expected_status
    assert (directory/'analysis.md').read_text() == run.markdown(report)
    run.checked()
    return dict(passed=True,rows_checked=sum(map(len,parts.values())),groups_checked=groups_checked,
                summaries_checked=summaries_checked,conditional_AUROC_checked=auc_checked,
                report_sha256=run.sha(directory/'report.json'),status=report['status'],TEST_read=False,
                weights_loaded=False,independent_scalar_counts_and_pair_AUROC=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory'); parser.add_argument('--receipt',required=True)
    args = parser.parse_args()
    value = audit(args.directory)
    run.write(args.receipt,value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
