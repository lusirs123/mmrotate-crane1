#!/usr/bin/env python3
"""Independent scalar states, matched controls and NumPy head replay."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_simple_anchor_v1 as run


def wrong(r):
    if r['pred'] is None:
        return None
    return any(abs(p-g) > .1*g for p,g in zip(sorted(r['pred'][2:4]),sorted(r['gt'][2:4])))


def summary(rows, kept, expected):
    counts = Counter(dict(FA=0,FR=0,ED=0,CR=0,MISSING=0)); longest = streak = 0; previous = None
    for r in sorted(rows,key=lambda r:(r['sequence'],r['frame_id'],r['image'])):
        w = wrong(r)
        if w is None:
            assert r['image'] not in kept
            state = 'MISSING'
        else:
            state = ('FA' if w else 'CR') if r['image'] in kept else ('ED' if w else 'FR')
        counts[state] += 1
        if previous != (r['sequence'],r['frame_id']-1):
            streak = 0
        streak = streak+1 if state=='FR' else 0
        longest = max(longest,streak); previous = r['sequence'],r['frame_id']
    assert dict(counts) == expected['states']
    assert longest == expected['runs']['correct_rejection']['longest']
    n = counts['FA']+counts['CR']; outputs = len(rows)-counts['MISSING']; good = counts['CR']+counts['FR']
    assert expected['frames']==len(rows) and expected['outputs']==outputs and expected['accepted_outputs']==n
    ratio = lambda a,b:a/b if b else None
    for k,v in dict(output_coverage=ratio(outputs,len(rows)),acceptance_coverage_all_frames=ratio(n,len(rows)),
        correct_coverage_all_frames=ratio(counts['CR'],len(rows)),good_retention=ratio(counts['CR'],good),accepted_error_fraction=ratio(counts['FA'],n)).items():
        assert expected[k]==v


def pair_auc(rows, method):
    bad = np.asarray([r['risks'][method] for r in rows if wrong(r) is True])
    good = np.asarray([r['risks'][method] for r in rows if wrong(r) is False])
    if not len(bad) or not len(good):
        return None
    return float(sum(np.sum(v>good)+.5*np.sum(v==good) for v in bad)/(len(bad)*len(good)))


def independent_risk(model,features,normalizer,base):
    h = ((features-np.asarray(normalizer['mean']))/np.asarray(normalizer['scale']))[:,1:]
    for layer in (0,2,4):
        h = h.dot(np.asarray(model['network.%d.weight'%layer]).T)+np.asarray(model['network.%d.bias'%layer])
        if layer != 4:
            h[h<0]=0
    d = .5*np.tanh(h[:,0]+model['beta'][0])
    # Intentionally replay through the logit/sigmoid formulation, not expm1.
    r = 1/(1+np.exp(-(np.log(base/(1-base))+d)))
    return r,d


def audit(directory):
    directory = Path(directory); report = json.loads((directory/'report.json').read_text())
    completion = json.loads((directory/'completion.json').read_text()); contract,manifest = run.checked()
    assert report['contract']==contract and report['sources']==manifest
    for file,pin in completion['artifacts'].items():
        if file=='final_heads.pth' and not (directory/file).exists():
            # Archive deliberately excludes weights; server must have reviewed bytes.
            server = json.loads((directory/'server_review.json').read_text())
            assert server['weights_sha256_verified']==pin
            continue
        assert run.sha(directory/file)==pin
    for file,pin in manifest['sources'].items():
        data = subprocess.check_output(['git','show',report['git_commit']+':'+file],cwd=str(ROOT))
        assert hashlib.sha256(data).hexdigest()==pin
    assert not report['TEST_read'] and not report['GT_online'] and not report['original_policy_changed']
    assert report['boxes_scores_output_center_angle_unchanged'] and not report['new_data_roles']
    assert report['detector_inferences_added']==report['feature_forwards_added']==0
    assert report['update_counts']==dict.fromkeys(run.core.ARMS,1000)
    assert report['parameter_counts']==dict.fromkeys(run.core.ARMS,4290)
    assert report['fixed_final_epoch']==100 and report['engineering']['save_reload_exact']
    initial = json.loads((directory/'initial_heads.json').read_text())
    assert initial['score_anchor']==initial['simple_anchor']
    model = json.loads((directory/'models.json').read_text())
    parts,policy = run.load_rows(contract); normalizer = run.normalizer_for(contract)
    assert model['normalizer']==normalizer and model['dtype']=='float64'
    order_hash = hashlib.sha256(); rng = np.random.RandomState(1701)
    for _ in range(100):
        order_hash.update(rng.permutation(2558).astype('<i8').tobytes())
    assert order_hash.hexdigest()==report['batch_order_sha256']
    records = [json.loads(s) for s in (directory/'train_log.jsonl').read_text().splitlines()]
    assert len(records)==2000
    for i,r in enumerate(records):
        assert (r['epoch'],r['slot'],r['arm'])==(i//20+1,(i//2)%10,run.core.ARMS[i%2])
        assert r['batch_size']==(254 if r['slot']==9 else 256)
        assert r['gradient_norm_after']<=min(5.,r['gradient_norm_before'])+1e-10
    max_difference = 0.; summaries = auc_count = 0; gate_failures = []; real_gain = False
    for role,original in parts.items():
        rows = run.jsonl(directory/('scored_'+role+'.jsonl.gz'))
        assert len(rows)==len(original)
        for r,o in zip(rows,original):
            assert {k:r[k] for k in run.FIELDS}=={k:o[k] for k in run.FIELDS}
            for a in run.core.ARMS:
                decision = r['candidate_decisions'][a]; base = r['original_simple_decision']
                for key in ('center_accepted','angle_accepted','final_box_original'):
                    assert decision[key]==base[key]
                assert decision['risks']['angle']==base['risks']['angle']
                expected = r['pred'] is not None and r['risks'][a]<=report['cutoffs'][a]['risk_le']
                assert decision['size_accepted']==expected
                assert decision['risks']['size']==r['risks'][a]
        features,outputs = run.matrix(contract,original,role,normalizer)
        for a in run.core.ARMS:
            base = np.asarray([1-r['pred'][5] if a=='score_anchor' else r['size_risks']['full_simple'] for r in outputs])
            risks,d = independent_risk(model['models'][a],features,normalizer,base)
            saved = np.asarray([r['risks'][a] for r in rows if r['pred'] is not None])
            difference = float(np.max(np.abs(saved-risks))); max_difference = max(max_difference,difference)
            assert difference<1e-10 and np.max(np.abs(d))<=.5
            _,neutral = independent_risk(initial[a],features,normalizer,base)
            assert np.array_equal(neutral,np.zeros_like(neutral))
        groups = {'all':rows}
        for field in ('domain','sequence'):
            groups.update({field+':'+v:[r for r in rows if r[field]==v] for v in sorted({r[field] for r in rows})})
        for method in run.core.METHODS:
            if role=='VAL':
                requirements = {}
                for group,values in groups.items():
                    good = sorted(r['risks'][method] for r in values if wrong(r) is False)
                    k = math.ceil(.95*len(good)); requirements[group]=dict(good=len(good),required=k,first_whole_tie_risk_le=good[k-1] if k else None)
                assert requirements==report['cutoffs'][method]['requirements']
                assert report['cutoffs'][method]['risk_le']==max(v['first_whole_tie_risk_le'] for v in requirements.values() if v['good'])
            for group,values in groups.items():
                stat = report['statistics'][role][method][group]
                kept = {r['image'] for r in values if r['pred'] is not None and r['risks'][method]<=report['cutoffs'][method]['risk_le']}
                summary(values,kept,stat['actual']); summaries+=1
                expected_auc = pair_auc(values,method); actual_auc = stat['error_AUROC']
                assert actual_auc is expected_auc if expected_auc is None else abs(actual_auc-expected_auc)<1e-12
                auc_count+=1
                for control,comp in stat['controls'].items():
                    out = sorted((r for r in values if r['pred'] is not None),key=lambda r:(r['risks'][control],r['image']))
                    n = stat['actual']['accepted_outputs']; target = stat['actual']['states']['CR']
                    summary(values,{r['image'] for r in out[:n]},comp['same_count']); summaries+=1
                    if n:
                        boundary = out[n-1]['risks'][control]
                        before = [r for r in out if r['risks'][control]<boundary]
                        tie = [r for r in out if r['risks'][control]==boundary]; slots=n-len(before)
                        bw = sum(wrong(r) for r in before)
                        assert comp['same_count_tie_bounds']['bad_min']==bw+max(0,slots-sum(not wrong(r) for r in tie))
                        assert comp['same_count_tie_bounds']['bad_max']==bw+min(slots,sum(wrong(r) for r in tie))
                    seen=0; threshold=None
                    for r in out:
                        if target==0:
                            break
                        seen+=int(not wrong(r))
                        if seen>=target:
                            threshold=r['risks'][control]; break
                    selected = {r['image'] for r in out if threshold is not None and r['risks'][control]<=threshold}
                    summary(values,selected,comp['same_CR']); summaries+=1
                    assert comp['same_CR_exact']==(comp['same_CR']['states']['CR']==target)
                    assert comp['same_CR_offline_risk_le']==threshold
                    if role=='VAL' and method=='simple_anchor':
                        c=stat['actual']['states']; actual=stat['actual']; sc=comp['same_CR']
                        conditions=dict(same_count_FA_nonincrease=c['FA']<=comp['same_count_tie_bounds']['bad_min'],
                            same_CR_exact=comp['same_CR_exact'],same_CR_FA_nonincrease=c['FA']<=sc['states']['FA'],
                            longest_correct_FR_nonincrease=actual['runs']['correct_rejection']['longest']<=sc['runs']['correct_rejection']['longest'])
                        if group=='all':
                            conditions.update(overall_strict_same_count_FA_gain=c['FA']<comp['same_count_tie_bounds']['bad_min'],overall_strict_same_CR_FA_gain=c['FA']<sc['states']['FA'])
                        gate_failures.extend(dict(group=group,control=control,check=k) for k,v in conditions.items() if not v)
                        if group.startswith('sequence:real_') and control=='full_simple' and comp['same_CR_exact'] and c['FA']<sc['states']['FA']:
                            real_gain=True
                if role=='VAL' and method=='simple_anchor':
                    c=stat['actual']['states']
                    if c['CR']<math.ceil(.95*(c['CR']+c['FR'])):
                        gate_failures.append(dict(group=group,check='correct_retention95'))
        for group,values in groups.items():
            kept={r['image'] for r in values if r['original_simple_decision']['size_accepted']}
            summary(values,kept,report['formal_policy_summary'][role][group]); summaries+=1
            out=[r for r in values if r['pred'] is not None]
            hits=sum(math.hypot(r['pred'][0]-r['gt'][0],r['pred'][1]-r['gt'][1])<15 for r in out)
            assert report['center'][role][group]==dict(frames=len(values),outputs=len(out),hits=hits,
                hit_rate_on_outputs=hits/len(out) if out else None,output_coverage=len(out)/len(values),correct_coverage_all_frames=hits/len(values))
    if not real_gain:
        gate_failures.append(dict(group='Real_videos',check='at_least_one_strict_same_CR_simple_gain'))
    canonical = lambda items:sorted(json.dumps(x,sort_keys=True) for x in items)
    assert canonical(gate_failures)==canonical(report['gate']['failures'])
    assert report['gate']['passed']==(not gate_failures)
    assert report['status']==('VAL_FAILED_STOP' if gate_failures else 'VAL_PASS_FROZEN_TEST_PENDING')
    assert (directory/'analysis.md').read_text()==run.core.markdown(report)
    return dict(passed=True,rows_checked=3445,summaries_checked=summaries,AUROC_checked=auc_count,
        independent_numpy_max_difference=max_difference,independent_gate=True,status=report['status'],
        weights_sha256_verified=completion['artifacts']['final_heads.pth'],
        weights_present=(directory/'final_heads.pth').exists(),report_sha256=run.sha(directory/'report.json'),TEST_read=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('directory',type=Path)
    parser.add_argument('--receipt',type=Path,required=True); args=parser.parse_args()
    result=audit(args.directory); run.write(args.receipt,result); print(json.dumps(result))
