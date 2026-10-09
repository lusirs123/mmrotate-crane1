#!/usr/bin/env python3
"""Independent NumPy/scalar replay; no fit, detector inference or selection."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import re
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_feature_source_v1 as run
from crane_project.tools.review_port_reliability_readout_compare_v1 import bad,counts,grouped,auc,digest
from crane_project.utils import port_simple_component_reliability_v1 as simple


def audit(directory):
    directory = Path(directory); run.checked_sources()
    receipt = json.loads((directory/'completion.json').read_text())
    for name,pin in receipt['artifacts'].items(): assert digest(directory/name)==pin,(name,'artifact SHA')
    report = json.loads((directory/'report.json').read_text()); saved = json.loads((directory/'models.json').read_text())
    pcas = json.loads((directory/'pca.json').read_text()); initial = json.loads((directory/'initial_heads.json').read_text())
    assert initial['native']==initial['midpoint']
    # A reviewer-only engineering correction may follow an immutable run.
    # Bind its original complete source closure to the recorded Git commit;
    # do not rewrite the report/checkpoint/source receipt to current bytes.
    commit=report['git_commit'];assert re.fullmatch('[0-9a-f]{40}',commit)
    for name,pin in report['sources']['sources'].items():
        content=subprocess.check_output(['git','show',commit+':'+name],cwd=str(ROOT))
        assert hashlib.sha256(content).hexdigest()==pin
    manifest=subprocess.check_output(['git','show',commit+':'+str(run.SOURCES.relative_to(ROOT))],cwd=str(ROOT))
    assert hashlib.sha256(manifest).hexdigest()==report['sources']['manifest_sha256']
    assert report['contract']==json.loads(run.PROTOCOL.read_text())
    assert report['input_sha256']==run.PINS and report['boxes_scores_output_center_angle_unchanged'] and not report['TEST_read']
    parts = {}; maximum = 0.; checks = 0; ranks = 0; failures = []; training = {}
    for role,expected in (('TRAIN',2558),('VAL',887)):
        rows = [json.loads(s) for s in gzip.open(directory/('scored_'+role+'.jsonl.gz'),'rt')]; parts[role] = rows
        assert len(rows)==expected and all(r['reliability_role']==role.lower() for r in rows)
        originals,source,ids = run.parent.checked_feature_rows(role,[{k:r[k] for k in run.prior.FIELDS} for r in rows])
        assert simple.fingerprint(originals)==simple.fingerprint([{k:r[k] for k in run.prior.FIELDS} for r in rows])
        native = np.load(directory/(role.lower()+'_native_raw.npz'),allow_pickle=False)
        assert native['images'].tolist()==ids and native['features'].shape==(len(ids),2304)
        traces = [json.loads(s) for s in gzip.open(directory/(role.lower()+'_native_trace.jsonl.gz'),'rt')]
        assert [r['image'] for r in traces]==[r['image'] for r in originals]
        idx = {image:i for i,image in enumerate(ids)}
        for trace in traces:
            assert trace['role']==role and not trace.get('GT_online',False)
            if not trace['present']: assert trace['image'] not in idx; continue
            value = native['features'][idx[trace['image']]]
            assert digest_bytes(value.astype('<f4').tobytes())==trace['patch_sha256']
            assert trace['actual_topk_observed'] and trace['feature_shape'][1]==256
            anchors = 3; width = trace['feature_shape'][3]
            assert trace['flat_anchor_index']==(trace['y']*width+trace['x'])*anchors+trace['anchor']
            assert 0<=trace['selected_surviving_index']<trace['surviving'] and trace['regression_max_replay_difference']<.01
        matrices = {}
        for arm,raw in (('midpoint',source[:,3:]),('native',native['features'])):
            raw=np.asarray(raw,dtype=np.float64)  # Match the fitted float64 PCA, not float32 np.mean.
            p = pcas[arm]; mean = np.asarray(p['mean']); components = np.asarray(p['components'])
            assert p['train_count']==2558 and p['dimensions']==256 and not p['whiten']
            np.testing.assert_allclose(components@components.T,np.eye(256),atol=1e-10,rtol=0)
            projected = np.c_[source[:,:3],(raw-mean)@components.T]
            data = np.load(directory/(role.lower()+'_'+arm+'_projected.npz'),allow_pickle=False)
            assert data['images'].tolist()==ids
            np.testing.assert_allclose(projected,data['features'],atol=1e-10,rtol=1e-12)
            matrices[arm] = projected
            normalizer = saved['normalizers'][arm]
            if role=='TRAIN':
                np.testing.assert_allclose(mean,raw.mean(0),atol=1e-12,rtol=0)
                np.testing.assert_allclose(projected[:,3:].var(0,ddof=1),p['eigenvalues'],atol=1e-9,rtol=1e-9)
                expected_scale=projected.std(0);expected_scale[expected_scale<1e-8]=1.
                np.testing.assert_allclose(normalizer['mean'],projected.mean(0),atol=1e-12,rtol=0)
                np.testing.assert_allclose(normalizer['scale'],expected_scale,atol=1e-12,rtol=0)
            h = ((projected-np.asarray(normalizer['mean']))/np.asarray(normalizer['scale']))[:,1:]
            model = saved['models'][arm]
            for layer in (0,2,4):
                h = h@np.asarray(model['network.%d.weight'%layer]).T+model['network.%d.bias'%layer]
                if layer!=4: h = np.maximum(h,0.)
            logits = -projected[:,0]+h[:,0]+model['beta'][0]
            risk = np.exp(-np.logaddexp(0.,-logits)); lookup = {r['image']:r for r in rows}
            for image,v in zip(ids,risk):
                error = abs(v-lookup[image]['experiment_risks'][arm]);maximum=max(maximum,float(error));assert error<1e-12
            if role=='TRAIN':
                y=np.asarray([bad(lookup[i]) for i in ids],dtype=float); assert int(y.sum())==71
                weights = np.where(y==1,len(y)/(2*y.sum()),len(y)/(2*(len(y)-y.sum())))
                training[arm]=dict(balanced_final_loss=float(np.mean((np.logaddexp(0.,logits)-y*logits)*weights)),
                    initial_loss=float(np.mean((np.logaddexp(0.,-projected[:,0])+y*projected[:,0])*weights)),
                    TRAIN_error_AUROC=auc(rows,arm))
            for r in rows:
                d=deepcopy(r['original_simple_decision'])
                if r['pred'] is not None:
                    d['risks']['size']=r['experiment_risks'][arm];d['size_accepted']=d['risks']['size']<=report['VAL_cutoffs'][arm]['risk_le']
                assert d==r['candidate_decisions'][arm]
                assert r['experiment_risks']['full_simple']==r['size_risks']['full_simple']
                assert r['experiment_risks']['score_only']==(None if r['pred'] is None else 1-r['pred'][5])
        stats = report['train_statistics'] if role=='TRAIN' else report['VAL_statistics']
        for arm in run.core.ARMS:
            for name,group in grouped(rows).items():
                out=[r for r in group if r['pred'] is not None];point=stats[arm][name]
                kept={r['image'] for r in out if r['experiment_risks'][arm]<=report['VAL_cutoffs'][arm]['risk_le']}
                c,l=counts(group,kept);assert c==point['actual']['states'] and l==point['actual']['runs']['correct_rejection']['longest'];checks+=1
                for method,rank in point['rank'].items():
                    v=auc(group,method);assert (v is None and rank['error_auroc'] is None) or abs(v-rank['error_auroc'])<1e-12;ranks+=1
                for method in point['matched_CR_controls']:
                    ordered=sorted(out,key=lambda r:(r['experiment_risks'][method],r['image']));k=len(kept)
                    same={r['image'] for r in ordered[:k]};c2,_=counts(group,same)
                    assert c2==point['same_count_controls'][method]['states'];checks+=1
                    target=c['CR'];good=0;boundary=None
                    if target:
                        for r in ordered:
                            good+=int(not bad(r))
                            if good>=target:boundary=r['experiment_risks'][method];break
                    matched={r['image'] for r in out if boundary is not None and r['experiment_risks'][method]<=boundary}
                    c3,l3=counts(group,matched);assert c3==point['matched_CR_controls'][method]['states'] and l3==point['matched_CR_controls'][method]['runs']['correct_rejection']['longest'];checks+=1
                    if k:
                        threshold=ordered[k-1]['experiment_risks'][method]
                        before=[r for r in ordered if r['experiment_risks'][method]<threshold];tie=[r for r in ordered if r['experiment_risks'][method]==threshold]
                        lower=sum(bad(r) for r in before)+max(0,k-len(before)-sum(not bad(r) for r in tie))
                    else:lower=0
                    assert lower==point['control_tie_bounds'][method]['bad_min_over_tie']
                    if role=='VAL' and arm=='native':
                        if c['FA']>lower:failures.append(dict(group=name,control=method,check='same_count_FA_nonincrease'))
                        if name=='all' and c['FA']>=lower:failures.append(dict(group=name,control=method,check='strict_overall_same_count_FA_gain'))
                        if c['FA']>c3['FA']:failures.append(dict(group=name,control=method,check='same_CR_FA_nonincrease'))
                        if name=='all' and c['FA']>=c3['FA']:failures.append(dict(group=name,control=method,check='strict_overall_same_CR_FA_gain'))
                        if c3['CR']!=target:failures.append(dict(group=name,control=method,check='same_CR_tie_inconclusive'))
                        if l>l3:failures.append(dict(group=name,control=method,check='longest_FR_same_CR_nonincrease'))
    for arm,cut in report['VAL_cutoffs'].items():
        thresholds=[]
        for name,group in grouped(parts['VAL']).items():
            good=sorted(r['experiment_risks'][arm] for r in group if bad(r) is False)
            if good:thresholds.append(good[int(np.ceil(.95*len(good)))-1])
        assert cut['risk_le']==max(thresholds)
    assert sorted(failures,key=lambda d:json.dumps(d,sort_keys=True))==sorted(report['gate']['failures'],key=lambda d:json.dumps(d,sort_keys=True))
    assert report['gate']['passed']==(not failures) and report['parameter_counts']==dict(midpoint=4290,native=4290)
    assert report['status']==('VAL_FAILED_STOP' if failures else 'VAL_PASS_FROZEN_TEST_PENDING')
    logs=[json.loads(s) for s in (directory/'train_log.jsonl').read_text().splitlines()]
    assert len(logs)==2000 and Counter(r['arm'] for r in logs)==dict(midpoint=1000,native=1000)
    assert {(r['epoch'],r['slot'],r['arm']) for r in logs}=={(e,s,a) for e in range(1,101) for s in range(10) for a in run.core.ARMS}
    assert all(r['batch_count']==(254 if r['slot']==9 else 256) and np.isfinite(r['loss']) and r['gradient_norm_after']<5.0001 for r in logs)
    order=hashlib.sha256();rng=np.random.RandomState(1701)
    for _ in range(100):order.update(rng.permutation(2558).astype('<i8').tobytes())
    assert order.hexdigest()==report['batch_order_sha256']
    for arm in run.core.ARMS:
        assert not np.asarray(initial[arm]['network.4.weight']).any() and initial[arm]['network.4.bias']==[0.] and initial[arm]['beta']==[0.]
    return dict(passed=True,rows=3445,risk_replay_maximum=maximum,state_and_matched_groups_checked=checks,
        rank_AUROC_checks=ranks,gate_failures_exact=failures,training=training,updates_checked=2000,
        pca_fitted_only_full_TRAIN=True,original_M_simple_policy_preserved=True,
        report_sha256=digest(directory/'report.json'),models_sha256=digest(directory/'models.json'),TEST_read=False)


def digest_bytes(value): return hashlib.sha256(value).hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--output',default='independent_review.json');args=parser.parse_args()
    if Path(args.output).name!=args.output:raise ValueError('Review output stays in result directory')
    result=audit(args.directory);run.write(args.directory/args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
