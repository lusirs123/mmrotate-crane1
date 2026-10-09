#!/usr/bin/env python3
"""CPU independent audit of frozen TEST results. No fitting or threshold choice."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import eval_port_reliability_feature_source_test_v1 as runner
from crane_project.tools import review_port_reliability_redc_size_v1 as scalar
from crane_project.utils import port_simple_component_reliability_v1 as simple


def audit(directory):
    directory=Path(directory);receipt=json.loads((directory/'completion.json').read_text())
    if (not receipt['TEST_evaluated'] or receipt['automatic_promotion']
            or receipt['protocol']!=runner.core.VERSION
            or receipt['status']!='EXPOSED_TEST_DIAGNOSTIC_COMPLETE_KEEP_VAL_FAILURE'):
        raise ValueError('Not frozen TEST diagnostic')
    for name,pin in receipt['artifacts'].items():assert runner.sha(directory/name)==pin
    model,cutoffs,pcas,_,_=runner.checked_inputs(False)
    source=runner.checked_sources()[1];report=json.loads((directory/'report.json').read_text())
    assert report['frozen_cutoffs']==cutoffs and report['sources']==source
    assert report['parent_VAL_status']=='VAL_FAILED_STOP' and not report['VAL_failure_retracted']
    assert not report['TEST_used_for_selection'] and not report['automatic_promotion']
    assert all(report[k]==0 for k in ('detector_updates','head_updates','quality_updates','threshold_updates'))
    records=runner.read_rows(runner.ROOT/runner.BASE/'test_decisions.jsonl')
    native=runner.read_rows(runner.ROOT/runner.COLLECT/'port_test/predictions.jsonl')
    original=runner.core.bind_rows(records,native)
    rows=[json.loads(s) for s in gzip.open(directory/'scored_TEST.jsonl.gz','rt')]
    assert [{k:r[k] for k in original[0]} for r in rows]==original
    stem=np.load(runner.ROOT/runner.STEM/'test_features.npz',allow_pickle=False)
    ids=stem['images'].tolist();x=stem['features'];lookup={r['image']:r for r in rows}
    raw=np.load(directory/'test_native_raw.npz',allow_pickle=False);native_raw=raw['features']
    assert raw['images'].tolist()==ids and native_raw.shape==(len(ids),2304)
    traces=[json.loads(s) for s in gzip.open(directory/'test_native_trace.jsonl.gz','rt')]
    import hashlib
    assert [t['image'] for t in traces]==[r['image'] for r in rows]
    for t in traces:
        assert t['role']=='TEST' and not t.get('GT_online',False)
        if not t['present']: assert lookup[t['image']]['pred'] is None;continue
        v=native_raw[ids.index(t['image'])]
        assert hashlib.sha256(v.astype('<f4').tobytes()).hexdigest()==t['patch_sha256']
        assert t['actual_topk_observed'] and t['regression_max_replay_difference']<.01
        assert t['flat_anchor_index']==(t['y']*t['feature_shape'][3]+t['x'])*3+t['anchor']
    matrices={}
    for arm,source_raw in dict(midpoint=x[:,3:],native=native_raw).items():
        p=pcas[arm];components=np.asarray(p['components']);mean=np.asarray(p['mean'])
        np.testing.assert_allclose(components@components.T,np.eye(256),atol=1e-10,rtol=0)
        matrices[arm]=np.c_[x[:,:3],(np.asarray(source_raw,dtype=np.float64)-mean)@components.T]
        projected=np.load(directory/('test_'+arm+'_projected.npz'),allow_pickle=False)
        assert projected['images'].tolist()==ids
        np.testing.assert_allclose(matrices[arm],projected['features'],atol=1e-10,rtol=1e-12)
    assert ids==[r['image'] for r in rows if r['pred'] is not None]
    index={image:i for i,image in enumerate(ids)}
    assert x.shape==(len(ids),291) and np.isfinite(x).all()
    assert report['input_sha256']==runner.PINS
    proof=report['feature_proof']
    assert proof['frames']==1440 and proof['outputs']==len(ids) and proof['new_detector_inferences']==1440
    assert proof['detector_updates']==0 and not proof['GT_online']
    maximum_B=np.asarray(proof['maximum_original_B_replay_difference'])
    assert maximum_B.shape==(6,) and np.isfinite(maximum_B).all() and maximum_B[4]<=2e-6 and maximum_B[5]==0
    assert not report['GT_online'] and report['boxes_scores_center_angle_output_unchanged']
    maximum=0.;status_checks=0;rank_checks=0
    for arm in runner.core.scoring.ARMS:
        normalizer=model['normalizers'][arm];projected=matrices[arm]
        h=((projected-np.asarray(normalizer['mean']))/np.asarray(normalizer['scale']))[:,1:]
        weights=model['models'][arm]
        for layer in (0,2,4):
            h=h@np.asarray(weights['network.%d.weight'%layer]).T+weights['network.%d.bias'%layer]
            if layer!=4:h=np.maximum(h,0.)
        risk=np.exp(-np.logaddexp(0.,-(-projected[:,0]+h[:,0]+weights['beta'][0])))
        for image,value in zip(ids,risk):
            delta=abs(float(value)-lookup[image]['experiment_risks'][arm]);maximum=max(maximum,delta)
            assert delta<1e-12
    for r in rows:
        if r['pred'] is not None:
            np.testing.assert_allclose(x[index[r['image']],:3],simple.descriptor(r['pred'],r['image_size']),rtol=0,atol=1e-12)
            assert r['experiment_risks']['full_simple']==r['original_simple_decision']['risks']['size']
            assert r['experiment_risks']['score_only']==1-r['pred'][5]
        else:assert all(v is None for v in r['experiment_risks'].values())
        for method in cutoffs:
            d=deepcopy(r['original_simple_decision'])
            if r['pred'] is not None:
                d['risks']['size']=r['experiment_risks'][method]
                d['size_accepted']=d['risks']['size']<=cutoffs[method]['risk_le']
            assert d==r['candidate_decisions'][method]
    for name,part in scalar.grouped(rows).items():
        out=[r for r in part if r['pred'] is not None]
        for method in cutoffs:
            kept={r['image'] for r in out if r['experiment_risks'][method]<=cutoffs[method]['risk_le']}
            c,l=scalar.counts(part,kept);v=report['fixed_VAL_cutoff_summary'][name][method]
            assert c==v['states'] and l==v['runs']['correct_rejection']['longest'];status_checks+=1
        for arm in runner.core.scoring.ARMS:
            point=report['statistics'][arm][name]
            kept={r['image'] for r in out if r['experiment_risks'][arm]<=cutoffs[arm]['risk_le']}
            counts,longest=scalar.counts(part,kept)
            assert counts==point['actual']['states'] and longest==point['actual']['runs']['correct_rejection']['longest'];status_checks+=1
            for method,v in point['rank'].items():
                a=scalar.auc(part,method)
                assert a is None and v['error_auroc'] is None or a is not None and abs(a-v['error_auroc'])<1e-12
                rank_checks+=1
            for method,v in point['same_count_controls'].items():
                ranked=sorted(out,key=lambda r:(r['experiment_risks'][method],r['image']))
                counts2,_=scalar.counts(part,{r['image'] for r in ranked[:len(kept)]})
                assert counts2==v['states'];status_checks+=1
                tie=point['control_tie_bounds'][method];k=len(kept)
                if k:
                    boundary=ranked[k-1]['experiment_risks'][method]
                    before_tie=[r for r in ranked if r['experiment_risks'][method]<boundary]
                    tied=[r for r in ranked if r['experiment_risks'][method]==boundary]
                    selected=k-len(before_tie);bad_before=sum(scalar.bad(r) for r in before_tie)
                    bad_tie=sum(scalar.bad(r) for r in tied)
                    assert tie['bad_min_over_tie']==bad_before+max(0,selected-(len(tied)-bad_tie))
                    assert tie['bad_max_over_tie']==bad_before+min(selected,bad_tie)
                else:assert tie['bad_min_over_tie']==tie['bad_max_over_tie']==0
                good=0;bound=None
                if counts['CR']:
                    for r in ranked:
                        good+=int(not scalar.bad(r))
                        if good==counts['CR']:bound=r['experiment_risks'][method];break
                same_good={r['image'] for r in out if bound is not None and r['experiment_risks'][method]<=bound}
                c,l=scalar.counts(part,same_good);v3=point['matched_CR_controls'][method]
                assert c==v3['states'] and l==v3['runs']['correct_rejection']['longest'];status_checks+=1
                assert v3['requested_CR']==counts['CR'] and v3['exact_CR']==(c['CR']==counts['CR'])
        counts,_=scalar.counts(part,{r['image'] for r in part if r['original_simple_decision']['size_accepted']})
        assert counts==report['original_policy_summary'][name]['states']
        counts,_=scalar.counts(part,{r['image'] for r in out})
        assert counts==report['raw_size_summary'][name]['states']
        hits=sum(simple.geometry_errors(r['gt'],r['pred'])['center_px']<15 for r in out)
        assert report['center'][name]==dict(frames=len(part),outputs=len(out),hits=hits,output_coverage=len(out)/len(part),
            hit_rate_on_outputs=hits/len(out) if out else None,correct_coverage_all_frames=hits/len(part))
    return dict(passed=True,rows=len(rows),outputs=len(ids),risk_replay_maximum=maximum,
        state_and_matched_checks=status_checks,rank_AUROC_checks=rank_checks,
        parent_VAL_failure_preserved=True,threshold_updates=0,training_updates=0,
        TEST_is_frozen_diagnostic=True,report_sha256=runner.sha(directory/'report.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    result=audit(args.directory);runner.write(args.directory/'independent_review.json',result)
    print(json.dumps(result,indent=2))
