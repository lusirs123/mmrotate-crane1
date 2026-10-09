#!/usr/bin/env python3
"""Independent NumPy/scalar audit of labels, feature transforms and decisions."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_box_contrast_v1 as run
from crane_project.tools.review_port_reliability_readout_compare_v1 import bad,counts,grouped,auc,digest
from crane_project.utils import port_simple_component_reliability_v1 as simple


def independent_logits(model, matrix, normalizer):
    normalized=(matrix-np.asarray(normalizer['mean']))/np.asarray(normalizer['scale'])
    h=normalized[...,1:]
    for layer in (0,2):
        h=np.maximum(h@np.asarray(model['network.%d.weight'%layer]).T+model['network.%d.bias'%layer],0.)
    logits=-matrix[...,0]+(h@np.asarray(model['network.4.weight']).T+model['network.4.bias'])[...,0]+model['beta'][0]
    return logits,h


def independent_variants(pred, role):
    result=[list(pred)];keys=['M']
    if role=='VAL':return result,keys
    long_index=2 if pred[2]>=pred[3] else 3;short_index=5-long_index
    for mode in ('long','short','both'):
        for multiplier in (.85,.95,1.05,1.15):
            p=list(pred)
            if mode!='short':p[long_index]*=multiplier
            if mode!='long':p[short_index]*=multiplier
            if p[long_index]>=p[short_index]:result.append(p);keys.append(mode+'_'+str(multiplier))
    return result,keys


def audit(directory):
    directory=Path(directory);run.checked_sources()
    report=json.loads((directory/'report.json').read_text());receipt=json.loads((directory/'completion.json').read_text())
    excluded=[]
    for name,pin in receipt['artifacts'].items():
        if name=='final_heads.pth' and not (directory/name).exists():excluded.append(name);continue
        assert digest(directory/name)==pin,(name,'artifact SHA')
    commit=report['git_commit'];assert re.fullmatch('[0-9a-f]{40}',commit)
    assert set(report['sources']['sources'])==set(run.SOURCE_FILES)
    for name,pin in report['sources']['sources'].items():
        assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name],cwd=str(ROOT))).hexdigest()==pin
    source_bytes=subprocess.check_output(['git','show',commit+':'+str(run.SOURCES.relative_to(ROOT))],cwd=str(ROOT))
    assert hashlib.sha256(source_bytes).hexdigest()==report['sources']['manifest_sha256']
    assert report['contract']==json.loads(run.PROTOCOL.read_text()) and report['input_sha256']==run.PINS
    if report.get('reporting_recovery'):
        recovery=report['reporting_recovery'];assert recovery['training_updates_added']==0 and recovery['detector_inferences_added']==0
        for name,pin in recovery['original_outputs_sha256'].items():
            if name in excluded:
                assert receipt['artifacts'][name]==pin;continue
            assert digest(directory/name)==pin
        for name,pin in report['reporting_sources']['sources'].items():
            assert hashlib.sha256(subprocess.check_output(['git','show',recovery['reporting_commit']+':'+name],cwd=str(ROOT))).hexdigest()==pin
        extra='crane_project/tools/finalize_port_reliability_box_contrast_v1.py'
        assert hashlib.sha256(subprocess.check_output(['git','show',recovery['reporting_commit']+':'+extra],cwd=str(ROOT))).hexdigest()==recovery['reporting_source_sha256']
    unavailable_pins=[]
    for name,pin in run.PINS.items():
        if not (ROOT/name).exists() and name.endswith('.pth'):
            unavailable_pins.append(name);continue
        if not (ROOT/name).exists() and name==run.prior.CACHE+'/cache_manifest.json':
            assert digest(directory/'frozen_cache_manifest.json')==pin;continue
        assert run.sha(ROOT/name)==pin
    assert not report['TEST_read'] and not report['original_policy_changed']
    assert report['boxes_scores_output_center_angle_unchanged'] and report['pca_normalizer_fit_only_original_TRAIN']
    originals,_=run.load_rows();saved=json.loads((directory/'models.json').read_text())
    pca=json.loads((directory/'pca.json').read_text());normalizer=saved['normalizer']
    initial=json.loads((directory/'initial_heads.json').read_text())
    assert initial['original']==initial['contrast']
    assert pca['train_count']==2558 and pca['dimensions']==256 and pca['raw_dimensions']==2304 and not pca['whiten']
    components=np.asarray(pca['components']);mean=np.asarray(pca['mean'])
    np.testing.assert_allclose(components@components.T,np.eye(256),atol=1e-10,rtol=0)
    maximum=0.;groups_checked=ranks=contrasts=0;failures=[];training={};parts={};mechanisms={}
    for role,expected in (('TRAIN',2558),('VAL',887)):
        print('Independent box contrast review',role,'labels/PCA/network',flush=True)
        rows=[json.loads(s) for s in gzip.open(directory/('scored_'+role+'.jsonl.gz'),'rt')];parts[role]=rows
        assert len(rows)==expected and all(r['reliability_role']==role.lower() for r in rows)
        expected_rows=sorted([r for r in originals if r['reliability_role']==role.lower()],key=lambda r:r['image'])
        assert simple.fingerprint(expected_rows)==simple.fingerprint([{k:r[k] for k in run.prior.FIELDS} for r in rows])
        data=np.load(directory/(role.lower()+'_raw.npz'),allow_pickle=False)
        raw,descriptors,labels,mask,ids=[data[k] for k in ('features','descriptors','labels','mask','images')]
        ids=ids.tolist();assert ids==[r['image'] for r in rows if r['pred'] is not None]
        n,k=mask.shape;assert k==(13 if role=='TRAIN' else 1) and raw.shape==(n,k,2304)
        assert labels.shape==mask.shape and descriptors.shape==(n,k,3) and mask.dtype==np.bool_
        assert mask[:,0].all();by_id={r['image']:r for r in rows};index={name:i for i,name in enumerate(ids)}
        traces=[json.loads(s) for s in gzip.open(directory/(role.lower()+'_trace.jsonl.gz'),'rt')]
        assert [r['image'] for r in traces]==[r['image'] for r in rows]
        changed=paired=aux_bad=0
        for trace in traces:
            row=by_id[trace['image']];assert trace['role']==role and not trace['GT_online']
            assert trace['present']==(row['pred'] is not None)
            if not trace['present']:assert trace['image'] not in index and 'boxes' not in trace;continue
            i=index[trace['image']];boxes,keys=independent_variants(row['pred'],role);size=len(boxes)
            assert trace['boxes']==boxes and trace['keys']==keys
            assert mask[i,:size].all() and not mask[i,size:].any()
            np.testing.assert_array_equal(raw[i,size:],0);np.testing.assert_array_equal(labels[i,size:],0)
            assert hashlib.sha256(raw[i,:size].astype('<f4').tobytes()).hexdigest()==trace['feature_sha256']
            targets=[]
            for j,box in enumerate(boxes):
                p,g=sorted(box[2:4]),sorted(row['gt'][2:4])
                target=int(any(abs(a-b)/b>.1 for a,b in zip(p,g)));targets.append(target)
                assert labels[i,j]==target
                np.testing.assert_allclose(descriptors[i,j],simple.descriptor(box,row['image_size']),atol=1e-12,rtol=0)
                assert [box[t] for t in (0,1,4,5)]==[row['pred'][t] for t in (0,1,4,5)]
            assert trace['labels']==targets
            contrasts+=size-1;aux_bad+=sum(targets[1:])
            changed+=sum(not np.array_equal(raw[i,0],v) for v in raw[i,1:size])
            paired+=int(0 in targets and 1 in targets)
        proof=report['train_proof'] if role=='TRAIN' else report['val_proof']
        assert proof['changed_pooled_features']==changed and proof['same_image_both_class_frames']==paired
        assert proof['auxiliary_bad']==aux_bad and proof['valid_contrasts']==int(mask[:,1:].sum())
        assert proof['original_M_never_replaced'] and proof['detector_updates']==0 and proof['midpoint_updates']==0
        matrix=np.concatenate((descriptors,(raw.astype(np.float64)-mean)@components.T),axis=2)
        stored=np.load(directory/(role.lower()+'_projected.npz'),allow_pickle=False)
        assert stored['images'].tolist()==ids
        np.testing.assert_allclose(matrix if role=='TRAIN' else matrix[:,0],stored['features'],rtol=1e-12,atol=1e-10)
        if role=='TRAIN':
            assert int(labels[:,0].sum())==71
            np.testing.assert_allclose(mean,raw[:,0].astype(np.float64).mean(0),atol=1e-12,rtol=0)
            np.testing.assert_allclose(matrix[:,0,3:].var(0,ddof=1),pca['eigenvalues'],rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(normalizer['mean'],matrix[:,0].mean(0),atol=1e-12,rtol=0)
            scale=matrix[:,0].std(0);scale[scale<1e-8]=1.
            np.testing.assert_allclose(normalizer['scale'],scale,atol=1e-12,rtol=0)
        stats=report['train_statistics'] if role=='TRAIN' else report['VAL_statistics']
        for arm in run.core.ARMS:
            logits,hidden=independent_logits(saved['models'][arm],matrix,normalizer)
            risk=np.exp(-np.logaddexp(0.,-logits[:,0]))
            for image,value in zip(ids,risk):
                error=abs(float(value)-by_id[image]['experiment_risks'][arm]);maximum=max(maximum,error);assert error<1e-12
            if role=='TRAIN':
                yy=labels[:,0].astype(np.float64);w=np.where(yy==1,len(yy)/(2*yy.sum()),len(yy)/(2*(len(yy)-yy.sum())))
                real=float(np.mean((np.logaddexp(0.,logits[:,0])-yy*logits[:,0])*w))
                aux_values=np.where(mask[:,1:],np.logaddexp(0.,logits[:,1:])-labels[:,1:]*logits[:,1:],0.)
                aux=float(.25*np.mean(aux_values.sum(1)/np.maximum(mask[:,1:].sum(1),1)))
                training[arm]=dict(real_balanced_final_loss=real,weighted_auxiliary_loss=aux,
                    combined_objective=real+(aux if arm=='contrast' else 0),TRAIN_error_AUROC=auc(rows,arm))
                correct=ties=pairs=both=0
                for ls,ys,valid in zip(logits,labels,mask):
                    good=ls[valid & (ys==0)];wrong=ls[valid & (ys==1)]
                    if len(good) and len(wrong):
                        difference=wrong[:,None]-good[None,:];pairs+=difference.size
                        correct+=int((difference>0).sum());ties+=int((difference==0).sum());both+=1
                mechanisms[arm]=dict(frames_with_both_classes=both,pairs=int(pairs),correctly_ordered=correct,
                    ties=ties,pair_AUROC=(correct+.5*ties)/pairs if pairs else None,synthetic_mechanism_only=True)
                assert mechanisms[arm]==report['TRAIN_contrast_diagnostic'][arm]
            for row in rows:
                decision=deepcopy(row['original_simple_decision'])
                if row['pred'] is not None:
                    decision['risks']['size']=row['experiment_risks'][arm]
                    decision['size_accepted']=decision['risks']['size']<=report['VAL_cutoffs'][arm]['risk_le']
                assert decision==row['candidate_decisions'][arm]
                assert row['experiment_risks']['full_simple']==row['size_risks']['full_simple']
                assert row['experiment_risks']['score_only']==(None if row['pred'] is None else 1-row['pred'][5])
            for name,group in grouped(rows).items():
                out=[r for r in group if r['pred'] is not None];point=stats[arm][name]
                kept={r['image'] for r in out if r['experiment_risks'][arm]<=report['VAL_cutoffs'][arm]['risk_le']}
                c,longest=counts(group,kept)
                assert c==point['actual']['states'] and longest==point['actual']['runs']['correct_rejection']['longest'];groups_checked+=1
                for method,r in point['rank'].items():
                    value=auc(group,method);assert value==r['error_auroc'] or (value is not None and abs(value-r['error_auroc'])<1e-12);ranks+=1
                indices=[index[r['image']] for r in out];zero=np.all(hidden[indices,0]==0,axis=1)
                hd=report['hidden_diagnostic'][role][arm][name]
                assert hd['second_hidden_all_zero']==int(zero.sum())
                assert hd['zero_bad']==sum(bool(v) and bad(r) for v,r in zip(zero,out))
                for method in point['matched_CR_controls']:
                    ordered=sorted(out,key=lambda r:(r['experiment_risks'][method],r['image']));accepted_count=len(kept)
                    c2,_=counts(group,{r['image'] for r in ordered[:accepted_count]})
                    assert c2==point['same_count_controls'][method]['states'];groups_checked+=1
                    good=0;boundary=None;target=c['CR']
                    if target:
                        for row in ordered:
                            good+=int(not bad(row))
                            if good>=target:boundary=row['experiment_risks'][method];break
                    matched={r['image'] for r in out if boundary is not None and r['experiment_risks'][method]<=boundary}
                    c3,l3=counts(group,matched)
                    assert c3==point['matched_CR_controls'][method]['states'] and l3==point['matched_CR_controls'][method]['runs']['correct_rejection']['longest'];groups_checked+=1
                    if accepted_count:
                        threshold=ordered[accepted_count-1]['experiment_risks'][method]
                        lower_rows=[r for r in out if r['experiment_risks'][method]<threshold]
                        tie=[r for r in out if r['experiment_risks'][method]==threshold]
                        lower=sum(bad(r) for r in lower_rows)+max(0,accepted_count-len(lower_rows)-sum(not bad(r) for r in tie))
                    else:lower=0
                    assert lower==point['control_tie_bounds'][method]['bad_min_over_tie']
                    if role=='VAL' and arm=='contrast':
                        if c['FA']>lower:failures.append(dict(group=name,control=method,check='same_count_FA_nonincrease'))
                        if name=='all' and c['FA']>=lower:failures.append(dict(group=name,control=method,check='strict_overall_same_count_FA_gain'))
                        if c['FA']>c3['FA']:failures.append(dict(group=name,control=method,check='same_CR_FA_nonincrease'))
                        if name=='all' and c['FA']>=c3['FA']:failures.append(dict(group=name,control=method,check='strict_overall_same_CR_FA_gain'))
                        if c3['CR']!=target:failures.append(dict(group=name,control=method,check='same_CR_tie_inconclusive'))
                        if longest>l3:failures.append(dict(group=name,control=method,check='longest_FR_same_CR_nonincrease'))
                if role=='VAL' and arm=='contrast' and c['CR']<int(np.ceil(.95*(c['CR']+c['FR']))):
                    failures.append(dict(group=name,check='correct_retention'))
        if role=='VAL':
            for name,group in grouped(rows).items():
                outputs=sum(r['pred'] is not None for r in group)
                hits=sum(np.linalg.norm(np.array(r['pred'][:2])-np.array(r['gt'][:2]))<15 for r in group if r['pred'] is not None)
                c=report['center'][name];assert c['outputs']==outputs and c['hits']==hits
                assert c['hit_rate_on_outputs']==hits/outputs and c['output_coverage']==outputs/len(group)
                assert c['correct_coverage_all_frames']==hits/len(group)
        del raw,matrix,data,stored
    for arm,cutoff in report['VAL_cutoffs'].items():
        requirements=[]
        for name,group in grouped(parts['VAL']).items():
            correct=sorted(r['experiment_risks'][arm] for r in group if bad(r) is False)
            if correct:requirements.append(correct[int(np.ceil(.95*len(correct)))-1])
        assert cutoff['risk_le']==max(requirements)
    assert sorted(failures,key=lambda d:json.dumps(d,sort_keys=True))==sorted(report['gate']['failures'],key=lambda d:json.dumps(d,sort_keys=True))
    assert report['gate']['passed']==(not failures)
    assert report['status']==('VAL_FAILED_STOP' if failures else 'VAL_PASS_FROZEN_TEST_PENDING')
    assert report['parameter_counts']==dict(original=4290,contrast=4290)
    logs=[json.loads(s) for s in (directory/'train_log.jsonl').read_text().splitlines()]
    assert len(logs)==2000 and Counter(r['arm'] for r in logs)==dict(original=1000,contrast=1000)
    assert {(r['epoch'],r['slot'],r['arm']) for r in logs}=={(e,s,a) for e in range(1,101) for s in range(10) for a in run.core.ARMS}
    for value in logs:
        assert value['batch_count']==(254 if value['slot']==9 else 256)
        assert abs(value['loss']-(value['real_loss']+value['weighted_auxiliary_loss']))<2e-5
        assert np.isfinite(value['gradient_norm_before']) and value['gradient_norm_after']<5.0001
        if value['arm']=='original':assert value['weighted_auxiliary_loss']==0 and value['valid_contrasts']==0
        if value['slot']==0:assert 'components' in value
    order=hashlib.sha256();rng=np.random.RandomState(1701)
    for _ in range(100):order.update(rng.permutation(2558).astype('<i8').tobytes())
    assert order.hexdigest()==report['batch_order_sha256']
    for model in initial.values():
        assert not np.asarray(model['network.4.weight']).any() and model['network.4.bias']==[0.] and model['beta']==[0.]
    return dict(passed=True,rows=3445,contrasts_checked=contrasts,risk_replay_maximum=maximum,
        state_and_matched_groups_checked=groups_checked,rank_AUROC_checks=ranks,gate_failures_exact=failures,
        training=training,mechanisms=mechanisms,updates_checked=2000,
        pca_normalizer_fitted_only_original_TRAIN=True,original_M_simple_policy_preserved=True,
        checkpoint_excluded_from_return_archive=excluded,report_sha256=digest(directory/'report.json'),
        frozen_checkpoints_not_copied_locally=unavailable_pins,
        reviewer_source_sha256=digest(Path(__file__)),reviewer_sources=run.checked_sources()[1],
        reviewer_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),
        models_sha256=digest(directory/'models.json'),TEST_read=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--output',default='independent_review.json');args=parser.parse_args()
    if Path(args.output).name!=args.output:raise ValueError('Review file stays inside result directory')
    result=audit(args.directory);run.write(args.directory/args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
