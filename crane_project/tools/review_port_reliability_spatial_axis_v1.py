#!/usr/bin/env python3
"""Independent labels, conv replay, confusion/coverage/ties and gate review."""
import argparse,gzip,hashlib,json,math,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_spatial_axis_v1 as run
from crane_project.tools.review_port_reliability_within_video_rank_v1 import wrong,summary,pair_auc


def conv_ref(x,w,b,pad=0):
    if pad:x=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)))
    windows=np.lib.stride_tricks.sliding_window_view(x,(w.shape[2],w.shape[3]),axis=(2,3))
    return np.einsum('nchwij,ocij->nohw',windows,w,optimize=True)+b[None,:,None,None]


def forward_ref(model,roi,support,descriptors,norm,anchor,arm):
    m=np.asarray(norm['channel_mean']);s=np.asarray(norm['channel_scale'])
    # Production casts normalized maps to FP32. Preserve that input contract.
    h=(((roi-m[None,:,None,None])/s[None,:,None,None])*(support>0)).astype(np.float32)
    ds=((descriptors-np.asarray(norm['descriptor_mean']))/np.asarray(norm['descriptor_scale'])).astype(np.float32)
    if arm=='coarse_bce':
        z=np.empty_like(h)
        for i in range(3):
            for j in range(3):
                block=h[:,:,3*i:3*i+3,3*j:3*j+3]
                z[:,:,3*i:3*i+3,3*j:3*j+3]=block.mean((2,3))[:,:,None,None]
        h=z
    h=np.concatenate((h,support),1).astype(np.float64);w=lambda k:np.asarray(model[k])
    h=np.maximum(conv_ref(h,w('stem.0.weight'),w('stem.0.bias')),0.)
    h=np.maximum(conv_ref(h,w('stem.2.weight'),w('stem.2.bias'),1),0.)
    h=np.c_[h.reshape(len(h),-1),ds]
    h=np.maximum(h.dot(w('hidden.weight').T)+w('hidden.bias'),0.)
    delta=.5*np.tanh((h.dot(w('risk_head.weight').T)+w('risk_head.bias'))[:,0])
    axis=h.dot(w('axis_head.weight').T)+w('axis_head.bias')
    risk=1/(1+np.exp(-(np.log(anchor)-np.log1p(-anchor)+delta)))
    return risk,axis


def audit(directory):
    p=Path(directory);report=json.loads((p/'report.json').read_text());completion=json.loads((p/'completion.json').read_text())
    local_weights_missing=any(not (ROOT/name).exists() for name in (run.prior.HEAD,run.historical.historical.B_PATH,run.prior.CACHE+'/cache_manifest.json'))
    if local_weights_missing:
        server=json.loads((p/'server_review.json').read_text());assert server['input_pins_verified']==report['input_pins']
    contract,manifest=run.checked(local_weights_missing);assert report['contract']==contract and report['sources']==manifest
    for name,sha in completion['artifacts'].items():
        if name=='final_heads.pth' and not (p/name).exists():
            assert json.loads((p/'server_review.json').read_text())['weights_sha256_verified']==sha
        else:assert run.sha(p/name)==sha
    for name,sha in manifest['sources'].items():
        assert hashlib.sha256(subprocess.check_output(['git','show',report['git_commit']+':'+name])).hexdigest()==sha
    assert not report['TEST_read'] and not report['GT_online'] and not report['original_policy_changed']
    assert report['boxes_scores_output_center_angle_unchanged'] and not report['new_data_roles']
    assert report['update_counts']==dict.fromkeys(run.core.ARMS,1000)
    assert report['parameter_counts']==dict.fromkeys(run.core.ARMS,13115)
    assert report['fixed_final_epoch']==100 and report['engineering']['save_reload_exact']
    initial=json.loads((p/'initial_heads.json').read_text());assert all(x==initial[run.core.ARMS[0]] for x in initial.values())
    model=json.loads((p/'models.json').read_text());norm=model['normalizer'];assert norm==json.loads((p/'normalizer.json').read_text())
    originals,policy=run.parts();log=[json.loads(s) for s in (p/'train_log.jsonl').read_text().splitlines()]
    assert len(log)==3000
    rng=np.random.RandomState(1701);digest=hashlib.sha256()
    for _ in range(100):digest.update(rng.permutation(2558).astype('<i8').tobytes())
    assert digest.hexdigest()==report['batch_order_sha256']
    for i,r in enumerate(log):
        assert (r['epoch'],r['slot'],r['arm'])==(i//30+1,(i//3)%10,run.core.ARMS[i%3])
        assert r['batch_size']==(254 if r['slot']==9 else 256)
        assert r['gradient_norm_after']<=min(5.,r['gradient_norm_before'])+2e-5
        assert r['actual_clipping']==(r['gradient_norm_before']>5.)
        applied=r['arm']=='spatial_axis';assert r['auxiliary_applied']==applied
        assert abs(r['loss']-r['bce_loss']-(r['weighted_auxiliary_loss'] if applied else 0.))<1e-10
        if not applied:assert not any(k.startswith('axis_head') for k in r['gradient_by_parameter'])
        if r['slot']==0:
            d=r['component_gradients'];assert d and all(math.isfinite(d[k]) for k in ('bce_norm','weighted_aux_norm'))
            assert abs(d['shared_clip_retention']-min(1.,5./(r['gradient_norm_before']+1e-6)))<1e-12
        else:assert r['component_gradients'] is None
    smoke=json.loads((p/'smoke_report.json').read_text())
    for a,v in smoke.items():
        assert v['exact_neutral'] and len(v['discarded_updates'])==2
        assert v['discarded_updates'][0]['component_gradients']['weighted_aux_norm']>0
        step=v['discarded_updates'][0 if a=='spatial_axis' else 1]
        assert step['gradient_by_parameter']['stem.0.weight']>0
    totals=aucs=0;maxrisk=maxaxis=0.;gate_fail=[];real_gain=False
    for role,original in originals.items():
        rows=run.jsonl(p/('scored_'+role+'.jsonl.gz'));assert len(rows)==len(original)
        for r,o in zip(rows,original):
            assert {k:r[k] for k in run.prior.FIELDS}==o
            assert wrong(r)==run.metrics.bad(r)
            for a in run.core.ARMS:
                d=r['candidate_decisions'][a];base=r['original_simple_decision']
                assert all(d[k]==base[k] for k in ('center_accepted','angle_accepted','final_box_original'))
                assert d['risks']['angle']==base['risks']['angle'] and d['risks']['size']==r['risks'][a]
                assert d['size_accepted']==(r['pred'] is not None and r['risks'][a]<=report['cutoffs'][a]['risk_le'])
        outputs=[r for r in rows if r['pred'] is not None];ids=[r['image'] for r in outputs]
        with np.load(p/(role.lower()+'_spatial.npz'),allow_pickle=False) as f:
            roi=f['features'];support=f['support'];ds=f['descriptors'];assert f['images'].tolist()==ids
        traces=run.jsonl(p/(role.lower()+'_trace.jsonl.gz'));assert [r['image'] for r in traces]==[r['image'] for r in rows]
        lookup={t['image']:t for t in traces}
        for i,r in enumerate(outputs):
            tr=lookup[r['image']];assert tr['box_original']==r['pred'] and not tr['GT_online']
            assert hashlib.sha256(roi[i].astype('<f4').tobytes()).hexdigest()==tr['feature_sha256']
            assert hashlib.sha256(support[i].astype('<f4').tobytes()).hexdigest()==tr['support_sha256']
            np.testing.assert_allclose(ds[i],run.simple.descriptor(r['pred'],r['image_size'])[1:],rtol=0,atol=1e-12)
        target=np.array([[(p0-g0)/g0/.1 for p0,g0 in zip(sorted(r['pred'][2:4],reverse=True),sorted(r['gt'][2:4],reverse=True))] for r in outputs])
        np.testing.assert_array_equal(target,run.core.targets(outputs))
        if role=='TRAIN':
            valid=support>0;mass=float(valid.sum());mean=(roi.astype(float)*valid).sum((0,2,3))/mass
            var=(((roi.astype(float)-mean[None,:,None,None])*valid)**2).sum((0,2,3))/mass
            scale=np.sqrt(var);scale[scale<1e-8]=1.
            np.testing.assert_allclose(mean,norm['channel_mean'],rtol=0,atol=1e-12);np.testing.assert_allclose(scale,norm['channel_scale'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(ds.mean(0),norm['descriptor_mean'],rtol=0,atol=1e-12)
            scale=ds.std(0);scale[scale<1e-8]=1.;np.testing.assert_allclose(scale,norm['descriptor_scale'],rtol=0,atol=1e-12)
        for a in run.core.ARMS:
            predicted=[]
            for start in range(0,len(ids),128):
                ix=slice(start,start+128);out=outputs[ix];base=np.array([r['risks']['full_simple'] for r in out])
                risks,axis=forward_ref(model['models'][a],roi[ix],support[ix],ds[ix],norm,base,a)
                saved=np.array([r['risks'][a] for r in out]);sa=np.array([r['axis_predictions'][a] for r in out]);predicted.extend(sa)
                maxrisk=max(maxrisk,float(np.max(np.abs(risks-saved))));maxaxis=max(maxaxis,float(np.max(np.abs(axis-sa))))
                nr,_=forward_ref(initial[a],roi[ix],support[ix],ds[ix],norm,base,a)
                np.testing.assert_allclose(nr,base,rtol=0,atol=2e-16)
            assert maxrisk<2e-5 and maxaxis<2e-3
            pp=np.array(predicted)
            for group,values in run.metrics.grouped(outputs).items():
                ix=[ids.index(r['image']) for r in values];e=(pp[ix]-target[ix])*.1;s=report['axis_metrics'][role][a][group]
                np.testing.assert_allclose(e.mean(0),s['signed_bias'],atol=1e-12)
                np.testing.assert_allclose(np.abs(e).mean(0),s['relative_MAE'],atol=1e-12)
                np.testing.assert_allclose(np.quantile(np.abs(e),.95,axis=0),s['absolute_error_P95'],atol=1e-12)
        groups=run.metrics.grouped(rows)
        for method in run.core.METHODS:
            if role=='VAL':
                req={}
                for name,values in groups.items():
                    good=sorted(r['risks'][method] for r in values if wrong(r) is False);n=math.ceil(.95*len(good))
                    req[name]=dict(good=len(good),required=n,first_whole_tie_risk_le=good[n-1] if n else None)
                assert req==report['cutoffs'][method]['requirements']
                assert report['cutoffs'][method]['risk_le']==max(v['first_whole_tie_risk_le'] for v in req.values() if v['good'])
            for group,values in groups.items():
                stat=report['statistics'][role][method][group]
                kept={r['image'] for r in values if r['pred'] is not None and r['risks'][method]<=report['cutoffs'][method]['risk_le']}
                summary(values,kept,stat['actual']);totals+=1
                auc=pair_auc(values,method);actual=stat['error_AUROC'];assert actual is auc if auc is None else abs(actual-auc)<1e-12;aucs+=1
                for control,comp in stat['controls'].items():
                    out=sorted([r for r in values if r['pred'] is not None],key=lambda r:(r['risks'][control],r['image']))
                    n=stat['actual']['accepted_outputs'];target_cr=stat['actual']['states']['CR']
                    summary(values,{r['image'] for r in out[:n]},comp['same_count']);totals+=1
                    if n:
                        boundary=out[n-1]['risks'][control];before=[r for r in out if r['risks'][control]<boundary];tie=[r for r in out if r['risks'][control]==boundary];slots=n-len(before);bw=sum(wrong(r) for r in before)
                        assert comp['same_count_tie_bounds']['bad_min']==bw+max(0,slots-sum(not wrong(r) for r in tie))
                        assert comp['same_count_tie_bounds']['bad_max']==bw+min(slots,sum(wrong(r) for r in tie))
                    seen=0;threshold=None
                    for r in out:
                        if target_cr==0:break
                        seen+=int(not wrong(r))
                        if seen>=target_cr:threshold=r['risks'][control];break
                    chosen={r['image'] for r in out if threshold is not None and r['risks'][control]<=threshold}
                    summary(values,chosen,comp['same_CR']);totals+=1
                    assert comp['same_CR_exact']==(comp['same_CR']['states']['CR']==target_cr)
                    assert comp['same_CR_offline_risk_le']==threshold
                    if role=='VAL' and method=='spatial_axis' and control in ('spatial_bce','full_simple','score_only'):
                        c=stat['actual']['states'];sc=comp['same_CR'];checks=dict(same_count_FA_nonincrease=c['FA']<=comp['same_count_tie_bounds']['bad_min'],same_CR_exact=comp['same_CR_exact'],same_CR_FA_nonincrease=c['FA']<=sc['states']['FA'],longest_correct_FR_nonincrease=stat['actual']['runs']['correct_rejection']['longest']<=sc['runs']['correct_rejection']['longest'])
                        if group=='all':checks.update(overall_strict_same_count_FA_gain=c['FA']<comp['same_count_tie_bounds']['bad_min'],overall_strict_same_CR_FA_gain=c['FA']<sc['states']['FA'])
                        gate_fail.extend(dict(group=group,control=control,check=k) for k,v in checks.items() if not v)
                        if group.startswith('sequence:real_') and control=='full_simple' and comp['same_CR_exact'] and c['FA']<sc['states']['FA']:
                            refs=[report['statistics']['VAL'][m][group]['error_AUROC'] for m in ('spatial_bce','full_simple')]
                            if auc is not None and all(v is not None and auc>v for v in refs):real_gain=True
                if role=='VAL' and method=='spatial_axis':
                    c=stat['actual']['states']
                    if c['CR']<math.ceil(.95*(c['CR']+c['FR'])):gate_fail.append(dict(group=group,check='correct_retention95'))
            videos=report['ranking'][role][method]['videos'];eligible=[]
            for name,v in videos.items():
                rr=groups[name];a=pair_auc(rr,method);assert v['error_AUROC'] is a if a is None else abs(v['error_AUROC']-a)<1e-12
                assert v['bad']==sum(wrong(r) is True for r in rr) and v['good']==sum(wrong(r) is False for r in rr)
                if a is not None:eligible.append((a,v['bad']*v['good']))
            rk=report['ranking'][role][method]
            assert abs(rk['within_video_macro_AUROC']-np.mean([a for a,n in eligible]))<1e-12
            assert abs(rk['within_video_pair_weighted_AUROC']-sum(a*n for a,n in eligible)/sum(n for a,n in eligible))<1e-12
        for group,values in groups.items():
            summary(values,{r['image'] for r in values if r['original_simple_decision']['size_accepted']},report['formal_policy_summary'][role][group]);totals+=1
            out=[r for r in values if r['pred'] is not None];hits=sum(math.hypot(r['pred'][0]-r['gt'][0],r['pred'][1]-r['gt'][1])<15 for r in out)
            assert report['center'][role][group]==dict(frames=len(values),outputs=len(out),hits=hits,hit_rate_on_outputs=hits/len(out),output_coverage=len(out)/len(values),correct_coverage_all_frames=hits/len(values))
    if not real_gain:gate_fail.append(dict(group='Real_videos',check='same_CR_FA_and_video_AUROC_gain'))
    canon=lambda v:sorted(json.dumps(x,sort_keys=True) for x in v)
    assert canon(gate_fail)==canon(report['gate']['failures'])
    assert report['gate']['passed']==(not gate_fail)
    assert report['status']==('VAL_FAILED_STOP' if gate_fail else 'VAL_PASS_FROZEN_TEST_PENDING')
    assert (p/'analysis.md').read_text()==run.core.markdown(report)
    return dict(passed=True,rows_checked=3445,summaries_checked=totals,AUROC_checked=aucs,independent_gate=True,
      independent_spatial_risk_max_error=maxrisk,independent_axis_max_error=maxaxis,status=report['status'],TEST_read=False,
      input_pins_verified=report['input_pins'],weights_sha256_verified=completion['artifacts']['final_heads.pth'],weights_present=(p/'final_heads.pth').exists(),report_sha256=run.sha(p/'report.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
    result=audit(args.directory);run.write(args.receipt,result);print(json.dumps(result))
