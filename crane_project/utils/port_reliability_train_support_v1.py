"""Offline TRAIN error-support audit; no online model, fitting or GT routing."""
from collections import Counter
from copy import deepcopy
import math
import numpy as np
from crane_project.utils import port_reliability_temporal_evidence_v1 as temporal
from crane_project.utils import port_simple_component_reliability_v1 as simple
VERSION='port_reliability_train_support_v1'
SETTINGS=dict(size_log_tolerance=.1,local_exclusion=3,level_cutoff=.0785511443740936,
    minimum_errors=20,minimum_real_videos=2,minimum_errors_per_video=5,minimum_error_runs_per_video=2)
STATES=('FA','FR','ED','CR','MISSING')


def classify(row):
    if row['pred'] is None:return dict(state='MISSING',errors=None,axes=None,direction=None,signature=None)
    pe,ge=[sorted(row[k][2:4],reverse=True) for k in ('pred','gt')]
    if any(not math.isfinite(x) or x<=0 for x in pe+ge):raise ValueError('Positive finite canonical edges required')
    errors=[(p-g)/g for p,g in zip(pe,ge)];bad=[abs(x)>.1 for x in errors]
    axes='both' if all(bad) else 'long' if bad[0] else 'short' if bad[1] else 'correct'
    direction='both_small' if max(errors)<-.1 else 'both_large' if min(errors)>.1 else 'mixed_or_single' if any(bad) else 'correct'
    sign=['small' if x<-.1 else 'large' if x>.1 else 'within' for x in errors]
    accept=row['original_simple_decision']['size_accepted']
    return dict(state=('FA' if any(bad) else 'CR') if accept else ('ED' if any(bad) else 'FR'),
        errors=errors,axes=axes,direction=direction,signature='/'.join(sign))


def regime(row):
    t=row['temporal']
    return 'unsupported' if not t['available'] else 'low' if t['primary']<=SETTINGS['level_cutoff'] else 'high'


def validate(rows):
    if not rows or any(r['split'] not in ('train','train_sim') or r['reliability_role']!='train' for r in rows):raise ValueError('TRAIN only')
    if len({r['image'] for r in rows})!=len(rows) or len({(r['sequence'],r['frame_id']) for r in rows})!=len(rows):raise ValueError('Duplicate identity')
    if rows!=sorted(rows,key=lambda r:(r['sequence'],r['frame_id'],r['image'])):raise ValueError('Canonical chronological order required')


def error_runs(rows):
    """Maximal contiguous true-size-error runs; accept flags do not split runs."""
    validate(rows);runs=[];current=None;previous=None
    for r in rows:
        bad=classify(r)['state'] in ('FA','ED')
        adjacent=previous is not None and previous['sequence']==r['sequence'] and r['frame_id']==previous['frame_id']+1 and not r['temporal'].get('acquisition_boundary',False)
        if not bad or not adjacent:current=None
        if bad:
            if current is None:
                current=dict(id='error_run_%03d'%len(runs),sequence=r['sequence'],start=r['frame_id'],end=r['frame_id'],images=[],FA=0,ED=0)
                runs.append(current)
            current['end']=r['frame_id'];current['images'].append(r['image']);current[classify(r)['state']]+=1
        previous=r
    return runs


def quantiles(values):
    return np.quantile(values,[0,.05,.5,.95,1]).tolist() if len(values) else None


def summarize(rows,runs):
    c=Counter(classify(r)['state'] for r in rows);result=dict(frames=len(rows),states={k:c[k] for k in STATES},states_detail={})
    for state in STATES:
        part=[r for r in rows if classify(r)['state']==state]
        labels=[classify(r) for r in part]
        result['states_detail'][state]=dict(frames=len(part),axes=dict(Counter(v['axes'] for v in labels if v['axes'] is not None)),
            signatures=dict(Counter(v['signature'] for v in labels if v['signature'] is not None)),directions=dict(Counter(v['direction'] for v in labels if v['direction'] is not None)),
            regimes=dict(Counter(regime(r) for r in part)),signed_error_quantiles={axis:quantiles([v['errors'][i] for v in labels if v['errors'] is not None]) for i,axis in enumerate(('long','short'))})
    ids={r['image'] for r in rows}
    result['true_error_runs']=[v for v in runs if set(v['images'])&ids]
    result['FA_containing_error_runs']=sum(any(classify(r)['state']=='FA' and r['image'] in v['images'] for r in rows) for v in result['true_error_runs'])
    return result


def audit(rows,features,ids):
    validate(rows);out=[r for r in rows if r['pred'] is not None];x=np.asarray(features,dtype=float)
    if ids!=[r['image'] for r in out] or x.shape!=(len(out),259) or not np.isfinite(x).all():raise ValueError('Full frozen native feature/output pairing required')
    desc=np.array([simple.descriptor(r['pred'],r['image_size']) for r in out])
    if not np.allclose(x[:,:3],desc,rtol=0,atol=1e-12):raise ValueError('M descriptor pairing differs')
    mean=x.mean(0);std=x.std(0);std=np.where(std<1e-8,1.,std);z=(x-mean)/std
    vectors=dict(zip(ids,z));sizes=dict(zip(ids,desc[:,1]));labels={r['image']:classify(r) for r in rows}
    runs=error_runs(rows);run_of={i:v['id'] for v in runs for i in v['images']}
    fa=[r for r in out if labels[r['image']]['state']=='FA'];ed=[r for r in out if labels[r['image']]['state']=='ED'];cr=[r for r in out if labels[r['image']]['state']=='CR']
    details=[]
    for r in fa:
        def compatible(q):
            return q['sequence']==r['sequence'] and regime(q)==regime(r) and abs(sizes[q['image']]-sizes[r['image']])<=SETTINGS['size_log_tolerance'] and abs(q['frame_id']-r['frame_id'])>SETTINGS['local_exclusion']
        semantic=[q for q in ed if labels[q['image']]['signature']==labels[r['image']]['signature']]
        matched=[q for q in semantic if compatible(q) and run_of[q['image']]!=run_of[r['image']]]
        correct=[q for q in cr if compatible(q)]
        nearest={}
        for block,sl in [('static3',slice(0,3)),('native256',slice(3,None))]:
            def closest(part):
                pairs=[(float(np.sqrt(np.mean((vectors[r['image']][sl]-vectors[q['image']][sl])**2))),q['image']) for q in part]
                return dict(distance=min(pairs)[0],image=min(pairs)[1]) if pairs else None
            donor,good=closest(matched),closest(correct)
            nearest[block]=dict(rejected_error=donor,accepted_correct=good,error_closer_than_correct=(donor['distance']<good['distance']) if donor and good else None)
        details.append(dict(image=r['image'],sequence=r['sequence'],signature=labels[r['image']]['signature'],regime=regime(r),
            error_run=run_of[r['image']],same_signature_ED_images=[q['image'] for q in semantic],
            cross_video_same_signature_ED_images=[q['image'] for q in semantic if q['sequence']!=r['sequence']],
            matched_ED_images=[q['image'] for q in matched],matched_ED_error_runs=sorted({run_of[q['image']] for q in matched}),
            matched_CR_images=[q['image'] for q in correct],nearest=nearest))
    groups={g:summarize(part,runs) for g,part in temporal.metrics.grouped(rows).items()}
    usable={q for d in details for q in d['matched_ED_images']}
    expanded=fa+[r for r in ed if r['image'] in usable]
    per_video={v:dict(errors=sum(r['sequence']==v for r in expanded),runs=len({run_of[r['image']] for r in expanded if r['sequence']==v})) for v in sorted({r['sequence'] for r in expanded})}
    supported=[v for v,d in per_video.items() if v.startswith('real_') and d['errors']>=SETTINGS['minimum_errors_per_video'] and d['runs']>=SETTINGS['minimum_error_runs_per_video']]
    sufficient=len(expanded)>=SETTINGS['minimum_errors'] and len(supported)>=SETTINGS['minimum_real_videos']
    return dict(groups=groups,error_runs=runs,remaining_FA=details,
        normalizer=dict(mean=mean.tolist(),scale=std.tolist(),role='full_TRAIN_only_descriptive_distance',not_new_PCA=True),
        support=dict(remaining_FA=len(fa),rejected_errors=len(ed),FA_with_same_signature_ED=sum(bool(d['same_signature_ED_images']) for d in details),
            FA_with_cross_video_same_signature_ED=sum(bool(d['cross_video_same_signature_ED_images']) for d in details),
            FA_with_matched_ED=sum(bool(d['matched_ED_images']) for d in details),unique_matched_ED=len(usable),expanded_error_rows=len(expanded),
            per_video=per_video,supported_real_videos=supported,material_precondition_passed=sufficient,
            direct_FA_only_precondition_passed=len(fa)>=SETTINGS['minimum_errors'] and len([v for v in groups if v.startswith('sequence:real') and groups[v]['states']['FA']>=SETTINGS['minimum_errors_per_video'] and groups[v]['FA_containing_error_runs']>=SETTINGS['minimum_error_runs_per_video']])>=SETTINGS['minimum_real_videos']),
        proximity={block:dict(comparable_FA=sum(d['nearest'][block]['error_closer_than_correct'] is not None for d in details),
            ED_closer=sum(d['nearest'][block]['error_closer_than_correct'] is True for d in details)) for block in ('static3','native256')},
        status='TRAIN_SUPPORT_AUDIT_COMPLETE_NO_MODEL',candidate_training_authorized_by_audit=False,
        limitations=['Error runs are correlated observations, not verified independent events.',
            'GT signatures/matches are offline descriptions, never online features or a learned classifier.',
            'Feature distance is scale-standardized descriptive geometry, not semantic equivalence, separability or performance.',
            'In-training detector/PCA features cannot establish unseen-video generalization.',
            'A passed material count condition does not establish useful feature information or authorize training.'])
