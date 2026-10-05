"""Frozen SIZE ranking, residual-distribution diagnostics and fixed gates."""
from copy import deepcopy
import math
import numpy as np
from crane_project.utils import port_size_residual_v4 as core
from crane_project.utils import port_reliability_separability_v1 as rank
from crane_project.utils import port_simple_component_reliability_v1 as simple

METHODS=('score','simple','a0','a1')
ROLES=('residual_holdout_train','val')


def annotate(source, evidence, role):
    if role not in ROLES or source['split'] not in (('train','train_sim') if role!= 'val' else ('val',)):
        raise ValueError('Only residual-held TRAIN and VAL may be assessed')
    if evidence['frozen']['final_box_original']!=source['pred'] or evidence['frozen_score_only']['final_box_original']!=source['pred']:
        raise ValueError('Evidence used another final box')
    error=simple.geometry_errors(source['gt'],source['pred']) if source['pred'] is not None else None
    return dict(deepcopy(source),**deepcopy(evidence),assessment_role=role,
        size_bad=error['size_max_relative']>.1 if error is not None else None,
        center_hit=error['center_px']<15. if error is not None else False,
        residual=None if error is None else core.residual_target(source['pred'],source['gt']).tolist(),
        GT_used_only_offline=True)


def risks(row):
    if row['pred'] is None: return {m:None for m in METHODS}
    return dict(score=1-row['pred'][5],simple=row['frozen']['risks']['size'],
        **{a:row['arms'][a]['risk'] if row['arms'][a]['defined'] else None for a in core.ARMS})


def state(rows, kept, frames):
    if len(rows)!=len(kept) or any(r['pred'] is None for r in rows):
        raise ValueError('State counts require the complete output population')
    good=[not r['size_bad'] for r in rows]
    cr=sum(g and k for g,k in zip(good,kept)); fa=sum(not g and k for g,k in zip(good,kept))
    fr=sum(g and not k for g,k in zip(good,kept)); ed=sum(not g and not k for g,k in zip(good,kept))
    def ratio(a,b):return a/b if b else None
    return dict(FA=fa,FR=fr,ED=ed,CR=cr,accepted=fa+cr,output_frames=len(rows),all_frames=frames,
        acceptance_coverage=ratio(fa+cr,frames),correctness_on_accepted=ratio(cr,fa+cr),
        good_retention_rate=ratio(cr,cr+fr),bad_detection_rate=ratio(ed,fa+ed),
        full_frame_correct_size_coverage=ratio(cr,frames),state_accuracy_on_outputs=ratio(cr+ed,len(rows)))


def matched(rows,count,frames):
    if count<0 or count>len(rows):raise ValueError('Invalid matched acceptance count')
    result={}
    for method in METHODS:
        available=[r for r in rows if risks(r)[method] is not None]
        ordered=sorted(available,key=lambda r:(risks(r)[method],r['image']))
        kept={r['image'] for r in ordered[:count]}; actual=min(count,len(available))
        value=state(rows,[r['image'] in kept for r in rows],frames)
        bounds=rank.same_count_reference([risks(r)[method] for r in available],
            [r['size_bad'] for r in available],[r['image'] for r in available],[actual])[actual]
        value.update(requested_count=count,reachable=len(available)>=count,available_outputs=len(available),
            accepted_images_sha256=simple.fingerprint(sorted(kept)),tie_bounds=bounds)
        # At fixed k on the same complete output population, FR = FA + good - k.
        value['FR_tie_bounds']={label:bounds['bad_'+label+'_over_tie']+sum(not r['size_bad'] for r in rows)-actual
                              for label in ('min','max')}
        result[method]=value
    return result


def comparison_cutoffs(records):
    val=[r for r in records if r['assessment_role']=='val']; count=math.ceil(.95*len(val)); result={}
    for arm in core.ARMS:
        values=[risks(r)[arm] for r in val if risks(r)[arm] is not None]
        result[arm]=dict(requested_count=count,available_outputs=len(values),reachable=len(values)>=count,
            risk_le=float(np.sort(values)[count-1]) if len(values)>=count else None,
            GT_used=False,role='comparison_only_no_deployment_policy',ties='accept_whole_boundary_tie')
    return result


def distribution_stats(rows,arm):
    values=[r for r in rows if r['arms'][arm]['defined']]
    if not values:return dict(outputs=0,mean_nll=None)
    e=np.asarray([r['residual'] for r in values]); sigma=np.asarray([r['arms'][arm]['sigma'] for r in values])
    if e.shape!=sigma.shape or not np.isfinite(sigma).all() or np.any(sigma<=0):
        raise ValueError('Invalid saved dispersion')
    nll=.5*((e/sigma)**2+2*np.log(sigma)+math.log(2*math.pi)).mean(1)
    truth=np.asarray([not r['size_bad'] for r in values],dtype=float)
    good=np.asarray([r['arms'][arm]['model_good_mass'] for r in values]); bins=[]
    for i in range(10):
        selected=(good>=i/10)&((good<(i+1)/10) if i!=9 else (good<=1))
        bins.append(dict(lower=i/10,upper=(i+1)/10,outputs=int(selected.sum()),
            mean_model_good_mass=float(good[selected].mean()) if selected.any() else None,
            actual_good_fraction=float(truth[selected].mean()) if selected.any() else None))
    correlation=float(np.corrcoef(e.T)[0,1]) if len(e)>1 and np.all(e.std(0)>1e-12) else None
    distance=np.sum((e/sigma)**2,1)
    return dict(outputs=len(values),mean_nll=float(nll.mean()),p90_nll=float(np.percentile(nll,90)),
        signed_residual_mean=e.mean(0).tolist(),residual_axis_correlation=correlation,
        mean_sigma=sigma.mean(0).tolist(),mean_squared_residual=np.mean(e**2,0).tolist(),
        mean_variance=np.mean(sigma**2,0).tolist(),
        model_good_mass_brier=float(np.mean((good-truth)**2)),model_good_mass_bins=bins,
        joint_zero_mean_ellipse_coverage={str(q):float(np.mean(distance<=-2*math.log(1-q)))
            for q in (.5,.8,.9,.95,.99)},
        probability_calibrated=False,not_a_threshold_fitting_set=True)


def group_summary(values,cutoffs):
    rows=[r for r in values if r['pred'] is not None]; frames=len(values)
    common=[r for r in rows if all(r['arms'][a]['defined'] for a in core.ARMS)]
    count=sum(r['frozen']['size_accepted'] for r in rows)
    workpoints={m:state(rows,[r[field]['size_accepted'] for r in rows],frames)
                for m,field in (('simple','frozen'),('score_only','frozen_score_only'))}
    workpoints['raw']=state(rows,[True]*len(rows),frames)
    for arm in core.ARMS:
        cutoff=cutoffs[arm]['risk_le']
        workpoints[arm]=state(rows,[cutoff is not None and risks(r)[arm] is not None and risks(r)[arm]<=cutoff for r in rows],frames)
        workpoints[arm]['coverage_goal_reachable']=cutoffs[arm]['reachable']
    ranks={name:dict(outputs=len(v),bad=sum(r['size_bad'] for r in v),
        support_sha256=simple.fingerprint([r['image'] for r in v]),
        methods={m:rank.rank_metrics([risks(r)[m] for r in v],[r['size_bad'] for r in v])
                 for m in METHODS if all(risks(r)[m] is not None for r in v)})
        for name,v in [('common',common),('all_outputs',rows)]}
    good=sum(not r['size_bad'] for r in rows); requested=math.ceil(.95*frames)
    return dict(frames=frames,outputs=len(rows),missing_outputs=frames-len(rows),
        output_coverage=len(rows)/frames if frames else None,
        center_hit_rate_on_outputs=sum(r['center_hit'] for r in rows)/len(rows) if rows else None,
        full_frame_center_correct_coverage=sum(r['center_hit'] for r in rows)/frames if frames else None,
        good_size_outputs=good,bad_size_outputs=len(rows)-good,
        availability={a:dict(defined=sum(r['arms'][a]['defined'] for r in rows),
            unavailable_good=sum(not r['size_bad'] and not r['arms'][a]['defined'] for r in rows),
            unavailable_bad=sum(r['size_bad'] and not r['arms'][a]['defined'] for r in rows)) for a in core.ARMS},
        ranks=ranks,common_distribution={a:distribution_stats(common,a) for a in core.ARMS},
        own_distribution={a:distribution_stats(rows,a) for a in core.ARMS},
        frozen_simple_count=count,primary_same_count=matched(rows,count,frames),
        fixed_coverage_points={str(f):dict(requested_count=math.ceil(f*frames),
            full_output=matched(rows,min(len(rows),math.ceil(f*frames)),frames)) for f in core.SETTINGS['coverage_fractions']},
        global_comparison_workpoints=workpoints,
        candidate_actual_count_matched={a:matched(rows,workpoints[a]['accepted'],frames) for a in core.ARMS},
        comparison_95pct_bound=dict(requested_count=requested,reachable=len(rows)>=requested,
            minimum_FA=max(0,requested-good) if len(rows)>=requested else None),
        no_final_workpoint=True)


def continuation_review(summary):
    val=summary['val'];tol=core.CONTINUATION['tolerance'];Q={};D={};strict=[]
    for name,g in val.items():
        if name=='all':continue
        points=g['primary_same_count']; candidate=points['a1'];controls=('score','simple','a0')
        reachable=all(points[m]['reachable'] for m in controls+('a1',))
        gain=(min(points[m]['tie_bounds']['bad_min_over_tie'] for m in controls)
              -candidate['tie_bounds']['bad_max_over_tie']) if reachable else None
        min_gain=core.CONTINUATION['real_FA_gain'] if name=='domain:real' else 0
        Q[name+'_conservative_FA_gain']=gain is not None and gain>=min_gain
        Q[name+'_conservative_FR_nonincrease']=bool(reachable and candidate['FR_tie_bounds']['max']<=
            min(points[m]['FR_tie_bounds']['min'] for m in controls))
        Q[name+'_availability_all_outputs']=g['availability']['a1']['defined']==g['outputs']
        if name.startswith('sequence:real_'):strict.append(gain is not None and gain>0)
        if name.startswith('domain:'):
            domain=name.split(':')[1]
            Q[domain+'_frozen_primary_count']=g['frozen_simple_count']==core.CONTINUATION['primary_accept_counts'][domain]
    aucs=val['domain:real']['ranks']['common']['methods']
    controls=[aucs[m]['error_auroc'] for m in ('score','simple','a0')]; candidate=aucs['a1']['error_auroc']
    Q['real_AUROC_gain_002']=bool(candidate is not None and all(x is not None for x in controls) and
        candidate>=max(controls)+core.CONTINUATION['real_AUROC_gain']-tol)
    Q['one_real_video_strict_gain']=any(strict)
    for role,groups in summary.items():
        for name in ('domain:real','domain:sim'):
            d=groups[name]['common_distribution'];a,b=d['a0']['mean_nll'],d['a1']['mean_nll']
            D[role+'_'+name+'_NLL_nonincrease']=a is not None and b is not None and b<=a+tol
    return dict(predeclared=core.CONTINUATION,Q1=dict(conditions=Q,passed=all(Q.values())),
        D=dict(conditions=D,passed=all(D.values()),probability_calibration_proven=False),
        eligible_to_design_separate_constrained_workpoint=all(Q.values()) and all(D.values()),
        final_policy_created=False,automatic_deployment_or_TEST=False)


def summarize(records):
    if len({r['image'] for r in records})!=len(records) or {r['assessment_role'] for r in records}!=set(ROLES):
        raise ValueError('Unique fixed images and both assessment roles required')
    for r in records:
        if r['split'] not in (('train','train_sim') if r['assessment_role']!='val' else ('val',)):
            raise ValueError('TEST/unknown split forbidden')
    cutoffs=comparison_cutoffs(records);summary={}
    for role in ROLES:
        values=[r for r in records if r['assessment_role']==role];groups={'all':values}
        for field in ('domain','sequence'):
            groups.update({field+':'+key:[r for r in values if r[field]==key] for key in sorted({r[field] for r in values})})
        summary[role]={name:group_summary(rows,cutoffs) for name,rows in groups.items()}
    return dict(summary=summary,comparison_workpoints=cutoffs,
        prespecified_continuation_review=continuation_review(summary),
        final_deployment_policy_created=False,calibration_parameters_fitted=False)
