"""Existing fixed comparison plus signed size/profile and preset R/Q1 review."""
from collections import Counter
import numpy as np
from crane_project.utils import port_size_core_comparison_v2 as original

annotate = original.annotate
CONTINUATION = dict(real_mean_relative_reduction=.2, real_correct_fraction_gain=.1,
    real_FA_gain=5, real_AUROC_gain=.02, real_probe_both_min=.6,
    primary_accept_counts={'real':333,'sim':510}, tolerance=1e-10,
    roles='R_reference_Q1_ranking_Q2_95pct_comparison_only')


def diagnostics(rows):
    """No fitted corrections: measured ratios/availability/proxy losses only."""
    output=[r for r in rows if r['pred'] is not None];result={}
    common=[r for r in output if all(r['arms'][a]['reference']['defined'] for a in ('a0','a1'))]
    for arm in ('a0','a1'):
        ratios={edge:[r['arms'][arm]['reference_error'][edge+'_ratio'] for r in common]
                for edge in ('long','short')}
        proxy=[r['arms'][arm]['offline_range_proxy'] for r in rows
               if r['arms'][arm].get('offline_range_proxy',{}).get('eligible')]
        result[arm]=dict(common_outputs=len(common),signed_reference_to_GT={
            k:dict(p10=float(np.percentile(v,10)),median=float(np.median(v)),p90=float(np.percentile(v,90)))
            if v else None for k,v in ratios.items()},
            unavailable_reasons=dict(Counter(r['arms'][arm]['reference']['reason']
                for r in output if not r['arms'][arm]['reference']['defined'])),
            offline_GT_proxy_frames=len(proxy),proxy_total_frames=len(rows),
            proxy_mean={k:float(np.mean([p[k] for p in proxy])) if proxy else None
                        for k in ('inner','outer','mean_absolute_profile_error')},
            proxy_contrast_floor_active=sum(p['contrast_floor_active'] for p in proxy),
            GT_proxy_used_for_online_flags=False)
    return result


def continuation_review(summary):
    """Preset practical conditions, not significance or automatic deployment."""
    tol=CONTINUATION['tolerance'];R={};Q={};val=summary['val'];held=summary['reference_holdout_train']
    real=val['domain:real'];r0=real['common_reference']['a0'];r1=real['common_reference']['a1']
    def available_pair(a,b,field):
        return a[field] is not None and b[field] is not None
    R['real_common_mean_20pct']=bool(available_pair(r0,r1,'mean') and r0['mean']>0 and
        r1['mean']<=r0['mean']*(1-CONTINUATION['real_mean_relative_reduction'])+tol)
    R['real_common_p90_nonincrease']=bool(available_pair(r0,r1,'p90') and r1['p90']<=r0['p90']+tol)
    R['real_common_correct_gain_10pp']=bool(available_pair(r0,r1,'within_10_fraction') and
        r1['within_10_fraction']>=r0['within_10_fraction']+CONTINUATION['real_correct_fraction_gain']-tol)
    R['real_all_output_correct_gain_10pp']=bool(real['references']['a1']['all_output_reference_correct_coverage'] is not None and
        real['references']['a1']['all_output_reference_correct_coverage']>=
        real['references']['a0']['all_output_reference_correct_coverage']+CONTINUATION['real_correct_fraction_gain']-tol)
    for role,groups,names in [('val',val,['domain:sim']),('holdout',held,['domain:real','domain:sim'])]:
        for name in names:
            group=groups[name];a,b=(group['common_reference'][m] for m in ('a0','a1'))
            for field in ('mean','p90'):
                R[role+'_'+name+'_'+field+'_nonincrease']=bool(available_pair(a,b,field) and b[field]<=a[field]+tol)
            a,b=(group['references'][m] for m in ('a0','a1'))
            R[role+'_'+name+'_correct_coverage_nondecrease']=bool(a['all_output_reference_correct_coverage'] is not None and
                b['all_output_reference_correct_coverage'] is not None and
                b['all_output_reference_correct_coverage']>=a['all_output_reference_correct_coverage']-tol)
    for role,groups in [('val',val),('holdout',held)]:
        for name in ('domain:real','domain:sim'):
            a,b=(groups[name]['references'][m] for m in ('a0','a1'))
            R[role+'_'+name+'_availability_nondecrease']=b['defined']>=a['defined']
    for role,groups in [('val',val),('holdout',held)]:
        a,b=(groups['domain:real']['common_reference'][m] for m in ('a0','a1'))
        for field in ('long_both_fraction','short_both_fraction'):
            R[role+'_real_'+field]=bool(available_pair(a,b,field) and
                b[field]>=CONTINUATION['real_probe_both_min']-tol and b[field]>=a[field]-tol)
    for name,group in val.items():
        if name.startswith('sequence:'):
            a,b=(group['references'][m] for m in ('a0','a1'))
            R[name+'_correct_coverage_nondecrease']=bool(available_pair(a,b,'all_output_reference_correct_coverage') and
                b['all_output_reference_correct_coverage']>=a['all_output_reference_correct_coverage']-tol)
    def conservative_gain(group,controls):
        points=group['primary_same_count'];candidate=points['a1']
        if not all(points[m]['reachable'] for m in ('a1',)+controls):return None
        return min(points[m]['tie_bounds']['bad_min_over_tie'] for m in controls)-candidate['tie_bounds']['bad_max_over_tie']
    for domain in ('real','sim'):
        group=val['domain:'+domain];gain=conservative_gain(group,('score','simple','a0'))
        Q[domain+'_fixed_primary_count']=group['frozen_simple_count']==CONTINUATION['primary_accept_counts'][domain]
        Q[domain+'_same_count_gain']=gain is not None and gain>=(CONTINUATION['real_FA_gain'] if domain=='real' else 0)
    gains=[]
    for name,group in val.items():
        if name.startswith('sequence:'):
            gain=conservative_gain(group,('score','simple'))
            Q[name+'_not_worse_than_score_simple']=gain is not None and gain>=0
            if name.startswith('sequence:real_'):gains.append(gain)
    Q['at_least_one_real_video_strict_gain']=any(g is not None and g>0 for g in gains)
    rank=real['ranks']['common']['rank'];aucs={k:rank[k]['error_auroc'] for k in ('score','simple','a1')}
    Q['real_common_AUROC_gain_002']=bool(all(v is not None for v in aucs.values()) and
        aucs['a1']>=max(aucs['score'],aucs['simple'])+CONTINUATION['real_AUROC_gain']-tol)
    rpass=all(R.values());qpass=all(Q.values())
    return dict(predeclared=CONTINUATION,R=dict(conditions=R,passed=rpass),Q1=dict(conditions=Q,passed=qpass),
        eligible_to_design_separate_constrained_final_workpoint=rpass and qpass,
        final_policy_created=False,automatic_deployment_or_TEST=False)


def summarize(records):
    result=original.summarize(records)
    extra={}
    for role,groups in result['summary'].items():
        values=[r for r in records if r['assessment_role']==role];extra[role]={}
        for name in groups:
            subset=values if name=='all' else [r for r in values if r[name.split(':',1)[0]]==name.split(':',1)[1]]
            extra[role][name]=diagnostics(subset)
    result['range_and_signed_diagnostics']=extra
    result['prespecified_continuation_review']=continuation_review(result['summary'])
    return result
