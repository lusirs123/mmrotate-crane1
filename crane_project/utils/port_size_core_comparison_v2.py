"""Paired reference accuracy and SIZE ranking; fixed 95% comparison, no deployment."""
import math
import numpy as np
from crane_project.utils import port_size_core_curvature_v2 as core
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_separability_v1 as rank

METHODS = ('score','simple','a0','a1')


def state(rows, kept, frames):
    if len(rows) != len(kept): raise ValueError('Wrong decision population')
    good = [not r['size_bad'] for r in rows]
    cr = sum(g and k for g,k in zip(good,kept)); fa = sum(not g and k for g,k in zip(good,kept))
    fr = sum(g and not k for g,k in zip(good,kept)); ed = sum(not g and not k for g,k in zip(good,kept))
    def ratio(a,b): return a/b if b else None
    return dict(FA=fa,FR=fr,ED=ed,CR=cr,output_frames=len(rows),all_frames=frames,
        accepted=fa+cr,acceptance_coverage=ratio(fa+cr,frames),
        full_frame_correct_coverage=ratio(cr,frames),state_accuracy_on_outputs=ratio(cr+ed,len(rows)),
        bad_detection_rate=ratio(ed,ed+fa),good_retention_rate=ratio(cr,cr+fr),
        correctness_on_accepted=ratio(cr,cr+fa))


def annotate(source, readings, role):
    if role not in ('reference_holdout_train','val') or source['split'] not in (
            ('train','train_sim') if role != 'val' else ('val',)):
        raise ValueError('No TEST or unknown assessment role')
    g = simple.canonical(source['gt']); output=source['pred'] is not None
    errors = simple.geometry_errors(source['gt'],source['pred']) if output else None
    result=dict(source,assessment_role=role,size_bad=None if errors is None else errors['size_max_relative']>.1,
        center_hit=False if errors is None else errors['center_px']<15.,arms=readings,
        frozen=readings['a0']['frozen'],GT_used_only_offline=True)
    if (readings['a0']['frozen'] != readings['a1']['frozen'] or
            readings['a0']['frozen_score_only'] != readings['a1']['frozen_score_only'] or
            readings['a0']['frozen']['final_box_original'] != source['pred']):
        raise ValueError('Arms changed frozen baseline flags')
    result['frozen_score_only']=readings['a0']['frozen_score_only']
    for arm in core.ARMS:
        ref=readings[arm]['reference']; item=result['arms'][arm]
        item['reference_error']=None; item['probes']=None
        if ref['defined']:
            lengths=np.array([ref['long_original_px'],ref['short_original_px']])
            item['reference_error']=dict(max_relative=float(np.max(np.abs(lengths/g[2:4]-1.))),
                long_ratio=float(lengths[0]/g[2]),short_ratio=float(lengths[1]/g[3]))
            # Near-squares stay in ALL real-frame metrics, just excluded from edge-order probes.
            if g[2]/g[3]>=1.2:
                base=np.abs(np.log(g[2:4]/lengths)); gaps={}
                for edge,label in ((0,'long'),(1,'short')):
                    gaps[label]=[float(abs(math.log(g[2+edge]*f/lengths[edge]))-base[edge]) for f in (.85,1.15)]
                item['probes']=dict(gaps=gaps,long_both=min(gaps['long'])>1e-6,
                    short_both=min(gaps['short'])>1e-6)
    return result


def risks(row):
    if row['pred'] is None: return {m:None for m in METHODS}
    return dict(score=1-row['pred'][5],simple=row['frozen']['risks']['size'],
                **{a:row['arms'][a]['reading']['risk'] if row['arms'][a]['reference']['defined'] else None
                   for a in core.ARMS})


def diagnostic_cutoffs(records):
    val=[r for r in records if r['assessment_role']=='val']
    result={}
    for arm in core.ARMS:
        values=[risks(r)[arm] for r in val if risks(r)[arm] is not None]
        count=math.ceil(.95*len(val))
        reachable=len(values)>=count
        result[arm]=dict(reachable=reachable,requested_count=count,available_outputs=len(values),
            risk_le=float(np.sort(values)[count-1]) if reachable else None,
            role='fixed_comparison_workpoint_not_final_deployment_policy',GT_labels_used=False,
            ties='accept_entire_cutoff_tie')
    return result


def matched(rows,count,frames,methods=METHODS):
    report={}
    for method in methods:
        available=[r for r in rows if risks(r)[method] is not None]
        ordered=sorted(available,key=lambda r:(risks(r)[method],r['image']))
        kept={r['image'] for r in ordered[:count]}
        value=state(rows,[r['image'] in kept for r in rows],frames)
        value.update(requested_count=count,reachable=len(available)>=count,
                     available_outputs=len(available),accepted_images_sha256=simple.fingerprint(sorted(kept)))
        # Both extremes at boundary; never tie-break by GT.
        value['tie_bounds']=rank.same_count_reference([risks(r)[method] for r in available],
            [r['size_bad'] for r in available],[r['image'] for r in available],[min(count,len(available))])[min(count,len(available))]
        report[method]=value
    return report


def reference_stats(rows,arm,total,all_output_count=None):
    available=[r for r in rows if r['arms'][arm]['reference']['defined']]
    unavailable=[r for r in rows if not r['arms'][arm]['reference']['defined']]
    errors=[r['arms'][arm]['reference_error']['max_relative'] for r in available]
    probes=[r['arms'][arm]['probes'] for r in available if r['arms'][arm]['probes'] is not None]
    hits=sum(e<=.1 for e in errors)
    denominator=len(rows) if all_output_count is None else all_output_count
    return dict(support_outputs=len(rows),all_output_denominator=denominator,
        defined=len(available),unavailable_good=sum(not r['size_bad'] for r in unavailable),
        unavailable_bad=sum(r['size_bad'] for r in unavailable),
        unavailable_images=[r['image'] for r in unavailable],
        support_sha256=simple.fingerprint([r['image'] for r in available]),
        mean=float(np.mean(errors)) if errors else None,p90=float(np.percentile(errors,90)) if errors else None,
        within_10=hits,within_10_fraction=hits/len(available) if available else None,
        all_output_reference_correct_coverage=hits/denominator if denominator else None,
        all_frame_reference_correct_coverage=hits/total if total else None,
        probe_eligible=len(probes),long_both_fraction=sum(p['long_both'] for p in probes)/len(probes) if probes else None,
        short_both_fraction=sum(p['short_both'] for p in probes)/len(probes) if probes else None)


def group_summary(values,cutoffs):
    rows=[r for r in values if r['pred'] is not None]; frames=len(values)
    common=[r for r in rows if all(r['arms'][a]['reference']['defined'] for a in core.ARMS)]
    count=sum(r['frozen']['size_accepted'] for r in rows)
    own={a:[r for r in rows if r['arms'][a]['reference']['defined']] for a in core.ARMS}
    support={'common':common,'a0_own':own['a0'],'a1_own':own['a1']}
    ranks={}
    for name,population in support.items():
        ranks[name]=dict(outputs=len(population),bad=sum(r['size_bad'] for r in population),
            support_sha256=simple.fingerprint([r['image'] for r in population]),
            rank={m:rank.rank_metrics([risks(r)[m] for r in population],[r['size_bad'] for r in population])
                  for m in METHODS if all(risks(r)[m] is not None for r in population)})
    workpoints={'simple':state(rows,[r['frozen']['size_accepted'] for r in rows],frames),
                'score_only':state(rows,[r['frozen_score_only']['size_accepted'] for r in rows],frames),
                'raw':state(rows,[True]*len(rows),frames)}
    for arm in core.ARMS:
        threshold=cutoffs[arm]['risk_le']
        workpoints[arm]=state(rows,[threshold is not None and risks(r)[arm] is not None
                                   and risks(r)[arm]<=threshold for r in rows],frames)
        workpoints[arm]['coverage_goal_reachable']=cutoffs[arm]['reachable']
    points={}
    for fraction in (.9,.95):
        k=min(len(rows),math.ceil(frames*fraction))
        points[str(fraction)]=dict(requested_count=math.ceil(frames*fraction),
            capped_output_count=k,full_output=matched(rows,k,frames),
            common_support=matched(common,min(k,len(common)),frames))
    all_frame_points={m:matched(rows,workpoints[m]['accepted'],frames) for m in ('a0','a1')}
    hits=sum(r['center_hit'] for r in rows); good=sum(not r['size_bad'] for r in rows)
    lower_bound = core.acceptance_bound(frames,len(rows),good,min(len(rows),math.ceil(.95*frames)))
    return dict(frames=frames,outputs=len(rows),missing_outputs=frames-len(rows),
        output_coverage=len(rows)/frames if frames else None,
        center_hit_rate_on_outputs=hits/len(rows) if rows else None,
        full_frame_center_correct_coverage=hits/frames if frames else None,
        correct_size_outputs=good,comparison_95pct_acceptance_bound=lower_bound,
        ranks=ranks,references={a:reference_stats(rows,a,frames) for a in core.ARMS},
        common_reference={a:reference_stats(common,a,frames,len(rows)) for a in core.ARMS},
        frozen_simple_count=count,primary_same_count=matched(rows,count,frames),
        fixed_coverage_points=points,global_comparison_workpoints=workpoints,
        candidate_actual_count_matched=all_frame_points,
        workpoint_role='comparison_only_final_accuracy_workpoint_deferred')


def summarize(records):
    if len({r['image'] for r in records}) != len(records): raise ValueError('Duplicate image identity')
    if {r['assessment_role'] for r in records} != {'reference_holdout_train','val'}:
        raise ValueError('Requires both fixed assessment roles')
    cutoffs=diagnostic_cutoffs(records); summaries={}
    for role in ('reference_holdout_train','val'):
        values=[r for r in records if r['assessment_role']==role]
        groups={'all':values}
        for field in ('domain','sequence'):
            groups.update({field+':'+k:[r for r in values if r[field]==k] for k in sorted({r[field] for r in values})})
        summaries[role]={name:group_summary(v,cutoffs) for name,v in groups.items()}
    return dict(summary=summaries,comparison_workpoints=cutoffs,
                final_deployment_policy_created=False,
                final_workpoint_requires_new_prespecified_TRAIN_VAL_good_retention_or_false_rejection_version=True)
