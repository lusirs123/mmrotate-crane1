"""Offline direction groups/weights and a single-factor loss contrast.

Online inference remains exactly ContinuousBoundarySizeHead.forward().
Groups/GT/weights exist only in offline TRAIN preparation and loss evaluation.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math

VERSION='port_geometry_size_direction_balance_v1'
ARMS=('uniform_control','direction_balanced')
GROUPS=('need_enlarge','need_shrink','near_correct','mixed')
SETTINGS=dict(near_relative=.03,weight_cap=4.,metric_tolerance=1e-6)
BASELINE_SHA='8f225449749466923527449b4af34035dce5c4a031b9c3a1bbc1eedb8329f288'
SCHEDULE_SHA='50de269e32e1d5c7280fcbe7523bde8c5c5fe8a48837d72a6f56fccfbd347615'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def edge_errors(gt,pred):
    if pred is None:return None
    if len(gt)!=5 or len(pred)!=6 or not all(math.isfinite(v) for v in list(gt)+list(pred)) or min(gt[2:4]+pred[2:4])<=0:
        raise ValueError('Expected finite original OBB and score')
    g=sorted(gt[2:4],reverse=True);p=sorted(pred[2:4],reverse=True)
    return [x/y-1. for x,y in zip(p,g)]


def group(gt,midpoint):
    errors=edge_errors(gt,midpoint)
    if errors is None:return 'missing'
    if max(map(abs,errors))<=SETTINGS['near_relative']:return 'near_correct'
    if max(errors)<0:return 'need_enlarge'
    if min(errors)>0:return 'need_shrink'
    return 'mixed'


def capped_weights(counts):
    """Equalize total group contribution per cell, subject to weight<=4.

    Solve sum_g min(cap*n_g, lambda)=sum_g n_g. Weights are constant per
    cell/group, mean one across the complete fixed schedule, never per-batch
    normalized. Empty groups cannot be repaired by weights: reject contract.
    """
    if set(counts)!=set(GROUPS) or any(not isinstance(n,int) or n<=0 for n in counts.values()):
        raise ValueError('All four groups require actual scheduled exposure')
    total=sum(counts.values());lo,hi=0.,float(total)
    for _ in range(100):
        mid=(lo+hi)/2
        if sum(min(SETTINGS['weight_cap']*n,mid) for n in counts.values())<total:lo=mid
        else:hi=mid
    weights={g:min(SETTINGS['weight_cap'],hi/counts[g]) for g in GROUPS}
    if not math.isclose(sum(counts[g]*weights[g] for g in GROUPS),total,rel_tol=1e-12,abs_tol=1e-9):
        raise ValueError('Mean-one weighted exposure failed')
    return weights


def exposure_contract(rows,schedule):
    if len(rows)!=5116:raise ValueError('Full standard/half TRAIN rows required')
    indexed={};fit=defaultdict(Counter);fit_unique=defaultdict(lambda:defaultdict(set))
    for row in rows:
        key=(row['image'],row['scale'])
        if key in indexed or row['shard'] not in ('train_s1','train_s05') or row['domain'] not in ('real','sim') or row['scale'] not in (1.,.5):
            raise ValueError('TRAIN view/identity differs; no VAL or TEST admitted')
        label=group(row['gt'],row['midpoint']);cell=row['domain']+'/'+str(row['scale'])
        indexed[key]=dict(image=row['image'],scale=row['scale'],domain=row['domain'],sequence=row['sequence'],
            role=row['sample_role'],eligible=row['eligible_train'],group=label)
        if row['sample_role']=='fit' and row['eligible_train']:
            fit[cell][label]+=1;fit_unique[cell][label].add(row['image'])
    batches=schedule['batches']
    if len(batches)!=200 or any(len(b)!=8 for b in batches):raise ValueError('Exactly fixed200x8 schedule required')
    counts=defaultdict(Counter);unique=defaultdict(lambda:defaultdict(set));chosen=Counter()
    for batch in batches:
        if len({(r['image'],r['scale']) for r in batch})!=8:raise ValueError('Duplicate view in batch')
        cells=Counter()
        for entry in batch:
            key=(entry['image'],entry['scale']);r=indexed[key]
            if r['role']!='fit' or not r['eligible'] or r['group']=='missing':raise ValueError('Non-fit/ineligible/missing exposure')
            cell=r['domain']+'/'+str(r['scale']);counts[cell][r['group']]+=1;unique[cell][r['group']].add(r['image']);chosen[key]+=1;cells[cell]+=1
        if cells!=Counter({d+'/'+str(s):2 for d in ('real','sim') for s in (1.,.5)}):raise ValueError('Domain/scale slots changed')
    cell_reports={}
    for cell in sorted(counts):
        weights=capped_weights(dict(counts[cell]))
        cell_reports[cell]=dict(eligible_fit_views=dict(fit[cell]),slots=dict(counts[cell]),
            unique_scheduled_views={g:len(unique[cell][g]) for g in GROUPS},
            weights=weights,weighted_slots={g:counts[cell][g]*weights[g] for g in GROUPS})
    scheduled=[dict(indexed[k],times=n,weight=cell_reports[indexed[k]['domain']+'/'+str(k[1])]['weights'][indexed[k]['group']]) for k,n in sorted(chosen.items())]
    return dict(protocol=VERSION,settings=SETTINGS,original_baseline_sha256=BASELINE_SHA,
        original_schedule_sha256=SCHEDULE_SHA,slots=1600,unique_scheduled_views=len(chosen),cells=cell_reports,
        scheduled_views=scheduled,all_train_group_identity_sha256=fingerprint([indexed[k] for k in sorted(indexed)]),
        original_batches_fingerprint=fingerprint(batches),test_or_val_used=False,
        note='Counts describe exposures, not gradient signs or independent events. Existing TRAIN roles, sampler, order and repeats unchanged.')


def loss(output,gt,midpoint,boxes_model,weights,arm):
    if arm not in ARMS:raise ValueError('Unknown contrast arm')
    import torch
    from torch.nn import functional as F
    from crane_project.utils import port_geometry_size_boundary_continuous_v1 as model
    terms=model.continuous_loss(output,gt,midpoint,boxes_model)
    delta=output['delta_roi'];w=delta.new_tensor(weights).detach()
    if w.shape!=(len(delta),) or not bool(torch.isfinite(w).all()) or not bool(((w>0)&(w<=SETTINGS['weight_cap'])).all()):
        raise ValueError('Invalid fixed TRAIN weights')
    if arm=='uniform_control' and not bool((w==1).all()):raise ValueError('Control weights must be exactly one')
    pointwise=F.smooth_l1_loss(delta,terms['target']['log_residual'],beta=model.SETTINGS['decoded_beta'],reduction='none')
    if arm=='direction_balanced':terms['total']=(pointwise*w[:,None]).mean()
    terms.update(unweighted=terms['decoded'],decoded=terms['total'],per_view_unweighted=pointwise.mean(-1),weights=w)
    return terms


def direction_summary(rows):
    """Fixed M group membership; evaluate actual guarded delivered sizes."""
    groups=defaultdict(list)
    for r in rows:
        if r['sample_role'] not in ('fit','probe') or (r['sample_role']=='fit' and not r['eligible_train']):continue
        label=group(r['gt'],r['midpoint'])
        if label=='missing':continue
        groups[(r['shard'],r['sample_role'],r['domain'],label)].append(r)
    summaries={}
    def stats(selected,method):
        values=[edge_errors(r['gt'],r[method]) for r in selected]
        if any(v is None for v in values):raise ValueError('Delivery lost original output')
        logs=[[math.log1p(e) for e in v] for v in values]
        def p95(v):
            # Match the existing metrics percentile (linear interpolated p95).
            s=sorted(v);pos=(len(s)-1)*.95;low=int(pos);return s[low]+(s[min(low+1,len(s)-1)]-s[low])*(pos-low)
        out=dict(frames=len(values),joint10=sum(max(map(abs,e))<=.1 for e in values),
            common_log_mae=sum(abs(sum(e)/2) for e in logs)/len(logs),
            abs_common_log_bias=abs(sum(sum(e)/2 for e in logs)/len(logs)),
            abs_short_log_bias=abs(sum(e[1] for e in logs)/len(logs)),
            ratio_log_mae=sum(abs(e[0]-e[1]) for e in logs)/len(logs),
            ratio_log_p95=p95([abs(e[0]-e[1]) for e in logs]))
        for i,edge in enumerate(('long','short')):
            out[edge+'_mae']=sum(abs(e[i]) for e in values)/len(values)
            out[edge+'_p95']=p95([abs(e[i]) for e in values])
        return out
    for key,selected in sorted(groups.items()):
        a,b=stats(selected,'midpoint'),stats(selected,'edge_residual')
        wrong=sum(max(map(abs,edge_errors(r['gt'],r['midpoint'])))<=.1 and max(map(abs,edge_errors(r['gt'],r['edge_residual'])))>.1 for r in selected)
        summaries['/'.join(key)]=dict(midpoint=a,corrected=b,new_size10_errors_from_correct=wrong,
            delivered_group_identity=fingerprint(sorted(r['image'] for r in selected)))
    return summaries


def direction_gates(summary):
    checks={};tol=SETTINGS['metric_tolerance']
    for shard in ('train_s1','train_s05'):
        for role in ('fit','probe'):
            for domain in ('real','sim'):
                for label in GROUPS:
                    key='/'.join((shard,role,domain,label));value=summary.get(key)
                    # Empty probe groups are explicitly unassessed, never passed.
                    if value is None:
                        checks[key+'/support_present']=False;continue
                    a,b=value['midpoint'],value['corrected']
                    for metric in ('common_log_mae','abs_common_log_bias','abs_short_log_bias','ratio_log_mae','ratio_log_p95',
                                   'long_mae','long_p95','short_mae','short_p95'):
                        checks[key+'/'+metric]=math.isfinite(b[metric]) and b[metric]<=a[metric]+tol
                    checks[key+'/joint10']=b['joint10']>=a['joint10']
                    if label=='near_correct':checks[key+'/no_new_size10_error']=value['new_size10_errors_from_correct']==0
                    if role=='fit' and label in ('need_enlarge','need_shrink'):
                        checks[key+'/common_scale_strict_gain']=b['common_log_mae']<a['common_log_mae']-tol
    failed=[k for k,v in checks.items() if not v]
    return dict(checks=checks,failed_checks=failed,direction_gate_pass=not failed,
        status='DIRECTION_GATE_PASS' if not failed else 'DIRECTION_GATE_FAIL_KEEP_M',
        note='Direction is assessed by error reduction, not requiring every box to have a particular update sign.')
