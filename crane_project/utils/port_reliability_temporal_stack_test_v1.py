"""Frozen temporal stack replay. Scoring consumes no GT; TEST never fits."""
from copy import deepcopy
import math
import numpy as np
from crane_project.utils import port_reliability_partial_auc_v1 as pauc
from crane_project.utils import port_reliability_image_temporal_crossfit_v1 as fusion
from crane_project.utils import port_reliability_readout_test_v1 as test
from crane_project.utils import port_reliability_fallback_decision_v1 as fallback

VERSION='port_reliability_temporal_stack_test_v1'
METHODS=('simple','temporal_veto','fusion4','partial_auc4','crossfit_fusion','fallback_margin')
CONTROLS=('full_simple','score_only','fusion4','partial_auc4','crossfit_fusion','fallback_margin')
base=pauc.base


def score(pred,image_size,point,image_risk,bundle):
    """Current M/history/image risk/frozen parameters only. Never accepts labels."""
    risks=base.online_risks(pred,image_size,point,bundle['models'])
    risks['partial_auc4']=pauc.online_risk(pred,image_size,point,bundle['models'],bundle['pauc'])
    risks['crossfit_fusion']=fusion.fused_risk(risks['partial_auc4'],image_risk,bundle['fusion'],point['available'])
    original=base.simple.SimpleComponentReliability(bundle['policy']).decide(pred,image_size)
    original['final_box_original']=original.pop('raw_b_output')
    risks['full_simple']=original['risks']['size'] if pred is not None else None
    risks['score_only']=1-pred[5] if pred is not None else None
    c=bundle['cutoffs'];branch=c['crossfit_fusion'] if point['available'] else c['partial_auc4']
    risks['fallback_margin']=None if pred is None else risks['crossfit_fusion']/branch
    flags=dict(simple=original['size_accepted'],temporal_veto=original['size_accepted'] and
        (not point['available'] or point['primary']<=c['temporal_veto']))
    for m in ('fusion4','partial_auc4','crossfit_fusion','fallback_margin'):
        flags[m]=risks[m] is not None and risks[m]<=c[m]
    if flags['fallback_margin']!=fallback.decide_size(risks['crossfit_fusion'],point['available'],c['crossfit_fusion'],c['partial_auc4']):
        raise ValueError('Branch decision identity differs')
    return risks,flags,original


def replay(rows,image_risks,bundle):
    result=[];tracker=None;previous=None
    ordered=sorted(rows,key=lambda r:(r['sequence'],r['frame_id'],r['image']))
    if len({r['image'] for r in rows})!=len(rows) or len({(r['sequence'],r['frame_id']) for r in rows})!=len(rows):
        raise ValueError('Duplicate frame identity')
    for r in ordered:
        reset=previous is None or previous!=r['sequence']
        if reset:tracker=base.level.CausalSizeEvidence()
        point=tracker.observe(r['pred'],r['frame_id'],r['image_size'],reset)
        point['acquisition_boundary']=False  # No known joins in these fixed VAL/TEST videos.
        ri=image_risks[r['image']]
        if (ri is None)!=(r['pred'] is None):raise ValueError('Image-risk output membership differs')
        risks,flags,original=score(r['pred'],r['image_size'],point,ri,bundle)
        saved=r['original_simple_decision']
        if any(original[k]!=saved[k] for k in original if k!='risks') or any(
            (original['risks'][k] is None)!=(saved['risks'][k] is None) or
            (original['risks'][k] is not None and abs(original['risks'][k]-saved['risks'][k])>1e-12) for k in original['risks']):
            raise ValueError('Formal policy replay differs')
        item=deepcopy(r);item.update(temporal=point,risks=risks,flags=flags,image_crossfit_risk=ri)
        result.append(item);previous=r['sequence']
    return result


def summary(rows,ids):
    if all(r['split']=='test' for r in rows):return test.test_summary(rows,ids)
    if all(r['split']=='val' for r in rows):return base.level.summarize(rows,ids)
    raise ValueError('Separate complete VAL and TEST roles')


def matched(rows,method,count=None,correct=None):
    out=sorted((r for r in rows if r['pred'] is not None),key=lambda r:(r['risks'][method],r['image']))
    if any(not math.isfinite(r['risks'][method]) for r in out):raise ValueError('Finite ranking required')
    if count is not None:
        if not 0<=count<=len(out):raise ValueError('Invalid count')
        ids={r['image'] for r in out[:count]};bound=out[count-1]['risks'][method] if count else None
        before=[r for r in out if bound is not None and r['risks'][method]<bound]
        tied=[r for r in out if bound is not None and r['risks'][method]==bound]
        slots=count-len(before);wrong=sum(base.level.metrics.bad(r) for r in before)
        value=summary(rows,ids);value['tie_bounds']=dict(FA_min=wrong+max(0,slots-sum(not base.level.metrics.bad(r) for r in tied)),
            FA_max=wrong+min(slots,sum(base.level.metrics.bad(r) for r in tied)),partial_tie=0<slots<len(tied),tie_count=len(tied))
    else:
        if correct is None or not 0<=correct<=sum(not base.level.metrics.bad(r) for r in out):raise ValueError('Invalid CR')
        seen=0;bound=None
        if correct:
            for r in out:
                seen+=int(not base.level.metrics.bad(r))
                if seen>=correct:bound=r['risks'][method];break
        value=summary(rows,{r['image'] for r in out if bound is not None and r['risks'][method]<=bound})
        value.update(exact_CR=value['states']['CR']==correct,offline_GT_threshold=bound)
    value['offline_only']=True
    return value


def evaluate(rows):
    groups={}
    for g,part in base.level.metrics.grouped(rows).items():
        actual={m:summary(part,{r['image'] for r in part if r['flags'][m]}) for m in METHODS}
        comparison={m:{control:dict(same_count=matched(part,control,count=v['accepted_outputs']),
            same_CR=matched(part,control,correct=v['states']['CR'])) for control in CONTROLS}
            for m,v in actual.items()}
        outputs=[r for r in part if r['pred'] is not None]
        hits=sum(base.simple.geometry_errors(r['gt'],r['pred'])['center_px']<15 for r in outputs)
        groups[g]=dict(actual=actual,comparisons=comparison,raw=summary(part,{r['image'] for r in outputs}),
            AUROC={m:base.level.metrics.auc(part,m) for m in CONTROLS},
            partial_AUC={m:pauc.partial_auc(part,m) for m in CONTROLS},
            center=dict(frames=len(part),outputs=len(outputs),hits=hits,output_coverage=len(outputs)/len(part),
                hit_rate_on_outputs=hits/len(outputs) if outputs else None,correct_coverage_all_frames=hits/len(part)),
            FA_regimes={m:{k:sum(r['flags'][m] and base.level.metrics.bad(r) is True and
                ('unsupported' if not r['temporal']['available'] else 'low' if r['temporal']['primary']<=base.LEVEL_CUTOFF else 'high')==k
                for r in part) for k in ('low','high','unsupported')} for m in METHODS})
    return dict(groups=groups)
