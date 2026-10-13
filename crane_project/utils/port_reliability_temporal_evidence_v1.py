"""Causal size-change evidence on saved M boxes; no detector or learned head."""
from collections import Counter, deque
from copy import deepcopy
import math
from statistics import median
from crane_project.utils import port_reliability_complementarity_v1 as metrics
from crane_project.utils import port_reliability_state_continuity_v1 as states

VERSION = 'port_reliability_temporal_evidence_v1'
SETTINGS = dict(history=3, train_correct_retention=.95, long_error_run_min=5,
    primary='past3_log_edge_deviation', diagnostic='previous_log_edge_change',
    minimum_frame_step=1, training_updates=0, TEST_allowed=False)


class CausalSizeEvidence:
    """Online inputs contain no truth, label, future frame, or domain router."""
    def __init__(self):
        self.history = deque(maxlen=SETTINGS['history'])
        self.previous_frame = None
        self.image_size = None

    def reset(self):
        self.history.clear()
        self.previous_frame = None
        self.image_size = None

    def observe(self, pred, frame_id, image_size, reset=False):
        if not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError('Invalid observation index')
        if len(image_size) != 2 or any(not math.isfinite(x) or x <= 0 for x in image_size):
            raise ValueError('Invalid original image size')
        if self.previous_frame is not None and frame_id <= self.previous_frame and not reset:
            raise ValueError('Observations must increase; reset at video boundaries')
        reason = None
        if reset:
            self.reset(); reason = 'explicit_boundary'
        elif self.previous_frame is not None and frame_id != self.previous_frame+1:
            self.history.clear(); reason = 'observation_gap'
        elif self.image_size is not None and list(image_size) != self.image_size:
            self.history.clear(); reason = 'resolution_change'
        self.previous_frame = frame_id; self.image_size = list(image_size)
        if pred is None:
            self.history.clear()
            return dict(available=False, reason='missing_output', primary=None, step=None,
                        signed_log_change=None, history_frames=[])
        if (len(pred) != 6 or any(not math.isfinite(x) for x in pred)
                or min(pred[2:4]) <= 0 or not .05 < pred[5] <= 1):
            raise ValueError('Expected finite original M box and score')
        edges = [math.log(x) for x in sorted(pred[2:4], reverse=True)]
        step = max(abs(a-b) for a,b in zip(edges,self.history[-1][1])) if self.history else None
        available = len(self.history) == SETTINGS['history']
        signed = [edges[i]-median(h[1][i] for h in self.history) for i in (0,1)] if available else None
        result = dict(available=available, reason=None if available else (reason or 'history_warmup'),
            primary=max(map(abs,signed)) if signed is not None else None,
            step=step, signed_log_change=signed, history_frames=[h[0] for h in self.history])
        self.history.append((frame_id,edges))
        return result


def enrich(rows, boundaries):
    ordered = states.checked_order(rows)
    outputs=[]; tracker=None; previous=None
    for row in ordered:
        new_video = previous is None or row['sequence'] != previous['sequence']
        if new_video: tracker=CausalSizeEvidence()
        point=tracker.observe(row['pred'],row['frame_id'],row['image_size'],
            reset=new_video or row['frame_id'] in boundaries.get(row['sequence'], []))
        point['acquisition_boundary']=row['frame_id'] in boundaries.get(row['sequence'], [])
        item=deepcopy(row); item['temporal']=point
        item['risks']=dict(full_simple=row['size_risks']['full_simple'],
            score_only=1-row['pred'][5] if row['pred'] is not None else None,
            temporal=point['primary'])
        outputs.append(item); previous=row
    return outputs


def summarize(rows, accepted):
    """Known acquisition joins also break runs, retaining original reported indices."""
    copied=[];positions={};offsets=Counter()
    for r in states.checked_order(rows):
        if r.get('temporal',{}).get('acquisition_boundary'):offsets[r['sequence']]+=1
        item=dict(r);item['frame_id']=r['frame_id']+offsets[r['sequence']]
        positions[(item['sequence'],item['frame_id'])]=r['frame_id'];copied.append(item)
    result=states.summarize(copied,accepted);seen=set()
    for kind in result['runs'].values():
        for segment in kind['segments']+kind['longest_segments']:
            if id(segment) in seen:continue
            seen.add(id(segment))
            for field in ('start_frame','end_frame'):
                segment[field]=positions[(segment['sequence'],segment[field])]
    result['known_acquisition_boundaries_reset_runs']=True
    return result


def formal_accept(rows):
    return {r['image'] for r in rows if r['original_simple_decision']['size_accepted']}


def fit_cutoff(train):
    if any(r['split'] not in ('train','train_sim') or r['reliability_role'] != 'train' for r in train):
        raise ValueError('Temporal threshold fitting is TRAIN only')
    # Protect each group's already-retained correct observations, with one global bound.
    eligible=[r for r in train if r['temporal']['available'] and
        r['original_simple_decision']['size_accepted'] and metrics.bad(r) is False]
    if not eligible: raise ValueError('No eligible TRAIN correct observations')
    points={}
    for group,part in metrics.grouped(eligible).items():
        values=sorted(r['temporal']['primary'] for r in part)
        required=math.ceil(SETTINGS['train_correct_retention']*len(values))
        points[group]=dict(correct_outputs=len(values),required=required,cutoff=values[required-1])
    threshold=max(p['cutoff'] for p in points.values())
    return dict(risk_le=threshold,group_protection=points,training_only=True,
        single_global_threshold=True,not_a_calibrated_error_probability=True)


def veto_accept(rows, cutoff):
    return {r['image'] for r in rows if r['original_simple_decision']['size_accepted'] and
        (not r['temporal']['available'] or r['temporal']['primary'] <= cutoff)}


def point(rows, method, count=None, correct=None):
    out=sorted((r for r in rows if r['pred'] is not None),key=lambda r:(r['risks'][method],r['image']))
    if any(r['risks'][method] is None or not math.isfinite(r['risks'][method]) for r in out):
        raise ValueError('Ranking requires actual finite evidence')
    if count is not None:
        if not 0 <= count <= len(out): raise ValueError('Invalid acceptance count')
        selected=out[:count]
        result=summarize(rows,{r['image'] for r in selected})
        boundary=selected[-1]['risks'][method] if selected else None
        ties=[r for r in out if r['risks'][method]==boundary] if boundary is not None else []
        before=[r for r in out if r['risks'][method]<boundary] if boundary is not None else []
        slots=count-len(before); wrong=sum(metrics.bad(r) for r in before)
        result['tie_bounds']=dict(FA_min=wrong+max(0,slots-sum(not metrics.bad(r) for r in ties)),
            FA_max=wrong+min(slots,sum(metrics.bad(r) for r in ties)),
            partial_tie=0 < slots < len(ties),tie_count=len(ties),selection='image_identity_not_GT')
    else:
        if correct is None or not 0 <= correct <= sum(metrics.bad(r) is False for r in out):
            raise ValueError('Invalid correct retention target')
        target=None;seen=0
        if correct:
            for r in out:
                seen+=int(metrics.bad(r) is False)
                if seen>=correct: target=r['risks'][method];break
        result=summarize(rows,{r['image'] for r in out if target is not None and r['risks'][method]<=target})
        result.update(exact_CR=result['states']['CR']==correct,offline_GT_threshold=target)
    return result


def raw_error_profile(rows, cutoff):
    summary=summarize(rows,{r['image'] for r in rows if r['pred'] is not None})
    segments=summary['runs']['incorrect_acceptance']['segments'];by_id={r['image']:r for r in rows}
    long_ids={r['image'] for r in rows for seg in segments if seg['length']>=SETTINGS['long_error_run_min']
        and r['sequence']==seg['sequence'] and seg['start_frame']<=r['frame_id']<=seg['end_frame']}
    categories={}
    for name,part in [('short_1_to_3',[s for s in segments if s['length']<=3]),
                      ('length4',[s for s in segments if s['length']==4]),
                      ('long5plus',[s for s in segments if s['length']>=5])]:
        ids={r['image'] for r in rows for seg in part if r['sequence']==seg['sequence'] and
            seg['start_frame']<=r['frame_id']<=seg['end_frame']}
        badrows=[by_id[i] for i in ids];supported=[r for r in badrows if r['temporal']['available']]
        categories[name]=dict(segments=len(part),error_frames=len(ids),history_supported=len(supported),
            temporal_below_cutoff=sum(r['temporal']['primary']<=cutoff for r in supported),
            formal_FA=sum(r['original_simple_decision']['size_accepted'] for r in badrows))
    types=Counter()
    for r in rows:
        if metrics.bad(r) is not True:continue
        p,g=sorted(r['pred'][2:4],reverse=True),sorted(r['gt'][2:4],reverse=True)
        e=[(a-b)/b for a,b in zip(p,g)]
        types['both_small_gt10' if max(e)<-.1 else 'both_large_gt10' if min(e)>.1 else 'other_error']+=1
    return dict(longest_error_run=summary['runs']['incorrect_acceptance']['longest'],categories=categories,
        error_direction_counts=dict(types),long_error_frames=len(long_ids),
        run_units='adjacent_observation_indices_not_seconds',GT_diagnostic_only=True)


def evaluate(rows, cutoff):
    baseline=formal_accept(rows); proposed=veto_accept(rows,cutoff)
    full={};evidence={}
    for group,part in metrics.grouped(rows).items():
        actual=summarize(part,proposed & {r['image'] for r in part})
        base=summarize(part,baseline & {r['image'] for r in part})
        controls={m:dict(same_count=point(part,m,count=actual['accepted_outputs']),
            same_CR=point(part,m,correct=actual['states']['CR'])) for m in ('full_simple','score_only')}
        full[group]=dict(formal=base,temporal_veto=actual,controls=controls,
            raw_error_profile=raw_error_profile(part,cutoff),
            changes=dict(rescued_FA=base['states']['FA']-actual['states']['FA'],
                         added_FR=actual['states']['FR']-base['states']['FR']))
        supported=[r for r in part if r['temporal']['available']]
        supported_ids={r['image'] for r in supported}
        ref=summarize(supported,baseline & supported_ids)
        good=sum(metrics.bad(r) is False for r in supported);target=math.ceil(.95*good)
        comparison={m:dict(AUROC=metrics.auc(supported,m),
            same_formal_count=point(supported,m,count=ref['accepted_outputs']),
            same_formal_CR=point(supported,m,correct=ref['states']['CR']),
            correct_retention95=point(supported,m,correct=target)) for m in ('temporal','full_simple','score_only')}
        conditional=[r for r in supported if r['image'] in baseline]
        evidence[group]=dict(frames=len(part),history_supported=len(supported),
            unsupported_outputs=sum(r['pred'] is not None and not r['temporal']['available'] for r in part),
            support_coverage=len(supported)/len(part),supported_good=good,supported_bad=len(supported)-good,
            formal_supported=ref,comparisons=comparison,
            within_formal_accept_AUROC={m:metrics.auc(conditional,m) for m in comparison})
    failures=[]
    for group,v in full.items():
        a=v['temporal_veto'];c=a['states']; good=c['CR']+c['FR']
        if good and c['CR']<math.ceil(.95*good):failures.append(dict(group=group,check='correct_retention95'))
        for method,p in v['controls'].items():
            if c['FA']>p['same_count']['tie_bounds']['FA_min']:
                failures.append(dict(group=group,control=method,check='same_count_FA_nonincrease'))
            if group=='all' and c['FA']>=p['same_count']['tie_bounds']['FA_min']:
                failures.append(dict(group=group,control=method,check='overall_strict_same_count_gain'))
            if not p['same_CR']['exact_CR']:
                failures.append(dict(group=group,control=method,check='same_CR_tie_inconclusive'))
            if c['FA']>p['same_CR']['states']['FA']:
                failures.append(dict(group=group,control=method,check='same_CR_FA_nonincrease'))
            if group=='all' and c['FA']>=p['same_CR']['states']['FA']:
                failures.append(dict(group=group,control=method,check='overall_strict_same_CR_gain'))
            if a['runs']['correct_rejection']['longest']>p['same_CR']['runs']['correct_rejection']['longest']:
                failures.append(dict(group=group,control=method,check='longest_correct_FR_nonincrease'))
    return dict(full_groups=full,supported_evidence=evidence,
        gate=dict(passed=not failures,failures=failures,evidence_only=True,candidate_adopted=False),
        center=metrics.center(rows),fallback='keep_original_flag_when_history_unavailable',
        no_threshold_or_feature_selected_on_VAL=True)
