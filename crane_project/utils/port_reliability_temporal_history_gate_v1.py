"""Past-only history-linearity gate between two frozen temporal veto rules."""
from copy import deepcopy
import math
from crane_project.utils import port_reliability_temporal_trend_v1 as trend

VERSION = 'port_reliability_temporal_history_gate_v1'
LEVEL_CUTOFF = 0.0785511443740936
TREND_CUTOFF = 0.10526070116970931
SETTINGS = dict(trend.SETTINGS, primary='past3_history_gated_normalized_residual',
                history_gate_quantile=.95, level_cutoff=LEVEL_CUTOFF, trend_cutoff=TREND_CUTOFF)


def select(point, q_cutoff):
    """No current truth, past truth, future, sequence or domain input."""
    if not math.isfinite(q_cutoff) or q_cutoff < 0: raise ValueError('Invalid history gate cutoff')
    result = deepcopy(point)
    if not result['available']:
        result.update(branch='unsupported', selected_raw_risk=None, selected_cutoff=None)
        return result
    is_linear = result['history_linearity'] <= q_cutoff
    raw = result['trend_primary'] if is_linear else result['level_primary']
    cutoff = TREND_CUTOFF if is_linear else LEVEL_CUTOFF
    normalized = raw/cutoff
    # Preserve the frozen raw <= cutoff decision even at a rounding boundary.
    if raw > cutoff and normalized <= 1: normalized = 1+2.**-52
    if raw <= cutoff and normalized > 1: normalized = 1.
    result.update(branch='trend' if is_linear else 'level', selected_raw_risk=raw,
                  selected_cutoff=cutoff, primary=normalized)
    return result


class CausalHistoryGateEvidence(trend.CausalTrendEvidence):
    def __init__(self, q_cutoff=None):
        if q_cutoff is not None and (not math.isfinite(q_cutoff) or q_cutoff < 0):
            raise ValueError('Invalid history gate cutoff')
        super().__init__(); self.q_cutoff = q_cutoff

    def observe(self, pred, frame_id, image_size, reset=False):
        past = list(self.history)
        point = super().observe(pred, frame_id, image_size, reset=reset)
        point.update(trend_primary=point['primary'], history_linearity=None,
                     signed_history_linearity=None, branch='unassigned' if point['available'] else 'unsupported')
        if point['available']:
            signed = [past[1][1][j]-(past[0][1][j]+past[2][1][j])/2 for j in (0,1)]
            point.update(history_linearity=max(map(abs,signed)), signed_history_linearity=signed)
        return select(point,self.q_cutoff) if self.q_cutoff is not None else point


def enrich(rows, boundaries):
    result = trend.enrich(rows,boundaries); tracker = None; sequence = None
    for r in result:
        new = sequence != r['sequence']
        if new: tracker = CausalHistoryGateEvidence()
        r['trend_temporal'] = deepcopy(r['temporal'])
        r['temporal'] = tracker.observe(r['pred'],r['frame_id'],r['image_size'],
            reset=new or r['frame_id'] in boundaries.get(r['sequence'],[]))
        r['temporal']['acquisition_boundary'] = r['trend_temporal']['acquisition_boundary']
        if r['temporal']['trend_primary'] != r['trend_temporal']['primary']:
            raise ValueError('Frozen trend evidence changed')
        sequence = r['sequence']
    return result


def apply_gate(rows, q_cutoff):
    result = deepcopy(rows)
    for r in result:
        r['temporal'] = select(r['temporal'],q_cutoff)
        r['risks']['temporal'] = r['temporal']['primary']
    return result


def trend_view(rows):
    result = deepcopy(rows)
    for r in result:
        r['temporal'] = r['trend_temporal']
        r['risks']['temporal'] = r['temporal']['primary']
    return result


def fit_history_gate(rows):
    if any(r['split'] not in ('train','train_sim') or r['reliability_role']!='train' for r in rows):
        raise ValueError('History gate fitting is TRAIN only')
    lookup={(r['sequence'],r['frame_id']):r for r in rows}
    eligible=[]
    for r in rows:
        if not r['temporal']['available'] or not r['original_simple_decision']['size_accepted']:
            continue
        if trend.previous.metrics.bad(r) is not False:continue
        past=[lookup[(r['sequence'],i)] for i in r['temporal']['history_frames']]
        if all(trend.previous.metrics.bad(p) is False for p in past):eligible.append(r)
    if not eligible:raise ValueError('No correct TRAIN history support')
    points={};thresholds=[]
    for group,part in trend.previous.metrics.grouped(rows).items():
        ids={r['image'] for r in part}
        values=sorted(r['temporal']['history_linearity'] for r in eligible if r['image'] in ids)
        needed=math.ceil(.95*len(values))
        cutoff=values[needed-1] if values else None
        points[group]=dict(eligible_histories=len(values),required=needed,cutoff=cutoff)
        if cutoff is not None:thresholds.append(cutoff)
    return dict(q_le=max(thresholds),group_support=points,TRAIN_only=True,single_global_gate=True,
                eligible_definition='current_correct_and_original_simple_accept_and_all_three_history_boxes_correct',
                no_correctness_or_coverage_guarantee=True,not_a_probability_of_correct_history=True)


def branch_diagnostics(rows, q_cutoff):
    lookup={(r['sequence'],r['frame_id']):r for r in rows};result={}
    for group,part in trend.previous.metrics.grouped(rows).items():
        counts={branch:dict(frames=0,current_good=0,current_bad=0,history_all_good=0,history_has_error=0)
                for branch in ('trend','level','unsupported')}
        for r in part:
            branch=r['temporal']['branch'];c=counts[branch];c['frames']+=1
            if r['pred'] is not None:c['current_bad' if trend.previous.metrics.bad(r) else 'current_good']+=1
            if r['temporal']['available']:
                past=[lookup[(r['sequence'],i)] for i in r['temporal']['history_frames']]
                c['history_all_good' if all(trend.previous.metrics.bad(p) is False for p in past) else 'history_has_error']+=1
        result[group]=counts
    return dict(groups=result,GT_diagnostic_only=True,q_le=q_cutoff,
                linear_history_does_not_imply_correct_history=True)


# Evaluation uses the same full-frame metrics and old-rule fair comparisons.
level_view = trend.level_view
previous = trend.previous
comparison = trend.comparison
evaluate = trend.evaluate
gate = trend.gate
