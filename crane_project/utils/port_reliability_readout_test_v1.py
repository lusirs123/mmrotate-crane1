"""Frozen exposed-TEST diagnostics, never a TRAIN/VAL selection replacement."""
from copy import deepcopy
from collections import Counter

import numpy as np

from crane_project.utils import port_reliability_readout_compare_v1 as scoring
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_feature_ablation_v1 as groups
from crane_project.utils import port_reliability_state_continuity_v1 as states
from crane_project.utils import port_reliability_separability_v1 as sep

VERSION = 'port_reliability_readout_test_v1'
COUNTS = {'real_seq03':200,'real_seq04':668,'sim_seq09':572}
INPUT_NAMES = ('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original')


def frozen_bundle(model, cutoffs, report):
    """Called before opening TEST inputs. A failed VAL is admissible here."""
    if (model['protocol'] != scoring.VERSION or report['protocol'] != scoring.VERSION
            or model['epoch'] != 100 or report['fixed_final_epoch'] != 100
            or model['update_counts'] != dict(temperature=1000,residual=1000)
            or report['update_counts'] != model['update_counts'] or report['TEST_read']
            or report['original_policy_changed'] or not report['boxes_scores_output_center_angle_unchanged']
            or report['status'] != 'VAL_FAILED_STOP' or report['gate']['passed']
            or set(model['models']) != set(scoring.ARMS) or cutoffs != report['VAL_cutoffs']
            or set(cutoffs) != {'temperature','residual','full_simple','score_only'}):
        raise ValueError('Requires exact final100 failed-VAL comparison and frozen VAL thresholds')
    for value in cutoffs.values():
        if not value.get('single_global_cutoff') or not 0 <= value['risk_le'] <= 1:
            raise ValueError('Invalid frozen global VAL cutoff')
    return deepcopy(model),deepcopy(cutoffs)


def bind_rows(records, native):
    """Offline pairing only. Historical size candidate is never loaded."""
    if (len(records) != 1440 or len(native) != 1440
            or Counter(r['sequence'] for r in records) != Counter(COUNTS)
            or [r['image'] for r in records] != sorted({r['image'] for r in records})):
        raise ValueError('Requires complete original1440 TEST with unique ordered identities')
    result = []
    for r,p in zip(records,native):
        if (r['split'] != 'test' or r['image'] != p['image'] or r['frame_id'] != p['frame_id']
                or r['image_sha256'] != p['image_sha256']
                or r['image_size'] != p['original_size_hw'][::-1]
                or r['domain'] != r['sequence'].split('_')[0]
                or r['image'] != '%s_%05d'%(r['sequence'],r['frame_id'])
                or r['pred'] != p['midpoint'] or not p['original_b_raw_exact_before_after']
                or (p['b'] is None) != (p['midpoint'] is None)
                or (p['b'] is not None and p['b'][5] != p['midpoint'][5])):
            raise ValueError('Reviewed M, score, count, image or native pairing changed')
        d = r['methods']['simple']
        if d['final_box_original'] != r['pred'] or d['center_accepted'] != (r['pred'] is not None):
            raise ValueError('Original three-component output identity differs')
        row = {k:deepcopy(r[k]) for k in ('image','sequence','frame_id','domain','split','gt','pred',
            'image_size','angle_axis_well_defined','image_sha256','annotation_sha256')}
        row.update(original_simple_decision=deepcopy(d),b_original=deepcopy(p['b']))
        result.append(row)
    return result


def replay_box(stored, actual):
    """Replayed head is evidence only; never replaces delivered M."""
    if (stored is None) != (actual is None):
        raise ValueError('Missing output changed')
    if stored is None: return np.zeros(6)
    a,b = np.asarray(stored,dtype=float),np.asarray(actual,dtype=float)
    if a.shape != (6,) or b.shape != (6,) or not np.isfinite(b).all():
        raise ValueError('Invalid replay box')
    delta = np.abs(a-b);delta[4]=abs((a[4]-b[4]+np.pi/2)%np.pi-np.pi/2)
    if (not np.allclose(a[:4],b[:4],atol=5e-4,rtol=2e-5) or delta[4]>2e-6 or a[5]!=b[5]):
        raise ValueError('Frozen midpoint replay differs')
    return delta


def score_rows(rows, features, ids, model, cutoffs):
    """No GT supplied to scoring/decision functions; no threshold fitting."""
    present = [r['image'] for r in rows if r['pred'] is not None]
    x = np.asarray(features,dtype=float)
    if ids != present or x.shape != (len(present),291) or not np.isfinite(x).all():
        raise ValueError('Complete TEST feature membership differs')
    risks = {m:{r['image']:None for r in rows} for m in cutoffs}
    for arm in scoring.ARMS:
        values = simple.sigmoid(scoring.numpy_logits(model['models'][arm],x,model['normalizer'],arm))
        risks[arm].update(dict(zip(ids,map(float,values))))
    for r in rows:
        if r['pred'] is not None:
            risks['full_simple'][r['image']] = r['original_simple_decision']['risks']['size']
            risks['score_only'][r['image']] = 1-r['pred'][5]
    decisions = []
    for r in rows:
        # All four use VAL cutoffs. The unchanged original simple point is
        # reported separately so it cannot be confused with this VAL95 point.
        methods = {m:scoring.decide(r['original_simple_decision'],risks[m][r['image']],cutoffs[m]['risk_le']) for m in risks}
        decisions.append(dict(image=r['image'],methods=methods))
    return risks,decisions


def test_order(rows):
    """TEST-only offline order; the original TRAIN/VAL guard stays intact."""
    ids=set();positions=set()
    for row in rows:
        pos=row['sequence'],row['frame_id']
        if (row['split']!='test' or row['image'] in ids or pos in positions
                or not isinstance(row['frame_id'],int)):
            raise ValueError('Invalid or duplicate frozen TEST identity')
        ids.add(row['image']);positions.add(pos)
    return sorted(rows,key=lambda r:(r['sequence'],r['frame_id'],r['image']))


def test_summary(rows, accepted):
    ordered = test_order(rows)
    ids = {r['image'] for r in rows}
    if not set(accepted) <= ids:
        raise ValueError('Acceptance contains an unknown image')
    observations = [(r, states.size_state(r, r['image'] in accepted)) for r in ordered]
    counts = {key: sum(s == key for _, s in observations) for key in states.STATES}
    runs = {}
    for kind, members in states.RUN_KINDS.items():
        segments = []; active = None; previous = None
        for row, state in observations:
            adjacent = (previous is not None and row['sequence'] == previous['sequence']
                        and row['frame_id'] == previous['frame_id']+1)
            if not adjacent or state not in members:
                if active is not None:
                    segments.append(active); active = None
            if state in members:
                if active is None:
                    active = dict(sequence=row['sequence'], start_frame=row['frame_id'],
                                  end_frame=row['frame_id'], length=0,
                                  state_counts={s: 0 for s in states.STATES})
                active['end_frame'] = row['frame_id']; active['length'] += 1
                active['state_counts'][state] += 1
            previous = row
        if active is not None:
            segments.append(active)
        longest = max((r['length'] for r in segments), default=0)
        runs[kind] = dict(longest=longest, segment_count=len(segments),
                          total_frames=sum(r['length'] for r in segments),
                          length_histogram=dict(sorted(Counter(r['length'] for r in segments).items())),
                          longest_segments=[r for r in segments if r['length'] == longest],
                          segments=segments)
    n = len(rows); outputs = n-counts['MISSING']; good = counts['CR']+counts['FR']
    kept = counts['CR']+counts['FA']
    adjacent_pairs = switches = 0
    for (a, sa), (b, sb) in zip(observations, observations[1:]):
        if a['sequence'] == b['sequence'] and a['frame_id']+1 == b['frame_id']:
            adjacent_pairs += 1
            switches += int((sa in ('FA', 'CR')) != (sb in ('FA', 'CR')))
    ratio = lambda a, b: a/b if b else None
    return dict(frames=n, outputs=outputs, states=counts,
                accepted_outputs=kept, output_coverage=ratio(outputs, n),
                acceptance_coverage_all_frames=ratio(kept, n),
                correct_coverage_all_frames=ratio(counts['CR'], n),
                good_retention=ratio(counts['CR'], good),
                accepted_error_fraction=ratio(counts['FA'], kept), runs=runs,
                adjacent_observed_pairs=adjacent_pairs, flag_switches=switches,
                flag_switch_rate=ratio(switches, adjacent_pairs),
                missing_excluded_from_confusion=True, gaps_reset_runs=True,
                observed_frame_adjacency_only=True)


def test_describe(rows, scores, arm, cutoff):
    result = {}
    controls = tuple(k for k in scores if k != arm)
    for name, group in groups.groups(rows).items():
        out = [r for r in group if r['pred'] is not None]
        ids = [r['image'] for r in out]
        bad = [groups.size_bad(r) for r in out]
        accepted = {i for i in ids if scores[arm][i]<=cutoff}
        value = dict(actual=test_summary(group,accepted),same_count_controls={},
                     comparison_is_offline=True,candidate_selected=False)
        for m in controls:
            ranked_ids=sorted(ids,key=lambda i:(scores[m][i],i));k=len(accepted)
            value['same_count_controls'][m]=test_summary(group,set(ranked_ids[:k]))
            if 0<k<len(ids) and scores[m][ranked_ids[k-1]]==scores[m][ranked_ids[k]]:
                boundary=scores[m][ranked_ids[k-1]]
                value['same_count_controls'][m]['partial_boundary_tie']=dict(risk=boundary,
                    tie_count=sum(scores[m][i]==boundary for i in ids),
                    selection='image_order_only_not_GT',continuity_order_sensitive=True)
        value['rank'] = {m:sep.rank_metrics([scores[m][i] for i in ids],bad) for m in scores}
        value['control_tie_bounds'] = {}
        value['matched_CR_controls'] = {}
        for m in controls:
            k = len(accepted)
            value['control_tie_bounds'][m] = sep.same_count_reference(
                [scores[m][i] for i in ids],bad,ids,[k])[k]
            target = value['actual']['states']['CR']
            ranked = sorted(out,key=lambda r:(scores[m][r['image']],r['image']))
            good = 0; bound = None
            if target:
                for r in ranked:
                    good += int(not groups.size_bad(r))
                    if good >= target:
                        bound = scores[m][r['image']]; break
            kept = {i for i in ids if bound is not None and scores[m][i]<=bound}
            matched = test_summary(group,kept)
            matched.update(offline_GT_reference_only=True, whole_boundary_tie=True,
                           requested_CR=target, exact_CR=matched['states']['CR']==target)
            value['matched_CR_controls'][m] = matched
        result[name] = value
    return result


def summarize(rows, risks, cutoffs):
    stats = {arm:test_describe(rows,risks,arm,cutoffs[arm]['risk_le']) for arm in scoring.ARMS}
    original = {}; raw = {}; center = {}; fixed = {}
    for name,part in groups.groups(rows).items():
        fixed[name] = {m:test_summary(part,{r['image'] for r in part
            if r['pred'] is not None and risks[m][r['image']]<=cutoffs[m]['risk_le']}) for m in cutoffs}
        original[name] = test_summary(part,{r['image'] for r in part if r['original_simple_decision']['size_accepted']})
        raw[name] = test_summary(part,{r['image'] for r in part if r['pred'] is not None})
        outputs = sum(r['pred'] is not None for r in part)
        hits = sum(simple.geometry_errors(r['gt'],r['pred'])['center_px']<15 for r in part if r['pred'] is not None)
        center[name] = dict(frames=len(part),outputs=outputs,hits=hits,output_coverage=outputs/len(part),
            hit_rate_on_outputs=hits/outputs if outputs else None,correct_coverage_all_frames=hits/len(part))
    return dict(statistics=stats,fixed_VAL_cutoff_summary=fixed,
        original_policy_summary=original,raw_size_summary=raw,center=center,
        offline_same_count_and_same_CR_only=True,thresholds_from_TEST=False,
        status='EXPOSED_TEST_DIAGNOSTIC_COMPLETE_KEEP_VAL_FAILURE',automatic_promotion=False)
