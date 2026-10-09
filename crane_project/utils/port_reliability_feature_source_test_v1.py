"""Frozen feature-source TEST supplement; no fitting or candidate selection."""
from copy import deepcopy
import numpy as np
from crane_project.utils import port_reliability_feature_source_v1 as scoring
from crane_project.utils import port_reliability_readout_test_v1 as old
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_feature_ablation_v1 as groups

VERSION = 'port_reliability_feature_source_test_v1'
COUNTS = old.COUNTS
bind_rows = old.bind_rows
test_summary = old.test_summary
test_describe = old.test_describe

def frozen_bundle(model, cutoffs, report):
    if (model['protocol'] != scoring.VERSION or report['protocol'] != scoring.VERSION
            or model['epoch'] != 100 or report['fixed_final_epoch'] != 100
            or model['update_counts'] != dict(midpoint=1000,native=1000)
            or report['update_counts'] != model['update_counts'] or report['TEST_read']
            or report['original_policy_changed'] or not report['boxes_scores_output_center_angle_unchanged']
            or report['status'] != 'VAL_FAILED_STOP' or report['gate']['passed']
            or set(model['models']) != set(scoring.ARMS) or set(model['normalizers']) != set(scoring.ARMS)
            or cutoffs != report['VAL_cutoffs']
            or set(cutoffs) != {'midpoint','native','full_simple','score_only'}):
        raise ValueError('Requires exact final100 failed-VAL sources and frozen VAL cutoffs')
    for value in cutoffs.values():
        if not value.get('single_global_cutoff') or not np.isfinite(value['risk_le']) or not 0 <= value['risk_le'] <= 1:
            raise ValueError('Invalid frozen global VAL cutoff')
    return deepcopy(model),deepcopy(cutoffs)

def score_rows(rows, matrices, ids, model, cutoffs):
    present = [r['image'] for r in rows if r['pred'] is not None]
    if ids != present or set(matrices) != set(scoring.ARMS):
        raise ValueError('Complete TEST feature membership differs')
    risks = {m:{r['image']:None for r in rows} for m in cutoffs}
    for arm in scoring.ARMS:
        x = np.asarray(matrices[arm],dtype=float)
        if x.shape != (len(ids),259) or not np.isfinite(x).all():
            raise ValueError('Fixed259 dimensional source features required')
        values = simple.sigmoid(scoring.numpy_logits(model['models'][arm],x,model['normalizers'][arm]))
        risks[arm].update(dict(zip(ids,map(float,values))))
    decisions=[]
    for r in rows:
        if r['pred'] is not None:
            risks['full_simple'][r['image']]=r['original_simple_decision']['risks']['size']
            risks['score_only'][r['image']]=1-r['pred'][5]
        methods={m:scoring.decide(r['original_simple_decision'],risks[m][r['image']],cutoffs[m]['risk_le']) for m in risks}
        decisions.append(dict(image=r['image'],methods=methods))
    return risks,decisions

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
