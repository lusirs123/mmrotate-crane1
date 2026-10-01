#!/usr/bin/env python3
"""Fixed B/E-H VAL-selected artifacts, original metrics/gates; no TEST or inference."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools.preflight_port_shape_e_h_v1 import check_configs, CONTROL, EXPERIMENT
from crane_project.tools.diagnose_port_center_size_d_v1 import load_frozen_val, val_analysis, runs
from crane_project.tools.audit_port_train_val_geometry_v1 import summarize, write_new
from crane_project.tools.preflight_port_shape_e_v1 import reference_riou


def rename_e(value):
    """Preserve old diagnostics; name E-H explicitly in this separate report."""
    if isinstance(value,dict):
        names={'d':'e_h','d_output':'e_h_output','d_center_hit':'e_h_center_hit','d_minus_b':'e_h_minus_b'}
        return {names.get(k,k.replace('compensated_d','shape_e_h')):rename_e(v) for k,v in value.items()}
    if isinstance(value,list):return [rename_e(v) for v in value]
    return value


def coverage(rows):
    n=len(rows);out=sum(r['metrics']['output'] for r in rows)
    hit=sum(bool(r['metrics']['output'] and r['metrics']['center_hit']) for r in rows)
    return dict(frames=n,output_frames=out,output_center_correct_frames=hit,
        output_center_hit_pct=100*hit/out if out else None,
        output_coverage_pct=100*out/n if n else None,
        all_frame_center_correct_coverage_pct=100*hit/n if n else None)


def conditions(b,e):
    groups={arm:{dom:[r for r in rr if r['domain']==dom] for dom in ('real','sim')}
            for arm,rr in [('b',b),('e_h',e)]}
    cov={arm:{dom:coverage(rr) for dom,rr in dd.items()} for arm,dd in groups.items()}
    longest=lambda rr,predicate:max([r['length'] for r in runs(rr,predicate)],default=0)
    rb,re=groups['b']['real'],groups['e_h']['real']
    checks=dict(real_outputs_ge374=cov['e_h']['real']['output_frames']>=374,
        real_all_frame_centres_ge360=cov['e_h']['real']['output_center_correct_frames']>=360,
        sim_outputs_and_centres_512=(cov['e_h']['sim']['output_frames']==512 and
                                    cov['e_h']['sim']['output_center_correct_frames']==512),
        real_no_output_run_not_worse=longest(re,lambda r:not r['metrics']['output'])<=longest(rb,lambda r:not r['metrics']['output']),
        real_riou_failure_run_not_worse=longest(re,lambda r:r['metrics']['riou']<.5)<=longest(rb,lambda r:r['metrics']['riou']<.5))
    for dom in ('real','sim'):
        shared=[(a,c) for a,c in zip(b,e) if a['domain']==dom and a['metrics']['output'] and c['metrics']['output']]
        a,c=summarize([x for x,y in shared]),summarize([y for x,y in shared])
        for field in ('long_edge_relative_error','short_edge_relative_error'):
            checks[dom+'_shared_'+field+'_decreases']=bool(shared) and c[field]['mean']<a[field]['mean']
        checks[dom+'_shared_center_not_worse']=bool(shared) and c['center_error_px']['mean']<=a['center_error_px']['mean']
        checks[dom+'_all_frame_riou_not_worse']=summarize(groups['e_h'][dom])['all_frame_mean_riou']>=summarize(groups['b'][dom])['all_frame_mean_riou']
    sim_angle=summarize(groups['e_h']['sim'])['angle_error_deg']['rmse']
    baseline_angle=summarize(groups['b']['sim'])['angle_error_deg']['rmse']
    checks['sim_pure_angle_rmse_lt_frozen_2_1072']=sim_angle is not None and sim_angle<2.1072
    checks['sim_pure_angle_rmse_lt_exact_b']=sim_angle is not None and baseline_angle is not None and sim_angle<baseline_angle
    return dict(coverage=cov,checks=checks,all_conditions_met=all(checks.values()),
        note='Pre-registered VAL conditions only; not automatic replacement, significance, depth or TEST evidence.')


def riou_crosscheck(rows):
    deltas=[];threshold_changes=0
    for r in rows:
        if not r['metrics']['output']:continue
        reference=reference_riou(r['pred'][:5],r['gt'])
        old=r['metrics']['riou'];deltas.append(abs(reference-old))
        threshold_changes+=int((reference>=.5)!=(old>=.5))
    return dict(output_pairs=len(deltas),max_abs_delta=max(deltas,default=0.),
        mean_abs_delta=sum(deltas)/len(deltas) if deltas else None,
        above_1e_3=sum(x>1e-3 for x in deltas),half_threshold_changes=threshold_changes,
        note='Independent float64 check only; original selection and primary metrics are unchanged.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--b-sweep',default='work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/val_sweep_port_v1')
    ap.add_argument('--e-sweep',default='work_dirs/crane_symeood_k1_port_day2night_shape_e_h_v1/val_sweep_port_v1')
    ap.add_argument('--out-json',required=True)
    args=ap.parse_args();os.chdir(ROOT)
    out=Path(args.out_json).resolve()
    if out.exists():raise FileExistsError('Preserve old report; choose a new output name')
    b,e=check_configs()
    bs,es=Path(args.b_sweep).resolve(),Path(args.e_sweep).resolve()
    selections=[json.loads((s/'sweep_results.json').read_text()) for s in (bs,es)]
    sb,se=selections
    epochs=[16,18,20,22,24]
    if (se.get('candidate_epochs')!=epochs or se.get('selected_checkpoint') not in ['epoch_'+str(x) for x in epochs]
            or set(se['all_checkpoints'])!=set('epoch_'+str(x) for x in epochs)
            or se.get('selection_config')!=sb.get('selection_config')
            or se.get('metric_protocol_version')!=sb.get('metric_protocol_version')
            or se.get('center_thresh_px')!=15. or sb.get('center_thresh_px')!=15.):
        raise ValueError('Expected original fixed VAL selection protocol/epoch set')
    rb,ib=load_frozen_val(b,CONTROL,bs,'epoch_24')
    re,ie=load_frozen_val(e,EXPERIMENT,es,se['selected_checkpoint'])
    analysis=rename_e(val_analysis(rb,re))
    cross={'b':riou_crosscheck(rb),'e_h':riou_crosscheck(re)}
    report=dict(protocol='port_shape_e_h_v1_val_compare',evidence_role='source_val_only_fixed_experiment_comparison',
        identities=dict(b=ib,e_h=ie),selection_info=se['selection_info'],
        geometry=analysis,pre_registered_conditions=conditions(rb,re),riou_crosscheck=cross,
        metric_consistency_review_required=any(x['above_1e_3'] or x['half_threshold_changes'] for x in cross.values()),
        limitations=['TEST has been exposed repeatedly and is not used here.',
                     'Common-output geometry does not substitute for all-frame coverage.',
                     'Single seed does not establish stable/significant improvement; depth has no independent GT check.'])
    write_new(out,report)
    print(json.dumps(report['pre_registered_conditions'],ensure_ascii=False,indent=2))
    print('VAL selection:',report['selection_info'])
    print('RIoU crosscheck:',cross)
    print('Saved',out)


if __name__=='__main__':main()
