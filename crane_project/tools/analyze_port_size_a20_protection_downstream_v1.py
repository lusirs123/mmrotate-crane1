#!/usr/bin/env python3
"""Verify completed frozen S043 downstream rows; no model, fitting or selection."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import eval_port_size_a20_protection_downstream_v1 as e
from crane_project.tools.analyze_port_size_qalign_v1 import csv_write
from crane_project.tools.analyze_port_size_proposal_downstream_v1 import aggregate_temporal


def detection_row(group,arm,cell,split):
    c=cell['components'];s=cell['sizes'];n,o=c['frames'],c['outputs'];angle_n=c['angle_eligible_outputs']
    row=dict(split=split,group=group,arm=arm,frames=n,outputs=o,missing=c['missing'],coverage=o/n,
        mean_RIoU=cell['mean_riou_full_frame'],AP50=cell['native_ap11']['0.5']['ap_percent'],AP75=cell['native_ap11']['0.75']['ap_percent'],
        center_correct=c['center_correct'],center_denominator=o,center_rate=c['center_correct']/o if o else None,
        size_correct=c['size_correct'],size_denominator=o,size_rate=c['size_correct']/o if o else None,
        angle_correct=c['angle_correct'],angle_eligible_outputs=angle_n,angle_eligible_frames=c['angle_eligible_frames'],
        angle_rate=c['angle_correct']/angle_n if angle_n else None,
        center_full=c['center_correct']/n,size_full=c['size_correct']/n,angle_full=c['angle_correct']/n)
    for k in ('a','r'):
        for stat in ('abs_mae','bias','abs_p95'):row[k+'_'+stat]=s[k][stat]
    for edge in ('long','short'):
        for stat in ('mae','p95','bias'):
            value=s['edges'][edge][stat];row[edge+'_'+stat+'_pct']=100*value if value is not None else None
    row.update(aggregate_temporal(cell['temporal']))
    return row


def assert_numbers(actual,saved,key=''):
    if isinstance(actual,dict):
        if not isinstance(saved,dict) or actual.keys()!=saved.keys():raise ValueError('Report keys differ '+key)
        for k in actual:assert_numbers(actual[k],saved[k],k)
    elif isinstance(actual,(tuple,list)):
        if not isinstance(saved,(tuple,list)) or len(actual)!=len(saved):raise ValueError('Report count differs '+key)
        for a,b in zip(actual,saved):assert_numbers(a,b,key)
    elif isinstance(actual,float):
        tolerance=5e-6 if key in ('riou','mean_riou_full_frame') else 1e-10
        if not isinstance(saved,(int,float)) or not math.isclose(actual,saved,abs_tol=tolerance,rel_tol=1e-12):raise ValueError('Report arithmetic differs '+key)
    elif actual!=saved:raise ValueError('Report value differs '+key)


def verify_detection(rows,saved):
    if saved['native_ap_verified'] is not True:raise ValueError('Native AP verification incomplete')
    fresh=e.detection_report(rows,verify_native=False)
    for group,methods in fresh['metrics'].items():
        for key,value in methods.items():
            reference=saved['metrics'][group][key]
            if key in e.ARMS:
                reference={k:v for k,v in reference.items() if k!='native_ap11'}
                for t in ('0.5','0.75'):
                    if abs(saved['metrics'][group][key]['native_ap11'][t]['ap_percent']-value['cache_gt_ap11'][t])>1e-4:raise ValueError('Native/cached AP disagrees')
            assert_numbers(value,reference)
    assert_numbers(fresh['diagnostics'],saved['diagnostics'])


def verify_depth(rows,saved,identity,calibration):
    predictions=[];truths=[]
    for r in rows:
        p=dict(frame_id=r['frame_id'],**{arm:r[arm]['box'] for arm in e.ARMS});predictions.append(p)
        cx,cy,w,h,angle=r['gt_box']
        truths.append(dict(camera_geometry=dict(z_cg_opt_m=r['truth_z_m']),pivot_relative=dict(theta_total_deg=r['theta_total_deg']),
            obb_geometry=dict(cx_px=cx,cy_px=cy,w_px=w,h_px=h,gamma_deg=math.degrees(angle))))
    manifest=dict(sequence_id=saved['sequence'],split=saved['split'],camera=identity['camera'],obb_reference_geometry=identity['physical_reference'])
    replay,summary=e.depth_report(predictions,truths,manifest,calibration)
    assert_numbers(replay,rows);assert_numbers(summary,saved)


def analyze(directory,out):
    if out.exists() or out.parent!=directory.parent:raise ValueError('Fresh sibling analysis result directory required')
    index=e.load(directory/'artifacts.json')['files']
    for n,h in index.items():
        if e.depth.sha(directory/n)!=h:raise ValueError('Artifact SHA mismatch '+n)
    done=e.load(directory/'completion.json')
    if done['protocol']!=e.VERSION or done['status']!='FROZEN_NATIVE_VAL_TEST_DEPTH_COMPLETE':raise ValueError('Incomplete frozen evaluation')
    if done['training_updates']!=0 or any(done[k] for k in ('selection','formula_refit','reliability_filter','automatic_adoption','test_selection','depth_selection')):raise ValueError('Scope changed')
    if done['state_before']!=done['state_after'] or done['source_identity']!=done['source_after']:raise ValueError('Frozen state/source drift')
    if any(done['weights'][arm]['sha256']!=pin for arm,pin in e.PINS.items()) or done['calibration_sha256']!=e.depth.CAL_SHA:raise ValueError('Frozen weight/formula identity differs')
    if done['counters']!=dict(feature_extractions=3424,native_head_calls=10272,size_head_calls=13696):raise ValueError('Native evaluation work count differs')
    e.require_val_proof(done['val_replay'])
    if e.depth.sha(ROOT/e.depth.CALIBRATION)!=e.depth.CAL_SHA:raise ValueError('Calibration changed')
    detection=[];near=[];cross=[]
    for split,count in (('val',887),('test',1440)):
        rows=e.rows(directory/split/'evaluation_rows.jsonl');saved=e.load(directory/split/'summary.json')
        if len(rows)!=count or len({r['image'] for r in rows})!=count:raise ValueError('Complete detection split required')
        verify_detection(rows,saved)
        for group,methods in saved['metrics'].items():
            for arm in e.ARMS:
                cell=methods[arm];row=detection_row(group,arm,cell,split)
                if arm in e.HEADS:row.update({k:methods[arm+'_paired_vs_M'][k] for k in ('wrong_to_right','right_to_wrong','new_output','lost_output')})
                detection.append(row)
            for arm,d in saved['diagnostics'][group].items():
                for edge,value in d['near_and_joint']['near'].items():near.append(dict(split=split,group=group,arm=arm,edge=edge,**value))
                cross.append(dict(split=split,group=group,arm=arm,statistics=d['cross_edge']))
    name=e.SEQUENCE;rows=e.rows(directory/name/'depth_rows.jsonl');saved=e.load(directory/name/'summary.json');identity=e.load(directory/name/'input_identity.json')
    if len(rows)!=1097 or saved['sequence']!=name or identity['calibration_sha256']!=e.depth.CAL_SHA:raise ValueError('Unknown-02 identity/count differs')
    constant=e.load(ROOT/'docs/data/webots_depth_formal_frame_index_v1.manifest.json')['constant_contract']
    pin=e.load(ROOT/'tools/port_frozen_downstream_v1_protocol.json')['sequences'][name]
    if identity['camera']!=constant['camera'] or any(identity['physical_reference'][k]!=constant['obb_reference_geometry'][k] for k in ('long_edge_mean_m','short_edge_mean_m')):raise ValueError('Frozen camera/physical reference differs')
    if any(identity[k]!=v for k,v in pin.items()) or saved['split']!=pin['split']:raise ValueError('Unknown sequence role/source identity differs')
    verify_depth(rows,saved,identity,e.load(ROOT/e.depth.CALIBRATION))
    depth=[];paired=[];geometry=[];propagation=[]
    for arm,value in saved['methods'].items():
        depth.append(dict(sequence=name,role=saved['role'],arm=arm,frames=value['frame_count'],outputs=value['output_frame_count'],
            valid=value['numeric_depth_count'],coverage=value['numeric_depth_coverage'],**value['depth_metrics']))
    for pair,value in saved['paired'].items():paired.append(dict(sequence=name,pair=pair,**{k:v for k,v in value.items() if k not in ('common_metrics','frame_ids','new_numeric_frame_ids','lost_numeric_frame_ids')}))
    for arm,value in saved['size_geometry'].items():
        for component,v in value.items():geometry.append(dict(sequence=name,arm=arm,component=component,**v))
    for arm,value in saved['propagation'].items():propagation.append(dict(arm=arm,statistics=value))
    out.mkdir()
    csv_write(out/'detection_components_custom.csv',detection);csv_write(out/'fixed_M_near_edges.csv',near)
    csv_write(out/'unknown_depth.csv',depth);csv_write(out/'unknown_paired_counts.csv',paired);csv_write(out/'unknown_size_geometry.csv',geometry)
    e.write(out/'cross_edge.json',cross);e.write(out/'unknown_paired_common_frames.json',saved['paired'])
    e.write(out/'unknown_propagation.json',propagation);e.write(out/'all_versions_common_valid.json',saved['all_versions_common_valid'])
    e.write(out/'analysis.json',dict(protocol=e.VERSION,source_artifacts_sha256=e.depth.sha(directory/'artifacts.json'),
        detection_frames=dict(val=887,test=1440),depth_frames=1097,independent_arithmetic_replay=True,
        boundaries='Frozen descriptive reports only; M remains formal, no selection/refit. Signed propagation is not additive MAE attribution.',
        source_sha256=e.depth.sha(__file__)))
    e.write(out/'artifacts.json',dict(files={p.name:e.depth.sha(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();analyze(args.run.resolve(),args.out.resolve())
