#!/usr/bin/env python3
"""Verify received artifacts and report fixed four-arm results without selection."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import eval_port_size_proposal_downstream_v1 as e


def numeric_equal(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(numeric_equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(numeric_equal(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10)
    return a==b


def csv_write(path,records):
    fields=list(dict.fromkeys(k for r in records for k in r))
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)


def aggregate_temporal(values):
    def average(key,denominator):
        cells=[v for v in values.values() if v[key] is not None and v[denominator]]
        n=sum(v[denominator] for v in cells)
        return sum(v[key]*v[denominator] for v in cells)/n if n else None
    runs=[v['mcml_max'] for v in values.values() if v['mcml_max'] is not None]
    return dict(DFR_percent_per_frame=average('dfr_percent_per_frame','dfr_adjacent_pairs'),
        DFR_pairs=sum(v['dfr_adjacent_pairs'] for v in values.values()),
        ACI=average('aci','aci_adjacent_pairs'),ACI_pairs=sum(v['aci_adjacent_pairs'] for v in values.values()),
        MCML_max=max(runs) if runs else None,MCML_mean=average('mcml_mean','mcml_segments'),
        MCML_segments=sum(v['mcml_segments'] for v in values.values()),
        TDR_hits=sum(v['tdr_w10']['numerator'] for v in values.values()),
        TDR_denominator=sum(v['tdr_w10']['denominator'] for v in values.values()))


def run(directory):
    index=e.load(directory/'artifacts.json')['files']
    for p,h in index.items():
        if e.depth.sha(directory/p)!=h:raise ValueError('Artifact mismatch: '+p)
    done=e.load(directory/'completion.json')
    if done['status']!='FROZEN_NATIVE_VAL_TEST_DEPTH_COMPLETE':raise ValueError('Incomplete evaluation')
    output=directory.parent/'analysis';output.mkdir(exist_ok=False)
    detection=[];baseline_replay={};near_edges=[]
    for split in ('val','test'):
        report=e.load(directory/split/'summary.json')
        saved_rows=e.rows(directory/split/'evaluation_rows.jsonl')
        if len(saved_rows)!=(887 if split=='val' else 1440):raise ValueError('Full detection denominator differs')
        if split=='test':
            path=ROOT/'docs/detection/ap_evidence_20261010/test_source/test_rows.jsonl'
            if e.depth.sha(path)!='13c4cfa7f1f67a7bc2cad36da1ec067adbeae74080b387d39efe39fdc41a788d':raise ValueError('Historical TEST reference changed')
            historical=e.rows(path);maxima={a:0. for a in ('b','midpoint')}
            if [r['image'] for r in historical]!=[r['image'] for r in saved_rows]:raise ValueError('Historical baseline identities differ')
            for r,old in zip(saved_rows,historical):
                for a in ('b','midpoint'):
                    x,y=r[a],old[a]
                    if (x is None)!=(y is None):raise ValueError('Historical baseline presence differs')
                    if x is not None:
                        maxima[a]=max(maxima[a],max(abs(v-w) for v,w in zip(x[:5],y[:5])))
                        if x[5]!=y[5] or any(abs(v-w)>1e-4+1e-6*abs(w) for v,w in zip(x[:5],y[:5])):raise ValueError('Historical baseline boxes differ')
            baseline_replay=dict(pass_check=True,max_component_difference=maxima,frames=1440)

        for group,methods in report.items():
            for a in e.ARMS:
                m=methods[a];c=m['components'];sz=m['sizes'];total_frames=c['frames'];out=c['outputs']
                canonical=e.depth.canonical
                sub=[r for r in saved_rows if group=='overall' or r['domain']==group or r['sequence']==group]
                if split=='test':
                    for j,edge in [(2,'long'),(3,'short')]:
                        near=[r for r in sub if r['midpoint'] is not None and abs(canonical(r['midpoint'])[j]/canonical(r['gt'])[j]-1)<=.03]
                        errors=[abs(canonical(r[a])[j]/canonical(r['gt'])[j]-1) for r in near if r[a] is not None]
                        near_edges.append(dict(group=group,arm=a,edge=edge,fixed_M_near_count=len(near),outputs=len(errors),harmed=sum(abs(canonical(r[a])[j]/canonical(r['gt'])[j]-1)>abs(canonical(r['midpoint'])[j]/canonical(r['gt'])[j]-1)+1e-12 for r in near if r[a] is not None),improved=sum(abs(canonical(r[a])[j]/canonical(r['gt'])[j]-1)<abs(canonical(r['midpoint'])[j]/canonical(r['gt'])[j]-1)-1e-12 for r in near if r[a] is not None),MAE_percent=100*math.fsum(errors)/len(errors) if errors else None))
                count=dict(frames=len(sub),outputs=0,missing=0,center_correct=0,size_correct=0,angle_eligible_frames=0,angle_eligible_outputs=0,angle_correct=0)
                for row in sub:
                    gt=canonical(row['gt']);eligible=gt[2]/gt[3]>=1.2;count['angle_eligible_frames']+=int(eligible)
                    if row[a] is None:count['missing']+=1;continue
                    count['outputs']+=1;pbox=canonical(row[a])
                    err=dict(center_px=math.hypot(pbox[0]-gt[0],pbox[1]-gt[1]),
                        size_max_relative=max(abs(pbox[j]-gt[j])/gt[j] for j in (2,3)),
                        angle_deg=abs((pbox[4]-gt[4]+math.pi/2)%math.pi-math.pi/2)*180/math.pi)
                    count['center_correct']+=int(err['center_px']<15)
                    count['size_correct']+=int(err['size_max_relative']<=.1)
                    count['angle_eligible_outputs']+=int(eligible)
                    count['angle_correct']+=int(eligible and err['angle_deg']<=3)
                    if a in ('c20','p07'):e.identity(row['midpoint'],row[a])
                if count!=c:raise ValueError('Three-component full-denominator replay differs')
                signed={edge:[] for edge in ('long','short')};a_values=[];r_values=[]
                for row in sub:
                    if row[a] is None:continue
                    gt,pbox=canonical(row['gt']),canonical(row[a]);logs=[]
                    for j,edge in [(2,'long'),(3,'short')]:
                        signed[edge].append(pbox[j]/gt[j]-1);logs.append(math.log(pbox[j]/gt[j]))
                    a_values.append(sum(logs)/2);r_values.append(logs[0]-logs[1])
                for edge,values in signed.items():
                    n=len(values);expected=dict(mae=math.fsum(map(abs,values))/n if n else None,bias=math.fsum(values)/n if n else None,p95=e.depth.percentile(list(map(abs,values)),.95))
                    if not numeric_equal(expected,{k:sz['edges'][edge][k] for k in expected}):raise ValueError('Continuous edge errors differ')
                for key,values in [('a',a_values),('r',r_values)]:
                    n=len(values);expected=dict(abs_mae=math.fsum(map(abs,values))/n if n else None,bias=math.fsum(values)/n if n else None,abs_p95=e.depth.percentile(list(map(abs,values)),.95))
                    if not numeric_equal(expected,sz[key]):raise ValueError('Common scale/aspect errors differ')

                r=dict(split=split,group=group,arm=a,frames=total_frames,outputs=out,missing=c['missing'],
                    mean_RIoU=m['mean_riou_full_frame'],AP50=m['native_ap11']['0.5']['ap_percent'],AP75=m['native_ap11']['0.75']['ap_percent'],
                    center_correct=c['center_correct'],center_denominator=out,center_rate=c['center_correct']/out if out else None,
                    size_correct=c['size_correct'],size_denominator=out,size_rate=c['size_correct']/out if out else None,
                    angle_correct=c['angle_correct'],angle_denominator=c['angle_eligible_outputs'],angle_rate=c['angle_correct']/c['angle_eligible_outputs'] if c['angle_eligible_outputs'] else None,
                    angle_eligible_frames=c['angle_eligible_frames'],coverage=out/total_frames,
                    center_full=c['center_correct']/total_frames,size_full=c['size_correct']/total_frames,angle_full_all_frames=c['angle_correct']/total_frames,angle_full=c['angle_correct']/c['angle_eligible_frames'] if c['angle_eligible_frames'] else None,
                    a_meanabs=sz['a']['abs_mae'],r_meanabs=sz['r']['abs_mae'])
                for edge in ('long','short'):
                    for metric in ('mae','p95','bias'):r[edge+'_'+metric+'_percent']=100*sz['edges'][edge][metric]
                    for metric in ('mean','p95'):r[edge+'_projected_gt_ratio_'+metric]=sz['projected_gt_axes'][edge][metric]
                if r['frames']!=count['frames'] or r['outputs']+r['missing']!=r['frames']:raise ValueError('Exported full-frame denominator differs')
                if not numeric_equal(r['coverage'],r['outputs']/count['frames']):raise ValueError('Exported coverage denominator differs')
                r.update(aggregate_temporal(m['temporal']))
                for domain,t in m['temporal'].items():
                    for k in ('dfr_percent_per_frame','dfr_adjacent_pairs','aci','aci_adjacent_pairs','mcml_max','mcml_mean','mcml_segments','mrf_mean','mrf_events'):r[domain+'_'+k]=t[k]
                    r[domain+'_tdr_hits']=t['tdr_w10']['numerator'];r[domain+'_tdr_denominator']=t['tdr_w10']['denominator']
                if a in ('c20','p07'):
                    r.update({k:methods[a+'_paired_vs_M'][k] for k in ('wrong_to_right','right_to_wrong','new_output','lost_output')})
                detection.append(r)
    csv_write(output/'detection_components_custom.csv',detection)
    csv_write(output/'test_fixed_M_near_edges.csv',near_edges)
    depth=[];pairs=[];geometry=[];propagation=[];historical_depth_replay={}
    names=['unknown_test_02_coverage_retry']
    for name in names:
        report=e.load(directory/name/'summary.json')
        if name!=e.depth.SEQUENCE:
            historical_root=ROOT/'work_dirs/port_depth_four_arm_v1/frozen_downstream_20261007_163625_1260566'
            key=name+'/summary.json';old_index=e.load(historical_root/'artifacts.json')['files']
            if e.depth.sha(historical_root/key)!=old_index[key]:raise ValueError('Historical depth summary SHA drift')
            old=e.load(historical_root/key)
            comparisons={a:numeric_equal(report['methods'][a]['depth_metrics'],old['groups'][historic]['depth_metrics']) for a,historic in [('b','symeood_b'),('midpoint','symeood_b_midpoint')]}
            if not all(comparisons.values()):raise ValueError('Same frozen B/M historical depth metrics differ')
            historical_depth_replay[name]=dict(pass_check=True,comparison=comparisons,summary_sha256=old_index[key])
        else:
            oracle=report['methods']['gt_obb']['depth_metrics']
            if not numeric_equal(oracle,e.depth.ORACLE):raise ValueError('Frozen Train03 GT oracle differs')
        rs=e.rows(directory/name/'depth_rows.jsonl')
        fixed=e.load(ROOT/'docs/data/webots_depth_formal_frame_index_v1.manifest.json')['constant_contract']
        calibration=e.load(ROOT/e.depth.CALIBRATION)
        for r in rs:
            for a in (*e.ARMS,'gt_obb'):
                box=r[a]['box']
                if box is not None:
                    actual=e.depth.depth(box,fixed['camera']['intrinsics'],fixed['obb_reference_geometry'],calibration['parameters'])
                    if not numeric_equal(actual,r[a]['depth']):raise ValueError('Raw-opt formula replay differs')
        for a,m in report['methods'].items():
            # Independently re-evaluate saved metre values using the frozen metric function.
            finite=[r for r in rs if r[a]['depth']['z_m'] is not None]
            expected=e.depth.error_metrics([r[a]['depth']['z_m'] for r in finite],[r['truth_z_m'] for r in finite])
            if not numeric_equal(expected,m['depth_metrics']):raise ValueError('Saved metric arithmetic differs')
            depth.append(dict(sequence=name,role=report['split'],arm=a,frames=m['frame_count'],outputs=m['output_frame_count'],numeric=m['numeric_depth_count'],numeric_failures=m['numeric_failure_count'],numeric_coverage=m['numeric_depth_coverage'],q_support=m['q_in_fit_support_count'],**m['depth_metrics']))
            finite_geometry=[r[a]['geometry'] for r in rs if r[a]['geometry'] is not None]
            geometry.append(dict(sequence=name,arm=a,count=len(finite_geometry),summary=m.get('geometry')))
        for a,value in report['propagation'].items():
            propagation.append(dict(sequence=name,arm=a,**value['signed_error_component_mean_m'],count=value['count'],max_identity_residual=value['max_signed_identity_residual_m']))
        common=[r for r in rs if all(r[a]['depth']['z_m'] is not None for a in e.ARMS)]
        if len(common)!=report['all_four_common_valid']['count']:raise ValueError('Common coverage differs')
        for a in e.ARMS:
            actual=e.depth.error_metrics([r[a]['depth']['z_m'] for r in common],[r['truth_z_m'] for r in common])
            if not numeric_equal(actual,report['all_four_common_valid']['methods'][a]):raise ValueError('Common metric replay differs')
        for key,p in report['paired'].items():
            before,after=key.split('->')
            if not numeric_equal(e.pair_depth(rs,before,after),p):raise ValueError('Paired depth changes differ')
            pairs.append(dict(sequence=name,pair=key,**p))
    csv_write(output/'depth_metric_errors.csv',depth);csv_write(output/'depth_paired_changes.csv',pairs)
    csv_write(output/'depth_geometry.csv',geometry);csv_write(output/'depth_signed_propagation.csv',propagation)
    e.write(output/'verification.json',dict(status='RECEIVED_ARTIFACTS_AND_METRE_ERRORS_VERIFIED',files_verified=len(index),analysis_script_sha256=e.depth.sha(__file__),calibration_sha256=e.depth.CAL_SHA,
        val_replay=done['val_replay'],historical_test_baseline_replay=baseline_replay,historical_depth_baseline_replay=historical_depth_replay,counters=done['counters'],peak_torch_allocated_mib=done['peak_torch_allocated_mib'],peak_torch_reserved_mib=done['peak_torch_reserved_mib'],
        max_sampled_process_mib=max(v['process_mib'] for v in done['memory_samples']),min_sampled_free_mib=min(v['free_mib'] for v in done['memory_samples'])))
    e.write(output/'artifacts.json',dict(files={p.name:e.depth.sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(detection=[r for r in detection if r['split']=='test' and r['group'] in ('overall','real','sim')],depth=depth,paired=pairs),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);run(p.parse_args().directory)
