#!/usr/bin/env python3
"""CPU replay of completed S026 custom metrics and Unknown baseline identities.

No weights, fitting, selection or image inference. Run after primary analysis.
"""
from pathlib import Path
import sys,json,math,hashlib,csv
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import eval_port_size_proposal_downstream_v1 as e
from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
from crane_project.tools.analyze_port_size_proposal_downstream_v1 import numeric_equal,csv_write
r=Path(sys.argv[1]);out=r.parent/'analysis';checks=0;transitions=[];riou_differences=[];rounded_differences=[]
from copy import deepcopy
import numpy as np
from crane_project.tools.eval_crane_offline import compute_riou
server=e.load(out/'server_cpu_riou_replay.json');frame_differences=[]
for split in ('val','test'):
 rows=e.rows(r/split/'evaluation_rows.jsonl');saved=e.load(r/split/'summary.json')
 if [x['image'] for x in rows]!=[x['image'] for x in server['values'][split]]:raise ValueError('Server RIoU row identity')
 for x,y in zip(rows,server['values'][split]):
  for a in e.ARMS:
   v=compute_riou(np.asarray(metrics.old.canonical(x[a])),np.asarray(metrics.old.canonical(x['gt']))) if x[a] is not None else 0.
   frame_differences.append(abs(v-y['values'][a]))
   if (v>=.5)!=(y['values'][a]>=.5):raise ValueError('Cross-platform continuity hit differs')
 for arm in ('c20','p07'):
  value=metrics.summary([dict(x,edge_residual=x[arm]) for x in rows])
  for group,ms in value.items():
   for key,public in [('b','b'),('midpoint','midpoint'),('edge_residual',arm)]:
    target=saved[group][public]
    # OpenCV rotated intersection can round differently across platforms.
    # Keep all counts/flags and other values under the strict old comparator;
    # Continuous mean_RIoU uses the existing 1e-5 numeric tolerance. The
    # rounded legacy display may differ by one final 1e-4 unit. Formal values
    # remain the reproducible server values, never these local recomputations.
    actual=deepcopy(ms[key]);expected={k:deepcopy(target[k]) for k in actual}
    x,y=actual.pop('mean_riou_full_frame'),expected.pop('mean_riou_full_frame')
    riou_differences.append(abs(x-y))
    for domain,cell in list(actual['metric_protocol_v2'].items()):
     if domain.endswith('/mean_RIoU'):
      z=abs(actual['metric_protocol_v2'].pop(domain)-expected['metric_protocol_v2'].pop(domain));rounded_differences.append(z)
      if z>.0001+1e-12:raise ValueError('Rounded RIoU differs beyond final digit')
     elif isinstance(cell,dict) and 'mean_RIoU' in cell:
      z=abs(cell.pop('mean_RIoU')-expected['metric_protocol_v2'][domain].pop('mean_RIoU'));rounded_differences.append(z)
      if z>.0001+1e-12:raise ValueError('Rounded RIoU differs beyond final digit')
    if abs(x-y)>metrics.RIOU_TOL or not numeric_equal(actual,expected):raise ValueError('Custom metric replay mismatch '+split+'/'+group+'/'+public)
    checks+=1
 for group in saved:
  sub=[x for x in rows if group=='overall' or x['domain']==group or x['sequence']==group]
  correct=lambda x,a:max(abs(v/g-1) for v,g in zip(sorted(x[a][2:4]),sorted(x['gt'][2:4])))<=.1
  for before,after in [('b','midpoint'),('b','c20'),('b','p07'),('midpoint','c20'),('midpoint','p07'),('c20','p07')]:
   common=[x for x in sub if x[before] is not None and x[after] is not None]
   transitions.append(dict(split=split,group=group,pair=before+'->'+after,common_outputs=len(common),wrong_to_right=sum(not correct(x,before) and correct(x,after) for x in common),right_to_wrong=sum(correct(x,before) and not correct(x,after) for x in common),new_output=sum(x[before] is None and x[after] is not None for x in sub),lost_output=sum(x[before] is not None and x[after] is None for x in sub)))
csv_write(out/'all_pair_size_transitions.csv',transitions)
# Frozen Unknown B/M row-for-row, stronger than summary equality.
name='unknown_test_02_coverage_retry';depth=e.rows(r/name/'depth_rows.jsonl');old=e.rows((ROOT/'work_dirs/port_size_frozen_downstream_v1/run_20261017_v5_gpu3')/name/'depth_rows.jsonl')
if len(depth)!=1097 or len(old)!=1097:raise ValueError('Unknown full length')
for a,b in zip(depth,old):
 if any(not numeric_equal(a[k],b[k]) for k in ('frame_id','truth_z_m','gt_box','b','midpoint')):raise ValueError('Unknown B/M saved-row identity differs')
 checks+=1
s=e.load(r/name/'summary.json');geometry=[]
for arm in e.ARMS:
 sub=[x[arm]['geometry'] for x in depth if x[arm]['geometry'] is not None]
 geometry.append(dict(arm=arm,count=len(sub),long_MAE_percent=100*sum(abs(g['long_relative_error']) for g in sub)/len(sub),short_MAE_percent=100*sum(abs(g['short_relative_error']) for g in sub)/len(sub),long_bias_percent=100*sum(g['long_relative_error'] for g in sub)/len(sub),short_bias_percent=100*sum(g['short_relative_error'] for g in sub)/len(sub),aspect_meanabs=sum(abs(g['log_aspect_residual']) for g in sub)/len(sub)))
 for edge in ('long','short'):
  near=[x for x in depth if x['midpoint']['geometry'] is not None and abs(x['midpoint']['geometry'][edge+'_relative_error'])<=.03]
  vals=[x[arm]['geometry'][edge+'_relative_error'] for x in near if x[arm]['geometry'] is not None]
  geometry[-1][edge+'_M_near_n']=len(near);geometry[-1][edge+'_M_near_MAE_percent']=100*sum(map(abs,vals))/len(vals) if vals else None
  geometry[-1][edge+'_M_near_harmed']=sum(abs(x[arm]['geometry'][edge+'_relative_error'])>abs(x['midpoint']['geometry'][edge+'_relative_error'])+1e-12 for x in near if x[arm]['geometry'] is not None)
csv_write(out/'unknown_size_interpretation.csv',geometry)
e.write(out/'additional_verification.json',dict(status='ALL_CUSTOM_METRICS_AND_UNKNOWN_BASELINE_ROWS_REPLAYED',custom_cells=checks-1097,unknown_baseline_rows=1097,metric_protocol=2,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),transitions=len(transitions),mean_riou_cross_platform_atol=metrics.RIOU_TOL,max_mean_riou_cross_platform_difference=max(riou_differences),max_rounded_riou_difference=max(rounded_differences),max_frame_riou_difference=max(frame_differences),continuity_hit_flags_exact=True,server_cpu_mean_replay_difference=server['max_mean_difference'],note='Post-return OpenCV platform diagnostic only; server formal values and all adoption/selection contracts unchanged. Other custom values and all counts use original strict comparator.'))
print(json.dumps(dict(custom_replay_cells=checks-1097,geometry=geometry),indent=2))
