#!/usr/bin/env python3
"""S037 frozen response analysis. Saved TRAIN-only tables; no model or fit."""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools.analyze_port_size_joint_v1 import sha,read,write,csv_write


def boolean(x):
    if x in (True,'True'):return True
    if x in (False,'False'):return False
    raise ValueError('Not a boolean '+str(x))


def stream(path):
    with path.open() as f:
        yield from csv.DictReader(f)


def strata(r):
    for group in ('overall',r['domain'],r['sequence']):
        yield r['scale'],group


def finish(cells,keys):
    rows=[]
    for k,v in sorted(cells.items()):
        r=dict(zip(keys,k),**dict(v))
        n=r.get('count',r.get('pairs',r.get('paired_count',0)))
        if 'direction_evaluable' in r:
            r['direction_correct_rate']=(r.get('direction_correct',0)/r['direction_evaluable']
                if 'direction_correct' in r and r['direction_evaluable'] else None)
        for name in ('before_abs_error_sum_pct','after_abs_error_sum_pct'):
            if name in r:r[name.replace('_sum','_mean')]=r[name]/n if n else None
        for name in ('both_directions_correct','both_errors_improved'):
            if name in r:r[name+'_rate']=r[name]/n if n else None
        for name in ('final_size_secant_sum','correction_secant_sum','abs_log_response_sum'):
            if name in r:r[name.replace('_sum','_mean')]=r[name]/n if n else None
        rows.append(r)
    return rows


def add_effect(c,prefix,delta):
    c[prefix+'_gain_sum']+=max(0.,-delta)
    c[prefix+'_harm_sum']+=max(0.,delta)
    # delta is in percentage points, frozen1e-6 relative tolerance =>1e-4pp.
    c[prefix+'_improved']+=int(delta < -1e-4)
    c[prefix+'_harmed']+=int(delta > 1e-4)


def analyze(source,out):
    if out.exists() or out.parent!=source.parent:raise ValueError('Fresh sibling analysis required')
    for n,s in read(source/'artifacts.json')['files'].items():
        if sha(source/n)!=s:raise ValueError('Response artifact SHA '+n)
    e=read(source/'engineering.json');p=read(source/'protocol.json')
    if e['status']!='FROZEN_TRAIN_RESPONSE_COMPLETE' or not e['real_frozen_state_unchanged'] or e['replay_count']!=5112:
        raise ValueError('Real replay not complete')
    if p['proposals']!=__import__('crane_project.tools.audit_port_size_direction_evidence_v1',fromlist=['protocol']).protocol()['proposals']:
        raise ValueError('Frozen proposal drift')
    response=defaultdict(Counter);other=defaultdict(Counter);visual=defaultdict(Counter);opposite=defaultdict(Counter)
    targets={};inventory=defaultdict(Counter);views=set();count=0
    # Only one scalar per perturbed-target response retained for cross-edge join.
    for r in stream(source/'responses.csv'):
        count+=1;views.add((r['image'],r['scale']))
        natural=r['slot']=='natural';edge=r['edge'];stage=r['stage'];slot=r['slot']
        stem=(r['image'],r['scale'],r['visual'],slot,stage)
        if not natural and not boolean(r['other_edge_response']):targets[stem]=boolean(r['improved'])
        for sc,group in strata(r):
            for qg in ('all',r['required_group']):
                c=response[sc,group,r['visual'],slot,stage,edge,qg]
                c['count']+=1;c['before_abs_error_sum_pct']+=100*abs(float(r['before_error']))
                c['after_abs_error_sum_pct']+=100*abs(float(r['after_error']))
                c['direction_evaluable']+=int(r['direction_correct']!='')
                c['direction_correct']+=int(r['direction_correct']!='' and boolean(r['direction_correct']))
                c['improved']+=boolean(r['improved']);c['harmed']+=boolean(r['harmed']);c['hold']+=boolean(r['hold'])
                c['direction_correct_but_harmed']+=int(r['direction_correct']!='' and boolean(r['direction_correct']) and boolean(r['harmed']))
                add_effect(c,'error_pp_edges',100*float(r['abs_error_delta']))
            if stage=='preguard' and edge=='long':
                c=inventory[sc,group,r['visual'],slot];c['valid_view_slots']+=1
                c['fallback']+=boolean(r['fallback']);c['accepted']+=boolean(r['delivery_accepted']);c['neutral']+=boolean(r['neutral'])
                if r['donor_support_absmean']!='':c['donor_support_difference_sum']+=float(r['donor_support_absmean'])
    for r in stream(source/'responses.csv'):
        if not boolean(r['other_edge_response']):continue
        key=(r['image'],r['scale'],r['visual'],r['slot'],r['stage'])
        target=targets.get(key)
        for sc,group in strata(r):
            for mg in ('all',r['fixed_M_group']):
                c=other[sc,group,r['visual'],r['slot'],r['stage'],r['edge'],mg]
                c['count']+=1
                delta=float(r['error_vs_same_visual_natural'])
                add_effect(c,'error_vs_natural_pp_edges',100*delta)
                add_effect(c,'error_vs_Q_GT_pp_edges',100*float(r['abs_error_delta']))
                c['abs_log_response_sum']+=abs(float(r['response_vs_same_visual_natural_log']))
                c['paired_target_evaluable']+=int(target is not None)
                c['target_improved']+=int(target is True)
                c['target_improved_other_harmed_vs_natural']+=int(target is True and delta>1e-6)
    for r in stream(source/'visual_pairs.csv'):
        for sc,group in strata(r):
            for qg in ('all',r['Q_group']):
                c=visual[sc,group,r['control'],r['slot'],r['stage'],r['edge'],qg]
                c['count']+=1;c['direction_evaluable']+=boolean(r['direction_evaluable'])
                c['only_normal_direction_correct']+=boolean(r['only_normal_direction_correct'])
                c['only_control_direction_correct']+=boolean(r['only_control_direction_correct'])
                a,b=boolean(r['normal_improved']),boolean(r['control_improved'])
                c['only_normal_error_improved']+=int(a and not b);c['only_control_error_improved']+=int(b and not a)
                c['both_error_improved']+=int(a and b);c['neither_error_improved']+=int(not a and not b)
                add_effect(c,'normal_pp_edges',100*float(r['normal_abs_error_delta']))
                add_effect(c,'control_pp_edges',100*float(r['control_abs_error_delta']))
                add_effect(c,'normal_minus_control_pp_edges',100*float(r['normal_minus_control_abs_error']))
                c['normal_fallback']+=boolean(r['normal_fallback']);c['control_fallback']+=boolean(r['control_fallback'])
    op_index={}
    for r in stream(source/'opposite_pairs.csv'):
        k=(r['image'],r['scale'],r['stage'],r['edge'],r['visual'])
        if k in op_index:raise ValueError('Duplicate opposite response pair')
        op_index[k]=r
        for sc,group in strata(r):
            for label,valid in (('all_axis_safe',True),('crossed_GT',boolean(r['crossed_GT'])),('both_nonnear',boolean(r['both_nonnear']))):
                if not valid:continue
                c=opposite[sc,group,r['visual'],r['stage'],r['edge'],label]
                c['pairs']+=1;c['both_directions_correct']+=boolean(r['both_directions_correct']);c['both_errors_improved']+=boolean(r['both_errors_improved'])
                c['final_size_secant_sum']+=float(r['final_size_secant']);c['correction_secant_sum']+=float(r['correction_secant'])
                c['either_fallback']+=int(boolean(r['minus_fallback']) or boolean(r['plus_fallback']))
    paired_opposite=defaultdict(Counter)
    for k,r in op_index.items():
        if r['visual']!='normal':continue
        for control in ('zero_visual_ablation','within_video_halfcycle'):
            x=op_index.get(k[:-1]+(control,))
            if x is None:continue
            for sc,group in strata(r):
                for label,valid in (('crossed_GT',boolean(r['crossed_GT'])),('both_nonnear',boolean(r['both_nonnear']))):
                    if not valid:continue
                    c=paired_opposite[sc,group,control,r['stage'],r['edge'],label];c['paired_count']+=1
                    for metric in ('both_directions_correct','both_errors_improved'):
                        a,b=boolean(r[metric]),boolean(x[metric]);c['only_normal_'+metric]+=int(a and not b);c['only_control_'+metric]+=int(b and not a)
    if len(views)!=5112:raise ValueError('Full output view coverage differs')
    source_identity=read(source/'source_identity.json')
    result=dict(protocol='port_size_direction_evidence_analysis_v1',source_artifacts_sha256=sha(source/'artifacts.json'),
        response_rows=count,observed_output_views=len(views),missing_views=e['missing_views'],
        source_identity=source_identity,replay=dict(count=e['replay_count'],maximum_absolute_difference=e['maximum_saved_replay_absolute_difference'],frozen_state_unchanged=e['real_frozen_state_unchanged']),
        unavailable_counts=dict(Counter(r['reason'] for r in read(source/'unavailable.json'))),
        analysis_code_sha256=sha(Path(__file__)),
        units='MAE fields retain sum_pct / count; edge gains/harms pp*edges; cross-GT pairs not independent samples; stages/slots never pooled as new images.',
        boundaries='TRAIN-only fixedV20. Primary preguard, delivered/fallback separate. Zero and donor may be OOD, do not prove ROI information sufficiency or insufficiency. No train/VAL/TEST/depth or new selector.')
    out.mkdir()
    for name,cells,keys in [
        ('response_summary.csv',response,('scale','group','visual','slot','stage','edge','Q_group')),
        ('cross_edge_summary.csv',other,('scale','group','visual','slot','stage','edge','fixed_M_group')),
        ('visual_summary.csv',visual,('scale','group','control','slot','stage','edge','Q_group')),
        ('bidirectional_summary.csv',opposite,('scale','group','visual','stage','edge','pair_group')),
        ('bidirectional_visual_pairs.csv',paired_opposite,('scale','group','control','stage','edge','pair_group')),
        ('inventory.csv',inventory,('scale','group','visual','slot'))]:csv_write(out/name,finish(cells,keys))
    write(out/'analysis.json',result);write(out/'artifacts.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(dict(response_rows=count,output_views=len(views),unavailable=result['unavailable_counts']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.source.resolve(),a.out.resolve())
