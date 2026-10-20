#!/usr/bin/env python3
"""Post-fit S044 tables and frozen-reference diagnostics. No model/selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools.analyze_port_size_qalign_v1 import csv_write
from crane_project.utils import port_size_a20_protection_review_v1 as review
from crane_project.utils.port_size_a20_protection_downstream_v1 import ARMS,HEADS,SEQUENCE


def load(path):return json.loads(path.read_text())
def rows(path):return [json.loads(v) for v in path.read_text().splitlines()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verified(path):
    files=load(path/'artifacts.json')['files']
    if any(sha(path/n)!=h for n,h in files.items()):raise ValueError('Frozen artifact differs')
    return len(files)


def fixed_diagnostics(records,group,split):
    present=[r for r in records if r['midpoint'] is not None];out=[]
    errors={arm:[review.relative_errors(r[arm],r['gt'])['absolute'] for r in present] for arm in ARMS}
    for j,edge in enumerate(('long','short')):
        m,a=[v[j] for v in errors['midpoint']],[v[j] for v in errors['a20']]
        mask=[i for i,v in enumerate(m) if v<=.03]
        for arm in ARMS:
            new=[v[j] for v in errors[arm]];before=[m[i] for i in mask];after=[new[i] for i in mask]
            ret=review.capped_retention(m,a,new)
            out.append(dict(split=split,group=group,arm=arm,edge=edge,outputs=len(present),near_count=len(mask),
                near_before_mae=sum(before)/len(before) if before else None,near_after_mae=sum(after)/len(after) if after else None,
                near_harmed=sum(y>x+1e-12 for x,y in zip(before,after)),near_improved=sum(y<x-1e-12 for x,y in zip(before,after)),
                near_positive_harm_sum=sum(max(0.,y-x) for x,y in zip(before,after)),**{'gain_'+k:v for k,v in ret.items()}))
    return out


def summarize(run,analysis,historical,out):
    if out.exists() or out.parent!=run.parent:raise ValueError('Fresh sibling output required')
    counts=dict(run=verified(run),analysis=verified(analysis),historical=verified(historical))
    done=load(run/'completion.json')
    if done['status']!='FROZEN_NATIVE_VAL_TEST_DEPTH_COMPLETE' or done['training_updates']!=0:raise ValueError('Incomplete or changed scope')
    detection=list(csv.DictReader((analysis/'detection_components_custom.csv').open()))
    test=[r for r in detection if r['split']=='test'];unknown=rows(run/SEQUENCE/'depth_rows.jsonl')
    near=[];proof={}
    for split,count in [('val',887),('test',1440)]:
        current=rows(run/split/'evaluation_rows.jsonl');old=rows(historical/split/'evaluation_rows.jsonl')
        if len(current)!=len(old) or len(current)!=count:raise ValueError('Historical split differs')
        maxima={a:0. for a in ('b','midpoint','a20')}
        for r,s in zip(current,old):
            for k in ('image','gt','image_sha256','native_meta','sequence','domain'):
                if r[k]!=s[k]:raise ValueError('Historical identity/resize drift '+k)
            for arm in maxima:
                if (r[arm] is None)!=(s[arm] is None):raise ValueError('Historical presence differs')
                if r[arm] is not None:maxima[arm]=max(maxima[arm],max(abs(x-y) for x,y in zip(r[arm],s[arm])))
        if any(maxima.values()):raise ValueError('Historical baseline not exactly reproduced')
        proof[split]=dict(frames=count,max_box_component_abs_diff=maxima,frame_image_gt_resize_exact=True)
        if split=='test':
            groups={'overall':current}
            groups.update({d:[r for r in current if r['domain']==d] for d in ('real','sim')})
            groups.update({v:[r for r in current if r['sequence']==v] for v in sorted({r['sequence'] for r in current})})
            for name,rr in groups.items():near+=fixed_diagnostics(rr,name,'test')
    previous=rows(historical/SEQUENCE/'depth_rows.jsonl');maxima={a:0. for a in ('b','midpoint','a20')}
    if len(unknown)!=1097 or len(previous)!=1097:raise ValueError('Historical Unknown count differs')
    adapted=[]
    for r,s in zip(unknown,previous):
        for key in ('frame_id','truth_z_m','gt_box','theta_total_deg'):
            if r[key]!=s[key]:raise ValueError('Unknown truth/source drift')
        for arm in maxima:
            if r[arm]!=s[arm]:raise ValueError('Historical Unknown baseline differs '+arm)
        adapted.append(dict(gt=r['gt_box'],**{a:r[a]['box'] for a in ARMS}))
    proof['unknown']=dict(frames=1097,baseline_boxes_depth_geometry_exact=True)
    near+=fixed_diagnostics(adapted,SEQUENCE,'unknown')
    summary=load(run/SEQUENCE/'summary.json')
    for arm,value in summary['propagation'].items():
        bias=summary['methods'][arm]['depth_metrics']['bias_m']
        if abs(sum(value['signed_error_component_mean_m'].values())-bias)>1e-10:raise ValueError('Signed bias identity differs')
    inter=rows(run/'val/intermediate_replay.jsonl')
    proof['val_intermediates']={k:max(r[k] for r in inter) for k in inter[0] if k!='image'}
    out.mkdir()
    fields=['arm','group','AP75','mean_RIoU','DFR_percent_per_frame','frames','outputs','missing','coverage']
    csv_write(out/'test_overall_core.csv',[{k:r[k] for k in fields} for r in test if r['group']=='overall'])
    fields=['arm','group','long_mae_pct','short_mae_pct','a_abs_mae','r_abs_mae','size_correct','size_denominator','size_rate','size_full']
    csv_write(out/'test_coordination_core.csv',[{k:r[k] for k in fields} for r in test if r['group'] in ('real','sim')])
    csv_write(out/'fixed_M_near_A20_gain.csv',near)
    proof.update(verified_artifact_counts=counts,formula_signed_bias_verified=True,source_sha256=sha(Path(__file__)),
        reference_boundary='TEST/Unknown A20 gain sets are computed once only for frozen descriptive diagnostics; no training/reference batching/selection. Near harm tolerance1e-12 matches frozen S041 reports; gain membership1e-6 matches S043 capped definition.',
        no_model_inference_or_training=True)
    (out/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    (out/'artifacts.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['run','analysis','historical','out']:p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();summarize(a.run.resolve(),a.analysis.resolve(),a.historical.resolve(),a.out.resolve())
