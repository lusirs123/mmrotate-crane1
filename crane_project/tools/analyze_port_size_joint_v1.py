#!/usr/bin/env python3
"""S027 saved TRAIN/VAL joint audit. CPU only; no model/cache/TEST inference.

Artificial slots have only aggregate error sums. Never reconstruct individual
responses from those sums or reinterpret them as delivered natural outputs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
VERSION = 'port_size_joint_v1'
EPS = 1e-12
PINS = dict(c20='fefbe410d84a3728137d0603ff8b0713bc0dd145fbc3727c4bb4aa114a70e974',
            p07='d1fa5587fb42fe30db6994a14eb2e345cd00c552b1718ba45e5c3f2e8b5ec6fd')
PROBE_SHA = 'deea12ebf763582b8685638db78f7911315b767471d4bea804918f21b1ceadca'
SLOTS = ('natural', 'long_plus', 'long_minus', 'short_plus', 'short_minus', 'long_exact', 'short_exact')
ROLES = dict(calibration_train_03_tilt_obb='calibration_diagnostic_exposed',
    fixed_dev_01_unseen_depth_swing='formula_development_exposed',
    unknown_test_02_coverage_retry='frozen_descriptive_test_exposed',
    calibration_train_01='historical_power_law_calibration_excluded',
    calibration_train_02_yaw_small_step='historical_yaw_audit_excluded',
    unknown_test_01_mixed_depth_swing='coverage_failed_no_reassignment',
    depth_pilot_04='storage_truth_pilot_excluded', depth_pilot_05_compact='storage_pilot_excluded')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())
def write(p, v):
    with Path(p).open('x') as f:json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False);f.write('\n')
def mean(v):return math.fsum(v)/len(v) if v else None
def pct(v):return 100*mean(v) if v else None
def quantile(v):
    if not v:return None
    v=sorted(v);k=(len(v)-1)*.95;i=int(k);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(k-i)
def csv_write(p, rs):
    fields=list(dict.fromkeys(k for r in rs for k in r))
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rs)


def canonical(box):
    if box is None:return None
    p=[float(x) for x in box[:5]]
    if len(p)!=5 or not all(math.isfinite(v) for v in p) or min(p[2:4])<=0:raise ValueError('Invalid box')
    if p[2]<p[3]:p[2],p[3],p[4]=p[3],p[2],p[4]+math.pi/2
    p[4]=(p[4]+math.pi/2)%math.pi-math.pi/2
    return p


def geometry(gt, pred, riou):
    if pred is None:return None
    g,p=canonical(gt),canonical(pred)
    e=[p[j]/g[j]-1 for j in (2,3)];logs=[math.log(p[j]/g[j]) for j in (2,3)]
    return dict(e=e,a=sum(logs)/2,r=logs[0]-logs[1],riou=riou(p,g),
                correct=max(map(abs,e))<=.1)


def state(delta):return 'better' if delta < -EPS else 'worse' if delta > EPS else 'tied'
def membership(g):
    if g is None:return 'missing'
    a,b=[abs(v)<=.03 for v in g['e']]
    return 'both_near' if a and b else 'long_near_short_error' if a else 'short_near_long_error' if b else 'both_error'


def bind(control, candidate, riou):
    if len(control)!=len(candidate):raise ValueError('Input counts differ')
    seen=set();result=[]
    for a,b in zip(control,candidate):
        for k in ('split','scale','image','sequence','frame_id','domain','gt','b','midpoint'):
            if a[k]!=b[k]:raise ValueError('Paired identity mismatch '+k)
        key=(a['split'],a['scale'],a['image'])
        if key in seen:raise ValueError('Duplicate paired view')
        seen.add(key)
        gt=a['gt'];m=a['midpoint'];c=a['edge_residual'];p=b['edge_residual']
        for x in (c,p):
            if (m is None)!=(x is None):raise ValueError('Output identity changed')
            if m is not None and (any(m[j]!=x[j] for j in (0,1,4,5)) or (m[2]<m[3])!=(x[2]<x[3])):raise ValueError('Geometry identity changed')
        geoms={k:geometry(gt,x,riou) for k,x in [('m',m),('c20',c),('p07',p)]}
        result.append(dict(split=a['split'],scale=a['scale'],image=a['image'],domain=a['domain'],sequence=a['sequence'],
            member=membership(geoms['m']),geoms=geoms,delivery=dict(c20=a['size_delivery'],p07=b['size_delivery'])))
    return result


def transition(x,y):
    return '/'.join(state(abs(b)-abs(a)) for a,b in zip(x['e'],y['e']))


def summarize(rs, before, after):
    present=[r for r in rs if r['geoms'][before] is not None and r['geoms'][after] is not None]
    x=[r['geoms'][before] for r in present];y=[r['geoms'][after] for r in present]
    result=dict(frames=len(rs),outputs=len(present),missing=len(rs)-len(present),
        new_output=sum(r['geoms'][before] is None and r['geoms'][after] is not None for r in rs),
        lost_output=sum(r['geoms'][before] is not None and r['geoms'][after] is None for r in rs),
        wrong_to_right=sum(not a['correct'] and b['correct'] for a,b in zip(x,y)),
        right_to_wrong=sum(a['correct'] and not b['correct'] for a,b in zip(x,y)),
        before_RIoU_full=math.fsum(g['riou'] for g in x)/len(rs) if rs else None,
        after_RIoU_full=math.fsum(g['riou'] for g in y)/len(rs) if rs else None,
        RIoU_improved=sum(b['riou']>a['riou']+EPS for a,b in zip(x,y)),RIoU_harmed=sum(b['riou']<a['riou']-EPS for a,b in zip(x,y)))
    for label,gs in [('before',x),('after',y)]:
        for j,edge in enumerate(('long','short')):
            es=[g['e'][j] for g in gs];result.update({label+'_'+edge+'_MAE_pct':pct(list(map(abs,es))),
                label+'_'+edge+'_bias_pct':pct(es),label+'_'+edge+'_P95_pct':100*quantile(list(map(abs,es))) if es else None})
        for key in ('a','r'):
            vs=[g[key] for g in gs];result[label+'_'+key+'_meanabs']=mean(list(map(abs,vs)));result[label+'_'+key+'_bias']=mean(vs)
    result['both_changed']=sum(all(abs(math.log1p(b['e'][j])-math.log1p(a['e'][j]))>EPS for j in (0,1)) for a,b in zip(x,y))
    if rs and rs[0]['member'] in ('long_near_short_error','short_near_long_error') and all(r['member']==rs[0]['member'] for r in rs):
        near=0 if rs[0]['member']=='long_near_short_error' else 1;wrong=1-near
        # Correct/wrong membership is FIXED M; C20->P07 improvements still
        # compare those two outputs, never relabel the samples by C20.
        corrected=[(a,b) for a,b in zip(x,y) if abs(b['e'][wrong])<abs(a['e'][wrong])-EPS]
        result.update(error_edge_improved=len(corrected),accurate_edge_harmed=sum(abs(b['e'][near])>abs(a['e'][near])+EPS for a,b in zip(x,y)),
            error_improved_accurate_harmed=sum(abs(b['e'][near])>abs(a['e'][near])+EPS for a,b in corrected),
            error_improved_accurate_not_harmed=sum(abs(b['e'][near])<=abs(a['e'][near])+EPS for a,b in corrected))
    return result


def natural_audit(records):
    summary=[];patterns=[];scale_aspect=[]
    for split in ('train','val'):
        for scale in sorted({r['scale'] for r in records if r['split']==split}):
            base=[r for r in records if r['split']==split and r['scale']==scale]
            groups=dict(overall=base)
            groups.update({d:[r for r in base if r['domain']==d] for d in ('real','sim')})
            groups.update({v:[r for r in base if r['sequence']==v] for v in sorted({r['sequence'] for r in base})})
            for group,rs in groups.items():
                for before,after in [('m','c20'),('m','p07'),('c20','p07')]:
                    stem=dict(split=split,scale=scale,group=group,pair=before+'->'+after)
                    for member in ('all','both_near','long_near_short_error','short_near_long_error','both_error','missing'):
                        sub=rs if member=='all' else [r for r in rs if r['member']==member]
                        summary.append(dict(stem,M_membership=member,**summarize(sub,before,after)))
                    p={a+'/'+b:[] for a in ('better','worse','tied') for b in ('better','worse','tied')}
                    ar={a+'/'+b:[] for a in ('better','worse','tied') for b in ('better','worse','tied')}
                    for row in rs:
                        a,b=row['geoms'][before],row['geoms'][after]
                        if a is None or b is None:continue
                        p[transition(a,b)].append(row)
                        ar[state(abs(b['a'])-abs(a['a']))+'/'+state(abs(b['r'])-abs(a['r']))].append(row)
                    for label,sub in p.items():patterns.append(dict(stem,long_short_change=label,**summarize(sub,before,after)))
                    for label,sub in ar.items():scale_aspect.append(dict(stem,scale_aspect_change=label,**summarize(sub,before,after)))
                    count=sum(len(v) for v in p.values());assert count==sum(r['geoms'][before] is not None for r in rs)
    return summary,patterns,scale_aspect


def artificial(probe):
    result=[]
    if probe['status']!='FROZEN_COMMON_TRAIN_PROPOSAL_PROBE_COMPLETE':raise ValueError('Not common-input probe')
    for arm,public in [('repeat7','c20'),('proposal7','p07')]:
        cells=probe['inventories'][arm]
        for domain in ('real','sim'):
            for scale in (.5,1.):
                prefix=domain+'/'+str(scale)+'/';natural=cells[prefix+'natural'];n=natural['count']
                for slot in SLOTS:
                    cell=cells[prefix+slot]
                    if cell['count']!=n:raise ValueError('Unequal artificial paired exposure counts')
                    # All seven slots are the same complete TRAIN view set.
                    # This is a change in mean error, not mean |response|.
                    for j,edge in enumerate(('long','short')):
                        before=cell['before_abs_sum'][j]/n;after=cell['after_abs_sum'][j]/n
                        untouched=(slot.startswith('long_') and edge=='short') or (slot.startswith('short_') and edge=='long')
                        if untouched and abs(before-natural['before_abs_sum'][j]/n)>1e-8:raise ValueError('Untouched input edge errors changed')
                        result.append(dict(arm=public,domain=domain,scale=scale,slot=slot,edge=edge,count=n,untouched_edge=untouched,
                            before_MAE_pct=100*before,after_MAE_pct=100*after,
                            after_minus_natural_MAE_pp=100*(after-natural['after_abs_sum'][j]/n),
                            harmed_vs_slot_input=cell['harmed'][j],improved_vs_slot_input=cell['improved'][j],
                            natural_harmed_vs_input=natural['harmed'][j],axis_cross=cell['axis_cross'],fallback_whole_pair=cell['fallback'],
                            output_stage='PRE_DELIVERY_FIXED_NATURAL_AXIS',individual_response='MISSING',near_conditioned_response='MISSING'))
    return result


def depth_inventory(root, formal, local_catalog):
    result=[];root=Path(root)
    known={s['sequence_id']:s for s in formal['sequence_summaries']};constant=formal['constant_contract']
    actual={p.name:p for p in root.iterdir() if p.is_dir()}
    union=set(actual)|set(local_catalog.get('sequences',[]))|set(ROLES)
    for name in sorted(union):
        p=actual.get(name);f=p/'metadata/manifest.json' if p else None;tp=p/'metadata/frames.jsonl' if p else None
        m=read(f) if f and f.is_file() else None;entry=known.get(name)
        r=dict(sequence=name,server_present=p is not None,local_catalog_present=name in local_catalog.get('sequences',[]),
            historical_role=ROLES.get(name,'UNREVIEWED_NO_AUTOMATIC_USE'),new_independent_validation=False,
            manifest_sha256=sha(f) if m else None,frames_sha256=sha(tp) if tp and tp.is_file() else None,
            metric_depth_evaluations=0)
        if m:
            r.update(split=m.get('split'),camera_contract_match=m.get('camera')==constant['camera'],
                physical_reference_match=all(math.isclose(m['obb_reference_geometry'][k],constant['obb_reference_geometry'][k],abs_tol=1e-10) for k in ('long_edge_mean_m','short_edge_mean_m')),
                frame_count=sum(1 for s in tp.open() if s.strip()) if tp.is_file() else 0,
                image_count=len(list((p/'images').glob('*.jpg'))),label_count=len(list((p/'labels').glob('*.txt'))))
            if entry:
                r['formal_metadata_sha_match']=r['manifest_sha256']==entry['manifest_sha256'] and r['frames_sha256']==entry['frames_jsonl_sha256']
                if not r['formal_metadata_sha_match'] or r['frame_count']!=entry['frame_count']:raise ValueError('Formal depth metadata drift')
        result.append(r)
    return result


def run(args):
    import numpy as np
    from crane_project.tools.eval_crane_offline import compute_riou
    source=read(ROOT/'crane_project/tools/port_size_joint_v1_sources.json')
    if any(sha(ROOT/n)!=h for n,h in source['sources'].items()):raise ValueError('Source drift')
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    run=args.proposal_run;index=read(run/'artifacts.json')['files'];used={};records=[]
    def checked(name):
        p=run/name
        if sha(p)!=index[name]:raise ValueError('Saved input SHA drift '+name)
        used[name]=index[name];return p
    done=read(checked('completion.json'))
    for arm,public,ep in [('repeat7','c20',20),('proposal7','p07',7)]:
        pick=done['selections'][arm]['selected'] or done['selections'][arm]['representative']
        if pick['epoch']!=ep or pick['checkpoint']['sha256']!=PINS[public] or index[arm+'/head_epoch_%02d.pth'%ep]!=PINS[public]:raise ValueError('Frozen representative identity')
    for split,n in [('train',5116),('val',887)]:
        paired=[]
        for arm,ep in [('repeat7',20),('proposal7',7)]:
            with gzip.open(checked(arm+'/'+split+'_epoch_%02d.rows.jsonl.gz'%ep),'rt') as f:rs=[json.loads(s) for s in f]
            if len(rs)!=n:raise ValueError('Incomplete '+split)
            paired.append(rs)
        records.extend(bind(*paired,lambda p,g:compute_riou(np.asarray(p),np.asarray(g))))
    if Counter((r['split'],r['scale']) for r in records)!=Counter({('train',.5):2558,('train',1.):2558,('val',1.):887}):raise ValueError('Paired view coverage')
    if sha(args.probe)!=PROBE_SHA:raise ValueError('Common probe bytes drift')
    probe=read(args.probe)
    if any(probe['selections'][a]['sha256']!=PINS[p] for a,p in [('repeat7','c20'),('proposal7','p07')]):raise ValueError('Common probe model identity')
    summary,patterns,scale_aspect=natural_audit(records)
    csv_write(out/'natural_joint_groups.csv',summary);csv_write(out/'natural_edge_patterns.csv',patterns);csv_write(out/'natural_scale_aspect.csv',scale_aspect)
    csv_write(out/'artificial_aggregate_responses.csv',artificial(probe))
    with gzip.open(out/'natural_joint_rows.jsonl.gz','xt') as f:
        for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
    formal=read(ROOT/'docs/data/webots_depth_formal_frame_index_v1.manifest.json')
    local=read(args.local_catalog);csv_write(out/'depth_data_inventory.csv',depth_inventory(args.depth_root,formal,local))
    report=dict(protocol=VERSION,status='SAVED_JOINT_AUDIT_COMPLETE_NO_NEW_INFERENCE',utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(Path(__file__)),source_manifest_sha256=sha(ROOT/'crane_project/tools/port_size_joint_v1_sources.json'),
        input_artifact_index_sha256=sha(run/'artifacts.json'),input_files=used,common_probe_sha256=sha(args.probe),
        heads=PINS,views=len(records),unique_train_images=len({r['image'] for r in records if r['split']=='train'}),
        outputs=sum(r['geoms']['m'] is not None for r in records),training_updates=0,model_inference_calls=0,weights_loaded=False,
        gpu_used=False,test_read=False,unknown_predictions_read=False,depth_error_recomputed=False,selection=False,
        missing_evidence=['Artificial paired per-frame outputs/deltas across slots are not saved.',
          'No fixed-M-near conditional artificial cross-edge responses; aggregate mean-error changes cannot supply damage intersections.',
          'Pre-delivery artificial error sums are not natural delivered-box RIoU or per-frame response magnitudes.'],
        cohort_count=len(summary),edge_pattern_count=len(patterns),scale_aspect_count=len(scale_aspect),
        local_catalog_sha256=sha(args.local_catalog),root_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip())
    write(out/'analysis.json',report);write(out/'artifacts.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_files',)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('proposal-run','probe','depth-root','local-catalog','out'):p.add_argument('--'+n,required=True,type=Path)
    run(p.parse_args())
