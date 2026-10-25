"""Independent local VAL verification from PKL, exported corners and GT.

Uses the protocol's canonical OBB normalization, with raw corner/file identity
and envelope checks, then independent polygon clipping and metric arithmetic.
No OpenCV or server metric helpers are imported locally. The project canonical
normalization is shared, not claimed as a second independent label decoder.
Does not load model weights, images, TEST, or deep-learning frameworks.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/media/omnisky/personal_files/ljj/symEOOD/'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rectangle(points):
    p=np.asarray(points,dtype=float).reshape(4,2)
    candidates=[]
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        edge=b-a;length=float(np.linalg.norm(edge))
        if length<1e-10:continue
        u=edge/length;v=np.array([-u[1],u[0]])
        x=p@u;y=p@v;w=float(x.max()-x.min());h=float(y.max()-y.min())
        center=u*(x.max()+x.min())/2+v*(y.max()+y.min())/2
        theta=math.atan2(u[1],u[0])
        if w<h:w,h,theta=h,w,theta+math.pi/2
        candidates.append((w*h,np.array([*center,w,h,(theta+math.pi/2)%math.pi-math.pi/2])))
    assert candidates
    return min(candidates,key=lambda z:z[0])[1]


def load_rect(path):
    words=path.read_text().split()
    if not words:return None
    return rectangle([float(x) for x in words[:8]])


def canonical_box(path,record):
    assert sha(path)==record['sha256'],path
    words=path.read_text().split()
    if not words:
        assert record['box'] is None
        return None
    box=np.asarray(record['box'],float)
    assert np.isfinite(box).all() and min(box[2:4])>0
    points=np.asarray([float(x) for x in words[:8]],float).reshape(4,2)
    c,s=math.cos(box[4]),math.sin(box[4]);relative=points-box[:2]
    projected=np.column_stack((relative@np.array([c,s]),relative@np.array([-s,c])))
    # OpenCV normalizes float32 corners. Preserve that protocol; verify that
    # the supplied rectangle is the actual raw-corner envelope at its axes.
    assert np.max(np.abs(projected.min(0)+box[2:4]/2))<.002,path
    assert np.max(np.abs(projected.max(0)-box[2:4]/2))<.002,path
    return box


def corners(b):
    c,s=math.cos(b[4]),math.sin(b[4]);x,y,w,h=b[:4]
    return [(x+c*u-s*v,y+s*u+c*v) for u,v in [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]]


def cross(a,b):return a[0]*b[1]-a[1]*b[0]
def sub(a,b):return a[0]-b[0],a[1]-b[1]
def polygon_area(p):return abs(sum(cross(a,b) for a,b in zip(p,p[1:]+p[:1])))/2 if len(p)>=3 else 0.


def overlap(a,b):
    if a is None:return 0.
    poly=corners(a);clip=corners(b)
    for q0,q1 in zip(clip,clip[1:]+clip[:1]):
        edge=sub(q1,q0);new=[]
        if not poly:break
        for p0,p1 in zip(poly[-1:]+poly[:-1],poly):
            d0=cross(edge,sub(p0,q0));d1=cross(edge,sub(p1,q0))
            inside0=d0>=-1e-10;inside1=d1>=-1e-10
            if inside0!=inside1:
                t=d0/(d0-d1)
                new.append((p0[0]+t*(p1[0]-p0[0]),p0[1]+t*(p1[1]-p0[1])))
            if inside1:new.append(p1)
        poly=new
    area=min(max(polygon_area(poly),0.),float(a[2]*a[3]),float(b[2]*b[3]))
    return area/max(a[2]*a[3]+b[2]*b[3]-area,1e-12)


def angle(a,b):return abs((a-b+math.pi/2)%math.pi-math.pi/2)


def summarize(rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[r['domain'],r['seq']].append(r)
    maxima=[];dfr=[];aci=[];windows=[];events=[];recovered=[]
    for rs in groups.values():
        rs.sort(key=lambda r:r['fid']);segments=[]
        for r in rs:
            if not segments or r['fid']!=segments[-1][-1]['fid']+1:segments.append([])
            segments[-1].append(r)
        for seg in segments:
            runs=[];run=0
            for r in seg:
                if r['riou_export']<.5:run+=1
                elif run:runs.append(run);recovered.append(run);run=0
            if run:runs.append(run)
            events.extend(runs);maxima.append(max(runs,default=0))
            hits=[r['riou_export']>=.5 for r in seg]
            windows.extend(any(hits[i:i+10]) for i in range(max(0,len(seg)-9)))
            for p,q in zip(seg,seg[1:]):
                if p['output'] and q['output']:
                    dfr.append(100*abs(q['diag_export']-p['diag_export'])/p['diag_export'])
                    aci.append(max(0.,1-angle(q['angle_export'],p['angle_export'])/(math.radians(35)+1e-9)))
    valid=[r for r in rows if r['output']];mean=lambda xs:statistics.mean(xs) if xs else None
    result=dict(frames=len(rows),outputs=len(valid),missing=len(rows)-len(valid),
                R_center_pct=100*sum(r['center_export']<15 for r in rows)/len(rows),
                center_correct=sum(r['center_error']<15 for r in valid),
                localization_hits=sum(r['riou_export']>=.5 for r in rows),
                mean_RIoU=mean([r['riou_export'] for r in rows]),
                mean_RIoU_native=mean([r['riou_native'] for r in rows]),
                MCML_max=max(maxima),MCML_mean=mean(maxima),MCML_segments=len(maxima),
                failure_events=len(events),MRF=mean(recovered),DFR=mean(dfr),DFR_pairs=len(dfr),ACI=mean(aci),
                TDR_hits=sum(windows),TDR_windows=len(windows),TDR_pct=100*mean(windows) if windows else None,
                gated_angle_RMSE_deg=math.sqrt(mean([r['gated_angle_sq'] for r in rows])),
                size_correct=sum(max(r['long_error'],r['short_error'])<=.1 for r in valid),
                angle_eligible=sum(r['angle_eligible'] for r in valid),
                angle_correct=sum(r['angle_eligible'] and r['angle_error_deg']<=3 for r in valid),
                meanabs_a=mean([abs(r['a']) for r in valid]),meanabs_r=mean([abs(r['r']) for r in valid]))
    for k in ('long','short'):
        xs=[r[k+'_error'] for r in valid]
        result[k+'_MAE_pct']=100*mean(xs) if xs else None
        result[k+'_P95_pct']=100*float(np.percentile(xs,95)) if xs else None
        result[k+'_bias_pct']=100*mean([r[k+'_bias'] for r in valid]) if valid else None
    for thr,name in [(.5,'AP50_independent'),(.75,'AP75_independent')]:
        ordered=sorted(valid,key=lambda r:-r['score']);tp=np.array([r['riou_native']>=thr for r in ordered],float)
        if not len(tp):result[name]=0.;continue
        recall=tp.cumsum()/len(rows);precision=tp.cumsum()/np.arange(1,len(tp)+1)
        result[name]=100*sum(max(precision[recall>=x],default=0.) for x in np.arange(0.,1.1,.1))/11
    return result


def main(base,out):
    out.mkdir(exist_ok=False);checks=0;maxdiff=collections.defaultdict(float)
    contract=json.loads((base/'protocol.json').read_text());assert contract['seeds']==[0]
    canonical=json.loads((base/'canonical_geometry_20261025.json').read_text())
    assert canonical['normalizer_sha256']==sha(ROOT/'crane_project/tools/analyze_symeood_ablation_first4_v1.py')
    for name,h in contract['sources'].items():assert sha(ROOT/name)==h
    prov=json.loads((base/'analysis_v1/provenance.json').read_text())
    for name,h in prov['outputs'].items():assert sha(base/'analysis_v1'/name)==h
    gtfiles=sorted(f for f in (ROOT/'crane_project/data/crane_grab_port_day2night_v1/val/annfiles').glob('*.txt') if not f.name.startswith('._'))
    assert len(gtfiles)==887
    inv={r['frame']:r for r in contract['data_inventory']}
    ground_truth={}
    for f in gtfiles:
        assert sha(f)==inv[f.stem]['annotation_sha256']
        ground_truth[f.stem]=canonical_box(f,canonical['GT'][f.stem])
    expected={(r['run'],r['epoch'],r['frame']):r for r in csv.DictReader((base/'analysis_v1/per_frame.csv').open())}
    official={(r['regression'],r['objective'],r['epoch'],r['domain']):r for r in csv.DictReader((base/'analysis_v1/metrics_all_epochs.csv').open())}
    metrics=[];cases=0
    for reg in ('iou','kld'):
        for obj in ('U','S','G'):
            run='s0_'+reg+'_'+obj.lower();work=base/run
            sweep=json.loads((work/'ckpt_sweep/sweep_results.json').read_text())
            for epoch,cp in sweep['all_checkpoints'].items():
                assert cp['results_pkl'].startswith(REMOTE)
                p=ROOT/cp['results_pkl'][len(REMOTE):];assert sha(p)==cp['results_pkl_sha256']
                with p.open('rb') as f:preds=pickle.load(f)
                assert len(preds)==887;rows=[]
                for gtfile,pred in zip(gtfiles,preds):
                    b=pred[0];assert len(b)<=1
                    gt=ground_truth[gtfile.stem]
                    e=canonical_box(p.parent/'Task1_grab'/gtfile.name,canonical['predictions'][run][epoch][gtfile.stem])
                    native=np.asarray(b[0][:5],float) if len(b) else None
                    assert (native is None)==(e is None)
                    domain,seq,fid=gtfile.stem.split('_');center=float(np.linalg.norm(e[:2]-gt[:2])) if e is not None else float('inf')
                    r=dict(frame=gtfile.stem,domain=domain,seq=seq,fid=int(fid),output=e is not None,
                           center_export=center,score=float(b[0][-1]) if len(b) else 0.,
                           riou_export=overlap(e,gt),riou_native=overlap(native,gt),
                           gated_angle_sq=(math.degrees(angle(e[4],gt[4])) if center<10 else 90.)**2)
                    if native is not None:
                        lp,sp=sorted(native[2:4],reverse=True);lg,sg=gt[2:4];el,es=math.log(lp/lg),math.log(sp/sg)
                        r.update(center_error=float(np.linalg.norm(native[:2]-gt[:2])),
                            long_error=abs(lp-lg)/lg,short_error=abs(sp-sg)/sg,long_bias=(lp-lg)/lg,short_bias=(sp-sg)/sg,a=(el+es)/2,r=el-es,
                            diag_export=float(np.hypot(e[2],e[3])),angle_export=float(e[4]),angle_eligible=lg/sg>=1.2,
                            angle_error_deg=math.degrees(angle(native[4]+(math.pi/2 if native[2]<native[3] else 0),gt[4])))
                    old=expected[run,epoch,r['frame']]
                    assert r['output']==(old['output']=='True')
                    for k,tol in [('center_export',.003),('riou_export',.0002),('riou_native',.0002),('center_error',.003),('angle_error_deg',.003)]:
                        if k not in r:continue
                        a0=float(old[k]);b0=r[k]
                        if math.isinf(a0):assert math.isinf(b0)
                        else:
                            d=abs(a0-b0);maxdiff[k]=max(maxdiff[k],d)
                            assert d<=tol,(run,epoch,r['frame'],k,a0,b0)
                        checks+=1
                    rows.append(r)
                for domain in ('real','sim','all'):
                    m=summarize([r for r in rows if domain=='all' or r['domain']==domain]);old=official[reg,obj,epoch,domain]
                    for key,value in m.items():
                        original=None if not old[key] else float(old[key])
                        if value is None:assert original is None
                        else:
                            # Independent double-precision geometry versus server float32.
                            tol=0 if key in ('frames','outputs','missing','center_correct','localization_hits','MCML_max','MCML_segments','failure_events','DFR_pairs','TDR_hits','TDR_windows','size_correct','angle_eligible','angle_correct') else .003 if key=='gated_angle_RMSE_deg' else .001
                            assert abs(value-original)<=tol,(run,epoch,domain,key,value,original)
                            maxdiff['metric_'+key]=max(maxdiff['metric_'+key],abs(value-original))
                        checks+=1
                    metrics.append(dict(run=run,epoch=epoch,domain=domain,**m))
                cases+=1
    receipt=dict(passed=True,cases=cases,frames=cases*887,aggregate_groups=len(metrics),checks=checks,
                 max_abs_differences=dict(maxdiff),method='original canonical OBB normalization (shared), raw file/envelope checks, independent double-precision polygon clipping and temporal metrics; no cv2/server metric helper/model imports',
                 canonical_geometry_sha256=sha(base/'canonical_geometry_20261025.json'),
                 source_sha256=sha(Path(__file__)),protocol_sha256=sha(base/'protocol.json'),
                 local_checkpoint_weights_loaded=False,TEST=0,depth=0)
    with (out/'verification.json').open('x') as f:json.dump(receipt,f,indent=2)
    with (out/'metrics_independent.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in metrics for k in r}));w.writeheader();w.writerows(metrics)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.base.resolve(),a.out.resolve())
