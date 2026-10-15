#!/usr/bin/env python3
"""Fixed TRAIN visual review of the old pixel-border reader; no fitting."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_reliability_pixel_border_v1 as old
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.tools.analyze_port_reliability_severity_v1 import describe

VERSION = 'port_reliability_boundary_readout_review_v1'
GROUPS = ('correct', 'small_10_25', 'large_or_mixed')
SETTINGS = dict(per_group=12, selection='equal-sequence-round-robin-quota_then_even-frame-indices',
    tangents_source='unchanged_pixel_border_v1', no_fit=True, VAL_opened=False, TEST_opened=False,
    GT_scope='offline_selection_overlay_and_review_only', blind_panels=True,
    categories=['A_wrong_edge', 'B_not_visible', 'C_scope_mismatch', 'D_readout_loss', 'U_uncertain'])
MANIFEST = ROOT/'crane_project/tools/port_reliability_boundary_readout_review_v1_sources.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def group(row):
    d = describe(row)
    if d['severity'] == 'correct': return 'correct'
    if d['error'] is None: return None
    if .1 < d['error'] <= .25 and 'large' not in d['sign']: return 'small_10_25'
    if 'large' in d['sign']: return 'large_or_mixed'
    return None


def select(rows):
    """Offline GT strata, no quality-score or visual cherry-picking."""
    if len({r['image'] for r in rows}) != len(rows): raise ValueError('Duplicate frame')
    chosen, counts = [], {}
    for name in GROUPS:
        pools = {}
        for r in rows:
            if group(r) == name: pools.setdefault(r['sequence'], []).append(r)
        pools = {s: sorted(v, key=lambda r:(r['frame_id'], r['image'])) for s,v in sorted(pools.items())}
        counts[name] = {s:len(v) for s,v in pools.items()}
        quota = {s:0 for s in pools}
        for unused in range(min(SETTINGS['per_group'], sum(map(len, pools.values())))):
            available = [s for s in pools if quota[s] < len(pools[s])]
            s = min(available, key=lambda s:(quota[s], s))
            quota[s] += 1
        for s, pool in pools.items():
            n = quota[s]
            indices = [len(pool)//2] if n == 1 else np.linspace(0, len(pool)-1, n).astype(int).tolist() if n else []
            if len(set(indices)) != n: raise ValueError('Selection duplicate')
            chosen.extend(dict(group=name, row=deepcopy(pool[i])) for i in indices)
    return chosen, counts


def frame(box):
    cx,cy,L,S,a = map(float, box[:5])
    if L < S: L,S,a = S,L,a+math.pi/2
    a = (a+math.pi/2)%math.pi-math.pi/2
    u = np.array([math.cos(a), math.sin(a)])
    return np.array([cx,cy]), L,S,u,np.array([-u[1],u[0]])


def corners(box):
    c,L,S,u,v = frame(box)
    return np.array([c+x*L*u/2+y*S*v/2 for x,y in [(-1,-1),(1,-1),(1,1),(-1,1)]])


def profiles(gray, pred):
    """GT-free exact old sampling, retaining all profile locations/responses."""
    c,L,S,u,v = frame(pred)
    t = np.linspace(-old.SETTINGS['tangent_extent'], old.SETTINGS['tangent_extent'], old.SETTINGS['tangents'])
    radius = np.linspace(old.SETTINGS['radial_min'], old.SETTINGS['radial_max'], old.SETTINGS['radial_points'])
    result = {}
    for axis,normal,tangent,size,span in [('L',u,v,L,S),('S',v,u,S,L)]:
        sides = []
        for sign in (-1,1):
            origins = c+t[:,None]*span*tangent
            pts = origins[:,None,:]+sign*radius[None,:,None]*size*normal
            delta = old.SETTINGS['derivative_pixels']
            hi,a = old.sample(gray,pts+delta*normal)
            lo,b = old.sample(gray,pts-delta*normal)
            gradient = np.abs(hi-lo)/(2*delta)
            valid = (a & b).all(axis=1) & (gradient.max(axis=1)>1e-8)
            peaks = [float(np.median(radius[np.isclose(g,g.max(),rtol=0,atol=1e-12)])) if ok else None
                     for g,ok in zip(gradient,valid)]
            sides.append(dict(sign=sign, points=pts.tolist(), origin=origins.tolist(),
                normal=(sign*normal).tolist(), size=size, radius=radius.tolist(),
                gradient=gradient.tolist(), valid=valid.tolist(), peak=peaks))
        result[axis] = sides
    return result


def check_profiles(detail, ev):
    for axis in ('L','S'):
        for a,b in zip(detail[axis],ev[axis]['sides']):
            peaks = [x for x in a['peak'] if x is not None]
            if len(peaks) != b['valid_profiles'] or not np.allclose(peaks,b['peak_fractions'],atol=1e-14,rtol=0):
                raise ValueError('Profile peak replay differs')


def gt_interval(origin, direction, size, gt):
    """Offline rectangle intersection, not a claim of a visible object edge."""
    c,L,S,u,v = frame(gt)
    lower,upper = -math.inf, math.inf
    for axis,extent in ((u,L/2),(v,S/2)):
        a = float(np.dot(np.asarray(origin)-c,axis))
        b = float(np.dot(direction,axis))*size
        if abs(b) < 1e-12:
            if abs(a) > extent: return None
            continue
        lo,hi = sorted(((-extent-a)/b,(extent-a)/b))
        lower,upper = max(lower,lo),min(upper,hi)
    return [lower,upper] if lower <= upper else None


def font(size):
    for name in ('DejaVuSans.ttf','/System/Library/Fonts/Supplemental/Arial.ttf'):
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default()


COLORS = ['#ffbb33','#44ddff','#ff77aa','#bb88ff','#88ee88','#ffff88','#ffffff']


def render(image, pred, gt, detail, destination, title):
    canvas = Image.new('RGB',(1800,1120),'#181818'); draw=ImageDraw.Draw(canvas)
    draw.text((20,12),title,fill='white',font=font(23))
    pts = np.concatenate([corners(pred),corners(gt)]+[np.asarray(s['points']).reshape(-1,2) for a in detail.values() for s in a])
    low = np.floor(pts.min(0)-12).astype(int); high=np.ceil(pts.max(0)+12).astype(int)
    low=np.maximum(low,0);high=np.minimum(high,np.array(image.size))
    if np.any(high<=low):raise ValueError('Empty crop')
    crop=image.crop((*low,*high))
    scale=min(950/crop.width,910/crop.height)
    dims=tuple(max(1,int(round(v*scale))) for v in crop.size)
    crop=crop.resize(dims,Image.Resampling.NEAREST if hasattr(Image,'Resampling') else Image.NEAREST)
    origin=np.array([20+(950-dims[0])/2,60+(910-dims[1])/2])
    canvas.paste(crop,tuple(origin.astype(int)))
    def xy(p):return tuple(((np.asarray(p)-low)*scale+origin).tolist())
    for box,color in ((gt,'#55ff55'),(pred,'#ff5555')):
        polygon=[xy(p) for p in corners(box)]
        draw.line(polygon+[polygon[0]],fill=color,width=3)
    for ai,axis in enumerate(('L','S')):
        for si,side in enumerate(detail[axis]):
            x0,y0,w,h=1030,60+(ai*2+si)*235,735,190
            draw.rectangle((x0,y0,x0+w,y0+h),outline='#999999')
            draw.text((x0,y0-21),'%s %s | gradient vs radius / predicted axis'%(axis,'-' if si==0 else '+'),font=font(15),fill='white')
            radius=np.array(side['radius']); gradient=np.array(side['gradient']);ymax=max(float(gradient.max()),1e-8)
            def xx(r):return x0+(r-.25)/.5*w
            draw.line((xx(.5),y0,xx(.5),y0+h),fill='#ff5555',width=2)
            for i,(line,g,peak,valid) in enumerate(zip(side['points'],gradient,side['peak'],side['valid'])):
                color=COLORS[i]
                draw.line([xy(line[0]),xy(line[-1])],fill=color,width=1)
                draw.line([(xx(r),y0+h-float(z)/ymax*h) for r,z in zip(radius,g)],fill=color,width=1)
                interval=gt_interval(side['origin'][i],side['normal'],side['size'],gt)
                if interval is not None and .25<=interval[1]<=.75:
                    gx=xx(interval[1]); draw.line((gx,y0+h-12,gx,y0+h),fill='#55ff55',width=2)
                if valid:
                    p=np.array(side['origin'][i])+np.array(side['normal'])*side['size']*peak
                    x,y=xy(p);draw.ellipse((x-4,y-4,x+4,y+4),fill=color,outline='black')
                    px=xx(peak);draw.line((px,y0,px,y0+9),fill=color,width=2)
            for r in (.25,.5,.75):draw.text((xx(r)-13,y0+h+3),str(r),fill='white',font=font(12))
    draw.text((20,1000),'Red: delivered M | Green: GT rectangle (offline reference, not visible-edge truth)',fill='white',font=font(20))
    draw.text((20,1035),'7 colored profiles / side; dots: old selected peaks; green plot ticks: GT exit intersections',fill='white',font=font(18))
    draw.text((20,1070),'Native pixels enlarged for viewing only; no scores, error groups or accept flags shown.',fill='white',font=font(18))
    canvas.save(destination)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();manifest=json.loads(MANIFEST.read_text())
    if manifest['version']!=VERSION or manifest['scope']!='TRAIN_ONLY_VISUAL_NO_FIT':raise ValueError('Scope differs')
    for p,digest in manifest['pins'].items():
        if sha(ROOT/p)!=digest:raise ValueError('Pinned source differs: '+p)
    rows=list(map(json.loads,gzip.open(ROOT/manifest['rows'],'rt')))
    if len(rows)!=2558 or any(r['split'] not in ('train','train_sim') or r['reliability_role']!='train' for r in rows):raise ValueError('Full TRAIN only')
    states=Counter(describe(r)['state'] for r in rows)
    if dict(states)!=dict(FA=13,FR=1037,ED=58,CR=1450):raise ValueError('Frozen TRAIN states differ')
    selected,counts=select(rows)
    out=args.out.resolve()
    if out.name!='result' or out.parent.parent!=ROOT/'work_dirs'/VERSION:raise ValueError('Wrong output root')
    out.mkdir(parents=True,exist_ok=False);(out/'panels').mkdir()
    write(out/'selection.json',dict(settings=SETTINGS,available=counts,selected=[dict(image=x['row']['image'],group=x['group'],sequence=x['row']['sequence']) for x in selected]))
    pixel={r['image']:r for r in json.loads((ROOT/manifest['old_rows']).read_text())}
    data=json.loads((ROOT/manifest['data_manifest']).read_text())
    records={r['id']:r for r in data['records'] if r['split'] in ('train','train_sim')}
    policy=json.loads((ROOT/manifest['policy']).read_text())
    judge=simple.SimpleComponentReliability(policy['simple_policy'])
    details=[];review=[];timings=[]
    for i,item in enumerate(selected):
        row=item['row'];spec=records[row['image']]
        path=(ROOT/manifest['data_root']/spec['images']['path']).resolve()
        if (ROOT/manifest['data_root']).resolve() not in path.parents or sha(path)!=spec['images']['sha256']:raise ValueError('Image identity changed')
        cached=pixel[row['image']]
        if cached['pred']!=row['pred'] or cached['image_sha256']!=spec['images']['sha256']:raise ValueError('Cache binding differs')
        decision=judge.decide(row['pred'],row['image_size'])
        if any(decision[k]!=row['original_simple_decision'][k] for k in ('center_accepted','size_accepted','angle_accepted')):raise ValueError('Frozen flag differs')
        with Image.open(path) as im:
            im=im.convert('RGB')
            if list(im.size)!=row['image_size']:raise ValueError('Image dimensions differ')
            gray=np.asarray(im.convert('L'),dtype=float)
            start=time.perf_counter();ev=old.evidence(gray,row['pred']);detail=profiles(gray,row['pred']);timings.append(time.perf_counter()-start)
            if ev!=cached['evidence']:raise ValueError('Old full evidence replay differs: '+row['image'])
            check_profiles(detail,ev)
            panel='panels/%02d_%s.png'%(i+1,row['image'])
            render(im,row['pred'],row['gt'],detail,out/panel,'%02d | %s'%(i+1,row['image']))
        info=dict(image=row['image'],group=item['group'],sequence=row['sequence'],domain=row['domain'],
            pred=row['pred'],gt=row['gt'],image_size=row['image_size'],image_sha256=spec['images']['sha256'],
            errors=describe(row),simple_flags=decision,evidence=ev,profiles=detail,panel=panel)
        details.append(info)
        review.append(dict(image=row['image'],axis={a:dict(codes=[],reason='',confidence='unreviewed') for a in ('L','S')}))
        print('REPLAY',i+1,len(selected),row['image'],flush=True)
    # Separate blinded visual panels from GT-error strata/flags in numeric details.
    write(out/'details.json',details);write(out/'visual_review_template.json',review)
    for start in range(0,len(details),6):
        sheet=Image.new('RGB',(1800,1680),'#222222')
        for j,row in enumerate(details[start:start+6]):
            with Image.open(out/row['panel']) as im:
                im=im.resize((900,560));sheet.paste(im,((j%2)*900,(j//2)*560))
        sheet.save(out/('contact_%02d.jpg'%(start//6+1)),quality=94)
    report=dict(version=VERSION,settings=SETTINGS,selected=len(details),available=counts,
        selected_counts=dict(Counter(d['group'] for d in details)),full_TRAIN_states=dict(states),
        exact_old_evidence_replays=len(details),profile_replays=len(details)*28,flags_replays=len(details)*3,
        source_pins=manifest['pins'],manifest_sha256=sha(MANIFEST),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        fits=0,detector_inferences=0,VAL_opened=False,TEST_opened=False,TEED_loaded=False,
        CPU_evidence_plus_diagnostic_profile_ms=dict(mean=float(np.mean(timings)*1000),p95=float(np.quantile(timings,.95)*1000)),
        status='VISUAL_REVIEW_REQUIRED_NO_AUTOMATIC_CAUSAL_CLASSIFICATION',
        limitations=['GT rectangles are not pixel boundary annotations.','Selected TRAIN frames are descriptive, correlated, and not a population estimate.',
          'Category C and D cannot be established from a weak/ambiguous picture alone.','No reliability improvement or TEED benefit measured.'])
    write(out/'report.json',report)
    write(out/'artifacts.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    for p,digest in manifest['pins'].items():
        if sha(ROOT/p)!=digest:raise ValueError('Source changed during run: '+p)
    print(report['status'],flush=True)


if __name__=='__main__':main()
