#!/usr/bin/env python3
"""Bounded S024 saved-output transfer audit; no model/inference/fitting/selection.

S020 cutpoints are reproduced from standard Sim TRAIN, never from Webots.
All depth rows are historical frozen outputs; no depth formula is rerun.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'port_size_transfer_v1'
EPS = 1e-12
ARMS = ('b', 'midpoint', 'i07', 'axis15')
DEPTH = ('calibration_train_03_tilt_obb', 'fixed_dev_01_unseen_depth_swing', 'unknown_test_02_coverage_retry')
CUTS = (4.088494867552162, 1.0736503335414151)
RANGES = ((3.4349903337863372, 4.8714420418175575), (0.9766445904730451, 1.1638008750148097),
          (-0.06782563249980392, 0.07654093330308595), (-0.0420899172724367, 0.04035168897038428))
PINS = dict(i07='affe4795b73cae325f878b15c2d81f942461bd965bae0b3c3ca6a7a944eb0464',
            axis15='7af0022adf214750789dbf71490ec9a07de69e14d89b8167c3d0e62cdc3950db')

ANNOTATION_REVIEW = {'location': 'Local read-only source review; server receives interpretation and source SHA, not external source copies', 'source_sha256': {'/Users/mac/Documents/fangzhen/grab_副本/worlds/crane.wbt': '124d5d50300dca5e54a033d9a2ef30ecb8cd14ca2a3543bbc58f98968c295074', '/Users/mac/Documents/fangzhen/grab_副本2/worlds/crane.wbt': '96ca6422cfa9625051b9895130b97fb074e68d88d76fd1b892f60c41b69f8750', '/Users/mac/Documents/fangzhen/grab_副本/controllers/crane_driver_obb/crane_driver_obb.py': 'b6f5769b6b8fa81fb126509188549d93f5dd53761937f7d59911e22659ad75bc', '/Users/mac/Documents/fangzhen/grab_副本2/controllers/crane_driver_obb/crane_driver_obb.py': 'a8885c1cde1bba9dc1d05ebeaeb3290ec3932c72ce88a5f4607403d8db4368c0', '/Users/mac/Documents/paper/dates/webots_obb_dataset/convert_yolo_obb_to_mmrotate.py': '0a7d75483b73e8fc61b816862d38c2c67ffedb1deaf339c696d51a41adf1a808'}, 'corner_translation_blocks_identical': True, 'existing_label_correspondence': {'sim_seq08': {'frames': 748, 'byte_identical': 18, 'source_files_present': 18, 'source_files_missing': 730, 'source_format_tokens': [10]}, 'sim_seq10': {'frames': 512, 'byte_identical': 0, 'source_files_present': 512, 'source_files_missing': 0, 'source_format_tokens': [6], 'normalized_YOLO_to_DOTA_max_coordinate_deviation_px': 0.04998915293799655, 'matched_with_coordinate_rounding_0p051px': 512}, 'sim_seq09': {'frames': 572, 'byte_identical': 0, 'source_files_present': 0, 'source_files_missing': 572, 'source_format_tokens': []}}, 'limits': 'Current source compatibility is not acquisition-time binding. Detection Sim capture controller/world snapshot and original projected corners remain missing. A049 internal physical/camera audit reused, not rerun. Available old source-label folder is partial and mixed DOTA/YOLO; byte differences across formats do not establish a label inconsistency.'}

def annotation_contract():
    sources = "A049; tools/data/prepare_port_size_reference_review_v1.py; local source SHA in analysis.json"
    return [
        dict(item="object_part", detection_sim="central top-beam reference; README and representative overlays", webots="Target_Box; GRAB_PT_TL/TR/BL/BR top-beam points", status="COMPATIBLE_NOT_FULLY_CONFIRMED", source=sources),
        dict(item="GT_generation", detection_sim="existing DOTA four-point labels preserved; original capture binding missing", webots="3D corner projection -> cv2.minAreaRect -> long-axis canonical box", status="UNCONFIRMED_ACQUISITION_BINDING", source="dates/webots_obb_dataset/convert_yolo_obb_to_mmrotate.py; crane_driver_obb.py:206-295,409-471"),
        dict(item="physical_reference", detection_sim="no per-frame original 3D-corner/world snapshot", webots="A049: 1.7750000000 x 0.6100409809 m; current two world corner translations match", status="CURRENT_DEFINITIONS_MATCH_HISTORICAL_BINDING_UNCONFIRMED", source=sources),
        dict(item="data_identity", detection_sim="sim_seq08/10/09 distinct detection sequences", webots="Train03/Fixed-dev/Unknown02 retain separate original roles", status="DIFFERENT_SEQUENCES_NOT_A_LABEL_CONFLICT", source="dataset README section 5; frozen S023 input_identity.json"),
    ]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):h.update(chunk)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())
def mean(xs):return math.fsum(xs)/len(xs) if xs else None


def percentile(xs, q=.95):
    if not xs:return None
    xs=sorted(xs); k=(len(xs)-1)*q; i=int(k)
    return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(k-i)


def canonical(box):
    if box is None:return None
    p=list(box[:5])
    if not all(math.isfinite(x) for x in p) or min(p[2:4])<=0:raise ValueError('Invalid box')
    if p[2]<p[3]:p[2],p[3],p[4]=p[3],p[2],p[4]+math.pi/2
    p[4]=(p[4]+math.pi/2)%math.pi-math.pi/2
    return p


def features(row):
    m,b=canonical(row['midpoint']),canonical(row['b'])
    return (math.log(math.sqrt(m[2]*m[3])),math.log(m[2]/m[3]),math.log(m[2]/b[2]),math.log(m[3]/b[3]))


def errors(row, arm):
    p,g=canonical(row[arm]),canonical(row['gt'])
    if p is None:return None
    es=[p[j]/g[j]-1 for j in (2,3)]; ls=[math.log1p(x) for x in es]
    return dict(e=es,a=sum(ls)/2,r=ls[0]-ls[1])


def grid(row):return '/'.join('high' if v>c else 'low' for v,c in zip(features(row),CUTS))
def inside(row, ranges):return all(lo<=v<=hi for v,(lo,hi) in zip(features(row),ranges))
def edge_group(row, j):
    e=errors(row,'midpoint')['e'][j]
    return 'near3' if abs(e)<=.03 else 'undersized' if e<0 else 'oversized'


def check_identity(row):
    for arm in ('i07','axis15'):
        m,p=row['midpoint'],row[arm]
        if (m is None)!=(p is None) or (m is not None and any(m[k]!=p[k] for k in (0,1,4,5))):
            raise ValueError('Size-head output identity drift')
    for a in ARMS:canonical(row[a])
    canonical(row['gt'])


def stats(rs, arm):
    es=[errors(r,arm) for r in rs if r[arm] is not None]
    ans=dict(frames=len(rs),outputs=len(es),missing=len(rs)-len(es),size10=sum(max(map(abs,x['e']))<=.1 for x in es),
             a_meanabs=mean([abs(x['a']) for x in es]),r_meanabs=mean([abs(x['r']) for x in es]))
    for j,edge in enumerate(('long','short')):
        vals=[x['e'][j] for x in es]
        ans.update({edge+'_mae_pct':100*mean(list(map(abs,vals))) if vals else None,
                    edge+'_bias_pct':100*mean(vals) if vals else None,
                    edge+'_p95_pct':100*percentile(list(map(abs,vals))) if vals else None})
    return ans


def paired(rs, arm, j):
    common=[r for r in rs if r['midpoint'] is not None and r[arm] is not None]
    before=[errors(r,'midpoint') for r in common]; after=[errors(r,arm) for r in common]
    ds=[abs(b['e'][j])-abs(a['e'][j]) for a,b in zip(before,after)]
    near=[(a,b) for a,b in zip(before,after) if abs(a['e'][j])<=.03]
    result=dict(common=len(common),improved=sum(x < -EPS for x in ds),worsened=sum(x>EPS for x in ds),tied=sum(abs(x)<=EPS for x in ds),
        near_n=len(near),near_harmed=sum(abs(b['e'][j])-abs(a['e'][j])>EPS for a,b in near),
        near_mae_before_pct=100*mean([abs(a['e'][j]) for a,b in near]) if near else None,
        near_mae_after_pct=100*mean([abs(b['e'][j]) for a,b in near]) if near else None,
        near_to_over3=sum(abs(b['e'][j])>.03 for a,b in near),near_to_over10=sum(abs(b['e'][j])>.1 for a,b in near),
        wrong_to_right=sum(max(map(abs,a['e']))>.1 and max(map(abs,b['e']))<=.1 for a,b in zip(before,after)),
        right_to_wrong=sum(max(map(abs,a['e']))<=.1 and max(map(abs,b['e']))>.1 for a,b in zip(before,after)))
    for direction in ('enlarge','shrink','unchanged'):
        selected=[]
        for r,d in zip(common,ds):
            update=canonical(r[arm])[j+2]/canonical(r['midpoint'])[j+2]-1
            if ('enlarge' if update>EPS else 'shrink' if update < -EPS else 'unchanged')==direction:selected.append(d)
        result.update({direction+'_n':len(selected),direction+'_improved':sum(x < -EPS for x in selected),direction+'_harmed':sum(x>EPS for x in selected)})
    return result


def indexed(directory, name, used):
    directory=Path(directory);idx=read(directory/'artifacts.json')['files'];p=directory/name
    if name not in idx or sha(p)!=idx[name]:raise ValueError('Missing/changed evidence '+str(p))
    used[str(p)]=idx[name]
    if name.endswith('jsonl.gz'):
        with gzip.open(p,'rt') as f:return [json.loads(s) for s in f]
    if name.endswith('jsonl'):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    return read(p)


def join_training(irows, arows, expected):
    key=lambda r:(r['image'],r['scale'])
    ix={key(r):r for r in irows};ax={key(r):r for r in arows}
    if len(ix)!=expected or len(irows)!=expected or len(arows)!=expected or ix.keys()!=ax.keys():raise ValueError('Incomplete/duplicate/mismatched saved rows')
    result=[]
    for k,r in ix.items():
        s=ax[k]
        for f in ('gt','b','midpoint','domain','sequence','split','shard','target_log'):
            if r[f]!=s[f]:raise ValueError('Baseline drift '+f)
        row=dict(r,i07=r['edge_residual'],axis15=s['edge_residual']);check_identity(row);result.append(row)
    return result


def cache_metadata(directory, training, used):
    """Read detached cache on CPU, never deserialize a model/checkpoint."""
    import torch
    manifest=read(directory/'cache_manifest.json')
    if sha(directory/'cache_manifest.json')!='046c5998dee0ba3703f1ae4e08fc6a02e9804d216ea12241357f69f2a4afd1e3':raise ValueError('Cache manifest drift')
    used[str(directory/'cache_manifest.json')]=sha(directory/'cache_manifest.json');found={}
    for name in ('train_s1.pt','train_s05.pt','val_s1.pt'):
        p=directory/name
        if sha(p)!=manifest['files'][name]:raise ValueError('Cache shard drift')
        used[str(p)]=manifest['files'][name]
        payload=torch.load(p,map_location='cpu')
        for r in payload['records']:
            n=len(r['boxes_original']);key=(r['image'],r['scale'])
            if key in found:raise ValueError('Duplicate cache metadata')
            found[key]=dict(b=r['boxes_original'][0].tolist() if n else None,
                xy=r['scale_xy'][0].tolist() if n else None,
                support_mean=float(r['support'].mean()) if n else None,
                support_shape=list(r['support'].shape),image_size=r['image_size'])
        del payload
    for row in training:
        meta=found[(row['image'],row['scale'])]
        if meta['b']!=row['b']:raise ValueError('Cache B row differs')
        row['roi_meta']=meta
    return dict(status='CPU_CACHE_METADATA_ONLY',shards=3,records=len(found),cuda_used=False)


def write_csv(p, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def run(args):
    if args.out.exists():raise ValueError('Refuse existing output')
    used={};allrows={};ir=args.i_run;ar=args.axis_run;fr=args.frozen_run
    for folder,arm,ep in ((ir,'i07',7),(ar,'axis15',15)):
        idx=read(folder/'artifacts.json')['files']
        name='candidate/head_epoch_%02d.pth'%ep
        if idx.get(name)!=PINS[arm] or sha(folder/name)!=PINS[arm]:raise ValueError('Wrong frozen head pin')
        used[str(folder/name)]=PINS[arm]
    for split,expected in (('train',5116),('val',887)):
        rs=join_training(indexed(ir,'candidate/'+split+'_epoch_07.rows.jsonl.gz',used),indexed(ar,'candidate/'+split+'_epoch_15.rows.jsonl.gz',used),expected)
        for domain in ('real','sim'):
            for scale in ((1.,.5) if split=='train' else (1.,)):
                allrows[split+'/'+domain+'/'+str(scale)]=[r for r in rs if r['domain']==domain and r['scale']==scale]
    standard=allrows['train/sim/1.0'];xs=[features(r) for r in standard if r['midpoint'] is not None]
    if len(xs)!=748 or any(abs(percentile([x[j] for x in xs],.5)-CUTS[j])>1e-12 for j in (0,1)):
        raise ValueError('S020 cutpoint mismatch')
    simranges=[(min(x[j] for x in xs),max(x[j] for x in xs)) for j in range(4)]
    if any(abs(a-b)>1e-12 for x,y in zip(simranges,RANGES) for a,b in zip(x,y)):raise ValueError('S020 ranges mismatch')
    fullranges={};domainranges={}
    for scale in (1.,.5):
        full=[r for d in ('real','sim') for r in allrows['train/'+d+'/'+str(scale)] if r['midpoint'] is not None]
        fullranges[scale]=[(min(features(r)[j] for r in full),max(features(r)[j] for r in full)) for j in range(4)]
        ss=allrows['train/sim/'+str(scale)]
        domainranges[scale]=[(min(features(r)[j] for r in ss),max(features(r)[j] for r in ss)) for j in range(4)]
    cache=cache_metadata(args.cache_dir,[r for rs in allrows.values() for r in rs],used) if args.cache_dir else dict(status='NOT_AVAILABLE')
    done=indexed(fr,'completion.json',used)
    if done['status']!='FROZEN_NATIVE_VAL_TEST_DEPTH_COMPLETE' or any(done['weights'][a]['sha256']!=PINS[a] for a in PINS):raise ValueError('Not frozen S023 completion')
    rs=indexed(fr,'test/evaluation_rows.jsonl',used)
    if len(rs)!=1440:raise ValueError('Incomplete TEST')
    for domain in ('real','sim'):
        allrows['test/'+domain+'/1.0']=[dict(r,scale=1.) for r in rs if r['domain']==domain]
    propagation={}
    for name,n in zip(DEPTH,(981,893,1097)):
        saved=indexed(fr,name+'/depth_rows.jsonl',used);preds=indexed(fr,name+'/predictions.jsonl',used);summ=indexed(fr,name+'/summary.json',used)
        if len(saved)!=n or len(preds)!=n or [r['frame_id'] for r in saved]!=[r['frame_id'] for r in preds]:raise ValueError('Incomplete depth identities')
        converted=[]
        for r,p in zip(saved,preds):
            row=dict(image=r['frame_id'],sequence=name,scale=1.,domain='webots',gt=r['gt_box'],
                     **{a:r[a]['box'] for a in ARMS})
            if any(row[a]!=p[a] for a in ARMS):raise ValueError('Depth predicted box drift')
            row['saved_depth']=r;row['support_mean_saved']=p.get('support_mean');check_identity(row);converted.append(row)
        allrows['depth/'+name]=converted
        propagation[name]=dict(role=summ['split'],reused_global_propagation=summ['propagation'],reused_paired=summ['paired'])
    support=[];gains=[];conditional_depth=[]
    for dataset,rs in allrows.items():
        scale=rs[0]['scale'];sr=domainranges[scale];full=fullranges[scale]
        present=[r for r in rs if r['midpoint'] is not None and r['b'] is not None]
        for row in rs:check_identity(row)
        for label in ('all','low/low','low/high','high/low','high/high','missing'):
            sub=rs if label=='all' else [r for r in present if grid(r)==label] if label!='missing' else [r for r in rs if r not in present]
            valid=[r for r in sub if r['midpoint'] is not None and r['b'] is not None]
            s=dict(dataset=dataset,cell=label,frames=len(sub),unique_frames=len({r['image'] for r in sub}),outputs=len(valid),missing=len(sub)-len(valid),
                sim_train_inside=sum(inside(r,sr) for r in valid),full_train_inside=sum(inside(r,full) for r in valid))
            for j,key in enumerate(('log_geomean','log_aspect','M_B_long','M_B_short')):
                vals=[features(r)[j] for r in valid];s.update({key+'_min':min(vals) if vals else None,key+'_max':max(vals) if vals else None,
                    key+'_outside_sim':sum(v<sr[j][0] or v>sr[j][1] for v in vals),key+'_outside_full':sum(v<full[j][0] or v>full[j][1] for v in vals)})
            for j,edge in enumerate(('long','short')):
                for g in ('undersized','oversized','near3'):s[edge+'_'+g]=sum(edge_group(r,j)==g for r in valid)
            meta=[r for r in valid if r.get('roi_meta') and r['roi_meta']['xy']]
            s['actual_resize_rows']=len(meta);s['actual_roi_shape']='256x9x9' if meta else 'MISSING'
            for j,edge in enumerate(('long','short')):
                lengths=[]
                for r in meta:
                    m=canonical(r['midpoint']);sx,sy=r['roi_meta']['xy'];c,t=math.cos(m[4]),math.sin(m[4])
                    factor=math.hypot(c*sx,t*sy) if j==0 else math.hypot(t*sx,c*sy)
                    lengths.append(m[j+2]*factor)
                s['M_'+edge+'_actual_model_px_min']=min(lengths) if lengths else None;s['M_'+edge+'_actual_model_px_max']=max(lengths) if lengths else None
            supports=[r['roi_meta']['support_mean'] for r in meta]
            supports += [r['support_mean_saved'] for r in valid if r.get('support_mean_saved') is not None]
            s['support_rows']=len(supports);s['support_mean']=mean(supports);s['support_min']=min(supports) if supports else None
            support.append(s)
            # All/cell slices plus inside/outside SIM TRAIN marginals. No inference filter.
            for support_group in ('all','inside_sim','outside_sim'):
                slice_rows=sub if support_group=='all' else [r for r in valid if inside(r,sr)==(support_group=='inside_sim')]
                for j,edge in enumerate(('long','short')):
                    for eg in ('all','undersized','oversized','near3'):
                        selected=slice_rows if eg=='all' else [r for r in slice_rows if r['midpoint'] is not None and edge_group(r,j)==eg]
                        for arm in ARMS:
                            out=dict(dataset=dataset,cell=label,support_group=support_group,edge=edge,M_edge_group=eg,arm=arm,**stats(selected,arm))
                            if arm in ('i07','axis15'):out.update(paired(selected,arm,j))
                            gains.append(out)
                if dataset.startswith('depth/'):
                    for arm in ARMS:
                        z=[r['saved_depth'] for r in slice_rows]
                        delta=[abs(r[arm]['depth']['z_m']-r['truth_z_m'])-abs(r['midpoint']['depth']['z_m']-r['truth_z_m']) for r in z if r[arm]['depth']['z_m'] is not None and r['midpoint']['depth']['z_m'] is not None]
                        comp=[r for r in z if r[arm]['depth']['z_m'] is not None and r['gt_obb']['depth']['z_m'] is not None and r[arm]['box'] is not None]
                        # Reuse saved z and q; short-only effect relative to saved GT-formula z.
                        short_terms=[r['gt_obb']['depth']['z_m']*(canonical(r['gt_box'])[3]/canonical(r[arm]['box'])[3]-1) for r in comp]
                        q_delta=[r[arm]['depth']['q_signed']**2-r['gt_obb']['depth']['q_signed']**2 for r in comp]
                        conditional_depth.append(dict(dataset=dataset,cell=label,support_group=support_group,arm=arm,frames=len(z),common=len(delta),
                            short_scale_signed_component_mean_m=mean(short_terms),q_squared_minus_GT_mean=mean(q_delta),
                            saved_depth_abs_delta_mean_m=mean(delta),depth_improved=sum(x < -EPS for x in delta),depth_worsened=sum(x>EPS for x in delta)))
    result=dict(protocol=VERSION,status='SAVED_TRANSFER_AUDIT_COMPLETE',utc=datetime.now(timezone.utc).isoformat(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),script_sha256=sha(__file__),
        executable=__import__('sys').executable,strata=dict(cuts=CUTS,sim_ranges_by_scale=domainranges,full_ranges_by_scale=fullranges,fit='S020 standard Sim TRAIN; exact replay'),
        dataset_counts={k:len(v) for k,v in allrows.items()},cache=cache,inputs=used,propagation_reused=propagation,annotation_local_review=ANNOTATION_REVIEW,
        limits=dict(training_updates=0,model_loads=0,inference_calls=0,feature_extractions=0,depth_formula_calls=0,epoch_reselection=False,
            causal_root_determined=False,actual_webots_resize_roi_metadata='MISSING; support_mean only; original geometry is not feature resolution',
            support='Four marginal ranges and coarse cells only; no joint/visual support proof',two_train_scales='Paired views, reported separately, never independent samples',
            depth='Three historical sequences for this explicitly requested conditional explanation only. Future fresh depth evaluation: Unknown-02 only.'))
    args.out.mkdir(parents=True)
    write_csv(args.out/'annotation_contract.csv',annotation_contract())
    write_csv(args.out/'train_support.csv',support);write_csv(args.out/'correction_gains.csv',gains);write_csv(args.out/'depth_conditional_terms.csv',conditional_depth)
    (args.out/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (args.out/'artifacts.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in args.out.iterdir() if p.is_file()}),indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],counts=result['dataset_counts'],cache=cache,outputs=str(args.out)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('i-run','axis-run','frozen-run','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cache-dir',type=Path)
    run(p.parse_args())
