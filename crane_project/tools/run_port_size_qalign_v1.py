#!/usr/bin/env python3
"""S038 paired fixed-B / Q-aligned cached 2D reads. Engineering gates before fit."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools.run_port_size_native_train_v1 import (
    sha,read,write,rows_write,seed,rng_get,rng_set,equal,near_joint,eligibility,ranking)
VERSION='port_size_qalign_v1'


def protocol():
    from crane_project.tools.run_port_size_local_vectors_v1 import protocol as parent
    p=parent()
    p.update(protocol=VERSION, hypothesis='Only visual 2D area reading coordinates change: fixed B versus B-to-Q warp within same cached B-P3. Not new source information.',
        representation='Same V2800 network/reference vectors/logical positions/decoder; both use W=min(S_A,S_Q). Only visual samples differ.',
        support='Outside old B cache zero, no extrapolation. Common logical-slot support, NOT physical intersection. Natural TRAIN zero W => stop, no filtering.',
        loss='Both arms ONLY final size SmoothL1 beta0.1 mean2N; no geometry/local/proposal/weight loss.',
        geometry_value='Q vs A selected-or-descriptive: Sim both lower>1e-6,Real<=A+0.001,overallRIoU>=A-1e-5; four fixed E_j intersection harm / fixed |E_j| <= A+1e-6. No posthoc reselection.',
        engineering='TRAIN-only full5116 support/neutral audit before any VAL read; two fixed8view batches, discard updates, Adam/RNG reload exact. 20warmup/100alternating paired GPU timings, read+head batch1, mean delta<=5ms,allocator delta<=128MiB.',
        resource='One physical GPU2or3; own process<7000MiB,stop6800,allocator6000.',
        papers=['https://github.com/csuhan/s2anet/blob/e463c2bb17e161a6909e3d83410b49697dce770a/mmdet/models/anchor_heads_rotated/s2anet_head.py',
                'https://github.com/YiqunChen1999/RefineBox/blob/0a9feb76857adb879225d536a92797d3f884522c/refinebox/models/layers/res_refiner.py'])
    p['fit']['arms']=['fixed_b','q_aligned']
    p['architecture']='Same V267->8 1x1,8->8 3x3,8->8 1x1 zero terminal,2800params; source grid A=u,Q=g H_Q T_B; both same u/M anchor and common W.'
    p['correspondence']='Fixed original B/Q raw axes, model B canonical source frame only; original points transformed before actual anisotropic resize; no GT input.'
    p['labels']='GT canonical long/short assigned once to M raw/modelB axes. No projected GT extent.'
    return p


def checked_sources():
    p=ROOT/'crane_project/tools/port_size_qalign_v1_sources.json';v=read(p)
    if read(ROOT/'crane_project/tools/port_size_qalign_v1_protocol.json')!=protocol():raise ValueError('Protocol drift')
    for n,s in v['sources'].items():
        if sha(ROOT/n)!=s:raise ValueError('Source drift '+n)
    return dict(manifest_sha256=sha(p),**v)


def batch(rows,torch):
    keys=('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original','gt_original')
    return [torch.cat([r[k] for r in rows]).cuda() for k in keys]


def update(h,opt,x,arm,torch):
    from crane_project.utils import port_size_qalign_v1 as model
    h.train();opt.zero_grad();o=h(*x[:-1]);size=model.size_loss(o,x[-1],x[-2],x[3]);loss=size;geo=size.detach()*0
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in h.parameters()):raise ValueError('Invalid gradient')
    stem=math.sqrt(sum(float(p.grad.double().square().sum()) for p in h.stem.parameters()))
    terminal=math.sqrt(sum(float(p.grad.double().square().sum()) for p in h.terminal.parameters()))
    norm=float(torch.nn.utils.clip_grad_norm_(h.parameters(),10.));opt.step()
    if any(not bool(torch.isfinite(p).all()) for p in h.parameters()):raise ValueError('Invalid parameters')
    return dict(loss=float(loss.detach()),size_loss=float(size.detach()),geometry_loss=float(geo.detach()),stem_grad_norm=stem,terminal_grad_norm=terminal,grad_norm=norm)


def evaluate(h,rows,torch):
    from crane_project.utils import port_size_qalign_v1 as model
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity
    h.eval();pred={}
    with torch.no_grad():
        pp=[r for r in rows if len(r['midpoint_original'])]
        for start in range(0,len(pp),8):
            rr=pp[start:start+8];x=batch(rr,torch);o=h(*x[:-1])
            t=model.residual.size_targets(x[-1],x[-2],x[3])['log_residual']
            loss=torch.nn.functional.smooth_l1_loss(o['delta_roi'],t,beta=.1,reduction='none')
            swap=(x[3][:,2]<x[3][:,3]) ^ (x[-2][:,2]<x[-2][:,3]);loss=torch.where(swap[:,None],loss.flip(-1),loss)
            for j,r in enumerate(rr):
                pred[r['image'],r['scale']]=dict(edge_residual=o['boxes_original'][j].cpu().tolist(),delta_roi=o['delta_roi'][j].cpu().tolist(),
                    loss_edges_canonical=loss[j].cpu().tolist(),vector_valid=bool(o['vector_valid'][j]),local_size_variance=o['local_size_variance'][j].cpu().tolist(),
                    size_delivery='neutral' if bool(o['neutral_bypass'][j]) else 'square' if bool(o['square_bypass'][j]) else 'accepted' if bool(o['accepted'][j]) else 'fallback',
                    failed_checks=[k for k,v in o['checks'].items() if not bool(v[j])])
    out=[]
    for r in rows:
        v={k:r[k] for k in ('image','sequence','domain','frame_id','scale','split','shard')}
        v.update(gt=r['gt_original'][0].tolist(),b=r['boxes_original'][0].tolist() if len(r['boxes_original']) else None,
                 midpoint=r['midpoint_original'][0].tolist() if len(r['midpoint_original']) else None)
        v.update(pred.get((r['image'],r['scale']),dict(edge_residual=None,delta_roi=None,loss_edges_canonical=None,size_delivery='no_output',failed_checks=[])))
        if not frozen_identity(v):raise ValueError('M identity changed')
        out.append(v)
    return out


def engineering(rows,out,torch):
    from crane_project.tools.engineer_port_size_qalign_v1 import engineering as check
    return check(rows,out,torch)


def load_records(directory,cache,frozen,device,torch,model,references,labels):
    from collections import Counter
    from crane_project.tools import run_port_geometry_size_distribution_v1 as prior
    sealed=prior.sealed;result=[];byte_count=0
    for label in labels:
        path=directory/(label+'.pt')
        if sha(path)!=cache['files'][path.name]:raise ValueError('Immutable shard SHA '+label)
        payload=sealed.torch_load(torch,path);expected,role,_=sealed.SHARDS[label]
        records=payload['records']
        counts=sealed.TRAIN_COUNTS if role=='train' else sealed.VAL_COUNTS
        if payload['identity']!=cache['identity'] or payload['detector_state']!=cache['detector_state'] or len(records)!=expected or len({r['image'] for r in records})!=expected or Counter(r['sequence'] for r in records)!=Counter(counts):raise ValueError('Shard identity/count')
        with torch.no_grad():
            for r in records:
                byte_count+=sealed.checked_record(r,label,torch,model.residual.original)
                inputs=[r[k].to(device) for k in sealed.TENSOR_KEYS[:-1]]
                o=frozen(*inputs);m=o['boxes_original'].detach().cpu()
                rr=dict(r,shard=label,midpoint_original=m)
                if role=='val':
                    source=dict(image=r['image'],sequence=r['sequence'],domain=r['domain'],frame_id=r['frame_id'],scale=r['scale'],
                        gt=r['gt_original'][0].tolist(),b=r['boxes_original'][0].tolist() if len(m) else None,midpoint=m[0].tolist() if len(m) else None,
                        diagnosis=dict(baseline_joint10=False,baseline_center_hit=False))
                    if len(m):
                        source['diagnosis']=dict(baseline_joint10=max(abs(x/y-1) for x,y in zip(sorted(source['midpoint'][2:4]),sorted(source['gt'][2:4])))<=.1,
                            baseline_center_hit=math.hypot(m[0,0]-r['gt_original'][0,0],m[0,1]-r['gt_original'][0,1])<15.)
                    sealed.compare_reference(source,references[r['image']])
                    if (bool(o['accepted'][0]) if len(m) else None)!=references[r['image']]['accepted']:raise ValueError('M accepted replay')
                result.append(rr)
        print('loaded/replayed',label,len(records),flush=True)
    return sorted(result,key=lambda r:(r['scale'],r['image'])),byte_count


def run(args,out,torch,proof):
    from crane_project.tools import run_port_geometry_size_distribution_v1 as prior
    from crane_project.utils import port_size_qalign_v1 as model
    from crane_project.utils import port_geometry_midpoint_sigma_v1 as frozen_model
    from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
    from crane_project.tools.run_port_geometry_size_condition_v1 import components
    sealed=prior.sealed;historical,differences=prior.historical_identity()
    if sha(args.b_weight)!=prior.B_SHA:raise ValueError('B actual SHA')
    cache=sealed.checked_cache(args.cache_dir,historical)
    selection,references,indexed=sealed.checked_selection(args.head_dir,historical)
    frozen=sealed.load_head(torch,frozen_model,args.head_dir,selection,cache,'cuda:0')
    frozen_digest=sealed.state_digest(frozen)
    train,bytes_read=load_records(args.cache_dir,cache,frozen,'cuda:0',torch,model,references,('train_s1','train_s05'))
    if len(train)!=5116 or sum(bool(len(r['midpoint_original'])) for r in train)!=5112:raise ValueError('TRAIN counts')
    write(out/'engineering_input_identity.json',dict(b_sha256=sha(args.b_weight),m_sha256=sha(args.head_dir/sealed.FIXED['path']),cache_manifest_sha256=sha(args.cache_dir/'cache_manifest.json'),shards=cache['files'],historical=historical,unused_detector_differences=differences,val_shards_loaded=0))
    engineering(train,out,torch)
    write(out/'ENGINEERING_PASSED.json',dict(status='PASS',formal_updates=0,val_shards_loaded=0))
    if args.engineering_only:return
    val,val_bytes=load_records(args.cache_dir,cache,frozen,'cuda:0',torch,model,references,('val_s1',));bytes_read+=val_bytes
    if (len(train),sum(bool(len(r['midpoint_original'])) for r in train),len(val),sum(bool(len(r['midpoint_original'])) for r in val))!=(5116,5112,887,886):raise ValueError('Full view counts')
    if any(float(r['support'].sum())<=0 for r in train+val if len(r['midpoint_original'])):raise ValueError('Natural output has no support; stop without GT filtering')
    inputs=dict(b_sha256=sha(args.b_weight),m_sha256=sha(args.head_dir/sealed.FIXED['path']),cache_manifest_sha256=sha(args.cache_dir/'cache_manifest.json'),shards=cache['files'],reference_artifacts=indexed,
                historical_identity=historical,unused_detector_historical_differences=differences,cpu_input_tensor_bytes=bytes_read)
    write(out/'input_identity.json',inputs)
    report=dict(protocol=VERSION,status='RUNNING',source_proof=proof,source_commit=args.source_commit,inputs=inputs,arms={},formal_head_updates=0,test_reads=0,depth_reads=0,feature_extractions=0)
    pp=[r for r in train if len(r['midpoint_original'])];riou_cache={}
    for arm in model.ARMS:
        dest=out/arm;dest.mkdir();seed(torch);h=model.AlignedSizeHead(arm).cuda();opt=torch.optim.Adam(h.parameters(),lr=.001)
        initial=sealed.state_digest(h);events=[]
        zv=evaluate(h,val,torch);zt=evaluate(h,train,torch)
        if any(r['edge_residual']!=r['midpoint'] for r in zv+zt):raise ValueError('Full neutral replay')
        if arm==model.ARMS[0]:
            rows_write(out/'baseline_val.rows.jsonl.gz',zv);rows_write(out/'baseline_train.rows.jsonl.gz',zt)
            bs=metrics.summary(zv,riou_cache);write(out/'baseline_val.summary.json',dict(metrics=bs,components=components(zv),diagnostics=near_joint(zv)))
            for d,count,correct in [('real',374,199),('sim',512,464)]:
                if bs[d]['midpoint']['sizes']['outputs']!=count or bs[d]['midpoint']['sizes']['joint10']!=correct:raise ValueError('M reference counts')
            for d,expected in [('real',(.082201951,.084057508)),('sim',(.04321642,.04345043))]:
                if any(abs(bs[d]['midpoint']['sizes']['edges'][e]['mae']-v)>1e-6 for e,v in zip(('long','short'),expected)):raise ValueError('M reference MAE')
        del zv,zt
        for epoch in range(1,25):
            began=time.monotonic();ids=list(range(len(pp)));random.Random(1703+epoch).shuffle(ids)
            loss_sums={k:0. for k in ('loss','size_loss','geometry_loss')};count=0
            for start in range(0,len(ids),8):
                rr=[pp[j] for j in ids[start:start+8]];x=batch(rr,torch);step=update(h,opt,x,arm,torch)
                for k in loss_sums:loss_sums[k]+=step[k]*len(rr)
                count+=len(rr);report['formal_head_updates']+=1
                if start%1600==0:print(json.dumps(dict(stage='train',arm=arm,epoch=epoch,view=start,**step)),flush=True)
            vr=evaluate(h,val,torch);s=metrics.summary(vr,riou_cache);diag=near_joint(vr);comp=components(vr);gate=eligibility(s,diag);cross=model.cross_edge(vr)
            tr=evaluate(h,train,torch);td=dict(size_groups=metrics.train_report(tr),joint=near_joint(tr),cross_edge_by_scale={str(scale):model.cross_edge([r for r in tr if r['scale']==scale]) for scale in (.5,1.)})
            saved_rng=rng_get(torch);payload=dict(protocol=VERSION,arm=arm,epoch=epoch,updates=epoch*639,source_commit=args.source_commit,input_identity=inputs,
                head_state={k:v.detach().cpu().clone() for k,v in h.state_dict().items()},optimizer_state=deepcopy(opt.state_dict()),rng=saved_rng)
            path=dest/('head_epoch_%02d.pth'%epoch)
            with path.open('xb') as f:torch.save(payload,f)
            if not equal(payload,torch.load(str(path),map_location='cpu'),torch):raise ValueError('Checkpoint serialization')
            rng_set(saved_rng,torch)
            ep=dict(epoch=epoch,selection_key=ranking(s,epoch),eligibility=gate,metrics=s,components=comp,diagnostics=diag,
                cross_edge=cross,training_losses={k:v/count for k,v in loss_sums.items()},schedule_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                checkpoint_sha256=sha(path),epoch_seconds=time.monotonic()-began)
            rows_write(dest/('epoch_%02d_val.rows.jsonl.gz'%epoch),vr);write(dest/('epoch_%02d_val.json'%epoch),ep)
            rows_write(dest/('epoch_%02d_train.rows.jsonl.gz'%epoch),tr);write(dest/('epoch_%02d_train.json'%epoch),td)
            events.append(ep)
            print(json.dumps(dict(stage='epoch_finished',arm=arm,epoch=epoch,eligible=gate['pass_all'],failed=gate['failed'],selection_key=ep['selection_key'],elapsed_s=ep['epoch_seconds'])),flush=True)
            del x,vr,tr,td
        eligible=[e for e in events if e['eligibility']['pass_all']]
        result=dict(initial_state_digest=initial,representative=min(events,key=lambda e:e['selection_key'])['epoch'],selected=min(eligible,key=lambda e:e['selection_key'])['epoch'] if eligible else None,eligible_epochs=[e['epoch'] for e in eligible],epochs=events)
        report['arms'][arm]=result;write(dest/'selection.json',result);del h,opt;torch.cuda.empty_cache()
    a,b=[report['arms'][k] for k in model.ARMS]
    if report['formal_head_updates']!=30672 or a['initial_state_digest']!=b['initial_state_digest'] or [e['schedule_sha256'] for e in a['epochs']]!=[e['schedule_sha256'] for e in b['epochs']]:raise ValueError('Paired budget/init/order')
    if frozen_digest!=sealed.state_digest(frozen):raise ValueError('Frozen M changed')
    if sha(args.b_weight)!=prior.B_SHA or sha(args.head_dir/sealed.FIXED['path'])!=sealed.FIXED['sha256']:raise ValueError('Final B/M bytes changed')
    if sha(args.cache_dir/'cache_manifest.json')!=cache['manifest_sha256'] or any(sha(args.cache_dir/n)!=h for n,h in cache['files'].items()):raise ValueError('Final cache bytes changed')
    report['status']='COMPLETE';write(out/'training.json',report)


def main(args):
    out=args.out_dir.resolve()
    if out.parent!=(ROOT/'work_dirs'/VERSION).resolve() or out.exists():raise ValueError('Fresh result-only output required')
    out.mkdir(parents=True);stop=None;report=dict(protocol=VERSION,status='FAILED',started_utc=datetime.now(timezone.utc).isoformat())
    try:
        proof=checked_sources();write(out/'source_identity.json',proof);write(out/'protocol.json',protocol())
        from crane_project.tools.audit_port_size_native_p3_v1 import setup_gpu
        resource,stop,thread=setup_gpu(out,7000,6800,6000);report['resource']=resource
        import torch
        torch.set_num_threads(1);torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(6000*2**20/torch.cuda.get_device_properties(0).total_memory,0);seed(torch)
        report['runtime']=dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
        run(args,out,torch,proof)
        if checked_sources()!=proof:raise ValueError('End source drift')
        report.update(status='COMPLETE',allocator_peak_mib=torch.cuda.max_memory_allocated()/2**20,reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20)
    except Exception as e:
        report['error']=type(e).__name__+': '+str(e);raise
    finally:
        if stop is not None:stop.set();thread.join(timeout=11)
        report['ended_utc']=datetime.now(timezone.utc).isoformat();write(out/'finished.json',report)
        write(out/'artifacts.json',dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--cache-dir',type=Path,default=ROOT/'work_dirs/port_results/geometry/port_geometry_midpoint_formal_v1_roi_cache')
    p.add_argument('--head-dir',type=Path,default=ROOT/'work_dirs/crane_symeood_k1_port_day2night_midpoint_sigma_v1/sigma_1p5')
    p.add_argument('--engineering-only',action='store_true')
    p.add_argument('--b-weight',type=Path,default=ROOT/'work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth')
    main(p.parse_args())
