#!/usr/bin/env python3
"""S034 two fresh local-vector arms. Frozen ROI/M, TRAIN+VAL only, one GPU."""
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
VERSION='port_size_local_vectors_v1'


def protocol():
    return dict(protocol=VERSION,status='FROZEN_BEFORE_FIT',
        hypothesis='Local 2D supervision of original GT rectangle side midpoints adds direction evidence beyond final size supervision; no guarantee directions become correct.',
        representation='81 existing fixed ROI points, 4x2 normalized continuous vectors per point, M-vector residual reference; no new features/points/multiscale.',
        correspondence='OFFLINE per-position minimum of mean8 SmoothL1 over8 D4 cyclic/reverse permutations preserving opposite pairs; deterministic first tie, no GT matching in forward.',
        decode='Inverse actual anisotropic resize/ROI affine before opposite-vector Euclidean norms; sort local long/short, support-weighted arithmetic mean; reference ratio makes zero residual exactly M.',
        support='Cached geometric support only, no GT overlap or quality head. Supported nonfinite or length<=1e-6 => whole-pair M fallback. Zero support fallback, training invalidity stops; never GT-filter.',
        labels='Original annotated OBB midpoints, not visible physical keypoints; labels outside ROI unbounded; local directions can pollute recovered lengths.',
        architecture='267->8 pointwise conv,8->8 3x3,8->8 terminal1x1;2800 parameters. Both arms same complete forward and geometry inputs, zero terminal.',
        loss='Control: mean(2N) SmoothL1 beta0.1 log-size residual. Candidate: same +0.1*meanN support-normalized minD4 mean8 SmoothL1 beta0.1 vector loss. No batch weight-sum rescaling beyond per-view support.',
        data='Full TRAIN2558 image identities x2 scales=5116 views,5112 outputs; full natural VAL887 each epoch,886 outputs. No TRAIN admission. Artificial proposals not used.',
        fit=dict(arms=['size_only','size_plus_geometry'],epochs_per_arm=24,batch_size=8,updates_per_epoch=639,seed=1703,optimizer='Adam',lr=.001,weight_decay=0.,clip_norm=10.,shuffle='Random(1703+epoch)',fresh_initialization=True),
        selection='Eligible first, lexicographic Sim longMAE,Sim shortMAE,Real longMAE,Real shortMAE,epoch. None eligible => same-key descriptive representative, not adopted.',
        adoption='Sim eachMAE<M-1e-6; Real each<=M+0.001; Sim meanabs(a/r)<=M+1e-6; four fixedM near<=3% edgeMAEs<=M+0.001; overall fullRIoU>=M-1e-5; eachdomain>=M-0.001. Identity/coverage exact; domainDFR/MCMLmax/mean nonincrease; TDRhitsnondecrease samewindows; ACI equal. Relative tolerances0.001 permit0.1pp cost, not strict protection.',
        geometry_value='Candidate vs common selected-or-descriptive control: bothSimMAEs lower by>1e-6,Real each<=control+0.001,overallRIoU>=control-1e-5; also report full adoption separately. Shared structure gains not attributable to added supervision; equal parameter count does not eliminate constraint/capacity differences.',
        diagnostics='MAE/P95/bias/a/r;near,bothnear,wrong->right/right->wrong,joint gain/tradeoff;wrong-direction harm and useful correction retention explanatory only;fullcustoms/AP11supplement/threegeometries/coverage;overall/domain/video,TRAINscalesseparate.',
        stop='Invalid coordinate/SHA/support/nonfinite/engineering stops;24epochs/arm only. No new lambda/grid/loss/module/epoch search. No TEST/depth or automatic M promotion.',
        resource='One physical GPU2or3 only; own process<7000MiB,stop6800MiB,allocator6000MiB.',
        papers=['https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_D2Det_Towards_High_Quality_Object_Detection_and_Instance_Segmentation_CVPR_2020_paper.pdf','https://arxiv.org/pdf/2008.07043'])


def checked_sources():
    p=ROOT/'crane_project/tools/port_size_local_vectors_v1_sources.json';v=read(p)
    if read(ROOT/'crane_project/tools/port_size_local_vectors_v1_protocol.json')!=protocol():raise ValueError('Protocol drift')
    for n,s in v['sources'].items():
        if sha(ROOT/n)!=s:raise ValueError('Source drift '+n)
    return dict(manifest_sha256=sha(p),**v)


def batch(rows,torch):
    keys=('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original','gt_original')
    return [torch.cat([r[k] for r in rows]).cuda() for k in keys]


def update(h,opt,x,arm,torch):
    from crane_project.utils import port_size_local_vectors_v1 as model
    h.train();opt.zero_grad();o=h(*x[:-1]);loss,size,geo=model.losses(o,x[-1],x[1],*x[2:6],arm)
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in h.parameters()):raise ValueError('Invalid gradient')
    stem=math.sqrt(sum(float(p.grad.double().square().sum()) for p in h.stem.parameters()))
    terminal=math.sqrt(sum(float(p.grad.double().square().sum()) for p in h.terminal.parameters()))
    norm=float(torch.nn.utils.clip_grad_norm_(h.parameters(),10.));opt.step()
    if any(not bool(torch.isfinite(p).all()) for p in h.parameters()):raise ValueError('Invalid parameters')
    return dict(loss=float(loss.detach()),size_loss=float(size.detach()),geometry_loss=float(geo.detach()),stem_grad_norm=stem,terminal_grad_norm=terminal,grad_norm=norm)


def evaluate(h,rows,torch):
    from crane_project.utils import port_size_local_vectors_v1 as model
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
    from crane_project.utils import port_size_local_vectors_v1 as model
    pp=[r for r in rows if len(r['midpoint_original'])];results={}
    for arm in model.ARMS:
        seed(torch);h=model.LocalVectorSizeHead().cuda();opt=torch.optim.Adam(h.parameters(),lr=.001)
        x=batch(pp[:8],torch)
        with torch.no_grad():
            neutral=h(*x[:-1]);assert torch.equal(neutral['boxes_original'],x[-2])
            original=model.encode(x[-1],*x[2:5]);decoded=model.decode(original,x[1],*x[2:5])
            gt=x[-1][:,2:4].sort(descending=True)[0]
            assert torch.allclose(decoded['sizes'],gt,atol=1e-4,rtol=2e-6)
        first=update(h,opt,x,arm,torch);assert first['stem_grad_norm']==0 and first['terminal_grad_norm']>0
        snapshot=dict(head=deepcopy(h.state_dict()),opt=deepcopy(opt.state_dict()),rng=rng_get(torch))
        stream=io.BytesIO();torch.save(snapshot,stream);stream.seek(0);got=torch.load(stream)
        clone=model.LocalVectorSizeHead().cuda();clone.load_state_dict(got['head']);op=torch.optim.Adam(clone.parameters(),lr=.001);op.load_state_dict(got['opt'])
        y=batch(pp[8:16],torch);rng_set(snapshot['rng'],torch);second=update(h,opt,y,arm,torch);state=rng_get(torch)
        rng_set(got['rng'],torch);other=update(clone,op,y,arm,torch)
        assert second['stem_grad_norm']>0 and equal(second,other,torch) and equal(h.state_dict(),clone.state_dict(),torch) and equal(opt.state_dict(),op.state_dict(),torch) and equal(state,rng_get(torch),torch)
        with torch.no_grad():assert torch.equal(h(*y[:-1])['vectors'],clone(*y[:-1])['vectors'])
        results[arm]=dict(first=first,second=second,adam_rng_reload_and_next_update_exact=True,discarded=True)
    # Same forward in both arms; cached ROI transfer is excluded explicitly.
    z=batch(pp[:1],torch);times=[]
    with torch.no_grad():
        for _ in range(20):h(*z[:-1])
        for _ in range(100):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();h(*z[:-1]);b.record();torch.cuda.synchronize();times.append(a.elapsed_time(b))
    results['efficiency']=dict(scope='Shared head only, batch1 cached tensors already onGPU;20warmups100repeats, shared GPU; excludes B/M/ROI transfers. Not deployment throughput.',head_ms=times,mean_head_ms=sum(times)/len(times),parameter_count=model.PARAMETER_COUNT)
    del h,opt,clone,op,x,y,z;torch.cuda.empty_cache();seed(torch);write(out/'engineering.json',results)


def run(args,out,torch,proof):
    from crane_project.tools import run_port_geometry_size_distribution_v1 as prior
    from crane_project.utils import port_size_local_vectors_v1 as model
    from crane_project.utils import port_geometry_midpoint_sigma_v1 as frozen_model
    from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
    from crane_project.tools.run_port_geometry_size_condition_v1 import components
    sealed=prior.sealed;historical,differences=prior.historical_identity()
    if sha(args.b_weight)!=prior.B_SHA:raise ValueError('B actual SHA')
    cache=sealed.checked_cache(args.cache_dir,historical)
    selection,references,indexed=sealed.checked_selection(args.head_dir,historical)
    frozen=sealed.load_head(torch,frozen_model,args.head_dir,selection,cache,'cuda:0')
    frozen_digest=sealed.state_digest(frozen)
    train,val,bytes_read=prior.load_records(args.cache_dir,cache,frozen,'cuda:0',torch,model,references)
    if (len(train),sum(bool(len(r['midpoint_original'])) for r in train),len(val),sum(bool(len(r['midpoint_original'])) for r in val))!=(5116,5112,887,886):raise ValueError('Full view counts')
    if any(float(r['support'].sum())<=0 for r in train+val if len(r['midpoint_original'])):raise ValueError('Natural output has no support; stop without GT filtering')
    inputs=dict(b_sha256=sha(args.b_weight),m_sha256=sha(args.head_dir/sealed.FIXED['path']),cache_manifest_sha256=sha(args.cache_dir/'cache_manifest.json'),shards=cache['files'],reference_artifacts=indexed,
                historical_identity=historical,unused_detector_historical_differences=differences,cpu_input_tensor_bytes=bytes_read)
    write(out/'input_identity.json',inputs);engineering(train,out,torch)
    report=dict(protocol=VERSION,status='RUNNING',source_proof=proof,source_commit=args.source_commit,inputs=inputs,arms={},formal_head_updates=0,test_reads=0,depth_reads=0,feature_extractions=0)
    pp=[r for r in train if len(r['midpoint_original'])];riou_cache={}
    for arm in model.ARMS:
        dest=out/arm;dest.mkdir();seed(torch);h=model.LocalVectorSizeHead().cuda();opt=torch.optim.Adam(h.parameters(),lr=.001)
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
            vr=evaluate(h,val,torch);s=metrics.summary(vr,riou_cache);diag=near_joint(vr);comp=components(vr);gate=eligibility(s,diag)
            tr=evaluate(h,train,torch);td=dict(size_groups=metrics.train_report(tr),joint=near_joint(tr))
            saved_rng=rng_get(torch);payload=dict(protocol=VERSION,arm=arm,epoch=epoch,updates=epoch*639,source_commit=args.source_commit,input_identity=inputs,
                head_state={k:v.detach().cpu().clone() for k,v in h.state_dict().items()},optimizer_state=deepcopy(opt.state_dict()),rng=saved_rng)
            path=dest/('head_epoch_%02d.pth'%epoch)
            with path.open('xb') as f:torch.save(payload,f)
            if not equal(payload,torch.load(str(path),map_location='cpu'),torch):raise ValueError('Checkpoint serialization')
            rng_set(saved_rng,torch)
            ep=dict(epoch=epoch,selection_key=ranking(s,epoch),eligibility=gate,metrics=s,components=comp,diagnostics=diag,
                training_losses={k:v/count for k,v in loss_sums.items()},schedule_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
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
    p.add_argument('--b-weight',type=Path,default=ROOT/'work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth')
    main(p.parse_args())
