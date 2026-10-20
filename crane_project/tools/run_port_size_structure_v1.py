#!/usr/bin/env python3
"""Frozen S045 contract: engineering then fresh V/D24epochs, all48 full VAL."""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools.run_port_size_native_train_v1 import sha,read,write,rows_write,seed,rng_get,rng_set,equal,near_joint,eligibility,ranking
from crane_project.tools import run_port_size_qalign_v1 as parent
from crane_project.utils import port_size_structure_v1 as model
VERSION='port_size_structure_v1'


def protocol():
    p=parent.protocol()
    p.update(protocol=VERSION,hypothesis='Same frozen visual/conditional input267 and common stem: local-vector representation+geometric decode vs direct canonical-log residual regression; not isolate one vector operator.',
        representation='V_structure: sealed fixed-B vector head2800params. D_direct: identical stem +8to2 terminal,2746params; per-slot canonical long/short log residual weighted by identical commonW. Same M8reference channels.',
        architecture='267to8 1x1+ReLU,8to8 3x3+ReLU,8to8(V)/8to2(D)1x1; parameter gap54=1.93% of V; same stem near-parameter control, not equal effective capacity.',
        decode='V unchanged inverse-anisotropic-affine/opposite norms/local sort/weighted arithmetic sizes/neutral reference normalization. D canonical log weighted mean then canonical-to-Mraw-to-modelBroi; same deliver_sizes.',
        support='Both same fixed-B samples and W=min(S_A,S_Q) from sealed read_pair; same logical coords and M reference vectors. Q-grid only contributes existing W, not new visual feature. Natural zero W stops; no GT filtering.',
        loss='Only final delivered-interface PRE-GUARD delta_roi SmoothL1 beta0.1 mean2N, no ratio/local/geometry/protection/weights/proposals. Invalid numerical representation stops, guard rejection alone is recorded not loss-filtered.',
        geometry_value='Separate adoption and structure evidence. Selected-or-descriptive V vs D: Sim both MAE<D-1e-6; Real each<=D+0.001; four Mnear<=D+0.001; Sim a/r<=D+1e-6; overallRIoU>=D-1e-5,domain>=D-0.001; DFR not increased. All24 same-epoch deltas reported with no posthoc persistence threshold or seed significance claim.',
        engineering='Before VAL load: full5116 TRAIN views, identical all267 stem inputs, M neutral/missing exact, W positive, source SHA; fixed first16output views for 2steps+reload next update per arm; discard all6updates incl clones. 20warmup/100randomized pairs read+head batch1 costs descriptive, own limits hard.',
        resource='Only physical GPU3 (last card); singleGPU own<=7000MiB,stop6800,allocator6000,free>=3072.',
        papers=[])
    p['fit'].update(arms=list(model.ARMS),adam_betas=[.9,.999],adam_eps=1e-8,amsgrad=False,lr_schedule='constant',drop_last=False,shuffle='Random(1703+epoch)',common_stem_exact=True,terminal_weight_bias_zero=True,load_historical_checkpoint=False)
    p['parameter_counts']=model.PARAMETERS
    p['adoption_checks']=dict(reference='M each epoch, same for both arms',
        sim_each_mae_strict_improvement=1e-6,real_each_mae_allowed_increase=.001,
        M_near_threshold=.03,near_each_domain_edge_mae_allowed_increase=.001,
        domain_full_RIoU_allowed_decrease=.001,overall_full_RIoU_allowed_decrease=1e-5,
        sim_a_r_allowed_increase=1e-6,output_identity_exact=True,DFR_MCML_nonincrease=True,
        TDR_hits_nondecrease_same_windows=True,ACI_and_adjacent_pairs_equal=True,
        other_diagnostics_not_hard_gates=['AP75','P95','bias','near_damage_count','positive_harm','cross_edge_harm'])
    p['limits']=dict(test_reads=0,depth_reads=0,feature_extractions=0,automatic_adoption=False,epoch_search_beyond24=False)
    return p


def checked_sources():
    path=ROOT/'crane_project/tools/port_size_structure_v1_sources.json';proof=read(path)
    if proof['protocol']!=VERSION or read(ROOT/'crane_project/tools/port_size_structure_v1_protocol.json')!=protocol():raise ValueError('Frozen protocol drift')
    for n,h in proof['sources'].items():
        if sha(ROOT/n)!=h:raise ValueError('Source drift '+n)
    return dict(manifest_sha256=sha(path),**proof)


def optimizer(h,torch):
    return torch.optim.Adam(h.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False)


def state_digest(state):
    # Match sealed state identity without requiring a complete matching head.
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fresh(arm,torch):
    seed(torch);return model.head(arm).cuda()


def batch(rows,torch):return parent.batch(rows,torch)


def update(h,opt,x,arm,torch):
    h.train();opt.zero_grad();o=h(*x[:-1]);loss=model.size_loss(o,x[-1],x[-2],x[3],arm)
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in h.parameters()):raise ValueError('Invalid gradient')
    norms={k:math.sqrt(sum(float(p.grad.double().square().sum()) for p in layer.parameters())) for k,layer in [('stem',h.stem),('terminal',h.terminal)]}
    norm=float(torch.nn.utils.clip_grad_norm_(h.parameters(),10.));opt.step()
    if any(not bool(torch.isfinite(p).all()) for p in h.parameters()):raise ValueError('Nonfinite parameter')
    return dict(loss=float(loss.detach()),stem_grad_norm=norms['stem'],terminal_grad_norm=norms['terminal'],grad_norm=norm)


def evaluate(h,records,arm,torch):
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity
    h.eval();pred={}
    with torch.no_grad():
        pp=[r for r in records if len(r['midpoint_original'])]
        for start in range(0,len(pp),8):
            rr=pp[start:start+8];x=batch(rr,torch);o=h(*x[:-1]);target=model.residual.size_targets(x[-1],x[-2],x[3])['log_residual']
            loss=torch.nn.functional.smooth_l1_loss(o['delta_roi'],target,beta=.1,reduction='none')
            swap=(x[3][:,2]<x[3][:,3]) ^ (x[-2][:,2]<x[-2][:,3]);loss=torch.where(swap[:,None],loss.flip(-1),loss)
            for j,r in enumerate(rr):
                ok=bool(model.valid(o,arm)[j]);cv=bool(o['candidate_valid'][j]);dv=o['delta_roi'][j]
                cell=dict(edge_residual=o['boxes_original'][j].cpu().tolist(),pre_guard=o['candidate_original'][j].cpu().tolist() if cv else None,
                    delta_roi=dv.cpu().tolist() if bool(torch.isfinite(dv).all()) else None,
                    loss_edges_canonical=loss[j].cpu().tolist() if bool(torch.isfinite(loss[j]).all()) else None,
                    representation_valid=ok,candidate_valid=cv,
                    size_delivery='neutral' if bool(o['neutral_bypass'][j]) else 'square' if bool(o['square_bypass'][j]) else 'accepted' if bool(o['accepted'][j]) else 'fallback',
                    failed_checks=[k for k,v in o['checks'].items() if not bool(v[j])],representation_failures=[] if ok else ['invalid_vector' if arm==model.ARMS[0] else 'invalid_direct_regression'])
                if arm==model.ARMS[0]:cell.update(vector_valid=ok,local_size_variance=o['local_size_variance'][j].cpu().tolist())
                else:cell.update(regression_valid=ok,local_log_variance=o['local_log_variance'][j].cpu().tolist())
                pred[r['image'],r['scale']]=cell
    output=[]
    for r in records:
        v={k:r[k] for k in ('image','sequence','domain','frame_id','scale','split','shard')}
        v.update(gt=r['gt_original'][0].tolist(),b=r['boxes_original'][0].tolist() if len(r['boxes_original']) else None,
            midpoint=r['midpoint_original'][0].tolist() if len(r['midpoint_original']) else None)
        v.update(pred.get((r['image'],r['scale']),dict(edge_residual=None,pre_guard=None,delta_roi=None,loss_edges_canonical=None,size_delivery='no_output',failed_checks=[],representation_failures=[])))
        if not frozen_identity(v):raise ValueError('M identity changed')
        output.append(v)
    return output


def near_report(rows):
    report=near_joint(rows)
    for group,cell in report.items():
        selected=[r for r in rows if r['midpoint'] is not None and (group=='overall' or r['domain']==group or r['sequence']==group or r['domain']+'/'+str(r['scale'])==group)]
        for j,edge in enumerate(('long','short')):
            values=[]
            for r in selected:
                gt=sorted(r['gt'][2:4],reverse=True)[j];m=abs(sorted(r['midpoint'][2:4],reverse=True)[j]/gt-1)
                c=abs(sorted(r['edge_residual'][2:4],reverse=True)[j]/gt-1)
                if m<=.03:values.append(max(0.,c-m))
            cell['near'][edge]['positive_harm_sum']=sum(values)
        cell['fallbacks']=sum(r['size_delivery']=='fallback' for r in selected)
        cell['invalid_representations']=sum(not r.get('representation_valid',True) for r in selected)
    return report


def structure_value(v,d):
    checks={}
    for dom in ('real','sim'):
        for edge in ('long','short'):
            a,b=[e['metrics'][dom]['edge_residual']['sizes']['edges'][edge]['mae'] for e in (v,d)]
            checks[dom+'/'+edge]=a<b-1e-6 if dom=='sim' else a<=b+.001
            checks[dom+'/near_'+edge]=v['diagnostics'][dom]['near'][edge]['after_mae']<=d['diagnostics'][dom]['near'][edge]['after_mae']+.001
        for metric in ['dfr_percent_per_frame']:
            checks[dom+'/'+metric]=v['metrics'][dom]['edge_residual']['temporal'][dom][metric]<=d['metrics'][dom]['edge_residual']['temporal'][dom][metric]
        checks[dom+'/riou']=v['metrics'][dom]['edge_residual']['mean_riou_full_frame']>=d['metrics'][dom]['edge_residual']['mean_riou_full_frame']-.001
    for k in ('a','r'):checks['sim/'+k]=v['metrics']['sim']['edge_residual']['sizes'][k]['abs_mae']<=d['metrics']['sim']['edge_residual']['sizes'][k]['abs_mae']+1e-6
    checks['overall/riou']=v['metrics']['overall']['edge_residual']['mean_riou_full_frame']>=d['metrics']['overall']['edge_residual']['mean_riou_full_frame']-1e-5
    return dict(pass_all=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],note='Limited single-seed structure+decoder comparison; not adoption or independent statistical significance. See all24 same-epoch deltas.')


def engineering(records,out,torch):
    present=[r for r in records if len(r['midpoint_original'])];heads={a:fresh(a,torch).cpu().eval() for a in model.ARMS}
    stems=[h.stem.state_dict() for h in heads.values()]
    if not equal(*stems,torch):raise ValueError('Initial common stems differ')
    views=[];captured={}
    hooks=[h.stem.register_forward_pre_hook(lambda mod,x,a=a:captured.update({a:x[0].detach().clone()})) for a,h in heads.items()]
    with torch.no_grad():
        for i,r in enumerate(records):
            x=[r[k] for k in ('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original')]
            pair=model.read_pair(*x);o={a:h(*x) for a,h in heads.items()}
            if any(not torch.equal(v['boxes_original'],x[-1]) for v in o.values()):raise ValueError('Full TRAIN neutral/missing mismatch')
            valid=bool(len(x[-1]));cell={k:r[k] for k in ('image','sequence','domain','scale','shard')};cell.update(missing=not valid,neutral_exact=True)
            if valid:
                if not torch.equal(captured[model.ARMS[0]],captured[model.ARMS[1]]):raise ValueError('All267 inputs differ')
                cell.update(W_sum=float(pair['support'].sum()),W_active=int((pair['support']>0).sum()),input_max_abs_diff=0.,
                    support_A=float(pair['source_support']['fixed_b'].sum()),support_Q=float(pair['source_support']['q_aligned'].sum()),
                    outside_A=int((~pair['inside']['fixed_b']).sum()),outside_Q=int((~pair['inside']['q_aligned']).sum()),
                    source_min_xy=pair['source_local']['fixed_b'].amin(1)[0].tolist(),source_max_xy=pair['source_local']['fixed_b'].amax(1)[0].tolist())
                if cell['W_sum']<=0 or any(not bool(model.valid(o[a],a).all()) for a in model.ARMS):raise ValueError('Natural TRAIN zero support or invalid neutral')
            views.append(cell)
            if i%500==0:print('TRAIN common-input/neutral',i,len(records),flush=True)
    for h in hooks:h.remove()
    write(out/'train_support_views.json',views)
    grouped=[]
    for dom in ('real','sim'):
        for scale in (.5,1.):
            sub=[v for v in views if v['domain']==dom and v['scale']==scale];active=[v for v in sub if not v['missing']]
            grouped.append(dict(domain=dom,scale=scale,views=len(sub),outputs=len(active),missing=len(sub)-len(active),neutral_all_exact=True,all267_exact=True,zero_W=0,W_sum=sum(v['W_sum'] for v in active)))
    write(out/'train_support_summary.json',grouped)
    report=dict(status='PASS',views=len(records),outputs=len(present),common_stem_initial_sha256=state_digest(stems[0]),parameter_counts=model.PARAMETERS,
        full_neutral_missing_exact=True,all267_inputs_exact=True,val_shards_loaded=0,temporary_updates_discarded=True,fixed_batch_members=[[dict(image=r['image'],scale=r['scale']) for r in present[i:i+8]] for i in (0,8)],arms={})
    for arm in model.ARMS:
        h=fresh(arm,torch);opt=optimizer(h,torch);x=batch(present[:8],torch);first=update(h,opt,x,arm,torch)
        if first['stem_grad_norm']!=0 or first['terminal_grad_norm']<=0:raise ValueError('First gradient connection')
        snap=dict(head=deepcopy(h.state_dict()),opt=deepcopy(opt.state_dict()),rng=rng_get(torch));buf=io.BytesIO();torch.save(snap,buf);buf.seek(0);got=torch.load(buf)
        if not equal(snap,got,torch):raise ValueError('Serialization/RNG drift')
        clone=fresh(arm,torch);clone.load_state_dict(got['head']);other=optimizer(clone,torch);other.load_state_dict(got['opt']);y=batch(present[8:16],torch)
        rng_set(snap['rng'],torch);second=update(h,opt,y,arm,torch);after=rng_get(torch)
        rng_set(got['rng'],torch);repeat=update(clone,other,y,arm,torch)
        if second['stem_grad_norm']<=0 or not equal(second,repeat,torch) or not equal(h.state_dict(),clone.state_dict(),torch) or not equal(opt.state_dict(),other.state_dict(),torch) or not equal(after,rng_get(torch),torch):raise ValueError('Actual Adam/RNG next step mismatch')
        report['arms'][arm]=dict(first=first,second=second,actual_adam_rng_resume_exact=True)
        del h,opt,clone,other,x,y
    # Each arm has 3 temporary updates including resume clone = 6 total.
    report['temporary_updates']=6
    hs={a:fresh(a,torch).eval() for a in model.ARMS};x=batch(present[:1],torch);timings=[];peaks={};order_rng=random.Random(1703)
    with torch.no_grad():
        for _ in range(20):
            for h in hs.values():h(*x[:-1])
        for i in range(100):
            order=list(model.ARMS);order_rng.shuffle(order);cell=dict(index=i,order=order)
            for arm in order:
                torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();hs[arm](*x[:-1]);end.record();torch.cuda.synchronize();cell[arm]=begin.elapsed_time(end);peaks[arm]=max(peaks.get(arm,0),(torch.cuda.max_memory_allocated()-before)/2**20)
            cell['V_minus_D_ms']=cell[model.ARMS[0]]-cell[model.ARMS[1]];timings.append(cell)
    report['cost']=dict(scope='read+head batch1 excluding frozen frontend/transfers; shared GPU load descriptive',warmups=20,repeats=100,paired=timings,
        mean_ms={a:statistics.mean(v[a] for v in timings) for a in model.ARMS},V_minus_D_mean_ms=statistics.mean(v['V_minus_D_ms'] for v in timings),allocator_increment_mib=peaks)
    write(out/'engineering.json',report);del heads,hs,x;torch.cuda.empty_cache();seed(torch)
    return report


def execute(args,out,torch,proof):
    from crane_project.tools import run_port_geometry_size_distribution_v1 as prior
    from crane_project.utils import port_geometry_midpoint_sigma_v1 as frozen_model
    from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
    from crane_project.tools.run_port_geometry_size_condition_v1 import components
    sealed=prior.sealed;history,differences=prior.historical_identity()
    if sha(args.b_weight)!=prior.B_SHA:raise ValueError('Actual B SHA')
    cache=sealed.checked_cache(args.cache_dir,history);selection,references,indexed=sealed.checked_selection(args.head_dir,history)
    frozen=sealed.load_head(torch,frozen_model,args.head_dir,selection,cache,'cuda:0');frozen_digest=sealed.state_digest(frozen)
    train,nbytes=parent.load_records(args.cache_dir,cache,frozen,'cuda:0',torch,model,references,('train_s1','train_s05'))
    if len(train)!=5116 or sum(bool(len(r['midpoint_original'])) for r in train)!=5112:raise ValueError('Complete TRAIN identity')
    inputs=dict(b_sha256=sha(args.b_weight),m_sha256=sha(args.head_dir/sealed.FIXED['path']),cache_manifest_sha256=sha(args.cache_dir/'cache_manifest.json'),shards=cache['files'],reference_artifacts=indexed,historical_identity=history,unused_detector_differences=differences)
    write(out/'input_identity.json',inputs);eng=engineering(train,out,torch);write(out/'ENGINEERING_PASSED.json',dict(status='PASS',formal_updates=0,val_shards_loaded=0,temporary_updates_discarded=True))
    if args.engineering_only:return
    val,nv=parent.load_records(args.cache_dir,cache,frozen,'cuda:0',torch,model,references,('val_s1',))
    if len(val)!=887 or sum(bool(len(r['midpoint_original'])) for r in val)!=886:raise ValueError('Complete VAL identity')
    report=dict(protocol=VERSION,status='RUNNING',formal_updates=0,inputs=inputs,source_proof=proof,source_commit=args.source_commit,common_stem_initial_sha256=eng['common_stem_initial_sha256'],arms={},test_reads=0,depth_reads=0,feature_extractions=0,automatic_adoption=False)
    present=[r for r in train if len(r['midpoint_original'])];cache_iou={};representatives={}
    for arm in model.ARMS:
        dest=out/arm;dest.mkdir();h=fresh(arm,torch);opt=optimizer(h,torch);initial=state_digest(h.state_dict());stem=state_digest(h.stem.state_dict())
        if opt.state or stem!=report['common_stem_initial_sha256']:raise ValueError('Fresh common initialization differs')
        zv=evaluate(h,val,arm,torch);zt=evaluate(h,train,arm,torch)
        if any(r['edge_residual']!=r['midpoint'] for r in zv+zt):raise ValueError('All natural neutral mismatch')
        if arm==model.ARMS[0]:
            rows_write(out/'baseline_val.rows.jsonl.gz',zv);rows_write(out/'baseline_train.rows.jsonl.gz',zt)
            summary=metrics.summary(zv,cache_iou);write(out/'baseline_val.summary.json',dict(metrics=summary,components=components(zv),diagnostics=near_report(zv)))
            for d,correct in [('real',199),('sim',464)]:
                if summary[d]['midpoint']['sizes']['joint10']!=correct:raise ValueError('M count mismatch')
        del zv,zt;events=[]
        for epoch in range(1,25):
            began=time.monotonic();ids=list(range(len(present)));random.Random(1703+epoch).shuffle(ids);loss_sum=0.;updates=0
            for start in range(0,len(ids),8):
                rr=[present[j] for j in ids[start:start+8]];x=batch(rr,torch);step=update(h,opt,x,arm,torch);report['formal_updates']+=1;updates+=1;loss_sum+=step['loss']*len(rr)
                if start%1600==0:print(json.dumps(dict(stage='train',arm=arm,epoch=epoch,view=start,**step)),flush=True)
            if updates!=639:raise ValueError('Fixed update budget')
            vr=evaluate(h,val,arm,torch);summary=metrics.summary(vr,cache_iou);diag=near_report(vr);comp=components(vr);gate=eligibility(summary,diag)
            tr=evaluate(h,train,arm,torch);td=dict(size_groups=metrics.train_report(tr),diagnostics=near_report(tr),cross_edge_by_scale={str(sc):model.cross_edge([r for r in tr if r['scale']==sc]) for sc in (.5,1.)})
            saved_rng=rng_get(torch);checkpoint=dict(protocol=VERSION,arm=arm,epoch=epoch,updates=epoch*639,input_identity=inputs,source_commit=args.source_commit,
                head_state={k:v.detach().cpu().clone() for k,v in h.state_dict().items()},optimizer_state=deepcopy(opt.state_dict()),rng=saved_rng)
            path=dest/('head_epoch_%02d.pth'%epoch)
            with path.open('xb') as f:torch.save(checkpoint,f)
            if not equal(checkpoint,torch.load(path,map_location='cpu'),torch):raise ValueError('Saved checkpoint differs')
            rng_set(saved_rng,torch)
            ep=dict(epoch=epoch,metrics=summary,components=comp,diagnostics=diag,cross_edge=model.cross_edge(vr),eligibility=gate,selection_key=ranking(summary,epoch),checkpoint_sha256=sha(path),schedule_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),training_loss=loss_sum/len(present),epoch_seconds=time.monotonic()-began)
            write(dest/('epoch_%02d_val.json'%epoch),ep);rows_write(dest/('epoch_%02d_val.rows.jsonl.gz'%epoch),vr);write(dest/('epoch_%02d_train.json'%epoch),td);rows_write(dest/('epoch_%02d_train.rows.jsonl.gz'%epoch),tr)
            events.append(ep);write(dest/'progress.json',dict(last_complete_epoch=epoch,updates=epoch*639,epochs=events));print(json.dumps(dict(stage='epoch_finished',arm=arm,epoch=epoch,eligible=gate['pass_all'],failed=gate['failed'],selection_key=ep['selection_key'],elapsed=ep['epoch_seconds'])),flush=True)
            del x,vr,tr,td
        eligible=[e for e in events if e['eligibility']['pass_all']];chosen=min(eligible or events,key=lambda e:e['selection_key'])
        result=dict(initial_state_digest=initial,initial_stem_digest=stem,representative=chosen['epoch'],selected=chosen['epoch'] if eligible else None,eligible_epochs=[e['epoch'] for e in eligible],epochs=events)
        representatives[arm]=chosen;report['arms'][arm]=result;write(dest/'selection.json',result);write(out/'training.json',report);del h,opt;torch.cuda.empty_cache()
    v,d=[report['arms'][a] for a in model.ARMS]
    if report['formal_updates']!=30672 or v['initial_stem_digest']!=d['initial_stem_digest'] or [e['schedule_sha256'] for e in v['epochs']]!=[e['schedule_sha256'] for e in d['epochs']]:raise ValueError('Fair budget/stem/schedule')
    if frozen_digest!=sealed.state_digest(frozen) or sha(args.b_weight)!=prior.B_SHA or sha(args.head_dir/sealed.FIXED['path'])!=sealed.FIXED['sha256']:raise ValueError('Frontend state or bytes changed')
    if sha(args.cache_dir/'cache_manifest.json')!=cache['manifest_sha256'] or any(sha(args.cache_dir/n)!=h for n,h in cache['files'].items()):raise ValueError('Cache bytes changed')
    report.update(status='COMPLETE',structure_value=structure_value(*[representatives[a] for a in model.ARMS]));write(out/'training.json',report)


def main(args):
    out=args.out_dir.resolve()
    if out.parent!=(ROOT/'work_dirs'/VERSION).resolve() or out.exists():raise ValueError('Fresh result-only directory required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3' or os.environ.get('WORLD_SIZE','1')!='1':raise ValueError('Only single physicalGPU3 authorized')
    out.mkdir(parents=True);stop=None;report=dict(protocol=VERSION,status='FAILED',started_utc=datetime.now(timezone.utc).isoformat())
    try:
        proof=checked_sources();write(out/'source_identity.json',proof);write(out/'protocol.json',protocol())
        from crane_project.tools.audit_port_size_native_p3_v1 import setup_gpu
        resource,stop,thread=setup_gpu(out,7000,6800,6000);report['resource']=resource
        import torch
        torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(6000*2**20/torch.cuda.get_device_properties(0).total_memory,0);seed(torch)
        report['runtime']=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),python=sys.version,executable=sys.executable)
        execute(args,out,torch,proof)
        if checked_sources()!=proof:raise ValueError('Source changed during run')
        report.update(status='ENGINEERING_ONLY_COMPLETE' if args.engineering_only else 'COMPLETE',allocator_peak_mib=torch.cuda.max_memory_allocated()/2**20,reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20)
    except Exception as exc:report['error']=repr(exc);raise
    finally:
        if stop is not None:stop.set();thread.join(timeout=11)
        report['ended_utc']=datetime.now(timezone.utc).isoformat();write(out/'finished.json',report)
        write(out/'artifacts.json',dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--source-commit',required=True);p.add_argument('--engineering-only',action='store_true')
    p.add_argument('--cache-dir',type=Path,default=ROOT/'work_dirs/port_results/geometry/port_geometry_midpoint_formal_v1_roi_cache')
    p.add_argument('--head-dir',type=Path,default=ROOT/'work_dirs/crane_symeood_k1_port_day2night_midpoint_sigma_v1/sigma_1p5')
    p.add_argument('--b-weight',type=Path,default=ROOT/'work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth')
    return p


if __name__=='__main__':main(parser().parse_args())
