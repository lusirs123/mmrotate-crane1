#!/usr/bin/env python3
"""One fixed24 candidate vs verified saved uniform-I; full VAL every epoch."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_geometry_size_condition_v1 as condition
from crane_project.utils import port_geometry_size_axis_balance_v1 as axis

old, sealed, metrics = condition.old, condition.sealed, condition.metrics
VERSION = axis.VERSION
MANIFEST = ROOT/('crane_project/tools/'+VERSION+'_sources.json')
CONTRACT = ROOT/('crane_project/tools/'+VERSION+'_protocol.json')


def protocol():
    return dict(protocol=VERSION, settings=old.SETTINGS, parameter_count=2712,
        timing='After S019/S020; before any axis-balanced fitting. Does not change I07 identity.',
        change='Only fixed TRAIN canonical-axis weights; same I forward/init/targets/SmoothL1/delivery. near3 coefficient1, other two error-direction sample masses balanced within cap4 and complete cell mean1. No gradient/loss equality claimed.',
        loss='L=sum_ij(w_ij*SmoothL1_beta0.1(delta_hat_ij-target_ij))/(2*N); N=present supervised views in current batch. No batch-weight-sum normalization. Canonical weights -> M raw -> B-model ROI axes.',
        weights='Full5112 present TRAIN views; per domain/scale/edge, fixed M signed relative error: abs<=.03 near3, else negative undersized/positive oversized. near=1; sum_g min(4*n_g,lambda)=n_under+n_over. Labels/weights absent from inference. Both directions required in8 cells.',
        control='Verify complete S019 uniform-I run/source/cache/runtime/init/schedule/all24 heads+rows+summaries. Replay01/07/21/24. No silent retraining on mismatch.',
        evaluation='Full TRAIN diagnostic and all887 VAL each24 epochs, no TRAIN gate. All custom detection metrics,3 geometry rates/denominators/coverage, video/domain/overall; AP11 cached-GT supplement.',
        selection='Each arm descriptive Sim longMAE, shortMAE, earliest epoch over24; independently same ordering within adoption-passing set. Fixed original I07 reference is never reselected.',
        adoption=dict(sim='Both MAE<min(B,I07)-1e-6; abs(a)<=M+1e-6; full meanRIoU>I07+1e-5; each fixed-M single-edge near3 MAE<=I07+1e-6.',
            real='Both MAE<=I07+1e-6; full meanRIoU>=I07-1e-5.',
            protection='Exact M center/raw and canonical angle/score/output. Per domain MCML max/mean<=I07, TDR identical denominator and hits>=I07, ACI and adjacent-pair identity.',
            diagnostic='P95/bias/r,4 groups a,joint10/paired flips,DFR/MRF/AP and per-video costs; no extra rejection gates.'),
        effects='Separate from adoption: compare each arm selected adoption-passing else descriptive representative under same fixed key. Report per-edge Sim MAE gain>1e-6 and both-edge flag, domain RIoU/Real costs independently. A failure of strict adoption is not proof of no local effect. No alternate epoch hunting.',
        engineering='Full real neutral TRAIN/VAL, actual raw-axis map roundtrip, full cell fixed weights, real all-one vs original-I2updates bit-exact losses/gradients/Adam/clipping/state;2weighted smoke updates; save/restore optimizer/RNG. Six total discarded engineering head updates, fresh seed/state for formal15336.',
        stop='Engineering failure stops. One24-epoch balanced candidate; no added module/loss/data/epochs or TEST selection. No automatic promotion. TRAIN diagnostic never skips VAL.',
        limits=dict(test_access=False, depth_access=False, feature_extractions=0, detector_updates=0,
                    frozen_midpoint_updates=0, cache_rebuilt=False, automatic_promotion=False))


def checked_sources():
    parent = condition.checked_sources(); source = sealed.read(MANIFEST)
    required = {'crane_project/utils/'+VERSION+'.py', 'crane_project/tools/run_'+VERSION+'.py',
                'tests/test_'+VERSION+'.py', 'crane_project/tools/'+VERSION+'_protocol.json'}
    if set(source['sources']) != required or source['protocol'] != VERSION or sealed.read(CONTRACT) != protocol():
        raise ValueError('Axis source/protocol scope differs')
    for name, digest in source['sources'].items():
        if sealed.sha(ROOT/name) != digest: raise ValueError('Axis source drift: '+name)
    return dict(condition=parent, sources=source, manifest_sha256=sealed.sha(MANIFEST))


def near_summary(rows):
    result = {}
    for domain in ('real', 'sim'):
        result[domain] = {}
        for j, edge in enumerate(('long', 'short')):
            cell = [r for r in rows if r['domain']==domain and r['midpoint'] is not None and axis.labels(r['gt'],r['midpoint'])[j]=='near3']
            result[domain][edge] = dict(count=len(cell), methods={m:metrics.size_summary(cell,m)['edges'][edge] for m in metrics.METHODS})
    return result


def adoption(value, near, reference, reference_near):
    checks = {}
    for domain in ('real','sim'):
        c,b,m = [value[domain][k] for k in ('edge_residual','b','midpoint')]
        ref = reference[domain]['edge_residual']; p = domain+'/'
        for edge in ('long','short'):
            x,y=c['sizes']['edges'][edge]['mae'],ref['sizes']['edges'][edge]['mae']
            checks[p+edge]=(x<min(y,b['sizes']['edges'][edge]['mae'])-1e-6 if domain=='sim' else x<=y+1e-6)
            if domain=='sim':
                checks[p+'near_'+edge]=(near[domain][edge]['count']==reference_near[domain][edge]['count'] and
                    near[domain][edge]['methods']['edge_residual']['mae']<=reference_near[domain][edge]['methods']['edge_residual']['mae']+1e-6)
        if domain=='sim': checks[p+'a']=c['sizes']['a']['abs_mae']<=m['sizes']['a']['abs_mae']+1e-6
        checks[p+'riou']=(c['mean_riou_full_frame']>ref['mean_riou_full_frame']+1e-5 if domain=='sim' else c['mean_riou_full_frame']>=ref['mean_riou_full_frame']-1e-5)
        ct,rt=c['temporal'][domain],ref['temporal'][domain]
        checks[p+'continuity']=(ct['mcml_max']<=rt['mcml_max'] and ct['mcml_mean']<=rt['mcml_mean'] and
            ct['tdr_w10']['denominator']==rt['tdr_w10']['denominator'] and ct['tdr_w10']['numerator']>=rt['tdr_w10']['numerator'])
        checks[p+'identity']=value[domain]['paired']['frozen_identity'] and value[domain]['paired']['new_output']==value[domain]['paired']['lost_output']==0
        checks[p+'angle_center']=(ct['aci']==rt['aci'] and ct['aci_adjacent_pairs']==rt['aci_adjacent_pairs'] and
            c['center_correct_full_frame']==ref['center_correct_full_frame'] and c['pure_angle_rmse_deg']==ref['pure_angle_rmse_deg'])
    return dict(pass_all=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v])


def selection_key(value, epoch):
    x=value['sim']['edge_residual']['sizes']['edges']
    return [x['long']['mae'],x['short']['mae'],epoch]


def choose(results):
    if sorted(r['epoch'] for r in results)!=list(range(1,25)):raise ValueError('All24 VAL epochs required')
    good=[r for r in results if r['adoption']['pass_all']]
    return dict(representative=min(results,key=lambda r:r['selection_key']),
                selected=min(good,key=lambda r:r['selection_key']) if good else None,
                passing_epochs=[r['epoch'] for r in good],epochs=results)


def update(head,opt,packed,indices,weights,torch,compare_original=False):
    ts=[t[indices] for t in packed['tensors']];head.train();opt.zero_grad()
    output=head(*ts[:-1]);terms=axis.loss(output,ts[-1],ts[-2],ts[3],weights[indices])
    if not bool(torch.isfinite(terms['total'])):raise ValueError('Nonfinite weighted loss')
    terms['total'].backward()
    params=list(head.parameters())
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):raise ValueError('Invalid gradients')
    gradients=[p.grad.detach().clone() for p in params] if compare_original else None
    norm=lambda ps:math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in ps))
    stem,terminal=norm(head.stem.parameters()),norm(list(head.u.parameters())+list(head.v.parameters()))
    before=float(torch.nn.utils.clip_grad_norm_(params,10.));opt.step()
    if any(not bool(torch.isfinite(p).all()) for p in params):raise ValueError('Invalid updated head')
    return dict(loss=float(terms['total'].detach()),unweighted_loss=float(terms['unweighted'].detach()),
                stem_grad_norm=stem,terminal_grad_norm=terminal,
                grad_norm_before_clip=before,clipped=before>10.,
                pointwise=terms['canonical_pointwise'].detach().cpu().tolist(),gradients=gradients)


def group_losses(acc,event,batch,prepared):
    for position,i in enumerate(batch):
        r=prepared['records'][i]
        for j,edge in enumerate(('long','short')):
            key=r['cell']+'/'+edge+'/'+r['groups'][j]
            cell=acc.setdefault(key,dict(count=0,unweighted_sum=0.,weighted_sum=0.))
            x=event['pointwise'][position][j];w=prepared['canonical_weights'][i][j]
            cell['count']+=1;cell['unweighted_sum']+=x;cell['weighted_sum']+=x*w


def save(path,head,opt,epoch,proof,torch):
    from crane_project.utils.port_geometry_size_condition_v1 import GeometryConditionSizeHead
    import random
    rng=dict(python=random.getstate(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    payload=dict(protocol=VERSION,arm='axis_balanced_I',epoch=epoch,updates=epoch*639,proof=proof,
        head_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
        optimizer_state=deepcopy(opt.state_dict()),rng=rng,head_digest=sealed.state_digest(head))
    with path.open('xb') as stream:torch.save(payload,stream)
    restored=sealed.torch_load(torch,path);reload=GeometryConditionSizeHead().to(next(head.parameters()).device)
    reload.load_state_dict(restored['head_state'],strict=True)
    replay_opt=torch.optim.Adam(reload.parameters(),lr=.001,weight_decay=0.);replay_opt.load_state_dict(restored['optimizer_state'])
    if sealed.state_digest(reload)!=payload['head_digest']:raise ValueError('Head serialization differs')
    for a,b in zip(opt.state_dict()['state'].values(),replay_opt.state_dict()['state'].values()):
        for k in a:
            if torch.is_tensor(a[k]):
                if not torch.equal(a[k].cpu(),b[k].cpu()):raise ValueError('Optimizer serialization differs')
            elif a[k]!=b[k]:raise ValueError('Optimizer step serialization differs')
    if restored['rng']['python']!=rng['python'] or not torch.equal(restored['rng']['cpu'],rng['cpu']) or any(not torch.equal(a,b) for a,b in zip(restored['rng']['cuda'],rng['cuda'])):raise ValueError('RNG serialization differs')
    random.setstate(rng['python']);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
    return reload,dict(sha256=sealed.sha(path),head_digest=payload['head_digest'],save_reload_exact=True)


def run(args):
    sources=checked_sources();out=Path(args.out_dir).resolve()
    if out.parent!=(ROOT/'work_dirs'/VERSION).resolve() or out.exists():raise ValueError('Fresh direct output path required')
    out.mkdir(parents=True);started=time.monotonic()
    proof=dict(source_identity=sources,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip())
    report=dict(protocol=VERSION,status='FAILED',started_utc=datetime.now(timezone.utc).isoformat(),**protocol()['limits'])
    try:
        sealed.write(out/'protocol.json',protocol());sealed.write(out/'source_identity.json',sources)
        if args.stage=='check':report.update(status='STATIC_CONTRACT_PASS',head_updates=0);return
        import torch
        from crane_project.utils import port_geometry_midpoint_sigma_v1 as frozen_model
        from crane_project.utils.port_geometry_size_condition_v1 import GeometryConditionSizeHead
        if not torch.cuda.is_available() or len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))!=1 or not os.environ.get('CUDA_VISIBLE_DEVICES') or os.environ.get('WORLD_SIZE','1')!='1':raise ValueError('Explicit single GPU required')
        torch.set_num_threads(1);device=torch.device('cuda:0');torch.cuda.set_device(device);old.seed_all(torch)
        control,cache_dir,head_dir,b_weight=[Path(getattr(args,n)).resolve() for n in ('control_run','cache_dir','head_dir','b_weight')]
        for p in (control,cache_dir,head_dir,b_weight):
            if out==p or out in p.parents or p in out.parents:raise ValueError('Input/output overlap')
        index=sealed.read(control/'artifacts.json')['files']
        file=lambda n:condition.val_replay.checked_file(control,index,n)
        for name in index:file(name)
        cp=sealed.read(file('input_runtime_proof.json'));cc=sealed.read(file('completion.json'))
        if cp['source_identity']!=sources['condition'] or cp['git_head']!='57427341162a9434c8d65419d5b1175ecdd6052b' or cc['formal_head_updates']!=15336 or cc['candidate']['selected']['epoch']!=7:raise ValueError('Uniform-I control identity differs')
        runtime=dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            tf32_matmul=getattr(torch.backends.cuda.matmul,'allow_tf32',None),tf32_cudnn=getattr(torch.backends.cudnn,'allow_tf32',None))
        for k,v in runtime.items():
            if k!='cuda_visible_devices' and v!=cp['runtime'][k]:raise ValueError('Runtime differs: '+k)
        historical,differences=old.historical_identity();cache=sealed.checked_cache(cache_dir,historical)
        selection,references,reference_index=sealed.checked_selection(head_dir,historical)
        if sealed.sha(b_weight)!=old.B_SHA or cache['files']!=cp['shards']:raise ValueError('B/cache differs')
        frozen=sealed.load_head(torch,frozen_model,head_dir,selection,cache,device);frozen_digest=sealed.state_digest(frozen)
        train,val,_=old.load_records(cache_dir,cache,frozen,device,torch,old.head_module(),references)
        tp,vp=old.pack(train,device,torch),old.pack(val,device,torch)
        if (len(train),len(tp['indices']),len(val),len(vp['indices']))!=(5116,5112,887,886):raise ValueError('Incomplete TRAIN/VAL')
        schedule=sealed.read(file('schedule.json'))
        batches=[old.epoch_batches(5112,e) for e in range(1,25)]
        identities=[dict(image=train[i]['image'],scale=train[i]['scale']) for i in tp['indices']]
        if schedule['batches']!=batches or schedule['train_view_identity']!=identities:raise ValueError('Control schedule differs')
        sealed.write(out/'schedule.json',schedule)
        old.seed_all(torch);h=GeometryConditionSizeHead().to(device);initial=sealed.state_digest(h)
        if initial!=sealed.read(file('preflight.json'))['initial_state']:raise ValueError('Control initial state differs')
        bt,bv=old.evaluate_rows(h,train,tp,torch),old.evaluate_rows(h,val,vp,torch)
        if any(r['midpoint']!=r['edge_residual'] for r in bt+bv):raise ValueError('Neutral full replay differs')
        for s,rs in [('train',bt),('val',bv)]:
            condition.val_replay.compare_numeric(rs,old.read_rows(file('baseline_'+s+'.rows.jsonl.gz')))
            old.write_rows(out/('baseline_'+s+'.rows.jsonl.gz'),rs)
        prepared=axis.prepare(bt,tp['indices']);weights=torch.tensor(prepared['canonical_weights'],device=device,dtype=tp['tensors'][0].dtype)
        if not torch.equal(weights,axis.permute_axes(axis.permute_axes(weights,tp['tensors'][-2],tp['tensors'][3]),tp['tensors'][-2],tp['tensors'][3])):raise ValueError('Actual axis mapping roundtrip failed')
        # Each weight uses exactly the same raw-axis involution as target assignment.
        target=old.head_module().size_targets(tp['tensors'][-1],tp['tensors'][-2],tp['tensors'][3])['log_residual']
        canonical_log=axis.permute_axes(target,tp['tensors'][-2],tp['tensors'][3])
        g=tp['tensors'][-1][:,2:4].sort(-1,descending=True).values;m=tp['tensors'][-2][:,2:4].sort(-1,descending=True).values
        if not torch.equal(canonical_log,(g/m).log()):raise ValueError('Target/weight axis association differs')
        sealed.write(out/'train_weights.json',prepared)
        riou_cache={};base=metrics.summary(bv,riou_cache)
        condition.val_replay.compare_numeric(base,sealed.read(file('baseline_val_summary.json')))
        sealed.write(out/'baseline_val_summary.json',base);sealed.write(out/'baseline_components.json',condition.components(bv))
        ref_rows=old.read_rows(file('candidate/val_epoch_07.rows.jsonl.gz'));reference=sealed.read(file('candidate/val_epoch_07.summary.json'));ref_near=near_summary(ref_rows)
        sealed.write(out/'fixed_I07_summary.json',reference);sealed.write(out/'fixed_I07_near.json',ref_near)
        proof.update(runtime=runtime,control_run=str(control),control_artifacts_sha256=sealed.sha(control/'artifacts.json'),
            original_control_proof=cp,cache_manifest_sha256=sealed.CACHE_SHA,shards=cache['files'],fixed_b_sha256=old.B_SHA,
            fixed_m=sealed.FIXED,reference_artifacts=reference_index,initial_state=initial,unused_historical_differences=differences)
        control_results=[];dest=out/'control';dest.mkdir()
        for epoch in range(1,25):
            prefix='candidate/';meta=sealed.read(file(prefix+'epoch_%02d.json'%epoch));ck=file(prefix+'head_epoch_%02d.pth'%epoch)
            payload=sealed.torch_load(torch,ck)
            if sealed.sha(ck)!=meta['checkpoint']['sha256'] or payload['protocol']!=condition.VERSION or payload['epoch']!=epoch or payload['updates']!=epoch*639 or payload['proof']!=cp:raise ValueError('Control checkpoint metadata differs')
            rows=old.read_rows(file(prefix+'val_epoch_%02d.rows.jsonl.gz'%epoch));value=sealed.read(file(prefix+'val_epoch_%02d.summary.json'%epoch))
            condition.val_replay.compare_numeric(metrics.summary(rows,riou_cache),value)
            if epoch in (1,7,21,24):
                rh=GeometryConditionSizeHead().to(device);rh.load_state_dict(payload['head_state'],strict=True)
                if sealed.state_digest(rh)!=payload['head_digest']:raise ValueError('Control head digest differs')
                replay=old.evaluate_rows(rh,val,vp,torch)
                condition.val_replay.compare_numeric([{k:v for k,v in r.items() if k!='metrics'} for r in rows],replay)
            near=near_summary(rows)
            result=dict(epoch=epoch,selection_key=selection_key(value,epoch),adoption=adoption(value,near,reference,ref_near),checkpoint_sha256=sealed.sha(ck),reused=True)
            control_results.append(result)
            for suffix,data in [('summary',value),('components',condition.components(rows)),('near',near)]:sealed.write(dest/('val_epoch_%02d.%s.json'%(epoch,suffix)),data)
            sealed.write(dest/('epoch_%02d.json'%epoch),result)
        controls=choose(control_results);sealed.write(dest/'selection.json',controls)
        # Real-cache all-one branch vs original updates, exact gradients/state.
        old.seed_all(torch);a=GeometryConditionSizeHead().to(device)
        old.seed_all(torch);b=GeometryConditionSizeHead().to(device)
        oa,ob=[torch.optim.Adam(x.parameters(),lr=.001,weight_decay=0.) for x in (a,b)]
        ones=torch.ones_like(weights);equivalence=[]
        for batch in batches[0][:2]:
            event=update(a,oa,tp,batch,ones,torch,True)
            # Obtain original pre-clip grads separately without an extra update.
            ts=[t[batch] for t in tp['tensors']];ob.zero_grad()
            original_terms=old.head_module().loss(b(*ts[:-1]),ts[-1],ts[-2],ts[3],'decoded_control');original_terms['total'].backward()
            if event['loss']!=float(original_terms['total'].detach()) or any(not torch.equal(x,p.grad) for x,p in zip(event['gradients'],b.parameters())):raise ValueError('All-one loss/gradient equivalence failed')
            torch.nn.utils.clip_grad_norm_(b.parameters(),10.);ob.step()
            if sealed.state_digest(a)!=sealed.state_digest(b):raise ValueError('All-one Adam/clipping update differs')
            for left,right in zip(oa.state_dict()['state'].values(),ob.state_dict()['state'].values()):
                for k in left:
                    if torch.is_tensor(left[k]):
                        if not torch.equal(left[k],right[k]):raise ValueError('All-one optimizer moments differ')
                    elif left[k]!=right[k]:raise ValueError('All-one optimizer steps differ')
            equivalence.append(dict(loss=event['loss'],loss_grad_state_optimizer_exact=True))
        old.seed_all(torch);smoke=GeometryConditionSizeHead().to(device);opt=torch.optim.Adam(smoke.parameters(),lr=.001,weight_decay=0.)
        events=[]
        for batch in batches[0][:2]:
            event=update(smoke,opt,tp,batch,weights,torch);events.append({k:v for k,v in event.items() if k not in ('gradients','pointwise')})
        if events[0]['stem_grad_norm']!=0 or events[0]['terminal_grad_norm']<=0 or events[1]['stem_grad_norm']<=0:raise ValueError('Weighted gradient smoke failed')
        reload,ck=save(out/'smoke.pth',smoke,opt,0,proof,torch)
        with torch.no_grad():
            for k in ('bucket_logits','boxes_original'):
                if not torch.equal(smoke(*[t[:8] for t in tp['tensors'][:-1]])[k],reload(*[t[:8] for t in tp['tensors'][:-1]])[k]):raise ValueError('Smoke replay differs')
        sealed.write(out/'preflight.json',dict(status='ENGINEERING_PASS',initial_state=initial,full_neutral_exact=True,
            all_one_equivalence=equivalence,axis_mapping_exact=True,control_replay_epochs=[1,7,21,24],discarded_head_updates=6,weighted_smoke=events,checkpoint=ck))
        sealed.write(out/'input_runtime_proof.json',proof)
        print('ENGINEERING_PASS control uniform-I reused24; balanced fresh24',flush=True)
        old.seed_all(torch);h=GeometryConditionSizeHead().to(device)
        if sealed.state_digest(h)!=initial:raise ValueError('Engineering state inherited')
        opt=torch.optim.Adam(h.parameters(),lr=.001,weight_decay=0.)
        results=[];dest=out/'candidate';dest.mkdir()
        with (dest/'progress.jsonl').open('x') as log:
            for epoch,batchlist in enumerate(batches,1):
                losses=[];acc={};tick=time.monotonic()
                for step,batch in enumerate(batchlist,1):
                    event=update(h,opt,tp,batch,weights,torch);group_losses(acc,event,batch,prepared)
                    event.pop('gradients');event.pop('pointwise');losses.append(event['loss'])
                    log.write(json.dumps(dict(epoch=epoch,step=step,**event),allow_nan=False)+'\n')
                    if step%100==0 or step==639:log.flush();print('balanced',epoch,step,event['loss'],flush=True)
                if sum(v['count'] for v in acc.values())!=10224 or not math.isclose(sum(v['weighted_sum'] for v in acc.values())/10224,sum(losses)/len(losses),rel_tol=1e-6,abs_tol=1e-9):raise ValueError('Training loss accounting differs')
                for cell,data in prepared['cells'].items():
                    for g,n in data['counts'].items():
                        if acc[cell+'/'+g]['count']!=n:raise ValueError('Epoch group exposure differs')
                sealed.write(dest/('train_epoch_%02d.loss_groups.json'%epoch),dict(groups=acc,
                    unweighted_sum=sum(v['unweighted_sum'] for v in acc.values()),weighted_sum=sum(v['weighted_sum'] for v in acc.values()),
                    interpretation='Actual sequential training losses, not equalized gradients. Each view/axis once; denominators2N.'))
                tr=old.evaluate_rows(h,train,tp,torch);tr_summary=metrics.train_report(tr)
                old.write_rows(dest/('train_epoch_%02d.rows.jsonl.gz'%epoch),tr);sealed.write(dest/('train_epoch_%02d.summary.json'%epoch),tr_summary)
                _,ck=save(dest/('head_epoch_%02d.pth'%epoch),h,opt,epoch,proof,torch)
                rows=old.evaluate_rows(h,val,vp,torch);value=metrics.summary(rows,riou_cache);near=near_summary(rows)
                old.write_rows(dest/('val_epoch_%02d.rows.jsonl.gz'%epoch),rows)
                for suffix,data in [('summary',value),('components',condition.components(rows)),('near',near)]:sealed.write(dest/('val_epoch_%02d.%s.json'%(epoch,suffix)),data)
                result=dict(epoch=epoch,updates=epoch*639,checkpoint=ck,selection_key=selection_key(value,epoch),adoption=adoption(value,near,reference,ref_near),
                    train_pass_diagnostic_only=tr_summary['train_pass'],val_evaluated=True,mean_weighted_loss=sum(losses)/len(losses),elapsed_seconds=time.monotonic()-tick)
                results.append(result);sealed.write(dest/('epoch_%02d.json'%epoch),result)
                if sealed.state_digest(frozen)!=frozen_digest or any(p.grad is not None or p.requires_grad for p in frozen.parameters()):raise ValueError('Frozen M changed')
                print('EPOCH_COMPLETE',epoch,'key',result['selection_key'],'adoption',result['adoption']['pass_all'],flush=True)
        candidates=choose(results);sealed.write(dest/'selection.json',candidates)
        chosen=lambda r:(r['selected'] or r['representative'])['epoch']
        ce,re=chosen(candidates),chosen(controls)
        c=sealed.read(dest/('val_epoch_%02d.summary.json'%ce));r=sealed.read(out/'control'/('val_epoch_%02d.summary.json'%re))
        effects={}
        for domain in ('real','sim'):
            effects[domain]=dict(mae_delta={e:c[domain]['edge_residual']['sizes']['edges'][e]['mae']-r[domain]['edge_residual']['sizes']['edges'][e]['mae'] for e in ('long','short')},
                riou_delta=c[domain]['edge_residual']['mean_riou_full_frame']-r[domain]['edge_residual']['mean_riou_full_frame'])
        effects['sim_edge_gain']={e:effects['sim']['mae_delta'][e]<-1e-6 for e in ('long','short')}
        effects['sim_both_edge_gain']=all(effects['sim_edge_gain'].values())
        # Freeze and replay the reported candidate even if no adoption version.
        payload=sealed.torch_load(torch,dest/('head_epoch_%02d.pth'%ce));rh=GeometryConditionSizeHead().to(device);rh.load_state_dict(payload['head_state'])
        for p in rh.parameters():p.requires_grad_(False)
        replay=old.evaluate_rows(rh,val,vp,torch);expected=old.read_rows(dest/('val_epoch_%02d.rows.jsonl.gz'%ce))
        condition.val_replay.compare_numeric(replay,[{k:v for k,v in x.items() if k!='metrics'} for x in expected])
        condition.val_replay.compare_numeric(metrics.summary(replay,riou_cache),c)
        sealed.write(out/'frozen_cached_val_replay.json',dict(status='PASS',epoch=ce,frames=887,head_sha256=sealed.sha(dest/('head_epoch_%02d.pth'%ce)),native_image_inference=False,updates=0))
        if checked_sources()!=sources or sealed.sha(b_weight)!=old.B_SHA or sealed.sha(head_dir/sealed.FIXED['path'])!=sealed.FIXED['sha256']:raise ValueError('Source/weight drift')
        for name,digest in cache['files'].items():
            if sealed.sha(cache_dir/name)!=digest:raise ValueError('Cache drift')
        for name in index:file(name)
        report.update(status='COMPLETE_REVIEW_REQUIRED' if candidates['selected'] else 'COMPLETE_NO_ADOPTED_VERSION_KEEP_M_I07',
            control=controls,candidate=candidates,effects_vs_control=effects,formal_head_updates=15336,control_training_updates=0,
            discarded_engineering_head_updates=6,complete_val_epochs_per_arm=24,control_reused=True,frozen_m_unchanged=True)
    except BaseException as exc:report['error']=repr(exc);raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        sealed.write(out/'completion.json',report)
        sealed.write(out/'artifacts.json',dict(protocol=VERSION,files={str(p.relative_to(out)):sealed.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifacts.json'}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('check','run'),required=True);p.add_argument('--out-dir',required=True)
    for n in ('control-run','cache-dir','head-dir','b-weight'):p.add_argument('--'+n)
    run(p.parse_args())


if __name__=='__main__':main()
