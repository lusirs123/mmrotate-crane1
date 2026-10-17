#!/usr/bin/env python3
"""Two fresh 24-epoch TRAIN-only proposal arms; natural complete VAL/epoch."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import json
import math
import os
import random
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_geometry_size_condition_v1 as condition
from crane_project.utils import port_geometry_size_proposal_v1 as model
old,metrics,sealed=condition.old,condition.metrics,condition.sealed
VERSION=model.VERSION
ARMS=('repeat7','proposal7')


def protocol():
    return dict(protocol=VERSION,parameter_count=2712,h=.0625,slots=list(model.SLOTS),arms=list(ARMS),
        training=dict(epochs=24,batch_size=8,updates_per_epoch=639,seed=1703,lr=.001,clip_norm=10.,weight_decay=0.),
        data='Full2558 TRAIN, standard/half5116 paired views,5112 output views; no GT/center/old eligible filtering. Four missing views retained in natural reports. No new independent images.',
        change='Only TRAIN Q organization. repeat7: seven natural M slots; proposal7: natural, long+/-h,short+/-h,long GT exact,short GT exact. Same frozen visual inputs, unchanged I structure/readout and unweighted SmoothL1 beta.1. Stem deterministic and shared once/view in BOTH arms.',
        labels='GT long/short assigned ONCE by natural M raw axes, then actual B-model swap. log(GT_axis/Q_axis), no rematch on Q crossing, no clipping. GT-exact named axis zero. No slot/type/h/GT flag input to scorer.',
        inference='Unchanged original I.forward on natural M ONCE; original B range/raw axis/canonical angle/ROI/whole-pair fallback M; training Q is never mistaken for frozen M.',
        control='Both arms freshly trained24; no reuse of historical I even if forward/loss close. Compare repeated reduction/gradients/updates and record max drift. Adam can amplify roundoff in mathematically constant logit biases.',
        engineering='Frozen source/cache/B/M/I07 SHA; saved M baseline -> raw cache B/GT/image identities; real full neutral natural train/VAL; full7slot labels/support/axis-crossing/unreachable inventory; natural-wrapper agreement, repeated2updates and candidate2updates discarded, exact checkpoint/optimizer/RNG replay; no TRAIN admission gate.',
        evaluation='Every24 natural full887 VAL per arm, overall/Real/Sim/video custom metrics, AP11 cached-GT supplement, center<15,size10,angle<=3 with GT aspect>=1.2; outputs/missing/full denominators. Natural TRAIN groups and7slot recovery diagnostic only.',
        selection='Both arms same lexicographic Sim longMAE,shortMAE,earliest epoch, independent of TRAIN. Select best intrinsic-adoption-passing epoch; else descriptive best and KEEP M. No composite score or TEST/Unknown access.',
        adoption=dict(sim='Both MAE < min(B,I07)-1e-6; meanabs(a)<=M+1e-6; fullRIoU>max(M,I07)+1e-5.',
            real='Both MAE<=I07+1e-6; fullRIoU>=I07-1e-5; no substantive tolerated degradation.',
            near='Both domains, each fixed-M near3 single-edge MAE<=M+1e-6. Membership fixed from M, not I07 or candidate. Counts exact.',
            protection='M center/raw/canonical angle/score/presence exact. Per-domain ACI unchanged; MCML max/mean<=M; TDR hits>=M with identical denominators. DFR/MRF, P95/bias/r and size10 transitions disclosed, not extra gates.',
            mechanism='Chosen proposal vs independently chosen repeat control: Sim BOTH MAE lower>1e-6 and fullRIoU higher>1e-5; Real both MAE/RIoU no worse within1e-6/1e-5; near MAE no worse control+1e-6. Adoption and usefulness separately reported.'),
        resources=dict(physical_gpu=3,max_process_mib=1536,min_free_mib=2048,torch_allocator_cap_mib=1100),
        stop='Engineering/nonfinite/resource failure stops. Otherwise fixed24 per arm, all VAL even if TRAIN harms; no added amplitude/module/weight/epochs/TEST/depth, no automatic promotion.',
        limits=dict(detector_updates=0,midpoint_updates=0,feature_extractions=0,test_access=False,depth_access=False,control_reused=False))


def check_sources():
    parent=condition.checked_sources();p=ROOT/'crane_project/tools'/ (VERSION+'_protocol.json');s=ROOT/'crane_project/tools'/(VERSION+'_sources.json')
    if sealed.read(p)!=protocol():raise ValueError('Proposal protocol drift')
    manifest=sealed.read(s)
    for f,h in manifest['sources'].items():
        if sealed.sha(ROOT/f)!=h:raise ValueError('Proposal source drift '+f)
    return dict(parent=parent,proposal=manifest,manifest_sha256=sealed.sha(s))


def memory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    gpu=next(x.split(', ') for x in raw.splitlines() if int(x.split(',')[0])==3)
    own=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    cells=[x.split(', ') for x in own.splitlines() if x.strip()]
    value=dict(free_mib=int(gpu[2]),process_mib=sum(int(x[2]) for x in cells if int(x[0])==os.getpid() and x[1]==gpu[1]))
    if value['free_mib']<2048 or value['process_mib']>1536:raise RuntimeError('RESOURCE_STOP '+str(value))
    return value


def near(rows):
    result={}
    for d in ('real','sim'):
        result[d]={}
        for j,e in enumerate(('long','short')):
            rs=[r for r in rows if r['domain']==d and r['midpoint'] is not None and abs(sorted(r['midpoint'][2:4],reverse=True)[j]/sorted(r['gt'][2:4],reverse=True)[j]-1)<=.03]
            result[d][e]=dict(count=len(rs),methods={k:metrics.size_summary(rs,k)['edges'][e] for k in metrics.METHODS})
    return result


def adoption(value,nr,ref):
    checks={}
    for d in ('real','sim'):
        c,b,m=[value[d][k] for k in ('edge_residual','b','midpoint')];i=ref[d]['edge_residual'];key=d+'/'
        for e in ('long','short'):
            x=c['sizes']['edges'][e]['mae'];z=i['sizes']['edges'][e]['mae']
            checks[key+e]=x<min(z,b['sizes']['edges'][e]['mae'])-1e-6 if d=='sim' else x<=z+1e-6
            checks[key+'near_'+e]=nr[d][e]['methods']['edge_residual']['mae']<=nr[d][e]['methods']['midpoint']['mae']+1e-6
        if d=='sim':checks[key+'a']=c['sizes']['a']['abs_mae']<=m['sizes']['a']['abs_mae']+1e-6
        checks[key+'riou']=c['mean_riou_full_frame']>max(m['mean_riou_full_frame'],i['mean_riou_full_frame'])+1e-5 if d=='sim' else c['mean_riou_full_frame']>=i['mean_riou_full_frame']-1e-5
        a,t=c['temporal'][d],m['temporal'][d]
        checks[key+'continuity']=a['mcml_max']<=t['mcml_max'] and a['mcml_mean']<=t['mcml_mean'] and a['tdr_w10']['denominator']==t['tdr_w10']['denominator'] and a['tdr_w10']['numerator']>=t['tdr_w10']['numerator']
        checks[key+'identity']=value[d]['paired']['frozen_identity'] and value[d]['paired']['new_output']==value[d]['paired']['lost_output']==0
        checks[key+'center_angle']=c['center_correct_full_frame']==m['center_correct_full_frame'] and c['pure_angle_rmse_deg']==m['pure_angle_rmse_deg'] and a['aci']==t['aci'] and a['aci_adjacent_pairs']==t['aci_adjacent_pairs']
    return dict(pass_all=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v])


def key(value,epoch):
    e=value['sim']['edge_residual']['sizes']['edges'];return [e['long']['mae'],e['short']['mae'],epoch]


def checked_file(run,name,used):
    idx=sealed.read(run/'artifacts.json')['files'];p=condition.val_replay.checked_file(run,idx,name)
    used[name]=idx[name];return p


def load_cache(args,torch,used):
    cache=sealed.read(args.cache_dir/'cache_manifest.json')
    if sealed.sha(args.cache_dir/'cache_manifest.json')!=sealed.CACHE_SHA:raise ValueError('Cache manifest differs')
    records={};baselines={}
    for split,n in [('train',5116),('val',887)]:
        baseline=old.read_rows(checked_file(args.i_run,'baseline_'+split+'.rows.jsonl.gz',used))
        if len(baseline)!=n:raise ValueError('Incomplete M baseline')
        keyed={(r['image'],r['scale']):r for r in baseline}
        if len(keyed)!=n:raise ValueError('Duplicate baseline')
        baselines[split]=keyed
    for label in ('train_s1','train_s05','val_s1'):
        path=args.cache_dir/(label+'.pt')
        if sealed.sha(path)!=cache['files'][path.name]:raise ValueError('Cache changed')
        raw=sealed.torch_load(torch,path);expected,split,scale=sealed.SHARDS[label]
        if raw['identity']!=cache['identity'] or raw['detector_state']!=cache['detector_state'] or len(raw['records'])!=expected:raise ValueError('Cache provenance/count drift')
        bucket=records.setdefault(split,[])
        if Counter(r['sequence'] for r in raw['records'])!=Counter(sealed.TRAIN_COUNTS if split=='train' else sealed.VAL_COUNTS):raise ValueError('Video counts differ')
        for row in raw['records']:
            src=baselines[split][(row['image'],row['scale'])];n=len(row['boxes_original'])
            if (row['boxes_original'][0].tolist() if n else None)!=src['b'] or row['gt_original'][0].tolist()!=src['gt']:raise ValueError('Cache B/GT baseline drift')
            if any(row[k]!=src[k] for k in ('image','sequence','domain','frame_id','split')):raise ValueError('Cache identity differs')
            m=torch.tensor([src['midpoint']],dtype=row['boxes_original'].dtype) if n else row['boxes_original'].clone()
            bucket.append(dict(row,shard=label,midpoint_original=m))
        del raw
    train=sorted(records['train'],key=lambda r:(r['scale'],r['image']));val=sorted(records['val'],key=lambda r:r['image'])
    return train,val,cache,baselines


def update(h,opt,packed,indices,arm,torch,keep=False):
    ts=[x[indices] for x in packed['tensors']];h.train();opt.zero_grad()
    q,t=model.proposals(ts[-2],ts[-1],ts[3],arm);out=model.readout(h,*ts[:-1],q)
    if keep:out['delta_roi'].retain_grad()
    terms=model.loss(out,t);terms['total'].backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in h.parameters()):raise ValueError('Invalid gradients')
    gradients={n:p.grad.detach().clone() for n,p in h.named_parameters()} if keep else None
    norm=float(torch.nn.utils.clip_grad_norm_(h.parameters(),10.));opt.step()
    if any(not bool(torch.isfinite(p).all()) for p in h.parameters()):raise ValueError('Nonfinite state')
    return dict(loss=float(terms['total'].detach()),grad_norm=norm,slot_losses=terms['pointwise'].detach().mean(0).mean(-1).cpu().tolist(),gradients=gradients)


def inventory(h,packed,records,torch,arm,recovery=False):
    result={};h.eval()
    with torch.no_grad():
        for start in range(0,len(packed['indices']),32):
            indices=list(range(start,min(start+32,len(packed['indices']))));ts=[x[indices] for x in packed['tensors']]
            q,t=model.proposals(ts[-2],ts[-1],ts[3],arm);out=model.readout(h,*ts[:-1],q)
            if arm=='proposal7':
                long_roi=((ts[-2][:,2]<ts[-2][:,3]) ^ (ts[3][:,2]<ts[3][:,3])).long()
                ids=torch.arange(len(indices),device=q.device)
                if not torch.equal(t[ids,5,long_roi],torch.zeros(len(indices),device=q.device)) or not torch.equal(t[ids,6,1-long_roi],torch.zeros(len(indices),device=q.device)):raise ValueError('GT-exact edge target not zero')
            assigned=model.base.distribution_targets(t.reshape(-1,2),out['bucket_logits'].reshape(-1,2,33),out['grid'])
            cov=out['border_support'].reshape(-1,2,33,2)
            l=cov.gather(2,assigned['left'][:,:,None,None].expand(-1,-1,1,2)).squeeze(2)
            r=cov.gather(2,assigned['right'][:,:,None,None].expand(-1,-1,1,2)).squeeze(2)
            support=(l*assigned['weight_left'][...,None]+r*assigned['weight_right'][...,None]).reshape(len(indices),7,2,2)
            if not recovery and not torch.equal(out['delta_roi'],torch.zeros_like(t)):raise ValueError('Not neutral')
            delivered=model.delivery(out,*ts[2:-1],q)
            before=(torch.exp(-t)-1).abs();after=(torch.exp(out['delta_roi']-t)-1).abs()
            # Map ROI errors back to natural canonical long/short for reporting.
            swap=(ts[3][:,2]<ts[3][:,3]) ^ (ts[-2][:,2]<ts[-2][:,3])
            before=torch.where(swap[:,None,None],before.flip(-1),before);after=torch.where(swap[:,None,None],after.flip(-1),after)
            if not recovery:
                natural=h(*ts[:-1]);wrapped=model.readout(h,*ts[:-1],ts[-2][:,None])
                if not torch.equal(natural['delta_roi'],wrapped['delta_roi'][:,0]):raise ValueError('Natural wrapper differs')
            for pos,packed_i in enumerate(indices):
                rec=records[packed['indices'][packed_i]]
                for slot,label in enumerate(model.SLOTS):
                    cell=result.setdefault(rec['domain']+'/'+str(rec['scale'])+'/'+label,dict(count=0,target_negative=[0,0],target_positive=[0,0],target_zero=[0,0],before_abs_sum=[0.,0.],after_abs_sum=[0.,0.],improved=[0,0],harmed=[0,0],support_sum=0.,zero_support=0,target_abs_max=0.,axis_cross=0,fallback=0))
                    rawt=t[pos,slot].cpu().tolist();a=before[pos,slot].cpu().tolist();b=after[pos,slot].cpu().tolist()
                    cell['count']+=1;cell['target_abs_max']=max(cell['target_abs_max'],max(map(abs,rawt)))
                    cell['axis_cross']+=int(bool((q[pos,slot,2]<q[pos,slot,3])!=(ts[-2][pos,2]<ts[-2][pos,3])))
                    for j in (0,1):
                        cell['target_negative'][j]+=int(rawt[j]<0);cell['target_positive'][j]+=int(rawt[j]>0);cell['target_zero'][j]+=int(rawt[j]==0)
                        cell['before_abs_sum'][j]+=a[j];cell['after_abs_sum'][j]+=b[j];cell['improved'][j]+=int(b[j]<a[j]-1e-12);cell['harmed'][j]+=int(b[j]>a[j]+1e-12)
                    sp=support[pos,slot];cell['support_sum']+=float(sp.sum());cell['zero_support']+=int((sp==0).sum())
                    status=pos*7+slot;cell['fallback']+=int(not bool(delivered['accepted'][status]) and not bool(delivered['neutral_bypass'][status]))
    return result


def save(path,h,opt,epoch,arm,torch,proof):
    rng=dict(python=random.getstate(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    payload=dict(protocol=VERSION,arm=arm,epoch=epoch,updates=epoch*639,proof=proof,
        head_state={k:v.detach().cpu().clone() for k,v in h.state_dict().items()},optimizer_state=deepcopy(opt.state_dict()),rng=rng,head_digest=sealed.state_digest(h))
    with path.open('xb') as f:torch.save(payload,f)
    got=sealed.torch_load(torch,path);reload=model.GeometryConditionSizeHead().to(next(h.parameters()).device);reload.load_state_dict(got['head_state'])
    if sealed.state_digest(reload)!=payload['head_digest']:raise ValueError('Checkpoint reload drift')
    op=torch.optim.Adam(reload.parameters(),lr=.001);op.load_state_dict(got['optimizer_state'])
    for a,b in zip(opt.state_dict()['state'].values(),op.state_dict()['state'].values()):
        for k in a:
            if torch.is_tensor(a[k]) and not torch.equal(a[k].cpu(),b[k].cpu()):raise ValueError('Adam reload drift')
    if rng['python']!=got['rng']['python'] or not torch.equal(rng['cpu'],got['rng']['cpu']) or any(not torch.equal(a,b) for a,b in zip(rng['cuda'],got['rng']['cuda'])):raise ValueError('RNG reload drift')
    random.setstate(rng['python']);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
    return dict(sha256=sealed.sha(path),head_digest=payload['head_digest'])


def run(args):
    sources=check_sources();out=args.out_dir.resolve()
    if out.parent!=(ROOT/'work_dirs'/VERSION).resolve() or out.exists():raise ValueError('Fresh result path required')
    out.mkdir(parents=True);started=time.monotonic();report=dict(status='FAILED',protocol=VERSION,**protocol()['limits'])
    used={};samples=[]
    try:
        import torch
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='3' or os.environ.get('WORLD_SIZE','1')!='1':raise ValueError('Use fourth physical card only')
        torch.set_num_threads(1);torch.cuda.set_device(0);device=torch.device('cuda:0');samples.append(memory())
        torch.cuda.set_per_process_memory_fraction(1100/(torch.cuda.get_device_properties(0).total_memory/2**20),0)
        old.seed_all(torch)
        for path,pin in [(args.b_weight,old.B_SHA),(args.m_weight,sealed.FIXED['sha256'])]:
            if sealed.sha(path)!=pin:raise ValueError('B/M weight SHA drift')
        origin=sealed.read(checked_file(args.i_run,'input_runtime_proof.json',used))
        finished=sealed.read(checked_file(args.i_run,'completion.json',used))
        if origin['source_identity']!=sources['parent'] or origin['cache_manifest_sha256']!=sealed.CACHE_SHA or finished['formal_head_updates']!=15336 or finished['candidate']['selected']['epoch']!=7:raise ValueError('Historical I provenance differs')
        train,val,cache,baselines=load_cache(args,torch,used);tp=old.pack(train,device,torch);vp=old.pack(val,device,torch)
        if [len(train),len(val),len(tp['indices']),len(vp['indices'])]!=[5116,887,5112,886]:raise ValueError('Full view counts differ')
        irows=old.read_rows(checked_file(args.i_run,'candidate/val_epoch_07.rows.jsonl.gz',used))
        ref=sealed.read(checked_file(args.i_run,'candidate/val_epoch_07.summary.json',used))
        pin='affe4795b73cae325f878b15c2d81f942461bd965bae0b3c3ca6a7a944eb0464'
        if sealed.sha(checked_file(args.i_run,'candidate/head_epoch_07.pth',used))!=pin:raise ValueError('I07 pin drift')
        schedule=[old.epoch_batches(5112,e) for e in range(1,25)]
        prior=sealed.read(checked_file(args.i_run,'schedule.json',used))
        identities=[dict(image=train[i]['image'],scale=train[i]['scale']) for i in tp['indices']]
        if schedule!=prior['batches'] or identities!=prior['train_view_identity']:raise ValueError('Schedule differs from full TRAIN')
        runtime=dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),physical_gpu=3,tf32_matmul=getattr(torch.backends.cuda.matmul,'allow_tf32',None),tf32_cudnn=getattr(torch.backends.cudnn,'allow_tf32',None))
        proof=dict(sources=sources,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),runtime=runtime,cache_manifest_sha256=sealed.CACHE_SHA,shards=cache['files'],B_sha256=old.B_SHA,M_sha256=sealed.FIXED['sha256'],I07_sha256=pin,input_files=used)
        sealed.write(out/'protocol.json',protocol());sealed.write(out/'input_runtime_proof.json',proof);sealed.write(out/'schedule.json',dict(train_view_identity=identities,batches=schedule))
        old.seed_all(torch);h=model.GeometryConditionSizeHead().to(device);initial=sealed.state_digest(h)
        bt,bv=old.evaluate_rows(h,train,tp,torch),old.evaluate_rows(h,val,vp,torch)
        if any(r['midpoint']!=r['edge_residual'] for r in bt+bv):raise ValueError('Neutral natural identity drift')
        for split,rs in [('train',bt),('val',bv)]:
            condition.val_replay.compare_numeric(rs,list(baselines[split].values()))
            old.write_rows(out/('baseline_'+split+'.rows.jsonl.gz'),rs)
        riou_cache={};baseval=metrics.summary(bv,riou_cache)
        # Baseline and I07 reference arithmetic verified without any new choices.
        computed=metrics.summary(irows,riou_cache);condition.val_replay.compare_numeric(computed,ref)
        inv={arm:inventory(h,tp,train,torch,arm) for arm in ARMS};sealed.write(out/'proposal_inventory.json',inv)
        # Real two-step comparison: exact state reuse is NOT assumed.
        old.seed_all(torch);oh=model.GeometryConditionSizeHead().to(device);nh=deepcopy(oh);oo=torch.optim.Adam(oh.parameters(),lr=.001);no=torch.optim.Adam(nh.parameters(),lr=.001);equivalence=[]
        for batch in schedule[0][:2]:
            a=old.update(oh,oo,tp,batch,'decoded_control',torch);b=update(nh,no,tp,batch,'repeat7',torch,keep=True)
            gradient_difference={name:float((p.grad-b['gradients'][name]*min(1.,10./(b['grad_norm']+1e-6))).abs().max()) for name,p in oh.named_parameters()}
            if max(gradient_difference.values())>1e-6:raise ValueError('Repeated-slot gradient inconsistency')
            states={name:float((p-dict(nh.named_parameters())[name]).abs().max()) for name,p in oh.named_parameters()}
            with torch.no_grad():
                xs=[v[batch] for v in tp['tensors'][:-1]];delta=float((oh(*xs)['delta_roi']-nh(*xs)['delta_roi']).abs().max())
            if abs(a['loss']-b['loss'])>1e-6 or delta>1e-5:raise ValueError('Repeated-slot function/update inconsistency')
            equivalence.append(dict(original_loss=a['loss'],repeated_loss=b['loss'],state_max_abs=states,gradient_max_abs=gradient_difference,delta_max_abs=delta,full_state_exact=all(x==0 for x in states.values())))
        # Discard candidate smoke and verify serialization before formal training.
        old.seed_all(torch);smoke=model.GeometryConditionSizeHead().to(device);op=torch.optim.Adam(smoke.parameters(),lr=.001)
        for batch in schedule[0][:2]:update(smoke,op,tp,batch,'proposal7',torch)
        save(out/'engineering_smoke.pth',smoke,op,0,'proposal7',torch,proof)
        sealed.write(out/'engineering.json',dict(status='ENGINEERING_PASS',initial_state=initial,full_neutral_natural_replay=True,repeat7_two_updates=equivalence,control_reuse=False,discarded_head_updates=6,proposal_inventory_complete=True))
        del h,oh,nh,smoke,oo,no,op;torch.cuda.empty_cache();samples.append(memory())
        print('ENGINEERING_PASS BOTH_ARMS_FRESH_24',flush=True)
        selections={}
        for arm in ARMS:
            old.seed_all(torch);h=model.GeometryConditionSizeHead().to(device)
            if sealed.state_digest(h)!=initial:raise ValueError('Formal init drift')
            opt=torch.optim.Adam(h.parameters(),lr=.001,weight_decay=0.);dest=out/arm;dest.mkdir();results=[]
            with (dest/'progress.jsonl').open('x') as log:
                for epoch,batches in enumerate(schedule,1):
                    tick=time.monotonic();losses=[];slot_losses=[0.]*7
                    for step,batch in enumerate(batches,1):
                        event=update(h,opt,tp,batch,arm,torch);event.pop('gradients');losses.append(event['loss'])
                        slot_losses=[a+b for a,b in zip(slot_losses,event['slot_losses'])]
                        log.write(json.dumps(dict(epoch=epoch,step=step,loss=event['loss'],grad_norm=event['grad_norm']))+'\n')
                        if step%100==0 or step==639:samples.append(memory());log.flush();print(arm,epoch,step,event['loss'],flush=True)
                    tr=old.evaluate_rows(h,train,tp,torch);trv=metrics.train_report(tr)
                    old.write_rows(dest/('train_epoch_%02d.rows.jsonl.gz'%epoch),tr);sealed.write(dest/('train_epoch_%02d.summary.json'%epoch),trv)
                    ck=save(dest/('head_epoch_%02d.pth'%epoch),h,opt,epoch,arm,torch,proof)
                    rows=old.evaluate_rows(h,val,vp,torch);value=metrics.summary(rows,riou_cache);nr=near(rows)
                    old.write_rows(dest/('val_epoch_%02d.rows.jsonl.gz'%epoch),rows)
                    sealed.write(dest/('val_epoch_%02d.summary.json'%epoch),value);sealed.write(dest/('val_epoch_%02d.components.json'%epoch),condition.components(rows));sealed.write(dest/('val_epoch_%02d.near.json'%epoch),nr)
                    event=dict(epoch=epoch,checkpoint=ck,selection_key=key(value,epoch),adoption=adoption(value,nr,ref),train_pass_diagnostic_only=trv['train_pass'],complete_val_frames=887,mean_loss=sum(losses)/len(losses),slot_mean_losses=[x/639 for x in slot_losses],elapsed_seconds=time.monotonic()-tick)
                    results.append(event);sealed.write(dest/('epoch_%02d.json'%epoch),event)
                    print('EPOCH_COMPLETE',arm,epoch,event['selection_key'],event['adoption']['pass_all'],'seconds',event['elapsed_seconds'],flush=True)
            selections[arm]=condition.choose(results);sealed.write(dest/'selection.json',selections[arm])
            chosen=selections[arm]['selected'] or selections[arm]['representative'];payload=sealed.torch_load(torch,dest/('head_epoch_%02d.pth'%chosen['epoch']));h.load_state_dict(payload['head_state'])
            sealed.write(dest/'chosen_proposal_recovery.json',inventory(h,tp,train,torch,arm,True))
            del h,opt,payload;torch.cuda.empty_cache()
        chosen={a:selections[a]['selected'] or selections[a]['representative'] for a in ARMS};values={a:sealed.read(out/a/('val_epoch_%02d.summary.json'%chosen[a]['epoch'])) for a in ARMS};nears={a:sealed.read(out/a/('val_epoch_%02d.near.json'%chosen[a]['epoch'])) for a in ARMS}
        mechanism={}
        for d in ('real','sim'):
            a,b=values['proposal7'][d]['edge_residual'],values['repeat7'][d]['edge_residual']
            for edge in ('long','short'):
                x,y=a['sizes']['edges'][edge]['mae'],b['sizes']['edges'][edge]['mae'];mechanism[d+'/'+edge]=x<y-1e-6 if d=='sim' else x<=y+1e-6
                mechanism[d+'/near_'+edge]=nears['proposal7'][d][edge]['methods']['edge_residual']['mae']<=nears['repeat7'][d][edge]['methods']['edge_residual']['mae']+1e-6
            mechanism[d+'/riou']=a['mean_riou_full_frame']>b['mean_riou_full_frame']+1e-5 if d=='sim' else a['mean_riou_full_frame']>=b['mean_riou_full_frame']-1e-5
        if check_sources()!=sources:raise ValueError('End source drift')
        for name,sha in cache['files'].items():
            if sealed.sha(args.cache_dir/name)!=sha:raise ValueError('End cache drift')
        for path,pin in [(args.b_weight,old.B_SHA),(args.m_weight,sealed.FIXED['sha256'])]:
            if sealed.sha(path)!=pin:raise ValueError('End B/M drift')
        report.update(status='COMPLETE_REVIEW_REQUIRED' if selections['proposal7']['selected'] and all(mechanism.values()) else 'COMPLETE_NO_ADOPTED_VERSION_KEEP_M',selections=selections,mechanism=mechanism,formal_updates_per_arm=15336,formal_updates_total=30672,engineering_updates_discarded=6,complete_val_epochs_per_arm=24,baseline=baseval,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    except BaseException as e:
        report.update(error=repr(e),status='RESOURCE_STOP' if 'RESOURCE_STOP' in str(e) else 'FAILED');raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,memory_samples=samples,finished_utc=datetime.now(timezone.utc).isoformat())
        sealed.write(out/'completion.json',report)
        sealed.write(out/'artifacts.json',dict(files={str(p.relative_to(out)):sealed.sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifacts.json'}))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('out-dir','cache-dir','i-run','b-weight','m-weight'):p.add_argument('--'+n,type=Path,required=True)
    return p

if __name__=='__main__':run(parser().parse_args())
