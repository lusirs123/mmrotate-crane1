"""CPU-only frozen LVSR feature/gradient audit. No optimizer or TEST reads."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
KEYS=('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original','gt_original')
SPEC=dict(version='lvsr_direction_representation_cpu_20261112_v1',CPU_threads=1,GPU=False,
    heads=['L5','RMS5'],TEST=False,training_updates=0,optimizer=False,
    feature_views=5999,feature_batch=8,nearest_k=5,nearest_metric='cosine of sqrt(common W)-weighted flattened maps',
    feature_stages=['visual256','geometry11','joint267','first_linear8','first_relu8','last_relu8'],
    neighbor_pools=['cross_video','same_domain_cross_video','same_video_gap30'],
    reference='TRAIN same scale only; VAL uses native TRAIN; exclude same image; no fitted probe',
    direction='GT/M >1.01 enlarge, <0.99 shrink, otherwise near; separate fixed-M error<=3% overlaps',
    gradient_sampling='up to32 views per TRAIN domain x scale x short-target-direction; evenly spaced within videos',
    gradient_weighting='population stratum mass / sampled stratum count; estimator, not actual full TRAIN gradient',
    natural_batch='first8 of unchanged epoch6 schedule after saved epoch5 state; CPU counterfactual, no update',
    response='Jacobian times proposed direction is first-order log-size response, not a new prediction or full trajectory',
    replay_atol=1e-4,replay_rtol=1e-6,CPU_budget_seconds=1800,
    download='core summaries only; no weights, features, caches, Jacobians or per-frame rows')


def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_digest(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def records_digest(records):
    h=hashlib.sha256()
    for r in records:
        h.update((r['image']+str(r['scale'])).encode())
        for k in KEYS:h.update(r[k].numpy().tobytes())
    return h.hexdigest()
def rows(path):
    with gzip.open(path,'rt') as f:return {(r['image'],r['scale']):r for r in map(json.loads,f)}


def direction(t):
    import torch
    return torch.where(t<math.log(.99),torch.zeros_like(t,dtype=torch.long),
        torch.where(t>math.log(1.01),torch.full_like(t,2,dtype=torch.long),torch.ones_like(t,dtype=torch.long)))


def canonical(value,bm,m):
    import torch
    swap=(bm[:,2]<bm[:,3])^(m[:,2]<m[:,3])
    return torch.where(swap[:,None],value.flip(-1),value)


def batch(rr):
    import torch
    return [torch.cat([r[k] for r in rr]) for k in KEYS]


def metadata(rr):
    import torch
    result=[]
    for r in rr:
        g=r['gt_original'][0,2:4].sort(descending=True)[0].double()
        m=r['midpoint_original'][0,2:4].sort(descending=True)[0].double()
        result.append(dict(image=r['image'],sequence=r['sequence'],frame_id=r['frame_id'],domain=r['domain'],
            scale=r['scale'],split=r['split'],target=(g/m).log().tolist(),M_error=(m/g-1).abs().tolist()))
    return result


def balanced_sample(rr,per_cell=32):
    import torch
    cells=defaultdict(list)
    mm=metadata(rr)
    for i,r in enumerate(mm):cells[(r['domain'],r['scale'],int(direction(torch.tensor(r['target'][1]))))].append(i)
    chosen=[];weights=[];counts=[]
    for key,ii in sorted(cells.items()):
        byseq=defaultdict(list)
        for i in ii:byseq[mm[i]['sequence']].append(i)
        quota=min(per_cell,len(ii));allocation={q:0 for q in sorted(byseq)}
        while sum(allocation.values())<quota:
            for q in allocation:
                if allocation[q]<len(byseq[q]) and sum(allocation.values())<quota:allocation[q]+=1
        pick=[]
        for q,k in allocation.items():
            pool=sorted(byseq[q],key=lambda i:mm[i]['frame_id'])
            pick.extend(pool[round(j*(len(pool)-1)/(k-1))] for j in range(k)) if k>1 else pick.extend(pool[:k])
        assert len(pick)==len(set(pick))==quota
        chosen.extend(pick);weights.extend([len(ii)/quota]*quota)
        counts.append(dict(domain=key[0],scale=key[1],short_direction=['shrink','near','enlarge'][key[2]],population=len(ii),sample=quota))
    return chosen,weights,counts


def adam_direction(parameters,gradient,opt_state,drop_old_first_moment=False):
    """Functional next Adam direction; no model/optimizer mutation."""
    import torch
    group=opt_state['param_groups'][0]
    assert len(opt_state['param_groups'])==1 and group['weight_decay']==0 and not group.get('amsgrad',False)
    norm=gradient.norm();clip=min(1.,10./(float(norm)+1e-6));g=gradient*clip
    beta1,beta2=group['betas'];out=[];offset=0
    for p,i in zip(parameters,group['params']):
        state=opt_state['state'][i];n=p.numel();gg=g[offset:offset+n].reshape(p.shape).to(p.dtype);offset+=n
        old=state['exp_avg'];old2=state['exp_avg_sq'];step=int(state['step'])+1
        moment=(torch.zeros_like(old) if drop_old_first_moment else old)*beta1+gg*(1-beta1)
        variance=old2*beta2+gg.square()*(1-beta2)
        denom=variance.sqrt()/math.sqrt(1-beta2**step)+group['eps']
        out.append((-group['lr']/(1-beta1**step)*moment/denom).reshape(-1).double())
    assert offset==len(gradient)
    return torch.cat(out),dict(gradient_norm=float(norm),clip_coefficient=clip,saved_step=int(next(iter(opt_state['state'].values()))['step']))


def extract(head,rr,capture=False):
    import torch
    x=batch(rr);seen={};handles=[]
    if capture:
        def input_hook(module,args):seen['input']=args[0].detach()
        handles.append(head.stem.branches[0][0].register_forward_pre_hook(input_hook))
        for i,arm in enumerate(head.stem.branches):
            for j,label in ((0,'linear'),(1,'relu')):
                def hook(module,args,out,i=i,label=label):seen[label+str(i)]=out.detach()
                handles.append(arm[j].register_forward_hook(hook))
        def stem_hook(module,args,out):seen['last']=out.detach()
        handles.append(head.stem.register_forward_hook(stem_hook))
    try:out=head(*x[:-1])
    finally:
        for h in handles:h.remove()
    pred=canonical(out['delta_roi'],x[3],x[5])
    target=canonical(__import__('crane_project.utils.port_size_qalign_v1',fromlist=['residual']).residual.size_targets(x[-1],x[-2],x[3])['log_residual'],x[3],x[5])
    if capture:
        w=out['common_support'].sqrt()
        maps={'visual256':seen['input'][:,:256],'geometry11':seen['input'][:,256:],'joint267':seen['input'],
            'first_linear8':torch.cat((seen['linear0'],seen['linear1']),1),
            'first_relu8':torch.cat((seen['relu0'],seen['relu1']),1),'last_relu8':seen['last']}
        seen={k:(v*w).flatten(1).float() for k,v in maps.items()}
    return out,pred,target,seen


def score_probe(pred,target):
    import torch
    if not len(target):return dict(n=0)
    c,p=direction(target),direction(pred);conf=torch.zeros(3,3,dtype=torch.long)
    for a in range(3):
        for b in range(3):conf[a,b]=((c==a)&(p==b)).sum()
    support=conf.sum(1);recall=[float(conf[i,i]/support[i]) if support[i] else None for i in range(3)]
    return dict(n=len(target),log_MAE=float((pred-target).abs().mean()),confusion_shrink_near_enlarge=conf.tolist(),
        accuracy=float((c==p).float().mean()),balanced_accuracy=sum(v for v in recall if v is not None)/sum(v is not None for v in recall),
        shrink_recall=recall[0],enlarge_recall=recall[2],class_counts=support.tolist())


def knn(features,meta,progress=None):
    import torch
    # Exact cosine neighbors of full flattened maps, no projection or fitted scaler.
    z=features/features.norm(dim=1,keepdim=True).clamp_min(1e-12);good=features.norm(dim=1)>1e-12
    target=torch.tensor([r['target'] for r in meta],dtype=torch.float64)
    result=[]
    for split,scale in (('train',1.),('train',.5),('val',1.)):
        refs=[i for i,r in enumerate(meta) if r['split']!='val' and r['scale']==scale]
        queries=[i for i,r in enumerate(meta) if (r['split']=='val')==(split=='val') and r['scale']==scale]
        ref_z=z[refs];ref_good=good[refs]
        seqs=sorted({meta[i]['sequence'] for i in refs+queries});seqmap={v:i for i,v in enumerate(seqs)}
        ref_seq=torch.tensor([seqmap[meta[i]['sequence']] for i in refs]);ref_dom=torch.tensor([meta[i]['domain']=='real' for i in refs])
        ref_frame=torch.tensor([meta[i]['frame_id'] for i in refs]);ref_targets=target[refs]
        entries={pool:defaultdict(list) for pool in SPEC['neighbor_pools']}
        for start in range(0,len(queries),64):
            qi=queries[start:start+64];sim=z[qi]@ref_z.T
            for a,i in enumerate(qi):
                if not bool(good[i]):continue
                r=meta[i];other=ref_seq!=seqmap[r['sequence']]
                for pool in SPEC['neighbor_pools']:
                    eligible=other if pool=='cross_video' else other&(ref_dom==(r['domain']=='real')) if pool=='same_domain_cross_video' else (~other)&((ref_frame-r['frame_id']).abs()>30)
                    eligible=eligible&ref_good
                    n=int(eligible.sum())
                    if n<5:continue
                    ranks=sim[a].masked_fill(~eligible,-float('inf')).topk(5).indices
                    # References never include any frame from a held-out query video.
                    if pool!='same_video_gap30':assert all(meta[refs[int(j)]]['sequence']!=r['sequence'] for j in ranks)
                    value=(i,ref_targets[ranks].mean(0),ref_targets[eligible].mean(0),float(sim[a,ranks[0]]),
                        float((ref_targets[ranks]-target[i]).abs().mean()))
                    entries[pool][r['sequence']].append(value)
            if progress is not None and start%512==0:progress(split=split,scale=scale,done=start+len(qi),total=len(queries))
        for pool,byvideo in entries.items():
            groups=dict(byvideo)
            all_values=[v for vv in byvideo.values() for v in vv];groups['overall']=all_values
            for domain in ('real','sim'):groups[domain]=[v for v in all_values if meta[v[0]]['domain']==domain]
            for group,vv in groups.items():
                if not vv:
                    result.append(dict(split=split,scale=scale,pool=pool,group=group,n=0));continue
                indices=[v[0] for v in vv];prediction=torch.stack([v[1] for v in vv]);prior=torch.stack([v[2] for v in vv])
                scores={e:score_probe(prediction[:,j],target[indices,j]) for j,e in enumerate(('long','short'))}
                priors={e:score_probe(prior[:,j],target[indices,j]) for j,e in enumerate(('long','short'))}
                result.append(dict(split=split,scale=scale,pool=pool,group=group,n=len(vv),edges=scores,reference_prior=priors,
                    mean_top1_cosine=sum(v[3] for v in vv)/len(vv),mean_neighbor_target_log_L1=sum(v[4] for v in vv)/len(vv)))
    return dict(zero_feature_views=int((~good).sum()),metrics=result,
        interpretation='nearest-neighbor probe only; no proof of information loss or collapse from a lower probe score')


def jacobians(head,rr,verify_first=16,progress=None):
    import torch
    params=list(head.parameters());js=[];preds=[];targets=[];max_rebuild=0.
    for i,r in enumerate(rr):
        out,d,t,_=extract(head,[r]);j=[]
        for edge in range(2):
            grad=torch.autograd.grad(d[0,edge],params,retain_graph=True)
            j.append(torch.cat([v.reshape(-1) for v in grad]).double())
        jj=torch.stack(j);deriv=((d[0]-t[0])/.1).clamp(-1,1).double()
        if i<verify_first:
            loss=torch.nn.functional.smooth_l1_loss(d,t,beta=.1)
            direct=torch.cat([v.reshape(-1) for v in torch.autograd.grad(loss,params)]).double()
            reconstructed=(jj*deriv[:,None]).sum(0)/2
            max_rebuild=max(max_rebuild,float((direct-reconstructed).abs().max()))
            if not torch.allclose(direct,reconstructed,atol=1e-6,rtol=1e-5):raise ValueError('Jacobian/loss gradient reconstruction differs')
        js.append(jj);preds.append(d.detach()[0].double());targets.append(t.detach()[0].double())
        if progress is not None and (i%64==0 or i==len(rr)-1):progress(done=i+1,total=len(rr))
    return torch.stack(js),torch.stack(preds),torch.stack(targets),max_rebuild


def gradient_report(head,payload,rr,weights,progress=None):
    import torch
    j,d,t,rebuild=jacobians(head,rr,progress=progress);w=torch.tensor(weights,dtype=torch.float64)
    derivative=((d-t)/.1).clamp(-1,1);parts=j*derivative[:,:,None]*w[:,None,None]/(2*w.sum())
    total=parts.sum((0,1));cls=direction(t);mm=metadata(rr)
    layer_masks={};offset=0
    for name,p in head.named_parameters():
        kind='stem' if name.startswith('stem') else 'terminal_bias' if name.endswith('bias') else 'terminal_weight'
        layer_masks.setdefault(kind,[]).extend(range(offset,offset+p.numel()));offset+=p.numel()
    groups={};vectors={}
    for edge,label in enumerate(('long','short')):
        for k,suffix in enumerate(('shrink','near','enlarge')):
            mask=cls[:,edge]==k;key=label+'_'+suffix
            vectors[key]=parts[mask,edge].sum(0)
            groups[key]=dict(sample_n=int(mask.sum()),estimated_population_mass=float(w[mask].sum()),
                mean_pred_log=float((d[mask,edge]*w[mask]).sum()/w[mask].sum()) if mask.any() else None,
                mean_target_log=float((t[mask,edge]*w[mask]).sum()/w[mask].sum()) if mask.any() else None)
        near=torch.tensor([m['M_error'][edge]<=.03 for m in mm]);key=label+'_accurate_M_le3pct'
        vectors[key]=parts[near,edge].sum(0);groups[key]=dict(sample_n=int(near.sum()),estimated_population_mass=float(w[near].sum()))
    reconstructed=sum(vectors[e+'_'+s] for e in ('long','short') for s in ('shrink','near','enlarge'))
    assert torch.allclose(total,reconstructed,atol=1e-10,rtol=1e-10)
    def cos(a,b):return float(a.dot(b)/(a.norm()*b.norm())) if a.norm()>0 and b.norm()>0 else None
    for key,g in vectors.items():
        groups[key].update(contribution_norm=float(g.norm()),cosine_with_total=cos(g,total),
            inner_product_with_total=float(g.dot(total)),layer_contribution_norm={k:float(g[v].norm()) for k,v in layer_masks.items()})
    pair_cos={a+'/'+b:cos(vectors[a],vectors[b]) for a,b in (('long_shrink','long_enlarge'),('short_shrink','short_enlarge'),
        ('long_accurate_M_le3pct','short_enlarge'),('short_accurate_M_le3pct','long_enlarge'))}
    adam,adam_meta=adam_direction(list(head.parameters()),total,payload['optimizer_state'])
    no_old_m,_=adam_direction(list(head.parameters()),total,payload['optimizer_state'],True)
    directions={'SGD_same_lr':-payload['optimizer_state']['param_groups'][0]['lr']*total,
        'Adam_saved_state':adam,'Adam_drop_old_first_moment':no_old_m}
    responses={}
    for name,step in directions.items():
        delta=(j*step[None,None]).sum(-1);cells={}
        for edge,label in enumerate(('long','short')):
            masks={s:cls[:,edge]==k for k,s in enumerate(('shrink','near','enlarge'))}
            masks['accurate_M_le3pct']=torch.tensor([m['M_error'][edge]<=.03 for m in mm])
            for suffix,mask in masks.items():
                mass=w[mask].sum();response=delta[mask,edge];dw=w[mask]
                cells[label+'_'+suffix]=dict(n=int(mask.sum()),estimated_population_mass=float(mass),
                    mean_first_order_delta=float((response*dw).sum()/mass) if mass>0 else None,
                    mass_shrink_direction=float(dw[response< -1e-12].sum()/mass) if mass>0 else None,
                    mass_local_loss_improved=float(dw[(derivative[mask,edge]*response)< -1e-12].sum()/mass) if mass>0 else None,
                    group_dot_step=float(vectors[label+'_'+suffix].dot(step)))
        responses[name]=dict(step_norm=float(step.norm()),groups=cells)
    return dict(n=len(rr),weight_mass=float(w.sum()),loss_gradient_reconstruction_max_abs=rebuild,
        total_gradient_norm=float(total.norm()),groups=groups,group_cosines=pair_cos,adam=adam_meta,responses=responses,
        derivative_signed_mean=(derivative*w[:,None]).sum(0).div(w.sum()).tolist(),
        neutral_output_derivative_signed_mean=((-t/.1).clamp(-1,1)*w[:,None]).sum(0).div(w.sum()).tolist())


def run(args):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(1)
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    from crane_project.tools import train_symeood_r4_lvsr_pair_v1 as trainer
    from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
    from crane_project.utils.port_size_pair_roi_rms_v1 import PairROIRMSSizeHead
    from crane_project.tools import train_lvsr_roi_rms_v1 as rms_training
    assert not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES']==''
    identity=p.checked_sources();out=ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/roi_rms_001'
    prefix='direction_repr_pilot' if args.stage=='pilot' else 'direction_repr'
    sources={'crane_project/tools/diagnose_lvsr_direction_representation_v1.py':sha(__file__),
        'tests/test_lvsr_direction_representation_v1.py':sha(ROOT/'tests/test_lvsr_direction_representation_v1.py')}
    if args.stage=='run':
        pilot=read(out/'direction_repr_pilot_completion.json')
        assert pilot['passed'] and pilot['elapsed_seconds']<120 and pilot['sources']==sources
    save(out/(prefix+'_protocol.json'),dict(settings=SPEC,stage=args.stage,sources=sources,parent_identity=identity))
    start=time.monotonic();status_file=out/(prefix+'_status.json')
    def status(stage,**extra):
        if time.monotonic()-start>SPEC['CPU_budget_seconds']:raise RuntimeError('CPU diagnostic budget exceeded')
        value=dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.monotonic()-start,**extra)
        tmp=status_file.with_suffix('.tmp');tmp.write_text(json.dumps(value));tmp.replace(status_file)
        print(json.dumps(value),flush=True)
    original_optimizer=torch.optim.Optimizer.__init__
    def forbidden(*a,**kw):raise RuntimeError('No optimizer or parameter updates in diagnostic')
    torch.optim.Optimizer.__init__=forbidden
    try:
        train,val,inputs=trainer.load_inputs(p.paths(),identity,type('Engine',(),{'torch':torch})())
        records=train+val;cache_before=records_digest(records)
        train_present=[r for r in train if len(r['midpoint_original'])]
        present=train_present+[r for r in val if len(r['midpoint_original'])]
        assert len(present)==5999 and len(train_present)==5112
        indices,weights,strata=balanced_sample(train_present)
        if args.stage=='pilot':
            pick=[round(j*(len(indices)-1)/15) for j in range(16)]
            indices=[indices[j] for j in pick];weights=[weights[j] for j in pick]
        probe=[train_present[i] for i in indices]
        specifications=[('L5',p.BASE/'lvsr/attempt_001/train/head_epoch_05.pth',PairSpecificSizeHead,
            ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/attempt_001/L5_all.rows.jsonl.gz',
            'b272c50dfe6329414803d5f22c6e1e84d9c3db37b6bb2e31f1c7ed140f662c3e'),
            ('RMS5',out/'head_epoch_05.pth',PairROIRMSSizeHead,out/'representative_TRAIN_VAL.rows.jsonl.gz',
            '1f821419f49201fb5970feb0829dc62a6b0458cff9f373d5abc815d7cbb3ffbd')]
        parent_pins=[ROOT/p.parent.R4_CHECKPOINT,p.midpoint_directory()/'train/head_epoch_20.pth',
            p.paths()['inputs']/'train.pt',p.paths()['inputs']/'val.pt']
        parent_sha={str(v.relative_to(ROOT)):p.sha(v) for v in parent_pins}
        assert parent_sha[str(parent_pins[0].relative_to(ROOT))]==p.parent.R4_SHA
        assert p.sha(parent_pins[1])==p.MIDPOINT_SHA
        assert rms_training.source_hashes()==read(out/'core_completion.json')['source_SHA']
        result={};proof={};checkpoints={};forwards=0
        for label,path,cls,refpath,expected in specifications:
            assert p.sha(path)==expected;checkpoints[label]=dict(path=str(path.relative_to(ROOT)),sha256=expected)
            payload=torch.load(str(path),map_location='cpu');assert payload['epoch']==5 and payload['updates']==3195
            head=cls().cpu().eval();head.load_state_dict(payload['head_state'],strict=True)
            before=tensor_digest(head.state_dict());reference=rows(refpath)
            chosen=present if args.stage=='run' else probe
            features=defaultdict(list);observed=[];max_box=0.;max_delta=0.
            status('features',head=label,views=len(chosen))
            with torch.no_grad():
                for start_index in range(0,len(chosen),8):
                    rr=chosen[start_index:start_index+8];o,d,t,ff=extract(head,rr,True);forwards+=len(rr)
                    for k,v in ff.items():features[k].append(v)
                    observed.extend(metadata(rr))
                    for i,r in enumerate(rr):
                        ref=reference[r['image'],r['scale']];actual=o['boxes_original'][i]
                        expected_box=actual.new_tensor(ref['edge_residual'])
                        max_box=max(max_box,float((actual-expected_box).abs().max()))
                        assert torch.allclose(actual,expected_box,atol=SPEC['replay_atol'],rtol=SPEC['replay_rtol'])
                        assert torch.equal(actual[[0,1,4,5]],r['midpoint_original'][0,[0,1,4,5]])
                        exp_delta=canonical(actual.new_tensor([ref['delta_roi']]),r['boxes_model'],r['midpoint_original'])[0]
                        max_delta=max(max_delta,float((d[i]-exp_delta).abs().max()))
                        assert torch.allclose(d[i],exp_delta,atol=1e-5,rtol=1e-5)
                        assert torch.allclose(t[i].double(),torch.tensor(observed[-len(rr)+i]['target'],dtype=torch.float64),atol=2e-7,rtol=1e-5)
                    if start_index%512==0:status('features',head=label,done=start_index+len(rr),total=len(chosen))
            feature_result={}
            if args.stage=='run':
                for stage,chunks in features.items():
                    status('neighbors',head=label,feature=stage)
                    feature_result[stage]=knn(torch.cat(chunks),observed,lambda **kw:status('neighbors',head=label,feature=stage,**kw))
            del features
            status('gradient_probe',head=label,views=len(probe))
            grad=gradient_report(head,payload,probe,weights,lambda **kw:status('gradient_probe',head=label,**kw));forwards+=len(probe)
            order=list(range(len(train_present)));random.Random(1703+6).shuffle(order)
            natural=[train_present[i] for i in order[:8]]
            status('natural_epoch6_batch',head=label)
            natural_grad=gradient_report(head,payload,natural,[1.]*8);forwards+=8
            # No in-place parameter updates, .grad assignments, upstream forwards or CUDA use.
            assert before==tensor_digest(head.state_dict()) and all(q.grad is None for q in head.parameters())
            assert p.sha(path)==expected
            result[label]=dict(features=feature_result,gradient_population_estimator=grad,natural_next_batch=natural_grad)
            proof[label]=dict(state_unchanged=True,parameter_grad_buffers_empty=True,CPU_replay_max_abs_box=max_box,CPU_replay_max_abs_delta=max_delta,
                replay_views=len(chosen),saved_optimizer_step=3195,gradient_probe_views=len(probe))
            del payload,head
        assert records_digest(records)==cache_before and p.checked_sources()==identity
        assert all(p.sha(ROOT/n)==h for n,h in parent_sha.items())
        assert sources['crane_project/tools/diagnose_lvsr_direction_representation_v1.py']==sha(__file__)
        assert not torch.cuda.is_initialized()
        summary=dict(protocol=SPEC,stage=args.stage,results=result,verification=proof,sampling_strata=strata,
            unique_probe_images=len({r['image'] for r in probe}),gradient_probe_views=len(probe),
            TRAIN_present=5112,VAL_present=887,checkpoints=checkpoints,parent_SHA=parent_sha,sources=sources,
            weights_downloaded=False,features_downloaded=False,TEST=False)
        save(out/(prefix+'_summary.json'),summary)
        save(out/(prefix+'_completion.json'),dict(passed=True,CPU_only=True,CUDA_initialized=False,training_updates=0,
            optimizer_created=False,TEST=False,inputs_unchanged=True,parents_unchanged=True,weights_unchanged=True,
            model_view_forwards=forwards,elapsed_seconds=time.monotonic()-start,sources=sources))
        names=[prefix+s for s in ('_protocol.json','_summary.json','_completion.json')]
        save(out/(prefix+'_manifest.json'),{n:sha(out/n) for n in names})
        status('COMPLETE')
    except BaseException as error:
        save(out/(prefix+'_failure.json'),dict(error=repr(error),training_updates=0,TEST=False));status('FAILED',error=repr(error));raise
    finally:torch.optim.Optimizer.__init__=original_optimizer


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('pilot','run'),required=True)
    run(parser.parse_args())
