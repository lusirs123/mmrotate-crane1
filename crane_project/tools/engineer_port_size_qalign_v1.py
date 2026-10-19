"""Full TRAIN-only engineering gates; fixed batches, discarded updates."""
from copy import deepcopy
import io
import random
import statistics
import torch
from crane_project.tools.run_port_size_native_train_v1 import seed,rng_get,rng_set,equal,write
from crane_project.utils import port_size_qalign_v1 as model


def engineering(rows,out,torch):
    from crane_project.tools.run_port_size_qalign_v1 import batch,update
    views=[];present=[r for r in rows if len(r['midpoint_original'])]
    # CPU inventory, including missing views; no VAL construction or evaluation.
    seed(torch);neutral={arm:model.AlignedSizeHead(arm).eval() for arm in model.ARMS}
    stem_inputs={}
    handles=[h.stem.register_forward_pre_hook(lambda module,args,arm=arm:stem_inputs.update({arm:args[0].detach().clone()})) for arm,h in neutral.items()]
    with torch.no_grad():
        for index,r in enumerate(rows):
            x=[r[k] for k in ('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original')]
            pair=model.read_pair(*x)
            cell={k:r[k] for k in ('image','domain','sequence','scale','shard')}
            cell.update(missing=not len(x[-1]),nonvisual_exact=True)
            if len(x[-1]):
                w=pair['support'];cell['W_sum']=float(w.sum());cell['W_active']=int((w>0).sum())
                cell['zero_common_support']=not bool((w>0).any())
                for arm in model.ARMS:
                    cell[arm+'_support_sum']=float(pair['source_support'][arm].sum())
                    cell[arm+'_outside']=int((~pair['inside'][arm]).sum())
                    for coords,label in ((pair['source_local'][arm],'cache'),(pair['original_points'][arm],'original')):
                        cell[arm+'_'+label+'_min_xy']=coords.amin(dim=1)[0].tolist()
                        cell[arm+'_'+label+'_max_xy']=coords.amax(dim=1)[0].tolist()
                    result=neutral[arm](*x)
                    if not torch.equal(result['boxes_original'],x[-1]):raise ValueError('TRAIN neutral M identity')
                    cell[arm+'_neutral_exact']=True
                    cell[arm+'_fallback']=not bool(result['vector_valid'].all())
                if not torch.equal(stem_inputs['fixed_b'][:,256:],stem_inputs['q_aligned'][:,256:]):raise ValueError('Nonvisual inputs differ')
                diff=pair['features']['q_aligned']-pair['features']['fixed_b']
                cell['visual_abs_mean']=float(diff.abs().mean());cell['visual_abs_max']=float(diff.abs().max())
            else:
                for arm in model.ARMS:
                    if not torch.equal(neutral[arm](*x)['boxes_original'],x[-1]):raise ValueError('Missing identity')
            views.append(cell)
            if index%500==0:print('TRAIN support/neutral',index,len(rows),flush=True)
    for handle in handles:handle.remove()
    write(out/'train_support_views.json',views)
    summary=[]
    for domain in ('overall','real','sim'):
        for scale in (.5,1.):
            group=[r for r in views if r['scale']==scale and (domain=='overall' or r['domain']==domain)]
            active=[r for r in group if not r['missing']]
            summary.append(dict(domain=domain,scale=scale,views=len(group),missing=len(group)-len(active),
                zero_common_support=sum(r['zero_common_support'] for r in active),
                sums={k:sum(r[k] for r in active) for k in ('W_sum','W_active','fixed_b_support_sum','q_aligned_support_sum','fixed_b_outside','q_aligned_outside')},
                visual_abs_mean=sum(r['visual_abs_mean'] for r in active)/len(active),
                neutral_all_exact=all(r['fixed_b_neutral_exact'] and r['q_aligned_neutral_exact'] for r in active)))
    write(out/'train_support_summary.json',summary)
    if any(r.get('zero_common_support') for r in views):raise ValueError('Natural TRAIN zero common support; pause, do not filter')
    report=dict(status='PASS',view_count=len(rows),val_shards_loaded=0,formal_updates=0,updates_discarded=True,arms={})
    chosen=present[:16]
    report['fixed_update_batches']=[[dict(image=r['image'],scale=r['scale']) for r in chosen[i:i+8]] for i in (0,8)]
    for arm in model.ARMS:
        seed(torch);h=model.AlignedSizeHead(arm).cuda();opt=torch.optim.Adam(h.parameters(),lr=.001)
        x=batch(chosen[:8],torch);first=update(h,opt,x,arm,torch)
        if first['stem_grad_norm']!=0 or first['terminal_grad_norm']<=0:raise ValueError('Neutral gradient connectivity')
        snap=dict(head=deepcopy(h.state_dict()),opt=deepcopy(opt.state_dict()),rng=rng_get(torch))
        stream=io.BytesIO();torch.save(snap,stream);stream.seek(0);got=torch.load(stream)
        clone=model.AlignedSizeHead(arm).cuda();clone.load_state_dict(got['head']);otheropt=torch.optim.Adam(clone.parameters(),lr=.001);otheropt.load_state_dict(got['opt'])
        y=batch(chosen[8:],torch);rng_set(snap['rng'],torch);second=update(h,opt,y,arm,torch);after=rng_get(torch)
        rng_set(got['rng'],torch);other=update(clone,otheropt,y,arm,torch)
        if second['stem_grad_norm']<=0 or not equal(second,other,torch) or not equal(h.state_dict(),clone.state_dict(),torch) or not equal(opt.state_dict(),otheropt.state_dict(),torch) or not equal(after,rng_get(torch),torch):raise ValueError('Adam/RNG resume mismatch')
        report['arms'][arm]=dict(first=first,second=second,adam_weights_rng_reload_exact=True)
        del h,opt,clone,otheropt,x,y
    heads={a:model.AlignedSizeHead(a).cuda().eval() for a in model.ARMS};x=batch(present[:1],torch)
    pairs=[];peaks={};order_rng=random.Random(1703)
    with torch.no_grad():
        for _ in range(20):
            for h in heads.values():h(*x[:-1])
        for i in range(100):
            order=list(model.ARMS);order_rng.shuffle(order);times={}
            for arm in order:
                torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();heads[arm](*x[:-1]);end.record();torch.cuda.synchronize()
                times[arm]=begin.elapsed_time(end)
                peaks[arm]=max(peaks.get(arm,0),(torch.cuda.max_memory_allocated()-base)/2**20)
            pairs.append(dict(index=i,order=order,**times,delta_ms=times['q_aligned']-times['fixed_b']))
    mean=statistics.mean(r['delta_ms'] for r in pairs)
    report['efficiency']=dict(device='GPU',scope='read_pair + common head, batch1 on GPU; excludes frozen detector/M/transfers',
        warmups_per_arm=20,paired_repeats=100,pairs=pairs,mean_delta_ms=mean,
        median_delta_ms=statistics.median(r['delta_ms'] for r in pairs),min_delta_ms=min(r['delta_ms'] for r in pairs),max_delta_ms=max(r['delta_ms'] for r in pairs),
        incremental_allocator_peak_mib=peaks,allocator_delta_mib=peaks['q_aligned']-peaks['fixed_b'],
        mean_latency_pass=mean<=5,allocator_pass=peaks['q_aligned']-peaks['fixed_b']<=128)
    write(out/'engineering.json',report)
    if not report['efficiency']['mean_latency_pass'] or not report['efficiency']['allocator_pass']:raise ValueError('Engineering resource delta gate')
    del heads,x,neutral;torch.cuda.empty_cache();seed(torch)
