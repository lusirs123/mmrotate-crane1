"""Frozen CPU perturbation audit; no optimizer, candidate, search or TEST."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from collections import Counter
import json,math
from pathlib import Path
import random,sys,time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/partial_visual_norm_001'
SPEC=dict(version='lvsr_bsam_review_20261113_v1',CPU_threads=1,GPU=False,TEST=False,budget_seconds=600,
    head_snapshots=['L5','L24'],model='unchanged PairSpecificSizeHead2480',optimizer_updates=0,
    density='all5112 original TRAIN views, canonical2D logGT/M, Linf radius log1.02; include self; no VAL-derived weights',
    weights='inverse sqrt TRAIN neighbor count, fixed global mean1 and cap4; no batch renormalization',
    rarity='TRAIN density lowest quintile vs highest quintile, tie-inclusive; hard>10% and accurate<=3% kept separate',
    radius='one fixed diagnostic radius per checkpoint:0.001*L2norm(all2480 parameters), not a tuned training hyperparameter',
    perturbations=['SAM_unweighted_global_gradient','BSAM_weighted_global_gradient'],
    natural_batch_probe='original epoch6 first16 batches, unchanged order/batch8; compare unweighted, paper W->U and author W->W gradients; no update',
    interpretation='positive adversarial loss change is sensitivity, not trained-model damage; first-order remainder not a Hessian estimate; no formal training gate from this radius',
    selection=False,adoption=False,download='grouped core JSON only; no weights, embeddings, gradients or per-frame rows')


def cosine(a,b):
    n=float(a.norm()*b.norm());return float(a.dot(b))/n if n else None


def evaluate(head,rr):
    import torch
    from crane_project.utils import lvsr_bsam_review_v1 as r
    from crane_project.tools.diagnose_lvsr_direction_representation_v1 import batch
    losses=[];boxes=[];delivered=Counter()
    with torch.no_grad():
        for start in range(0,len(rr),8):
            x=batch(rr[start:start+8]);ll,o=r.loss_edges(head,x)
            assert torch.equal(o['boxes_original'][:,[0,1,4,5]],x[-2][:,[0,1,4,5]])
            assert torch.equal(o['boxes_original'][:,2]<o['boxes_original'][:,3],x[-2][:,2]<x[-2][:,3])
            losses.append(ll.double());boxes.append(o['boxes_original'].detach())
            delivered['accepted']+=int(o['accepted'].sum());delivered['fallback_or_bypass']+=int((~o['accepted']).sum())
    return torch.cat(losses),torch.cat(boxes),dict(delivered)


def full_gradient(head,rr,weights):
    import torch
    from crane_project.utils import lvsr_bsam_review_v1 as r
    from crane_project.tools.diagnose_lvsr_direction_representation_v1 import batch
    params=list(head.parameters());total=torch.zeros(2480,dtype=torch.float64)
    for start in range(0,len(rr),8):
        x=batch(rr[start:start+8]);ll,_=r.loss_edges(head,x);w=weights[start:start+len(ll)].to(ll)
        total+=r.flat_gradient((ll*w[:,None]).sum()/(2*len(rr)),params).double()
    return total


def groups_summary(rr,losses,boxes,masks,base=None):
    import torch
    gt=torch.tensor([sorted(r['gt_original'][0,2:4].tolist(),reverse=True) for r in rr],dtype=torch.float64)
    sizes=boxes[:,2:4].double().sort(descending=True)[0];error=(sizes/gt-1).abs();result={}
    for name,mask in masks.items():
        n=int(mask.sum())
        if not n:continue
        current=losses[mask].mean(0)
        cell=dict(n=n,long_loss=float(current[0]),short_loss=float(current[1]),loss=float(current.mean()),
            long_MAE=float(error[mask,0].mean()),short_MAE=float(error[mask,1].mean()),joint10=int((error[mask].amax(-1)<=.1).sum()))
        if base is not None:
            delta=current-base[mask].mean(0)
            cell.update(long_loss_change=float(delta[0]),short_loss_change=float(delta[1]),loss_change=float(delta.mean()))
        result[name]=cell
    return result


def summarize_values(values):
    from crane_project.tools.diagnose_lvsr_conr_failure_v1 import mean,quantile
    values=[v for v in values if v is not None]
    return dict(n=len(values),mean=mean(values),median=quantile(values,.5),min=min(values) if values else None,max=max(values) if values else None)


def run():
    import torch
    torch.set_num_threads(1)
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    from crane_project.utils import lvsr_bsam_review_v1 as r
    from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
    from crane_project.tools import train_symeood_r4_lvsr_pair_v1 as trainer
    from crane_project.tools import diagnose_lvsr_direction_representation_v1 as d
    from crane_project.tools import diagnose_lvsr_conr_failure_v1 as io
    started=time.monotonic();identity=p.checked_sources()
    source_names=[__file__,r.__file__,ROOT/'tests/test_lvsr_bsam_review_v1.py']
    sources={str(Path(q).resolve().relative_to(ROOT)):io.sha(q) for q in source_names}
    io.save(OUT/'bsam_review_protocol.json',dict(settings=SPEC,source_SHA=sources))
    try:
        train,val,_=trainer.load_inputs(p.paths(),identity,SimpleNamespace(torch=torch));records=train+val
        before=d.records_digest(records);train=[q for q in train if len(q['midpoint_original'])]
        assert len(train)==5112 and len(val)==887
        mm=d.metadata(train);targets=torch.tensor([v['target'] for v in mm],dtype=torch.float64)
        density=r.target_density(targets);weights=r.density_weights(density)
        ordered=density.sort()[0];low=int(ordered[math.ceil(.2*len(ordered))-1]);high=int(ordered[math.ceil(.8*len(ordered))-1]);assert low<high
        masks={'overall':torch.ones(len(train),dtype=torch.bool),'rare_density':density<=low,'common_density':density>=high,
            'M_joint_hard_gt10pct':torch.tensor([max(v['M_error'])>.1 for v in mm]),
            'M_both_accurate_le3pct':torch.tensor([max(v['M_error'])<=.03 for v in mm])}
        for j,edge in enumerate(('long','short')):
            masks[edge+'_needs_shrink']=targets[:,j]<math.log(.99)
            masks[edge+'_accurate_le3pct']=torch.tensor([v['M_error'][j]<=.03 for v in mm])
        for q in ('real','sim'):masks[q]=torch.tensor([v['domain']==q for v in mm])
        for q in (1.,.5):masks['scale'+str(q)]=torch.tensor([v['scale']==q for v in mm])
        order=list(range(5112));random.Random(1709).shuffle(order)
        snapshots={};hashes={str(p.paths()['inputs']/n):p.sha(p.paths()['inputs']/n) for n in ('train.pt','val.pt')}
        for epoch,label,expected in [(5,'L5','b272c50dfe6329414803d5f22c6e1e84d9c3db37b6bb2e31f1c7ed140f662c3e'),(24,'L24','4e6baa0dd4bf568db68f63f39a835c2cbc4bb0ba08f2358b0ff3ed9a9ac8c5b3')]:
            checkpoint=p.BASE/'lvsr/attempt_001/train'/('head_epoch_%02d.pth'%epoch);assert io.sha(checkpoint)==expected
            hashes[str(checkpoint)]=expected
            payload=torch.load(str(checkpoint),map_location='cpu');head=PairSpecificSizeHead();head.load_state_dict(payload['head_state'],strict=True);head.eval()
            saved=d.tensor_digest(head.state_dict());params=list(head.parameters());norm=float(torch.cat([v.detach().reshape(-1) for v in params]).norm());rho=.001*norm
            before_loss,before_boxes,before_delivery=evaluate(head,train);vl,vb,vd=evaluate(head,val)
            saved_rows=d.rows(OUT.parent/'attempt_001'/('L%d_all.rows.jsonl.gz'%epoch))
            original=torch.tensor([saved_rows[q['image'],q['scale']]['edge_residual'] for q in train])
            replay_max=float((before_boxes-original).abs().max());assert torch.allclose(before_boxes,original,atol=1e-4,rtol=1e-6)
            grads={'SAM_unweighted_global_gradient':full_gradient(head,train,torch.ones(5112)),
                'BSAM_weighted_global_gradient':full_gradient(head,train,weights)}
            perturb_reports={}
            for method,g in grads.items():
                eps=r.perturbation(g,rho).float()
                with r.displaced(params,eps):
                    ll,bb,delivery=evaluate(head,train);lv,bv,dv=evaluate(head,val)
                assert d.tensor_digest(head.state_dict())==saved
                perturb_reports[method]=dict(perturbation_norm=float(eps.norm()),TRAIN=groups_summary(train,ll,bb,masks,before_loss),
                    VAL=groups_summary(val,lv,bv,{'overall':torch.ones(887,dtype=torch.bool)},vl),TRAIN_delivery=delivery,VAL_delivery=dv,
                    first_order_global_loss_change=float(grads['SAM_unweighted_global_gradient'].dot(eps.double())),
                    observed_global_loss_change=float((ll-before_loss).mean()))
            batches=[]
            for start in range(0,128,8):
                ii=order[start:start+8];rr=[train[i] for i in ii];x=d.batch(rr);w=weights[ii].float();ll,_=r.loss_edges(head,x)
                gu=r.flat_gradient(ll.mean(),params);ll,_=r.loss_edges(head,x);gw=r.flat_gradient((ll*w[:,None]).mean(),params)
                eps=r.perturbation(gw,rho)
                with r.displaced(params,eps):
                    ll,_=r.loss_edges(head,x);paper=r.flat_gradient(ll.mean(),params)
                    ll,_=r.loss_edges(head,x);author=r.flat_gradient((ll*w[:,None]).mean(),params)
                assert d.tensor_digest(head.state_dict())==saved
                batches.append(dict(batch=start//8,n=8,weight_mean=float(w.mean()),rare=int((density[ii]<=low).sum()),hard=int(masks['M_joint_hard_gt10pct'][ii].sum()),
                    weighted_unweighted_gradient_cosine=cosine(gw,gu),paper_original_gradient_cosine=cosine(paper,gu),paper_author_gradient_cosine=cosine(paper,author),
                    paper_gradient_change_norm=float((paper-gu).norm()),original_gradient_norm=float(gu.norm())))
            snapshots[label]=dict(epoch=epoch,checkpoint_SHA=expected,parameter_norm=norm,rho=rho,original_row_replay_max=replay_max,
                baseline_TRAIN=groups_summary(train,before_loss,before_boxes,masks),baseline_VAL=groups_summary(val,vl,vb,{'overall':torch.ones(887,dtype=torch.bool)}),
                baseline_TRAIN_delivery=before_delivery,baseline_VAL_delivery=vd,global_gradient_cosine=cosine(grads['SAM_unweighted_global_gradient'],grads['BSAM_weighted_global_gradient']),
                perturbations=perturb_reports,natural_batches=batches,
                natural_summary={key:summarize_values([v[key] for v in batches]) for key in ('weighted_unweighted_gradient_cosine','paper_original_gradient_cosine','paper_author_gradient_cosine')},
                state_exactly_restored=True)
            print(json.dumps(dict(stage='bsam_review_snapshot_complete',head=label,rho=rho,replay_max=replay_max,global_gradient_cosine=snapshots[label]['global_gradient_cosine'],elapsed=time.monotonic()-started)),flush=True)
            if time.monotonic()-started>600:raise RuntimeError('BSAM CPU review budget exceeded')
        assert d.records_digest(records)==before and p.checked_sources()==identity and not torch.cuda.is_initialized()
        assert all(io.sha(Path(q))==h for q,h in hashes.items()) and all(io.sha(ROOT/q)==h for q,h in sources.items())
        io.save(OUT/'bsam_review_summary.json',dict(settings=SPEC,TRAIN_views=5112,VAL_views=887,
            density=dict(window=r.WINDOW,min=int(density.min()),max=int(density.max()),low_quintile_threshold=low,high_quintile_threshold=high,
                rare=int((density<=low).sum()),common=int((density>=high).sum()),weights_min=float(weights.min()),weights_max=float(weights.max()),weights_mean=float(weights.mean()),weights_at_cap=int((weights>=4.-1e-12).sum()),
                rare_hard_overlap=int(((density<=low)&masks['M_joint_hard_gt10pct']).sum())),snapshots=snapshots))
        io.save(OUT/'bsam_review_completion.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,optimizer_updates=0,TEST_access=False,
            elapsed_seconds=time.monotonic()-started,source_SHA=sources,pinned_SHA=hashes,cache_digest_before=before,cache_digest_after=before,
            saved_checkpoint_unchanged=True,temporary_perturbations_restored_exactly=True,script_SHA=io.sha(__file__)))
        files=['bsam_review_'+s+'.json' for s in ('protocol','summary','completion')]
        io.save(OUT/'bsam_review_manifest.json',{n:io.sha(OUT/n) for n in files})
        print(json.dumps(dict(completed=True,updates=0,seconds=time.monotonic()-started)),flush=True)
    except BaseException as e:
        io.save(OUT/'bsam_review_failure.json',dict(error=repr(e),retry=False,TEST=False));raise


if __name__=='__main__':run()
