"""Read-only CPU audit: frozen loss gradients, batch coverage and error transport.

No fitting, optimizer, new TEST inference, calibration or checkpoint selection.
All per-view arrays remain server-side; only grouped JSON/CSV are transferable.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/conr_001'
SPEC=dict(version='conr_failure_cpu_20261112_v1',CPU_threads=1,GPU=False,optimizer=False,updates=0,
    TEST='existing fixed rows only; no images or model forward',budget_seconds=600,
    snapshots=['initial','L5','ConR5','ConR24'],gradient_batches='first16 natural batches of original epoch6 schedule (128 TRAIN views); pilot first2',
    full_pair_coverage='all639 natural epoch6 batches; stored ConR5 outputs and fixed TRAIN labels',
    delivery_decomposition='ConR5 VAL887 only, original/bias_only/without_bias; untrained diagnostic interventions',
    selection=False,adoption=False,download='grouped core only; weights/cache/prediction rows/gradients stay server')


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(4*1024*1024),b''):h.update(v)
    return h.hexdigest()
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def load_rows(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open()) as f:return [json.loads(v) for v in f if v.strip()]
def mean(a):return sum(a)/len(a) if a else None

def quantile(v,q):
    if not v:return None
    s=sorted(v);a=(len(s)-1)*q;i=int(a);return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(a-i)


def gradient_stats(a,b):
    """b is already weighted; compare norms, never squared-norm share to benefit."""
    na=float(a.norm());nb=float(b.norm());dot=float(a.dot(b))
    return dict(size_norm=na,weighted_aux_norm=nb,aux_to_size_norm=nb/na if na else None,
        cosine=dot/(na*nb) if na and nb else None,inner_product=dot,
        aux_unit_descent_size_loss_derivative=-dot/nb if nb else 0.)


def masks(target,prediction,window):
    import torch
    dt=(target[:,None]-target[None,:]).abs().amax(-1)/window
    dp=(prediction.detach()[:,None]-prediction.detach()[None,:]).abs().amax(-1)/window
    other=~torch.eye(len(target),device=target.device,dtype=torch.bool)
    positive=(dt<=1)&other;negative=(dt>1)&(dp<=1)&other
    return positive,negative,positive.any(1)&negative.any(1)


def error_transport(old,new):
    """Exact signed MAE-difference sum by correction direction transitions."""
    index={(r['image'],r.get('scale',1.)):r for r in old}
    groups=defaultdict(list)
    for r in new:
        b=index[r['image'],r.get('scale',1.)]
        for k in ('gt','b','midpoint'):assert r[k]==b[k],k
        if r['midpoint'] is None:continue
        for j,edge in enumerate(('long','short')):
            g,m,p,q=[sorted(v[2:4],reverse=True)[j] for v in (r['gt'],r['midpoint'],b['edge_residual'],r['edge_residual'])]
            target=g/m-1.;ca=p/m-1.;cb=q/m-1.;ea=abs(p/g-1.);eb=abs(q/g-1.)
            sign=lambda x:'wrong' if x*target< -1e-8 else 'toward' if x*target>1e-8 else 'neutral'
            # Disjoint target bins, unlike the separately reported near-accurate subset.
            need='shrink' if target<-.01 else 'enlarge' if target>.01 else 'near'
            v=dict(delta=eb-ea,old=ea,new=eb,target=target,old_change=ca,new_change=cb,
                overshoot_old=ca*target>0 and abs(ca)>abs(target),overshoot_new=cb*target>0 and abs(cb)>abs(target))
            for group in ('overall',r['domain'],r['sequence']):
                for transition in ('all',sign(ca)+'_to_'+sign(cb)):
                    groups[group,edge,need,transition].append(v)
    result=[]
    for (group,edge,need,transition),vv in sorted(groups.items()):
        result.append(dict(group=group,edge=edge,target_group=need,transition=transition,n=len(vv),
            error_delta_sum=sum(v['delta'] for v in vv),old_MAE=mean([v['old'] for v in vv]),new_MAE=mean([v['new'] for v in vv]),
            worsened=sum(v['delta']>1e-6 for v in vv),improved=sum(v['delta']< -1e-6 for v in vv),
            overshoot_old=sum(v['overshoot_old'] for v in vv),overshoot_new=sum(v['overshoot_new'] for v in vv),
            old_change_mean=mean([v['old_change'] for v in vv]),new_change_mean=mean([v['new_change'] for v in vv]),
            worst10_positive_delta_sum=sum(sorted([max(v['delta'],0.) for v in vv],reverse=True)[:10])))
    for group,edge,need in {(v['group'],v['edge'],v['target_group']) for v in result}:
        rr=[v for v in result if (v['group'],v['edge'],v['target_group'])==(group,edge,need)]
        allv=next(v for v in rr if v['transition']=='all');parts=[v for v in rr if v['transition']!='all']
        assert sum(v['n'] for v in parts)==allv['n']
        assert abs(sum(v['error_delta_sum'] for v in parts)-allv['error_delta_sum'])<1e-10
    return result


def coverage(records,saved,schedule,window):
    import torch
    from crane_project.utils import port_size_pair_conr_v1 as c
    totals=defaultdict(lambda:defaultdict(int))
    for start in range(0,len(schedule),8):
        rr=[records[i] for i in schedule[start:start+8]]
        target=[];pred=[];near=[]
        for r in rr:
            q=saved[r['image'],r['scale']]
            g,m,p=[torch.tensor(sorted(q[k][2:4],reverse=True),dtype=torch.float64) for k in ('gt','midpoint','edge_residual')]
            t=c.model.residual.size_targets(r['gt_original'],r['midpoint_original'],r['boxes_model'])['log_residual']
            target.append(c.canonical_pair(t,r['boxes_model'],r['midpoint_original'])[0])
            # Pair selection uses pre-delivery deltas, including fallback frames.
            delta=torch.tensor(q['delta_roi'],dtype=t.dtype).reshape(1,2)
            pred.append(c.canonical_pair(delta,r['boxes_model'],r['midpoint_original'])[0])
            near.append((m/g-1).abs())
        t,p=torch.stack(target),torch.stack(pred)
        for label,pp in [('neutral',torch.zeros_like(p)),('ConR5',p)]:
            pos,neg,active=masks(t,pp,window)
            for i,r in enumerate(rr):
                labels=['overall',r['domain'],r['sequence']]
                for j,e in enumerate(('long','short')):
                    need='shrink' if t[i,j]<math.log(.99) else 'enlarge' if t[i,j]>math.log(1.01) else 'near'
                    labels.extend([e+'_'+need])
                    if near[i][j]<=.03:labels.append(e+'_accurate_M_le3pct')
                    if near[i][j]>.1:labels.append(e+'_M_gt10pct')
                for group in labels:
                    v=totals[label,group];v['n']+=1;v['active']+=int(active[i]);v['no_positive']+=int(not bool(pos[i].any()));v['no_negative']+=int(not bool(neg[i].any()))
                    v['positive_pairs']+=int(pos[i].sum());v['negative_pairs']+=int(neg[i].sum())
                    for j in range(len(rr)):
                        if pos[i,j]:
                            v['positive_same_video']+=int(r['sequence']==rr[j]['sequence'])
                            v['positive_same_image']+=int(r['image']==rr[j]['image'])
    return [dict(snapshot=k[0],group=k[1],**v,active_fraction=v['active']/v['n']) for k,v in sorted(totals.items())]


def run(args):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(1)
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    from crane_project.utils import port_size_pair_conr_v1 as c
    from crane_project.tools import train_symeood_r4_lvsr_pair_v1 as trainer
    from crane_project.tools import train_lvsr_conr_v1 as training
    from crane_project.tools import diagnose_lvsr_direction_representation_v1 as d
    from crane_project.tools.run_symeood_110_lvsr_v1 import schedule_indices
    from crane_project.tools.run_port_size_native_train_v1 import seed
    from types import SimpleNamespace
    started=time.monotonic();identity=p.checked_sources();assert not torch.cuda.is_initialized()
    prefix='failure_audit_pilot' if args.stage=='pilot' else 'failure_audit'
    sources={str(Path(__file__).resolve().relative_to(ROOT)):sha(__file__),'tests/test_lvsr_conr_failure_v1.py':sha(ROOT/'tests/test_lvsr_conr_failure_v1.py')}
    if args.stage=='run':
        pilot=read(OUT/'failure_audit_pilot_completion.json');assert pilot['passed'] and pilot['source_SHA']==sources
    save(OUT/(prefix+'_protocol.json'),dict(**SPEC,source_SHA=sources,stage=args.stage))
    old_init=torch.optim.Optimizer.__init__
    def forbidden(*a,**kw):raise RuntimeError('Read-only diagnosis forbids optimizer creation')
    torch.optim.Optimizer.__init__=forbidden
    try:
        summary=read(OUT/'core_summary.json');assert summary['source_SHA']==training.source_hashes()
        train,val,_=trainer.load_inputs(p.paths(),identity,SimpleNamespace(torch=torch));present=[r for r in train if len(r['midpoint_original'])]
        cache_before=d.records_digest(train+val)
        pinned=[p.paths()['inputs']/'train.pt',p.paths()['inputs']/'val.pt',ROOT/p.parent.R4_CHECKPOINT,
            p.midpoint_directory()/'train/head_epoch_20.pth',p.BASE/'lvsr/attempt_001/train/head_epoch_05.pth',OUT/'head_epoch_05.pth',OUT/'head_epoch_24.pth']
        hashes={str(q.relative_to(ROOT)):sha(q) for q in pinned}
        assert hashes[str((OUT/'head_epoch_05.pth').relative_to(ROOT))]==summary['selection']['representative_checkpoint']['sha256']
        assert hashes[str((OUT/'head_epoch_24.pth').relative_to(ROOT))]==summary['epoch24']['checkpoint']['sha256']
        saved={ (q['image'],q['scale']):q for q in load_rows(OUT/'representative_TRAIN_VAL.rows.jsonl.gz') }
        oldsaved={ (q['image'],q['scale']):q for q in load_rows(OUT.parent/'attempt_001/L5_all.rows.jsonl.gz') }
        order=schedule_indices(len(present),6);nb=2 if args.stage=='pilot' else 16
        named=[(f'batch{i+1}',[present[j] for j in order[i*8:i*8+8]]) for i in range(nb)]
        gradients=[];max_replay=0.;views=0
        for label,path in [('initial',None),('L5',p.BASE/'lvsr/attempt_001/train/head_epoch_05.pth'),('ConR5',OUT/'head_epoch_05.pth'),('ConR24',OUT/'head_epoch_24.pth')]:
            seed(torch);head=c.PairConRSizeHead().cpu().train()
            if path:head.load_state_dict(torch.load(str(path),map_location='cpu')['head_state'],strict=True)
            before=d.tensor_digest(head.state_dict());params=list(head.parameters());names=[n for n,_ in head.named_parameters()]
            for batchlabel,rr in named:
                x=d.batch(rr);output=head(*x[:-1]);views+=len(rr)
                pred=c.canonical_pair(output['delta_roi'],x[3],x[-2]);target=c.canonical_pair(c.model.residual.size_targets(x[-1],x[-2],x[3])['log_residual'],x[3],x[-2])
                size=c.model.size_loss(output,x[-1],x[-2],x[3]);aux,stats=c.contrastive_loss(output['training_features'],output['common_support'],target,pred)
                gs=torch.autograd.grad(size,params,retain_graph=True)
                ga=torch.autograd.grad(c.CONR_WEIGHT*aux,params,retain_graph=True,allow_unused=True)
                ga=[torch.zeros_like(q) if g is None else g for g,q in zip(ga,params)]
                flat=lambda vv:torch.cat([v.reshape(-1) for v in vv]).double()
                a,b=flat(gs),flat(ga)
                total=flat(torch.autograd.grad(size+c.CONR_WEIGHT*aux,params,retain_graph=True))
                assert torch.allclose(a+b,total,atol=1e-6,rtol=1e-5)
                blocks={}
                for block in ('all','stem','terminal_weight','terminal_bias'):
                    idx=[i for i,n in enumerate(names) if block=='all' or block=='stem' and n.startswith('stem') or block=='terminal_weight' and n.startswith('terminal') and n.endswith('weight') or block=='terminal_bias' and n.startswith('terminal') and n.endswith('bias')]
                    blocks[block]=gradient_stats(flat([gs[i] for i in idx]),flat([ga[i] for i in idx]))
                assert blocks['terminal_weight']['weighted_aux_norm']==blocks['terminal_bias']['weighted_aux_norm']==0.
                pos,neg,active=masks(target.detach(),pred.detach(),c.WINDOW);assert int(active.sum())==stats['active_anchors']
                # Unit-norm auxiliary and size directions separate direction from raw scale.
                responses=defaultdict(list)
                for j in range(len(rr)):
                    for edge,ename in enumerate(('long','short')):
                        jac=flat(torch.autograd.grad(pred[j,edge],params,retain_graph=True))
                        derivative=float(((pred[j,edge]-target[j,edge])/.1).clamp(-1,1).detach())
                        delta_aux=-float(jac.dot(b))/float(b.norm()) if float(b.norm()) else 0.
                        delta_size=-float(jac.dot(a))/float(a.norm()) if float(a.norm()) else 0.
                        need='shrink' if target[j,edge]<math.log(.99) else 'enlarge' if target[j,edge]>math.log(1.01) else 'near'
                        responses[ename+'_'+need].append(dict(aux=delta_aux,size=delta_size,aux_loss_derivative=derivative*delta_aux,size_loss_derivative=derivative*delta_size))
                response={k:dict(n=len(v),mean_aux_unit_delta=mean([q['aux'] for q in v]),mean_size_unit_delta=mean([q['size'] for q in v]),aux_unit_loss_derivative=mean([q['aux_loss_derivative'] for q in v]),size_unit_loss_derivative=mean([q['size_loss_derivative'] for q in v]),aux_harms=sum(q['aux_loss_derivative']>1e-12 for q in v)) for k,v in responses.items()}
                if label in ('L5','ConR5'):
                    reference=oldsaved if label=='L5' else saved
                    for j,r in enumerate(rr):
                        q=reference[r['image'],r['scale']];expected=torch.tensor(q['edge_residual'],dtype=output['boxes_original'].dtype)
                        err=float((output['boxes_original'][j]-expected).abs().max());max_replay=max(err,max_replay)
                        assert torch.allclose(output['boxes_original'][j],expected,atol=1e-4,rtol=1e-6)
                gradients.append(dict(snapshot=label,batch=batchlabel,n=len(rr),size_loss=float(size.detach()),weighted_aux_loss=float((c.CONR_WEIGHT*aux).detach()),blocks=blocks,response=response,**stats))
                assert not any(q.grad is not None for q in params)
                if time.monotonic()-started>SPEC['budget_seconds']:raise RuntimeError('CPU budget exceeded')
            assert d.tensor_digest(head.state_dict())==before
            print(json.dumps(dict(stage='gradients',snapshot=label,batches=nb,elapsed=time.monotonic()-started)),flush=True)
        result=dict(gradient_batches=gradients,CPU_replay_max_abs=max_replay)
        if args.stage=='run':
            result['pair_coverage']=coverage(present,saved,order,c.WINDOW)
            transport={}
            for split in ('train_s1','train_s05','val'):
                choose=lambda q: q['split']=='val' if split=='val' else q['split']!='val' and q['scale']==(1. if split=='train_s1' else .5)
                transport[split]=error_transport([q for q in oldsaved.values() if choose(q)],[q for q in saved.values() if choose(q)])
            oldtest=ROOT/'work_dirs/symeood_r4_adaptation_20261031/lvsr/test/attempt_002/test_rows.jsonl'
            newtest=OUT/'test_fixed_rows.jsonl';testreceipt=read(OUT/'test_fixed_completion.json')
            from crane_project.tools.eval_lvsr_conr_fixed_test_v1 import L5_ROWS_SHA
            assert sha(oldtest)==L5_ROWS_SHA and testreceipt['passed']
            hashes[str(oldtest.relative_to(ROOT))]=sha(oldtest);hashes[str(newtest.relative_to(ROOT))]=sha(newtest)
            transport['test']=error_transport(load_rows(oldtest),load_rows(newtest));result['error_transport']=transport
            # Feature/bias interventions keep actual weight tensors unchanged.
            head=c.PairConRSizeHead().eval();head.load_state_dict(torch.load(str(OUT/'head_epoch_05.pth'),map_location='cpu')['head_state'])
            before=d.tensor_digest(head.state_dict());biases=[q.bias.detach() for q in head.terminal.branches]
            bias=torch.cat((biases[0][:2],biases[1][:2],biases[0][2:],biases[1][2:]))[None,:,None,None]
            arms={};max_val_replay=0.
            from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity
            for arm in ('original','bias_only','without_bias'):
                handle=None;rr=[]
                if arm!='original':
                    def hook(module,args,out,arm=arm):return bias.expand_as(out) if arm=='bias_only' else out-bias
                    handle=head.terminal.register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        for start in range(0,len(val),8):
                            vv=val[start:start+8];x=d.batch(vv);o=head(*x[:-1]);views+=len(vv)
                            for j,r in enumerate(vv):
                                q=dict(saved[r['image'],r['scale']],edge_residual=o['boxes_original'][j].tolist())
                                assert frozen_identity(q);rr.append(q)
                                if arm=='original':
                                    expected=torch.tensor(saved[r['image'],r['scale']]['edge_residual'])
                                    err=float((o['boxes_original'][j]-expected).abs().max());max_val_replay=max(max_val_replay,err)
                                    assert torch.allclose(o['boxes_original'][j],expected,atol=1e-4,rtol=1e-6)
                finally:
                    if handle:handle.remove()
                arms[arm]=trainer.summarize(rr)
            result['VAL_frozen_terminal_decomposition']=arms;result['CPU_VAL_replay_max_abs']=max_val_replay
            assert d.tensor_digest(head.state_dict())==before
        assert all(sha(ROOT/q)==h for q,h in hashes.items()) and d.records_digest(train+val)==cache_before
        assert p.checked_sources()==identity and summary['source_SHA']==training.source_hashes()
        assert not torch.cuda.is_initialized() and time.monotonic()-started<SPEC['budget_seconds']
        save(OUT/(prefix+'_summary.json'),result)
        save(OUT/(prefix+'_completion.json'),dict(passed=True,CPU_only=True,CUDA_initialized=False,optimizer_created=False,training_updates=0,new_TEST_inference=False,
            elapsed_seconds=time.monotonic()-started,forward_views=views,weights_unchanged=True,inputs_unchanged=True,source_SHA=sources,pinned_SHA=hashes,
            representative_unchanged=summary['selection'],production_sources_unchanged=True))
        files=[prefix+'_'+suffix+'.json' for suffix in ('protocol','summary','completion')]
        save(OUT/(prefix+'_manifest.json'),{n:sha(OUT/n) for n in files})
        print(json.dumps(dict(passed=True,stage=args.stage,elapsed=time.monotonic()-started,forward_views=views)),flush=True)
    except BaseException as error:
        save(OUT/(prefix+'_failure.json'),dict(error=repr(error),updates=0,automatic_retry=False));raise
    finally:torch.optim.Optimizer.__init__=old_init


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('pilot','run'),default='pilot');run(p.parse_args())
