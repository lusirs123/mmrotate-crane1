"""Bounded CPU learning diagnostic, not a new candidate or TEST selector."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/conr_001'
SPEC=dict(version='lvsr_small_fit_20261112_v1',CPU_threads=1,GPU=False,TEST=False,budget_seconds=600,
    fit_views=128,native_scale_only=True,domain_quota=64,
    holdout='lexicographically last real TRAIN video, fixed before selection; sim has one TRAIN video so no sim TRAIN-video holdout; full VAL videos evaluated without fitting',
    selection='domain x joint long/short direction (GT/M <.99 shrink, >1.01 enlarge, else near), equal round-robin cell quota; within cell round-robin video then evenly spaced frame identities',
    arms=['original_full2480','terminal_bias_only8'],initialization='same original seed1703 tensors; fresh Adam; original zero terminal',
    steps_per_arm=500,batch_size=8,seed=1703,lr=.001,weight_decay=0.,betas=[.9,.999],eps=1e-8,clip_norm=10.,
    loss='unchanged SmoothL1 beta.1 log-size pre-delivery; no ConR',
    checkpoints=[0,50,100,250,500],representative='final step500 only; no best-step search, no adoption',
    clear_fit_signal='at500 both delivered fit edge MAEs at least10% lower than trained bias control and lower than M; joint10 no smaller than bias control; report direction/amplitude/near separately',
    limits='finite data/budget can demonstrate fit but failure cannot uniquely distinguish feature information, representation capacity or optimization; no full-TRAIN generalization claim',
    downloads='core grouped JSON/CSV only; weights, optimizer, inputs and frame identities stay server')


def direction(value):return 0 if value<math.log(.99) else 2 if value>math.log(1.01) else 1


def select_fit(records,per_domain=64):
    """Labels and metadata only; never consult VAL or learned predictions."""
    native=[r for r in records if r['split']!='val' and r['scale']==1. and len(r['midpoint_original'])]
    real=sorted({r['sequence'] for r in native if r['domain']=='real'})
    if len(real)<2:raise ValueError('Need at least two real TRAIN videos')
    held=real[-1];pool=[r for r in native if r['sequence']!=held];selected=[];counts=[]
    for domain in ('real','sim'):
        cells=defaultdict(list)
        for r in pool:
            if r['domain']!=domain:continue
            g=sorted(r['gt_original'][0,2:4].tolist(),reverse=True);m=sorted(r['midpoint_original'][0,2:4].tolist(),reverse=True)
            key=tuple(direction(math.log(a/b)) for a,b in zip(g,m));cells[key].append(r)
        if sum(map(len,cells.values()))<per_domain:raise ValueError('Insufficient TRAIN domain pool')
        quota={k:0 for k in sorted(cells)}
        while sum(quota.values())<per_domain:
            for k in quota:
                if quota[k]<len(cells[k]) and sum(quota.values())<per_domain:quota[k]+=1
        for k,n in quota.items():
            byvideo=defaultdict(list)
            for r in cells[k]:byvideo[r['sequence']].append(r)
            alloc={v:0 for v in sorted(byvideo)}
            while sum(alloc.values())<n:
                for v in alloc:
                    if alloc[v]<len(byvideo[v]) and sum(alloc.values())<n:alloc[v]+=1
            for video,nn in alloc.items():
                rr=sorted(byvideo[video],key=lambda r:(r['frame_id'],r['image']))
                take=[rr[round(j*(len(rr)-1)/(nn-1))] for j in range(nn)] if nn>1 else rr[:nn]
                selected.extend(take)
            counts.append(dict(domain=domain,long_direction=k[0],short_direction=k[1],available=len(cells[k]),selected=n))
    selected.sort(key=lambda r:(r['domain'],r['sequence'],r['frame_id'],r['image']))
    assert len(selected)==2*per_domain and len({r['image'] for r in selected})==len(selected)
    holdout=[r for r in native if r['sequence']==held]
    assert not {r['image'] for r in selected}&{r['image'] for r in holdout}
    return selected,holdout,counts,held


def configure(head,arm):
    if arm not in SPEC['arms']:raise ValueError('Unknown diagnostic arm')
    for name,p in head.named_parameters():
        p.requires_grad_(arm=='original_full2480' or name.startswith('terminal.') and name.endswith('.bias'))
    count=sum(p.numel() for p in head.parameters() if p.requires_grad)
    if count!=(2480 if arm=='original_full2480' else 8):raise ValueError('Wrong trainable parameter set')
    return [p for p in head.parameters() if p.requires_grad]


def step(head,opt,x):
    import torch
    from crane_project.utils import port_size_qalign_v1 as model
    head.train();opt.zero_grad();output=head(*x[:-1]);loss=model.size_loss(output,x[-1],x[-2],x[3])
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite diagnostic loss')
    loss.backward();params=[p for p in head.parameters() if p.requires_grad]
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):raise ValueError('Invalid diagnostic gradient')
    norm=float(torch.nn.utils.clip_grad_norm_(params,10.));opt.step()
    if any(not bool(torch.isfinite(p).all()) for p in params):raise ValueError('Invalid diagnostic parameters')
    return dict(loss=float(loss.detach()),gradient_norm=norm)


def evaluate(head,records):
    import torch
    from crane_project.tools import diagnose_lvsr_direction_representation_v1 as d
    from crane_project.utils import port_size_qalign_v1 as model
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity
    head.eval();pred={}
    with torch.no_grad():
        pp=[r for r in records if len(r['midpoint_original'])]
        for start in range(0,len(pp),8):
            rr=pp[start:start+8];x=d.batch(rr);o=head(*x[:-1])
            target=model.residual.size_targets(x[-1],x[-2],x[3])['log_residual']
            ll=torch.nn.functional.smooth_l1_loss(o['delta_roi'],target,beta=.1,reduction='none')
            ll=d.canonical(ll,x[3],x[-2])
            for j,r in enumerate(rr):
                pred[r['image'],r['scale']]=dict(edge_residual=o['boxes_original'][j].tolist(),delta_roi=o['delta_roi'][j].tolist(),loss_edges_canonical=ll[j].tolist(),
                    size_delivery='neutral' if bool(o['neutral_bypass'][j]) else 'square' if bool(o['square_bypass'][j]) else 'accepted' if bool(o['accepted'][j]) else 'fallback',
                    failed_checks=[k for k,v in o['checks'].items() if not bool(v[j])])
    result=[]
    for r in records:
        q={k:r[k] for k in ('image','sequence','domain','frame_id','scale','split','shard')}
        q.update(gt=r['gt_original'][0].tolist(),b=r['boxes_original'][0].tolist() if len(r['boxes_original']) else None,
            midpoint=r['midpoint_original'][0].tolist() if len(r['midpoint_original']) else None)
        q.update(pred.get((r['image'],r['scale']),dict(edge_residual=None,delta_roi=None,loss_edges_canonical=None,size_delivery='no_output',failed_checks=[])))
        assert frozen_identity(q);result.append(q)
    return result


def amplitude(rows):
    import torch
    from crane_project.tools.diagnose_lvsr_direction_representation_v1 import score_probe
    out={}
    rr=[r for r in rows if r['edge_residual'] is not None]
    for i,edge in enumerate(('long','short')):
        g,m,p=[torch.tensor([sorted(r[k][2:4],reverse=True)[i] for r in rr],dtype=torch.float64) for k in ('gt','midpoint','edge_residual')]
        target=(g/m).log();pred=(p/m).log();merror=(m/g-1).abs();cerror=(p/g-1).abs()
        centered=target-target.mean();slope=float((centered*(pred-pred.mean())).sum()/centered.square().sum()) if float(centered.square().sum()) else None
        bins={}
        for name,mask in [('needs_shrink',target<math.log(.99)),('needs_enlarge',target>math.log(1.01)),('accurate_M_le3pct',merror<=.03)]:
            n=int(mask.sum());tt=target[mask];pp=pred[mask]
            bins[name]=dict(n=n,M_MAE=float(merror[mask].mean()) if n else None,head_MAE=float(cerror[mask].mean()) if n else None,
                target_log_mean=float(tt.mean()) if n else None,prediction_log_mean=float(pp.mean()) if n else None,
                correct_direction=int((tt*pp>1e-8).sum()),wrong_direction=int((tt*pp< -1e-8).sum()),
                harmed=int((cerror[mask]-merror[mask]>1e-6).sum()))
        out[edge]=dict(direction=score_probe(pred,target),predicted_target_slope=slope,log_MAE=float((pred-target).abs().mean()),bins=bins)
    return out


def report(rows,common):
    groups={'overall':rows}
    for domain in ('real','sim'):
        rr=[r for r in rows if r['domain']==domain]
        if rr:groups[domain]=rr
    for seq in sorted({r['sequence'] for r in rows}):groups[seq]=[r for r in rows if r['sequence']==seq]
    return {g:dict(summary=common['summary'](rr),amplitude=amplitude(rr)) for g,rr in groups.items()}


def run():
    import torch,runpy
    torch.set_num_threads(1)
    from types import SimpleNamespace
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
    from crane_project.tools import train_symeood_r4_lvsr_pair_v1 as trainer
    from crane_project.tools import diagnose_lvsr_direction_representation_v1 as d
    from crane_project.tools import diagnose_lvsr_conr_failure_v1 as io
    started=time.monotonic();identity=p.checked_sources();assert not torch.cuda.is_initialized()
    sources={str(Path(__file__).resolve().relative_to(ROOT)):io.sha(__file__),'tests/test_lvsr_small_fit_v1.py':io.sha(ROOT/'tests/test_lvsr_small_fit_v1.py')}
    io.save(OUT/'small_fit_protocol.json',dict(**SPEC,sources=sources))
    try:
        train,val,inputs=trainer.load_inputs(p.paths(),identity,SimpleNamespace(torch=torch));records=train+val
        before_cache=d.records_digest(records);fit,holdout,strata,held=select_fit(train)
        assert len(fit)==128 and held=='real_seq13' and len(holdout)==304 and len(val)==887
        pools={'fit_TRAIN128':fit,'heldout_TRAIN_real_video':holdout,'VAL_unfitted_videos':val}
        manifest=dict(fit=[dict(image=r['image'],sequence=r['sequence'],frame_id=r['frame_id'],domain=r['domain'],scale=r['scale']) for r in fit],
            heldout_TRAIN_video=held,heldout_TRAIN_frames=len(holdout),VAL_frames=len(val),strata=strata)
        io.save(OUT/'small_fit_server_only_samples.json',manifest)
        pins=[p.paths()['inputs']/'train.pt',p.paths()['inputs']/'val.pt',ROOT/p.parent.R4_CHECKPOINT,
            p.midpoint_directory()/'train/head_epoch_20.pth',p.BASE/'lvsr/attempt_001/train/head_epoch_05.pth',p.BASE/'lvsr/attempt_001/train/selection.json']
        hashes={str(q.relative_to(ROOT)):io.sha(q) for q in pins}
        assert io.sha(pins[2])==p.parent.R4_SHA and io.sha(pins[3])==p.MIDPOINT_SHA
        assert io.sha(pins[4])=='b272c50dfe6329414803d5f22c6e1e84d9c3db37b6bb2e31f1c7ed140f662c3e'
        common=runpy.run_path(str(OUT.parent/'diagnose.py'))
        # Shared deterministic 4000 sample slots, no oversampling of one arm.
        schedule=[];epoch=1
        while len(schedule)<4000:
            order=list(range(len(fit)));random.Random(1703+epoch).shuffle(order);schedule.extend(order);epoch+=1
        schedule=schedule[:4000]
        schedule_sha=__import__('hashlib').sha256(json.dumps(schedule).encode()).hexdigest()
        events={};frozen_checks={};cp_info={};views=0;total_updates=0;initial_digest=None
        baseline_rows=None
        for arm in SPEC['arms']:
            random.seed(1703);torch.manual_seed(1703);head=PairSpecificSizeHead();params=configure(head,arm)
            digest=d.tensor_digest(head.state_dict())
            if initial_digest is None:initial_digest=digest
            assert digest==initial_digest and sum(p.numel() for p in head.parameters())==2480
            frozen={k:v.detach().clone() for k,v in head.state_dict().items()
                    if arm=='terminal_bias_only8' and not (k.startswith('terminal.') and k.endswith('.bias'))}
            if baseline_rows is None:
                baseline_rows=evaluate(head,records);views+=5999
                assert len(baseline_rows)==6003 and all(r['midpoint']==r['edge_residual'] for r in baseline_rows)
            base={(r['image'],r['scale']):r for r in baseline_rows}
            current={name:[base[r['image'],r['scale']] for r in rr] for name,rr in pools.items()}
            history=[dict(step=0,metrics={name:report(rr,common) for name,rr in current.items()})]
            opt=torch.optim.Adam(params,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
            assert not opt.state
            with (OUT/('small_fit_'+arm+'_server_only_progress.jsonl')).open('x') as log:
                for update in range(1,501):
                    rr=[fit[i] for i in schedule[(update-1)*8:update*8]]
                    st=step(head,opt,d.batch(rr));views+=8;total_updates+=1
                    assert all(int(opt.state[q]['step'])==update for q in params)
                    log.write(json.dumps(dict(step=update,**st),allow_nan=False)+'\n')
                    if update in SPEC['checkpoints']:
                        state_before=d.tensor_digest(head.state_dict());current={}
                        for name,rr in pools.items():
                            current[name]=evaluate(head,rr);views+=len(rr)
                        assert d.tensor_digest(head.state_dict())==state_before
                        event=dict(step=update,metrics={name:report(rr,common) for name,rr in current.items()},last_update=st)
                        history.append(event)
                        a=event['metrics']['fit_TRAIN128']['overall']['summary']['methods']['edge_residual']
                        print(json.dumps(dict(stage='diagnostic_fit',arm=arm,step=update,fit_long_MAE=a['long_MAE'],fit_short_MAE=a['short_MAE'],fit_joint=a['joint10'],elapsed=time.monotonic()-started)),flush=True)
                        log.flush()
                    if time.monotonic()-started>SPEC['budget_seconds']:raise RuntimeError('CPU budget exceeded; no retry')
            for k,v in frozen.items():assert torch.equal(v,head.state_dict()[k]),k
            frozen_checks[arm]=dict(unchanged_parameter_tensors=len(frozen),trainable_parameters=sum(q.numel() for q in params))
            checkpoint=OUT/('small_fit_'+arm+'_server_only_step500.pth')
            payload=dict(arm=arm,step=500,head_state={k:v.detach().clone() for k,v in head.state_dict().items()},optimizer_state=deepcopy(opt.state_dict()),initial_digest=initial_digest,schedule_sha256=schedule_sha,diagnostic_only=True,adoption=False)
            torch.save(payload,str(checkpoint));reload=torch.load(str(checkpoint),map_location='cpu')
            from crane_project.tools.run_port_size_native_train_v1 import equal
            assert equal(payload,reload,torch)
            clone=PairSpecificSizeHead();clone.load_state_dict(reload['head_state'],strict=True)
            assert d.tensor_digest(clone.state_dict())==d.tensor_digest(head.state_dict())
            a=evaluate(head,fit[:8]);b=evaluate(clone,fit[:8]);views+=16;assert a==b
            cp_info[arm]=dict(path=checkpoint.name,sha256=io.sha(checkpoint),step=500,diagnostic_only=True)
            events[arm]=history
        full=events['original_full2480'][-1]['metrics']['fit_TRAIN128']['overall']['summary']['methods']
        bias=events['terminal_bias_only8'][-1]['metrics']['fit_TRAIN128']['overall']['summary']['methods']
        clear=all(full['edge_residual'][e+'_MAE']<=.9*bias['edge_residual'][e+'_MAE'] and full['edge_residual'][e+'_MAE']<full['midpoint'][e+'_MAE'] for e in ('long','short')) and full['edge_residual']['joint10']>=bias['edge_residual']['joint10']
        assert all(io.sha(ROOT/q)==h for q,h in hashes.items()) and d.records_digest(records)==before_cache and p.checked_sources()==identity
        assert all(io.sha(ROOT/q)==h for q,h in sources.items()) and not torch.cuda.is_initialized()
        io.save(OUT/'small_fit_summary.json',dict(settings=SPEC,sample_counts=dict(fit=128,heldout_TRAIN_real_video=304,VAL=887),strata=strata,heldout_TRAIN_video=held,
            histories=events,clear_fit_signal_at_final=clear,final_checkpoints=cp_info,initial_digest=initial_digest,schedule_sha256=schedule_sha,frozen_checks=frozen_checks,
            interpretation='diagnostic-only real optimization; final500 fixed; no production replacement or TEST selection'))
        io.save(OUT/'small_fit_completion.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,optimizer_updates=total_updates,updates_per_arm=500,formal_training=False,TEST_access=False,
            elapsed_seconds=time.monotonic()-started,forward_views=views,parent_weights_unchanged=True,inputs_unchanged=True,original_selection_unchanged=True,sources=sources,pinned_SHA=hashes,
            fit_samples_manifest_sha256=io.sha(OUT/'small_fit_server_only_samples.json'),clear_fit_signal=clear,download_weights=False))
        files=['small_fit_'+v+'.json' for v in ('protocol','summary','completion')]
        io.save(OUT/'small_fit_manifest.json',{n:io.sha(OUT/n) for n in files})
        print(json.dumps(dict(completed=True,clear_fit_signal=clear,updates=total_updates,elapsed=time.monotonic()-started)),flush=True)
    except BaseException as error:
        io.save(OUT/'small_fit_failure.json',dict(error=repr(error),TEST=False,retry=False));raise

if __name__=='__main__':run()
