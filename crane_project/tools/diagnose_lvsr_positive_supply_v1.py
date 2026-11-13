"""CPU-only admission check for ONE conditional positive-supply trial."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/conr_001'
SPEC=dict(version='positive_supply_admission_20261112_v1',CPU_threads=1,GPU=False,optimizer=False,updates=0,TEST=False,budget_seconds=600,
    pair_rule='only missing-native-positive anchors; one external TRAIN positive; different image, same-video gap>30; cross-video preferred then nearest Linf distance then index; exclude main-batch images',
    unchanged='main8,size loss,negative pool,loss weight .001,temperature .07,window log1.02,pushing .01; no new parameters',
    coverage='all639 unchanged epoch6 batches on sealed L5 and ConR5 pre-delivery outputs',
    gradients='first16 natural epoch6 batches plus all TRAIN views with either M-edge error>10%, sorted by domain/scale/sequence/frame/image, batch8; two frozen snapshots',
    gate='strict predeclared conjunction: identities valid; hard long and short active anchors both increase on both snapshots; for each snapshot and natural/hard pool, mean whole-size first-order derivative under unit auxiliary descent <=0 and mean long/short-shrink derivative <=0, each with >=8 observations; each shrink derivative no worse than original within1e-10',
    consequence='if any gate fails: no training, no hyperparameter or pairing search; preserve failure',
    interpretation='first-order SGD-direction probe on frozen states, not Adam trajectory or validation gain; admission evidence only')


def run():
    import torch
    torch.set_num_threads(1)
    from crane_project.utils import lvsr_conr_positive_supply_v1 as supply
    from crane_project.utils import port_size_pair_conr_v1 as c
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    from crane_project.tools import train_symeood_r4_lvsr_pair_v1 as trainer
    from crane_project.tools import diagnose_lvsr_direction_representation_v1 as d
    from crane_project.tools import diagnose_lvsr_conr_failure_v1 as audit
    from crane_project.tools.run_symeood_110_lvsr_v1 import schedule_indices
    from types import SimpleNamespace
    started=time.monotonic();identity=p.checked_sources()
    sources={str(Path(__file__).resolve().relative_to(ROOT)):audit.sha(__file__),
        'crane_project/utils/lvsr_conr_positive_supply_v1.py':audit.sha(ROOT/'crane_project/utils/lvsr_conr_positive_supply_v1.py'),
        'tests/test_lvsr_conr_positive_supply_v1.py':audit.sha(ROOT/'tests/test_lvsr_conr_positive_supply_v1.py'),
        'crane_project/utils/port_size_pair_conr_v1.py':audit.sha(ROOT/'crane_project/utils/port_size_pair_conr_v1.py')}
    audit.save(OUT/'positive_supply_protocol.json',dict(**SPEC,sources=sources))
    old_init=torch.optim.Optimizer.__init__
    def forbidden(*a,**kw):raise RuntimeError('Admission prohibits optimizer')
    torch.optim.Optimizer.__init__=forbidden
    try:
        train,val,_=trainer.load_inputs(p.paths(),identity,SimpleNamespace(torch=torch));records=[r for r in train if len(r['midpoint_original'])]
        memory=d.records_digest(train+val);idx=supply.PositiveIndex(records);order=schedule_indices(len(records),6)
        errors=torch.stack([(r['midpoint_original'][0,2:4].sort(descending=True)[0]/r['gt_original'][0,2:4].sort(descending=True)[0]-1).abs() for r in records])
        hard=[i for i in range(len(records)) if bool((errors[i]>.1).any())]
        hard.sort(key=lambda i:(records[i]['domain'],records[i]['scale'],records[i]['sequence'],records[i]['frame_id'],records[i]['image']))
        pools={'natural':[order[i:i+8] for i in range(0,128,8)],'hard':[hard[i:i+8] for i in range(0,len(hard),8)]}
        paths={'L5':p.BASE/'lvsr/attempt_001/train/head_epoch_05.pth','ConR5':OUT/'head_epoch_05.pth'}
        saved_paths={'L5':OUT.parent/'attempt_001/L5_all.rows.jsonl.gz','ConR5':OUT/'representative_TRAIN_VAL.rows.jsonl.gz'}
        pins={str(q.relative_to(ROOT)):audit.sha(q) for q in list(paths.values())+list(saved_paths.values())+[p.paths()['inputs']/'train.pt',p.paths()['inputs']/'val.pt']}
        coverage=[];samples=[];views=0;max_replay=0.;index_detail=defaultdict(int)
        for i,candidates in enumerate(idx.candidates):
            index_detail['n']+=1;index_detail['with_candidate']+=int(bool(candidates))
            for edge,name in enumerate(('long','short')):
                if errors[i,edge]>.1:
                    index_detail[name+'_hard_n']+=1;index_detail[name+'_hard_with_candidate']+=int(bool(candidates))
                    index_detail[name+'_hard_with_crossvideo']+=int(any(records[j]['sequence']!=records[i]['sequence'] for j in candidates))
        for label,path in paths.items():
            saved={(q['image'],q['scale']):q for q in audit.load_rows(saved_paths[label])}
            prediction=torch.stack([c.canonical_pair(torch.tensor(saved[r['image'],r['scale']]['delta_roi']).reshape(1,2),r['boxes_model'],r['midpoint_original'])[0] for r in records])
            counts=defaultdict(lambda:defaultdict(int))
            for start in range(0,len(order),8):
                ii=order[start:start+8];tt=idx.targets[ii];pp=prediction[ii]
                pos,neg,active=audit.masks(tt,pp,c.WINDOW);chosen=idx.select(ii)
                modified=pos.any(1).clone();role_by_anchor={a:chosen['roles'][k] for k,(a,b) in enumerate(chosen['links'])}
                for a,b in chosen['links']:
                    assert not pos[a].any();modified[a]=True
                    mate=chosen['partner_indices'][b]
                    assert records[mate]['image']!=records[ii[a]]['image']
                    assert float((idx.targets[mate]-tt[a]).abs().max())<=c.WINDOW
                now=modified&neg.any(1)
                for j,i in enumerate(ii):
                    groups=['overall',records[i]['domain']]
                    for edge,name in enumerate(('long','short')):
                        if errors[i,edge]>.1:groups.append(name+'_M_gt10pct')
                        if errors[i,edge]<=.03:groups.append(name+'_accurate_M_le3pct')
                        if tt[j,edge]<math.log(.99):groups.append(name+'_shrink')
                    for group in groups:
                        a=counts[group];a['n']+=1;a['old_active']+=int(active[j]);a['new_active']+=int(now[j]);a['native_positive']+=int(pos[j].any())
                        a['external_positive']+=int(j in role_by_anchor);a['external_crossvideo']+=int(role_by_anchor.get(j)=='cross_video');a['external_samevideo_gap30']+=int(role_by_anchor.get(j)=='same_video_gap30')
                        a['no_positive_after']+=int(not modified[j])
            coverage.extend(dict(snapshot=label,group=k,**v) for k,v in sorted(counts.items()))
            head=c.PairConRSizeHead().train();head.load_state_dict(torch.load(str(path),map_location='cpu')['head_state'],strict=True)
            state=d.tensor_digest(head.state_dict());params=list(head.parameters())
            flat=lambda vv:torch.cat([(torch.zeros_like(q) if v is None else v).reshape(-1) for v,q in zip(vv,params)]).double()
            for pool,batches in pools.items():
                for bi,ii in enumerate(batches):
                    rr=[records[i] for i in ii];x=d.batch(rr);main=head(*x[:-1]);views+=len(ii)
                    for j,r in enumerate(rr):
                        expected=torch.tensor(saved[r['image'],r['scale']]['edge_residual'])
                        max_replay=max(max_replay,float((main['boxes_original'][j]-expected).abs().max()))
                        assert torch.allclose(main['boxes_original'][j],expected,atol=1e-4,rtol=1e-6)
                    target=c.canonical_pair(c.model.residual.size_targets(x[-1],x[-2],x[3])['log_residual'],x[3],x[-2]);pred=c.canonical_pair(main['delta_roi'],x[3],x[-2])
                    size=c.model.size_loss(main,x[-1],x[-2],x[3]);gs=flat(torch.autograd.grad(size,params,retain_graph=True))
                    old,oldstats=c.contrastive_loss(main['training_features'],main['common_support'],target,pred)
                    selected=idx.select(ii);partner=None;pt=None
                    if selected['partner_indices']:
                        xp=d.batch([records[i] for i in selected['partner_indices']]);partner=head(*xp[:-1]);views+=len(selected['partner_indices'])
                        pt=idx.targets[selected['partner_indices']]
                    new,stats=supply.loss(main,target,pred,partner,pt,selected['links'])
                    assert stats['negative_pairs']==oldstats['negative_pairs']
                    for name,aux,st in [('old',old,oldstats),('supplied',new,stats)]:
                        gg=flat(torch.autograd.grad(c.CONR_WEIGHT*aux,params,retain_graph=True,allow_unused=True));gn=float(gg.norm())
                        unit=-gg/gn if gn else gg
                        value=dict(snapshot=label,pool=pool,batch=bi+1,arm=name,n=len(ii),active=st['active_anchors'],external_views=len(selected['partner_indices']) if name=='supplied' else 0,
                            auxiliary_loss=float(aux.detach()),gradient_norm=gn,size_loss_unit_derivative=float(gs.dot(unit)),groups={})
                        for edge,ename in enumerate(('long','short')):
                            for group,mask in [('shrink',target[:,edge]<math.log(.99)),('M_gt10pct',errors[ii,edge]>.1),('accurate_M_le3pct',errors[ii,edge]<=.03)]:
                                if not bool(mask.any()):continue
                                edge_loss=torch.nn.functional.smooth_l1_loss(pred[mask,edge],target[mask,edge],beta=.1)
                                grad=flat(torch.autograd.grad(edge_loss,params,retain_graph=True))
                                jac=flat(torch.autograd.grad(pred[mask,edge].mean(),params,retain_graph=True))
                                value['groups'][ename+'_'+group]=dict(n=int(mask.sum()),unit_loss_derivative=float(grad.dot(unit)),unit_log_size_delta=float(jac.dot(unit)))
                        samples.append(value)
                    assert all(q.grad is None for q in params)
                    if time.monotonic()-started>SPEC['budget_seconds']:raise RuntimeError('CPU budget exceeded')
            assert d.tensor_digest(head.state_dict())==state
            print(json.dumps(dict(stage='frozen_check',snapshot=label,batches_per_arm=sum(map(len,pools.values())),elapsed=time.monotonic()-started)),flush=True)
        aggregate=[];checks={}
        for label in paths:
            for edge in ('long','short'):
                a=next(q for q in coverage if q['snapshot']==label and q['group']==edge+'_M_gt10pct')
                checks[label+'/'+edge+'_hard_active_increases']=a['new_active']>a['old_active']
            for pool in pools:
                v={}
                for arm in ('old','supplied'):
                    rr=[q for q in samples if q['snapshot']==label and q['pool']==pool and q['arm']==arm]
                    n=sum(q['n'] for q in rr);a=dict(snapshot=label,pool=pool,arm=arm,n=n,whole_size_unit_derivative=sum(q['n']*q['size_loss_unit_derivative'] for q in rr)/n,groups={})
                    for group in sorted({g for q in rr for g in q['groups']}):
                        gg=[q['groups'][group] for q in rr if group in q['groups']];nn=sum(q['n'] for q in gg)
                        a['groups'][group]=dict(n=nn,unit_loss_derivative=sum(q['n']*q['unit_loss_derivative'] for q in gg)/nn,unit_log_size_delta=sum(q['n']*q['unit_log_size_delta'] for q in gg)/nn)
                    aggregate.append(a);v[arm]=a
                checks[label+'/'+pool+'/whole_size_nonharm']=v['supplied']['whole_size_unit_derivative']<=1e-10
                for edge in ('long','short'):
                    a=v['supplied']['groups'].get(edge+'_shrink',{});b=v['old']['groups'].get(edge+'_shrink',{})
                    checks[label+'/'+pool+'/'+edge+'_shrink_supported']=a.get('n',0)>=8
                    checks[label+'/'+pool+'/'+edge+'_shrink_nonharm']=a.get('unit_loss_derivative',float('inf'))<=1e-10
                    checks[label+'/'+pool+'/'+edge+'_shrink_no_worse_than_old']=a.get('unit_loss_derivative',float('inf'))<=b.get('unit_loss_derivative',-float('inf'))+1e-10
        admitted=all(checks.values());assert all(audit.sha(ROOT/q)==h for q,h in pins.items()) and d.records_digest(train+val)==memory
        assert all(audit.sha(ROOT/q)==h for q,h in sources.items()) and p.checked_sources()==identity
        assert not torch.cuda.is_initialized()
        audit.save(OUT/'positive_supply_summary.json',dict(index=dict(index_detail),coverage=coverage,gradient_batches=samples,aggregate=aggregate,checks=checks,admitted=admitted,failed=[k for k,v in checks.items() if not v],CPU_replay_max_abs=max_replay))
        audit.save(OUT/'positive_supply_completion.json',dict(passed=True,admitted_to_training=admitted,CPU_only=True,CUDA_initialized=False,optimizer_created=False,training_updates=0,TEST_access=False,
            elapsed_seconds=time.monotonic()-started,forward_views=views,weights_unchanged=True,inputs_unchanged=True,sources=sources,pinned_SHA=pins,
            hard_views=len(hard),natural_views=128,original_selection_unchanged=True,automatic_search=False))
        files=['positive_supply_'+s+'.json' for s in ('protocol','summary','completion')]
        audit.save(OUT/'positive_supply_manifest.json',{n:audit.sha(OUT/n) for n in files})
        print(json.dumps(dict(completed=True,admitted=admitted,failed=[k for k,v in checks.items() if not v],elapsed=time.monotonic()-started)),flush=True)
    except BaseException as error:
        audit.save(OUT/'positive_supply_failure.json',dict(error=repr(error),training_updates=0,TEST=False,retry=False));raise
    finally:torch.optim.Optimizer.__init__=old_init

if __name__=='__main__':run()
