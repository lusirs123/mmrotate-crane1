"""One fresh24-epoch training-only ConR-inspired LVSR trial. Default CPU static check.

Reuses original batching/order with one training-only auxiliary objective, neutral/gradient/reload checks and VAL
evaluation. Frozen R4/M20 inputs are reused without copying or rebuilding.
"""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
CONFIG=ROOT/'crane_project/configs/crane_lvsr_r4_conr_v1.json'
SOURCES=('crane_project/configs/crane_lvsr_r4_conr_v1.json',
    'crane_project/utils/port_size_pair_conr_v1.py',
    'crane_project/tools/train_lvsr_conr_v1.py',
    'tests/test_lvsr_conr_v1.py',
    'work_dirs/lvsr_diagnostic_20261112_v1/diagnose.py')


def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def source_hashes():
    from crane_project.utils.symeood_r4_lvsr_protocol_v1 import sha
    return {n:sha(ROOT/n) for n in SOURCES}


def assess(metrics,l5):
    """Frozen before run; L5 adoption distinct from M increment and R4 chain."""
    checks={}
    for domain in ('overall','real','sim'):
        v=metrics[domain];m=v['midpoint'];c=v['edge_residual'];b=l5[domain]['edge_residual']
        checks[domain+'/identity']=v['frozen_identity']
        checks[domain+'/joint_vs_L5']=c['joint10']>b['joint10'] if domain=='overall' else c['joint10']>=b['joint10']
        for edge in ('long','short'):checks[domain+'/'+edge+'_MAE_vs_L5']=c[edge+'_MAE']<=b[edge+'_MAE']+1e-6
        checks[domain+'/AP75_vs_M_L5']=c['AP75']>=max(m['AP75'],b['AP75'])-.0001
        checks[domain+'/RIoU_vs_M_L5']=c['mean_RIoU']>=max(m['mean_RIoU'],b['mean_RIoU'])-1e-5
        checks[domain+'/TP50_vs_M_L5']=c['TP50']>=max(m['TP50'],b['TP50'])
    return dict(pass_all=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v])


def increment_gate(metrics):
    checks={}
    for d in ('overall','real','sim'):
        v=metrics[d];a=v['edge_residual'];b=v['midpoint'];checks[d+'/identity']=v['frozen_identity']
        checks[d+'/joint']=a['joint10']>b['joint10'] if d=='overall' else a['joint10']>=b['joint10']
        for e in ('long','short'):checks[d+'/'+e]=a[e+'_MAE']<=b[e+'_MAE']+1e-6
        checks[d+'/AP75']=a['AP75']>=b['AP75']-.0001
        checks[d+'/RIoU']=a['mean_RIoU']>=b['mean_RIoU']-1e-5
        checks[d+'/TP50']=a['TP50']>=b['TP50']
    return dict(pass_all=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v])


def choose(events):
    eligible=[e for e in events if e['eligibility_vs_L5']['pass_all']]
    best=min(events,key=lambda e:e['selection_key'])
    passed=best['eligibility_vs_L5']['pass_all']
    return dict(selected_epoch=best['epoch'] if passed else None,representative_epoch=best['epoch'],
        eligible_epochs=[e['epoch'] for e in eligible],selected_checkpoint=best['checkpoint'] if passed else None,
        representative_checkpoint=best['checkpoint'],automatic_adoption=False,selection_on_test=False,
        representative_policy='all24 epochs predeclared ranking; gates diagnostic only')


def check_config():
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    cfg=read(CONFIG)
    for k in ('epochs','batch_size','seed','lr','weight_decay','betas','eps','clip_norm','parameters','alignment'):
        if cfg[k]!=p.SETTINGS[k]:raise ValueError('Single-factor training setting differs: '+k)
    if cfg['TEST'] or cfg['continuation'] or cfg['physical_gpu']!=2:raise ValueError('Protocol boundary differs')
    if cfg['head']!='PairConRSizeHead':raise ValueError('Wrong head')
    from crane_project.utils.port_size_pair_conr_v1 import CONR_WEIGHT,TEMPERATURE,WINDOW,PUSHING
    if cfg['conr']!=dict(weight=CONR_WEIGHT,temperature=TEMPERATURE,window_log=WINDOW,pushing=PUSHING):raise ValueError('ConR settings differ')
    return cfg


def report_rows(rows,common):
    subsets={'val':[r for r in rows if r['split']=='val'],
        'train_s1':[r for r in rows if r['split']!='val' and r['scale']==1.],
        'train_s05':[r for r in rows if r['split']!='val' and r['scale']==.5]}
    return {k:common['grouped'](v) for k,v in subsets.items()}


def run(args):
    cfg=check_config();hashes=source_hashes()
    if args.stage=='check':
        return dict(passed=True,settings=cfg,sources=hashes,training_started=False)
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    from crane_project.tools import train_symeood_r4_lvsr_pair_v1 as original
    from crane_project.tools import run_symeood_110_lvsr_v1 as legacy
    from crane_project.utils.port_size_pair_conr_v1 import update
    from types import SimpleNamespace
    base=SimpleNamespace(evaluate=legacy.evaluate,batch=legacy.batch,schedule_indices=legacy.schedule_indices,update=update)
    from crane_project.tools.run_port_size_native_train_v1 import seed,rng_get,equal,near_joint
    parent=p.checked_sources();root=ROOT/cfg['output_root'];root.mkdir(exist_ok=False)
    source=p.BASE/'lvsr/attempt_001';started=time.monotonic()
    identity=dict(protocol=cfg['protocol'],sources=hashes,settings=cfg,parent_identity=parent)
    save(root/'protocol.json',identity)
    def status(stage,**extra):
        value=dict(stage=stage,pid=os.getpid(),time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),**extra)
        temporary=root/'status.tmp';temporary.write_text(json.dumps(value));temporary.replace(root/'status.json')
        print(json.dumps(value),flush=True)
    try:
        status('engineering')
        engine,_,_,vector,runtime=original.runtime(2);torch=engine.torch
        from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
        from crane_project.utils.port_size_pair_conr_v1 import PairConRSizeHead
        resources=[]
        def resource():
            v=engine.resource_check(0)
            import subprocess
            q=subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
            v['GPU2_compute_pids']=[int(x.strip()) for x in q.splitlines() if x.strip()]
            v['coexistence_user_authorized']=True
            resources.append(v);return v
        # Keep the engine's existing resource policy; add contention separately.
        train,val,inputs=original.load_inputs(p.paths(),parent,engine)
        if inputs['counts']['train_present']!=5112 or len(val)!=887:raise ValueError('Wrong inputs')
        import runpy
        common=runpy.run_path(str(ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/diagnose.py'))
        records=train+val;memory_before=common['memory_digest'](records)
        old_selection=read(source/'train/selection.json')
        l5_event=read(source/'train/epoch_05_VAL.json');l5=l5_event['metrics']
        pins=[ROOT/p.parent.R4_CHECKPOINT,p.midpoint_directory()/'train/head_epoch_20.pth',
            source/'train/head_epoch_05.pth',source/'train/head_epoch_24.pth',
            source/'inputs/train.pt',source/'inputs/val.pt',source/'train/selection.json']
        frozen_sha={str(q.relative_to(ROOT)):p.sha(q) for q in pins}
        if p.sha(pins[0])!=p.parent.R4_SHA or p.sha(pins[1])!=p.MIDPOINT_SHA:raise ValueError('Parents differ')
        if p.sha(pins[2])!='b272c50dfe6329414803d5f22c6e1e84d9c3db37b6bb2e31f1c7ed140f662c3e':raise ValueError('L5 differs')
        if old_selection['selected_epoch'] is not None:raise ValueError('Preserve original L5 null selection')
        # Nonneutral legacy weights verify that eval has exactly the original graph.
        trial=[r for d in ('real','sim') for sc in (.5,1.) for r in [v for v in train if v['domain']==d and v['scale']==sc and len(v['midpoint_original'])][:4]]
        loaded=torch.load(str(pins[2]),map_location='cpu')
        control=PairConRSizeHead().cuda().eval();control.load_state_dict(loaded['head_state'],strict=True)
        baseline_head=PairSpecificSizeHead().cuda().eval();baseline_head.load_state_dict(loaded['head_state'],strict=True)
        a=base.evaluate(baseline_head,trial,torch);b=base.evaluate(control,trial,torch)
        max_control=max(abs(x-y) for r,s in zip(a,b) for x,y in zip(r['edge_residual'],s['edge_residual']))
        if max_control>1e-4 or any(r['size_delivery']!=s['size_delivery'] or r['failed_checks']!=s['failed_checks'] for r,s in zip(a,b)):
            raise ValueError('Inference control differs from original')
        del loaded,control,baseline_head,a,b
        seed(torch);old=PairSpecificSizeHead().cuda();old_state={k:v.detach().cpu().clone() for k,v in old.state_dict().items()};rng=torch.get_rng_state()
        seed(torch);candidate=PairConRSizeHead().cuda()
        if not equal(old_state,candidate.state_dict(),torch) or not torch.equal(rng,torch.get_rng_state()):raise ValueError('Initialization differs')
        del old,candidate
        paths=dict(attempt=root,verify=root/'verification',inputs=source/'inputs')
        original.verify(paths,parent,engine,base,PairConRSizeHead,train,val,inputs)
        proof=read(paths['verify']/'completion.json')
        if source_hashes()!=hashes:raise ValueError('Sources changed during engineering')
        save(root/'engineering.json',dict(passed=True,head=cfg['head'],initial_parameters_equal=True,initial_rng_equal=True,
            inference_control_frames=len(trial),inference_control_max_abs=max_control,neutral_views=6003,
            trial_optimizer_updates=4,trial_discarded=True,original_engineering_receipt='verification/completion.json'))
        seed(torch);head=PairConRSizeHead().cuda()
        if engine.g.state_digest(head)!=proof['initial_state']:raise ValueError('Trial state reused or fresh seed changed')
        opt=torch.optim.Adam(head.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'])
        if opt.state:raise ValueError('Fresh Adam required')
        present=[r for r in train if len(r['midpoint_original'])];steps=math.ceil(len(present)/cfg['batch_size'])
        if steps!=639:raise ValueError('Wrong update budget')
        baseline=read(paths['verify']/'baseline_VAL.json');events=[];updates=0
        for epoch in range(1,25):
            tick=time.monotonic();order=base.schedule_indices(len(present),epoch);total=0.;maxgrad=0.
            aux_total=0.;size_total=0.;anchors=0;active=0;positive=0;negative=0;zeros=0
            status('train',epoch=epoch,step=0,updates=updates)
            with (root/'progress.jsonl').open('a') as stream:
                for start in range(0,len(order),8):
                    rr=[present[i] for i in order[start:start+8]]
                    st=base.update(head,opt,base.batch(rr,torch),'fixed_b',torch);updates+=1
                    if any(int(opt.state[q]['step'])!=updates for q in head.parameters()):raise ValueError('Adam counter differs')
                    if any(g['lr']!=.001 or g['weight_decay']!=0. for g in opt.param_groups):raise ValueError('Optimizer schedule drift')
                    total+=st['loss'];maxgrad=max(maxgrad,st['grad_norm'])
                    aux_total+=st['conr_loss'];size_total+=st['size_loss'];anchors+=st['anchors'];active+=st['active_anchors']
                    positive+=st['positive_pairs'];negative+=st['negative_pairs'];zeros+=st['zero_embeddings']
                    stream.write(json.dumps(dict(epoch=epoch,step=start//8+1,updates=updates,**st),allow_nan=False)+'\n')
                    if start//8%100==0:
                        stream.flush();resource();status('train',epoch=epoch,step=start//8+1,total_steps=steps,updates=updates)
            if updates!=epoch*639:raise ValueError('Update budget differs')
            vr=base.evaluate(head,val,torch);metrics=original.summarize(vr)
            for domain in metrics:
                if not metrics[domain]['frozen_identity']:raise ValueError('Frozen M identity changed')
                for arm in ('b','midpoint'):
                    if metrics[domain][arm]!=baseline[domain][arm]:raise ValueError('Upstream metrics changed')
            cp=root/('head_epoch_%02d.pth'%epoch)
            payload=dict(protocol=cfg['protocol'],head_type=cfg['head'],identity=identity,epoch=epoch,updates=updates,
                head_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},optimizer_state=deepcopy(opt.state_dict()),
                rng=rng_get(torch),parameters=2480,detector_sha256=p.parent.R4_SHA,midpoint_sha256=p.MIDPOINT_SHA,
                parent_checkpoint=None,input_completion_sha256=p.sha(source/'inputs/completion.json'))
            engine.atomic_save(cp,payload)
            if not equal(payload,torch.load(str(cp),map_location='cpu'),torch):raise ValueError('Checkpoint reload differs')
            event=dict(epoch=epoch,updates=updates,metrics=metrics,eligibility_vs_L5=assess(metrics,l5),increment_vs_M=increment_gate(metrics),
                original_chain_gate=p.assess(metrics),selection_key=p.selection_key(metrics,epoch),
                checkpoint=dict(path=cp.name,sha256=p.sha(cp),epoch=epoch),near_edges=near_joint(vr),cross_edge=vector.cross_edge(vr),
                auxiliary=dict(conr_loss_mean=aux_total/steps,size_loss_mean=size_total/steps,anchors=anchors,active_anchors=active,
                    active_fraction=active/anchors,positive_pairs=positive,negative_pairs=negative,zero_embeddings=zeros),
                loss_mean=total/steps,max_grad_norm=maxgrad,seconds=time.monotonic()-tick,resource=resource())
            save(root/('epoch_%02d_VAL.json'%epoch),event);original.write_rows(root/('epoch_%02d_VAL.rows.jsonl.gz'%epoch),vr)
            events.append(event)
            if source_hashes()!=hashes:raise ValueError('Trial source drift')
            a=metrics['overall']['edge_residual']
            status('epoch_complete',epoch=epoch,updates=updates,loss=event['loss_mean'],long_MAE=a['long_MAE'],short_MAE=a['short_MAE'],
                joint10=a['joint10'],active_fraction=event['auxiliary']['active_fraction'],eligible_vs_L5=event['eligibility_vs_L5']['pass_all'],increment_M=event['increment_vs_M']['pass_all'],seconds=event['seconds'])
        selection=choose(events);save(root/'selection.json',dict(**selection,protocol=cfg['protocol'],rules=cfg['selection']))
        chosen=next(e for e in events if e['epoch']==selection['representative_epoch'])
        status('final_analysis',representative_epoch=chosen['epoch'])
        replay=PairConRSizeHead().cuda();loaded=torch.load(str(root/chosen['checkpoint']['path']),map_location='cpu')
        replay.load_state_dict(loaded['head_state'],strict=True)
        if original.summarize(base.evaluate(replay,val,torch))!=chosen['metrics']:raise ValueError('Representative reload VAL differs')
        final_rows=base.evaluate(replay,records,torch)
        from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity
        if not all(frozen_identity(r) for r in final_rows):raise ValueError('Final TRAIN/VAL identity differs')
        original.write_rows(root/'representative_TRAIN_VAL.rows.jsonl.gz',final_rows)
        detail=report_rows(final_rows,common)
        save(root/'core_detail.json',dict(epoch=chosen['epoch'],metrics=detail))
        if common['memory_digest'](records)!=memory_before:raise ValueError('Input tensors changed')
        if any(p.sha(ROOT/n)!=h for n,h in frozen_sha.items()):raise ValueError('Frozen upstream or L5 files changed')
        if p.checked_sources()!=parent or source_hashes()!=hashes:raise ValueError('Final source identity drift')
        save(root/'core_summary.json',dict(protocol=cfg['protocol'],settings=cfg,selection=selection,representative=chosen,
            baseline_L5=l5_event,epoch24=events[-1],epochs=events,source_SHA=hashes,upstream_SHA=frozen_sha,
            original_L5_selected_epoch=None,weight_download=False,TEST=False))
        columns=['epoch','loss','long_MAE_pct','short_MAE_pct','joint10','RIoU','AP75','real_joint','sim_joint','eligible_L5','increment_M','original_chain']
        with (root/'core_epochs.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
            for e in events:
                a=e['metrics']['overall']['edge_residual'];w.writerow(dict(epoch=e['epoch'],loss=e['loss_mean'],long_MAE_pct=100*a['long_MAE'],
                    short_MAE_pct=100*a['short_MAE'],joint10=a['joint10'],RIoU=a['mean_RIoU'],AP75=a['AP75'],
                    real_joint=e['metrics']['real']['edge_residual']['joint10'],sim_joint=e['metrics']['sim']['edge_residual']['joint10'],
                    eligible_L5=e['eligibility_vs_L5']['pass_all'],increment_M=e['increment_vs_M']['pass_all'],original_chain=e['original_chain_gate']['pass_all']))
        save(root/'core_completion.json',dict(passed=True,epochs_completed=24,formal_optimizer_updates=updates,discarded_engineering_updates=4,
            selected_epoch=selection['selected_epoch'],representative_epoch=selection['representative_epoch'],
            checkpoint=chosen['checkpoint'],head_type=cfg['head'],parents_unchanged=True,inputs_unchanged=True,source_SHA=hashes,
            detector_updates=0,midpoint_updates=0,TEST=False,automatic_adoption=False,original_L5_selection_unchanged=True,
            runtime=runtime,resources=resources,elapsed_seconds=time.monotonic()-started))
        core=['core_summary.json','core_detail.json','core_epochs.csv','core_completion.json','engineering.json','protocol.json','selection.json']
        save(root/'core_artifacts.json',{n:p.sha(root/n) for n in core})
        status('COMPLETE',selected_epoch=selection['selected_epoch'],representative_epoch=selection['representative_epoch'],updates=updates)
        return dict(completed=True,selected_epoch=selection['selected_epoch'],TEST=False)
    except BaseException as error:
        save(root/'failure.json',dict(error=repr(error),TEST=False,automatic_retry=False));status('FAILED',error=repr(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('check','run'),default='check')
    print(json.dumps(run(parser.parse_args()),ensure_ascii=False,indent=2))
