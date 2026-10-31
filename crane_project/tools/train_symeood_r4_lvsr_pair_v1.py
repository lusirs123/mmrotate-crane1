"""Fresh dual-branch LVSR on frozen R4/M20; fixed24 TRAIN/VAL, no TEST/depth."""
import argparse
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p


def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
def write_rows(path,rows):
    with gzip.open(str(path),'xt') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')


def runtime(gpu):
    from crane_project.tools import train_symeood_r4_midpoint_v1 as adapted
    adapted.resource_check(gpu,before=True)
    engine=adapted.make_engine(gpu);rt=engine.runtime(0)
    from crane_project.tools import run_symeood_110_lvsr_v1 as base
    from crane_project.tools.run_port_size_native_train_v1 import seed
    from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
    from crane_project.utils import port_size_qalign_v1 as vector
    torch=engine.torch;torch.set_num_threads(1);seed(torch)
    return engine,base,PairSpecificSizeHead,vector,rt


def status(folder,stage,**details):
    row=dict(stage=stage,pid=os.getpid(),time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        TEST=False,depth=False,**details)
    tmp=folder/'stage_status.tmp';tmp.write_text(json.dumps(row,indent=2));tmp.replace(folder/'stage_status.json')


def prepare():
    identity=p.checked_sources();paths=p.paths()
    paths['preparation'].mkdir(parents=True,exist_ok=False)
    save(paths['preparation']/'protocol.json',dict(identity=identity,settings=p.SETTINGS,gate=p.GATE,
        stages=['inputs','verify','train'],detector_updates=0,midpoint_updates=0,TEST=False,depth=False,
        old_LVSR_weights=False,shared_head_control=False,continuation=False,
        memory=p.parent.MEMORY,root=str(p.BASE.relative_to(ROOT))))
    print('R4_M20_LVSR_PREPARED_NO_GPU_OR_DATA_ACCESS',flush=True)


def build_inputs(paths,identity,engine,base,new_head,vector,rt):
    torch=engine.torch;dest=paths['inputs'];dest.mkdir()
    # Validated cache was generated from this exact frozen R4; never old110.
    root=p.midpoint_directory();parent_identity=identity['midpoint_training_identity']
    train,val,cache=engine.load_cache(root/'roi_cache',parent_identity)
    if rt!=cache['runtime']:raise ValueError('R4 runtime differs from cache')
    path=root/'train/head_epoch_20.pth'
    if p.sha(path)!=p.MIDPOINT_SHA:raise ValueError('Selected M20 checkpoint SHA differs')
    payload=torch.load(str(path),map_location='cpu');cp=read(root/'train/selection.json')['selected_checkpoint']
    if (payload['identity']!=parent_identity or payload['epoch']!=20 or payload['updates']!=17980
            or payload['frozen_b']!=p.parent.frontend_identity()
            or payload['context']['manifest_sha256']!=p.PINS['roi_cache/cache_manifest.json']
            or payload['context']['detector_state']!=cache['detector_state']):
        raise ValueError('M20/cache provenance differs')
    frozen=engine.new_head().cuda().eval().requires_grad_(False)
    frozen.load_state_dict(payload['head_state'],strict=True)
    before=engine.g.state_digest(frozen)
    if before!=cp['head_digest']:raise ValueError('M20 state digest differs')
    references={r['image']:r for r in map(json.loads,(root/'train/val_epoch_20.rows.jsonl').read_text().splitlines())}
    inventory=[]
    for name,records in (('train',train),('val',val)):
        with torch.no_grad():
            for index,r in enumerate(records):
                x=[r[k].cuda() for k in ('roi','support','boxes_original','boxes_model','scale_xy')]
                m=frozen(*x)['boxes_original'].detach().cpu()
                if len(m)!=len(r['boxes_original']) or not torch.equal(m[:,5],r['boxes_original'][:,5]):
                    raise ValueError('M output count/score changed')
                r['midpoint_original']=m
                r['shard']='val_s1' if r['role']=='val' else 'train_s1' if r['scale']==1. else 'train_s05'
                if name=='val':
                    ref=references[r['image']];actual=m[0].tolist() if len(m) else None
                    if actual is None or ref['midpoint'] is None:
                        if actual!=ref['midpoint']:raise ValueError('VAL output presence differs')
                    elif actual[5]!=ref['midpoint'][5] or not torch.allclose(m[0,:5],torch.tensor(ref['midpoint'][:5]),atol=1e-4,rtol=1e-6):
                        raise ValueError('Full VAL M20 replay differs')
                if len(m):
                    pair=vector.read_pair(*[r[k] for k in ('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original')])
                    if not bool((pair['support'].sum((1,2,3))>0).all()):raise ValueError('Natural output has no common support')
                    target=vector.residual.size_targets(r['gt_original'],m,r['boxes_model'])['log_residual']
                    if not bool(torch.isfinite(target).all()):raise ValueError('Nonfinite TRAIN/VAL size target')
                inventory.append(dict(image=r['image'],scale=r['scale'],split=r['split'],shard=r['shard'],
                    domain=r['domain'],midpoint=m[0].tolist() if len(m) else None))
                if index%200==0 or index==len(records)-1:
                    status(paths['attempt'],'inputs',split=name,done=index+1,total=len(records));engine.resource_check(0)
                    print('M20_INPUTS',name,index+1,'/',len(records),flush=True)
        records.sort(key=lambda r:(r['scale'],r['image']))
    if engine.g.state_digest(frozen)!=before or any(q.grad is not None or q.requires_grad for q in frozen.parameters()):
        raise ValueError('Frozen M20 changed during input generation')
    cache_sha=p.PINS['roi_cache/cache_manifest.json'];counts=dict(train_views=len(train),val_views=len(val),
        train_present=sum(bool(len(r['midpoint_original'])) for r in train),
        val_present=sum(bool(len(r['midpoint_original'])) for r in val),
        detector_feature_extractions=0,midpoint_forwards=6003,midpoint_updates=0)
    if counts['train_views']!=5116 or counts['val_views']!=887 or counts['train_present']!=5112 or counts['val_present']!=887:
        raise ValueError('R4/M20 view/output counts differ')
    counts['updates_per_epoch']=math.ceil(counts['train_present']/8)
    files={}
    for name,records in (('train',train),('val',val)):
        path=dest/(name+'.pt');engine.atomic_save(path,dict(identity=identity,cache_sha256=cache_sha,records=records))
        files[path.name]=p.sha(path)
    write_rows(dest/'midpoint_inputs.rows.jsonl.gz',inventory)
    save(dest/'completion.json',dict(status='R4_M20_LVSR_INPUTS_COMPLETE',identity=identity,
        files=files,cache_manifest_sha256=cache_sha,counts=counts,M20_state_before=before,M20_state_after=engine.g.state_digest(frozen),
        full_VAL_M20_replay=True,TRAIN_center_filter=False,old_cache=False,TEST=False,depth=False))
    print('R4_M20_LVSR_INPUTS_COMPLETE',json.dumps(counts),flush=True)


def load_inputs(paths,identity,engine):
    completion=read(paths['inputs']/'completion.json')
    if completion['identity']!=identity or completion['status']!='R4_M20_LVSR_INPUTS_COMPLETE':
        raise ValueError('Matching completed LVSR inputs required')
    result=[]
    for name in ('train','val'):
        path=paths['inputs']/(name+'.pt')
        if p.sha(path)!=completion['files'][path.name]:raise ValueError('LVSR input shard SHA differs')
        payload=engine.torch.load(str(path),map_location='cpu')
        if payload['identity']!=identity or payload['cache_sha256']!=p.PINS['roi_cache/cache_manifest.json']:
            raise ValueError('LVSR input provenance differs')
        records=payload['records']
        if len(records)!=completion['counts'][name+'_views']:raise ValueError('Input count differs')
        for r in records:
            if (r['role']=='val')!=(name=='val'):raise ValueError('TRAIN/VAL role leakage')
            if any(v.device.type!='cpu' or v.requires_grad or v.grad_fn is not None for v in
                (r[k] for k in ('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original','gt_original'))):
                raise ValueError('Detached CPU inputs required')
        result.append(records)
    return *result,completion


def summarize(rows):
    import numpy as np
    from crane_project.tools.eval_crane_offline import compute_riou
    from crane_project.tools.eval_port_geometry_ap_v1 import ap11
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import canonical,row_metrics,frozen_identity
    groups=dict(overall=rows,real=[r for r in rows if r['domain']=='real'],sim=[r for r in rows if r['domain']=='sim'])
    result={}
    for name,group in groups.items():
        cell=dict(frozen_identity=all(frozen_identity(r) for r in group))
        for method in ('b','midpoint','edge_residual'):
            values=[]
            for r in group:
                pred=r[method];riou=compute_riou(np.asarray(canonical(pred)),np.asarray(canonical(r['gt']))) if pred else 0.
                v=row_metrics(r['gt'],pred,riou)
                values.append(dict(v,score=pred[5] if pred else 0.))
            present=[r for r in values if r['output']]
            item=dict(frames=len(group),outputs=len(present),TP50=sum(r['riou']>=.5 for r in values),
                TP75=sum(r['riou']>=.75 for r in values),mean_RIoU=sum(r['riou'] for r in values)/len(group),
                joint10=sum(r['joint_size10'] for r in values),
                long_MAE=sum(r['long_relative_error'] for r in present)/len(present),
                short_MAE=sum(r['short_relative_error'] for r in present)/len(present),
                center_MAE_px=sum(r['center_error_px'] for r in present)/len(present),
                center_correct=sum(r['center_hit'] for r in values),
                pure_angle_RMSE_deg=math.sqrt(sum(r['angle_error_deg']**2 for r in present)/len(present)))
            ordered=np.argsort(-np.asarray([r['score'] for r in present],dtype=np.float32))
            for threshold in (.5,.75):
                hits=np.asarray([r['riou']>=threshold for r in present],dtype=np.float32)[ordered]
                tp=hits.cumsum();fp=(1-hits).cumsum()
                item['AP'+str(round(threshold*100))]=100*ap11(tp.astype(np.float64)/len(group),tp/np.maximum(tp+fp,np.finfo(np.float32).eps))
            cell[method]=item
        result[name]=cell
    return result


def verify(paths,identity,engine,base,new_head,train,val,inputs):
    from crane_project.tools.run_port_size_native_train_v1 import seed,rng_get,rng_set,equal
    torch=engine.torch;dest=paths['verify'];dest.mkdir()
    present=[r for r in train if len(r['midpoint_original'])]
    chosen=[r for d in ('real','sim') for sc in (.5,1.) for r in
        [v for v in present if v['domain']==d and v['scale']==sc][:4]]
    if len(chosen)!=16:raise ValueError('Need16 TRAIN trial views')
    vv=[r for d in ('real','sim') for r in [v for v in val if v['domain']==d][:2]]
    seed(torch);h=new_head().cuda();initial=engine.g.state_digest(h)
    if initial['parameter_count']!=2480:raise ValueError('Wrong LVSR architecture')
    neutral=base.evaluate(h,train+val,torch)
    if any(r['midpoint']!=r['edge_residual'] for r in neutral):raise ValueError('Full6003-view neutral LVSR changed M')
    baseline=summarize([r for r in neutral if r['split']=='val'])
    from crane_project.tools import eval_port_geometry_midpoint_formal_v1 as evaluator
    saved=read(p.midpoint_directory()/'train/selected_val_compare.json')
    for domain in ('overall','real','sim'):
        if abs(baseline[domain]['midpoint']['center_MAE_px']-saved['groups'][domain]['midpoint']['center_error_px']['mean'])>1e-6:
            raise ValueError('Baseline M geometry differs')
    opt=torch.optim.Adam(h.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    logs=[];branch_gradients=[]
    for index,rr in enumerate((chosen[:8],chosen[8:])):
        stats=base.update(h,opt,base.batch(rr,torch),'fixed_b',torch);logs.append(stats)
        norms=[dict(stem=math.sqrt(sum(float(q.grad.double().square().sum()) for q in s.parameters())),
            terminal=math.sqrt(sum(float(q.grad.double().square().sum()) for q in t.parameters())))
            for s,t in zip(h.stem.branches,h.terminal.branches)]
        if any(n['terminal']<=0 or index==1 and n['stem']<=0 for n in norms):raise ValueError('Missing pair-branch gradient')
        branch_gradients.append(norms)
    snapshot=dict(head=deepcopy(h.state_dict()),optimizer=deepcopy(opt.state_dict()),rng=rng_get(torch))
    engine.atomic_save(dest/'discarded_trial.pth',snapshot)
    loaded=torch.load(str(dest/'discarded_trial.pth'),map_location='cpu')
    clone=new_head().cuda();clone.load_state_dict(loaded['head'],strict=True)
    other=torch.optim.Adam(clone.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.);other.load_state_dict(loaded['optimizer'])
    if not equal(snapshot,loaded,torch) or base.evaluate(h,vv,torch)!=base.evaluate(clone,vv,torch):raise ValueError('Head/optimizer save reload differs')
    rng_set(snapshot['rng'],torch);a=base.update(h,opt,base.batch(chosen[:8],torch),'fixed_b',torch)
    rng_set(loaded['rng'],torch);b=base.update(clone,other,base.batch(chosen[:8],torch),'fixed_b',torch)
    if not equal(a,b,torch) or not equal(h.state_dict(),clone.state_dict(),torch) or not equal(opt.state_dict(),other.state_dict(),torch):
        raise ValueError('Exact optimizer continuation replay failed')
    save(dest/'baseline_VAL.json',baseline)
    save(dest/'completion.json',dict(status='R4_M20_LVSR_VERIFY_PASS_DISCARDED',identity=identity,
        input_completion_sha256=p.sha(paths['inputs']/'completion.json'),initial_state=initial,
        neutral_views=len(neutral),parameters=2480,actual_optimizer_updates=4,discarded=True,
        optimizer_restore_exact=True,branch_gradients=branch_gradients,losses=logs,
        resource=engine.resource_check(0),TEST=False,depth=False))
    print('R4_M20_LVSR_VERIFY_PASS_DISCARDED',flush=True)


def fit(paths,identity,engine,base,new_head,vector,train,val,inputs):
    from crane_project.tools.run_port_size_native_train_v1 import seed,rng_get,equal,near_joint
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity
    torch=engine.torch;dest=paths['train'];dest.mkdir()
    proof=read(paths['verify']/'completion.json')
    if (proof['identity']!=identity or proof['status']!='R4_M20_LVSR_VERIFY_PASS_DISCARDED'
            or proof['input_completion_sha256']!=p.sha(paths['inputs']/'completion.json') or not proof['discarded']):
        raise ValueError('Matching successful engineering trial required')
    seed(torch);h=new_head().cuda();initial=engine.g.state_digest(h)
    if initial!=proof['initial_state']:raise ValueError('Fresh initialization differs from discarded trial')
    opt=torch.optim.Adam(h.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    if opt.state:raise ValueError('Fresh Adam required')
    present=[r for r in train if len(r['midpoint_original'])];steps=math.ceil(len(present)/8)
    baseline=read(paths['verify']/'baseline_VAL.json')
    events=[];updates=0
    for epoch in range(1,25):
        tick=time.monotonic();order=base.schedule_indices(len(present),epoch);total=0.;maxgrad=0.
        status(paths['attempt'],'train',epoch=epoch,step=0,updates=updates)
        with (dest/'progress.jsonl').open('a') as stream:
            for start in range(0,len(order),8):
                rr=[present[i] for i in order[start:start+8]]
                stats=base.update(h,opt,base.batch(rr,torch),'fixed_b',torch);updates+=1
                if any(int(opt.state[q]['step'])!=updates for q in h.parameters()):raise ValueError('Adam step counter differs')
                total+=stats['loss'];maxgrad=max(maxgrad,stats['grad_norm'])
                stream.write(json.dumps(dict(epoch=epoch,step=start//8+1,updates=updates,batch_size=len(rr),**stats),allow_nan=False)+'\n')
                if start//8%100==0:
                    stream.flush();engine.resource_check(0)
                    status(paths['attempt'],'train',epoch=epoch,step=start//8+1,total_steps=steps,updates=updates)
            if updates!=epoch*steps:raise ValueError('Epoch budget differs')
        vr=base.evaluate(h,val,torch)
        if not all(frozen_identity(r) for r in vr):raise ValueError('Size head changed M center/angle/score/presence')
        metrics=summarize(vr)
        for domain in metrics:
            if metrics[domain]['b']!=baseline[domain]['b'] or metrics[domain]['midpoint']!=baseline[domain]['midpoint']:
                raise ValueError('Fixed VAL baseline changed')
        gate=p.assess(metrics)
        state={k:v.detach().cpu().clone() for k,v in h.state_dict().items()}
        payload=dict(protocol=p.VERSION,identity=identity,epoch=epoch,updates=updates,updates_per_epoch=steps,
            head_state=state,head_digest=engine.g.state_digest(h),optimizer_state=deepcopy(opt.state_dict()),rng=rng_get(torch),
            detector_sha256=p.parent.R4_SHA,midpoint_sha256=p.MIDPOINT_SHA,parameters=2480,
            input_completion_sha256=p.sha(paths['inputs']/'completion.json'),parent_checkpoint=None)
        path=dest/('head_epoch_%02d.pth'%epoch);engine.atomic_save(path,payload)
        if not equal(payload,torch.load(str(path),map_location='cpu'),torch):raise ValueError('Saved checkpoint replay differs')
        event=dict(epoch=epoch,updates=updates,updates_per_epoch=steps,metrics=metrics,eligibility=gate,
            selection_key=p.selection_key(metrics,epoch),checkpoint=dict(path=path.name,sha256=p.sha(path),epoch=epoch),
            near_edges=near_joint(vr),cross_edge=vector.cross_edge(vr),loss_mean=total/steps,max_grad_norm=maxgrad,
            resource=engine.resource_check(0),seconds=time.monotonic()-tick)
        save(dest/('epoch_%02d_VAL.json'%epoch),event);write_rows(dest/('epoch_%02d_VAL.rows.jsonl.gz'%epoch),vr)
        events.append(event)
        status(paths['attempt'],'epoch_complete',epoch=epoch,updates=updates,qualified=gate['pass_all'])
        print(json.dumps(dict(epoch=epoch,updates=updates,qualified=gate['pass_all'],loss=event['loss_mean'],
            VAL_AP75=metrics['overall']['edge_residual']['AP75'],joint10=metrics['overall']['edge_residual']['joint10'],seconds=event['seconds'])),flush=True)
    selection=dict(**p.choose(events),protocol=p.VERSION,identity=identity,gate=p.GATE,
        selected_checkpoint=None,representative_checkpoint=None,epochs=events,TEST=False,depth=False)
    chosen=next(e for e in events if e['epoch']==selection['representative_epoch'])
    selection['representative_checkpoint']=chosen['checkpoint']
    if selection['selected_epoch'] is not None:selection['selected_checkpoint']=chosen['checkpoint']
    replay=new_head().cuda();replay.load_state_dict(torch.load(str(dest/chosen['checkpoint']['path']),map_location='cpu')['head_state'],strict=True)
    if summarize(base.evaluate(replay,val,torch))!=chosen['metrics']:raise ValueError('Chosen head reload VAL differs')
    backups=paths['attempt']/'backups';backups.mkdir()
    for epoch in sorted({selection['representative_epoch'],24}):
        source=dest/('head_epoch_%02d.pth'%epoch);target=backups/source.name
        with source.open('rb') as src,target.open('xb') as dst:
            import shutil
            shutil.copyfileobj(src,dst)
        if p.sha(source)!=p.sha(target):raise ValueError('Backup SHA differs')
    if p.checked_sources()!=identity:raise ValueError('Source identity changed')
    save(dest/'selection.json',selection)
    save(dest/'completion.json',dict(status='R4_M20_LVSR_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED',identity=identity,
        epochs_completed=24,updates=updates,selected_epoch=selection['selected_epoch'],
        representative_epoch=selection['representative_epoch'],detector_updates=0,midpoint_updates=0,
        TEST=False,depth=False,automatic_adoption=False,continuation=False))
    print('R4_M20_LVSR_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED',flush=True)


def run(args):
    identity=p.checked_sources();paths=p.paths(args.attempt)
    if args.stage=='check':return dict(passed=True,identity=identity,training_started=False)
    if args.stage=='prepare':prepare();return dict(prepared=True,training_started=False)
    if read(paths['preparation']/'protocol.json')['identity']!=identity:raise ValueError('Prepared identity differs')
    if paths[args.stage].exists():raise ValueError('Preserve existing stage; no overwrite/retry')
    paths['attempt'].mkdir(parents=True,exist_ok=True)
    engine,base,new_head,vector,rt=runtime(args.gpu)
    try:
        if args.stage=='inputs':build_inputs(paths,identity,engine,base,new_head,vector,rt)
        else:
            train,val,inputs=load_inputs(paths,identity,engine)
            if args.stage=='verify':verify(paths,identity,engine,base,new_head,train,val,inputs)
            elif args.stage=='train':fit(paths,identity,engine,base,new_head,vector,train,val,inputs)
        if p.checked_sources()!=identity:raise ValueError('Source drift during stage')
        return dict(stage=args.stage,completed=True,TEST=False,depth=False)
    except BaseException as error:
        save(paths['attempt']/(args.stage+'_failure.json'),dict(error=repr(error),stage=args.stage,retry=False))
        raise


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--stage',choices=('check','prepare','inputs','verify','train'),default='check')
    result.add_argument('--gpu',type=int,choices=(2,3),default=2)
    result.add_argument('--attempt',default='attempt_001')
    return result


if __name__=='__main__':print(json.dumps(run(parser().parse_args()),indent=2))
