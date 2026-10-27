#!/usr/bin/env python3
"""New110+M21 LVSR: copied fixed_b math, conditional original-loss C continuation."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools.run_port_size_native_train_v1 import (
    sha,read,write,rows_write,seed,rng_get,rng_set,equal,near_joint,eligibility,ranking)
VERSION='symeood_110_lvsr_v1'
CONTRACT=ROOT/'crane_project/data_contracts/symeood_110_lvsr_v1.json'
FORMAL=ROOT/'work_dirs/symeood_110_retrain_seed0_20261026_v1/downstream/midpoint/formal'
DETECTOR_SHA='7f35dacc308943949a8fd2e6aaad312d14651564903ec87c8c8dcec0552ec054'
MIDPOINT_SHA='7aaaacc6c666116e466c9a99ac59d71acf7f77312dc86de224fcf965569fd9ac'


def checked_sources():
    from crane_project.tools import train_symeood_110_midpoint_v1 as f
    value=read(CONTRACT)
    if (value['protocol']!=VERSION or value['detector_sha256']!=DETECTOR_SHA or
            value['midpoint_sha256']!=MIDPOINT_SHA or value['common_arms']!=['fixed_b'] or
            value['continuation_arms']!=['C'] or value['common_epochs']!=24 or value['continuation_epochs']!=6):
        raise ValueError('Frozen new110 LVSR contract differs')
    for name,h in value['sources'].items():
        if sha(ROOT/name)!=h:raise ValueError('LVSR source differs: '+name)
    f.checked_sources()
    for name,h in value['input_metadata'].items():
        if sha(FORMAL/name)!=h:raise ValueError('Frozen input metadata differs: '+name)
    return value


def schedule_indices(count,epoch,continuation=False):
    if count<=0:raise ValueError('No present TRAIN views')
    indices=list(range(count));random.Random(1703+(20 if continuation else 0)+epoch).shuffle(indices)
    return indices


def choose(events):
    eligible=[e for e in events if e['eligibility']['pass_all']]
    representative=min(eligible or events,key=lambda e:e['selection_key'])
    return dict(selected=representative['epoch'] if eligible else None,
        representative=representative['epoch'],eligible_epochs=[e['epoch'] for e in eligible])


def batch(rows,torch):
    keys=('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original','gt_original')
    return [torch.cat([r[k] for r in rows]).cuda() for k in keys]


def update(h,opt,x,arm,torch):
    from crane_project.utils import port_size_qalign_v1 as model
    h.train();opt.zero_grad();o=h(*x[:-1]);size=model.size_loss(o,x[-1],x[-2],x[3]);loss=size;geo=size.detach()*0
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in h.parameters()):raise ValueError('Invalid gradient')
    stem=math.sqrt(sum(float(p.grad.double().square().sum()) for p in h.stem.parameters()))
    terminal=math.sqrt(sum(float(p.grad.double().square().sum()) for p in h.terminal.parameters()))
    norm=float(torch.nn.utils.clip_grad_norm_(h.parameters(),10.));opt.step()
    if any(not bool(torch.isfinite(p).all()) for p in h.parameters()):raise ValueError('Invalid parameters')
    return dict(loss=float(loss.detach()),size_loss=float(size.detach()),geometry_loss=float(geo.detach()),stem_grad_norm=stem,terminal_grad_norm=terminal,grad_norm=norm)


def evaluate(h,rows,torch):
    from crane_project.utils import port_size_qalign_v1 as model
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity
    h.eval();pred={}
    with torch.no_grad():
        pp=[r for r in rows if len(r['midpoint_original'])]
        for start in range(0,len(pp),8):
            rr=pp[start:start+8];x=batch(rr,torch);o=h(*x[:-1])
            t=model.residual.size_targets(x[-1],x[-2],x[3])['log_residual']
            loss=torch.nn.functional.smooth_l1_loss(o['delta_roi'],t,beta=.1,reduction='none')
            swap=(x[3][:,2]<x[3][:,3]) ^ (x[-2][:,2]<x[-2][:,3]);loss=torch.where(swap[:,None],loss.flip(-1),loss)
            for j,r in enumerate(rr):
                pred[r['image'],r['scale']]=dict(edge_residual=o['boxes_original'][j].cpu().tolist(),delta_roi=o['delta_roi'][j].cpu().tolist(),
                    loss_edges_canonical=loss[j].cpu().tolist(),vector_valid=bool(o['vector_valid'][j]),local_size_variance=o['local_size_variance'][j].cpu().tolist(),
                    size_delivery='neutral' if bool(o['neutral_bypass'][j]) else 'square' if bool(o['square_bypass'][j]) else 'accepted' if bool(o['accepted'][j]) else 'fallback',
                    failed_checks=[k for k,v in o['checks'].items() if not bool(v[j])])
    out=[]
    for r in rows:
        v={k:r[k] for k in ('image','sequence','domain','frame_id','scale','split','shard')}
        v.update(gt=r['gt_original'][0].tolist(),b=r['boxes_original'][0].tolist() if len(r['boxes_original']) else None,
                 midpoint=r['midpoint_original'][0].tolist() if len(r['midpoint_original']) else None)
        v.update(pred.get((r['image'],r['scale']),dict(edge_residual=None,delta_roi=None,loss_edges_canonical=None,size_delivery='no_output',failed_checks=[])))
        if not frozen_identity(v):raise ValueError('M identity changed')
        out.append(v)
    return out


def load_inputs(torch,out):
    """Reuse only the completed new110 formal ROI shards; no detector build."""
    from crane_project.tools import train_symeood_110_midpoint_v1 as f
    from crane_project.utils import port_size_qalign_v1 as model
    f.runtime_imports();f.seed_all()
    identity=f.checked_sources();cache_dir=FORMAL/'roi_cache'
    train,val,cache=f.load_cache(cache_dir,identity)
    selection=read(FORMAL/'train/selection.json');cp=selection['selected_checkpoint']
    if cp['epoch']!=21 or cp['sha256']!=MIDPOINT_SHA or sha(FORMAL/'train'/cp['path'])!=MIDPOINT_SHA:
        raise ValueError('Requires original fixed midpoint ep21')
    if sha(Path(read(f.BASE_CONTRACT)['checkpoint']))!=DETECTOR_SHA:
        raise ValueError('New110 checkpoint SHA differs')
    payload=torch.load(str(FORMAL/'train'/cp['path']),map_location='cpu')
    if (payload['identity']!=identity or payload['sigma_cells']!=1.5 or payload['frozen_b']!=f.frontend_identity() or
            payload['context']['manifest_sha256']!=sha(cache_dir/'cache_manifest.json') or
            payload['context']['detector_state']!=cache['detector_state']):
        raise ValueError('Frozen M/cache parent identity differs')
    frozen=f.new_head().cuda().eval().requires_grad_(False);frozen.load_state_dict(payload['head_state'],strict=True)
    digest=f.g.state_digest(frozen)
    if digest!=cp['head_digest']:raise ValueError('Frozen M head digest differs')
    references={r['image']:r for r in (json.loads(x) for x in (FORMAL/'train/val_epoch_21.rows.jsonl').read_text().splitlines())}
    inventories=[]
    with torch.no_grad():
        for records in (train,val):
            for r in records:
                x=[r[k].cuda() for k in ('roi','support','boxes_original','boxes_model','scale_xy')]
                prediction=frozen(*x);m=prediction['boxes_original'].detach().cpu()
                r['midpoint_original']=m
                r['shard']='val_s1' if r['role']=='val' else ('train_s1' if r['scale']==1. else 'train_s05')
                if len(m)!=len(r['boxes_original']):raise ValueError('M changed output identity')
                if r['role']=='val':
                    ref=references[r['image']]
                    actual=m[0].tolist() if len(m) else None
                    if (actual is None)!=(ref['midpoint'] is None):raise ValueError('M VAL presence replay differs')
                    if actual is not None and (actual[5]!=ref['midpoint'][5] or not torch.allclose(m[0,:5],torch.tensor(ref['midpoint'][:5]),atol=1e-4,rtol=1e-6)):
                        raise ValueError('Cached M VAL replay differs')
                cell={k:r[k] for k in ('image','scale','domain','sequence','split','shard')}
                cell.update(midpoint=m[0].tolist() if len(m) else None,present=bool(len(m)))
                if len(m):
                    pair=model.read_pair(*[r[k] for k in ('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original')])
                    if not bool((pair['support'].sum((1,2,3))>0).all()):raise ValueError('Natural zero common support; do not filter')
                    target=model.residual.size_targets(r['gt_original'],m,r['boxes_model'])['log_residual']
                    cell.update(common_support_sum=float(pair['support'].sum()),reference_vectors_shape=[1,81,4,2],
                        target_log_residual=target.tolist())
                inventories.append(cell)
    if f.g.state_digest(frozen)!=digest or any(p.requires_grad or p.grad is not None for p in frozen.parameters()):
        raise ValueError('Frozen midpoint changed')
    rows_write(out/'midpoint_inputs.rows.jsonl.gz',inventories)
    train=sorted(train,key=lambda r:(r['scale'],r['image']));val=sorted(val,key=lambda r:(r['scale'],r['image']))
    counts=dict(train_views=len(train),val_views=len(val),train_present=sum(bool(len(r['midpoint_original'])) for r in train),
        val_present=sum(bool(len(r['midpoint_original'])) for r in val),detector_feature_extractions=0,midpoint_updates=0,
        cache_manifest_sha256=sha(cache_dir/'cache_manifest.json'),shards=cache['files'])
    counts['updates_per_epoch']=math.ceil(counts['train_present']/8)
    write(out/'input_counts.json',counts)
    return train,val,frozen,digest,counts


def short_trial(train,val,out,torch):
    """Bounded original fixed_b updates, exact Adam/RNG reload; discarded."""
    from crane_project.utils import port_size_qalign_v1 as model
    from crane_project.tools import train_symeood_110_midpoint_v1 as f
    present=[r for r in train if len(r['midpoint_original'])]
    chosen=[r for d in ('real','sim') for sc in (.5,1.)
        for r in [v for v in present if v['domain']==d and v['scale']==sc][:4]]
    vv=[r for d in ('real','sim') for r in [v for v in val if v['domain']==d][:2]]
    if len(chosen)!=16 or len(vv)!=4:raise ValueError('Expected16 TRAIN and4 VAL trial views')
    seed(torch);h=model.AlignedSizeHead('fixed_b').cuda();opt=torch.optim.Adam(h.parameters(),lr=.001,weight_decay=0.)
    initial=f.g.state_digest(h)
    if any(r['midpoint']!=r['edge_residual'] for r in evaluate(h,chosen+vv,torch)):
        raise ValueError('Zero LVSR output not exact M')
    logs=[]
    for rr in (chosen[:8],chosen[8:]):
        logs.append(update(h,opt,batch(rr,torch),'fixed_b',torch))
        if math.sqrt(sum(float(p.grad.double().square().sum()) for p in h.parameters()))>10.0001:
            raise ValueError('Trial gradient clipping differs')
    if logs[0]['terminal_grad_norm']<=0 or logs[1]['stem_grad_norm']<=0:raise ValueError('Trial gradients ineffective')
    snap=dict(head=deepcopy(h.state_dict()),opt=deepcopy(opt.state_dict()),rng=rng_get(torch))
    path=out/'discarded_trial.pth'
    with path.open('xb') as stream:torch.save(snap,stream)
    got=torch.load(str(path),map_location='cpu')
    clone=model.AlignedSizeHead('fixed_b').cuda();clone.load_state_dict(got['head'],strict=True)
    other=torch.optim.Adam(clone.parameters(),lr=.001,weight_decay=0.);other.load_state_dict(got['opt'])
    if not equal(snap,got,torch) or evaluate(h,vv,torch)!=evaluate(clone,vv,torch):raise ValueError('Trial save/reload differs')
    rng_set(snap['rng'],torch);a=update(h,opt,batch(chosen[:8],torch),'fixed_b',torch)
    rng_set(got['rng'],torch);z=update(clone,other,batch(chosen[:8],torch),'fixed_b',torch)
    if not equal(a,z,torch) or not equal(h.state_dict(),clone.state_dict(),torch) or not equal(opt.state_dict(),other.state_dict(),torch):
        raise ValueError('Trial next-update restoration differs')
    report=dict(passed=True,discarded=True,actual_optimizer_updates=4,initial_digest=initial,
        train_samples=[dict(image=r['image'],scale=r['scale']) for r in chosen],val_samples=[r['image'] for r in vv],
        losses=logs,checkpoint_sha256=sha(path),optimizer_restore_exact=True,neutral_exact=True)
    write(out/'trial.json',report)
    del h,opt,clone,other;torch.cuda.empty_cache();seed(torch)
    return report


def backup(path,stage,out,torch,role):
    """Two independent file-system copies of the exact selected head."""
    import shutil
    from crane_project.utils.port_size_qalign_v1 import AlignedSizeHead
    payload=torch.load(str(path),map_location='cpu');h=AlignedSizeHead('fixed_b')
    keys=h.load_state_dict(payload['head_state'],strict=True)
    digest=sha(path);copies=[]
    for folder in (out/'backups'/stage,Path('/home/omnisky/symeood_model_backups/symeood_110_retrain_seed0_20261026_v1/lvsr')/stage):
        folder.mkdir(parents=True,exist_ok=False);target=folder/('selected_head.pth' if role=='eligible_selected' else 'descriptive_head.pth')
        with path.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
        if sha(target)!=digest:raise ValueError('Backup SHA differs')
        copies.append(dict(path=str(target),sha256=digest,device=target.stat().st_dev))
    if len({r['device'] for r in copies})!=2:raise ValueError('Backups must use independent filesystems')
    proof=dict(source=str(path),role=role,sha256=digest,copies=copies,missing_keys=keys.missing_keys,unexpected_keys=keys.unexpected_keys)
    write(out/(stage+'_backup.json'),proof)
    return proof


def fit_stage(train,val,out,stage,inputs,torch,parent=None):
    from crane_project.tools import run_port_size_a20_protection_v1 as continuation
    from crane_project.utils import port_size_qalign_v1 as model
    from crane_project.utils import port_size_a20_protection_train_v1 as objective
    from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
    from crane_project.tools.run_port_geometry_size_condition_v1 import components
    from crane_project.tools import train_symeood_110_midpoint_v1 as f
    dest=out/stage;dest.mkdir();seed(torch)
    h=model.AlignedSizeHead('fixed_b').cuda()
    if parent is not None:
        payload=torch.load(str(parent['path']),map_location='cpu');h.load_state_dict(payload['head_state'],strict=True)
        if sha(parent['path'])!=parent['sha256']:raise ValueError('Continuation parent SHA differs')
    opt=torch.optim.Adam(h.parameters(),lr=.0001 if parent else .001,weight_decay=0.)
    if opt.state:raise ValueError('Adam must start empty')
    initial=f.g.state_digest(h);events=[];updates=0;riou_cache={}
    present=[r for r in train if len(r['midpoint_original'])]
    steps=math.ceil(len(present)/8);epochs=6 if parent else 24
    tables=None
    if parent:
        tables={s:objective.reference_table(parent[s+'_rows']) for s in ('train','val')}
        for s,table in tables.items():write(dest/(s+'_fixed_references.json'),list(table.values()))
    update_status(out,'training',stage_name=stage,epoch=0,updates=0)
    for epoch in range(1,epochs+1):
        tick=time.monotonic();ids=schedule_indices(len(present),epoch,continuation=parent is not None)
        totals={};samples=0;maxgrad=0.
        with (dest/'progress.jsonl').open('a') as log:
            for start in range(0,len(ids),8):
                rr=[present[i] for i in ids[start:start+8]];x=batch(rr,torch)
                if parent:
                    ref=objective.pack_references(rr,tables['train'],x[0].device,torch,training=True)
                    stats,_=continuation.update(h,opt,x,ref,'C',torch)
                else:stats=update(h,opt,x,'fixed_b',torch)
                updates+=1;samples+=len(rr);maxgrad=max(maxgrad,stats.get('grad_norm',0.))
                for k,v in stats.items():totals[k]=totals.get(k,0.)+v
                log.write(json.dumps(dict(epoch=epoch,step=start//8+1,updates=updates,batch_size=len(rr),**stats),allow_nan=False)+'\n')
        if samples!=len(present) or updates!=epoch*steps:raise ValueError('Actual exposure/update count differs')
        vr=continuation.evaluate(h,val,tables['val'],torch) if parent else evaluate(h,val,torch)
        summary=metrics.summary(vr,riou_cache);diag=near_joint(vr);comp=components(vr)
        gate=objective.eligibility(summary,diag,parent['summary']) if parent else eligibility(summary,diag)
        tr=continuation.evaluate(h,train,tables['train'],torch) if parent else evaluate(h,train,torch)
        td=dict(size_groups=metrics.train_report(tr),joint=near_joint(tr),
            cross_edge_by_scale={str(sc):model.cross_edge([r for r in tr if r['scale']==sc]) for sc in (.5,1.)})
        saved_rng=rng_get(torch)
        payload=dict(protocol=VERSION,stage=stage,arm='C' if parent else 'fixed_b',epoch=epoch,
            updates=updates,updates_per_epoch=steps,input_identity=inputs,parent_epoch=parent['epoch'] if parent else None,
            parent_sha256=parent['sha256'] if parent else None,head_state={k:v.detach().cpu().clone() for k,v in h.state_dict().items()},
            optimizer_state=deepcopy(opt.state_dict()),rng=saved_rng)
        path=dest/('head_epoch_%02d.pth'%epoch)
        with path.open('xb') as stream:torch.save(payload,stream)
        if not equal(payload,torch.load(str(path),map_location='cpu'),torch):raise ValueError('Checkpoint serialization differs')
        rng_set(saved_rng,torch)
        ep=dict(epoch=epoch,eligibility=gate,selection_key=ranking(summary,epoch),metrics=summary,components=comp,
            diagnostics=diag,cross_edge=model.cross_edge(vr),updates=updates,updates_per_epoch=steps,
            schedule_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
            checkpoint_sha256=sha(path),training_losses={k:v/steps for k,v in totals.items()},max_grad_norm=maxgrad,
            epoch_seconds=time.monotonic()-tick)
        write(dest/('epoch_%02d_val.json'%epoch),ep);rows_write(dest/('epoch_%02d_val.rows.jsonl.gz'%epoch),vr)
        write(dest/('epoch_%02d_train.json'%epoch),td);rows_write(dest/('epoch_%02d_train.rows.jsonl.gz'%epoch),tr)
        events.append(ep)
        update_status(out,'training',stage_name=stage,epoch=epoch,updates=updates,failed=gate['failed'])
        print(json.dumps(dict(stage=stage,epoch=epoch,updates=updates,eligible=gate['pass_all'],failed=gate['failed'],seconds=ep['epoch_seconds'])),flush=True)
        f.resource_check(0)
    result=dict(initial_digest=initial,**choose(events),formal_updates=updates,updates_per_epoch=steps,epochs=events)
    write(dest/'selection.json',result)
    representative=result['selected'] or result['representative']
    proof=backup(dest/('head_epoch_%02d.pth'%representative),stage,out,torch,
        'eligible_selected' if result['selected'] is not None else 'descriptive_only')
    return result,proof


def update_status(out,state,**details):
    import os
    value=dict(state=state,time=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),physical_gpu=0,
        TEST=False,depth=False,detector_updates=0,midpoint_updates=0,automatic_retry=False,**details)
    temporary=out/'status.tmp';temporary.write_text(json.dumps(value,indent=2));temporary.replace(out/'status.json')


def main(args):
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('WORLD_SIZE','1')!='1':
        raise ValueError('Only physical GPU0/logical0 authorized')
    proof=checked_sources();out=Path(args.out_dir).resolve()
    if out!=ROOT/proof['output_directory'] or out.exists():raise ValueError('New result directory required')
    out.mkdir(parents=True,exist_ok=False)
    try:
        import torch
        from crane_project.tools import train_symeood_110_midpoint_v1 as f
        from crane_project.utils import port_size_qalign_v1 as model
        torch.set_num_threads(1);torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(min(1.,6000*2**20/torch.cuda.get_device_properties(0).total_memory),0)
        update_status(out,'loading_formal_ROI')
        train,val,frozen,digest,counts=load_inputs(torch,out)
        inputs=dict(detector_sha256=DETECTOR_SHA,midpoint_sha256=MIDPOINT_SHA,**counts)
        write(out/'identity.json',dict(contract=proof,inputs=inputs))
        update_status(out,'short_trial')
        trial=short_trial(train,val,out,torch)
        seed(torch);head=model.AlignedSizeHead('fixed_b').cuda()
        if f.g.state_digest(head)!=trial['initial_digest']:raise ValueError('Formal must use fresh original initialization')
        baseline=evaluate(head,val,torch)
        from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
        write(out/'baseline_VAL.json',metrics.summary(baseline,{}));del head
        common,common_backup=fit_stage(train,val,out,'fixed_b',inputs,torch)
        final=dict(protocol=VERSION,common=common,common_backup=common_backup,continuation_run=False)
        if common['selected'] is not None:
            epoch=common['selected'];folder=out/'fixed_b';path=folder/('head_epoch_%02d.pth'%epoch)
            import gzip
            def read_rows(p):
                with gzip.open(p,'rt') as stream:return [json.loads(x) for x in stream]
            parent=dict(epoch=epoch,path=path,sha256=sha(path),summary=read(folder/('epoch_%02d_val.json'%epoch))['metrics'],
                train_rows=read_rows(folder/('epoch_%02d_train.rows.jsonl.gz'%epoch)),val_rows=read_rows(folder/('epoch_%02d_val.rows.jsonl.gz'%epoch)))
            continuation,continuation_backup=fit_stage(train,val,out,'C',inputs,torch,parent)
            final.update(continuation_run=True,continuation=continuation,continuation_backup=continuation_backup,
                parent_epoch=epoch,parent_sha256=parent['sha256'])
        else:final['continuation_stop_reason']='NO_COMMON_EPOCH_PASSES_ORIGINAL_ADOPTION_GATES'
        if f.g.state_digest(frozen)!=digest or any(p.requires_grad or p.grad is not None for p in frozen.parameters()):
            raise ValueError('Frozen M changed')
        if checked_sources()!=proof:raise ValueError('Source changed during training')
        write(out/'completion.json',final)
        update_status(out,'complete',common_selected=common['selected'],common_representative=common['representative'],
            common_updates=common['formal_updates'],continuation_run=final['continuation_run'],review_required=True)
    except BaseException as error:
        update_status(out,'stopped_on_error',error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',required=True)
    main(p.parse_args())


