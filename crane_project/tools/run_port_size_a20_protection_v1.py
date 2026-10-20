#!/usr/bin/env python3
"""S043: SHA-bound engineering -> C/R/P six epochs each, all18 full VAL.

No TEST/depth, feature extraction, model selector or alternate hyperparameters.
"""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_size_qalign_v1 as parent
from crane_project.tools.run_port_size_native_train_v1 import sha,read,write,rows_write,seed,rng_get,rng_set,equal,near_joint,ranking
from crane_project.utils import port_size_a20_protection_train_v1 as objective
VERSION=objective.VERSION


def checked_sources():
    path=ROOT/'crane_project/tools/port_size_a20_protection_train_v1_sources.json';proof=read(path)
    if proof['protocol']!=VERSION:raise ValueError('Source protocol differs')
    for n,h in proof['sources'].items():
        if sha(ROOT/n)!=h:raise ValueError('Source drift '+n)
    contract=read(ROOT/'crane_project/tools/port_size_a20_protection_train_v1_protocol.json')
    if (contract['arms']!=list(objective.ARMS) or contract['epochs_per_arm']!=6 or contract['updates_per_epoch']!=639
            or contract['lr']!=.0001 or contract['a20_sha256']!=objective.A20_SHA):raise ValueError('Run contract drift')
    parent.checked_sources()
    return dict(manifest_sha256=sha(path),**proof)


def load_rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]


def checked_start(folder,torch):
    index=read(folder/'artifacts.json')['files']
    files=['training.json','finished.json','fixed_b/head_epoch_20.pth',
           'fixed_b/epoch_20_train.rows.jsonl.gz','fixed_b/epoch_20_val.rows.jsonl.gz']
    if any(sha(folder/n)!=index[n] for n in files):raise ValueError('A20 sealed artifacts differ')
    if index[files[2]]!=objective.A20_SHA:raise ValueError('Requires fixed A20')
    training=read(folder/'training.json')
    if training['status']!='COMPLETE' or read(folder/'finished.json')['status']!='COMPLETE' or training['formal_head_updates']!=30672:raise ValueError('Incomplete S039')
    payload=torch.load(str(folder/files[2]),map_location='cpu')
    if (payload['protocol']!='port_size_qalign_v1' or payload['arm']!='fixed_b' or payload['epoch']!=20
            or payload['updates']!=12780):raise ValueError('A20 metadata differs')
    rows={s:load_rows(folder/('fixed_b/epoch_20_'+s+'.rows.jsonl.gz')) for s in ('train','val')}
    if [len(rows[s]) for s in ('train','val')]!=[5116,887]:raise ValueError('Complete references required')
    return payload,rows,dict(index_sha256=sha(folder/'artifacts.json'),files={n:index[n] for n in files})


def head(state,torch):
    from crane_project.utils.port_size_qalign_v1 import AlignedSizeHead
    h=AlignedSizeHead('fixed_b').cuda();h.load_state_dict(state,strict=True)
    if sum(p.numel() for p in h.parameters())!=2800:raise ValueError('Head capacity changed')
    return h


def update(h,opt,x,refs,arm,torch):
    h.train();opt.zero_grad();o=h(*x[:-1]);terms=objective.losses(o,x[-1],x[-2],x[3],refs,arm,torch)
    loss=terms['total']
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in h.parameters()):raise ValueError('Invalid gradients')
    norm=float(torch.nn.utils.clip_grad_norm_(h.parameters(),10.));opt.step()
    if any(not bool(torch.isfinite(p).all()) for p in h.parameters()):raise ValueError('Invalid parameters')
    stats={k:float(terms[k].detach()) for k in ('total','size','ratio','protect')}
    stats.update(grad_norm=norm,activated=int(terms['active'].sum()),protected=int(refs['active'].sum()),edges=int(terms['active'].numel()))
    points={k:terms[k].detach().cpu().tolist() for k in ('point_size','point_ratio','hinge','active')}
    return stats,points


def evaluate(h,records,table,torch):
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity
    h.eval();pred={}
    with torch.no_grad():
        pp=[r for r in records if len(r['midpoint_original'])]
        for start in range(0,len(pp),8):
            rr=pp[start:start+8];x=parent.batch(rr,torch);ref=objective.pack_references(rr,table,x[0].device,torch)
            o=h(*x[:-1]);terms=objective.losses(o,x[-1],x[-2],x[3],ref,'C',torch)
            for j,r in enumerate(rr):
                pred[objective.key(r)]=dict(edge_residual=o['boxes_original'][j].cpu().tolist(),pre_guard=o['candidate_original'][j].cpu().tolist(),
                    delta_roi=o['delta_roi'][j].cpu().tolist(),loss_edges_canonical=terms['point_size'][j].cpu().tolist(),
                    vector_valid=bool(o['vector_valid'][j]),local_size_variance=o['local_size_variance'][j].cpu().tolist(),
                    size_delivery='neutral' if bool(o['neutral_bypass'][j]) else 'square' if bool(o['square_bypass'][j]) else 'accepted' if bool(o['accepted'][j]) else 'fallback',
                    failed_checks=[k for k,v in o['checks'].items() if not bool(v[j])],
                    protect_hinge=terms['hinge'][j].cpu().tolist(),protect_activated=terms['active'][j].cpu().tolist())
        result=[]
        for r in records:
            v={k:r[k] for k in ('image','sequence','domain','frame_id','scale','split','shard')}
            v.update(gt=r['gt_original'][0].tolist(),b=r['boxes_original'][0].tolist() if len(r['boxes_original']) else None,
                     midpoint=r['midpoint_original'][0].tolist() if len(r['midpoint_original']) else None)
            v.update(pred.get(objective.key(r),dict(edge_residual=None,pre_guard=None,delta_roi=None,loss_edges_canonical=None,size_delivery='no_output',failed_checks=[])))
            if not frozen_identity(v):raise ValueError('Frozen M identity differs')
            result.append(v)
    return result


def replay(actual,reference,torch):
    lookup={objective.key(r):r for r in reference};maximum=0.;dmax=0.
    if len(lookup)!=len(actual):raise ValueError('Replay full count')
    for r in actual:
        old=lookup[objective.key(r)]
        for k in ('image','scale','sequence','domain','frame_id','split','gt'):
            if r[k]!=old[k]:raise ValueError('Replay metadata '+k)
        for k in ('b','midpoint','edge_residual'):
            if (r[k] is None)!=(old[k] is None):raise ValueError('Replay presence')
            if r[k] is not None:
                a,b=torch.tensor(r[k]),torch.tensor(old[k]);maximum=max(maximum,float((a-b).abs().max()))
                if not torch.allclose(a,b,atol=1e-4,rtol=1e-6) or r[k][5]!=old[k][5]:raise ValueError('A20 box replay '+k)
        if r['size_delivery']!=old['size_delivery'] or r['failed_checks']!=old['failed_checks']:raise ValueError('Discrete delivery replay differs')
        if r['delta_roi'] is not None:
            a,b=torch.tensor(r['delta_roi']),torch.tensor(old['delta_roi']);dmax=max(dmax,float((a-b).abs().max()))
            if not torch.allclose(a,b,atol=1e-6,rtol=1e-6):raise ValueError('A20 delta replay differs')
    return dict(frames=len(actual),max_box_abs=maximum,max_delta_abs=dmax,score_presence_delivery_exact=True)


def close_state(a,b,torch,atol):
    errors=[]
    def walk(x,y):
        if torch.is_tensor(x):
            if not torch.is_tensor(y) or x.shape!=y.shape:raise ValueError('State tensor shape')
            xx,yy=x.detach().cpu(),y.detach().cpu()
            if xx.is_floating_point():
                errors.append(float((xx.double()-yy.double()).abs().max()) if xx.numel() else 0.)
                if not torch.allclose(xx,yy,rtol=1e-6,atol=atol):raise ValueError('State numerical mismatch')
            elif not torch.equal(xx,yy):raise ValueError('Integer/RNG mismatch')
        elif isinstance(x,dict):
            if x.keys()!=y.keys():raise ValueError('State keys')
            for k in x:walk(x[k],y[k])
        elif isinstance(x,(tuple,list)):
            if len(x)!=len(y):raise ValueError('State sequence')
            for u,v in zip(x,y):walk(u,v)
        elif x!=y:raise ValueError('State value mismatch')
    walk(a,b);return max(errors,default=0.)


def engineering(train,table,state,out,torch):
    present=[r for r in train if len(r['midpoint_original'])];chosen=present[:16]
    batches=[parent.batch(chosen[i:i+8],torch) for i in (0,8)]
    refs=[objective.pack_references(chosen[i:i+8],table,'cuda:0',torch,training=True) for i in (0,8)]
    seed(torch);a=head(state,torch);b=head(state,torch)
    oa,ob=[torch.optim.Adam(h.parameters(),lr=.0001) for h in (a,b)]
    comparisons=[]
    for x,ref in zip(batches,refs):
        old=parent.update(a,oa,x,'fixed_b',torch);new,_=update(b,ob,x,ref,'C',torch)
        if old['loss']!=new['total']:raise ValueError('C scalar graph differs')
        comparisons.append(dict(loss=old['loss'],gradient_max_abs=close_state([p.grad for p in a.parameters()],[p.grad for p in b.parameters()],torch,1e-10),
            state_max_abs=close_state(a.state_dict(),b.state_dict(),torch,1e-7),adam_max_abs=close_state(oa.state_dict(),ob.state_dict(),torch,1e-10)))
    del a,b,oa,ob
    report=dict(control_original_two_updates=comparisons,arms={},temporary_updates_discarded=True,
        fixed_batches=[[dict(image=r['image'],scale=r['scale']) for r in chosen[i:i+8]] for i in (0,8)],
        tolerances=dict(rtol=1e-6,gradient_atol=1e-10,state_atol=1e-7,adam_atol=1e-10,rng_serialization_exact=True))
    for arm in objective.ARMS:
        seed(torch);h=head(state,torch);opt=torch.optim.Adam(h.parameters(),lr=.0001)
        before=time.monotonic();first,_=update(h,opt,batches[0],refs[0],arm,torch)
        snap=dict(head=deepcopy(h.state_dict()),opt=deepcopy(opt.state_dict()),rng=rng_get(torch))
        stream=io.BytesIO();torch.save(snap,stream);stream.seek(0);got=torch.load(stream,map_location='cpu')
        clone=head(state,torch);clone.load_state_dict(got['head']);other=torch.optim.Adam(clone.parameters(),lr=.0001);other.load_state_dict(got['opt'])
        if not equal(snap,got,torch):raise ValueError('Exact checkpoint/RNG serialization differs')
        rng_set(snap['rng'],torch);second,_=update(h,opt,batches[1],refs[1],arm,torch);after=rng_get(torch)
        rng_set(got['rng'],torch);repeated,_=update(clone,other,batches[1],refs[1],arm,torch)
        if not equal(after,rng_get(torch),torch):raise ValueError('RNG resume differs')
        report['arms'][arm]=dict(first=first,second=second,state_max_abs=close_state(h.state_dict(),clone.state_dict(),torch,1e-7),
            adam_max_abs=close_state(opt.state_dict(),other.state_dict(),torch,1e-10),serialization_rng_exact=True,elapsed_seconds=time.monotonic()-before)
        del h,opt,clone,other,snap,got
    write(out/'engineering.json',report);del batches,refs;torch.cuda.empty_cache();seed(torch)
    return report


def summarize(rows,table,cache):
    from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
    from crane_project.tools.run_port_geometry_size_condition_v1 import components
    from crane_project.utils.port_size_qalign_v1 import cross_edge
    return dict(metrics=metrics.summary(rows,cache),components=components(rows),diagnostics=near_joint(rows),
                protection=objective.protection_report(rows,table),cross_edge=cross_edge(rows))


def execute(args,out,torch,source):
    from crane_project.tools import run_port_geometry_size_distribution_v1 as prior
    from crane_project.utils import port_size_qalign_v1 as model
    from crane_project.utils import port_geometry_midpoint_sigma_v1 as frozen_model
    from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
    sealed=prior.sealed;history,differences=prior.historical_identity()
    cache=sealed.checked_cache(args.cache_dir,history);selection,references,indexed=sealed.checked_selection(args.head_dir,history)
    if sha(args.b_weight)!=prior.B_SHA:raise ValueError('B weight SHA')
    frozen=sealed.load_head(torch,frozen_model,args.head_dir,selection,cache,'cuda:0');frozen_state=sealed.state_digest(frozen)
    payload,saved,startproof=checked_start(args.a20_run,torch);state=payload['head_state']
    for k,v in [('b_sha256',prior.B_SHA),('m_sha256',sealed.FIXED['sha256'])]:
        if payload['input_identity'][k]!=v:raise ValueError('A20 frontend identity')
    tables={s:objective.reference_table(saved[s]) for s in ('train','val')}
    for s,table in tables.items():
        write(out/(s+'_fixed_references.json'),list(table.values()))
    reference_hashes={s:sha(out/(s+'_fixed_references.json')) for s in tables}
    train,_=parent.load_records(args.cache_dir,cache,frozen,'cuda:0',torch,model,references,('train_s1','train_s05'))
    if len(train)!=5116 or sum(bool(len(r['midpoint_original'])) for r in train)!=5112:raise ValueError('Full TRAIN identity')
    if any(float(r['support'].sum())<=0 for r in train if len(r['midpoint_original'])):raise ValueError('Zero natural support')
    seed(torch);h=head(state,torch);initial_digest=sealed.state_digest(h)
    replay_train=evaluate(h,train,tables['train'],torch);train_replay=replay(replay_train,saved['train'],torch)
    fallbacks=[r for r in replay_train if r['size_delivery']=='fallback']
    if len(fallbacks)!=2:raise ValueError('Historical two TRAIN fallbacks differ')
    write(out/'a20_train_replay.json',dict(proof=train_replay,fallbacks=fallbacks,missing=4))
    rows_write(out/'a20_train.rows.jsonl.gz',replay_train)
    engineering(train,tables['train'],state,out,torch)
    val,_=parent.load_records(args.cache_dir,cache,frozen,'cuda:0',torch,model,references,('val_s1',))
    a20_val=evaluate(h,val,tables['val'],torch);val_replay=replay(a20_val,saved['val'],torch)
    a20_summary=summarize(a20_val,tables['val'],{})
    rows_write(out/'a20_val.rows.jsonl.gz',a20_val);write(out/'a20_val.json',a20_summary)
    write(out/'ENGINEERING_PASSED.json',dict(train=train_replay,val=val_replay,temporary_updates_discarded=True,initial_digest=initial_digest))
    del h,a20_val,replay_train;torch.cuda.empty_cache()
    inputs=dict(a20=startproof,frontend=payload['input_identity'],reference_hashes=reference_hashes,source=source,
                cache_manifest_sha256=sha(args.cache_dir/'cache_manifest.json'),shards=cache['files'],unused_detector_differences=differences)
    write(out/'input_identity.json',inputs)
    report=dict(protocol=VERSION,status='RUNNING',formal_updates=0,initial_A20_updates=12780,arms={},input_identity=inputs,
                test_reads=0,depth_reads=0,feature_extractions=0,automatic_promotion=False)
    present=[r for r in train if len(r['midpoint_original'])];riou_cache={}
    representatives={}
    for arm in objective.ARMS:
        dest=out/arm;dest.mkdir();seed(torch);h=head(state,torch);opt=torch.optim.Adam(h.parameters(),lr=.0001)
        if sealed.state_digest(h)!=initial_digest or opt.state:raise ValueError('Fresh identical A20/Adam failed')
        events=[]
        for epoch in range(1,7):
            tick=time.monotonic();ids=list(range(len(present)));random.Random(1703+20+epoch).shuffle(ids)
            totals=defaultdict(float);cells=defaultdict(lambda:defaultdict(float))
            for start in range(0,len(ids),8):
                rr=[present[j] for j in ids[start:start+8]];x=parent.batch(rr,torch)
                refs=objective.pack_references(rr,tables['train'],x[0].device,torch,training=True)
                stats,points=update(h,opt,x,refs,arm,torch);report['formal_updates']+=1
                for k,v in stats.items():totals[k]+=v
                for i,r in enumerate(rr):
                    for j,e in enumerate(('long','short')):
                        c=cells[r['domain']+'/'+str(r['scale'])+'/'+e];c['count']+=1
                        c['size_sum']+=points['point_size'][i][j];c['ratio_sum']+=points['point_ratio'][i]
                        c['hinge_sum']+=points['hinge'][i][j];c['activated']+=points['active'][i][j]
                if start%1600==0:print(json.dumps(dict(stage='train',arm=arm,epoch=epoch,view=start,**stats)),flush=True)
            vr=evaluate(h,val,tables['val'],torch);summary=summarize(vr,tables['val'],riou_cache)
            gate=objective.eligibility(summary['metrics'],summary['diagnostics'],a20_summary['metrics'])
            tr=evaluate(h,train,tables['train'],torch)
            td=dict(size_groups=metrics.train_report(tr),diagnostics=near_joint(tr),protection=objective.protection_report(tr,tables['train']),
                    cross_edge_by_scale={str(sc):model.cross_edge([r for r in tr if r['scale']==sc]) for sc in (.5,1.)})
            saved_rng=rng_get(torch)
            checkpoint=dict(protocol=VERSION,arm=arm,epoch=epoch,continuation_updates=epoch*639,source_A20_sha256=objective.A20_SHA,
                input_identity=inputs,head_state={k:v.detach().cpu().clone() for k,v in h.state_dict().items()},optimizer_state=deepcopy(opt.state_dict()),rng=saved_rng)
            path=dest/('head_epoch_%02d.pth'%epoch)
            with path.open('xb') as f:torch.save(checkpoint,f)
            if not equal(checkpoint,torch.load(str(path),map_location='cpu'),torch):raise ValueError('Checkpoint serialization differs')
            rng_set(saved_rng,torch)
            ep=dict(summary,epoch=epoch,eligibility=gate,selection_key=ranking(summary['metrics'],epoch),checkpoint_sha256=sha(path),
                schedule_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                training_losses={k:(v/639 if k not in ('activated','protected','edges') else v) for k,v in totals.items()},
                actual_training_cells=dict(cells),elapsed_seconds=time.monotonic()-tick)
            write(dest/('epoch_%02d_val.json'%epoch),ep);rows_write(dest/('epoch_%02d_val.rows.jsonl.gz'%epoch),vr)
            write(dest/('epoch_%02d_train.json'%epoch),td);rows_write(dest/('epoch_%02d_train.rows.jsonl.gz'%epoch),tr)
            if sum(c['count'] for c in cells.values())!=10224:raise ValueError('Full single-edge exposure')
            events.append(ep);print(json.dumps(dict(stage='epoch_finished',arm=arm,epoch=epoch,failed=gate['failed'],selection_key=ep['selection_key'],elapsed_seconds=ep['elapsed_seconds'])),flush=True)
        eligible=[e for e in events if e['eligibility']['pass_all']]
        representative=min(eligible or events,key=lambda e:e['selection_key']);representatives[arm]=representative
        result=dict(initial_digest=initial_digest,selected=representative['epoch'] if eligible else None,representative=representative['epoch'],
                    eligible_epochs=[e['epoch'] for e in eligible],epochs=events)
        write(dest/'selection.json',result);report['arms'][arm]=result;del h,opt;torch.cuda.empty_cache()
    if report['formal_updates']!=11502 or len({tuple(e['schedule_sha256'] for e in report['arms'][a]['epochs']) for a in objective.ARMS})!=1:raise ValueError('Paired budget/order mismatch')
    if frozen_state!=sealed.state_digest(frozen):raise ValueError('Frozen M changed')
    if checked_start(args.a20_run,torch)[2]!=startproof or checked_sources()!=source:raise ValueError('End source/start drift')
    if any(sha(out/(s+'_fixed_references.json'))!=h for s,h in reference_hashes.items()):raise ValueError('Frozen references changed')
    if sha(args.b_weight)!=prior.B_SHA or sha(args.head_dir/sealed.FIXED['path'])!=sealed.FIXED['sha256']:raise ValueError('Frontend weight changed')
    if any(sha(args.cache_dir/n)!=v for n,v in cache['files'].items()):raise ValueError('Source cache changed')
    report.update(status='COMPLETE',mechanism=objective.mechanism(representatives));write(out/'training.json',report)


def main(args):
    out=args.out_dir.resolve()
    if out.exists() or out.parent!=(ROOT/'work_dirs'/VERSION).resolve():raise ValueError('Fresh result-only directory required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3') or os.environ.get('WORLD_SIZE','1')!='1':raise ValueError('One physical GPU2/3 required')
    out.mkdir(parents=True);stop=None;watcher=None
    report=dict(protocol=VERSION,status='FAILED',started_utc=datetime.now(timezone.utc).isoformat())
    try:
        proof=checked_sources();write(out/'source_identity.json',proof);write(out/'protocol.json',read(ROOT/'crane_project/tools/port_size_a20_protection_train_v1_protocol.json'))
        from crane_project.tools.audit_port_size_native_p3_v1 import setup_gpu
        resource,stop,watcher=setup_gpu(out,7000,6800,6000);report['resource']=resource
        import torch
        torch.set_num_threads(1);torch.cuda.set_device(0);seed(torch)
        torch.cuda.set_per_process_memory_fraction(6000*2**20/torch.cuda.get_device_properties(0).total_memory,0)
        report['runtime']=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
        execute(args,out,torch,proof);report.update(status='COMPLETE',allocator_peak_mib=torch.cuda.max_memory_allocated()/2**20,
            reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20)
    except Exception as e:report['error']=repr(e);raise
    finally:
        if stop is not None:stop.set();watcher.join(timeout=11)
        report['ended_utc']=datetime.now(timezone.utc).isoformat();write(out/'finished.json',report)
        write(out/'artifacts.json',dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--a20-run',type=Path,default=ROOT/'work_dirs/port_size_qalign_v1/train_20261019_v1')
    p.add_argument('--cache-dir',type=Path,default=ROOT/'work_dirs/port_results/geometry/port_geometry_midpoint_formal_v1_roi_cache')
    p.add_argument('--head-dir',type=Path,default=ROOT/'work_dirs/crane_symeood_k1_port_day2night_midpoint_sigma_v1/sigma_1p5')
    p.add_argument('--b-weight',type=Path,default=ROOT/'work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth')
    main(p.parse_args())
