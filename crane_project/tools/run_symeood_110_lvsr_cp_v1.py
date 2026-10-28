#!/usr/bin/env python3
"""Independent CP. Default check is static; smoke/train require explicit --run."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import run_symeood_110_lvsr_c_ep7 as control
from crane_project.utils import symeood_110_lvsr_cp_v1 as cp
from crane_project.tools.run_port_size_native_train_v1 import sha,read,write,rows_write,seed,rng_get,rng_set,equal,near_joint

CONTRACT=ROOT/'crane_project/data_contracts/symeood_110_lvsr_cp_v1.json'
OUT=control.COMMON/'continuation_CP_ep7'


def status(stage,**details):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    value=dict(state=stage,time=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        physical_gpu=int(visible) if visible in ('0','1') else None,TEST=False,depth=False,
        detector_updates=0,midpoint_updates=0,automatic_retry=False,**details)
    temporary=OUT/'status.tmp';temporary.write_text(json.dumps(value,indent=2));temporary.replace(OUT/'status.json')


def checked_sources():
    proof=read(CONTRACT);control.checked_sources()
    science={k:v for k,v in proof.items() if k not in ('implementation','status','scope_this_turn')}
    digest=hashlib.sha256(json.dumps(science,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=proof['implementation']['scientific_sha256']:raise ValueError('Frozen CP scientific contract changed')
    if (proof['candidate']!='CP' or proof['parent']['sha256']!=control.PARENT_SHA or proof['parent']['epoch']!=7 or
        proof['settings']['epochs']!=6 or proof['change']['ratio_weight']!=0. or proof['change']['protect_weight']!=.1):
        raise ValueError('Frozen CP contract differs')
    for path,h in proof['implementation']['sources'].items():
        if sha(ROOT/path)!=h:raise ValueError('CP source differs: '+path)
    for path,h in proof['implementation']['control_files'].items():
        if sha(ROOT/path)!=h:raise ValueError('Existing C evidence differs: '+path)
    for split in ('train','val'):
        if sha(ROOT/proof['inputs'][split+'_fixed_references'])!=proof['inputs'][split+'_reference_sha256']:
            raise ValueError('Fixed CP reference differs')
    return proof


def load_tables(proof):
    return {s:{(r['image'],r['scale']):r for r in read(ROOT/proof['inputs'][s+'_fixed_references'])} for s in ('train','val')}


def batch(rows,torch,device):
    return [torch.cat([r[k] for r in rows]).to(device) for k in
        ('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original','gt_original')]


def new_head(state,torch,device):
    from crane_project.utils.port_size_qalign_v1 import AlignedSizeHead
    h=AlignedSizeHead('fixed_b').to(device);h.load_state_dict(state,strict=True);return h


def small_inputs(proof,torch):
    """Only bounded views run through the head; no full TRAIN/VAL replay."""
    parent=read(control.COMMON/'input_counts.json')
    with __import__('gzip').open(control.COMMON/'fixed_b/epoch_07_train.rows.jsonl.gz','rt') as f:tr=[json.loads(x) for x in f]
    with __import__('gzip').open(control.COMMON/'fixed_b/epoch_07_val.rows.jsonl.gz','rt') as f:vr=[json.loads(x) for x in f]
    ordered=sorted([r for r in tr if r['midpoint'] is not None],key=lambda r:(r['scale'],r['image']))
    ids=control.base.schedule_indices(len(ordered),1,continuation=True)
    first=[ordered[i] for i in ids[:8]]
    chosen=[r for d in ('real','sim') for sc in (.5,1.) for r in
        [v for v in ordered if v['domain']==d and v['scale']==sc][:4]]
    vv=[r for d in ('real','sim') for r in [v for v in vr if v['domain']==d and v['midpoint'] is not None][:2]]
    fallbacks=[r for r in tr if r['size_delivery']=='fallback']
    selected={s:{(r['image'],r['scale']):r for r in rr} for s,rr in [('train',first+chosen+fallbacks[:1]),('val',vv)]}
    manifest=read(control.base.FORMAL/'roi_cache/cache_manifest.json')
    if sha(control.base.FORMAL/'roi_cache/cache_manifest.json')!=proof['inputs']['cache_manifest_sha256']:raise ValueError('Formal cache differs')
    found={s:{} for s in selected}
    for name in ('train_s1.pt','train_s05.pt','val_s1.pt'):
        path=control.base.FORMAL/'roi_cache'/name
        if sha(path)!=manifest['files'][name]:raise ValueError('Formal shard differs')
        payload=torch.load(str(path),map_location='cpu');split='val' if name.startswith('val') else 'train'
        if payload['detector_state']!=manifest['detector_state'] or payload['identity']!=manifest['identity']:raise ValueError('Shard identity differs')
        for r in payload['records']:
            key=(r['image'],r['scale'])
            if key not in selected[split]:continue
            v=selected[split][key]
            if r['gt_original'][0].tolist()!=v['gt'] or r['boxes_original'][0].tolist()!=v['b']:raise ValueError('Selected cache row differs')
            r['midpoint_original']=r['boxes_original'].new_tensor([v['midpoint']]);r['shard']=v['shard'];found[split][key]=r
        del payload
    for s in found:
        if found[s].keys()!=selected[s].keys():raise ValueError('Missing limited view')
    lookup=lambda rr,s:[found[s][r['image'],r['scale']] for r in rr]
    return lookup(first,'train'),lookup(chosen,'train'),lookup(vv,'val')+lookup(fallbacks[:1],'train'),dict(views_available=parent['train_present'],fallback_available=bool(fallbacks),fallback_selected=[dict(image=r['image'],scale=r['scale']) for r in fallbacks[:1]])


def smoke(proof,torch,device):
    first,train,val,counts=small_inputs(proof,torch);tables=load_tables(proof)
    payload=torch.load(str(ROOT/proof['parent']['path']),map_location='cpu')
    if sha(ROOT/proof['parent']['path'])!=control.PARENT_SHA:raise ValueError('Fixed ep7 bytes differ')
    from crane_project.tools import train_symeood_110_midpoint_v1 as f
    f.runtime_imports();seed(torch);h=new_head(payload['head_state'],torch,device)
    initial=f.g.state_digest(h);expected=read(control.OUT/'C/selection.json')['initial_digest']
    if initial!=expected:raise ValueError('CP/C parent parameters differ')
    opt=torch.optim.Adam(h.parameters(),lr=.0001,weight_decay=0.)
    if opt.state:raise ValueError('CP Adam must start empty')
    x=batch(first,torch,device);ref=control.objective.pack_references(first,tables['train'],device,torch,training=True)
    h.train();out=h(*x[:-1]);terms=cp.losses(out,x[-1],x[-2],x[3],ref,torch)
    recorded=json.loads((control.OUT/'C/progress.jsonl').open().readline())
    base_grads=torch.autograd.grad(terms['size'],tuple(h.parameters()),retain_graph=True)
    c_norm=math.sqrt(sum(float(g.double().square().sum()) for g in base_grads))
    if not math.isclose(float(terms['size']),recorded['size'],rel_tol=1e-6,abs_tol=1e-7):raise ValueError('First-batch C scalar replay differs')
    if not math.isclose(c_norm,recorded['grad_norm'],rel_tol=1e-5,abs_tol=1e-6):raise ValueError('First-batch C gradient replay differs')
    del out,terms,base_grads,x,ref;logs=[]
    for rr in (train[:8],):
        x=batch(rr,torch,device);ref=control.objective.pack_references(rr,tables['train'],device,torch,training=True)
        logs.append(cp.update(h,opt,x,ref,torch))
        if math.sqrt(sum(float(p.grad.double().square().sum()) for p in h.parameters()))>10.0001:raise ValueError('CP clipping differs')
    saved=dict(head=deepcopy(h.state_dict()),opt=deepcopy(opt.state_dict()),rng=rng_get(torch))
    stream=io.BytesIO();torch.save(saved,stream);stream.seek(0);got=torch.load(stream,map_location='cpu')
    if not equal(saved,got,torch):raise ValueError('CP save/reload differs')
    clone=new_head(got['head'],torch,device);other=torch.optim.Adam(clone.parameters(),lr=.0001,weight_decay=0.);other.load_state_dict(got['opt'])
    rr=train[8:];x=batch(rr,torch,device);ref=control.objective.pack_references(rr,tables['train'],device,torch,training=True)
    rng_set(saved['rng'],torch);a=cp.update(h,opt,x,ref,torch);after=rng_get(torch)
    rng_set(got['rng'],torch);b=cp.update(clone,other,x,ref,torch)
    if not equal(a,b,torch) or not equal(after,rng_get(torch),torch) or not equal(h.state_dict(),clone.state_dict(),torch) or not equal(opt.state_dict(),other.state_dict(),torch):raise ValueError('CP next-update restoration differs')
    with torch.no_grad():
        for rr in (val,):
            x=batch(rr,torch,device);v=h(*x[:-1]);m=x[-2]
            if not torch.equal(v['boxes_original'][:,[0,1,4,5]],m[:,[0,1,4,5]]):raise ValueError('CP altered frozen identity')
    return dict(passed=True,device=device,torch=str(torch.__version__),actual_optimizer_updates=3,discarded=True,
        initial_digest=initial,C_first_batch=dict(loss=float(recorded['size']),gradient_norm=c_norm,CP_uses_no_C_optimizer_update=True),
        train_samples=[dict(image=r['image'],scale=r['scale']) for r in train],val_samples=[r['image'] for r in val],counts=counts,
        model_optimizer_RNG_restore_exact=True,contract_sha256=sha(CONTRACT),GPU_training_accepted=device.startswith('cuda'),runtime_GPU_validation_pending=device=='cpu',gpu_context_initialized=torch.cuda.is_initialized())


def check_schedule(train,proof):
    present=[r for r in train if len(r['midpoint_original'])]
    if len(present)!=proof['settings']['expected_present_train_views']:raise ValueError('Present TRAIN count differs')
    values={}
    for n in range(1,7):
        ids=control.base.schedule_indices(len(present),n,continuation=True)
        h=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        if h!=proof['control']['epoch_schedule_sha256'][str(n)]:raise ValueError('C/CP schedule differs')
        values[str(n)]=dict(index_sha256=h,ordered_view_sha256=hashlib.sha256(json.dumps([(present[i]['image'],present[i]['scale']) for i in ids],separators=(',',':')).encode()).hexdigest())
    return present,values


def fit(proof,torch):
    from crane_project.tools import train_symeood_110_midpoint_v1 as f
    from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
    from crane_project.tools.run_port_geometry_size_condition_v1 import components
    train,val,parent,payload=control.load_inputs(torch);present,schedules=check_schedule(train,proof)
    runtime=f.runtime(0);recorded=read(control.OUT/'identity.json')['runtime']
    if any(runtime[k]!=v for k,v in recorded.items()):raise ValueError('CP GPU numeric environment differs from C')
    cache_runtime=read(control.base.FORMAL/'roi_cache/cache_manifest.json')['runtime']
    if runtime!=cache_runtime:raise ValueError('CP software environment differs from formal cache/C source context')
    tables=load_tables(proof);dest=OUT/'CP';dest.mkdir();seed(torch)
    h=new_head(payload['head_state'],torch,'cuda:0');opt=torch.optim.Adam(h.parameters(),lr=.0001,weight_decay=0.)
    if opt.state or f.g.state_digest(h)!=read(control.OUT/'C/selection.json')['initial_digest']:raise ValueError('CP/C initialization differs')
    write(OUT/'identity.json',dict(runtime=runtime,contract_sha256=sha(CONTRACT),parent_epoch=7,parent_sha256=control.PARENT_SHA,
        schedules=schedules,optimizer='fresh Adam',initial_digest=f.g.state_digest(h),feature_extractions=0,midpoint_updates=0))
    events=[];riou_cache={};updates=0;steps=math.ceil(len(present)/8)
    for n in range(1,7):
        tick=time.monotonic();sums={};samples=0;order=control.base.schedule_indices(len(present),n,continuation=True)
        with (dest/'progress.jsonl').open('a') as log:
            for start in range(0,len(present),8):
                ids=order[start:start+8]
                rr=[present[i] for i in ids];x=batch(rr,torch,'cuda:0')
                ref=control.objective.pack_references(rr,tables['train'],'cuda:0',torch,training=True)
                stats=cp.update(h,opt,x,ref,torch);updates+=1;samples+=len(rr)
                for k,v in stats.items():sums[k]=sums.get(k,0.)+v
                log.write(json.dumps(dict(epoch=n,step=start//8+1,updates=updates,batch_size=len(rr),**stats),allow_nan=False)+'\n')
        if samples!=len(present) or updates!=n*steps:raise ValueError('CP exposure count differs')
        vr=control.old.evaluate(h,val,tables['val'],torch);summary=metrics.summary(vr,riou_cache);near=near_joint(vr);damages=cp.damage(vr)
        crows=control.old.load_rows(control.OUT/('C/epoch_%02d_val.rows.jsonl.gz'%n));csummary=read(control.OUT/('C/epoch_%02d_val.json'%n))['metrics']
        gate=cp.eligibility(summary,near,damages,csummary,cp.damage(crows),parent['summary'])
        tr=control.old.evaluate(h,train,tables['train'],torch)
        rng=rng_get(torch);checkpoint=dict(protocol='symeood_110_lvsr_CP_v1',arm='CP',epoch=n,updates=updates,updates_per_epoch=steps,
            parent_epoch=7,parent_sha256=control.PARENT_SHA,input_identity=payload['input_identity'],contract_sha256=sha(CONTRACT),
            head_state={k:v.detach().cpu().clone() for k,v in h.state_dict().items()},optimizer_state=deepcopy(opt.state_dict()),rng=rng)
        path=dest/('head_epoch_%02d.pth'%n)
        with path.open('xb') as stream:torch.save(checkpoint,stream)
        if not equal(checkpoint,torch.load(str(path),map_location='cpu'),torch):raise ValueError('CP checkpoint serialization differs')
        rng_set(rng,torch)
        ep=dict(epoch=n,updates=updates,metrics=summary,components=components(vr),diagnostics=near,damage=damages,eligibility=gate,
            selection_key=cp.selection_key(summary,damages,n),training_losses={k:v/steps for k,v in sums.items()},
            training_count_totals={k:sums[k] for k in ('activated','protected','edges')},schedule=schedules[str(n)],checkpoint_sha256=sha(path),epoch_seconds=time.monotonic()-tick)
        write(dest/('epoch_%02d_val.json'%n),ep);rows_write(dest/('epoch_%02d_val.rows.jsonl.gz'%n),vr)
        write(dest/('epoch_%02d_train.json'%n),dict(size_groups=metrics.train_report(tr),damage=cp.damage(tr),diagnostics=near_joint(tr)))
        rows_write(dest/('epoch_%02d_train.rows.jsonl.gz'%n),tr);events.append(ep);f.resource_check(0)
        status('training_CP',epoch=n,updates=updates,eligible=gate['pass_all'])
        print(json.dumps(dict(epoch=n,updates=updates,eligible=gate['pass_all'],failed=gate['failed'])),flush=True)
    result=dict(**cp.select(events),formal_updates=updates,epochs=events);write(dest/'selection.json',result)
    backups={}
    for role,n in [('selected',result['selected']),('representative',result['representative']),('fixed_CP6',6)]:
        if n is None:continue
        if n not in backups:backups[n]=control.base.backup(dest/('head_epoch_%02d.pth'%n),'CP_ep7_epoch%02d'%n,OUT,torch,'eligible_selected' if n==result['selected'] else 'descriptive_only')
    return dict(versions={k:result[k] for k in ('selected','representative','fixed_final','eligible_epochs')},formal_updates=updates,
        fixed_CP6_vs_C6_pass=events[5]['eligibility']['pass_all'],backups=backups,TEST=False,depth=False)


def main(args):
    proof=checked_sources()
    if args.stage=='check':
        print(json.dumps(dict(status='CP_STATIC_CHECK_PASS_NO_GPU',candidate='CP',epochs=6,training_started=False)));return
    if not args.run:raise ValueError('Explicit --run required; default never starts smoke or training')
    if args.device=='cuda' and (args.gpu is None or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu)):raise ValueError('Select one explicitly authorized physical GPU')
    if args.stage=='train' and args.device!='cuda':raise ValueError('Formal CP training uses original CUDA path only')
    # The legacy RNG helper queries every visible CUDA device. Hide them before
    # importing torch for CPU acceptance, so even RNG capture cannot claim a GPU.
    if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    device='cpu' if args.device=='cpu' else 'cuda:0'
    if args.device=='cuda':torch.cuda.set_device(0)
    try:
        if args.stage=='smoke':
            target=OUT/('cpu_smoke.json' if device=='cpu' else 'gpu_smoke.json')
            if target.exists():raise ValueError('Passed smoke receipt preserved; no automatic repeat')
            status('CP_'+args.device+'_smoke')
            receipt=smoke(proof,torch,device)
            if device=='cpu' and receipt['gpu_context_initialized']:raise ValueError('CPU acceptance initialized a CUDA context')
            write(target,receipt);status('smoke_pass_wait_authorization',training_started=False,device=device)
        else:
            acceptance=read(OUT/'gpu_smoke.json')
            if not acceptance['passed'] or acceptance['contract_sha256']!=sha(CONTRACT):raise ValueError('Current GPU smoke must pass before formal train')
            if (OUT/'CP').exists() or (OUT/'formal_start.json').exists():raise ValueError('Existing formal run preserved; no automatic retry')
            write(OUT/'formal_start.json',dict(contract_sha256=sha(CONTRACT),gpu=args.gpu,epochs=6))
            result=fit(proof,torch)
            if checked_sources()!=proof:raise ValueError('CP source/control changed')
            write(OUT/'completion.json',dict(state='complete',**result));status('complete',**result['versions'])
    except BaseException as error:
        status('stopped_on_error',error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('check','smoke','train'),default='check')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--gpu',type=int,choices=(0,1))
    p.add_argument('--run',action='store_true');main(p.parse_args())
