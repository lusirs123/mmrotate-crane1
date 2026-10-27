#!/usr/bin/env python3
"""Explicit six-epoch C research continuation from the descriptive new-chain ep7."""
from copy import deepcopy
import gzip
import io
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import run_symeood_110_lvsr_v1 as base
from crane_project.tools import run_port_size_a20_protection_v1 as old
from crane_project.tools.run_port_size_native_train_v1 import sha,read,write,seed,rng_get,rng_set,equal,ranking
from crane_project.utils import port_size_a20_protection_train_v1 as objective

CONTRACT=ROOT/'crane_project/data_contracts/symeood_110_lvsr_c_ep7.json'
COMMON=ROOT/'work_dirs/symeood_110_retrain_seed0_20261026_v1/downstream/lvsr'
PARENT_SHA='dfa6206dd9eadd6875bd9523afa409a59d797bde3db926a41a8d0e33ea08eee6'
OUT=COMMON/'continuation_ep7'


def checked_sources():
    proof=read(CONTRACT);base.checked_sources()
    if (proof['parent_epoch']!=7 or proof['parent_sha256']!=PARENT_SHA or
            proof['arms']!=['C'] or proof['epochs']!=6 or proof['physical_gpu'] not in (0,1)):
        raise ValueError('Fixed C continuation identity differs')
    for name,h in proof['sources'].items():
        if sha(ROOT/name)!=h:raise ValueError('C source differs: '+name)
    for name,h in proof['common_artifacts'].items():
        if sha(COMMON/name)!=h:raise ValueError('Original common artifact differs: '+name)
    done=read(COMMON/'completion.json')
    if done['common']['selected'] is not None or done['common']['representative']!=7 or done['continuation_run']:
        raise ValueError('Requires original descriptive ep7 and completed stop decision')
    return proof


def versions(events):
    eligible=[e for e in events if e['eligibility']['pass_all']]
    return dict(selected=min(eligible,key=lambda e:e['selection_key'])['epoch'] if eligible else None,
        representative=min(events,key=lambda e:e['selection_key'])['epoch'],fixed_final=6,
        eligible_epochs=[e['epoch'] for e in eligible])


def load_inputs(torch):
    """Attach already saved M21 outputs to the unchanged formal ROI tensors."""
    from crane_project.tools import train_symeood_110_midpoint_v1 as f
    f.runtime_imports();identity=f.checked_sources()
    train,val,manifest=f.load_cache(base.FORMAL/'roi_cache',identity)
    payload=torch.load(str(COMMON/'fixed_b/head_epoch_07.pth'),map_location='cpu')
    if (sha(COMMON/'fixed_b/head_epoch_07.pth')!=PARENT_SHA or payload['epoch']!=7 or
            payload['arm']!='fixed_b' or payload['input_identity']['detector_sha256']!=base.DETECTOR_SHA or
            payload['input_identity']['midpoint_sha256']!=base.MIDPOINT_SHA):raise ValueError('Parent checkpoint differs')
    if sha(Path(read(f.BASE_CONTRACT)['checkpoint']))!=base.DETECTOR_SHA:raise ValueError('Detector bytes differ')
    if sha(base.FORMAL/'train/head_epoch_21.pth')!=base.MIDPOINT_SHA:raise ValueError('M21 bytes differ')
    saved={}
    for split,rows in (('train',train),('val',val)):
        with gzip.open(COMMON/('fixed_b/epoch_07_'+split+'.rows.jsonl.gz'),'rt') as stream:saved[split]=[json.loads(x) for x in stream]
        lookup={objective.key(r):r for r in saved[split]}
        if len(lookup)!=len(rows):raise ValueError('Saved input count differs')
        for r in rows:
            v=lookup[objective.key(r)]
            for key in ('image','scale','domain','sequence','frame_id','split'):
                if r[key]!=v[key]:raise ValueError('Input metadata differs: '+key)
            b=r['boxes_original'][0].tolist() if len(r['boxes_original']) else None
            if b!=v['b'] or r['gt_original'][0].tolist()!=v['gt']:raise ValueError('Saved B/GT differs')
            r['midpoint_original']=torch.tensor([v['midpoint']],dtype=r['boxes_original'].dtype) if v['midpoint'] is not None else r['boxes_original'].clone()
            r['shard']=v['shard']
    train=sorted(train,key=lambda r:(r['scale'],r['image']));val=sorted(val,key=lambda r:(r['scale'],r['image']))
    expected=read(COMMON/'input_counts.json')
    if sum(bool(len(r['midpoint_original'])) for r in train)!=expected['train_present']:raise ValueError('TRAIN count differs')
    parent=dict(epoch=7,path=COMMON/'fixed_b/head_epoch_07.pth',sha256=PARENT_SHA,
        summary=read(COMMON/'fixed_b/epoch_07_val.json')['metrics'],train_rows=saved['train'],val_rows=saved['val'])
    return train,val,parent,payload


def trial(train,val,parent,state,torch):
    """Three discarded C updates; only the newly exercised C path is tested."""
    present=[r for r in train if len(r['midpoint_original'])]
    rr=[r for d in ('real','sim') for sc in (.5,1.) for r in [v for v in present if v['domain']==d and v['scale']==sc][:4]]
    vv=[r for d in ('real','sim') for r in [v for v in val if v['domain']==d and len(v['midpoint_original'])][:2]]
    tables={s:objective.reference_table(parent[s+'_rows']) for s in ('train','val')}
    seed(torch);h=old.head(state,torch);opt=torch.optim.Adam(h.parameters(),lr=.0001,weight_decay=0.)
    if opt.state:raise ValueError('Trial Adam must start fresh')
    replay=old.replay(old.evaluate(h,vv,tables['val'],torch),
        [v for v in parent['val_rows'] if objective.key(v) in {objective.key(r) for r in vv}],torch)
    def step(head,optimizer,rows):
        x=base.batch(rows,torch);ref=objective.pack_references(rows,tables['train'],x[0].device,torch,training=True)
        stats,_=old.update(head,optimizer,x,ref,'C',torch)
        if stats['total']!=stats['size']:raise ValueError('C total must be original size loss')
        if sum(float(p.grad.double().square().sum()) for p in head.parameters())**.5>10.0001:raise ValueError('Clipping differs')
        return stats
    first=step(h,opt,rr[:8]);snap=dict(head=deepcopy(h.state_dict()),opt=deepcopy(opt.state_dict()),rng=rng_get(torch))
    stream=io.BytesIO();torch.save(snap,stream);stream.seek(0);got=torch.load(stream,map_location='cpu')
    if not equal(snap,got,torch):raise ValueError('Save/reload differs')
    clone=old.head(got['head'],torch);other=torch.optim.Adam(clone.parameters(),lr=.0001,weight_decay=0.);other.load_state_dict(got['opt'])
    rng_set(snap['rng'],torch);second=step(h,opt,rr[8:]);after=rng_get(torch)
    rng_set(got['rng'],torch);repeated=step(clone,other,rr[8:])
    if not equal(after,rng_get(torch),torch) or not equal(second,repeated,torch):raise ValueError('Resumed update differs')
    if not equal(h.state_dict(),clone.state_dict(),torch) or not equal(opt.state_dict(),other.state_dict(),torch):raise ValueError('Resume model/Adam differs')
    old.evaluate(h,vv,tables['val'],torch)
    report=dict(passed=True,discarded=True,actual_optimizer_updates=3,train_samples=[dict(image=r['image'],scale=r['scale']) for r in rr],
        val_samples=[r['image'] for r in vv],parent_replay=replay,first=first,second=second,model_optimizer_rng_restore_exact=True)
    del h,opt,clone,other;torch.cuda.empty_cache();seed(torch)
    return report


def main():
    proof=checked_sources();gpu=proof['physical_gpu']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu) or os.environ.get('WORLD_SIZE','1')!='1':raise ValueError('Single authorized GPU required')
    if OUT.exists():raise ValueError('Existing C continuation is preserved; no automatic retry')
    OUT.mkdir()
    try:
        import torch
        torch.set_num_threads(1);torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(min(1.,6000*2**20/torch.cuda.get_device_properties(0).total_memory),0)
        base.update_status(OUT,'loading_existing_inputs')
        train,val,parent,payload=load_inputs(torch)
        write(OUT/'identity.json',dict(contract=proof,parent_epoch=7,parent_role='descriptive_only',parent_sha256=PARENT_SHA,
            detector_sha256=base.DETECTOR_SHA,midpoint_sha256=base.MIDPOINT_SHA,feature_extractions=0,midpoint_forwards=0,
            common_original_status_sha256=sha(COMMON/'status.json'),runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0))))
        base.update_status(OUT,'short_C_trial')
        write(OUT/'trial.json',trial(train,val,parent,payload['head_state'],torch))
        result,backup=base.fit_stage(train,val,OUT,'C',payload['input_identity'],torch,parent)
        recorded=versions(result['epochs']);backups=dict(primary=backup)
        primary=result['selected'] or result['representative']
        for role,n in (('representative',recorded['representative']),('fixed_C6',6)):
            if n==primary:backups[role]=backup
            else:backups[role]=base.backup(OUT/'C'/('head_epoch_%02d.pth'%n),'C_ep7_'+role,OUT,torch,'descriptive_only')
        if recorded['selected'] is not None:backups['selected']=backup
        if checked_sources()!=proof:raise ValueError('Source or original common artifacts changed')
        write(OUT/'completion.json',dict(state='complete',authorization='Separate fixed-budget research continuation; common non-adoption preserved',
            parent_epoch=7,parent_role='descriptive_only',parent_sha256=PARENT_SHA,versions=recorded,formal_updates=result['formal_updates'],backups=backups,
            selected_rule='Original eligibility then ranking',representative_rule='Original ranking over all six epochs independently of eligibility',TEST=False,depth=False))
        base.update_status(OUT,'complete',formal_updates=result['formal_updates'],**recorded)
    except BaseException as error:
        base.update_status(OUT,'stopped_on_error',error=repr(error));raise


if __name__=='__main__':main()
