"""Supplemental R4+LB ablation: same LVSR, fresh training conditioned on B."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import gzip
import json
import math
from pathlib import Path
import random
import runpy
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import train_lvsr_vd_control_v1 as io
CONFIG=ROOT/'crane_project/configs/crane_r4_lvsr_without_midpoint_v1.json'
SOURCES=['crane_project/configs/crane_r4_lvsr_without_midpoint_v1.json',
 'crane_project/tools/train_r4_lvsr_without_midpoint_v1.py','tests/test_r4_lvsr_without_midpoint_v1.py',
 'crane_project/tools/train_lvsr_vd_control_v1.py','work_dirs/lvsr_diagnostic_20261112_v1/diagnose.py']
read,save=io.read,io.save


def checked_config():
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    c=read(CONFIG)
    for key in ('epochs','batch_size','seed','lr','weight_decay','betas','eps','clip_norm','parameters'):
        if c[key]!=p.SETTINGS[key]:raise ValueError('Original training setting differs: '+key)
    if c['physical_gpu']!=3 or c['updates']!=15336 or c['TEST']:raise ValueError('Wrong ablation scope')
    return c


def sources():
    return {n:io.sha(ROOT/n) for n in SOURCES}


def condition_on_B(records):
    """Only replace the internal condition alias. Original input shards stay intact."""
    derived=[]
    for r in records:
        b=r['boxes_original'];m=r['midpoint_original']
        if b.shape!=m.shape or not b[:,5].equal(m[:,5]):raise ValueError('B/M count or score differs')
        new=dict(r,midpoint_original=b.detach().clone(),geometry_condition='B')
        if new['midpoint_original'].requires_grad or new['midpoint_original'].grad_fn is not None:
            raise ValueError('Condition must be detached')
        derived.append(new)
    return derived


def measure(rows,method):
    import numpy as np
    from crane_project.tools.eval_crane_offline import compute_riou
    from crane_project.tools.eval_port_geometry_ap_v1 import ap11
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import canonical,row_metrics
    values=[]
    for r in rows:
        pred=r[method]
        iou=compute_riou(np.asarray(canonical(pred)),np.asarray(canonical(r['gt']))) if pred else 0.
        values.append(dict(row_metrics(r['gt'],pred,iou),score=pred[5] if pred else 0.))
    pp=[v for v in values if v['output']];n=len(rows);out=len(pp)
    result=dict(frames=n,outputs=out,TP50=sum(v['riou']>=.5 for v in values),
        TP75=sum(v['riou']>=.75 for v in values),joint10=sum(v['joint_size10'] for v in values),
        mean_RIoU=sum(v['riou'] for v in values)/n,
        long_MAE=sum(v['long_relative_error'] for v in pp)/out,
        short_MAE=sum(v['short_relative_error'] for v in pp)/out,
        center_MAE_px=sum(v['center_error_px'] for v in pp)/out,
        pure_angle_RMSE_deg=math.sqrt(sum(v['angle_error_deg']**2 for v in pp)/out))
    order=np.argsort(-np.asarray([v['score'] for v in pp],dtype=np.float32))
    for threshold in (.5,.75):
        hits=np.asarray([v['riou']>=threshold for v in pp],dtype=np.float32)[order]
        tp=hits.cumsum();fp=(1-hits).cumsum()
        result['AP'+str(round(threshold*100))]=100*ap11(tp.astype(np.float64)/n,tp/np.maximum(tp+fp,np.finfo(np.float32).eps))
    return result


def four_paths(current,original_rows):
    if len(current)!=887 or len(original_rows)!=887:raise ValueError('Full VAL required')
    lookup={r['image']:r for r in original_rows};merged=[]
    for r in current:
        old=lookup[r['image']]
        if r['gt']!=old['gt'] or r['b']!=old['b'] or r['midpoint']!=r['b']:
            raise ValueError('Four-way original GT/B or B condition differs')
        merged.append(dict(r,M20=old['midpoint'],LM=old['edge_residual']))
    groups=dict(overall=merged)
    groups.update({d:[r for r in merged if r['domain']==d] for d in ('real','sim')})
    groups.update({s:[r for r in merged if r['sequence']==s] for s in sorted({r['sequence'] for r in merged})})
    return {g:{name:measure(rr,method) for name,method in
        [('R4','b'),('R4_LB','edge_residual'),('R4_M20','M20'),('R4_M20_LM','LM')]} for g,rr in groups.items()}


def engineering(engine,base,Head,train,val,out,seed,equal,rng_get,rng_set,original_initial):
    torch=engine.torch;seed(torch);head=Head().cuda();initial=engine.g.state_digest(head)
    if initial!=original_initial:raise ValueError('Original2480 initialization differs')
    records=train+val
    neutral=base.evaluate(head,records,torch)
    if any(r['b']!=r['midpoint'] or r['edge_residual']!=r['b'] for r in neutral):
        raise ValueError('All-view zero LB must equal B exactly')
    empty=[r for r in train if not len(r['boxes_original'])]
    if len(empty)!=4:raise ValueError('Expected4 empty TRAIN views')
    with torch.no_grad():
        for r in empty:
            x=[r[k].cuda() for k in ('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original')]
            if len(head(*x)['boxes_original']):raise ValueError('Empty forward created output')
    chosen=[r for d in ('real','sim') for sc in (.5,1.) for r in
        [q for q in train if len(q['boxes_original']) and q['domain']==d and q['scale']==sc][:4]]
    opt=torch.optim.Adam(head.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    logs=[base.update(head,opt,base.batch(chosen[i:i+8],torch),'fixed_b',torch) for i in (0,8)]
    if logs[0]['terminal_grad_norm']<=0 or logs[1]['stem_grad_norm']<=0:raise ValueError('Ineffective head gradients')
    snapshot=dict(head=deepcopy(head.state_dict()),optimizer=deepcopy(opt.state_dict()),rng=rng_get(torch))
    engine.atomic_save(out/'discarded_smoke.pth',snapshot)
    loaded=torch.load(str(out/'discarded_smoke.pth'),map_location='cpu')
    if not equal(snapshot,loaded,torch):raise ValueError('Checkpoint differs')
    clone=Head().cuda();clone.load_state_dict(loaded['head'],strict=True)
    other=torch.optim.Adam(clone.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    other.load_state_dict(loaded['optimizer'])
    if base.evaluate(head,chosen,torch)!=base.evaluate(clone,chosen,torch):raise ValueError('Reload inference differs')
    rng_set(snapshot['rng'],torch);a=base.update(head,opt,base.batch(chosen[:8],torch),'fixed_b',torch)
    rng_set(loaded['rng'],torch);b=base.update(clone,other,base.batch(chosen[:8],torch),'fixed_b',torch)
    if a!=b or not equal(head.state_dict(),clone.state_dict(),torch) or not equal(opt.state_dict(),other.state_dict(),torch):
        raise ValueError('Next Adam update differs after restore')
    proof=dict(passed=True,initial_state=initial,neutral_views=6003,empty_forwards=4,
        restore_head_Adam_RNG_and_next_update_exact=True,engineering_updates_discarded=4,
        midpoint_model_built=False,midpoint_forwards=0,TEST=False)
    save(out/'engineering.json',proof)
    return proof,neutral


def run(c,out):
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    from crane_project.tools import train_symeood_r4_lvsr_pair_v1 as original
    from crane_project.tools.run_port_size_native_train_v1 import seed,rng_get,rng_set,equal
    parent=p.checked_sources();hashes=sources()
    engine,base,Head,vector,runtime=original.runtime(3);torch=engine.torch
    import subprocess
    if subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()!=c['gpu_uuid']:
        raise ValueError('Physical GPU3 UUID differs')
    original_train,original_val,inputs=original.load_inputs(p.paths(),parent,engine)
    common=runpy.run_path(str(ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/diagnose.py'))
    original_digest=common['memory_digest'](original_train+original_val)
    train,val=condition_on_B(original_train),condition_on_B(original_val)
    derived_digest=common['memory_digest'](train+val)
    present=[r for r in train if len(r['boxes_original'])]
    if len(train)!=5116 or len(present)!=5112 or len(val)!=887:raise ValueError('Original frame/view counts differ')
    source=p.paths()['attempt']
    reference_rows=source/'train/epoch_05_VAL.rows.jsonl.gz'
    if io.sha(reference_rows)!='06fffb95f43304a6e3a88d5942cdee8bc97ecb1298edb939cbd677eecf448f56':
        raise ValueError('Historical M20/LM VAL rows differ')
    pins=[ROOT/p.parent.R4_CHECKPOINT,p.midpoint_directory()/'train/head_epoch_20.pth',
        source/'train/head_epoch_05.pth',source/'train/selection.json',source/'verification/completion.json',
        source/'inputs/train.pt',source/'inputs/val.pt',source/'inputs/completion.json',reference_rows]
    frozen_pins={str(q.relative_to(ROOT)):io.sha(q) for q in pins}
    expected=[p.parent.R4_SHA,p.MIDPOINT_SHA,'b272c50dfe6329414803d5f22c6e1e84d9c3db37b6bb2e31f1c7ed140f662c3e']
    if any(frozen_pins[str(q.relative_to(ROOT))]!=h for q,h in zip(pins[:3],expected)):raise ValueError('Frozen parent/control SHA differs')
    original_initial=read(source/'verification/completion.json')['initial_state']
    counts=dict(Counter(r['sequence']+'|'+str(r['scale']) for r in present));plans={}
    for epoch in range(1,25):
        order=base.schedule_indices(5112,epoch)
        reference=list(range(5112));random.Random(1703+epoch).shuffle(reference)
        if order!=reference:raise ValueError('Original order differs')
        plans[str(epoch)]=dict(frame_order_SHA=io.frame_hash(present,order),views=5112)
    identity=dict(settings=c,sources=hashes,source_cache_parent=parent,condition='B')
    save(out/'protocol.json',dict(identity=identity,frozen_pins=frozen_pins,runtime=runtime,
        original_input_digest=original_digest,derived_B_input_digest=derived_digest,
        planned_sampling=plans,video_scale_counts=counts,TEST=False,
        midpoint_model_built=False,old_controls_reinferred=False))
    io.status(out,'engineering')
    proof,neutral=engineering(engine,base,Head,train,val,out,seed,equal,rng_get,rng_set,original_initial)
    baseline=original.summarize([r for r in neutral if r['split']=='val'])
    save(out/'baseline_VAL.json',dict(condition_alias='midpoint field is B; no actual M forward',metrics=baseline))
    seed(torch);head=Head().cuda()
    if engine.g.state_digest(head)!=proof['initial_state']:raise ValueError('Fresh state differs; smoke must be discarded')
    opt=torch.optim.Adam(head.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    updates=0;events=[]
    for epoch in range(1,25):
        tick=time.monotonic();order=base.schedule_indices(5112,epoch);actual=[];exposure=Counter();loss_sum=0.
        for start in range(0,5112,8):
            ii=order[start:start+8];rr=[present[i] for i in ii]
            stats=base.update(head,opt,base.batch(rr,torch),'fixed_b',torch);updates+=1
            actual.extend(ii);exposure.update(r['sequence']+'|'+str(r['scale']) for r in rr);loss_sum+=stats['loss']
            if any(int(opt.state[q]['step'])!=updates for q in head.parameters()):raise ValueError('Adam counter differs')
            if start%800==0:engine.resource_check(0);io.status(out,'train_LB',epoch=epoch,step=start//8+1,updates=updates)
        if (updates!=epoch*639 or sorted(actual)!=list(range(5112))
            or io.frame_hash(present,actual)!=plans[str(epoch)]['frame_order_SHA'] or dict(exposure)!=counts):
            raise ValueError('Actual update/order/exposure differs')
        vr=base.evaluate(head,val,torch);metrics=original.summarize(vr)
        for domain in metrics:
            if not metrics[domain]['frozen_identity'] or any(metrics[domain][key]!=baseline[domain][key] for key in ('b','midpoint')):
                raise ValueError('B-only identity/baseline changed')
        cp=out/('head_epoch_%02d.pth'%epoch)
        payload=dict(protocol=c['protocol'],identity=identity,condition='B',epoch=epoch,updates=updates,parameters=2480,
            head_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
            optimizer_state=deepcopy(opt.state_dict()),rng=rng_get(torch),parent_checkpoint=None,
            detector_sha256=p.parent.R4_SHA,midpoint_model_used=False)
        engine.atomic_save(cp,payload)
        if not equal(payload,torch.load(str(cp),map_location='cpu'),torch):raise ValueError('Saved checkpoint differs')
        e=dict(epoch=epoch,updates=updates,metrics=metrics,gate_vs_input_B=p.assess(metrics),
            selection_key=p.selection_key(metrics,epoch),checkpoint=dict(path=cp.name,epoch=epoch,sha256=io.sha(cp)),
            loss_mean=loss_sum/639,sampling=dict(frame_order_SHA=io.frame_hash(present,actual),video_scale_counts=dict(exposure)),
            seconds=time.monotonic()-tick)
        save(out/('epoch_%02d_VAL.json'%epoch),e);original.write_rows(out/('epoch_%02d_VAL.rows.jsonl.gz'%epoch),vr)
        events.append(e);io.status(out,'epoch_complete_LB',epoch=epoch,updates=updates,metrics=metrics['overall']['edge_residual'])
    chosen=io.pick(events,p);selected=chosen['epoch'] if chosen['gate_vs_input_B']['pass_all'] else None
    save(out/'selection.json',dict(representative_epoch=chosen['epoch'],representative_checkpoint=chosen['checkpoint'],
        selected_epoch=selected,all24_ranking_gate_not_prefilter=True,gate_reference='R4 input B',TEST=False,automatic_adoption=False))
    replay=Head().cuda();replay.load_state_dict(torch.load(str(out/chosen['checkpoint']['path']),map_location='cpu')['head_state'],strict=True)
    final=base.evaluate(replay,train+val,torch);current=[r for r in final if r['split']=='val']
    if original.summarize(current)!=chosen['metrics']:raise ValueError('Representative replay differs')
    original.write_rows(out/'representative_TRAIN_VAL.rows.jsonl.gz',final)
    with gzip.open(reference_rows,'rt') as f:oldrows=list(map(json.loads,f))
    compare=four_paths(current,oldrows)
    detail=io.all_groups(final,common)
    save(out/'core_detail.json',dict(condition='B; midpoint alias denotes B, not M20',relative_to_B=detail,four_paths_VAL=compare))
    save(out/'core_summary.json',dict(settings=c,representative=chosen,epochs=events,
        representative_epoch=chosen['epoch'],selected_epoch=selected,reference_LM_epoch=5,
        four_paths_VAL=compare,source_SHA=hashes,frozen_pins=frozen_pins,TEST=False,automatic_adoption=False))
    with (out/'core_epochs.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['epoch','loss','long_MAE_pct','short_MAE_pct','joint10','RIoU','AP75','gate_vs_B'])
        for e in events:
            m=e['metrics']['overall']['edge_residual']
            w.writerow([e['epoch'],e['loss_mean'],100*m['long_MAE'],100*m['short_MAE'],m['joint10'],m['mean_RIoU'],m['AP75'],e['gate_vs_input_B']['pass_all']])
    with (out/'four_paths_VAL.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['group','method','frames','outputs','long_MAE_pct','short_MAE_pct','joint10','RIoU','AP75','center_MAE_px','angle_RMSE_deg'])
        for group,methods in compare.items():
            for method,m in methods.items():
                w.writerow([group,method,m['frames'],m['outputs'],100*m['long_MAE'],100*m['short_MAE'],m['joint10'],m['mean_RIoU'],m['AP75'],m['center_MAE_px'],m['pure_angle_RMSE_deg']])
    if (common['memory_digest'](original_train+original_val)!=original_digest
        or common['memory_digest'](train+val)!=derived_digest or sources()!=hashes or p.checked_sources()!=parent
        or any(io.sha(ROOT/n)!=h for n,h in frozen_pins.items())):raise ValueError('Original or derived inputs/source drift')
    save(out/'completion.json',dict(status='R4_LB_TRAIN_VAL_COMPLETE',epochs=24,updates=updates,
        representative_epoch=chosen['epoch'],checkpoint=chosen['checkpoint'],selected_epoch=selected,
        parameters=2480,condition='B',engineering_updates_discarded=4,midpoint_forwards=0,
        original_inputs_unchanged=True,derived_inputs_unchanged=True,actual_order_and_exposure_verified=True,
        TEST=False,automatic_adoption=False,detector_updates=0))
    core=['protocol.json','engineering.json','baseline_VAL.json','selection.json','core_detail.json',
        'core_summary.json','core_epochs.csv','four_paths_VAL.csv','completion.json']
    save(out/'core_manifest.json',{n:io.sha(out/n) for n in core})
    io.status(out,'R4_LB_COMPLETE',epoch=chosen['epoch'],updates=updates)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args();c=checked_config()
    if not args.run:print(json.dumps(dict(settings=c,sources=sources()),indent=2));return
    out=ROOT/c['output_root'];out.mkdir(exist_ok=False)
    def timeout(*unused):raise RuntimeError('Fixed7200-second ablation budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,c['wall_seconds'])
    try:run(c,out)
    except BaseException as e:save(out/'failure.json',dict(error=repr(e),retry=False,TEST=False));io.status(out,'FAILED',error=repr(e));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
