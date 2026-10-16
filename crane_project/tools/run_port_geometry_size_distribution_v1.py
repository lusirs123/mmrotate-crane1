#!/usr/bin/env python3
"""One complete-TRAIN, two-supervision, fixed24-epoch size-only contrast.

check reads source text only. run verifies sealed cache/weights, performs zero
and discarded2-update engineering checks, then fresh24 epochs per arm. No TEST,
detector construction, feature extraction, resume, sweeps or automatic adoption.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import diagnose_port_geometry_midpoint_edge_feasibility_v1 as sealed
from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics

VERSION = 'port_geometry_size_distribution_v1'
SETTINGS = dict(seed=1703, epochs=24, batch_size=8, lr=.001, weight_decay=0., clip_norm=10.)
ARMS = ('decoded_control','distribution')
PROTOCOL = ROOT/'crane_project/tools/port_geometry_size_distribution_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_geometry_size_distribution_v1_sources.json'
B_SHA = '8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23'
NEW_FILES = {
 'crane_project/utils/port_geometry_size_distribution_v1.py',
 'crane_project/utils/port_geometry_size_distribution_metrics_v1.py',
 'crane_project/tools/run_port_geometry_size_distribution_v1.py',
 'crane_project/tools/port_geometry_size_distribution_v1_protocol.json',
 'tests/test_port_geometry_size_distribution_v1.py',
 'crane_project/tools/run_port_geometry_size_distribution_v1.sh',
 'crane_project/tools/diagnose_port_geometry_midpoint_edge_feasibility_v1.py',
 'crane_project/utils/port_geometry_midpoint_edge_feasibility_v1.py',
 'crane_project/utils/port_geometry_midpoint_edge_residual_v1.py',
 'crane_project/utils/port_geometry_midpoint_edge_residual_metrics_v1.py',
 'crane_project/tools/eval_crane_offline.py','crane_project/tools/eval_port_geometry_ap_v1.py',
 sealed.FORMAL_SOURCES,sealed.SIGMA_SOURCES}


def protocol_document():
    return dict(protocol=VERSION,settings=SETTINGS,arms=list(ARMS),parameter_count=2712,
        grid=dict(bins=33,log_min=-1.,log_max=1.,uniform=True),
        inputs='Frozen B P3 9x9 ROI/support, B raw original/model boxes, actual sx/sy, frozen M. Original M axes sampled then B-model ROI swap. No new image evidence.',
        outputs='Two continuous log-size residuals from probability expectation minus same-shaped uniform expectation. M center, raw/canonical angle, score, output identity copied. Original delivery guards and whole-pair fallback. No fine branch.',
        supervision='distribution: two-adjacent-bin barycentric soft CE for log(GT/M); decoded_control: same-forward decoded SmoothL1 beta0.1. No combined objective, IoU weights, distillation, class loss or direction balancing.',
        reference='DFL-style continuous-target distribution supervision discussed in D-FINE Eq5; not full D-FINE/FGL or a claim of novel distribution supervision.',
        training='Full2558 TRAIN identities at standard and existing half scale. All5112 present views, no inherited center or fit/probe filtering;4 missing retained in reports. Each present view exactly once per epoch, seed1703+epoch Python shuffle, batch8 including final partial. Both arms same schedule/init;24 epochs, no sweep/resume.',
        train_gate='Each domain/scale fixed-M need_enlarge and need_shrink meanabs(a) strictly lower by>1e-6; near_correct<=3% no new joint10 errors and meanabs(a) nonincrease within1e-6; all groups present. Identity preserved. Mixed reported.',
        val_gate='Complete887 VAL only fitted TRAIN-passing epochs: Real both MAE lower than M and B; Real P95/abs(a)/abs(r) no worse M; Sim both MAE/P95/a/r no worse M. Each domain joint10>=M. Real fullRIoU>M+1e-5, Sim>=M-1e-5; MCML max/mean nonincrease, TDR nondecrease. Exact center/angle/score/output. DFR/MRF disclosed, AP11 supplemental cache GT.',
        selection='Among TRAIN+VAL passing epochs Real longMAE, shortMAE, earliest epoch lexicographic. Mechanism requires selected distribution Real longMAE<control-1e-6 and shortMAE<=control+1e-6. Control with no admissible epoch uses best TRAIN-passing descriptive reference, explicitly not adopted. No TEST or alternate objective search.',
        engineering='Verify cache byte SHA and3 shards, B weight SHA, selected M SHA/context and887 cached VAL replay. New zero full TRAIN/VAL identity;2updates/supervision discarded, initial stage1 terminal and stage2 stem gradients; exact head/optimizer/RNG save-load. Freeze M and verify digest before/after. Source closure bound to current bytes, historical provenance separately reconstructed.',
        limits=dict(detector_updates=0,frozen_midpoint_updates=0,feature_extractions=0,cache_rebuilt=False,
                    test_access=False,selection_on_test=False,automatic_promotion=False,depth_access=False),
        stop='Bad mappings/targets/SHA/nonfinite/engineering fail stop. Complete fixed budget if engineering passes; no fitted VAL when TRAIN fails. No admissible improvement retains M, no extra module or budget.',
        source_reference='https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/src/zoo/dfine/dfine_criterion.py')


def checked_sources():
    manifest = sealed.read(SOURCES)
    required = set(sealed.read(ROOT/sealed.SIGMA_SOURCES)['sources'])|NEW_FILES
    if manifest['protocol']!=VERSION or set(manifest['sources'])!=required or sealed.read(PROTOCOL)!=protocol_document():
        raise ValueError('Fixed source/protocol scope differs')
    for name,digest in manifest['sources'].items():
        if sealed.sha(ROOT/name)!=digest:
            raise ValueError('Current source SHA differs: '+name)
    return dict(manifest_sha256=sealed.sha(SOURCES),protocol_sha256=sealed.sha(PROTOCOL),sources=manifest['sources'])


def historical_identity():
    for name,digest in sealed.PARENT_HASHES.items():
        if sealed.sha(ROOT/name)!=digest:
            raise ValueError('Historical provenance document changed: '+name)
    formal,sigma = sealed.read(ROOT/sealed.FORMAL_SOURCES),sealed.read(ROOT/sealed.SIGMA_SOURCES)
    parent = dict(protocol_sha256=sealed.PARENT_HASHES[sealed.FORMAL_PROTOCOL],
        sources_sha256=sealed.PARENT_HASHES[sealed.FORMAL_SOURCES],sources=formal['sources'],
        frozen_b=sealed.read(ROOT/sealed.FORMAL_PROTOCOL)['frozen_b'])
    training = dict(protocol_sha256=sealed.PARENT_HASHES[sealed.SIGMA_PROTOCOL],
        sources_sha256=sealed.PARENT_HASHES[sealed.SIGMA_SOURCES],sources=sigma['sources'],parent_identity=parent)
    differences = {n:dict(historical=d,current=sealed.sha(ROOT/n)) for n,d in sigma['sources'].items() if sealed.sha(ROOT/n)!=d}
    allowed = {'mmrotate/core/bbox/assigners/sym_pola.py','mmrotate/models/losses/sym_kld_loss.py',
        'mmrotate/models/losses/sym_nfl_loss.py','mmrotate/models/dense_heads/sym_eood_head.py',
        'mmrotate/models/detectors/sym_eood_detector.py'}
    if set(differences)-allowed:
        raise ValueError('Executed historical dependency changed: '+str(set(differences)-allowed))
    return dict(training_identity=training),differences


def write_rows(path,rows):
    with gzip.open(str(path),'xt',encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')


def read_rows(path):
    with gzip.open(str(path),'rt',encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def seed_all(torch):
    random.seed(SETTINGS['seed']);torch.manual_seed(SETTINGS['seed']);torch.cuda.manual_seed_all(SETTINGS['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def epoch_batches(count,epoch):
    indices=list(range(count));random.Random(SETTINGS['seed']+epoch).shuffle(indices)
    return [indices[i:i+SETTINGS['batch_size']] for i in range(0,count,SETTINGS['batch_size'])]


def load_records(directory,cache,frozen,device,torch,model,references):
    train,val=[],[]
    bytes_read=0
    for label in ('train_s1','train_s05','val_s1'):
        path=directory/(label+'.pt')
        if sealed.sha(path)!=cache['files'][path.name]:
            raise ValueError('Immutable shard SHA differs: '+label)
        payload=sealed.torch_load(torch,path);expected,role,_=sealed.SHARDS[label]
        records=payload['records']
        if (payload['identity']!=cache['identity'] or payload['detector_state']!=cache['detector_state'] or
            len(records)!=expected or len({r['image'] for r in records})!=expected or
            Counter(r['sequence'] for r in records)!=Counter(sealed.TRAIN_COUNTS if role=='train' else sealed.VAL_COUNTS)):
            raise ValueError('Shard identity/count/state differs')
        prepared=[]
        with torch.no_grad():
            for r in records:
                bytes_read+=sealed.checked_record(r,label,torch,model.residual.original)
                inputs=[r[k].to(device) for k in sealed.TENSOR_KEYS[:-1]]
                output=frozen(*inputs);m=output['boxes_original'].detach().cpu()
                record=dict(r,shard=label,midpoint_original=m)
                if role=='val':
                    source=dict(image=r['image'],sequence=r['sequence'],domain=r['domain'],frame_id=r['frame_id'],
                        scale=r['scale'],gt=r['gt_original'][0].tolist(),b=inputs[2][0].cpu().tolist() if len(m) else None,
                        midpoint=m[0].tolist() if len(m) else None,
                        diagnosis=dict(baseline_joint10=False,baseline_center_hit=False))
                    if len(m):
                        source['diagnosis']=dict(baseline_joint10=max(abs(x/y-1) for x,y in zip(sorted(source['midpoint'][2:4]),sorted(source['gt'][2:4])))<=.1,
                            baseline_center_hit=math.hypot(m[0,0]-r['gt_original'][0,0],m[0,1]-r['gt_original'][0,1])<15.)
                        sealed.compare_reference(source,references[r['image']])
                    if (bool(output['accepted'][0]) if len(m) else None)!=references[r['image']]['accepted']:
                        raise ValueError('M accepted/fallback replay differs')
                prepared.append(record)
        (train if role=='train' else val).extend(prepared)
        print('loaded/replayed',label,len(records),flush=True)
        if bytes_read>1024**3:
            raise ValueError('CPU cache budget exceeded')
    return sorted(train,key=lambda r:(r['scale'],r['image'])),sorted(val,key=lambda r:r['image']),bytes_read


def pack(records,device,torch):
    indices=[i for i,r in enumerate(records) if len(r['midpoint_original'])]
    keys=sealed.TENSOR_KEYS[:-1]+('midpoint_original','gt_original')
    return dict(indices=indices,keys=keys,tensors=[torch.cat([records[i][k] for i in indices]).to(device) for k in keys])


def evaluate_rows(head,records,packed,torch,batch_size=64):
    head.eval();values={}
    with torch.no_grad():
        for start in range(0,len(packed['indices']),batch_size):
            tensors=[x[start:start+batch_size] for x in packed['tensors']]
            output=head(*tensors[:-1]);target=head_module().size_targets(tensors[-1],tensors[-2],tensors[3])['log_residual']
            assignment=head_module().distribution_targets(target,output['bucket_logits'],output['grid'])
            cov=output['border_support']
            left=cov.gather(2,assignment['left'][:,:,None,None].expand(-1,-1,1,2)).squeeze(2)
            right=cov.gather(2,assignment['right'][:,:,None,None].expand(-1,-1,1,2)).squeeze(2)
            weighted=left*assignment['weight_left'][...,None]+right*assignment['weight_right'][...,None]
            outputs={k:output[k].detach().cpu().tolist() for k in ('boxes_original','candidate_original','delta_roi','accepted','candidate_valid','neutral_bypass','square_bypass')}
            check={k:v.detach().cpu().tolist() for k,v in output['checks'].items()}
            target_cpu,coverage=target.cpu().tolist(),weighted.cpu().tolist()
            for j,index in enumerate(packed['indices'][start:start+batch_size]):
                status='square' if outputs['square_bypass'][j] else 'neutral' if outputs['neutral_bypass'][j] else 'accepted' if outputs['accepted'][j] else 'fallback'
                values[index]=dict(edge_residual=outputs['boxes_original'][j],
                    candidate=outputs['candidate_original'][j] if outputs['candidate_valid'][j] else None,
                    delta_roi=outputs['delta_roi'][j],size_delivery=status,
                    failed_checks=[k for k,v in check.items() if not v[j]],
                    target_log=target_cpu[j],target_weighted_border_support=coverage[j])
    rows=[]
    for i,r in enumerate(records):
        present=bool(len(r['midpoint_original']))
        row={k:r[k] for k in ('image','sequence','domain','frame_id','scale','split','shard')}
        row.update(gt=r['gt_original'][0].tolist(),b=r['boxes_original'][0].tolist() if present else None,
            midpoint=r['midpoint_original'][0].tolist() if present else None,
            **values.get(i,dict(edge_residual=None,candidate=None,delta_roi=None,size_delivery='no_output',
                               failed_checks=[],target_log=None,target_weighted_border_support=None)))
        if not metrics.old.frozen_identity(row):
            raise ValueError('M center/angle/score/output identity changed')
        rows.append(row)
    return rows


def head_module():
    from crane_project.utils import port_geometry_size_distribution_v1 as model
    return model


def update(head,opt,packed,indices,arm,torch):
    tensors=[x[indices] for x in packed['tensors']]
    head.train();opt.zero_grad()
    output=head(*tensors[:-1]);terms=head_module().loss(output,tensors[-1],tensors[-2],tensors[3],arm)
    if not bool(torch.isfinite(terms['total'])):
        raise ValueError('Nonfinite loss')
    terms['total'].backward()
    params=list(head.parameters())
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
        raise ValueError('Missing/nonfinite gradient')
    norm=lambda ps:math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in ps))
    stem,terminal=norm(head.stem.parameters()),norm(list(head.u.parameters())+list(head.v.parameters()))
    before=float(torch.nn.utils.clip_grad_norm_(params,SETTINGS['clip_norm']))
    opt.step()
    if any(not bool(torch.isfinite(p).all()) for p in params):
        raise ValueError('Nonfinite updated parameters')
    return dict(loss=float(terms['total'].detach()),soft_ce=float(terms['soft_ce'].detach()),
        decoded_loss=float(terms['decoded'].detach()),stem_grad_norm=stem,terminal_grad_norm=terminal,
        grad_norm_before_clip=before,clipped=before>SETTINGS['clip_norm'])


def save_checkpoint(path,head,opt,arm,epoch,updates,proof,torch):
    rng=dict(python=random.getstate(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    payload=dict(protocol=VERSION,arm=arm,epoch=epoch,updates=updates,proof=proof,
        head_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
        optimizer_state=deepcopy(opt.state_dict()) if opt is not None else None,rng=rng,head_digest=sealed.state_digest(head))
    with path.open('xb') as f:
        torch.save(payload,f)
    restored=sealed.torch_load(torch,path)
    reloaded=head_module().DistributionSizeHead().to(next(head.parameters()).device)
    reloaded.load_state_dict(restored['head_state'],strict=True)
    if sealed.state_digest(reloaded)!=payload['head_digest']:
        raise ValueError('Checkpoint state serialization differs')
    if opt is not None:
        replay_opt=torch.optim.Adam(reloaded.parameters(),lr=SETTINGS['lr'],weight_decay=0.)
        replay_opt.load_state_dict(restored['optimizer_state'])
        for a,b in zip(opt.state_dict()['state'].values(),replay_opt.state_dict()['state'].values()):
            for k in a:
                if hasattr(a[k],'shape'):
                    if not torch.equal(a[k].cpu(),b[k].cpu()):raise ValueError('Optimizer replay differs')
                elif a[k]!=b[k]:raise ValueError('Optimizer step replay differs')
    if restored['rng']['python']!=rng['python'] or not torch.equal(restored['rng']['cpu'],rng['cpu']) or any(not torch.equal(a,b) for a,b in zip(restored['rng']['cuda'],rng['cuda'])):
        raise ValueError('RNG serialization differs')
    random.setstate(rng['python']);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
    return reloaded,dict(path=path.name,sha256=sealed.sha(path),head_digest=payload['head_digest'],epoch=epoch,updates=updates,save_reload_state_optimizer_rng_exact=True)


def publish(out,report):
    sealed.write(out/'completion.json',report)
    files={str(p.relative_to(out)):sealed.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifacts.json'}
    sealed.write(out/'artifacts.json',dict(protocol=VERSION,files=files))


def check_out(path):
    out=Path(path).resolve()
    base=(ROOT/'work_dirs'/VERSION).resolve()
    if out.parent!=base or out.exists():
        raise ValueError('New direct child of work_dirs/'+VERSION+' required; no overwrite')
    return out


def run(args):
    identity=checked_sources();out=check_out(args.out_dir)
    if args.stage=='run':
        if not all((args.cache_dir,args.head_dir,args.b_weight)):
            raise ValueError('run requires explicit cache-dir, head-dir and b-weight')
        for name in (args.cache_dir,args.head_dir,args.b_weight):
            source=Path(name).resolve()
            if out==source or out in source.parents or source in out.parents:
                raise ValueError('Output overlaps an immutable input')
    out.mkdir(parents=True)
    proof=dict(source_identity=identity,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip())
    report=dict(protocol=VERSION,started_utc=datetime.now(timezone.utc).isoformat(),status='FAILED',
        stage=args.stage,proof=proof,**protocol_document()['limits'])
    started=time.monotonic()
    try:
        sealed.write(out/'protocol.json',protocol_document());sealed.write(out/'source_identity.json',identity)
        if args.stage=='check':
            report.update(status='STATIC_CONTRACT_PASS',head_updates=0,model_loaded=False)
            publish(out,report);print(report['status'],flush=True);return
        import torch
        from crane_project.utils import port_geometry_midpoint_sigma_v1 as frozen_model
        model=head_module()
        if os.environ.get('WORLD_SIZE','1')!='1' or not torch.cuda.is_available() or len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))!=1 or not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise ValueError('Explicit exactly one CUDA_VISIBLE_DEVICES GPU required')
        torch.set_num_threads(1);device=torch.device('cuda:0');torch.cuda.set_device(device);seed_all(torch)
        historical,differences=historical_identity()
        cache_dir=Path(args.cache_dir).resolve();head_dir=Path(args.head_dir).resolve();b_weight=Path(args.b_weight).resolve()
        if sealed.sha(b_weight)!=B_SHA:
            raise ValueError('B actual weight SHA differs')
        cache=sealed.checked_cache(cache_dir,historical)
        selection,references,indexed=sealed.checked_selection(head_dir,historical)
        runtime=dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            tf32_matmul=getattr(torch.backends.cuda.matmul,'allow_tf32',None),tf32_cudnn=getattr(torch.backends.cudnn,'allow_tf32',None))
        proof.update(cache_manifest_sha256=sealed.CACHE_SHA,shards=cache['files'],fixed_m=sealed.FIXED,
            fixed_b_sha256=B_SHA,reference_artifacts=indexed,runtime=runtime,
            historical_identity=historical,unused_detector_historical_differences=differences)
        sealed.write(out/'input_runtime_proof.json',proof)
        frozen=sealed.load_head(torch,frozen_model,head_dir,selection,cache,device)
        frozen_digest=sealed.state_digest(frozen)
        train,val,bytes_read=load_records(cache_dir,cache,frozen,device,torch,model,references)
        trainpack,valpack=pack(train,device,torch),pack(val,device,torch)
        if len(train)!=5116 or len(trainpack['indices'])!=5112 or len(val)!=887 or len(valpack['indices'])!=886:
            raise ValueError('Full TRAIN/VAL output counts differ')
        schedules=[epoch_batches(5112,e) for e in range(1,25)]
        sealed.write(out/'schedule.json',dict(train_view_identity=[dict(image=train[i]['image'],scale=train[i]['scale']) for i in trainpack['indices']],
            batches=schedules,epochs=24,steps_per_epoch=639,updates_per_arm=15336))
        seed_all(torch);neutral=model.DistributionSizeHead().to(device);initial=sealed.state_digest(neutral)
        baseline_train=evaluate_rows(neutral,train,trainpack,torch);baseline_val=evaluate_rows(neutral,val,valpack,torch)
        if any(r['midpoint']!=r['edge_residual'] for r in baseline_train+baseline_val):
            raise ValueError('Zero update full TRAIN/VAL not exactly M')
        targetvalues=[v for r in baseline_train if r['target_log'] is not None for v in r['target_log']]
        supportvalues=[v for r in baseline_train if r['target_weighted_border_support'] is not None for pair in r['target_weighted_border_support'] for v in pair]
        sealed.write(out/'support.json',dict(target_min=min(targetvalues),target_max=max(targetvalues),target_outside_log1=sum(abs(x)>1 for x in targetvalues),
            gt_weighted_support_mean=sum(supportvalues)/len(supportvalues),zero_target_border_support=sum(x==0 for x in supportvalues),target_border_count=len(supportvalues),
            input_tensor_bytes=bytes_read,train_present=5112,train_missing=4,val_present=886,val_missing=1,
            guards_target_b_outside=sum(any(not .8<=g/b<=1.25 for g,b in zip(model.size_targets(r['gt_original'],r['midpoint_original'],r['boxes_model'])['raw_edges'][0].tolist(),r['boxes_original'][0,2:4].tolist())) for r in train if len(r['midpoint_original'])),
            note='No target-support, center, role or geometry filtering of present TRAIN; support is interpolated evidence, not independent measurements.'))
        write_rows(out/'baseline_train.rows.jsonl.gz',baseline_train);write_rows(out/'baseline_val.rows.jsonl.gz',baseline_val)
        riou_cache={};baseline_summary=metrics.summary(baseline_val,riou_cache)
        sealed.write(out/'baseline_val_summary.json',baseline_summary)
        smoke=[]
        for arm in ARMS:
            seed_all(torch);h=model.DistributionSizeHead().to(device);opt=torch.optim.Adam(h.parameters(),lr=SETTINGS['lr'],weight_decay=0.)
            if sealed.state_digest(h)!=initial:raise ValueError('Unequal initial state')
            events=[update(h,opt,trainpack,b,arm,torch) for b in schedules[0][:2]]
            if events[0]['stem_grad_norm']!=0 or events[0]['terminal_grad_norm']<=0 or events[1]['stem_grad_norm']<=0:
                raise ValueError('Two-stage gradient path failed')
            replay,ckpt=save_checkpoint(out/('smoke_'+arm+'.pth'),h,opt,arm,0,2,proof,torch)
            checkpack=dict(trainpack,tensors=[x[:8] for x in trainpack['tensors']],indices=list(range(8)))
            with torch.no_grad():
                a=h(*checkpack['tensors'][:-1]);b=replay(*checkpack['tensors'][:-1])
                if not torch.equal(a['boxes_original'],b['boxes_original']) or not torch.equal(a['bucket_logits'],b['bucket_logits']):
                    raise ValueError('GPU smoke prediction replay differs')
            smoke.append(dict(arm=arm,events=events,checkpoint=ckpt,prediction_replay_exact=True,discarded=True))
        sealed.write(out/'preflight.json',dict(status='ENGINEERING_PASS',zero_full_identity=True,initial_state=initial,
            full_val_reference_replay=True,smoke=smoke,frozen_digest=frozen_digest,formal_updates=0))
        print('ENGINEERING_PASS; fresh formal initialization',flush=True)
        arm_results={}
        for arm in ARMS:
            directory=out/arm;directory.mkdir();seed_all(torch);h=model.DistributionSizeHead().to(device)
            if sealed.state_digest(h)!=initial:raise ValueError('Smoke state reused')
            opt=torch.optim.Adam(h.parameters(),lr=SETTINGS['lr'],weight_decay=0.)
            epochs=[];best=None;best_descriptive=None
            with (directory/'progress.jsonl').open('x',encoding='utf-8') as log:
                for epoch,batches in enumerate(schedules,1):
                    tick=time.monotonic();events=[]
                    for step,batch in enumerate(batches,1):
                        event=update(h,opt,trainpack,batch,arm,torch);events.append(event)
                        log.write(json.dumps(dict(epoch=epoch,step=step,update=(epoch-1)*639+step,**event),sort_keys=True,allow_nan=False)+'\n')
                        if step%100==0 or step==len(batches):
                            log.flush();print(arm,'epoch',epoch,'step',step,'loss',event['loss'],flush=True)
                    rows=evaluate_rows(h,train,trainpack,torch);tr=metrics.train_report(rows)
                    write_rows(directory/('train_epoch_%02d.rows.jsonl.gz'%epoch),rows)
                    sealed.write(directory/('train_epoch_%02d.summary.json'%epoch),tr)
                    _,checkpoint=save_checkpoint(directory/('head_epoch_%02d.pth'%epoch),h,opt,arm,epoch,epoch*639,proof,torch)
                    result=dict(epoch=epoch,updates=epoch*639,train_pass=tr['train_pass'],train_failed_checks=tr['failed_checks'],
                        checkpoint=checkpoint,mean_loss=sum(e['loss'] for e in events)/len(events),elapsed_seconds=time.monotonic()-tick,
                        val_evaluated=False)
                    if tr['train_pass']:
                        vr=evaluate_rows(h,val,valpack,torch);vs=metrics.summary(vr,riou_cache);gate=metrics.val_gate(vs)
                        write_rows(directory/('val_epoch_%02d.rows.jsonl.gz'%epoch),vr)
                        sealed.write(directory/('val_epoch_%02d.summary.json'%epoch),vs)
                        result.update(val_evaluated=True,val_gate=gate,selection_key=list(metrics.selection_key(vs,epoch)))
                        if best_descriptive is None or result['selection_key']<best_descriptive['selection_key']:
                            best_descriptive=deepcopy(result)
                        if gate['val_pass'] and (best is None or result['selection_key']<best['selection_key']):
                            best=deepcopy(result)
                    epochs.append(result);sealed.write(directory/('epoch_%02d.json'%epoch),result)
                    print('EPOCH_COMPLETE',arm,epoch,'TRAIN',tr['train_pass'],'VAL',result.get('val_gate',{}).get('val_pass'),
                          'failed',tr['failed_checks'],flush=True)
                    if sealed.state_digest(frozen)!=frozen_digest or any(p.requires_grad or p.grad is not None for p in frozen.parameters()):
                        raise ValueError('Frozen M updated')
            arm_results[arm]=dict(selected=best,best_train_qualified_descriptive=best_descriptive,epochs=epochs,updates=15336)
            sealed.write(directory/'selection.json',arm_results[arm])
        candidate=arm_results['distribution']['selected']
        control=arm_results['decoded_control']['selected'] or arm_results['decoded_control']['best_train_qualified_descriptive']
        mechanism = candidate is not None and control is not None and candidate['selection_key'][0]<control['selection_key'][0]-metrics.TOL and candidate['selection_key'][1]<=control['selection_key'][1]+metrics.TOL
        if checked_sources()!=identity or sealed.sha(head_dir/sealed.FIXED['path'])!=sealed.FIXED['sha256'] or sealed.sha(b_weight)!=B_SHA or sealed.sha(cache_dir/'cache_manifest.json')!=sealed.CACHE_SHA:
            raise ValueError('End-run source/weight/cache identity changed')
        report.update(status='COMPLETE_REVIEW_REQUIRED' if mechanism else 'COMPLETE_HYPOTHESIS_NOT_SUPPORTED_KEEP_M',
            arms=arm_results,mechanism_supported=mechanism,head_updates=30672,train_views=5116,val_frames=887,
            fitted_val_evaluations=sum(sum(e['val_evaluated'] for e in v['epochs']) for v in arm_results.values()),
            frozen_m_state_unchanged=sealed.state_digest(frozen)==frozen_digest,
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(0))
    except BaseException as exc:
        report['error']=repr(exc);report['elapsed_seconds']=time.monotonic()-started
        publish(out,report);raise
    report['elapsed_seconds']=time.monotonic()-started
    publish(out,report);print(report['status'],flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('check','run'),required=True)
    p.add_argument('--out-dir',required=True)
    p.add_argument('--cache-dir');p.add_argument('--head-dir');p.add_argument('--b-weight')
    return p


if __name__=='__main__':
    run(parser().parse_args())
