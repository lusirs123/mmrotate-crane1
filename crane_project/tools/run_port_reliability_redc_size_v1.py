#!/usr/bin/env python3
"""Full TRAIN -> full VAL, fixed final epoch; no historical sub-role filtering.

Only the predeclared ReDC arm may be selected. TEST remains gated and is never
used to alter features, budgets, weights or thresholds. Refuse any overwrite.
"""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.utils import port_reliability_redc_size_v1 as core
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_feature_ablation_v1 as ab
from crane_project.utils import port_reliability_state_continuity_v1 as states
from crane_project.utils import port_midpoint_sigma15_reliability_v1 as binding

PROTOCOL = ROOT/'crane_project/tools/port_reliability_redc_size_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_reliability_redc_size_v1_sources.json'
PRIMARY = 'work_dirs/port_reliability_feature_ablation_v1/20261008_size_features_v1/fit/scored_TRAIN_VAL.jsonl.gz'
CACHE = 'work_dirs/port_results/geometry/port_geometry_midpoint_formal_v1_roi_cache'
HEAD = 'work_dirs/crane_symeood_k1_port_day2night_midpoint_sigma_v1/sigma_1p5/head_epoch_03.pth'
POLICY = 'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit/policy.json'
PINS = {PRIMARY:'e804241c2b7d529ed816f9d2f33dedad1e2961a5b132426bc816777b324c03b4',
        HEAD:binding.HEAD_SHA,POLICY:'38d914f114fcdb257f33f1cd6390ebcb4e103adfa1d9022628875e137c916e1f',
        CACHE+'/cache_manifest.json':'046c5998dee0ba3703f1ae4e08fc6a02e9804d216ea12241357f69f2a4afd1e3',
        CACHE+'/train_s1.pt':'903ae2a2146665e6caa557fee6bed4cd6cadd51f842e6ee9e13bf82482976ec0',
        CACHE+'/val_s1.pt':'88a189cd707e2406acafbfabb198cbb71d72e118617a2cb1ed93702fff1c70db'}
FIELDS = ('image','sequence','domain','split','frame_id','image_size','gt','pred',
          'train_angle_eligible','angle_axis_well_defined','b_original','reliability_role',
          'size_risks','original_simple_decision')
SOURCE_FILES = [
    'crane_project/utils/port_reliability_redc_size_v1.py',
    'crane_project/utils/port_reliability_redc_size_v1_torch.py',
    'crane_project/tools/run_port_reliability_redc_size_v1.py',
    'crane_project/tools/port_reliability_redc_size_v1_protocol.json',
    'tests/test_port_reliability_redc_size_v1.py',
    'tools/run_port_reliability_redc_size_v1.sh',
    'crane_project/utils/port_simple_component_reliability_v1.py',
    'crane_project/utils/port_reliability_feature_ablation_v1.py',
    'crane_project/utils/port_reliability_separability_v1.py',
    'crane_project/utils/port_reliability_state_continuity_v1.py',
    'crane_project/utils/port_reliability_tradeoff_v1.py',
    'crane_project/utils/port_midpoint_sigma15_reliability_v1.py',
    'crane_project/utils/port_geometry_midpoint_v1.py',
    'crane_project/utils/port_geometry_midpoint_sigma_v1.py',
    'crane_project/utils/port_geometry_refine_g_v1.py',
    'crane_project/utils/port_structure_reliability_v1.py']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')


def checked_sources():
    protocol=json.loads(PROTOCOL.read_text()); manifest=json.loads(SOURCES.read_text())
    actual={name:sha(ROOT/name) for name in SOURCE_FILES}
    if (manifest!=dict(protocol=core.VERSION,sources=actual)
            or protocol['protocol']!=core.VERSION or protocol['settings']!=core.SETTINGS
            or protocol['input_pins']!=PINS):
        raise ValueError('Fixed source/design changed')
    return protocol,dict(manifest_sha256=sha(SOURCES),sources=actual)


def load_rows():
    policy=json.loads((ROOT/POLICY).read_text())
    runtime=binding.Sigma15Reliability(policy,policy['front_end'])
    with gzip.open(ROOT/PRIMARY,'rt') as f:
        rows=[{k:r[k] for k in FIELDS} for r in map(json.loads,f)]
    for role,counts in (('train',{'real':1810,'sim':748}),('val',{'real':375,'sim':512})):
        part=[r for r in rows if r['reliability_role']==role]
        if Counter(r['domain'] for r in part)!=counts:
            raise ValueError('Complete original split counts changed')
        states.checked_order(part)
    if len(rows)!=3445:
        raise ValueError('Unexpected TRAIN/VAL membership')
    for r in rows:
        if runtime.decide(r['pred'],r['image_size'])!=r['original_simple_decision']:
            raise ValueError('Original policy/box/flags no longer replay: '+r['image'])
        if r['size_risks']['full_simple']!=r['original_simple_decision']['risks']['size']:
            raise ValueError('Frozen size score differs')
    return rows,policy


def load_trusted(path):
    import torch
    # Only byte-pinned repository-owned tensor/checkpoint files, no user pickle.
    return torch.load(str(path),map_location='cpu')


def collect(rows, head, device, role, out):
    import torch
    from crane_project.utils import port_reliability_redc_size_v1_torch as native
    shard_name='train_s1.pt' if role=='train' else 'val_s1.pt'
    shard=load_trusted(ROOT/CACHE/shard_name)
    manifest=json.loads((ROOT/CACHE/'cache_manifest.json').read_text())
    if (shard['identity']!=manifest['identity'] or shard['detector_state']!=manifest['detector_state']):
        raise ValueError('Frozen cache identity differs')
    records=shard['records']; index={c['image']:c for c in records}
    if len(index)!=len(records) or set(index)!={r['image'] for r in rows}:
        raise ValueError('Complete frame cache identity differs')
    ordered=sorted(rows,key=lambda r:r['image']); records=[index[r['image']] for r in ordered]
    max_gt_angle=0.
    for r,c in zip(ordered,records):
        if (c['role']!=role or c['scale']!=1.
                or any(c[k]!=r[k] for k in ('image','sequence','domain','split','frame_id','image_size'))):
            raise ValueError('Cache metadata/scale differs')
        n=0 if r['pred'] is None else 1
        if (c['roi'].shape!=(n,256,9,9) or c['support'].shape!=(n,1,9,9)
                or c['boxes_original'].shape!=(n,6) or c['gt_original'].shape!=(1,5)):
            raise ValueError('Frozen cache shape/missing identity differs')
        g=c['gt_original'][0].tolist()
        d=abs((g[4]-r['gt'][4]+np.pi/2)%np.pi-np.pi/2)
        if g[:4]!=r['gt'][:4] or d>2e-6:
            raise ValueError('GT center/size or periodic-angle pairing changed')
        max_gt_angle=max(max_gt_angle,d)
        if n and c['boxes_original'][0].tolist()!=r['b_original']:
            raise ValueError('Original B cache differs')
        for k in ('roi','support','boxes_original','boxes_model','scale_xy'):
            if c[k].requires_grad or not bool(torch.isfinite(c[k]).all()):
                raise ValueError('Frozen finite detached inputs required')
    extracted=native.freeze_features(head,records,device)
    maximum=np.zeros(6); x=[]; selected=[]
    for r,(f,decoded) in zip(ordered,extracted):
        if r['pred'] is None:
            if len(decoded['boxes_original']) or f is not None:
                raise ValueError('Missing output restored')
            continue
        replay=decoded['boxes_original'][0].detach().cpu().numpy().astype(float)
        stored=np.asarray(r['pred'],dtype=float)
        delta=np.abs(replay-stored); delta[4]=abs((replay[4]-stored[4]+np.pi/2)%np.pi-np.pi/2)
        maximum=np.maximum(maximum,delta)
        if (not np.allclose(replay[:4],stored[:4],atol=5e-4,rtol=2e-5)
                or delta[4]>2e-6 or replay[5]!=stored[5]):
            raise ValueError('Frozen M replay differs: '+r['image'])
        if ab.size_bad(dict(r,pred=replay.tolist()))!=ab.size_bad(r):
            raise ValueError('Cross-device replay changes supervision label')
        # Store original delivered M values, never replace with this replay.
        x.append(core.descriptor(r['pred'],r['image_size'],f)); selected.append(r['image'])
    features=np.asarray(x,dtype=float)
    np.savez_compressed(out/(role+'_features.npz'),features=features,images=np.asarray(selected))
    print('Frozen geometry features',role,len(ordered),'outputs',len(x),flush=True)
    return ordered,features,selected,dict(frames=len(ordered),outputs=len(x),
        maximum_native_replay_difference=maximum.tolist(),maximum_GT_periodic_rounding=max_gt_angle,
        stored_M_never_replaced=True,new_detector_inferences=0,head_updates=0)


def score(rows, features, ids, models, normalizer):
    result={name:{r['image']:None for r in rows} for name in (*core.ARMS,'full_simple','score_only')}
    for arm in core.ARMS:
        values=simple.sigmoid(core.numpy_logits(models[arm],features,normalizer,arm))
        for image,v in zip(ids,values):result[arm][image]=float(v)
    for r in rows:
        if r['pred'] is not None:
            result['full_simple'][r['image']]=r['size_risks']['full_simple']
            result['score_only'][r['image']]=1-r['pred'][5]
    return result


def save_scored(path, rows, scores, cutoffs):
    with gzip.open(path,'xt') as stream:
        for r in rows:
            decisions={}
            for arm in core.ARMS:
                decisions[arm]=core.decide(r['original_simple_decision'],scores[arm][r['image']],cutoffs[arm]['risk_le'])
                original=deepcopy(r['original_simple_decision'])
                if r['pred'] is not None:
                    original['risks']['size']=scores[arm][r['image']]
                    original['size_accepted']=scores[arm][r['image']]<=cutoffs[arm]['risk_le']
                if decisions[arm]!=original:
                    raise ValueError('Non-size field changed')
            stream.write(json.dumps(dict(r,experiment_risks={k:v[r['image']] for k,v in scores.items()},
                                         candidate_decisions=decisions),allow_nan=False)+'\n')


def run(args):
    import torch
    from crane_project.utils import port_reliability_redc_size_v1_torch as native
    from crane_project.utils.port_geometry_midpoint_sigma_v1 import SigmaMidpointHead
    protocol,source=checked_sources()
    if args.out.resolve().parent.parent!=(ROOT/'work_dirs'/core.VERSION).resolve():
        raise ValueError('Use work_dirs/'+core.VERSION+'/RUN_ID/result')
    if args.out.exists():raise FileExistsError('Refuse overwrite')
    args.out.mkdir(parents=True)
    before={name:sha(ROOT/name) for name in PINS}
    if before!=PINS:raise ValueError('Frozen input changed')
    rows,policy=load_rows(); fingerprint=simple.fingerprint(rows)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    device='cpu' if args.gpu<0 else 'cuda:'+str(args.gpu)
    head=SigmaMidpointHead(1.5).to(device)
    payload=load_trusted(ROOT/HEAD)
    if (payload['sigma_cells']!=1.5 or payload['epoch']!=3 or payload['updates']!=2706
            or payload['frozen_b']['checkpoint_sha256']!=binding.B_SHA):
        raise ValueError('Wrong frozen head identity')
    head.load_state_dict(payload['head_state'],strict=True); head.eval(); head.requires_grad_(False)
    head_before={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    train,tx,tids,train_proof=collect([r for r in rows if r['reliability_role']=='train'],head,device,'train',args.out)
    indexed={r['image']:r for r in train}
    labels=np.asarray([ab.size_bad(indexed[i]) for i in tids],dtype=np.float32)
    if len(labels)!=2558 or int(labels.sum())!=71:
        raise ValueError('Complete TRAIN error support identity differs')
    normalizer=core.normalization(tx)
    x=torch.tensor(core.normalize(tx,normalizer),dtype=torch.float32,device=device)
    z=torch.tensor(tx[:,0],dtype=torch.float32,device=device)
    y=torch.tensor(labels,device=device)
    w=torch.tensor(core.class_weights(labels),dtype=torch.float32,device=device)
    # Smoke has separate freshly initialized heads/optimizers and is discarded.
    smoke=native.make_models(normalizer,device); checks={}
    idx=np.r_[np.flatnonzero(labels==0)[:4],np.flatnonzero(labels==1)[:4]]
    idx=torch.tensor(idx,dtype=torch.long,device=device)
    for arm,model in smoke.items():
        initial=model(x,z).detach()
        if not torch.allclose(initial,-z,rtol=0,atol=1e-5):
            raise ValueError('Neutral initialization does not reproduce score risk')
        opt=native.optimizer(model)
        updates=[native.update(model,opt,x[idx],z[idx],y[idx],w[idx]) for _ in range(2)]
        actual=model(x,z).detach().cpu().numpy()
        reference=core.numpy_logits(native.exported(model),tx,normalizer,arm)
        difference=float(np.max(np.abs(simple.sigmoid(actual)-simple.sigmoid(reference))))
        if difference>1e-4:raise ValueError('Torch/Numpy risk replay differs')
        if arm=='redc' and not any(v>0 for k,v in updates[1]['gradient_by_parameter'].items() if k.startswith('network.0')):
            raise ValueError('Feature-conditioned branch has no post-initialization gradient')
        checks[arm]=dict(neutral_score_identity=True,discarded_updates=updates,maximum_risk_replay_difference=difference)
    write(args.out/'smoke_report.json',checks)
    del smoke
    models=native.make_models(normalizer,device)
    optimizers={arm:native.optimizer(model) for arm,model in models.items()}
    counts={arm:0 for arm in core.ARMS}; rng=np.random.RandomState(1701)
    order_digest=hashlib.sha256(); start=time.monotonic()
    with (args.out/'train_log.jsonl').open('x') as stream:
        for epoch in range(1,101):
            order=rng.permutation(len(tx)); order_digest.update(order.astype('<i8').tobytes())
            for slot,offset in enumerate(range(0,len(tx),256)):
                indices=torch.tensor(order[offset:offset+256],dtype=torch.long,device=device)
                for arm,model in models.items():
                    value=native.update(model,optimizers[arm],x[indices],z[indices],y[indices],w[indices])
                    counts[arm]+=1
                    stream.write(json.dumps(dict(epoch=epoch,slot=slot,arm=arm,batch_count=len(indices),**value),allow_nan=False)+'\n')
            if epoch%10==0:
                print('ReDC TRAIN',epoch,'/100',counts,flush=True)
    exported={arm:native.exported(model) for arm,model in models.items()}
    for arm,model in models.items():
        actual=model(x,z).detach().cpu().numpy()
        reference=core.numpy_logits(exported[arm],tx,normalizer,arm)
        delta=float(np.max(np.abs(simple.sigmoid(actual)-simple.sigmoid(reference))))
        if delta>1e-4:raise ValueError('Final Torch/Numpy ranking readout differs')
        checks[arm]['final_maximum_risk_replay_difference']=delta
    write(args.out/'models.json',dict(protocol=core.VERSION,settings=core.SETTINGS,
        normalizer=normalizer,models=exported,epoch=100,update_counts=counts,probability_claim=False))
    torch.save(dict(models={k:v.state_dict() for k,v in models.items()},normalizer=normalizer,
                    epoch=100,protocol=core.VERSION),str(args.out/'final_heads.pth'))
    reloaded=load_trusted(args.out/'final_heads.pth')
    if any(not torch.equal(v.detach().cpu(),reloaded['models'][arm][k].cpu())
           for arm,m in models.items() for k,v in m.state_dict().items()):
        raise ValueError('Final checkpoint reload differs')
    # Complete VAL is read only after the fixed final TRAIN weights exist.
    val,vx,vids,val_proof=collect([r for r in rows if r['reliability_role']=='val'],head,device,'val',args.out)
    scores=score(val,vx,vids,exported,normalizer)
    cutoffs={arm:ab.calibrate(val,values,.95) for arm,values in scores.items()}
    statistics={arm:core.describe(val,scores,arm,cutoffs[arm]['risk_le']) for arm in core.ARMS}
    gate=core.gate(statistics['redc'])
    train_scores=score(train,tx,tids,exported,normalizer)
    train_statistics={arm:core.describe(train,train_scores,arm,cutoffs[arm]['risk_le']) for arm in core.ARMS}
    save_scored(args.out/'scored_TRAIN.jsonl.gz',train,train_scores,cutoffs)
    save_scored(args.out/'scored_VAL.jsonl.gz',val,scores,cutoffs)
    write(args.out/'cutoffs.json',cutoffs)
    fixed={name:{arm:states.summarize(group,{r['image'] for r in group if r['pred'] is not None
                and scores[arm][r['image']]<=cutoffs[arm]['risk_le']}) for arm in scores}
           for name,group in ab.groups(val).items()}
    formal={name:states.summarize(group,{r['image'] for r in group if r['original_simple_decision']['size_accepted']})
            for name,group in ab.groups(val).items()}
    centers={name:dict(frames=len(group),outputs=sum(r['pred'] is not None for r in group),
        hits=sum(simple.geometry_errors(r['gt'],r['pred'])['center_px']<15 for r in group if r['pred'] is not None))
        for name,group in ab.groups(val).items()}
    for c in centers.values():
        c.update(output_coverage=c['outputs']/c['frames'],
                 hit_rate_on_outputs=c['hits']/c['outputs'] if c['outputs'] else None,
                 correct_coverage_all_frames=c['hits']/c['frames'])
    if (simple.fingerprint(rows)!=fingerprint or any(not torch.equal(v,head.state_dict()[k].cpu()) for k,v in head_before.items())
            or any(p.grad is not None for p in head.parameters()) or {name:sha(ROOT/name) for name in PINS}!=before
            or checked_sources()[1]!=source or counts!={'linear':1000,'redc':1000}):
        raise ValueError('Frozen inputs/sources/state or fixed update budget changed')
    report=dict(protocol=core.VERSION,status='VAL_PASS_FROZEN_TEST_PENDING' if gate['passed'] else 'VAL_FAILED_STOP',
        sources=source,contract=protocol,input_sha256=before,train_feature_proof=train_proof,val_feature_proof=val_proof,
        train_statistics=train_statistics,VAL_statistics=statistics,VAL_cutoffs=cutoffs,VAL_fixed_points=fixed,
        original_policy_summary=formal,center=centers,gate=gate,engineering=checks,
        update_counts=counts,batch_order_sha256=order_digest.hexdigest(),train_bad=int(labels.sum()),
        train_rows=len(train),val_rows=len(val),parameter_counts={arm:sum(p.numel() for p in m.parameters()) for arm,m in models.items()},
        elapsed_seconds=time.monotonic()-start,GT_online=False,TEST_read=False,
        original_policy_changed=False,boxes_scores_output_center_angle_unchanged=True,
        VAL_used_for_thresholds_and_development=True,TEST_repeatedly_exposed=True,
        fixed_final_epoch=100,smoke_reused=False,new_data_roles=False,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip())
    write(args.out/'report.json',report)
    if gate['passed']:
        write(args.out/'selected_policy.json',dict(protocol=core.VERSION,model=exported['redc'],
              normalizer=normalizer,cutoff=cutoffs['redc'],original_policy=policy,arm='redc'))
    else:
        (args.out/'TEST_SKIPPED.txt').write_text('VAL gate failed. No TEST reads, added epochs or linear-arm reselection.\n')
    write(args.out/'completion.json',dict(protocol=core.VERSION,status=report['status'],
        TEST_read=False,selected_arm=gate['selected_arm'],artifacts={p.name:sha(p) for p in args.out.iterdir() if p.is_file()}))
    print(report['status'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args)
    except Exception as e:
        if args.out.exists() and not (args.out/'failure.json').exists():
            write(args.out/'failure.json',dict(error=repr(e),status='ENGINEERING_FAILED_STOP'))
        raise


if __name__=='__main__':main()
