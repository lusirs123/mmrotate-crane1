#!/usr/bin/env python3
"""Fixed full TRAIN -> full VAL box-conditioned size judgement comparison."""
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
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_reliability_box_contrast_v1 as core
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_feature_ablation_v1 as ab
from crane_project.utils import port_reliability_state_continuity_v1 as states
from crane_project.tools import run_port_reliability_feature_source_v1 as historical
prior = historical.prior
sha, write = prior.sha, prior.write
PROTOCOL = ROOT/'crane_project/tools/port_reliability_box_contrast_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_reliability_box_contrast_v1_sources.json'
NEW_FILES = [
    'crane_project/utils/port_reliability_box_contrast_v1.py',
    'crane_project/utils/port_reliability_box_contrast_v1_torch.py',
    'crane_project/tools/run_port_reliability_box_contrast_v1.py',
    'crane_project/tools/review_port_reliability_box_contrast_v1.py',
    'crane_project/tools/port_reliability_box_contrast_v1_protocol.json',
    'tests/test_port_reliability_box_contrast_v1.py',
    'tools/run_port_reliability_box_contrast_v1.sh']
SOURCE_FILES = list(dict.fromkeys(historical.SOURCE_FILES+[
    'crane_project/tools/port_reliability_feature_source_v1_sources.json']+NEW_FILES))
PINS = {p:historical.PINS[p] for p in (prior.PRIMARY, prior.HEAD, prior.POLICY,
                                    prior.CACHE+'/cache_manifest.json', historical.B_PATH)}


def checked_sources():
    historical.checked_sources()
    protocol = json.loads(PROTOCOL.read_text()); actual = {p:sha(ROOT/p) for p in SOURCE_FILES}
    if (json.loads(SOURCES.read_text()) != dict(protocol=core.VERSION, sources=actual)
            or protocol['settings'] != core.SETTINGS or protocol['input_pins'] != PINS
            or protocol['protocol'] != core.VERSION): raise ValueError('Fixed sources/contract differs')
    return protocol, dict(manifest_sha256=sha(SOURCES), sources=actual)


def load_rows():
    """Keep saved policy risks; tolerate only cross-libm last-bit replay noise."""
    policy=json.loads((ROOT/prior.POLICY).read_text())
    runtime=prior.binding.Sigma15Reliability(policy,policy['front_end'])
    with gzip.open(ROOT/prior.PRIMARY,'rt') as stream:
        rows=[{k:r[k] for k in prior.FIELDS} for r in map(json.loads,stream)]
    if len(rows)!=3445:raise ValueError('Full TRAIN/VAL membership differs')
    for role,counts in (('train',dict(real=1810,sim=748)),('val',dict(real=375,sim=512))):
        part=[r for r in rows if r['reliability_role']==role]
        if Counter(r['domain'] for r in part)!=counts:raise ValueError('Split counts differ')
        states.checked_order(part)
    for row in rows:
        computed=runtime.decide(row['pred'],row['image_size']);stored=row['original_simple_decision']
        if computed.get('risks') is not None:
            for key,value in computed['risks'].items():
                target=stored['risks'][key]
                if value is None or target is None:
                    if value is not target:raise ValueError('Missing risk identity differs')
                elif abs(value-target)>1e-12:raise ValueError('Policy numerical replay differs')
            computed['risks']=deepcopy(stored['risks'])
        if computed!=stored:raise ValueError('Original box/decision changed: '+row['image'])
        if row['size_risks']['full_simple']!=stored['risks']['size']:raise ValueError('Saved simple score differs')
    return rows,policy


def collect(rows, role, gpu, out):
    """One frozen forward/image. GT is excluded from sampling/online APIs."""
    import torch
    from torch.nn import functional as F
    from mmcv.parallel import collate, scatter
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as frozen
    from crane_project.utils.port_geometry_midpoint_sigma_v1 import SigmaMidpointHead
    from crane_project.utils import port_reliability_box_contrast_v1_torch as native
    if role not in ('TRAIN', 'VAL'): raise ValueError('TRAIN/VAL only; no TEST access')
    cfg = frozen.g.check_cfg(); sources, data_identity = frozen.checked_data()
    source = {r['image']:r for r in sources}; lookup = {r['image']:r for r in rows}
    detector = frozen.build_detector(cfg, gpu); before = frozen.g.state_digest(detector)
    manifest = json.loads((ROOT/prior.CACHE/'cache_manifest.json').read_text())
    if before != manifest['detector_state']: raise ValueError('Frozen B state differs')
    head = SigmaMidpointHead(1.5).cuda(gpu)
    payload = prior.load_trusted(ROOT/prior.HEAD)
    if (payload['sigma_cells'] != 1.5 or payload['epoch'] != 3 or payload['updates'] != 2706
            or payload['frozen_b']['checkpoint_sha256'] != historical.PINS[historical.B_PATH]):
        raise ValueError('Wrong frozen midpoint identity')
    head.load_state_dict(payload['head_state'], strict=True); head.eval(); head.requires_grad_(False)
    head_before = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    dataset = frozen.dataset_for(cfg, role.lower(), 1.)
    names = [Path(info['filename']).stem for info in frozen.infos(dataset)]
    if len(names) != len(lookup) or set(names) != set(lookup): raise ValueError('Complete split membership differs')
    captured = {}; traces = []; maximum_b = np.zeros(6); maximum_m = np.zeros(6)
    changed_features = valid_variants = auxiliary_bad = paired_frames = 0; started = time.monotonic()
    for index, name in enumerate(names):
        row = lookup[name]; value = scatter(collate([dataset[index]], samples_per_gpu=1), [gpu])[0]
        image, metas = value['img'][0], value['img_metas'][0]; meta = metas[0]
        if (len(value['img']) != 1 or image.shape != (1,3,1024,1024) or len(metas) != 1
                or Path(meta['filename']).stem != name or meta.get('flip', False)
                or list(meta['ori_shape'][:2][::-1]) != row['image_size']):
            raise ValueError('Deterministic original view differs: '+name)
        frozen.g.checked_meta(meta); frozen.g.assert_detector_frozen(detector)
        with torch.no_grad():
            fpn = detector.extract_feat(image)
            if fpn[0].shape != (1,256,128,128): raise ValueError('P3 shape differs')
            raw = frozen.g.flatten_prediction(detector.simple_test_from_features(fpn, metas, rescale=False))
            b = frozen.g.flatten_prediction(detector.simple_test_from_features(fpn, metas, rescale=True))
            if len(b) != int(row['pred'] is not None): raise ValueError('Output/missing identity differs')
            trace = dict(image=name, role=role, image_sha256=source[name]['image_sha256'],
                         present=bool(len(b)), scale_factor=np.asarray(meta['scale_factor']).tolist(), GT_online=False)
            if len(b):
                stored_b = np.asarray(row['b_original']); delta = np.abs(b[0]-stored_b)
                delta[4] = abs((b[0,4]-stored_b[4]+np.pi/2)%np.pi-np.pi/2)
                maximum_b = np.maximum(maximum_b, delta)
                if (not np.allclose(b[0,:4], stored_b[:4], atol=5e-4, rtol=2e-5)
                        or delta[4] > 2e-6 or b[0,5] != stored_b[5]): raise ValueError('Original B replay differs: '+name)
                bt, rt = image.new_tensor(b), image.new_tensor(raw)
                if (not torch.equal(bt[:,5], rt[:,5]) or not torch.allclose(
                        frozen.g.map_boxes(rt[:,:5], meta, inverse=True), bt[:,:5], atol=1e-4, rtol=1e-6)):
                    raise ValueError('Original/model coordinate restoration differs')
                roi, support, _ = frozen.g.sample_local(fpn[0], rt[:,:5], meta, 'aligned')
                xy = image.new_tensor(np.asarray(meta['scale_factor'])[:2]).reshape(1,2)
                decoded = head(roi, support, bt, rt[:,:5], xy)['boxes_original'][0].cpu().numpy()
                stored_m = np.asarray(row['pred']); delta = np.abs(decoded-stored_m)
                delta[4] = abs((decoded[4]-stored_m[4]+np.pi/2)%np.pi-np.pi/2)
                maximum_m = np.maximum(maximum_m, delta)
                if (not np.allclose(decoded[:4], stored_m[:4], atol=5e-4, rtol=2e-5)
                        or delta[4] > 2e-6 or decoded[5] != stored_m[5]): raise ValueError('Original M replay differs: '+name)
                if ab.size_bad(dict(row, pred=decoded.tolist())) != ab.size_bad(row):
                    raise ValueError('Replay changes size label')
                # Always sample STORED M; frozen replays above never replace it.
                boxes, keys = core.training_boxes(row['pred']) if role=='TRAIN' else (stored_m[None].copy(), ['M'])
                pooled, candidate_support = native.box_features(fpn[0], boxes, meta)
                after = frozen.g.flatten_prediction(detector.simple_test_from_features(fpn, metas, rescale=False))
                if not np.array_equal(raw, after): raise ValueError('Sampling changed frozen B output')
                # Offline labels only AFTER the GT-free input features are produced.
                targets = core.offline_labels(boxes, row['gt'])
                descriptors = np.stack([simple.descriptor(p, row['image_size']) for p in boxes])
                trace.update(boxes=boxes.tolist(), keys=keys, labels=targets.tolist(),
                    feature_sha256=hashlib.sha256(pooled.astype('<f4').tobytes()).hexdigest(),
                    support_mean=candidate_support.mean((1,2,3)).tolist())
                captured[name] = (pooled, descriptors, targets)
                changed_features += sum(not np.array_equal(pooled[0],v) for v in pooled[1:])
                valid_variants += len(boxes)-1; auxiliary_bad += int(targets[1:].sum())
                paired_frames += int(bool(np.any(targets==0) and np.any(targets==1)))
            traces.append(trace)
        if index%100 == 0 or index==len(names)-1: print('Box-conditioned', role, index+1, '/', len(names), flush=True)
        del value, image, metas, fpn
    if (frozen.g.state_digest(detector) != before or any(p.grad is not None for p in head.parameters())
            or any(not torch.equal(v,head.state_dict()[k].cpu()) for k,v in head_before.items())):
        raise ValueError('Frozen detector/midpoint changed')
    ids = sorted(captured); k = 13 if role=='TRAIN' else 1
    raws = np.zeros((len(ids),k,2304), dtype=np.float32)
    descriptors = np.zeros((len(ids),k,3), dtype=np.float64)
    labels = np.zeros((len(ids),k), dtype=np.float32); mask = np.zeros((len(ids),k), dtype=bool)
    for i, name in enumerate(ids):
        values, ds, ys = captured[name]; size=len(values)
        raws[i,:size], descriptors[i,:size], labels[i,:size], mask[i,:size] = values, ds, ys, True
    np.savez_compressed(out/(role.lower()+'_raw.npz'), features=raws, descriptors=descriptors,
                        labels=labels, mask=mask, images=np.asarray(ids))
    with gzip.open(out/(role.lower()+'_trace.jsonl.gz'), 'xt') as stream:
        for trace in sorted(traces, key=lambda r:r['image']): stream.write(json.dumps(trace, allow_nan=False)+'\n')
    proof = dict(frames=len(rows), outputs=len(ids), data_identity=data_identity, detector_state=before,
        maximum_B_replay_difference=maximum_b.tolist(), maximum_M_replay_difference=maximum_m.tolist(),
        valid_contrasts=valid_variants, changed_pooled_features=changed_features, auxiliary_bad=auxiliary_bad,
        same_image_both_class_frames=paired_frames, original_M_never_replaced=True,
        one_feature_extraction_per_frame=True, elapsed_seconds=time.monotonic()-started,
        detector_updates=0, midpoint_updates=0, GT_online=False)
    if role=='TRAIN' and (not valid_variants or changed_features!=valid_variants or not paired_frames):
        raise ValueError('TRAIN contrast inputs/labels have no actual usable change')
    del detector, head, dataset, captured; torch.cuda.empty_cache()
    return raws, descriptors, labels, mask, ids, proof


def projected(raws, ds, pca):
    n, k, _ = raws.shape
    return core.features(ds.reshape(n*k,3), raws.reshape(n*k,2304), pca).reshape(n,k,259)


def score(rows, matrix, ids, models, normalizer):
    scores = {a:{r['image']:None for r in rows} for a in (*core.ARMS,'full_simple','score_only')}
    for arm in core.ARMS:
        values = simple.sigmoid(core.numpy_logits(models[arm], matrix, normalizer))
        for image, v in zip(ids, values): scores[arm][image] = float(v)
    for row in rows:
        if row['pred'] is not None:
            scores['full_simple'][row['image']] = row['size_risks']['full_simple']
            scores['score_only'][row['image']] = 1-row['pred'][5]
    return scores


def save_scored(path, rows, scores, cutoffs):
    with gzip.open(path, 'xt') as stream:
        for row in sorted(rows, key=lambda r:r['image']):
            decisions = {a:core.decide(row['original_simple_decision'], scores[a][row['image']], cutoffs[a]['risk_le']) for a in core.ARMS}
            stream.write(json.dumps(dict(row, experiment_risks={a:s[row['image']] for a,s in scores.items()},
                                         candidate_decisions=decisions), allow_nan=False)+'\n')


def run(args):
    import torch
    from crane_project.utils import port_reliability_box_contrast_v1_torch as native
    protocol, source = checked_sources()
    if args.out.resolve().parent.parent != (ROOT/'work_dirs'/core.VERSION).resolve(): raise ValueError('Unified experiment/RUN_ID/result required')
    if args.out.exists(): raise FileExistsError('Refuse overwrite')
    before = {p:sha(ROOT/p) for p in PINS}
    if before != PINS: raise ValueError('Frozen inputs changed')
    rows, policy = load_rows(); fingerprint = simple.fingerprint(rows)
    train = [r for r in rows if r['reliability_role']=='train']; val = [r for r in rows if r['reliability_role']=='val']
    args.out.mkdir(parents=True); started=time.monotonic()
    (args.out/'frozen_cache_manifest.json').write_bytes((ROOT/prior.CACHE/'cache_manifest.json').read_bytes())
    torch.set_num_threads(1); torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True); device='cuda:'+str(args.gpu)
    raw, ds, labels, mask, ids, train_proof = collect(train, 'TRAIN', args.gpu, args.out)
    y = labels[:,0]
    if len(y)!=2558 or int(y.sum())!=71: raise ValueError('Genuine complete TRAIN label identity differs')
    print('PCA on original TRAIN M only', raw[:,0].shape, flush=True)
    pca=core.fit_pca(raw[:,0], 'TRAIN'); matrix=projected(raw,ds,pca)
    normalizer=core.normalization(matrix[:,0]); write(args.out/'pca.json',pca)
    np.savez_compressed(args.out/'train_projected.npz', features=matrix, images=np.asarray(ids))
    tensors = [torch.tensor(v, dtype=torch.float32, device=device) for v in
               (core.normalize(matrix,normalizer),ds[:,:,0],labels,core.class_weights(y))]
    x,z,targets,w=tensors; valid=torch.tensor(mask,dtype=torch.bool,device=device)
    idx=torch.tensor(np.r_[np.flatnonzero(y==0)[:4],np.flatnonzero(y==1)[:4]],dtype=torch.long,device=device)
    smoke=native.make_models(device); engineering={}
    for arm,model in smoke.items():
        if not torch.equal(model(x[:,0],z[:,0]),-z[:,0]): raise ValueError('Neutral score identity is not exact')
        opt=native.optimizer(model)
        updates=[native.update(model,opt,x[idx,0],z[idx,0],targets[idx,0],w[idx],
                    x[idx,1:],z[idx,1:],targets[idx,1:],valid[idx,1:],arm,component_gradients=True) for _ in range(2)]
        if not any(v>0 for k,v in updates[1]['gradient_by_parameter'].items() if k.startswith('network.0')):
            raise ValueError('Post-initialization hidden gradient disconnected')
        if arm=='contrast' and updates[1]['components']['weighted_auxiliary_gradient_norm']<=0:
            raise ValueError('Contrast supervision has no actual gradient')
        engineering[arm]=dict(neutral_identity=True,discarded_updates=updates)
    write(args.out/'smoke_report.json',engineering); del smoke
    models=native.make_models(device); initial={a:native.exported(m) for a,m in models.items()}
    if initial['original']!=initial['contrast']: raise ValueError('Initial weights differ')
    write(args.out/'initial_heads.json',initial)
    optimizers={a:native.optimizer(m) for a,m in models.items()}; counts={a:0 for a in core.ARMS}
    order_hash=hashlib.sha256(); rng=np.random.RandomState(core.SETTINGS['seed'])
    with (args.out/'train_log.jsonl').open('x') as stream:
        for epoch in range(1,101):
            order=rng.permutation(len(ids)); order_hash.update(order.astype('<i8').tobytes())
            for slot,offset in enumerate(range(0,len(ids),256)):
                ix=torch.tensor(order[offset:offset+256],dtype=torch.long,device=device)
                for arm,model in models.items():
                    v=native.update(model,optimizers[arm],x[ix,0],z[ix,0],targets[ix,0],w[ix],
                                    x[ix,1:],z[ix,1:],targets[ix,1:],valid[ix,1:],arm,component_gradients=(slot==0))
                    counts[arm]+=1
                    stream.write(json.dumps(dict(epoch=epoch,slot=slot,arm=arm,batch_count=len(ix),**v),allow_nan=False)+'\n')
            if epoch%10==0: print('Box contrast TRAIN',epoch,'/100',counts,flush=True)
    exported={a:native.exported(m) for a,m in models.items()}
    for arm,model in models.items():
        replay=simple.sigmoid(core.numpy_logits(exported[arm],matrix[:,0],normalizer))
        error=float(np.max(np.abs(torch.sigmoid(model(x[:,0],z[:,0])).detach().cpu().numpy()-replay)))
        if error>1e-4: raise ValueError('Torch/NumPy final risk differs')
        engineering[arm]['final_risk_replay_difference']=error
    write(args.out/'models.json',dict(protocol=core.VERSION,models=exported,normalizer=normalizer,
                                    epoch=100,update_counts=counts))
    torch.save(dict(protocol=core.VERSION,models={a:m.state_dict() for a,m in models.items()},
                    normalizer=normalizer,epoch=100),str(args.out/'final_heads.pth'))
    loaded=prior.load_trusted(args.out/'final_heads.pth')
    if any(not torch.equal(v.cpu(),loaded['models'][a][k].cpu()) for a,m in models.items() for k,v in m.state_dict().items()):
        raise ValueError('Checkpoint reload differs')
    # Final epoch is fixed before any VAL feature tensor is sampled.
    vraw,vds,vy,vmask,vids,val_proof=collect(val,'VAL',args.gpu,args.out)
    vmat=projected(vraw,vds,pca)[:,0]
    np.savez_compressed(args.out/'val_projected.npz',features=vmat,images=np.asarray(vids))
    vscores=score(val,vmat,vids,exported,normalizer); tscores=score(train,matrix[:,0],ids,exported,normalizer)
    cutoffs={a:ab.calibrate(val,s,.95) for a,s in vscores.items()}
    vstats={a:core.describe(val,vscores,a,cutoffs[a]['risk_le']) for a in core.ARMS}
    tstats={a:core.describe(train,tscores,a,cutoffs[a]['risk_le']) for a in core.ARMS}
    gate=core.gate(vstats['contrast'])
    save_scored(args.out/'scored_TRAIN.jsonl.gz',train,tscores,cutoffs)
    save_scored(args.out/'scored_VAL.jsonl.gz',val,vscores,cutoffs); write(args.out/'cutoffs.json',cutoffs)
    diagnostics={role:{a:core.hidden_diagnostic(part,identities,m,exported[a],normalizer) for a in core.ARMS}
                 for role,part,identities,m in [('TRAIN',train,ids,matrix[:,0]),('VAL',val,vids,vmat)]}
    contrast_checks={a:core.contrast_diagnostic(core.numpy_logits(exported[a],matrix.reshape(-1,259),normalizer).reshape(mask.shape),labels,mask) for a in core.ARMS}
    fixed={name:{a:states.summarize(group,{r['image'] for r in group if r['pred'] is not None and vscores[a][r['image']]<=cutoffs[a]['risk_le']}) for a in vscores} for name,group in ab.groups(val).items()}
    formal={name:states.summarize(group,{r['image'] for r in group if r['original_simple_decision']['size_accepted']}) for name,group in ab.groups(val).items()}
    centers={}
    for name,group in ab.groups(val).items():
        outputs=sum(r['pred'] is not None for r in group)
        hits=sum(simple.geometry_errors(r['gt'],r['pred'])['center_px']<15 for r in group if r['pred'] is not None)
        centers[name]=dict(frames=len(group),outputs=outputs,hits=hits,output_coverage=outputs/len(group),
            hit_rate_on_outputs=hits/outputs if outputs else None,correct_coverage_all_frames=hits/len(group))
    if (simple.fingerprint(rows)!=fingerprint or {p:sha(ROOT/p) for p in PINS}!=before
            or checked_sources()[1]!=source or counts!=dict(original=1000,contrast=1000)):
        raise ValueError('Inputs/source/order/budget changed')
    report=dict(protocol=core.VERSION,status='VAL_PASS_FROZEN_TEST_PENDING' if gate['passed'] else 'VAL_FAILED_STOP',
        contract=protocol,sources=source,input_sha256=before,train_proof=train_proof,val_proof=val_proof,
        train_statistics=tstats,VAL_statistics=vstats,VAL_cutoffs=cutoffs,VAL_fixed_points=fixed,gate=gate,
        original_policy_summary=formal,center=centers,engineering=engineering,hidden_diagnostic=diagnostics,
        TRAIN_contrast_diagnostic=contrast_checks,parameter_counts={a:sum(p.numel() for p in m.parameters()) for a,m in models.items()},
        update_counts=counts,batch_order_sha256=order_hash.hexdigest(),train_rows=2558,train_bad=71,val_rows=887,
        fixed_final_epoch=100,smoke_reused=False,new_data_roles=False,GT_online=False,TEST_read=False,
        TEST_repeatedly_exposed=True,original_policy_changed=False,boxes_scores_output_center_angle_unchanged=True,
        synthetic_samples_are_not_new_real_error_scenes=True,pca_normalizer_fit_only_original_TRAIN=True,
        elapsed_seconds=time.monotonic()-started,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip())
    write(args.out/'report.json',report)
    (args.out/'TEST_NOT_RUN.txt').write_text('No TEST selection. Failed candidate stops; pass requires a frozen TEST reporting entry.\n')
    write(args.out/'completion.json',dict(protocol=core.VERSION,status=report['status'],TEST_read=False,
        artifacts={p.name:sha(p) for p in args.out.iterdir() if p.is_file()}))
    print(report['status'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=0); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if Path.cwd().resolve()!=ROOT: raise SystemExit('Run in original project root')
    existed=args.out.exists()
    try: run(args)
    except Exception as error:
        if not existed and args.out.exists() and not (args.out/'failure.json').exists():
            write(args.out/'failure.json',dict(error=type(error).__name__+': '+str(error)))
        raise
