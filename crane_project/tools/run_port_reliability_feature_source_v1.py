#!/usr/bin/env python3
"""One fixed TRAIN->VAL source comparison; only native arm may advance."""
import argparse
from collections import Counter
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
from crane_project.utils import port_reliability_feature_source_v1 as core
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_feature_ablation_v1 as ab
from crane_project.utils import port_reliability_state_continuity_v1 as states
from crane_project.tools import run_port_reliability_readout_compare_v1 as parent
prior = parent.prior
sha,write = prior.sha,prior.write
PROTOCOL = ROOT/'crane_project/tools/port_reliability_feature_source_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_reliability_feature_source_v1_sources.json'
NEW_FILES = [
    'crane_project/utils/port_reliability_feature_source_v1.py',
    'crane_project/utils/port_reliability_feature_source_v1_torch.py',
    'crane_project/tools/run_port_reliability_feature_source_v1.py',
    'crane_project/tools/review_port_reliability_feature_source_v1.py',
    'crane_project/tools/port_reliability_feature_source_v1_protocol.json',
    'tests/test_port_reliability_feature_source_v1.py',
    'tools/run_port_reliability_feature_source_v1.sh']
SOURCE_FILES = list(dict.fromkeys(parent.SOURCE_FILES+[
    'crane_project/tools/port_reliability_readout_compare_v1_sources.json',
    'mmrotate/models/dense_heads/sym_eood_head.py',
    'mmrotate/models/dense_heads/rotated_retina_head.py',
    'mmrotate/models/detectors/sym_eood_detector.py',
    'crane_project/tools/train_port_geometry_midpoint_formal_v1.py',
    'crane_project/tools/eval_port_geometry_midpoint_v1_test.py',
    'crane_project/tools/diagnose_port_shape_e_h_train_gradients_v1.py',
    'crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py',
    'crane_project/configs/crane_symeood_k1_port_day2night_v1.py',
    'crane_project/configs/crane_symeood_k1.py',
    'configs/_base_/schedules/schedule_1x.py',
    'configs/_base_/default_runtime.py']+NEW_FILES))
B_PATH = 'work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth'
PINS = dict(parent.PINS,**{B_PATH:'8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23'})


def checked_sources():
    parent.checked_sources()
    protocol = json.loads(PROTOCOL.read_text()); actual = {p:sha(ROOT/p) for p in SOURCE_FILES}
    if (json.loads(SOURCES.read_text()) != dict(protocol=core.VERSION,sources=actual)
            or protocol['settings'] != core.SETTINGS or protocol['input_pins'] != PINS
            or protocol['protocol'] != core.VERSION): raise ValueError('Source/contract differs')
    return protocol,dict(manifest_sha256=sha(SOURCES),sources=actual)


def collect(rows, role, gpu, out):
    import torch
    from mmcv.parallel import collate,scatter
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as frozen
    from crane_project.utils import port_reliability_feature_source_v1_torch as native
    if role not in ('TRAIN','VAL'): raise ValueError('No unapproved data role')
    cfg = frozen.g.check_cfg(); sources,data_identity = frozen.checked_data()
    source = {r['image']:r for r in sources}; lookup = {r['image']:r for r in rows}
    detector = frozen.build_detector(cfg,gpu); before = frozen.g.state_digest(detector)
    cache = json.loads((ROOT/prior.CACHE/'cache_manifest.json').read_text())
    if before != cache['detector_state']: raise ValueError('Frozen B state differs from source cache')
    dataset = frozen.dataset_for(cfg,role.lower(),1.)
    names = [Path(info['filename']).stem for info in frozen.infos(dataset)]
    if len(names) != len(lookup) or set(names) != set(lookup): raise ValueError('Full dataset membership differs')
    captured = {}; maximum = np.zeros(6); traces = []
    for index,name in enumerate(names):
        row = lookup[name]; value = scatter(collate([dataset[index]],samples_per_gpu=1),[gpu])[0]
        image,metas = value['img'][0],value['img_metas'][0]
        if (len(value['img']) != 1 or image.shape != (1,3,1024,1024) or len(metas) != 1
                or Path(metas[0]['filename']).stem != name or metas[0].get('flip',False)
                or list(metas[0]['ori_shape'][:2][::-1]) != row['image_size']):
            raise ValueError('Deterministic native view differs: '+name)
        frozen.g.checked_meta(metas[0]); frozen.g.assert_detector_frozen(detector)
        prediction,feature,trace = native.trace_native(detector,image,metas)
        b = frozen.g.flatten_prediction(prediction)
        if len(b) != int(row['pred'] is not None) or (feature is None) != (not len(b)):
            raise ValueError('Native B/M output/missing identity differs: '+name)
        if len(b):
            stored = np.asarray(row['b_original']); delta = np.abs(b[0]-stored)
            delta[4] = abs((b[0,4]-stored[4]+np.pi/2)%np.pi-np.pi/2)
            maximum = np.maximum(maximum,delta)
            if (not np.allclose(b[0,:4],stored[:4],atol=5e-4,rtol=2e-5)
                    or delta[4]>2e-6 or b[0,5] != stored[5]):
                raise ValueError('Native original B replay differs: '+name)
            captured[name] = feature
            trace['patch_sha256'] = hashlib.sha256(feature.astype('<f4').tobytes()).hexdigest()
        traces.append(dict(image=name,role=role,image_sha256=source[name]['image_sha256'],**trace))
        if index%50 == 0 or index==len(names)-1: print('Native source',role,index+1,'/',len(names),flush=True)
        del value,image,metas,prediction,b
    if frozen.g.state_digest(detector) != before: raise ValueError('Detector state changed')
    ids = sorted(captured); raw = np.stack([captured[i] for i in ids])
    np.savez_compressed(out/(role.lower()+'_native_raw.npz'),features=raw,images=np.asarray(ids))
    with gzip.open(out/(role.lower()+'_native_trace.jsonl.gz'),'xt') as stream:
        for r in sorted(traces,key=lambda r:r['image']): stream.write(json.dumps(r,allow_nan=False)+'\n')
    proof = dict(frames=len(rows),outputs=len(ids),data_identity=data_identity,detector_state=before,
        maximum_original_B_replay_difference=maximum.tolist(),actual_native_topk=True,
        new_detector_inferences=len(rows),detector_updates=0,GT_online=False)
    del detector,dataset; torch.cuda.empty_cache()
    return raw,ids,proof


def score(rows, matrices, ids, models, normalizers):
    scores = {a:{r['image']:None for r in rows} for a in (*core.ARMS,'full_simple','score_only')}
    for arm in core.ARMS:
        risks = simple.sigmoid(core.numpy_logits(models[arm],matrices[arm],normalizers[arm]))
        for image,v in zip(ids,risks): scores[arm][image] = float(v)
    for r in rows:
        if r['pred'] is not None:
            scores['full_simple'][r['image']] = r['size_risks']['full_simple']
            scores['score_only'][r['image']] = 1-r['pred'][5]
    return scores


def save_scored(path,rows,scores,cutoffs):
    with gzip.open(path,'xt') as stream:
        for r in rows:
            decisions = {a:core.decide(r['original_simple_decision'],scores[a][r['image']],cutoffs[a]['risk_le']) for a in core.ARMS}
            stream.write(json.dumps(dict(r,experiment_risks={a:s[r['image']] for a,s in scores.items()},candidate_decisions=decisions),allow_nan=False)+'\n')


def run(args):
    import torch
    from crane_project.utils import port_reliability_feature_source_v1_torch as native
    protocol,source = checked_sources()
    if args.out.resolve().parent.parent != (ROOT/'work_dirs'/core.VERSION).resolve(): raise ValueError('Use unified work_dirs experiment/RUN_ID/result')
    if args.out.exists(): raise FileExistsError('Refuse overwrite')
    before = {p:sha(ROOT/p) for p in PINS}
    if before != PINS: raise ValueError('Frozen input SHA differs')
    rows,policy = prior.load_rows(); fingerprint = simple.fingerprint(rows)
    args.out.mkdir(parents=True); started = time.monotonic()
    torch.set_num_threads(1); torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = 'cuda:'+str(args.gpu)
    train,stored,ids = parent.checked_feature_rows('TRAIN',[r for r in rows if r['reliability_role']=='train'])
    raw,tids,train_proof = collect(train,'TRAIN',args.gpu,args.out)
    if tids != ids: raise ValueError('Paired TRAIN sources differ')
    raw_sources = dict(midpoint=stored[:,3:],native=raw)
    pcas = {}; normalizers = {}; matrices = {}
    for arm in core.ARMS:
        print('TRAIN-only PCA',arm,raw_sources[arm].shape,flush=True)
        pcas[arm] = core.fit_pca(raw_sources[arm],'TRAIN')
        matrices[arm] = core.features(stored[:,:3],raw_sources[arm],pcas[arm])
        normalizers[arm] = core.normalization(matrices[arm])
        np.savez_compressed(args.out/('train_'+arm+'_projected.npz'),features=matrices[arm],images=np.asarray(ids))
    write(args.out/'pca.json',pcas)
    by_id = {r['image']:r for r in train}
    labels = np.asarray([ab.size_bad(by_id[i]) for i in ids],dtype=np.float32)
    if len(labels) != 2558 or int(labels.sum()) != 71: raise ValueError('Full TRAIN size targets differ')
    x = {a:torch.tensor(core.normalize(matrices[a],normalizers[a]),dtype=torch.float32,device=device) for a in core.ARMS}
    z,y,w = [torch.tensor(v,dtype=torch.float32,device=device) for v in (stored[:,0],labels,core.class_weights(labels))]
    idx = torch.tensor(np.r_[np.flatnonzero(labels==0)[:4],np.flatnonzero(labels==1)[:4]],dtype=torch.long,device=device)
    smoke = native.make_models(device); checks = {}
    for arm,model in smoke.items():
        if not torch.equal(model(x[arm],z),-z): raise ValueError('Neutral risk not exactly score')
        opt = native.optimizer(model)
        updates = [native.update(model,opt,x[arm][idx],z[idx],y[idx],w[idx]) for _ in range(2)]
        if not any(v>0 for k,v in updates[1]['gradient_by_parameter'].items() if k.startswith('network.0')): raise ValueError('Hidden gradient disconnected')
        checks[arm] = dict(neutral_identity=True,discarded_updates=updates)
    del smoke
    write(args.out/'smoke_report.json',checks)
    models = native.make_models(device); initial = {a:native.exported(m) for a,m in models.items()}
    if initial['native'] != initial['midpoint']: raise ValueError('Initial weights differ')
    write(args.out/'initial_heads.json',initial)
    optimizers = {a:native.optimizer(m) for a,m in models.items()}; counts = {a:0 for a in core.ARMS}
    order_hash = hashlib.sha256(); rng = np.random.RandomState(1701)
    with (args.out/'train_log.jsonl').open('x') as stream:
        for epoch in range(1,101):
            order = rng.permutation(len(ids)); order_hash.update(order.astype('<i8').tobytes())
            for slot,offset in enumerate(range(0,len(ids),256)):
                indices = torch.tensor(order[offset:offset+256],dtype=torch.long,device=device)
                for arm,model in models.items():
                    value = native.update(model,optimizers[arm],x[arm][indices],z[indices],y[indices],w[indices]); counts[arm] += 1
                    stream.write(json.dumps(dict(epoch=epoch,slot=slot,arm=arm,batch_count=len(indices),**value),allow_nan=False)+'\n')
            if epoch%10 == 0: print('Source TRAIN',epoch,'/100',counts,flush=True)
    exported = {a:native.exported(m) for a,m in models.items()}
    for arm,model in models.items():
        replay = simple.sigmoid(core.numpy_logits(exported[arm],matrices[arm],normalizers[arm]))
        error = float(np.max(np.abs(torch.sigmoid(model(x[arm],z)).detach().cpu().numpy()-replay)))
        if error>1e-4: raise ValueError('Torch/NumPy final risk replay differs')
        checks[arm]['final_risk_replay_difference'] = error
    write(args.out/'models.json',dict(protocol=core.VERSION,models=exported,normalizers=normalizers,epoch=100,update_counts=counts))
    torch.save(dict(protocol=core.VERSION,models={a:m.state_dict() for a,m in models.items()},normalizers=normalizers,epoch=100),str(args.out/'final_heads.pth'))
    loaded = prior.load_trusted(args.out/'final_heads.pth')
    if any(not torch.equal(v.cpu(),loaded['models'][a][k].cpu()) for a,m in models.items() for k,v in m.state_dict().items()): raise ValueError('Checkpoint reload differs')
    # Only final frozen weights are evaluated; no TRAIN performance gate or probe.
    val,vstored,vids = parent.checked_feature_rows('VAL',[r for r in rows if r['reliability_role']=='val'])
    vraw,nids,val_proof = collect(val,'VAL',args.gpu,args.out)
    if nids != vids: raise ValueError('Paired VAL sources differ')
    vmatrices = {a:core.features(vstored[:,:3],dict(midpoint=vstored[:,3:],native=vraw)[a],pcas[a]) for a in core.ARMS}
    for arm in core.ARMS: np.savez_compressed(args.out/('val_'+arm+'_projected.npz'),features=vmatrices[arm],images=np.asarray(vids))
    scores = score(val,vmatrices,vids,exported,normalizers)
    cutoffs = {a:ab.calibrate(val,s,.95) for a,s in scores.items()}
    statistics = {a:core.describe(val,scores,a,cutoffs[a]['risk_le']) for a in core.ARMS}
    gate = core.gate(statistics['native'])
    tscores = score(train,matrices,ids,exported,normalizers)
    tstats = {a:core.describe(train,tscores,a,cutoffs[a]['risk_le']) for a in core.ARMS}
    save_scored(args.out/'scored_TRAIN.jsonl.gz',train,tscores,cutoffs)
    save_scored(args.out/'scored_VAL.jsonl.gz',val,scores,cutoffs)
    write(args.out/'cutoffs.json',cutoffs)
    previous = json.loads((ROOT/parent.SOURCE_DIR/'report.json').read_text())
    fixed = {name:{a:states.summarize(group,{r['image'] for r in group if r['pred'] is not None and scores[a][r['image']]<=cutoffs[a]['risk_le']}) for a in scores} for name,group in ab.groups(val).items()}
    if (simple.fingerprint(rows) != fingerprint or {p:sha(ROOT/p) for p in PINS} != before
            or checked_sources()[1] != source or counts != dict(midpoint=1000,native=1000)): raise ValueError('Source/inputs/update budget changed')
    report = dict(protocol=core.VERSION,status='VAL_PASS_FROZEN_TEST_PENDING' if gate['passed'] else 'VAL_FAILED_STOP',
        contract=protocol,sources=source,input_sha256=before,train_proof=train_proof,val_proof=val_proof,
        train_statistics=tstats,VAL_statistics=statistics,VAL_cutoffs=cutoffs,VAL_fixed_points=fixed,
        gate=gate,engineering=checks,parameter_counts={a:sum(p.numel() for p in m.parameters()) for a,m in models.items()},
        update_counts=counts,batch_order_sha256=order_hash.hexdigest(),train_rows=2558,train_bad=71,val_rows=887,
        original_policy_summary=previous['original_policy_summary'],center=previous['center'],
        fixed_final_epoch=100,smoke_reused=False,new_data_roles=False,GT_online=False,TEST_read=False,
        TEST_repeatedly_exposed=True,original_policy_changed=False,boxes_scores_output_center_angle_unchanged=True,
        source_scope='Native B FPN regression receptive field, supervised against delivered M size; shared FPN is not independent regression tower',
        elapsed_seconds=time.monotonic()-started,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip())
    write(args.out/'report.json',report)
    (args.out/'TEST_NOT_RUN.txt').write_text('VAL fail stops; pass requires frozen TEST entry, never TEST selection.\n')
    write(args.out/'completion.json',dict(protocol=core.VERSION,status=report['status'],TEST_read=False,artifacts={p.name:sha(p) for p in args.out.iterdir() if p.is_file()}))
    print(report['status'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=0); parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT: raise SystemExit('Run in original project root')
    existed = args.out.exists()
    try: run(args)
    except Exception as error:
        if not existed and args.out.exists() and not (args.out/'failure.json').exists(): write(args.out/'failure.json',dict(error=type(error).__name__+': '+str(error)))
        raise
