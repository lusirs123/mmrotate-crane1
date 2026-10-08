#!/usr/bin/env python3
"""Fixed temperature vs signed residual, complete TRAIN -> VAL, no TEST."""
import argparse
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
from crane_project.utils import port_reliability_readout_compare_v1 as core
from crane_project.utils import port_reliability_redc_size_v1 as old_core
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_feature_ablation_v1 as ab
from crane_project.utils import port_reliability_state_continuity_v1 as states
from crane_project.tools import run_port_reliability_redc_size_v1 as prior

SOURCE_DIR = 'work_dirs/port_reliability_redc_size_v1/20261008_redc_v1/result'
PROTOCOL = ROOT/'crane_project/tools/port_reliability_readout_compare_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_reliability_readout_compare_v1_sources.json'
FEATURE_PINS = {SOURCE_DIR+'/'+name:pin for name,pin in {
    'completion.json':'faebcd5ef0b64bbe86ebc86027eb2503aea7fa25969da06b99c895131f122880',
    'report.json':'e18466a22f053501db264a8201421c0ffdcdcb519ac5978bf287b91e3ef981f8',
    'models.json':'1e648ace3f2f350c284dc27a61f8a356f366f95a088b8569900492a71c1c4417',
    'train_features.npz':'1d4a0b5e127f7f674bc18e881a49343c0b7277ed0c679bfdb2245c19511350b9',
    'val_features.npz':'801d4c4ee44af61797ff326e495800f1a427b65d7b30c9e0f44bf9a926e19fef',
    'scored_TRAIN.jsonl.gz':'dc1c4ebeafee9ade68594e87ca4b213bf8a5a2345196673e93f297a77ea362f9',
    'scored_VAL.jsonl.gz':'4ca676dfdcc06bddff8dfcae808f947b2f9c4492de81e029e6ddaec720fc6dad'
}.items()}
PINS = dict(prior.PINS, **FEATURE_PINS)
SOURCE_FILES = list(dict.fromkeys(prior.SOURCE_FILES + [
    'crane_project/tools/port_reliability_redc_size_v1_sources.json',
    'crane_project/utils/port_reliability_readout_compare_v1.py',
    'crane_project/utils/port_reliability_readout_compare_v1_torch.py',
    'crane_project/tools/run_port_reliability_readout_compare_v1.py',
    'crane_project/tools/review_port_reliability_readout_compare_v1.py',
    'crane_project/tools/review_port_reliability_redc_size_v1.py',
    'crane_project/tools/port_reliability_readout_compare_v1_protocol.json',
    'tests/test_port_reliability_readout_compare_v1.py',
    'tools/run_port_reliability_readout_compare_v1.sh']))
sha = prior.sha
write = prior.write


def checked_sources():
    protocol = json.loads(PROTOCOL.read_text())
    actual = {name:sha(ROOT/name) for name in SOURCE_FILES}
    if (json.loads(SOURCES.read_text()) != dict(protocol=core.VERSION,sources=actual)
            or protocol['protocol'] != core.VERSION or protocol['settings'] != core.SETTINGS
            or protocol['input_pins'] != PINS):
        raise ValueError('Fixed source/design changed')
    prior.checked_sources()
    return protocol,dict(manifest_sha256=sha(SOURCES),sources=actual)


def checked_feature_rows(role, original_rows):
    """Reuse the already extracted features, with byte and per-frame pairing."""
    if role not in ('TRAIN','VAL'):
        raise ValueError('Only original TRAIN/VAL; no new role or TEST')
    source = ROOT/SOURCE_DIR
    rows = [{k:r[k] for k in prior.FIELDS} for r in
            (json.loads(s) for s in gzip.open(source/('scored_'+role+'.jsonl.gz'),'rt'))]
    expected = sorted(original_rows,key=lambda r:r['image'])
    if simple.fingerprint(rows) != simple.fingerprint(expected):
        raise ValueError('Cached delivered boxes/labels/flags differ')
    data = np.load(source/(role.lower()+'_features.npz'),allow_pickle=False)
    x, ids = data['features'], data['images'].tolist()
    if x.shape != (sum(r['pred'] is not None for r in rows),291) or not np.isfinite(x).all():
        raise ValueError('Frozen feature dimensions/values differ')
    if ids != [r['image'] for r in rows if r['pred'] is not None]:
        raise ValueError('Complete feature/image order differs')
    by_id = {r['image']:r for r in rows}
    for i,image in enumerate(ids):
        # The stored bytes are SHA-pinned; log/sqrt may differ in last bits
        # between NumPy/libm versions. Do not replace the stored descriptors.
        np.testing.assert_allclose(x[i,:3],simple.descriptor(by_id[image]['pred'],by_id[image]['image_size']),rtol=0,atol=1e-12)
    return rows,x,ids


def score(rows, features, ids, models, normalizer):
    result = {name:{r['image']:None for r in rows} for name in (*core.ARMS,'full_simple','score_only')}
    for arm in core.ARMS:
        values = simple.sigmoid(core.numpy_logits(models[arm],features,normalizer,arm))
        for image,v in zip(ids,values): result[arm][image] = float(v)
    for r in rows:
        if r['pred'] is not None:
            result['full_simple'][r['image']] = r['size_risks']['full_simple']
            result['score_only'][r['image']] = 1-r['pred'][5]
    return result


def save_scored(path, rows, scores, cutoffs):
    with gzip.open(path,'xt') as stream:
        for r in rows:
            decisions = {arm:core.decide(r['original_simple_decision'],scores[arm][r['image']],cutoffs[arm]['risk_le'])
                         for arm in core.ARMS}
            for arm in core.ARMS:
                d = deepcopy(r['original_simple_decision'])
                if r['pred'] is not None:
                    d['risks']['size'] = scores[arm][r['image']]
                    d['size_accepted'] = d['risks']['size'] <= cutoffs[arm]['risk_le']
                if decisions[arm] != d: raise ValueError('Non-size field changed')
            stream.write(json.dumps(dict(r,experiment_risks={k:v[r['image']] for k,v in scores.items()},
                candidate_decisions=decisions),allow_nan=False)+'\n')


def run(args):
    import torch
    from crane_project.utils import port_reliability_readout_compare_v1_torch as native
    protocol,source = checked_sources()
    if args.out.resolve().parent.parent != (ROOT/'work_dirs'/core.VERSION).resolve():
        raise ValueError('Use work_dirs/'+core.VERSION+'/RUN_ID/result')
    if args.out.exists(): raise FileExistsError('Refuse overwrite')
    before = {name:sha(ROOT/name) for name in PINS}
    if before != PINS: raise ValueError('Frozen input or feature evidence changed')
    receipt = json.loads((ROOT/SOURCE_DIR/'completion.json').read_text())
    for name,pin in receipt['artifacts'].items():
        if sha(ROOT/SOURCE_DIR/name) != pin: raise ValueError('Prior artifact SHA differs')
    previous = json.loads((ROOT/SOURCE_DIR/'report.json').read_text())
    if previous['TEST_read'] or not previous['boxes_scores_output_center_angle_unchanged']:
        raise ValueError('Prior feature identity invalid')
    old_models = json.loads((ROOT/SOURCE_DIR/'models.json').read_text())
    rows,policy = prior.load_rows(); fingerprint = simple.fingerprint(rows)
    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = 'cpu' if args.gpu < 0 else 'cuda:'+str(args.gpu)
    train,tx,tids = checked_feature_rows('TRAIN',[r for r in rows if r['reliability_role']=='train'])
    by_id = {r['image']:r for r in train}
    labels = np.asarray([ab.size_bad(by_id[i]) for i in tids],dtype=np.float32)
    if len(labels) != 2558 or int(labels.sum()) != 71: raise ValueError('TRAIN membership changed')
    normalizer = core.normalization(tx)
    if simple.fingerprint(normalizer) != simple.fingerprint(old_models['normalizer']):
        raise ValueError('Identical TRAIN standardization not reproduced')
    x = torch.tensor(core.normalize(tx,normalizer),dtype=torch.float32,device=device)
    z = torch.tensor(tx[:,0],dtype=torch.float32,device=device)
    y = torch.tensor(labels,device=device)
    w = torch.tensor(core.class_weights(labels),dtype=torch.float32,device=device)
    idx = torch.tensor(np.r_[np.flatnonzero(labels==0)[:4],np.flatnonzero(labels==1)[:4]],dtype=torch.long,device=device)
    smoke = native.make_models(normalizer,device); checks = {}
    for arm,model in smoke.items():
        if not torch.allclose(model(x,z),-z,rtol=0,atol=1e-5): raise ValueError('Neutral score identity failed')
        opt = native.optimizer(model)
        updates = [native.update(model,opt,x[idx],z[idx],y[idx],w[idx]) for _ in range(2)]
        if not any(v>0 for k,v in updates[1]['gradient_by_parameter'].items() if k.startswith('network.0')):
            raise ValueError('Feature-conditioned gradient missing after neutral step')
        reference = core.numpy_logits(native.exported(model),tx,normalizer,arm)
        delta = float(np.max(np.abs(simple.sigmoid(model(x,z).detach().cpu().numpy())-simple.sigmoid(reference))))
        if delta > 1e-4: raise ValueError('Smoke Torch/NumPy replay differs')
        checks[arm] = dict(neutral_score_identity=True,discarded_updates=updates,maximum_risk_replay_difference=delta)
    write(args.out/'smoke_report.json',checks); del smoke
    models = native.make_models(normalizer,device)
    initial = {arm:native.exported(m) for arm,m in models.items()}
    write(args.out/'initial_heads.json',initial)
    optimizers = {arm:native.optimizer(m) for arm,m in models.items()}
    counts = {arm:0 for arm in core.ARMS}; rng = np.random.RandomState(1701)
    order_digest = hashlib.sha256(); start = time.monotonic()
    with (args.out/'train_log.jsonl').open('x') as stream:
        for epoch in range(1,101):
            order = rng.permutation(len(tx)); order_digest.update(order.astype('<i8').tobytes())
            for slot,offset in enumerate(range(0,len(tx),256)):
                indices = torch.tensor(order[offset:offset+256],dtype=torch.long,device=device)
                for arm,model in models.items():
                    value = native.update(model,optimizers[arm],x[indices],z[indices],y[indices],w[indices])
                    counts[arm] += 1
                    stream.write(json.dumps(dict(epoch=epoch,slot=slot,arm=arm,batch_count=len(indices),**value),allow_nan=False)+'\n')
            if epoch%10 == 0: print('Readout TRAIN',epoch,'/100',counts,flush=True)
    exported = {arm:native.exported(m) for arm,m in models.items()}
    replication = max(float(np.max(np.abs(np.asarray(v)-np.asarray(old_models['models']['redc'][k]))))
                      for k,v in exported['temperature'].items())
    if replication > 1e-6 or order_digest.hexdigest() != previous['batch_order_sha256']:
        raise ValueError('Temperature control did not reproduce the fixed prior training')
    for arm,model in models.items():
        delta = float(np.max(np.abs(simple.sigmoid(model(x,z).detach().cpu().numpy())-
            simple.sigmoid(core.numpy_logits(exported[arm],tx,normalizer,arm)))))
        if delta > 1e-4: raise ValueError('Final Torch/NumPy readout differs')
        checks[arm]['final_maximum_risk_replay_difference'] = delta
    write(args.out/'models.json',dict(protocol=core.VERSION,settings=core.SETTINGS,normalizer=normalizer,
        models=exported,epoch=100,update_counts=counts,probability_claim=False))
    torch.save(dict(protocol=core.VERSION,models={k:v.state_dict() for k,v in models.items()},
                    normalizer=normalizer,epoch=100),str(args.out/'final_heads.pth'))
    reloaded = prior.load_trusted(args.out/'final_heads.pth')
    if any(not torch.equal(v.detach().cpu(),reloaded['models'][arm][k].cpu())
           for arm,m in models.items() for k,v in m.state_dict().items()): raise ValueError('Checkpoint reload differs')
    # Final TRAIN weights are fixed before opening the VAL feature array.
    val,vx,vids = checked_feature_rows('VAL',[r for r in rows if r['reliability_role']=='val'])
    scores = score(val,vx,vids,exported,normalizer)
    cutoffs = {arm:ab.calibrate(val,values,.95) for arm,values in scores.items()}
    statistics = {arm:core.describe(val,scores,arm,cutoffs[arm]['risk_le']) for arm in core.ARMS}
    gate = core.gate(statistics['residual'])
    train_scores = score(train,tx,tids,exported,normalizer)
    train_statistics = {arm:core.describe(train,train_scores,arm,cutoffs[arm]['risk_le']) for arm in core.ARMS}
    diagnostics = {}
    for role,part,features,ids,risk in (('TRAIN',train,tx,tids,train_scores),('VAL',val,vx,vids,scores)):
        diagnostics[role] = {arm:dict(hidden=core.hidden_diagnostic(part,ids,features,exported[arm],normalizer),
            cross_sign={name:core.cross_sign_diagnostic(group,risk,arm) for name,group in ab.groups(part).items()})
            for arm in core.ARMS}
        # The mathematical sign barrier must also hold in the serialized model.
        if any(v['correctly_ordered'] or v['ties'] for v in diagnostics[role]['temperature']['cross_sign'].values()):
            raise ValueError('Temperature sign-barrier invariant violated')
    save_scored(args.out/'scored_TRAIN.jsonl.gz',train,train_scores,cutoffs)
    save_scored(args.out/'scored_VAL.jsonl.gz',val,scores,cutoffs)
    write(args.out/'cutoffs.json',cutoffs)
    fixed = {name:{arm:states.summarize(group,{r['image'] for r in group if r['pred'] is not None
        and scores[arm][r['image']]<=cutoffs[arm]['risk_le']}) for arm in scores} for name,group in ab.groups(val).items()}
    if (simple.fingerprint(rows) != fingerprint or {name:sha(ROOT/name) for name in PINS} != before
            or checked_sources()[1] != source or counts != dict(temperature=1000,residual=1000)):
        raise ValueError('Frozen source/input/update budget changed')
    report = dict(protocol=core.VERSION,status='VAL_PASS_FROZEN_TEST_PENDING' if gate['passed'] else 'VAL_FAILED_STOP',
        sources=source,contract=protocol,input_sha256=before,feature_source=SOURCE_DIR,
        train_feature_proof=previous['train_feature_proof'],val_feature_proof=previous['val_feature_proof'],
        train_statistics=train_statistics,VAL_statistics=statistics,VAL_cutoffs=cutoffs,VAL_fixed_points=fixed,
        original_policy_summary=previous['original_policy_summary'],center=previous['center'],gate=gate,
        engineering=checks,diagnostics=diagnostics,temperature_replication_max_parameter_difference=replication,
        update_counts=counts,batch_order_sha256=order_digest.hexdigest(),train_bad=71,train_rows=2558,val_rows=887,
        parameter_counts={arm:sum(p.numel() for p in m.parameters()) for arm,m in models.items()},
        elapsed_seconds=time.monotonic()-start,GT_online=False,TEST_read=False,original_policy_changed=False,
        boxes_scores_output_center_angle_unchanged=True,VAL_used_for_thresholds_and_development=True,
        TEST_repeatedly_exposed=True,fixed_final_epoch=100,smoke_reused=False,new_data_roles=False,
        new_detector_inferences=0,new_geometry_feature_forwards=0,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip())
    write(args.out/'report.json',report)
    if gate['passed']:
        write(args.out/'selected_policy.json',dict(protocol=core.VERSION,model=exported['residual'],normalizer=normalizer,
            cutoff=cutoffs['residual'],original_policy=policy,arm='residual'))
    (args.out/'TEST_NOT_RUN.txt').write_text('Finite TRAIN/VAL comparison only. No TEST access or extra epoch/selection.\n')
    write(args.out/'completion.json',dict(protocol=core.VERSION,status=report['status'],TEST_read=False,
        selected_arm=gate['selected_arm'],artifacts={p.name:sha(p) for p in args.out.iterdir() if p.is_file()}))
    print(report['status'],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=0); parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    try: run(args)
    except Exception as e:
        if args.out.exists() and not (args.out/'failure.json').exists():
            write(args.out/'failure.json',dict(error=repr(e),status='ENGINEERING_FAILED_STOP'))
        raise


if __name__ == '__main__': main()
