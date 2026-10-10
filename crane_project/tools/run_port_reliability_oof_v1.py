#!/usr/bin/env python3
"""Offline auxiliary training -> genuine Real OOF -> same-simple full VAL."""
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
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_reliability_oof_v1 as core
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_complementarity_v1 as metrics
from crane_project.utils import port_reliability_within_video_rank_v1 as ranking
from crane_project.tools import run_port_reliability_box_contrast_v1 as original

DATA = ROOT/'crane_project/data/crane_grab_port_day2night_v1'
CONFIG = ROOT/'crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py'
CONTRACT = ROOT/'crane_project/tools/port_reliability_oof_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_reliability_oof_v1_sources.json'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def emit(value):
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def dump_rows(path, rows):
    with gzip.open(path, 'xt') as f:
        for r in rows: f.write(json.dumps(r, sort_keys=True, allow_nan=False)+'\n')


def checked():
    contract = json.loads(CONTRACT.read_text())
    manifest = json.loads(SOURCES.read_text())
    if contract['protocol'] != core.VERSION or contract['settings'] != core.SETTINGS:
        raise ValueError('OOF contract changed')
    for p, expected in manifest['sources'].items():
        if sha(ROOT/p) != expected: raise ValueError('Source changed: '+p)
    # Baseline bytes are retained; none of these checkpoints initialize auxiliaries.
    for p, expected in contract['input_pins'].items():
        if sha(ROOT/p) != expected: raise ValueError('Formal evidence changed: '+p)
    rows, policy = original.load_rows()
    plan = core.plan(rows)
    data = json.loads((DATA/'manifest.json').read_text())
    source = {r['id']: r for r in data['records'] if r['split'] in ('train', 'train_sim', 'val')}
    if len(source) != 3445: raise ValueError('Original membership changed')
    for r in rows:
        s = source[r['image']]
        if s['split'] != r['split'] or s['sequence'] != r['sequence']:
            raise ValueError('Source roles changed')
        for field in ('images', 'annfiles'):
            if sha(DATA/s[field]['path']) != s[field]['sha256']:
                raise ValueError('Image/GT bytes changed: '+r['image'])
    hashes = [{source[n]['images']['sha256'] for n in plan[g]['predict_images']} for g in ('A', 'B')]
    if hashes[0] & hashes[1]: raise ValueError('Cross-fold identical image leakage')
    return rows, policy, plan, source, dict(contract_sha256=sha(CONTRACT), sources_sha256=sha(SOURCES),
        data_manifest_sha256=sha(DATA/'manifest.json'), cross_fold_exact_duplicates=0,
        acquisition_event_independence_confirmed=False, TEST_opened=False)


def prepare(out):
    rows, policy, plan, source, proof = checked()
    out.mkdir(parents=True, exist_ok=False)
    from mmcv import Config
    for fold, specification in plan.items():
        folder = out/fold; folder.mkdir()
        root = folder/'data'; root.mkdir()
        for split in ('train', 'train_sim'):
            for field in ('images', 'annfiles'): (root/split/field).mkdir(parents=True)
        for name in specification['train_images']:
            record = source[name]
            for field in ('images', 'annfiles'):
                src = DATA/record[field]['path']
                (root/record['split']/field/src.name).symlink_to(src)
        cfg = Config.fromfile(str(CONFIG))
        for spec, split in zip(cfg.data.train, ('train', 'train_sim')):
            spec.data_root = str(root.resolve())+'/'
            spec.ann_file = split+'/annfiles/'
            spec.img_prefix = split+'/images/'
        # This auxiliary trainer never constructs original VAL or TEST datasets.
        cfg.data.pop('val', None); cfg.data.pop('test', None)
        cfg.data.samples_per_gpu = core.SETTINGS['detector_batch']
        cfg.data.workers_per_gpu = 2
        cfg.data.train_dataloader = dict(samples_per_gpu=core.SETTINGS['detector_batch'], workers_per_gpu=2)
        cfg.workflow = [('train', 1)]
        cfg.runner.max_epochs = core.SETTINGS['detector_epochs']
        cfg.load_from = None; cfg.resume_from = None; cfg.auto_resume = False
        cfg.checkpoint_config = dict(interval=24, max_keep_ckpts=1)
        cfg.log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
        cfg.evaluation = None
        cfg.work_dir = str((folder/'detector').resolve())
        cfg.seed = core.SETTINGS['seed']
        cfg.dump(str(folder/'detector_config.py'))
        specification = dict(specification, resolved_config_sha256=sha(folder/'detector_config.py'))
        write(folder/'membership.json', specification)
    write(out/'prepare.json', dict(protocol=core.VERSION, proof=proof, plan=plan,
        formal_frontend_frozen=True, auxiliary_warm_start='ImageNet_ResNet50_only',
        original_training=core.support([r for r in rows if r['reliability_role']=='train'])))
    return out


def check_fold(out, fold):
    rows, policy, plan, source, proof = checked()
    saved = json.loads((out/'prepare.json').read_text())
    if saved['proof'] != proof or saved['plan'] != plan: raise ValueError('Prepared source changed')
    folder = out/fold
    membership = json.loads((folder/'membership.json').read_text())
    if membership['resolved_config_sha256'] != sha(folder/'detector_config.py'):
        raise ValueError('Auxiliary configuration changed')
    found = []
    for split in ('train', 'train_sim'):
        for ann in sorted((folder/'data'/split/'annfiles').glob('*.txt')):
            name = ann.stem; record = source[name]
            image = folder/'data'/split/'images'/(name+'.jpg')
            if record['split'] != split or ann.resolve() != (DATA/record['annfiles']['path']).resolve() or image.resolve() != (DATA/record['images']['path']).resolve():
                raise ValueError('Auxiliary symlink source mismatch')
            if sha(ann) != record['annfiles']['sha256'] or sha(image) != record['images']['sha256']:
                raise ValueError('Auxiliary data mutated')
            found.append(name)
    if sorted(found) != plan[fold]['train_images'] or set(found) & set(plan[fold]['predict_images']):
        raise ValueError('Excluded sequences entered auxiliary training')
    return rows, plan[fold], folder


def model_for(cfg, checkpoint=None):
    import torch
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from mmcv.runner import load_checkpoint
    import_modules_from_strings(**cfg.custom_imports)
    if cfg.load_from is not None or cfg.resume_from is not None:
        raise ValueError('Task-trained auxiliary initialization prohibited')
    spec = deepcopy(cfg.model)
    if checkpoint:
        spec.backbone.pop('init_cfg', None)
        spec.backbone.pretrained = None
    model = build_detector(spec, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    if checkpoint:
        payload = load_checkpoint(model, str(checkpoint), map_location='cpu', strict=True)
        if payload['meta']['epoch'] != 24: raise ValueError('Final fixed auxiliary epoch required')
    else:
        if spec.backbone.init_cfg.checkpoint != 'torchvision://resnet50':
            raise ValueError('Only generic ImageNet initialization allowed')
        model.init_weights()
    return model


def preflight(out, gpu):
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import build_optimizer
    from mmdet.apis import set_random_seed
    from mmrotate.datasets import build_dataset
    from mmdet.datasets import build_dataloader
    check_fold(out, 'A')
    cfg = Config.fromfile(str(out/'A/detector_config.py'))
    set_random_seed(core.SETTINGS['seed'], deterministic=True)
    torch.cuda.set_device(gpu)
    model = model_for(cfg).cuda(gpu)
    data = build_dataset(cfg.data.train)
    actual = {Path(info['filename']).stem for info in data.data_infos} if hasattr(data, 'data_infos') else {Path(info['filename']).stem for d in data.datasets for info in d.data_infos}
    if actual != set(json.loads((out/'A/membership.json').read_text())['train_images']):
        raise ValueError('Runtime training dataset differs')
    model.train(); wrapped = MMDataParallel(model, device_ids=[gpu])
    opt = build_optimizer(wrapped, cfg.optimizer)
    loader = build_dataloader(data, samples_per_gpu=2, workers_per_gpu=2, num_gpus=1, dist=False, shuffle=True, seed=core.SETTINGS['seed'])
    torch.cuda.reset_peak_memory_stats(gpu)
    records = []; before = time.monotonic()
    for i, batch in enumerate(loader):
        if i >= core.SETTINGS['preflight_steps']: break
        torch.cuda.synchronize(); started = time.monotonic()
        opt.zero_grad(); value = wrapped.train_step(batch, opt)
        loss = value['loss']
        if not torch.isfinite(loss): raise ValueError('Nonfinite detector preflight loss')
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 10.))
        if not np.isfinite(norm): raise ValueError('Nonfinite detector gradient')
        opt.step(); torch.cuda.synchronize()
        records.append(dict(step=i+1, seconds=time.monotonic()-started, loss=float(loss), gradient_before_clip=norm,
                            logs={k:float(v) for k,v in value['log_vars'].items()}))
        emit(dict(stage='discarded_detector_preflight', **records[-1]))
    total = torch.cuda.get_device_properties(gpu).total_memory
    result = dict(passed=len(records)==8 and torch.cuda.max_memory_reserved(gpu)<.95*total,
        steps=records, discarded=True, detector_parameters=sum(p.numel() for p in model.parameters()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(gpu), peak_reserved_bytes=torch.cuda.max_memory_reserved(gpu),
        device_total_bytes=total, device=torch.cuda.get_device_name(gpu),
        elapsed=time.monotonic()-before, batch=2, input=[1024,1024], precision='FP32',
        no_VAL_or_TEST_dataloader=True)
    write(out/'resource_preflight.json', result)
    if not result['passed']: raise ValueError('Resource preflight failed; do not change batch after launch')
    return result


def train_detector(out, fold):
    check_fold(out, fold)
    if not json.loads((out/'resource_preflight.json').read_text())['passed']:
        raise ValueError('Detector training requires discarded resource preflight')
    folder = out/fold
    emit(dict(stage='auxiliary_detector_training', fold=fold, epochs=24,
              config_sha256=sha(folder/'detector_config.py')))
    subprocess.run([sys.executable, str(ROOT/'tools/train.py'), str(folder/'detector_config.py'),
        '--no-validate', '--seed', str(core.SETTINGS['seed']), '--deterministic'], cwd=str(ROOT), check=True)
    from mmcv import Config
    import torch
    final = folder/'detector/epoch_24.pth'
    payload = torch.load(str(final), map_location='cpu')
    if payload['meta']['epoch'] != 24: raise ValueError('Auxiliary training did not complete fixed budget')
    cfg = Config.fromfile(str(folder/'detector_config.py'))
    expected = core.plan(original.load_rows()[0])[fold]
    write(folder/'detector_completion.json', dict(checkpoint_sha256=sha(final), epoch=24,
        checkpoint_meta_iteration=payload['meta']['iter'], actual_training_images=expected['train_images'],
        excluded_sequences=expected['excluded_sequences'], checkpoint_selection=False,
        original_VAL_TEST_used=False, config_sha256=sha(folder/'detector_config.py')))


def cache_views(out, fold, model, gpu, allowed_rows):
    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmrotate.datasets import build_dataset
    from mmdet.datasets.dataset_wrappers import ConcatDataset
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as frozen
    cfg = Config.fromfile(str(CONFIG))
    folder = out/fold
    lookup = {r['image']:r for r in allowed_rows}
    result = []
    for scale in (1., .5):
        specs = frozen.g.fixed_specs(cfg, scale)
        for spec, split in zip(specs, ('train', 'train_sim')):
            spec['data_root'] = str((folder/'data').resolve())+'/'
        dataset = ConcatDataset([build_dataset(s) for s in specs])
        names = [Path(i['filename']).stem for i in frozen.infos(dataset)]
        if set(names) != set(lookup): raise ValueError('Auxiliary midpoint training data leaked')
        for i, name in enumerate(names):
            data = scatter(collate([dataset[i]], samples_per_gpu=1), [gpu])[0]
            image, metas = data['img'][0], data['img_metas'][0]
            tensors = frozen.capture_cache(model, image, metas)
            old = lookup[name]; gt = torch.tensor(old['gt'], dtype=torch.float32).reshape(1,5)
            error = float((tensors['boxes_original'][0,:2]-gt[0,:2]).norm()) if len(tensors['boxes_original']) else None
            result.append(dict(image=name, domain=old['domain'], sequence=old['sequence'], role='train',
                scale=scale, eligible=error is not None and error<15., gt_original=gt, **tensors))
            if i%100 == 0: emit(dict(stage='auxiliary_midpoint_cache', fold=fold, scale=scale, done=i+1, total=len(names)))
    return result


def midpoint_predict(out, fold, gpu):
    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmdet.apis import set_random_seed
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as frozen
    from crane_project.utils.port_geometry_midpoint_sigma_v1 import SigmaMidpointHead
    rows, specification, folder = check_fold(out, fold)
    detector_path = folder/'detector/epoch_24.pth'
    completion = json.loads((folder/'detector_completion.json').read_text())
    if sha(detector_path) != completion['checkpoint_sha256']: raise ValueError('Auxiliary final detector changed')
    cfg = Config.fromfile(str(folder/'detector_config.py'))
    model = model_for(cfg, detector_path).cuda(gpu).eval(); model.requires_grad_(False)
    before = frozen.g.state_digest(model)
    by_name = {r['image']:r for r in rows}
    allowed = [by_name[n] for n in specification['train_images']]
    cache = cache_views(out, fold, model, gpu, allowed)
    if frozen.g.state_digest(model) != before: raise ValueError('Frozen auxiliary detector mutated')
    pools = {d:[r for r in cache if r['domain']==d and r['eligible']] for d in ('real','sim')}
    if any(not p for p in pools.values()): raise ValueError('Auxiliary midpoint has no eligible domain support')
    set_random_seed(core.SETTINGS['midpoint_seed'], deterministic=True)
    head = SigmaMidpointHead(1.5).cuda(gpu)
    opt = frozen.optimizer_for(head)
    updates = 0
    with (folder/'midpoint_train.jsonl').open('x') as log:
        for epoch in range(1,4):
            batches = frozen.epoch_batches(cache, epoch)
            for slot, indices in enumerate(batches):
                value = frozen.update(head, opt, cache, indices, torch.device('cuda', gpu)); updates += 1
                log.write(json.dumps(dict(epoch=epoch, slot=slot, **value))+'\n')
            emit(dict(stage='auxiliary_midpoint_epoch', fold=fold, epoch=epoch, updates=updates))
    eligible_support = {d:sum(r['domain']==d and r['eligible'] for r in cache) for d in ('real','sim')}
    payload = dict(protocol=core.VERSION, fold=fold, epoch=3, updates=updates, sigma_cells=1.5,
        head_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
        detector_sha256=sha(detector_path), train_images=specification['train_images'])
    frozen.atomic_save(folder/'midpoint_epoch_03.pth', payload)
    if not frozen.tree_equal(payload, torch.load(str(folder/'midpoint_epoch_03.pth'), map_location='cpu')):
        raise ValueError('Auxiliary midpoint save/reload differs')
    head.eval(); head.requires_grad_(False); del cache, pools; torch.cuda.empty_cache()
    # Held-out images enter only frozen inference. No GT-dependent routing/selection.
    original_cfg = Config.fromfile(str(CONFIG))
    dataset = frozen.dataset_for(original_cfg, 'train', 1.)
    names = [Path(i['filename']).stem for i in frozen.infos(dataset)]
    requested = set(specification['predict_images']); outputs = []
    with torch.no_grad():
        for index, name in enumerate(names):
            if name not in requested: continue
            value = scatter(collate([dataset[index]], samples_per_gpu=1), [gpu])[0]
            tensors = frozen.capture_cache(model, value['img'][0], value['img_metas'][0])
            pred = None; base = None
            if len(tensors['boxes_original']):
                base = tensors['boxes_original'][0].tolist()
                inputs = [tensors[k].cuda(gpu) for k in ('roi','support','boxes_original','boxes_model','scale_xy')]
                pred = head(*inputs)['boxes_original'][0].cpu().tolist()
                if pred[5] != base[5]: raise ValueError('Auxiliary midpoint score changed')
            old = by_name[name]
            outputs.append({k:deepcopy(old[k]) for k in ('image','sequence','domain','split','frame_id','image_size','gt')})
            outputs[-1].update(pred=pred, auxiliary_B=base, excluded_fold=fold)
            if len(outputs)%100 == 0: emit(dict(stage='OOF_predict', fold=fold, done=len(outputs), total=len(requested)))
    if {r['image'] for r in outputs} != requested: raise ValueError('OOF prediction coverage incomplete')
    if frozen.g.state_digest(model) != before: raise ValueError('Auxiliary inference detector changed')
    dump_rows(folder/'oof_predictions.jsonl.gz', sorted(outputs, key=lambda r:r['image']))
    write(folder/'midpoint_completion.json', dict(updates=updates, epochs=3, detector_state_before=before,
        detector_state_after=frozen.g.state_digest(model), detector_sha256=sha(detector_path),
        midpoint_sha256=sha(folder/'midpoint_epoch_03.pth'), support=core.support(outputs),
        auxiliary_training_center_eligible_views=eligible_support,
        prediction_GT_routing=False, VAL_TEST_read=False))


def assess(out):
    rows, policy, plan, source, proof = checked()
    auxiliary=[]
    for fold in ('A','B'):
        with gzip.open(out/fold/'oof_predictions.jsonl.gz','rt') as f: auxiliary += [json.loads(l) for l in f]
    train = [r for r in rows if r['reliability_role']=='train']
    oof = core.materialize_training(train, auxiliary)
    source_gate = core.source_gate(oof)
    write(out/'training_material.json', dict(source_gate=source_gate, original=core.support(train), oof=core.support(oof)))
    dump_rows(out/'OOF_TRAIN.jsonl.gz', oof)
    if not source_gate['passed']:
        write(out/'completion.json', dict(status='AUXILIARY_SOURCE_FAILED_STOP', adopted=False, VAL_scored=False, TEST_read=False))
        return
    common = {r['image'] for r in oof if r['pred'] is not None} & {r['image'] for r in train if r['pred'] is not None}
    models = dict(in_sample=core.fit(train), oof=core.fit(oof),
        common_in_sample=core.fit([r for r in train if r['image'] in common]),
        common_oof=core.fit([r for r in oof if r['image'] in common]))
    for key in ('weights','mean','scale'):
        if not np.allclose(models['in_sample'][key],policy['simple_policy']['models']['size'][key],atol=1e-10,rtol=1e-10):
            raise ValueError('Same-TRAIN simple control did not reproduce frozen model')
    write(out/'quality_models.json', models)
    val = core.score([r for r in rows if r['reliability_role']=='val'], models)
    cutoffs = {name:ranking.calibrate(val,name) for name in core.METHODS}
    stats = core.statistics(val, cutoffs); gate = core.gate(stats)
    protections = {}
    for method in models:
        decisions = []
        for r in val:
            decision = ranking.decide(r['original_simple_decision'], r['risks'][method], cutoffs[method]['risk_le'])
            for key in ('final_box_original','center_accepted','angle_accepted'):
                if decision[key] != r['original_simple_decision'][key]: raise ValueError('Formal protection changed')
            decisions.append(decision)
        protections[method] = dict(box_score_count_center_angle_preserved=True)
    report = dict(protocol=core.VERSION, status='VAL_PASS_FROZEN_TEST_PENDING' if gate['passed'] else 'VAL_FAILED_STOP',
        proof=proof, models=models, cutoffs=cutoffs, statistics=stats, gate=gate,
        ranking={m:ranking.ranking_summary(val,m) for m in core.METHODS},
        protections=protections, center=metrics.center(val), common_TRAIN_outputs=len(common),
        formal_policy_sha256=sha(ROOT/original.prior.POLICY), TEST_read=False, original_policy_replaced=False)
    write(out/'report.json', report); dump_rows(out/'scored_VAL.jsonl.gz', val)
    candidate_policy=deepcopy(policy)
    candidate_policy['simple_policy']['models']['size']=deepcopy(models['oof'])
    candidate_policy['simple_policy']['cutoffs']['simple']['size']=deepcopy(cutoffs['oof'])
    candidate_policy['candidate_identity']=dict(protocol=core.VERSION,gate_passed=gate['passed'],
        automatically_deployed=False,auxiliary_models_required_online=False,
        report_sha256=sha(out/'report.json'))
    simple.validate_policy(candidate_policy['simple_policy'])
    write(out/'candidate_policy.json',candidate_policy)
    write(out/'completion.json', dict(status=report['status'], adopted=False, TEST_read=False,
        report_sha256=sha(out/'report.json')))
    lines=['# 按Real序列排除的预测来源对照', '', '**状态：'+report['status']+'**', '',
        '| 方法 | 接受数 | FA | FR | ED | CR | AUROC |','|---|---:|---:|---:|---:|---:|---:|']
    for name in core.METHODS:
        v=stats[name]['all'];c=v['actual']['states']
        lines.append('|%s|%d|%d|%d|%d|%d|%.6f|'%(name,v['actual']['accepted_outputs'],c['FA'],c['FR'],c['ED'],c['CR'],v['error_AUROC']))
    (out/'analysis.md').write_text('\n'.join(lines)+'\n')
    emit(dict(stage='complete', status=report['status'], failure_checks=len(gate['failures'])))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare','preflight','detector','midpoint','assess','all'), required=True)
    parser.add_argument('--out', required=True);parser.add_argument('--fold',choices=('A','B'));parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();out=Path(args.out).resolve()
    if ROOT/'work_dirs' not in out.parents: raise ValueError('All artifacts must remain in project work_dirs')
    try:
        if args.mode=='prepare':prepare(out)
        elif args.mode=='preflight':preflight(out,args.gpu)
        elif args.mode=='detector':train_detector(out,args.fold)
        elif args.mode=='midpoint':midpoint_predict(out,args.fold,args.gpu)
        elif args.mode=='assess':assess(out)
        else:
            prepare(out)
            subprocess.run([sys.executable,__file__,'--mode','preflight','--gpu',str(args.gpu),'--out',str(out)],check=True)
            # Preflight subprocess isolation prevents caching its models/optimizer in full training.
            for fold in ('A','B'):
                subprocess.run([sys.executable,__file__,'--mode','detector','--fold',fold,'--out',str(out)],check=True)
                subprocess.run([sys.executable,__file__,'--mode','midpoint','--fold',fold,'--gpu',str(args.gpu),'--out',str(out)],check=True)
            assess(out)
    except Exception as e:
        if out.exists() and not (out/'failure.json').exists():
            write(out/'failure.json',dict(mode=args.mode,fold=args.fold,error=str(e),traceback=traceback.format_exc(),TEST_read=False))
        raise


if __name__=='__main__':main()
