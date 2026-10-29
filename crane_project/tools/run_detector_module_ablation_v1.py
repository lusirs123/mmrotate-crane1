#!/usr/bin/env python3
"""Missing E+M, E+L, E+M+L and S+L arms, reusing sealed head mathematics.

Default check is stdlib-only and writes nothing. Runtime stages are separate,
explicit commands; there is no queue, detector training, continuation or TEST.
E is scale-augmented EOOD; S is the archived new SymEOOD-110.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONTRACT = ROOT/'crane_project/data_contracts/detector_module_ablation_v1.json'
ARMS = ('eood_m', 'eood_l', 'eood_ml', '110_l')
SHARDS = ('train_s1.pt', 'train_s05.pt', 'val_s1.pt')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def checked_contract(path=CONTRACT):
    p = read(path)
    expected_arms = {
        'eood_m': dict(frontend='eood', midpoint=True, lvsr=False),
        'eood_l': dict(frontend='eood', midpoint=False, lvsr=True),
        'eood_ml': dict(frontend='eood', midpoint=True, lvsr=True),
        '110_l': dict(frontend='110', midpoint=False, lvsr=True)}
    if p['protocol'] != 'detector_module_ablation_v1' or p['arms'] != expected_arms:
        raise ValueError('Unknown module-ablation protocol')
    for stage in ('midpoint', 'lvsr'):
        settings = p[stage]
        if (settings['epochs'] != 24 or settings['batch_size'] != 8 or
                settings['seed'] != 1703 or settings['lr'] != .001 or
                settings['weight_decay'] != 0. or settings['clip_norm'] != 10. or
                settings['optimizer'] != 'Adam' or settings['adam'] != dict(betas=[.9, .999], eps=1e-8)):
            raise ValueError('Head hyperparameters differ: '+stage)
    if (p['midpoint']['sigma_cells'] != 1.5 or p['midpoint']['parameters'] != 17696 or
            p['midpoint']['train_center_match_px'] != 15. or p['lvsr']['parameters'] != 2480 or
            p['train_scales'] != [1., .5] or p['val_scale'] != 1.):
        raise ValueError('Architecture/views differ')
    for name, digest in p['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Source identity differs: '+name)
    return p


def arm_spec(proof, arm):
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    return dict(proof['arms'][arm], name=arm)


def output_root(proof):
    return ROOT/proof['output_directory']


def cache_path(proof, frontend):
    # S+L uses the unchanged existing S cache. All E arms share one E cache.
    if frontend == '110':
        return ROOT/proof['frontends']['110']['cache']
    return output_root(proof)/'eood_cache'


def runtime_imports():
    from crane_project.tools import train_symeood_110_midpoint_v1 as midpoint
    midpoint.runtime_imports()
    return midpoint, midpoint.torch


def top1(result, np):
    """Same one-class task Top1, AFTER native threshold/NMS; stable first tie."""
    if len(result) != 1 or len(result[0]) != 1:
        raise ValueError('Expected one image / one grab class')
    a = np.asarray(result[0][0])
    if a.ndim != 2 or a.shape[1] != 6 or not np.isfinite(a).all():
        raise ValueError('Invalid native predictions')
    return [[a[int(np.argmax(a[:, 5])):int(np.argmax(a[:, 5]))+1].copy() if len(a) else a.copy()]]


class NativeFrontendAdapter:
    """Reuse midpoint capture_cache without changing either detector's source.

    Eood's original simple_test is extract_feat -> bbox_head -> get_bboxes ->
    rbbox2result. Only the already-extracted features are supplied here.
    """
    def __init__(self, detector, family, np):
        self.detector, self.family, self.np = detector, family, np

    def __getattr__(self, name):
        return getattr(self.detector, name)

    def simple_test_from_features(self, features, metas, rescale=False):
        if self.family == 'eood':
            from mmrotate.core import rbbox2result
            outs = self.detector.bbox_head(features)
            boxes = self.detector.bbox_head.get_bboxes(*outs, metas, rescale=rescale)
            result = [rbbox2result(b, labels, self.detector.bbox_head.num_classes) for b, labels in boxes]
        else:
            result = self.detector.simple_test_from_features(features, metas, rescale=rescale)
        return top1(result, self.np)


def build_frontend(proof, family, f, torch):
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmcv.runner import load_checkpoint
    from mmrotate.models import build_detector
    spec = proof['frontends'][family]
    cfg = Config.fromfile(str(ROOT/spec['config']))
    checkpoint = ROOT/spec['checkpoint']
    if sha(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('Frozen detector checkpoint differs')
    import_modules_from_strings(**cfg.custom_imports)
    model_cfg = deepcopy(cfg.model)
    model_cfg.pretrained = None
    # Prevent torchvision initialization/download; all tensors come from weight.
    if (model_cfg.backbone.get('init_cfg') or {}).get('type') == 'Pretrained':
        model_cfg.backbone.init_cfg = None
    # Preserve train_cfg as in the original builder: some heads use it at init.
    detector = build_detector(model_cfg)
    payload = load_checkpoint(detector, str(checkpoint), map_location='cpu', strict=True)
    if payload.get('meta', {}).get('epoch') != spec['epoch']:
        raise ValueError('Detector epoch metadata differs')
    # Require the existing config's actual inference/architecture fields. Work
    # directory and training-only runtime hook path are deliberately irrelevant.
    meta_text = payload.get('meta', {}).get('config')
    if not meta_text:
        raise ValueError('Missing detector config metadata')
    meta_cfg = Config.fromstring(meta_text, '.py')
    for field in ('model', 'data', 'optimizer', 'optimizer_config', 'lr_config', 'runner'):
        if cfg[field] != meta_cfg[field]:
            raise ValueError('Checkpoint/config field differs: '+field)
    f.g.freeze_detector(detector.cuda())
    adapter = NativeFrontendAdapter(detector, family, f.np)
    # The reused capture boundary expects this receipt field, not its name.
    adapter.new110_load_receipt = dict(checkpoint_sha256=spec['checkpoint_sha256'],
        strict=True, epoch=spec['epoch'], family=family)
    return cfg, adapter


def cache_identity(proof, family):
    return dict(protocol=proof['protocol'], frontend=family,
        checkpoint_sha256=proof['frontends'][family]['checkpoint_sha256'],
        config_sha256=proof['sources'][proof['frontends'][family]['config']],
        views=dict(train=proof['train_scales'], val=proof['val_scale']))


def make_cache(proof, f, torch):
    """Same original data/view/capture functions; only frontend binding varies."""
    from mmcv.parallel import collate, scatter
    directory = cache_path(proof, 'eood')
    directory.mkdir(parents=True, exist_ok=False)
    sources, data = f.checked_data()
    f.seed_all()
    cfg, detector = build_frontend(proof, 'eood', f, torch)
    before = f.g.state_digest(detector)
    identity = cache_identity(proof, 'eood')
    manifest = dict(protocol=proof['protocol'], identity=identity, data_identity=data,
        detector_state=before, files={}, record_counts={}, tensor_bytes=0,
        detector_updates=0, test_access=False, contract_sha256=sha(CONTRACT),
        runtime=f.runtime(0), feature_extractions=0, native_head_calls=0,
        native_top1='highest score AFTER native threshold/NMS; no cfg overrides')
    lookup = {r['image']: r for r in sources}
    for split, scale, label in (('train', 1., 'train_s1'), ('train', .5, 'train_s05'), ('val', 1., 'val_s1')):
        dataset = f.dataset_for(cfg, split, scale)
        names = [Path(v['filename']).stem for v in f.infos(dataset)]
        expected = {r['image'] for r in sources if (r['split'] == 'val') == (split == 'val')}
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError('Full dataset view identity differs')
        records = []
        for index, name in enumerate(names):
            source = lookup[name]
            sample = scatter(collate([dataset[index]], samples_per_gpu=1), [0])[0]
            image, metas = sample['img'][0], sample['img_metas'][0]
            if (image.shape != (1, 3, 1024, 1024) or len(metas) != 1 or
                    Path(metas[0]['filename']).stem != name or metas[0].get('flip', False) or
                    list(metas[0]['ori_shape'][:2][::-1]) != source['image_size']):
                raise ValueError('Deterministic view differs')
            tensors = f.capture_cache(detector, image, metas)
            manifest['feature_extractions'] += 1
            manifest['native_head_calls'] += 3
            gt = torch.tensor(source['gt'], dtype=torch.float32).reshape(1, 5)
            b = tensors['boxes_original']
            center = float((b[0, :2]-gt[0, :2]).norm()) if len(b) else None
            records.append(dict(source, scale=scale, role=split, gt_original=gt,
                eligible=split == 'train' and center is not None and center < 15.,
                center_error_px=center, **tensors))
            if index % 50 == 0:
                print(json.dumps(dict(stage='cache', shard=label, done=index+1, total=len(names))), flush=True)
            f.resource_check(0)
        manifest['tensor_bytes'] += f.tensor_bytes(records)
        if manifest['tensor_bytes'] > f.SETTINGS['max_cpu_tensor_bytes']:
            raise ValueError('CPU ROI tensor cache exceeds inherited limit')
        path = directory/(label+'.pt')
        f.atomic_save(path, dict(identity=identity, detector_state=before, records=records))
        manifest['files'][path.name] = sha(path)
        manifest['record_counts'][label] = len(records)
        del dataset, records
    if f.g.state_digest(detector) != before:
        raise ValueError('Frozen detector changed')
    write(directory/'cache_manifest.json', manifest)


def load_cache(proof, family, f, torch):
    directory = cache_path(proof, family)
    manifest = read(directory/'cache_manifest.json')
    if family == '110':
        if manifest['identity']['frozen_b']['checkpoint_sha256'] != proof['frontends']['110']['checkpoint_sha256']:
            raise ValueError('Wrong S cache detector identity')
    elif manifest['identity'] != cache_identity(proof, family):
        raise ValueError('Wrong E cache identity')
    elif manifest['contract_sha256'] != sha(CONTRACT):
        raise ValueError('Wrong E cache preparation contract')
    if (manifest['record_counts'] != dict(train_s1=2558, train_s05=2558, val_s1=887) or
            set(manifest['files']) != set(SHARDS) or manifest['detector_updates'] != 0 or manifest['test_access']):
        raise ValueError('Full frozen TRAIN/VAL cache required')
    shards = {}
    for name in SHARDS:
        path = directory/name
        if sha(path) != manifest['files'][name]:
            raise ValueError('Cache shard SHA differs')
        payload = torch.load(str(path), map_location='cpu')
        if payload['identity'] != manifest['identity'] or payload['detector_state'] != manifest['detector_state']:
            raise ValueError('Cache shard parent differs')
        if len(payload['records']) != manifest['record_counts'][Path(name).stem]:
            raise ValueError('Cache shard record count differs')
        shards[Path(name).stem] = payload['records']
    train, val = shards['train_s1']+shards['train_s05'], shards['val_s1']
    f.validate_records(train, val)
    if f.tensor_bytes(train+val) != manifest['tensor_bytes']:
        raise ValueError('Cache byte count differs')
    # Same annotations across E/S; don't require identical predictions or masks.
    sources, actual = f.checked_data()
    if actual != manifest['data_identity']:
        raise ValueError('Cached data identity differs from frozen TRAIN/VAL')
    by_name = {r['image']: r for r in sources}
    for row in train+val:
        source = by_name[row['image']]
        if any(row[k] != source[k] for k in ('image_sha256', 'annotation_sha256', 'gt', 'image_size')):
            raise ValueError('Cached source identity differs')
        if row['gt_original'][0].tolist() != torch.tensor(source['gt'], dtype=torch.float32).tolist():
            raise ValueError('Cached GT tensor differs')
    return train, val, dict(cache_manifest_sha256=sha(directory/'cache_manifest.json'),
        detector_sha256=proof['frontends'][family]['checkpoint_sha256'],
        detector_state=manifest['detector_state'], data_identity=actual, shards=manifest['files'],
        cache_runtime=manifest['runtime'])


def attach_reference(records, midpoint, device, torch):
    """L-only uses Q=B. No zero/random/trained midpoint is silently executed."""
    result = []
    with torch.no_grad():
        for source in records:
            row = dict(source)
            if midpoint is None:
                reference = row['boxes_original'].detach().cpu().clone()
            else:
                x = [row[k].to(device) for k in ('roi', 'support', 'boxes_original', 'boxes_model', 'scale_xy')]
                reference = midpoint(*x)['boxes_original'].detach().cpu()
            if len(reference) != len(row['boxes_original']) or not torch.equal(reference[:, 5], row['boxes_original'][:, 5]):
                raise ValueError('Reference changed output count/score')
            row['midpoint_original'] = reference
            row['shard'] = 'val_s1' if row['role'] == 'val' else 'train_s1' if row['scale'] == 1. else 'train_s05'
            result.append(row)
    return sorted(result, key=lambda r: (r['scale'], r['image']))


def save_head(path, head, optimizer, epoch, updates, identity, f, torch):
    payload = dict(protocol='detector_module_ablation_v1', identity=identity,
        epoch=epoch, updates=updates, parent_checkpoint=None,
        head_state={k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        optimizer_state=deepcopy(optimizer.state_dict()), rng=f.rng_state())
    f.atomic_save(path, payload)
    if not f.tree_equal(payload, torch.load(str(path), map_location='cpu')):
        raise ValueError('Head/Adam/RNG serialization differs')
    return dict(path=path.name, sha256=sha(path), epoch=epoch, updates=updates)


def midpoint_train(proof, f, torch):
    train, val, identity = load_cache(proof, 'eood', f, torch)
    dest = output_root(proof)/'eood_m'
    dest.mkdir(parents=True, exist_ok=False)
    identity.update(frontend='eood', module='midpoint', contract_sha256=sha(CONTRACT))
    f.seed_all()
    h = f.new_head().cuda()
    opt = f.optimizer_for(h)
    initial = f.g.state_digest(h)
    neutral = f.evaluate(h, val, 'cuda:0')
    if any(r['b'] != r['midpoint'] for r in neutral):
        raise ValueError('Midpoint neutral delivery differs')
    write(dest/'baseline_VAL.json', f.summaries(neutral))
    checkpoints, updates = {}, 0
    for epoch in range(1, 25):
        batches = f.epoch_batches(train, epoch)
        losses = []
        for indices in batches:
            stats = f.update(h, opt, train, indices, 'cuda:0')
            losses.append(stats['loss'])
            updates += 1
        value = f.summaries(f.evaluate(h, val, 'cuda:0'))
        cp = save_head(dest/('head_epoch_%02d.pth'%epoch), h, opt, epoch, updates, identity, f, torch)
        write(dest/('val_epoch_%02d.json'%epoch), value)
        checkpoints[cp['path']] = dict(checkpoint=cp, metrics=value['groups']['overall']['midpoint']['metric_protocol_v2'])
        write(dest/('epoch_%02d_log.json'%epoch), dict(epoch=epoch, updates=updates,
            epoch_steps=len(batches), loss_mean=sum(losses)/len(losses),
            batch_order_sha256=hashlib.sha256(json.dumps(batches, separators=(',', ':')).encode()).hexdigest()))
        print(json.dumps(dict(stage='midpoint', epoch=epoch, updates=updates)), flush=True)
    chosen, _, info = f.select_best_checkpoint(checkpoints, f.SELECTION_CONFIG)
    if chosen is None:
        raise ValueError('No VAL-selected midpoint')
    result = dict(identity=identity, selected_checkpoint=checkpoints[chosen]['checkpoint'],
        selection_info=info, selection_config=f.SELECTION_CONFIG, all_checkpoints=checkpoints,
        initial_state=initial, fixed_epoch24='head_epoch_24.pth', TEST=False)
    write(dest/'selection.json', result)
    write(dest/'completion.json', dict(epochs=24, updates=updates, detector_updates=0, TEST=False))


def lvsr_selection_key(summary, epoch):
    from crane_project.tools.train_symeood_110_lvsr_pair_v1 import selection_key
    return selection_key(summary, epoch)


def select_lvsr(events):
    # All24 epochs, joint-first; not different gates/criteria per arm. This is
    # an exploratory ablation selection, not retroactive old-gate acceptance.
    best = min(events, key=lambda e: e['selection_key'])
    return dict(selected_epoch=best['epoch'], selected_checkpoint=best['checkpoint'],
        rule='joint10_desc_sim_long_sim_short_real_long_real_short_epoch_asc',
        fixed_epoch24='head_epoch_24.pth', original_adoption_gate_passed=None,
        comparison_scope='prospectively frozen new arms; archived full arm is descriptive', TEST=False)


def load_eood_midpoint(proof, cache_identity_value, f, torch):
    folder = output_root(proof)/'eood_m'
    result = read(folder/'selection.json')
    cp = result['selected_checkpoint']
    path = folder/cp['path']
    if sha(path) != cp['sha256']:
        raise ValueError('EOOD midpoint SHA differs')
    payload = torch.load(str(path), map_location='cpu')
    expected = dict(cache_identity_value, frontend='eood', module='midpoint', contract_sha256=sha(CONTRACT))
    if payload['identity'] != expected or payload['epoch'] != cp['epoch']:
        raise ValueError('EOOD midpoint/cache identity differs')
    h = f.new_head().cuda().eval().requires_grad_(False)
    h.load_state_dict(payload['head_state'], strict=True)
    return h, dict(checkpoint=str(path.relative_to(ROOT)), sha256=cp['sha256'], epoch=cp['epoch'])


def lvsr_train(proof, arm, f, torch):
    from crane_project.tools import run_symeood_110_lvsr_v1 as base
    from crane_project.tools.run_port_size_native_train_v1 import rows_write, seed, near_joint
    from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
    from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
    spec = arm_spec(proof, arm)
    if not spec['lvsr']:
        raise ValueError('eood_m is a midpoint stage, not LVSR')
    train, val, identity = load_cache(proof, spec['frontend'], f, torch)
    midpoint, midpoint_proof = (load_eood_midpoint(proof, identity, f, torch)
        if spec['midpoint'] else (None, None))
    md = f.g.state_digest(midpoint) if midpoint is not None else None
    train = attach_reference(train, midpoint, 'cuda:0', torch)
    val = attach_reference(val, midpoint, 'cuda:0', torch)
    if midpoint is not None and f.g.state_digest(midpoint) != md:
        raise ValueError('Frozen midpoint changed')
    del midpoint
    present = [r for r in train if len(r['midpoint_original'])]
    steps = math.ceil(len(present)/8)
    if not steps:
        raise ValueError('No present TRAIN views')
    identity.update(arm=arm, midpoint=midpoint_proof, reference='M' if spec['midpoint'] else 'B',
        contract_sha256=sha(CONTRACT))
    dest = output_root(proof)/arm
    dest.mkdir(parents=True, exist_ok=False)
    seed(torch)
    h = PairSpecificSizeHead().cuda()
    opt = torch.optim.Adam(h.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    neutral = base.evaluate(h, val, torch)
    if any(r['midpoint'] != r['edge_residual'] for r in neutral):
        raise ValueError('LVSR neutral delivery differs')
    write(dest/'baseline_VAL.json', metrics.summary(neutral, {}))
    write(dest/'input_counts.json', dict(train_views=len(train), train_present=len(present),
        val_frames=len(val), val_present=sum(bool(len(r['midpoint_original'])) for r in val),
        updates_per_epoch=steps, identity=identity))
    events, updates, riou_cache = [], 0, {}
    for epoch in range(1, 25):
        order = base.schedule_indices(len(present), epoch)
        totals = {}
        for start in range(0, len(order), 8):
            rr = [present[i] for i in order[start:start+8]]
            stats = base.update(h, opt, base.batch(rr, torch), 'fixed_b', torch)
            updates += 1
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.)+v
        rows = base.evaluate(h, val, torch)
        summary = metrics.summary(rows, riou_cache)
        cp = save_head(dest/('head_epoch_%02d.pth'%epoch), h, opt, epoch, updates, identity, f, torch)
        event = dict(epoch=epoch, updates=updates, updates_per_epoch=steps, checkpoint=cp,
            metrics=summary, diagnostics=near_joint(rows), selection_key=lvsr_selection_key(summary, epoch),
            schedule_sha256=hashlib.sha256(json.dumps(order, separators=(',', ':')).encode()).hexdigest(),
            training_losses={k: v/steps for k, v in totals.items()})
        write(dest/('epoch_%02d_val.json'%epoch), event)
        rows_write(dest/('epoch_%02d_val.rows.jsonl.gz'%epoch), rows)
        events.append(event)
        print(json.dumps(dict(arm=arm, epoch=epoch, updates=updates,
            joint=summary['overall']['edge_residual']['sizes']['joint10'])), flush=True)
    write(dest/'selection.json', dict(identity=identity, **select_lvsr(events)))
    write(dest/'completion.json', dict(epochs=24, updates=updates, detector_updates=0, midpoint_updates=0, TEST=False))


def smoke_views(f):
    """Two fixed TRAIN images/domain at two scales; two VAL/domain: 12 views.

    Selection uses names/split/domain only, never GT error or TEST.
    """
    manifest = read(f.g.ready.DATA/'manifest.json')
    result = []
    for role in ('train', 'val'):
        for domain in ('real', 'sim'):
            pool = sorted([r for r in manifest['records']
                if (r['split'] in ('train', 'train_sim') if role == 'train' else r['split'] == 'val')
                and r['sequence'].startswith(domain+'_')], key=lambda r: r['id'])[:2]
            if len(pool) != 2:
                raise ValueError('Insufficient fixed smoke images')
            for scale in ((1., .5) if role == 'train' else (1.,)):
                result.extend((r, role, scale) for r in pool)
    return result


def capture_smoke(proof, family, f, torch):
    """Selected images only; no dataset construction/full data scan/cache shard."""
    from mmcv.parallel import collate, scatter
    from mmdet.datasets.pipelines import Compose
    cfg, detector = build_frontend(proof, family, f, torch)
    before = f.g.state_digest(detector)
    train, val, native_checks = [], [], 0
    for source, role, scale in smoke_views(f):
        image_path = f.g.ready.DATA/source['images']['path']
        ann_path = f.g.ready.DATA/source['annfiles']['path']
        if sha(image_path) != source['images']['sha256'] or sha(ann_path) != source['annfiles']['sha256']:
            raise ValueError('Selected smoke image/annotation differs')
        pipeline = cfg.data.val.pipeline if role == 'val' else f.g.fixed_specs(cfg, scale)[0]['pipeline']
        sample = Compose(deepcopy(pipeline))(dict(img_info=dict(filename=str(image_path)), img_prefix=None))
        data = scatter(collate([sample], samples_per_gpu=1), [0])[0]
        image, metas = data['img'][0], data['img_metas'][0]
        tensors = f.capture_cache(detector, image, metas)
        if native_checks == 0:
            with torch.no_grad():
                native = top1(detector.detector.simple_test(image, metas, rescale=True), f.np)[0][0]
            actual = tensors['boxes_original'].numpy()
            if actual.shape != native.shape or not f.np.allclose(actual, native, atol=1e-4, rtol=1e-6):
                raise ValueError('Adapter differs from original native simple_test')
            native_checks += 1
        gt_values = f.old.parse_dota_txt(str(ann_path))
        if len(gt_values) != 1:
            raise ValueError('Smoke requires original single GT')
        gt = torch.tensor(gt_values[0], dtype=torch.float32).reshape(1, 5)
        b = tensors['boxes_original']
        center = float((b[0, :2]-gt[0, :2]).norm()) if len(b) else None
        row = dict(image=source['id'], domain=source['sequence'].split('_')[0],
            sequence=source['sequence'], frame_id=int(source['id'].rsplit('_', 1)[1]),
            split=source['split'], role=role, scale=scale, gt_original=gt,
            eligible=role == 'train' and center is not None and center < 15., **tensors)
        (train if role == 'train' else val).append(row)
    if f.g.state_digest(detector) != before:
        raise ValueError('Smoke changed frozen detector')
    del detector
    torch.cuda.empty_cache()
    return train, val, dict(views=len(train)+len(val), native_forward_checks=native_checks,
        detector_updates=0, detector_state_unchanged=True)


def finite_smoke(proof, f, torch):
    """Eight discarded head updates total; no fit, cache, VAL selection or TEST."""
    from crane_project.tools import run_symeood_110_lvsr_v1 as base
    from crane_project.tools.run_port_size_native_train_v1 import seed
    from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
    started = time.monotonic()
    out = output_root(proof)/('smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True, exist_ok=False)
    report = dict(protocol=proof['protocol'], contract_sha256=sha(CONTRACT),
        runtime=f.runtime(0), scope='discarded_engineering_trial_not_performance',
        formal_training_started=False, full_cache_created=False, TEST=False, depth=False,
        temporary_optimizer_updates=0, cases={}, frontends={})
    for family in ('eood', '110'):
        train, val, receipt = capture_smoke(proof, family, f, torch)
        report['frontends'][family] = receipt
        references = [(family+'_l' if family == 'eood' else '110_l', None)]
        midpoint = None
        if family == 'eood':
            f.seed_all()
            midpoint = f.new_head().cuda()
            opt = f.optimizer_for(midpoint)
            initial = f.evaluate(midpoint, val, 'cuda:0')
            if any(r['b'] != r['midpoint'] for r in initial):
                raise ValueError('Smoke M neutral output differs')
            logs = [f.update(midpoint, opt, train, f.epoch_batches(train, epoch)[0], 'cuda:0')
                for epoch in (1, 2)]
            identity = dict(smoke=True, frontend=family, module='midpoint')
            cp = save_head(out/'temporary_midpoint.pth', midpoint, opt, 0, 2, identity, f, torch)
            clone = f.new_head().cuda()
            clone.load_state_dict(torch.load(str(out/cp['path']), map_location='cpu')['head_state'], strict=True)
            if f.evaluate(clone, val, 'cuda:0') != f.evaluate(midpoint, val, 'cuda:0'):
                raise ValueError('Smoke M save/reload forward differs')
            midpoint.eval().requires_grad_(False)
            for parameter in midpoint.parameters():
                parameter.grad = None
            references.append(('eood_ml', midpoint))
            report['cases']['eood_m'] = dict(updates=2, neutral_exact=True,
                save_reload_forward_exact=True, losses=[r['loss'] for r in logs])
            report['temporary_optimizer_updates'] += 2
            del clone, opt
        for arm, reference_head in references:
            md = f.g.state_digest(reference_head) if reference_head is not None else None
            tr = attach_reference(train, reference_head, 'cuda:0', torch)
            vr = attach_reference(val, reference_head, 'cuda:0', torch)
            seed(torch)
            h = PairSpecificSizeHead().cuda()
            opt = torch.optim.Adam(h.parameters(), lr=.001, weight_decay=0.)
            neutral = base.evaluate(h, vr, torch)
            if any(r['midpoint'] != r['edge_residual'] for r in neutral):
                raise ValueError('Smoke L neutral output differs')
            present = [r for r in tr if len(r['midpoint_original'])]
            if not present:
                raise ValueError('No present smoke TRAIN outputs')
            logs = [base.update(h, opt, base.batch(present[:8], torch), 'fixed_b', torch) for _ in range(2)]
            cp = save_head(out/('temporary_'+arm+'.pth'), h, opt, 0, 2,
                dict(smoke=True, arm=arm), f, torch)
            clone = PairSpecificSizeHead().cuda()
            clone.load_state_dict(torch.load(str(out/cp['path']), map_location='cpu')['head_state'], strict=True)
            if base.evaluate(clone, vr, torch) != base.evaluate(h, vr, torch):
                raise ValueError('Smoke L save/reload forward differs')
            if reference_head is not None and f.g.state_digest(reference_head) != md:
                raise ValueError('Smoke updated frozen M')
            report['cases'][arm] = dict(updates=2, neutral_exact=True, frozen_reference=True,
                save_reload_forward_exact=True, losses=[r['loss'] for r in logs])
            report['temporary_optimizer_updates'] += 2
            del h, clone, opt, tr, vr
        del train, val, midpoint, references
        torch.cuda.empty_cache()
    report.update(passed=True, elapsed_seconds=time.monotonic()-started,
        temporary_states_discarded=True, temporary_files='Trial-only names; never formal initialization',
        physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'])
    write(out/'completion.json', report)
    print(json.dumps(dict(passed=True, receipt=str(out/'completion.json'),
        updates=report['temporary_optimizer_updates'], formal_training_started=False)), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('check', 'smoke', 'cache', 'midpoint', 'lvsr'), default='check')
    p.add_argument('--arm', choices=ARMS, default='eood_m')
    p.add_argument('--execute', action='store_true', help='Explicit future execution; check is always read-only')
    return p


def main(args):
    proof = checked_contract()
    if args.stage == 'check':
        print(json.dumps(dict(status='LOCAL_CODE_PREPARED_RUNTIME_NOT_VALIDATED',
            arms=proof['arms'], training_started=False, cache_created=False,
            note='Existing detector batches are frozen historical controls, not new paired detector training'), ensure_ascii=False))
        return
    if not args.execute:
        raise ValueError('Runtime stages require explicit --execute; default check never runs them')
    if (os.environ.get('WORLD_SIZE', '1') != '1' or
            not os.environ.get('CUDA_VISIBLE_DEVICES') or ',' in os.environ['CUDA_VISIBLE_DEVICES']):
        raise ValueError('One explicitly selected physical GPU required')
    if args.stage == 'midpoint' and args.arm != 'eood_m':
        raise ValueError('Train E midpoint once as eood_m; eood_ml reuses its frozen selected head')
    if args.stage == 'cache' and arm_spec(proof, args.arm)['frontend'] != 'eood':
        raise ValueError('S+L reuses the sealed existing S ROI cache')
    if args.stage == 'lvsr' and not arm_spec(proof, args.arm)['lvsr']:
        raise ValueError('eood_m has no LVSR stage')
    f, torch = runtime_imports()
    torch.set_num_threads(1)
    if f.runtime(0) != proof['runtime_reference']:
        raise ValueError('Runtime differs from the archived new110 head environment')
    if args.stage == 'smoke':
        finite_smoke(proof, f, torch)
    elif args.stage == 'cache':
        make_cache(proof, f, torch)
    elif args.stage == 'midpoint':
        midpoint_train(proof, f, torch)
    else:
        lvsr_train(proof, args.arm, f, torch)


if __name__ == '__main__':
    main(parser().parse_args())
