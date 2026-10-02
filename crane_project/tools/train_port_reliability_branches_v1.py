#!/usr/bin/env python3
"""Fixed-budget paired side-branch TRAIN runner. Never trains the B detector.

check is CPU/data/contract only. smoke is four persistent updates plus reload,
discarded and ineligible for VAL. train is eight complete paired TRAIN epochs.
Resume preserves old artifacts by requiring a NEW --work-dir.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import check_port_reliability_readiness_v1 as ready
from crane_project.tools import check_port_reliability_train_support_v1 as support
from crane_project.tools import preflight_port_structure_reliability_v1 as prior
from crane_project.utils.port_reliability_train_policy_v1 import (
    POLICY, direction_qualification, train_quality_targets_original)
from crane_project.utils.port_structure_reliability_v1 import (
    SETTINGS, assert_detector_frozen, canonical_boxes, component_probes_original,
    freeze_detector, map_boxes, map_points, response_targets)
from crane_project.utils import port_reliability_branches_v1 as branch

PROTOCOL = ROOT/'crane_project/tools/port_reliability_branches_v1_protocol.json'
MANIFEST = ROOT/'crane_project/tools/port_reliability_branches_v1_sources.json'
SNAPSHOT_SHA = '27643a9e00eaef5f26221dd8c8d6b52ede991ad372f369d645b8d1a17b131af0'
VAL_REPORT_SHA = '474434c3d8ca9967939dfdecc0e0c16aabdc985bd8bd0bcdc13f8c43d6e568f4'


def checked_sources():
    old = support.checked_sources()
    manifest = json.loads(MANIFEST.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    current = {p: branch.sha(ROOT/p) for p in manifest['sources']}
    if (manifest['protocol'] != branch.VERSION or manifest['sources'] != current
            or manifest['protocol_sha256'] != branch.sha(PROTOCOL)
            or protocol['epochs'] != branch.EPOCHS or protocol['seed'] != branch.SEED
            or protocol['steps_per_arm'] != 2558*branch.EPOCHS
            or protocol['arms'] != list(branch.ARMS)
            or protocol['coverages'] != list(branch.COVERAGES)):
        raise ValueError('Reviewed formal source/protocol contract differs')
    return dict(manifest_sha256=branch.sha(MANIFEST), sources=current,
                previous_47_contract=old), protocol


def checked_inputs(snapshot_path, cache, structure_report):
    if branch.sha(snapshot_path) != SNAPSHOT_SHA:
        raise ValueError('Requires the exact returned 2558-frame numeric snapshot')
    snapshot = json.loads(snapshot_path.read_text())
    inputs, identity = snapshot['inputs'], snapshot['identity']
    if support.fingerprint(inputs) != identity['input_manifest_sha256']:
        raise ValueError('Numeric input manifest differs')
    if identity['source_contract'] != support.checked_sources():
        raise ValueError('Numeric snapshot uses another source contract')
    predictions, complete = support.read_cache(cache, identity, inputs)
    prerequisite = prior.prerequisites(structure_report)
    if identity['train_sources'] != prerequisite['source_identities']:
        raise ValueError('Snapshot TRAIN source byte identities differ')
    current = support.train_inputs()
    if len(current) != 2558 or len(inputs) != 2558:
        raise ValueError('Expected 2558 TRAIN frames')
    max_corner = 0.
    for actual, saved in zip(current, inputs):
        for key in ('image', 'sequence', 'domain', 'split', 'image_size', 'image_sha256',
                    'annotation_sha256', 'axis_json_sha256', 'dota_difficulty'):
            if actual.get(key) != saved.get(key):
                raise ValueError('Current TRAIN bytes/role differ: '+saved['image'])
        for key in ('train_angle_eligible', 'evaluation_angle_eligible', 'direction_gt_source', 'structure_source'):
            if actual['direction_qualification'][key] != saved['direction_qualification'][key]:
                raise ValueError('Current TRAIN direction policy differs')
        delta = ready.corner_difference(ready.box_polygon(actual['gt_original']),
                                        ready.box_polygon(saved['gt_original']))
        if delta > .001:
            raise ValueError('Current/saved numerical GT differs materially: '+saved['image'])
        max_corner = max(delta, max_corner)
    proof = dict(snapshot_sha256=SNAPSHOT_SHA, input_manifest_sha256=support.fingerprint(inputs),
        identity_sha256=support.fingerprint(identity), all_row_input_hashes_exact=True,
        current_train_source_bytes_match=True, local_vs_saved_max_corner_px=max_corner,
        train_frames=2558, qualification=support.qualification_counts(inputs),
        clean_support=support.support_report(support.support_rows(inputs, predictions)),
        b_cache_state=complete['b_state_before'],
        cache_sha256={p: branch.sha(cache/p) for p in ('identity.json', 'complete.json', 'predictions.jsonl')})
    return inputs, proof


def axis_for(source):
    if source['domain'] == 'sim':
        return ready.box_axis(source['gt_original'])
    seq = source['sequence']
    folder = 'axis_legacy_train_v1' if seq in ready.LEGACY_TRAIN else 'axis_k2p1'
    axis, _ = ready.read_axis(ready.DATA/'provenance'/folder/seq/'axis_json'/(source['image']+'.json'),
        source['image'], source['image_size'], legacy=seq in ready.LEGACY_TRAIN)
    return axis


def build_runtime(cfg, checkpoint, gpu):
    import cv2
    import mmcv
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    if not torch.cuda.is_available():
        raise RuntimeError('GPU modes require the server CUDA environment')
    if branch.sha(checkpoint) != ready.FROZEN_B['checkpoint_sha256']:
        raise ValueError('Frozen B checkpoint SHA differs')
    import_modules_from_strings(**cfg.custom_imports)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    branch.seed_all(branch.SEED)
    detector = build_detector(deepcopy(cfg.model))
    loaded = load_checkpoint(detector, str(checkpoint), map_location='cpu', strict=True)
    if loaded.get('meta', {}).get('epoch') != 24:
        raise ValueError('Frozen B epoch differs')
    del loaded
    freeze_detector(detector.cuda(gpu))
    runtime = dict(torch=torch.__version__, mmcv=mmcv.__version__, numpy=np.__version__,
        opencv=cv2.__version__, cuda_device=torch.cuda.get_device_name(gpu),
        visible_cuda_devices=torch.cuda.device_count(), cudnn_deterministic=True, cudnn_benchmark=False)
    return detector, runtime


def datasets_for(specs, inputs):
    from mmrotate.datasets import build_dataset
    datasets = [build_dataset(spec) for spec in specs]
    indexed = [(part, index) for part in datasets for index in range(len(part))]
    if len(indexed) != len(inputs):
        raise ValueError('Actual TRAIN loader count differs')
    for (part, index), source in zip(indexed, inputs):
        if Path(part.data_infos[index]['filename']).stem != source['image']:
            raise ValueError('Actual TRAIN loader order differs')
        gt = canonical_boxes(torch.tensor(part.get_ann_info(index)['bboxes']))
        if gt.shape != (1, 5) or not np.allclose(gt[0].numpy(), source['gt_original'], atol=1e-5, rtol=1e-6):
            raise ValueError('Actual GPU-run loader GT differs from saved numeric GT')
    return indexed


def training_view(detector, indexed, source, gpu, view_seed):
    from mmcv.parallel import collate, scatter
    branch.seed_all(view_seed)
    part, index = indexed
    batch = scatter(collate([part[index]], samples_per_gpu=1), [gpu])[0]
    image, metas = batch['img'], batch['img_metas']
    if len(metas) != 1 or Path(metas[0]['filename']).stem != source['image']:
        raise ValueError('TRAIN pipeline silently replaced/dropped a frame')
    meta = metas[0]
    if list(meta['ori_shape'][:2][::-1]) != source['image_size']:
        raise ValueError('TRAIN original image geometry differs')
    gt = image.new_tensor(source['gt_original']).reshape(1, 5)
    expected = canonical_boxes(map_boxes(gt, meta, size_mode='annotation'))
    actual = canonical_boxes(batch['gt_bboxes'][0])
    if actual.shape != (1, 5) or not torch.allclose(expected, actual, atol=1e-3, rtol=1e-5):
        raise ValueError('TRAIN GT shrink/resize/reflection mapping differs')
    axis = image.new_tensor(axis_for(source))
    qualification = direction_qualification(source['gt_original'], source['domain'],
        axis.cpu().numpy() if source['domain'] == 'real' else None,
        ready.NATIVE_K[source['sequence']] if source['domain'] == 'real' else None)
    if qualification['train_angle_eligible'] != source['direction_qualification']['train_angle_eligible']:
        raise ValueError('Native direction qualification changed')
    with torch.no_grad():
        feats = detector.extract_feat(image)
        raw = prior.flatten_prediction(detector.simple_test_from_features(feats, metas, rescale=False))
    if any(f.requires_grad or f.grad_fn is not None for f in feats):
        raise ValueError('Frozen detector graph retained')
    p3 = feats[0]
    if p3.shape != (1, 256, 128, 128) or tuple(image.shape[-2:]) != (1024, 1024):
        raise ValueError('B feature/padding contract differs')
    b_model = image.new_tensor(raw[:, :5])
    b_original = map_boxes(b_model, meta, inverse=True)
    if not meta.get('flip', False):
        with torch.no_grad():
            native = prior.flatten_prediction(detector.simple_test_from_features(feats, metas, rescale=True))
        if not np.allclose(b_original.cpu().numpy(), native[:, :5], atol=1e-4, rtol=1e-6):
            raise ValueError('Wrapper original coordinates differ from native B')
    probes, names = component_probes_original(gt)
    boxes_original = torch.cat((b_original, probes))
    boxes = torch.cat((b_model, map_boxes(probes, meta)))
    scores = image.new_full((len(boxes),), float(raw[0, 5]) if len(raw) else .5)
    target, mask, errors = train_quality_targets_original(boxes_original, gt, qualification)
    response, valid = response_targets(map_points(axis, meta), p3.shape[-2:], meta)
    b_saved = np.concatenate((b_original.cpu().numpy(), raw[:, 5:6]), axis=1).tolist()
    record = dict(image=source['image'], sequence=source['sequence'], domain=source['domain'],
        input_sha256=support.fingerprint(source), view_seed=view_seed, view_image_sha256=prior.tensor_sha(image),
        scale_factor=np.asarray(meta['scale_factor']).tolist(), img_shape=list(meta['img_shape']),
        pad_shape=list(meta['pad_shape']), flip=bool(meta.get('flip', False)),
        flip_direction=meta.get('flip_direction'), gt_original=gt.cpu().tolist()[0],
        qualification=qualification, genuine_count=len(raw), probes=len(names),
        genuine_b_original=b_saved, genuine_errors_original=errors[:len(raw)].cpu().tolist(),
        genuine_targets=target[:len(raw)].cpu().tolist(), train_angle_mask=float(mask[0, 2]))
    return (p3, boxes, scores, meta, target, mask, len(raw), response, valid), record, (feats, raw, metas)


def checked_smoke(path, sources, protocol, proof, frozen):
    report = json.loads(path.read_text())
    if (report.get('status') != 'PAIRED_SMOKE_SAVE_RELOAD_PASS_DISCARDED'
            or report['sources'] != sources or report['protocol'] != protocol
            or report['snapshot_sha256'] != SNAPSHOT_SHA or report['b_state'] != frozen
            or report['optimizer_steps_per_arm'] != 4
            or report['cache_sha256'] != proof['cache_sha256']
            or not report['save_reload_quality_exact'] or not report['b_raw_unchanged']):
        raise ValueError('New paired smoke did not pass the identical current contract')
    payload = branch.read_checkpoint(Path(report['checkpoint']))
    if (payload['contract']['role'] != 'discarded_smoke'
            or payload['optimizer_steps_per_arm'] != 4
            or branch.sha(report['checkpoint']) != report['checkpoint_sha256']):
        raise ValueError('Paired smoke checkpoint identity differs')
    return branch.sha(path)


def run(args, inputs, proof, cfg, sources, protocol):
    detector, runtime = build_runtime(cfg, args.b_checkpoint, args.gpu)
    frozen = prior.state_digest(detector)
    if frozen != proof['b_cache_state']:
        raise ValueError('Loaded B state differs from reviewed genuine TRAIN cache')
    role = 'formal_train' if args.mode == 'train' else 'discarded_smoke'
    smoke_sha = checked_smoke(args.smoke_report, sources, protocol, proof, frozen) if role == 'formal_train' else None
    arms = branch.make_arms('cuda:'+str(args.gpu))
    optimizers = branch.make_optimizers(arms)
    contract = dict(protocol=protocol, sources=sources, role=role, frozen_b=ready.FROZEN_B,
        frozen_b_state=frozen, runtime=runtime, numeric_input_snapshot_sha256=SNAPSHOT_SHA,
        train_input_manifest_sha256=proof['input_manifest_sha256'],
        train_cache_sha256=proof['cache_sha256'], direction_policy=POLICY,
        architecture=branch.architecture(arms), paired_smoke_sha256=smoke_sha,
        training_pipeline_sha256=support.fingerprint([dict(spec) for spec in cfg.data.train]))
    epoch_start, steps, chain = 0, 0, support.fingerprint(dict(seed=branch.SEED))
    if args.resume:
        if role != 'formal_train':
            raise ValueError('Only formal epoch-boundary resume supported')
        payload = branch.read_checkpoint(args.resume, contract)
        if payload['epoch'] >= branch.EPOCHS:
            raise ValueError('Formal budget already completed')
        branch.load_bundle(payload, arms, optimizers, restore_random=True)
        epoch_start, steps, chain = payload['epoch'], payload['optimizer_steps_per_arm'], payload['view_chain_sha256']
        del payload
    args.work_dir.mkdir(parents=True, exist_ok=False)
    branch.write_new(args.work_dir/'contract.json', contract)
    branch.write_new(args.work_dir/'input_check.json', proof)
    branch.save_checkpoint(args.work_dir/('epoch_%02d.pth' % epoch_start), arms, optimizers,
                           contract, epoch_start, steps, frozen, chain)
    torch.cuda.reset_peak_memory_stats(args.gpu)
    if role == 'formal_train':
        indexed = datasets_for(deepcopy(cfg.data.train), inputs)
    else:
        smoke_sets = {}
        for _, scale, flip in protocol['smoke_views']:
            key = (scale, flip)
            if key not in smoke_sets:
                smoke_sets[key] = datasets_for(prior.fixed_specs(cfg, scale, flip), inputs)
        positions = {r['image']: i for i, r in enumerate(inputs)}
    final_epoch = branch.EPOCHS if role == 'formal_train' else 1
    reload_exact = False
    with (args.work_dir/'train_steps.jsonl').open('x') as log:
        for epoch in range(epoch_start+1, final_epoch+1):
            order = np.random.RandomState(branch.SEED+epoch).permutation(len(inputs)) if role == 'formal_train' else range(4)
            for slot, index in enumerate(order):
                if role == 'formal_train':
                    source, item = inputs[index], indexed[index]
                else:
                    name, scale, flip = protocol['smoke_views'][slot]
                    index = positions[name]
                    source, item = inputs[index], smoke_sets[(scale, flip)][index]
                view_seed = branch.SEED+100000*epoch+slot
                values, record, raw_state = training_view(detector, item, source, args.gpu, view_seed)
                record.update(epoch=epoch, slot=slot, dataset_index=int(index), optimizer_step=steps+1)
                record['arms'] = branch.update_arms(arms, optimizers, *values)
                assert_detector_frozen(detector)
                steps += 1
                chain = branch.fingerprint(dict(previous=chain, view={k: record[k] for k in
                    ('epoch', 'slot', 'image', 'view_seed', 'view_image_sha256', 'genuine_b_original')}))
                record['paired_view_chain_sha256'] = chain
                log.write(json.dumps(record, allow_nan=False)+'\n')
                log.flush()
                # New persistent-update save/load check; no old 14-view repetition.
                if role == 'discarded_smoke' and slot == 1:
                    path = args.work_dir/'smoke_step_02.pth'
                    before_q = branch.online_qualities(arms, *values[:4])
                    branch.save_checkpoint(path, arms, optimizers, contract, 0, steps, frozen, chain)
                    fresh = branch.make_arms('cuda:'+str(args.gpu))
                    fresh_opt = branch.make_optimizers(fresh)
                    payload = branch.read_checkpoint(path, contract)
                    branch.load_bundle(payload, fresh, fresh_opt, restore_random=True)
                    if before_q != branch.online_qualities(fresh, *values[:4]):
                        raise ValueError('Reload changed same-view quality scores')
                    arms, optimizers = fresh, fresh_opt
                    del payload, fresh, fresh_opt
                if (slot+1) % 100 == 0 or role == 'discarded_smoke' or slot+1 == len(order):
                    print(role, 'epoch', epoch, 'view', slot+1, '/', len(order),
                        source['image'], 'loss', {k: round(v['loss_total'], 6) for k, v in record['arms'].items()}, flush=True)
                if slot+1 != len(order):
                    del values, raw_state
            assert_detector_frozen(detector)
            if prior.state_digest(detector) != frozen:
                raise ValueError('B parameter/buffer changed during paired epoch')
            path = args.work_dir/('epoch_%02d.pth' % epoch)
            branch.save_checkpoint(path, arms, optimizers, contract, epoch, steps, frozen, chain)
            before_q = branch.online_qualities(arms, *values[:4])
            payload = branch.read_checkpoint(path, contract)
            branch.load_bundle(payload, arms, optimizers, restore_random=True)
            reload_exact = before_q == branch.online_qualities(arms, *values[:4])
            if not reload_exact:
                raise ValueError('Epoch bundle reload changed same-view quality')
            del payload
            if role == 'discarded_smoke':
                feats, raw, metas = raw_state
                with torch.no_grad():
                    repeated = prior.flatten_prediction(detector.simple_test_from_features(feats, metas, rescale=False))
                if not np.array_equal(raw, repeated):
                    raise ValueError('Branch update/reload changed raw B prediction')
            del values, raw_state
    if checked_sources() != (sources, protocol):
        raise ValueError('Sources changed during training')
    _, final_proof = checked_inputs(args.input_snapshot, args.train_cache, args.structure_report)
    if final_proof != proof:
        raise ValueError('TRAIN sources/cache changed during training')
    report = dict(status='PAIRED_FORMAL_TRAIN_COMPLETE_FIXED_EPOCH8' if role == 'formal_train' else
        'PAIRED_SMOKE_SAVE_RELOAD_PASS_DISCARDED', sources=sources, protocol=protocol,
        snapshot_sha256=SNAPSHOT_SHA, cache_sha256=proof['cache_sha256'], b_state=frozen,
        detector_optimizer_steps=0, optimizer_steps_per_arm=steps, completed_epoch=final_epoch,
        paired_view_chain_sha256=chain, save_reload_quality_exact=reload_exact,
        b_raw_unchanged=True, checkpoint=str(path.resolve()), checkpoint_sha256=branch.sha(path),
        train_log_sha256=branch.sha(args.work_dir/'train_steps.jsonl'), resume_from=str(args.resume) if args.resume else None,
        runtime=runtime, max_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20,
        max_reserved_mib=torch.cuda.max_memory_reserved(args.gpu)/2**20,
        val_or_test_read=False, test_repeatedly_exposed=True)
    branch.write_new(args.work_dir/'completion.json', report)
    print('Saved', args.work_dir/'completion.json', report['status'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'smoke', 'train'), default='check')
    parser.add_argument('--input-snapshot', type=Path, default=Path('work_dirs/port_reliability_train_support_v1/train_input_snapshot.json'))
    parser.add_argument('--train-cache', type=Path, default=Path('work_dirs/port_reliability_train_support_v1_cache'))
    parser.add_argument('--structure-report', type=Path, default=Path('work_dirs/port_reliability_readiness_v1_structure_complete/train_structure_check.json'))
    parser.add_argument('--b-checkpoint', type=Path, default=prior.CHECKPOINT)
    parser.add_argument('--smoke-report', type=Path, default=Path('work_dirs/port_reliability_branches_v1_smoke/completion.json'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    import os
    os.chdir(ROOT)
    if args.work_dir.exists():
        raise FileExistsError('Use a NEW --work-dir; existing evidence is preserved: '+str(args.work_dir))
    if args.mode != 'train' and args.resume:
        raise ValueError('Resume is only for formal training')
    sources, protocol = checked_sources()
    inputs, proof = checked_inputs(args.input_snapshot, args.train_cache, args.structure_report)
    cfg = prior.check_cfg()
    if args.mode == 'check':
        branch.write_new(args.work_dir/'input_check.json', dict(proof, sources=sources, protocol=protocol,
            architecture=branch.architecture(branch.make_arms()),
            status='FORMAL_TRAIN_CONTRACT_PASS_CPU_ONLY', val_or_test_read=False, optimizer_steps=0))
        print('Saved', args.work_dir/'input_check.json', 'CPU only; no fitting/inference')
    else:
        run(args, inputs, proof, cfg, sources, protocol)


if __name__ == '__main__':
    main()
