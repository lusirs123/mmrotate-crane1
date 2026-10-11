#!/usr/bin/env python3
"""Bounded, discarded two-image TRAIN learning check. No VAL/TEST or adoption."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_reliability_oof_v1 as run
from crane_project.utils import port_reliability_oof_training_v2 as repair

STEPS = 256
CHECKPOINTS = (0, 64, 128, 256)


def selected_names(rows, allowed):
    """Select by identity only: midpoint index of first sequence in each domain."""
    result = []
    for domain in ('real', 'sim'):
        available = [r for r in rows if r['image'] in allowed and r['domain'] == domain]
        sequence = min(r['sequence'] for r in available)
        ordered = sorted(r['image'] for r in available if r['sequence'] == sequence)
        result.append(ordered[len(ordered)//2])
    return result


def deterministic_specs(cfg):
    result = deepcopy(cfg.data.train)
    for spec in result:
        for op in spec['pipeline']:
            if op['type'] == 'PortIsotropicShrink': op['prob'] = 0.
            if op['type'] == 'RRandomFlip': op['flip_ratio'] = 0.
    return result


def learning_rate(cfg, step):
    """Same early-iteration LR as MMCV's configured linear warmup."""
    schedule = cfg.lr_config
    if schedule['warmup'] != 'linear' or schedule.get('warmup_by_epoch', False):
        raise ValueError('Expected original iteration-based linear warmup')
    if step >= schedule['warmup_iters']: return cfg.optimizer.lr
    return cfg.optimizer.lr * (schedule['warmup_ratio'] +
        (1. - schedule['warmup_ratio']) * step / schedule['warmup_iters'])


def main(args):
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import build_optimizer
    from mmdet.apis import set_random_seed
    from mmdet.datasets.dataset_wrappers import ConcatDataset
    from mmrotate.datasets import build_dataset
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as frozen

    rows, _, plan, _, proof = run.checked()
    source = args.source.resolve(); folder = source/'A'
    saved = json.loads((folder/'membership.json').read_text())
    if saved['train_images'] != plan['A']['train_images']:
        raise ValueError('Diagnostic source TRAIN membership changed')
    cfg_path = folder/'detector_config.py'
    if run.sha(cfg_path) != saved['resolved_config_sha256']:
        raise ValueError('Diagnostic source config changed')
    cfg = Config.fromfile(str(cfg_path)); repair.validate_config(cfg)
    checkpoint = folder/'detector/epoch_24.pth'
    pin = json.loads((folder/'detector_completion.json').read_text())['checkpoint_sha256']
    if run.sha(checkpoint) != pin: raise ValueError('Diagnostic source checkpoint changed')
    args.out.mkdir(parents=True, exist_ok=False)
    names = selected_names(rows, set(saved['train_images']))
    contract = dict(steps=STEPS, evaluations=CHECKPOINTS, names=names,
        purpose='discarded fixed-batch TRAIN learning diagnosis, not generalization',
        seed=run.core.SETTINGS['seed'], batch=2, image_size=[1024,1024], precision='FP32',
        augmentation=False, learning_rate='original linear warmup, no schedule search',
        optimizer=dict(cfg.optimizer), clip=10., generic_initialization='ImageNet ResNet50 only',
        discarded=True, save_weights=False, automatic_long_training=False,
        VAL_scored=False, VAL_TEST_tensors_loaded=False, TEST_read=False,
        auxiliary_inference_revision=repair.INFERENCE_REVISION,
        source_config_sha256=run.sha(cfg_path), source_checkpoint_sha256=pin, proof=proof)
    contract['continuation'] = 'Both fixed TRAIN images must pass center<15px and both edges<=10% at final step; short pass alone never launches long training'
    run.write(args.out/'contract.json', contract)
    set_random_seed(contract['seed'], deterministic=True); torch.cuda.set_device(args.gpu)
    data = ConcatDataset([build_dataset(s) for s in deterministic_specs(cfg)])
    actual = [Path(i['filename']).stem for i in frozen.infos(data)]
    if set(actual) != set(saved['train_images']): raise ValueError('TRAIN dataset drift')
    indices = [actual.index(n) for n in names]
    batch = scatter(collate([data[i] for i in indices], samples_per_gpu=2), [args.gpu])[0]
    gt = batch['gt_bboxes']; metas = batch['img_metas']
    if any(len(b) != 1 for b in gt) or any(y.tolist() != [0] for y in batch['gt_labels']):
        raise ValueError('Expected one single-class GT per diagnostic image')
    specs = repair.inference_specs(frozen.g.fixed_specs(Config.fromfile(str(run.CONFIG)), 1.), cfg)
    for spec in specs: spec['data_root'] = str((folder/'data').resolve())+'/'
    infer = ConcatDataset([build_dataset(s) for s in specs])
    infer_names = [Path(i['filename']).stem for i in frozen.infos(infer)]
    infer_data = [scatter(collate([infer[infer_names.index(n)]], samples_per_gpu=1), [args.gpu])[0] for n in names]
    for i, value in enumerate(infer_data):
        if not torch.equal(batch['img'][i], value['img'][0][0]):
            raise ValueError('Actual TRAIN/inference pixel tensors differ')
    lookup = {r['image']:r for r in rows}
    center_restore_error = max(float(torch.abs(
        frozen.g.map_boxes(gt[i], metas[i], inverse=True)[0,:2].cpu() -
        torch.tensor(lookup[n]['gt'][:2])).max()) for i,n in enumerate(names))
    if center_restore_error > .001: raise ValueError('TRAIN GT center restoration differs')

    def evaluate(model, step, identity):
        model.eval(); result = []
        with torch.no_grad():
            for n, value in zip(names, infer_data):
                pred = frozen.g.flatten_prediction(model.simple_test(value['img'][0], value['img_metas'][0], rescale=True))
                truth = np.asarray(lookup[n]['gt'], dtype=np.float32)
                item = dict(image=n, gt=truth.tolist(), pred=pred.tolist(), output=bool(len(pred)))
                if len(pred):
                    error = float(np.linalg.norm(pred[0,:2]-truth[:2]))
                    dims = np.sort(pred[0,2:4])[::-1]; target = np.sort(truth[2:4])[::-1]
                    rel = np.abs(dims/target-1.)
                    item.update(center_error=error, center_correct=error<15.,
                        long_relative_error=float(rel[0]), short_relative_error=float(rel[1]),
                        size_correct=bool((rel<=.1).all()))
                result.append(item)
        record = dict(step=step, identity=identity, frames=result)
        run.emit(dict(stage='fixed_TRAIN_readout', **record)); return record

    reference = run.model_for(cfg, checkpoint).cuda(args.gpu)
    reference.requires_grad_(False); reference_state = frozen.g.state_digest(reference)
    evaluations = [evaluate(reference, 24, 'failed_A24_read_only')]
    if frozen.g.state_digest(reference) != reference_state: raise ValueError('Reference model mutated')
    del reference; torch.cuda.empty_cache()
    # Initialize once, no use of the failed checkpoint or formal weights.
    set_random_seed(contract['seed'], deterministic=True)
    model = run.model_for(cfg).cuda(args.gpu)
    optimizer = build_optimizer(model, cfg.optimizer)
    evaluations.append(evaluate(model, 0, 'fresh_discarded'))
    target_stats = []; original_loss_single = model.bbox_head.loss_single

    def inspected_loss_single(head, *values, **kw):
        with torch.no_grad():
            cls, deltas, anchors, labels, _, targets = values[:6]
            labels = labels.reshape(2,-1); per_image = labels.shape[1]
            flat_anchors = anchors.reshape(-1,5)
            decoded_gt = head.bbox_coder.decode(flat_anchors, targets.reshape(-1,5)).reshape(2,per_image,5)
            decoded_pred = head.bbox_coder.decode(flat_anchors, deltas.permute(0,2,3,1).reshape(-1,5)).reshape(2,per_image,5)
            for i in range(2):
                positive = (labels[i]>=0)&(labels[i]<head.num_classes)
                if not positive.any(): continue
                boxes = decoded_gt[i,positive]
                cerror = float((boxes[:,:2]-gt[i][0,:2]).abs().max())
                dim_error = float((boxes[:,2:4].sort(dim=1)[0]-gt[i][0,2:4].sort()[0]).abs().max())
                long_angles = boxes[:,4] + (boxes[:,2]<boxes[:,3]).to(boxes.dtype)*np.pi/2
                gt_long_angle = gt[i][0,4] + (gt[i][0,2]<gt[i][0,3]).to(boxes.dtype)*np.pi/2
                angle_error = float(torch.remainder(long_angles-gt_long_angle+np.pi/2,np.pi).sub(np.pi/2).abs().max())
                target_error = max(cerror, dim_error, angle_error)
                if target_error > .01: raise ValueError('Assigned encoded GT does not reconstruct own-image GT')
                pred = decoded_pred[i,positive]
                saturated = ((pred[:,:2]<0).any(1)|(pred[:,:2]>1023).any(1)|
                    (pred[:,2:4]<1).any(1)|(pred[:,2:4]>1024).any(1))
                target_stats.append(dict(image=names[i], positives=int(positive.sum()),
                    encoded_target_max_error=target_error, pre_clamp_saturated_positive_count=int(saturated.sum())))
        return original_loss_single(*values, **kw)

    model.bbox_head.loss_single = types.MethodType(inspected_loss_single, model.bbox_head)
    guard = repair.MemoryPlateau(); trace = []
    with (args.out/'updates.jsonl').open('x') as log:
        for step in range(STEPS):
            model.train(); optimizer.zero_grad(); target_stats.clear()
            lr = learning_rate(cfg, step)
            for group in optimizer.param_groups: group['lr'] = lr
            value = model.train_step(batch, optimizer); loss = value['loss']
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite fixed-batch loss')
            loss.backward()
            gradients = {key:float(torch.norm(torch.stack([p.grad.detach().norm() for p in module.parameters()
                if p.grad is not None]))) for key,module in (
                ('main_classification',model.bbox_head.retina_cls), ('main_regression',model.bbox_head.retina_reg),
                ('backbone',model.backbone), ('neck',model.neck), ('ATSS',model.aux_heads))}
            if not all(np.isfinite(g) for g in gradients.values()): raise FloatingPointError('Nonfinite module gradient')
            params = [p for p in model.parameters() if p.grad is not None]
            if not all(torch.isfinite(p.grad).all() for p in params): raise FloatingPointError('Nonfinite parameter gradient')
            before = float(torch.nn.utils.clip_grad_norm_(params,10.))
            after = float(torch.norm(torch.stack([p.grad.detach().norm() for p in params])))
            optimizer.step(); torch.cuda.synchronize(args.gpu)
            record = dict(step=step+1, lr=lr, loss=float(loss), gradient_by_module=gradients,
                gradient_before_clip=before, gradient_after_clip=after,
                gradient_clip_fraction=min(1.,10./(before+1e-6)), targets=deepcopy(target_stats),
                logs={k:float(v) for k,v in value['log_vars'].items()})
            # All inspection output is CPU scalar; no per-step CUDA graph is retained.
            del value, loss
            if step+1 in CHECKPOINTS:
                evaluations.append(evaluate(model,step+1,'fresh_discarded'))
            record['memory'] = guard.observe(torch.cuda.memory_allocated(args.gpu),torch.cuda.memory_reserved(args.gpu))
            trace.append(record); log.write(json.dumps(record,allow_nan=False)+'\n'); log.flush()
            if (step+1)%16==0: run.emit(dict(stage='fixed_TRAIN_learning',**record))
    summary = dict(contract=contract, steps_completed=len(trace), evaluations=evaluations,
        TRAIN_inference_pixels_equal=True, GT_center_restore_max_error=center_restore_error,
        target_decode_max_error=max(r['encoded_target_max_error'] for t in trace for r in t['targets']),
        gradient_nonzero_steps={k:sum(t['gradient_by_module'][k]>0 for t in trace) for k in trace[0]['gradient_by_module']},
        memory_guard_passed=True, peak_reserved_mib=torch.cuda.max_memory_reserved(args.gpu)/2**20,
        source_checkpoint_unchanged=run.sha(checkpoint)==pin, weights_discarded=True,
        tiny_batch_learned=all(r.get('center_correct',False) and r.get('size_correct',False)
                               for r in evaluations[-1]['frames']),
        status='DIAGNOSTIC_COMPLETE_NOT_LONG_TRAIN_AUTHORIZATION',
        VAL_scored=False, VAL_TEST_tensors_loaded=False, TEST_read=False)
    run.write(args.out/'report.json',summary); run.emit(summary)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    try: main(args)
    except Exception as exc:
        if args.out.exists() and not (args.out/'failure.json').exists():
            run.write(args.out/'failure.json',dict(error=repr(exc),VAL_read=False,TEST_read=False))
        raise
