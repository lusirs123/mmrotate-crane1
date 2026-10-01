#!/usr/bin/env python3
"""Frozen B ep24 / E-H ep22, eight bounded TRAIN forwards, zero updates.

Inspect the actual main-KLD nodes; B's H is explicitly counterfactual.
No coefficient search, new training, checkpoint selection, VAL/TEST inference.
"""
import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.analyze_port_shape_e_h_train_logs_v1 import config_value, NAMES
from crane_project.tools.preflight_port_center_size_v1 import (
    CONTROL, fixed_train_specs, set_assignment_phase, sum_loss)
from crane_project.tools.preflight_port_shape_e_h_v1 import (
    check_configs, EXPERIMENT, EXPECTED, SOURCES as EXISTING_SOURCES)
from crane_project.tools.preflight_port_shape_e_v1 import capture_main_positives
from crane_project.tools.diagnose_port_center_size_d_v1 import (
    sha, gradient_vector, grad_norm, grad_cos, parameter_digest, ordered_data_infos)

PAIRS = ((0, 1810), (905, 2184))
IMAGES = (('real_seq01_00000', 'sim_seq08_00000'),
          ('real_seq06_00006', 'sim_seq08_00374'))
EPOCHS = dict(b=24, e_h=22)
WEIGHT_SHAS = dict(
    b='8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23',
    e_h='b61c5fed3fdec9b70b0c1ecf91699bff1a01e05ee665fb0eff593fb7e7d1ca5a')
SOURCES = tuple(dict.fromkeys(EXISTING_SOURCES + (
    'crane_project/tools/diagnose_port_shape_e_h_train_gradients_v1.py',
    'crane_project/tools/analyze_port_shape_e_h_train_logs_v1.py',
    'crane_project/tools/ckpt_sweep.py',
    'crane_project/configs/crane_symeood_k1_port_day2night_center_size_d_v1.py',
    'configs/_base_/schedules/schedule_1x.py',
    'configs/_base_/default_runtime.py',
    'mmrotate/datasets/crane_custom_dota.py',
    'mmrotate/datasets/pipelines/transforms.py',
    'mmrotate/core/bbox/assigners/sym_pola.py',
    'mmrotate/models/losses/sym_nfl_loss.py')))
MANIFEST = ROOT/'crane_project/tools/port_shape_e_h_train_gradients_v1_sources.json'


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def checkpoint_contract(meta, cfg, arm):
    """Strict selected fields, safe AST only; never execute checkpoint config."""
    fields = ('model', 'data', 'optimizer', 'optimizer_config', 'lr_config',
              'runner', 'load_from', 'resume_from')
    parsed = {}
    for node in ast.parse(meta.get('config', '')).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in fields):
            key = node.targets[0].id
            if key in parsed:
                raise ValueError('Duplicate saved config field: ' + key)
            parsed[key] = config_value(node.value)
    if (set(parsed) != set(fields)
            or parsed != plain({k: cfg.to_dict()[k] for k in fields})
            or meta.get('seed') != 0 or meta.get('epoch') != EPOCHS[arm]
            or meta.get('iter') != 640 * EPOCHS[arm]):
        raise ValueError('Saved checkpoint config/seed/epoch/iter does not match ' + arm)
    return dict(status='MATCH', checked_fields=list(fields), seed=0,
                epoch=meta['epoch'], iter=meta['iter'],
                config_text_sha256=hashlib.sha256(meta['config'].encode()).hexdigest())


def frozen_identity(sweep, arm, config_path):
    selection_path = Path(sweep).resolve()/'sweep_results.json'
    selection = json.loads(selection_path.read_text())
    key = 'epoch_' + str(EPOCHS[arm])
    if (selection.get('evidence_role') != 'source_val_checkpoint_selection'
            or selection.get('selected_checkpoint') != key
            or selection.get('config_sha256') != sha(config_path)):
        raise ValueError('Frozen selection/config identity differs for ' + arm)
    record = selection['all_checkpoints'][key]
    checkpoint = Path(record['checkpoint']).resolve()
    digest = sha(checkpoint)
    if (Path(selection['selected_path']).resolve() != checkpoint
            or record['checkpoint_sha256'] != digest or digest != WEIGHT_SHAS[arm]):
        raise ValueError('Checkpoint is not the previously reviewed frozen ' + arm)
    # Only identity fields are retained/read for use; no VAL metrics or caches.
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=digest,
                selected_epoch=key, selection_path=str(selection_path),
                selection_sha256=sha(selection_path), config_sha256=sha(config_path))


@contextmanager
def late_state(model):
    """Restore all counters/buffers even on failure; reject BN/other mutation."""
    import torch
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    scheduled = set()
    counter = model.bbox_head.assigner._local_call_count
    try:
        if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training
               for m in model.modules()):
            raise ValueError('TRAIN probe requires the frozen norm_eval BN state')
        set_assignment_phase(model.bbox_head, 'late_o2o')
        for name, module in model.named_modules():
            if hasattr(module, '_local_iter') and hasattr(module, 'loss_call_factor'):
                scheduled.add(name + '._local_iter')
                module._local_iter.fill_(module.warmup_iters * module.loss_call_factor)
                if module._get_current_tau() != module.tau_min:
                    raise RuntimeError('Classification temperature is not in late state')
        yield
        current = dict(model.named_buffers())
        if any(not torch.equal(current[k], v) for k, v in buffers.items() if k not in scheduled):
            raise RuntimeError('Probe changed a non-schedule model buffer')
    finally:
        with torch.no_grad():
            for k, value in model.named_buffers():
                value.copy_(buffers[k])
        model.bbox_head.assigner._local_call_count = counter


@contextmanager
def observe_nodes(head):
    """Match per-level image/anchor identities to actual KLD and optional H nodes."""
    levels, extras = [], []
    original = head.loss_single
    extra = head.shape_compensation
    original_extra = extra.forward if extra is not None else None

    def observed(self, *args, **kwargs):
        import torch
        cls, reg, anchors, labels = args[:4]
        with torch.no_grad():
            labels = labels.reshape(-1)
            indices = torch.nonzero((labels >= 0) & (labels < self.num_classes)).reshape(-1)
            per_image = labels.numel() // cls.shape[0]
            flat = reg.permute(0, 2, 3, 1).reshape(-1, 5)
            before = self.bbox_coder.decode(anchors.reshape(-1, 5), flat)[indices]
            max_wh = float(kwargs.get('decode_max_size') or 2048.)
            levels.append(dict(level=len(levels), n=len(indices),
                identities=[(int(i)//per_image, int(i)%per_image) for i in indices],
                delta_wh_clip_coordinates=int((flat[indices, 2:4].abs() > abs(math.log(16/1000))).sum()),
                edge_clamp_boxes=int(((before[:, 2:4] <= 1.) | (before[:, 2:4] >= max_wh)).any(-1).sum()),
                center_clamp_boxes=int(((before[:, :2] <= 0.) | (before[:, :2] >= max_wh-1.)).any(-1).sum())
                    if kwargs.get('decode_max_size') is not None else 0))
        return original(*args, **kwargs)

    def observed_extra(pred, target, weight=None, avg_factor=None, **kwargs):
        value = original_extra(pred, target, weight=weight, avg_factor=avg_factor, **kwargs)
        extras.append((pred, target, weight, avg_factor, value))
        return value

    head.loss_single = MethodType(observed, head)
    if extra is not None:
        extra.forward = observed_extra
    try:
        yield levels, extras
    finally:
        head.loss_single = original
        if extra is not None:
            extra.forward = original_extra


def original_box(box, scale_factor):
    """Invert TRAIN RResize (rounded sx/sy) then the common isotropic shrink.

    This reports coordinates; it does not replace the inference restoration code.
    RResize scales centres by sx/sy and both edges by sqrt(sx*sy).
    """
    import numpy as np
    factor = np.asarray(scale_factor, dtype=float).reshape(-1)
    if (len(factor) != 4 or not np.isfinite(factor).all() or (factor <= 0).any()
            or not np.allclose(factor[:2], factor[2:], rtol=0, atol=1e-7)):
        raise ValueError('Invalid RResize scale_factor')
    edge = math.sqrt(factor[0]*factor[1])
    return [float(v)/s for v, s in zip(box, (factor[0], factor[1], edge, edge, 1.))]


def signed_response(pred, target, gradient):
    """Infinitesimal decoded-box SGD direction at frozen assignment, no step.

    Error*gradient>0 means local correction. Log-edge and rad-angle gradients
    are reported separately; their raw norms are never ranked against each other.
    """
    pw, ph, pa = (float(v) for v in pred[2:])
    tw, th, ta = (float(v) for v in target[2:])
    long_index, short_index = ((2, 3) if pw >= ph else (3, 2))
    gt_long, gt_short = max(tw, th), min(tw, th)
    long_error, short_error = math.log(max(pw, ph)/gt_long), math.log(min(pw, ph)/gt_short)
    long_grad = float(pred[long_index]*gradient[long_index])
    short_grad = float(pred[short_index]*gradient[short_index])
    difference = pa + (math.pi/2 if pw < ph else 0.) - ta - (math.pi/2 if tw < th else 0.)
    angle_error = .5*math.atan2(math.sin(2*difference), math.cos(2*difference))
    angle_grad = float(gradient[4])
    return dict(log_long_error=long_error, log_short_error=short_error,
        log_long_gradient=long_grad, log_short_gradient=short_grad,
        long_error_times_gradient=long_error*long_grad,
        short_error_times_gradient=short_error*short_grad,
        canonical_angle_error_rad=angle_error, canonical_angle_error_deg=math.degrees(angle_error),
        angle_gradient_rad=angle_grad, angle_error_times_gradient=angle_error*angle_grad,
        gt_log_aspect=math.log(gt_long/gt_short), pred_log_aspect=abs(math.log(pw/ph)),
        angle_away_from_wrap_boundary=abs(abs(angle_error)-math.pi/2) > 1e-6,
        angle_defined_for_both=abs(math.log(pw/ph)) > 1e-6 and abs(math.log(tw/th)) > 1e-6,
        xy_gradient=[float(v) for v in gradient[:2]])


def probe_batch(model, batch, arm, scale):
    import torch
    from mmrotate.models.losses.covariance_shape_loss import CovarianceShapeLoss, covariance_shape_terms
    from mmrotate.models.losses.sym_kld_calculator import sym_kld
    head = model.bbox_head
    if any(m.get('flip', False) for m in batch['img_metas']):
        raise ValueError('Unexpected flip in fixed TRAIN views')
    if any(len(gt) != 1 for gt in batch['gt_bboxes']):
        raise ValueError('Expected the previous one-GT-per-image TRAIN samples')
    with late_state(model), capture_main_positives(head) as main, observe_nodes(head) as (levels, extras):
        losses = model(return_loss=True, **batch)
        base_terms = {k: sum_loss(v) for k, v in losses.items()
                      if 'loss' in k and k != 'loss_shape_compensation'}
        bbox = base_terms['loss_bbox']
        base_total = sum(base_terms.values())
        if len(main) != len(levels) or len(main) != 5:
            raise RuntimeError('Main per-level node capture differs')
        if not torch.allclose(sum(x[4] for x in main), bbox, rtol=1e-6, atol=1e-8):
            raise RuntimeError('Main loss capture differs or a protection masked the loss')
        reference = CovarianceShapeLoss(**{k: v for k, v in EXPECTED.items() if k != 'type'})
        hypothetical = sum(reference(p, t, weight=w, avg_factor=n) for p, t, w, n, _ in main)
        extra_loss = hypothetical
        if arm == 'e_h':
            if len(extras) != len(main) or any(
                    p is not ep or t is not et or not torch.equal(w, ew) or n != en
                    for (p, t, w, n, _), (ep, et, ew, en, _) in zip(main, extras)):
                raise RuntimeError('H did not reuse actual main nodes/weights/normalizers')
            extra_loss = sum_loss(losses['loss_shape_compensation'])
            if not torch.allclose(extra_loss, hypothetical, rtol=1e-6, atol=1e-8):
                raise RuntimeError('Actual H differs from fixed .05 formula')
        if not all(bool(torch.isfinite(v).all()) for v in list(base_terms.values()) + [extra_loss]):
            raise RuntimeError('Nonfinite actual diagnostic loss')
        nodes = [x[0] for x in main]
        direct = {name: gradient_vector(value, nodes) for name, value in
                  (('main_kld', bbox), ('shape_005', extra_loss))}
        if any(g is None for gs in direct.values() for g in gs):
            raise RuntimeError('Disconnected actual main nodes, including empty levels')
        if any(torch.count_nonzero(g[:, :2]) for g in direct['shape_005']):
            raise RuntimeError('Shape term has a direct centre gradient')
        positives = []
        per_image = [0]*len(batch['img_metas'])
        for j, ((pred, target, weight, normalizer, _), level) in enumerate(zip(main, levels)):
            if len(pred) != level['n'] or float(normalizer) != 2.:
                raise RuntimeError('Late O2O count/normalizer differs')
            for i, (image_index, anchor) in enumerate(level['identities']):
                per_image[image_index] += 1
                if not torch.isfinite(pred[i]).all() or not torch.isfinite(target[i]).all():
                    raise RuntimeError('Nonfinite positive box')
                # Canonical OBB representation may swap edges during encode/decode.
                source = batch['gt_bboxes'][image_index][0]
                from mmrotate.models.losses.sym_kld_calculator import _xywha_to_gaussian
                with torch.no_grad():
                    _, target_cov = _xywha_to_gaussian(target[i:i+1])
                    _, source_cov = _xywha_to_gaussian(source[None])
                if (not torch.allclose(target[i, :2], source[:2], atol=1e-3, rtol=1e-5)
                        or not torch.allclose(target_cov, source_cov, atol=1e-3, rtol=1e-5)):
                    raise RuntimeError('Positive target does not match this image GT')
                meta = batch['img_metas'][image_index]
                response = {k: signed_response(pred[i].detach(), target[i].detach(), gs[j][i])
                            for k, gs in direct.items()}
                original_pred = original_box(pred[i].detach().cpu().tolist(), meta['scale_factor'])
                original_gt = original_box(target[i].detach().cpu().tolist(), meta['scale_factor'])
                positives.append(dict(image=Path(meta['filename']).stem, level=j, anchor=anchor,
                    weight=float(weight[i]), normalizer=float(normalizer),
                    input_pred=pred[i].detach().cpu().tolist(), input_gt=target[i].detach().cpu().tolist(),
                    original_pred=original_pred, original_gt=original_gt,
                    center_distance_original_px=math.hypot(original_pred[0]-original_gt[0], original_pred[1]-original_gt[1]),
                    direct=response))
        if per_image != [1, 1]:
            raise RuntimeError('Frozen late assignment must have one positive per image')
        gradients = {}
        groups = dict(regression_conv=[head.retina_reg.weight, head.retina_reg.bias],
                      classification_conv=[head.retina_cls.weight, head.retina_cls.bias],
                      full_fpn=[p for p in model.neck.parameters() if p.requires_grad])
        for name, params in groups.items():
            gs = {k: gradient_vector(value, params) for k, value in
                  (('main_kld', bbox), ('shape_005', extra_loss), ('base_total', base_total))}
            norms = {k: grad_norm(v) for k, v in gs.items()}
            gradients[name] = dict(norms=norms,
                shape_to_main=norms['shape_005']/norms['main_kld'] if norms['main_kld'] else None,
                shape_to_base_total=norms['shape_005']/norms['base_total'] if norms['base_total'] else None,
                cos_shape_main=grad_cos(gs['shape_005'], gs['main_kld']),
                cos_shape_base_total=grad_cos(gs['shape_005'], gs['base_total']))
        if gradients['classification_conv']['norms']['shape_005'] != 0:
            raise RuntimeError('Unexpected direct H classification-conv signal')
        # Only detached concatenations are used here, never as autograd targets.
        with torch.no_grad():
            pred, target = torch.cat(nodes).detach(), torch.cat([x[1] for x in main]).detach()
            _, guards = covariance_shape_terms(pred, target, return_guards=True)
            raw = sym_kld(pred, target, eps=head.loss_bbox.eps)
            if not torch.isfinite(raw).all():
                raise RuntimeError('Nonfinite raw main KLD, possibly masked by the head')
            # KLD uses float32; H's double guard counts cannot certify KLD guards.
            from mmrotate.models.losses.sym_kld_calculator import _inv2x2_safe, _xywha_to_gaussian
            _, p_cov = _xywha_to_gaussian(pred)
            _, t_cov = _xywha_to_gaussian(target)
            determinant_floors = inverse_limits = 0
            for cov in (p_cov, t_cov):
                det = cov[:, 0, 0]*cov[:, 1, 1]-cov[:, 0, 1]*cov[:, 1, 0]
                determinant_floors += int((det <= head.loss_bbox.eps).sum())
                inverse_limits += int((_inv2x2_safe(cov, head.loss_bbox.eps).abs() >= 1e4).any(-1).any(-1).sum())
            main_guards = dict(determinant_floors=determinant_floors, inverse_limits=inverse_limits,
                               raw_caps=int((raw >= 1e4).sum()), loss_caps=int((raw >= 120.).sum()))
        return dict(arm=arm, scale=scale, phase='late_o2o',
            shape_role='actual_fixed_e_h' if arm == 'e_h' else 'counterfactual_only_not_B_training',
            classification_tau=float(head.loss_cls._get_current_tau()),
            assigner_call_count=int(head.assigner._local_call_count),
            losses={k: float(v.detach()) for k, v in base_terms.items()},
            loss_shape_005=float(extra_loss.detach()), actual_node_reuse=True,
            per_image_positive_counts=per_image, positives=positives, gradients=gradients,
            decoder_guards=levels, covariance_guards_double=guards,
            main_raw_kld=raw.cpu().tolist(), main_guards_float32=main_guards)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def runtime_probe(configs, identities, gpu, progress):
    import numpy as np
    import torch
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    from crane_project.tools.ckpt_sweep import annotation_set_sha256
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    datasets = {s: build_dataset(fixed_train_specs(configs['b'], s)) for s in (1., .5)}
    if any([len(x) for x in ds.datasets] != [1810, 748] for ds in datasets.values()):
        raise ValueError('TRAIN split/count differs')
    infos = [ordered_data_infos(datasets[s]) for s in (1., .5)]
    if [x['filename'] for x in infos[0]] != [x['filename'] for x in infos[1]]:
        raise ValueError('TRAIN ordering differs between views')
    for indices, images in zip(PAIRS, IMAGES):
        if tuple(Path(infos[0][i]['filename']).stem for i in indices) != images:
            raise ValueError('Fixed TRAIN sample identity differs')
    rows, states, inputs = [], {}, {}
    for arm in ('b', 'e_h'):
        model = build_detector(deepcopy(configs[arm].model))
        loaded = load_checkpoint(model, identities[arm]['checkpoint'], map_location='cpu', strict=True)
        contract = checkpoint_contract(loaded.get('meta', {}), configs[arm], arm)
        del loaded
        model.cuda(gpu).train()
        before = parameter_digest(model)
        buffers = {k: tensor_sha(v) for k, v in model.named_buffers()}
        torch.cuda.reset_peak_memory_stats(gpu)
        for s_index, scale in enumerate((1., .5)):
            for p_index, (pair, images) in enumerate(zip(PAIRS, IMAGES)):
                seed = 1701 + s_index*2 + p_index
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                batch = scatter(collate([datasets[scale][i] for i in pair], samples_per_gpu=2), [gpu])[0]
                if tuple(Path(m['filename']).stem for m in batch['img_metas']) != images:
                    raise ValueError('TRAIN loader substituted an image')
                original_gt = [original_box(gt[0].detach().cpu().tolist(), m['scale_factor'])
                               for gt, m in zip(batch['gt_bboxes'], batch['img_metas'])]
                record = dict(input_sha256=tensor_sha(batch['img']), original_gt=original_gt,
                    gt_sha256=[tensor_sha(gt) for gt in batch['gt_bboxes']],
                    image_sha256=[sha(m['filename']) for m in batch['img_metas']],
                    scale_factors=[plain(np.asarray(m['scale_factor']).tolist()) for m in batch['img_metas']],
                    input_shapes=[list(m['img_shape']) for m in batch['img_metas']])
                key = (scale, p_index)
                if arm == 'b':
                    inputs[key] = record
                    if scale == .5:
                        clean = inputs[(1., p_index)]
                        if (not np.allclose(record['original_gt'], clean['original_gt'], rtol=1e-5, atol=1e-3)
                                or not np.allclose(record['scale_factors'], np.asarray(clean['scale_factors'])*.5, rtol=1e-6, atol=1e-7)):
                            raise ValueError('Half-scale GT or common scale restoration differs')
                elif record != inputs[key]:
                    raise ValueError('B/E TRAIN input tensors/GT/scale/image hashes differ')
                row = probe_batch(model, batch, arm, scale)
                row.update(record, images=list(images), source_indices=list(pair), seed=seed)
                rows.append(row)
                with Path(progress).open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                print('TRAIN', arm, scale, list(images), 'H', row['loss_shape_005'],
                      'FPN H/main', row['gradients']['full_fpn']['shape_to_main'], flush=True)
                del batch, row
        if before != parameter_digest(model) or buffers != {k: tensor_sha(v) for k, v in model.named_buffers()}:
            raise RuntimeError('TRAIN probe changed saved parameters/buffers')
        states[arm] = dict(checkpoint_contract=contract, parameter_sha256=before,
            parameters_and_buffers_unchanged=True, peak_gpu_bytes=torch.cuda.max_memory_allocated(gpu))
        del model
        torch.cuda.empty_cache()
    pairing = []
    for b, e in zip(rows[:4], rows[4:]):
        pairing.append(dict(images=b['images'], scale=b['scale'], inputs_identical=True,
            selected_positive_anchors_identical=[
                (x['level'], x['anchor']) == (y['level'], y['anchor'])
                for x, y in zip(sorted(b['positives'], key=lambda x: x['image']),
                                sorted(e['positives'], key=lambda x: x['image']))]))
    return dict(rows=rows, states=states, pairing=pairing, optimizer_steps=0,
        annotation_sha256={d: annotation_set_sha256(ROOT/'crane_project/data/crane_grab_port_day2night_v1'/d/'annfiles')
                           for d in ('train', 'train_sim')},
        gpu_name=torch.cuda.get_device_name(gpu), full_train_audit=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', type=int, default=0)
    for arm, option in (('b', 'b'), ('e_h', 'e')):
        ap.add_argument('--'+option+'-sweep', default=str(ROOT/'work_dirs'/NAMES[arm]/'val_sweep_port_v1'))
    ap.add_argument('--config-only', action='store_true')
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()
    out = Path(args.out_json).resolve()
    progress, artifacts = out.with_suffix('.progress.jsonl'), out.with_suffix('.artifacts.json')
    if any(p.exists() for p in (out, progress, artifacts)):
        raise FileExistsError('Keep existing outputs; use a new output name')
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='CHECK_STARTED', evidence_role='bounded_train_gradient_diagnosis',
        sources={s: sha(ROOT/s) for s in SOURCES}, formula_and_coefficient='FROZEN_E_H_005',
        expected_epochs=EPOCHS, expected_images=IMAGES, expected_scales=[1., .5],
        optimizer_steps=0, test_repeatedly_exposed=True,
        limitations=[
            'Four TRAIN images, frozen snapshots; no full-run causality or statistical benefit claim.',
            'Decoded local directions hold assignment fixed; decoder guards/shared layers can alter parameter response.',
            'Log-edge gradients and rad-angle gradients have distinct units; do not rank their raw norms.',
            'Near-square orientation is weakly identifiable; inspect aspect and canonical-equivalence boundaries.',
            'TRAIN positives are not inference outputs; no output coverage or independent depth accuracy claim.',
            'Different B/E positive anchors are not strictly paired gradient targets.',
            'B H term is counterfactual .05 only; never a new trained B loss or a coefficient recommendation.'])
    try:
        manifest = json.loads(MANIFEST.read_text())
        if manifest.get('sources') != report['sources']:
            raise ValueError('Source manifest differs; use the reviewed code bundle, do not bypass identity checks')
        report['source_manifest_sha256'] = sha(MANIFEST)
        b, e = check_configs()
        configs = dict(b=b, e_h=e)
        report['config_sha256'] = dict(b=sha(CONTROL), e_h=sha(EXPERIMENT))
        if args.config_only:
            report['status'] = 'CONFIG_ONLY_RUNTIME_UNVERIFIED'
        else:
            report['identities'] = dict(b=frozen_identity(args.b_sweep, 'b', CONTROL),
                                       e_h=frozen_identity(args.e_sweep, 'e_h', EXPERIMENT))
            with artifacts.open('x') as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
            with progress.open('x'):
                pass
            report['train'] = runtime_probe(configs, report['identities'], args.gpu, progress)
            if report['sources'] != {s: sha(ROOT/s) for s in SOURCES}:
                raise RuntimeError('Probe source changed during execution')
            report['status'] = 'TRAIN_GRADIENT_CHECK_COMPLETE_REVIEW_REQUIRED'
    except Exception as error:
        report.update(status='CHECK_FAILED', error=type(error).__name__+': '+str(error))
        raise
    finally:
        with out.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print('Saved', out, 'status', report['status'], flush=True)


if __name__ == '__main__':
    main()
