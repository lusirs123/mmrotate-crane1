#!/usr/bin/env python3
"""F-S: four frozen B TRAIN probes + four original-init probes; zero updates.

Fixed candidate beta/lambda=.1. No E/D checkpoint, VAL/TEST inference, sweep,
coefficient fitting or formal training. Sources and prerequisite math are bound.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.check_port_size_f_s_v1 import (
    EXPECTED, EXPERIMENT, FIXTURE, MANIFEST, SOURCES, candidate_loss, checked_sources,
    reduction_multiplier)
from crane_project.tools.preflight_port_center_size_v1 import (
    CONTROL, fixed_train_specs, set_assignment_phase, sum_loss)
from crane_project.tools.preflight_port_shape_e_v1 import capture_main_positives
from crane_project.tools.preflight_port_shape_e_h_v1 import initialize_matched_models
from crane_project.tools.diagnose_port_shape_e_h_train_gradients_v1 import (
    PAIRS, IMAGES, frozen_identity, checkpoint_contract, observe_nodes,
    original_box, signed_response, tensor_sha)
from crane_project.tools.diagnose_port_center_size_d_v1 import (
    sha, parameter_digest, gradient_vector, grad_norm, grad_cos, ordered_data_infos)


def check_configs():
    from mmcv import Config
    b, f = (Config.fromfile(str(p)) for p in (CONTROL, EXPERIMENT))
    baseline, actual = deepcopy(b.to_dict()), deepcopy(f.to_dict())
    if actual['model']['bbox_head'].pop('shape_compensation') != EXPECTED:
        raise ValueError('F-S candidate differs from beta/lambda=.1 design')
    actual['work_dir'] = baseline['work_dir']
    if actual != baseline:
        raise ValueError('F-S differs from B beyond size loss/work_dir')
    if (b.model.bbox_head.get('shape_compensation') is not None
            or b.model.bbox_head.get('center_size_compensation') is not None
            or f.model.bbox_head.get('center_size_compensation') is not None
            or b.load_from is not None or b.resume_from is not None
            or b.model.bbox_head.loss_bbox != dict(type='SymKLDLoss', eps=1e-6,
                                                  reduction='mean', loss_weight=2.)
            or b.model.backbone.init_cfg != dict(type='Pretrained', checkpoint='torchvision://resnet50')
            or b.runner.max_epochs != 24 or b.data.samples_per_gpu != 2
            or dict(b.optimizer) != dict(type='SGD', lr=.0025, momentum=.9, weight_decay=.0001)
            or dict(b.optimizer_config.grad_clip) != dict(max_norm=10, norm_type=2)
            or b.model.test_cfg.score_thr != .05 or b.model.test_cfg.max_per_img != 1):
        raise ValueError('Original B initialization/loss/training/inference contract differs')
    if len(b.data.train) != 2:
        raise ValueError('Expected real + sim TRAIN')
    for entry, split in zip(b.data.train, ('train', 'train_sim')):
        if (entry.data_root != 'crane_project/data/crane_grab_port_day2night_v1/'
                or entry.ann_file != split+'/annfiles/' or entry.img_prefix != split+'/images/'
                or [s for s in entry.pipeline if s['type'] == 'PortIsotropicShrink'] !=
                   [dict(type='PortIsotropicShrink', prob=.5, scale_range=(.5, 1.))]):
            raise ValueError('TRAIN data/isotropic augmentation differs')
    return b, f


def rng_snapshot():
    import numpy as np
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


@contextmanager
def phase_state(model, phase):
    """Restore counters/buffers/RNG on every exit, reject non-schedule mutation."""
    import torch
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    counter = model.bbox_head.assigner._local_call_count
    rng = rng_snapshot()
    scheduled = set()
    try:
        if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training
               for m in model.modules()):
            raise ValueError('TRAIN probe requires original norm_eval BN state')
        set_assignment_phase(model.bbox_head, phase)
        for name, module in model.named_modules():
            if hasattr(module, '_local_iter') and hasattr(module, 'loss_call_factor'):
                scheduled.add(name+'._local_iter')
                module._local_iter.fill_(0 if phase == 'warmup_o2m' else
                                        module.warmup_iters*module.loss_call_factor)
        yield
        current = dict(model.named_buffers())
        if current.keys() != buffers.keys() or any(
                not torch.equal(current[k], v) for k, v in buffers.items() if k not in scheduled):
            raise RuntimeError('Probe changed a non-schedule buffer')
    finally:
        with torch.no_grad():
            for k, value in model.named_buffers():
                if k in buffers:
                    value.copy_(buffers[k])
        model.bbox_head.assigner._local_call_count = counter
        restore_rng(rng)


def main_guards(pred, target, eps):
    import torch
    from mmrotate.models.losses.sym_kld_calculator import _xywha_to_gaussian, _inv2x2_safe
    with torch.no_grad():
        mu_p, pc = _xywha_to_gaussian(pred)
        mu_t, tc = _xywha_to_gaussian(target)
        ip, it = _inv2x2_safe(pc, eps), _inv2x2_safe(tc, eps)
        delta = (mu_p-mu_t).unsqueeze(-1)
        raw = .5*(torch.einsum('nij,nji->n', it, pc)+torch.einsum('nij,nji->n', ip, tc)-4.
                  +(delta.transpose(-1, -2)@(ip+it)@delta).reshape(-1))
        if not torch.isfinite(raw).all():
            raise RuntimeError('Nonfinite raw KLD could have been masked by original protections')
        return dict(raw_before_scalar_clamp=raw.cpu().tolist(),
            determinant_floors=sum(int((c[:, 0, 0]*c[:, 1, 1]-c[:, 0, 1]*c[:, 1, 0] <= eps).sum()) for c in (pc, tc)),
            inverse_limits=sum(int((i.abs() >= 1e4).any(-1).any(-1).sum()) for i in (ip, it)),
            raw_negative_roundoff=int((raw < 0).sum()), raw_caps=int((raw >= 1e4).sum()),
            loss_caps=int((raw >= 120.).sum()),
            gaussian_edge_floors=int(((pred[:, 2:4] < 1.) | (target[:, 2:4] < 1.)).sum()))


def probe_batch(model, batch, cfg, stage, scale):
    if stage not in ('frozen_b', 'initialization') or scale not in (1., .5):
        raise ValueError('Only the two bounded TRAIN stages/scales are allowed')
    try:
        return _probe_batch(model, batch, cfg, stage, scale)
    finally:
        model.zero_grad(set_to_none=True)


def _probe_batch(model, batch, cfg, stage, scale):
    import torch
    from mmrotate.models.losses.sym_kld_calculator import _xywha_to_gaussian
    phase = 'late_o2o' if stage == 'frozen_b' else 'warmup_o2m'
    head = model.bbox_head
    if any(m.get('flip', False) for m in batch['img_metas']) or any(len(gt) != 1 for gt in batch['gt_bboxes']):
        raise ValueError('Fixed TRAIN probe must have unflipped one-GT images')
    model.zero_grad(set_to_none=True)
    with phase_state(model, phase):
        forward_rng = rng_snapshot()
        with capture_main_positives(head) as nodes, observe_nodes(head) as (levels, extras):
            losses = model(return_loss=True, **batch)
        base = {k: sum_loss(v) for k, v in losses.items()
                if 'loss' in k and k != 'loss_shape_compensation'}
        bbox, size = base['loss_bbox'], sum_loss(losses['loss_shape_compensation'])
        total_b = sum(base.values())
        if len(nodes) != 5 or len(nodes) != len(extras) or len(nodes) != len(levels):
            raise RuntimeError('Main/size/level capture differs')
        independent = []
        for main, extra in zip(nodes, extras):
            p, t, w, n, _ = main
            if (p is not extra[0] or t is not extra[1] or n != extra[3]
                    or not torch.equal(w, extra[2])):
                raise RuntimeError('Size loss did not reuse actual KLD nodes/weights/normalizer')
            independent.append(candidate_loss()(p, t, weight=w, avg_factor=n))
        for captured, actual in ((sum(x[4] for x in nodes), bbox),
                                 (sum(x[4] for x in extras), size), (sum(independent), size)):
            if not torch.allclose(captured, actual, rtol=1e-6, atol=1e-8):
                raise RuntimeError('Captured/independent loss differs or original loss was masked')
        if any(not torch.isfinite(v).all() for v in list(base.values())+[size, total_b+size]):
            raise RuntimeError('Nonfinite integrated objective')
        count = sum(len(x[0]) for x in nodes)
        if count <= 0 or int(losses['shape_positive_count']) != count:
            raise RuntimeError('Positive count differs')
        direct = {k: gradient_vector(value, [x[0] for x in nodes]) for k, value in
                  (('main_kld', bbox), ('size', size), ('main_plus_size', bbox+size))}
        if any(g is None for gs in direct.values() for g in gs):
            raise RuntimeError('Disconnected actual positive nodes, including empty levels')
        if any(torch.count_nonzero(g[:, [0, 1, 4]]) for g in direct['size']):
            raise RuntimeError('Unexpected direct centre/angle size gradient')
        positives, per_image = [], [0]*len(batch['img_metas'])
        for j, ((p, t, w, n, _), level) in enumerate(zip(nodes, levels)):
            if len(p) != level['n']:
                raise RuntimeError('Anchor capture differs')
            for i, (image_index, anchor) in enumerate(level['identities']):
                per_image[image_index] += 1
                source = batch['gt_bboxes'][image_index][0]
                if not torch.isfinite(p[i]).all() or not torch.isfinite(t[i]).all():
                    raise RuntimeError('Nonfinite actual positive')
                with torch.no_grad():
                    _, cov = _xywha_to_gaussian(t[i:i+1])
                    _, original_cov = _xywha_to_gaussian(source[None])
                if (not torch.allclose(t[i, :2], source[:2], rtol=1e-5, atol=1e-3)
                        or not torch.allclose(cov, original_cov, rtol=1e-5, atol=1e-3)):
                    raise RuntimeError('Target does not match this image GT')
                response = {k: signed_response(p[i].detach(), t[i].detach(), gs[j][i])
                            for k, gs in direct.items()}
                for e, g in (('log_long_error', 'log_long_gradient'), ('log_short_error', 'log_short_gradient')):
                    expected = EXPECTED['loss_weight']*float(w[i])*reduction_multiplier(float(n), p.dtype)/2*max(-1., min(1., response['size'][e]/EXPECTED['beta']))
                    if abs(response['size'][g]-expected) > 1e-6:
                        raise RuntimeError('Actual size gradient differs from independent derivative')
                meta = batch['img_metas'][image_index]
                positives.append(dict(image=Path(meta['filename']).stem, level=j, anchor=anchor,
                    weight=float(w[i]), normalizer=float(n), input_pred=p[i].detach().cpu().tolist(),
                    input_gt=t[i].detach().cpu().tolist(), direct=response,
                    original_pred=original_box(p[i].detach().cpu().tolist(), meta['scale_factor']),
                    original_gt=original_box(t[i].detach().cpu().tolist(), meta['scale_factor'])))
        expected_n = sum(max(v, 1) for v in per_image)
        if any(float(x[3]) != expected_n for x in nodes) or (phase == 'late_o2o' and per_image != [1, 1]):
            raise RuntimeError('Original global positive normalizer/late assignment differs')
        gradients = {}
        groups = dict(regression_conv=[head.retina_reg.weight, head.retina_reg.bias],
                      classification_conv=[head.retina_cls.weight, head.retina_cls.bias],
                      full_fpn=[p for p in model.neck.parameters() if p.requires_grad])
        for name, params in groups.items():
            gs = {k: gradient_vector(value, params) for k, value in
                  (('main_kld', bbox), ('size', size), ('complete_b', total_b))}
            ns = {k: grad_norm(v) for k, v in gs.items()}
            gradients[name] = dict(norms=ns, size_to_main=ns['size']/ns['main_kld'] if ns['main_kld'] else None,
                size_to_complete_b=ns['size']/ns['complete_b'] if ns['complete_b'] else None,
                cos_size_main=grad_cos(gs['size'], gs['main_kld']),
                cos_size_complete_b=grad_cos(gs['size'], gs['complete_b']))
            if name == 'regression_conv':
                gradients[name]['size_xy_angle_output_rows_norm'] = grad_norm(
                    [g[index::5] for g in gs['size'] if g is not None for index in (0, 1, 4)])
        if (gradients['classification_conv']['norms']['size'] != 0
                or gradients['regression_conv']['size_xy_angle_output_rows_norm'] != 0
                or any(gradients[g]['norms']['size'] <= 0 for g in ('regression_conv', 'full_fpn'))):
            raise RuntimeError('Designed gradient scope or effective size signal differs')
        clipping = {}
        params = [p for p in model.parameters() if p.requires_grad]
        for name, value in (('complete_b', total_b), ('complete_f_s', total_b+size)):
            model.zero_grad(set_to_none=True)
            value.backward(retain_graph=True)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise RuntimeError('Nonfinite complete objective gradient')
            clip = cfg.optimizer_config.grad_clip
            before = float(torch.nn.utils.clip_grad_norm_(params, clip.max_norm, norm_type=clip.norm_type))
            if not math.isfinite(before):
                raise RuntimeError('Nonfinite complete clip norm')
            clipping[name] = dict(before=before, after=grad_norm([p.grad for p in params]),
                                  multiplier=min(1., float(clip.max_norm)/(before+1e-6)))
        pred, target = torch.cat([x[0] for x in nodes]).detach(), torch.cat([x[1] for x in nodes]).detach()
        guards = main_guards(pred, target, head.loss_bbox.eps)
        errors = pred[:, 2:4].sort(-1, descending=True).values.log()-target[:, 2:4].sort(-1, descending=True).values.log()
        size_guards = dict(eps_floor_edges=int(((pred[:, 2:4] <= EXPECTED['eps']) | (target[:, 2:4] <= EXPECTED['eps'])).sum()),
                           linear_region_edges=int((errors.abs() >= EXPECTED['beta']).sum()),
                           sorted_pred_ties=int((pred[:, 2] == pred[:, 3]).sum()))
        tau, assigner_count = float(head.loss_cls._get_current_tau()), int(head.assigner._local_call_count)
        if phase == 'late_o2o' and tau != head.loss_cls.tau_min:
            raise RuntimeError('Late classification temperature differs')
        # Same parameters, inputs, RNG and scheduling, with only F-S disabled.
        saved = head.shape_compensation
        try:
            head.shape_compensation = None
            restore_rng(forward_rng)
            with phase_state(model, phase):
                off = model(return_loss=True, **batch)
                off_terms = {k: sum_loss(v).detach() for k, v in off.items() if 'loss' in k}
                del off
        finally:
            head.shape_compensation = saved
        if off_terms.keys() != base.keys() or any(
                not torch.allclose(off_terms[k], v, rtol=1e-5, atol=1e-7) for k, v in base.items()):
            raise RuntimeError('B original losses changed with F-S enabled')
        model.zero_grad(set_to_none=True)
        return dict(stage=stage, scale=scale, phase=phase,
            size_role='counterfactual_on_frozen_b_not_trained_f' if stage == 'frozen_b' else 'actual_original_seed0_initialization',
            losses={k: float(v.detach()) for k, v in base.items()}, loss_size=float(size.detach()),
            loss_complete_f_s=float((total_b+size).detach()), per_image_positive_counts=per_image,
            normalizers=[float(x[3]) for x in nodes], classification_tau=tau, assigner_call_count=assigner_count,
            actual_node_reuse=True, base_losses_unchanged=True, direct_coordinate_norms={key:grad_norm([g[:,i:i+1] for g in direct['size']])
                for i,key in enumerate(('x','y','w','h','angle'))}, positives=positives,
            gradients=gradients, clipping=clipping, decoder_guards=levels, main_guards=guards, size_guards=size_guards)


def input_record(batch):
    import numpy as np
    return dict(input_sha256=tensor_sha(batch['img']),
        original_gt=[original_box(gt[0].detach().cpu().tolist(), m['scale_factor'])
                     for gt, m in zip(batch['gt_bboxes'], batch['img_metas'])],
        gt_sha256=[tensor_sha(gt) for gt in batch['gt_bboxes']],
        image_sha256=[sha(m['filename']) for m in batch['img_metas']],
        scale_factors=[np.asarray(m['scale_factor']).tolist() for m in batch['img_metas']],
        input_shapes=[list(m['img_shape']) for m in batch['img_metas']])


def library_contract():
    import mmcv
    import mmdet
    import torch
    import mmdet.models.losses.utils as utils
    return dict(torch=torch.__version__, mmcv=mmcv.__version__, mmdet=mmdet.__version__,
                weight_reduce_loss_source_sha256=sha(utils.__file__),
                normalizer_2_multiplier_float32=reduction_multiplier(2., torch.float32),
                normalizer_2_multiplier_float64=reduction_multiplier(2., torch.float64))


def training_identity():
    return json.loads(FIXTURE.read_text())['training_identity']


def runtime_probe(b, f, identity, gpu, progress):
    import numpy as np
    import torch
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    from crane_project.tools.ckpt_sweep import annotation_set_sha256
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False  # original tools/train.py initialization setting
    datasets = {s: build_dataset(fixed_train_specs(b, s)) for s in (1., .5)}
    if any([len(x) for x in ds.datasets] != [1810, 748] for ds in datasets.values()):
        raise ValueError('TRAIN split/count differs')
    infos = [ordered_data_infos(datasets[s]) for s in (1., .5)]
    if [x['filename'] for x in infos[0]] != [x['filename'] for x in infos[1]]:
        raise ValueError('TRAIN ordering differs')
    if any(tuple(Path(infos[0][i]['filename']).stem for i in pair) != images for pair, images in zip(PAIRS, IMAGES)):
        raise ValueError('Fixed TRAIN image names differ')
    annotation_shas = {d:annotation_set_sha256(ROOT/'crane_project/data/crane_grab_port_day2night_v1'/d/'annfiles')
                       for d in ('train', 'train_sim')}
    expected_data = training_identity()
    if annotation_shas != expected_data['annotation_sha256']:
        raise ValueError('TRAIN annotation sets differ from previous reviewed evidence')
    inputs, rows, states = {}, [], {}
    for stage in ('frozen_b', 'initialization'):
        if stage == 'frozen_b':
            model = build_detector(deepcopy(f.model))
            loaded = load_checkpoint(model, identity['checkpoint'], map_location='cpu', strict=True)
            contract = checkpoint_contract(loaded.get('meta', {}), b, 'b')
            del loaded
        else:
            model, digest = initialize_matched_models(b, f)
            contract = dict(b_f_initialization_equal=True, initial_parameter_sha256=digest,
                            procedure='seed0 build_detector init_weights with original ImageNet; no B resume')
        model.cuda(gpu).train()
        before = parameter_digest(model)
        buffers = {k:tensor_sha(v) for k,v in model.named_buffers()}
        torch.cuda.reset_peak_memory_stats(gpu)
        for s_index, scale in enumerate((1., .5)):
            for p_index, (pair, images) in enumerate(zip(PAIRS, IMAGES)):
                seed = 1701+s_index*2+p_index
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                batch = scatter(collate([datasets[scale][i] for i in pair], samples_per_gpu=2), [gpu])[0]
                if tuple(Path(m['filename']).stem for m in batch['img_metas']) != images:
                    raise ValueError('TRAIN loader substituted an image')
                record, key = input_record(batch), (scale, p_index)
                if record['image_sha256'] != [expected_data['image_sha256'][name] for name in images]:
                    raise ValueError('TRAIN image bytes differ from previous reviewed evidence')
                if stage == 'frozen_b':
                    inputs[key] = record
                    if scale == .5:
                        clean = inputs[(1., p_index)]
                        if (not np.allclose(record['original_gt'], clean['original_gt'], rtol=1e-5, atol=1e-3)
                                or not np.allclose(record['scale_factors'], np.asarray(clean['scale_factors'])*.5, rtol=1e-6, atol=1e-7)):
                            raise ValueError('Half-scale GT/common isotropic restoration differs')
                elif record != inputs[key]:
                    raise ValueError('Frozen/init TRAIN inputs/GT/metadata differ')
                row = probe_batch(model, batch, f, stage, scale)
                row.update(record, images=list(images), source_indices=list(pair), seed=seed)
                rows.append(row)
                with Path(progress).open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                print('TRAIN', stage, scale, list(images), 'F-S', row['loss_size'],
                      'FPN size/main', row['gradients']['full_fpn']['size_to_main'],
                      'clip', row['clipping'], flush=True)
                del batch, row
        if before != parameter_digest(model) or buffers != {k:tensor_sha(v) for k,v in model.named_buffers()}:
            raise RuntimeError('Probe changed saved parameters/buffers')
        states[stage] = dict(contract=contract, parameters_and_buffers_unchanged=True,
                             parameter_sha256=before, peak_gpu_bytes=torch.cuda.max_memory_allocated(gpu))
        del model
        torch.cuda.empty_cache()
    if sha(identity['checkpoint']) != identity['checkpoint_sha256']:
        raise RuntimeError('Frozen B checkpoint changed during probe')
    return dict(rows=rows, states=states, inputs_identical_across_stages=True, optimizer_steps=0,
        annotation_sha256=annotation_shas, previous_train_identity_matched=True,
        checkpoint_bytes_unchanged=True, gpu_name=torch.cuda.get_device_name(gpu),
        torch_version=torch.__version__, full_train_audit=False)


def prerequisite(path, sources):
    report = json.loads(Path(path).read_text())
    if (report.get('status') != 'SAVED_BOX_FORMULA_CHECK_COMPLETE_REVIEW_REQUIRED'
            or report.get('candidate_settings') != EXPECTED or report.get('sources') != sources
            or report.get('source_manifest_sha256') != sha(MANIFEST)
            or report['mathematics'].get('fixture_sha256') != sha(FIXTURE)
            or report['mathematics'].get('all_mathematical_checks_passed') is not True
            or report.get('optimizer_steps') != 0):
        raise ValueError('Prerequisite saved-box report is not for this reviewed F-S candidate')
    return dict(path=str(Path(path).resolve()), sha256=sha(path), status=report['status'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-only', action='store_true')
    parser.add_argument('--math-report')
    parser.add_argument('--b-sweep', default=str(ROOT/'work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/val_sweep_port_v1'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    out = Path(args.out_json).resolve()
    artifacts, progress = out.with_suffix('.artifacts.json'), out.with_suffix('.progress.jsonl')
    if any(p.exists() for p in (out, artifacts, progress)):
        raise FileExistsError('Keep existing results; use a new output name')
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='CHECK_STARTED', protocol='port_size_f_s_v1_preflight',
        evidence_role='bounded_train_candidate_integration_check', candidate_settings=EXPECTED,
        formula='B + .1 * weighted_mean(mean(SmoothL1(log sorted edge ratios, beta=.1)))',
        formal_training_authorized=False, optimizer_steps=0, test_repeatedly_exposed=True,
        limitations=['Four TRAIN images, eight batches; no optimization or accuracy claim.',
                     'Frozen B with F-S is counterfactual; initialization is not historical TRAIN reconstruction.',
                     'Direct size correction does not imply composed KLD correction or preserved shared-network coverage.',
                     'Current seed0 initialization identity is not an archived historical initial parameter comparison.',
                     'No TEST tuning, inference, output-center coverage or independent depth evaluation.'])
    try:
        report['sources'] = checked_sources()
        report['source_manifest_sha256'] = sha(MANIFEST)
        report['library_contract'] = library_contract()
        b, f = check_configs()
        report['config_sha256'] = dict(b=sha(CONTROL), f_s=sha(EXPERIMENT))
        if args.config_only:
            report['status'] = 'CONFIG_ONLY_RUNTIME_UNVERIFIED'
        else:
            if not args.math_report:
                raise ValueError('--math-report is required before GPU probes')
            report['prerequisite'] = prerequisite(args.math_report, report['sources'])
            report['identity'] = frozen_identity(args.b_sweep, 'b', CONTROL)
            with artifacts.open('x') as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
            with progress.open('x'):
                pass
            report['train'] = runtime_probe(b, f, report['identity'], args.gpu, progress)
            if checked_sources() != report['sources']:
                raise RuntimeError('Sources changed during probe')
            report['status'] = 'TRAIN_PREFLIGHT_COMPLETE_REVIEW_REQUIRED'
    except Exception as error:
        report.update(status='CHECK_FAILED', error=type(error).__name__+': '+str(error))
        raise
    finally:
        with out.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print('Saved', out, 'status', report['status'], flush=True)


if __name__ == '__main__':
    main()
