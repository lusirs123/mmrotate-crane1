"""Frozen-detector side branches, checkpoint contract, and component evaluation.

No detector, GT, domain, sequence, or probe identity enters the online API.
The sigmoid outputs regress exp(-error/reference), not calibrated probabilities.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import nn

from crane_project.utils.port_structure_reliability_v1 import (
    SETTINGS, StructureComponentReliability, balanced_response_loss,
    canonical_boxes, checked_meta, quality_loss)

VERSION = 'port_reliability_branches_v1'
ARMS = ('geometry', 'roi', 'structure')
COMPONENTS = ('center', 'size', 'angle')
EPOCHS = 8
SEED = 1701
COVERAGES = (1., .99, .98, .95, .90, .80)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def geometry_descriptor(boxes, scores, meta):
    checked_meta(meta)
    b = canonical_boxes(boxes.detach())
    scores = scores.detach().reshape(-1)
    if len(scores) != len(b) or not bool(torch.isfinite(scores).all()) or bool(((scores < 0) | (scores > 1)).any()):
        raise ValueError('Expected one finite score per box')
    ph, pw = meta['pad_shape'][:2]
    return torch.stack((scores, b[:, 0]/pw, b[:, 1]/ph,
        torch.log(b[:, 2]/max(ph, pw)), torch.log(b[:, 3]/max(ph, pw)),
        torch.log(b[:, 2]/b[:, 3]), torch.sin(2*b[:, 4]), torch.cos(2*b[:, 4])), dim=1)


class GeometryQuality(nn.Module):
    """Same eight online box/score features as the ROI heads; no image input."""
    def __init__(self):
        super().__init__()
        self.quality = nn.Sequential(nn.Linear(8, 64), nn.ReLU(inplace=False), nn.Linear(64, 3))

    def forward(self, p3, boxes, scores, meta):
        return dict(qualities=self.quality(geometry_descriptor(boxes, scores, meta)).sigmoid())


def make_arms(device='cpu'):
    seed_all(SEED)
    structure = StructureComponentReliability(True)
    roi = StructureComponentReliability(False)
    roi.load_state_dict(structure.state_dict(), strict=True)
    # A genuine ordinary ROI control: no response input AND no auxiliary loss.
    for p in roi.response.parameters():
        p.requires_grad_(False)
    seed_all(SEED+1)
    geometry = GeometryQuality()
    arms = dict(geometry=geometry.to(device), roi=roi.to(device), structure=structure.to(device))
    seed_all(SEED)
    return arms


def make_optimizers(arms):
    return {name: torch.optim.SGD([p for p in arm.parameters() if p.requires_grad],
        lr=.001, momentum=0., weight_decay=0.) for name, arm in arms.items()}


def architecture(arms):
    return {name: dict(parameters=sum(p.numel() for p in arm.parameters()),
        trainable_parameters=sum(p.numel() for p in arm.parameters() if p.requires_grad),
        structure_input=name == 'structure', structure_loss=name == 'structure')
        for name, arm in arms.items()}


def update_arms(arms, optimizers, p3, boxes, scores, meta, target, mask,
                genuine_count, response_target, valid):
    if p3.requires_grad or p3.grad_fn is not None:
        raise ValueError('Detector computation graph must be absent')
    result = {}
    # Only one head's autograd graph is live at a time; shared P3 is detached.
    for name in ARMS:
        arm, optimizer = arms[name], optimizers[name]
        arm.train()
        optimizer.zero_grad()
        online = arm(p3, boxes, scores, meta)
        qloss, parts = quality_loss(online['qualities'], target, mask, genuine_count)
        sloss = qloss.new_zeros(())
        if name == 'structure':
            sloss, _ = balanced_response_loss(online['response_logits'], response_target, valid)
        loss = qloss+sloss
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite branch loss')
        loss.backward()
        final_gradient = arm.quality[-1].weight.grad
        if final_gradient is None or not bool(torch.isfinite(final_gradient).all()):
            raise ValueError('Missing/nonfinite quality gradient')
        task_norms = final_gradient.double().square().sum(dim=1).sqrt().cpu().tolist()
        response_norms = None
        if name == 'structure':
            gradient = arm.response.weight.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise ValueError('Missing/nonfinite structure gradient')
            response_norms = gradient.double().flatten(1).square().sum(dim=1).sqrt().cpu().tolist()
        if name == 'roi' and any(p.grad is not None for p in arm.response.parameters()):
            raise ValueError('Ordinary ROI response received gradients')
        trainable = [p for p in arm.parameters() if p.requires_grad]
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, 10.))
        if not math.isfinite(norm):
            raise ValueError('Nonfinite branch gradient')
        result[name] = dict(loss_total=float(loss.detach()), loss_quality=float(qloss.detach()),
            loss_structure=float(sloss.detach()), loss_genuine=float(parts['genuine'].detach()),
            loss_probes=float(parts['probe'].detach()), preclip_norm=norm,
            clip_multiplier=min(1., 10./(norm+1e-6)), quality_task_gradient_norms=task_norms,
            response_task_gradient_norms=response_norms,
            genuine_quality_before_update=online['qualities'][:genuine_count].detach().cpu().tolist())
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in arm.parameters()):
            raise ValueError('Nonfinite updated weights')
        del online, loss, qloss, sloss, parts, final_gradient
    return result


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume CUDA RNG/device count differs')
        torch.cuda.set_rng_state_all(state['cuda'])


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_checkpoint(path, arms, optimizers, contract, epoch, steps, b_state, view_chain):
    """Commit a new bundle and SHA marker; never replace an existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or Path(str(path)+'.sha.json').exists():
        raise FileExistsError('Preserve existing checkpoint: '+str(path))
    payload = dict(protocol=VERSION, contract=contract, contract_sha256=fingerprint(contract),
        epoch=epoch, optimizer_steps_per_arm=steps, view_chain_sha256=view_chain,
        arms={k: cpu_tree(v.state_dict()) for k, v in arms.items()},
        optimizers={k: cpu_tree(v.state_dict()) for k, v in optimizers.items()},
        rng=rng_state(), frozen_b_state=b_state)
    fd, temporary = tempfile.mkstemp(prefix='.checkpoint-', dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.link(temporary, str(path))  # exclusive atomic commit on same filesystem
    finally:
        os.unlink(temporary)
    write_new(str(path)+'.sha.json', dict(checkpoint_sha256=sha(path),
        protocol=VERSION, contract_sha256=fingerprint(contract), epoch=epoch,
        optimizer_steps_per_arm=steps, role=contract['role']))
    return sha(path)


def read_checkpoint(path, expected_contract=None, for_val=False):
    path = Path(path)
    marker = json.loads(Path(str(path)+'.sha.json').read_text())
    if marker['checkpoint_sha256'] != sha(path):
        raise ValueError('Checkpoint SHA marker mismatch')
    payload = torch.load(str(path), map_location='cpu')
    contract = payload['contract']
    if (payload['protocol'] != VERSION or marker['protocol'] != VERSION
            or payload['contract_sha256'] != fingerprint(contract)
            or marker['contract_sha256'] != fingerprint(contract)
            or marker['epoch'] != payload['epoch']
            or marker['optimizer_steps_per_arm'] != payload['optimizer_steps_per_arm']
            or marker['role'] != contract['role']
            or set(payload['arms']) != set(ARMS) or set(payload['optimizers']) != set(ARMS)):
        raise ValueError('Checkpoint protocol/contract mismatch')
    if expected_contract is not None and expected_contract != contract:
        raise ValueError('Resume requires identical source/data/protocol/runtime contract')
    if contract['role'] == 'formal_train':
        if not 0 <= payload['epoch'] <= EPOCHS or payload['optimizer_steps_per_arm'] != payload['epoch']*2558:
            raise ValueError('Only complete formal epochs can be resumed')
    elif contract['role'] != 'discarded_smoke':
        raise ValueError('Unrecognized checkpoint role')
    if for_val and (contract['role'] != 'formal_train' or payload['epoch'] != EPOCHS):
        raise ValueError('VAL only accepts fixed formal epoch8, never smoke/selected epochs')
    return payload


def load_bundle(payload, arms, optimizers=None, restore_random=False):
    for name in ARMS:
        arms[name].load_state_dict(payload['arms'][name], strict=True)
        if optimizers is not None:
            optimizers[name].load_state_dict(payload['optimizers'][name])
            for group in optimizers[name].param_groups:
                if (group['lr'], group['momentum'], group['weight_decay']) != (.001, 0., 0.):
                    raise ValueError('Loaded optimizer differs from fixed SGD protocol')
            device = next(arms[name].parameters()).device
            for state in optimizers[name].state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
    if restore_random:
        restore_rng(payload['rng'])


def online_qualities(arms, p3, boxes, scores, meta):
    with torch.no_grad():
        result = {}
        for name in ARMS:
            arms[name].eval()
            result[name] = arms[name](p3, boxes, scores, meta)['qualities'].cpu().tolist()
    return result


def selection_stats(rows, component, accepted):
    """One component at a time; partial outputs are never declared complete OBBs."""
    error_key = dict(center='center_px', size='size_max_relative', angle='angle_deg')[component]
    def good(r):
        value = r['errors'][error_key]
        return value < 15 if component == 'center' else value <= (.10 if component == 'size' else 3.)
    eligible = [r for r in rows if component != 'angle' or r['angle_axis_well_defined']]
    outputs = [r for r in eligible if r['pred'] is not None]
    kept = [r for r in outputs if r['image'] in accepted]
    correct = sum(good(r) for r in kept)
    errors = np.asarray([r['errors'][error_key] for r in kept])
    run = best = 0
    previous = None
    for r in sorted(eligible, key=lambda r: (r['sequence'], r['frame_id'])):
        if previous is None or r['sequence'] != previous['sequence'] or r['frame_id'] != previous['frame_id']+1:
            run = 0
        run = 0 if r['image'] in accepted else run+1
        best = max(best, run)
        previous = r
    return dict(total_frames=len(rows), eligible_frames=len(eligible), output_frames=len(outputs),
        accepted_frames=len(kept), raw_output_coverage=len(outputs)/len(eligible) if eligible else None,
        component_accepted_coverage=len(kept)/len(eligible) if eligible else None,
        correct_accepted=correct, incorrect_accepted=len(kept)-correct,
        correct_rejected=sum(good(r) and r['image'] not in accepted for r in outputs),
        incorrect_rejected=sum(not good(r) and r['image'] not in accepted for r in outputs),
        missing_outputs=len(eligible)-len(outputs),
        correctness_on_accepted=correct/len(kept) if kept else None,
        all_eligible_frame_correct_component_coverage=correct/len(eligible) if eligible else None,
        longest_unaccepted_consecutive_frames=best,
        error_mean=float(errors.mean()) if len(errors) else None,
        error_rmse=float(np.sqrt((errors**2).mean())) if len(errors) else None,
        error_p90=float(np.percentile(errors, 90)) if len(errors) else None,
        error_p95=float(np.percentile(errors, 95)) if len(errors) else None,
        center_valid_accepted=sum(r['errors']['center_px'] < 15 for r in kept),
        joint_correct_accepted_on_angle_eligible=sum(r['angle_axis_well_defined'] and
            r['errors']['center_px'] < 15 and r['errors']['size_max_relative'] <= .10 and
            r['errors']['angle_deg'] <= 3 for r in kept),
        joint_assessed_accepted=sum(r['angle_axis_well_defined'] for r in kept))


def compare_component_rankings(rows, coverages=COVERAGES):
    """GT eligibility is an offline evaluation stratum, never an online filter.

    Keep ceil(coverage * eligible frames), capped at eligible output count.
    All four rankers have identical actual counts in every stratum/component.
    Image names break ties deterministically; no threshold is fitted/exported.
    """
    methods = ('score',)+ARMS
    groups = dict(all=rows)
    for field in ('domain', 'sequence'):
        for value in sorted({r[field] for r in rows}):
            groups[field+':'+value] = [r for r in rows if r[field] == value]
    result = {}
    for group, values in groups.items():
        raw = [r for r in values if r['pred'] is not None]
        raw_hits = sum(r['errors']['center_px'] < 15 for r in raw)
        summary = dict(frames=len(values), output_frames=len(raw), output_coverage=len(raw)/len(values),
            center_hit_rate_on_outputs=raw_hits/len(raw) if raw else None,
            all_frame_center_correct_coverage=raw_hits/len(values), component_curves={})
        for index, component in enumerate(COMPONENTS):
            eligible = [r for r in values if component != 'angle' or r['angle_axis_well_defined']]
            out = [r for r in eligible if r['pred'] is not None]
            curves = []
            for coverage in coverages:
                count = min(len(out), int(math.ceil(coverage*len(eligible))))
                comparison = {}
                for method in methods:
                    def value(r):
                        return r['pred'][5] if method == 'score' else r['qualities'][method][index]
                    ordered = sorted(out, key=lambda r: (-value(r), r['image']))
                    accepted = {r['image'] for r in ordered[:count]}
                    stats = selection_stats(values, component, accepted)
                    cutoff = value(ordered[count-1]) if count else None
                    stats.update(cutoff_score_descriptive_only=cutoff,
                        total_at_cutoff=sum(value(r) == cutoff for r in ordered) if count else 0,
                        accepted_at_cutoff=sum(value(r) == cutoff for r in ordered[:count]) if count else 0)
                    comparison[method] = stats
                increments = {reference: dict(
                    incorrect_accepted_structure_minus_reference=comparison['structure']['incorrect_accepted']-comparison[reference]['incorrect_accepted'],
                    correct_rejected_structure_minus_reference=comparison['structure']['correct_rejected']-comparison[reference]['correct_rejected'],
                    center_valid_accepted_structure_minus_reference=comparison['structure']['center_valid_accepted']-comparison[reference]['center_valid_accepted'])
                    for reference in ('score', 'geometry', 'roi')}
                curves.append(dict(requested_eligible_frame_coverage=coverage,
                    matched_actual_accept_count=count, methods=comparison,
                    structure_increments=increments))
            summary['component_curves'][component] = curves
        result[group] = summary
    return result
