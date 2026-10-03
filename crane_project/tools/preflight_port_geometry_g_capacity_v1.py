#!/usr/bin/env python3
"""Fixed TRAIN capacity check: saved FC64 reference vs new matched FC16.

Only FC16 is updated (200 fixed batches). Reuse the exact CPU ROI cache and
reviewed wide result. No B extraction, donor shuffle, fitted checkpoint,
VAL/TEST access, step/seed search or automatic formal-training approval.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import preflight_port_geometry_g_roi_ablation_v1 as a
from crane_project.utils.port_geometry_refine_g_capacity_v1 import (
    CompactLocalGeometryRefiner, HIDDEN_FC, PARAMETER_COUNT, compact_initial_state)

g = a.g
VERSION = 'port_geometry_g_capacity_v1_train_preflight'
STEPS_PER_ARM = 200
PROTOCOL = ROOT/'crane_project/tools/port_geometry_g_capacity_v1_protocol.json'
MANIFEST = ROOT/'crane_project/tools/port_geometry_g_capacity_v1_sources.json'
A_MANIFEST_SHA = '0250eafb9feef5fff88a4072dca02b1a0cb71f345f6fa1e0b866c2d8a2ada170'


def protocol_document(samples):
    return dict(protocol=VERSION, evidence_role='TRAIN_only_single_FC_capacity_intervention',
        reference_g_settings=deepcopy(g.SETTINGS), changed_setting=dict(hidden_fc=HIDDEN_FC),
        conditions=['fc64_saved_reference', 'fc16_matched'],
        parameter_counts=dict(fc64=183907, fc16=PARAMETER_COUNT),
        baseline_report_sha256=a.BASELINE_SHA, cache_manifest_sha256=a.CACHE_MANIFEST_SHA,
        prior_ablation_manifest_sha256=A_MANIFEST_SHA,
        fixed_sample_counts={role: dict(Counter(s['domain'] for s in samples if s['role'] == role)) for role in ('fit','probe')},
        intervention='Only hidden FC width64 to16. Same 9x9 spatial flatten, stem, recipient box descriptor and three bounded residuals.',
        initialization='Generate seed1703 UNTRAINED FC64; require its full initial digest to match saved reference. Copy identical stem and first16 hidden units including biases; zero final layer. No learned pruning or data-selected units.',
        sampling='Use existing matched ordinary ROI/support only; no new feature extraction, aligned run or donor permutation.',
        optimizer='Same Adam lr.001 weight_decay0 clip10 and original three-component normalized SmoothL1 loss.',
        runtime_policy='Same python/torch/cuda/cudnn/numpy/opencv version strings as the reviewed reference; physical GPU number may differ.',
        scope=dict(detector_forward_calls=0, detector_updates=0, additional_reference_updates=0,
                   new_head_updates_total=STEPS_PER_ARM, formal_training=False, checkpoint_export=False,
                   val_access=False, test_access=False),
        evaluation='Initial exact B and fixed final step200; same fit/probe/domain/scale edge MAE/p90, pure angle RMSE/p90, RIoU, tails and three coverage denominators. Report FC16 minus B and FC16 minus saved FC64. No step, width or seed selection.',
        review='Completion is not approval. Preserve original joint geometry conditions; lower fit loss or beating FC64 alone cannot replace joint improvement over B.',
        limitations=['One fixed capacity point16 is a hypothesis, not a validated best width or proof of overfitting.',
                     'TRAIN probe has16 image identities and32 correlated views; B trained on all, no independent VAL.',
                     'Fixed centers/output decisions cannot improve localization distance or recover missed frames.',
                     'Component loss and overlap under fixed centers remain coupled; no continuity or depth guarantee.',
                     'TEST repeatedly exposed previously; no TEST access or tuning in this check.'])


def checked_contract(baseline_path):
    manifest, protocol = [json.loads(p.read_text()) for p in (MANIFEST, PROTOCOL)]
    if (manifest.get('protocol') != VERSION or manifest.get('reference_g_settings') != g.SETTINGS or
            manifest.get('changed_setting') != dict(hidden_fc=16)):
        raise ValueError('Capacity source/settings contract differs')
    actual = {p:g.ready.sha(ROOT/p) for p in manifest['sources']}
    if actual != manifest['sources']:
        raise ValueError('Reviewed capacity source SHA differs')
    if g.ready.sha(a.MANIFEST) != A_MANIFEST_SHA:
        raise ValueError('Requires unchanged reviewed ROI ablation provenance')
    # Reuse its source/TRAIN/baseline verification, not its donor adapter or fit.
    _, samples, prior_identity, baseline = a.checked_contract(baseline_path)
    if protocol != protocol_document(samples):
        raise ValueError('Predeclared single-capacity protocol differs')
    identity = dict(manifest_sha256=g.ready.sha(MANIFEST), protocol_sha256=g.ready.sha(PROTOCOL),
        sources=actual, prior_identity=prior_identity, baseline_report_sha256=a.BASELINE_SHA,
        cache_manifest_sha256=a.CACHE_MANIFEST_SHA)
    return protocol, samples, identity, baseline


def checked_initialization(reference):
    g.seed_all()
    wide = g.LocalGeometryRefiner()
    wide_digest = g.state_digest(wide)
    if wide_digest != reference['initial_state']:
        raise ValueError('Seed1703 untrained FC64 initialization differs from reference')
    initial = compact_initial_state(wide.state_dict())
    small = CompactLocalGeometryRefiner(); small.load_state_dict(initial, strict=True)
    small_digest = g.state_digest(small)
    if small_digest['parameter_count'] != PARAMETER_COUNT or small_digest['buffer_count'] != 0:
        raise ValueError('Fixed FC16 architecture differs')
    stem_equal = all(torch.equal(small.state_dict()[k],v) for k,v in wide.state_dict().items() if k.startswith('stem.'))
    if not stem_equal:
        raise ValueError('Common stem initialization differs')
    evidence = dict(fc64_initial_state=wide_digest, fc16_initial_state=small_digest,
                    common_stem_exact=True, predetermined_first16_units=True, output_zero=True)
    del wide, small
    return initial, evidence


def fit_compact_arm(records, batches, initial, arm, device, progress):
    # Same update/evaluation body as g.fit_arm; only head construction changes.
    head = CompactLocalGeometryRefiner().to(device); head.load_state_dict(initial, strict=True)
    before = g.state_digest(head); start = g.evaluate(head, records, arm, device)
    if any(not np.array_equal(np.asarray(row['pred']), r['boxes_original'][0].numpy())
           for row, r in zip(start['rows'], records) if row['pred'] is not None):
        raise ValueError('Zero-initialized G is not exactly B')
    optimizer = torch.optim.Adam(head.parameters(), lr=g.SETTINGS['lr'], weight_decay=0.)
    logs = []
    head.train()
    for step, indices in enumerate(batches, 1):
        roi, support, b, bm, gt = g.batch_tensors(records, indices, arm, device)
        optimizer.zero_grad(); output = head(roi, support, b, bm)
        loss, parts = g.regression_loss(output['boxes_original'], gt)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite G loss')
        tasks = None
        if step == 1:
            tasks = [float(torch.autograd.grad(p, head.output.weight, retain_graph=True)[0].double().norm()) for p in parts]
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in head.parameters()):
            raise ValueError('Missing/nonfinite head gradient')
        stem_norm = sum(float(p.grad.double().square().sum()) for p in head.stem.parameters())**.5
        norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), g.SETTINGS['clip_norm']))
        if not math.isfinite(norm):
            raise ValueError('Nonfinite G clipping norm')
        optimizer.step()
        progress(dict(stage='update', arm=arm, step=step))
        if any(not bool(torch.isfinite(p).all()) for p in head.parameters()):
            raise ValueError('Nonfinite updated head')
        if step in (1, 2) or step % 25 == 0:
            log = dict(stage='fit', arm=arm, step=step, loss=float(loss.detach()),
                       parts=parts.detach().cpu().tolist(), grad_norm_before_clip=norm,
                       clip_multiplier=min(1., g.SETTINGS['clip_norm']/(norm+1e-6)), stem_grad_norm=stem_norm,
                       final_layer_task_grad_norms=tasks)
            logs.append(log); progress(log)
    final = g.evaluate(head, records, arm, device)
    result = dict(initial_state=before, final_state=g.state_digest(head), initial=start, final=final, logs=logs,
        finite_gradients=True, initial_all_task_gradients_effective=all(x > 0 for x in logs[0]['final_layer_task_grad_norms']),
        stem_gradient_after_zero_output_step_effective=logs[1]['stem_grad_norm'] > 0,
        fit_loss_decreased_by_group={k: final['eligible_loss'][k] < v-1e-6 for k, v in start['eligible_loss'].items() if k.startswith('fit/')})
    del head, optimizer; torch.cuda.empty_cache()
    return result


def publish_artifacts(out_dir, report):
    files = sorted(p for p in out_dir.rglob('*') if p.is_file() and p.name != 'artifacts.json')
    g.write_new(out_dir/'artifacts.json', dict(protocol=VERSION, status=report['status'],
        files={str(p.relative_to(out_dir)):g.ready.sha(p) for p in files}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-only', action='store_true', help='Source/TRAIN/baseline contract only; no cache load, GPU or updates.')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--baseline-report', default=str(a.BASELINE))
    ap.add_argument('--cache-dir', default='work_dirs/port_geometry_g_v1_roi_cache')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args(); os.chdir(ROOT)
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    report = dict(protocol=VERSION, evidence_role='TRAIN_only_single_FC_capacity_intervention',
        status='STARTED', formal_training=False, detector_forward_calls=0, detector_updates=0,
        additional_reference_updates=0, head_updates_total=0, checkpoint_exported=False, gpu_devices_used=0)
    with (out/'progress.jsonl').open('x') as stream:
        def progress(value):
            if value['stage'] == 'update':
                report['head_updates_total'] += 1
                return
            stream.write(json.dumps(g.json_native(value), ensure_ascii=False, allow_nan=False)+'\n'); stream.flush()
            print('TRAIN fc16 step', value['step'], 'loss', value['loss'], flush=True)
        try:
            protocol, samples, identity, baseline = checked_contract(Path(args.baseline_report).resolve())
            report.update(identity=identity, reference_g_settings=g.SETTINGS,
                changed_setting=dict(hidden_fc=16), parameter_counts=protocol['parameter_counts'],
                fixed_sample_counts=protocol['fixed_sample_counts'], limitations=protocol['limitations'])
            g.write_new(out/'protocol.json', protocol); shutil.copyfile(str(MANIFEST), str(out/'sources.json'))
            print('G capacity check: saved FC64 updates=0; matched FC16 updates='+str(STEPS_PER_ARM)+'; B inference/updates=0. Not formal training.', flush=True)
            if args.check_only:
                report['status'] = 'STATIC_CAPACITY_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES'
            else:
                if int(os.environ.get('WORLD_SIZE','1')) != 1:
                    raise ValueError('Single-process finite check; do not use torchrun/DDP')
                report['runtime'] = a.checked_runtime(baseline)
                cache_dir = Path(args.cache_dir).resolve()
                if g.ready.sha(cache_dir/'cache_manifest.json') != a.CACHE_MANIFEST_SHA:
                    raise ValueError('Requires the exact reviewed complete ROI cache')
                payload, cache = g.checked_cache(cache_dir, identity['prior_identity']['g_identity'], samples)
                records = payload['records']; g.validate_records(records, samples)
                a.check_cache_reference(records, cache, baseline)
                if any(len(r['boxes_original']) != 1 or not r['eligible'] for r in records):
                    raise ValueError('Expected all128 reviewed eligible B outputs')
                g.write_new(out/'cache_manifest.json', cache)
                report.update(cache=cache, cache_reused=True, detector_state_unchanged=True,
                    original_cpu_roi_bytes=g.cpu_roi_bytes(records),
                    train_support=g.support_report(records,json.loads(g.PROTOCOL.read_text())))
                if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
                    raise ValueError('A valid logical CUDA device is required')
                torch.cuda.set_device(args.gpu); device = torch.device('cuda',args.gpu)
                report['gpu_devices_used'] = 1; report['runtime']['gpu'] = torch.cuda.get_device_name(args.gpu)
                initial, initialization = checked_initialization(baseline['arms']['ordinary'])
                batches = g.schedule(records)
                if len(batches) != STEPS_PER_ARM or batches != baseline['minibatches']:
                    raise ValueError('Fixed200 reference batches differ')
                report.update(initialization=initialization, minibatches=batches,
                    reference_fc64=baseline['arms']['ordinary'], reference_origin='reviewed saved ordinary; no new FC64 updates')
                print('Reusing reviewed CPU ordinary ROIs. One logical GPU:',args.gpu,'; only FC16 is fitted.',flush=True)
                result, cost = g.measured(lambda:fit_compact_arm(records,batches,initial,'ordinary',device,progress),args.gpu)
                report.update(compact_fc16=result, refiner_cost=cost)
                g.write_new(out/'compact_fc16.json',dict(protocol=VERSION,identity=identity,
                    completed_head_updates=STEPS_PER_ARM,formal_training=False,checkpoint_exported=False,result=result,cost=cost))
                if (result['initial'] != report['reference_fc64']['initial'] or
                        result['initial_state'] != initialization['fc16_initial_state']):
                    raise ValueError('FC16 initial B or initialization digest differs')
                if report['head_updates_total'] != STEPS_PER_ARM:
                    raise ValueError('New update budget differs')
                report.update(fc16_minus_b=g.geometry_deltas(result['initial'],result['final']),
                    fc16_minus_fc64=g.geometry_deltas(report['reference_fc64']['final'],result['final']),
                    status='TRAIN_CAPACITY_CHECK_COMPLETE_REVIEW_REQUIRED')
            g.write_new(out/'completion.json',report)
        except Exception as error:
            report['status']='FAILED_REVIEW_REQUIRED'; report['error']=type(error).__name__+': '+str(error)
            destination=out/('failure.json' if (out/'completion.json').exists() else 'completion.json')
            try:
                g.write_new(destination,report)
            except (TypeError,ValueError) as serialization_error:
                g.write_new(destination,dict(protocol=VERSION,status=report['status'],formal_training=False,
                    detector_updates=0,detector_forward_calls=0,additional_reference_updates=0,
                    head_updates_total=report['head_updates_total'],error=report['error'],
                    serialization_error=str(serialization_error),full_report_available=False))
            stream.flush(); publish_artifacts(out,report)
            raise
    publish_artifacts(out,report)
    print('Saved',out/'completion.json','status',report['status'],flush=True)


if __name__=='__main__':
    main()
