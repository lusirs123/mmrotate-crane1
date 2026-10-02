#!/usr/bin/env python3
"""Fixed TRAIN ROI correspondence ablation: matched vs shuffled ordinary ROIs.

Reuse the reviewed detached G cache, preserve recipient geometry/GT, and run
200 identical head batches per condition. No B inference, formal training,
checkpoint export, VAL/TEST reads or step/seed search.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import preflight_port_geometry_g_v1 as g

VERSION = 'port_geometry_g_roi_ablation_v1_train_preflight'
CONDITIONS = ('matched', 'shuffled')
PERMUTATION_SEED = 1704
BASELINE_SHA = 'ce38c0f399027de2a72bad8ff94fa5691cb1aa0d1e66048c8ef32d025487bc71'
CACHE_MANIFEST_SHA = '7dae321ea2c0fb3f60eb4eb20cee9ebfb8f404d5c0780308485a24ed16f5108d'
G_MANIFEST_SHA = '7840ab1c8d68d327d0ac6793d59b4e6cc0340d846087499f2d350692155dc200'
PROTOCOL = ROOT/'crane_project/tools/port_geometry_g_roi_ablation_v1_protocol.json'
MANIFEST = ROOT/'crane_project/tools/port_geometry_g_roi_ablation_v1_sources.json'
BASELINE = ROOT/'work_dirs/port_geometry_g_v1_train_preflight_report_fix_v1.json'


def donor_map(samples):
    """Predeclared image-level derangement; no GT/features used to choose donors."""
    if len({s['image'] for s in samples}) != len(samples):
        raise ValueError('Duplicate recipient image')
    rng = np.random.RandomState(PERMUTATION_SEED)
    result = {}
    for role in ('fit', 'probe'):
        for domain in ('real', 'sim'):
            names = [s['image'] for s in samples if s['role'] == role and s['domain'] == domain]
            if len(names) < 2:
                raise ValueError('Each donor stratum needs at least two images')
            for _ in range(1000):
                order = rng.permutation(len(names))
                if bool((order != np.arange(len(names))).all()):
                    break
            else:
                raise ValueError('Unable to produce the fixed derangement')
            result.update({name: names[int(j)] for name, j in zip(names, order)})
    if set(result) != {s['image'] for s in samples}:
        raise ValueError('Undeclared role/domain')
    return result


def protocol_document(samples):
    return dict(protocol=VERSION, evidence_role='TRAIN_only_ROI_correspondence_ablation',
        g_settings=deepcopy(g.SETTINGS), conditions=list(CONDITIONS), permutation_seed=PERMUTATION_SEED,
        donor_mapping=donor_map(samples),
        g_manifest_sha256=G_MANIFEST_SHA, baseline_report_sha256=BASELINE_SHA,
        cache_manifest_sha256=CACHE_MANIFEST_SHA,
        sampling='Both conditions use the existing ordinary ROI; no aligned sampling or new B extraction.',
        intervention='Move donor ordinary ROI and support together. Retain recipient original/model B box, score, GT and metadata.',
        donor_policy='Fixed bijection without self-donors in each fit/probe x real/sim stratum; same donor image at both scales; never cross roles/domains/scales.',
        replay_policy='Matched result must exactly reproduce the reviewed ordinary result after log condition relabeling; otherwise stop before shuffled fitting.',
        runtime_policy='GPU fitting requires identical python/torch/cuda/cudnn/numpy/opencv version strings to the reviewed run.',
        scope=dict(detector_updates=0, detector_forward_calls=0,
                   head_updates_total=g.SETTINGS['steps_per_arm']*len(CONDITIONS),
                   formal_training=False, checkpoint_export=False, val_access=False, test_access=False),
        evaluation='Same initial B and fixed final step200; all fit/probe/domain/scale groups, pure angle RMSE/p90, edge errors, RIoU and all three center/output denominators. No step or seed selection.',
        review='Completion is not formal-training approval. Compare matched/shuffled probe and each against B; preserve original joint geometry conditions.',
        limitations=['TRAIN probe contains 16 image identities with two correlated views, B trained on all; not independent VAL.',
                     'Single within-stratum derangement preserves domain/scale distributions and may retain similar same-video object content.',
                     'This tests the whole ROI/support correspondence; it does not separately isolate appearance, context or geometric ROI extent.',
                     'No gain cannot prove that image information is absent; a relative gain over shuffled cannot replace joint improvement over B.',
                     'Centers are copied from B; no center correction, video continuity or independent depth validation.',
                     'TEST repeatedly exposed previously; no TEST access or tuning here.'])


def checked_contract(baseline_path):
    manifest = json.loads(MANIFEST.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    if manifest.get('protocol') != VERSION or manifest.get('g_settings') != g.SETTINGS:
        raise ValueError('Ablation source/settings contract differs')
    actual = {p: g.ready.sha(ROOT/p) for p in manifest['sources']}
    if actual != manifest['sources']:
        raise ValueError('Reviewed ablation source SHA differs')
    if g.ready.sha(g.MANIFEST) != G_MANIFEST_SHA:
        raise ValueError('Requires the unchanged reviewed G report-fix sources')
    cfg, _, samples, g_identity = g.checked_inputs()
    for scale in g.SETTINGS['scales']:
        g.fixed_specs(cfg, scale)
    if protocol != protocol_document(samples):
        raise ValueError('Predeclared ablation/donor protocol differs')
    if g.ready.sha(baseline_path) != BASELINE_SHA:
        raise ValueError('Requires the exact reviewed completed TRAIN baseline report')
    baseline = json.loads(baseline_path.read_text())
    if (baseline['identity'] != g_identity or baseline['settings'] != g.SETTINGS or
            baseline['status'] != 'TRAIN_SHORT_FIT_COMPLETE_REVIEW_REQUIRED' or
            baseline['head_updates_total'] != 400 or baseline['detector_updates'] != 0 or
            baseline['formal_training'] or baseline['checkpoint_exported'] or
            not baseline['detector_state_unchanged']):
        raise ValueError('Reviewed baseline scope/identity differs')
    encoded = json.dumps(baseline['cache'], ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if hashlib.sha256(encoded.encode()).hexdigest() != CACHE_MANIFEST_SHA:
        raise ValueError('Baseline cache manifest content differs')
    g.accepted_cache_identity(baseline['cache']['identity'], g_identity)
    identity = dict(manifest_sha256=g.ready.sha(MANIFEST), protocol_sha256=g.ready.sha(PROTOCOL),
        sources=actual, g_identity=g_identity, baseline_report_sha256=BASELINE_SHA,
        cache_manifest_sha256=CACHE_MANIFEST_SHA)
    return protocol, samples, identity, baseline


def assemble_records(records, samples, mapping):
    """Shallow immutable views: new condition keys reference existing CPU tensors."""
    if mapping != donor_map(samples):
        raise ValueError('Donor substitution differs from the fixed mapping')
    lookup = {(r['image'], r['scale']): r for r in records}
    expected = [(s['image'], scale, s['role']) for scale in g.SETTINGS['scales'] for s in samples]
    if len(lookup) != len(records) or [(r['image'], r['scale'], r['role']) for r in records] != expected:
        raise ValueError('Ablation cache view identity/order differs')
    prepared, assignments = [], []
    for r in records:
        donor = lookup[(mapping[r['image']], r['scale'])]
        if (donor['role'] != r['role'] or donor['domain'] != r['domain'] or donor['image'] == r['image']):
            raise ValueError('Invalid donor role/domain/self identity')
        # This exact reviewed cache has 128 genuine, center-eligible outputs.
        # Stop on substitutions/missing views rather than alter supervision.
        if len(r['boxes_original']) != 1 or not r['eligible']:
            raise ValueError('Expected the reviewed 128 genuine eligible B outputs')
        current = dict(r, local=dict(r['local']))
        current['local']['matched'] = r['local']['ordinary']
        current['local']['shuffled'] = donor['local']['ordinary']
        prepared.append(current)
        assignments.append(dict(recipient=r['image'], donor=donor['image'], role=r['role'],
                                domain=r['domain'], scale=r['scale']))
    return prepared, assignments


def check_cache_reference(records, cache, baseline):
    if cache != baseline['cache']:
        raise ValueError('Actual cache differs from the reviewed baseline cache')
    reference = baseline['arms']['ordinary']['initial']['rows']
    if len(records) != len(reference):
        raise ValueError('Baseline/cache row count differs')
    for row, old in zip(records, reference):
        if (any(row[k] != old[k] for k in ('image', 'role', 'domain', 'sequence', 'scale', 'frame_id', 'eligible')) or
                row['boxes_original'].tolist() != [old['pred']] or
                row['gt_original'].tolist() != [old['gt_original']] or
                row['parsed_gt_original'] != old['parsed_gt_original'] or
                row['reference_gt_absolute_delta'] != old['reference_gt_absolute_delta']):
            raise ValueError('Cached recipient B/GT differs from reviewed baseline')


def verify_matched_result(result, baseline):
    replay = deepcopy(result)
    for log in replay['logs']:
        log['arm'] = 'ordinary'
    if replay != baseline['arms']['ordinary']:
        raise ValueError('Matched replay differs from reviewed ordinary; stop comparison')


def checked_runtime(baseline):
    runtime = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), numpy=np.__version__, opencv=g.cv2.__version__,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if any(runtime[k] != baseline['runtime'][k] for k in runtime if k != 'cuda_visible_devices'):
        raise ValueError('Fitting runtime differs from the reviewed replay environment')
    return runtime


def copy_previews(cache_dir, cache, out_dir):
    previews = [p for p in cache['files'] if p.startswith('previews/')]
    if len(previews) != 12 or any(Path(p).suffix != '.png' for p in previews):
        raise ValueError('Expected the original twelve spatial support PNGs')
    (out_dir/'previews').mkdir()
    for name in previews:
        target = out_dir/name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write((cache_dir/name).read_bytes())
        if g.ready.sha(target) != cache['files'][name]:
            raise ValueError('Copied preview identity differs')
    g.write_new(out_dir/'cache_manifest.json', cache)


def publish_artifacts(out_dir, report):
    paths = sorted(p for p in out_dir.rglob('*') if p.is_file() and p.name != 'artifacts.json')
    g.write_new(out_dir/'artifacts.json', dict(protocol=VERSION, status=report['status'],
        files={str(p.relative_to(out_dir)): g.ready.sha(p) for p in paths}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-only', action='store_true', help='Validate source/TRAIN/baseline/donor contract; no feature cache load or GPU.')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--baseline-report', default=str(BASELINE))
    ap.add_argument('--cache-dir', default='work_dirs/port_geometry_g_v1_roi_cache')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args(); os.chdir(ROOT)
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    cache_dir = Path(args.cache_dir).resolve()
    report = dict(protocol=VERSION, evidence_role='TRAIN_only_ROI_correspondence_ablation',
        status='STARTED', formal_training=False, detector_updates=0, detector_forward_calls=0,
        head_updates_total=0, checkpoint_exported=False, gpu_devices_used=0)
    with (out/'progress.jsonl').open('x') as stream:
        def progress(value):
            if value['stage'] == 'update':
                report['head_updates_total'] += 1
                return
            stream.write(json.dumps(g.json_native(value), ensure_ascii=False, allow_nan=False)+'\n'); stream.flush()
            if value['stage'] == 'fit':
                print('TRAIN', value['arm'], 'step', value['step'], 'loss', value['loss'], flush=True)
        try:
            protocol, samples, identity, baseline = checked_contract(Path(args.baseline_report).resolve())
            report.update(identity=identity, g_settings=g.SETTINGS, donor_mapping=protocol['donor_mapping'],
                fixed_sample_counts={role: dict(Counter(s['domain'] for s in samples if s['role'] == role)) for role in ('fit', 'probe')},
                limitations=protocol['limitations'])
            # These are frozen contract copies, not input files mutated in place.
            g.write_new(out/'protocol.json', protocol)
            shutil.copyfile(str(MANIFEST), str(out/'sources.json'))
            print('G ROI information check: matched/shuffled ordinary; 200 updates each; B inference/updates=0.', flush=True)
            if args.check_only:
                report['status'] = 'STATIC_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES'
            else:
                if int(os.environ.get('WORLD_SIZE', '1')) != 1:
                    raise ValueError('Single-process finite check; do not launch with torchrun/DDP')
                report['runtime'] = checked_runtime(baseline)
                if g.ready.sha(cache_dir/'cache_manifest.json') != CACHE_MANIFEST_SHA:
                    raise ValueError('Requires the exact reviewed complete feature cache')
                payload, cache = g.checked_cache(cache_dir, identity['g_identity'], samples)
                records = payload['records']; g.validate_records(records, samples)
                check_cache_reference(records, cache, baseline)
                prepared, assignments = assemble_records(records, samples, protocol['donor_mapping'])
                report.update(cache=cache, cache_reused=True, detector_state_unchanged=True,
                              original_cpu_roi_bytes=g.cpu_roi_bytes(records), train_support=g.support_report(records, json.loads(g.PROTOCOL.read_text())))
                g.write_new(out/'donor_assignment.json', dict(protocol=VERSION, assignments=assignments))
                copy_previews(cache_dir, cache, out)
                if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
                    raise ValueError('A valid logical CUDA device is required')
                torch.cuda.set_device(args.gpu); device = torch.device('cuda', args.gpu)
                report['gpu_devices_used'] = 1
                report['runtime']['gpu'] = torch.cuda.get_device_name(args.gpu)
                print('Reusing reviewed CPU ROI cache; no B extraction. One logical GPU:', args.gpu, flush=True)
                g.seed_all(); initial = g.LocalGeometryRefiner().state_dict()
                batches = g.schedule(prepared)
                if len(batches) != g.SETTINGS['steps_per_arm'] or batches != baseline['minibatches']:
                    raise ValueError('Balanced batch schedule differs from reviewed replay')
                report.update(minibatches=batches, arms={}, refiner_costs={})
                for condition in CONDITIONS:
                    result, cost = g.measured(lambda: g.fit_arm(prepared, batches, initial, condition, device, progress), args.gpu)
                    report['arms'][condition] = result; report['refiner_costs'][condition] = cost
                    g.write_new(out/(condition+'.json'), dict(protocol=VERSION, identity=identity,
                        condition=condition, completed_head_updates=len(batches), formal_training=False,
                        checkpoint_exported=False, result=result, cost=cost))
                    if condition == 'matched':
                        verify_matched_result(result, baseline)
                        report['matched_reference_replay'] = 'EXACT_REPLAY_PASS'
                if report['arms']['matched']['initial'] != report['arms']['shuffled']['initial']:
                    raise ValueError('Zero-residual condition baselines differ')
                if report['head_updates_total'] != len(CONDITIONS)*g.SETTINGS['steps_per_arm']:
                    raise ValueError('Condition update budget differs')
                report['final_geometry_deltas'] = {c: g.geometry_deltas(report['arms'][c]['initial'], report['arms'][c]['final']) for c in CONDITIONS}
                report['matched_minus_shuffled_final'] = g.geometry_deltas(report['arms']['shuffled']['final'], report['arms']['matched']['final'])
                report['status'] = 'TRAIN_ROI_ABLATION_COMPLETE_REVIEW_REQUIRED'
            g.write_new(out/'completion.json', report)
        except Exception as error:
            report['status'] = 'FAILED_REVIEW_REQUIRED'; report['error'] = type(error).__name__+': '+str(error)
            destination = out/('failure.json' if (out/'completion.json').exists() else 'completion.json')
            try:
                g.write_new(destination, report)
            except (TypeError, ValueError) as serialization_error:
                g.write_new(destination, dict(protocol=VERSION, status=report['status'],
                    formal_training=False, detector_updates=0, detector_forward_calls=0,
                    head_updates_total=report['head_updates_total'], error=report['error'],
                    serialization_error=str(serialization_error), full_report_available=False))
            stream.flush(); publish_artifacts(out, report)
            raise
    publish_artifacts(out, report)
    print('Saved', out/'completion.json', 'status', report['status'], flush=True)


if __name__ == '__main__':
    main()
