#!/usr/bin/env python3
"""Two bounded checks on completed ep08 reliability heads. No fit or TEST.

review: CPU-only full genuine-output logs / locked VAL associations.
probe: at most eight predeclared TRAIN views, fixed heads, response input
sensitivity / annotation evidence / shared-stem loss gradients, zero updates.
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
from crane_project.tools import eval_port_reliability_branches_v1_cached_val as previous
from crane_project.utils import port_reliability_mechanism_v1 as mechanism
from crane_project.utils.port_structure_reliability_v1 import (
    assert_detector_frozen, map_boxes, map_points)

branch, train, ready, prior = previous.branch, previous.train, previous.ready, previous.prior
VERSION = 'port_reliability_mechanism_v1'
PROTOCOL = ROOT/'crane_project/tools/port_reliability_mechanism_v1_protocol.json'
MANIFEST = ROOT/'crane_project/tools/port_reliability_mechanism_v1_sources.json'
DESCRIPTORS = ('B_score', 'cx_over_pad', 'cy_over_pad', 'log_long_over_pad',
               'log_short_over_pad', 'log_aspect', 'sin_2theta', 'cos_2theta')


def assert_same_statistics(actual, expected):
    """Only permit roundoff in recomputing saved statistics, not changed counts."""
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or actual.keys() != expected.keys():
            raise ValueError('Cached statistic keys differ')
        for key in actual:
            assert_same_statistics(actual[key], expected[key])
    elif isinstance(actual, list):
        if not isinstance(expected, list) or len(actual) != len(expected):
            raise ValueError('Cached statistic lists differ')
        for a, b in zip(actual, expected):
            assert_same_statistics(a, b)
    elif isinstance(actual, float):
        if not isinstance(expected, (float, int)) or not np.isfinite([actual, expected]).all() or abs(actual-expected) > 1e-9:
            raise ValueError('Cached floating statistic differs')
    elif actual != expected:
        raise ValueError('Cached counts/roles differ')


def checked_sources():
    training_sources, training_protocol, previous_sources = previous.checked_sources()
    manifest = json.loads(MANIFEST.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    current = {p: branch.sha(ROOT/p) for p in manifest['sources']}
    if (manifest['protocol'] != VERSION or manifest['sources'] != current
            or manifest['protocol_sha256'] != branch.sha(PROTOCOL)
            or manifest['parent_cached_val_manifest_sha256'] != previous_sources['manifest_sha256']
            or protocol['protocol'] != VERSION or protocol['train_epoch'] != 8
            or protocol['max_gpu_views'] != 8 or protocol['optimizer_steps'] != 0
            or protocol['test_read'] is not False):
        raise ValueError('Reviewed mechanism source/protocol contract differs')
    return dict(manifest_sha256=branch.sha(MANIFEST), sources=current,
        training_sources=training_sources, previous_cached_val_sources=previous_sources), protocol, training_protocol


def evidence_paths(args):
    return {name: folder/name for folder, names in (
        (args.train_dir, ('completion.json', 'contract.json', 'train_steps.jsonl')),
        (args.val_dir, ('val_compare.json', 'val_qualities.jsonl', 'parity_diagnosis.json')))
        for name in names}


def checked_evidence(args, sources, protocol, training_protocol):
    paths = evidence_paths(args)
    proof = {name: branch.sha(path) for name, path in paths.items()}
    if proof != protocol['evidence_sha256']:
        raise ValueError('Requires the exact reviewed completed TRAIN / cached VAL artifacts')
    completion, contract, val, parity = (json.loads(paths[n].read_text()) for n in
        ('completion.json', 'contract.json', 'val_compare.json', 'parity_diagnosis.json'))
    logs = [json.loads(l) for l in paths['train_steps.jsonl'].open()]
    rows = [json.loads(l) for l in paths['val_qualities.jsonl'].open()]
    if (completion['status'] != 'PAIRED_FORMAL_TRAIN_COMPLETE_FIXED_EPOCH8'
            or completion['completed_epoch'] != 8 or completion['optimizer_steps_per_arm'] != 20464
            or completion['resume_from'] is not None or contract['role'] != 'formal_train'
            or len(logs) != 20464 or len(rows) != 887
            or completion['detector_optimizer_steps'] != 0
            or not completion['save_reload_quality_exact'] or not completion['b_raw_unchanged']
            or completion['val_or_test_dataset_read'] or completion['val_or_test_metrics_used_for_training']
            or val['evidence_role'] != 'component_quality_conditional_on_locked_native_B_outputs_source_VAL_only'
            or val['fixed_epoch'] != 8 or val['test_read']
            or completion['train_log_sha256'] != proof['train_steps.jsonl']
            or val['rows_sha256'] != proof['val_qualities.jsonl']
            or val['parity_diagnosis_sha256'] != proof['parity_diagnosis.json']
            or completion['checkpoint_sha256'] != val['branch_checkpoint_sha256']
            or completion['checkpoint_sha256'] != protocol['branch_checkpoint_sha256']
            or parity['branch_checkpoint_sha256'] != completion['checkpoint_sha256']):
        raise ValueError('Completed epoch8 / locked VAL evidence differs')
    for current in (completion['sources'], contract['sources'], val['training_sources'], parity['training_sources']):
        if current != sources['training_sources']:
            raise ValueError('Completed training source identity differs')
    for current in (completion['protocol'], contract['protocol'], val['training_protocol_unchanged']):
        if current != training_protocol:
            raise ValueError('Completed training protocol differs')
    if (val['repair_sources'] != sources['previous_cached_val_sources']
            or parity['repair_sources'] != val['repair_sources']
            or contract['frozen_b'] != ready.FROZEN_B
            or completion['b_state'] != contract['frozen_b_state']
            or completion['b_state'] != val['frozen_b_state']
            or completion['b_state'] != parity['frozen_b_state']):
        raise ValueError('Frozen B / cached VAL source identity differs')
    if branch.sha(args.input_snapshot) != train.SNAPSHOT_SHA:
        raise ValueError('Requires unchanged original numeric TRAIN snapshot')
    inputs = json.loads(args.input_snapshot.read_text())['inputs']
    if branch.fingerprint(inputs) != contract['train_input_manifest_sha256']:
        raise ValueError('TRAIN input snapshot differs from formal contract')
    chain = branch.fingerprint(dict(seed=branch.SEED))
    permutations = {ep: np.random.RandomState(branch.SEED+ep).permutation(2558) for ep in range(1, 9)}
    for step, r in enumerate(logs, 1):
        ep, slot = (step-1)//2558+1, (step-1) % 2558
        index = int(permutations[ep][slot]); source = inputs[index]
        if ((r['epoch'], r['slot'], r['optimizer_step'], r['dataset_index']) != (ep, slot, step, index)
                or r['view_seed'] != branch.SEED+100000*ep+slot
                or r['image'] != source['image'] or r['input_sha256'] != branch.fingerprint(source)
                or r['gt_original'] != source['gt_original'] or r['probes'] != 13):
            raise ValueError('Original paired TRAIN view stream differs')
        chain = branch.fingerprint(dict(previous=chain, view={k: r[k] for k in
            ('epoch', 'slot', 'image', 'view_seed', 'view_image_sha256', 'genuine_b_original')}))
        if r['paired_view_chain_sha256'] != chain:
            raise ValueError('Paired TRAIN stream chain differs')
    if chain != completion['paired_view_chain_sha256']:
        raise ValueError('Completed TRAIN stream end differs')
    assert_same_statistics(branch.compare_component_rankings(rows), val['strata'])
    return logs, rows, inputs, proof, completion


def choose_cases(logs):
    """Selection uses ONLY fixed TRAIN ep08 errors, never VAL/TEST quality."""
    cases = {}
    final = [r for r in logs if r['epoch'] == 8 and r['genuine_count']]
    for domain in ('real', 'sim'):
        rows = [r for r in final if r['domain'] == domain]
        if not rows:
            raise ValueError('No genuine TRAIN support for '+domain)
        for name, component in (('first_dataset_index', None), ('max_center_error', 0),
                                ('max_size_error', 1), ('max_angle_error', 2)):
            row = (min(rows, key=lambda r: (r['dataset_index'], r['image'])) if component is None
                   else min(rows, key=lambda r: (-r['genuine_errors_original'][0][component], r['image'])))
            key = row['optimizer_step']
            if key not in cases:
                cases[key] = dict(image=row['image'], domain=domain, sequence=row['sequence'],
                    dataset_index=row['dataset_index'], slot=row['slot'], view_seed=row['view_seed'],
                    optimizer_step=key, logged_view_image_sha256=row['view_image_sha256'],
                    selection_roles=[], logged_errors_original=row['genuine_errors_original'])
            cases[key]['selection_roles'].append(name)
    return sorted(cases.values(), key=lambda r: (r['domain'], r['slot']))


def descriptor(box, score):
    x, y, w, h, theta = map(float, box)
    if w < h:
        w, h, theta = h, w, theta+np.pi/2
    return [score, x/1024., y/1024., np.log(w/1024.), np.log(h/1024.),
            np.log(w/h), np.sin(2*theta), np.cos(2*theta)]


def log_rows(records, inputs):
    result = []
    for r in records:
        source = inputs[r['dataset_index']]
        pred = r['genuine_b_original'][0] if r['genuine_count'] else None
        meta = dict(scale_factor=r['scale_factor'], img_shape=r['img_shape'], pad_shape=r['pad_shape'],
            ori_shape=tuple(source['image_size'][::-1])+ (3,), flip=r['flip'], flip_direction=r['flip_direction'])
        model = map_boxes(torch.tensor([pred[:5]], dtype=torch.float32), meta)[0].tolist() if pred else None
        result.append(dict(image=r['image'], sequence=r['sequence'], domain=r['domain'], pred=pred,
            errors=r['genuine_errors_original'][0] if pred else None,
            qualities={arm: r['arms'][arm]['genuine_quality_before_update'][0] if pred else None for arm in branch.ARMS},
            angle_axis_well_defined=r['qualification']['evaluation_angle_eligible'],
            train_angle_eligible=bool(r['train_angle_mask']),
            descriptor=descriptor(model, pred[5]) if pred else None))
    return result


def profiles(rows, train_role):
    groups = dict(all=rows)
    for field in ('domain', 'sequence'):
        for value in sorted({r[field] for r in rows}):
            groups[field+':'+value] = [r for r in rows if r[field] == value]
    result = {}
    for name, group in groups.items():
        outputs = [r for r in group if r['pred'] is not None]
        errors = np.asarray([r['errors'] for r in outputs], dtype=float).reshape(-1, 3)
        d = np.asarray([r['descriptor'] for r in outputs], dtype=float).reshape(-1, 8)
        by_method = {}
        for arm in ('score',)+branch.ARMS:
            q = np.asarray([[r['pred'][5]]*3 if arm == 'score' else r['qualities'][arm] for r in outputs]).reshape(-1, 3)
            comp = {}
            for component in mechanism.COMPONENTS:
                eligible = None if component != 'angle' else [
                    r['train_angle_eligible'] if train_role else r['angle_axis_well_defined'] for r in outputs]
                comp[component] = mechanism.quality_profile(errors, q, component, eligible,
                    {n: d[:, i] for i, n in enumerate(DESCRIPTORS)})
                if arm == 'score':
                    comp[component]['quality_target_mae'] = None
                    comp[component]['quality_target_rmse'] = None
                    comp[component]['note'] = 'B classification score reference: no component regression or calibration claim.'
            by_method[arm] = comp
        correct = sum(r['errors'][0] < 15 for r in outputs)
        result[name] = dict(frames=len(group), output_frames=len(outputs),
            output_coverage=len(outputs)/len(group),
            center_hit_rate_on_outputs=correct/len(outputs) if outputs else None,
            all_frame_center_correct_coverage=correct/len(group), methods=by_method,
            angle_role='TRAIN native real axis / Webots OBB qualification' if train_role else 'unchanged GT aspect >=1.2 evaluation')
    return result


def cpu_review(args, logs, rows, inputs, proof, completion, sources, protocol):
    train_results = {}
    for epoch in (1, 8):
        selected = [r for r in logs if r['epoch'] == epoch]
        train_results['epoch_'+str(epoch)] = profiles(log_rows(selected, inputs), True)
    val_rows = [dict(r, errors=[r['errors'][k] for k in ('center_px', 'size_max_relative', 'angle_deg')]
                    if r['pred'] is not None else None,
                    descriptor=descriptor(r['cached_b_model'][0], r['pred'][5]) if r['pred'] is not None else None)
                for r in rows]
    probe_loss = {}
    for epoch in range(1, 9):
        selected = [r for r in logs if r['epoch'] == epoch]
        probe_loss[str(epoch)] = {arm: {k: float(np.mean([r['arms'][arm][k] for r in selected]))
            for k in ('loss_quality', 'loss_genuine', 'loss_probes', 'loss_structure')}
            for arm in branch.ARMS}
    report = dict(protocol=VERSION, status='CPU_GENUINE_SUPPORT_REVIEW_COMPLETE', sources=sources,
        protocol_contract=protocol, evidence_sha256=proof,
        branch_checkpoint_sha256=completion['checkpoint_sha256'],
        gpu_cases=choose_cases(logs), train_genuine=train_results, cached_val_genuine=profiles(val_rows, False),
        train_genuine_probe_loss_means=probe_loss, training_stream_role='logged_online_pre_update_scores_not_fixed_epoch8_replay',
        probe_quality_arrays_in_original_log=False,
        detector_updates=0, head_updates=0, model_inference=False, test_read=False, test_repeatedly_exposed=True,
        limits=['No optimizer/model run; epoch stream includes successive weights and random views, not independent validation.',
            '13 probe predictions were not individually logged; only their aggregate loss is available here.',
            'GT/sequence/domain only define offline diagnostic strata; no GT/identity enters an online head.',
            'AUROC/rank correlations are descriptive, not calibration or significance; zero-class strata remain undefined.',
            'VAL remains conditioned on locked native B outputs; TEST is not read.',
            'Descriptor correlations and train/VAL error-support differences do not prove a shortcut or a root cause.'])
    if len(report['gpu_cases']) > protocol['max_gpu_views']:
        raise ValueError('GPU case budget exceeded')
    if (checked_sources()[:2] != (sources, protocol)
            or {n: branch.sha(p) for n, p in evidence_paths(args).items()} != proof
            or branch.sha(args.input_snapshot) != train.SNAPSHOT_SHA):
        raise ValueError('Inputs/sources changed during CPU review')
    args.out_dir.mkdir(parents=True, exist_ok=False)
    branch.write_new(args.out_dir/'review.json', report)
    print('Saved', args.out_dir/'review.json', report['status'], 'GPU views fixed', len(report['gpu_cases']), flush=True)
    return report


def validate_review(path, logs, proof, sources, protocol):
    review = json.loads(path.read_text())
    cases = choose_cases(logs)
    if (review['status'] != 'CPU_GENUINE_SUPPORT_REVIEW_COMPLETE' or review['sources'] != sources
            or review['evidence_sha256'] != proof or review['protocol_contract'] != protocol
            or review['gpu_cases'] != cases or not 0 < len(cases) <= 8):
        raise ValueError('Requires unchanged completed CPU review and predeclared TRAIN cases')
    return review


def prepare_review(args, logs, rows, inputs, proof, completion, sources, protocol):
    # A local review is optional: reconstruct it from verified server evidence.
    requested = args.review_report
    if requested is not None and requested.exists():
        validate_review(requested, logs, proof, sources, protocol)
        return
    review_dir = args.out_dir.with_name(args.out_dir.name + '_review')
    report_path = review_dir/'review.json'
    if requested is not None:
        print('Review report missing:', requested, flush=True)
    if review_dir.exists():
        if not report_path.is_file():
            raise FileExistsError('Preserve incomplete review directory; choose a new --out-dir: ' + str(review_dir))
        validate_review(report_path, logs, proof, sources, protocol)
        print('Reusing verified CPU review:', report_path, flush=True)
    else:
        review_args = deepcopy(args)
        review_args.out_dir = review_dir
        cpu_review(review_args, logs, rows, inputs, proof, completion, sources, protocol)
        validate_review(report_path, logs, proof, sources, protocol)
    args.review_report = report_path


def gpu_probe(args, logs, inputs, proof, completion, sources, protocol, training_protocol):
    validate_review(args.review_report, logs, proof, sources, protocol)
    cases = choose_cases(logs)
    path = args.train_dir/'epoch_08.pth'
    payload = previous.base.fixed_bundle(path, sources['training_sources'], training_protocol)
    if branch.sha(path) != protocol['branch_checkpoint_sha256']:
        raise ValueError('Requires the exact completed epoch08; no other weights')
    cfg = prior.check_cfg()
    _, input_proof = train.checked_inputs(args.input_snapshot, args.train_cache, args.structure_report)
    detector, runtime = train.build_runtime(cfg, args.b_checkpoint, args.gpu)
    frozen = prior.state_digest(detector)
    if frozen != completion['b_state'] or frozen != input_proof['b_cache_state'] or frozen != payload['frozen_b_state']:
        raise ValueError('Frozen B source differs')
    arms = branch.make_arms('cuda:'+str(args.gpu))
    if branch.architecture(arms) != payload['contract']['architecture']:
        raise ValueError('Quality architecture differs')
    branch.load_bundle(payload, arms)
    del payload
    for arm in arms.values():
        arm.eval()
    head_states = {n: prior.state_digest(a) for n, a in arms.items()}
    indexed = train.datasets_for(deepcopy(cfg.data.train), inputs)  # TRAIN only
    args.out_dir.mkdir(parents=True, exist_ok=False)
    previews = args.out_dir/'previews'; previews.mkdir()
    torch.cuda.reset_peak_memory_stats(args.gpu)
    selected = {r['optimizer_step']: r for r in logs if r['epoch'] == 8}
    records = []
    with (args.out_dir/'cases.jsonl').open('x') as stream:
        for case in cases:
            old = selected[case['optimizer_step']]
            source = inputs[case['dataset_index']]
            values, replay, raw_state = train.training_view(detector, indexed[case['dataset_index']],
                                                          source, args.gpu, case['view_seed'])
            p3, boxes, scores, meta, target, mask, genuine_count, response_target, valid = values
            if (replay['view_image_sha256'] != case['logged_view_image_sha256']
                    or replay['input_sha256'] != old['input_sha256']
                    or any(replay[k] != old[k] for k in ('scale_factor', 'img_shape', 'pad_shape', 'flip', 'flip_direction'))):
                raise ValueError('Predeclared augmented TRAIN view did not replay exactly: '+case['image'])
            q = branch.online_qualities(arms, p3, boxes, scores, meta)
            zero = mechanism.zero_response_qualities(arms['structure'], p3, boxes, scores, meta)
            with torch.no_grad():
                response = arms['structure'](p3, boxes, scores, meta)['response_logits'].sigmoid().cpu().numpy()[0]
            axis = map_points(p3.new_tensor(train.axis_for(source)), meta).cpu().numpy()
            evidence = mechanism.response_evidence(response, valid.cpu().numpy()[0, 0], axis,
                                                   boxes[:genuine_count].cpu().numpy())
            gradients = mechanism.structure_gradients(arms['structure'], *values)
            # Only for visualization: replay the same dataset seed, no second B forward.
            image, image_meta = replay_image(indexed[case['dataset_index']], args.gpu, case['view_seed'])
            if prior.tensor_sha(image) != replay['view_image_sha256']:
                raise ValueError('Preview image differs from the diagnostic TRAIN view')
            preview_path = previews/(case['image']+'.png')
            mechanism.preview(preview_path, image, image_meta, response, axis, boxes[:genuine_count].cpu().numpy())
            targets, errors = target.cpu().tolist(), replay['genuine_errors_original']
            qq, zz = np.asarray(q['structure']), np.asarray(zero)
            record = dict(case=case, replay=replay,
                logged_vs_runtime_B=previous.prediction_difference(replay['genuine_b_original'], old['genuine_b_original']),
                qualities=q, structure_response_input_zero_qualities=zero, continuous_targets=targets,
                quality_masks=mask.cpu().tolist(),
                probe_names=probe_names(source), genuine_count=genuine_count,
                genuine_errors_original=errors,
                response_zero_abs_quality_delta_by_component=np.abs(qq-zz).mean(0).tolist(),
                genuine_response_zero_abs_quality_delta_by_component=np.abs(qq[:genuine_count]-zz[:genuine_count]).mean(0).tolist() if genuine_count else None,
                genuine_target_mae_by_method={n: np.abs(np.asarray(z)[:genuine_count]-np.asarray(targets)[:genuine_count]).mean(0).tolist() if genuine_count else None for n, z in dict(q, structure_input_zero=zero).items()},
                probe_target_mae_by_method={n: np.abs(np.asarray(z)[genuine_count:]-np.asarray(targets)[genuine_count:]).mean(0).tolist() for n, z in dict(q, structure_input_zero=zero).items()},
                response_evidence=evidence, gradients=gradients,
                structure_source=source['direction_qualification']['structure_source'],
                preview=str(preview_path.relative_to(args.out_dir)), preview_sha256=branch.sha(preview_path),
                p3_sha256=prior.tensor_sha(p3), detector_updates=0, head_updates=0)
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False)+'\n')
            stream.flush()
            assert_detector_frozen(detector)
            if any(p.grad is not None for a in arms.values() for p in a.parameters()):
                raise ValueError('Diagnostic accumulated head gradients')
            print('TRAIN fixed-head probe', len(records), '/', len(cases), case['image'],
                  'quality/structure stem', gradients['shared_stem_quality_vs_structure'], flush=True)
            del values, raw_state, p3, boxes, scores, target, mask, response_target, valid, image
    if (prior.state_digest(detector) != frozen
            or head_states != {n: prior.state_digest(a) for n, a in arms.items()}
            or not arms['structure'].use_structure
            or checked_sources()[:2] != (sources, protocol)
            or {n: branch.sha(p) for n, p in evidence_paths(args).items()} != proof
            or branch.sha(args.input_snapshot) != train.SNAPSHOT_SHA
            or branch.sha(path) != protocol['branch_checkpoint_sha256']):
        raise ValueError('Fixed sources, checkpoint, or model state changed during probe')
    report = dict(protocol=VERSION, status='FIXED_EPOCH8_TRAIN_MECHANISM_PROBE_COMPLETE_REVIEW_REQUIRED',
        sources=sources, protocol_contract=protocol, evidence_sha256=proof,
        review_sha256=branch.sha(args.review_report), branch_checkpoint_sha256=branch.sha(path),
        runtime=runtime, cases=len(records), cases_sha256=branch.sha(args.out_dir/'cases.jsonl'),
        frozen_b_state=frozen, heads_before=head_states, heads_after=head_states,
        detector_updates=0, head_updates=0, optimizer_created=False, test_read=False, test_repeatedly_exposed=True,
        max_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20,
        max_reserved_mib=torch.cuda.max_memory_reserved(args.gpu)/2**20,
        limits=['Only predeclared selected TRAIN diagnostic views, not a performance estimate or independent validation.',
            'Native real axes are annotation reference, not visibility or pixel evidence guarantees; sim lines are Webots OBB proxies.',
            'Zero response input is distribution-changing sensitivity, not an equal-budget retrained ablation.',
            'PCA response orientation is offline evidence only, never a new online angle or quality gate.',
            'Local gradient ratios/cosines do not identify a unique cause or justify a new loss coefficient.',
            'B and trained heads receive no updates; all existing output, transform, depth and evaluation rules persist.',
            'No TEST or new VAL model inference.'])
    branch.write_new(args.out_dir/'probe.json', report)
    print('Saved', args.out_dir/'probe.json', report['status'], flush=True)


def probe_names(source):
    return train.component_probes_original(torch.tensor(source['gt_original'], dtype=torch.float32))[1]


def replay_image(indexed, gpu, view_seed):
    from mmcv.parallel import collate, scatter
    branch.seed_all(view_seed)
    part, index = indexed
    batch = scatter(collate([part[index]], samples_per_gpu=1), [gpu])[0]
    return batch['img'], batch['img_metas'][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('review', 'probe'), default='review')
    parser.add_argument('--train-dir', type=Path, default=Path('work_dirs/port_reliability_branches_v1_train'))
    parser.add_argument('--val-dir', type=Path, default=Path('work_dirs/port_reliability_branches_v1_val_cached_v1'))
    parser.add_argument('--input-snapshot', type=Path, default=Path('work_dirs/port_reliability_train_support_v1/train_input_snapshot.json'))
    parser.add_argument('--train-cache', type=Path, default=Path('work_dirs/port_reliability_train_support_v1_cache'))
    parser.add_argument('--structure-report', type=Path, default=Path('work_dirs/port_reliability_readiness_v1_structure_complete/train_structure_check.json'))
    parser.add_argument('--b-checkpoint', type=Path, default=prior.CHECKPOINT)
    parser.add_argument('--review-report', type=Path,
        help='Existing CPU review; if omitted or missing, build <out-dir>_review/review.json from verified evidence')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    import os
    os.chdir(ROOT)
    if args.out_dir.exists():
        raise FileExistsError('Preserve existing evidence; choose a new --out-dir')
    sources, protocol, training_protocol = checked_sources()
    logs, rows, inputs, proof, completion = checked_evidence(args, sources, protocol, training_protocol)
    if args.mode == 'review':
        cpu_review(args, logs, rows, inputs, proof, completion, sources, protocol)
    else:
        prepare_review(args, logs, rows, inputs, proof, completion, sources, protocol)
        gpu_probe(args, logs, inputs, proof, completion, sources, protocol, training_protocol)


if __name__ == '__main__':
    main()
