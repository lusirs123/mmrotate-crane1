"""One explicitly authorized TEST of the original VAL-selected R4 epoch16.

No training, checkpoint scan, score calibration, new threshold, or adoption.
The native test entry point and its preprocessing/output contract are retained.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'crane_project/tools'))
BASE = ROOT / 'work_dirs/symeood_110_reg_tower_20261030_v1'
OUT = BASE / 'fixed_TEST_20261030_v1'
REFERENCE = ROOT / 'work_dirs/smallpaper_readiness_20261028_v1/saved_predictions_unified.rows.jsonl'
REFERENCE_SHA = '1abae6afe6ece14c356c7b5608a5ceb692316172605d970444c41a2a12fa8481'
CHECKPOINT_SHA = 'b5ee405108970552717b3b7c8f0e56c6813232e19aebccff82a5c7183511b13b'
ANNOTATIONS_SHA = 'e0dbb1bd8aea7209314d8ed60bc44e965550ed606135cc0016e1075d717de13e'
COUNTS = dict(real_seq03=200, real_seq04=668, sim_seq09=572)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def selected(sweep):
    if sweep['selected_checkpoint'] != 'epoch_16':
        raise ValueError('Only the original source-VAL selected epoch16 is authorized')
    cp = sweep['all_checkpoints']['epoch_16']
    if cp['checkpoint_sha256'] != CHECKPOINT_SHA:
        raise ValueError('Frozen R4 checkpoint identity differs')
    return cp


def native_arguments(config, checkpoint):
    return ['tools/test.py', str(config), str(checkpoint), '--work-dir', str(OUT / 'native'),
            '--out', str(OUT / 'preds/results.pkl'),
            '--result-identity-out', str(OUT / 'runtime_result_identity.json'),
            '--runtime-audit-out', str(OUT / 'runtime_audit.json')]


def annotation_sha(directory):
    digest = hashlib.sha256()
    for path in sorted(Path(directory).glob('*.txt')):
        if path.name.startswith('._'):
            continue
        digest.update(path.name.encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runner():
    spec = importlib.util.spec_from_file_location('sealed_r4_runner', ROOT / 'crane_project/tools/run_symeood_110_reg_tower_v1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    from crane_project.tools.ckpt_sweep import dataset_paths, get_val_img_ids
    control = runner()
    c = control.integrity(BASE)
    cp = selected(read(BASE / 'R4/ckpt_sweep/sweep_results.json'))
    assert sha(cp['checkpoint']) == CHECKPOINT_SHA
    assert read(BASE / 'training_verification.json')['passed']
    assert read(BASE / 'target_VAL/summary.json')['passed']
    assert sha(REFERENCE) == REFERENCE_SHA
    rows = [json.loads(line) for line in REFERENCE.read_text().splitlines()]
    ann, images = dataset_paths(c['config'], 'test')
    assert annotation_sha(ann) == ANNOTATIONS_SHA
    assert len(rows) == 1440 and Counter(r['sequence'] for r in rows) == Counter(COUNTS)
    assert all(r['split'] == 'test' for r in rows)
    assert get_val_img_ids(ann) == [r['image'] for r in rows]
    OUT.mkdir(exist_ok=False)
    (OUT / 'preds').mkdir()
    save(OUT / 'protocol.json', dict(protocol='symeood_r4_original_VAL_ep16_fixed_exposed_TEST_v1',
        frozen_UTC=datetime.now(timezone.utc).isoformat(),
        user_authorized=True, checkpoint=cp['checkpoint'], checkpoint_sha256=CHECKPOINT_SHA,
        selected_epoch=16, selection_source=str(BASE / 'R4/ckpt_sweep/sweep_results.json'),
        selection_source_sha256=sha(BASE / 'R4/ckpt_sweep/sweep_results.json'),
        source_VAL_target_selection=read(BASE / 'target_VAL/task_target_selection.json')['selected'],
        source_VAL_all_guards_passed=False, TEST_is_not_a_replacement_for_failed_VAL_guards=True,
        config=c['config'], config_sha256=c['config_sha256'],
        evaluator_source_sha256=sha(__file__), training_protocol_sha256=sha(BASE / 'protocol.json'),
        all_training_sources_verified=len(c['sources']),
        sources={name: sha(ROOT / name) for name in (
            'tools/test.py', 'crane_project/tools/ckpt_sweep.py',
            'crane_project/tools/eval_crane_offline.py', 'crane_project/tools/eval_port_geometry_ap_v1.py',
            'crane_project/utils/port_geometry_midpoint_edge_residual_metrics_v1.py')},
        annotations=ann, annotation_sha256=ANNOTATIONS_SHA, images=images,
        reference_rows=str(REFERENCE), reference_rows_sha256=REFERENCE_SHA,
        counts=COUNTS, frames=1440, physical_gpu=0, allocator_limit_mib=3072,
        inherited_head_constructor_allocator_limit_mib=6144,
        memory_policy='Apply 3072 MiB before native entry and again after model construction, before any inference forward; OOM stops without retry.',
        inference='native tools/test.py, same config; state/gradient guards surround native single_gpu_test',
        native_arguments=native_arguments(c['config'], cp['checkpoint']),
        TRAIN_access=False, new_VAL_inference=False, checkpoint_scan=False, score_changes=False,
        training_updates=0, TEST_repeatedly_exposed=True, unbiased_holdout_claim=False,
        automatic_adoption=False, downstream_training=False, retry=False))
    print('FIXED_TEST_PREPARED', flush=True)


def checked():
    p = read(OUT / 'protocol.json')
    runner().integrity(BASE)
    assert selected(read(p['selection_source']))['checkpoint'] == p['checkpoint']
    assert sha(p['selection_source']) == p['selection_source_sha256']
    assert sha(p['checkpoint']) == CHECKPOINT_SHA == p['checkpoint_sha256']
    assert sha(p['config']) == p['config_sha256']
    assert sha(__file__) == p['evaluator_source_sha256']
    assert sha(BASE / 'protocol.json') == p['training_protocol_sha256']
    assert sha(REFERENCE) == REFERENCE_SHA
    assert annotation_sha(p['annotations']) == ANNOTATIONS_SHA
    for name, digest in p['sources'].items():
        assert sha(ROOT / name) == digest, name
    return p


def infer():
    p = checked()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert not (OUT / 'preds/results.pkl').exists(), 'Never overwrite or repeat a completed TEST'
    import torch
    import mmdet.apis
    original = mmdet.apis.single_gpu_test
    calls = []

    def enforce_inference_memory_limit():
        limit = p['allocator_limit_mib'] * 2**20
        total = torch.cuda.get_device_properties(0).total_memory
        assert torch.cuda.memory_allocated(0) < limit
        torch.cuda.set_per_process_memory_fraction(limit / total, 0)

    enforce_inference_memory_limit()

    def digest(model):
        h = hashlib.sha256()
        for name, value in model.state_dict().items():
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def guarded(model, loader, *args, **kwargs):
        # The inherited head constructor sets its original 6 GiB ceiling.
        # Tighten it again after construction and before the first forward.
        enforce_inference_memory_limit()
        assert len(loader.dataset) == 1440 and len(loader) == 1440
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        before = digest(model)
        counts = dict(forward_calls=0)

        def audit_forward(module, inputs):
            assert not torch.is_grad_enabled() and not module.training
            counts['forward_calls'] += 1

        handle = model.register_forward_pre_hook(audit_forward)
        started = time.perf_counter()
        try:
            outputs = original(model, loader, *args, **kwargs)
        finally:
            handle.remove()
        after = digest(model)
        assert before == after and len(outputs) == counts['forward_calls'] == 1440
        save(OUT / 'frozen_model_state.json', dict(passed=True, before=before, after=after,
            parameters_and_buffers_unchanged=True, gradients_disabled_all_forwards=True,
            training_updates=0, optimizer_construction_forbidden=True,
            forward_calls=counts['forward_calls'], frames=1440,
            inference_allocator_limit_mib=p['allocator_limit_mib'],
            elapsed_seconds=time.perf_counter() - started,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20))
        calls.append(True)
        return outputs

    def forbidden_optimizer(*args, **kwargs):
        raise RuntimeError('No optimizer may be constructed during frozen TEST')

    torch.optim.Optimizer.__init__ = forbidden_optimizer
    mmdet.apis.single_gpu_test = guarded
    sys.argv = p['native_arguments']
    runpy.run_path(str(ROOT / 'tools/test.py'), run_name='__main__')
    assert calls == [True]
    from crane_project.tools.ckpt_sweep import check_or_record_prediction
    check_or_record_prediction(p['config'], p['checkpoint'], str(OUT / 'preds/results.pkl'),
                               p['annotations'], 'fixed_test', generated=True)
    assert sha(p['checkpoint']) == CHECKPOINT_SHA
    print('FIXED_TEST_INFERENCE_COMPLETE', flush=True)


def longest_failure(rows):
    current = maximum = 0
    previous = None
    for row in sorted(rows, key=lambda r: (r['sequence'], r['frame_id'])):
        if previous is None or row['sequence'] != previous['sequence'] or row['frame_id'] != previous['frame_id'] + 1:
            current = 0
        current = current + 1 if row['riou'] < .5 else 0
        maximum = max(maximum, current)
        previous = row
    return maximum


def analyze():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    p = checked()
    import numpy as np
    from crane_project.tools.eval_crane_offline import compute_riou
    from crane_project.tools.eval_port_geometry_ap_v1 import evaluate_images, validate_saved_gt
    from crane_project.tools.analyze_symeood_ablation_first4_v1 import txtbox
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import canonical, row_metrics
    raw_path = OUT / 'preds/results.pkl'
    provenance = read(str(raw_path) + '.provenance.json')
    assert provenance['checkpoint_sha256'] == CHECKPOINT_SHA
    assert provenance['annotations_sha256'] == ANNOTATIONS_SHA
    assert provenance['results_pkl_sha256'] == sha(raw_path)
    identity = read(OUT / 'runtime_result_identity.json')
    assert identity['results_sha256'] == sha(raw_path) and identity['checkpoint_sha256'] == CHECKPOINT_SHA
    refs = [json.loads(line) for line in REFERENCE.read_text().splitlines()]
    assert [r['frame_key'] for r in identity['runtime_dataset_order']] == [r['image'] for r in refs]
    with raw_path.open('rb') as stream:
        predictions = pickle.load(stream)
    assert len(predictions) == 1440 and all(len(r[0]) <= 1 for r in predictions)
    metrics, per_frame, cache = [], [], {}
    out = OUT / 'analysis'
    out.mkdir(exist_ok=False)
    for ref in refs:
        ann = Path(p['annotations']) / (ref['image'] + '.txt')
        poly = np.asarray([float(v) for v in ann.read_text().split()[:8]]).reshape(4, 2)
        validate_saved_gt(ref['gt'], poly, txtbox(ann))
    for side, field in [('S', 'b'), ('E', 'eood_aug'), ('R4', None)]:
        rows = []
        for ref, native in zip(refs, predictions):
            pred = (native[0][0].tolist() if len(native[0]) else None) if field is None else ref[field]
            value = {k: ref[k] for k in ('image', 'sequence', 'domain', 'frame_id', 'gt')}
            value.update(side=side, prediction=pred, score=pred[5] if pred is not None else 0.)
            riou = compute_riou(np.asarray(canonical(pred)), np.asarray(canonical(ref['gt']))) if pred is not None else 0.
            value.update(row_metrics(ref['gt'], pred, riou))
            rows.append(value)
        cache[side] = rows
        per_frame.extend(rows)
        for domain in ('all', 'real', 'sim', *COUNTS):
            group = [r for r in rows if domain == 'all' or domain in (r['domain'], r['sequence'])]
            valid = [r for r in group if r['output']]
            sim = [r for r in valid if r['domain'] == 'sim']
            value = dict(side=side, domain=domain, frames=len(group), outputs=len(valid),
                TP50=sum(r['riou'] >= .5 for r in group), TP75=sum(r['riou'] >= .75 for r in group),
                Recall_pct=100 * sum(r['riou'] >= .5 for r in group) / len(group),
                center_correct=sum(r['center_hit'] for r in group),
                center_MAE_px=sum(r['center_error_px'] for r in valid) / len(valid),
                center_median_px=float(np.median([r['center_error_px'] for r in valid])),
                center_P95_px=float(np.percentile([r['center_error_px'] for r in valid], 95)),
                mean_RIoU=sum(r['riou'] for r in group) / len(group),
                long_MAE_pct=100 * sum(r['long_relative_error'] for r in valid) / len(valid),
                short_MAE_pct=100 * sum(r['short_relative_error'] for r in valid) / len(valid),
                joint10=sum(r['joint_size10'] for r in group),
                joint10_pct=100 * sum(r['joint_size10'] for r in group) / len(group),
                Sim_angle_RMSE_deg=(math.sqrt(sum(r['angle_error_deg']**2 for r in sim) / len(sim)) if sim else None),
                MCML_max_native=longest_failure(group))
            dets = [[canonical(r['prediction']) + [r['score']]] if r['output'] else [] for r in group]
            gt = [[canonical(r['gt'])] for r in group]
            for threshold in (.5, .75):
                result = evaluate_images(dets, gt, threshold)
                assert result['tp'] == value['TP50' if threshold == .5 else 'TP75']
                value['AP' + str(round(threshold * 100))] = result['ap_percent']
            metrics.append(value)
    paired = []
    for side in ('S', 'E'):
        for domain in ('all', 'real', 'sim', *COUNTS):
            pairs = [(a, b) for a, b in zip(cache[side], cache['R4']) if domain == 'all' or domain in (a['domain'], a['sequence'])]
            common = [(a, b) for a, b in pairs if a['output'] and b['output']]
            paired.append(dict(reference=side, domain=domain, frames=len(pairs), common_outputs=len(common),
                recovered50=sum(a['riou'] < .5 <= b['riou'] for a, b in pairs),
                lost50=sum(b['riou'] < .5 <= a['riou'] for a, b in pairs),
                recovered75=sum(a['riou'] < .75 <= b['riou'] for a, b in pairs),
                lost75=sum(b['riou'] < .75 <= a['riou'] for a, b in pairs),
                recovered_size10=sum(not a['joint_size10'] and b['joint_size10'] for a, b in pairs),
                lost_size10=sum(a['joint_size10'] and not b['joint_size10'] for a, b in pairs),
                common_center_MAE_before=sum(a['center_error_px'] for a, b in common) / len(common),
                common_center_MAE_after=sum(b['center_error_px'] for a, b in common) / len(common)))
    save(out / 'summary.json', dict(passed=True, TEST=True, exposed_TEST=True, training_updates=0,
        selected_epoch=16, checkpoint_sha256=CHECKPOINT_SHA, results_pkl_sha256=sha(raw_path),
        GT_source_sha256=REFERENCE_SHA, annotation_sha256=ANNOTATIONS_SHA,
        AP='Top1 rotated VOC07 eleven-point, existing evaluator', metrics=metrics, paired=paired,
        source_VAL_guards_passed=False, automatic_adoption=False, reselection=False,
        note='Descriptive frozen exposed TEST; no checkpoint scan or TEST-driven tuning. Sim-only pure angle.'))
    with (out / 'metrics.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (out / 'per_frame.jsonl').open('x') as stream:
        for row in per_frame:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    assert read(OUT / 'frozen_model_state.json')['passed']
    save(OUT / 'completion.json', dict(status='FIXED_R4_EP16_TEST_COMPLETE',
        completed_UTC=datetime.now(timezone.utc).isoformat(), frames=1440, training_updates=0,
        frozen_model_state_unchanged=True, runtime_frame_order_verified=True,
        selected_epoch=16, checkpoint_sha256=CHECKPOINT_SHA, results_pkl_sha256=sha(raw_path),
        automatic_adoption=False, repeated_exposed_TEST=True, no_new_training=True))
    print('FIXED_R4_EP16_TEST_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('prepare', 'infer', 'analyze'))
    args = parser.parse_args()
    globals()[args.stage]()
