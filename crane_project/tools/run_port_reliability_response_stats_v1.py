"""TRAIN fit -> full VAL -> gated frozen TEST; fixed midpoint response evidence."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
from crane_project.utils import port_reliability_response_stats_v1 as core
from crane_project.utils import port_reliability_response_stats_v1_torch as native
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_simple_pilot_increment_v1 as base
from crane_project.utils.port_reliability_paired_ranking_audit_v1 import Ranking, bad, groups
from crane_project.tools import run_port_reliability_redc_size_v1 as history

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'crane_project/tools/port_reliability_response_stats_v1_manifest.json'
KEYS = ('roi', 'support', 'boxes_original', 'boxes_model', 'scale_xy')


def sha(path):
    return history.sha(path)


def write(path, value):
    return history.write(path, value)


def pins(values):
    for path, expected in values.items():
        if sha(ROOT/path) != expected:
            raise ValueError('Source/input SHA mismatch: '+path)


def load_rows(role, manifest):
    item = manifest['roles'][role]
    with gzip.open(ROOT/item['rows'], 'rt') as f:
        rows = [json.loads(s) for s in f]
    base.checked_rows(rows, role, item['frames'])
    return rows


def online_maps(head, inputs):
    """GT-free inputs only. Frozen B-local priors, not final-box re-sampling."""
    from crane_project.utils import port_geometry_midpoint_v1 as midpoint
    from crane_project.utils.port_geometry_midpoint_sigma_v1 import point_prior
    if len(inputs) != 5 or head.training or any(p.requires_grad for p in head.parameters()):
        raise ValueError('Frozen head/GT-free five tensors required')
    decoded = head(*inputs)
    if not len(inputs[2]):
        return decoded, None, None
    b = midpoint.frame(inputs[2], inputs[3], inputs[4])[0]
    reference = midpoint.original_to_roi(midpoint.box_midpoints(b[:, :5]), b, inputs[3], inputs[4])
    prior = point_prior(reference, 1.5)[1].softmax(-1).reshape(-1, 4, 9, 9)
    return decoded, decoded['probabilities'], prior


def collect(rows, role, head, device, out, manifest):
    item = manifest['roles'][role]
    # Role tensors are hashed/loaded only when this stage is permitted.
    pins({item['cache']: item['cache_sha256']})
    shard = torch.load(str(ROOT/item['cache']), map_location='cpu')
    if role == 'TEST':
        from crane_project.utils.port_reliability_readout_test_v1_torch import validate_record
        if shard['GT_online'] or shard['protocol'] != 'port_size_candidate_downstream_v1_fixed_diagnostic':
            raise ValueError('Wrong TEST GT-free cache provenance')
        # The historical ROI list is in the same ordered frame identity as its predictions.
        cache = shard['rows']
        predictions = [json.loads(s) for s in (ROOT/item['inventory']).read_text().splitlines() if s.strip()]
        if len(cache) != len(rows) or len(predictions) != len(rows):
            raise ValueError('TEST cache membership differs')
        index = {}
        for record, inventory in zip(cache, predictions):
            image = inventory['image']
            row = next((r for r in rows if r['image'] == image), None)
            if row is None or image in index:
                raise ValueError('TEST missing or duplicate identity')
            validate_record(record, row['b_original'], row['pred'], inventory['scale_xy'])
            index[image] = record
    else:
        cached_manifest = json.loads((ROOT/history.CACHE/'cache_manifest.json').read_text())
        if shard['identity'] != cached_manifest['identity'] or shard['detector_state'] != cached_manifest['detector_state']:
            raise ValueError('Frozen cache identity differs')
        index = {r['image']: r for r in shard['records']}
        if len(index) != len(shard['records']):
            raise ValueError('Duplicate cache identity')
    if set(index) != {r['image'] for r in rows}:
        raise ValueError('Complete role cache membership differs')
    ordered = sorted(rows, key=lambda r: r['image'])
    ids, ps, qs, features, traces = [], [], [], [], []
    maximum = np.zeros(6); statistic_error = 0.
    before = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    for i, row in enumerate(ordered):
        record = index[row['image']]; present = row['pred'] is not None
        n = int(present)
        if role != 'TEST':
            if (record['role'] != role.lower() or record['scale'] != 1.
                    or any(record[k] != row[k] for k in ('image', 'sequence', 'domain', 'split', 'frame_id', 'image_size'))):
                raise ValueError('TRAIN/VAL cache metadata differs')
            if n and record['boxes_original'][0].tolist() != row['b_original']:
                raise ValueError('B cache/output identity differs')
            gt = record['gt_original'][0].tolist()
            if gt[:4] != row['gt'][:4] or abs((gt[4]-row['gt'][4]+np.pi/2)%np.pi-np.pi/2) > 2e-6:
                raise ValueError('Offline annotation pairing differs')
        shapes = ((n, 256, 9, 9), (n, 1, 9, 9), (n, 6), (n, 5), (n, 2))
        for key, shape in zip(KEYS, shapes):
            t = record[key]
            if tuple(t.shape) != shape or t.requires_grad or not bool(torch.isfinite(t).all()):
                raise ValueError('Invalid cached input '+key)
        inputs = [record[k].to(device) for k in KEYS]
        with torch.no_grad():
            decoded, p, q = online_maps(head, inputs)
        trace = dict(image=row['image'], present=present, GT_online=False)
        if n:
            replay = decoded['boxes_original'][0].cpu().numpy().astype(float)
            expected = np.asarray(row['pred']); delta = np.abs(replay-expected)
            delta[4] = abs((replay[4]-expected[4]+np.pi/2)%np.pi-np.pi/2)
            maximum = np.maximum(maximum, delta)
            if (not np.allclose(replay[:4], expected[:4], atol=5e-4, rtol=2e-5)
                    or delta[4] > 2e-6 or replay[5] != expected[5]
                    or bad(dict(row, pred=replay.tolist())) != bad(row)):
                raise ValueError('M output/label replay differs: '+row['image'])
            pv, qv = p.cpu().numpy(), q.cpu().numpy()
            response = core.response_features(pv, qv)
            with torch.no_grad():
                tv = native.response_features(p, q).cpu().numpy()
            statistic_error = max(statistic_error, float(np.max(np.abs(response-tv))))
            if statistic_error > 1e-10:
                raise ValueError('Torch/NumPy statistics disagree')
            ids.append(row['image']); ps.append(pv[0]); qs.append(qv[0])
            features.append(np.r_[simple.descriptor(row['pred'], row['image_size']), response[0]])
            trace.update(replay=replay.tolist(), probability_sha256=hashlib.sha256(pv.tobytes()).hexdigest(),
                         prior_sha256=hashlib.sha256(qv.tobytes()).hexdigest())
        elif len(decoded['boxes_original']):
            raise ValueError('Missing output restored')
        traces.append(trace)
        if i % 200 == 0 or i+1 == len(ordered):
            print('Response cache', role, i+1, '/', len(ordered), flush=True)
    if any(p.grad is not None for p in head.parameters()) or any(not torch.equal(before[k], v.cpu()) for k, v in head.state_dict().items()):
        raise ValueError('Frozen M changed')
    raw = np.asarray(features)
    np.savez_compressed(out/(role.lower()+'_responses.npz'), features=raw, images=np.asarray(ids),
                        probability=np.asarray(ps), prior=np.asarray(qs))
    with gzip.open(out/(role.lower()+'_replay.jsonl.gz'), 'xt') as f:
        for trace in traces:
            f.write(json.dumps(trace, allow_nan=False)+'\n')
    del shard, index
    return raw, ids, dict(frames=len(rows), outputs=len(ids), maximum_M_replay_difference=maximum.tolist(),
                         statistics_torch_numpy_max_abs=statistic_error, head_updates=0, detector_inferences=0,
                         original_M_never_replaced=True, GT_online=False)


def add_scores(rows, raw, ids, models, policy):
    from crane_project.utils.port_midpoint_sigma15_reliability_v1 import Sigma15Reliability
    api = Sigma15Reliability(policy, policy['front_end'])
    lookup = {image: i for i, image in enumerate(ids)}
    values = {name: core.numpy_risk(model, raw[:, :model['dimensions']]) for name, model in models.items()}
    for row in rows:
        if api.decide(row['pred'], row['image_size']) != row['original_simple_decision']:
            raise ValueError('Original policy no longer replays')
        original = row['original_simple_decision']['risks']['size']
        row['audit_risks'] = dict(formal=original,
                                 score=1-row['pred'][5] if row['pred'] is not None else None)
        for name, scores in values.items():
            row['audit_risks'][name] = float(scores[lookup[row['image']]]) if row['pred'] is not None else None


def benchmark(head, model, train, device):
    """Three fixed TRAIN images, paired full forward incl. CPU postprocessing.

    Dataset loading is outside timing; this is inference pipeline latency, not
    camera/decode/control end-to-end. Baseline includes original three flags.
    """
    from mmcv.parallel import collate, scatter
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as frozen
    from crane_project.utils.port_midpoint_sigma15_reliability_v1 import Sigma15Reliability
    cfg = frozen.g.check_cfg()
    detector = frozen.build_detector(cfg, int(device.split(':')[1]))
    before = frozen.g.state_digest(detector)
    dataset = frozen.dataset_for(cfg, 'train', 1.)
    names = [Path(info['filename']).stem for info in frozen.infos(dataset)]
    valid = sorted(r['image'] for r in train if r['pred'] is not None)
    selected = [valid[0], valid[len(valid)//2], valid[-1]]
    policy = json.loads((ROOT/history.POLICY).read_text()); api = Sigma15Reliability(policy, policy['front_end'])
    records = []
    for name in selected:
        value = scatter(collate([dataset[names.index(name)]], samples_per_gpu=1), [int(device.split(':')[1])])[0]
        image, metas = value['img'][0], value['img_metas'][0]
        meta = metas[0]; image_size = list(meta['ori_shape'][:2][::-1])
        def forward(enhanced):
            with torch.no_grad():
                fpn = detector.extract_feat(image)
                raw = frozen.g.flatten_prediction(detector.simple_test_from_features(fpn, metas, rescale=False))
                b = frozen.g.flatten_prediction(detector.simple_test_from_features(fpn, metas, rescale=True))
                if not len(b):
                    return api.decide(None, image_size)
                bt, rt = image.new_tensor(b), image.new_tensor(raw)
                roi, support, _ = frozen.g.sample_local(fpn[0], rt[:, :5], meta, 'aligned')
                xy = image.new_tensor(np.asarray(meta['scale_factor'])[:2]).reshape(1, 2)
                inputs = [roi, support, bt, rt[:, :5], xy]
                if enhanced:
                    decoded, p, q = online_maps(head, inputs)
                else:
                    decoded = head(*inputs)
                pred = decoded['boxes_original'][0].cpu().numpy().astype(float).tolist()
                decision = api.decide(pred, image_size)
                if enhanced:
                    response = core.response_features(p.cpu().numpy(), q.cpu().numpy())
                    x = np.c_[simple.descriptor(pred, image_size)[None], response]
                    risk = float(core.numpy_risk(model, x)[0])
                    decision = core.size_decision(decision, risk, .5)
                return decision
        baseline_identity = forward(False)
        candidate_identity = forward(True)
        for key in ('final_box_original', 'center_accepted', 'angle_accepted'):
            if baseline_identity[key] != candidate_identity[key]:
                raise ValueError('Timing implementation changes detector/other flags')
        for _ in range(10):
            forward(False); forward(True)
        times = dict(baseline=[], response=[])
        for repeat in range(30):
            for enhanced in ((False, True) if repeat % 2 == 0 else (True, False)):
                torch.cuda.synchronize(); start = time.perf_counter()
                forward(enhanced); torch.cuda.synchronize()
                times['response' if enhanced else 'baseline'].append(1000*(time.perf_counter()-start))
        summary = {arm: dict(median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)), samples_ms=t)
                   for arm, t in times.items()}
        summary['image'] = name
        summary['median_overhead_fraction'] = summary['response']['median_ms']/summary['baseline']['median_ms']-1
        records.append(summary)
    if frozen.g.state_digest(detector) != before:
        raise ValueError('Timing changes frozen detector')
    all_times = {a: [v for r in records for v in r[a]['samples_ms']] for a in ('baseline', 'response')}
    medians = {a: float(np.median(v)) for a, v in all_times.items()}
    del detector, dataset
    torch.cuda.empty_cache()
    return dict(images=records, batch=1, input=[1, 3, 1024, 1024], precision='M FP32 / scorer NumPy float64',
        hardware=torch.cuda.get_device_name(), warmups_per_image_per_arm=10, repeats_per_image_per_arm=30,
        aggregate_median_ms=medians, aggregate_median_overhead_fraction=medians['response']/medians['baseline']-1,
        scope='Tensor-to-box/three-flags, excludes image IO/camera/decode. Three TRAIN images only; shared host load.',
        scoring_cutoff=.5, cutoff_is_timing_placeholder_not_policy=True, measured_before_VAL=True)


def run(out, gpu):
    if out.exists() or Path.cwd().resolve() != ROOT or out.parent.parent != ROOT/'work_dirs'/core.VERSION:
        raise ValueError('Use new unified experiment/RUN_ID/result from project root')
    manifest = json.loads(MANIFEST.read_text())
    if manifest['settings'] != core.SETTINGS:
        raise ValueError('Frozen settings differ')
    pins(manifest['source_pins']); pins(manifest['development_pins'])
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(gpu); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = 'cuda:'+str(gpu)
    out.mkdir(parents=True); write(out/'protocol.json', manifest)
    write(out/'runtime.json', dict(cwd=str(ROOT), environment=os.environ.get('CONDA_DEFAULT_ENV'),
        python=platform.python_version(), torch=torch.__version__, GPU=torch.cuda.get_device_name(gpu),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        HEAD=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()))
    from crane_project.utils.port_geometry_midpoint_sigma_v1 import SigmaMidpointHead
    payload = torch.load(str(ROOT/history.HEAD), map_location='cpu')
    if payload['epoch'] != 3 or payload['sigma_cells'] != 1.5 or payload['updates'] != 2706:
        raise ValueError('Wrong midpoint checkpoint identity')
    head = SigmaMidpointHead(1.5).to(device).eval().requires_grad_(False)
    head.load_state_dict(payload['head_state'], strict=True)
    train = load_rows('TRAIN', manifest)
    raw, ids, proof_train = collect(train, 'TRAIN', head, device, out, manifest)
    lookup = {r['image']: r for r in train}; labels = np.array([bad(lookup[i]) for i in ids], dtype=float)
    if raw.shape != (2558, 19) or labels.sum() != 71:
        raise ValueError('Full TRAIN material differs')
    models, objects = {}, {}
    with (out/'train_log.jsonl').open('x') as f:
        for seed in core.SEEDS:
            states = {}
            for arm, dimensions in (('control', 3), ('response', 19)):
                name = '%s_%d' % (arm, seed)
                obj, model, records = native.fit(raw[:, :dimensions], labels, seed)
                models[name] = model; objects[name] = obj; states[arm] = obj.state_dict()
                for record in records:
                    f.write(json.dumps(dict(method=name, **record), allow_nan=False)+'\n')
                print('TRAIN', name, 'iterations', model['optimizer_iterations'], 'gradient', model['max_abs_gradient'], flush=True)
            path = out/('seed_%d.pth' % seed); torch.save(states, str(path))
            reloaded = torch.load(str(path), map_location='cpu')
            if any(not torch.equal(v, reloaded[a][k]) for a, state in states.items() for k, v in state.items()):
                raise ValueError('Saved checkpoint reload differs')
    write(out/'models.json', models)
    # Timing uses TRAIN images only and frozen1701; never select a faster seed.
    latency = benchmark(head, models['response_1701'], train, device)
    write(out/'latency.json', latency)
    write(out/'TRAIN_frozen.json', dict(models_sha256=sha(out/'models.json'),
        protocol_sha256=sha(MANIFEST), all_training_complete_before_VAL=True,
        TEST_read=False, input_rows_sha256=manifest['development_pins'][manifest['roles']['TRAIN']['rows']]))
    # Only now read VAL rows and tensors. All architecture/statistics/weights are frozen.
    val = load_rows('VAL', manifest)
    pins(manifest['val_pins'])
    vraw, vids, proof_val = collect(val, 'VAL', head, device, out, manifest)
    policy = json.loads((ROOT/history.POLICY).read_text())
    cuts = dict(formal=policy['size']['threshold'] if 'size' in policy else None)
    # Exact original deployment threshold is copied from the previously frozen evaluation.
    cuts['formal'] = manifest['formal_cutoff']
    add_scores(train, raw, ids, models, policy); add_scores(val, vraw, vids, models, policy)
    for name in models:
        cuts[name] = Ranking(val, name).same_cr(core.SETTINGS['primary_CR'])['risk_le']
    cuts['score'] = Ranking(val, 'score').same_cr(core.SETTINGS['primary_CR'])['risk_le']
    methods = ('formal', 'score')+tuple(models)
    roles = dict(TRAIN=train, VAL=val)
    results = {role: {name: base.diagnostics(rows, name, cuts) for name in methods} for role, rows in roles.items()}
    decision = core.gate(val, models, cuts)
    # Latency is a separate feasibility gate, not used to tune the scorer.
    decision['latency_within_5_percent'] = latency['aggregate_median_overhead_fraction'] <= .05
    decision['performance_passed'] = decision['passed']
    decision['passed'] = decision['passed'] and decision['latency_within_5_percent']
    write(out/'freeze.json', dict(decision=decision, cutoffs=cuts, models_sha256=sha(out/'models.json'),
        protocol_sha256=sha(MANIFEST), TEST_read=False, all_models_cutoffs_gate_frozen_before_TEST=True))
    proofs = dict(TRAIN=proof_train, VAL=proof_val)
    if decision['passed']:
        pins(manifest['test_pins']); test = load_rows('TEST', manifest)
        traw, tids, proofs['TEST'] = collect(test, 'TEST', head, device, out, manifest)
        test_models = {k: v for k, v in models.items() if k.endswith('_1701')}
        add_scores(test, traw, tids, test_models, policy); roles['TEST'] = test
        results['TEST'] = {name: base.diagnostics(test, name, cuts) for name in ('formal', 'score')+tuple(test_models)}
    else:
        (out/'TEST_NOT_RUN.txt').write_text('VAL/latency gate failed. No TEST rows, tensors or TEST input hashes read.\n')
    for role, rows in roles.items():
        with gzip.open(out/('scored_'+role+'.jsonl.gz'), 'xt') as f:
            for row in rows:
                row['response_decisions'] = {name: core.size_decision(row['original_simple_decision'],
                    row['audit_risks'][name], cuts[name]) for name in models if name in row['audit_risks']}
                f.write(json.dumps(row, allow_nan=False)+'\n')
    report = dict(status='VAL_PASS_FROZEN_TEST_REPORTED' if decision['passed'] else 'VAL_FAILED_STOP',
        results=results, cutoffs=cuts, decision=decision, proofs=proofs, latency=latency,
        TEST_read=bool(decision['passed']), TEST_repeatedly_exposed=True, TEST_used_for_selection=False,
        detector_updates=0, midpoint_updates=0, scorer_fits=6, pilot_used=False, GT_online=False,
        boxes_scores_output_center_angle_unchanged=True, original_policy_changed=False,
        source_manifest_sha256=sha(MANIFEST), model_sha256=sha(out/'models.json'))
    write(out/'report.json', report)
    with (out/'metrics.csv').open('x', newline='') as f:
        columns = ['role', 'method', 'group', 'FA', 'FR', 'ED', 'CR', 'MISSING', 'correct_retention',
                   'accepted_error_rate', 'acceptance_coverage', 'full_frame_correct_coverage', 'longest_correct_FR',
                   'longest_FR_observations', 'longest_without_CR_observations']
        writer = csv.DictWriter(f, columns); writer.writeheader()
        for role, methods_result in results.items():
            for name, value in methods_result.items():
                for group, d in value['at_frozen_VAL_cut'].items():
                    line = dict(role=role, method=name, group=group, **d['states'])
                    line.update({k: d[k] for k in columns if k in d}); writer.writerow(line)
    lines = ['# 冻结midpoint响应统计对照', '', report['status'], '',
             '|角色|方法|FA|FR|ED|CR|Real CR|', '|---|---|---:|---:|---:|---:|---:|']
    for role, values in results.items():
        for name, value in values.items():
            d = value['at_frozen_VAL_cut']; s = d['all']['states']
            lines.append('|%s|%s|%d|%d|%d|%d|%d|' % (role, name, s['FA'], s['FR'], s['ED'], s['CR'], d['domain:real']['states']['CR']))
    lines += ['', '16维响应统计减对应B先验统计；3→8→1控制41参数，19→8→1候选169参数。',
        '共同描述量路径/输出层配对初始化；候选新增输入权重初始为零。非纯容量消融。',
        '完整TRAIN拟合，完整VAL统一CR>=653；三种子均须比正式simple和对应控制少至少3FA，RealCR>=189，且同接受数有收益。',
        '损失为类平衡BCE+L2=.1；相同系数不等于相同函数正则。无误差概率校准保证。',
        '响应形态是学习用代理，不把尖峰解释为尺寸可靠。TEST只在性能及5%时延门槛通过后报告。',
        '推理时延见latency.json：3张TRAIN图、batch1、10次预热/30次交替测量，不含图片IO。',
        json.dumps(decision, ensure_ascii=False)]
    (out/'analysis.md').write_text('\n'.join(lines)+'\n')
    pins(manifest['source_pins']); pins(manifest['development_pins']); pins(manifest['val_pins'])
    write(out/'completion.json', dict(status=report['status'], artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(report['status'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0); args = parser.parse_args()
    try:
        run(args.out.resolve(), args.gpu)
    except Exception as error:
        if args.out.is_dir() and not (args.out/'failure.json').exists() and not (args.out/'completion.json').exists():
            write(args.out/'failure.json', dict(status='ENGINEERING_FAILED', error=repr(error)))
        raise
