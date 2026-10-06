"""Standard-library contracts for frozen TEST inference timing (no GT scoring)."""
import ast
import hashlib
import json
import math
from pathlib import Path

VERSION = 'port_detection_runtime_v1_four_arm'
ARMS = ('eood', 'symeood', 'symeood_b', 'symeood_b_midpoint')
COUNTS = {'real_seq03': 200, 'real_seq04': 668, 'sim_seq09': 572}
B_SHA = '8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23'
M_SHA = '16c2fb448ac4e1c53530b8086d547f6f6ccb8d6b0763a42391c34f9b337982d7'
IMAGE_SHA = '6448e47b9a715f47f04eacaaabb54a32b522c72208bf6e82793284de2aad665d'
M_ROWS_SHA = '13c4cfa7f1f67a7bc2cad36da1ec067adbeae74080b387d39efe39fdc41a788d'
EOOD_CONFIG = 'crane_project/configs/crane_eood_k1_port_day2night_v1.py'
SYM_CONFIG = 'crane_project/configs/crane_symeood_k1_port_day2night_v1.py'
B_CONFIG = 'crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py'
ARCHIVE = 'crane_project/data/crane_grab_port_day2night_v1/provenance/config_retirement_20260929'
# Exact immutable identities reviewed from the 20261006 server input receipt.
# Only these two migrations are accepted; unknown historical SHA stays an error.
MIGRATIONS = {
    'eood': dict(stem='crane_eood_k1',
        config_sha256='4265a9362b5e2dc99f273591a98342c9ba8deba3bbca8ae1059bf4a209a1a3c0',
        parent_sha256='dd71d89a26c47d25c129ec861b5f341b811c1f06069bb265259b838759bbdbfc',
        checkpoint_sha256='ee277d72cdf27d76e3216da4d17256b453b539d69d692cdd71960fadf7ebd2bf',
        selection_sha256='a0bf581181890dc65cc381812cff44e93dd5c6e5658857443c08bfebba39dc8a',
        test_report_sha256='5698fec53d161a8a2c6013322e851a1d84aa26a3a92b33798bfc0f6dbf2d6a3e',
        reference_sha256='4a1812c2207876c67a1c1d94c97311a6db8edfe5f43a40ca7a3b248223a35408'),
    'symeood': dict(stem='crane_symeood_k1',
        config_sha256='ca8b950c8bba2775b20a075e619a24915bcee443c1a1e0bd1858a908d966ea45',
        parent_sha256='6e096c1e273b681583b029954eb617636adb801620651f5df21b35698b177e08',
        checkpoint_sha256='780a5de1a17b32041175bdf408a209872100f776de811075c55204bf333a92fe',
        selection_sha256='7f1439a451442572c9e35131a3b29312f0c37900a5414a59f8e98c3769ff502d',
        test_report_sha256='5b04937224cac601c9c1a2d5fcd50cccf9a473c6632e362a834b6a27bc58de64',
        reference_sha256='c31827aff1e80f9276e87a61b6af15f22963567abc309b7a0e014923ae892454'),
}


def config_assignments(path):
    """Compare archived MMCV override syntax without executing any config."""
    tree = ast.parse(Path(path).read_text())
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, (ast.Str, ast.Constant)):
            if isinstance(ast.literal_eval(node.value), str):
                continue
        if (not isinstance(node, ast.Assign) or len(node.targets) != 1 or
                not isinstance(node.targets[0], ast.Name) or node.targets[0].id in result):
            raise ValueError('Unexpected archived config statement')
        result[node.targets[0].id] = node.value
    return result


def archived_config_proof(root, arm):
    """Prove the old child/parent overrides equal current config except work_dir.

    This does not replace full resolved checkpoint-config validation at build.
    The inherited model/training config is still checked there against meta.
    """
    root = Path(root); migration = MIGRATIONS[arm]; stem = migration['stem']
    old = root/ARCHIVE/(stem+'_port_day2night_seq06_v1.py.txt')
    parent = root/ARCHIVE/(stem+'_port_day2night_v1.py.txt')
    if sha(old) != migration['config_sha256'] or sha(parent) != migration['parent_sha256']:
        raise ValueError('Archived migration config SHA differs')
    child = config_assignments(old)
    if (set(child) != {'_base_', 'work_dir'} or
            literal(child['_base_']) != ['./'+stem+'_port_day2night_v1.py'] or
            literal(child['work_dir']) != 'work_dirs/'+stem+'_port_day2night_seq06_v1'):
        raise ValueError('Archived child has an unreviewed override')
    current = config_assignments(root/detector_identity(arm)['config'])
    previous = config_assignments(parent)
    expected_keys = {'_base_', 'data_root', 'data'}
    if (set(previous) != expected_keys or set(current) != expected_keys | {'work_dir'} or
            literal(current['work_dir']) != 'work_dirs/'+stem+'_port_day2night_v1' or
            any(ast.dump(previous[k]) != ast.dump(current[k]) for k in expected_keys)):
        raise ValueError('Archived/current config overrides are not equivalent')
    return dict(archived_config_sha256=sha(old), archived_parent_sha256=sha(parent),
        current_config_sha256=sha(root/detector_identity(arm)['config']),
        override_equivalence=True, ignored_changes=['docstrings', 'work_dir', 'inheritance_flattening'],
        resolved_checkpoint_config_check='Required separately before timing each arm')


def migration_proof(root, arm, recorded_config_sha, weight_sha, selection_path,
                    receipt_path, predictions):
    """Accept only the reviewed byte-identical history, with a proven config chain."""
    if recorded_config_sha == sha(Path(root)/detector_identity(arm)['config']):
        return None
    expected = MIGRATIONS[arm]
    actual = dict(config_sha256=recorded_config_sha, checkpoint_sha256=weight_sha,
        selection_sha256=sha(selection_path), test_report_sha256=sha(receipt_path),
        reference_sha256=sha(predictions))
    if any(actual[k] != expected[k] for k in actual):
        raise ValueError('Unreviewed historical config/weight/selection/TEST identity: '+
                         json.dumps(actual, sort_keys=True))
    proof = archived_config_proof(root, arm)
    proof.update(actual, mode='reviewed_seq06_directory_migration')
    return proof


def check_recorded_path(value, actual, root, arm, migration, suffix):
    """An old missing pathname is accepted only under the pinned migration proof."""
    if not isinstance(value, str) or not value:
        raise ValueError('Missing recorded artifact path')
    if migration is None:
        if Path(value).resolve() != Path(actual).resolve():
            raise ValueError('Recorded artifact points to a different path')
    else:
        old = Path(root)/('work_dirs/'+MIGRATIONS[arm]['stem']+'_port_day2night_seq06_v1')/suffix
        # No basename-only, arbitrary-prefix or another-arm relocation.
        if Path(value) != old:
            raise ValueError('Recorded artifact is outside the reviewed old directory')


def detector_identity(arm):
    if arm == 'eood':
        return dict(config=EOOD_CONFIG, selected_epoch=24, augmentation_b=False)
    if arm == 'symeood':
        return dict(config=SYM_CONFIG, selected_epoch=20, augmentation_b=False)
    if arm in ('symeood_b', 'symeood_b_midpoint'):
        return dict(config=B_CONFIG, selected_epoch=24, augmentation_b=True)
    raise ValueError('Unknown fixed benchmark arm')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def protocol_document():
    return dict(protocol=VERSION, arms=list(ARMS), split='test', frames=1440,
        sequence_counts=COUNTS, image_identity_sha256=IMAGE_SHA,
        eood=dict(config=EOOD_CONFIG, selected_epoch=24, augmentation_b=False,
            role='Existing project EOOD K=1 baseline; preserve native multi-candidate/NMS.'),
        symeood=dict(config=SYM_CONFIG, selected_epoch=20, augmentation_b=False,
            role='Existing VAL-selected SymEOOD K=1 control without B; native single-output inference.'),
        symeood_b=dict(config=B_CONFIG, selected_epoch=24, checkpoint_sha256=B_SHA),
        midpoint=dict(sigma_cells=1.5, epoch=3, updates=2706,
            checkpoint_sha256=M_SHA, test_rows_sha256=M_ROWS_SHA, parameters=17696),
        batch_size=1, precision='FP32', warmup_frames=50, repeats=3,
        cpu_threads=1, gpu_devices=1, cudnn_benchmark=False,
        numerical_settings=dict(cudnn_deterministic=True, cuda_matmul_allow_tf32=False,
            cudnn_allow_tf32=True,
            reason='Restore seed_all cuDNN determinism and unchanged torch1.13 convolution default from sealed sigma15 evaluation; same flags for all four arms. No autocast/half.'),
        output_preflight='For each arm, replay all1440 frames under its original numerical settings with exact historical scores, then replay under one common timing setting. Each phase has 50 warmup frames. Cross-setting scores are descriptive; output counts/order/raw geometry/threshold side must remain compatible. Save common-setting outputs for strict repeated-score checks. No timed arm starts until every history and common-setting check passes.',
        historical_numerical_settings={arm:historical_settings(arm) for arm in ARMS},
        score_validation=dict(historical_same_setting='Exact score equality; no new tolerance.',
            cross_setting='Measure and report score differences; no magnitude cutoff or score adjustment. Valid scores, count, candidate order, raw geometry and fixed threshold side are gated separately. Exact historical reproduction must pass first.',
            same_setting_repeats='Exact score equality against the full current-run preflight outputs.',
            midpoint='Exact equality to the native B scores from the same extraction/forward.',
            output_count_order_threshold='Unchanged count, ordered raw boxes and side of fixed score threshold0.05; no threshold tuning.'),
        timing=dict(decoded_frame_to_obb='Decoded BGR CPU frame -> preprocessing, H2D, native detector, optional reused-P3 ROI/midpoint, original-coordinate CPU OBB; serial CUDA-synchronized wall time.',
            model_and_postprocess='Prepared CUDA image -> original-coordinate CPU OBB; includes native postprocessing/D2H and Python, not GPU-kernel-only time.',
            file_to_obb='JPEG read/decode plus decoded-frame-to-OBB (filesystem cache may be warm).',
            excluded='Model/source/reference loading, GT/metric evaluation, consistency checks, logging, drawing and result-file writes.',
            fps='n / sum(per-frame seconds); never mean(1 / per-frame seconds).'),
        comparisons=['EOOD -> SymEOOD: whole model/inference route change.',
            'SymEOOD -> SymEOOD+B: same deployed structure; B is training-only, not an inference transform.',
            'SymEOOD+B -> SymEOOD+B+midpoint: frozen detector plus the fixed geometry head.'],
        input_diagnosis='Optional inputs stage reads existing metadata and streams file SHA only. No pickle/torch load, GPU, images, annotations, prediction generation or reselection. Missing or inconsistent provenance is reported, never repaired automatically.',
        historical_migration='Only the two reviewed seq06 config/weight/selection/TEST/prediction byte identities are accepted. Archived override AST must equal current config apart from work_dir and inheritance flattening; full resolved checkpoint config is still required. Old records remain unchanged; recorded paths are limited to the original named seq06 directories.',
        artifact_layout='Unified shell runner writes tests/logs/check/inputs/benchmark below one new work_dirs/port_detection_runtime_v1 task directory and packages it once on success or failure. Old task directories are retained.',
        validation='Original-mode full outputs must reproduce sealed historical scores exactly. Common-mode repeats must match their common-mode full replay scores exactly; raw geometry, count/order/threshold-side remain compatible with sealed outputs. A single backbone and native detector-head call per production frame. Extra audit forwards are not timed.',
        scope=dict(training_updates=0, gt_online=False, ground_truth_scoring=False,
            selection_on_test=False, test_repeatedly_exposed=True,
            reliability_included=False, size_residual_candidate_included=False,
            fp16=False, fusion=False, export=False, automatic_promotion=False))


def percentile(values, q):
    xs = sorted(values)
    position = (len(xs)-1)*q
    left = int(position)
    fraction = position-left
    return xs[left]*(1-fraction)+xs[min(left+1, len(xs)-1)]*fraction


def timing_summary(seconds):
    if not seconds or any(not math.isfinite(v) or v <= 0 for v in seconds):
        raise ValueError('Timing samples must be nonempty, finite and positive')
    total = math.fsum(seconds)
    return dict(frames=len(seconds), total_seconds=total, fps=len(seconds)/total,
        mean_ms=1000*total/len(seconds), p50_ms=1000*percentile(seconds, .5),
        p95_ms=1000*percentile(seconds, .95), max_ms=1000*max(seconds))


def historical_settings(arm):
    detector_identity(arm)
    # tools/test.py does not enable deterministic cuDNN; sigma15 eval calls seed_all.
    return dict(cudnn_benchmark=False, cudnn_deterministic=arm in ('symeood_b','symeood_b_midpoint'),
                cuda_matmul_allow_tf32=False, cudnn_allow_tf32=True)


def check_boxes(actual, expected, cross_setting=False):
    """Raw w/h-angle and score; no GT, canonical reordering or best-candidate pick."""
    if len(actual) != len(expected):
        raise ValueError('Output presence/candidate count differs from sealed result')
    worst = 0.
    for a, b in zip(actual, expected):
        if len(a) != 6 or len(b) != 6:
            raise ValueError('Expected Nx6 raw boxes')
        if any(not math.isfinite(float(v)) for v in list(a)+list(b)):
            raise ValueError('Nonfinite output/reference')
        if a[2] <= 0 or a[3] <= 0:
            raise ValueError('Nonpositive output size')
        if cross_setting and (not 0 <= a[5] <= 1 or not 0 <= b[5] <= 1 or
                                  (a[5] > .05) != (b[5] > .05)):
            raise ValueError('Score range or fixed threshold side differs')
        for index, (x, y) in enumerate(zip(a, b)):
            error = abs(float(x)-float(y))
            if index == 5 and cross_setting:
                continue  # score drift is recorded; exact historical mode was gated separately
            tolerance = 0. if index == 5 else 1e-4+1e-6*abs(float(y))
            if error > tolerance:
                raise ValueError('Output component %d differs from sealed result: actual=%.17g, '
                    'expected=%.17g, abs_error=%.17g, tolerance=%.17g' %
                    (index, float(x), float(y), error, tolerance))
            worst = max(worst, error)
    return worst


def check_midpoint_scores(actual, native_scores):
    if len(actual) != len(native_scores) or any(float(box[5]) != float(score)
                                              for box,score in zip(actual,native_scores)):
        raise ValueError('Midpoint changed the same-forward native B score/count')


def literal(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict' and not node.args:
        if any(k.arg is None for k in node.keywords):
            raise ValueError('Saved-config expansion is unsupported')
        return {k.arg: literal(k.value) for k in node.keywords}
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple)):
        return [literal(v) for v in node.elts]
    return ast.literal_eval(node)


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def check_checkpoint_meta(meta, cfg, epoch=24):
    fields = ('model', 'data', 'optimizer', 'optimizer_config', 'lr_config',
              'runner', 'load_from', 'resume_from')
    saved = {}
    for node in ast.parse(meta.get('config', '')).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and node.targets[0].id in fields):
            key = node.targets[0].id
            if key in saved:
                raise ValueError('Duplicate checkpoint config field')
            saved[key] = literal(node.value)
    if saved != plain({k: cfg[k] for k in fields}) or meta.get('epoch') != epoch or meta.get('seed') != 0:
        raise ValueError('Checkpoint is not the original selected detector/config')
    return dict(epoch=epoch, seed=0, saved_config_sha256=hashlib.sha256(meta['config'].encode()).hexdigest())


def selection_checks(selection, config_sha, checkpoint_sha, epoch=24):
    """Report each original strict field; missing legacy fields remain failures."""
    key = 'epoch_%d' % epoch
    selection = selection if isinstance(selection, dict) else {}
    records = selection.get('all_checkpoints')
    record = records.get(key) if isinstance(records, dict) else None
    record = record if isinstance(record, dict) else {}
    path = selection.get('selected_path')
    actual = dict(evidence_role=selection.get('evidence_role'),
        selected_checkpoint=selection.get('selected_checkpoint'),
        config_sha256=selection.get('config_sha256'),
        checkpoint_record_sha256=record.get('checkpoint_sha256'),
        selected_path_name=Path(path).name if isinstance(path, str) and path else None)
    expected = dict(evidence_role='source_val_checkpoint_selection', selected_checkpoint=key,
        config_sha256=config_sha, checkpoint_record_sha256=checkpoint_sha,
        selected_path_name=key+'.pth')
    return {name:dict(expected=expected[name], actual=actual[name],
        passed=actual[name] is not None and expected[name] is not None and
               actual[name] == expected[name]) for name in expected}


def check_selection(selection, config_sha, checkpoint_sha, epoch=24):
    checks = selection_checks(selection, config_sha, checkpoint_sha, epoch)
    failed = {name:value for name, value in checks.items() if not value['passed']}
    if failed:
        raise ValueError('Existing VAL selection mismatch for epoch_%d; no reselection: '%epoch+
            json.dumps(failed, ensure_ascii=False, sort_keys=True))


def unique_directory(candidates, required):
    """Allow identical copied/aliased inputs, reject ambiguous different reports."""
    seen = {}
    for value in candidates:
        p = Path(value).resolve()
        if p in seen or not all((p/name).is_file() for name in required):
            continue
        seen[p] = tuple(sha(p/name) for name in required)
    if not seen:
        raise FileNotFoundError('Required frozen input directory not found: '+', '.join(map(str, candidates)))
    if len(set(seen.values())) != 1:
        raise ValueError('Different frozen input copies; provide the intended explicit directory')
    return next(iter(seen))
