"""Standard-library contracts for frozen TEST inference timing (no GT scoring)."""
import ast
import hashlib
import json
import math
from pathlib import Path

VERSION = 'port_detection_runtime_v1'
ARMS = ('eood', 'symeood_b', 'symeood_b_midpoint')
COUNTS = {'real_seq03': 200, 'real_seq04': 668, 'sim_seq09': 572}
B_SHA = '8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23'
M_SHA = '16c2fb448ac4e1c53530b8086d547f6f6ccb8d6b0763a42391c34f9b337982d7'
IMAGE_SHA = '6448e47b9a715f47f04eacaaabb54a32b522c72208bf6e82793284de2aad665d'
M_ROWS_SHA = '13c4cfa7f1f67a7bc2cad36da1ec067adbeae74080b387d39efe39fdc41a788d'
EOOD_CONFIG = 'crane_project/configs/crane_eood_k1_port_day2night_v1.py'
B_CONFIG = 'crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py'


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
        symeood_b=dict(config=B_CONFIG, selected_epoch=24, checkpoint_sha256=B_SHA),
        midpoint=dict(sigma_cells=1.5, epoch=3, updates=2706,
            checkpoint_sha256=M_SHA, test_rows_sha256=M_ROWS_SHA, parameters=17696),
        batch_size=1, precision='FP32', warmup_frames=50, repeats=3,
        cpu_threads=1, gpu_devices=1, cudnn_benchmark=False,
        timing=dict(decoded_frame_to_obb='Decoded BGR CPU frame -> preprocessing, H2D, native detector, optional reused-P3 ROI/midpoint, original-coordinate CPU OBB; serial CUDA-synchronized wall time.',
            model_and_postprocess='Prepared CUDA image -> original-coordinate CPU OBB; includes native postprocessing/D2H and Python, not GPU-kernel-only time.',
            file_to_obb='JPEG read/decode plus decoded-frame-to-OBB (filesystem cache may be warm).',
            excluded='Model/source/reference loading, GT/metric evaluation, consistency checks, logging, drawing and result-file writes.',
            fps='n / sum(per-frame seconds); never mean(1 / per-frame seconds).'),
        validation='All1440 outputs of every repeat must match sealed predictions. A single backbone and native detector-head call per production frame. Extra audit forwards are not timed.',
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


def check_boxes(actual, expected):
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
        for index, (x, y) in enumerate(zip(a, b)):
            error = abs(float(x)-float(y))
            tolerance = 0. if index == 5 else 1e-4+1e-6*abs(float(y))
            if error > tolerance:
                raise ValueError('Output component %d differs from sealed result' % index)
            worst = max(worst, error)
    return worst


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


def check_checkpoint_meta(meta, cfg):
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
    if saved != plain({k: cfg[k] for k in fields}) or meta.get('epoch') != 24 or meta.get('seed') != 0:
        raise ValueError('Checkpoint is not the original selected detector/config')
    return dict(epoch=24, seed=0, saved_config_sha256=hashlib.sha256(meta['config'].encode()).hexdigest())


def check_selection(selection, config_sha, checkpoint_sha):
    if (selection.get('evidence_role') != 'source_val_checkpoint_selection' or
            selection.get('selected_checkpoint') != 'epoch_24' or
            selection.get('config_sha256') != config_sha or
            selection['all_checkpoints']['epoch_24']['checkpoint_sha256'] != checkpoint_sha or
            Path(selection['selected_path']).name != 'epoch_24.pth'):
        raise ValueError('Requires the existing VAL-selected epoch24; no new selection')


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
