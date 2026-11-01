"""Explicit frozen R4/M20/L5 loading, independent of historical trainers.

Configuration inspection uses only stdlib. Runtime imports are lazy; importing
this module never touches CUDA, datasets, checkpoints, or old experiment gates.
"""
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = 'crane_project/configs/crane_sgar_midpoint_lvsr_pipeline_v1.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def literal(node):
    """Read our literal configs/constants without executing Python modules."""
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == 'dict' and not node.args:
            if any(k.arg is None for k in node.keywords):
                raise ValueError('Config dict expansion is unsupported')
            return {k.arg: literal(k.value) for k in node.keywords}
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'math' and node.func.attr == 'log'
                and len(node.args) == 1 and not node.keywords):
            return math.log(literal(node.args[0]))
        raise ValueError('Executable config expression is forbidden')
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == 'math' and node.attr == 'pi':
            return math.pi
    if isinstance(node, ast.BinOp):
        a, b = literal(node.left), literal(node.right)
        if isinstance(node.op, ast.Div): return a / b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.Pow): return a ** b
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        raise ValueError('Unsupported config arithmetic')
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.List):
        return [literal(v) for v in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(literal(v) for v in node.elts)
    return ast.literal_eval(node)


def read_constants(path, names=None):
    result = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if names is None or name in names:
                if name in result: raise ValueError('Duplicate config key: ' + name)
                result[name] = literal(node.value)
    return result


def plain(value):
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    return value


def merge_config(base, child):
    """MMCV-style dict merge: dictionaries recurse; sequences replace."""
    result = deepcopy(base)
    for key, value in child.items():
        if isinstance(value, dict):
            value = deepcopy(value)
            replace = value.pop('_delete_', False)
            result[key] = merge_config({} if replace else result.get(key, {}), value)
        else:
            result[key] = deepcopy(value)
    return result


def read_resolved_config(path, stack=()):
    """Resolve literal _base_ inheritance without importing MMCV or Torch."""
    path = Path(path).resolve()
    if path in stack: raise ValueError('Cyclic config inheritance')
    child = read_constants(path)
    bases = child.pop('_base_', [])
    if isinstance(bases, str): bases = [bases]
    if not isinstance(bases, list) or any(not isinstance(b, str) for b in bases):
        raise ValueError('Expected _base_ paths')
    result = {}
    for name in bases:
        base = read_resolved_config(path.parent/name, stack + (path,))
        if set(result) & set(base):
            raise ValueError('Duplicate keys across base configs')
        result.update(base)
    return merge_config(result, child)


def local_path(name):
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def resolve_checkpoint(spec):
    """An existing wrong file fails; do not silently hide it with a mirror."""
    for name in spec['paths']:
        path = local_path(name)
        if path.is_file():
            if sha(path) != spec['sha256']:
                raise ValueError('Checkpoint SHA differs: ' + str(path))
            return path
    raise FileNotFoundError('Checkpoint missing: ' + ', '.join(spec['paths']))


def read_config(path=DEFAULT_CONFIG):
    cfg = json.loads(local_path(path).read_text())
    if cfg['schema'] != 'crane_sgar_midpoint_lvsr_pipeline_v1':
        raise ValueError('Unknown pipeline schema')
    if [(s['stage'], s['epoch']) for s in cfg['stages']] != [('detector', 16), ('midpoint', 20), ('lvsr', 5)]:
        raise ValueError('Frozen pipeline stages differ')
    for name, digest in cfg['source_sha256'].items():
        if sha(local_path(name)) != digest:
            raise ValueError('Core source/config SHA differs: ' + name)
    configs = [read_resolved_config(local_path(s['config'])) for s in cfg['stages']]
    r, m, l = configs
    if (r['model']['type'] != 'SymEOOD' or r['model']['bbox_head']['type'] != 'SymEOODRegTowerHeadV1'
            or r['model']['bbox_head']['loss_cls']['use_geometry_weight'] is not False
            or r['model']['test_cfg']['max_per_img'] != 1):
        raise ValueError('R4 model identity differs')
    if (m['head'] != dict(type='SigmaMidpointHead', prior_sigma_cells=1.5, parameters=17696)
            or l['head'] != dict(type='PairSpecificSizeHead', alignment='fixed_b', parameters=2480)):
        raise ValueError('Refinement architecture differs')
    return cfg, configs


def inspect(path=DEFAULT_CONFIG):
    cfg, configs = read_config(path)
    weights = {s['stage']: str(resolve_checkpoint(s)) for s in cfg['stages']}
    return dict(passed=True, identity=cfg['identity'], configs=[s['config'] for s in cfg['stages']],
                checkpoints=weights, core_sources_verified=len(cfg['source_sha256']),
                training_updates=0, model_forwards=0, GPU_access=False)


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FrozenCranePipeline:
    """Image -> R4 -> midpoint -> LVSR, with no GT or old trainer input."""
    def __init__(self, config=DEFAULT_CONFIG, device='cuda:0'):
        import torch
        from mmcv import Config
        from mmcv.runner import load_checkpoint
        from mmcv.utils import import_modules_from_strings
        from mmrotate.models import build_detector
        from crane_project.utils import port_geometry_midpoint_v1 as midpoint
        from crane_project.utils import port_geometry_refine_g_v1 as geometry
        from crane_project.utils import port_geometry_midpoint_edge_residual_v1 as size
        from crane_project.utils.port_geometry_midpoint_sigma_v1 import SigmaMidpointHead
        from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
        self.torch, self.geometry, self.device = torch, geometry, device
        self.config, self.stage_configs = read_config(config)
        self.weights = {s['stage']: resolve_checkpoint(s) for s in self.config['stages']}
        r, m, l = self.stage_configs
        if (plain(midpoint.SETTINGS) != m['implementation_settings'] or
                plain(size.SETTINGS) != l['delivery_settings'] or
                {k: geometry.SETTINGS[k] for k in m['sampling']} != m['sampling']):
            raise ValueError('Implementation constants differ from formal config')
        self.cfg = Config.fromfile(str(local_path(self.config['stages'][0]['config'])))
        import_modules_from_strings(**self.cfg.custom_imports)
        spec = deepcopy(self.cfg.model)
        spec.pretrained = None
        spec.backbone.init_cfg = None
        spec.train_cfg = None
        self.detector = build_detector(spec)
        payload = load_checkpoint(self.detector, str(self.weights['detector']), map_location='cpu', strict=True)
        meta = payload['meta']
        if meta.get('epoch') != 16 or meta.get('iter') != 10240 or meta.get('seed') != 0:
            raise ValueError('R4 checkpoint metadata differs')
        recorded = {}
        for node in ast.parse(meta['config']).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                key = node.targets[0].id
                if key in ('model', 'data', 'optimizer', 'optimizer_config', 'lr_config', 'runner'):
                    recorded[key] = literal(node.value)
        if recorded != {k: r[k] for k in recorded} or len(recorded) != 6:
            raise ValueError('R4 checkpoint and formal config differ')
        self.midpoint = SigmaMidpointHead(m['head']['prior_sigma_cells'])
        self.lvsr = PairSpecificSizeHead()
        for name, head, epoch in [('midpoint', self.midpoint, 20), ('lvsr', self.lvsr, 5)]:
            payload = torch.load(str(self.weights[name]), map_location='cpu')
            if payload['epoch'] != epoch: raise ValueError('Wrong refinement epoch')
            head.load_state_dict(payload['head_state'], strict=True)
        for model in self.models():
            model.to(device).eval().requires_grad_(False)
        if [sum(p.numel() for p in model.parameters()) for model in self.models()] != [33921112, 17696, 2480]:
            raise ValueError('Frozen model parameter counts differ')
        self.counts = dict(feature_extractions=0, native_head_calls=0, midpoint_calls=0, lvsr_calls=0)

    def models(self):
        return self.detector, self.midpoint, self.lvsr

    def states(self):
        return [state_digest(m) for m in self.models()]

    def load_image(self, path):
        from mmdet.datasets.pipelines import Compose
        from mmcv.parallel import collate, scatter
        data = dict(img_info=dict(filename=str(Path(path).resolve())), img_prefix=None,
                    bbox_fields=[], mask_fields=[], seg_fields=[])
        value = Compose(self.cfg.data.test.pipeline)(data)
        gpu_index = self.torch.device(self.device).index
        value = scatter(collate([value], samples_per_gpu=1), [0 if gpu_index is None else gpu_index])[0]
        image, metas = value['img'][0], value['img_metas'][0]
        if image.shape != (1, 3, 1024, 1024) or len(metas) != 1 or metas[0].get('flip', False):
            raise ValueError('Expected native single-scale 1024 batch1 input')
        self.geometry.checked_meta(metas[0])
        return image, metas

    def predict(self, image, metas):
        import numpy as np
        torch, g = self.torch, self.geometry
        if any(m.training or any(p.requires_grad for p in m.parameters()) for m in self.models()):
            raise ValueError('Frozen inference models required')
        with torch.no_grad():
            features = self.detector.extract_feat(image)
            self.counts['feature_extractions'] += 1
            def native(rescale):
                self.counts['native_head_calls'] += 1
                value = self.detector.simple_test_from_features(features, metas, rescale=rescale)
                boxes = np.asarray(value[0][0])
                if boxes.ndim != 2 or boxes.shape[1] != 6 or len(boxes) > 1 or not np.isfinite(boxes).all():
                    raise ValueError('Invalid native R4 result')
                return boxes
            raw_np, original_np = native(False), native(True)
            raw, original = image.new_tensor(raw_np), image.new_tensor(original_np)
            restored = g.map_boxes(raw[:, :5], metas[0], inverse=True)
            if (not torch.allclose(restored, original[:, :5], atol=1e-4, rtol=1e-6)
                    or not torch.equal(raw[:, 5], original[:, 5])):
                raise ValueError('Native box coordinate restoration differs')
            roi, support, _ = g.sample_local(features[0], raw[:, :5], metas[0], 'aligned')
            xy = original.new_tensor([np.asarray(metas[0]['scale_factor']).reshape(-1)[:2]]).reshape(1, 2)
            xy = xy.expand(len(original), -1)
            mid = self.midpoint(roi, support, original, raw[:, :5], xy)
            self.counts['midpoint_calls'] += 1
            size = self.lvsr(roi, support, original, raw[:, :5], xy, mid['boxes_original'])
            self.counts['lvsr_calls'] += 1
            if not np.array_equal(raw_np, native(False)):
                raise ValueError('Refinement changed native detector output')
            for output in [mid['boxes_original'], size['boxes_original']]:
                if len(output) != len(original) or not torch.equal(output[:, 5], original[:, 5]):
                    raise ValueError('Refinement changed presence/score')
            if not torch.equal(mid['boxes_original'][:, [0, 1, 4, 5]], size['boxes_original'][:, [0, 1, 4, 5]]):
                raise ValueError('LVSR changed midpoint center/angle/score')
            present = bool(len(original))
            row = {name: value[0].cpu().tolist() if present else None for name, value in
                   [('b', original), ('midpoint', mid['boxes_original']), ('edge_residual', size['boxes_original'])]}
            row.update(accepted=bool(mid['accepted'][0]) if present else None,
                failed_checks=[k for k, v in mid['checks'].items() if present and not bool(v[0])],
                size_delivery='no_output' if not present else 'neutral' if bool(size['neutral_bypass'][0]) else
                    'square' if bool(size['square_bypass'][0]) else 'accepted' if bool(size['accepted'][0]) else 'fallback',
                size_failed_checks=[k for k, v in size['checks'].items() if present and not bool(v[0])],
                vector_valid=bool(size['vector_valid'][0]) if present else None)
            return row
