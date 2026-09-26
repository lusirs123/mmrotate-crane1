"""Validate the formal 24-epoch object/background relation A/C contract."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crane_project.tools.preflight_k1_dino_warmstart_v2 import (
    EXPECTED_BASELINE_SHA256, sha256)


def validate(config_a, config_c):
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings

    a = Config.fromfile(str(config_a))
    c = Config.fromfile(str(config_c))
    shared = ('data', 'train_pipeline', 'test_pipeline', 'optimizer',
              'optimizer_config', 'lr_config', 'runner', 'checkpoint_config',
              'evaluation', 'load_from', 'resume_from')
    for key in shared:
        if a.get(key) != c.get(key):
            raise ValueError('A/C differ in ' + key)
    for name, cfg in (('A', a), ('C', c)):
        if cfg.runner.max_epochs != 24 or cfg.optimizer.lr != 0.0025:
            raise ValueError(name + ' does not use the K1 24-epoch budget')
        if cfg.resume_from is not None:
            raise ValueError(name + ' unexpectedly resumes training')
        if cfg.model.backbone.frozen_stages != 1:
            raise ValueError(name + ' changes K1 backbone freezing')
        if cfg.model.bbox_head.use_semantic_cls_adapter is not True:
            raise ValueError(name + ' lacks the classification adapter')
        if cfg.model.get('semantic_distillation') is not None:
            raise ValueError(name + ' enables the old feature loss')
        if sum(step.get('type') == 'LoadDinoFeatureFromCache'
               for step in cfg.train_pipeline) != 1:
            raise ValueError(name + ' must load exactly one DINO cache')
    if a.model.get('object_background_relation') is not None:
        raise ValueError('A must not contain relation supervision')
    relation = dict(c.model.get('object_background_relation') or {})
    expected = dict(enabled=True, loss_weight=0.05, feature_level=0,
                    protect_geometry=True)
    if relation != expected:
        raise ValueError('C relation configuration does not match the contract')
    imports = list(a.get('custom_imports', {}).get('imports', []))
    import_modules_from_strings(imports, allow_failed_imports=False)
    from mmrotate.models import build_detector
    built = {}
    for name, cfg in (('A', a), ('C', c)):
        model = build_detector(cfg.model,
                               train_cfg=cfg.get('train_cfg'),
                               test_cfg=cfg.get('test_cfg'))
        built[name] = type(model).__name__
        enabled = getattr(model, 'object_background_relation', None) is not None
        if enabled != (name == 'C'):
            raise ValueError(name + ' relation module construction mismatch')
    checkpoint = Path(a.load_from).resolve()
    if sha256(checkpoint) != EXPECTED_BASELINE_SHA256:
        raise ValueError('K1 epoch_20 checkpoint SHA256 mismatch')
    return dict(
        protocol='k1_dino_object_background_relation_v5_formal_preflight',
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        config_a_sha256=sha256(config_a), config_c_sha256=sha256(config_c),
        epochs=24, optimizer=dict(a.optimizer), relation=relation,
        built_models=built,
        status='READY_FOR_FORMAL_TRAINING')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-a', required=True)
    parser.add_argument('--config-c', required=True)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    report = validate(Path(args.config_a), Path(args.config_c))
    output = Path(args.out_json)
    if output.exists():
        raise FileExistsError('Refusing to overwrite ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
