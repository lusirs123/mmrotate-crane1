"""EOOD-background SGAR: exactly two training-geometry replacements.

Stdlib only. Importing this module does not prepare or launch an experiment.
The historical EOOD and R4 are reused; R4-Ref is not an experiment arm.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = 'work_dirs/crane_eood_k1_port_day2night_aug_b_v1'
REFERENCE_CONFIG = REFERENCE + '/crane_eood_k1_port_day2night_aug_b_v1.py'
IMPORTS = (
    'mmrotate.core.bbox.assigners.pola_kld_geometry_v1',
    'mmrotate.models.losses.eood_sym_kld_positive_loss_v1',
)
EPOCHS = (16, 18, 20, 22, 24)


def candidate_model(reference):
    model = deepcopy(reference)
    if model['type'] != 'Eood' or model['bbox_head']['type'] != 'EoodHead':
        raise ValueError('Requires the original EOOD skeleton, not R4')
    main = model['bbox_head']['predictors'][0]
    if main['train_cfg']['pola'] != dict(type='PolaAssigner', cost_class=1.,
                                       cost_bbox=1., cost_riou=1.):
        raise ValueError('Unexpected original POLA configuration')
    if main['train_cfg']['init_epoch'] != 4:
        raise ValueError('Original four-epoch MaxIoU phase required')
    if main['loss_iou'] != dict(type='RotatedIoULoss', loss_weight=2.,
                                mode='linear', diou=True):
        raise ValueError('Unexpected effective main regression')
    main['train_cfg']['pola'].update(type='PolaKLDGeometryV1',
                                    geometry_mode='kld', eps=1e-6)
    main['loss_iou'] = dict(type='EOODSymKLDPositiveLossV1', loss_weight=2.,
                           eps=1e-6, reduction='mean', strict_finite=True)
    return model


def check_model_delta(reference, candidate):
    if candidate != candidate_model(reference):
        raise ValueError('Unexpected change beyond the two SGAR geometry slots')


def candidate_config(reference, work_dir):
    cfg = deepcopy(reference)
    cfg['model'] = candidate_model(reference['model'])
    for name in IMPORTS:
        if name not in cfg['custom_imports']['imports']:
            cfg['custom_imports']['imports'].append(name)
    cfg['work_dir'] = str(work_dir)
    cfg['seed'] = 0
    if cfg.get('load_from') is not None or cfg.get('resume_from') is not None:
        raise ValueError('Training must start from initialization, not a trained arm')
    return cfg


def check_config_delta(reference, candidate):
    expected = candidate_config(reference, candidate['work_dir'])
    if candidate != expected:
        raise ValueError('Candidate changes EOOD settings beyond the allowed delta')


def training_arguments(config, work_dir, backend='nccl'):
    if backend not in ('nccl', 'gloo'):
        raise ValueError('Unsupported backend')
    args = ['tools/train.py', str(config), '--launcher', 'pytorch', '--seed', '0',
            '--work-dir', str(work_dir)]
    if backend != 'nccl':
        args += ['--cfg-options', 'dist_params.backend=' + backend]
    return args


def describe_change(candidate, baseline):
    """Descriptive VAL deltas, never an adoption gate or a checkpoint selector."""
    if candidate['frames'] != 887 or baseline['frames'] != 887:
        raise ValueError('Complete 887-frame VAL required')
    fields = ('AP75', 'Recall_pct', 'center_MAE_px', 'mean_RIoU')
    if not all(math.isfinite(row[k]) for row in (candidate, baseline) for k in fields):
        raise ValueError('Nonfinite comparison metric')
    return dict(candidate_minus_baseline={k: candidate[k]-baseline[k] for k in fields},
                success_threshold=None, selection_changed=False, TEST=False,
                strict_initialization_pair=False, automatic_adoption=False)


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
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify(base, launch=False):
    c = read(base / ('launch_protocol.json' if launch else 'protocol.json'))
    if c['protocol'] != 'eood_sgar_three_arm_v1':
        raise ValueError('Not the EOOD-background SGAR protocol')
    if sha(ROOT / c['config']) != c['config_sha256']:
        raise ValueError('Candidate config changed')
    for name, digest in {**c['sources'], **c.get('source_tree', {})}.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Source changed: ' + name)
    if launch and sha(base / 'protocol.json') != c['preparation_protocol_sha256']:
        raise ValueError('Preparation protocol changed')
    return c
