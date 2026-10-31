"""Exactly one B regression change on the effective EOOD skeleton."""
from copy import deepcopy
import math

LOSS_IMPORT = 'mmrotate.models.losses.eood_sym_kld_positive_loss_v1'
EPOCHS = (16, 18, 20, 22, 24)


def training_arguments(config, work, backend='nccl'):
    if backend not in ('nccl', 'gloo'):
        raise ValueError('Unsupported distributed communication backend')
    args = ['tools/train.py', str(config), '--launcher', 'pytorch', '--seed', '0',
            '--work-dir', str(work)]
    if backend != 'nccl':
        args += ['--cfg-options', 'dist_params.backend=' + backend]
    return args


def candidate_model(reference):
    model = deepcopy(reference)
    if model['type'] != 'Eood' or model['bbox_head']['type'] != 'EoodHead':
        raise ValueError('Original EOOD skeleton required')
    head = model['bbox_head']['predictors'][0]
    if head['train_cfg']['pola']['type'] != 'PolaAssigner':
        raise ValueError('Original POLA must remain unchanged')
    old = head['loss_iou']
    if old != dict(type='RotatedIoULoss', loss_weight=2.0, mode='linear', diou=True):
        raise ValueError('Unexpected effective baseline regression')
    head['loss_iou'] = dict(type='EOODSymKLDPositiveLossV1', loss_weight=2.0,
        eps=1e-6, reduction='mean', strict_finite=True)
    return model


def check_model_delta(reference, candidate):
    if candidate != candidate_model(reference):
        raise ValueError('Changed beyond the effective main regression loss')


def assess_detection(candidate, baseline):
    if candidate['frames'] != 887 or baseline['frames'] != 887:
        raise ValueError('Complete fixed VAL required')
    for name in ('AP75','Recall_pct','center_MAE_px','mean_RIoU'):
        if not all(math.isfinite(row[name]) for row in (candidate,baseline)):
            raise ValueError('Nonfinite metric')
    tol=1e-5
    checks=dict(AP75_gain=candidate['AP75']>=baseline['AP75']+1.-tol,
        Recall=candidate['Recall_pct']>=baseline['Recall_pct']-tol,
        center=candidate['center_MAE_px']<=baseline['center_MAE_px']+tol,
        RIoU=candidate['mean_RIoU']>=baseline['mean_RIoU']-tol)
    return dict(checks=checks,eligible=all(checks.values()),
        AP75_gain_pp=candidate['AP75']-baseline['AP75'],TEST=False,
        strict_initialization_pair=False,automatic_adoption=False)
