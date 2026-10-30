"""Only two planned factors: B loss and zero/four regression layers."""
from copy import deepcopy
import math

ARMS = dict(B2R0=0, B2R4=4)
GEOMETRY_WEIGHT = 0.25
EPOCHS = (16, 18, 20, 22, 24)


def candidate_model(reference, arm):
    if arm not in ARMS:
        raise ValueError('Unknown B2 arm')
    model = deepcopy(reference)
    head = model['bbox_head']
    if head['type'] != 'SymEOODHead' or head['loss_cls']['use_geometry_weight']:
        raise ValueError('Requires the sealed A+B, C-off reference')
    head['type'] = 'SymEOODB2HeadV1'
    head['reg_tower_layers'] = ARMS[arm]
    head['memory_limit_mib'] = 6144
    head['loss_bbox'] = dict(type='SymIoUHybridLossV1', geometry_weight=GEOMETRY_WEIGHT,
        loss_weight=2.0, eps=1e-6, reduction='mean', strict_finite=True)
    return model


def check_model_delta(reference, candidate, arm):
    if candidate != candidate_model(reference, arm):
        raise ValueError('Unexpected change beyond B2 loss and regression tower')


def assess_detection(candidate, reference):
    """New EOOD-oriented development goal; sizes do not select a detector."""
    if candidate['frames'] != 887 or reference['frames'] != 887:
        raise ValueError('Requires the complete fixed VAL')
    for name in ('AP75', 'Recall_pct', 'center_MAE_px', 'mean_RIoU'):
        if not math.isfinite(candidate[name]) or not math.isfinite(reference[name]):
            raise ValueError('Nonfinite detection metric')
    checks = dict(AP75_gain=candidate['AP75'] >= reference['AP75'] + 1.0 - 1e-5,
        Recall=candidate['Recall_pct'] >= reference['Recall_pct'] - 1e-5,
        center=candidate['center_MAE_px'] <= reference['center_MAE_px'] + 1e-5,
        RIoU=candidate['mean_RIoU'] >= reference['mean_RIoU'] - 1e-5)
    return dict(checks=checks, eligible=all(checks.values()),
        AP75_gain_pp=candidate['AP75'] - reference['AP75'], TEST=False)
