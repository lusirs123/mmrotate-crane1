"""One E+A candidate versus the frozen augmented EOOD; no control retrain."""
from copy import deepcopy
import math

ASSIGNER_IMPORT = 'mmrotate.core.bbox.assigners.pola_kld_geometry_v1'
EPOCHS = (16, 18, 20, 22, 24)


def candidate_model(reference):
    model = deepcopy(reference)
    if model['type'] != 'Eood' or model['bbox_head']['type'] != 'EoodHead':
        raise ValueError('Requires the effective original EOOD skeleton')
    main = model['bbox_head']['predictors'][0]
    if main['train_cfg']['pola']['type'] != 'PolaAssigner':
        raise ValueError('Reference POLA differs')
    if main['train_cfg']['pola'].get('o2m', False):
        raise ValueError('Requires one-to-one POLA')
    main['train_cfg']['pola'].update(type='PolaKLDGeometryV1',
                                    geometry_mode='kld', eps=1e-6)
    return model


def check_model_delta(reference, candidate):
    if candidate != candidate_model(reference):
        raise ValueError('Unexpected change beyond POLA geometry')


def assess_detection(candidate, baseline):
    if candidate['frames'] != 887 or baseline['frames'] != 887:
        raise ValueError('Complete fixed VAL required')
    fields = ('AP75', 'Recall_pct', 'center_MAE_px', 'mean_RIoU')
    for name in fields:
        if not all(math.isfinite(r[name]) for r in (candidate, baseline)):
            raise ValueError('Nonfinite detection metric')
    tol = 1e-5
    checks = dict(AP75_gain=candidate['AP75'] >= baseline['AP75'] + 1. - tol,
        Recall=candidate['Recall_pct'] >= baseline['Recall_pct'] - tol,
        center=candidate['center_MAE_px'] <= baseline['center_MAE_px'] + tol,
        RIoU=candidate['mean_RIoU'] >= baseline['mean_RIoU'] - tol)
    return dict(checks=checks, eligible=all(checks.values()),
                AP75_gain_pp=candidate['AP75'] - baseline['AP75'], TEST=False,
                strict_initialization_pair=False, automatic_adoption=False)
