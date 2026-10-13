"""Frozen image/temporal fusion with complete partial-AUC fallback decisions."""
from copy import deepcopy
import math
from crane_project.utils import port_reliability_image_temporal_crossfit_v1 as prior

VERSION = 'port_reliability_fallback_decision_v1'
METHOD = 'fallback_decision'
CONTROLS = ('crossfit_fusion', 'in_sample_fusion', 'partial_auc4', 'fusion4', 'full_simple', 'score_only')


def decide_size(risk, history_available, supported_cutoff, fallback_cutoff):
    for cutoff in (supported_cutoff, fallback_cutoff):
        if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
            raise ValueError('Finite frozen cutoff required')
    if not isinstance(history_available, bool):
        raise ValueError('Boolean history availability required')
    if risk is None:
        return False
    if not math.isfinite(risk) or not 0 <= risk <= 1:
        raise ValueError('Finite risk in [0,1] required')
    return risk <= (supported_cutoff if history_available else fallback_cutoff)


class CausalFallbackDecision(prior.CausalImageTemporal):
    def __init__(self, temporal_models, temporal_model, pca, head,
                 head_normalizer, fusion, cutoff, original_policy, fallback_cutoff):
        super().__init__(temporal_models, temporal_model, pca, head,
                         head_normalizer, fusion, cutoff, original_policy)
        decide_size(None, False, cutoff, fallback_cutoff)
        self.fallback_cutoff = fallback_cutoff

    def decide(self, pred, image_size, frame_id, native_patch=None, reset=False):
        result = super().decide(pred, image_size, frame_id, native_patch, reset)
        available = result['size_image_temporal_history_available']
        result['size_accepted'] = decide_size(result['size_image_temporal_risk'],
            available, self.cutoff, self.fallback_cutoff)
        result['size_fallback_decision_version'] = VERSION
        result['size_fallback_decision_cutoff'] = self.cutoff if available else self.fallback_cutoff
        return result


def score(rows, supported_cutoff, fallback_cutoff):
    result = deepcopy(rows)
    for row in result:
        risk = row['risks']['crossfit_fusion']
        if not row['temporal']['available'] and risk != row['risks']['partial_auc4']:
            raise ValueError('Frozen fallback risk identity differs')
        row[METHOD] = decide_size(risk, row['temporal']['available'],
                                 supported_cutoff, fallback_cutoff)
        if row['pred'] is None and (risk is not None or row[METHOD]):
            raise ValueError('Missing output cannot receive a usable flag')
    return result


def evaluate(rows, supported_cutoff, fallback_cutoff):
    base = prior.temporal.base
    groups = {}
    for group, part in base.level.metrics.grouped(rows).items():
        ids = {r['image'] for r in part if r[METHOD]}
        summary = base.level.summarize(part, ids)
        controls = {m: dict(same_count=base.level.point(part, m, count=len(ids)),
            same_CR=base.level.point(part, m, correct=summary['states']['CR'])) for m in CONTROLS}
        actual = {m: base.level.summarize(part, base.accepted(part, m,
            supported_cutoff if m == 'crossfit_fusion' else fallback_cutoff))
            for m in ('crossfit_fusion', 'partial_auc4')}
        changes = {}
        for m, cutoff in (('crossfit_fusion', supported_cutoff), ('partial_auc4', fallback_cutoff)):
            changes[m] = [dict(image=r['image'], history_available=r['temporal']['available'],
                wrong=base.level.metrics.bad(r), before=r['pred'] is not None and r['risks'][m] <= cutoff,
                after=r[METHOD]) for r in part
                if (r['pred'] is not None and r['risks'][m] <= cutoff) != r[METHOD]]
        regimes = {k: sum(r['image'] in ids and base.level.metrics.bad(r) is True and
            ('unsupported' if not r['temporal']['available'] else
             'low' if r['temporal']['primary'] <= base.LEVEL_CUTOFF else 'high') == k for r in part)
            for k in ('low', 'high', 'unsupported')}
        groups[group] = dict(candidate=summary, controls=controls, frozen_actual=actual,
                             changes=changes, FA_regimes=regimes)
    return dict(groups=groups, center=base.level.metrics.center(rows))


def gate(evaluation):
    # Retain the preceding experiment's strict group/continuity requirements.
    return prior.gate({prior.METHOD: evaluation})
