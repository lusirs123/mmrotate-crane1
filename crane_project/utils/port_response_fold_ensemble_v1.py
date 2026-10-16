"""Frozen scorer inference only. Inputs are cached/current 19D descriptors, never GT."""
import numpy as np
from crane_project.utils import port_reliability_response_stats_v1 as response
from crane_project.utils import port_simple_component_reliability_v1 as simple

VERSION = 'port_response_fold_ensemble_v1'
NAMES = ('formal', 'bce_1701', 'soft_1701', 'bce0_1701', 'soft0_1701')
SEQUENCES = ('real_seq01', 'real_seq05', 'real_seq06', 'real_seq12', 'real_seq13', 'sim_seq08')
SETTINGS = dict(seed=1701, aggregation='sigmoid(mean(six fold final fusion logits))',
    simple_control='sigmoid(mean(six fold original simple logits))', folds=list(SEQUENCES),
    new_fits=0, CR=653, Real_CR=189, equal_weights=True, subset_selection=False,
    TRAIN='Original single held-video scores only; not six-fold average; meta was fitted on these labels',
    TEST='All four prespecified methods after VAL cutoffs/judgement freeze; exposed reporting only')


def validate(raw):
    x = np.asarray(raw, dtype=float)
    if x.ndim != 2 or x.shape[1] != 19 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite Nx19 online descriptors required')
    return x


def simple_logits(model, raw):
    x = validate(raw)
    return model['weights'][0]+((x[:, :3]-model['mean'])/model['scale'])@np.asarray(model['weights'][1:])


def base_logits(models, raw):
    x = validate(raw)
    return np.column_stack([simple_logits(models[n], x) if n == 'formal'
                            else response.numpy_logits(models[n], x) for n in NAMES])


def fusion_logits(meta, mapped):
    x = response.normalize(mapped, meta['normalizer'])
    return meta['weights'][0]+x@np.asarray(meta['weights'][1:])


def fold_simple(raw, folds):
    if set(folds) != set(SEQUENCES):
        raise ValueError('All six frozen folds required')
    return simple.sigmoid(np.mean([simple_logits(folds[s]['models']['formal'], raw) for s in SEQUENCES], axis=0))


def full_fusion(raw, bundle):
    z = base_logits(bundle['full'], raw)
    return simple.sigmoid(fusion_logits(bundle['meta']['fusion_oof'], response.normalize(z, bundle['full_mapping'])))


def ensemble(raw, folds, meta):
    if set(folds) != set(SEQUENCES):
        raise ValueError('All six frozen folds required')
    z = np.stack([base_logits(folds[s]['models'], raw) for s in SEQUENCES])
    mapped = np.stack([response.normalize(a, folds[s]['mapping']) for s, a in zip(SEQUENCES, z)])
    h = np.stack([fusion_logits(meta, a) for a in mapped])
    result = dict(fold_simple=simple.sigmoid(z[:, :, 0].mean(0)), fold_fusion=simple.sigmoid(h.mean(0)))
    return result, dict(fold_base_logits=z, fold_mapped=mapped, fold_fusion_logits=h)


def held_training(held_logits, held_mapped, meta):
    """Preserve previously generated single held-video TRAIN scores, no averaging."""
    if held_logits.shape != held_mapped.shape or held_logits.ndim != 2 or held_logits.shape[1] != 5:
        raise ValueError('Original paired five held-out scores required')
    return dict(fold_simple=simple.sigmoid(held_logits[:, 0]),
                fold_fusion=simple.sigmoid(fusion_logits(meta, held_mapped)))
