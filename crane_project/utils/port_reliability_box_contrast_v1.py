"""Fixed same-image size supervision; no online box correction or GT input."""
from copy import deepcopy
import numpy as np
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_feature_source_v1 as source
from crane_project.utils import port_reliability_readout_compare_v1 as previous
from crane_project.utils import port_reliability_feature_ablation_v1 as ab

VERSION = 'port_reliability_box_contrast_v1'
ARMS = ('original', 'contrast')
CONTROLS = ('original', 'full_simple', 'score_only')
SETTINGS = dict(source.SETTINGS, auxiliary_weight=.25,
                multipliers=[.85, .95, 1.05, 1.15], modes=['long', 'short', 'both'],
                roi_size=9, context_multiplier=1.5, feature_channels=256,
                raw_dimensions=2304, maximum_contrasts=12)
fit_pca, project, features = source.fit_pca, source.project, source.features
normalization, normalize = source.normalization, source.normalize
numpy_logits, decide, describe = source.numpy_logits, source.decide, source.describe
class_weights = source.class_weights


def training_boxes(pred):
    """GT-free generation in raw w/h representation; never mutate delivered M.

    Return original first, then valid long/short/both variants in fixed order.
    The original angle/center/score are byte-preserved. Reject axis inversion,
    do not silently swap dimensions and thereby alter the fixed long direction.
    """
    p = simple.prediction(pred)
    if p is None:
        return np.empty((0, 6)), []
    long_index, short_index = (2, 3) if p[2] >= p[3] else (3, 2)
    result, keys = [p.copy()], ['M']
    for mode in SETTINGS['modes']:
        for multiplier in SETTINGS['multipliers']:
            b = p.copy()
            if mode in ('long', 'both'): b[long_index] *= multiplier
            if mode in ('short', 'both'): b[short_index] *= multiplier
            if (not np.isfinite(b).all() or min(b[2:4]) <= 0
                    or b[long_index] < b[short_index]):
                continue
            result.append(b); keys.append(mode+'_'+str(multiplier))
    return np.stack(result), keys


def offline_labels(boxes, gt):
    """Existing canonical <=10% label, no GT-conditioned candidate generation."""
    b = np.asarray(boxes)
    if b.ndim != 2 or b.shape[1] != 6:
        raise ValueError('Expected Nx6 original-coordinate boxes')
    return np.array([simple.geometry_errors(gt, p)['size_max_relative'] > .1
                     for p in b], dtype=np.float32)


def pool_reference(roi):
    a = np.asarray(roi, dtype=np.float64)
    if a.ndim != 4 or a.shape[1:] != (256, 9, 9) or not np.isfinite(a).all():
        raise ValueError('Expected finite N256x9x9 detached P3 ROI')
    return a.reshape(len(a), 256, 3, 3, 3, 3).mean((3, 5)).reshape(len(a), 2304)


def gate(statistics):
    renamed = deepcopy(statistics)
    for value in renamed.values():
        for key in ('control_tie_bounds', 'matched_CR_controls'):
            value[key]['temperature'] = value[key].pop('original')
    result = previous.gate(renamed)
    for item in result['failures']:
        if item.get('control') == 'temperature': item['control'] = 'original'
    result['selected_arm'] = 'contrast' if result['passed'] else None
    return result


def contrast_diagnostic(logits, labels, mask):
    """Same-image good/bad ordering is a mechanism check, not VAL evidence."""
    a, y, m = np.asarray(logits), np.asarray(labels), np.asarray(mask, dtype=bool)
    if a.shape != y.shape or m.shape != a.shape or a.ndim != 2:
        raise ValueError('Invalid contrast diagnostic')
    pairs = correct = ties = frames = 0
    for scores, targets, valid in zip(a, y, m):
        good, bad = scores[valid & (targets == 0)], scores[valid & (targets == 1)]
        if len(good) and len(bad):
            difference = bad[:, None]-good[None, :]
            pairs += difference.size; correct += int((difference > 0).sum())
            ties += int((difference == 0).sum()); frames += 1
    return dict(frames_with_both_classes=frames, pairs=int(pairs), correctly_ordered=correct,
                ties=ties, pair_AUROC=(correct+.5*ties)/pairs if pairs else None,
                synthetic_mechanism_only=True)


def hidden_diagnostic(rows, ids, matrix, model, normalizer):
    """Use this head's 259-dimensional normalization, not old 291-D stem."""
    hidden=normalize(matrix,normalizer)[:,1:]
    for layer in (0,2):
        hidden=np.maximum(hidden@np.asarray(model['network.%d.weight'%layer]).T+model['network.%d.bias'%layer],0.)
    output=(hidden@np.asarray(model['network.4.weight']).T+model['network.4.bias'])[:,0]
    zero=np.all(hidden==0,axis=1);lookup={r['image']:r for r in rows};indices={name:i for i,name in enumerate(ids)}
    result={}
    for name,group in ab.groups(rows).items():
        selected=[indices[r['image']] for r in group if r['pred'] is not None]
        result[name]=dict(outputs=len(selected),bad=sum(ab.size_bad(lookup[ids[i]]) for i in selected),
            second_hidden_all_zero=int(zero[selected].sum()),
            zero_bad=sum(bool(zero[i]) and ab.size_bad(lookup[ids[i]]) for i in selected),
            raw_output_range=[float(output[selected].min()),float(output[selected].max())] if selected else None)
    return result
