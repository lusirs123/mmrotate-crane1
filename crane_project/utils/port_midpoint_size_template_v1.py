"""Frozen size-reader assessment; all working points are offline diagnostics.

GT supplies error labels only. Reader availability is not a correctness label.
The reference-held TRAIN role is held out from the reference learner, not B or
the simple risk models. There is no training, threshold fitting or TEST route.
"""
from collections import Counter
from copy import deepcopy
import math

import numpy as np

from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_separability_v1 as rank
from crane_project.utils import port_size_reference_v1 as size
from crane_project.utils import port_midpoint_reliability_v1 as midpoint

VERSION = 'port_midpoint_size_template_v1'
METHODS = ('score', 'simple', 'moments', 'template')
READERS = ('moments', 'template')


def paired_prediction(expected, actual, image, atol=1e-4, rtol=1e-6):
    """Do not replace locked formal geometry with fresh numerical variants."""
    a, b = simple.prediction(expected), simple.prediction(actual)
    if (a is None) != (b is None):
        raise ValueError('Online formal output availability differs: '+image)
    if a is None:
        return 0.
    if a[5] != b[5] or not np.allclose(a[:5], b[:5], atol=atol, rtol=rtol):
        raise ValueError('Online formal box/score differs from locked collection: '+image)
    return float(np.max(np.abs(a[:5]-b[:5])))


def online_readings(probability, final_box, image_size, meta, runtime, settings):
    """No GT, role, domain, sequence or history is accepted by this function."""
    before = deepcopy(final_box)
    decisions = {m: runtime.decide(final_box, image_size, m)
                 for m in ('raw', 'score_only', 'simple')}
    if any(d['final_box_original'] != before or d['center_accepted'] != (before is not None)
           for d in decisions.values()):
        raise ValueError('Existing reliability changed final box or center retention')
    references, readings = {}, {}
    for method in READERS:
        function = size.reference_from_map if method == 'moments' else midpoint.template_reference
        references[method] = (function(probability, final_box[:5], meta, settings)
            if final_box is not None else dict(defined=False, reason='no_detector_output'))
        readings[method] = (size.size_reading(final_box[:5], references[method], settings['risk_scale'])
            if final_box is not None else dict(defined=False, risk=None,
                long_log_ratio=None, short_log_ratio=None))
    if final_box != before:
        raise ValueError('Reader changed its final-box context')
    risks = dict(score=decisions['score_only']['risks']['size'],
                 simple=decisions['simple']['risks']['size'],
                 **{m: readings[m]['risk'] for m in READERS})
    return dict(references=references, readings=readings, risks=risks,
                frozen_decisions=decisions, new_reader_flags_created=False)


def record(source, evidence, role):
    """Attach offline labels AFTER image-only reading and fixed simple scoring."""
    if role not in ('reference_holdout_train', 'val'):
        raise ValueError('Only fixed reference-held TRAIN and VAL allowed')
    if source['split'] not in (('train', 'train_sim') if role != 'val' else ('val',)):
        raise ValueError('Assessment role differs from source split')
    output = source['pred'] is not None
    errors = simple.geometry_errors(source['gt'], source['pred']) if output else None
    g = simple.canonical(source['gt'])
    reference_errors = {}
    for method in READERS:
        ref = evidence['references'][method]
        if ref['defined']:
            delta = np.array([ref['long_original_px'], ref['short_original_px']])/g[2:4]-1
            reference_errors[method] = dict(long_ratio=float(delta[0]+1),
                short_ratio=float(delta[1]+1), max_relative=float(np.abs(delta).max()))
        else:
            reference_errors[method] = None
    return dict(deepcopy(source), **deepcopy(evidence), assessment_role=role,
        offline_errors=errors, size_bad=None if errors is None else errors['size_max_relative'] > .1,
        reference_errors=reference_errors, GT_used_only_offline=True)


def distribution(values):
    a = np.asarray(values, float)
    return dict(count=len(a), mean=float(a.mean()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,
        p90=float(np.percentile(a, 90)) if len(a) else None,
        maximum=float(a.max()) if len(a) else None,
        within_10_percent=int(np.sum(a <= .1)))


def state_metrics(values, accepted, frames):
    """Names explicitly distinguish bad acceptance from false rejection of good."""
    stats = rank.confusion([r['size_bad'] for r in values], accepted, frames, frames)
    stats['good_false_rejected'] = stats.pop('correct_rejected')
    stats['bad_correctly_rejected'] = stats.pop('incorrect_rejected')
    stats['bad_accepted'] = stats.pop('incorrect_accepted')
    stats['good_accepted'] = stats.pop('correct_accepted')
    return stats


def subset_report(output, support, methods, frames, fractions):
    """Same-support ranks AND same-count diagnostics, without label-based choices."""
    values = [output[i] for i in support]
    images = [r['image'] for r in values]
    bad = [r['size_bad'] for r in values]
    risks = {m: [r['risks'][m] for r in values] for m in methods}
    stats = dict(support_outputs=len(values), support_bad=sum(bad),
        support_good=len(bad)-sum(bad), support_images_sha256=simple.fingerprint(images),
        rank={m: rank.rank_metrics(risks[m], bad) for m in methods},
        same_count_points=[])
    for fraction in fractions:
        requested = min(len(output), math.ceil(fraction*frames))
        count = min(requested, len(values))
        matched, full = {}, {}
        for method in methods:
            order = sorted(range(len(values)), key=lambda i: (risks[method][i], images[i]))
            accepted = set(order[:count])
            metric = state_metrics(values, [i in accepted for i in range(len(values))], frames)
            metric['tie_bounds'] = rank.same_count_reference(risks[method], bad, images, [count])[count]
            metric['accepted_images_sha256'] = simple.fingerprint(sorted(images[i] for i in accepted))
            matched[method] = metric
            # Separate end-to-end view: undefined outputs count as rejected.
            # Score/simple may use all outputs at the same accepted count here.
            population = output if method in ('score', 'simple') else values
            full_order = sorted(range(len(population)),
                key=lambda i: (population[i]['risks'][method], population[i]['image']))
            kept = {population[i]['image'] for i in full_order[:count]}
            full[method] = state_metrics(output, [r['image'] in kept for r in output], frames)
            full[method]['tie_bounds_on_available_population'] = rank.same_count_reference(
                [r['risks'][method] for r in population], [r['size_bad'] for r in population],
                [r['image'] for r in population], [count])[count]
        stats['same_count_points'].append(dict(nominal_all_frame_fraction=fraction,
            requested_count=requested, actual_count=count,
            availability_limited=count < requested, methods=matched,
            all_output_same_count_availability_as_reject=full,
            diagnostic_only_not_deployment_cutoffs=True))
    return stats


def summarize_group(values, fractions):
    output = [r for r in values if r['pred'] is not None]
    hits = sum(r['offline_errors']['center_px'] < 15 for r in output)
    common = [i for i, r in enumerate(output) if all(r['references'][m]['defined'] for m in READERS)]
    stats = dict(frames=len(values), outputs=len(output), missing_outputs=len(values)-len(output),
        center_hits=hits, output_coverage=rank.ratio(len(output), len(values)),
        center_hit_rate_on_outputs=rank.ratio(hits, len(output)),
        all_frame_center_correct_coverage=rank.ratio(hits, len(values)),
        raw_size_bad=sum(r['size_bad'] for r in output),
        center_flag_retained_on_every_output=True,
        full_output_baselines=subset_report(output, list(range(len(output))),
            ('score', 'simple'), len(values), fractions),
        common_defined=subset_report(output, common, METHODS, len(values), fractions), readers={},
        frozen_workpoints={})
    for method in ('score_only', 'simple'):
        stats['frozen_workpoints'][method] = state_metrics(output,
            [r['frozen_decisions'][method]['size_accepted'] for r in output], len(values))
    for method in READERS:
        support = [i for i, r in enumerate(output) if r['references'][method]['defined']]
        unavailable = [r for r in output if not r['references'][method]['defined']]
        stats['readers'][method] = dict(defined=len(support),
            availability_on_outputs=rank.ratio(len(support), len(output)),
            availability_on_all_frames=rank.ratio(len(support), len(values)),
            unavailable_outputs=len(unavailable), unavailable_good=sum(not r['size_bad'] for r in unavailable),
            unavailable_bad=sum(r['size_bad'] for r in unavailable),
            unavailable_reasons=dict(Counter(r['references'][method]['reason'] for r in unavailable)),
            reference_error_relative_to_GT=distribution([output[i]['reference_errors'][method]['max_relative'] for i in support]),
            common_reference_error_relative_to_GT=distribution([output[i]['reference_errors'][method]['max_relative'] for i in common]),
            own_defined=subset_report(output, support, ('score', 'simple', method), len(values), fractions))
    return stats


def summarize(records, fractions=(.9, .95)):
    if len({r['image'] for r in records}) != len(records):
        raise ValueError('Duplicate image identity in full assessment')
    result = {}
    for role in ('reference_holdout_train', 'val'):
        values = [r for r in records if r['assessment_role'] == role]
        if not values:
            raise ValueError('Both fixed assessment roles are required')
        groups = {'all': values}
        for field in ('domain', 'sequence'):
            groups.update({field+':'+v: [r for r in values if r[field] == v]
                for v in sorted({r[field] for r in values})})
        result[role] = {name: summarize_group(rows, fractions) for name, rows in groups.items()}
    return result
