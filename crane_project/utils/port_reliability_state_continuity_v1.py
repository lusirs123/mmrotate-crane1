"""Offline size-state continuity. Missing outputs never enter confusion counts."""
from collections import Counter
import math

STATES = ('FA', 'FR', 'ED', 'CR', 'MISSING')
RUN_KINDS = dict(incorrect_acceptance=('FA',), correct_rejection=('FR',),
                 error_detection=('ED',), correct_retention=('CR',),
                 missing_output=('MISSING',), unavailable=('FR', 'ED', 'MISSING'))


def size_state(row, accepted):
    if not isinstance(accepted, bool):
        raise ValueError('Acceptance must be boolean')
    pred = row['pred']
    if pred is None:
        if accepted:
            raise ValueError('A missing output cannot be accepted')
        return 'MISSING'
    gt = row['gt']
    edges = sorted(pred[2:4]), sorted(gt[2:4])
    if any(not math.isfinite(x) or x <= 0 for pair in edges for x in pair):
        raise ValueError('Size assessment requires finite positive edges')
    # Same canonical-edge and relative-error contract as the sealed evaluator.
    bad = max(abs(b-g)/g for b, g in zip(*edges)) > .1
    return ('FA' if bad else 'CR') if accepted else ('ED' if bad else 'FR')


def checked_order(rows):
    ids, positions = set(), set()
    for row in rows:
        if row['split'] not in ('train', 'train_sim', 'val'):
            raise ValueError('This supplemental review must not read TEST')
        expected_role = 'val' if row['split'] == 'val' else 'train'
        if row.get('reliability_role', expected_role) != expected_role:
            raise ValueError('Source split and reliability role disagree')
        pos = row['sequence'], row['frame_id']
        if row['image'] in ids or pos in positions or not isinstance(row['frame_id'], int):
            raise ValueError('Duplicate identity/position or invalid frame index')
        ids.add(row['image']); positions.add(pos)
    return sorted(rows, key=lambda r: (r['sequence'], r['frame_id'], r['image']))


def summarize(rows, accepted):
    ordered = checked_order(rows)
    ids = {r['image'] for r in rows}
    if not set(accepted) <= ids:
        raise ValueError('Acceptance contains an unknown image')
    states = [(r, size_state(r, r['image'] in accepted)) for r in ordered]
    counts = {key: sum(s == key for _, s in states) for key in STATES}
    runs = {}
    for kind, members in RUN_KINDS.items():
        segments = []; active = None; previous = None
        for row, state in states:
            adjacent = (previous is not None and row['sequence'] == previous['sequence']
                        and row['frame_id'] == previous['frame_id']+1)
            if not adjacent or state not in members:
                if active is not None:
                    segments.append(active); active = None
            if state in members:
                if active is None:
                    active = dict(sequence=row['sequence'], start_frame=row['frame_id'],
                                  end_frame=row['frame_id'], length=0,
                                  state_counts={s: 0 for s in STATES})
                active['end_frame'] = row['frame_id']; active['length'] += 1
                active['state_counts'][state] += 1
            previous = row
        if active is not None:
            segments.append(active)
        longest = max((r['length'] for r in segments), default=0)
        runs[kind] = dict(longest=longest, segment_count=len(segments),
                          total_frames=sum(r['length'] for r in segments),
                          length_histogram=dict(sorted(Counter(r['length'] for r in segments).items())),
                          longest_segments=[r for r in segments if r['length'] == longest],
                          segments=segments)
    n = len(rows); outputs = n-counts['MISSING']; good = counts['CR']+counts['FR']
    kept = counts['CR']+counts['FA']
    adjacent_pairs = switches = 0
    for (a, sa), (b, sb) in zip(states, states[1:]):
        if a['sequence'] == b['sequence'] and a['frame_id']+1 == b['frame_id']:
            adjacent_pairs += 1
            switches += int((sa in ('FA', 'CR')) != (sb in ('FA', 'CR')))
    ratio = lambda a, b: a/b if b else None
    return dict(frames=n, outputs=outputs, states=counts,
                accepted_outputs=kept, output_coverage=ratio(outputs, n),
                acceptance_coverage_all_frames=ratio(kept, n),
                correct_coverage_all_frames=ratio(counts['CR'], n),
                good_retention=ratio(counts['CR'], good),
                accepted_error_fraction=ratio(counts['FA'], kept), runs=runs,
                adjacent_observed_pairs=adjacent_pairs, flag_switches=switches,
                flag_switch_rate=ratio(switches, adjacent_pairs),
                missing_excluded_from_confusion=True, gaps_reset_runs=True,
                observed_frame_adjacency_only=True)


def compare(rows, accepted, risks, control_names=('full_simple', 'score_only')):
    """Offline same-count controls, label-free image tie order; no deployment."""
    main = summarize(rows, accepted)
    controls = {}
    ids = [r['image'] for r in rows if r['pred'] is not None]
    k = main['accepted_outputs']
    for name in control_names:
        values = risks[name]
        if any(not math.isfinite(values[i]) for i in ids):
            raise ValueError('Nonfinite control risk')
        ranked = sorted(ids, key=lambda i: (values[i], i))
        controls[name] = summarize(rows, set(ranked[:k]))
        if 0 < k < len(ids) and values[ranked[k-1]] == values[ranked[k]]:
            boundary = values[ranked[k-1]]
            controls[name]['partial_boundary_tie'] = dict(
                risk=boundary, tie_count=sum(values[i] == boundary for i in ids),
                selection='image_order_only_not_GT', continuity_order_sensitive=True)
    return dict(actual=main, same_count_controls=controls,
                comparison_is_offline=True, candidate_selected=False)
