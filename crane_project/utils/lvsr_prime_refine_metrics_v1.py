"""Fixed10% boundary audit using exactly the original row-metric semantics."""
from collections import defaultdict

from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import row_metrics


def boundary_pairs(before, after):
    key = lambda r: (r['image'], r['scale'])
    index = {key(r): r for r in before}
    if len(index) != len(before) or len(after) != len(index) or set(index) != set(map(key, after)):
        raise ValueError('Unique, identical frame/view sets required')
    groups = defaultdict(list)
    for b in after:
        a = index[key(b)]
        if any(a[k] != b[k] for k in ('gt', 'b', 'midpoint', 'sequence', 'domain')):
            raise ValueError('Frozen GT/R4/M identity differs')
        old = row_metrics(a['gt'], a['edge_residual'], 0.)
        new = row_metrics(b['gt'], b['edge_residual'], 0.)
        transition = ('pass' if old['joint_size10'] else 'fail')+'_'+('pass' if new['joint_size10'] else 'fail')
        entry = dict(transition=transition)
        if old['output'] and new['output']:
            errors = [new[edge+'_relative_error'] for edge in ('long', 'short')]
            entry.update(old_worst=max(old['long_relative_error'], old['short_relative_error']), new_worst=max(errors),
                         long_crossed=new['long_relative_error'] > .1 and old['long_relative_error'] <= .1,
                         short_crossed=new['short_relative_error'] > .1 and old['short_relative_error'] <= .1)
        for group in ('overall', b['domain'], b['sequence']):
            groups[group].append(entry)
    result = {}
    for name, entries in groups.items():
        broken = [r for r in entries if r['transition'] == 'pass_fail']
        numeric = [r for r in broken if 'old_worst' in r]
        result[name] = dict(frames=len(entries), repairs=sum(r['transition'] == 'fail_pass' for r in entries),
            broken=len(broken), broken_new_error_le11pct=sum(r.get('new_worst', 1.) <= .11 for r in broken),
            broken_new_error_le12pct=sum(r.get('new_worst', 1.) <= .12 for r in broken),
            broken_short_crossings=sum(r.get('short_crossed', False) for r in broken),
            broken_long_crossings=sum(r.get('long_crossed', False) for r in broken),
            broken_numeric_outputs=len(numeric),
            broken_old_margin_mean=sum(.1-r['old_worst'] for r in numeric)/len(numeric) if numeric else None,
            broken_new_excess_mean=sum(r['new_worst']-.1 for r in numeric)/len(numeric) if numeric else None)
    return result
