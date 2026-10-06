"""Simple independent flags bound to the sealed sigma1.5/epoch03 frontend.

Online input is exclusively the delivered original-coordinate OBB+score and
original image size. Neither annotations nor temporal/domain state enter it.
"""
from copy import deepcopy

from crane_project.utils import port_simple_component_reliability_v1 as simple

VERSION = 'port_midpoint_sigma15_reliability_v1'
B_SHA = '8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23'
HEAD_SHA = '16c2fb448ac4e1c53530b8086d547f6f6ccb8d6b0763a42391c34f9b337982d7'


def checked_frontend(front_end):
    selected = front_end['midpoint_checkpoint']
    if (front_end['frozen_b']['checkpoint_sha256'] != B_SHA or
            selected['sha256'] != HEAD_SHA or selected['path'] != 'head_epoch_03.pth' or
            selected['epoch'] != 3 or selected['sigma_cells'] != 1.5 or
            selected['updates'] != 2706):
        raise ValueError('Requires frozen B24 + sigma1.5/epoch03')


class Sigma15Reliability:
    def __init__(self, policy, front_end):
        checked_frontend(front_end)
        if policy.get('protocol') != VERSION or policy.get('front_end') != front_end:
            raise ValueError('Policy belongs to another frontend; refit on matched TRAIN/VAL')
        self.base = simple.SimpleComponentReliability(policy['simple_policy'])

    def decide(self, final_box_original, image_size, method='simple'):
        value = self.base.decide(final_box_original, image_size, method)
        value['final_box_original'] = value.pop('raw_b_output')
        return value


def evaluate(rows, policy, old_simple):
    """Offline same-frontend controls; legacy policy is never deployed here."""
    decisions, summaries = simple.evaluate(rows, policy['simple_policy'])
    _, transferred = simple.evaluate(rows, old_simple)
    old_runtime = simple.SimpleComponentReliability(old_simple)
    old_risks = {r['image']: old_runtime.decide(r['pred'], r['image_size'])['risks']
                 for r in rows}
    indexed = {r['image']: r['methods'] for r in decisions}
    for group, summary in summaries.items():
        if group == 'all':
            values = rows
        else:
            field, name = group.split(':', 1)
            values = [r for r in rows if r[field] == name]
        for component in simple.COMPONENTS:
            stats = summary['components'][component]
            stats['old_sigma1_policy_transfer_diagnostic'] = deepcopy(
                transferred[group]['components'][component]['simple'])
            out = [r for r in values if r['pred'] is not None and
                   (component != 'angle' or r['angle_axis_well_defined'])]
            ranked = sorted(out, key=lambda r: (old_risks[r['image']][component], r['image'])) if component != 'center' else out
            count = stats['simple']['accepted_frames']
            stats['matched_old_simple_diagnostic'] = simple._component_stats(
                values, component, {r['image'] for r in ranked[:count]})
            for method, item in list(stats.items()):
                if not isinstance(item, dict):
                    continue
                # Four states are conditional on an output, with missing separate.
                item.update(false_accept=item['incorrect_accepted'],
                    false_reject=item['correct_rejected'], error_detected=item['incorrect_rejected'],
                    correct_retained=item['correct_accepted'],
                    judgement_accuracy_on_outputs=(item['correct_accepted']+item['incorrect_rejected'])/
                        item['output_frames'] if item['output_frames'] else None)
                if method in ('raw', 'score_only', 'simple'):
                    n = sum(indexed[r['image']][method][component+'_accepted'] for r in values)
                    item.update(online_accepted_frames_all=n, online_accepted_coverage_all_frames=n/len(values))
            good = stats['raw']['correct_accepted']
            kept = stats['simple']['accepted_frames']
            summary['components'][component]['coverage_false_accept_lower_bound'] = max(0, kept-good)
    for record in decisions:
        for decision in record['methods'].values():
            decision['final_box_original'] = decision.pop('raw_b_output')
    return decisions, summaries
