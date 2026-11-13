"""Freeze the all-epoch-ranked checkpoint before exactly one TEST run."""
from copy import deepcopy
import unittest
from crane_project.tools import eval_lvsr_decoded_edge_proxy_fixed_test_v1 as e
from crane_project.tools import train_lvsr_shared_spatial_v1 as t


def fixture():
    events = [dict(epoch=i, eligibility_vs_L5=dict(pass_all=(i == 8)),
                   selection_key=[-680 if i == 5 else -670, i],
                   checkpoint=dict(path='head_epoch_%02d.pth' % i,
                                   sha256=('%064x' % i), epoch=i)) for i in range(1, 25)]
    events[4]['checkpoint'] = deepcopy(e.EXPECTED_CHECKPOINT)
    selection = t.choose(events)
    cp = selection['representative_checkpoint']
    summary = dict(selection=selection, epochs=events, representative=events[4], TEST=False)
    complete = dict(passed=True, epochs_completed=24, formal_optimizer_updates=15336, checkpoint=cp, TEST=False,physical_gpu=3,training_parameters=15440)
    return summary, complete


class FixedTestChecks(unittest.TestCase):
    def test_evaluator_uses_the_current_training_directory_and_single_gpu(self):
        cfg = e.training.check_config()
        self.assertEqual(e.RUN, e.ROOT/cfg['output_root'])
        self.assertEqual(cfg['physical_gpu'], 3)
        self.assertEqual(e.EXPECTED_CHECKPOINT['epoch'], 5)

    def test_ranked_representative_does_not_require_diagnostic_gate(self):
        s, c = fixture()
        cp = e.validate_training(s, c)
        self.assertEqual(cp['epoch'], 5)
        self.assertIsNone(s['selection']['selected_epoch'])
        self.assertEqual(s['selection']['eligible_epochs'], [8])

    def test_rejects_changed_checkpoint_or_posthoc_ranking(self):
        for field, value in (('epoch', 17), ('sha256', 'wrong'), ('path', 'head_epoch_17.pth')):
            s, c = deepcopy(fixture())
            s['selection']['representative_checkpoint'][field] = value
            with self.assertRaises(ValueError): e.validate_training(s, c)
        s, c = fixture(); s['epochs'][-1]['selection_key'] = [-999, 24]
        with self.assertRaises(ValueError): e.validate_training(s, c)
        s, c = fixture(); s['epochs'].pop()
        with self.assertRaises(ValueError): e.validate_training(s, c)

    def test_metrics_keep_missing_output_in_full_frame_denominator(self):
        rr = [dict(image='sim_a_%05d' % i, sequence='sim_a', frame_id=i, domain='sim',
                   gt=[0, 0, 100, 50, 0], prediction=[0, 0, 100, 50, 0, .9] if i == 0 else None)
              for i in range(2)]
        m, _, _, _, _ = e.metric(rr, 'prediction')
        self.assertEqual(m['frames'], 2); self.assertEqual(m['outputs'], 1)
        self.assertEqual(m['joint10'], 1); self.assertEqual(m['RIoU'], .5)
        self.assertEqual(m['long_MAE_pct'], 0.); self.assertEqual(m['short_MAE_pct'], 0.)
        self.assertEqual(m['MCML_max'], 1)


if __name__ == '__main__': unittest.main()
