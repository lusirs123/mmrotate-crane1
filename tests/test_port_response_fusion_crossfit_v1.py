import copy
import unittest
import numpy as np
from crane_project.utils import port_response_fusion_crossfit_v1 as core
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_response_stats_v1 as response
from crane_project.utils import port_reliability_response_tail_v1 as tail
from crane_project.utils.port_reliability_response_stats_v1_torch import Risk


class CrossfitTest(unittest.TestCase):
    def test_whole_video_partition_and_material_isolation(self):
        seq = np.repeat(core.SEQUENCES, 4)
        ownership = np.zeros(len(seq), dtype=int)
        raw = np.random.RandomState(1701).normal(size=(len(seq), 19))
        labels = np.tile([0, 1, 0, 0], 6)
        for s, fit, held in core.folds(seq):
            self.assertFalse(set(fit) & set(held)); self.assertEqual(set(seq[held]), {s})
            ownership[held] += 1
            baseline = simple.fit_linear_risk(raw[fit, :3], labels[fit], core.LINEAR)
            changed = raw.copy(); changed[held] += 10000
            self.assertEqual(baseline, simple.fit_linear_risk(changed[fit, :3], labels[fit], core.LINEAR))
            self.assertEqual(core.mapping_fit(raw[fit]), core.mapping_fit(changed[fit]))
        np.testing.assert_array_equal(ownership, 1)
        with self.assertRaises(ValueError):
            core.folds(seq[:-4])

    def test_convex_fusion_export_gradient_and_scale(self):
        rng = np.random.RandomState(1701)
        x = rng.normal(size=(100, 5)); y = (x[:, 0]+x[:, 1] > .5).astype(float)
        fitted = core.fit_linear(x, y)
        self.assertEqual(fitted['parameters'], 6)
        self.assertLess(fitted['max_abs_gradient'], 1e-8)
        shifted = core.fit_linear(x*np.arange(1, 6)+100, y)
        np.testing.assert_allclose(core.fusion_risk(fitted, x), core.fusion_risk(shifted, x*np.arange(1, 6)+100), atol=1e-10, rtol=0)
        summary = core.conditional_summary(np.tile(x[:, :1], (1, 5)), y, np.repeat('real_one', 100))
        good = x[y == 0, 0]; wrong = x[y == 1, 0]
        expected = sum(sum(v > a for a in good)+.5*sum(v == a for a in good) for v in wrong)/(len(good)*len(wrong))
        self.assertAlmostEqual(summary['all']['AUROC']['formal'], expected)

    def test_continuation_parent_and_export(self):
        rng = np.random.RandomState(1701); x = rng.normal(size=(20, 19)); y = np.tile([0., 1.], 10)
        obj = Risk(1701, 19)
        parent = dict(normalizer=response.normalize_fit(x, 'TRAIN'), final_state_sha256=tail.state_digest(obj),
                      weights={k: v.detach().tolist() for k, v in obj.state_dict().items()})
        a = core.continuation(x, y, parent, 'bce', .1, .1, 'cpu', steps=2)
        self.assertNotEqual(a['initial_state_sha256'], a['final_state_sha256'])
        self.assertEqual(a['initial_state_sha256'], parent['final_state_sha256'])
        bad = copy.deepcopy(parent); bad['normalizer']['mean'][0] += 1
        with self.assertRaises(ValueError):
            core.continuation(x, y, bad, 'bce', .1, .1, 'cpu', steps=2)

    def test_flags_and_missing_preserved(self):
        original = dict(final_box_original=[1, 2, 10, 5, 0, .5], center_accepted=True,
                        size_accepted=True, angle_accepted=False, arbitrary={'preserved': 1})
        changed = response.size_decision(original, .8, .5)
        self.assertFalse(changed['size_accepted'])
        changed['size_accepted'] = True
        self.assertEqual(changed, original)
        missing = dict(original, final_box_original=None, center_accepted=False, size_accepted=False)
        self.assertEqual(response.size_decision(missing, None, .5), missing)


if __name__ == '__main__':
    unittest.main()
