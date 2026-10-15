import unittest
from crane_project.tools import eval_port_reliability_response_warm_test_v1 as supplement

class FrozenSupplementTest(unittest.TestCase):
    def test_seed_and_thresholds_preselected(self):
        models={k:{'sentinel':k} for k in supplement.METHODS[2:]}
        models['soft_1702']={'sentinel':'excluded'}
        cuts={k:i/10 for i,k in enumerate(models)}
        cuts.update(formal=.7,score=.8)
        chosen, frozen=supplement.selected(models,cuts)
        self.assertEqual(tuple(chosen),supplement.METHODS[2:])
        self.assertNotIn('soft_1702',chosen)
        self.assertEqual(frozen,{k:cuts[k] for k in supplement.METHODS})
        self.assertEqual(supplement.SEED,1701)

    def test_missing_frozen_method_fails(self):
        with self.assertRaises(KeyError): supplement.selected({}, {})
