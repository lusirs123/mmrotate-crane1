"""Prevent shared or continued weights from being reported as pair ep11."""
import unittest
from unittest.mock import patch
from crane_project.tools import eval_symeood_110_lvsr_pair_ep11_test_v1 as evaluation


class FixedPairIdentity(unittest.TestCase):
    def payload(self):
        return dict(protocol='symeood_110_lvsr_pair_train_v1', epoch=11,
            parent_checkpoint=None, updates=7029, updates_per_epoch=639)

    def test_only_original_pair_ep11(self):
        evaluation.validate_payload(self.payload())
        for changes in (dict(protocol='symeood_110_lvsr_v1'), dict(epoch=24),
                        dict(parent_checkpoint='ep11_C1'), dict(updates=0)):
            with self.assertRaises(ValueError):
                evaluation.validate_payload(dict(self.payload(), **changes))

    def test_VAL_alias_is_pair_prediction(self):
        ref = evaluation.val_reference(dict(edge_residual=[1, 2, 3, 4, .5, .9]))
        self.assertEqual(ref['pair_ep11'], ref['edge_residual'])
        self.assertNotIn('ep11_c1', ref)

    def test_report_state_restores_on_failure(self):
        existing = evaluation.existing
        heads, arms = existing.HEADS, existing.ARMS
        def fail(*args):
            self.assertEqual(set(existing.HEADS), {'pair_ep11'})
            self.assertEqual(existing.ARMS, ('b', 'midpoint', 'pair_ep11'))
            raise RuntimeError('report failure')
        with patch.object(existing, 'depth_report', fail):
            with self.assertRaises(RuntimeError):
                evaluation.depth_report([], [], {}, {})
        self.assertIs(existing.HEADS, heads)
        self.assertIs(existing.ARMS, arms)


if __name__ == '__main__':
    unittest.main()
