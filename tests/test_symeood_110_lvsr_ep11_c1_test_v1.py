"""Disambiguate the new ep11-start C1 from shared ep11 and old ep7-start C1."""
import unittest
from unittest.mock import patch
from crane_project.tools import eval_symeood_110_lvsr_ep11_c1_test_v1 as evaluation


class FixedC1Identity(unittest.TestCase):
    def payload(self):
        return dict(epoch=1, arm='C', parent_epoch=11,
            parent_sha256='cb0d65914a073306559a6e5ebcb5a4186676922cb4ae9cb23dff39e202cdb80a')

    def test_accepts_only_ep11_start_C1(self):
        evaluation.validate_payload(self.payload())
        for changes in (dict(parent_epoch=7), dict(epoch=11, arm='fixed_b'),
                        dict(epoch=6), dict(parent_sha256='wrong-parent')):
            with self.assertRaises(ValueError):
                evaluation.validate_payload(dict(self.payload(), **changes))

    def test_report_keys_restore_after_failure(self):
        existing = evaluation.existing
        heads, arms = existing.HEADS, existing.ARMS
        def fail(*args):
            self.assertEqual(existing.ARMS, ('b', 'midpoint', 'ep11_c1'))
            self.assertEqual(set(existing.HEADS), {'ep11_c1'})
            raise RuntimeError('report failure')
        with patch.object(existing, 'depth_report', fail):
            with self.assertRaises(RuntimeError):
                evaluation.depth_report([], [], {}, {})
        self.assertIs(existing.HEADS, heads)
        self.assertIs(existing.ARMS, arms)

    def test_VAL_alias_uses_C1_prediction_field(self):
        box = [1., 2., 3., 4., .5, .9]
        ref = evaluation.val_reference(dict(image='frame', edge_residual=box))
        self.assertEqual(ref['ep11_c1'], box)
        self.assertNotIn('ep11', ref)


if __name__ == '__main__':
    unittest.main()
