"""Bounded CPU tests for fixed selection, native Top1 and sequence boundaries."""
from copy import deepcopy
import unittest
import numpy as np

from crane_project.tools.eval_eood_sgar_fixed_test_v1 import (
    CHECKPOINT_SHA, fixed_selection, top1, longest_failure)


class FixedTestContractTests(unittest.TestCase):
    def test_only_original_ep24_and_selection_rule(self):
        config = {'frozen_rule': 1}
        source = dict(selected_checkpoint='epoch_24', selection_config=config,
                      all_checkpoints={'epoch_24': dict(checkpoint='fixed.pth', checkpoint_sha256=CHECKPOINT_SHA)})
        self.assertEqual(fixed_selection(source, config)['checkpoint'], 'fixed.pth')
        for value in (None, 'epoch_22'):
            changed = deepcopy(source); changed['selected_checkpoint'] = value
            with self.assertRaises(ValueError): fixed_selection(changed, config)
        with self.assertRaises(ValueError): fixed_selection(source, {'changed': 1})
        changed = deepcopy(source); changed['all_checkpoints']['epoch_24']['checkpoint_sha256'] = 'wrong'
        with self.assertRaises(ValueError): fixed_selection(changed, config)

    def test_native_multi_output_uses_highest_score_without_altering_output(self):
        boxes = np.asarray([[1., 2., 10., 5., 0., .2], [3., 4., 12., 6., .1, .9]], dtype=np.float32)
        saved = boxes.copy()
        self.assertEqual(top1([boxes]), boxes[1].tolist())
        self.assertTrue(np.array_equal(boxes, saved))
        self.assertIsNone(top1([np.empty((0, 6), dtype=np.float32)]))
        invalid = boxes.copy(); invalid[0, 0] = np.nan
        with self.assertRaises(ValueError): top1([invalid])
        with self.assertRaises(ValueError): top1([boxes, boxes])

    def test_failure_run_breaks_on_sequence_and_frame_gaps(self):
        rows = [dict(sequence='a', frame_id=1, riou=0.), dict(sequence='a', frame_id=2, riou=0.),
                dict(sequence='a', frame_id=4, riou=0.), dict(sequence='b', frame_id=1, riou=0.)]
        self.assertEqual(longest_failure(rows), 2)
        rows[1]['riou'] = .5
        self.assertEqual(longest_failure(rows), 1)


if __name__ == '__main__':
    unittest.main()
