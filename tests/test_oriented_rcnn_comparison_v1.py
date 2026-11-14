"""Fail-closed checks for the authorized external comparison recipe."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crane_project.tools import check_oriented_rcnn_comparison_v1 as check


class RecipeTests(unittest.TestCase):
    def test_author_global_batch_and_schedule(self):
        cfg, _ = check.check_config()
        self.assertEqual(cfg['optimizer']['lr'], 0.005)
        self.assertEqual(cfg['data']['samples_per_gpu'] * 2, 2)
        self.assertEqual(cfg['runner']['max_epochs'], 12)
        self.assertEqual(cfg['lr_config']['step'], [8, 11])
        self.assertEqual(cfg['lr_config']['warmup_ratio'], 0.001)
        self.assertEqual(check.EPOCHS, [8, 9, 10, 11, 12])

    def reject_changed_config(self, mutate):
        cfg, closure = check.check_config()
        cfg = deepcopy(cfg)
        mutate(cfg)
        original = check.resolve
        def resolve(path):
            if path == check.ROOT/check.CONFIG:
                return cfg, closure
            return original(path)
        with patch.object(check, 'resolve', resolve):
            with self.assertRaises(ValueError):
                check.check_config()

    def test_reject_task_scale_augmentation(self):
        self.reject_changed_config(lambda c: c['data']['train'][1]['pipeline'].insert(
            3, dict(type='PortIsotropicShrink', prob=0.5, scale_range=(0.5, 1.0))))

    def test_reject_faa_model_substitution(self):
        self.reject_changed_config(lambda c: c['model']['neck'].update(type='FAAFusionFPN'))

    def test_reject_wrong_learning_rate(self):
        self.reject_changed_config(lambda c: c['optimizer'].update(lr=0.0025))

    def test_reject_wrong_split(self):
        self.reject_changed_config(lambda c: c['data']['val'].update(img_prefix='test/images/'))

    def test_native_source_identity(self):
        provenance = check.check_sources()
        self.assertEqual(provenance['tag'], 'v0.3.4')
        self.assertEqual(len(provenance['files']), 7)


if __name__ == '__main__':
    unittest.main()
