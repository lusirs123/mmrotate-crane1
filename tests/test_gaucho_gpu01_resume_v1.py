from copy import deepcopy
import unittest

from crane_project.tools.resume_gaucho_no_port_shrink_gpu01_v1 import check_training_config


class ResumeConfigTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(seed=0, resume_from=None, work_dir='old', gpu_ids=[0, 1],
            custom_hooks=[dict(type='ExternalTrainingObserverV1', priority='LOWEST')],
            model=dict(loss='probiou'), runner=dict(max_epochs=24))
        self.parent = dict(protocol='gaucho_no_port_scale_native_resume_v1',
                           resume=dict(checkpoint='/old/epoch_2.pth'))
        self.historical = deepcopy(self.base)
        self.historical.update(resume_from='/old/epoch_2.pth', work_dir='resume002', auto_resume=False)
        self.historical['custom_hooks'].append(dict(type='GauChoResumeAuditV1', priority='LOWEST'))

    def test_accepts_native_resume_metadata_changes(self):
        check_training_config(self.base, self.historical, self.parent)

    def test_accepts_mmcv_preserved_original_config(self):
        historical = deepcopy(self.base)
        historical['work_dir'] = 'original_attempt'
        self.assertEqual(check_training_config(self.base, historical, self.parent),
                         'original_config_preserved_by_mmcv')

    def test_does_not_ignore_inconsistent_audit_hook_and_ancestry(self):
        self.historical['resume_from'] = None
        with self.assertRaises(ValueError):
            check_training_config(self.base, self.historical, self.parent)

    def test_rejects_training_recipe_drift(self):
        self.historical['model']['loss'] = 'other'
        with self.assertRaises(ValueError):
            check_training_config(self.base, self.historical, self.parent)

    def test_rejects_wrong_resume_ancestry(self):
        self.historical['resume_from'] = '/different/epoch_2.pth'
        with self.assertRaises(ValueError):
            check_training_config(self.base, self.historical, self.parent)

    def test_rejects_changed_seed(self):
        self.historical['seed'] = 42
        with self.assertRaises(ValueError):
            check_training_config(self.base, self.historical, self.parent)

    def test_rejects_extra_hook(self):
        self.historical['custom_hooks'].append(dict(type='UnrelatedHook'))
        with self.assertRaises(ValueError):
            check_training_config(self.base, self.historical, self.parent)


if __name__ == '__main__':
    unittest.main()
