"""Bounded config migration and immutable-loading checks; no CUDA/model run."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from crane_project.utils import crane_frozen_pipeline_v1 as p


class FormalPipelineChecks(unittest.TestCase):
    def test_r4_keeps_effective_model_data_and_training_recipe(self):
        old = p.read_constants(p.ROOT/'work_dirs/symeood_110_reg_tower_20261030_v1/config.py')
        new = p.read_resolved_config(p.ROOT/'crane_project/configs/crane_sgar_r4_v1.py')
        changed = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
        self.assertEqual(changed, {'load_from','work_dir','custom_hooks','custom_imports'})
        self.assertIsNone(new['load_from'])
        self.assertEqual(new['custom_hooks'], [])
        self.assertNotIn('/media/', json.dumps(new))
        self.assertEqual(new['model'], old['model'])
        self.assertEqual(new['data'], old['data'])
        self.assertIsInstance(new['data']['test']['pipeline'][1]['img_scale'], tuple)
        self.assertIsInstance(new['model']['bbox_head']['bbox_coder']['target_stds'], tuple)

    def test_all_configuration_inheritance_stays_in_configs(self):
        root = (p.ROOT/'crane_project/configs').resolve()
        for name in ['crane_sgar_r4_v1.py','crane_midpoint_r4_v1.py','crane_lvsr_r4_midpoint_v1.py']:
            path = root/name
            entry = p.read_constants(path)
            self.assertNotIn('artifact_reference',entry)
            self.assertTrue(entry['_base_'])
            for parent in entry['_base_']:
                resolved = (path.parent/parent).resolve()
                self.assertIn(root,resolved.parents)
                self.assertTrue(resolved.is_file())

    def test_refinement_training_and_constants_match_original_records(self):
        _, (_, m, l) = p.read_config()
        for filename, actual in [('symeood_r4_midpoint_v1.json',m['training']),
                                 ('symeood_r4_lvsr_pair_v1.json',l['training'])]:
            contract = json.loads((p.ROOT/'crane_project/data_contracts'/filename).read_text())
            self.assertEqual(actual, contract['settings'])
        original = p.read_constants(p.ROOT/'crane_project/utils/port_geometry_midpoint_v1.py', {'SETTINGS'})
        self.assertEqual(m['implementation_settings'], original['SETTINGS'])
        self.assertEqual(l['branches']['hidden_channels'], 4)
        self.assertIsNone(l['checkpoint']['selected_epoch'])

    def test_static_check_does_not_import_torch_or_old_trainers(self):
        before = set(sys.modules)
        result = p.inspect()
        self.assertTrue(result['passed'])
        self.assertEqual(result['model_forwards'], 0)
        added = set(sys.modules)-before
        self.assertNotIn('torch', added)
        self.assertFalse(any('train_symeood' in n for n in added))

    def test_wrong_existing_checkpoint_not_hidden_by_good_mirror(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, mirror = (Path(tmp)/n for n in ('first.pth','mirror.pth'))
            first.write_bytes(b'bad'); mirror.write_bytes(b'good')
            spec = dict(paths=[str(first),str(mirror)],sha256=p.sha(mirror))
            with self.assertRaisesRegex(ValueError, 'SHA differs'):
                p.resolve_checkpoint(spec)
            first.unlink()
            self.assertEqual(p.resolve_checkpoint(spec),mirror)

    def test_config_reader_rejects_executable_expression(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.py'
            path.write_text('model = __import__("os").getcwd()')
            with self.assertRaisesRegex(ValueError, 'Executable'):
                p.read_constants(path)

    def test_source_drift_is_rejected(self):
        cfg, _ = p.read_config()
        target = p.local_path(next(iter(cfg['source_sha256'])))
        real = p.sha
        with patch.object(p,'sha',side_effect=lambda path:'0'*64 if Path(path)==target else real(path)):
            with self.assertRaisesRegex(ValueError,'Core source/config SHA'):
                p.read_config()

    def test_no_training_or_test_stage_and_default_check(self):
        from crane_project.tools.run_crane_frozen_pipeline_v1 import parser
        args = parser().parse_args([])
        self.assertEqual(args.stage,'check')
        self.assertIsNone(args.gpu)
        self.assertIsNone(args.out_dir)
        self.assertEqual(parser()._option_string_actions['--stage'].choices,('check','predict','verify-val'))

    def test_inheritance_merge_delete_and_cycle_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, child = Path(tmp)/'base.py', Path(tmp)/'child.py'
            base.write_text('model = dict(head=dict(type="Old", channels=32), levels=[1,2])')
            child.write_text('_base_ = ["base.py"]\nmodel = dict(head=dict(type="New"), levels=[3])')
            self.assertEqual(p.read_resolved_config(child)['model'],
                             dict(head=dict(type='New',channels=32),levels=[3]))
            child.write_text('_base_ = ["base.py"]\nmodel = dict(head=dict(_delete_=True,type="New"))')
            self.assertEqual(p.read_resolved_config(child)['model']['head'],dict(type='New'))
            base.write_text('_base_ = ["child.py"]')
            with self.assertRaisesRegex(ValueError,'Cyclic'):
                p.read_resolved_config(child)


if __name__ == '__main__':
    unittest.main()
