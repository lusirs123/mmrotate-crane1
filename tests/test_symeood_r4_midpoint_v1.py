"""Local identity, control and no-launch checks; no torch, data or GPU needed."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import pprint
import sys
import tempfile
import unittest
from unittest.mock import patch

from crane_project.utils import symeood_r4_midpoint_protocol_v1 as p
from crane_project.tools import train_symeood_r4_midpoint_v1 as t


class PreparationChecks(unittest.TestCase):
    def test_identity_and_original_math_closure_without_torch(self):
        before = 'torch' in sys.modules
        identity = p.checked_sources()
        self.assertEqual(identity['frozen_b']['checkpoint_sha256'], p.R4_SHA)
        self.assertEqual(identity['frozen_b']['selected_epoch'], 'epoch_16')
        self.assertEqual(identity['sigma_cells'], 1.5)
        self.assertEqual(identity['settings']['epochs'], 24)
        self.assertEqual('torch' in sys.modules, before)

    def test_config_controls_and_r4_confounders(self):
        a = p.ablation_audit(); sig = a['signatures']
        self.assertEqual(sig['E']['classification_tower_layers'], 4)
        self.assertEqual(sig['R4']['classification_tower_layers'], 0)
        self.assertEqual(sig['110']['regression_tower_layers'], 0)
        self.assertEqual(sig['R4']['regression_tower_layers'], 4)
        self.assertEqual(sig['R4']['main_classification']['loss_weight'], .25)
        self.assertEqual(sig['E']['main_classification']['loss_weight'], 2.)
        self.assertEqual(sig['R4']['auxiliary_classification']['loss_weight'], .01)
        self.assertEqual(sig['E']['auxiliary_classification']['loss_weight'], 1.)
        self.assertEqual(sig['R4']['inference']['max_per_img'], 1)
        self.assertEqual(sig['E']['inference']['max_per_img'], 2000)
        self.assertIn('cannot form', a['invalid_factorial'])

    def test_extra_loss_or_schedule_changes_break_tower_control(self):
        original, r4 = p.config(p.S_CONFIG), p.config(p.R4_CONFIG)
        p.assert_r4_structure_control(original, r4)
        wrong = deepcopy(r4); wrong['model']['aux_bbox_head'][0]['loss_cls']['loss_weight'] = 1.
        with self.assertRaises(ValueError):
            p.assert_r4_structure_control(original, wrong)
        wrong = deepcopy(r4); wrong['lr_config']['warmup_iters'] = 999
        with self.assertRaises(ValueError):
            p.assert_r4_structure_control(original, wrong)

    def test_checkpoint_meta_wrong_frontend_epoch_and_code_are_rejected(self):
        cfg = p.config(p.R4_CONFIG)
        text = '\n'.join(k+' = '+pprint.pformat(cfg[k]) for k in p.META_FIELDS)
        meta = dict(epoch=16, iter=10240, seed=0, config=text)
        p.check_checkpoint_metadata(meta, cfg)
        for key, value in [('epoch', 20), ('iter', 12800), ('seed', 1)]:
            with self.assertRaises(ValueError):
                p.check_checkpoint_metadata(dict(meta, **{key: value}), cfg)
        with self.assertRaises(ValueError):
            p.check_checkpoint_metadata(dict(meta, config='model = __import__("os").getcwd()'), cfg)

    def test_artifact_attempts_remain_below_one_root(self):
        a, b = (p.artifact_paths(p.DEFAULT_ROOT, name) for name in ('attempt_001', 'attempt_002'))
        self.assertEqual(a['root'], b['root'])
        self.assertEqual(a['preparation'], b['preparation'])
        self.assertNotEqual(a['cache'], b['cache'])
        for name in ('cache', 'verify', 'train'):
            self.assertIn(a['attempt'], a[name].parents)
        for root, attempt in [(Path('/tmp/outside'), 'attempt_001'), (p.DEFAULT_ROOT, '../escape')]:
            with self.assertRaises(ValueError):
                p.artifact_paths(root, attempt)

    def test_default_check_and_prepare_never_enter_runtime(self):
        self.assertEqual(t.parser().parse_args([]).stage, 'check')
        with tempfile.TemporaryDirectory(dir=p.ROOT/'work_dirs') as tmp:
            root = Path(tmp)/'one_experiment'
            args = argparse.Namespace(root=root, attempt='attempt_001', stage='check', gpu=0)
            with patch.object(t, 'make_engine', side_effect=AssertionError('runtime forbidden')):
                result = t.run(args)
                self.assertEqual(result['optimizer_updates'], 0)
                self.assertFalse(root.exists())
                args.stage = 'prepare'; t.run(args)
            self.assertTrue((root/'preparation/review.json').exists())
            self.assertFalse((root/'midpoint').exists())
            with self.assertRaises(FileExistsError):
                t.run(args)

    def test_foreign_cache_rejected_before_cuda_or_optimizer(self):
        identity = {'frozen_b': p.frontend_identity()}
        with tempfile.TemporaryDirectory(dir=p.ROOT/'work_dirs') as tmp:
            paths = p.artifact_paths(Path(tmp)/'experiment', 'attempt_001')
            paths['cache'].mkdir(parents=True)
            (paths['cache']/'cache_manifest.json').write_text(json.dumps(dict(
                protocol='symeood_110_midpoint_v1', identity=identity, test_access=False, detector_updates=0)))
            with self.assertRaisesRegex(ValueError, 'Old110'):
                t.runtime_prerequisites(paths, 'verify', identity)

    def test_verification_bound_to_source_cache_and_zero_updates(self):
        identity = {'frontend': p.R4_SHA}
        good = dict(status='R4_MIDPOINT_ZERO_UPDATE_VERIFIED', identity=identity,
            cache_manifest_sha256='c'*64, detector_updates=0, optimizer_updates=0, test_access=False)
        p.verification_binding(good, identity, 'c'*64)
        for key, value in [('optimizer_updates', 1), ('test_access', True),
                           ('cache_manifest_sha256', 'd'*64), ('identity', {'frontend': 'old110'})]:
            with self.assertRaises(ValueError):
                p.verification_binding(dict(good, **{key: value}), identity, 'c'*64)

    def test_source_drift_is_rejected(self):
        original = p.sha
        target = p.ROOT/'crane_project/tools/train_symeood_r4_midpoint_v1.py'
        with patch.object(p, 'sha', side_effect=lambda path: '0'*64 if Path(path)==target else original(path)):
            with self.assertRaisesRegex(ValueError, 'source SHA'):
                p.checked_sources()

    def test_no_test_smoke_or_lvsr_dispatch(self):
        stages = t.parser()._option_string_actions['--stage'].choices
        self.assertEqual(stages, ('check', 'prepare', 'cache', 'verify', 'train'))
        self.assertNotIn('test', stages)
        self.assertNotIn('smoke', stages)  # Historical smoke performs optimizer steps.
        self.assertNotIn('lvsr', stages)

    def test_authorized_rear_gpu_mapping_and_controller_stages(self):
        from crane_project.tools.run_symeood_r4_midpoint_v1 import command
        for gpu in (2, 3):
            self.assertEqual(t.parser().parse_args(['--gpu', str(gpu)]).gpu, gpu)
            for stage in ('cache', 'verify', 'train'):
                args = t.parser().parse_args(command(stage, p.DEFAULT_ROOT, 'attempt_001', gpu)[2:])
                self.assertEqual((args.stage, args.gpu), (stage, gpu))
        with self.assertRaises(ValueError):
            command('test', p.DEFAULT_ROOT, 'attempt_001', 3)

    def test_server_original_and_local_mirror_need_identical_receipt_bytes(self):
        original = p.selection_receipt_path().read_bytes()
        with tempfile.TemporaryDirectory(dir=p.ROOT/'work_dirs') as tmp:
            first, mirror = Path(tmp)/'server.json', Path(tmp)/'download.json'
            mirror.write_bytes(original)
            with patch.object(p, 'R4_SELECTION', str(first.relative_to(p.ROOT))), \
                    patch.object(p, 'R4_SELECTION_MIRROR', str(mirror.relative_to(p.ROOT))):
                self.assertEqual(p.selection_receipt_path(), mirror)
                first.write_bytes(original)
                self.assertEqual(p.selection_receipt_path(), first)
                first.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'receipt SHA'):
                    p.selection_receipt_path()  # Do not hide a bad server receipt with the mirror.


if __name__ == '__main__':
    unittest.main()
