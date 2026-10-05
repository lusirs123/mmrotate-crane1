"""Destructive maintenance contract tests using disposable synthetic outputs."""
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / 'crane_project/tools/organize_port_work_dirs_v1.py'
spec = importlib.util.spec_from_file_location('organize_port', str(SOURCE))
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class OrganizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name) / 'work_dirs'
        self.work.mkdir()

    def put(self, name, data=b'original evidence'):
        p = self.work / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def test_plan_is_read_only(self):
        self.put('port_geometry_example/report.json')
        before = sorted(str(p) for p in self.work.rglob('*'))
        plan = tool.plan(self.work)
        self.assertEqual(len(plan['moves']), 1)
        self.assertEqual(before, sorted(str(p) for p in self.work.rglob('*')))

    def test_consolidate_prune_verify_and_repeat(self):
        protected = (
            'crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth',
            'crane_symeood_k1_port_day2night_v1/epoch_20.pth',
            'crane_eood_k1_port_day2night_v1/epoch_24.pth',
            'crane_eood_k1_port_day2night_aug_b_v1/epoch_24.pth',
            'crane_symeood_k1_port_day2night_midpoint_formal_v1/head_epoch_23.pth',
            'crane_symeood_k1_port_day2night_midpoint_sigma_v1/sigma_1p5/head_epoch_03.pth',
            'crane_symeood_k1_dino_historical/epoch_01.pth',
            '_geometry_archive/old_review.tar.gz',
        )
        for p in protected:
            self.put(p)
        for name in tool.MODELS:
            self.put(name + '/epoch_01.pth', b'unused model')
            self.put(name + '/training.log.json')
            self.put(name + '/val/predictions.txt')
            (self.work / name / 'latest.pth').symlink_to('epoch_01.pth')
        self.put(tool.DIAGNOSTICS[0] + '/completion.json')
        cache = 'port_geometry_midpoint_formal_v1_roi_cache'
        self.put(cache + '/train_s1.pt', b'cache bytes')
        policy = 'port_midpoint_reliability_v1_fit'
        self.put(policy + '/policy.json', b'{"immutable": true}')
        self.put('port_size_core_curvature_v2_train_gradfix1/a1/epoch_01.pth', b'keep failed reliability checkpoint')
        self.put('port_geometry_midpoint_sigma15_v1_test_eval/completion.json')
        self.put('port_unknown_result.json')
        self.put('port_sigma_review.tar.gz')
        result = tool.apply(self.work, tool.plan(self.work))
        for p in protected:
            self.assertEqual((self.work / p).read_bytes(), b'original evidence')
        for name in tool.MODELS + (tool.DIAGNOSTICS[0],):
            self.assertFalse((self.work / name).exists())
        with tarfile.open(str(self.work / result['verified_archive'])) as tf:
            names = tf.getnames()
            self.assertFalse(any(n.endswith(('.pth', '.pt')) for n in names))
            for name in tool.MODELS:
                self.assertEqual(tf.extractfile(name + '/val/predictions.txt').read(), b'original evidence')
        manifest_path = self.work / (result['verified_archive'][:-7] + '.manifest.json')
        manifest = json.loads(manifest_path.read_text())
        self.assertTrue(manifest['originals'][tool.MODELS[0] + '/epoch_01.pth']['omitted_weight'])
        self.assertTrue((self.work / cache).is_symlink())
        self.assertEqual((self.work / cache / 'train_s1.pt').read_bytes(), b'cache bytes')
        self.assertEqual((self.work / policy / 'policy.json').read_bytes(), b'{"immutable": true}')
        failed = self.work / 'port_results/reliability/port_size_core_curvature_v2_train_gradfix1/a1/epoch_01.pth'
        self.assertEqual(failed.read_bytes(), b'keep failed reliability checkpoint')
        self.assertFalse((self.work / 'port_geometry_midpoint_sigma15_v1_test_eval').exists())
        repeated = tool.plan(self.work)
        self.assertEqual(repeated['moves'], [])
        self.assertEqual(repeated['archive_then_remove'], [])
        tool.apply(self.work, repeated)

    def test_collision_and_external_symlinks_rejected(self):
        p = self.put('port_geometry_example/a.json')
        dest = tool.destination(self.work, p.parent.name)
        dest.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'collision'):
            tool.plan(self.work)
        p.unlink()
        p.parent.rmdir()
        (self.work / tool.MODELS[0]).symlink_to(self.work / 'port_results', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'real directory'):
            tool.plan(self.work)

    def test_output_parent_symlink_refused_even_without_moves(self):
        (self.work / 'port_results').symlink_to(Path(self.tmp.name), target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink parent'):
            tool.plan(self.work)

    def test_nested_links_cannot_break_after_move(self):
        self.put('port_keep_model/epoch_01.pth')
        (self.work / 'port_keep_model/latest.pth').symlink_to('epoch_01.pth')
        tool.apply(self.work, tool.plan(self.work))
        self.assertEqual((self.work / 'port_results/other/port_keep_model/latest.pth').read_bytes(), b'original evidence')
        self.put('port_external/a.json')
        (self.work / 'port_external/outside').symlink_to('../../external')
        with self.assertRaisesRegex(ValueError, 'External nested link'):
            tool.plan(self.work)

    def test_retained_nonweight_symlink_in_closed_dir_refused(self):
        name = tool.MODELS[0]
        self.put(name + '/epoch_01.pth')
        (self.work / name / 'predictions').symlink_to('/tmp')
        with self.assertRaisesRegex(ValueError, 'Non-weight symlink'):
            tool.apply(self.work, tool.plan(self.work))
        self.assertTrue((self.work / name / 'epoch_01.pth').exists())

    def test_changed_original_blocks_deletion(self):
        name = tool.MODELS[0]
        p = self.put(name + '/report.json')
        original = tool.snapshot
        count = [0]

        def changing(work, names):
            count[0] += 1
            if count[0] == 2:
                p.write_bytes(b'changed while archiving')
            return original(work, names)

        with mock.patch.object(tool, 'snapshot', side_effect=changing):
            with self.assertRaisesRegex(ValueError, 'verification failed'):
                tool.apply(self.work, tool.plan(self.work))
        self.assertTrue(p.exists())

    def test_active_training_process_is_detected(self):
        lines = ('101 python crane_project/tools/run_port_midpoint_reliability_v1.py --mode train\n'
                 '102 python crane_project/tools/organize_port_work_dirs_v1.py --apply\n'
                 '103 python tools/train.py config.py\n')
        with mock.patch.object(tool.subprocess, 'check_output', return_value=lines):
            self.assertEqual(len(tool.active_jobs()), 2)


if __name__ == '__main__':
    unittest.main()
