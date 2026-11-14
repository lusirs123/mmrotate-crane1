import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from crane_project.tools import eval_oriented_rcnn_comparison_v1 as variant


class EvaluationTests(unittest.TestCase):
    def test_five_preregistered_candidates_and_true_model_identity(self):
        result = variant.static_check()
        self.assertEqual(result['candidates'], [8, 9, 10, 11, 12])
        self.assertEqual(result['model'], 'oriented_rcnn')
        self.assertEqual(result['physical_gpus'], [0, 1])
        self.assertIn(variant.SELF, variant.configuration(variant.MODEL))
        with self.assertRaises(ValueError):
            variant.configuration('gaucho')

    def test_dispatch_changes_only_model_adapter_preserving_locked_weight(self):
        command = ['python', str(variant.ROOT/'crane_project/tools/export_external_obb_v1.py'),
            '--model', 'gaucho', '--split', 'test', '--checkpoint', '/frozen/epoch_10.pth',
            '--selection-lock', '/locked.json', '--execute']
        result = variant.route_command(command)
        self.assertEqual(result[2:4], ['--stage', 'export'])
        expected = command[2:].copy(); expected[1] = variant.MODEL
        self.assertEqual(result[4:], expected)

    def test_dispatch_stdio_and_timing_api(self):
        api = variant.child_process_api()
        self.assertEqual(api.DEVNULL, subprocess.DEVNULL)
        self.assertEqual(api.STDOUT, subprocess.STDOUT)
        self.assertIs(api.TimeoutExpired, subprocess.TimeoutExpired)
        command = ['python', str(variant.ROOT/'crane_project/tools/benchmark_crane_final_models_v1.py'),
            '--stage', 'worker', '--config', '/profile.json', '--out-dir', '/timing']
        with patch.object(variant.subprocess, 'Popen', return_value='child') as launch:
            self.assertEqual(api.Popen(command, stdin=api.DEVNULL), 'child')
        self.assertEqual(launch.call_args.args[0][2:4], ['--stage', 'timing_worker'])
        self.assertEqual(launch.call_args.args[0][4:], command[4:])

    def test_refuses_training_or_wrong_export_child(self):
        with self.assertRaises(ValueError):
            variant.route_command(['python', 'tools/train.py'])
        with self.assertRaises(ValueError):
            variant.route_command(['python', str(variant.ROOT/'crane_project/tools/export_external_obb_v1.py'),
                                   '--model', 'yolo26'])
        with self.assertRaises(ValueError):
            variant.checkpoint(24)

    def test_common_selector_recipe_redirect_is_specific(self):
        spec = variant.experiment_read(variant.ROOT/'configs/gaucho/experiment.json')
        self.assertEqual(spec['model'], variant.MODEL)
        self.assertEqual(spec['selection']['candidate_completed_epochs'], list(variant.EPOCHS))
        path = variant.ROOT/variant.SPEC
        self.assertEqual(variant.experiment_read(path), json.loads(path.read_text()))

    def test_test_requires_all_val_metrics_selection_lock_and_exclusive_reservation(self):
        trace = []
        protocol = dict(weights={'10': 'sha'}, configuration={'source': 'sha'}, training_protocol_sha256='sha')
        lock = dict(selected_completed_epoch=10, checkpoint='/fixed/epoch_10.pth', locked_Taipei='fixed')
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch.object(variant, 'read', side_effect=lambda p: protocol if Path(p).name == 'protocol.json' else lock), \
                 patch.object(variant, 'verify'), patch.object(variant, 'sha', return_value='sha'), \
                 patch.object(variant.monitor, 'atomic_json'), patch.object(variant, 'write_new'), \
                 patch.object(variant.workflow, 'infer', side_effect=lambda o,j,p: trace.append(('infer',j))), \
                 patch.object(variant.saved, 'evaluate', side_effect=lambda p: trace.append(('metric',p))), \
                 patch.object(variant.selector, 'select', side_effect=lambda *a: trace.append(('select',a[0]))), \
                 patch.object(variant.exporter, 'validate_test_lock', side_effect=lambda *a: trace.append(('lock',a[1]))), \
                 patch.object(variant.workflow, 'reserve_test', side_effect=lambda *a: trace.append(('reserve',a[0]))), \
                 patch.object(variant.efficiency, 'run', side_effect=lambda *a,**k: trace.append(('timing',k['gpu']))):
                variant.controller(out)
        val = [job for kind,jobs in trace if kind=='infer' for job in jobs if job[0]=='val']
        tests = [job for kind,jobs in trace if kind=='infer' for job in jobs if job[0]=='test']
        self.assertEqual([job[1] for job in val], list(variant.EPOCHS))
        self.assertEqual(tests, [('test',10,0)])
        positions = {kind: next(i for i,v in enumerate(trace) if v[0]==kind) for kind in ('select','lock','reserve','timing')}
        test_index = next(i for i,v in enumerate(trace) if v[0]=='infer' and v[1][0][0]=='test')
        self.assertLess(positions['select'], positions['lock'])
        self.assertLess(positions['lock'], positions['reserve'])
        self.assertLess(positions['reserve'], test_index)
        self.assertLess(test_index, positions['timing'])

    def test_invalid_lock_prevents_test_reservation_and_timing(self):
        protocol = dict(weights={'10':'sha'}, configuration={}, training_protocol_sha256='sha')
        lock = dict(selected_completed_epoch=10, checkpoint='/fixed/epoch_10.pth', locked_Taipei='fixed')
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(variant, 'read', side_effect=lambda p: protocol if Path(p).name=='protocol.json' else lock), \
                 patch.object(variant, 'verify'), patch.object(variant, 'sha', return_value='sha'), \
                 patch.object(variant.monitor, 'atomic_json'), patch.object(variant.workflow, 'infer'), \
                 patch.object(variant.saved, 'evaluate'), patch.object(variant.selector, 'select'), \
                 patch.object(variant.exporter, 'validate_test_lock', side_effect=ValueError('bad lock')), \
                 patch.object(variant.workflow, 'reserve_test') as reserve, \
                 patch.object(variant.efficiency, 'run') as timing:
                with self.assertRaises(ValueError):
                    variant.controller(Path(directory))
                reserve.assert_not_called(); timing.assert_not_called()

    def test_timing_cannot_silently_reduce_repeats_or_change_precision(self):
        for key, value in [('repeats',1), ('warmup_frames',0), ('precision','FP16')]:
            config = variant.read(variant.ROOT/variant.efficiency.TEMPLATE)
            config['models'] = {variant.MODEL: {}}
            config[key] = value
            with self.assertRaises(ValueError):
                variant.validate_profile(config)


if __name__ == '__main__':
    unittest.main()
