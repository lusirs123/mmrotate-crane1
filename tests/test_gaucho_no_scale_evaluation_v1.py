import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch
from crane_project.tools import eval_gaucho_no_port_shrink_v1 as variant


class NoScaleEvaluationTests(unittest.TestCase):
    def test_child_dispatch_preserves_stdio_and_timeout_api(self):
        api=variant.child_process_api()
        self.assertEqual(api.DEVNULL,subprocess.DEVNULL)
        self.assertEqual(api.STDOUT,subprocess.STDOUT)
        self.assertIs(api.TimeoutExpired,subprocess.TimeoutExpired)
        with patch.object(variant.subprocess,'Popen',return_value='child') as launch:
            cmd=['python',str(variant.ROOT/'crane_project/tools/export_external_obb_v1.py'),'--model','gaucho']
            self.assertEqual(api.Popen(cmd,stdin=api.DEVNULL,stderr=api.STDOUT),'child')
            self.assertEqual(launch.call_args.args[0][2:4],['--stage','export'])

    def test_export_routes_to_variant_without_changing_checkpoint_or_split(self):
        command=['python',str(variant.ROOT/'crane_project/tools/export_external_obb_v1.py'),
            '--model','gaucho','--split','test','--checkpoint','/frozen/epoch_20.pth',
            '--selection-lock','/locked.json','--execute']
        result=variant.route_command(command)
        self.assertEqual(result[2:4],['--stage','export'])
        self.assertEqual(result[4:],command[2:])

    def test_timing_worker_routes_without_changing_profile(self):
        command=['python',str(variant.ROOT/'crane_project/tools/benchmark_crane_final_models_v1.py'),
            '--stage','worker','--config','/profile.json','--out-dir','/timing']
        result=variant.route_command(command)
        self.assertEqual(result[2:4],['--stage','timing_worker'])
        self.assertEqual(result[4:],command[4:])

    def test_other_commands_cannot_be_launched(self):
        with self.assertRaises(ValueError):
            variant.route_command(['python','some_training_script.py'])
        with self.assertRaises(ValueError):
            variant.route_command(['python',str(variant.ROOT/'crane_project/tools/benchmark_crane_final_models_v1.py'),
                '--stage','benchmark'])

    def test_configuration_pins_variant_dispatcher(self):
        with patch.object(variant,'_native_identity',return_value={'base.py':'sha'}), \
                patch.object(variant,'sha',return_value='variant-sha'):
            identity=variant.configuration('gaucho')
        self.assertEqual(identity,{'base.py':'sha',variant.SELF:'variant-sha'})

    def test_evaluation_uses_completed_no_scale_run(self):
        self.assertEqual(variant.TRAIN.name,'resume_003_gpu01')
        self.assertIn('no_port_shrink',variant.CONFIG)
        self.assertEqual(variant.workflow.EPOCHS,(16,18,20,22,24))


if __name__ == '__main__':
    unittest.main()
