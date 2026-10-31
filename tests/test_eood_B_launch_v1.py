"""Resource routing and owned-process stopping for the E+B controller."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crane_project.tools.run_eood_B_regression_v1 import gpus, env_for, stop_group
from crane_project.utils.eood_B_regression_protocol_v1 import training_arguments


class LaunchPolicyTests(unittest.TestCase):
    def test_gloo_fallback_changes_only_explicit_communication_option(self):
        normal=training_arguments('config.py','train','nccl')
        self.assertEqual(training_arguments('config.py','train','gloo'),
                         normal+['--cfg-options','dist_params.backend=gloo'])
        self.assertNotIn('--resume-from',normal)
        self.assertNotIn('--autoscale-lr',normal)
        with self.assertRaises(ValueError):
            training_arguments('config.py','train','unknown')

    def test_resource_query_checks_only_authorized_cards(self):
        def query(command, **kwargs):
            # Reproduce the real failure if the query reaches unrelated GPU2.
            output='0, 4000, 4192\n1, 4000, 4192\n'
            if '-i' not in command or command[command.index('-i')+1] != '0,1':
                output+='Unable to determine the device handle for GPU2: 0000:09:00.0: Unknown Error\n'
            return output
        with patch('subprocess.check_output', side_effect=query):
            self.assertEqual(gpus(), {0:{'free_mib':4000,'used_mib':4192},
                                      1:{'free_mib':4000,'used_mib':4192}})

    def test_authorized_device_missing_or_faulty_still_stops(self):
        for output in ('0, 4000, 4192\n', '0, 4000, 4192\n1, N/A, N/A\n',
                       '0, 4000, 4192\n0, 4000, 4192\n',
                       '0, 4000, 4192\n1, -1, 0\n'):
            with self.subTest(output=output), patch('subprocess.check_output', return_value=output):
                with self.assertRaises(ValueError):
                    gpus()

    def test_gpu_and_cpu_environments_do_not_mutate_parent(self):
        c={'launch_environment':{'CUDA_VISIBLE_DEVICES':'0,1','OMP_NUM_THREADS':'1'}}
        before=dict(os.environ)
        self.assertEqual(env_for(c)['CUDA_VISIBLE_DEVICES'],'0,1')
        self.assertEqual(env_for(c,cpu=True)['CUDA_VISIBLE_DEVICES'],'')
        self.assertEqual(dict(os.environ),before)

    def test_stopping_targets_only_the_owned_running_process_group(self):
        calls=[]
        process=SimpleNamespace(pid=12345,poll=lambda:None,wait=lambda timeout:calls.append(timeout))
        with patch('os.killpg') as kill:
            stop_group(process)
            self.assertEqual(kill.call_args.args[0],12345)
            self.assertEqual(calls,[15])
        with patch('os.killpg') as kill:
            stop_group(SimpleNamespace(poll=lambda:0))
            kill.assert_not_called()


if __name__=='__main__':
    unittest.main()
