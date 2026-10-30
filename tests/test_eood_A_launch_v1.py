"""Resource routing and owned-process stopping for the E+A controller."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crane_project.tools.run_eood_A_geometry_v1 import gpus, env_for, stop_group


class LaunchPolicyTests(unittest.TestCase):
    def test_resource_query_checks_only_authorized_cards(self):
        output='0, 4000, 4192\n1, 4000, 4192\n2, 7600, 592\n3, 8000, 192\n'
        with patch('subprocess.check_output', return_value=output):
            self.assertEqual(gpus(), {2:{'free_mib':7600,'used_mib':592},
                                      3:{'free_mib':8000,'used_mib':192}})

    def test_gpu_and_cpu_environments_do_not_mutate_parent(self):
        c={'launch_environment':{'CUDA_VISIBLE_DEVICES':'2,3','OMP_NUM_THREADS':'1'}}
        before=dict(os.environ)
        self.assertEqual(env_for(c)['CUDA_VISIBLE_DEVICES'],'2,3')
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
