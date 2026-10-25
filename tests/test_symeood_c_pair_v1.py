"""Paired launcher must reproduce gate environment, not inherit ambient drift."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().parents[1]/'crane_project/tools/run_symeood_c_pair_v1.py'
SPEC=importlib.util.spec_from_file_location('c_pair_runner',PATH)
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


class PairedEnvironment(unittest.TestCase):
    def test_frozen_parent_replaces_ambient_prefix_without_duplication(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0','LD_LIBRARY_PATH':'/ambient','CONDA_PREFIX':'/other'},clear=False):
            env=MODULE.paired_environment(dict(CUDA_VISIBLE_DEVICES='2,3',LD_LIBRARY_PATH='/verified/cv2:',CONDA_PREFIX=None))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'2,3')
        self.assertEqual(env['LD_LIBRARY_PATH'],'/verified/cv2:')
        self.assertNotIn('CONDA_PREFIX',env)
        self.assertEqual(env['OMP_NUM_THREADS'],'1')

    def test_wrong_physical_cards_are_not_silently_accepted(self):
        with self.assertRaises(AssertionError):MODULE.paired_environment({'CUDA_VISIBLE_DEVICES':'0,1'})


if __name__=='__main__':unittest.main()
