"""Guard the one-run budget, config identity and fail-stop orchestration."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().parents[1]/'crane_project/tools/run_symeood_110_retrain_v1.py'
spec=importlib.util.spec_from_file_location('single110',PATH);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Contracts(unittest.TestCase):
    def test_only_paths_may_change(self):
        old=dict(work_dir='old',custom_hooks=[dict(type='trace',runtime_lock='old')],model=dict(C=False),runner=dict(max_epochs=24))
        new=copy.deepcopy(old);new['work_dir']='new';new['custom_hooks'][0]['runtime_lock']='new'
        m.assert_config_identity(old,new)
        new['runner']['max_epochs']=25
        with self.assertRaises(AssertionError):m.assert_config_identity(old,new)
        self.assertEqual(old['custom_hooks'][0]['runtime_lock'],'old')

    def test_formal_native_command_has_no_resume_determinism_or_gate(self):
        cmd=m.native_command('cfg','dest')
        self.assertIn('tools/train.py',cmd)
        self.assertIn('--nproc_per_node=2',cmd)
        self.assertNotIn('--deterministic',cmd);self.assertNotIn('--resume-from',cmd)
        self.assertNotIn('check_symeood_110_no_update_v1.py',cmd)
        self.assertIn('crane_project/tools/check_symeood_110_no_update_v1.py',m.native_command('cfg','audit',True))

    def test_explicit_01_mapping_and_original_parent_environment(self):
        with patch.dict(os.environ,{'LD_LIBRARY_PATH':'ambient','CONDA_PREFIX':'ambient','CUDA_VISIBLE_DEVICES':'3'}):
            env=m.environment(dict(launch_environment=dict(LD_LIBRARY_PATH='original:',CONDA_PREFIX=None,OMP_NUM_THREADS='1')))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0,1');self.assertEqual(env['LD_LIBRARY_PATH'],'original:')
        self.assertNotIn('CONDA_PREFIX',env)

    def test_nonfinite_stops_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);(work/'train.log.json').write_text('{"mode":"train","epoch":1,"loss":NaN}\n')
            with self.assertRaises(AssertionError):m.validate_train_logs(work,partial=True)

    def test_existing_destination_is_never_silently_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'claim.json';m.save(p,dict(formal_runs=1))
            with self.assertRaises(FileExistsError):m.save(p,dict(formal_runs=2))


if __name__=='__main__':unittest.main()
