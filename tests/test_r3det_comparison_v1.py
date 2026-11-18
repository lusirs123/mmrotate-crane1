from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from crane_project.tools import check_r3det_comparison_v1 as check
from crane_project.tools import run_r3det_comparison_v1 as run


class R3DetChecks(unittest.TestCase):
    def test_native_model_recipe_and_effective_batch(self):
        cfg,_=check.check_config()
        self.assertEqual(cfg['optimizer']['lr'],.0025)
        self.assertEqual(cfg['runner']['max_epochs'],12)
        self.assertEqual(cfg['model']['num_refine_stages'],1)
        self.assertEqual(cfg['optimizer_config']['cumulative_iters'],2)
        self.assertEqual(cfg['lr_config']['warmup_iters'],1000)
        self.assertEqual(cfg['data']['samples_per_gpu'],1)

    def reject(self,mutate):
        cfg,closure=check.check_config();cfg=deepcopy(cfg);mutate(cfg);original=check.resolve
        with patch.object(check,'resolve',side_effect=lambda p:(cfg,closure) if p==check.ROOT/check.CONFIG else original(p)):
            with self.assertRaises(ValueError):check.check_config()

    def test_reject_paper_scale_augmentation(self):
        self.reject(lambda c:c['data']['train'][0]['pipeline'].insert(3,dict(type='PortIsotropicShrink')))

    def test_reject_le90_dataset(self):
        self.reject(lambda c:c['data']['train'][0].update(version='le90'))

    def test_reject_unscaled_warmup_or_wrong_accumulation(self):
        self.reject(lambda c:c['lr_config'].update(warmup_iters=500))
        self.reject(lambda c:c['optimizer_config'].update(cumulative_iters=1))

    def test_refuse_busy_selected_gpu_without_touching_other_gpu(self):
        rows=[dict(index=0,uuid='a',pci='00',free_mib=8100),dict(index=1,uuid='b',pci='01',free_mib=1000)]
        with patch.object(run.io,'gpu_rows',return_value=rows),patch.object(run.io,'processes',return_value=[dict(uuid='b')]):
            row,ordinal=run.idle(0);self.assertEqual(ordinal,0)
        with patch.object(run.io,'gpu_rows',return_value=rows),patch.object(run.io,'processes',return_value=[dict(uuid='a')]):
            with self.assertRaises(RuntimeError):run.idle(0)

    def memory_guard_case(self,used,free):
        class Child:
            pid=123
            def poll(self):return None
            def wait(self):return -15
        row=dict(index=0,uuid='a',pci='00',free_mib=free,used_mib=used)
        protocol=dict(physical_gpu=0,pci_cuda_ordinal=0,before=row,driver_mib=6656,minimum_free_mib=1536)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(run,'assert_sources'),patch.object(run,'idle'), \
                 patch.object(run.io,'gpu_rows',return_value=[row]), \
                 patch.object(run.io,'processes',return_value=[dict(pid=123,uuid='a',used_mib=used)]), \
                 patch.object(run.subprocess,'Popen',return_value=Child()),patch.object(run,'stop_group') as stop:
                self.assertFalse(run.guarded_run(Path(folder),protocol,smoke=True))
                stop.assert_called_once()
                self.assertEqual(run.read(Path(folder)/'result.json')['reason'],'MEMORY_GUARD')

    def test_process_memory_limit_stops(self):
        self.memory_guard_case(6657,2000)

    def test_low_free_memory_stops(self):
        self.memory_guard_case(1000,1535)


if __name__=='__main__':unittest.main()
