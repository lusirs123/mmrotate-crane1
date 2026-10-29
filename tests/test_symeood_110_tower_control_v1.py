"""Control identity checks; optional real head gradients in mmrotljj runtime."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('t4_run',ROOT/'crane_project/tools/run_symeood_110_tower_control_v1.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)


class ControlTests(unittest.TestCase):
    def test_data_optimizer_and_output_contract_changes_are_rejected(self):
        import runpy
        old=runpy.run_path(str(ROOT/'work_dirs/symeood_110_retrain_seed0_20261026_v1/configs/Coff_seed0.py'))
        old={k:v for k,v in old.items() if not k.startswith('__')}
        new=deepcopy(old)
        new['model']['bbox_head']['type']='SymEOODTowerHeadV1'
        new['custom_imports']['imports'] += [
            'mmrotate.models.dense_heads.sym_eood_tower_head_v1',
            'crane_project.utils.symeood_tower_trace_v1']
        new['custom_hooks'][0].update(type='SymEOODTowerTraceV1',runtime_lock='new/lock.json',
            reference_work=str(t.REFERENCE/'110'))
        new.update(load_from='new/initial_T4.pth',work_dir='new/T4')
        t.assert_config_delta(old,new)
        for key,value in [('optimizer',dict(type='SGD',lr=.1)),
                          ('runner',dict(type='EpochBasedRunner',max_epochs=25))]:
            bad=deepcopy(new);bad[key]=value
            with self.assertRaises(AssertionError):t.assert_config_delta(old,bad)
        bad=deepcopy(new);bad['data']['samples_per_gpu']=1
        with self.assertRaises(AssertionError):t.assert_config_delta(old,bad)
        bad=deepcopy(new);bad['model']['test_cfg']['score_thr']=.1
        with self.assertRaises(AssertionError):t.assert_config_delta(old,bad)

    def test_auxiliary_or_loss_change_is_rejected(self):
        a=dict(bbox_head=dict(type='SymEOODHead',stacked_convs=4,loss_bbox={'loss_weight':2}),aux_bbox_head=[{'weight':.01}])
        b=deepcopy(a);b['bbox_head']['type']='SymEOODTowerHeadV1';t.assert_model_delta(a,b)
        b['aux_bbox_head'][0]['weight']=1
        with self.assertRaises(AssertionError):t.assert_model_delta(a,b)
        b=deepcopy(a);b['bbox_head'].update(type='SymEOODTowerHeadV1',loss_bbox={'loss_weight':3})
        with self.assertRaises(AssertionError):t.assert_model_delta(a,b)

    def test_no_test_or_resume_stage(self):
        text=(ROOT/'crane_project/tools/run_symeood_110_tower_control_v1.py').read_text()
        self.assertNotIn('--run-final-test',text)
        self.assertNotIn('--resume-from',text)
        self.assertNotIn('--auto-resume',text)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Runtime test requires mmrotljj')
    def test_both_towers_receive_gradients_and_preserve_predictor_shapes(self):
        import torch
        from mmcv import Config
        from mmcv.utils import import_modules_from_strings
        from mmrotate.models import build_head
        torch.set_num_threads(1)
        cfg=Config.fromfile(str(ROOT/'work_dirs/symeood_110_retrain_seed0_20261026_v1/configs/Coff_seed0.py'))
        import_modules_from_strings(**cfg.custom_imports)
        import mmrotate.models.dense_heads.sym_eood_tower_head_v1
        a=deepcopy(cfg.model.bbox_head);base=build_head(a)
        a.type='SymEOODTowerHeadV1';head=build_head(a)
        self.assertEqual(sum(p.numel() for p in head.parameters())-sum(p.numel() for p in base.parameters()),4720640)
        base_state=base.state_dict();state=head.state_dict()
        self.assertTrue(all(state[k].shape==v.shape for k,v in base_state.items()))
        x=torch.randn(2,256,8,8);cs,rg=head.forward_single(x)
        self.assertEqual(tuple(cs.shape),(2,3,8,8));self.assertEqual(tuple(rg.shape),(2,15,8,8))
        (cs.square().mean()+rg.square().mean()).backward()
        for tower in (head.cls_convs,head.reg_convs):
            self.assertEqual(len(tower),4)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in tower.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in tower.parameters()),0)


if __name__=='__main__':unittest.main()
