import unittest
from copy import deepcopy
from crane_project.utils.port_oof_momentum_restart_v1 import VERSION
try:
    import torch
except ImportError:
    torch = None


class ContractTests(unittest.TestCase):
    def test_separate_revision(self):
        self.assertEqual(VERSION,'opposing_sgd_momentum_restart_v1')


@unittest.skipIf(torch is None,'Torch required')
class RestartTests(unittest.TestCase):
    def make(self, gradient, momentum, **kwargs):
        from mmrotate.core.hooks.port_oof_momentum_restart_hook import PortOOFMomentumRestartSGD
        p = torch.nn.Parameter(torch.tensor([.25,-.5]))
        opt = PortOOFMomentumRestartSGD([p], momentum_restart_revision=VERSION,
            lr=.0025,momentum=.9,**kwargs)
        p.grad = torch.tensor(gradient)
        opt.state[p]['momentum_buffer'] = torch.tensor(momentum)
        return p,opt

    def test_decision_never_mutates_state(self):
        from crane_project.utils.port_oof_momentum_restart_v1 import momentum_restart_decision
        p,opt = self.make([1.,2.],[-3.,-4.])
        before = p.clone(); old = opt.state[p]['momentum_buffer'].clone(); grad = p.grad.clone()
        decision = momentum_restart_decision(opt)
        self.assertTrue(decision['restart_required'])
        self.assertTrue(torch.equal(before,p));self.assertTrue(torch.equal(old,opt.state[p]['momentum_buffer']))
        self.assertTrue(torch.equal(grad,p.grad));self.assertEqual(opt.actual_steps,0)

    def test_aligned_momentum_is_bitwise_ordinary_sgd(self):
        p,opt = self.make([1.,2.],[3.,4.],weight_decay=.0001)
        q = torch.nn.Parameter(p.detach().clone())
        old = torch.optim.SGD([q],lr=.0025,momentum=.9,weight_decay=.0001)
        q.grad = p.grad.clone(); old.state[q]['momentum_buffer'] = opt.state[p]['momentum_buffer'].clone()
        opt.step();old.step()
        self.assertFalse(opt.last_restart['momentum_restarted'])
        self.assertTrue(torch.equal(p,q))
        self.assertTrue(torch.equal(opt.state[p]['momentum_buffer'],old.state[q]['momentum_buffer']))

    def test_opposing_momentum_removed_without_changing_gradient(self):
        p,opt = self.make([1.,2.],[-30.,-40.],weight_decay=.0001)
        before = p.detach().clone(); grad = p.grad.clone()
        expected = grad+.0001*before
        opt.step()
        self.assertTrue(opt.last_restart['momentum_restarted'])
        torch.testing.assert_close(p,before-.0025*expected)
        torch.testing.assert_close(opt.state[p]['momentum_buffer'],expected)
        self.assertTrue(torch.equal(grad,p.grad));self.assertEqual(opt.restart_count,1)

    def test_no_history_and_zero_dot_do_not_restart(self):
        p,opt = self.make([1.,0.],[0.,2.])
        opt.step();self.assertFalse(opt.last_restart['momentum_restarted'])
        p,opt = self.make([1.,2.],[3.,4.]);opt.state[p].clear()
        opt.step();self.assertFalse(opt.last_restart['momentum_restarted'])

    def test_unused_parameter_momentum_retained(self):
        p,opt = self.make([1.,2.],[-3.,-4.])
        q = torch.nn.Parameter(torch.ones(2));opt.add_param_group(dict(params=[q]))
        opt.state[q]['momentum_buffer'] = torch.tensor([7.,8.])
        before = q.clone();old = opt.state[q]['momentum_buffer'].clone()
        opt.step()
        self.assertTrue(torch.equal(q,before));self.assertTrue(torch.equal(opt.state[q]['momentum_buffer'],old))

    def test_nonfinite_and_unsupported_contract_stop_before_writes(self):
        from crane_project.utils.port_oof_momentum_restart_v1 import momentum_restart_decision
        p,opt = self.make([1.,2.],[float('nan'),4.])
        before = p.clone()
        with self.assertRaises(FloatingPointError):opt.step()
        self.assertTrue(torch.equal(p,before));self.assertEqual(opt.actual_steps,0)
        for field,value in [('nesterov',True),('dampening',.1)]:
            p,opt = self.make([1.,2.],[3.,4.]);opt.param_groups[0][field] = value
            with self.assertRaises(ValueError):momentum_restart_decision(opt)

    def test_config_is_opt_in_and_preserves_original_contract(self):
        from mmcv import Config
        from pathlib import Path
        from crane_project.utils.port_oof_momentum_restart_v1 import configure,check_config
        from crane_project.utils.port_oof_task_budget_v1 import check_task_config
        from crane_project.utils.port_reliability_oof_training_v2 import repair_config
        from crane_project.utils.port_reliability_oof_prevention_v1 import configure as prevention
        from crane_project.utils.port_oof_task_budget_v1 import configure as budget
        path = Path(__file__).resolve().parents[1]/'crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py'
        cfg = Config.fromfile(str(path));repair_config(cfg)
        cfg.data.val = {};cfg.data.test = {};cfg.evaluation = None
        cfg.workflow = [('train',1)];cfg.load_from = cfg.resume_from = None;cfg.auto_resume = False
        cfg.seed = 1701;cfg.runner.max_epochs = 24
        prevention(cfg,'unused');budget(cfg)
        original = deepcopy(cfg)
        configure(cfg);check_config(cfg);check_task_config(original)
        for field in ('model','data','lr_config','seed','runner','load_from','resume_from'):
            self.assertEqual(cfg[field],original[field])
        cfg.optimizer.lr = .001
        with self.assertRaises(ValueError):check_config(cfg)

    def test_real_hook_loss_and_memory_guards_do_not_mutate_momentum(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from mmrotate.core.hooks.port_oof_momentum_restart_hook import PortOOFMomentumRestartHook
        from mmrotate.core.hooks.port_oof_prevention_hook import PreventionStopped
        from crane_project.utils.port_oof_task_budget_v1 import VERSION as TASK, MEMORY_REVISION
        from crane_project.utils.port_reliability_oof_prevention_v1 import SETTINGS,MEMORY_SETTINGS
        class Tiny(torch.nn.Module):
            def __init__(self,bad):
                super().__init__();self.p = torch.nn.Parameter(torch.zeros(2));self.bad = bad
                self.bbox_head = SimpleNamespace(assigner=SimpleNamespace(_local_call_count=0))
            def forward_train(self,img_metas):
                return dict(loss_cls=self.p[0]*1000+(5. if self.bad else 0.),
                            loss_bbox=self.p[1]*1000,aux0_loss_cls=self.p.sum()*.1)
        for failure in ('normal','loss','memory'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as d:
                m = Tiny(failure=='loss')
                from mmrotate.core.hooks.port_oof_momentum_restart_hook import PortOOFMomentumRestartSGD
                opt = PortOOFMomentumRestartSGD(m.parameters(),momentum_restart_revision=VERSION,
                    lr=.0025,momentum=.9)
                opt.state[m.p]['momentum_buffer'] = torch.tensor([-100.,-100.])
                old = opt.state[m.p]['momentum_buffer'].clone()
                runner = SimpleNamespace(model=m,optimizer=opt,iter=0,epoch=0,
                    log_buffer=SimpleNamespace(update=lambda *a:None))
                hook = PortOOFMomentumRestartHook(momentum_restart_revision=VERSION,
                    task_budget_revision=TASK,task_memory_revision=MEMORY_REVISION,
                    directory=str(Path(d)/'result'),settings=SETTINGS,memory_guard=MEMORY_SETTINGS,
                    grad_clip=dict(max_norm=10,norm_type=2))
                hook.before_run(runner);hook.before_train_iter(runner)
                losses = m.forward_train([dict(filename='sim_train.jpg')])
                runner.outputs = dict(loss=sum(losses.values()),num_samples=1)
                with patch('torch.cuda.synchronize'),patch('torch.cuda.memory_stats',return_value={}),\
                     patch('torch.cuda.memory_allocated',return_value=0),\
                     patch('torch.cuda.memory_reserved',return_value=0),\
                     patch('torch.cuda.max_memory_allocated',return_value=0),\
                     patch('torch.cuda.max_memory_reserved',return_value=0):
                    if failure=='memory':
                        hook.memory_guard.observe = lambda *a,**k:(_ for _ in ()).throw(RuntimeError('resource fixture'))
                    if failure=='normal':
                        hook.after_train_iter(runner)
                        self.assertEqual(opt.actual_steps,1);self.assertEqual(hook.completed,1)
                        self.assertTrue(opt.last_restart['momentum_restarted'])
                        self.assertGreater(float(m.p.norm()),0.)
                        hook.finish('TEST_FIXTURE',runner)
                        import json
                        rows = [json.loads(x) for x in (Path(d)/'result/updates.jsonl').read_text().splitlines()]
                        self.assertTrue(rows[0]['momentum_restart']['momentum_restarted'])
                        report = json.loads((Path(d)/'result/completion.json').read_text())
                        self.assertEqual(report['momentum_restart_count'],1)
                    else:
                        with self.assertRaises(PreventionStopped if failure=='loss' else RuntimeError):
                            hook.after_train_iter(runner)
                if failure!='normal':
                    self.assertTrue(torch.equal(old,opt.state[m.p]['momentum_buffer']))
                    self.assertEqual(opt.actual_steps,0);self.assertEqual(hook.completed,0)


if __name__ == '__main__':
    unittest.main()
