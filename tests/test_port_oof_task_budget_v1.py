import unittest
from crane_project.utils.port_oof_task_budget_v1 import BUDGET,TASKS,VERSION
try:
    import torch
except ImportError:
    torch=None


class ContractTests(unittest.TestCase):
    def test_fixed_budget_and_revision(self):
        self.assertEqual(BUDGET,10.)
        self.assertEqual(TASKS,('main_cls','main_reg','auxiliary'))
        self.assertEqual(VERSION,'per_task_gradient_budget_v1')


@unittest.skipIf(torch is None,'Torch required')
class BackwardTests(unittest.TestCase):
    def backward(self,terms,params):
        from crane_project.utils.port_oof_task_budget_v1 import task_budget_backward
        return task_budget_backward(terms,params)

    def test_large_classification_and_regression_are_both_bounded(self):
        p=torch.nn.Parameter(torch.zeros(3))
        terms={n:p[i]*v for i,(n,v) in enumerate(zip(TASKS,(6570.,3048.,27.6)))}
        s=self.backward(terms,[p])
        torch.testing.assert_close(p.grad,torch.full((3,),10.))
        self.assertLess(s['gradient_before_clip'],30.)
        self.assertTrue(all(x['budgeted_norm']<=10. for x in s['component_gradients'].values()))

    def test_small_gradients_equal_ordinary_total_backward(self):
        p=torch.nn.Parameter(torch.tensor([.2,-.4]))
        terms=dict(main_cls=p.square().sum(),main_reg=(p*.1).sum(),auxiliary=(p*.3).sum())
        expected=torch.autograd.grad(sum(terms.values()),p,retain_graph=True)[0]
        self.backward(terms,[p])
        torch.testing.assert_close(p.grad,expected)

    def test_cancellation_zero_and_unused_parameters(self):
        p=torch.nn.Parameter(torch.tensor(1.))
        q=torch.nn.Parameter(torch.tensor(2.))
        terms=dict(main_cls=p*1e12,main_reg=-p*1e12,auxiliary=p*0.)
        s=self.backward(terms,[p,q])
        self.assertEqual(float(p.grad),0.)
        self.assertIsNone(q.grad)
        self.assertEqual(s['component_gradients']['auxiliary']['scale'],1.)

    def test_reject_stale_gradients_and_invalid_task_contract(self):
        p=torch.nn.Parameter(torch.tensor(1.))
        p.grad=torch.ones_like(p)
        with self.assertRaisesRegex(ValueError,'Clear gradients'):
            self.backward({n:p for n in TASKS},[p])
        p.grad=None
        with self.assertRaisesRegex(ValueError,'three reviewed'):
            self.backward({'main_reg':p},[p])

    def test_nonfinite_component_cannot_update(self):
        p=torch.nn.Parameter(torch.tensor(1.))
        for bad in (float('nan'),float('inf')):
            p.grad=None
            with self.assertRaises(FloatingPointError):
                self.backward(dict(main_cls=p*bad,main_reg=p,auxiliary=p),[p])

    def test_actual_hook_keeps_loss_and_resource_stops_before_optimizer(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from mmrotate.core.hooks.port_oof_task_budget_hook import PortOOFTaskBudgetHook
        from mmrotate.core.hooks.port_oof_prevention_hook import PreventionStopped
        from crane_project.utils.port_reliability_oof_prevention_v1 import SETTINGS,MEMORY_SETTINGS
        class Tiny(torch.nn.Module):
            def __init__(self,bad_loss):
                super().__init__()
                self.p=torch.nn.Parameter(torch.zeros(3))
                self.bad_loss=bad_loss
                self.bbox_head=SimpleNamespace(assigner=SimpleNamespace(_local_call_count=0))
            def forward_train(self,img_metas):
                return dict(loss_cls=self.p[0]*1000+(5. if self.bad_loss else 0.),
                            loss_bbox=self.p[1]*1000,aux0_loss_cls=self.p[2]*1000)
        class Log:
            def update(self,*args):pass
        for case in ('normal','loss','memory'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as d:
                m=Tiny(case=='loss')
                opt=torch.optim.SGD(m.parameters(),lr=.1,momentum=.9)
                runner=SimpleNamespace(model=m,optimizer=opt,iter=0,epoch=0,log_buffer=Log())
                hook=PortOOFTaskBudgetHook(task_budget_revision=VERSION,directory=str(Path(d)/'trace'),
                    settings=SETTINGS,memory_guard=MEMORY_SETTINGS,grad_clip=dict(max_norm=10,norm_type=2))
                hook.before_run(runner)
                hook.before_train_iter(runner)
                terms=m.forward_train([dict(filename='real_train.jpg')])
                runner.outputs=dict(loss=sum(terms.values()),num_samples=1)
                with patch('torch.cuda.synchronize'),patch('torch.cuda.memory_stats',return_value={}),\
                     patch('torch.cuda.memory_allocated',return_value=0),patch('torch.cuda.memory_reserved',return_value=0),\
                     patch('torch.cuda.max_memory_allocated',return_value=0),patch('torch.cuda.max_memory_reserved',return_value=0):
                    if case=='loss':
                        with self.assertRaises(PreventionStopped):hook.after_train_iter(runner)
                    elif case=='memory':
                        with patch.object(hook.memory_guard,'observe',side_effect=RuntimeError('resource cap')):
                            with self.assertRaisesRegex(RuntimeError,'resource cap'):hook.after_train_iter(runner)
                    else:
                        hook.after_train_iter(runner)
                        self.assertAlmostEqual(float(m.p.norm()),1.,places=5)
                        self.assertEqual(hook.completed,1)
                        hook.finish('TEST_FIXTURE',runner)
                if case!='normal':
                    self.assertEqual(float(m.p.norm()),0.)
                    self.assertEqual(len(opt.state),0)
                    self.assertEqual(hook.completed,0)
                report=json.loads((Path(d)/'trace/completion.json').read_text())
                self.assertEqual(report['task_budget_revision'],VERSION)


if __name__=='__main__':unittest.main()
