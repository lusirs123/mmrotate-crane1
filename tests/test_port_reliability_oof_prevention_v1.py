import math
import unittest
from crane_project.utils import port_reliability_oof_prevention_v1 as prevention

try:
    import torch
except ImportError:
    torch = None


class ContractTests(unittest.TestCase):
    def test_cache_growth_and_short_active_spikes_are_allowed_below_cap(self):
        guard=prevention.CacheAwareMemoryGuard()
        for step in range(1,193):
            active=(450 if step in range(80,84) else 390)*2**20
            record=guard.observe(active,(2878 if step<90 else 3006)*2**20)
        self.assertEqual(guard.cache_growth_events,1)
        self.assertEqual(record['sustained_growth_checks'],0)

    def test_single_stable_active_jump_is_not_persistent_growth_at_any_alignment(self):
        for change_at in range(33,65):
            guard=prevention.CacheAwareMemoryGuard()
            for step in range(1,193):
                guard.observe((390 if step<change_at else 450)*2**20,3100*2**20)
            self.assertEqual(guard.growth_checks,0)

    def test_persistent_active_growth_stops(self):
        guard=prevention.CacheAwareMemoryGuard()
        with self.assertRaisesRegex(RuntimeError,'Persistent active'):
            for step in range(1,193):
                guard.observe((390+max(0,step-32))*2**20,3100*2**20)
        self.assertGreaterEqual(guard.growth_checks,2)

    def test_fixed_cap_includes_transient_peaks(self):
        for kw in (dict(reserved=4*2**30+1),dict(peak_reserved=4*2**30+1),dict(peak_allocated=4*2**30+1)):
            guard=prevention.CacheAwareMemoryGuard()
            arguments=dict(allocated=390*2**20,reserved=3006*2**20);arguments.update(kw)
            with self.assertRaisesRegex(RuntimeError,'fixed4GiB'):guard.observe(**arguments)

    def test_invalid_settings_and_counters_are_rejected(self):
        with self.assertRaises(ValueError):prevention.CacheAwareMemoryGuard(absolute_limit_bytes=8*2**30)
        with self.assertRaises(ValueError):prevention.CacheAwareMemoryGuard().observe(10,9)
    def test_fixed_limits(self):
        self.assertEqual(prevention.SETTINGS['max_updates'], 7000)
        self.assertEqual(prevention.SETTINGS['regression_budget'], 10.)
        self.assertEqual(prevention.SETTINGS['seed'], 1701)

    def test_mixed_gradient_and_original_loss_limits(self):
        losses = dict(main_cls=.5, main_reg=8., total=9.)
        self.assertEqual(prevention.reasons(losses, 60.), [])
        self.assertEqual(prevention.reasons(losses, 1000.), ['gradient_before_clip'])
        self.assertEqual(prevention.reasons(dict(losses, main_cls=5.), 60.), ['main_cls'])
        self.assertEqual(prevention.reasons(dict(losses, main_reg=10.), 60.), ['main_reg'])

    def test_nonfinite_total_and_auxiliary_cannot_pass(self):
        losses = dict(main_cls=.5, main_reg=8., total=float('nan'))
        self.assertIn('nonfinite', prevention.reasons(losses, 60.))
        self.assertIn('nonfinite', prevention.reasons(dict(losses, total=9.), float('inf')))


@unittest.skipIf(torch is None, 'Torch required')
class BackwardTests(unittest.TestCase):
    def test_scalar_finite_checks_reject_nan_inf_and_missing(self):
        for gradients in ([None], [torch.tensor([float('nan')])], [torch.tensor([float('inf')])]):
            with self.assertRaises(FloatingPointError):prevention.finite_gradient_norm(gradients)
        g=torch.tensor([3.,4.])
        self.assertEqual(float(prevention.finite_gradient_norm([g,None])),5.)

    def test_all_finite_checks_are_scalar_and_update_is_identical(self):
        from unittest.mock import patch
        from crane_project.utils.port_reliability_oof_gradient_chain_v1 import regression_budget_backward
        original=torch.isfinite;seen=[]
        def check(value):
            seen.append(value.numel());return original(value)
        p=torch.arange(64,dtype=torch.float32).requires_grad_(True)
        q=p.detach().clone().requires_grad_(True)
        weights=torch.arange(1,65,dtype=torch.float32)
        regression=(p*weights).sum()
        with patch('torch.isfinite',side_effect=check):
            prevention.measured_budget_backward(regression+3*p.sum(),regression,[p])
        r=(q*weights).sum();regression_budget_backward(r+3*q.sum(),r,[q])
        torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
        self.assertTrue(seen and all(n==1 for n in seen))

    def test_same_repair_and_raw_mixed_measurements(self):
        from crane_project.utils.port_reliability_oof_gradient_chain_v1 import regression_budget_backward
        p = torch.tensor([1., 2.], requires_grad=True)
        q = p.detach().clone().requires_grad_(True)
        stats = prevention.measured_budget_backward(103*p[0]+196*p[1], 100*p[0]+200*p[1], [p])
        regression_budget_backward(103*q[0]+196*q[1], 100*q[0]+200*q[1], [q])
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
        self.assertAlmostEqual(stats['raw_gradient_before_budget'], math.hypot(103,196), places=4)
        self.assertAlmostEqual(stats['gradient_before_clip'], float(p.grad.norm()), places=6)

    def test_large_fp32_regression_is_not_a_stop_if_mixed_is_small(self):
        p = torch.tensor(1., requires_grad=True)
        regression = 1e8*p
        stats = prevention.measured_budget_backward(regression+3*p, regression, [p])
        self.assertEqual(float(p.grad), 13.)
        self.assertEqual(prevention.reasons(dict(main_cls=.5, main_reg=1., total=1.5), stats['gradient_before_clip']), [])
        self.assertGreater(stats['raw_gradient_before_budget'], 1000.)

    def test_small_gradients_unchanged(self):
        p = torch.tensor([1.,2.], requires_grad=True)
        stats = prevention.measured_budget_backward(p.square().sum()+3*p.sum(), p.square().sum(), [p])
        torch.testing.assert_close(p.grad, torch.tensor([5.,7.]))
        self.assertEqual(stats['regression_gradient_scale'], 1.)

    def test_other_branch_explosion_still_stops(self):
        p = torch.tensor(1., requires_grad=True)
        stats = prevention.measured_budget_backward(100*p+10000*p, 100*p, [p])
        self.assertEqual(float(p.grad), 10010.)
        self.assertIn('gradient_before_clip', prevention.reasons(dict(main_cls=.5,main_reg=1.,total=1.5), stats['gradient_before_clip']))

    def test_nonfinite_regression_rejected(self):
        p = torch.tensor(1., requires_grad=True)
        with self.assertRaises(FloatingPointError):
            prevention.measured_budget_backward(p*float('inf'), p*float('inf'), [p])
        self.assertIsNone(p.grad)

    def test_missing_other_gradients(self):
        p = torch.tensor(1.,requires_grad=True); q = torch.tensor(1.,requires_grad=True)
        stats = prevention.measured_budget_backward(100*p+3*q, 100*p, [p,q])
        self.assertEqual(float(p.grad),10.)
        self.assertEqual(float(q.grad),3.)
        self.assertAlmostEqual(stats['raw_gradient_before_budget'], math.hypot(100,3), places=5)

    def test_hook_limits_prevent_optimizer_and_restore_preforward_buffers(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from mmrotate.core.hooks.port_oof_prevention_hook import PortOOFPreventionHook, PreventionStopped
        from crane_project.utils.port_reliability_oof_prevention_v1 import MEMORY_SETTINGS as MEMORY_GUARD
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.p=torch.nn.Parameter(torch.tensor(1.))
                self.register_buffer('running', torch.tensor(0.))
                self.bbox_head=SimpleNamespace(assigner=SimpleNamespace(_local_call_count=0))
            def forward_train(self, img_metas):
                self.running.add_(1); self.bbox_head.assigner._local_call_count+=1
                return dict(loss_bbox=self.p*2,loss_cls=self.p*5)
        class Log:
            def update(self,*args): pass
        with tempfile.TemporaryDirectory() as directory:
            m=Tiny(); optimizer=torch.optim.SGD(m.parameters(),lr=.1,momentum=.9)
            runner=SimpleNamespace(model=m,optimizer=optimizer,iter=0,epoch=0,log_buffer=Log())
            hook=PortOOFPreventionHook(directory=str(Path(directory)/'trace'),settings=prevention.SETTINGS,
                memory_guard=MEMORY_GUARD,grad_clip=dict(max_norm=10,norm_type=2))
            hook.before_run(runner); hook.before_train_iter(runner)
            terms=m.forward_train([dict(filename='real_train.jpg')])
            runner.outputs=dict(loss=sum(terms.values()),num_samples=1)
            with self.assertRaises(PreventionStopped): hook.after_train_iter(runner)
            self.assertEqual(float(m.p),1.)
            self.assertEqual(hook.completed,0)
            before=torch.load(str(Path(directory)/'trace/before_anomaly.pt'),map_location='cpu')
            self.assertEqual(float(before['model']['running']),0.)
            self.assertEqual(before['runtime']['assigner_local_call_count'],0)
            report=json.loads((Path(directory)/'trace/completion.json').read_text())
            self.assertEqual(report['status'],'ANOMALY_QUARANTINED_BEFORE_OPTIMIZER')

    def test_actual_hook_budgeted_update_and_resource_stop_before_step(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from mmrotate.core.hooks.port_oof_prevention_hook import PortOOFPreventionHook
        from crane_project.utils.port_reliability_oof_prevention_v1 import MEMORY_SETTINGS as MEMORY_GUARD
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.p=torch.nn.Parameter(torch.tensor(0.))
                self.bbox_head=SimpleNamespace(assigner=SimpleNamespace(_local_call_count=0))
            def forward_train(self, img_metas):
                return dict(loss_bbox=100*self.p,loss_cls=3*self.p)
        class Log:
            def update(self,*args): pass
        for resource_failure in (False, True):
            with self.subTest(resource_failure=resource_failure), tempfile.TemporaryDirectory() as directory:
                m=Tiny(); optimizer=torch.optim.SGD(m.parameters(),lr=.1,momentum=.9)
                runner=SimpleNamespace(model=m,optimizer=optimizer,iter=0,epoch=0,log_buffer=Log())
                hook=PortOOFPreventionHook(directory=str(Path(directory)/'trace'),settings=prevention.SETTINGS,
                    memory_guard=MEMORY_GUARD,grad_clip=dict(max_norm=10,norm_type=2))
                hook.before_run(runner); hook.before_train_iter(runner)
                terms=m.forward_train([dict(filename='real_train.jpg')])
                runner.outputs=dict(loss=sum(terms.values()),num_samples=1)
                with patch('torch.cuda.synchronize'), patch('torch.cuda.memory_allocated',return_value=0), patch('torch.cuda.memory_reserved',return_value=0):
                    if resource_failure:
                        with patch.object(hook.memory_guard,'observe',side_effect=RuntimeError('memory grows')):
                            with self.assertRaises(RuntimeError):hook.after_train_iter(runner)
                        self.assertEqual(float(m.p),0.)
                        self.assertEqual(hook.completed,0)
                        self.assertEqual(len(optimizer.state),0)
                    else:
                        hook.after_train_iter(runner)
                        self.assertAlmostEqual(float(m.p),-1.,places=6)
                        self.assertEqual(hook.completed,1)
                        self.assertIn(m.p,optimizer.state)
                        hook.finish('TEST_FIXTURE',runner)


if __name__ == '__main__':
    unittest.main()
