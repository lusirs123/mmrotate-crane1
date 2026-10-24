"""Runtime checks for the observation-only C contract (mmrotljj)."""
import ast
import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[1]/'crane_project/tools/diagnose_symeood_c_frozen_v1.py'
spec=importlib.util.spec_from_file_location('c_frozen',PATH)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    import torch
except ImportError:
    torch=None


class SourceChecks(unittest.TestCase):
    def test_no_optimizer_or_update_calls(self):
        tree=ast.parse(PATH.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,('step','backward','clip_grad_norm_','unlink','rmtree'))
                if node.func.attr=='remove':
                    # Removing observation hooks is required; no file remove.
                    self.assertIn(ast.unparse(node.func.value) if hasattr(ast,'unparse') else
                                  node.func.value.attr,('self.h1','self.h2','h1','h2'))

    def test_split_scope_is_train_val(self):
        source=PATH.read_text()
        self.assertIn("for phase in ('val','train','gradient')",source)
        self.assertNotIn("data.test",source)


@unittest.skipIf(torch is None,'PyTorch unavailable locally; run in mmrotljj')
class RuntimeChecks(unittest.TestCase):
    def test_cross_ties_and_missing_own(self):
        a=torch.tensor([1.,2.,3.,float('inf'),1.])
        b=torch.tensor([2.,1.,3.,1.,float('inf')])
        self.assertEqual(module.tie_category(a,b,1e-6,1e-5).tolist(),[0,1,2,3,4])

    def test_numerically_close_reference_is_tie(self):
        a=torch.tensor([1.]);b=torch.tensor([1.+1e-7])
        self.assertEqual(module.tie_category(a,b,1e-6,1e-5).tolist(),[2])

    def losses(self):
        module.imports()
        from mmrotate.models.losses.sym_nfl_loss import SymNFLLoss
        logits=torch.tensor([[.3],[-.2],[.7],[.1]],requires_grad=True)
        boxes=torch.tensor([[0.,0.,10.,5.,0.],[1.,0.,10.,5.,0.],[100.,100.,10.,5.,0.],[200.,200.,10.,5.,0.]],requires_grad=True)
        gt=torch.tensor([[0.,0.,10.,5.,0.]])
        labels=torch.tensor([0,1,1,0]);weight=torch.tensor([1.,1.,.5,0.])
        kwargs=dict(tau_init=1.,tau_min=1.,warmup_iters=0,spatial_topk=2,loss_weight=.25)
        u=SymNFLLoss(use_geometry_weight=False,**kwargs)(logits,labels,boxes,gt,weight,avg_factor=2)
        g=SymNFLLoss(use_geometry_weight=True,**kwargs)(logits,labels,boxes,gt,weight,avg_factor=2)
        return logits,boxes,u,g

    def test_detached_geometry_and_strength_bounds(self):
        logits,boxes,u,g=self.losses()
        self.assertGreaterEqual(float(g),float(u))
        self.assertLessEqual(float(g),2*float(u))
        self.assertIsNone(torch.autograd.grad(g,boxes,retain_graph=True,allow_unused=True)[0])

    def test_negative_candidates_are_also_reweighted(self):
        logits,boxes,u,g=self.losses()
        gu=torch.autograd.grad(u,logits,retain_graph=True)[0]
        gg=torch.autograd.grad(g,logits)[0]
        self.assertGreater(float(gg[1].abs()),float(gu[1].abs()))
        self.assertEqual(float(gg[3]),0.)

    def test_loss_matching_is_not_gradient_matching(self):
        logits,boxes,u,g=self.losses()
        alpha=(g/u).detach();s=alpha*u
        torch.testing.assert_close(s,g)
        gg=torch.autograd.grad(g,logits,retain_graph=True)[0]
        gs=torch.autograd.grad(s,logits)[0]
        self.assertFalse(torch.allclose(gg,gs))

    def test_gradient_read_does_not_fill_grad_fields(self):
        logits,boxes,u,g=self.losses()
        torch.autograd.grad(g,logits)
        self.assertIsNone(logits.grad)
        self.assertIsNone(boxes.grad)


if __name__=='__main__':unittest.main()
