"""Focused U/S/G arithmetic, gradient and state tests; run in mmrotljj."""
import unittest
import json
import torch
from crane_project.utils.symeood_usg_v1 import SymNFLUSGV1, matched_objectives,encode_rng,decode_rng
from mmrotate.models.losses.sym_nfl_loss import SymNFLLoss


class USGTests(unittest.TestCase):
    def sample(self):
        z = torch.tensor([[-2.], [1.], [.3], [-5.]], requires_grad=True)
        labels = torch.tensor([0, 1, 1, 1])
        boxes = torch.tensor([[5.,5.,4.,2.,0.], [7.,5.,4.,2.,0.], [50.,5.,4.,2.,0.], [6.,5.,4.,2.,0.]], requires_grad=True)
        return z, labels, boxes, boxes[:1].detach()

    def test_native_u_g_loss_and_logit_gradient(self):
        for mode in ('U', 'G'):
            for it in (0, 4999, 20000):
                for empty in (False, True):
                    z, labels, boxes, gt = self.sample()
                    if empty:gt = gt[:0]
                    old = SymNFLLoss(use_geometry_weight=mode == 'G', loss_weight=.25)
                    new = SymNFLUSGV1(objective=mode, loss_weight=.25)
                    old._local_iter.fill_(it); new._local_iter.fill_(it)
                    kwargs = dict(weight=torch.tensor([1., .5, 1., 0.]), avg_factor=1.)
                    a = old(z, labels, boxes, gt, **kwargs)
                    b = new(z, labels, boxes, gt, **kwargs)
                    self.assertTrue(torch.equal(a, b))
                    self.assertTrue(torch.equal(torch.autograd.grad(a,z,retain_graph=True)[0], torch.autograd.grad(b,z,retain_graph=True)[0]))
                    self.assertTrue(torch.equal(old._local_iter,new._local_iter))

    def test_single_batch_scalar_not_levelwise(self):
        u = torch.tensor([1., 3.], requires_grad=True)
        g = u * torch.tensor([2., 1.])
        alpha, s, degenerate = matched_objectives(u.sum(),g.sum())
        self.assertFalse(degenerate); self.assertFalse(alpha.requires_grad)
        self.assertTrue(torch.equal(s,g.sum()))
        self.assertTrue(torch.equal(torch.autograd.grad(s,u)[0], torch.tensor([1.25,1.25])))

    def test_geometry_is_detached_from_box_branch(self):
        z,y,b,gt = self.sample(); loss=SymNFLUSGV1(objective='G')
        g=loss(z,y,b,gt,weight=torch.ones(4),avg_factor=1.)
        self.assertIsNone(torch.autograd.grad(g,b,allow_unused=True)[0])

    def test_no_positive_all_negative_and_empty_predictions(self):
        z,y,b,gt=self.sample(); y.fill_(1)
        for mode in ('U','G','S'):
            loss=SymNFLUSGV1(objective=mode)
            value=loss(z,y,b,gt[:0],weight=torch.ones(4),avg_factor=1.)
            self.assertTrue(torch.isfinite(value))
            empty=loss(z[:0],y[:0],b[:0],gt[:0],weight=torch.ones(0),avg_factor=1.)
            self.assertEqual(float(empty),0.)

    def test_zero_tiny_and_nonfinite_policy(self):
        for x in (0.,1e-14,1e-12):
            u=torch.tensor(x,requires_grad=True); alpha,s,d=matched_objectives(u,u*2)
            self.assertTrue(d); self.assertEqual(float(alpha),1.)
            self.assertEqual(float(s),float(u))
        with self.assertRaises(FloatingPointError):matched_objectives(torch.tensor(float('nan')),torch.tensor(1.))
        with self.assertRaises(FloatingPointError):matched_objectives(torch.tensor(1.),torch.tensor(3.))

    def test_nonuniform_weight_and_ignored_gradient(self):
        z,y,b,gt=self.sample();loss=SymNFLUSGV1(objective='S')
        loss(z,y,b,gt,weight=torch.tensor([1.,.5,1.,0.]),avg_factor=2.)
        r=loss.records[0];alpha,s,d=matched_objectives(r['U'],r['G'])
        grad=torch.autograd.grad(s,z)[0]
        self.assertEqual(float(grad[-1]),0.)
        self.assertIsNone(z.grad)

    def test_rng_payload_is_lossless_and_json_safe(self):
        original={'torch':torch.get_rng_state(),'counter':35}
        encoded=encode_rng(original)
        json.dumps({'rng':encoded})
        restored=decode_rng(encoded)
        self.assertTrue(torch.equal(original['torch'],restored['torch']))
        self.assertEqual(restored['counter'],35)


if __name__ == '__main__':unittest.main()
