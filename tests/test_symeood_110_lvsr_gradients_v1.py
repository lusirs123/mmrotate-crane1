import unittest

from crane_project.tools import diagnose_symeood_110_lvsr_gradients_v1 as d


class GradientDiagnosisTests(unittest.TestCase):
    def test_canonical_mapping_and_subset_denominators(self):
        import torch
        from crane_project.utils.port_size_qalign_v1 import residual
        # All four M/model-B order combinations, GT raw order also mixed.
        m = torch.tensor([[0., 0., 12., 6., 0., .8], [0., 0., 6., 12., 0., .8]]*2)
        bm = torch.tensor([[0., 0., 12., 6., 0.]]*2+[[0., 0., 6., 12., 0.]]*2)
        gt = torch.tensor([[0., 0., 11., 5., 0.], [0., 0., 5., 11., 0.]]*2)
        delta = torch.tensor([[.05, -.03], [.02, .07], [-.08, .01], [.03, -.05]], requires_grad=True)
        near = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.bool)
        terms, swap = d.decompose(dict(delta_roi=delta, vector_valid=torch.ones(4, dtype=torch.bool)), gt, m, bm, near, torch)
        self.assertEqual(swap.tolist(), [False, True, True, False])
        target = residual.size_targets(gt, m, bm)['log_residual']
        point_roi = torch.nn.functional.smooth_l1_loss(delta, target, reduction='none', beta=.1)
        point_canonical = torch.where(swap[:, None], point_roi.flip(-1), point_roi)
        gradients = {k:torch.autograd.grad(v, delta, retain_graph=True)[0] for k, v in terms.items()}
        self.assertTrue(torch.allclose(gradients['size'], (gradients['long']+gradients['short'])/2))
        for j, edge in enumerate(('long', 'short')):
            self.assertTrue(torch.allclose(terms[edge+'_near'], (point_canonical[:, j]*near[:, j]).sum()/4))
            self.assertTrue(torch.allclose(gradients[edge], gradients[edge+'_near']+gradients[edge+'_other']))
        self.assertIsNone(delta.grad)

    def test_sampling_is_fixed_stratified_and_quality_blind(self):
        rows = [dict(domain=dom, scale=sc, image=dom+str(i), midpoint=[1])
                for dom in ('real', 'sim') for sc in (1., .5) for i in range(80)]
        a = d.choose(rows)
        self.assertEqual(a, d.choose(list(reversed(rows))))
        self.assertEqual(len(a), 32)
        self.assertEqual(len({(r['image'],r['scale']) for b in a for r in b['views']}), 256)

    def test_zero_and_empty_relations_do_not_fabricate_cosines(self):
        import torch
        a = torch.tensor([1., 0.]); b = torch.tensor([-2., 0.]); total=(a+b)/2
        r = d.relation(a, b, total)
        self.assertEqual(r['cosine'], -1.)
        self.assertLess(r['a_dot_total'], 0)
        self.assertIsNone(d.relation(a, b*0, total)['cosine'])
        self.assertIsNone(d.relation(a, b, total, applicable=False)['dot'])


if __name__ == '__main__':
    unittest.main()
