"""Train-only state isolation, exact controls, Adam restoration and inference."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from crane_project.utils import port_size_pair_decoded_edge_proxy_v1 as p
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
from crane_project.tools.run_port_size_native_train_v1 import equal, rng_get, rng_set
from tests.test_lvsr_conr_v1 import inputs


def centers():
    return torch.stack((torch.linspace(-.1, .1, 20), torch.linspace(.08, -.08, 20)), 1).double()


def optimizer(head, bank):
    return torch.optim.Adam([{'params': head.parameters()}, {'params': bank.parameters()}], lr=.001)


class DecodedEdgeProxyTrainChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_head_init_rng_and_inference_same_with_training_state_external(self):
        torch.manual_seed(1703); old = PairSpecificSizeHead().double(); rng = torch.get_rng_state()
        torch.manual_seed(1703); head = p.PairDecodedEdgeProxyHead().double()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        bank = p.ProxyBank(centers())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(equal(old.state_dict(), head.state_dict(), torch))
        self.assertEqual(sum(q.numel() for q in head.parameters()), 2480)
        self.assertEqual(sum(q.numel() for q in bank.parameters()), 12960)
        x = inputs()
        for training in (False, True):
            old.train(training); head.train(training)
            a, b = old(*x[:-1]), head(*x[:-1])
            for key in a:
                if isinstance(a[key], torch.Tensor):self.assertTrue(torch.equal(a[key], b[key]), key)
            self.assertEqual('training_features' in b, training)
            self.assertEqual(len(head.stem._forward_hooks), 0)
        empty = head(*[q[:0] for q in x[:-1]])
        self.assertEqual(len(empty['boxes_original']), 0)

    def test_zero_weights_exact_two_adam_steps_and_original_clipping(self):
        from crane_project.tools.run_symeood_110_lvsr_v1 import update as original_update
        torch.manual_seed(1703); old = PairSpecificSizeHead().double()
        torch.manual_seed(1703); head = p.PairDecodedEdgeProxyHead().double(); bank = p.ProxyBank(centers())
        a = torch.optim.Adam(old.parameters(), lr=.001); b = optimizer(head, bank)
        before = deepcopy(bank.state_dict())
        for _ in range(2):
            aa = original_update(old, a, inputs(), 'fixed_b', torch)
            bb = p.update(head, bank, b, inputs(), 0., 0.)
            self.assertTrue(equal(aa, bb, torch))
            self.assertTrue(equal(old.state_dict(), head.state_dict(), torch))
            for q, r in zip(old.parameters(), head.parameters()):
                self.assertTrue(equal(a.state[q], b.state[r], torch))
        self.assertTrue(equal(before, bank.state_dict(), torch))
        self.assertFalse(any(q in b.state for q in bank.parameters()))

    def test_joint_update_changes_both_states_but_preserves_input_and_fixed_targets(self):
        torch.manual_seed(1703); head = p.PairDecodedEdgeProxyHead().double(); bank = p.ProxyBank(centers())
        x = inputs(); before_x = [q.clone() for q in x]
        before_head = deepcopy(head.state_dict()); before_proxy = bank.features.clone(); targets = bank.targets.clone()
        opt = optimizer(head, bank)
        stats = p.update(head, bank, opt, x)
        self.assertTrue(torch.isfinite(torch.tensor(stats['loss'])))
        self.assertFalse(equal(before_head, head.state_dict(), torch))
        self.assertFalse(torch.equal(before_proxy, bank.features)); self.assertTrue(torch.equal(targets, bank.targets))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(x, before_x)))
        self.assertEqual({int(opt.state[q]['step']) for q in list(head.parameters())+list(bank.parameters())}, {1})
        out = head(*x[:-1])
        self.assertTrue(torch.equal(out['boxes_original'][:, [0, 1, 4, 5]], x[-2][:, [0, 1, 4, 5]]))

    def test_exact_joint_optimizer_rng_reload_and_next_update(self):
        torch.manual_seed(1703); head = p.PairDecodedEdgeProxyHead().double(); bank = p.ProxyBank(centers())
        opt = optimizer(head, bank); x = inputs()
        for _ in range(2):p.update(head, bank, opt, x)
        state = dict(head=deepcopy(head.state_dict()), proxy=deepcopy(bank.state_dict()),
                     optimizer=deepcopy(opt.state_dict()), rng=rng_get(torch))
        clone = p.PairDecodedEdgeProxyHead().double(); other = p.ProxyBank(centers())
        clone.load_state_dict(state['head']); other.load_state_dict(state['proxy'])
        newopt = optimizer(clone, other); newopt.load_state_dict(state['optimizer'])
        rng_set(state['rng'], torch); a = p.update(head, bank, opt, x)
        rng_set(state['rng'], torch); b = p.update(clone, other, newopt, x)
        self.assertTrue(equal(a, b, torch))
        self.assertTrue(equal(head.state_dict(), clone.state_dict(), torch))
        self.assertTrue(equal(bank.state_dict(), other.state_dict(), torch))
        self.assertTrue(equal(opt.state_dict(), newopt.state_dict(), torch))

    def test_deployment_does_not_depend_on_proxy_bank(self):
        torch.manual_seed(1703); head = p.PairDecodedEdgeProxyHead().double(); bank = p.ProxyBank(centers())
        p.update(head, bank, optimizer(head, bank), inputs())
        deploy = PairSpecificSizeHead().double().eval(); deploy.load_state_dict(head.state_dict(), strict=True)
        with torch.no_grad():
            a = head.eval()(*inputs()[:-1]); b = deploy(*inputs()[:-1])
        for key in a:
            if isinstance(a[key], torch.Tensor):self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_matches_reviewed_objective_and_does_not_route_through_terminal(self):
        from crane_project.utils import lvsr_decoded_edge_proxy_review_v1 as r
        from crane_project.utils import lvsr_prime_review_v1 as prime
        from crane_project.utils import port_size_qalign_v1 as g
        torch.manual_seed(1703);head=p.PairDecodedEdgeProxyHead().double();bank=p.ProxyBank(centers())
        x=inputs();_,alignment,proxy,_=p.objectives(head,bank,x)
        out=head(*x[:-1]);n=len(x[-2])
        v=g.residual.original.roi_delta_to_original(out['vectors'].reshape(n,324,2),x[2],x[3],x[4]).reshape(n,81,4,2)
        lengths=torch.stack(((v[:,:,0]-v[:,:,2]).norm(dim=-1),(v[:,:,1]-v[:,:,3]).norm(dim=-1)),-1)
        masks,_,_=r.masks(lengths,out['common_support'])
        target=prime.canonical_pair(g.residual.size_targets(x[-1],x[-2],x[3])['log_residual'],x[3],x[-2])
        ref=r.objectives(out['training_features'],masks,target,bank.features,bank.targets)
        self.assertTrue(torch.equal(alignment,ref['alignment'].mean()))
        self.assertTrue(torch.equal(proxy,ref['proxy'].mean()))
        self.assertTrue(all(v is None for v in torch.autograd.grad(alignment,tuple(head.terminal.parameters()),allow_unused=True)))

    def test_zero_selected_descriptor_stops_before_any_optimizer_step(self):
        torch.manual_seed(1703);head=p.PairDecodedEdgeProxyHead().double();bank=p.ProxyBank(centers())
        with torch.no_grad():
            for branch in head.stem.branches:
                branch[2].weight.zero_();branch[2].bias.fill_(-1)
        before=deepcopy(head.state_dict());proxy=deepcopy(bank.state_dict());opt=optimizer(head,bank)
        with self.assertRaises(p.InvalidAuxiliaryDescriptor):p.update(head,bank,opt,inputs())
        self.assertFalse(opt.state)
        self.assertTrue(equal(before,head.state_dict(),torch));self.assertTrue(equal(proxy,bank.state_dict(),torch))

    def test_zero_selected_proxy_stops_before_any_optimizer_step(self):
        torch.manual_seed(1703);head=p.PairDecodedEdgeProxyHead().double();bank=p.ProxyBank(centers())
        with torch.no_grad():bank.features.zero_()
        before=deepcopy(head.state_dict());opt=optimizer(head,bank)
        with self.assertRaises(p.InvalidAuxiliaryDescriptor):p.update(head,bank,opt,inputs())
        self.assertFalse(opt.state);self.assertTrue(equal(before,head.state_dict(),torch))

    def test_deployment_does_not_construct_auxiliary_masks(self):
        torch.manual_seed(1703);head=p.PairDecodedEdgeProxyHead().double().eval()
        with patch('crane_project.utils.lvsr_decoded_edge_proxy_review_v1.masks',side_effect=AssertionError('Inference called auxiliary routing')):
            out=head(*inputs()[:-1])
        self.assertEqual(len(out['boxes_original']),len(inputs()[-2]))


if __name__ == '__main__':
    unittest.main()
