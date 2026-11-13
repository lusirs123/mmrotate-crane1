"""Sampling isolation and the eight-bias training control."""
import unittest
import torch
from crane_project.tools import diagnose_lvsr_small_fit_v1 as d
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead


def record(seq,domain,i):
    m=torch.tensor([[100.,100.,100.,60.,.2,.9]])
    gt=m[:,:5].clone()
    gt[:,2:4]*=torch.tensor([.95,1.,1.05])[torch.tensor([i%3,(i//3)%3])]
    return dict(image=seq+str(i),sequence=seq,domain=domain,frame_id=i*10,split='train',scale=1.,
                midpoint_original=m,gt_original=gt)


class SmallFitChecks(unittest.TestCase):
    def test_sampling_excludes_entire_holdout_and_non_native_views(self):
        rr=[record(seq,domain,i) for seq,domain in [('real_a','real'),('real_z','real'),('sim_a','sim')] for i in range(90)]
        extra=dict(rr[0],image='half',scale=.5);val=dict(rr[0],image='VAL',split='val')
        fit,held,cells,name=d.select_fit(rr+[extra,val])
        self.assertEqual(name,'real_z');self.assertEqual(len(held),90);self.assertEqual(len(fit),128)
        self.assertEqual(sum(r['domain']=='sim' for r in fit),64)
        self.assertEqual(len({r['image'] for r in fit}),128)
        self.assertTrue(all(r['scale']==1. and r['split']=='train' and r['sequence']!=name for r in fit))
        again=d.select_fit(list(reversed(rr))+[val,extra])[0]
        self.assertEqual([r['image'] for r in fit],[r['image'] for r in again])
        self.assertEqual(sum(q['selected'] for q in cells),128)

    def test_bias_control_keeps_same_initial_tensors_and_only_updates_eight_biases(self):
        torch.manual_seed(1703);a=PairSpecificSizeHead();state=torch.get_rng_state()
        torch.manual_seed(1703);b=PairSpecificSizeHead();active=d.configure(b,'terminal_bias_only8')
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in active),8)
        self.assertEqual(sum(p.numel() for p in d.configure(a,'original_full2480')),2480)
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]))
        # Geometry and forward are inherited; exercise the real update on CPU.
        from tests.test_lvsr_conr_v1 import inputs
        b=b.double();active=[p for p in b.parameters() if p.requires_grad]
        before={k:v.clone() for k,v in b.state_dict().items()}
        opt=torch.optim.Adam(active,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        x=inputs();stats=d.step(b,opt,x)
        self.assertGreater(stats['gradient_norm'],0.)
        changed=[]
        for k,v in b.state_dict().items():
            if not torch.equal(v,before[k]):changed.append(k)
        self.assertEqual(set(changed),{'terminal.branches.0.bias','terminal.branches.1.bias'})
        self.assertEqual({int(opt.state[p]['step']) for p in active},{1})

    def test_direction_uses_neutral_band(self):
        self.assertEqual([d.direction(v) for v in (-.1,0.,.1)],[0,1,2])


if __name__=='__main__':torch.set_num_threads(1);unittest.main()
