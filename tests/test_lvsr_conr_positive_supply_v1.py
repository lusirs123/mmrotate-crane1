"""Index isolation, exact empty control and positive-only rectangular loss."""
import unittest
import torch
from crane_project.utils import lvsr_conr_positive_supply_v1 as p
from crane_project.utils import port_size_pair_conr_v1 as c


def record(i,sequence,frame,target=0.,scale=1.):
    m=torch.tensor([[150.,180.,100.,60.,.2,.9]])
    gt=m[:,:5].clone();gt[:,2:4]*=torch.exp(torch.tensor(target))
    return dict(image='img'+str(i),sequence=sequence,frame_id=frame,split='train',scale=scale,
        midpoint_original=m,gt_original=gt,boxes_model=m[:,:5].clone())


def output(f):return dict(training_features=f,common_support=torch.ones(len(f),1,1,1,dtype=f.dtype))


class PositiveSupplyChecks(unittest.TestCase):
    def test_index_cross_video_priority_and_image_gap_exclusion(self):
        rr=[record(0,'a',0),record(1,'a',50,.001),record(2,'b',5,.01),record(3,'a',1),record(0,'a',0,scale=.5),record(4,'b',8,.2)]
        idx=p.PositiveIndex(rr)
        self.assertEqual(idx.candidates[0][0],2)
        self.assertNotIn(3,idx.candidates[0]);self.assertNotIn(4,idx.candidates[0])
        result=idx.select([0,5]);self.assertEqual(result['partner_indices'][0],2)
        self.assertEqual(result['roles'][0],'cross_video')
        self.assertEqual(idx.select([0,1])['partner_indices'],[])
        rr[0]['split']='val'
        with self.assertRaises(ValueError):p.PositiveIndex(rr)

    def test_no_partner_exact_legacy_value_and_feature_gradient(self):
        torch.manual_seed(4)
        f=torch.randn(3,2,1,1,dtype=torch.float64,requires_grad=True)
        t=torch.tensor([[0.,0.],[.001,0.],[.1,.1]],dtype=f.dtype);pred=torch.zeros_like(t)
        a,_=c.contrastive_loss(f,torch.ones(3,1,1,1),t,pred)
        b,_=p.loss(output(f),t,pred,None,None,[])
        self.assertTrue(torch.equal(a,b))
        ga=torch.autograd.grad(a,f,retain_graph=True)[0];gb=torch.autograd.grad(b,f)[0]
        self.assertTrue(torch.equal(ga,gb))

    def test_external_positive_adds_missing_anchor_without_negative_pool_change(self):
        f=torch.tensor([[[[1.]],[[.4]]],[[[.2]],[[1.]]],[[[.3]],[[.9]]]],dtype=torch.float64,requires_grad=True)
        partner=torch.tensor([[[[.8]],[[.1]]]],dtype=f.dtype,requires_grad=True)
        t=torch.tensor([[0.,0.],[.1,0.],[.105,0.]],dtype=f.dtype);pred=torch.zeros_like(t)
        old,stats=c.contrastive_loss(f,torch.ones(3,1,1,1),t,pred)
        new,ss=p.loss(output(f),t,pred,output(partner),torch.tensor([[.001,0.]],dtype=f.dtype),[(0,0)])
        self.assertEqual(stats['active_anchors'],2);self.assertEqual(ss['active_anchors'],3)
        self.assertEqual(ss['negative_pairs'],stats['negative_pairs']);self.assertEqual(ss['positive_pairs'],stats['positive_pairs']+1)
        new.backward();self.assertTrue(torch.isfinite(partner.grad).all());self.assertGreater(partner.grad.abs().sum().item(),0.)
        with self.assertRaises(ValueError):p.loss(output(f),t,pred,output(partner),t[:1],[(1,0)])
        with self.assertRaises(ValueError):p.loss(output(f),t,pred,output(partner),t[1:2],[(0,0)])

    def test_external_loss_matches_explicit_formula(self):
        f=torch.tensor([[[[1.]],[[0.]]],[[[0.]],[[1.]]]],dtype=torch.float64)
        ff=torch.tensor([[[[.8]],[[.6]]]],dtype=f.dtype)
        t=torch.tensor([[0.,0.],[.1,0.]],dtype=f.dtype)
        loss,stats=p.loss(output(f),t,torch.zeros_like(t),output(ff),t[:1],[(0,0)])
        # Anchor0 now has external positive, original anchor1 remains inactive.
        pos=.8/c.TEMPERATURE;neg=.01*(.1/c.WINDOW)
        expected=(torch.logsumexp(torch.tensor([pos,neg],dtype=f.dtype),0)-pos)/2
        torch.testing.assert_close(loss,expected)
        self.assertEqual(stats['active_anchors'],1)

if __name__=='__main__':torch.set_num_threads(1);unittest.main()
