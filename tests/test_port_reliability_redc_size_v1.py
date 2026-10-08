import inspect
import math
import unittest
from copy import deepcopy

import numpy as np

from crane_project.utils import port_reliability_redc_size_v1 as m
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_feature_ablation_v1 as ab

try:
    import torch
    from crane_project.utils import port_reliability_redc_size_v1_torch as native
except ImportError:
    torch = None


def row(i, bad=False, role='val', sequence='real_seq01'):
    return dict(image='%s_%05d'%(sequence,i),sequence=sequence,domain='real',
        split=role,frame_id=i,reliability_role='train' if role=='train' else 'val',
        pred=[0.,0.,12. if bad else 10.,4.,0.,.6],gt=[0.,0.,10.,4.,0.])


class NumericContract(unittest.TestCase):
    def test_spatial_pool_keeps_channels_and_positions(self):
        x=np.arange(32*81,dtype=float).reshape(32,9,9)
        expected=np.array([x[c,r*3:r*3+3,k*3:k*3+3].mean()
            for c in range(32) for r in range(3) for k in range(3)])
        np.testing.assert_array_equal(m.pooled_feature(x),expected)

    def test_no_global_channel_mean(self):
        x=np.zeros((32,9,9));x[1,0,0]=9.
        y=m.pooled_feature(x).reshape(32,3,3)
        self.assertEqual(y[1,0,0],1.);self.assertEqual(y[0].sum(),0.)

    def test_invalid_features_fail(self):
        for x in (np.zeros((256,9,9)),np.full((32,9,9),np.nan)):
            with self.assertRaises(ValueError):m.pooled_feature(x)

    def test_descriptor_matches_simple_and_width_swap(self):
        a=[20.,30.,10.,4.,.2,.8];b=[20.,30.,4.,10.,.2-math.pi/2,.8]
        x=m.descriptor(a,[100,200],np.arange(288))
        np.testing.assert_allclose(x,m.descriptor(b,[100,200],np.arange(288)))
        np.testing.assert_allclose(x[:3],simple.descriptor(a,[100,200]))

    def test_isotropic_descriptor_units(self):
        a=[20.,30.,10.,4.,.2,.8];b=[40.,60.,20.,8.,.2,.8]
        np.testing.assert_allclose(m.descriptor(a,[100,200],np.zeros(288)),
                                   m.descriptor(b,[200,400],np.zeros(288)))

    def test_no_GT_in_online_functions(self):
        for f in (m.pooled_feature,m.descriptor,m.decide):
            self.assertNotIn('gt',inspect.signature(f).parameters)
            self.assertNotIn('domain',inspect.signature(f).parameters)

    def test_missing_bypasses_features(self):
        self.assertIsNone(m.descriptor(None,[1,1],None))
        with self.assertRaises(ValueError):m.descriptor(None,[1,1],np.zeros(288))

    def test_only_size_changes(self):
        d=dict(final_box_original=[0,0,10,4,0,.8],center_accepted=True,
               size_accepted=True,angle_accepted=False,risks=dict(size=.1,angle=.9))
        before=deepcopy(d);v=m.decide(d,.7,.5)
        self.assertEqual(d,before);self.assertFalse(v['size_accepted'])
        v['size_accepted']=d['size_accepted'];v['risks']['size']=d['risks']['size']
        self.assertEqual(v,d)

    def test_missing_decision_preserves_identity(self):
        d=dict(final_box_original=None,center_accepted=False,size_accepted=False,
               angle_accepted=False,risks=dict(size=None,angle=None))
        self.assertEqual(m.decide(d,None,.5),d)
        with self.assertRaises(ValueError):m.decide(d,.1,.5)

    def test_normalizer_is_TRAIN_only_fixed(self):
        x=np.arange(3*291,dtype=float).reshape(3,291);x[:,20]=1
        n=m.normalization(x);self.assertEqual(n['scale'][20],1)
        np.testing.assert_allclose(m.normalize(x,n).mean(0),0.,atol=1e-12)
        before=deepcopy(n);m.normalize(x+1e6,n);self.assertEqual(n,before)

    def test_full_class_weights(self):
        y=np.array([0]*98+[1]*2);w=m.class_weights(y)
        self.assertAlmostEqual(w[y==0].sum()/100,.5)
        self.assertAlmostEqual(w[y==1].sum()/100,.5)
        losses=np.linspace(.1,2.,100)
        self.assertAlmostEqual(np.mean(losses*w),.5*losses[:98].mean()+.5*losses[98:].mean())
        with self.assertRaises(ValueError):m.class_weights([0,0])

    def test_threshold_VAL_only(self):
        rows=[row(i) for i in range(20)]+[row(20,True)]
        risks={r['image']:i/25 for i,r in enumerate(rows)}
        point=ab.calibrate(rows,risks,.95)
        self.assertEqual(point['risk_le'],18/25)
        with self.assertRaises(ValueError):ab.calibrate([row(0,role='train')],{rows[0]['image']:.1})

    def test_threshold_protects_each_video(self):
        rows=[row(i) for i in range(20)]+[row(i,sequence='real_seq02') for i in range(20)]
        risks={r['image']:(.8 if r['sequence']=='real_seq02' else .1) for r in rows}
        self.assertEqual(ab.calibrate(rows,risks,.95)['risk_le'],.8)

    def test_correct_rejection_separate_from_error_detection(self):
        rows=[row(0,True),row(1,True),row(2),row(3)]
        scores={k:{r['image']:.5 for r in rows} for k in ('redc',*m.CONTROLS)}
        stats=m.describe(rows,scores,'redc',.4)['all']['actual']
        self.assertEqual(stats['states'],dict(FA=0,FR=2,ED=2,CR=0,MISSING=0))
        self.assertEqual(stats['runs']['correct_rejection']['longest'],2)
        self.assertEqual(stats['runs']['unavailable']['longest'],4)

    def test_same_CR_accepts_whole_tie(self):
        rows=[row(0),row(1,True),row(2)]
        scores={k:{r['image']:.5 for r in rows} for k in ('redc',*m.CONTROLS)}
        scores['redc']={rows[0]['image']:.1,rows[1]['image']:.8,rows[2]['image']:.8}
        stats=m.describe(rows,scores,'redc',.2)['all']
        self.assertEqual(stats['matched_CR_controls']['full_simple']['states']['CR'],2)
        self.assertFalse(stats['matched_CR_controls']['full_simple']['exact_CR'])

    def test_same_count_GT_never_selects_members(self):
        rows=[row(0,True),row(1),row(2)]
        scores={k:{r['image']:.5 for r in rows} for k in ('redc',*m.CONTROLS)}
        scores['redc'][rows[0]['image']]=.1
        v=m.describe(rows,scores,'redc',.2)['all']
        self.assertEqual(v['same_count_controls']['full_simple']['states']['FA'],1)
        self.assertEqual(v['control_tie_bounds']['full_simple']['bad_min_over_tie'],0)

    def test_labels_are_final_size_not_score(self):
        a=row(0,True);a['pred'][5]=.999
        self.assertTrue(ab.size_bad(a))
        a['pred'][2]=10.;a['pred'][5]=.06
        self.assertFalse(ab.size_bad(a))

    def test_missing_not_confusion(self):
        rows=[row(0),dict(row(1),pred=None)]
        scores={k:{rows[0]['image']:.1,rows[1]['image']:None} for k in ('redc',*m.CONTROLS)}
        s=m.describe(rows,scores,'redc',.5)['all']['actual']
        self.assertEqual(s['states']['CR'],1);self.assertEqual(s['states']['MISSING'],1)
        self.assertEqual(s['correct_coverage_all_frames'],.5)


@unittest.skipIf(torch is None,'Torch unavailable locally; full tests required on server')
class TorchContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.raw=np.random.RandomState(1701).normal(size=(12,291));self.raw[:,0]=np.linspace(-4,4,12)
        self.n=m.normalization(self.raw)
        self.x=torch.tensor(m.normalize(self.raw,self.n),dtype=torch.float32)
        self.z=torch.tensor(self.raw[:,0],dtype=torch.float32)
        self.models=native.make_models(self.n,'cpu')

    def test_initial_score_identity_and_parameter_counts(self):
        for model in self.models.values():
            torch.testing.assert_close(model(self.x,self.z),-self.z,rtol=0,atol=1e-5)
        self.assertEqual(sum(p.numel() for p in self.models['linear'].parameters()),292)
        self.assertEqual(sum(p.numel() for p in self.models['redc'].parameters()),4802)

    def test_numpy_matches_nontrivial_network(self):
        model=self.models['redc']
        with torch.no_grad():model.network[4].weight.fill_(.13);model.beta.fill_(.2)
        ref=m.numpy_logits(native.exported(model),self.raw,self.n,'redc')
        np.testing.assert_allclose(model(self.x,self.z).detach().numpy(),ref,rtol=1e-5,atol=1e-5)

    def test_feature_gradient_after_neutral_step(self):
        model=self.models['redc'];opt=native.optimizer(model)
        y=torch.tensor([0,1]*6,dtype=torch.float32);w=torch.ones(12)
        native.update(model,opt,self.x,self.z,y,w)
        v=native.update(model,opt,self.x,self.z,y,w)
        self.assertGreater(v['gradient_by_parameter']['network.0.weight'],0)

    def test_clipping_and_input_detachment(self):
        model=self.models['linear'];opt=native.optimizer(model)
        x=self.x.clone().requires_grad_();z=self.z.clone().requires_grad_()
        v=native.update(model,opt,x,z,torch.ones(12),torch.full((12,),1e5))
        self.assertIsNone(x.grad);self.assertIsNone(z.grad)
        self.assertGreater(v['gradient_norm_before'],5.)
        self.assertLessEqual(v['gradient_norm_after'],5.0001)

    def test_balanced_loss_gradient_matches_finite_difference(self):
        u=torch.tensor([-.7,.3,1.2],dtype=torch.float64,requires_grad=True)
        y=torch.tensor([0.,1.,0.],dtype=torch.float64);w=torch.tensor(m.class_weights(y.numpy()))
        loss=native.balanced_loss(u,y,w);loss.backward()
        expected=(simple.sigmoid(u.detach().numpy())-y.numpy())*w.numpy()/3
        np.testing.assert_allclose(u.grad.numpy(),expected,atol=1e-12)
        for i in range(3):
            a=u.detach().clone();b=a.clone();a[i]+=1e-6;b[i]-=1e-6
            numeric=(native.balanced_loss(a,y,w)-native.balanced_loss(b,y,w))/(2e-6)
            self.assertAlmostEqual(float(numeric),float(u.grad[i]),places=9)

    def test_biases_excluded_from_weight_decay(self):
        model=self.models['redc'];opt=native.optimizer(model)
        exempt={id(p) for g in opt.param_groups if g['weight_decay']==0 for p in g['params']}
        self.assertIn(id(model.beta),exempt)
        for n,p in model.named_parameters():
            self.assertEqual(id(p) in exempt,not n.endswith('weight'))

    def test_temperature_floor_is_positive(self):
        model=self.models['redc']
        with torch.no_grad():model.network[4].bias.fill_(-1000.)
        torch.testing.assert_close(model(self.x,self.z),-self.z/.25)

    def test_serialized_heads_roundtrip(self):
        for arm,model in self.models.items():
            reference=m.numpy_logits(native.exported(model),self.raw,self.n,arm)
            np.testing.assert_allclose(reference,model(self.x,self.z).detach().numpy(),atol=1e-5)


if __name__=='__main__':unittest.main()
