import inspect
import json
from copy import deepcopy
import unittest

import numpy as np

from crane_project.utils import port_reliability_readout_compare_v1 as core
from crane_project.utils import port_reliability_redc_size_v1 as old
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.tools import run_port_reliability_readout_compare_v1 as runner
from crane_project.tools import review_port_reliability_readout_compare_v1 as review

try:
    import torch
    from crane_project.utils import port_reliability_readout_compare_v1_torch as native
except ImportError:
    torch = None


def row(i, bad=False, score=.8):
    return dict(image='real_seq01_%05d'%i,sequence='real_seq01',domain='real',
        split='val',frame_id=i,reliability_role='val',pred=[0.,0.,12. if bad else 10.,4.,0.,score],
        gt=[0.,0.,10.,4.,0.])


def zero_model(bias=0., beta=0.):
    return {'network.0.weight':np.zeros((16,290)).tolist(),'network.0.bias':np.zeros(16).tolist(),
        'network.2.weight':np.zeros((8,16)).tolist(),'network.2.bias':np.zeros(8).tolist(),
        'network.4.weight':np.zeros((1,8)).tolist(),'network.4.bias':[bias],'beta':[beta]}


class NumericContract(unittest.TestCase):
    def setUp(self):
        self.f=np.zeros((2,291));self.f[:,0]=[2.,-2.]
        self.n=dict(mean=np.zeros(291).tolist(),scale=np.ones(291).tolist())

    def test_neutral_both_readouts(self):
        temp=zero_model(float(np.log(np.expm1(.75))))
        np.testing.assert_allclose(core.numpy_logits(temp,self.f,self.n,'temperature'),-self.f[:,0],atol=1e-14)
        np.testing.assert_array_equal(core.numpy_logits(zero_model(),self.f,self.n,'residual'),-self.f[:,0])

    def test_same_capacity_parameter_shapes(self):
        a=zero_model();b=zero_model(.5)
        self.assertEqual({k:np.shape(v) for k,v in a.items()},{k:np.shape(v) for k,v in b.items()})
        self.assertEqual(sum(np.size(v) for v in a.values()),4802)

    def test_temperature_cannot_cross_score_half(self):
        # Give the positive-score item a huge T and negative-score item a tiny T.
        m=zero_model();m['network.0.weight'][0][0]=1.;m['network.2.weight'][0][0]=1.
        m['network.4.weight'][0][0]=100.;f=self.f.copy();f[:,1]=[1.,0.]
        logits=core.numpy_logits(m,f,self.n,'temperature')
        self.assertLess(logits[0],logits[1])

    def test_residual_can_cross_score_half(self):
        m=zero_model();m['network.0.weight'][0][0]=1.;m['network.2.weight'][0][0]=1.
        m['network.4.weight'][0][0]=5.;f=self.f.copy();f[:,1]=[1.,0.]
        logits=core.numpy_logits(m,f,self.n,'residual')
        self.assertGreater(logits[0],logits[1])
        np.testing.assert_array_equal(logits,[3.,2.])

    def test_scalar_reviewer_independent_formula(self):
        rng=np.random.RandomState(4);f=rng.normal(size=(5,291));m=zero_model(.31,.17)
        for name,value in m.items():
            if name.endswith('weight'):m[name]=rng.normal(size=np.shape(value)).tolist()
        for arm in core.ARMS:
            ref,h=review.independent_logits(m,f,self.n,arm)
            np.testing.assert_allclose(ref,core.numpy_logits(m,f,self.n,arm),atol=1e-12)
            self.assertEqual(h.shape,(5,8))

    def test_GT_free_online_size_only(self):
        self.assertNotIn('gt',inspect.signature(core.decide).parameters)
        d=dict(final_box_original=[0,0,10,4,0,.8],center_accepted=True,
               size_accepted=True,angle_accepted=True,risks=dict(size=.1,angle=.2))
        before=deepcopy(d);out=core.decide(d,.9,.5)
        self.assertEqual(d,before);self.assertFalse(out['size_accepted'])
        out['risks']['size']=d['risks']['size'];out['size_accepted']=d['size_accepted']
        self.assertEqual(out,d)

    def test_missing_stays_missing(self):
        d=dict(final_box_original=None,center_accepted=False,size_accepted=False,
               angle_accepted=False,risks=dict(size=None,angle=None))
        self.assertEqual(core.decide(d,None,.5),d)
        with self.assertRaises(ValueError):core.decide(d,.7,.5)

    def test_cross_sign_counts_and_ties(self):
        rows=[row(0,True,.9),row(1,False,.1),row(2,True,.5),dict(row(3),pred=None)]
        scores={'residual':{r['image']:.3 for r in rows}}
        d=core.cross_sign_diagnostic(rows,scores,'residual')
        self.assertEqual((d['pairs'],d['ties'],d['correctly_ordered']),(1,1,0))
        self.assertEqual(d['pair_AUROC'],.5);self.assertEqual(d['score_exact_half'],1)
        self.assertIsNone(core.cross_sign_diagnostic([rows[0]],scores,'residual')['pair_AUROC'])

    def test_hidden_errors_use_original_size_labels(self):
        rows=[row(0,True),row(1)]
        d=core.hidden_diagnostic(rows,[r['image'] for r in rows],self.f,zero_model(),self.n)['all']
        self.assertEqual((d['outputs'],d['bad'],d['second_hidden_all_zero'],d['zero_bad']),(2,1,2,1))

    def test_fr_different_from_ed_and_missing(self):
        rows=[row(0,True),row(1),dict(row(2),pred=None)]
        counts,longest=review.counts(rows,set())
        self.assertEqual(counts,dict(FA=0,FR=1,ED=1,CR=0,MISSING=1));self.assertEqual(longest,1)

    def test_only_original_TRAIN_VAL_roles(self):
        for role in ('test','probe','fit','legacy','purged'):
            with self.assertRaises(ValueError):runner.checked_feature_rows(role,[])

    def test_pinned_features_tamper_detected(self):
        path=runner.ROOT/runner.SOURCE_DIR/'train_features.npz'
        self.assertEqual(runner.sha(path),runner.FEATURE_PINS[str(path.relative_to(runner.ROOT))])
        self.assertNotIn('TEST',''.join(runner.PINS))

    def test_fixed_contract_sources_and_budget(self):
        protocol,source=runner.checked_sources()
        self.assertEqual(protocol['settings']['epochs'],100)
        self.assertEqual(protocol['settings']['batch_size'],256)
        self.assertEqual(core.CONTROLS,('full_simple','score_only','temperature'))
        self.assertEqual(source['sources'][str(runner.PROTOCOL.relative_to(runner.ROOT))],runner.sha(runner.PROTOCOL))

    def test_cached_row_pairing_complete(self):
        import gzip
        rows=[{k:r[k] for k in runner.prior.FIELDS} for r in
              (json.loads(s) for s in gzip.open(runner.ROOT/runner.SOURCE_DIR/'scored_TRAIN.jsonl.gz','rt'))]
        part,x,ids=runner.checked_feature_rows('TRAIN',rows)
        self.assertEqual(len(part),2558);self.assertEqual(x.shape,(2558,291))
        changed=deepcopy(rows);changed[0]['pred'][2]*=1.01
        with self.assertRaises(ValueError):runner.checked_feature_rows('TRAIN',changed)

    def test_gate_requires_strict_gain_and_no_fr_relaxation(self):
        rows=[row(i) for i in range(20)]+[row(20,True)]
        scores={k:{r['image']:i/30 for i,r in enumerate(rows)} for k in ('residual',*core.CONTROLS)}
        stat=core.describe(rows,scores,'residual',.9)
        result=core.gate(stat)
        self.assertFalse(result['passed']);self.assertIsNone(result['selected_arm'])
        self.assertIn('strict_overall_same_count_FA_gain',[v['check'] for v in result['failures']])


@unittest.skipIf(torch is None,'Torch unavailable locally; required on server')
class TorchContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.raw=np.random.RandomState(1701).normal(size=(12,291));self.raw[:,0]=np.linspace(-4,4,12)
        self.n=core.normalization(self.raw)
        self.x=torch.tensor(core.normalize(self.raw,self.n),dtype=torch.float32)
        self.z=torch.tensor(self.raw[:,0],dtype=torch.float32)
        self.models=native.make_models(self.n,'cpu')

    def test_exact_neutral_and_shared_hidden(self):
        for model in self.models.values():
            torch.testing.assert_close(model(self.x,self.z),-self.z,rtol=0,atol=1e-5)
            self.assertEqual(sum(p.numel() for p in model.parameters()),4802)
        for n,p in self.models['temperature'].network.named_parameters():
            if not n.startswith('4.'):self.assertTrue(torch.equal(p,self.models['residual'].network.state_dict()[n]))

    def test_hidden_identical_to_old_seed_consumption(self):
        from crane_project.utils import port_reliability_redc_size_v1_torch as old_native
        previous=old_native.make_models(self.n,'cpu')['redc']
        for k,v in previous.state_dict().items():self.assertTrue(torch.equal(v,self.models['temperature'].state_dict()[k]))

    def test_distinct_parameters_and_optimizers(self):
        ids=[{id(p) for p in m.parameters()} for m in self.models.values()]
        self.assertFalse(ids[0]&ids[1])

    def test_numpy_torch_both_readouts(self):
        for arm,model in self.models.items():
            with torch.no_grad():model.network[4].weight.fill_(.13);model.beta.fill_(.2)
            ref=core.numpy_logits(native.exported(model),self.raw,self.n,arm)
            np.testing.assert_allclose(model(self.x,self.z).detach().numpy(),ref,rtol=1e-5,atol=1e-5)

    def test_feature_gradient_both_after_neutral_step(self):
        y=torch.tensor([0,1]*6,dtype=torch.float32);w=torch.ones(12)
        for arm,model in self.models.items():
            opt=native.optimizer(model)
            native.update(model,opt,self.x,self.z,y,w)
            v=native.update(model,opt,self.x,self.z,y,w)
            self.assertGreater(v['gradient_by_parameter']['network.0.weight'],0.,arm)

    def test_residual_readout_gradient_finite_difference(self):
        model=self.models['residual'].double();x=self.x.double();z=self.z.double()
        with torch.no_grad():model.network[4].weight.fill_(.13)
        loss=model(x,z).sum();loss.backward()
        p=model.network[4].weight;analytic=float(p.grad[0,0]);original=float(p[0,0])
        with torch.no_grad():
            p[0,0]=original+1e-6;plus=float(model(x,z).sum())
            p[0,0]=original-1e-6;minus=float(model(x,z).sum())
            p[0,0]=original
        self.assertAlmostEqual(analytic,(plus-minus)/2e-6,places=7)

    def test_inputs_detached_clip_and_beta_no_decay(self):
        for model in self.models.values():
            opt=native.optimizer(model);x=self.x.clone().requires_grad_();z=self.z.clone().requires_grad_()
            exempt={id(p) for g in opt.param_groups if g['weight_decay']==0 for p in g['params']}
            self.assertIn(id(model.beta),exempt)
            v=native.update(model,opt,x,z,torch.ones(12),torch.full((12,),1e5))
            self.assertIsNone(x.grad);self.assertIsNone(z.grad)
            self.assertGreater(v['gradient_norm_before'],5.)
            self.assertLessEqual(v['gradient_norm_after'],5.0001)


if __name__=='__main__':unittest.main()
