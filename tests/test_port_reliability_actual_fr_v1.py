import unittest
from copy import deepcopy
import importlib.util
import numpy as np
from crane_project.utils import port_reliability_actual_fr_v1 as c
from test_port_reliability_video_transfer_v1 import row

class TestActualFR(unittest.TestCase):
    def material(self):return [row(i,'a' if i<20 else 'b',error=i in (0,21)) for i in range(42)]
    def test_budget_uses_exact_floor_not_surrogate(self):
        data=c.build(self.material())
        self.assertEqual(data['limits']['a']['max_FR'],0)
        self.assertEqual(data['limits']['b']['max_FR'],1)
    def test_bigm_bounds_every_corner_of_weight_box(self):
        data=c.build(self.material());d=data['design'];m=data['big_m']
        from itertools import product
        for w in product([-10.,10.],repeat=4):self.assertTrue(np.all(np.abs(d@np.array(w))+1e-4<=m+1e-12))
    def test_all_accept_is_feasible_but_detects_no_error(self):
        data=c.build(self.material());cert=c.validate_solution(data,[-1e-4,0,0,0],np.zeros(42))
        self.assertEqual(cert['FA'],2);self.assertEqual(cert['ED'],0);self.assertEqual(cert['FR'],0)
    def test_zero_and_wrong_binary_assignments_rejected(self):
        data=c.build(self.material())
        with self.assertRaises(ValueError):c.validate_solution(data,[0,0,0,0],np.zeros(42))
        with self.assertRaises(ValueError):c.validate_solution(data,[-1e-4,0,0,0],np.ones(42))
    def test_fractional_binary_and_coefficient_overflow_fail(self):
        data=c.build(self.material())
        with self.assertRaises(ValueError):c.validate_solution(data,[-1,0,0,0],np.full(42,.1))
        with self.assertRaises(ValueError):c.validate_solution(data,[-11,0,0,0],np.zeros(42))
    def test_actual_per_video_fr_budget_enforced(self):
        data=c.build(self.material())
        with self.assertRaises(ValueError):c.validate_solution(data,[1,0,0,0],np.ones(42))
    def test_fit_normalizer_and_membership(self):
        data=c.build(self.material());x=np.array([c.video.fusion.simple.descriptor(r['pred'],r['image_size']) for r in data['rows']])
        np.testing.assert_allclose(data['mean'],x.mean(0),atol=1e-12)
    def test_single_class_or_video_without_correct_support_fail(self):
        with self.assertRaises(ValueError):c.build([row(i) for i in range(4)])
        with self.assertRaises(ValueError):c.build([row(0,'a'),row(1,'b',error=True)])
    def test_missing_separate_and_zero_rejected(self):
        r=row(0);r['risks']['actual_count3']=0.;missing=row(1);missing['pred']=None
        keep=c.accepted([r,missing],'actual_count3',{'actual_count3':{'logit_lt':0.}})
        self.assertEqual(keep,set());self.assertEqual(c.video.counts([r,missing],keep)['states']['MISSING'],1)
    def test_held_GT_not_used_by_scoring(self):
        model=dict(weights=[-1.,0,0,0],mean=[0.,0,0],scale=[1.,1,1]);models=dict(control3=model,actual_count3=model)
        r=row(0);saved=deepcopy(r);a=c.score([r],models);r['gt'][2]=200;b=c.score([r],models)
        self.assertEqual(a[0]['risks'],b[0]['risks']);self.assertEqual(a[0]['gt'],saved['gt'])
    def test_independent_review_detects_corrupted_actual_counts(self):
        from crane_project.tools import review_port_reliability_actual_fr_v1 as review
        data=c.build(self.material());n=len(data['rows'])
        control=c.video.fusion.simple.fit_linear_risk(
            np.array([c.video.fusion.simple.descriptor(r['pred'],r['image_size']) for r in data['rows']]),
            data['labels'].astype(float),dict(l2=.1,max_iterations=100,gradient_tolerance=1e-8))
        w=np.array([-1e-4,0.,0.,0.]);certificate=c.validate_solution(data,w,np.zeros(n))
        m=dict(mean=data['mean'].tolist(),scale=data['scale'].tolist(),weights=w.tolist(),
            train_rows=n,bad_rows=2,good_rows=n-2,fit_rejections={r['image']:False for r in data['rows']},
            fitting_videos=list(data['limits']),certificate=certificate,
            NP_statistical_guarantee=False,unrestricted_capacity_claim=False,
            solver=dict(raw_dual_bound=None,FA_upper=2,FA_lower=0,absolute_FA_gap=2,
                optimal_within_fixed_domain=False,incumbent=False,status=1,
                outcome='NO_SOLVER_INCUMBENT_ALL_ACCEPT_REFERENCE',
                options=dict(time_limit=60.,node_limit=100000,mip_rel_gap=0.,presolve=True)))
        models=dict(control3=control,actual_count3=m)
        self.assertEqual(review.check_models(models,data['rows'],dict(models=dict(control3=control))),2)
        m['certificate']['FA']=1
        with self.assertRaises(ValueError):review.check_models(models,data['rows'],dict(models=dict(control3=control)))

    @unittest.skipUnless(importlib.util.find_spec('scipy') is not None,'Local numeric runtime has no SciPy; solver integration is mandatory on server')
    def test_solver_known_separable_and_identical_features(self):
        data=self.material();m=c.solve(data)
        self.assertTrue(m['solver']['incumbent']);self.assertTrue(m['solver']['optimal_within_fixed_domain']);self.assertEqual(m['certificate']['FA'],0)
        for r in data:
            r['pred']=[0.,0.,100.,40.,0.,.8];r['gt'][2]=80. if r['frame_id'] in (0,21) else 100.
        m=c.solve(data)
        self.assertTrue(m['solver']['optimal_within_fixed_domain']);self.assertEqual(m['certificate']['FA'],2);self.assertEqual(m['certificate']['FR'],0)

if __name__=='__main__':unittest.main()
