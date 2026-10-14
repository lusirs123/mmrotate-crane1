import unittest
from copy import deepcopy
import math
import numpy as np
from crane_project.utils import port_reliability_fr_constraint_v1 as c
from test_port_reliability_video_transfer_v1 import row

class TestConstrained(unittest.TestCase):
    def material(self):
        return [row(i, seq='v%d' % (i%2), error=i%4 == 0) for i in range(24)]

    def test_objective_and_constraint_analytic_derivatives(self):
        _, _, _, e, groups = c.prepare(self.material()); w=np.array([-4., .1, -.03, .01]); step=1e-5
        loss, grad, hess=c.barrier(w, e, groups, .01)
        ng=[]; nh=[]
        for i in range(4):
            d=np.eye(4)[i]*step
            ng.append((c.barrier(w+d,e,groups,.01)[0]-c.barrier(w-d,e,groups,.01)[0])/(2*step))
            nh.append((c.barrier(w+d,e,groups,.01)[1]-c.barrier(w-d,e,groups,.01)[1])/(2*step))
        np.testing.assert_allclose(ng,grad,atol=1e-8,rtol=1e-7)
        np.testing.assert_allclose(np.array(nh).T,hess,atol=1e-8,rtol=1e-7)
        self.assertGreater(np.linalg.eigvalsh(hess).min(),0)

    def test_loss_signs_and_unpenalized_intercept(self):
        e=np.array([[1.,0.,0.,0.]])
        self.assertLess(c.objective(np.array([1.,0.,0.,0.]),e)[0],c.objective(np.array([-1.,0.,0.,0.]),e)[0])
        groups={'v':e}
        self.assertGreater(c.constraints(np.array([1.,0.,0.,0.]),groups)[0][0],c.constraints(np.array([-1.,0.,0.,0.]),groups)[0][0])
        self.assertAlmostEqual(c.objective(np.array([1.,0.,0.,0.]),e)[0],float(np.logaddexp(0,-1)))

    def test_certified_solution_and_empirical_FR_bound(self):
        data=self.material(); m=c.fit_constrained(data)
        self.assertTrue(m['converged']); self.assertFalse(m['certificate']['guarantee_on_unseen_video'])
        for seq,q in m['certificate']['surrogate_by_video'].items():
            good=[r for r in data if r['sequence']==seq and not c.video.wrong(r)]
            fr=sum(c.logit(m,r)>=0 for r in good)/len(good)
            self.assertLessEqual(fr,q+1e-12); self.assertLessEqual(q,.05)

    def test_degenerate_uninformative_features_accept_all(self):
        rows=self.material()
        for r in rows: r['pred']=[0.,0.,100.,40.,0.,.8]; r['gt'][2]=80. if r['frame_id']%4==0 else 100.
        m=c.fit_constrained(rows)
        self.assertTrue(all(c.logit(m,r)<0 for r in rows))
        self.assertAlmostEqual(m['weights'][0],math.log(math.expm1(.05*math.log(2))),places=4)

    def test_held_GT_cannot_change_models_or_thresholds(self):
        fit=self.material(); held=[row(100,'held')]
        a=c.fit_score(fit,held); held[0]['gt'][2]=200; b=c.fit_score(fit,held)
        self.assertEqual(a[0],b[0]); self.assertEqual(a[1],b[1])

    def test_fit_normalizer_and_inputs_unchanged(self):
        rows=self.material(); saved=deepcopy(rows); m,_,_,_=c.fit_score(rows,[])
        x=np.array([c.video.fusion.simple.descriptor(r['pred'],r['image_size']) for r in rows])
        np.testing.assert_allclose(m['constrained3']['mean'],x.mean(0),atol=1e-12)
        self.assertEqual(rows,saved)

    def test_zero_boundary_reject_and_missing_separate(self):
        r=row(0); r['risks']['constrained3']=0; missing=row(1); missing['pred']=None
        t={'constrained3':{'logit_lt':0}}
        self.assertEqual(c.accepted([r,missing],'constrained3',t),set())
        self.assertEqual(c.video.counts([r,missing],set())['states']['MISSING'],1)

    def test_both_classes_and_each_correct_video_required(self):
        with self.assertRaises(ValueError): c.fit_constrained([row(i) for i in range(3)])
        with self.assertRaises(ValueError): c.fit_constrained([row(0,'a'),row(1,'b',error=True)])

    def test_infeasible_certificate_fails(self):
        _,_,_,e,g=c.prepare(self.material())
        with self.assertRaises(ValueError): c.certificate(np.zeros(4),e,g,1e-7)

    def test_independent_review_catches_multiplier_corruption(self):
        from crane_project.tools import review_port_reliability_fr_constraint_v1 as review
        rows=self.material(); models,_,_,_=c.fit_score(rows,[])
        self.assertEqual(review.check_models(models,rows),2)
        broken=deepcopy(models)
        for v in broken['constrained3']['certificate']['dual_multipliers']:
            broken['constrained3']['certificate']['dual_multipliers'][v]=0.
        with self.assertRaises(ValueError): review.check_models(broken,rows)

    def test_independent_review_catches_wrong_fit_normalizer(self):
        from crane_project.tools import review_port_reliability_fr_constraint_v1 as review
        rows=self.material(); models,_,_,_=c.fit_score(rows,[])
        models['constrained3']['mean'][0]+=1e-3
        with self.assertRaises(ValueError): review.check_models(models,rows)

    def test_gradient_reaches_all_features(self):
        rows=self.material(); _,_,_,e,g=c.prepare(rows)
        _,grad,_=c.objective(np.zeros(4),e)
        self.assertTrue(np.isfinite(grad).all()); self.assertGreater(np.linalg.norm(grad[1:]),0)

if __name__=='__main__': unittest.main()
