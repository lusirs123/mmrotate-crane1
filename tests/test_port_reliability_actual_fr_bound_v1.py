from copy import deepcopy
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from crane_project.utils import port_reliability_actual_fr_bound_v1 as c
from crane_project.tools import review_port_reliability_actual_fr_bound_v1 as review
from test_port_reliability_video_transfer_v1 import row


class TestBoundedActualFR(unittest.TestCase):
    def material(self):
        return [row(i,'a' if i<20 else 'b',error=i in (0,21)) for i in range(42)]

    def test_upper_requires_verified_feasible_witness(self):
        data=c.prior.build(self.material())
        self.assertEqual(c.witness_certificate(data,[-1e-4,0,0,0],2)['FA'],2)
        with self.assertRaises(ValueError):c.witness_certificate(data,[-1e-4,0,0,0],1)
        with self.assertRaises(ValueError):c.witness_certificate(data,[0,0,0,0],2)

    def test_witness_validates_FR_not_only_FA(self):
        with self.assertRaises(ValueError):c.witness_certificate(c.prior.build(self.material()),[1.,0,0,0],2)

    def test_no_new_budget_or_feature_expansion(self):
        for key in c.prior.SETTINGS:self.assertEqual(c.SETTINGS[key],c.prior.SETTINGS[key])
        self.assertEqual(c.SETTINGS['solver_calls'],1)
        self.assertFalse(c.SETTINGS['known_solution_warm_start'])

    def test_independent_certificate_rejects_corruption(self):
        rows=self.material();data=c.prior.build(rows);w=[-1e-4,0,0,0]
        cert=c.prior.validate_solution(data,w,np.zeros(len(rows)))
        model=dict(weights=w,mean=data['mean'].tolist(),scale=data['scale'].tolist(),
            certificate=cert,fit_rejections={r['image']:False for r in rows},train_rows=42,bad_rows=2,good_rows=40)
        self.assertEqual(review.check_model(model,rows),42)
        model['certificate']['FA']=1
        with self.assertRaises(ValueError):review.check_model(model,rows)

    @unittest.skipUnless(importlib.util.find_spec('scipy') is not None,'SciPy solver integration runs on server')
    def test_new_cut_small_separable_and_identical_features(self):
        rows=self.material();before=deepcopy(rows)
        answer=c.solve(rows,[-1e-4,0,0,0],upper=2)
        self.assertEqual(answer['model']['certificate']['FA'],0)
        self.assertTrue(answer['solver']['optimal_within_fixed_domain'])
        self.assertEqual(rows,before)
        for r in rows:
            r['pred']=[0.,0.,100.,40.,0.,.8]
            r['gt'][2]=80. if r['frame_id'] in (0,21) else 100.
        answer=c.solve(rows,[-1e-4,0,0,0],upper=2)
        self.assertEqual(answer['model']['certificate']['FA'],2)
        self.assertTrue(answer['solver']['optimal_within_fixed_domain'])

    @unittest.skipUnless(importlib.util.find_spec('scipy') is not None,'SciPy solver integration runs on server')
    def test_no_incumbent_is_not_control_or_all_accept_candidate(self):
        def fake(**kw):
            self.assertEqual(len(kw['constraints']),3)
            cut=kw['constraints'][2]
            np.testing.assert_equal(cut.A.toarray()[0,4:],c.prior.build(self.material())['labels'].astype(float))
            self.assertEqual(float(cut.lb[0]),0.)
            kw['options'].pop('node_limit')
            return SimpleNamespace(x=None,status=1,message='mock timeout',mip_dual_bound=-2.,mip_node_count=0)
        with patch('scipy.optimize.milp',side_effect=fake):answer=c.solve(self.material(),[-1e-4,0,0,0],upper=2)
        self.assertIsNone(answer['model'])
        self.assertEqual(answer['solver']['outcome'],'NO_SOLVER_CANDIDATE')
        self.assertEqual(answer['solver']['options']['node_limit'],100000)
        self.assertFalse(answer['known_control_is_candidate'])

    @unittest.skipUnless(importlib.util.find_spec('scipy') is not None,'SciPy solver integration runs on server')
    def test_false_infeasibility_contradicts_witness(self):
        fake=SimpleNamespace(x=None,status=2,message='mock infeasible',mip_dual_bound=None)
        with patch('scipy.optimize.milp',return_value=fake):
            with self.assertRaises(ValueError):c.solve(self.material(),[-1e-4,0,0,0],upper=2)


if __name__=='__main__':unittest.main()
