import unittest
from copy import deepcopy
from crane_project.tools import analyze_port_reliability_frozen_workpoints_v1 as a

class TestFrozenWorkpoints(unittest.TestCase):
    def example(self):
        m=dict(mean=[0.,0.,0.],scale=[1.,1.,1.],weights=[0.,0.,0.,0.])
        frozen=dict(control_model=deepcopy(m),candidate_model=deepcopy(m),
            formal_risk_le=.5,protected_risk_le=.7,candidate_logit_lt=0.,score_risk_le=.6)
        r=dict(pred=[0.,0.,100.,40.,0.,.8],gt=[0.,0.,100.,40.,0.],image_size=[1000,600],
            risks=dict(full_simple=.5,score_only=.2),original_simple_decision=dict(size_accepted=True))
        return r,frozen
    def test_zero_is_a_threshold_and_tie_rejected(self):
        r,f=self.example(); z=a.apply([r],f)[0]
        self.assertFalse(z['supplement_flags']['constrained_zero'])
        self.assertTrue(z['supplement_flags']['protected_simple'])
    def test_negative_risk_accepts(self):
        r,f=self.example(); f['candidate_model']['weights'][0]=-1
        self.assertTrue(a.apply([r],f)[0]['supplement_flags']['constrained_zero'])
    def test_GT_does_not_enter_flags(self):
        r,f=self.example(); p=a.apply([r],f); r['gt'][2]=200.
        self.assertEqual(p[0]['supplement_flags'],a.apply([r],f)[0]['supplement_flags'])
    def test_no_prediction_is_not_a_reliability_rejection(self):
        r,f=self.example(); r['pred']=None
        self.assertFalse(any(a.apply([r],f)[0]['supplement_flags'].values()))
    def test_inputs_are_not_mutated(self):
        r,f=self.example(); saved=deepcopy((r,f));a.apply([r],f)
        self.assertEqual((r,f),saved)
    def test_native_sim_train_identity_preserved(self):
        r={'split':'train_sim','reliability_role':'train','domain':'sim'}
        a.validate_role('TRAIN',[r],1)
        with self.assertRaises(ValueError):a.validate_role('VAL',[r],1)
    def test_external_role_mismatch_fails(self):
        r={'split':'train','reliability_role':'val','domain':'real'}
        with self.assertRaises(ValueError):a.validate_role('TRAIN',[r],1)
    def test_inconsistent_formal_scoring_fails(self):
        r,f=self.example(); r['risks']['full_simple']=.6
        with self.assertRaises(ValueError):a.apply([r],f)
if __name__=='__main__':unittest.main()
