import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from crane_project.utils import port_reliability_feature_source_test_v1 as core
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.tools import eval_port_reliability_feature_source_test_v1 as runner


def bundle():
    head={'network.0.weight':np.zeros((16,258)).tolist(),'network.0.bias':[0.]*16,
        'network.2.weight':np.zeros((8,16)).tolist(),'network.2.bias':[0.]*8,
        'network.4.weight':np.zeros((1,8)).tolist(),'network.4.bias':[0.],'beta':[0.]}
    model=dict(protocol=core.scoring.VERSION,epoch=100,update_counts=dict(midpoint=1000,native=1000),
        models={a:deepcopy(head) for a in core.scoring.ARMS},normalizers={a:dict(mean=[0.]*259,scale=[1.]*259) for a in core.scoring.ARMS})
    cutoffs={m:dict(single_global_cutoff=True,risk_le=.5) for m in ('midpoint','native','full_simple','score_only')}
    report=dict(protocol=core.scoring.VERSION,fixed_final_epoch=100,update_counts=model['update_counts'],TEST_read=False,
        original_policy_changed=False,boxes_scores_output_center_angle_unchanged=True,status='VAL_FAILED_STOP',gate=dict(passed=False),VAL_cutoffs=deepcopy(cutoffs))
    return model,cutoffs,report


def row(i,bad=False,missing=False):
    pred=None if missing else [0.,0.,12. if bad else 10.,4.,0.,.8]
    d=dict(final_box_original=pred,center_accepted=not missing,size_accepted=not missing,angle_accepted=not missing,
        risks=dict(size=None if missing else .2,angle=None if missing else .3))
    return dict(image='real_seq03_%05d'%i,sequence='real_seq03',frame_id=i,domain='real',split='test',
        pred=pred,gt=[0.,0.,10.,4.,0.],image_size=[100,80],original_simple_decision=d)

class ContractTests(unittest.TestCase):
    def test_frozen_bundle_copies_no_promotion(self):
        a,b,c=bundle();before=deepcopy((a,b,c));x,y=core.frozen_bundle(a,b,c)
        x['epoch']=1;y['native']['risk_le']=0.;self.assertEqual((a,b,c),before)

    def test_epoch_or_budget_changed_rejected(self):
        for key,value in [('epoch',99),('update_counts',dict(midpoint=1000,native=999))]:
            a,b,c=bundle();a[key]=value
            with self.assertRaises(ValueError):core.frozen_bundle(a,b,c)

    def test_val_cutoff_changed_rejected(self):
        a,b,c=bundle();b['native']['risk_le']=.9
        with self.assertRaises(ValueError):core.frozen_bundle(a,b,c)

    def test_history_cannot_be_promoted(self):
        for key,value in [('TEST_read',True),('status','VAL_PASSED'),('original_policy_changed',True)]:
            a,b,c=bundle();c[key]=value
            with self.assertRaises(ValueError):core.frozen_bundle(a,b,c)

    def test_no_fit_or_gate_only_size_changes(self):
        a,b,_=bundle();rows=[row(1),row(2,True),row(3,missing=True)]
        matrices={arm:np.zeros((2,259)) for arm in core.scoring.ARMS}
        for x in matrices.values():x[:,0]=[2.,-2.]
        before=deepcopy((a,b,rows))
        with patch.object(core.scoring,'fit_pca',side_effect=AssertionError('TEST fit forbidden')),patch.object(core.scoring,'gate',side_effect=AssertionError('TEST gate forbidden')):
            risks,decisions=core.score_rows(rows,matrices,[r['image'] for r in rows[:2]],a,b)
            report=core.summarize(rows,risks,b)
        self.assertEqual((a,b,rows),before);self.assertFalse(report['automatic_promotion'])
        for r,d in zip(rows,decisions):
            for method,v in d['methods'].items():
                self.assertEqual(v['final_box_original'],r['pred']);self.assertEqual(v['center_accepted'],r['original_simple_decision']['center_accepted'])
                self.assertEqual(v['angle_accepted'],r['original_simple_decision']['angle_accepted'])
        self.assertTrue(all(v[rows[-1]['image']] is None for v in risks.values()))

    def test_bad_feature_membership_or_shape_rejected(self):
        a,b,_=bundle();rows=[row(1)];ids=[rows[0]['image']]
        with self.assertRaises(ValueError):core.score_rows(rows,{m:np.zeros((1,291)) for m in core.scoring.ARMS},ids,a,b)
        with self.assertRaises(ValueError):core.score_rows(rows,{m:np.zeros((1,259)) for m in core.scoring.ARMS},['unknown'],a,b)

    def test_missing_not_FR_and_center_three_denominators(self):
        rows=[row(1),row(2,True),row(3,missing=True)];r=core.test_summary(rows,{rows[0]['image']})
        self.assertEqual(r['states'],dict(FA=0,FR=0,ED=1,CR=1,MISSING=1))
        self.assertEqual(r['output_coverage'],2/3);self.assertEqual(r['good_retention'],1.)

    def test_sources_contract_and_exact_artifact_pins(self):
        runner.checked_sources();model,cutoffs,pcas,policy,proof=runner.checked_inputs(False)
        self.assertFalse(proof['TEST_evaluated']);self.assertEqual(set(pcas),set(core.scoring.ARMS))
        self.assertEqual(cutoffs['native']['risk_le'],.5361259910005223)

if __name__=='__main__':unittest.main()
