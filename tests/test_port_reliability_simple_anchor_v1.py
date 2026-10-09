import inspect
import math
import unittest
import numpy as np
from crane_project.utils import port_reliability_simple_anchor_v1 as core
from crane_project.utils import port_reliability_complementarity_v1 as metrics


def row(index, bad=False, risk=.2, present=True, sequence='real_seq01', split='val'):
    pred=[0,0,12 if bad else 10,2,0,.8] if present else None
    decision=dict(center_accepted=present,size_accepted=present,angle_accepted=present,
        risks=dict(size=risk if present else None,angle=.1 if present else None),final_box_original=pred)
    return dict(image=sequence+'_%05d'%index,sequence=sequence,frame_id=index,domain=sequence.split('_')[0],
        split=split,pred=pred,gt=[0,0,10,2,0],size_risks=dict(full_simple=risk if present else None),
        original_simple_decision=decision,risks={m:risk if present else None for m in core.METHODS})


def neutral():
    model=dict(beta=[0.])
    for n,shape in ((0,(16,258)),(2,(8,16)),(4,(1,8))):
        model['network.%d.weight'%n]=np.zeros(shape).tolist()
        model['network.%d.bias'%n]=np.zeros(shape[0]).tolist()
    return model


class AnchorTests(unittest.TestCase):
    def test_neutral_exact(self):
        r=np.array([.01582384,.17637164,.41665113433410333,.77898824])
        np.testing.assert_array_equal(core.readout(r,np.zeros(len(r))),r)

    def test_equivalent_logit(self):
        r=np.linspace(.001,.999,200); d=np.linspace(-.5,.5,200)
        expected=1/(1+np.exp(-(np.log(r)-np.log1p(-r)+d)))
        np.testing.assert_allclose(core.readout(r,d),expected,rtol=1e-14,atol=1e-15)

    def test_bound(self):
        r=np.array([.01,.3,.99]); d=np.array([-.5,0,.5]); p=core.readout(r,d)
        np.testing.assert_allclose(np.log(p/(1-p))-np.log(r/(1-r)),d,atol=3e-14)

    def test_bad_anchors(self):
        for r in ([0.],[1.],[float('nan')],[-.1]):
            with self.assertRaises(ValueError): core.readout(r,[0.])

    def test_bad_delta(self):
        for d in ([.501],[float('inf')],[]):
            with self.assertRaises(ValueError): core.readout([.2],d)

    def test_positive_negative_correction(self):
        r=core.readout([.2,.2],[-.1,.1]); self.assertLess(r[0],.2); self.assertGreater(r[1],.2)

    def test_online_signature(self):
        for function in (core.readout,core.numpy_forward,core.decide):
            self.assertFalse({'gt','domain','sequence'} & set(inspect.signature(function).parameters))

    def test_same_capacity_numpy(self):
        model=neutral(); self.assertEqual(sum(np.asarray(v).size for v in model.values()),4290)
        x=np.zeros((2,259)); normalizer=dict(mean=np.zeros(259),scale=np.ones(259))
        r,d=core.numpy_forward(model,x,normalizer,[.2,.4])
        np.testing.assert_array_equal(r,[.2,.4]); np.testing.assert_array_equal(d,[0.,0.])

    def test_unchanged_center_angle_box(self):
        original=row(1)['original_simple_decision']; changed=core.decide(original,.8,.4)
        self.assertFalse(changed['size_accepted'])
        self.assertTrue(original['size_accepted'])
        for k in ('center_accepted','angle_accepted','final_box_original'):
            self.assertEqual(original[k],changed[k])

    def test_missing(self):
        r=row(1,present=False); self.assertEqual(core.decide(r['original_simple_decision'],None,.4),r['original_simple_decision'])
        self.assertIsNone(metrics.bad(r))
        with self.assertRaises(ValueError): core.decide(r['original_simple_decision'],.2,.4)

    def test_calibration_no_test(self):
        with self.assertRaises(ValueError): core.calibrate([row(1,split='test')],'full_simple')

    def test_global_group_requirement(self):
        values=[row(i,risk=.1) for i in range(20)]+[row(1,risk=.8,sequence='sim_seq02')]
        value=core.calibrate(values,'full_simple'); self.assertEqual(value['risk_le'],.8)
        self.assertTrue(value['uses_VAL_GT']); self.assertTrue(value['calibration_not_independent_validation'])

    def test_whole_ties(self):
        rows=[row(1),row(2,bad=True)]
        cutoff=core.calibrate(rows,'full_simple')['risk_le']
        self.assertEqual(len(metrics.accepted(rows,'full_simple',cutoff)),2)

    def test_same_count_tie_bounds(self):
        values=[row(1),row(2,bad=True)]
        candidate=dict(accepted_outputs=1,states=dict(CR=1))
        m=metrics.matched(values,'full_simple',candidate)
        self.assertEqual(m['same_count_tie_bounds']['bad_min'],0)
        self.assertEqual(m['same_count_tie_bounds']['bad_max'],1)
        self.assertTrue(m['same_count_tie_bounds']['partial_boundary_tie'])

    def test_correct_fr_not_error_rejections(self):
        from crane_project.utils import port_reliability_state_continuity_v1 as states
        values=[row(i,bad=True) for i in range(1,6)]+[row(6)]
        v=states.summarize(values,set()); self.assertEqual(v['runs']['correct_rejection']['longest'],1)
        self.assertEqual(v['states']['ED'],5)

    def test_class_balance(self):
        y=np.array([0]*100+[1]*3); w=core.class_weights(y)
        self.assertAlmostEqual(w[y==0].sum(),len(y)/2)
        self.assertAlmostEqual(w[y==1].sum(),len(y)/2)

    def test_equal_scores_cannot_pass(self):
        rows=[row(i) for i in range(1,21)]+[row(21,bad=True)]
        cuts={m:core.calibrate(rows,m) for m in core.METHODS}
        gate=core.gate(core.statistics(rows,cuts))
        self.assertFalse(gate['passed']); self.assertIsNone(gate['selected_arm'])


if __name__=='__main__': unittest.main()
