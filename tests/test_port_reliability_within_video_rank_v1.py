import inspect
import math
import unittest
import numpy as np
from crane_project.utils import port_reliability_within_video_rank_v1 as core
from crane_project.utils import port_reliability_complementarity_v1 as metrics


def row(index, bad=False, risk=.2, present=True, sequence='real_seq01', split='val'):
    pred=[0,0,12 if bad else 10,2,0,.8] if present else None
    decision=dict(center_accepted=present,size_accepted=present,angle_accepted=present,
        risks=dict(size=risk if present else None,angle=.1 if present else None),final_box_original=pred)
    return dict(image=sequence+'_%05d'%index,sequence=sequence,frame_id=index,domain=sequence.split('_')[0],
        split=split,reliability_role='train' if split in ('train','train_sim') else split,
        pred=pred,gt=[0,0,10,2,0],size_risks=dict(full_simple=risk if present else None),
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

    def test_both_arms_same_simple_anchor(self):
        r=row(1,risk=.2)
        for arm in core.ARMS:
            np.testing.assert_array_equal(core.anchors([r],arm),[.2])

    def test_pairs_genuine_same_video_complete(self):
        rows=[row(1,split='train'),row(2,bad=True,split='train'),row(3,split='train'),
              row(1,bad=True,sequence='sim_seq08',split='train_sim'),
              row(2,sequence='sim_seq08',split='train_sim')]
        rows.sort(key=lambda r:r['image'])
        plan=core.build_pairs(rows)
        self.assertEqual(core.pair_counts(plan),{'real_seq01':dict(bad=1,good=2,pairs=2),
                                                'sim_seq08':dict(bad=1,good=1,pairs=1)})
        for name,v in plan.items():
            for b in v['bad']:
                for g in v['good']:
                    self.assertTrue(metrics.bad(rows[b])); self.assertFalse(metrics.bad(rows[g]))
                    self.assertEqual(rows[b]['sequence'],rows[g]['sequence'])

    def test_pairs_reject_val_test_missing_duplicate_single_class(self):
        for rows in ([row(1),row(2,bad=True)],
                     [row(1,split='test'),row(2,bad=True,split='test')],
                     [row(1,split='train',present=False)],
                     [row(1,split='train'),row(1,bad=True,split='train')],
                     [row(1,split='train')]):
            with self.assertRaises(ValueError): core.build_pairs(rows)

    def test_rank_macro_video_not_pair_count_weighting(self):
        plan={'a':dict(bad=[0],good=[1,2],pairs=2),'b':dict(bad=[3],good=[4],pairs=1)}
        loss,terms=core.numpy_rank_loss([2,0,0,-2,0],plan)
        self.assertAlmostEqual(loss,(terms['a']+terms['b'])/2)
        self.assertNotAlmostEqual(loss,(2*terms['a']+terms['b'])/3)

    def test_rank_shift_invariance_and_correct_gradient_direction(self):
        plan={'a':dict(bad=[0],good=[1],pairs=1)}; s=np.array([-.4,.3])
        value=lambda a:core.numpy_rank_loss(a,plan)[0]
        self.assertAlmostEqual(value(s),value(s-1000),places=12)
        eps=1e-6; grad=np.array([(value(s+eps*np.eye(2)[i])-value(s-eps*np.eye(2)[i]))/(2*eps) for i in range(2)])
        self.assertLess(grad[0],0); self.assertGreater(grad[1],0)
        self.assertAlmostEqual(grad.sum(),0,places=9)
        self.assertLess(value(s-.1*grad),value(s))

    def test_ranking_summary_separates_video_and_pooled(self):
        rows=[row(1,risk=.1),row(2,bad=True,risk=.2),
              row(1,risk=.8,sequence='real_seq05'),row(2,bad=True,risk=.9,sequence='real_seq05')]
        result=core.ranking_summary(rows,'full_simple')
        self.assertEqual(result['within_video_macro_AUROC'],1.)
        self.assertEqual(result['within_video_pair_weighted_AUROC'],1.)
        self.assertEqual(result['pooled_AUROC'],.75)

    def test_FA_gain_without_video_AUROC_gain_is_not_adopted(self):
        rows=[row(i) for i in range(1,21)]+[row(21,bad=True)]
        cuts={m:core.calibrate(rows,m) for m in core.METHODS}
        statistics=core.statistics(rows,cuts)
        # Isolate the mechanism gate: even a same-CR FA gain cannot substitute
        # for actual improvement in the corresponding video's ordering.
        value=statistics['within_video_rank']['sequence:real_seq01']
        comp=value['controls']['full_simple']
        comp['same_CR_exact']=True
        comp['same_CR']['states']['FA']=value['actual']['states']['FA']+1
        gate=core.gate(statistics)
        self.assertIn('at_least_one_same_CR_FA_and_within_video_AUROC_gain',
                      [f['check'] for f in gate['failures']])

    def test_missing_excluded_from_ranking_support(self):
        rows=[row(1),row(2,bad=True),row(3,present=False)]
        result=core.ranking_summary(rows,'full_simple')
        self.assertEqual(result['genuine_pair_count'],1)
        self.assertEqual(result['eligible_videos'],1)

    def test_source_contract_pins_and_real_pair_counts(self):
        from crane_project.tools import run_port_reliability_within_video_rank_v1 as run
        contract,_=run.checked()
        self.assertEqual(contract['settings'],core.SETTINGS)
        self.assertEqual(sum(v['pairs'] for v in contract['pair_counts'].values()),32437)
        self.assertEqual(sum(v['bad'] for v in contract['pair_counts'].values()),71)
        self.assertEqual(contract['settings']['rank_weight'],.25)

    def test_independent_labels_match_complete_frozen_data(self):
        from crane_project.tools import run_port_reliability_within_video_rank_v1 as run
        from crane_project.tools import review_port_reliability_within_video_rank_v1 as review
        contract,_=run.checked(); parts,_=run.load_rows(contract)
        self.assertEqual(sum(map(len,parts.values())),3445)
        for rows in parts.values():
            for r in rows:
                self.assertIs(review.wrong(r),metrics.bad(r))
        plan=run.pair_plan(contract,parts['TRAIN'])
        risks=core.anchors(parts['TRAIN'],'bce_control')
        logits=np.log(risks)-np.log1p(-risks)
        value,terms=core.numpy_rank_loss(logits,plan)
        scalar={name:sum(float(np.logaddexp(0.,logits[g]-logits[b])) for b in v['bad'] for g in v['good'])/v['pairs']
                for name,v in plan.items()}
        self.assertAlmostEqual(value,sum(scalar.values())/len(scalar),places=12)
        for name in scalar: self.assertAlmostEqual(terms[name],scalar[name],places=12)

    def test_independent_review_counts_exclude_missing_and_error_runs(self):
        from crane_project.tools import review_port_reliability_within_video_rank_v1 as review
        from crane_project.utils import port_reliability_state_continuity_v1 as states
        rows=[row(1),row(2,bad=True),row(3,bad=True),row(4,present=False)]
        expected=states.summarize(rows,set())
        review.summary(rows,set(),expected)
        self.assertEqual(expected['states'],dict(FA=0,FR=1,ED=2,CR=0,MISSING=1))

    def test_independent_review_AUROC_matches_tied_pairs(self):
        from crane_project.tools import review_port_reliability_within_video_rank_v1 as review
        rows=[row(1,risk=.2),row(2,risk=.8),row(3,bad=True,risk=.2),row(4,bad=True,risk=.9)]
        self.assertEqual(review.pair_auc(rows,'full_simple'),metrics.auc(rows,'full_simple'))


if __name__=='__main__': unittest.main()
