import unittest
from copy import deepcopy
from unittest.mock import patch
from crane_project.tools import eval_port_reliability_actual_fr_val_v1 as e


class TestActualFRVAL(unittest.TestCase):
    def fixture(self):
        model = dict(mean=[0.,0.,0.], scale=[1.,1.,1.], weights=[0.,0.,0.,0.])
        frozen = dict(control_model=deepcopy(model), candidate_model=deepcopy(model),
                      formal_risk_le=.5, protected_risk_le=.7, candidate_logit_lt=0., score_risk_le=.6)
        row = dict(image='real_seq01_00001', sequence='real_seq01', frame_id=1,
                   domain='real', split='val', reliability_role='val',
                   pred=[0.,0.,100.,40.,0.,.8], gt=[0.,0.,100.,40.,0.],
                   image_size=[1000,600], temporal={'acquisition_boundary':False},
                   risks=dict(full_simple=.5, score_only=.2),
                   original_simple_decision=dict(size_accepted=True))
        return row, frozen

    def test_zero_tie_rejected_and_negative_accepted(self):
        r, f = self.fixture()
        self.assertFalse(e.apply([r],f)[0]['supplement_flags']['actual_count_zero'])
        f['candidate_model']['weights'][0] = -1.
        self.assertTrue(e.apply([r],f)[0]['supplement_flags']['actual_count_zero'])

    def test_GT_and_sequence_do_not_enter_decisions(self):
        r, f = self.fixture(); original=e.apply([r],f)[0]
        r['gt'][2]=200.; r['sequence']='unseen'
        revised=e.apply([r],f)[0]
        self.assertEqual(original['supplement_flags'],revised['supplement_flags'])
        self.assertEqual(original['risks'],revised['risks'])

    def test_missing_is_separate_from_FR_ED(self):
        r,f=self.fixture();r['pred']=None
        output=e.apply([r],f)[0]
        self.assertFalse(any(output['supplement_flags'].values()))
        self.assertEqual(e.metrics.counts([output],set())['states'],dict(FA=0,FR=0,ED=0,CR=0,MISSING=1))

    def test_input_and_frozen_model_preserved(self):
        r,f=self.fixture();before=deepcopy((r,f));e.apply([r],f)
        self.assertEqual((r,f),before)

    def test_formal_replay_corruption_fails(self):
        r,f=self.fixture();r['risks']['full_simple']=.6
        with self.assertRaises(ValueError):e.apply([r],f)

    def test_curves_preserve_whole_ties_and_missing(self):
        r,f=self.fixture();bad=deepcopy(r);bad['image']='bad';bad['frame_id']=2;bad['pred'][2]=130.
        missing=deepcopy(r);missing['image']='missing';missing['frame_id']=3;missing['pred']=None
        points=e.curve([r,bad,missing],'full_simple')
        self.assertEqual(len(points),2)
        self.assertEqual(points[-1]['states'],dict(FA=1,FR=0,ED=0,CR=1,MISSING=1))
        self.assertEqual(points[-1]['acceptance_coverage'],2/3)

    def test_positive_wrong_risk_AUC_direction(self):
        r,f=self.fixture();bad=deepcopy(r);bad['image']='bad';bad['frame_id']=2;bad['pred'][2]=130.;bad['risks']['full_simple']=.9
        self.assertEqual(e.metrics.fusion.level.metrics.auc([r,bad],'full_simple'),1.)

    def test_VAL_evaluation_never_calls_fit_solver(self):
        r,f=self.fixture()
        with patch.object(e.scoring,'fit',side_effect=AssertionError('No fitting'),create=True):
            report,curves=e.evaluate([r],f)
        self.assertEqual(report['frames'],1)
        self.assertIn('all',curves)


if __name__=='__main__':unittest.main()
