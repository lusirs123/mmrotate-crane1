from copy import deepcopy
import inspect
import unittest
from unittest.mock import patch
import numpy as np
from crane_project.tools import run_port_reliability_opposite_border_v1_frozen_val as r
from crane_project.utils import port_reliability_opposite_border_v1 as m
from crane_project.tools import run_port_reliability_opposite_border_v1 as old


def row(i,bad=False,missing=False):
    p=None if missing else [100.,100.,100.,25. if bad else 20.,0.,.8]
    decision=dict(center_accepted=not missing,size_accepted=not missing,angle_accepted=not missing,
                  final_box_original=p,risks=dict(size=None if missing else .2,angle=None if missing else .3))
    return dict(image='real_seq01_'+str(i).zfill(3),sequence='real_seq01',domain='real',split='val',
        frame_id=i,pred=p,gt=[100.,100.,100.,20.,0.],image_size=[1000,500],reliability_role='val',
        sample_role='val',original_simple_decision=decision,size_risks=dict(full_simple=None if missing else .2),
        train_angle_eligible=True,angle_axis_well_defined=True,
        experiment_features={arm:None if missing else [0.]*len(schema) for arm,schema in m.SCHEMAS.items()})


def fixtures():
    models={arm:m.fit(np.zeros((6,len(schema))),[0,0,0,1,1,1],arm) for arm,schema in m.SCHEMAS.items()}
    points={arm:dict(risk_le=.5) for arm in (m.ARM,)+m.CONTROLS}
    return [row(0),row(1,bad=True),row(2,missing=True)],models,points


class FrozenVALTests(unittest.TestCase):
    def test_exact_failed_historical_source_is_explicitly_authorized(self):
        train=r.frozen_model_source()
        self.assertFalse(train['probe_gate']['passed'])
        self.assertEqual(train['conclusion'],'TRAIN_CAPABILITY_FAILED_STOP')
        self.assertEqual(train['models'],r.json.loads((r.TRAIN/'models.json').read_text()))

    def test_arbitrary_source_or_altered_bytes_rejected(self):
        with self.assertRaises(ValueError):r.frozen_model_source(r.TRAIN.parent/'unknown')
        with patch.object(old,'sha',return_value='changed'):
            with self.assertRaises(ValueError):r.frozen_model_source()

    def test_original_guard_remains_closed(self):
        with self.assertRaises(ValueError):old.frozen_train(r.TRAIN)

    def test_VAL_only_no_extra_roles_or_TEST(self):
        rows,models,points=fixtures()
        for role in ('train','test','probe'):
            x=deepcopy(rows);x[0]['reliability_role']=role
            with self.assertRaises(ValueError):r.evaluate(x,models,points)
        x=deepcopy(rows);x[0]['split']='test'
        with self.assertRaises(ValueError):r.evaluate(x,models,points)

    def test_evaluation_never_calls_fit_or_cutoff_solver(self):
        rows,models,points=fixtures();before=deepcopy((rows,models,points))
        with patch.object(m,'fit',side_effect=AssertionError('Forbidden fitting')),patch.object(m,'fit_cutoff',side_effect=AssertionError('Forbidden cutoff fitting')):
            report,scored=r.evaluate(rows,models,points)
        self.assertEqual((rows,models,points),before)
        self.assertEqual(report['diagnostics']['all']['actual']['states'],dict(FA=1,FR=0,ED=0,CR=1,MISSING=1))
        self.assertEqual(report['descriptive_checks']['gate_scope'],'descriptive_frozen_VAL_supplement_not_original_contract_pass')
        self.assertEqual(len(scored),3)

    def test_frozen_threshold_applies_without_recalibration(self):
        rows,models,points=fixtures();points[m.ARM]['risk_le']=.49
        report,scored=r.evaluate(rows,models,points)
        self.assertEqual(report['diagnostics']['all']['actual']['states'],dict(FA=0,FR=1,ED=1,CR=0,MISSING=1))
        self.assertEqual(points[m.ARM]['risk_le'],.49)
        self.assertEqual(report['formal_policy_summary']['all']['states'],dict(FA=1,FR=0,ED=0,CR=1,MISSING=1))
        self.assertEqual(report['diagnostics']['all']['actual']['runs']['correct_rejection']['longest'],1)
        self.assertEqual(report['diagnostics']['all']['actual']['runs']['error_detection']['longest'],1)

    def test_only_size_decision_changes_missing_has_no_quality(self):
        rows,models,points=fixtures();report,scored=r.evaluate(rows,models,points)
        for source,output in zip(rows,scored):
            a=deepcopy(source['original_simple_decision']);b=deepcopy(output['candidate_diagnostic_decision'])
            a.pop('size_accepted');b.pop('size_accepted');a['risks'].pop('size');b['risks'].pop('size')
            self.assertEqual(a,b)
        self.assertEqual(scored[2]['candidate_diagnostic_decision'],rows[2]['original_simple_decision'])

    def test_output_path_and_overwrite_guards(self):
        self.assertEqual(r.checked_out(r.OUT),r.OUT.resolve())
        with self.assertRaises(ValueError):r.checked_out(r.TRAIN.parent/'val_check')
        with patch.object(r.Path,'exists',return_value=True):
            with self.assertRaises(FileExistsError):r.checked_out(r.OUT)

    def test_sealed_source_closure_and_frozen_VAL_rows(self):
        _,proof=r.checked_sources();self.assertIn('tools/run_port_reliability_opposite_border_v1_frozen_val.sh',proof['sources'])
        rows,pins=old.load_val_rows();self.assertEqual(len(rows),887)
        self.assertEqual(sum(x['pred'] is not None for x in rows),886)
        self.assertEqual({x['sample_role'] for x in rows},{'val'})


if __name__=='__main__':unittest.main()
