from copy import deepcopy
import ast
import inspect
import unittest
import numpy as np
from crane_project.utils import port_reliability_remaining_error_support_v1 as c


def row(i,accept=True,bad=False):
    return dict(image='real_seq01_%05d'%i,sequence='real_seq01',frame_id=i,domain='real',split='train',reliability_role='train',
        pred=[100,100,40,10,0,.9],gt=[100,100,50 if bad else 40,10,0],image_size=[400,300],
        original_simple_decision=dict(size_accepted=accept),temporal=dict(available=True,primary=.01,acquisition_boundary=False))


def fixture():
    rows=[row(0,bad=True),row(1,False,True),row(2),row(3),row(4),row(5,False,True),row(10)]
    ids=[r['image'] for r in rows];desc=np.array([c.simple.descriptor(r['pred'],r['image_size']) for r in rows])
    raw=np.zeros((len(rows),256));raw[:,0]=[0,1,2,3,4,.1,10]
    return rows,np.c_[desc,raw],ids


class Test(unittest.TestCase):
    def test_four_states_and_missing(self):
        for accepted,bad,state in [(True,True,'FA'),(True,False,'CR'),(False,True,'ED'),(False,False,'FR')]:
            self.assertEqual(c.classify(row(0,accepted,bad))['state'],state)
        r=row(0);r['pred']=None;self.assertEqual(c.classify(r)['state'],'MISSING')

    def test_long_short_canonical_and_exact_boundary(self):
        r=row(0,True,True);a=c.classify(r);r['pred'][2:4]=r['pred'][3:1:-1];r['gt'][2:4]=r['gt'][3:1:-1];self.assertEqual(c.classify(r),a)
        r=row(0);r['pred'][2]=44.;self.assertEqual(c.classify(r)['state'],'CR')
        r['pred'][2]=44.001;self.assertEqual(c.classify(r)['state'],'FA')

    def test_error_runs_accept_change_does_not_split(self):
        a,b=row(0,True,True),row(1,False,True);runs=c.error_runs([a,b]);self.assertEqual(len(runs),1);self.assertEqual((runs[0]['FA'],runs[0]['ED']),(1,1))
        b['temporal']['acquisition_boundary']=True;self.assertEqual(len(c.error_runs([a,b])),2)

    def test_gaps_correct_frames_missing_break_runs(self):
        a,b=row(0,True,True),row(2,True,True);self.assertEqual(len(c.error_runs([a,b])),2)
        z=row(1);z['pred']=None;self.assertEqual(len(c.error_runs([a,z,b])),2)

    def test_only_train_unique_chronological(self):
        r=row(0);r['split']='test';r['reliability_role']='test'
        with self.assertRaises(ValueError):c.validate([r])
        r=row(0)
        with self.assertRaises(ValueError):c.validate([r,r])
        with self.assertRaises(ValueError):c.validate([row(1),row(0)])

    def test_matched_donor_excludes_same_run_and_near_frames(self):
        rows,x,ids=fixture();v=c.audit(rows,x,ids);d=v['remaining_FA'][0]
        self.assertEqual(d['matched_ED_images'],[row(5)['image']]);self.assertEqual(d['matched_CR_images'],[row(4)['image'],row(10)['image']])
        self.assertEqual(v['support']['unique_matched_ED'],1);self.assertEqual(v['proximity']['native256']['ED_closer'],1)
        self.assertFalse(v['support']['material_precondition_passed'])

    def test_incompatible_axis_direction_not_donor(self):
        rows,x,ids=fixture();rows[5]['gt'][2]=20.;v=c.audit(rows,x,ids);self.assertEqual(v['remaining_FA'][0]['matched_ED_images'],[])
        self.assertIsNone(v['remaining_FA'][0]['nearest']['native256']['rejected_error'])

    def test_history_regime_not_filled_zero(self):
        rows,x,ids=fixture();rows[5]['temporal']=dict(available=False,primary=None);v=c.audit(rows,x,ids)
        self.assertEqual(v['remaining_FA'][0]['matched_ED_images'],[])

    def test_full_cache_pairing_and_finite_values(self):
        rows,x,ids=fixture()
        with self.assertRaises(ValueError):c.audit(rows,x,ids[::-1])
        x[0,0]=np.nan
        with self.assertRaises(ValueError):c.audit(rows,x,ids)

    def test_no_input_changes_or_classifier_fit(self):
        rows,x,ids=fixture();a=deepcopy(rows);b=x.copy();c.audit(rows,x,ids);self.assertEqual(rows,a);np.testing.assert_array_equal(x,b)
        from crane_project.tools import run_port_reliability_remaining_error_support_v1 as runner
        calls=[n.func.attr for n in ast.walk(ast.parse(inspect.getsource(runner))) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(any(n in calls for n in ('fit','calibrate','select_cutoff','train_head')))
        self.assertNotIn('scored_VAL',inspect.getsource(runner));self.assertNotIn('scored_TEST',inspect.getsource(runner))


if __name__=='__main__':unittest.main()
