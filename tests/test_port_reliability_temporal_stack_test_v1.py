import ast
from copy import deepcopy
import inspect
import unittest
import numpy as np
from crane_project.utils import port_reliability_temporal_stack_test_v1 as c


def bundle():
    model=lambda d:dict(weights=[0.]+[.2]*d,mean=[0.]*d,scale=[1.]*d)
    # A real validated formal policy is used only as fixed configuration in unit tests.
    import json
    from pathlib import Path
    policy=json.loads(Path('work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit/policy.json').read_text())['simple_policy']
    return dict(models={k:model(d) for k,d in [('fusion4',4),('static3',3),('fallback3',3)]},pauc=model(4),
        fusion=dict(weights=[.4,.1,.2],normalizer=dict(mean=[0.,0.],scale=[1.,1.])),policy=policy,
        cutoffs=dict(fusion4=.7,partial_auc4=.4,crossfit_fusion=.9,fallback_margin=1.,temporal_veto=.08))


def row(i,pred=None,gt=None):
    p=pred if pred is not None else [100,100,40,10,0,.9]
    return dict(image='real_seq03_%05d'%i,sequence='real_seq03',domain='real',frame_id=i,split='test',image_size=[400,300],
        pred=p,gt=gt or p[:5],reliability_role='test')


def formal(r,b):
    v=c.base.simple.SimpleComponentReliability(b['policy']).decide(r['pred'],r['image_size']);v['final_box_original']=v.pop('raw_b_output');return v


class Test(unittest.TestCase):
    def test_missing_is_not_quality_error(self):
        b=bundle();point=c.base.level.CausalSizeEvidence().observe(None,0,[400,300])
        risks,flags,_=c.score(None,[400,300],point,None,b)
        self.assertTrue(all(v is None for v in risks.values()));self.assertFalse(any(flags.values()))
        r=row(0);r['pred']=None
        self.assertEqual(c.summary([r],set())['states']['MISSING'],1)

    def test_fallback_uses_original_branch_cutoff(self):
        b=bundle();p=row(0)['pred'];point=c.base.level.CausalSizeEvidence().observe(p,0,[400,300])
        risks,flags,_=c.score(p,[400,300],point,.1,b)
        self.assertEqual(flags['fallback_margin'],risks['partial_auc4']<=.4)
        self.assertEqual(risks['crossfit_fusion'],risks['partial_auc4'])
        self.assertNotEqual(flags['crossfit_fusion'],flags['fallback_margin'])

    def test_geometry_flags_input_preserved(self):
        b=bundle();p=row(0)['pred'];before=deepcopy(p);point=dict(available=True,primary=.2)
        risks,flags,original=c.score(p,[400,300],point,.5,b)
        self.assertEqual(p,before);self.assertEqual(original['final_box_original'],p)
        self.assertTrue(original['center_accepted']);self.assertEqual(flags['fallback_margin'],risks['crossfit_fusion']<=.9)

    def test_scoring_has_no_GT_or_group_inputs(self):
        self.assertEqual(list(inspect.signature(c.score).parameters),['pred','image_size','point','image_risk','bundle'])
        self.assertFalse(any(name in inspect.getsource(c.score) for name in ('gt','sequence','domain')))

    def test_causal_reset_gap_and_future(self):
        b=bundle();rows=[row(i) for i in range(5)]
        for r in rows:r['original_simple_decision']=formal(r,b)
        ri={r['image']:.3 for r in rows};a=c.replay(rows,ri,b)
        self.assertEqual([r['temporal']['available'] for r in a],[False,False,False,True,True])
        rows[-1]['pred'][2]=99;rows[-1]['original_simple_decision']=formal(rows[-1],b)
        z=c.replay(rows,ri,b);self.assertEqual(a[:4],z[:4])
        rows=rows[:2]+rows[3:];z=c.replay(rows,ri,b);self.assertFalse(any(r['temporal']['available'] for r in z))

    def test_GT_does_not_change_decision(self):
        b=bundle();r=row(0);r['original_simple_decision']=formal(r,b)
        a=c.replay([r],{r['image']:.3},b);r['gt'][2]=100
        z=c.replay([r],{r['image']:.3},b);self.assertEqual(a[0]['flags'],z[0]['flags']);self.assertEqual(a[0]['risks'],z[0]['risks'])

    def test_same_CR_whole_tie_and_zero(self):
        rows=[row(0),row(1,gt=[100,100,100,10,0]),row(2)]
        for r in rows:r['risks']=dict(full_simple=.5)
        v=c.matched(rows,'full_simple',correct=1);self.assertEqual(v['states']['CR'],2);self.assertFalse(v['exact_CR'])
        self.assertEqual(c.matched(rows,'full_simple',correct=0)['accepted_outputs'],0)
        v=c.matched(rows,'full_simple',count=1);self.assertEqual(v['tie_bounds']['FA_min'],0);self.assertEqual(v['tie_bounds']['FA_max'],1)

    def test_duplicate_or_mixed_roles_rejected(self):
        b=bundle();r=row(0)
        with self.assertRaises(ValueError):c.replay([r,r],{},b)
        v=row(1);v['split']='val'
        with self.assertRaises(ValueError):c.summary([r,v],set())

    def test_no_fit_or_calibration_called_by_runner(self):
        from crane_project.tools import eval_port_reliability_temporal_stack_test_v1 as runner
        tree=ast.parse(inspect.getsource(runner))
        calls=[node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
        self.assertFalse(any(x in calls for x in ('fit','calibrate','select_cutoff','train_head')))
        source=inspect.getsource(runner.run);self.assertLess(source.index("io.write(out/'frozen_bundle.json'"),source.index('test_rows(paths,bundle)'))


if __name__=='__main__':unittest.main()
