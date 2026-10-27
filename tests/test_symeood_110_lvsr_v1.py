"""Original-function reuse, conditional continuation and dynamic-count controls."""
import ast
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import random
import unittest

ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'crane_project/tools/run_symeood_110_lvsr_v1.py'
spec=importlib.util.spec_from_file_location('new_lvsr',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class LVSRControls(unittest.TestCase):
    def test_original_batch_loss_update_evaluate_unmodified(self):
        def funcs(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
        old=funcs(ROOT/'crane_project/tools/run_port_size_qalign_v1.py');new=funcs(p)
        for name in ('batch','update','evaluate'):self.assertEqual(old[name],new[name])

    def test_frozen_identity_and_original_method_sources(self):
        lock=m.checked_sources()
        self.assertEqual(lock['common_arms'],['fixed_b']);self.assertEqual(lock['continuation_arms'],['C'])
        self.assertEqual(lock['common_epochs'],24);self.assertEqual(lock['continuation_epochs'],6)
        self.assertEqual(lock['physical_gpu'],0)

    def test_schedule_matches_original_and_keeps_tail(self):
        for continuation in (False,True):
            expected=list(range(19));random.Random(1703+(20 if continuation else 0)+3).shuffle(expected)
            actual=m.schedule_indices(19,3,continuation)
            self.assertEqual(actual,expected)
            batches=[actual[i:i+8] for i in range(0,len(actual),8)]
            self.assertEqual([len(x) for x in batches],[8,8,3])
        with self.assertRaises(ValueError):m.schedule_indices(0,1)

    def test_no_eligible_epoch_is_descriptive_only(self):
        events=[dict(epoch=1,selection_key=[.2,.2,.2,.2,1],eligibility=dict(pass_all=False)),
                dict(epoch=2,selection_key=[.1,.1,.1,.1,2],eligibility=dict(pass_all=False))]
        value=m.choose(events);self.assertIsNone(value['selected']);self.assertEqual(value['representative'],2)
        events[0]['eligibility']['pass_all']=True
        value=m.choose(events);self.assertEqual(value['selected'],1)

    def test_original_adoption_rejects_dfr_cost(self):
        summary={};diag={}
        for d in ('real','sim'):
            base=dict(sizes=dict(edges={e:dict(mae=.1) for e in ('long','short')},a=dict(abs_mae=.1),r=dict(abs_mae=.1)),
                mean_riou_full_frame=.8,temporal={d:dict(dfr_percent_per_frame=2.,mcml_max=2,mcml_mean=1.,aci=.95,aci_adjacent_pairs=10,tdr_w10=dict(numerator=5,denominator=5))})
            candidate=deepcopy(base)
            for e in ('long','short'):candidate['sizes']['edges'][e]['mae']=.09
            summary[d]=dict(midpoint=base,edge_residual=candidate,paired=dict(frozen_identity=True,new_output=0,lost_output=0))
            diag[d]=dict(near={e:dict(before_mae=.01,after_mae=.01) for e in ('long','short')})
        summary['overall']=dict(midpoint=dict(mean_riou_full_frame=.8),edge_residual=dict(mean_riou_full_frame=.8))
        self.assertTrue(m.eligibility(summary,diag)['pass_all'])
        summary['real']['edge_residual']['temporal']['real']['dfr_percent_per_frame']=2.1
        gate=m.eligibility(summary,diag)
        self.assertFalse(gate['pass_all']);self.assertIn('real/dfr',gate['failed'])

    def test_no_historical_fixed_counter_or_second_arms(self):
        tree=ast.parse(p.read_text());fit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit_stage')
        self.assertNotIn(639,[n.value for n in ast.walk(fit) if isinstance(n,ast.Constant)])
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for forbidden in ('create_cache','build_detector','extract_feat','fixed_test_inputs','capture'):
            self.assertNotIn(forbidden,calls)


if __name__=='__main__':unittest.main()
