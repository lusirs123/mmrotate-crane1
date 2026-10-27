"""Controls specific to the separately authorized descriptive-parent C run."""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'crane_project/tools/run_symeood_110_lvsr_c_ep7.py'
spec=importlib.util.spec_from_file_location('continuation_ep7',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ContinuationControls(unittest.TestCase):
    def test_separate_authorization_and_unchanged_common_sources(self):
        p=m.checked_sources()
        self.assertEqual(p['parent_epoch'],7);self.assertEqual(p['arms'],['C']);self.assertEqual(p['epochs'],6)
        self.assertEqual(p['parent_role'],'descriptive_only')

    def test_eligible_selection_and_descriptive_ranking_are_separate(self):
        events=[dict(epoch=1,selection_key=[.1,1],eligibility=dict(pass_all=False)),
                dict(epoch=2,selection_key=[.2,2],eligibility=dict(pass_all=True))]
        result=m.versions(events)
        self.assertEqual(result['selected'],2);self.assertEqual(result['representative'],1);self.assertEqual(result['fixed_final'],6)
        events[1]['eligibility']['pass_all']=False
        self.assertIsNone(m.versions(events)['selected'])

    def test_reuses_training_without_detector_or_midpoint_forward(self):
        tree=ast.parse(PATH.read_text())
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertIn('fit_stage',calls);self.assertIn('update',calls);self.assertIn('evaluate',calls)
        for name in ('load_inputs','build_detector','extract_feat','new_head','create_cache','execute','engineering'):
            self.assertNotIn(name,calls)
        self.assertFalse(any(isinstance(n,ast.FunctionDef) and n.name in ('update','evaluate','losses') for n in tree.body))


if __name__=='__main__':unittest.main()
