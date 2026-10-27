"""Frozen versions, unchanged metric/formula paths and bounded inference gate."""
import ast
import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'crane_project/tools/eval_symeood_110_lvsr_downstream.py'
spec=importlib.util.spec_from_file_location('new_downstream',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class FrozenEvaluation(unittest.TestCase):
    def test_original_detection_and_depth_reporting_exact(self):
        def funcs(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
        old=funcs(ROOT/'crane_project/utils/port_size_a20_protection_downstream_v1.py');new=funcs(PATH)
        for name in ('detection_report','depth_report'):self.assertEqual(old[name],new[name])

    def test_fixed_versions_calibration_and_exposed_sequence(self):
        p=m.checked_sources()
        self.assertEqual(p['versions'],['b','midpoint','ep7','c3','c6'])
        self.assertEqual(p['sequence'],'unknown_test_02_coverage_retry')
        self.assertEqual(p['calibration_sha256'],m.depth.CAL_SHA)
        self.assertEqual(len(p['val_names']),8);self.assertFalse(p['full_VAL_replay'])

    def test_native_identity_rejects_presence_and_score_drift(self):
        p=dict(b=[1.,2.,10.,4.,.1,.8]);self.assertEqual(m.compare_frame(p,p,['b']),{'b':0.})
        with self.assertRaises(ValueError):m.compare_frame(p,dict(b=None),['b'])
        with self.assertRaises(ValueError):m.compare_frame(p,dict(b=[1.,2.,10.,4.,.1,.7]),['b'])

    def test_no_training_cache_or_full_old_queue(self):
        tree=ast.parse(PATH.read_text())
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for forbidden in ('backward','step','create_cache','load_cache','checked_references','load_native','require_val_proof','fit','execute'):
            self.assertNotIn(forbidden,calls)


if __name__=='__main__':unittest.main()
