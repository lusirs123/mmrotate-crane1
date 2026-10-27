"""Fixed inference boundaries; model numerics checked once by real8-frame gate."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'crane_project/tools/eval_symeood_110_midpoint_ep21_test.py'
spec=importlib.util.spec_from_file_location('fixed_ep21',p)
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


class FixedEvaluationTests(unittest.TestCase):
    def test_identity_fixes_original_selected_weight(self):
        _,lock,selection=e.checked_identity()
        self.assertEqual(selection['selected_checkpoint']['epoch'],21)
        self.assertEqual(selection['selected_checkpoint']['sha256'],e.HEAD_SHA)
        self.assertEqual(lock['test_frames'],1440)
        self.assertFalse(lock['full_VAL_replay'])

    def test_representatives_include_both_domains_and_exact_missing_frames(self):
        rows=[json.loads(x) for x in (e.FORMAL/'train/val_epoch_21.rows.jsonl').read_text().splitlines()]
        selected=e.representative_rows(rows)
        self.assertEqual(len(selected),8)
        self.assertEqual(sum(r['b'] is None for r in selected),2)
        self.assertEqual({r['domain'] for r in selected},{'real','sim'})
        broken=deepcopy(rows)
        next(r for r in broken if r['image']=='real_seq07_00167')['b']=[1]*6
        with self.assertRaisesRegex(ValueError,'missing outputs'):e.representative_rows(broken)

    def test_cli_cannot_override_weights_epoch_thresholds_or_train(self):
        options=e.parser()._option_string_actions
        self.assertEqual(set(options),{'-h','--help','--gpu','--out-dir'})

    def test_original_inference_and_gate_reused_without_optimizer_or_cache_calls(self):
        tree=ast.parse(p.read_text())
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for name in ('capture','prediction_row','verify_val_rows','fixed_test_inputs','test_summaries','no_grad'):
            self.assertIn(name,calls)
        for name in ('backward','step','create_cache','load_cache','train_formal','select_best_checkpoint','replay_fit'):
            self.assertNotIn(name,calls)
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and n.func.attr=='update' and isinstance(n.func.value,ast.Name) and n.func.value.id in ('f','t','optimizer')
            for n in ast.walk(tree)))
        source=p.read_text()
        self.assertLess(source.index('replay=e.verify_val_rows'),source.index('sources,data=old.fixed_test_inputs'))


if __name__=='__main__':unittest.main()
