"""Fixed TEST identity and no-optimization boundaries, standard library only."""
import ast
from copy import deepcopy
import json
import sys
import unittest
from crane_project.tools import eval_symeood_r4_midpoint_fixed_test_v1 as e


class FixedTestChecks(unittest.TestCase):
    def test_static_identity_has_no_tensor_runtime_and_preserves_original_selection(self):
        before='torch' in sys.modules
        identity=e.static_identity()
        self.assertEqual(before, 'torch' in sys.modules)
        self.assertEqual(identity['head_sha256'], e.HEAD_SHA)
        self.assertEqual(identity['detector_sha256'], e.p.R4_SHA)
        self.assertEqual(identity['selected_head_epoch'],20)
        self.assertEqual(identity['physical_gpu'],2)
        self.assertEqual(identity['test_frames'],1440)
        self.assertFalse(identity['selection_on_test'])

    def test_foreign_selection_or_incomplete_training_rejected(self):
        root=e.training_directory(); identity=e.p.checked_sources()
        original=e.read(root/'train/selection.json'); complete=e.read(root/'train/completion.json')
        for field,value in (('epoch',24),('sha256','0'*64),('updates',21576)):
            wrong=deepcopy(original); wrong['selected_checkpoint'][field]=value
            with self.assertRaises(ValueError):e.validate_selection(wrong,complete,identity)
        wrong=deepcopy(complete); wrong['epochs_completed']=23
        with self.assertRaises(ValueError):e.validate_selection(original,wrong,identity)

    def test_representatives_cover_domains_and_fallback(self):
        path=e.training_directory()/'train/val_epoch_20.rows.jsonl'
        rows=[json.loads(v) for v in path.read_text().splitlines()]
        selected=e.representative_rows(rows)
        self.assertEqual(len(selected),8)
        self.assertEqual(sum(r['accepted'] is False for r in selected),1)
        wrong=deepcopy(rows); next(r for r in wrong if r['image']=='real_seq07_00193')['accepted']=True
        with self.assertRaises(ValueError):e.representative_rows(wrong)

    def test_cli_cannot_change_weights_thresholds_or_train(self):
        self.assertEqual(set(e.parser()._option_string_actions),{'-h','--help','--stage'})
        self.assertEqual(e.parser().parse_args([]).stage,'check')

    def test_gate_precedes_test_and_existing_gt_free_capture_is_reused(self):
        source=e.Path(e.__file__).read_text(); tree=ast.parse(source)
        self.assertLess(source.index('replay = e.verify_val_rows'),source.index('sources, data_identity = old.fixed_test_inputs'))
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for name in ('capture','prediction_row','verify_val_rows','fixed_test_inputs','no_grad','load_selected_pipeline'):
            self.assertIn(name,calls)
        for name in ('backward','step','create_cache','load_cache','train_formal','select_best_checkpoint','replay_fit'):
            self.assertNotIn(name,calls)
        self.assertIn('torch.optim.Optimizer.__init__ = forbidden',source)


if __name__=='__main__':unittest.main()
