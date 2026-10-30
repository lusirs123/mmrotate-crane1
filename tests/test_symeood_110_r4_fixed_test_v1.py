"""Freeze checkpoint and native TEST invocation before accessing new predictions."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r4_fixed_test', ROOT / 'crane_project/tools/eval_symeood_110_r4_fixed_test_v1.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FrozenTestTests(unittest.TestCase):
    def test_only_original_VAL_epoch16_is_allowed(self):
        cp = dict(checkpoint_sha256=module.CHECKPOINT_SHA)
        self.assertEqual(module.selected(dict(selected_checkpoint='epoch_16', all_checkpoints={'epoch_16': cp})), cp)
        for epoch in ('epoch_18', 'epoch_20', 'epoch_22', 'epoch_24'):
            with self.assertRaises(ValueError):
                module.selected(dict(selected_checkpoint=epoch))
        with self.assertRaises(ValueError):
            module.selected(dict(selected_checkpoint='epoch_16', all_checkpoints={'epoch_16': dict(checkpoint_sha256='wrong')}))

    def test_native_arguments_preserve_model_and_dataset_config(self):
        args = module.native_arguments('sealed-config.py', 'selected.pth')
        self.assertEqual(args[:3], ['tools/test.py', 'sealed-config.py', 'selected.pth'])
        self.assertNotIn('--cfg-options', args)
        self.assertNotIn('--fuse-conv-bn', args)
        self.assertNotIn('--eval', args)
        self.assertIn('--result-identity-out', args)
        self.assertIn('--runtime-audit-out', args)

    def test_failure_run_does_not_cross_sequence_or_frame_gaps(self):
        rows = [dict(sequence='a', frame_id=i, riou=0.) for i in (1, 2, 4)]
        rows += [dict(sequence='b', frame_id=5, riou=0.)]
        self.assertEqual(module.longest_failure(rows), 2)


if __name__ == '__main__':
    unittest.main()
