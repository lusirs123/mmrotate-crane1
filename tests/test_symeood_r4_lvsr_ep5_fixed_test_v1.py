"""CPU checks for fixed representative identity and no-update TEST boundaries."""
import ast
from copy import deepcopy
import gzip
import json
import sys
import unittest
from crane_project.tools import eval_symeood_r4_lvsr_ep5_fixed_test_v1 as e


class FixedL5Checks(unittest.TestCase):
    def test_static_check_preserves_null_selection_without_torch(self):
        before='torch' in sys.modules
        value=e.static_identity()
        self.assertEqual(before,'torch' in sys.modules)
        self.assertIsNone(value['selected_epoch'])
        self.assertTrue(value['representative_only'])
        self.assertEqual(value['lvsr_sha256'],e.HEAD_SHA)
        self.assertFalse(value['depth'])
        self.assertFalse(value['selection_on_test'])

    def test_wrong_checkpoint_or_promoted_selection_is_rejected(self):
        folder=e.training_directory();s=e.read(folder/'selection.json');c=e.read(folder/'completion.json')
        identity=e.p.checked_sources()
        for key,value in [('selected_epoch',5),('representative_epoch',24),('TEST',True),('eligible_epochs',[5])]:
            wrong=deepcopy(s);wrong[key]=value
            with self.assertRaises(ValueError):e.validate_selection(wrong,c,identity)
        wrong=deepcopy(c);wrong['updates']=3195
        with self.assertRaises(ValueError):e.validate_selection(s,wrong,identity)

    def test_payload_rejects_old_frontend_continuation_or_final_epoch(self):
        identity={'fixed':'identity'}
        v=dict(protocol=e.p.VERSION,identity=identity,epoch=5,updates=3195,updates_per_epoch=639,
            parameters=2480,parent_checkpoint=None,detector_sha256=e.p.parent.R4_SHA,midpoint_sha256=e.p.MIDPOINT_SHA)
        e.validate_payload(v,identity)
        for key,value in [('epoch',24),('parent_checkpoint','old110'),('parameters',2800),('midpoint_sha256','0'*64)]:
            with self.assertRaises(ValueError):e.validate_payload(dict(v,**{key:value}),identity)

    def test_fixed_val_samples_and_decision_mismatch(self):
        with gzip.open(e.training_directory()/'epoch_05_VAL.rows.jsonl.gz','rt') as f:
            rows=[json.loads(v) for v in f]
        samples=e.sample_references(rows)
        self.assertEqual([r['image'] for r in samples],list(e.fixed.VAL_NAMES))
        self.assertEqual({r['domain'] for r in samples},{'real','sim'})
        with self.assertRaises(ValueError):e.sample_references(rows[:-1])
        for key,value in [('size_delivery','other'),('failed_checks',['wrong']),('vector_valid',False)]:
            ref=samples[0]
            with self.assertRaises(ValueError):e.compare_decisions(dict(ref,**{key:value}),ref)

    def test_cli_and_online_boundary_cannot_train_or_choose(self):
        self.assertEqual(set(e.parser()._option_string_actions),{'-h','--help','--stage'})
        source=e.Path(e.__file__).read_text();tree=ast.parse(source)
        self.assertLess(source.index('replay=legacy.e.verify_val_rows'),source.index('sources,data_identity=old.fixed_test_inputs'))
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for name in ('backward','step','load_cache','create_cache','fit'):
            self.assertNotIn(name,calls)
        self.assertIn('torch.optim.Optimizer.__init__ = forbidden',source)
        predict=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='predict')
        self.assertEqual([a.arg for a in predict.args.args],['image','metas'])


if __name__=='__main__':unittest.main()
