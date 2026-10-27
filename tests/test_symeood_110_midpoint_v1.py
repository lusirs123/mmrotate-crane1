"""Local identity/control tests; no torch, detector, dataset, GPU or remote IO."""
import ast
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('new110_midpoint',ROOT/'crane_project/tools/train_symeood_110_midpoint_v1.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)


class PreparationTests(unittest.TestCase):
    def test_local_contract_checks_real_source_bytes_without_runtime_imports(self):
        self.assertNotIn('torch',sys.modules)
        result=t.checked_sources()
        self.assertEqual(result['sigma_cells'],1.5)
        self.assertEqual(result['frozen_b']['checkpoint_sha256'],
            '7f35dacc308943949a8fd2e6aaad312d14651564903ec87c8c8dcec0552ec054')
        self.assertNotIn('torch',sys.modules)

    def test_original_calculations_are_copied_unchanged(self):
        names=t.unchanged_functions()
        self.assertIn('update',names);self.assertIn('epoch_batches',names)
        self.assertIn('evaluate',names);self.assertIn('dataset_for',names)

    def test_ordinary_focal_and_assignment_match_real_expanded_config(self):
        config=runpy.run_path(str(ROOT/json.loads(t.BASE_CONTRACT.read_text())['config']))
        self.assertFalse(config['model']['bbox_head']['loss_cls']['use_geometry_weight'])
        self.assertEqual(config['model']['bbox_head']['loss_bbox']['type'],'SymKLDLoss')
        tree=ast.parse(Path(t.__file__).read_text())
        cfg=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_cfg')
        self.assertIn(config['model']['train_cfg']['assigner']['type'],
            [n.value for n in ast.walk(cfg) if isinstance(n,ast.Constant) and isinstance(n.value,str)])

    def test_source_drift_is_rejected(self):
        original=t.sha
        target=next(iter(json.loads(t.SOURCES.read_text())['sources']))
        with patch.object(t,'sha',side_effect=lambda p:'0'*64 if Path(p)==ROOT/target else original(p)):
            with self.assertRaisesRegex(ValueError,'source SHA'):t.checked_sources()

    def test_lost_old110_or_changed_epoch_cannot_replace_new_base(self):
        base=json.loads(t.BASE_CONTRACT.read_text())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'base.json';base['selection']['epoch']=24;p.write_text(json.dumps(base))
            with patch.object(t,'BASE_CONTRACT',p):
                with self.assertRaisesRegex(ValueError,'sealed adopted'):t.frontend_identity()

    def test_relative_script_path_python38(self):
        with patch.object(t,'__file__','crane_project/tools/train_symeood_110_midpoint_v1.py'):
            self.assertEqual(t.checked_sources()['sigma_cells'],1.5)

    def test_test_and_lvsr_are_not_exposed(self):
        parser=t.parser()
        self.assertEqual(parser._option_string_actions['--stage'].choices,('check','cache','smoke','trial','formal'))
        for stage in ('train','test','lvsr'):
            with self.assertRaisesRegex(ValueError,'preparation stages'):
                t.run(type('Args',(),{'stage':stage})())

    def test_formal_requires_new_empty_directory_no_existing_cache(self):
        with patch.object(t,'runtime_imports',side_effect=AssertionError('runtime import')):
            with self.assertRaisesRegex(ValueError,'old/trial caches forbidden'):
                t.formal_run(type('Args',(),{'cache_dir':'old_cache','out_dir':'anything','gpu':0})())

    def test_formal_requires_physical_gpu2(self):
        contract=json.loads(t.SOURCES.read_text())['formal_run']
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaisesRegex(ValueError,'physical GPU2'):
                t.formal_run(type('Args',(),{'cache_dir':None,'out_dir':str(ROOT/contract['directory']),'gpu':0})())

    def test_formal_reuses_successful_discarded_trial_without_its_weights(self):
        trial=t.checked_formal_trial()
        self.assertEqual(trial['actual_optimizer_steps'],4)
        self.assertTrue(trial['discarded'])
        self.assertFalse(trial['formal_cache_reusable'])

    def test_formal_training_function_preserves_original_math(self):
        def find(path):
            return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='train_formal')
        old=ast.dump(find(ROOT/'crane_project/tools/train_port_geometry_midpoint_formal_v1.py'),include_attributes=False)
        new=ast.dump(find(Path(t.__file__)),include_attributes=False)
        new=new.replace("Call(func=Name(id='new_head', ctx=Load()), args=[], keywords=[])",
            "Call(func=Attribute(value=Name(id='m', ctx=Load()), attr='SpatialMidpointHead', ctx=Load()), args=[], keywords=[])")
        self.assertEqual(old,new)

    def test_counts_follow_new_cache_without_historical_639(self):
        train=[dict(domain=d,eligible=True,boxes_original=[1]) for d,n in (('real',9),('sim',5)) for _ in range(n)]
        train+=[dict(domain='real',eligible=False,boxes_original=[]),
                dict(domain='sim',eligible=False,boxes_original=[1])]
        out=t.actual_counts(train,[dict(boxes_original=[]),dict(boxes_original=[1])])
        self.assertEqual(out['midpoint_steps_per_epoch'],3)
        self.assertEqual(out['midpoint_formal_budget'],72)
        self.assertEqual(out['lvsr_projected_steps_per_epoch'],2)
        self.assertEqual(out['missing_train'],1)
        self.assertEqual(out['continuation_arms'],['C'])
        self.assertIsNone(out['lvsr_parent_epoch'])

    def test_empty_eligible_domain_stops(self):
        with self.assertRaisesRegex(ValueError,'No eligible'):
            t.actual_counts([dict(domain='real',eligible=True,boxes_original=[1])],[])

    def test_trial_rejects_old_cache_before_any_runtime_import(self):
        with patch.object(t,'runtime_imports',side_effect=AssertionError('runtime import')):
            with self.assertRaisesRegex(ValueError,'refuses existing cache'):
                t.finite_trial(type('Args',(),{'cache_dir':'old_cache'})())

    def test_static_check_does_not_create_output_or_import_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'must_not_exist'
            with patch.object(t,'runtime_imports',side_effect=AssertionError('runtime import')):
                with patch('builtins.print'):
                    result=t.run(type('Args',(),{'stage':'check','out_dir':str(out)})())
            self.assertFalse(out.exists())
            self.assertFalse(result['cache_created'])
            self.assertEqual(result['optimizer_updates'],0)

    def test_original_balanced_schedule_covers_eligible_views_only(self):
        # Execute the copied schedule alone, without loading training modules.
        import numpy as np
        function=next(n for n in ast.parse(Path(t.__file__).read_text()).body
            if isinstance(n,ast.FunctionDef) and n.name=='epoch_batches')
        env={'np':np,'math':math,'SETTINGS':t.SETTINGS}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<original_schedule>','exec'),env)
        rows=[dict(role='train',eligible=True,domain=d) for d,n in (('real',9),('sim',5)) for _ in range(n)]
        rows+=[dict(role='val',eligible=False,domain='real'),dict(role='train',eligible=False,domain='sim')]
        batches=env['epoch_batches'](rows,1)
        self.assertEqual(len(batches),3)
        self.assertEqual({i for batch in batches for i in batch},set(range(14)))
        self.assertTrue(all(sum(rows[i]['domain']=='real' for i in b)==4 and len(b)==8 for b in batches))
        self.assertEqual(batches,env['epoch_batches'](rows,1))
        self.assertNotEqual(batches,env['epoch_batches'](rows,2))


if __name__=='__main__':unittest.main()
