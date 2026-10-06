"""Necessary guards for frozen baseline extension, native top1 and error algebra."""
import ast
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crane_project.utils import port_depth_four_arm_v1 as c
from tools.depth import audit_port_depth_four_arm_v1 as r
from test_port_midpoint_depth_v1 import fixture,calibration


class ContractTests(unittest.TestCase):
    def test_only_two_new_fixed_baselines_no_selection_knobs(self):
        self.assertEqual(set(c.BASELINES),{'eood','symeood'})
        self.assertEqual(c.BASELINES['eood']['epoch'],24)
        self.assertEqual(c.BASELINES['symeood']['epoch'],20)
        flags={v for a in r.parser()._actions for v in a.option_strings}
        for v in ('--sigma','--epoch','--threshold','--limit','--fit-offset','--b-checkpoint'):
            self.assertNotIn(v,flags)
        self.assertIn('do not rerun',c.protocol_document()['inference'])

    def test_eood_native_top1_keeps_score_order_ties_and_empty(self):
        boxes=[[0.,0.,5.,2.,0.,.2],[1.,1.,5.,2.,0.,.9],[2.,2.,5.,2.,0.,.9]]
        before=deepcopy(boxes);box,index=c.top1(boxes,'eood')
        self.assertEqual(index,1);self.assertEqual(box,boxes[1]);self.assertEqual(boxes,before)
        self.assertEqual(c.top1([],'eood'),(None,None))
        with self.assertRaises(ValueError):c.top1(boxes,'symeood')

    def test_invalid_native_boxes_are_not_repaired_or_filtered(self):
        for b in ([0.,0.,0.,2.,0.,.9],[0.,0.,5.,2.,0.,math.nan],[0.,0.,5.,2.,0.,1.1]):
            with self.assertRaises(ValueError):c.top1([b],'eood')

    def test_original_config_migration_and_selection_identity(self):
        for arm,spec in c.BASELINES.items():
            key='epoch_%d'%spec['epoch']
            selection=dict(evidence_role='source_val_checkpoint_selection',selected_checkpoint=key,
                config_sha256=spec['historical_config_sha256'],all_checkpoints={key:dict(checkpoint_sha256=spec['checkpoint_sha256'])})
            self.assertTrue(c.selection_contract(r.ROOT,arm,selection)['archived_overrides_equivalent'])
            wrong=deepcopy(selection);wrong['selected_checkpoint']='epoch_1'
            with self.assertRaises(ValueError):c.selection_contract(r.ROOT,arm,wrong)

    def test_checkpoint_meta_validates_resolved_fields_and_rejects_execution(self):
        cfg={k:{} for k in ('model','data','optimizer','optimizer_config','lr_config','runner','load_from','resume_from')}
        text='\n'.join(k+' = {}' for k in cfg)
        self.assertEqual(c.checkpoint_meta(dict(config=text,epoch=20,seed=0),cfg,20)['epoch'],20)
        for epoch,seed in ((24,0),(20,1)):
            with self.assertRaises(ValueError):c.checkpoint_meta(dict(config=text,epoch=epoch,seed=seed),cfg,20)
        with self.assertRaises(ValueError):c.checkpoint_meta(dict(config="model = __import__('os').system('false')",epoch=20,seed=0),cfg,20)

    def test_error_propagation_identity_is_signed_not_mae_and_no_mutation(self):
        row=dict(truth_z_m=10.,gt_obb=dict(depth=dict(z_m=10.,q_signed=0.)),
                 b=dict(depth=dict(z_m=10/.95*math.exp(.2),q_signed=.1),
                        geometry=dict(short_relative_error=-.05)))
        before=deepcopy(row);x=c.propagation([row],dict(parameters=dict(beta=20.)))
        self.assertAlmostEqual(sum(x['signed_error_component_mean_m'].values()),row['b']['depth']['z_m']-10,places=12)
        self.assertLess(x['max_signed_identity_residual_m'],1e-12)
        self.assertEqual(row,before);self.assertIn('not to MAE',x['note'])

    def test_numeric_failure_never_improves_full_frame_error_coverage(self):
        m,truth,pred,oracle=fixture()
        pred[1].update(b=None,midpoint=None,accepted=None)
        with patch.object(c.original,'COUNT',2),patch.object(c.original,'ORACLE',oracle):
            rows,_=c.original.evaluate(pred,truth,m,calibration())
        x=c.direct_errors(rows)
        self.assertEqual(x['frame_count'],2);self.assertEqual(x['numeric_count'],1)
        self.assertEqual(x['abs_error_coverage']['1.0']['denominator'],2)
        self.assertEqual(x['abs_error_coverage']['1.0']['fraction'],.5)

    def test_counter_covers_direct_forward_and_restores_original_result(self):
        class Head:
            def forward(self,x,scale=1):return x*scale
        h=Head();counts=[0,0];native=r.count_main_forward(h,counts)
        self.assertEqual(h.forward(3,scale=2),6);self.assertEqual(counts,[0,1])
        h.forward=native;self.assertEqual(h.forward(4),4);self.assertEqual(counts,[0,1])

    def test_summary_reproduction_preserves_counts_and_detects_real_change(self):
        r.same_summary(dict(count=981,value=1.0),dict(count=981,value=1.0+1e-12))
        with self.assertRaises(ValueError):r.same_summary(dict(count=980,value=1.),dict(count=981,value=1.))
        with self.assertRaises(ValueError):r.same_summary(dict(value=1.01),dict(value=1.))

    def test_static_stage_never_loads_reference_data_models_and_artifact_protocol(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'load_reference') as archive,patch.object(r.prior,'checked_dataset') as data,patch.object(r,'native_baseline') as native:
            out=Path(tmp)/'static';r.run(r.parser().parse_args(['--stage','check','--out-dir',str(out)]))
            archive.assert_not_called();data.assert_not_called();native.assert_not_called()
            self.assertEqual(json.loads((out/'artifacts.json').read_text())['protocol'],c.VERSION)
            with self.assertRaises(FileExistsError):r.run(r.parser().parse_args(['--stage','check','--out-dir',str(out)]))

    def test_reference_rejects_unreviewed_archive_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'wrong.tar.gz';p.write_bytes(b'wrong historical result')
            with self.assertRaises(ValueError):r.load_reference(p)

    def test_new_predictions_precede_truth_parsing_and_eood_constructor_retained(self):
        tree=ast.parse(Path(r.__file__).read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        lines={name:[n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and getattr(n.func,'id','')==name] for name in ('native_baseline','evaluate')}
        self.assertLess(lines['native_baseline'][0],lines['evaluate'][0])
        source=Path(r.__file__).read_text();self.assertIn("if arm!='eood':model.train_cfg=None",source)
        self.assertIn('rescale=True',source)


NATIVE=all(importlib.util.find_spec(name) is not None for name in ('torch','numpy'))


@unittest.skipUnless(NATIVE,'Torch/NumPy not installed locally')
class NativeCounterTests(unittest.TestCase):
    def test_nn_call_forward_counting(self):
        import torch
        class Head(torch.nn.Module):
            def forward(self,x):return x+1
        h=Head();counts=[0,0];native=r.count_main_forward(h,counts)
        self.assertEqual(h(torch.tensor(2.)).item(),3.);self.assertEqual(counts,[0,1]);h.forward=native

    def test_direct_forward_counting(self):
        import torch
        class Head(torch.nn.Module):
            def forward(self,x):return x+1
        h=Head();counts=[0,0];native=r.count_main_forward(h,counts)
        self.assertEqual(h.forward(torch.tensor(2.)).item(),3.);self.assertEqual(counts,[0,1]);h.forward=native


if __name__=='__main__':unittest.main()
