"""Frozen L5 depth identity, full denominators, truth separation and formula."""
import ast
from copy import deepcopy
import sys
import unittest
from crane_project.tools import eval_symeood_r4_lvsr_ep5_depth_v1 as e


class L5DepthChecks(unittest.TestCase):
    def test_static_identity_does_not_import_tensor_runtime_or_enable_EOOD(self):
        before='torch' in sys.modules;value=e.static_identity()
        self.assertEqual(before,'torch' in sys.modules)
        self.assertEqual(value['pin']['count'],1097)
        self.assertEqual(value['arms'],['R4','R4_midpoint','R4_midpoint_L5'])
        self.assertEqual(value['parent']['lvsr_sha256'],e.fixed.HEAD_SHA)
        self.assertEqual(value['calibration_sha256'],e.depth.CAL_SHA)
        self.assertEqual(value['target'],'z_cg_opt_m')
        for key in ('EOOD','formula_refit','q_support_filter','reliability_filter','selection_on_test','depth_for_small_paper'):
            self.assertFalse(value[key])

    def test_cli_has_no_refit_or_checkpoint_scan(self):
        self.assertEqual(set(e.parser()._option_string_actions),{'-h','--help','--stage'})
        self.assertEqual(e.parser().parse_args([]).stage,'check')

    def test_gt_free_predict_and_closed_stream_before_truth(self):
        source=e.Path(e.__file__).read_text();tree=ast.parse(source)
        self.assertLess(source.index("save(OUT/'prediction_completion.json'"),source.index('truths=legacy.read_rows(truth_path)'))
        predict=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='predict')
        self.assertEqual([a.arg for a in predict.args.args],['image','metas'])
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for name in ('backward','step','create_cache','fit','select_best_checkpoint'):
            self.assertNotIn(name,calls)
        self.assertIn('torch.optim.Optimizer.__init__=forbidden',source)

    def test_missing_depth_and_q_outside_support_stay_in_full_denominators(self):
        calibration=deepcopy(e.read(e.ROOT/e.depth.CALIBRATION))
        calibration['parameters']['q_signed_min']=1.;calibration['parameters']['q_signed_max']=2.
        manifest=dict(camera=dict(intrinsics=dict(fx=937.2097135167754,fy=937.2097135167754)),
            obb_reference_geometry=dict(long_edge_mean_m=1.775,short_edge_mean_m=.6100409808533983))
        box=[512.,512.,100.,30.,0.,.8]
        preds=[dict(frame_id='frame_%05d'%i,**{a:box for a in e.ARMS}) for i in range(1097)]
        preds[0]=dict(preds[0],R4_midpoint_L5=None)
        truths=[dict(obb_geometry=dict(cx_px=512.,cy_px=512.,w_px=100.,h_px=30.,gamma_deg=0.),
            camera_geometry=dict(z_cg_opt_m=20.),pivot_relative=dict(theta_total_deg=0.)) for _ in range(1097)]
        rows,value=e.score(preds,truths,manifest,calibration)
        self.assertEqual(value['methods']['R4_midpoint_L5']['numeric_depth_count'],1096)
        self.assertEqual(value['methods']['R4_midpoint_L5']['q_out_of_fit_support_count'],1096)
        self.assertEqual(value['direct_depth_errors']['R4_midpoint_L5']['abs_error_coverage']['1.0']['denominator'],1097)
        self.assertEqual(value['common_valid']['count'],1096)
        self.assertEqual(value['paired']['R4_midpoint->R4_midpoint_L5']['tied'],1096)
        with self.assertRaises(ValueError):e.score(preds[:-1],truths,manifest,calibration)


if __name__=='__main__':unittest.main()
