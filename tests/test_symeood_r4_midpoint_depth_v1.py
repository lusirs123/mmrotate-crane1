"""Frozen depth target/formula and truth separation; no CUDA in these tests."""
import ast
from copy import deepcopy
import sys
import unittest
from crane_project.tools import eval_symeood_r4_midpoint_depth_v1 as e


class DepthChecks(unittest.TestCase):
    def test_static_scope_and_same_formula_without_torch(self):
        before='torch' in sys.modules;identity=e.static_identity()
        self.assertEqual(before,'torch' in sys.modules)
        self.assertEqual(identity['pin']['count'],1097)
        self.assertEqual(identity['target'],'z_cg_opt_m')
        self.assertEqual(identity['calibration_sha256'],e.depth.CAL_SHA)
        self.assertEqual(identity['arms'],['EOOD','R4','R4_midpoint'])
        for key in ('formula_refit','q_support_filter','reliability_filter','selection_on_test','LVSR'):
            self.assertFalse(identity[key])

    def test_cli_has_no_formula_weights_or_threshold_overrides(self):
        self.assertEqual(set(e.parser()._option_string_actions),{'-h','--help','--stage'})
        self.assertEqual(e.parser().parse_args([]).stage,'check')

    def test_truth_parsed_after_all_prediction_streams(self):
        source=e.Path(e.__file__).read_text();tree=ast.parse(source)
        self.assertLess(source.index("save(OUT/'EOOD.proof.json',"),source.index('truths=legacy.read_rows(truth_path)'))
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for name in ('capture','simple_test','depth','summarize','validate_truth','no_grad'):
            self.assertIn(name,calls)
        for name in ('backward','step','train_formal','create_cache','select_best_checkpoint'):
            self.assertNotIn(name,calls)

    def test_missing_output_denominator_and_out_of_support_are_retained(self):
        calibration=e.read(e.ROOT/e.depth.CALIBRATION)
        calibration=deepcopy(calibration)
        calibration['parameters']['q_signed_min']=1.
        calibration['parameters']['q_signed_max']=2.
        manifest=dict(camera=dict(intrinsics=dict(fx=937.2097135167754,fy=937.2097135167754)),
            obb_reference_geometry=dict(long_edge_mean_m=1.775,short_edge_mean_m=.6100409808533983))
        box=[512.,512.,100.,30.,0.,.8]
        predictions=[dict(frame_id='frame_%05d'%i,EOOD=box,R4=box,R4_midpoint=box) for i in range(1097)]
        predictions[0]=dict(predictions[0],EOOD=None)
        truths=[dict(obb_geometry=dict(cx_px=512.,cy_px=512.,w_px=100.,h_px=30.,gamma_deg=0.),
            camera_geometry=dict(z_cg_opt_m=20.),pivot_relative=dict(theta_total_deg=0.)) for _ in range(1097)]
        rows,summary=e.score(predictions,truths,manifest,calibration)
        self.assertEqual(summary['methods']['EOOD']['frame_count'],1097)
        self.assertEqual(summary['methods']['EOOD']['numeric_depth_count'],1096)
        self.assertEqual(summary['direct_depth_errors']['EOOD']['abs_error_coverage']['1.0']['denominator'],1097)
        self.assertEqual(summary['common_valid']['count'],1096)
        self.assertEqual(summary['methods']['R4']['depth_metrics'],summary['methods']['R4_midpoint']['depth_metrics'])
        self.assertEqual(summary['paired']['R4->R4_midpoint']['tied'],1097)
        self.assertEqual(summary['methods']['R4']['q_out_of_fit_support_count'],1097)
        self.assertEqual(summary['methods']['R4']['numeric_depth_count'],1097)


if __name__=='__main__':unittest.main()
