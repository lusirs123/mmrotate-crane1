"""Frozen-depth contract tests; optional native midpoint bridge uses CPU fixtures."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_midpoint_depth_v1 as c
from tools.depth import audit_port_midpoint_depth_v1 as runner


def calibration():
    return json.loads((ROOT/c.CALIBRATION).read_text())


def fixture(count=2):
    intr = dict(fx=937.2097135167754, fy=937.2097135167754)
    geometry = dict(long_edge_mean_m=1.775, short_edge_mean_m=.6100409808533983)
    manifest = dict(sequence_id=c.SEQUENCE, split='calibration_train', run_instance_id='synthetic',
                    camera=dict(intrinsics=intr), obb_reference_geometry=geometry)
    truths, predictions = [], []
    for i in range(count):
        w, h = intr['fx']*geometry['long_edge_mean_m']/20, intr['fy']*geometry['short_edge_mean_m']/20
        truth = dict(sequence_id=c.SEQUENCE, image_file='images/frame_%05d.jpg'%i,
            run_instance_id='synthetic', obb_valid=True, truth_valid=True,
            frame_truth_audit=dict(passed=True),
            obb_geometry=dict(cx_px=512., cy_px=512., w_px=w, h_px=h, gamma_deg=0.),
            camera_geometry=dict(z_cg_opt_m=20.), pivot_relative=dict(theta_total_deg=float(i)))
        box = [512., 512., w, h, 0., .9]
        truths.append(truth)
        predictions.append(dict(frame_id='frame_%05d'%i, b=box, midpoint=list(box), accepted=True))
    z = c.depth(predictions[0]['b'], intr, geometry, calibration()['parameters'])['z_m']
    oracle = c.error_metrics([z]*count, [20.]*count)
    return manifest, truths, predictions, oracle


class ContractTests(unittest.TestCase):
    def test_source_and_original_calibration_closure(self):
        identity, cal = runner.checked_sources()
        self.assertGreater(len(identity['sources']), 82)
        self.assertEqual(c.sha(ROOT/c.CALIBRATION), c.CAL_SHA)
        self.assertEqual(cal['parameters']['b_m'], 0.)
        self.assertFalse(cal['parameters']['fit_offset'])

    def test_protocol_is_train_only_and_has_no_reliability_or_fitting(self):
        p = c.protocol_document()
        self.assertEqual(p['frame_count'], 981)
        self.assertEqual(p['midpoint']['sigma_cells'], 1.5)
        self.assertEqual(p['midpoint']['epoch'], 3)
        self.assertFalse(p['scope']['parameter_refit'])
        self.assertFalse(p['scope']['reliability_filter'])
        self.assertFalse(p['scope']['fixed_dev_access'])
        self.assertFalse(p['scope']['unknown_access'])

    def test_no_parameter_or_epoch_or_limit_cli(self):
        actions = {s for a in runner.parser()._actions for s in a.option_strings}
        for flag in ('--sigma', '--epoch', '--threshold', '--beta', '--fit-offset', '--limit'):
            self.assertNotIn(flag, actions)

    def test_formula_matches_existing_implementation_for_non_square_intrinsics(self):
        tree = ast.parse((ROOT/'tools/depth/evaluate_detector_obb_depth.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'depth_from_obb_box')
        function.returns = None
        for arg in function.args.args: arg.annotation = None
        namespace = dict(math=math)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), 'original_function', 'exec'), namespace)
        intr = dict(fx=937., fy=1021.)
        geo = dict(long_edge_mean_m=1.775, short_edge_mean_m=.6100409808533983)
        params = calibration()['parameters']
        rng = random.Random(17)
        for i in range(40):
            box = [512., 512., rng.uniform(70, 120), rng.uniform(25, 40), rng.uniform(-4, 4)]
            if i % 2: box[2], box[3] = box[3], box[2]
            old_z, comp = namespace['depth_from_obb_box'](c.canonical(box), intr['fx'], intr['fy'],
                geo['long_edge_mean_m'], geo['short_edge_mean_m'], params)
            new = c.depth(box, intr, geo, params)
            self.assertTrue(math.isclose(new['z_m'], old_z, rel_tol=2e-12))
            self.assertAlmostEqual(new['q_signed'], comp['q_signed'], places=12)

    def test_raw_axis_swap_and_angle_period_are_equivalent(self):
        intr, geo = dict(fx=900., fy=1000.), dict(long_edge_mean_m=1.775, short_edge_mean_m=.61004)
        a = [1., 2., 90., 30., .2, .9]
        b = [1., 2., 30., 90., .2-math.pi/2, .9]
        x, y = c.depth(a, intr, geo, calibration()['parameters']), c.depth(b, intr, geo, calibration()['parameters'])
        self.assertAlmostEqual(x['z_m'], y['z_m'], places=9)
        b[4] += 5*math.pi
        self.assertAlmostEqual(x['z_m'], c.depth(b, intr, geo, calibration()['parameters'])['z_m'], places=9)

    def test_size_sensitivity_includes_ratio_correction(self):
        m, truths, pred, _ = fixture()
        b = pred[0]['b']; intr = m['camera']['intrinsics']; geo = m['obb_reference_geometry']; params = calibration()['parameters']
        start = c.depth(b, intr, geo, params)['z_m']
        changed = list(b); changed[2] *= 1.05; changed[3] *= .95
        got = c.depth(changed, intr, geo, params)
        expected = math.exp(params['beta']*math.log(.95/1.05)**2)/.95
        self.assertAlmostEqual(got['z_m']/start, expected, places=11)
        self.assertLess(got['q_signed'], params['q_signed_min'])

    def test_no_q_clamping_and_overflow_is_explicit(self):
        intr = dict(fx=900., fy=900.); geo = dict(long_edge_mean_m=1.775, short_edge_mean_m=.61004)
        result = c.depth([1., 2., 1e100, 1., 0.], intr, geo, calibration()['parameters'])
        self.assertEqual(result['status'], 'formula_overflow')
        self.assertIsNone(result['z_m'])
        self.assertLess(result['q_signed'], -200.)

    def test_invalid_box_and_offset_are_not_silently_repaired(self):
        m, _, p, _ = fixture()
        for box in ([0.,0.,0.,2.,0.], [0.,0.,math.nan,2.,0.], [0.,0.,2.,-1.,0.]):
            with self.assertRaises(ValueError):
                c.depth(box, m['camera']['intrinsics'], m['obb_reference_geometry'], calibration()['parameters'])
        changed = deepcopy(calibration()['parameters']); changed['b_m'] = .1
        with self.assertRaises(ValueError):
            c.depth(p[0]['b'], m['camera']['intrinsics'], m['obb_reference_geometry'], changed)

    def test_large_finite_depth_rms_does_not_overflow_by_squaring(self):
        metrics = c.error_metrics([1e200, 1e200], [20., 20.])
        self.assertTrue(math.isfinite(metrics['mae_m']))
        self.assertTrue(math.isfinite(metrics['rmse_m']))
        self.assertAlmostEqual(metrics['rmse_m']/1e200, 1.)

    def test_missing_outputs_keep_all_frame_denominators(self):
        m, truths, pred, oracle = fixture()
        pred[1].update(b=None, midpoint=None, accepted=None)
        with patch.object(c, 'COUNT', 2), patch.object(c, 'ORACLE', oracle):
            _, summary = c.evaluate(pred, truths, m, calibration())
        for arm in ('b', 'midpoint'):
            s = summary['groups'][arm]
            self.assertEqual(s['output_frame_count'], 1)
            self.assertEqual(s['output_coverage'], .5)
            self.assertEqual(s['center_correct_rate_output_frames'], 1.)
            self.assertEqual(s['center_correct_coverage_all_frames'], .5)
            self.assertEqual(s['depth_relative_step']['count'], 0)

    def test_out_of_support_finite_depth_is_in_main_metrics(self):
        m, truths, pred, oracle = fixture()
        pred[0]['midpoint'][2] *= 1.05; pred[0]['midpoint'][3] *= .95
        with patch.object(c, 'COUNT', 2), patch.object(c, 'ORACLE', oracle):
            rows, summary = c.evaluate(pred, truths, m, calibration())
        s = summary['groups']['midpoint']
        self.assertFalse(rows[0]['midpoint']['q_in_fit_support'])
        self.assertEqual(s['depth_metrics']['count'], 2)
        self.assertEqual(s['q_out_of_fit_support_count'], 1)
        self.assertEqual(s['q_fit_support_coverage_all_frames'], .5)

    def test_numeric_failure_is_counted_and_not_called_valid_depth(self):
        m, truths, pred, oracle = fixture()
        pred[0]['midpoint'][2] = 1e100; pred[0]['midpoint'][3] = 1.
        with patch.object(c, 'COUNT', 2), patch.object(c, 'ORACLE', oracle):
            _, summary = c.evaluate(pred, truths, m, calibration())
        s = summary['groups']['midpoint']
        self.assertEqual(s['numeric_failure_count'], 1)
        self.assertEqual(s['numeric_depth_coverage'], .5)
        self.assertEqual(summary['paired_numeric_depth_count'], 1)
        self.assertFalse(s['deployment_validity_gate_frozen'])

    def test_presence_scores_and_frame_order_are_protected(self):
        m, truths, pred, oracle = fixture()
        for mutation in ('missing', 'score', 'order'):
            wrong = deepcopy(pred)
            if mutation == 'missing': wrong[0]['midpoint'] = None
            elif mutation == 'score': wrong[0]['midpoint'][5] += .01
            else: wrong.reverse()
            with patch.object(c, 'COUNT', 2), patch.object(c, 'ORACLE', oracle), self.assertRaises(ValueError):
                c.evaluate(wrong, truths, m, calibration())

    def test_nontrain_and_truth_mismatch_rejected(self):
        m, truths, _, _ = fixture()
        with patch.object(c, 'COUNT', 2):
            for split in ('fixed_dev', 'unknown_test'):
                wrong = dict(m, split=split)
                with self.assertRaises(ValueError): c.validate_truth(truths, wrong)
            wrong = deepcopy(truths); wrong[0]['image_file'] = '../other.jpg'
            with self.assertRaises(ValueError): c.validate_truth(wrong, m)
            wrong = deepcopy(truths); wrong[0]['run_instance_id'] = 'another'
            with self.assertRaises(ValueError): c.validate_truth(wrong, m)

    def test_head_identity_rejects_wrong_sigma_epoch_and_context(self):
        payload = dict(protocol='port_geometry_midpoint_sigma_v1', sigma_cells=1.5, epoch=3,
            updates=2706, head_digest=deepcopy(c.HEAD_DIGEST), frozen_b={'identity':'b'}, head_state={},
            context=dict(manifest_sha256=c.CACHE_SHA, detector_state={'identity':'frozen'}))
        c.validate_head_header(payload, {'identity':'b'})
        for key, value in [('sigma_cells',1.), ('epoch',23), ('updates',200), ('head_digest',{})]:
            wrong = dict(payload); wrong[key] = value
            with self.assertRaises(ValueError): c.validate_head_header(wrong, {'identity':'b'})

    def test_weight_locator_uses_only_fixed_sha_and_does_not_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root/'work_dirs/relocated'; folder.mkdir(parents=True)
            good = folder/'head_epoch_03.pth'; good.write_bytes(b'fixed fixture')
            bad = root/'work_dirs/head_epoch_03.pth'; bad.write_bytes(b'other sigma')
            self.assertEqual(c.checked_weight(root, None, good.name, c.sha(good)), good.resolve())
            with self.assertRaises(ValueError): c.checked_weight(root, bad, good.name, c.sha(good))
            with self.assertRaises(FileNotFoundError): c.checked_weight(root, None, 'missing.pth', c.sha(good))

    def test_image_bytes_and_frame_set_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root/c.SEQUENCE/'images'; folder.mkdir(parents=True)
            digest = hashlib.sha256()
            for i in range(2):
                p = folder/('frame_%05d.jpg'%i); p.write_bytes(bytes([i]))
                digest.update(p.name.encode()); digest.update(b'\0'); digest.update(hashlib.sha256(p.read_bytes()).digest())
            with patch.object(c, 'COUNT', 2):
                self.assertEqual(len(c.image_sources(root, digest.hexdigest())), 2)
                p.write_bytes(b'changed')
                with self.assertRaises(ValueError): c.image_sources(root, digest.hexdigest())

    def test_check_stage_loads_no_models_or_dataset_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = runner.parser().parse_args(['--stage','check','--out-dir',str(Path(tmp)/'new')])
            with patch.object(runner, 'checked_dataset', side_effect=AssertionError('data read')), \
                 patch.object(runner, 'load_native', side_effect=AssertionError('model read')):
                runner.run(args)
                with self.assertRaises(FileExistsError): runner.run(args)

    def test_json_never_overwrites_or_serializes_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'a.json'; c.write_new(p, {'value':1})
            with self.assertRaises(FileExistsError): c.write_new(p, {'value':2})
            with self.assertRaises(ValueError): c.write_new(Path(tmp)/'b.json', {'value':math.nan})
            self.assertFalse((Path(tmp)/'b.json').exists())

    def test_historical_report_is_optional_and_cannot_be_legacy(self):
        self.assertEqual(c.historical_reference(None)['status'], 'not_provided')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'old.json'; p.write_text(json.dumps({'coordinate_contract':'legacy_plumb_opt_v1'}))
            with self.assertRaises(ValueError): c.historical_reference(p)

    def test_matching_historical_report_is_explicitly_unpaired(self):
        data = dict(coordinate_contract='raw_opt_v1', sequence_id=c.SEQUENCE, stage='calibration_train_audit',
            calibration_sha256=c.CAL_SHA, parameter_refit_performed=False, unknown_sequence_read=False,
            current=dict(frame_count=c.COUNT, prediction_sha256='old_prediction_fixture',
                truth_obb_oracle_depth_metrics=c.ORACLE,
                prediction_manifest=dict(processed_frame_count=c.COUNT, limit=None, metric_truth_read=False,
                    threshold_tuning_performed=False, checkpoint_sha256='old_weight_fixture')))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'old.json'; p.write_text(json.dumps(data))
            result = c.historical_reference(p)
            self.assertFalse(result['same_run']); self.assertFalse(result['truth_bytes_verified'])
            data['stage'] = 'fixed_dev_evaluation'; p.write_text(json.dumps(data))
            with self.assertRaises(ValueError): c.historical_reference(p)

    def test_prediction_seal_precedes_truth_loading(self):
        source = ast.parse(Path(runner.__file__).read_text())
        run = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        calls = [(n.lineno, getattr(n.func, 'id', '')) for n in ast.walk(run) if isinstance(n, ast.Call)]
        infer_line = next(line for line, name in calls if name == 'native_predictions')
        truth_line = next(line for line, name in calls if name == 'load_truth')
        self.assertLess(infer_line, truth_line)
        # The inherited online API has no truth/domain/sequence argument.
        old = ast.parse((ROOT/'crane_project/tools/eval_port_geometry_midpoint_v1_test.py').read_text())
        capture = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'capture')
        self.assertEqual([a.arg for a in capture.args.args], ['detector','head','image','metas','audit'])


NATIVE = all(importlib.util.find_spec(name) is not None for name in ('torch', 'numpy', 'mmcv', 'mmdet', 'mmrotate'))


@unittest.skipUnless(NATIVE, 'Native Torch/MMCV runtime unavailable locally')
class NativeBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import numpy
        from crane_project.tools import eval_port_geometry_midpoint_v1_test as old
        from crane_project.utils.port_geometry_midpoint_sigma_v1 import SigmaMidpointHead
        cls.t, cls.np, cls.old, cls.Head = torch, numpy, old, SigmaMidpointHead

    def bridge(self, present=True, scaled=False):
        t, np, old = self.t, self.np, self.old
        sx, sy = ((1024/2049, 1024/2048) if scaled else (1.,1.))
        raw = np.asarray([[400.,400.,90.,30.,.2,.9]], dtype=np.float32) if present else np.empty((0,6), dtype=np.float32)
        meta = dict(scale_factor=np.asarray([sx,sy,sx,sy],dtype=np.float32),
                    img_shape=(1024,1024,3), ori_shape=((2048,2049,3) if scaled else (1024,1024,3)),
                    pad_shape=(1024,1024,3), flip=False)
        original = old.g.map_boxes(t.tensor(raw[:,:5]), meta, inverse=True).numpy()
        result = np.column_stack([original, raw[:,5]]).astype(np.float32)
        class Detector(t.nn.Module):
            def __init__(self): super().__init__(); self.marker = t.nn.Parameter(t.zeros(1))
            def extract_feat(self, image): return [image.new_zeros(1,256,128,128)]
            def simple_test_from_features(self, features, metas, rescale=False):
                return [[result.copy() if rescale else raw.copy()]]
        detector = old.g.freeze_detector(Detector())
        head = self.Head(1.5).eval().requires_grad_(False)
        state = old.g.state_digest(head)
        events = []
        prediction = old.capture(detector, head, t.zeros(1,3,1024,1024), [meta], events.append)
        self.assertEqual(state, old.g.state_digest(head))
        self.assertEqual(len(events), 4)
        self.assertEqual(prediction['b'], prediction['midpoint'])
        return prediction

    def test_neutral_head_reuses_native_output_exactly(self): self.bridge()
    def test_empty_output_stays_empty(self): self.assertIsNone(self.bridge(present=False)['b'])
    def test_actual_sx_sy_rounding_is_preserved(self): self.bridge(scaled=True)


if __name__ == '__main__':
    unittest.main()
