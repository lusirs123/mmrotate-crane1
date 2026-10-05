"""Focused zero-update diagnosis tests (stdlib + optional original torch geometry).

Run: python -m unittest discover -s tests -p test_port_geometry_midpoint_edge_feasibility_v1.py -v
No image, checkpoint, real ROI shard, TEST or training is used by these tests.
"""
from contextlib import redirect_stderr
from copy import deepcopy
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from crane_project.utils import port_geometry_midpoint_edge_feasibility_v1 as g
from crane_project.tools import diagnose_port_geometry_midpoint_edge_feasibility_v1 as tool


def boxes(w=100., h=50., theta=0., xy=(1., 1.), center=(500., 400.)):
    b = [center[0], center[1], w, h, theta, .75]
    bm = [b[0]*xy[0], b[1]*xy[1], w*xy[0], h*xy[1], theta]
    return b, bm, list(xy)


def diagnose(b, bm, xy, gt=None, m=None, **kw):
    return g.diagnose_frame(b, m or b, gt or b[:5], bm, xy, **kw)


class GeometryTests(unittest.TestCase):
    def test_exact_gt_outside_b_but_joint10_reachable(self):
        b, bm, xy = boxes()
        result = diagnose(b, bm, xy, gt=[500., 400., 130., 50., 0.])
        self.assertFalse(result['exact_sorted_gt_size_inside_b_bound'])
        self.assertEqual(result['full_intersection']['width_interval_px'], [117., 125.])
        self.assertTrue(result['stages']['b_roi_and_order'])
        self.assertTrue(result['witness']['nominal']['passed'])
        self.assertEqual(result['constraint_status'], 'continuous_witness_only')
        self.assertIsNone(result['delivered_size_reachable'])

    def test_bound_impossibility_and_reasons(self):
        b, bm, xy = boxes()
        result = diagnose(b, bm, xy, gt=[500., 400., 160., 80., 0.])
        self.assertFalse(result['stages']['b_bounds_only'])
        self.assertEqual(set(result['full_intersection']['failed_constraints']),
            {'raw_width_interval_empty', 'raw_height_interval_empty'})
        self.assertEqual(result['constraint_status'], 'nominal_infeasible')
        self.assertFalse(result['delivered_size_reachable'])

    def test_roi_adds_a_real_constraint(self):
        b, bm, xy = boxes()
        m = [500., 414.9, 100., 40., 0., .75]
        result = diagnose(b, bm, xy, gt=[500., 400., 100., 55., 0.], m=m)
        self.assertTrue(result['stages']['b_bounds_and_order'])
        self.assertFalse(result['stages']['b_roi_and_order'])
        self.assertLess(result['roi']['raw_height_upper_px'], 49.5)
        self.assertFalse(result['delivered_size_reachable'])

    def test_baseline_bypass_survives_rebuilt_roi_failure(self):
        b, bm, xy = boxes()
        m = [500., 414.9, 100., 55., 0., .75]
        result = diagnose(b, bm, xy, gt=[500., 400., 100., 55., 0.], m=m)
        self.assertFalse(result['stages']['b_roi_and_order'])
        self.assertTrue(result['baseline_joint10'])
        self.assertTrue(result['baseline_bypass_reachable'])
        self.assertTrue(result['delivered_size_reachable'])
        self.assertFalse(result['baseline_rebuilt_midpoints_pass_nominal_constraints']['checks']['roi_midpoints'])

    def test_midpoints_are_not_a_corner_constraint(self):
        b, bm, xy = boxes(center=(0., 0.))
        m = [0., 0., 124., 62., math.pi/18, .75]
        frame = g.make_frame(b, bm, xy)
        self.assertTrue(g.check_sizes(m[2:4], b, m, m[2:4], frame)['passed'])
        c, s = math.cos(m[4]), math.sin(m[4])
        corner = (62*c-31*s, 62*s+31*c)
        self.assertGreater(abs(g.to_roi(corner, frame)[1]), .5+1e-5)

    def test_b_order_and_model_order_are_separate(self):
        b, bm, xy = boxes(w=100., h=100.05, theta=.34, xy=(1.01, .99))
        result = diagnose(b, bm, xy)
        self.assertTrue(result['frame']['original_b_swap'])
        self.assertFalse(result['frame']['model_b_swap'])
        self.assertTrue(result['stages']['b_roi_and_order'])

    def test_restoration_rejects_mean_scale_and_wrong_raw_angle(self):
        b, bm, xy = boxes(theta=.3, xy=(.5002, .4998))
        bad = [v*.5 for v in b[:4]]+[b[4]]
        with self.assertRaisesRegex(ValueError, 'restoration'):
            g.make_frame(b, bad, xy)
        wrong = list(bm); wrong[4] += math.pi/2
        with self.assertRaisesRegex(ValueError, 'angle'):
            g.make_frame(b, wrong, xy)

    def test_equivalent_raw_swaps_keep_geometric_roi_caps(self):
        b, bm, xy = boxes(theta=.31, xy=(.503, .498))
        frame = g.make_frame(b, bm, xy)
        swapped = [b[0], b[1], b[3], b[2], b[4]+math.pi/2, b[5]]
        cap = g.roi_caps(b, frame)
        other = g.roi_caps(swapped, frame)
        self.assertAlmostEqual(cap['raw_width_upper_px'], other['raw_height_upper_px'], places=10)
        self.assertAlmostEqual(cap['raw_height_upper_px'], other['raw_width_upper_px'], places=10)
        self.assertAlmostEqual(g.wrap_pi(.32+math.pi), .32)

    def test_context_minimum_and_center_outside(self):
        b, bm, xy = boxes(w=4., h=2., xy=(.5, .51), center=(0., 0.))
        frame = g.make_frame(b, bm, xy)
        self.assertEqual(frame['context_sides_model'], [16., 16.])
        m = deepcopy(b); m[0] = 100.
        caps = g.roi_caps(m, frame)
        self.assertEqual((caps['raw_width_upper_px'], caps['raw_height_upper_px']), (0., 0.))

    def test_closed_vs_strict_order_boundary(self):
        bounds = [[10., 10.], [10., 10.]]
        self.assertTrue(g.solve_intervals(bounds, False)['feasible'])
        strict = g.solve_intervals(bounds, True)
        self.assertFalse(strict['feasible'])
        self.assertIn('strict_order_unavailable', strict['failed_constraints'])
        self.assertTrue(strict['numeric_boundary'])
        self.assertFalse(g.solve_intervals([[9., 10.], [11., 12.]], False)['feasible'])

    def test_witness_construction_preserves_strict_order(self):
        bounds = [[10., 20.], [10., 20.]]
        candidates = g.certificate_candidates(bounds, True, [0., 0., 16., 17., 0., 1.])
        self.assertLess(candidates[0][0], candidates[0][1])
        self.assertLessEqual(len(candidates), 3)

    def test_exact_square_is_frozen_without_callback(self):
        b, bm, xy = boxes(w=100., h=100.)
        callback = unittest.mock.Mock()
        result = diagnose(b, bm, xy, gt=[500., 400., 115., 100., 0.], verify_witness=callback)
        callback.assert_not_called()
        self.assertTrue(result['stages']['b_roi_and_order'])
        self.assertEqual(result['constraint_status'], 'frozen_square_baseline')
        self.assertFalse(result['delivered_size_reachable'])

    def test_runtime_witness_required_and_failure_not_infeasibility(self):
        b, bm, xy = boxes()
        gt = [500., 400., 130., 50., 0.]
        for callback in (lambda p, q: None, lambda p, q: dict(passed=False)):
            result = diagnose(b, bm, xy, gt=gt, verify_witness=callback)
            self.assertEqual(result['constraint_status'], 'numeric_unresolved')
            self.assertIsNone(result['delivered_size_reachable'])
        result = diagnose(b, bm, xy, gt=gt, verify_witness=lambda p, q: dict(passed=True))
        self.assertEqual(result['constraint_status'], 'runtime_witness_verified')
        self.assertTrue(result['delivered_size_reachable'])

    def test_failed_tiny_boundary_does_not_widen_target(self):
        b, bm, xy = boxes()
        gt = [500., 400., (125.+1e-10)/.9, 50., 0.]
        result = diagnose(b, bm, xy, gt=gt)
        self.assertFalse(result['stages']['b_roi_and_order'])
        self.assertEqual(result['constraint_status'], 'numeric_unresolved')
        self.assertIsNone(result['delivered_size_reachable'])

    def test_goal_assignment_and_training_conflict_are_reported(self):
        b, bm, xy = boxes(w=50., h=100.)
        result = diagnose(b, bm, xy, matched_gt_raw=[100., 50.])
        self.assertEqual(result['goal_raw_width_height'], [50., 100.])
        self.assertFalse(result['matched_training_target']['agrees_with_metric_axis_target'])
        self.assertFalse(result['matched_training_target']['keeps_m_raw_order'])
        self.assertEqual(result['matched_training_target']['raw_width_height'], [100., 50.])

    def test_no_output_and_wrong_frame_identity(self):
        gt = [500., 400., 100., 50., 0.]
        result = g.diagnose_frame(None, None, gt)
        self.assertFalse(result['output'])
        self.assertFalse(result['delivered_size_reachable'])
        with self.assertRaisesRegex(ValueError, 'identity'):
            g.diagnose_frame(gt+[.5], None, gt)

    def test_source_guard_violation_is_an_error(self):
        b, bm, xy = boxes()
        for index, value in ((5, .5), (0, 550.), (4, .3), (2, 126.)):
            m = deepcopy(b); m[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                diagnose(b, bm, xy, m=m)
        for index, value in ((2, 0.), (0, float('nan')), (4, float('inf'))):
            bad = deepcopy(b); bad[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                diagnose(bad, bm, xy)

    def test_native_source_guard_basis_is_explicit_and_never_loosens_joint10(self):
        b, bm, xy = boxes(w=200., h=100., center=(0., 0.))
        m = deepcopy(b); m[0] = 30.000001907348633
        with self.assertRaisesRegex(ValueError, 'center_bound'):
            diagnose(b, bm, xy, m=m)
        native = dict(center_bound=True, angle_bound=True, edge_bound=True)
        result = diagnose(b, bm, xy, m=m, source_guard_checks=native)
        self.assertEqual(result['source_guards']['numeric_disagreements'], ['center_bound'])
        self.assertTrue(result['baseline_joint10'])
        self.assertFalse(g.joint_correct([110.00001, 50.], [100., 50.]))
        with self.assertRaisesRegex(ValueError, 'original B guard'):
            diagnose(b, bm, xy, m=m, source_guard_checks=dict(native, center_bound=False))

    def test_randomized_affine_caps_against_direct_four_midpoints(self):
        rng = random.Random(1703)
        for _ in range(1200):
            b, bm, xy = boxes(w=rng.uniform(4., 160.), h=rng.uniform(3., 110.),
                theta=rng.uniform(-1.55, 1.55), xy=(rng.uniform(.45, 1.1), rng.uniform(.45, 1.1)))
            frame = g.make_frame(b, bm, xy)
            m = deepcopy(b)
            m[0] += rng.uniform(-.2, .2)*min(b[2:4])
            m[1] += rng.uniform(-.2, .2)*min(b[2:4])
            m[4] += rng.uniform(-.15, .15)
            caps = g.roi_caps(m, frame)
            pair = [rng.uniform(.65, 1.5)*v for v in b[2:4]]
            expected = pair[0] <= caps['raw_width_upper_px'] and pair[1] <= caps['raw_height_upper_px']
            actual = g.check_sizes(pair, b, m, pair, frame)['checks']['roi_midpoints']
            self.assertEqual(expected, actual)


class ContractTests(unittest.TestCase):
    def test_sealed_source_contract_and_fixed_roles(self):
        identity = tool.checked_sources()
        self.assertEqual(identity['training_identity']['parent_identity']['frozen_b']['selected_epoch'], 'epoch_24')
        self.assertEqual(len(tool.sample_roles()), 64)
        self.assertEqual(tool.protocol_document()['stages'], ['check', 'diagnose'])
        self.assertFalse(tool.protocol_document()['scope']['test_access'])
        self.assertFalse(tool.protocol_document()['scope']['independent_size_head_implemented'])

    def test_cache_resolution_old_organized_symlink_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            new = root/'work_dirs/port_results/geometry'/tool.CACHE_BASENAME
            new.mkdir(parents=True)
            (new/'cache_manifest.json').write_text('{}')
            (new.parent.parent/'INDEX.md').write_text('- port_results/geometry/'+tool.CACHE_BASENAME+'\n')
            old = root/'work_dirs'/tool.CACHE_BASENAME
            old.symlink_to(new, target_is_directory=True)
            self.assertEqual(tool.resolve_cache(root=root), new.resolve())
            old.unlink(); old.mkdir(); (old/'cache_manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, '--cache-dir'):
                tool.resolve_cache(root=root)
            self.assertEqual(tool.resolve_cache(str(old), root=root), old.resolve())

    def test_missing_cache_stops_without_rebuilding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'INDEX.md'):
                tool.resolve_cache(root=root)
            self.assertEqual(list(root.iterdir()), [])

    def test_torch_load_compatibility(self):
        class Old:
            @staticmethod
            def load(path, map_location):
                return dict(path=path, device=map_location)
        class New:
            @staticmethod
            def load(path, map_location, weights_only=True):
                return dict(path=path, device=map_location, weights_only=weights_only)
        self.assertEqual(tool.torch_load(Old, Path('sealed.pt'))['device'], 'cpu')
        self.assertFalse(tool.torch_load(New, Path('sealed.pt'))['weights_only'])

    def test_selected_artifact_and_selection_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/tool.FIXED['path']).write_bytes(b'synthetic sealed weight fixture')
            fixed = dict(tool.FIXED, sha256=tool.sha(root/tool.FIXED['path']))
            identity = dict(training_identity={'sealed': 'fixture'})
            selection = dict(protocol='port_geometry_midpoint_sigma_v1', sigma_cells=1.5,
                split='val', selected_checkpoint=fixed, cache_manifest_sha256=tool.CACHE_SHA,
                proof=dict(identity=identity['training_identity'], cache_manifest_sha256=tool.CACHE_SHA),
                test_access=False, selection_on_test=False, automatic_promotion=False)
            tool.write(root/'selection.json', selection)
            refs = [dict(image='real_seq07_'+str(i).zfill(5), sequence='real_seq07', scale=1.) for i in (0, 1)]
            (root/'val_epoch_03.rows.jsonl').write_text('\n'.join(json.dumps(r) for r in refs)+'\n')
            names = ['selection.json', fixed['path'], 'val_epoch_03.rows.jsonl']
            def reindex():
                (root/'artifacts.json').write_text(json.dumps(dict(protocol='port_geometry_midpoint_sigma_v1', files={n: tool.sha(root/n) for n in names})))
            reindex()
            with patch.object(tool, 'FIXED', fixed), patch.object(tool, 'VAL_COUNTS', {'real_seq07': 2}):
                self.assertEqual(len(tool.checked_selection(root, identity)[1]), 2)
                selection['test_access'] = True
                (root/'selection.json').write_text(json.dumps(selection)); reindex()
                with self.assertRaisesRegex(ValueError, 'identity'):
                    tool.checked_selection(root, identity)
                selection['test_access'] = False
                (root/'selection.json').write_text(json.dumps(selection)); reindex()
                (root/fixed['path']).write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'artifact SHA'):
                    tool.checked_selection(root, identity)

    def test_failed_run_is_indexed_and_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'new'
            args = tool.parser().parse_args(['--stage', 'diagnose', '--out-dir', str(out)])
            with patch.object(tool, 'checked_sources', side_effect=ValueError('fixture mismatch')):
                with self.assertRaisesRegex(ValueError, 'fixture mismatch'):
                    tool.run(args)
            completion = tool.read(out/'completion.json')
            self.assertEqual(completion['status'], 'EDGE_FEASIBILITY_FAILED')
            index = tool.read(out/'artifacts.json')
            self.assertEqual(index['files']['completion.json'], tool.sha(out/'completion.json'))
            before = tool.sha(out/'completion.json')
            with self.assertRaises(FileExistsError):
                tool.run(args)
            self.assertEqual(before, tool.sha(out/'completion.json'))

    def test_test_stage_is_absent(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tool.parser().parse_args(['--stage', 'test'])

    def test_denominators_no_output_and_baseline_bypass(self):
        b, bm, xy = boxes()
        def row(diag, gt, m):
            return dict(diagnosis=diag, gt=gt, midpoint=m, domain='real', sequence='real_seq07')
        good = diagnose(b, bm, xy)
        bad = diagnose(b, bm, xy, gt=[500., 400., 150., 50., 0.])
        absent = g.diagnose_frame(None, None, b[:5])
        result = tool.summarize([row(good, b[:5], b), row(bad, [500., 400., 150., 50., 0.], b), row(absent, b[:5], None)])
        self.assertEqual(result['output_coverage'], dict(numerator=2, denominator=3, fraction=2/3))
        self.assertEqual(result['center_correct_conditional']['denominator'], 2)
        self.assertEqual(result['center_correct_full_frame']['denominator'], 3)
        self.assertEqual(result['joint_size10_full_frame']['numerator'], 1)
        self.assertEqual(result['baseline_size']['long']['n'], 2)
        self.assertEqual(result['no_output'], 1)
        self.assertEqual(tool.summarize([])['center_correct_conditional']['fraction'], None)

    def test_reference_replay_preserves_axis_and_joint_count(self):
        b, bm, xy = boxes(w=100., h=99.99999)
        diag = diagnose(b, bm, xy)
        row = dict(image='real_seq07_00000', sequence='real_seq07', domain='real', frame_id=0,
            scale=1., gt=b[:5], b=b, midpoint=b, diagnosis=diag)
        ref = deepcopy(row)
        tool.compare_reference(row, ref)
        ref['midpoint'][2] = 99.99998
        with self.assertRaisesRegex(ValueError, 'canonical axis'):
            tool.compare_reference(row, ref)

    def test_linear_p95(self):
        self.assertEqual(tool.percentile([], .95), None)
        self.assertAlmostEqual(tool.percentile([0., 1., 2.], .95), 1.9)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'torch unavailable: run native checks in mmrotljj')
class NativeTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from crane_project.utils import port_geometry_midpoint_sigma_v1 as model
        cls.torch, cls.model, cls.original = torch, model, model.original

    def tensors(self, raw=None, xy=(1., 1.)):
        torch = self.torch
        raw = raw or [500., 400., 100., 50., .25, .75]
        b = torch.tensor([raw], dtype=torch.float32)
        bm = b[:, :5].clone(); scale = b.new_tensor([xy])
        bm[:, :4] *= scale[:, [0, 1, 0, 1]]
        return b, bm, scale

    def test_native_log_exp_witness_and_independent_dtype_recheck(self):
        torch, original = self.torch, self.original
        b, bm, xy = self.tensors()
        _, _, _, sides, rotation = original.frame(b, bm, xy)
        frame = g.make_frame(b[0].tolist(), bm[0].tolist(), xy[0].tolist(),
            dict(context_sides_model=sides[0].tolist(), rotation_model=rotation[0].tolist()))
        with torch.no_grad():
            result = tool.tensor_witness(torch, original, b, bm, xy, b, [121., 50.], [130., 50.], frame)
        self.assertTrue(result['passed'])
        self.assertEqual(result['dtype'], 'torch.float32')
        with torch.no_grad():
            crossed = tool.tensor_witness(torch, original, b, bm, xy, b, [80., 90.], [90., 80.], frame)
        self.assertFalse(crossed['passed'])
        self.assertFalse(crossed['checks']['raw_order'])

    def test_native_mixed_swaps_and_target_points(self):
        torch = self.torch
        b, bm, xy = self.tensors([500., 400., 100., 100.05, .34, .75], xy=(1.01, .99))
        head = self.model.SigmaMidpointHead(1.5).eval()
        for p in head.parameters():
            p.requires_grad_(False)
        row = dict(roi=b.new_zeros(1, 256, 9, 9), support=b.new_ones(1, 1, 9, 9),
            boxes_original=b, boxes_model=bm, scale_xy=xy, gt_original=b[:, :5].clone(),
            image='real_seq07_00000', sequence='real_seq07', domain='real', frame_id=0,
            split='val', role='val', scale=1., eligible=False)
        before = tool.state_digest(head)
        with torch.no_grad():
            result = tool.diagnose_record(row, head, torch, self.original, torch.device('cpu'))
        self.assertEqual(result['midpoint'], b[0].tolist())
        self.assertTrue(result['diagnosis']['matched_training_target']['original_model_b_swap_differ'])
        self.assertTrue(result['diagnosis']['matched_training_target']['agrees_with_metric_axis_target'])
        self.assertEqual(before, tool.state_digest(head))
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_native_no_output_and_cached_eligibility(self):
        torch = self.torch
        head = self.model.SigmaMidpointHead(1.5).eval()
        for p in head.parameters():
            p.requires_grad_(False)
        row = dict(roi=torch.empty(0, 256, 9, 9), support=torch.empty(0, 1, 9, 9),
            boxes_original=torch.empty(0, 6), boxes_model=torch.empty(0, 5), scale_xy=torch.empty(0, 2),
            gt_original=torch.tensor([[500., 400., 100., 50., .25]], dtype=torch.float32),
            image='real_seq07_00000', sequence='real_seq07', domain='real', frame_id=0,
            split='val', role='val', scale=1., eligible=False)
        tool.checked_record(row, 'val_s1', torch, self.original)
        with torch.no_grad():
            result = tool.diagnose_record(row, head, torch, self.original, torch.device('cpu'))
        self.assertFalse(result['diagnosis']['output'])
        row['eligible'] = True
        with self.assertRaisesRegex(ValueError, 'eligibility'):
            tool.checked_record(row, 'val_s1', torch, self.original)

    def test_native_affine_with_unequal_scales_swaps_and_minimum_context(self):
        torch, original = self.torch, self.original
        cases = [(100., 50., .31, (.503, .498)), (50., 100., -.67, (.498, .503)),
                 (100., 100.05, .34, (1.01, .99)), (4., 2., 1.55, (.5, .51))]
        for w, h, theta, scale in cases:
            b, bm, xy = self.tensors([500., 400., w, h, theta, .75], xy=scale)
            m = b.clone(); m[:, 0] += .1*min(w, h); m[:, 1] -= .1*min(w, h)
            m[:, 4] += .08; m[:, 2:4] *= m.new_tensor([1.1, .92])
            _, _, _, sides, rotation = original.frame(b, bm, xy)
            frame = g.make_frame(b[0].tolist(), bm[0].tolist(), xy[0].tolist(),
                dict(context_sides_model=sides[0].tolist(), rotation_model=rotation[0].tolist()))
            nominal = g.check_sizes(m[0, 2:4].tolist(), b[0].tolist(), m[0].tolist(), m[0, 2:4].tolist(), frame)
            native = original.original_to_roi(original.box_midpoints(m[:, :5]), b, bm, xy)[0].tolist()
            for point in nominal['points_roi']:
                self.assertLess(min(math.hypot(point[0]-p[0], point[1]-p[1]) for p in native), 1e-5)
            self.assertEqual(nominal['checks']['roi_midpoints'], all(abs(v) <= .5+1e-5 for p in native for v in p))

    def test_native_source_guard_accepts_original_float32_boundary(self):
        torch, original = self.torch, self.original
        b, bm, xy = self.tensors([0., 0., 200., 100., 0., .75])
        m = b.clone(); m[:, 0] = 30.000001907348633
        native = tool.native_source_guards(torch, original, b, m)
        self.assertTrue(native['center_bound'])
        result = g.diagnose_frame(b[0].tolist(), m[0].tolist(), b[0, :5].tolist(),
            bm[0].tolist(), xy[0].tolist(), source_guard_checks=native)
        self.assertEqual(result['source_guards']['numeric_disagreements'], ['center_bound'])


if __name__ == '__main__':
    unittest.main()
