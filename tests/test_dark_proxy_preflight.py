import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from crane_project.tools.dark_proxy_preflight import (
    _draw_proxy_preview,
    _longest_run,
    _plateau_rows,
    _variant_name,
    decoded_box_to_original,
    evaluate_gate,
    select_preview_frames,
    validate_data_role,
)
from crane_project.utils.dark_degradation import (
    apply_dark_degradation,
    temporal_strength,
)


class DarkDegradationTest(unittest.TestCase):

    def setUp(self):
        x = np.linspace(20, 240, 64, dtype=np.uint8)
        xx, yy = np.meshgrid(x, x)
        self.image = np.stack([
            xx, yy, np.full((64, 64), 128, dtype=np.uint8)
        ], axis=-1)

    def _apply(self, family, severity=0.9, frame=5):
        return apply_dark_degradation(
            self.image, family=family, sequence='real_seq07', frame=frame,
            start=0, end=10, severity=severity, seed=0,
            profile='constant')

    def test_zero_severity_is_exact_identity(self):
        for family in ('photometric', 'dark_isp'):
            output, meta = self._apply(family, severity=0.0)
            np.testing.assert_array_equal(output, self.image)
            self.assertEqual(meta['strength'], 0.0)

    def test_families_are_deterministic_and_distinct(self):
        outputs = {}
        for family in ('photometric', 'dark_isp'):
            first, meta = self._apply(family)
            second, _ = self._apply(family)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(first.shape, self.image.shape)
            self.assertEqual(first.dtype, np.uint8)
            self.assertLess(float(first.mean()), float(self.image.mean()))
            self.assertTrue(meta['geometry_preserving'])
            outputs[family] = first
        self.assertFalse(np.array_equal(
            outputs['photometric'], outputs['dark_isp']))

    def test_ramp_plateau_is_temporally_coherent(self):
        edge = temporal_strength(0, 0, 100, 0.8, 'ramp-plateau')
        middle = temporal_strength(50, 0, 100, 0.8, 'ramp-plateau')
        neighbor = temporal_strength(51, 0, 100, 0.8, 'ramp-plateau')
        self.assertLess(edge, middle)
        self.assertAlmostEqual(middle, neighbor, places=6)


class ProxyIsolationTest(unittest.TestCase):

    def test_val_is_protocol_source(self):
        with tempfile.TemporaryDirectory() as root:
            policy = validate_data_role(
                root, 'source_val', 'val', 'real_seq07')
            self.assertTrue(policy['protocol_source'])
            self.assertTrue(policy['zero_shot_compliant'])

    def test_train_requires_explicit_non_authorizing_flag(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'requires --split val'):
                validate_data_role(
                    root, 'source_val', 'train', 'real_seq01')
            policy = validate_data_role(
                root, 'source_val', 'train', 'real_seq01',
                allow_train_proxy=True)
            self.assertFalse(policy['protocol_source'])

    def test_target_dev_is_explicitly_allowed_and_labelled(self):
        with tempfile.TemporaryDirectory() as root:
            policy = validate_data_role(
                root, 'target_dev', 'test', 'real_seq02',
                reference_only=True)
            self.assertTrue(policy['eligible_for_model_selection'])
            self.assertTrue(policy['uses_target_labels'])
            self.assertFalse(policy['zero_shot_compliant'])

    def test_role_sequence_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'permits only'):
                validate_data_role(
                    root, 'target_dev', 'test', 'real_seq03',
                    reference_only=True)

    def test_target_holdout_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'confirm-frozen-holdout'):
                validate_data_role(
                    root, 'target_holdout', 'test', 'real_seq03',
                    reference_only=True)
            policy = validate_data_role(
                root, 'target_holdout', 'test', 'real_seq03',
                reference_only=True, confirm_frozen_holdout=True)
            self.assertFalse(policy['eligible_for_model_selection'])

    def test_source_val_cannot_be_renamed_test_data(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'requires --split val'):
                validate_data_role(
                    root, 'source_val', 'test', 'real_seq07')
            with self.assertRaisesRegex(ValueError, 'target sequence'):
                validate_data_role(
                    root, 'source_val', 'train', 'real_seq02',
                    allow_train_proxy=True)


class ProxyGateTest(unittest.TestCase):

    def test_preview_frames_are_sampled_from_ramp_plateau(self):
        selected = select_preview_frames(
            list(range(101)), 3, 'ramp-plateau')
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[1], 50)
        for frame in selected:
            self.assertGreaterEqual(
                temporal_strength(
                    frame, 0, 100, 1.0, 'ramp-plateau'),
                0.95)

    def test_preview_box_mapping_and_overlay(self):
        box = decoded_box_to_original(
            [200.0, 100.0, 80.0, 40.0, 0.3],
            dict(scale_factor=[2.0, 2.0, 2.0, 2.0], flip=False))
        self.assertEqual(box, [100.0, 50.0, 40.0, 20.0, 0.3])
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        row = dict(
            variant='industrial_edges_d0p45_x1p00',
            main_silent=False,
            riou_thr=0.5,
            top1_candidate=dict(
                candidate_index=1, score=0.8, riou=0.0,
                box_ori=[40.0, 40.0, 30.0, 12.0, 0.0]),
            usable_candidate=dict(
                candidate_index=2, score=0.2, riou=0.7, rank=500,
                box_ori=[100.0, 70.0, 24.0, 10.0, 0.2]),
        )
        overlay = _draw_proxy_preview(
            image, dict(cx=100.0, cy=70.0, w=24.0, h=10.0, angle=10.0),
            row)
        self.assertEqual(overlay.shape, image.shape)
        self.assertGreater(int(overlay.sum()), 0)

    def test_longest_run_resets_on_frame_gap(self):
        rows = [
            dict(frame=1, silent=True),
            dict(frame=2, silent=True),
            dict(frame=4, silent=True),
            dict(frame=5, silent=False),
        ]
        self.assertEqual(_longest_run(rows, 'silent'), 2)

    def test_plateau_and_variant_name_use_both_independent_axes(self):
        rows = [
            dict(frame=1, degradation=dict(
                dark_severity=0.45, dark_strength=0.45,
                structure_severity=1.0, structure_strength=0.90)),
            dict(frame=2, degradation=dict(
                dark_severity=0.45, dark_strength=0.45,
                structure_severity=1.0, structure_strength=0.95)),
        ]
        self.assertEqual(
            [row['frame'] for row in _plateau_rows(rows)], [2])
        self.assertEqual(
            _variant_name('industrial_edges', 0.45, 1.0),
            'industrial_edges_d0p45_x1p00')

    def test_gate_requires_silence_geometry_and_rank(self):
        clean = dict(
            per_k={'10000': {'recall': 0.95}},
            dense_best_riou={'mean': 0.80},
            usable_rank={'median': 2.0},
        )
        degraded = dict(
            silence_rate=0.85,
            longest_silent_run=18,
            top1_recall=0.10,
            top1_mcml=20,
            per_k={'10000': {'recall': 0.90}},
            dense_best_riou={'mean': 0.72},
            usable_rank={'median': 2000.0},
        )
        args = SimpleNamespace(
            min_silence_rate=0.79,
            max_silence_rate=1.0,
            max_top1_recall=0.20,
            min_top1_error_run=16,
            min_rank_median=500.0,
            max_rank_median=8000.0,
            min_pool_oracle_recall=0.80,
            min_oracle_retention=0.80,
            min_dense_riou_retention=0.80,
        )
        result = evaluate_gate(degraded, clean, 10000, args)
        self.assertTrue(result['passed'])
        degraded['dense_best_riou']['mean'] = 0.20
        result = evaluate_gate(degraded, clean, 10000, args)
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['dense_riou_retention'])

    def test_target_reference_adds_signature_checks(self):
        clean = dict(
            per_k={'10000': {'recall': 0.95}},
            dense_best_riou={'mean': 0.80},
            usable_rank={'median': 2.0},
        )
        degraded = dict(
            silence_rate=0.85,
            longest_silent_run=20,
            top1_recall=0.10,
            top1_mcml=20,
            per_k={'10000': {'recall': 0.90}},
            dense_best_riou={'mean': 0.72},
            usable_rank={'median': 1000.0},
        )
        target = dict(
            silence_rate=0.75,
            longest_silent_run=30,
            per_k={'10000': {'recall': 0.94}},
            usable_rank={'median': 5000.0},
        )
        args = SimpleNamespace(
            min_silence_rate=0.79,
            max_silence_rate=1.0,
            max_top1_recall=0.20,
            min_top1_error_run=16,
            min_rank_median=500.0,
            max_rank_median=8000.0,
            min_pool_oracle_recall=0.80,
            min_oracle_retention=0.80,
            min_dense_riou_retention=0.80,
        )
        result = evaluate_gate(
            degraded, clean, 10000, args, target_reference=target)
        self.assertTrue(result['passed'])
        self.assertIsNotNone(result['target_match'])
        self.assertTrue(result['target_match']['informational_only'])

    def test_gate_rejects_good_top1_or_geometry_collapse(self):
        clean = dict(
            per_k={'10000': {'recall': 0.95}},
            dense_best_riou={'mean': 0.80},
            usable_rank={'median': 1.0},
        )
        degraded = dict(
            silence_rate=0.90,
            longest_silent_run=20,
            top1_recall=0.40,
            top1_mcml=20,
            per_k={'10000': {'recall': 0.90}},
            dense_best_riou={'mean': 0.72},
            usable_rank={'median': 2500.0},
        )
        args = SimpleNamespace(
            min_silence_rate=0.79,
            max_silence_rate=1.0,
            max_top1_recall=0.20,
            min_top1_error_run=16,
            min_rank_median=500.0,
            max_rank_median=8000.0,
            min_pool_oracle_recall=0.80,
            min_oracle_retention=0.80,
            min_dense_riou_retention=0.80,
        )
        result = evaluate_gate(degraded, clean, 10000, args)
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['top1_recall'])
        degraded['top1_recall'] = 0.20
        result = evaluate_gate(degraded, clean, 10000, args)
        self.assertFalse(result['checks']['top1_recall'])
        degraded['top1_recall'] = 0.10
        degraded['per_k']['10000']['recall'] = 0.30
        result = evaluate_gate(degraded, clean, 10000, args)
        self.assertFalse(result['checks']['pool_oracle_recall'])


if __name__ == '__main__':
    unittest.main()
