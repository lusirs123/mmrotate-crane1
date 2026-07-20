import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from crane_project.tools.dark_proxy_preflight import (
    _longest_run,
    evaluate_gate,
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

    def test_longest_run_resets_on_frame_gap(self):
        rows = [
            dict(frame=1, silent=True),
            dict(frame=2, silent=True),
            dict(frame=4, silent=True),
            dict(frame=5, silent=False),
        ]
        self.assertEqual(_longest_run(rows, 'silent'), 2)

    def test_gate_requires_silence_geometry_and_rank(self):
        clean = dict(
            per_k={'10000': {'recall': 0.95}},
            dense_best_riou={'mean': 0.80},
            usable_rank={'median': 2.0},
        )
        degraded = dict(
            silence_rate=0.35,
            longest_silent_run=8,
            per_k={'10000': {'recall': 0.90}},
            dense_best_riou={'mean': 0.72},
            usable_rank={'median': 200.0},
        )
        args = SimpleNamespace(
            min_silence_rate=0.20,
            max_silence_rate=0.60,
            min_silent_run=5,
            min_pool_oracle_recall=0.80,
            min_oracle_retention=0.80,
            min_dense_riou_retention=0.80,
            min_rank_ratio=10.0,
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
            silence_rate=0.70,
            longest_silent_run=20,
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
            min_silence_rate=0.20,
            max_silence_rate=0.60,
            min_silent_run=5,
            min_pool_oracle_recall=0.80,
            min_oracle_retention=0.80,
            min_dense_riou_retention=0.80,
            min_rank_ratio=10.0,
            max_target_silence_gap=0.15,
            min_target_silent_run_ratio=0.25,
            min_target_rank_ratio=0.10,
            max_target_rank_ratio=10.0,
            max_target_pool_recall_gap=0.15,
        )
        result = evaluate_gate(
            degraded, clean, 10000, args, target_reference=target)
        self.assertTrue(result['passed'])
        self.assertIsNotNone(result['target_match'])


if __name__ == '__main__':
    unittest.main()
