"""Meaningful conversion/geometry/selection guards, without trained models."""
import math
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from crane_project.tools.prepare_external_obb_v1 import polygon_label, check_configs, sha, MANIFEST_SHA
from crane_project.tools.export_external_obb_v1 import validate_test_lock
from crane_project.utils.external_obb_io_v1 import yolo_to_common


class ExternalOBBAdaptation(unittest.TestCase):
    def test_rectangular_image_coordinate_roundtrip_and_difficulty_one(self):
        text = '10 20 90 20 90 60 10 60 grab 1\n'
        label = polygon_label(text, 200, 80).split()
        self.assertEqual(label[0], '0')
        original = list(map(float, text.split()[:8]))
        restored = [float(v) * (200 if i % 2 == 0 else 80) for i, v in enumerate(label[1:])]
        np.testing.assert_allclose(restored, original, rtol=0, atol=1e-12)

    def test_converter_rejects_clipping_nonfinite_and_degenerate_polygons(self):
        for text in ('-1 0 10 0 10 10 -1 10 grab 0',
                     'nan 0 10 0 10 10 0 10 grab 0',
                     '0 0 10 10 0 10 10 0 grab 0',
                     '0 0 0 0 10 10 0 10 grab 0'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                polygon_label(text, 100, 100)

    def test_empty_native_yolo_result_keeps_missing_frame(self):
        result = yolo_to_common(np.empty((0, 5)), [], [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (0, 6))

    def test_width_angle_equivalence_and_score_top1(self):
        result = yolo_to_common([[300, 500, 10, 30, 0],
                                 [300, 500, 30, 10, math.pi / 2]], [.2, .9], [0, 0])[0]
        self.assertAlmostEqual(float(result[0, 5]), .9, places=6)
        np.testing.assert_allclose(result[0, :5], result[1, :5], rtol=0, atol=1e-6)
        # Original image pixels remain original pixels, with no second rescale.
        self.assertEqual(tuple(result[0, :2]), (300, 500))

    def test_unexpected_class_and_invalid_geometry_are_not_silently_dropped(self):
        for box, score, cls in (([0, 0, 10, 20, 0], .5, 1),
                                ([0, 0, -1, 20, 0], .5, 0),
                                ([0, 0, 10, 20, 0], float('nan'), 0)):
            with self.assertRaises(ValueError):
                yolo_to_common([box], [score], [cls])

    def test_test_cannot_unlock_without_val_or_with_substituted_weight(self):
        with self.assertRaisesRegex(ValueError, 'prior source-VAL'):
            validate_test_lock({}, 'yolo26', 'weight', {})
        lock = dict(evidence_role='source_val_checkpoint_selection',
                    TEST_used_for_selection=False, model='yolo26', checkpoint_sha256='a')
        with self.assertRaisesRegex(ValueError, 'weight/model'):
            validate_test_lock(lock, 'yolo26', 'b', {})

    def test_complete_val_lock_accepts_and_partial_candidate_set_rejects(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'selection.json'
            result = dict(model='yolo26', configuration={},
                candidate_completed_epochs=[16, 18, 20, 22, 24],
                selected_checkpoint='epoch_24', TEST_used_for_selection=False,
                all_checkpoints={'epoch_%d' % e: dict(checkpoint_sha256='chosen')
                                 for e in (16, 18, 20, 22, 24)})
            path.write_text(json.dumps(result))
            lock = dict(model='yolo26', configuration={}, checkpoint_sha256='chosen',
                        manifest_sha256=MANIFEST_SHA, selection_results=str(path),
                        selection_results_sha256=sha(path),
                        evidence_role='source_val_checkpoint_selection', TEST_used_for_selection=False)
            validate_test_lock(lock, 'yolo26', 'chosen', {})
            del result['all_checkpoints']['epoch_16']
            path.write_text(json.dumps(result))
            lock['selection_results_sha256'] = sha(path)
            with self.assertRaisesRegex(ValueError, 'candidate set'):
                validate_test_lock(lock, 'yolo26', 'chosen', {})

    def test_gaucho_model_remains_official_with_one_class(self):
        value = check_configs()
        self.assertEqual(value['gaucho_model_delta'], ['bbox_head.num_classes: 15 -> 1'])


if __name__ == '__main__':
    unittest.main()
