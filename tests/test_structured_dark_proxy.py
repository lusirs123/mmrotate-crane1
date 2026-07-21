import os
import tempfile
import unittest

import cv2
import numpy as np

from crane_project.utils.structured_dark_proxy import (
    apply_industrial_edge_interference,
    apply_source_background_interference,
    apply_structured_dark_proxy,
    build_background_patch_library,
    target_exclusion_mask,
)


class StructuredDarkProxyTest(unittest.TestCase):

    def setUp(self):
        x = np.linspace(25, 225, 192, dtype=np.uint8)
        xx, yy = np.meshgrid(x, x)
        texture = ((xx.astype(np.int16) + yy.astype(np.int16)) % 40)
        self.image = np.stack([
            xx,
            yy,
            np.clip(90 + texture, 0, 255).astype(np.uint8),
        ], axis=-1)
        self.gts = [dict(
            cx=96.0, cy=96.0, w=48.0, h=24.0,
            angle=15.0, cls='grab')]

    def _library(self):
        patch = self.image[8:72, 10:106].copy()
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return dict(
            patches=[dict(
                image=patch,
                stats=dict(
                    mean=float(gray.mean()), std=float(gray.std()),
                    gradient=float(np.sqrt(gx * gx + gy * gy).mean())),
                source_image='train/images/real_seq01_00000.jpg',
                source_annotation='train/annfiles/real_seq01_00000.txt',
                source_rect=[10, 8, 96, 64])],
            manifest=dict(sha256='fixture', split='train'))

    def test_industrial_layer_is_deterministic_and_avoids_target(self):
        first, meta = apply_industrial_edge_interference(
            self.image, self.gts, 'real_seq07', 10, 0, 30, 0.75, seed=0)
        second, _ = apply_industrial_edge_interference(
            self.image, self.gts, 'real_seq07', 10, 0, 30, 0.75, seed=0)
        np.testing.assert_array_equal(first, second)
        exclusion = target_exclusion_mask(self.image.shape, self.gts)
        np.testing.assert_array_equal(
            first[exclusion > 0], self.image[exclusion > 0])
        self.assertGreater(meta['structure_pixels'], 0)

    def test_source_background_layer_is_deterministic_and_avoids_target(self):
        library = self._library()
        first, meta = apply_source_background_interference(
            self.image, self.gts, library, 'real_seq07',
            10, 0, 30, 0.75, seed=0)
        second, _ = apply_source_background_interference(
            self.image, self.gts, library, 'real_seq07',
            10, 0, 30, 0.75, seed=0)
        np.testing.assert_array_equal(first, second)
        exclusion = target_exclusion_mask(self.image.shape, self.gts)
        np.testing.assert_array_equal(
            first[exclusion > 0], self.image[exclusion > 0])
        self.assertEqual(meta['library_sha256'], 'fixture')

    def test_source_background_state_keeps_donor_and_motion_continuous(self):
        library = self._library()
        state = {}
        _, first = apply_source_background_interference(
            self.image, self.gts, library, 'real_seq07',
            10, 0, 100, 0.75, seed=0, sequence_state=state)
        _, second = apply_source_background_interference(
            self.image, self.gts, library, 'real_seq07',
            11, 0, 100, 0.75, seed=0, sequence_state=state)
        self.assertEqual(
            [item['source_image'] for item in first['placements']],
            [item['source_image'] for item in second['placements']])
        for left, right in zip(first['placements'], second['placements']):
            lx, ly = left['destination_rect'][:2]
            rx, ry = right['destination_rect'][:2]
            self.assertLessEqual(abs(lx - rx), 2)
            self.assertLessEqual(abs(ly - ry), 2)

    def test_combined_proxy_keeps_shape_and_records_both_layers(self):
        output, meta = apply_structured_dark_proxy(
            self.image, self.gts, family='industrial_edges',
            sequence='real_seq07', frame=10, start=0, end=30,
            dark_severity=0.45, structure_severity=0.75,
            seed=0, dark_family='photometric',
            temporal_profile='constant')
        self.assertEqual(output.shape, self.image.shape)
        self.assertEqual(output.dtype, np.uint8)
        self.assertLess(float(output.mean()), float(self.image.mean()))
        self.assertFalse(meta['target_geometry_modified'])
        self.assertEqual(meta['dark_strength'], 0.45)
        self.assertEqual(meta['structure_strength'], 0.75)
        self.assertIn('darkening', meta)
        self.assertIn('structure', meta)

    def test_dark_and_structure_strengths_are_independent(self):
        low_structure, low_meta = apply_structured_dark_proxy(
            self.image, self.gts, family='industrial_edges',
            sequence='real_seq07', frame=10, start=0, end=30,
            dark_severity=0.45, structure_severity=0.35,
            seed=0, temporal_profile='constant')
        high_structure, high_meta = apply_structured_dark_proxy(
            self.image, self.gts, family='industrial_edges',
            sequence='real_seq07', frame=10, start=0, end=30,
            dark_severity=0.45, structure_severity=1.0,
            seed=0, temporal_profile='constant')

        exclusion = target_exclusion_mask(self.image.shape, self.gts)
        np.testing.assert_array_equal(
            low_structure[exclusion > 0], high_structure[exclusion > 0])
        self.assertFalse(np.array_equal(
            low_structure[exclusion == 0], high_structure[exclusion == 0]))
        self.assertEqual(low_meta['dark_strength'], high_meta['dark_strength'])
        self.assertNotEqual(
            low_meta['structure_strength'], high_meta['structure_strength'])
        self.assertGreater(
            high_meta['structure']['structure_pixels'],
            low_meta['structure']['structure_pixels'])

        _, darker_meta = apply_structured_dark_proxy(
            self.image, self.gts, family='industrial_edges',
            sequence='real_seq07', frame=10, start=0, end=30,
            dark_severity=0.65, structure_severity=1.0,
            seed=0, temporal_profile='constant')
        self.assertNotEqual(
            high_meta['dark_strength'], darker_meta['dark_strength'])
        self.assertEqual(
            high_meta['structure_strength'],
            darker_meta['structure_strength'])
        self.assertEqual(
            high_meta['structure']['placements'],
            darker_meta['structure']['placements'])

    def test_ramp_changes_alpha_without_changing_layout_count(self):
        _, edge = apply_structured_dark_proxy(
            self.image, self.gts, family='industrial_edges',
            sequence='real_seq07', frame=0, start=0, end=100,
            dark_severity=0.45, structure_severity=1.0,
            seed=0, temporal_profile='ramp-plateau')
        _, plateau = apply_structured_dark_proxy(
            self.image, self.gts, family='industrial_edges',
            sequence='real_seq07', frame=50, start=0, end=100,
            dark_severity=0.45, structure_severity=1.0,
            seed=0, temporal_profile='ramp-plateau')
        self.assertLess(
            edge['structure_strength'], plateau['structure_strength'])
        self.assertEqual(
            len(edge['structure']['placements']),
            len(plateau['structure']['placements']))
        self.assertEqual(edge['structure']['layout_strength'], 1.0)
        self.assertEqual(plateau['structure']['layout_strength'], 1.0)

    def test_background_library_uses_only_train_non_target_regions(self):
        with tempfile.TemporaryDirectory() as root:
            image_dir = os.path.join(root, 'train', 'images')
            ann_dir = os.path.join(root, 'train', 'annfiles')
            os.makedirs(image_dir)
            os.makedirs(ann_dir)
            image = np.tile(self.image, (3, 3, 1))
            image_path = os.path.join(
                image_dir, 'real_seq01_00000.jpg')
            ann_path = os.path.join(
                ann_dir, 'real_seq01_00000.txt')
            self.assertTrue(cv2.imwrite(image_path, image))
            with open(ann_path, 'w') as handle:
                handle.write(
                    '250 250 350 250 350 350 250 350 grab 0\n')
            library = build_background_patch_library(
                root, max_patches=4, seed=0,
                min_texture_std=1.0, min_gradient=0.1)
            self.assertGreater(len(library['patches']), 0)
            self.assertEqual(library['manifest']['split'], 'train')
            for donor in library['manifest']['donors']:
                self.assertTrue(donor['source_image'].startswith(
                    'train/images/'))
                self.assertTrue(donor['source_annotation'].startswith(
                    'train/annfiles/'))


if __name__ == '__main__':
    unittest.main()
