import unittest
from types import SimpleNamespace

import torch

from crane_project.tools.pqa_oracle_gate_probe import (
    pqa_oracle_volume_iou,
    validate_args,
)


class TestPqaOracleVolumeIou(unittest.TestCase):

    def setUp(self):
        self.gt = torch.tensor([[100.0, 80.0, 80.0, 20.0, 0.35]])

    def test_identical_box_has_unit_quality(self):
        quality = pqa_oracle_volume_iou(self.gt.clone(), self.gt)
        self.assertTrue(torch.allclose(
            quality, torch.ones_like(quality), atol=1e-6))

    def test_far_box_has_zero_quality(self):
        far = torch.tensor([[400.0, 400.0, 80.0, 20.0, 0.35]])
        quality = pqa_oracle_volume_iou(far, self.gt)
        self.assertEqual(float(quality.item()), 0.0)

    def test_candidate_specific_encoding_discriminates_boxes(self):
        candidates = torch.tensor([
            [100.0, 80.0, 80.0, 20.0, 0.35],
            [112.0, 80.0, 80.0, 20.0, 0.35],
            [100.0, 80.0, 80.0, 20.0, 1.20],
        ])
        quality = pqa_oracle_volume_iou(
            candidates, self.gt, grid_size=33, batch_size=2)
        self.assertGreater(float(quality[0]), float(quality[1]))
        self.assertGreater(float(quality[0]), float(quality[2]))

    def test_cls_times_quality_can_recover_deep_correct_candidate(self):
        candidates = torch.tensor([
            [400.0, 400.0, 80.0, 20.0, 0.35],
            [100.0, 80.0, 80.0, 20.0, 0.35],
        ])
        cls_scores = torch.tensor([0.9, 0.01])
        quality = pqa_oracle_volume_iou(candidates, self.gt)
        selected = int(torch.argmax(cls_scores * quality).item())
        self.assertEqual(selected, 1)

    def test_empty_gt_returns_zero(self):
        candidates = self.gt.clone()
        quality = pqa_oracle_volume_iou(
            candidates, torch.empty((0, 5)))
        self.assertEqual(float(quality.item()), 0.0)

    def test_protocol_guard_rejects_k1_checkpoint(self):
        args = SimpleNamespace(
            pool_size=10000, grid_size=25, quality_batch_size=512,
            riou_thr=0.5, final_score_thr=0.0, seed=0,
            config='crane_project/configs/crane_symeood_k1_brightaug.py',
            checkpoint='work_dirs/crane_symeood_k1/epoch_20.pth',
            split='test', seq='real_seq02', start=137, end=169,
            candidate_source='main', allow_non_brightaug=False)
        with self.assertRaisesRegex(ValueError, 'BrightAug epoch_20'):
            validate_args(args)


if __name__ == '__main__':
    unittest.main()
