"""The grouped AP readback must retain missing-output GT in recall denominators."""
import unittest
import numpy as np
from crane_project.tools.analyze_symeood_r4_midpoint_leavevideo_v1 import ap
from crane_project.tools.eval_port_geometry_ap_v1 import ap11


class AnalysisChecks(unittest.TestCase):
    def test_ap75_and_missing_frame_denominator(self):
        rows = [dict(midpoint=[0, 0, 20, 10, 0, score] if score is not None else None,
                     metrics=dict(midpoint=dict(riou=iou)))
                for score, iou in [(.9, .9), (.8, .6), (.7, .9), (None, 0.)]]
        result = ap(rows, 'midpoint', np, ap11)
        self.assertEqual((result['frames'], result['outputs'], result['TP75']), (4, 3, 2))
        self.assertAlmostEqual(result['AP75'], 100*5/11, places=4)
        self.assertAlmostEqual(result['AP50'], 100*8/11, places=4)

    def test_score_sort_matches_original_float32_convention(self):
        rows = [dict(midpoint=[0, 0, 20, 10, 0, score], metrics=dict(midpoint=dict(riou=iou)))
                for score, iou in [(.800000001, .2), (.8, .9), (.7, .9)]]
        result = ap(rows, 'midpoint', np, ap11)
        order = np.argsort(-np.asarray([r['midpoint'][5] for r in rows], dtype=np.float32))
        hits = np.asarray([0, 1, 1], dtype=np.float32)[order]
        tp, fp = hits.cumsum(), (1-hits).cumsum()
        self.assertEqual(result['AP75'], 100*ap11(tp.astype(np.float64)/3, tp/(tp+fp)))


if __name__ == '__main__':
    unittest.main()
