"""Analytic checks for independent result geometry and temporal boundaries."""
import math
import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crane_project/tools'))
from verify_symeood_usg_results_v1 import overlap,angle,summarize


class ResultChecks(unittest.TestCase):
    def test_identical_and_axis_equivalent_rectangles(self):
        a=np.array([0.,0.,4.,2.,.3])
        b=np.array([0.,0.,2.,4.,.3+math.pi/2])
        self.assertAlmostEqual(overlap(a,a),1.)
        self.assertAlmostEqual(overlap(a,b),1.)

    def test_disjoint_and_missing(self):
        a=np.array([0.,0.,4.,2.,0.]);b=a.copy();b[0]=10
        self.assertEqual(overlap(a,b),0.)
        self.assertEqual(overlap(None,b),0.)

    def test_half_shift_and_rotation_invariance(self):
        for theta in (0.,.4,1.2):
            a=np.array([0.,0.,4.,2.,theta]);b=a.copy()
            b[:2]=[2*math.cos(theta),2*math.sin(theta)]
            self.assertAlmostEqual(overlap(a,b),1/3,places=12)

    def test_angle_wrap(self):
        self.assertAlmostEqual(math.degrees(angle(math.radians(89),math.radians(-89))),2.)

    def row(self,fid,output=True,diag=100.):
        r=dict(domain='sim',seq='fixture',fid=fid,frame='f'+str(fid),output=output,
               center_export=0. if output else float('inf'),riou_export=.9 if output else 0.,
               gated_angle_sq=0. if output else 8100.,score=.9,riou_native=.9 if output else 0.)
        if output:r.update(center_error=0.,angle_eligible=True,angle_error_deg=0.,
            diag_export=diag,angle_export=0.,long_error=0.,short_error=0.,long_bias=0.,short_bias=0.,a=0.,r=0.)
        return r

    def test_all_missing_is_failure_not_zero_error(self):
        result=summarize([self.row(i,False) for i in range(5)])
        self.assertEqual(result['outputs'],0);self.assertEqual(result['MCML_max'],5)
        self.assertEqual(result['failure_events'],1);self.assertIsNone(result['MRF'])
        self.assertEqual(result['gated_angle_RMSE_deg'],90.)
        self.assertIsNone(result['DFR']);self.assertIsNone(result['long_MAE_pct'])

    def test_no_pair_crosses_gap_or_video(self):
        rows=[self.row(1),self.row(2,diag=110),self.row(4,diag=200)]
        extra=self.row(1,diag=500);extra['seq']='other';rows.append(extra)
        result=summarize(rows)
        self.assertEqual(result['DFR_pairs'],1);self.assertAlmostEqual(result['DFR'],10.)
        self.assertEqual(result['MCML_segments'],3)


if __name__=='__main__':unittest.main()
