"""Temporal audit edge cases: missing frames must never be reconnected."""
import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[1]/'crane_project/tools/verify_historical_b_ablation111_v1.py'
SPEC=importlib.util.spec_from_file_location('historical_b_verify',PATH)
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


def records(fids,present,seq='seq03'):
    return [dict(domain='real',seq=seq,fid=str(fid),output=str(ok)) for fid,ok in zip(fids,present)]


class PairAuditTests(unittest.TestCase):
    def test_missing_output_and_id_gap_break_pairs(self):
        rows=records([0,1,2,4,5],[True,False,True,True,True])
        self.assertEqual(MODULE.valid_pairs(rows),{('real','seq03',4,5)})

    def test_equal_output_counts_do_not_imply_equal_pair_denominators(self):
        a=MODULE.valid_pairs(records([0,1,2,3],[True,False,True,True]))
        b=MODULE.valid_pairs(records([0,1,2,3],[False,True,True,True]))
        self.assertEqual(len(a),1);self.assertEqual(len(b),2)
        self.assertEqual(a&b,{('real','seq03',2,3)})

    def test_video_boundaries_and_frame_order(self):
        rows=records([2,1,0],[True,True,True])+records([3,4],[True,True],seq='seq04')
        self.assertEqual(MODULE.valid_pairs(rows),{('real','seq03',0,1),('real','seq03',1,2),('real','seq04',3,4)})


if __name__=='__main__':unittest.main()
