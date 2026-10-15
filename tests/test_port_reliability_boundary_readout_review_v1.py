import math
import unittest
from copy import deepcopy
import numpy as np
from crane_project.tools import review_port_reliability_boundary_readout_v1 as m


def row(i,sequence='s',long=100,short=40):
    return dict(image='%s_%03d'%(sequence,i),sequence=sequence,frame_id=i,
        pred=[80,80,long,short,0,.8],gt=[80,80,100,40,0],
        original_simple_decision=dict(size_accepted=True))


class TestReview(unittest.TestCase):
    def test_groups_boundaries_and_sign(self):
        for l,s,g in [(90,40,'correct'),(89,40,'small_10_25'),(75,40,'small_10_25'),
                      (74,40,None),(120,40,'large_or_mixed'),(80,50,'large_or_mixed')]:
            self.assertEqual(m.group(row(0,long=l,short=s)),g)
    def test_selection_order_independent_and_balanced(self):
        rows=[row(i,s) for s in ['a','b','c'] for i in range(30)]
        a,_=m.select(rows);b,_=m.select(list(reversed(rows)))
        self.assertEqual(a,b);self.assertEqual(len(a),12)
        self.assertEqual(m.Counter(x['row']['sequence'] for x in a),dict(a=4,b=4,c=4))
        self.assertEqual([x['row']['frame_id'] for x in a[:4]],[0,9,19,29])
    def test_shortage_not_filled_from_other_group(self):
        chosen,_=m.select([row(i,long=80) for i in range(3)])
        self.assertEqual(len(chosen),3)
    def test_duplicate_rejected_and_no_mutation(self):
        rows=[row(i) for i in range(3)];before=deepcopy(rows);m.select(rows)
        self.assertEqual(rows,before)
        with self.assertRaises(ValueError):m.select(rows+rows[:1])
    def test_profile_replay_and_axis_exchange(self):
        y,x=np.indices((180,180));gray=((x//7+y//11)%2)*200.
        pred=[90,90,70,40,.23,.8]
        ev=m.old.evidence(gray,pred);d=m.profiles(gray,pred);m.check_profiles(d,ev)
        other=pred[:];other[2:4]=other[3:1:-1];other[4]-=math.pi/2
        dd=m.profiles(gray,other)
        for axis in ('L','S'):
            for a,b in zip(d[axis],dd[axis]):np.testing.assert_allclose(a['gradient'],b['gradient'],atol=1e-10)
    def test_flat_profiles_not_supported(self):
        gray=np.ones((180,180));p=[90,90,60,30,0]
        d=m.profiles(gray,p);m.check_profiles(d,m.old.evidence(gray,p))
        self.assertFalse(any(v for a in d.values() for s in a for v in s['valid']))
    def test_image_border_invalid(self):
        gray=np.arange(200*200,dtype=float).reshape(200,200);p=[0,0,90,50,0]
        d=m.profiles(gray,p);m.check_profiles(d,m.old.evidence(gray,p))
        self.assertFalse(all(v for a in d.values() for s in a for v in s['valid']))
    def test_GT_intersection_geometry(self):
        gt=[50,50,100,40,0]
        np.testing.assert_allclose(m.gt_interval([50,50],[1,0],100,gt),[-.5,.5])
        self.assertIsNone(m.gt_interval([50,100],[1,0],100,gt))
        np.testing.assert_allclose(m.gt_interval([50,50],[0,-1],40,gt),[-.5,.5])


if __name__=='__main__':unittest.main()
