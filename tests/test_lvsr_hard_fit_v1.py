"""Hard-view selection cannot reuse heldout images or change control scale."""
import unittest
import torch
from crane_project.tools import diagnose_lvsr_hard_fit_v1 as d


def record(seq,i,hard=False,scale=1.):
    m=torch.tensor([[100.,100.,100.,50.,0.,.9]])
    gt=m[:,:5].clone()
    if hard:gt[:,2]*=1.2
    return dict(image=seq+str(i),sequence=seq,domain='real',frame_id=i,scale=scale,split='train',midpoint_original=m,gt_original=gt)


class HardChecks(unittest.TestCase):
    def test_all_reachable_hard_and_matched_controls_only(self):
        rr=[record('real_a',i) for i in range(20)]+[record('real_a',100+i,True,s) for i,s in enumerate((1.,1.,.5))]
        rr += [record('real_a',200+i,False,.5) for i in range(10)]
        rr += [record('real_seq13',i,True) for i in range(5)]
        val=dict(record('real_a',999,True),split='val');rr.append(val)
        feasible={(r['image'],r['scale']) for r in rr if r['frame_id']!=101}
        hard,control,fit=d.select_fit(rr,feasible)
        self.assertEqual(len(hard),2);self.assertEqual(len(control),2)
        self.assertTrue(all(max(d.errors(r))>.1 for r in hard))
        self.assertTrue(all(max(d.errors(r))<=.03 for r in control))
        self.assertEqual(sorted(r['scale'] for r in hard),sorted(r['scale'] for r in control))
        self.assertTrue(all(r['sequence']!='real_seq13' and r['split']=='train' for r in fit))
        again=d.select_fit(list(reversed(rr)),feasible)[2]
        self.assertEqual([(r['image'],r['scale']) for r in fit],[(r['image'],r['scale']) for r in again])

    def test_missing_control_stops(self):
        rr=[record('real_a',0,True)]
        with self.assertRaises(ValueError):d.select_fit(rr,{(rr[0]['image'],1.)})


if __name__=='__main__':unittest.main()
