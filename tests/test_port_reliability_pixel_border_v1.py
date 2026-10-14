import unittest
import numpy as np
from crane_project.utils import port_reliability_pixel_border_v1 as c

class TestPixelBorder(unittest.TestCase):
    def setUp(self):
        self.im=np.zeros((160,200));self.im[60:101,50:151]=200
        self.box=[100,80,100,40,0,.8]
    def test_native_edges(self):
        e=c.evidence(self.im,self.box)
        for a in ('L','S'):
            self.assertTrue(e[a]['supported']);self.assertLess(e[a]['mismatch'],.04)
    def test_wrong_short_side(self):
        b=self.box[:];b[3]=30
        e=c.evidence(self.im,b)
        self.assertGreater(e['S']['signed_offset'],.2)
        self.assertGreater(e['S']['mismatch'],c.evidence(self.im,self.box)['S']['mismatch'])
    def test_too_wide(self):
        b=self.box[:];b[3]=52
        self.assertLess(c.evidence(self.im,b)['S']['signed_offset'],-.2)
    def test_swap_pi(self):
        e=c.evidence(self.im,self.box)
        for b in ([100,80,40,100,np.pi/2,.8],[100,80,100,40,np.pi,.8]):
            x=c.evidence(self.im,b)
            for a in ('L','S'):self.assertAlmostEqual(x[a]['mismatch'],e[a]['mismatch'],places=10)
    def test_rotation(self):
        x=c.evidence(self.im.T,[80,100,100,40,np.pi/2,.8]);e=c.evidence(self.im,self.box)
        for a in ('L','S'):self.assertAlmostEqual(x[a]['mismatch'],e[a]['mismatch'])
    def test_flat(self):
        e=c.evidence(np.ones((160,200)),self.box)
        self.assertFalse(e['S']['supported']);self.assertIsNone(e['S']['mismatch'])
    def test_boundary_missing(self):
        self.assertIsNone(c.evidence(self.im,None))
        self.assertFalse(c.evidence(self.im,[1,1,100,40,0])['S']['supported'])
    def test_input_immutable(self):
        im=self.im.copy();box=self.box[:];c.evidence(im,box)
        np.testing.assert_array_equal(im,self.im);self.assertEqual(box,self.box)
    def test_bilinear_and_invalid(self):
        x,mask=c.sample(np.arange(9).reshape(3,3),np.array([[.5,.5],[-1,1]]))
        self.assertEqual(x[0],2);self.assertEqual(mask.tolist(),[True,False])
        with self.assertRaises(ValueError):c.evidence(self.im,[1,1,0,3,0])
    def test_no_evidence_no_gate(self):
        r=dict(image='x',domain='real',sequence='real_a',low_temporal=True,wrong=True,error_type='S_only',evidence=c.evidence(np.ones((160,200)),self.box),scale=50,accepted=True)
        a=c.analyze([r]);self.assertFalse(a['feasibility_passed']);self.assertEqual(a['groups']['all']['wrong'],1)
    def test_fixed_direction_and_same_video_size_matching(self):
        rows=[]
        for video in ('real_a','real_b'):
            for i in range(35):
                wrong=i<11
                ev={a:dict(supported=True,mismatch=.4 if wrong else .1) for a in ('L','S')}
                rows.append(dict(domain='real',sequence=video,low_temporal=True,wrong=wrong,
                    error_type='both' if wrong else 'correct',evidence=ev,scale=50,accepted=True))
        report=c.analyze(rows)
        self.assertTrue(report['feasibility_passed'])
        self.assertEqual(report['groups']['all']['axes']['S']['size_matched_pairs'],2*11*24)
        self.assertEqual(report['groups']['all']['axes']['S']['size_matched_auc'],1.)
        for r in rows:
            for a in ('L','S'):r['evidence'][a]['mismatch']=.1 if r['wrong'] else .4
        self.assertFalse(c.analyze(rows)['feasibility_passed'])
    def test_no_history_no_quality_group(self):
        r=dict(domain='real',sequence='real_a',low_temporal=False,wrong=True)
        report=c.analyze([r]);self.assertEqual(report['groups']['all']['n'],0)
        self.assertFalse(report['feasibility_passed'])
if __name__=='__main__':unittest.main()
