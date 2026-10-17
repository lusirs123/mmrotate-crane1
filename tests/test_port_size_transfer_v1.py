"""Scope/denominator/geometry checks for the bounded transfer audit."""
import unittest
from copy import deepcopy
from crane_project.tools import analyze_port_size_transfer_v1 as a

class TransferAuditTests(unittest.TestCase):
    def row(self, short=50.):
        b=[0.,0.,100.,50.,0.,.8]
        return dict(gt=b[:5],b=list(b),midpoint=list(b),i07=[0.,0.,100.,short,0.,.8],axis15=list(b),image='f',scale=1.)

    def test_near_damage_despite_size10_staying_correct(self):
        r=self.row(49.);p=a.paired([r],'i07',1)
        self.assertEqual((p['near_n'],p['near_harmed'],p['right_to_wrong']),(1,1,0))
        self.assertEqual((p['shrink_n'],p['shrink_harmed']),(1,1))

    def test_enlargement_can_improve_or_harm(self):
        r=self.row(50.);r['midpoint'][3]=49.
        p=a.paired([r],'i07',1);self.assertEqual(p['enlarge_improved'],1)
        r['i07'][3]=52.;p=a.paired([r],'i07',1);self.assertEqual(p['enlarge_harmed'],1)

    def test_missing_denominator(self):
        r=self.row();r.update(midpoint=None,i07=None,axis15=None)
        a.check_identity(r);s=a.stats([r],'i07');self.assertEqual((s['frames'],s['outputs'],s['missing']),(1,0,1));self.assertIsNone(s['short_mae_pct'])

    def test_axis_swap_and_log_identity(self):
        r=self.row();r['i07']=[0.,0.,50.,100.,-a.math.pi/2,.8]
        self.assertEqual(a.errors(r,'i07')['e'],[0.,0.])
        r=self.row(48.);e=a.errors(r,'i07')
        self.assertAlmostEqual(e['a']-e['r']/2,a.math.log(.96))

    def test_readonly_and_guards(self):
        r=self.row(49.);before=deepcopy(r);a.stats([r],'i07');a.paired([r],'i07',1);self.assertEqual(r,before)
        r['i07'][0]=1.
        with self.assertRaises(ValueError):a.check_identity(r)
        with self.assertRaises(ValueError):a.canonical([0.,0.,0.,1.,0.])

    def test_support_is_marginal_and_cuts_fixed(self):
        r=self.row();x=a.features(r)
        self.assertTrue(a.inside(r,[(v,v) for v in x]))
        self.assertFalse(a.inside(r,[(x[0]+1,x[0]+2)]+[(v,v) for v in x[1:]]))
        self.assertEqual(a.grid(r),'high/low')

    def test_no_pooled_scales_and_duplicate_join(self):
        r=self.row();r.update(edge_residual=r['i07'],domain='sim',sequence='s',split='train',shard='s',target_log=[0.,0.])
        s=deepcopy(r);s['scale']=.5
        joined=a.join_training([r,s],[r,s],2);self.assertEqual({x['scale'] for x in joined},{1.,.5})
        with self.assertRaises(ValueError):a.join_training([r,r],[r,s],2)

if __name__=='__main__':unittest.main()
