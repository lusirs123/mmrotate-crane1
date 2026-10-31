"""Exact vector-norm reconstruction for diagnostic mixtures, no model."""
import math
import unittest
from crane_project.tools.audit_eood_B_balance_v1 import mixture


class MixtureChecks(unittest.TestCase):
    def test_zero_retains_DIoU(self):
        self.assertEqual(mixture(3.,2.,-.4,0.)['norm_over_DIoU'],1.)
        self.assertEqual(mixture(3.,2.,-.4,0.)['cosine_to_DIoU'],1.)

    def test_matches_explicit_vectors(self):
        for alpha in (0.,.25,1.):
            a=(2.,1.);b=(-.3,1.7)
            norm=lambda x:math.sqrt(sum(v*v for v in x))
            c=sum(x*y for x,y in zip(a,b))/(norm(a)*norm(b))
            r=mixture(norm(a),norm(b),c,alpha)
            z=tuple(x+alpha*y for x,y in zip(a,b))
            self.assertAlmostEqual(r['norm_over_DIoU'],norm(z)/norm(a))
            self.assertAlmostEqual(r['cosine_to_DIoU'],sum(x*y for x,y in zip(a,z))/(norm(a)*norm(z)))

    def test_invalid_or_cancelling_inputs_rejected(self):
        for args in ((0,1,0,1),(1,-1,0,1),(1,1,2,1),(1,1,0,-1),(1,1,-1,1)):
            with self.assertRaises(ValueError):mixture(*args)


if __name__=='__main__':unittest.main()
