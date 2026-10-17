"""Joint-group and missing-evidence checks; no model/cache/weights required."""
from copy import deepcopy
import math
import unittest
from crane_project.tools import analyze_port_size_joint_v1 as a


class JointAuditTests(unittest.TestCase):
    def fixture(self):
        m=[0.,0.,100.,40.,0.,.8];gt=[0.,0.,100.,50.,0.]
        c=[0.,0.,101.,45.,0.,.8];p=[0.,0.,103.,49.,0.,.8]
        base=dict(split='train',scale=1.,image='sim_seq08_00000',sequence='sim_seq08',frame_id=0,domain='sim',b=list(m),midpoint=m,gt=gt,size_delivery='accepted')
        return dict(base,edge_residual=c),dict(base,edge_residual=p)

    def rows(self):return a.bind(*[[x] for x in self.fixture()],lambda p,g:.9)

    def test_error_improvement_can_damage_accurate_edge(self):
        rs=self.rows();self.assertEqual(rs[0]['member'],'long_near_short_error')
        s=a.summarize(rs,'m','p07')
        self.assertEqual(s['error_edge_improved'],1)
        self.assertEqual(s['error_improved_accurate_harmed'],1)
        self.assertEqual(s['wrong_to_right'],1)

    def test_control_comparison_does_not_redefine_near_membership(self):
        rs=self.rows();s=a.summarize(rs,'c20','p07')
        self.assertEqual(rs[0]['member'],'long_near_short_error')
        self.assertEqual(s['error_improved_accurate_harmed'],1)

    def test_joint_scale_ratio_opposite_changes(self):
        g=a.geometry([0,0,100,50,0],[0,0,90,45,0],lambda p,g:.8)
        h=a.geometry([0,0,100,50,0],[0,0,100,45,0],lambda p,g:.9)
        self.assertLess(abs(h['a']),abs(g['a']));self.assertGreater(abs(h['r']),abs(g['r']))

    def test_ties_and_missing_are_not_forced_into_improvement(self):
        c,p=self.fixture();c['edge_residual']=c['midpoint'];p['edge_residual']=p['midpoint']
        rs=a.bind([c],[p],lambda p,g:.8)
        self.assertEqual(a.transition(rs[0]['geoms']['m'],rs[0]['geoms']['p07']),'tied/tied')
        for r in (c,p):r['midpoint']=None;r['edge_residual']=None
        missing=a.bind([c],[p],lambda p,g:.8);s=a.summarize(missing,'m','p07')
        self.assertEqual((s['frames'],s['outputs'],s['missing']),(1,0,1));self.assertEqual(s['after_RIoU_full'],0.)

    def test_output_identity_and_duplicate_views_rejected(self):
        c,p=self.fixture();p['edge_residual'][0]=1.
        with self.assertRaisesRegex(ValueError,'Geometry identity'):a.bind([c],[p],lambda p,g:.8)
        c,p=self.fixture()
        with self.assertRaisesRegex(ValueError,'Duplicate'):a.bind([c,c],[p,p],lambda p,g:.8)

    def test_both_near_and_pattern_partition(self):
        c,p=self.fixture()
        for r in (c,p):r['gt']=[0.,0.,100.,40.,0.]
        rs=a.bind([c],[p],lambda p,g:.8);self.assertEqual(rs[0]['member'],'both_near')
        s,edges,ar=a.natural_audit(rs)
        cells=[x for x in edges if x['group']=='overall' and x['pair']=='m->p07']
        self.assertEqual(sum(x['outputs'] for x in cells),1)

    def test_artificial_aggregate_does_not_claim_individual_response(self):
        cells={}
        for domain in ('real','sim'):
            for scale in (.5,1.):
                for slot in a.SLOTS:
                    cells[domain+'/'+str(scale)+'/'+slot]=dict(count=2,before_abs_sum=[.04,.06],after_abs_sum=[.02,.04],harmed=[1,1],improved=[1,1],axis_cross=0,fallback=0)
        probe=dict(status='FROZEN_COMMON_TRAIN_PROPOSAL_PROBE_COMPLETE',inventories=dict(repeat7=cells,proposal7=deepcopy(cells)))
        rs=a.artificial(probe);self.assertEqual(len(rs),112)
        self.assertTrue(all(r['individual_response']=='MISSING' for r in rs))
        self.assertTrue(all(abs(r['after_minus_natural_MAE_pp'])<1e-10 for r in rs))

    def test_canonical_edges_and_aspect(self):
        x=a.geometry([0,0,100,50,0],[0,0,50,100,-math.pi/2],lambda p,g:1.)
        self.assertEqual(x['e'],[0.,0.]);self.assertEqual(x['r'],0.)

    def test_train_sim_is_training_role_with_source_name_preserved(self):
        c,p=self.fixture()
        for r in (c,p):r['split']='train_sim'
        row=a.bind([c],[p],lambda p,g:.8)[0]
        self.assertEqual(row['split'],'train');self.assertEqual(row['source_split'],'train_sim')


if __name__=='__main__':unittest.main()
