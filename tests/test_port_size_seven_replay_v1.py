"""Replay axis/response semantics independent of CUDA or model training."""
from copy import deepcopy
import math
import unittest
from crane_project.tools import replay_port_size_seven_v1 as r
from crane_project.tools import analyze_port_size_seven_replay_v1 as a

SLOTS=('natural','long_plus','long_minus','short_plus','short_minus','long_exact','short_exact')


def view():
    m=[0.,0.,100.,40.,0.,.8];gt=[0.,0.,100.,50.,0.]
    ns=[0.,0.,102.,45.,0.,.8];q=[[list(m) for _ in SLOTS]]
    finals=[list(ns) for _ in SLOTS]
    d=dict(boxes_original=finals,candidate_original=deepcopy(finals),accepted=[True]*7,
           neutral_bypass=[False]*7,square_bypass=[False]*7,checks=dict(finite=[True]*7))
    rec=dict(image='sim_seq08_00000',sequence='sim_seq08',domain='sim',frame_id=0,split='train_sim',scale=1.,shard='train_s1')
    return r.serialize_view(rec,m,gt,m,m[:5],[1.,1.],q,[[[.01,.02] for _ in SLOTS]],d,0,SLOTS)


class ReplayTests(unittest.TestCase):
    def test_natural_damage_not_mistaken_for_extra_response(self):
        v=view();x=a.response_detail(v,'short_plus')
        self.assertTrue(x['natural_harmed']);self.assertFalse(x['extra_harmed'])
        self.assertAlmostEqual(x['log_response'],0.);self.assertAlmostEqual(x['error_change'],0.)

    def test_extra_harm_and_useful_recovery_are_separate(self):
        v=view();s=v['slots'][3];s['final_canonical'][2]=103.
        s['final_errors']=r.errors([103.,45.],v['gt'])
        x=a.response_detail(v,'short_plus')
        self.assertTrue(x['extra_harmed']);self.assertGreater(x['log_response'],0.)
        self.assertTrue(x['natural_harmed'])

    def test_roundoff_response_and_error_counts(self):
        v=view();s=v['slots'][3];s['final_canonical'][2]+=1e-6;s['final_errors']=r.errors(s['final_canonical'][2:4],v['gt'])
        x=a.response_detail(v,'short_plus');self.assertFalse(x['extra_harmed'])
        self.assertLess(abs(x['log_response']),a.EPS)

    def test_proposal_axis_cross_is_not_output_response(self):
        v=view();v['slots'][3]['input_axis_cross']=True;v['slots'][3]['pre_axis_cross']=True
        x=a.response_detail(v,'short_plus');self.assertTrue(x['input_axis_cross'])
        self.assertEqual(x['log_response'],0.);self.assertFalse(x['extra_harmed'])

    def test_fixed_axis_never_sorts_crossing_proposal(self):
        m=[0,0,100,99,0,.8];q=[0,0,100,105,0,.8]
        self.assertEqual(r.fixed_edges(q,m),[100,105])
        self.assertEqual(r.canonical(q)[2:4],[105,100])

    def test_roi_and_natural_raw_swaps_are_independent(self):
        v=view();self.assertFalse(v['natural_raw_swap']);self.assertFalse(v['model_b_swap'])
        self.assertEqual(v['slots'][0]['delta_fixed'],[.01,.02])
        self.assertEqual(r.fixed_edges([0,0,40,100,0,.8],[0,0,40,100,0,.8]),[100,40])

    def test_actual_resize_can_change_model_axis_order(self):
        m=[0.,0.,40.,100.,0.,.8];bm=[0.,0.,120.,100.,0.]
        ds=dict(boxes_original=[list(m) for _ in SLOTS],candidate_original=[list(m) for _ in SLOTS],
                accepted=[True]*7,neutral_bypass=[False]*7,square_bypass=[False]*7,checks=dict(finite=[True]*7))
        rec=dict(image='x',sequence='s',domain='real',frame_id=0,split='train',scale=1.,shard='train_s1')
        v=r.serialize_view(rec,m,[0,0,100,50,0],m,bm,[3.,1.],[[list(m) for _ in SLOTS]],[[[.01,.02] for _ in SLOTS]],ds,0,SLOTS)
        self.assertTrue(v['natural_raw_swap']);self.assertFalse(v['model_b_swap'])
        self.assertEqual(v['slots'][0]['delta_raw'],[.01,.02]);self.assertEqual(v['slots'][0]['delta_fixed'],[.02,.01])

    def test_final_identity_protection(self):
        v=view();self.assertTrue(all(not s['final_axis_cross'] for s in v['slots']))
        self.assertTrue(all(s['final_raw'][5]==v['midpoint'][5] for s in v['slots']))

    def test_gt_exact_and_ratio_algebra(self):
        e=r.errors([100.,50.],[0,0,100,50,0]);self.assertEqual(e['relative'],[0.,0.]);self.assertEqual(e['r'],0.)
        g=r.errors([100.,40.],[0,0,100,50,0]);self.assertAlmostEqual(g['r'],-math.log(.8))

    def test_saved_delta_has_tighter_tolerance_than_coordinates(self):
        self.assertGreater(r.compare_saved([10.00002],[10.],'box'),0.)
        with self.assertRaises(ValueError):r.compare_saved([.01002],[.01],'delta')

    def test_response_stats_distinguishes_amplitude_and_utility(self):
        v=view();x=a.response_detail(v,'short_plus');s=a.response_stats([x])
        self.assertEqual(s['n'],1);self.assertEqual(s['response_above_roundoff'],0)
        self.assertEqual(s['natural_harmed'],1);self.assertEqual(s['extra_harmed'],0)
        self.assertGreater(s['natural_damage_positive_mean_pp'],0.)


if __name__=='__main__':unittest.main()
