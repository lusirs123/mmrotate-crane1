"""Tests for offline denominators, geometry contracts and exact bookkeeping."""
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from tools.analysis import analyze_port_size_all_evidence_v1 as a


class EvidenceTests(unittest.TestCase):
    def test_canonical_swap_keeps_long_short_association(self):
        x=a.sizes([0,0,5,10,0,1],[0,0,10,5,1.5707963267948966])
        self.assertEqual(x['long_log'],0)
        self.assertEqual(x['short_log'],0)
        self.assertTrue(x['joint_correct'])

    def test_small_common_scale_can_pass_joint_ten_percent(self):
        x=a.sizes([0,0,95,19,0,1],[0,0,100,20,0])
        self.assertTrue(x['joint_correct'])
        self.assertAlmostEqual(x['ratio_log'],0)
        self.assertLess(x['short_log'],0)

    def test_missing_outputs_keep_full_frame_denominator(self):
        gt=[0,0,100,20,0]
        rows=[dict(size=dict(M=a.sizes([0,0,100,20,0,1],gt))),dict(size=dict(M=None))]
        g=a.geometry_group(rows,'M')
        self.assertEqual(g['frames'],2)
        self.assertEqual(g['outputs'],1)
        self.assertEqual(g['all_frame_joint_correct_coverage'],.5)

    def test_ordered_accounting_all_boolean_transitions(self):
        states=list(itertools.product((False,True),repeat=4))
        result=a.ordered_accounting(states)
        for key,v in result.items():
            self.assertEqual(v['delta'],v['correctness_change_old_flags']+v['flags_change_new_correctness'])
        x=a.ordered_accounting([(True,False,True,True)])
        self.assertEqual(x['FA']['correctness_change_old_flags'],1)
        self.assertEqual(x['FA']['flags_change_new_correctness'],0)
        x=a.ordered_accounting([(False,False,True,False)])
        self.assertEqual(x['FA']['correctness_change_old_flags'],0)
        self.assertEqual(x['FA']['flags_change_new_correctness'],-1)

    def test_duplicate_identity_and_output_mismatch_are_rejected(self):
        with self.assertRaises(ValueError):a.index([dict(id='x'),dict(id='x')],'id')
        with self.assertRaises(ValueError):a.equal_box(None,[0,0,1,1,0,1])
        with self.assertRaises(ValueError):a.equal_box([1],[2])

    def test_input_digest_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json';p.write_text('{}')
            spec=dict(path=str(p),sha256=hashlib.sha256(b'{}').hexdigest())
            self.assertEqual(a.checked_source(spec),b'{}')
            p.write_text('{"changed":true}')
            with self.assertRaises(ValueError):a.checked_source(spec)
            p=Path(d)/'output.json';a.dump(p,{})
            with self.assertRaises(FileExistsError):a.dump(p,{})

    def test_candidate_direction_guard(self):
        gt=[0,0,100,20,0]
        row=dict(boxes=dict(M=gt+[1],C=[1,0,99,19,0,1]),size=dict(M=a.sizes(gt+[1],gt),C=a.sizes([1,0,99,19,0,1],gt)))
        with self.assertRaises(ValueError):a.stage_pair([row],'M','C')

    def test_size_source_counterfactual_preserves_raw_and_canonical_direction(self):
        b=[0,0,100,20,0,.7];m=[3,4,18,95,-1.5,.7]
        h=a.substitute_size_source(b,m)
        self.assertEqual(h[:2],m[:2]);self.assertEqual(h[4:],m[4:])
        self.assertEqual(h[2:4],[20,100])
        self.assertEqual(a.depth.canonical(h)[4],a.depth.canonical(m)[4])
        self.assertIsNone(a.substitute_size_source(None,None))
        with self.assertRaises(ValueError):a.substitute_size_source(None,m)

    def test_exact_depth_change_under_equal_focal_lengths(self):
        gt=[0,0,100,20,0];b=gt+[1];c=[0,0,95,19,0,1]
        intr=dict(fx=900,fy=900);geom=dict(long_edge_mean_m=5,short_edge_mean_m=1)
        par=dict(c=1,beta=20,b_m=0)
        row=dict(boxes=dict(B=b,M=c),size=dict(B=a.sizes(b,gt),M=a.sizes(c,gt)),
            depth=dict(B=a.depth.depth(b,intr,geom,par),M=a.depth.depth(c,intr,geom,par)),beta=20,truth_z_m=45)
        x=a.stage_pair([row],'B','M')
        self.assertLess(x['exact_log_depth_change']['max_abs_identity_residual'],1e-14)
        self.assertAlmostEqual(x['exact_log_depth_change']['ratio_q']['mean'],0)
        self.assertEqual(x['depth_worse'],1)

    def test_contract_excludes_selection_and_candidate_val_substitution(self):
        doc=json.loads((a.ROOT/'tools/analysis/port_size_all_evidence_v1_inputs.json').read_text())
        for k in ('training','inference','policy_refit','formula_refit','parameter_selection'):
            self.assertFalse(doc['scope'][k])
        self.assertIn('Not run',doc['scope']['candidate_VAL'])
        self.assertEqual(doc['expected_counts']['port_TRAIN'],2558)
        self.assertEqual(doc['expected_counts']['port_TRAIN_half_view'],2558)


if __name__=='__main__':unittest.main()
