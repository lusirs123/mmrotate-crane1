"""Completion auditing must compare values and preserve sequence boundaries."""
import base64
import importlib.util
from pathlib import Path
import pickle
import unittest


def load(name):
    p=Path(__file__).resolve().parents[1]/'crane_project/tools'/name
    spec=importlib.util.spec_from_file_location(p.stem,p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


AUDIT=load('audit_symeood_c_pair_completion_v1.py')
VERIFY=load('verify_symeood_c_pair_completion_v1.py')


class ArrayState:
    def __init__(self,values,storage_id):self.values=values;self.storage_id=storage_id
    def tolist(self):return self.values


def payload(values,storage):
    return base64.b64encode(pickle.dumps(dict(cpu=ArrayState(values,storage)),protocol=4)).decode()


def row(fid,seq='s1',present=True,hit=True):
    return dict(domain='real',seq=seq,fid=fid,frame=seq+str(fid),output=present,riou_export=.8 if hit else 0.,
        riou_native=.8 if hit else 0.,diag_export=100.,angle_export=0.,center_export=0. if hit else float('inf'),
        center_error=0.,angle_eligible=True,angle_error_deg=0.,gated_angle_sq=0.,a=0.,r=0.,
        long_error=0.,short_error=0.,long_bias=0.,short_bias=0.)


class CompletionTests(unittest.TestCase):
    def test_rng_storage_identity_is_not_rng_value_identity(self):
        a,b=payload([1,2,3],1),payload([1,2,3],2)
        self.assertNotEqual(a,b)
        self.assertEqual(AUDIT.rng_identity(a),AUDIT.rng_identity(b))

    def test_rng_value_change_is_detected(self):
        self.assertNotEqual(AUDIT.rng_identity(payload([1,2],1)),AUDIT.rng_identity(payload([1,3],1)))

    def test_missing_output_and_id_gap_do_not_bridge_pairs(self):
        rr=[row(0),row(1,present=False,hit=False),row(2),row(4),row(5)]
        m,pairs=VERIFY.aggregate(rr)
        self.assertEqual(set(pairs),{('s14','s15')})
        self.assertEqual(m['failure_events'],1)
        self.assertEqual(m['MCML_max'],1)

    def test_failure_events_do_not_join_across_videos(self):
        rr=[row(0,hit=False),row(1,hit=False),row(2,seq='s2',hit=False),row(3,seq='s2')]
        m,pairs=VERIFY.aggregate(rr)
        self.assertEqual(m['failure_events'],2)
        self.assertEqual(m['MCML_max'],2)
        self.assertEqual(m['MCML_mean'],1.5)
        self.assertNotIn(('s11','s22'),pairs)


if __name__=='__main__':unittest.main()
