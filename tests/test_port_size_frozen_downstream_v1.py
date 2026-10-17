"""Meaningful protection checks for frozen downstream evaluation."""
from copy import deepcopy
import unittest
from crane_project.tools import eval_port_size_frozen_downstream_v1 as e

class FrozenEvaluationTests(unittest.TestCase):
    def fixture(self):
        box=[500.,500.,100.,50.,0.,.8]
        manifest=dict(sequence_id='fixture',split='fixed_dev',camera=dict(intrinsics=dict(fx=1000.,fy=1000.)),
            obb_reference_geometry=dict(long_edge_mean_m=2.,short_edge_mean_m=1.))
        calibration=dict(parameters=dict(c=1.,beta=2.,b_m=0.,q_signed_min=-.01,q_signed_max=.01))
        ps=[dict(frame_id='frame_00000',**{a:list(box) for a in e.ARMS})]
        ts=[dict(camera_geometry=dict(z_cg_opt_m=20.),pivot_relative=dict(theta_total_deg=1.),
            obb_geometry=dict(cx_px=500.,cy_px=500.,w_px=100.,h_px=50.,gamma_deg=0.))]
        return ps,ts,manifest,calibration

    def test_raw_opt_exact_fixture(self):
        _,s=e.depth_report(*self.fixture())
        for a in e.ARMS:self.assertEqual(s['methods'][a]['depth_metrics']['mae_m'],0.)

    def test_no_q_filter_and_full_denominator(self):
        p,t,m,c=self.fixture();p[0]['i07'][2]=70.;p[0]['axis15']=None
        _,s=e.depth_report(p,t,m,c)
        self.assertEqual(s['methods']['i07']['q_out_of_fit_support_count'],1)
        self.assertEqual(s['methods']['i07']['depth_metrics']['count'],1)
        self.assertEqual(s['methods']['axis15']['frame_count'],1)
        self.assertEqual(s['methods']['axis15']['numeric_depth_count'],0)
        self.assertEqual(s['paired']['midpoint->axis15']['lost_numeric'],1)

    def test_paired_improvement_and_worsening(self):
        p,t,m,c=self.fixture();p[0]['i07'][3]=49.;p[0]['axis15'][3]=48.
        _,s=e.depth_report(p,t,m,c)
        self.assertEqual(s['paired']['i07->axis15']['worsened'],1)
        self.assertEqual(s['paired']['midpoint->i07']['worsened'],1)

    def test_no_formula_or_box_mutation(self):
        args=self.fixture();before=deepcopy(args);e.depth_report(*args);self.assertEqual(args,before)

    def test_frame_order_and_missing_frame_rejected(self):
        args=self.fixture();args[0][0]['frame_id']='frame_00001'
        with self.assertRaises(ValueError):e.depth_report(*args)
        args=self.fixture();args[1].clear()
        with self.assertRaises(ValueError):e.depth_report(*args)

    def test_real_and_webots_original_resolution_contract(self):
        shape=(1,3,1024,1024)
        meta=dict(filename='/tmp/frame.jpg',ori_shape=(1080,1920,3),flip=False)
        src=dict(image_path='/tmp/frame.jpg',image_size=[1920,1080])
        e.validate_view(shape,[meta],src)
        with self.assertRaises(ValueError):e.validate_view(shape,[meta],dict(image_path='/tmp/frame.jpg'))
        meta['ori_shape']=(1024,1024,3)
        e.validate_view(shape,[meta],dict(image_path='/tmp/frame.jpg'))

    def test_depth_source_closure(self):
        value=e.checked_depth_sources()
        self.assertEqual(len(value["reviewed_training_only_differences"]),5)

    def test_resource_limits(self):
        e.check_memory(dict(process_mib=1536,free_mib=2048))
        for value in [dict(process_mib=1537,free_mib=4096),dict(process_mib=500,free_mib=2047)]:
            with self.assertRaisesRegex(RuntimeError,'RESOURCE_STOP'):e.check_memory(value)

    def test_size_identity_guards(self):
        b=[0.,0.,10.,5.,0.,.8];e.identity(b,[0.,0.,11.,6.,0.,.8]);e.identity(None,None)
        for i in (0,1,4,5):
            c=list(b);c[i]+=.1
            with self.assertRaises(ValueError):e.identity(b,c)
        with self.assertRaises(ValueError):e.identity(b,None)

if __name__=='__main__':unittest.main()
