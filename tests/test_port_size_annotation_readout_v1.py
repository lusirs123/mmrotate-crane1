import copy
import importlib.util
import math
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('size_readout',ROOT/'tools/data/diagnose_port_size_annotation_readout_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.points=[[10,10],[90,10],[90,50],[10,50]]
        self.frame=dict(image='images/f.png',size=[120,100],image_sha256='fixed')
        self.doc=dict(imagePath='f.png',imageWidth=120,imageHeight=100,
            flags=dict(independent_reviewed=True,unmeasurable=False),
            object_definition='',measurement_basis='',shapes=[dict(label=m.packet.LABEL,shape_type='polygon',points=self.points)])
        self.axis=dict(image_sha256='fixed',axis_points=[],axis_reviewed=False,
                       axis_source='PENDING_MANUAL_SAME_FRAME_AXIS',reference_k=None,k_basis='')

    def test_enclosing_virtual_corners_do_not_require_physical_corner_claim(self):
        self.assertEqual(m.enclosing_points(self.doc,self.frame),self.points)
        self.assertEqual(self.doc['measurement_basis'],'')

    def test_unreviewed_or_self_crossing_enclosure_rejected(self):
        self.doc['flags']['independent_reviewed']=False
        with self.assertRaises(ValueError):m.enclosing_points(self.doc,self.frame)
        self.doc['flags']['independent_reviewed']=True
        self.doc['shapes'][0]['points']=[self.points[i] for i in (0,2,1,3)]
        with self.assertRaises(ValueError):m.enclosing_points(self.doc,self.frame)

    def test_reference_rectangle_and_pi_invariance(self):
        r=m.reference_obb(self.points)
        self.assertEqual((r['long_px'],r['short_px']), (80,40))
        p=m.project(self.points,0); q=m.project(self.points,math.pi)
        self.assertAlmostEqual(p['transverse_px'],q['transverse_px'])
        self.assertAlmostEqual(m.angle_delta(0,math.pi),0)

    def test_rotation_translation_preserve_dimensions(self):
        theta=.35; c,s=math.cos(theta),math.sin(theta)
        points=[[x*c-y*s+100,x*s+y*c+90] for x,y in self.points]
        r=m.reference_obb(points)
        self.assertAlmostEqual(r['long_px'],80,places=8)
        self.assertAlmostEqual(r['short_px'],40,places=8)
        self.assertAlmostEqual(m.angle_delta(r['angle_rad'],theta),0,places=8)

    def test_proxy_is_order_invariant_but_not_independent_evidence(self):
        theta=m.proxy_axis(self.points)
        self.assertAlmostEqual(m.angle_delta(theta,m.proxy_axis(list(reversed(self.points)))),0)
        self.assertEqual(m.axis_measurement(self.axis,self.frame,self.points)['status'],'PENDING_INDEPENDENT_SAME_FRAME_AXIS')

    def test_fixed_axis_width_is_independent_of_k(self):
        self.axis.update(axis_points=[[10,30],[90,30]],axis_reviewed=True,axis_source='manual_same_frame_long_axis')
        a=m.axis_measurement(self.axis,self.frame,self.points)
        self.assertEqual(a['fixed_axis_enclosing']['transverse_px'],40)
        self.assertEqual(a['fixed_k_status'],'NOT_EVALUATED')
        self.axis.update(reference_k=2,k_basis='same-object reference confirmed by human')
        b=m.axis_measurement(self.axis,self.frame,self.points)
        self.assertEqual(b['fixed_axis_enclosing'],a['fixed_axis_enclosing'])
        self.assertEqual(b['generated_minus_measured_relative'],0)

    def test_unsupported_axis_k_or_wrong_pixels_rejected(self):
        for update in (dict(reference_k=2),dict(axis_reviewed='true'),dict(image_sha256='other'),
                       dict(axis_points=[[1,1],[2,2]])):
            axis=dict(self.axis,**update)
            with self.assertRaises(ValueError):m.axis_measurement(axis,self.frame,self.points)
        self.axis.update(axis_points=[[10,30],[90,30]],axis_reviewed=True,axis_source='quad_proxy')
        with self.assertRaises(ValueError):m.axis_measurement(self.axis,self.frame,self.points)

    def test_one_degree_perturbation_is_sensitivity_not_truth_error(self):
        p=m.project(self.points,0);q=m.project(self.points,math.pi/180)
        self.assertGreater(q['transverse_px'],p['transverse_px'])
        self.assertAlmostEqual(q['transverse_px'],40*math.cos(math.pi/180)+80*math.sin(math.pi/180))

    def test_reference_encloses_every_original_point(self):
        points=[[10,10],[92,14],[85,51],[11,49]]
        r=m.reference_obb(points)
        p=m.project(points,r['angle_rad'])
        self.assertAlmostEqual(r['long_px']*r['short_px'],p['along_axis_px']*p['transverse_px'])

    @unittest.skipUnless(importlib.util.find_spec('cv2') is not None and importlib.util.find_spec('numpy') is not None,'Native OpenCV not installed locally')
    def test_native_exact_repo_function_matches_rectangle_and_reference(self):
        read,info=m.native_loader();r=read(self.points)
        self.assertEqual((r['long_px'],r['short_px']), (80,40))
        self.assertEqual(info['function_file_sha256'],m.NATIVE_SHA)
        points=[[10,10],[92,14],[85,51],[11,49]]
        native=read(points);ref=m.reference_obb(points)
        self.assertAlmostEqual(native['long_px'],ref['long_px'],places=4)
        self.assertAlmostEqual(native['short_px'],ref['short_px'],places=4)
        self.assertLess(m.angle_delta(native['angle_rad'],ref['angle_rad']),1e-4)


if __name__=='__main__':unittest.main()
