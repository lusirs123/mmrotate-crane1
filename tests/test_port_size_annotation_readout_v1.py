import copy
import importlib.util
import math
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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

    def paired(self, points=None, k=2, endpoints=None, repeat=None):
        points = self.points if points is None else points
        record = dict(self.axis, axis_points=endpoints or [[10,30],[90,30]],
                      axis_reviewed=True, axis_source='manual_same_frame_long_axis',
                      reference_k=k, k_basis='same-object human reference')
        axis = m.axis_measurement(record,self.frame,points)
        return m.same_frame_pair(axis,points,m.reference_obb(points),record,repeat)

    def test_pair_equal_sizes_have_zero_discrepancy(self):
        p=self.paired()
        self.assertTrue(p['eligible'])
        self.assertEqual(p['primary']['long_relative'],0)
        self.assertEqual(p['primary']['short_relative'],0)
        self.assertEqual(p['primary']['short_log_decomposition']['total'],0)
        self.assertEqual(p['free_obb']['center_delta_px'],0)

    def test_fixed_k_discrepancy_and_long_extent_are_separated(self):
        points=[[10,14],[90,14],[90,46],[10,46]]
        p=self.paired(points)['primary']
        self.assertAlmostEqual(p['short_relative'],.25)
        self.assertEqual(p['long_relative'],0)
        self.assertEqual(p['short_log_decomposition']['long_extent_component'],0)
        self.assertAlmostEqual(p['short_log_decomposition']['projected_ratio_component'],math.log(1.25))
        # An axis with shorter extent can cancel the projected-ratio term.
        points=[[0,10],[100,10],[100,50],[0,50]]
        p=self.paired(points)['primary']; d=p['short_log_decomposition']
        self.assertAlmostEqual(p['long_relative'],-.2)
        self.assertAlmostEqual(p['short_relative'],0)
        self.assertAlmostEqual(d['long_extent_component'],math.log(.8))
        self.assertAlmostEqual(d['projected_ratio_component'],math.log(1.25))
        self.assertAlmostEqual(d['total'],p['short_log_ratio'])

    def test_pair_reversed_axis_and_point_order_preserve_dimensions(self):
        a=self.paired()['primary']
        b=self.paired(list(reversed(self.points)),endpoints=[[90,30],[10,30]])['primary']
        for key in ('axis_long_px','four_point_along_axis_px','four_point_transverse_px',
                    'long_relative','short_relative','short_log_ratio'):
            self.assertAlmostEqual(a[key],b[key])

    def test_pair_rotation_and_translation_preserve_signed_differences(self):
        theta=.23;c,s=math.cos(theta),math.sin(theta)
        def move(p):return [p[0]*c-p[1]*s+12,p[0]*s+p[1]*c+5]
        points=[move(p) for p in self.points]
        endpoints=[move(p) for p in [[10,30],[90,30]]]
        a=self.paired(k=2.5)['primary'];b=self.paired(points,k=2.5,endpoints=endpoints)['primary']
        self.assertAlmostEqual(a['short_relative'],b['short_relative'])
        self.assertAlmostEqual(a['long_relative'],b['long_relative'])

    def test_pending_axis_and_k_do_not_invent_short_evidence(self):
        axis=m.axis_measurement(self.axis,self.frame,self.points)
        p=m.same_frame_pair(axis,self.points,m.reference_obb(self.points),self.axis)
        self.assertFalse(p['eligible'])
        p=self.paired(k=None)
        self.assertTrue(p['eligible'])
        self.assertNotIn('short_relative',p['primary'])
        self.assertEqual(p['status'],'LONG_DIRECTION_COMPARISON_ONLY_K_UNCONFIRMED')

    def test_transverse_axis_is_excluded_from_long_identity_comparison(self):
        p=self.paired(endpoints=[[50,10],[50,50]])
        self.assertFalse(p['eligible'])
        self.assertEqual(p['status'],'AXIS_LONG_IDENTITY_CONFLICT_REVIEW_REQUIRED')

    def test_repeat_checks_observed_spread_without_truth_or_training_gate(self):
        p=self.paired(k=2.5,repeat=[[10,11],[90,11],[90,49],[10,49]])
        r=p['fixed_axis_repeat']
        self.assertAlmostEqual(r['short_relative'],-.05)
        self.assertTrue(r['short_gap_same_sign'])
        self.assertTrue(r['both_short_gaps_exceed_observed_repeat_difference'])
        self.assertTrue(r['no_systematic_bias_inference'])
        p=self.paired(k=2.5,repeat=[[10,16],[90,16],[90,44],[10,44]])
        self.assertFalse(p['fixed_axis_repeat']['short_gap_same_sign'])
        p=self.paired(repeat=self.points)
        self.assertTrue(p['fixed_axis_repeat']['points_identical_to_first'])
        self.assertFalse(p['fixed_axis_repeat']['short_gap_same_sign'])

    def test_video_groups_and_empty_values_are_not_pooled_truth(self):
        pending=m.same_frame_pair(m.axis_measurement(self.axis,self.frame,self.points),
                                 self.points,m.reference_obb(self.points),self.axis)
        rows=[dict(image='a',same_frame_pair=self.paired(k=2.5)),
              dict(image='b',same_frame_pair=self.paired(k=None)),
              dict(image='c',same_frame_pair=pending)]
        s=m.pair_summary(rows,{'a':'video08','b':'video08','c':'video11'})
        self.assertEqual((s['eligible_axis_count'],s['paired_k_count'],s['missing_axis_count']),(2,1,1))
        self.assertEqual(s['by_source_video']['video08']['short_relative']['negative'],1)
        self.assertIsNone(s['by_source_video']['video11']['short_relative']['mean'])
        self.assertFalse(s['supervision_only_training_contrast_ready'])
        self.assertTrue(s['systematic_bias_status'].startswith('NOT_ESTABLISHED'))
        self.assertEqual(m.descriptive([-.2,0,.1])['zero'],1)
        self.assertAlmostEqual(m.descriptive([-.2,0,.1])['abs_p95'],.19)

    def test_malformed_k_basis_rejected_cleanly(self):
        self.axis.update(axis_points=[[10,30],[90,30]],axis_reviewed=True,
                         axis_source='manual_same_frame_long_axis',reference_k=2,k_basis=4)
        with self.assertRaises(ValueError):m.axis_measurement(self.axis,self.frame,self.points)

    def test_imported_axis_source_binds_filename_image_and_exact_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'axis_annotations';folder.mkdir()
            path=folder/'f.json'
            document=dict(imagePath='../../images/f.png',imageWidth=120,imageHeight=100,
                shapes=[dict(label='axis',shape_type='line',points=[[10,30],[90,30]])])
            path.write_text(json.dumps(document))
            record=dict(self.axis,axis_points=[[10,30],[90,30]],
                axis_annotation_file='axis_annotations/f.json',axis_annotation_sha256=m.packet.sha(path))
            sources={};m.check_axis_annotation_source(record,self.frame,root,sources)
            self.assertEqual(sources[str(path.resolve())],record['axis_annotation_sha256'])
            for update in (dict(axis_points=[[11,30],[90,30]]),dict(axis_annotation_sha256='wrong')):
                with self.assertRaises(ValueError):
                    m.check_axis_annotation_source(dict(record,**update),self.frame,root,{})
            document['imageWidth']=121;path.write_text(json.dumps(document))
            record['axis_annotation_sha256']=m.packet.sha(path)
            with self.assertRaises(ValueError):m.check_axis_annotation_source(record,self.frame,root,{})
            record['axis_annotation_file']='../escaped.json'
            with self.assertRaises(ValueError):m.check_axis_annotation_source(record,self.frame,root,{})

    def test_paired_empty_inputs_stop_before_backend_or_report_creation(self):
        packet=ROOT/'annotation_materials/port_independent_size_labels_v1_20261006_final'
        if not packet.exists():self.skipTest('Local fixed packet not present')
        # Simulate the all-pending boundary without changing completed human inputs.
        out=ROOT/'work_dirs/port_size_annotation_readout_v1_preflight_test_no_output'
        with patch.object(m,'native_loader',side_effect=AssertionError('Backend must not load')), \
             patch.object(m,'axis_measurement',return_value=dict(status='PENDING_INDEPENDENT_SAME_FRAME_AXIS')):
            with self.assertRaisesRegex(ValueError,'SAME_FRAME_PAIRS_MISSING'):
                m.diagnose(packet,out,'native',require_axis=True)
        self.assertFalse(out.exists())

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
