import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('independent_size',
    ROOT/'tools/data/prepare_port_independent_size_labels_v1.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class IndependentLabelsTests(unittest.TestCase):
    def setUp(self):
        self.frame = dict(image='images/f.png', size=[100, 80])
        self.document = dict(imagePath='f.png', imageWidth=100, imageHeight=80,
            flags=dict(independent_reviewed=True, unmeasurable=False),
            object_definition='four visible corners of the same central face',
            measurement_basis='manual_visible_four_corners',
            shapes=[dict(shape_type='polygon', label=m.LABEL,
                         points=[[10, 10], [75, 12], [70, 40], [12, 45]])])

    def test_axis_agrees_with_generation_but_is_not_independent_short_edge(self):
        axis = dict(imageWidth=100, imageHeight=80,
            shapes=[dict(label='axis', shape_type='line', points=[[10, 20], [70, 20]])])
        result = m.axis_check(axis, '10 5 70 5 70 35 10 35 grab 0', 2, (100, 80))
        self.assertTrue(result['conversion_consistent'])
        self.assertFalse(result['independent_short_edge_measurement'])
        rotated = m.axis_check(axis, '25 -10 55 -10 55 50 25 50 grab 0', 2, (100, 80))
        self.assertFalse(rotated['conversion_consistent'])

    def test_axis_size_mismatch_and_nonfinite_are_rejected(self):
        axis = dict(imageWidth=100, imageHeight=80,
            shapes=[dict(label='axis', shape_type='line', points=[[10, 20], [70, 20]])])
        result = m.axis_check(axis, '10 0 70 0 70 40 10 40 grab 0', 2, (100, 80))
        self.assertFalse(result['conversion_consistent'])
        axis['shapes'][0]['points'][0][0] = float('nan')
        with self.assertRaises(ValueError): m.axis_check(axis, '10 5 70 5 70 35 10 35 grab 0', 2, (100, 80))

    def test_visible_projected_polygon_does_not_need_fixed_aspect_or_rectangle(self):
        result = m.validate_annotation(self.document, self.frame)
        self.assertTrue(result['independent_corner_observation'])
        reverse = copy.deepcopy(self.document)
        reverse['shapes'][0]['points'].reverse()
        self.assertEqual(result, m.validate_annotation(reverse, self.frame))

    def test_empty_or_unreviewed_is_pending_not_new_gt(self):
        self.document['flags']['independent_reviewed'] = False
        self.assertEqual(m.validate_annotation(self.document, self.frame)['status'], 'PENDING_REVIEW')
        self.document['shapes'] = []
        self.assertEqual(m.validate_annotation(self.document, self.frame)['status'], 'PENDING')

    def test_unmeasurable_needs_reason_and_cannot_guess_corners(self):
        self.document['flags']['unmeasurable'] = True
        self.document['review_note'] = 'Short side occluded; no guessed corner'
        with self.assertRaises(ValueError): m.validate_annotation(self.document, self.frame)
        self.document['shapes'] = []
        self.assertEqual(m.validate_annotation(self.document, self.frame)['status'], 'UNMEASURABLE')
        self.document['review_note'] = ''
        with self.assertRaises(ValueError): m.validate_annotation(self.document, self.frame)

    def test_crossed_concave_degenerate_nonfinite_and_outside_polygons_rejected(self):
        for points in ([[[10, 10], [70, 40], [75, 12], [12, 45]],
                        [[10, 10], [75, 12], [30, 20], [12, 45]],
                        [[10, 10], [10, 10], [75, 40], [12, 45]],
                        [[float('inf'), 10], [75, 12], [70, 40], [12, 45]],
                        [[-1, 10], [75, 12], [70, 40], [12, 45]]]):
            with self.subTest(points=points):
                self.document['shapes'][0]['points'] = points
                with self.assertRaises(ValueError): m.validate_annotation(self.document, self.frame)

    def test_axis_old_label_and_missing_object_definition_rejected(self):
        for key, value in (('shape_type', 'line'), ('label', 'grab')):
            doc = copy.deepcopy(self.document); doc['shapes'][0][key] = value
            with self.assertRaises(ValueError): m.validate_annotation(doc, self.frame)
        self.document['object_definition'] = ''
        with self.assertRaises(ValueError): m.validate_annotation(self.document, self.frame)

    def test_wrong_image_identity_dimensions_and_boolean_flags_rejected(self):
        for key, value in (('imagePath', 'wrong.png'), ('imageHeight', 79),
                           ('flags', dict(independent_reviewed='true'))):
            doc = copy.deepcopy(self.document); doc[key] = value
            with self.assertRaises(ValueError): m.validate_annotation(doc, self.frame)

    def test_frame_plan_predeclared_no_model_selection(self):
        self.assertEqual(m.select_frames(163, 4), [8, 57, 105, 154])
        self.assertEqual(len(set(m.select_frames(1104, 16))), 16)
        self.assertEqual(set(m.VIDEO_PLAN), {'1 (8).mp4', '1 (11).mp4'})

    def test_paths_cannot_overwrite_dataset_or_escape_packet(self):
        with self.assertRaises(ValueError): m.new_output(ROOT/'crane_project/data/new')
        with self.assertRaises(ValueError): m.owned(ROOT/'tests/example.json', ROOT/'work_dirs')
        with tempfile.TemporaryDirectory(prefix=m.VERSION+'_test_', dir=ROOT/'work_dirs') as tmp:
            with self.assertRaises(FileExistsError): m.new_output(tmp)

    def test_packet_check_image_and_manifest_tamper_no_promotion(self):
        with tempfile.TemporaryDirectory(prefix=m.VERSION+'_test_', dir=ROOT/'work_dirs') as tmp:
            out = Path(tmp); (out/'images').mkdir(); (out/'annotations').mkdir()
            image = out/'images/f.png'; image.write_bytes(b'immutable test image')
            frame = dict(self.frame, annotation='annotations/f.json', image_sha256=m.sha(image))
            m.write_json(out/'annotations/f.json', self.document)
            receipt = dict(protocol=m.VERSION, ready_for_training=False, split_assigned=False, frames=[frame])
            m.write_json(out/'packet.json', receipt)
            m.write_json(out/'artifacts.json', {'packet.json':m.sha(out/'packet.json')})
            result = m.check(out)
            self.assertFalse(result['ready_for_training'])
            self.assertFalse(result['split_assigned'])
            image.write_bytes(b'changed')
            with self.assertRaises(ValueError): m.check(out)
            image.write_bytes(b'immutable test image')
            receipt['split_assigned'] = True
            m.write_json(out/'packet.json', receipt)
            with self.assertRaises(ValueError): m.check(out)


if __name__ == '__main__': unittest.main()
