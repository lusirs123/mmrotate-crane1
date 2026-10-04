"""Independent CPU geometry, split, statistics, IO and stage-gate checks."""
from argparse import Namespace
from copy import deepcopy
import ast
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import importlib.util
import pickle
from types import SimpleNamespace
from contextlib import nullcontext

import numpy as np

from crane_project.tools import run_port_size_reference_v1 as entry
from crane_project.utils import port_size_reference_v1 as size

PROTOCOL = json.loads(entry.PROTOCOL.read_text())


def meta(scale=1., flip=False):
    w = int(256*scale)
    return dict(img_shape=[w, w, 3], ori_shape=[256, 256, 3], pad_shape=[256, 256, 3],
        scale_factor=[scale]*4, flip=flip, flip_direction='horizontal' if flip else None)


def reference(box=(128., 128., 80., 40., .37), transform=None):
    transform = transform or meta()
    target, _ = size.target_map(box, transform)
    return size.reference_from_map(target, box, transform, PROTOCOL)


class SizeReferenceTests(unittest.TestCase):
    def test_truncated_covariance_correction_has_independent_integral(self):
        t = np.linspace(-2., 2., 100001); w = np.exp(-t*t/2)
        numeric = np.sum((t[1:]**2*w[1:]+t[:-1]**2*w[:-1])/2)/np.sum((w[1:]+w[:-1])/2)
        self.assertAlmostEqual(size.TRUNCATED_VARIANCE, numeric, places=8)

    def test_ideal_sizes_are_correct_not_biased_by_gaussian_truncation(self):
        for theta in (0., .37, 1.1, -1.4):
            r = reference((128., 128., 80., 40., theta))
            self.assertTrue(r['defined'])
            self.assertLess(abs(r['long_original_px']/80-1), .04)
            self.assertLess(abs(r['short_original_px']/40-1), .04)

    def test_upscaled_grid_matches_align_corners_false_p3_coordinates(self):
        p, valid = size.grid(meta())
        self.assertEqual(p[0, 0].tolist(), [-3., -3.])
        self.assertFalse(valid[0, 0]); self.assertEqual(p[2, 2].tolist(), [1., 1.])

    def test_isotropic_scale_and_flip_restore_original_size(self):
        b = [128., 128., 80., 40., .3]
        for scale, flip in ((1., False), (.5, False), (.5, True)):
            r = reference(b, meta(scale, flip))
            self.assertTrue(r['defined'])
            self.assertLess(abs(r['long_original_px']/80-1), .05)
            self.assertLess(abs(r['short_original_px']/40-1), .05)
            self.assertLess(np.linalg.norm(np.asarray(r['center_original'])-b[:2]), 1.)

    def test_equivalent_width_height_and_pi_period_are_invariant(self):
        b = [128., 128., 80., 40., .3]; r = reference(b)
        expected = size.size_reading(b, r)
        for p in ([128., 128., 40., 80., .3+math.pi/2], [128., 128., 80., 40., .3+math.pi]):
            self.assertAlmostEqual(expected['risk'], size.size_reading(p, r)['risk'])

    def test_long_and_short_both_sides_detected_without_nuisance_response(self):
        row = dict(image='x', domain='real', sequence='r', gt=[128., 128., 80., 40., .3])
        r = entry.probe_record(row, reference(row['gt']), PROTOCOL)
        self.assertTrue(all(x > .3 for x in r['size_probe_risk_gaps'].values()))
        self.assertLess(r['nuisance_control_max_risk_difference'], 1e-12)
        result = entry.probe_summary([r])['all']
        self.assertEqual((result['correct_pairs'], result['long_both_sides'], result['short_both_sides']), (4, 1, 1))

    def test_flat_and_weak_maps_explicitly_unavailable(self):
        for value in (0., .01, .9):
            r = size.reference_from_map(np.full((128, 128), value), [128, 128, 80, 40, 0], meta(), PROTOCOL)
            self.assertFalse(r['defined']); self.assertIsNone(size.size_reading([0, 0, 10, 5, 0], r)['risk'])

    def test_unresolved_short_edge_is_unavailable(self):
        r = reference((128., 128., 80., 8., .3))
        self.assertFalse(r['defined']); self.assertEqual(r['reason'], 'unresolved_short_edge')

    def test_image_edge_truncation_does_not_become_trusted_size(self):
        r = reference((4., 128., 80., 40., 0.))
        self.assertFalse(r['defined']); self.assertEqual(r['reason'], 'context_clipped_or_contaminated')

    def test_invalid_and_anisotropic_maps_rejected(self):
        m = meta(); m['scale_factor'] = [.5, .8, .5, .8]
        with self.assertRaises(ValueError): size.grid(m)
        for array in (np.zeros((10, 10)), np.full((128, 128), np.nan), np.full((128, 128), 2.)):
            with self.assertRaises(ValueError): size.reference_from_map(array, [128, 128, 80, 40, 0], meta(), PROTOCOL)

    def test_soft_loss_preserves_fractional_targets_and_padding_is_excluded(self):
        target = np.array([[.2, .7, 0., 0.]])
        optimal = np.array([[math.log(.2/.8), math.log(.7/.3), -20., 100.]])
        mask = np.array([[True, True, True, False]])
        self.assertLess(size.loss_numpy(optimal, target, mask), 1e-15)
        self.assertGreater(size.loss_numpy(np.zeros_like(optimal), target, mask), .1)
        other = optimal.copy(); other[0, 3] = -100.
        self.assertEqual(size.loss_numpy(optimal, target, mask), size.loss_numpy(other, target, mask))

    def test_fit_split_never_uses_labels_and_preserves_guard(self):
        counts = dict(real_seq01=339, real_seq05=560, real_seq06=466, real_seq12=141, real_seq13=304, sim_seq08=748)
        rows = [dict(image=s+'_%05d' % i, sequence=s, frame_id=i, gt=None) for s, n in counts.items() for i in range(n)]
        split = size.partition(rows, PROTOCOL)
        self.assertEqual((len(split['fit']), len(split['holdout']), len(split['guard'])), (384, 432, 32))
        self.assertFalse(any(r['sequence'] == 'real_seq13' for r in split['fit']))
        fit_sim = [r['frame_id'] for r in split['fit'] if r['sequence'] == 'sim_seq08']
        held_sim = [r['frame_id'] for r in split['holdout'] if r['sequence'] == 'sim_seq08']
        self.assertEqual(min(held_sim)-max(fit_sim)-1, 32)
        changed = deepcopy(rows)
        for r in changed: r['gt'] = 'changed labels'
        self.assertEqual([r['image'] for r in split['fit']], [r['image'] for r in size.partition(changed, PROTOCOL)['fit']])

    def test_missing_unavailable_and_matched_count_denominators(self):
        model = dict(weights=[0., -1., 0., 0.], mean=[0., 0., 0.], scale=[1., 1., 1.])
        rows = []
        for i in range(4):
            p = [128, 128, 80 if i != 1 else 50, 40, 0, .8] if i != 3 else None
            rows.append(dict(image='r_%05d' % i, domain='real', sequence='r', gt=[128, 128, 80, 40, 0],
                pred=p, image_size=[256, 256], reading=dict(defined=i < 2, risk=.1+i*.2 if i < 2 else None),
                reference_size_error=0. if i < 2 else None))
        stats = entry.summary(rows, {'models': {'size': model}}, PROTOCOL, {})['all']
        self.assertEqual((stats['outputs'], stats['reference_defined'], stats['unavailable_good']), (3, 2, 1))
        self.assertEqual(stats['output_coverage'], .75)
        self.assertEqual(stats['center_correct_on_outputs'], 1.)
        self.assertEqual(stats['all_frame_center_correct_coverage'], .75)
        point = stats['coverage_points'][0]
        self.assertEqual(point['actual_count'], 2)
        for metrics in point['matched_count_methods'].values(): self.assertEqual(metrics['correct_accepted']+metrics['incorrect_accepted'], 2)
        self.assertEqual(point['score_tie_bounds']['boundary_tie_size'], 3)

    def test_error_distribution_does_not_clip_large_reference_errors(self):
        self.assertEqual(entry.error_distribution([.1, 3.])['max'], 3.)

    def test_stage_gate_is_exact_and_rejects_wrong_or_missing_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'r.json'; contract = {'fixed': 1}
            entry.write(p, dict(status='PASS', contract=contract, contract_sha256=entry.simple.fingerprint(contract)))
            entry.write(Path(directory)/'completion.json', dict(status='PASS', contract_sha256=entry.simple.fingerprint(contract),
                artifacts={'r.json': entry.parent.sha(p)}))
            self.assertEqual(entry.gate(p, 'PASS', contract), entry.parent.sha(p))
            with self.assertRaises(ValueError): entry.gate(p, 'OTHER', contract)
            with self.assertRaises(ValueError): entry.gate(p, 'PASS', {'fixed': 2})
            with self.assertRaises(FileExistsError): entry.write(p, {})

    def test_checkpoint_uses_stream_and_rejects_wrong_roles_and_modified_bytes(self):
        def save(value, stream):
            self.assertTrue(hasattr(stream, 'write'))
            pickle.dump(value, stream)
        def load(path, map_location):
            self.assertEqual(map_location, 'cpu')
            with open(path, 'rb') as stream: return pickle.load(stream)
        torch = SimpleNamespace(save=save, load=load)
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', {'torch': torch}):
            p = Path(directory)/'epoch.pth'; contract = {'fixed': 1}
            payload = dict(protocol=size.VERSION, contract=contract, role='smoke_discarded', epoch=1, state={'weights': [1.]})
            entry.save_checkpoint(p, payload)
            self.assertEqual(entry.load_checkpoint(p, contract, 'smoke_discarded'), payload)
            with self.assertRaises(ValueError): entry.load_checkpoint(p, contract, 'reference_train')
            with self.assertRaises(ValueError): entry.load_checkpoint(p, {'fixed': 2}, 'smoke_discarded')
            with self.assertRaises(FileExistsError): entry.save_checkpoint(p, payload)
            p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaises(ValueError): entry.load_checkpoint(p, contract, 'smoke_discarded')

    def test_incomplete_and_failed_stage_cannot_release_training(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'r.json'; contract = {'fixed': 1}
            entry.write(p, dict(status='PASS', contract=contract, contract_sha256=entry.simple.fingerprint(contract)))
            with self.assertRaises(ValueError): entry.gate(p, 'PASS', contract)
            entry.write(Path(directory)/'completion.json', dict(status='PASS', contract_sha256=entry.simple.fingerprint(contract),
                artifacts={'r.json': entry.parent.sha(p)}))
            entry.write(Path(directory)/'failure.json', {'error': 'failed'})
            with self.assertRaises(ValueError): entry.gate(p, 'PASS', contract)

    def test_python38_syntax_and_no_test_or_quality_fit_in_new_entries(self):
        for p in (entry.ROOT/'crane_project/utils/port_size_reference_v1.py',
                  entry.ROOT/'crane_project/utils/port_size_reference_v1_torch.py', Path(entry.__file__)):
            ast.parse(p.read_text(), feature_version=(3, 8))
        self.assertFalse(PROTOCOL['test_read']); self.assertFalse(PROTOCOL['quality_head_trained'])
        self.assertEqual(PROTOCOL['optimizer_steps'], 384*4)

    def test_assessment_route_keeps_missing_boxes_and_writes_frozen_evidence(self):
        class Tensor:
            def __init__(self, values): self.values = np.asarray(values)
            def sigmoid(self): return self
            def __getitem__(self, index): return Tensor(self.values[index])
            def cpu(self): return self
            def numpy(self): return self.values
        class Model:
            def load_state_dict(self, state, strict): pass
            def eval(self): pass
            def __call__(self, feature): return Tensor(feature[None, None])
        fake_torch = SimpleNamespace(no_grad=nullcontext,
            cuda=SimpleNamespace(max_memory_allocated=lambda gpu: 0, max_memory_reserved=lambda gpu: 0))
        detector = object(); model = Model(); contract = {'fixed': 1}; gates = {'ideal': 'x'}
        def row(image, pred, split):
            return dict(image=image, sequence=image.rsplit('_', 1)[0], domain='real', split=split,
                gt=[128., 128., 80., 40., .3], pred=pred, image_size=[256, 256])
        prediction = [128., 128., 80., 40., .3, .8]
        held = [row('real_seq13_00000', prediction, 'train'), row('real_seq13_00001', None, 'train')]
        val = [row('real_seq07_00000', prediction, 'val')]
        policy = {'models': {'size': dict(weights=[0., -1., 0., 0.], mean=[0.]*3, scale=[1.]*3)}}
        prepared = (PROTOCOL, {}, [], val, policy, {}, {'holdout': held}, {}, contract, {})
        def view(source, *args):
            target, _ = size.target_map(source['gt'], meta())
            return [target], meta(), [meta()]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint = root/'epoch_04.pth'; checkpoint.write_bytes(b'fixture')
            out = root/'assessment'; out.mkdir()
            report = dict(status='FIXED_EPOCH4_SIZE_REFERENCE_TRAIN_COMPLETE_REVIEW_REQUIRED', contract=contract,
                contract_sha256=entry.simple.fingerprint(contract), steps=1536, gates=gates,
                checkpoint_sha256=entry.parent.sha(checkpoint), b_state_before='b', b_state_after='b', save_reload_logits_exact=True)
            entry.write(root/'train_report.json', report)
            entry.write(root/'completion.json', dict(status=report['status'], contract_sha256=entry.simple.fingerprint(contract),
                artifacts={'train_report.json': entry.parent.sha(root/'train_report.json')}))
            saved = dict(epoch=4, steps=1536, gates=gates, b_state='b', state={})
            args = Namespace(reference_checkpoint=checkpoint, out_dir=out, gpu=0)
            guards = SimpleNamespace(assert_detector_frozen=lambda detector: None)
            with patch.object(entry, 'runtime', return_value=(fake_torch, detector, None, model)), \
                 patch.object(entry, 'view', side_effect=view), patch.object(entry, 'load_checkpoint', return_value=saved), \
                 patch.object(entry.parent, 'state_digest', side_effect=lambda obj: 'b' if obj is detector else 'r'), \
                 patch.dict('sys.modules', {'crane_project.utils.port_structure_reliability_v1': guards}):
                status = entry.assess(args, prepared)
            self.assertEqual(status, 'FROZEN_SIZE_REFERENCE_ASSESSMENT_COMPLETE_REVIEW_REQUIRED')
            actual = json.loads((out/'assessment.json').read_text())
            self.assertEqual(actual['summaries']['train_holdout']['all']['outputs'], 1)
            self.assertEqual(actual['summaries']['train_holdout']['all']['frames'], 2)
            self.assertEqual(actual['summaries']['train_holdout']['probe_diagnostics']['all']['short_both_sides'], 1)
            records = [json.loads(x) for x in (out/'reference_rows.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 3)
            self.assertIsNone(records[1]['pred']); self.assertIsNone(records[1]['reading']['risk'])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'server-only Torch checks; local Torch unavailable')
class TorchReferenceTests(unittest.TestCase):
    def test_numpy_torch_loss_and_actual_image_gradients(self):
        import torch
        from crane_project.utils.port_size_reference_v1_torch import SizeReference, reference_loss
        torch.manual_seed(1701)
        model = SizeReference(); features = torch.randn(1, 256, 8, 8)
        transform = dict(img_shape=[64, 64, 3], ori_shape=[64, 64, 3], pad_shape=[64, 64, 3], scale_factor=[1.]*4, flip=False)
        t, v = size.target_map([32, 32, 30, 16, .2], transform)
        target = torch.tensor(t)[None, None]; valid = torch.tensor(v, dtype=target.dtype)[None, None]
        z = model(features); self.assertEqual(z.shape, target.shape)
        loss = reference_loss(z, target, valid)
        self.assertAlmostEqual(float(loss), size.loss_numpy(z.detach().numpy()[0, 0], t, v), places=5)
        loss.backward()
        self.assertGreater(float(model.stem[0].weight.grad.norm()), 0)
        self.assertGreater(float(model.output.weight.grad.norm()), 0)
        self.assertFalse(features.requires_grad)
        with self.assertRaises(ValueError): model(features.requires_grad_(True))

    def test_real_torch_checkpoint_reload_preserves_logits_and_source_role(self):
        import torch
        from crane_project.utils.port_size_reference_v1_torch import SizeReference
        torch.manual_seed(1701); model = SizeReference().eval(); x = torch.randn(1, 256, 8, 8)
        contract = {'fixed': 1}
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'epoch.pth'
            entry.save_checkpoint(p, dict(protocol=size.VERSION, contract=contract, role='smoke_discarded', epoch=1, state=model.state_dict()))
            restored = SizeReference().eval(); restored.load_state_dict(entry.load_checkpoint(p, contract, 'smoke_discarded')['state'])
            self.assertTrue(torch.equal(model(x), restored(x)))


if __name__ == '__main__': unittest.main()
