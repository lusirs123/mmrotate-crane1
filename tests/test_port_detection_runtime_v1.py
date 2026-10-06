"""CPU contract checks; optional real-Torch pipeline checks use synthetic boxes."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_detection_runtime_v1 as c
from crane_project.tools import benchmark_port_detection_runtime_v1 as b


class ContractTests(unittest.TestCase):
    def test_fixed_identity_and_no_augmented_eood_baseline(self):
        p = c.protocol_document()
        self.assertEqual(p['arms'], ['eood', 'symeood', 'symeood_b', 'symeood_b_midpoint'])
        self.assertFalse(p['eood']['augmentation_b'])
        self.assertNotIn('aug_b', p['eood']['config'])
        self.assertFalse(p['symeood']['augmentation_b'])
        self.assertNotIn('aug_b', p['symeood']['config'])
        self.assertEqual(p['symeood']['selected_epoch'], 20)
        self.assertEqual(c.detector_identity('symeood')['config'], c.SYM_CONFIG)
        for arm in ('symeood_b', 'symeood_b_midpoint'):
            self.assertEqual(c.detector_identity(arm)['config'], c.B_CONFIG)
            self.assertEqual(c.detector_identity(arm)['selected_epoch'], 24)
        with self.assertRaises(ValueError): c.detector_identity('eood_b')
        self.assertEqual(p['midpoint']['sigma_cells'], 1.5)
        self.assertEqual(p['midpoint']['epoch'], 3)
        self.assertFalse(p['scope']['selection_on_test'])
        self.assertFalse(p['scope']['reliability_included'])
        self.assertEqual(p['warmup_frames'], 50)
        self.assertEqual(p['repeats'], 3)

    def test_fps_is_total_frames_over_total_time(self):
        s = c.timing_summary([.01, .09])
        self.assertAlmostEqual(s['fps'], 20.)
        self.assertAlmostEqual(s['mean_ms'], 50.)
        self.assertNotAlmostEqual(s['fps'], (100+1/.09)/2)

    def test_percentile_uses_linear_interpolation(self):
        s = c.timing_summary([.01, .02, .03, .04])
        self.assertAlmostEqual(s['p50_ms'], 25.)
        self.assertAlmostEqual(s['p95_ms'], 38.5)

    def test_invalid_timing_cannot_be_published(self):
        for values in ([], [0.], [-1.], [math.nan], [math.inf]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                c.timing_summary(values)

    def test_output_presence_and_score_are_protected(self):
        box = [[50., 40., 20., 10., .2, .8]]
        self.assertEqual(c.check_boxes(box, box), 0.)
        self.assertEqual(c.check_boxes([], []), 0.)
        with self.assertRaises(ValueError):
            c.check_boxes([], box)
        changed = deepcopy(box); changed[0][5] += 1e-8
        with self.assertRaises(ValueError):
            c.check_boxes(changed, box)

    def test_equivalent_axis_swap_cannot_silently_replace_raw_contract(self):
        with self.assertRaises(ValueError):
            c.check_boxes([[50., 40., 10., 20., .2+math.pi/2, .8]],
                          [[50., 40., 20., 10., .2, .8]])

    def test_geometry_tolerance_does_not_accept_a_different_output(self):
        ref = [[50., 40., 20., 10., .2, .8]]
        near = deepcopy(ref); near[0][0] += 1e-5
        c.check_boxes(near, ref)
        near[0][0] += .01
        with self.assertRaises(ValueError):
            c.check_boxes(near, ref)

    def test_nonfinite_or_nonpositive_boxes_rejected(self):
        for size in (0., -1., math.nan, math.inf):
            with self.subTest(size=size), self.assertRaises(ValueError):
                c.check_boxes([[1., 2., size, 5., 0., .8]], [[1., 2., 10., 5., 0., .8]])

    def test_existing_val_selection_required(self):
        s = dict(evidence_role='source_val_checkpoint_selection', selected_checkpoint='epoch_24',
            config_sha256='cfg', selected_path='/server/epoch_24.pth',
            all_checkpoints={'epoch_24':dict(checkpoint_sha256='weight')})
        c.check_selection(s, 'cfg', 'weight')
        for key, value in (('selected_checkpoint', 'epoch_22'), ('config_sha256', 'aug_b'),
                           ('evidence_role', 'selected_on_test')):
            other = deepcopy(s); other[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.check_selection(other, 'cfg', 'weight')

    def test_checkpoint_config_is_parsed_without_execution(self):
        keys = ('model', 'data', 'optimizer', 'optimizer_config', 'lr_config',
                'runner', 'load_from', 'resume_from')
        cfg = {key:None for key in keys}; cfg['model'] = dict(type='Eood')
        text = '\n'.join(key+' = '+repr(cfg[key]) for key in keys)
        meta = dict(config=text, epoch=24, seed=0)
        c.check_checkpoint_meta(meta, cfg)
        meta['config'] = text.replace("{'type': 'Eood'}", "__import__('os').system('false')")
        with self.assertRaises(ValueError):
            c.check_checkpoint_meta(meta, cfg)

    def test_unaugmented_symeood_keeps_its_val_epoch20(self):
        s = dict(evidence_role='source_val_checkpoint_selection', selected_checkpoint='epoch_20',
            config_sha256='cfg', selected_path='/server/epoch_20.pth',
            all_checkpoints={'epoch_20':dict(checkpoint_sha256='weight')})
        c.check_selection(s, 'cfg', 'weight', epoch=20)
        with self.assertRaises(ValueError): c.check_selection(s, 'cfg', 'weight', epoch=24)
        keys = ('model', 'data', 'optimizer', 'optimizer_config', 'lr_config',
                'runner', 'load_from', 'resume_from')
        cfg = {key:None for key in keys}
        meta = dict(config='\n'.join(key+' = None' for key in keys), epoch=20, seed=0)
        self.assertEqual(c.check_checkpoint_meta(meta, cfg, epoch=20)['epoch'], 20)
        with self.assertRaises(ValueError): c.check_checkpoint_meta(meta, cfg, epoch=24)

    def test_unaugmented_symeood_reference_is_bound_to_its_own_weight_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp); sweep = model/'ckpt_sweep'
            final = sweep/'final_test/epoch_20'; (final/'preds').mkdir(parents=True)
            weight = model/'epoch_20.pth'; weight.write_bytes(b'synthetic fixture only')
            predictions = final/'preds/results.pkl'
            with predictions.open('wb') as stream: pickle.dump([[] for _ in range(1440)], stream)
            config_sha = c.sha(ROOT/c.SYM_CONFIG)
            c.write_new(sweep/'sweep_results.json', dict(
                evidence_role='source_val_checkpoint_selection', selected_checkpoint='epoch_20',
                config_sha256=config_sha, selected_path=str(weight),
                all_checkpoints={'epoch_20':dict(checkpoint=str(weight), checkpoint_sha256=c.sha(weight))}))
            receipt = dict(evidence_role='fixed_test_after_source_val_selection',
                config_sha256=config_sha, checkpoint_sha256=c.sha(weight), frame_count=1440,
                results_pkl_sha256=c.sha(predictions))
            c.write_new(final/'final_test_metrics_v2.json', receipt)
            refs, actual_weight, proof = b.checked_detector_inputs('symeood', model)
            self.assertEqual((len(refs), actual_weight, proof['selected_epoch']), (1440, weight, 20))
            # A B-config reference cannot silently become the no-B control.
            receipt['config_sha256'] = c.sha(ROOT/c.B_CONFIG)
            (final/'final_test_metrics_v2.json').write_text(json.dumps(receipt))
            with self.assertRaises(ValueError): b.checked_detector_inputs('symeood', model)

    def test_directory_copies_must_be_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, d = Path(tmp)/'a', Path(tmp)/'d'
            a.mkdir(); d.mkdir()
            for p in (a, d): (p/'reference.json').write_text('{}')
            self.assertEqual(c.unique_directory([a, d], ['reference.json']), a.resolve())
            (d/'reference.json').write_text('{"different":true}')
            with self.assertRaises(ValueError):
                c.unique_directory([a, d], ['reference.json'])
            with self.assertRaises(FileNotFoundError):
                c.unique_directory([Path(tmp)/'missing'], ['reference.json'])

    def test_outputs_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'record.json'; c.write_new(p, dict(status='old'))
            with self.assertRaises(FileExistsError): c.write_new(p, dict(status='new'))
            self.assertEqual(c.read(p), dict(status='old'))

    def test_sources_match_and_static_check_needs_no_gpu_or_weights(self):
        self.assertEqual(b.source_contract()['protocol_sha256'], c.sha(b.PROTOCOL))
        with tempfile.TemporaryDirectory(dir=str(ROOT/'work_dirs')) as tmp:
            out = Path(tmp)/'new_static'
            b.run(b.parser().parse_args(['--stage', 'check', '--out-dir', str(out)]))
            result = c.read(out/'completion.json')
            self.assertEqual(result['status'], 'STATIC_RUNTIME_CONTRACT_PASS_NO_MODEL_DATA_GPU')
            self.assertFalse(result['gt_scoring'])
            with self.assertRaises(FileExistsError):
                b.run(b.parser().parse_args(['--stage', 'check', '--out-dir', str(out)]))

    def test_no_training_or_precision_parameter_sweep_cli(self):
        options = b.parser()._option_string_actions
        for name in ('--sigma', '--epoch', '--threshold', '--use-fp16', '--train', '--split'):
            self.assertNotIn(name, options)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'),
                     'Real-Torch synthetic pipeline checks require the server environment')
class TorchPipelineTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        from crane_project.utils import port_geometry_refine_g_v1 as geometry
        self.np, self.torch, self.g = np, torch, geometry
        self.meta = dict(scale_factor=np.asarray([.499, .5, .499, .5], dtype=np.float32),
            img_shape=(128, 128, 3), ori_shape=(256, 256, 3), pad_shape=(128, 128, 3), flip=False)
        self.image = torch.zeros(1, 3, 128, 128)

    def detector(self, empty=False):
        torch, np, geometry = self.torch, self.np, self.g
        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.extracts = 0; self.calls = 0; self.rescale_calls = 0
                self.raw = np.empty((0, 6), dtype=np.float32) if empty else np.asarray([[64., 64., 32., 16., .2, .8]], dtype=np.float32)
                self.register_buffer('frozen', torch.ones(1))
            def extract_feat(self, image):
                self.extracts += 1
                return (torch.zeros(1, 256, 16, 16),)
            def simple_test_from_features(self, features, metas, rescale=False):
                self.calls += 1
                boxes = torch.from_numpy(self.raw.copy())
                if rescale:
                    self.rescale_calls += 1
                    boxes = torch.cat((geometry.map_boxes(boxes[:, :5], metas[0], inverse=True), boxes[:, 5:]), 1)
                return [[boxes.numpy()]]
        return Fake().eval().requires_grad_(False)

    def test_one_backbone_one_native_call_and_separate_xy_restoration(self):
        d = self.detector(); before = b.state_sha(d)
        with self.torch.no_grad():
            result = b.Pipeline(d, None, self.torch, self.np, self.g)(self.image, [self.meta])
        self.assertEqual((d.extracts, d.calls), (1, 1))
        self.assertEqual(d.rescale_calls, 1)
        self.assertAlmostEqual(float(result[0, 0]), 64/.499, places=3)
        self.assertAlmostEqual(float(result[0, 1]), 128., places=4)
        self.assertEqual(b.state_sha(d), before)
        self.assertEqual(float(result[0, 5]), float(d.raw[0, 5]))

    def test_real_sigma_head_reuses_same_features_and_preserves_zero_identity(self):
        from crane_project.utils.port_geometry_midpoint_sigma_v1 import SigmaMidpointHead
        d = self.detector(); head = SigmaMidpointHead(1.5).eval().requires_grad_(False)
        before = b.state_sha(head)
        with self.torch.no_grad():
            result = b.Pipeline(d, head, self.torch, self.np, self.g)(self.image, [self.meta])
            raw = self.image.new_tensor(d.raw)
            ref = self.torch.cat((self.g.map_boxes(raw[:, :5], self.meta, inverse=True), raw[:, 5:]), 1).numpy()
        self.assertEqual((d.extracts, d.calls), (1, 1))
        self.assertEqual(d.rescale_calls, 0)
        self.assertTrue(self.np.array_equal(result, ref))
        self.assertEqual(b.state_sha(head), before)

    def test_empty_output_is_timed_without_adding_a_detection(self):
        from crane_project.utils.port_geometry_midpoint_sigma_v1 import SigmaMidpointHead
        d = self.detector(empty=True); head = SigmaMidpointHead(1.5).eval().requires_grad_(False)
        with self.torch.no_grad():
            result = b.Pipeline(d, head, self.torch, self.np, self.g)(self.image, [self.meta])
        self.assertEqual(result.shape, (0, 6))
        self.assertEqual((d.extracts, d.calls), (1, 1))

    def test_eood_native_multi_candidates_are_not_reduced_before_nms(self):
        np = self.np
        class EOOD:
            def __init__(self): self.calls = 0
            def simple_test(self, image, metas, rescale=False):
                self.calls += 1
                if not rescale: raise AssertionError('Requires native original output')
                return [[np.asarray([[1., 2., 20., 10., 0., .9], [3., 4., 18., 9., .1, .7]], dtype=np.float32)]]
        d = EOOD()
        result = b.Pipeline(d, None, self.torch, self.np, self.g)(self.image, [self.meta])
        self.assertEqual(result.shape, (2, 6)); self.assertEqual(d.calls, 1)


if __name__ == '__main__':
    unittest.main()
