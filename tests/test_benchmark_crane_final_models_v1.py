"""Identity, timing denominators, and the single-pass final inference path."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crane_project.tools import benchmark_crane_final_models_v1 as b


class FinalEfficiencyContracts(unittest.TestCase):
    def test_only_three_fixed_models_and_test_images_without_runtime_imports(self):
        before = set(sys.modules)
        cfg, _ = b.contract()
        refs = b.references(cfg)
        paths = b.images(cfg, [r['image'] for r in refs['final']])
        self.assertEqual(tuple(refs), ('final', 'eood', 'faa'))
        self.assertEqual(len(paths), 1440)
        self.assertEqual(sum(r['box'] is not None for r in refs['eood']), 1410)
        self.assertEqual(sum(r['box'] is not None for r in refs['faa']), 1429)
        self.assertEqual(sum(r['box'] is not None for r in refs['final']), 1440)
        self.assertTrue(all(set(r) == {'image', 'box'} for rows in refs.values() for r in rows))
        added = set(sys.modules)-before
        self.assertNotIn('torch', added)
        self.assertFalse(any('train_symeood' in name for name in added))

    def test_epoch_change_and_module_arm_cannot_enter_the_main_comparison(self):
        original = b.read(b.local(b.DEFAULT_CONFIG))
        for kind in ('epoch', 'arm', 'split'):
            cfg = deepcopy(original)
            if kind == 'epoch': cfg['models']['faa']['weights'][0]['epoch'] = 22
            elif kind == 'arm': cfg['models']['R4_only'] = deepcopy(cfg['models']['final'])
            else: cfg['split'] = 'val'
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'bad.json'; path.write_text(json.dumps(cfg))
                with self.assertRaisesRegex(ValueError, 'epoch|three-model'):
                    b.contract(str(path))

    def test_source_drift_and_workdir_inheritance_fail(self):
        cfg = b.read(b.local(b.DEFAULT_CONFIG)); target = next(iter(cfg['source_sha256']))
        original = b.frozen.sha
        with patch.object(b.frozen, 'sha', side_effect=lambda p:'0'*64 if p==b.local(target) else original(p)):
            with self.assertRaisesRegex(ValueError, 'SHA differs'):
                b.contract()
        with self.assertRaisesRegex(ValueError, 'stay in configs'):
            b.config_closure(b.ROOT/'work_dirs/config.py')

    def test_missing_detection_is_not_zero_latency_or_an_excluded_frame(self):
        stats = b.timing_summary([.01, .09])
        self.assertAlmostEqual(stats['mean_ms'], 50.)
        self.assertAlmostEqual(stats['fps'], 20.)
        self.assertAlmostEqual(stats['p95_ms'], 86.)
        for values in ([], [0.], [-1.], [math.nan]):
            with self.assertRaises(ValueError): b.timing_summary(values)
        self.assertEqual(b.compare_top1(None, None), 0.)
        with self.assertRaisesRegex(ValueError, 'presence'):
            b.compare_top1(None, [1., 2., 3., 2., 0., .9])

    def test_score_and_geometry_drift_are_not_silently_accepted(self):
        box = [50., 40., 20., 10., .2, .8]
        self.assertEqual(b.compare_top1(box, box), 0.)
        changed = box[:]; changed[0] += .1
        with self.assertRaisesRegex(ValueError, 'geometry'): b.compare_top1(changed, box)
        changed = box[:]; changed[5] += .01
        with self.assertRaisesRegex(ValueError, 'score'): b.compare_top1(changed, box)
        changed = box[:]; changed[5] += 1e-8
        with self.assertRaisesRegex(ValueError, 'score'): b.compare_top1(changed, box, score_atol=0.)

    def test_one_feature_head_and_sampling_per_final_frame(self):
        counts = dict.fromkeys(('features', 'detector_head', 'sampling', 'midpoint', 'lvsr'), 1440)
        b.verify_single_pass(counts, 1440)
        for key in counts:
            altered = counts.copy(); altered[key] += 1
            with self.assertRaisesRegex(ValueError, 'repeated/skipped'):
                b.verify_single_pass(altered, 1440)

    def test_default_does_not_start_gpu_or_select_new_weights(self):
        args = b.parser().parse_args([])
        self.assertEqual(args.stage, 'check')
        self.assertIsNone(args.gpu)
        self.assertIsNone(args.out_dir)
        self.assertFalse(args.require_weights)

    def test_stage_profile_reuses_same_test_and_frozen_prefixes(self):
        cfg, _ = b.contract(b.STAGE_CONFIG)
        refs = b.references(cfg)
        self.assertTrue(cfg['module_efficiency'])
        self.assertEqual(tuple(refs), ('r4','r4_m','final'))
        self.assertEqual([len(s['weights']) for s in cfg['models'].values()], [1,2,3])
        self.assertEqual([s['reference_key'] for s in cfg['models'].values()], ['b','midpoint','edge_residual'])
        self.assertEqual(len(b.images(cfg, [r['image'] for r in refs['r4']])), 1440)
        for rows in refs.values():
            self.assertEqual([r['image'] for r in rows], [r['image'] for r in refs['final']])
            self.assertTrue(all(set(r)=={'image','box'} for r in rows))

    def test_stage_cannot_load_a_different_head_or_detector_background(self):
        original = b.read(b.local(b.STAGE_CONFIG))
        for kind in ('weight', 'background', 'mode'):
            cfg = deepcopy(original)
            if kind == 'weight': cfg['models']['r4_m']['weights'][1]['sha256'] = '0'*64
            elif kind == 'background': cfg['models']['r4']['config'] = 'other.py'
            else: cfg['module_efficiency'] = False
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'bad.json'; path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError): b.contract(str(path))

    def test_inactive_stage_must_have_zero_calls(self):
        bare = dict(features=1,detector_head=1,sampling=0,midpoint=0,lvsr=0)
        mid = dict(features=1,detector_head=1,sampling=1,midpoint=1,lvsr=0)
        b.verify_single_pass(bare, 1, 1); b.verify_single_pass(mid, 1, 2)
        for counts,depth in [(bare,1),(mid,2)]:
            wrong = counts.copy(); wrong['lvsr'] = 1
            with self.assertRaisesRegex(ValueError, 'repeated/skipped'):
                b.verify_single_pass(wrong, 1, depth)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'),
                     'CPU Torch/NumPy runtime required; stdlib contract checks still run')
class FinalForwardCPU(unittest.TestCase):
    def test_fast_path_equals_audited_path_without_extra_detector_calls(self):
        import numpy as np
        import torch
        from crane_project.utils import port_geometry_refine_g_v1 as geometry
        from crane_project.utils.port_geometry_midpoint_sigma_v1 import SigmaMidpointHead
        from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
        torch.set_num_threads(1)
        class Detector(torch.nn.Module):
            def __init__(self, empty):
                super().__init__(); self.empty = empty; self.extractions = 0; self.head_calls = 0
            def extract_feat(self, image):
                self.extractions += 1
                return (image.new_zeros(1, 256, 8, 8),)
            def simple_test_from_features(self, features, metas, rescale=False):
                self.head_calls += 1
                boxes = np.empty((0, 6), dtype=np.float32) if self.empty else np.array(
                    [[30., 30., 20., 10., .1, .9]], dtype=np.float32)
                if rescale: boxes[:, :4] /= np.asarray(metas[0]['scale_factor'], dtype=np.float32)
                return [[boxes]]
        meta = dict(ori_shape=(128,128,3), img_shape=(64,64,3), pad_shape=(64,64,3),
                    scale_factor=np.array([.5,.5,.5,.5], dtype=np.float32), flip=False)
        for empty in (False, True):
            with self.subTest(no_output=empty), torch.no_grad():
                detector, mid, lvsr = Detector(empty), SigmaMidpointHead(1.5), PairSpecificSizeHead()
                for model in (detector, mid, lvsr): model.eval().requires_grad_(False)
                p = SimpleNamespace(torch=torch, geometry=geometry, detector=detector, midpoint=mid, lvsr=lvsr,
                    models=lambda:(detector,mid,lvsr),
                    counts=dict(feature_extractions=0,native_head_calls=0,midpoint_calls=0,lvsr_calls=0))
                image = torch.zeros(1, 3, 64, 64)
                audited = b.frozen.FrozenCranePipeline.predict(p, image, [meta])
                detector.extractions = detector.head_calls = 0
                for depth,key in [(1,'b'),(2,'midpoint'),(3,'edge_residual')]:
                    detector.extractions = detector.head_calls = 0
                    calls = dict(mid=0,lvsr=0)
                    handles = [mid.register_forward_pre_hook(lambda *a:calls.__setitem__('mid',calls['mid']+1)),
                               lvsr.register_forward_pre_hook(lambda *a:calls.__setitem__('lvsr',calls['lvsr']+1))]
                    with patch.object(geometry,'sample_local',wraps=geometry.sample_local) as sampler:
                        fast = b.FinalForward(p,depth); actual = fast(image, [meta]); fast.verify_invariants()
                        self.assertEqual(sampler.call_count,int(depth>=2))
                    for h in handles: h.remove()
                    self.assertEqual(detector.extractions, 1)
                    self.assertEqual(detector.head_calls, 1)
                    self.assertEqual(calls,dict(mid=int(depth>=2),lvsr=int(depth==3)))
                    b.verify_single_pass(fast.calls, 1, depth)
                    value = None if len(actual)==0 else actual[0].tolist()
                    b.compare_top1(value, audited[key], score_atol=0.)


if __name__ == '__main__': unittest.main()
