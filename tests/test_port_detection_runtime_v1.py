"""CPU contract checks; optional real-Torch pipeline checks use synthetic boxes."""
from copy import deepcopy
from contextlib import nullcontext
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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

    def test_numerical_settings_restore_sealed_determinism_for_all_arms(self):
        fake = SimpleNamespace(backends=SimpleNamespace(
            cudnn=SimpleNamespace(benchmark=True, deterministic=False, allow_tf32=False),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))))
        flags = b.numerical_settings(fake)
        self.assertEqual(flags, dict(cudnn_benchmark=False,cudnn_deterministic=True,
            cuda_matmul_allow_tf32=False,cudnn_allow_tf32=True))
        documented = {k:v for k,v in c.protocol_document()['numerical_settings'].items() if k != 'reason'}
        documented['cudnn_benchmark'] = c.protocol_document()['cudnn_benchmark']
        self.assertEqual(flags, documented)

    def test_score_mismatch_records_frame_exact_values_and_keeps_strict_gate(self):
        actual = [[50.,40.,20.,10.,.2,.80000001]]
        expected = [[50.,40.,20.,10.,.2,.8]]
        fake = SimpleNamespace(dtype='float32', tolist=lambda:actual)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'symeood_b image=frame.*actual='):
                b.verify_output(fake,expected,out,'symeood_b',dict(image='frame'),0,
                                'output_preflight',dict(cudnn_deterministic=True))
            report = c.read(out/'output_mismatch.json')
            self.assertEqual(report['actual'],actual)
            self.assertEqual(report['expected'],expected)
            self.assertEqual(report['score_policy'],'Exact; unchanged')
            self.assertGreater(report['component_absolute_errors'][0][5],0.)
            self.assertIn('tolerance=0', report['error'])

    def test_historical_fp32_compatibility_does_not_relax_same_run_scores_or_geometry(self):
        ref = [[50.,40.,20.,10.,.2,.992497980594635]]
        actual = deepcopy(ref); actual[0][5] = .9924980998039246
        c.check_boxes(actual,ref,historical_scores=True)
        with self.assertRaises(ValueError): c.check_boxes(actual,ref)
        actual[0][5] = ref[0][5]+2e-6
        with self.assertRaises(ValueError): c.check_boxes(actual,ref,historical_scores=True)
        actual = deepcopy(ref); actual[0][0] += .01
        with self.assertRaises(ValueError): c.check_boxes(actual,ref,historical_scores=True)
        actual = deepcopy(ref); actual[0][5] = 1.0000001
        with self.assertRaises(ValueError): c.check_boxes(actual,ref,historical_scores=True)
        with self.assertRaises(ValueError): c.check_boxes([],ref,historical_scores=True)
        with self.assertRaises(ValueError):
            c.check_boxes([[50.,40.,20.,10.,.2,.0500001]],
                          [[50.,40.,20.,10.,.2,.0499999]],historical_scores=True)

    def test_midpoint_preserves_native_scores_exactly_even_when_history_is_close(self):
        box = [[50.,40.,20.,10.,.2,.8]]
        c.check_midpoint_scores(box,[.8])
        c.check_midpoint_scores([],[])
        with self.assertRaises(ValueError): c.check_midpoint_scores(box,[.80000001])
        with self.assertRaises(ValueError): c.check_midpoint_scores(box,[])

    def test_full_output_preflight_checks_all_arms_and_all_frames_without_publishing_speed(self):
        rows = [dict(image='frame_%02d'%i,sequence='real_seq03') for i in range(31)]
        weights = {arm:None for arm in c.ARMS}
        torch = SimpleNamespace(no_grad=nullcontext,cuda=SimpleNamespace(empty_cache=lambda:None))
        cv2 = SimpleNamespace(imread=lambda path:object())
        current_arm = []; calls = {arm:0 for arm in c.ARMS}
        score = .9924980998039246; oldscore = .992497980594635
        class FakePipeline:
            def __init__(self,*args):
                self.arm = current_arm[-1]; self.native_scores = [score]
            def __call__(self,*args):
                calls[self.arm] += 1
                return SimpleNamespace(dtype='float32',tolist=lambda:[[50.,40.,20.,10.,.2,score]])
        def fakebuild(arm,*args):
            current_arm.append(arm); return object(),None,None,None,{}
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(b,
                build=fakebuild,Pipeline=FakePipeline,transform=lambda cfg:None,
                prepare=lambda *args:(object(),[]),state_sha=lambda model:'unchanged',
                expected_boxes=lambda *args:[[50.,40.,20.,10.,.2,oldscore]]):
            out = Path(tmp)
            proof,values = b.preflight_outputs(rows,{},weights,{},0,torch,None,cv2,None,out,{})
            self.assertEqual(set(proof['arms']),set(c.ARMS))
            for arm in c.ARMS:
                self.assertEqual(proof['arms'][arm]['verified_predictions'],31)
                self.assertEqual(calls[arm],62)  # all31 fixture warmup + all31 checks
                self.assertEqual(len(values[arm]),31)
                self.assertEqual(proof['arms'][arm]['historical_score_drift']['nonidentical_score_outputs'],31)
            self.assertFalse((out/'runtime_compare.json').exists())
            self.assertFalse((out/'timings.jsonl').exists())

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

    def test_selection_error_identifies_real_mismatch_without_claiming_epoch_is_wrong(self):
        s = dict(evidence_role='source_val_checkpoint_selection', selected_checkpoint='epoch_24',
            config_sha256='old_cfg', selected_path='/server/epoch_24.pth',
            all_checkpoints={'epoch_24':dict(checkpoint_sha256='weight')})
        checks = c.selection_checks(s, 'current_cfg', 'weight')
        self.assertEqual([key for key, value in checks.items() if not value['passed']], ['config_sha256'])
        self.assertTrue(checks['selected_checkpoint']['passed'])
        with self.assertRaisesRegex(ValueError, 'config_sha256.*current_cfg'):
            c.check_selection(s, 'current_cfg', 'weight')

    def test_missing_legacy_provenance_is_reported_and_not_silently_accepted(self):
        s = dict(selected_checkpoint='epoch_24', selected_path='/server/epoch_24.pth',
                 all_checkpoints={'epoch_24':{}})
        checks = c.selection_checks(s, 'cfg', 'weight')
        self.assertFalse(checks['evidence_role']['passed'])
        self.assertFalse(checks['config_sha256']['passed'])
        self.assertFalse(checks['checkpoint_record_sha256']['passed'])
        with self.assertRaises(ValueError): c.check_selection(s, 'cfg', 'weight')
        # Unknown current weight plus absent metadata must also remain a failure.
        self.assertFalse(c.selection_checks({}, 'cfg', None)['checkpoint_record_sha256']['passed'])

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
                protocol='crane_ckpt_sweep_final_test_v2', metric_protocol_version=2,
                config=str(ROOT/c.SYM_CONFIG), checkpoint=str(weight), results_pkl=str(predictions),
                config_sha256=config_sha, checkpoint_sha256=c.sha(weight), frame_count=1440,
                results_pkl_sha256=c.sha(predictions))
            c.write_new(final/'final_test_metrics_v2.json', receipt)
            refs, actual_weight, proof = b.checked_detector_inputs('symeood', model)
            self.assertEqual((len(refs), actual_weight, proof['selected_epoch']), (1440, weight, 20))
            # A B-config reference cannot silently become the no-B control.
            receipt['config_sha256'] = c.sha(ROOT/c.B_CONFIG)
            (final/'final_test_metrics_v2.json').write_text(json.dumps(receipt))
            with self.assertRaises(ValueError): b.checked_detector_inputs('symeood', model)

    def test_reviewed_archive_chain_matches_current_overrides(self):
        for arm in ('eood', 'symeood'):
            proof = c.archived_config_proof(ROOT, arm)
            self.assertTrue(proof['override_equivalence'])
            self.assertEqual(proof['archived_config_sha256'], c.MIGRATIONS[arm]['config_sha256'])
        # Even a small unreviewed dataset override must block compatibility.
        original = c.config_assignments
        with patch.object(c, 'config_assignments', wraps=original) as reader:
            def changed(path):
                values = original(path)
                if Path(path) == ROOT/c.SYM_CONFIG:
                    import ast
                    values['data_root'] = ast.parse("'other_dataset/'", mode='eval').body
                return values
            reader.side_effect = changed
            with self.assertRaisesRegex(ValueError, 'not equivalent'):
                c.archived_config_proof(ROOT, 'symeood')

    def test_migrated_reference_is_pinned_before_unpickling_and_keeps_old_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp); sweep = model/'val_sweep_port_v1'
            final = sweep/'final_test/epoch_20'; (final/'preds').mkdir(parents=True)
            weight = model/'epoch_20.pth'; weight.write_bytes(b'migration fixture only')
            predictions = final/'preds/results.pkl'
            with predictions.open('wb') as stream: pickle.dump([[] for _ in range(1440)], stream)
            old = ROOT/'work_dirs/crane_symeood_k1_port_day2night_seq06_v1'
            selection = dict(evidence_role='source_val_checkpoint_selection', selected_checkpoint='epoch_20',
                config_sha256=c.MIGRATIONS['symeood']['config_sha256'], selected_path=str(old/'epoch_20.pth'),
                all_checkpoints={'epoch_20':dict(checkpoint=str(old/'epoch_20.pth'), checkpoint_sha256=c.sha(weight))})
            c.write_new(sweep/'sweep_results.json', selection)
            receipt = dict(protocol='crane_ckpt_sweep_final_test_v2', metric_protocol_version=2,
                evidence_role='fixed_test_after_source_val_selection', frame_count=1440,
                config=str(ROOT/'crane_project/configs/crane_symeood_k1_port_day2night_seq06_v1.py'),
                config_sha256=selection['config_sha256'], checkpoint=str(old/'epoch_20.pth'),
                checkpoint_sha256=c.sha(weight), results_pkl=str(old/'val_sweep_port_v1/final_test/epoch_20/preds/results.pkl'),
                results_pkl_sha256=c.sha(predictions))
            receipt_path = final/'final_test_metrics_v2.json'; c.write_new(receipt_path, receipt)
            migration = dict(c.MIGRATIONS['symeood'], checkpoint_sha256=c.sha(weight),
                selection_sha256=c.sha(sweep/'sweep_results.json'), test_report_sha256=c.sha(receipt_path),
                reference_sha256=c.sha(predictions))
            with patch.dict(c.MIGRATIONS, symeood=migration):
                refs, actual_weight, proof = b.checked_detector_inputs('symeood', model)
                self.assertEqual((len(refs), actual_weight), (1440, weight))
                self.assertEqual(proof['migration']['mode'], 'reviewed_seq06_directory_migration')
                self.assertNotEqual(proof['recorded_config_sha256'], proof['config_sha256'])
                self.assertEqual(c.read(sweep/'sweep_results.json')['selected_path'], str(old/'epoch_20.pth'))
                for field in ('checkpoint_sha256', 'selection_sha256', 'test_report_sha256', 'reference_sha256'):
                    with patch.dict(migration, {field:'0'*64}), patch.object(b.pickle, 'load',
                            side_effect=AssertionError('Unpickle before validation')):
                        with self.assertRaisesRegex(ValueError, 'Unreviewed historical'):
                            b.checked_detector_inputs('symeood', model)
                with self.assertRaisesRegex(ValueError, 'Unreviewed historical'):
                    c.migration_proof(ROOT, 'symeood', c.MIGRATIONS['eood']['config_sha256'],
                        c.sha(weight), sweep/'sweep_results.json', receipt_path, predictions)

    def test_migration_paths_do_not_accept_basename_only_or_other_directories(self):
        old = ROOT/'work_dirs/crane_eood_k1_port_day2night_seq06_v1/epoch_24.pth'
        c.check_recorded_path(str(old), ROOT/'work_dirs/new/epoch_24.pth', ROOT, 'eood', {}, 'epoch_24.pth')
        for path in ('epoch_24.pth', '/elsewhere/epoch_24.pth', str(old).replace('eood_k1', 'symeood_k1')):
            with self.subTest(path=path), self.assertRaises(ValueError):
                c.check_recorded_path(path, old, ROOT, 'eood', {}, 'epoch_24.pth')

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

    def test_input_diagnosis_reports_legacy_fields_without_unpickling_or_modifying_sources(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'work_dirs')) as tmp:
            root = Path(tmp); model = root/'model'; sweep = model/'ckpt_sweep'
            sweep.mkdir(parents=True)
            weight = model/'epoch_24.pth'; weight.write_bytes(b'test fixture')
            selection = sweep/'sweep_results.json'
            c.write_new(selection, dict(selected_checkpoint='epoch_24', selected_path=str(weight),
                all_checkpoints={'epoch_24':dict(checkpoint=str(weight))}))
            final = sweep/'final_test/epoch_24'; (final/'preds').mkdir(parents=True)
            # This cannot be unpickled; diagnostic must only hash raw bytes.
            (final/'preds/results.pkl').write_bytes(b'not a pickle; hash only')
            before = c.sha(selection)
            with patch.object(b.pickle, 'load', side_effect=AssertionError('Unpickle forbidden')):
                with patch.multiple(b, EOOD_DIR=model, SYM_DIR=root/'absent_sym',
                                    B_WEIGHT=root/'absent_b.pth', M_WEIGHT=root/'absent_m.pth'):
                    args = b.parser().parse_args(['--stage','inputs','--out-dir',str(root/'diagnosis')])
                    b.run(args)
            report = c.read(root/'diagnosis/input_diagnosis.json')
            self.assertEqual(report['status'], 'INPUT_METADATA_DIAGNOSIS_COMPLETE_REVIEW_REQUIRED')
            result = report['input_diagnosis']
            for field in ('cuda_calls','model_loads','tensor_loads','prediction_unpickles','image_reads','annotation_reads'):
                self.assertEqual(result[field], 0)
            candidate = next(x for x in result['eood']['candidates'] if x['selection_exists'])
            self.assertEqual(set(candidate['failed_fields']),
                {'evidence_role','config_sha256','checkpoint_record_sha256'})
            self.assertTrue(candidate['selection_checks']['selected_checkpoint']['passed'])
            self.assertFalse(candidate['prediction_file']['unpickled'])
            self.assertEqual(c.sha(selection), before)

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
