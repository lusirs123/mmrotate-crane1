"""CPU/stdlib fixtures for TRAIN-only log review and evidence boundaries."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from crane_project.tools.analyze_port_shape_e_h_train_logs_v1 import (
    NAMES, SHAPE, analyze, comparison, discover, metadata_review)


def meta(arm='e_h'):
    model = dict(bbox_head=dict(loss_bbox=dict(type='SymKLDLoss', loss_weight=2)))
    if arm == 'e_h':
        model['bbox_head']['shape_compensation'] = SHAPE
    return dict(seed=0, exp_name=NAMES[arm]+'.py', config=(
        'model = '+repr(model)+'\nrunner = dict(type="EpochBasedRunner", max_epochs=24)\n'
        'optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))\n'
        'work_dir = '+repr('work_dirs/'+NAMES[arm])))


def row(epoch=17, iteration=50, shape=.01, grad=9.):
    return dict(mode='train', epoch=epoch, iter=iteration, lr=.00025,
                loss_cls=.2, loss_bbox=.1, loss_shape_compensation=shape,
                aux0_loss_cls=.02, aux0_loss_bbox=.03, loss=.35+shape,
                grad_norm=grad, shape_positive_count=40)


class LogReview(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def log(self, rows, name='run.log.json', arm='e_h', header=True):
        path = self.root/name
        items = ([meta(arm)] if header else []) + rows
        path.write_text('\n'.join(json.dumps(r) for r in items)+'\n')
        return path

    def test_train_only_weighted_shape_and_auxiliary_loss(self):
        path = self.log([row(), dict(mode='val', epoch=17, iter=887,
                                    loss_bbox=999, grad_norm=float('nan'))])
        result = analyze([path], 'e_h')
        self.assertEqual(len(result['trajectory']), 1)
        m = result['windows']['late_17_24']['metrics']
        self.assertAlmostEqual(m['logged_weighted_shape_to_main_kld_ratio']['median'], .1)
        self.assertAlmostEqual(m['aux_classification_sum']['median'], .02)
        self.assertAlmostEqual(m['logged_total_minus_component_sum']['median'], 0)
        self.assertEqual(result['nonfinite_or_nonnumeric_events'], [])
        self.assertEqual(result['sources'][0]['skipped_modes']['val'], 1)

    def test_grad_windows_not_actual_clip_frequency(self):
        # These two logged means could each hide some per-step norms above10.
        result = analyze([self.log([row(grad=9), row(iteration=100, grad=11)])], 'e_h')
        window = result['windows']['late_17_24']
        self.assertEqual(window['logged_windows_mean_preclip_norm_gt10_fraction'], .5)
        self.assertIn('NOT per-step', window['interpretation'])
        self.assertNotIn('clip_multiplier', window)
        self.assertAlmostEqual(window['metrics']['grad_norm']['p90'], 10.8)

    def test_nonfinite_missing_and_zero_denominator_preserved(self):
        r = row(shape=float('nan'), grad=float('inf'))
        r['loss_bbox'] = 0
        del r['loss_cls']
        result = analyze([self.log([r])], 'e_h')
        # shape=NaN also makes the logged total loss NaN in this fixture.
        self.assertEqual(len(result['nonfinite_or_nonnumeric_events']), 3)
        self.assertIsNone(result['trajectory'][0]['metrics']['loss_shape_compensation'])
        self.assertIsNone(result['trajectory'][0]['metrics']['logged_weighted_shape_to_main_kld_ratio'])
        self.assertEqual(result['windows']['late_17_24']['required_field_missing_or_nonfinite']['loss_cls'], 1)
        self.assertTrue(result['data_review_required'])
        json.dumps(result, allow_nan=False)

    def test_wrong_identity_and_coefficient_rejected(self):
        for change in ('seed', 'exp_name', 'config'):
            m = meta()
            if change == 'seed':
                m['seed'] = 1
            elif change == 'exp_name':
                m['exp_name'] = NAMES['b']+'.py'
            else:
                m['config'] = m['config'].replace("'loss_weight': 0.05", "'loss_weight': 0.25")
            with self.subTest(change=change), self.assertRaises(ValueError):
                metadata_review(m, 'e_h')

    def test_no_config_execution(self):
        marker = self.root/'executed'
        m = meta()
        m['config'] = 'model = __import__("pathlib").Path('+repr(str(marker))+').touch()'
        review = metadata_review(m, 'e_h')
        self.assertTrue(review['identity_review_required'])
        self.assertFalse(marker.exists())

    def test_overlapping_and_restarted_logs_rejected(self):
        a = self.log([row()], 'a.log.json')
        b = self.log([row()], 'b.log.json')
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            analyze([a, b], 'e_h')
        with self.assertRaisesRegex(ValueError, 'Non-increasing'):
            analyze([self.log([row(), row(iteration=25)], 'restart.log.json')], 'e_h')

    def test_disjoint_log_segments_sort_and_report_sources(self):
        a = self.log([row(epoch=18)], 'a.log.json')
        b = self.log([row(epoch=17)], 'b.log.json')
        result = analyze([a, b], 'e_h')
        self.assertEqual(result['epoch_coverage'], [17, 18])
        self.assertEqual(result['trajectory'][0]['epoch'], 17)
        self.assertEqual(len(result['sources']), 2)

    def test_ambiguous_discovery_and_absent_optional_b(self):
        self.log([row()], 'a.log.json')
        self.log([row()], 'b.log.json')
        with self.assertRaisesRegex(ValueError, 'Multiple logs'):
            discover(None, self.root, True)
        self.assertEqual(discover(None, self.root/'absent', False), [])
        with self.assertRaises(FileNotFoundError):
            discover(None, self.root/'absent', True)

    def test_partial_epoch_coverage_is_not_completion(self):
        result = analyze([self.log([row(epoch=24)])], 'e_h')
        self.assertEqual(result['missing_epochs'], list(range(1, 24)))
        self.assertTrue(result['data_review_required'])
        self.assertEqual(result['windows']['epoch_1']['logged_records'], 0)

    def test_optional_b_is_descriptive_not_paired(self):
        r = row()
        r.pop('loss_shape_compensation')
        r['loss'] = .35
        b = analyze([self.log([r], 'b.log.json', arm='b')], 'b')
        e = analyze([self.log([row()])], 'e_h')
        c = comparison({'b': b, 'e_h': e})
        self.assertTrue(c['available'])
        self.assertEqual(c['fixed_epoch_windows']['late_17_24']['loss_bbox']['e_h_to_b_median_ratio'], 1)
        self.assertFalse(comparison({'e_h': e})['available'])

    def test_bad_json_and_no_train_fail(self):
        path = self.root/'broken.log.json'
        path.write_text(json.dumps(meta())+'\n{bad}\n')
        with self.assertRaisesRegex(ValueError, 'invalid JSON'):
            analyze([path], 'e_h')
        with self.assertRaisesRegex(ValueError, 'No TRAIN'):
            analyze([self.log([dict(mode='val', epoch=24, iter=887)])], 'e_h')

    def test_cli_full_24_epochs_finite_report_and_no_overwrite(self):
        path = self.log([row(epoch=ep) for ep in range(1, 25)])
        out = self.root/'report.json'
        script = Path(__file__).resolve().parents[1]/'crane_project/tools/analyze_port_shape_e_h_train_logs_v1.py'
        cmd = [sys.executable, str(script), '--e-log', str(path), '--no-b', '--out-json', str(out)]
        run = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(out.read_text())
        self.assertFalse(report['arms']['e_h']['data_review_required'])
        self.assertEqual(report['arms']['e_h']['windows']['late_17_24']['logged_records'], 8)
        before = out.read_bytes()
        rerun = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(rerun.returncode, 0)
        self.assertEqual(before, out.read_bytes())


if __name__ == '__main__':
    unittest.main()
