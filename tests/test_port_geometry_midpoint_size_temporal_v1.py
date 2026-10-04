"""Residual meaning, temporal boundaries, VAL provenance and read-only gates."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crane_project.tools import analyze_port_geometry_midpoint_size_temporal_v1 as a


def row(seq, frame, gt_short=10., b_factor=1., midpoint_factor=1., missing=False, angle=0.):
    gt = [50., 50., gt_short*2, gt_short, angle]
    b = None if missing else [50., 50., gt_short*2*b_factor, gt_short*b_factor, angle, .8]
    m = None if missing else [50., 50., gt_short*2*midpoint_factor, gt_short*midpoint_factor, angle, .8]
    def metric(box):
        if box is None:
            return dict(output=False, center_hit=False, riou=0., angle_penalty_reason='no_output')
        factor = box[3]/gt_short
        return dict(output=True, center_hit=True, riou=.8, center_error_px=0., angle_error_deg=0.,
            short_edge_relative_error=abs(factor-1), long_edge_relative_error=abs(factor-1),
            short_edge_signed_log_ratio=math.log(factor), long_edge_signed_log_ratio=math.log(factor),
            protocol_angle_error_deg=0., angle_penalty_reason=None)
    return dict(image=seq+'_%05d' % frame, sequence=seq, domain=seq.split('_')[0], frame_id=frame,
        scale=1., gt=gt, b=b, midpoint=m, candidate=m, accepted=None if missing else True,
        candidate_valid=None if missing else True, failed_checks=[],
        metrics={'b': metric(b), 'midpoint': metric(m)})


def bundle():
    rs = [row(seq, i) for seq, n in a.VAL_COUNTS.items() for i in range(n)]
    frames = [a.static_frame(r) for r in rs]
    pairs, _ = a.temporal_pairs(frames)
    groups = {'overall': frames}
    groups.update({name: [r for r in frames if r['domain'] == name or r['sequence'] == name]
                   for name in ('real', 'sim')+tuple(a.VAL_COUNTS)})
    comparison = {'split': 'val', 'groups': {}}
    overall_metrics = {}
    for name, subset in groups.items():
        summary = a.group_summary(subset, [p for p in pairs if p['sequence'] in {r['sequence'] for r in subset}])
        domain = subset[0]['domain'] if name != 'overall' else None
        if name in ('real', 'sim'):
            overall_metrics.update({domain+'/TDR_w10(%)': 100., domain+'/R_center(%)': 100.,
                domain+'/MCML_max(frames)': 0})
        comparison['groups'][name] = {}
        for method in a.METHODS:
            hits = len(subset)
            comparison['groups'][name][method] = dict(
                output_coverage_fraction=dict(numerator=hits, denominator=hits),
                conditional_center_correct_fraction=dict(numerator=hits, denominator=hits),
                all_frame_center_correct_fraction=dict(numerator=hits, denominator=hits),
                short_edge_relative_error={'mean': 0.}, long_edge_relative_error={'mean': 0.},
                metric_protocol_v2=({domain+'/DFR(%/frame)': 0., domain+'/ACI': 1.} if domain else {}))
    overall_metrics.update({'sim/ACI': 1., 'sim/A-RMSE(deg)': 0.})
    for method in a.METHODS:
        comparison['groups']['overall'][method]['metric_protocol_v2'] = overall_metrics.copy()
    source_path = a.ROOT/'crane_project/tools/port_geometry_midpoint_formal_v1_sources.json'
    protocol_path = a.ROOT/'crane_project/tools/port_geometry_midpoint_formal_v1_protocol.json'
    identity = dict(sources=a.json_read(source_path.read_bytes())['sources'],
        sources_sha256=a.sha(source_path.read_bytes()), protocol_sha256=a.sha(protocol_path.read_bytes()),
        frozen_b=a.json_read(protocol_path.read_bytes())['frozen_b'])
    config, select = a.existing_selection_rule()
    entries = {}
    for epoch in range(1, 25):
        name = 'head_epoch_%02d.pth' % epoch
        checkpoint = dict(path=name, epoch=epoch, updates=epoch, sha256='b'*64,
                          head_digest={'synthetic': True})
        entries[name] = dict(checkpoint=checkpoint, metrics=overall_metrics.copy())
    chosen, _, info = select(entries, config)
    selection = dict(protocol=a.TRAIN_VERSION, split='val', test_access=False, selection_on_test=False,
        identity=identity, selected_checkpoint=entries[chosen]['checkpoint'], selection_config=config,
        all_checkpoints=entries, selection_info=info)
    completion = dict(status='FORMAL_MIDPOINT_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED', epochs_completed=24,
                      detector_updates=0, test_access=False, selection=selection)
    def encoded(x):
        return json.dumps(x, allow_nan=False).encode()
    files = {'training/selection.json': encoded(selection), 'training/completion.json': encoded(completion),
             'training/selected_val_compare.json': encoded(comparison),
             'training/val_epoch_01.rows.jsonl': b'\n'.join(encoded(r) for r in rs)}
    for epoch in range(1, 25):
        files['training/val_epoch_%02d.json' % epoch] = encoded(comparison)
    artifact = {name.split('/', 1)[1]: a.sha(data) for name, data in files.items()}
    artifact.update({name: 'b'*64 for name in entries})
    files['training/artifacts.json'] = encoded({'files': artifact})
    return files, rs, comparison


class ResidualTests(unittest.TestCase):
    def test_equivalent_box_and_scale(self):
        first = a.geometry([1., 2., 20., 10., .2])
        second = a.geometry([1., 2., 10., 20., .2-math.pi/2])
        for key in first:
            self.assertAlmostEqual(first[key], second[key])
        original = row('real_seq07', 0, b_factor=.8, midpoint_factor=.9)
        scaled = deepcopy(original)
        for key in ('gt', 'b', 'midpoint', 'candidate'):
            # Candidate/delivered lists deliberately alias in this fixture.
            scaled[key] = [v*.5 for v in original[key][:4]] + original[key][4:]
        x, y = a.static_frame(original), a.static_frame(scaled)
        self.assertEqual(x['b']['relative_error'], y['b']['relative_error'])
        self.assertEqual(x['midpoint']['log_ratio'], y['midpoint']['log_ratio'])

    def test_smaller_dfr_can_track_gt_worse(self):
        rs = [row('real_seq07', 0), row('real_seq07', 1, gt_short=12., midpoint_factor=10/12)]
        frames = [a.static_frame(r) for r in rs]
        pairs, _ = a.temporal_pairs(frames)
        v = a.group_summary(frames, pairs)
        self.assertAlmostEqual(v['b']['dfr_pct_per_frame'], 20.)
        self.assertAlmostEqual(v['midpoint']['dfr_pct_per_frame'], 0.)
        self.assertAlmostEqual(v['b']['temporal']['short']['log_increment_error']['rmse'], 0.)
        self.assertGreater(v['midpoint']['temporal']['short']['log_increment_error']['rmse'], .18)

    def test_static_bias_does_not_imply_temporal_error(self):
        frames = [a.static_frame(row('real_seq07', i, gt_short=10+i, b_factor=.8, midpoint_factor=.9))
                  for i in range(3)]
        pairs, _ = a.temporal_pairs(frames)
        value = a.group_summary(frames, pairs)
        self.assertAlmostEqual(value['b']['static']['short']['log_ratio']['mean'], math.log(.8))
        self.assertAlmostEqual(value['b']['temporal']['short']['log_increment_error']['rmse'], 0.)
        for p in pairs:
            self.assertAlmostEqual(p['midpoint']['short']['log_increment_error']-p['b']['short']['log_increment_error'],
                                   p['log_correction_change']['short'])

    def test_missing_gap_and_sequence_boundaries(self):
        rs = [row('real_seq07', 0), row('real_seq07', 1, missing=True), row('real_seq07', 2),
              row('real_seq07', 4), row('real_seq14', 5), row('real_seq14', 6)]
        pairs, support = a.temporal_pairs([a.static_frame(r) for r in rs])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['sequence'], 'real_seq14')
        self.assertEqual(support['missing_output_pairs'], 2)
        self.assertEqual(support['sequence_or_gap_boundaries'], 2)

    def test_angle_periodicity(self):
        frames = [a.static_frame(row('sim_seq10', i, angle=math.radians(angle)))
                  for i, angle in enumerate((89, -89))]
        pairs, _ = a.temporal_pairs(frames)
        self.assertAlmostEqual(pairs[0]['b']['angle_increment_deg'], 2.)
        self.assertAlmostEqual(pairs[0]['b']['angle_increment_error_deg'], 0.)
        self.assertGreater(pairs[0]['b']['aci'], .94)

    def test_fraction_empty_and_constant_correlation(self):
        self.assertIsNone(a.fraction(0, 0)['pct'])
        self.assertIsNone(a.correlation([1, 1], [2, 3]))
        self.assertAlmostEqual(a.correlation([1, 2], [4, 2]), -1.)
        self.assertIsNone(a.stats([])['rmse'])

    def test_analysis_does_not_mutate_inputs(self):
        rs = [row('real_seq07', 0, b_factor=.8, midpoint_factor=.9)]
        previous = deepcopy(rs)
        a.group_summary([a.static_frame(r) for r in rs], [])
        self.assertEqual(rs, previous)

    def test_nonfinite_and_nonpositive_box_rejected(self):
        for bad in ([0., 0., 2., 0., 0.], [0., 0., 2., 1., float('nan')]):
            with self.assertRaises(ValueError):
                a.geometry(bad)
        with self.assertRaises(ValueError):
            a.json_read(b'{"v": NaN}')


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files, cls.rows, cls.comparison = bundle()

    def test_valid_only_selected_rows_and_no_test_access(self):
        opened = []
        def read(name):
            opened.append(name)
            return self.files[name]
        rs, comparison, proof = a.validate_inputs(read)
        self.assertEqual(len(rs), 887)
        self.assertEqual(proof['selected_checkpoint']['epoch'], 1)
        self.assertEqual(sum(n.endswith('.rows.jsonl') for n in opened), 1)
        self.assertFalse(any('test' in n or n.endswith('.pth') for n in opened))
        summary, _, pairs = a.analyze(rs, comparison)
        self.assertEqual(len(pairs), 884)
        self.assertEqual(summary['groups']['real']['output_coverage']['numerator'], 375)

    def test_artifact_sha_tampering_rejected(self):
        files = dict(self.files)
        files['training/val_epoch_01.rows.jsonl'] += b' '
        with self.assertRaisesRegex(ValueError, 'artifact SHA'):
            a.validate_inputs(files.__getitem__)

    def test_forged_original_selection_rejected(self):
        files = dict(self.files)
        selection = json.loads(files['training/selection.json'])
        selection['selection_info']['feasible'] = 23
        files['training/selection.json'] = json.dumps(selection).encode()
        completion = json.loads(files['training/completion.json'])
        completion['selection'] = selection
        files['training/completion.json'] = json.dumps(completion).encode()
        with self.assertRaisesRegex(ValueError, 'Original full-VAL selection'):
            a.validate_inputs(files.__getitem__)

    def test_independent_online_val_exact_replay_and_tampering(self):
        files = dict(self.files)
        selection = json.loads(files['training/selection.json'])
        report = dict(split='val', status='FROZEN_FORMAL_MIDPOINT_VAL_COMPLETE_REVIEW_REQUIRED',
            test_access=False, selection_on_test=False, head_updates=0, detector_updates=0,
            frames=887, feature_extractions=887, native_head_calls=2661,
            identity={'training_identity': selection['identity']},
            state_before={'head': selection['selected_checkpoint']['head_digest']},
            state_after={'head': selection['selected_checkpoint']['head_digest']},
            selection=dict(selected_checkpoint=selection['selected_checkpoint'],
                selection_sha256=a.sha(files['training/selection.json']),
                completion_sha256=a.sha(files['training/completion.json'])),
            data_identity={'annotation_sha256': selection['identity']['frozen_b']['annotation_sha256']})
        files['online/completion.json'] = json.dumps(report).encode()
        files['online/val_compare.json'] = files['training/selected_val_compare.json']
        files['online/val_rows.jsonl'] = files['training/val_epoch_01.rows.jsonl']
        def artifacts():
            return json.dumps({'files': {name.split('/', 1)[1]: a.sha(data)
                for name, data in files.items() if name.startswith('online/') and
                not name.endswith('artifacts.json')}}).encode()
        files['online/artifacts.json'] = artifacts()
        _, _, proof = a.validate_inputs(files.__getitem__, online=True)
        self.assertTrue(proof['independent_online_val']['exact_rows_and_summary'])
        report['test_access'] = True
        files['online/completion.json'] = json.dumps(report).encode()
        files['online/artifacts.json'] = artifacts()
        with self.assertRaisesRegex(ValueError, 'online VAL identity'):
            a.validate_inputs(files.__getitem__, online=True)

    def test_test_frames_and_wrong_scale_rejected(self):
        for field, value in [('sequence', 'real_seq03'), ('scale', .5)]:
            rs = deepcopy(self.rows)
            rs[0][field] = value
            with self.assertRaises(ValueError):
                a.checked_rows(rs, self.comparison)

    def test_coverage_denominator_and_score_changes_rejected(self):
        changed = deepcopy(self.comparison)
        changed['groups']['real']['b']['conditional_center_correct_fraction']['denominator'] = 887
        with self.assertRaisesRegex(ValueError, 'denominator'):
            a.checked_rows(self.rows, changed)
        rs = deepcopy(self.rows)
        rs[0]['midpoint'][5] = .7
        with self.assertRaisesRegex(ValueError, 'Score'):
            a.checked_rows(rs, self.comparison)

    def test_static_cli_no_gpu_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'static'
            args = a.parser().parse_args(['--check-only', '--out-dir', str(out)])
            with patch.object(a, 'validate_inputs', return_value=(self.rows, self.comparison, {})), \
                    patch.object(a, 'analyze', side_effect=AssertionError('No analysis in static stage')):
                a.run(args)
                report = json.loads((out/'completion.json').read_text())
                self.assertFalse(report['test_access'])
                self.assertEqual(report['inference_calls'], 0)
                self.assertEqual(set(p.name for p in out.iterdir()), {'completion.json', 'artifacts.json'})
                with self.assertRaises(FileExistsError):
                    a.run(args)


if __name__ == '__main__':
    unittest.main()
