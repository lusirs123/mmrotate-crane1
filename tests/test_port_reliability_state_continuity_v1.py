"""State semantics and temporal mistakes that generic availability hides."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from crane_project.utils import port_reliability_state_continuity_v1 as c
from crane_project.tools import review_port_reliability_state_continuity_v1 as runner


def row(i, bad=False, missing=False, seq='seq', split='val'):
    return dict(image=seq+'_'+str(i), sequence=seq, frame_id=i, split=split,
                gt=[0., 0., 100., 20., 0.],
                pred=None if missing else [0., 0., 130. if bad else 100., 20., 0., .5])


class StateContinuityTests(unittest.TestCase):
    def test_four_states(self):
        self.assertEqual([c.size_state(row(0, b), a) for b, a in
                          [(True, True), (False, False), (True, False), (False, True)]],
                         ['FA', 'FR', 'ED', 'CR'])

    def test_missing_not_rejected_good_or_detected_bad(self):
        s = c.summarize([row(0, missing=True)], set())
        self.assertEqual(s['states'], dict(FA=0, FR=0, ED=0, CR=0, MISSING=1))
        self.assertIsNone(s['accepted_error_fraction'])
        with self.assertRaises(ValueError): c.size_state(row(0, missing=True), True)

    def test_correct_error_rejection_not_correct_observation_loss(self):
        s = c.summarize([row(i, bad=True) for i in range(5)], set())
        self.assertEqual(s['runs']['error_detection']['longest'], 5)
        self.assertEqual(s['runs']['correct_rejection']['longest'], 0)
        self.assertEqual(s['runs']['unavailable']['longest'], 5)

    def test_mixed_unavailable_composition(self):
        s = c.summarize([row(0, bad=True), row(1), row(2, missing=True)], set())
        run = s['runs']['unavailable']['longest_segments'][0]
        self.assertEqual(run['length'], 3)
        self.assertEqual(run['state_counts'], dict(FA=0, FR=1, ED=1, CR=0, MISSING=1))

    def test_missing_and_error_reset_FR_runs(self):
        s = c.summarize([row(0), row(1, missing=True), row(2), row(3, bad=True), row(4)], set())
        self.assertEqual(s['runs']['correct_rejection']['longest'], 1)

    def test_frame_gaps_and_sequences_reset(self):
        s = c.summarize([row(0), row(2), row(0, seq='next')], set())
        self.assertEqual(s['runs']['correct_rejection']['longest'], 1)

    def test_sorted_and_immutable(self):
        rows = [row(2), row(0), row(1)]; before = deepcopy(rows)
        self.assertEqual(c.summarize(rows, set())['runs']['correct_rejection']['longest'], 3)
        self.assertEqual(rows, before)

    def test_false_acceptance_runs(self):
        rows = [row(0, bad=True), row(1, bad=True), row(2)]
        s = c.summarize(rows, {r['image'] for r in rows})
        self.assertEqual(s['runs']['incorrect_acceptance']['longest'], 2)
        self.assertEqual(s['runs']['correct_retention']['longest'], 1)

    def test_size_swap_and_boundary(self):
        r = row(0); r['pred'][2:4] = [22., 110.]
        self.assertEqual(c.size_state(r, True), 'CR')
        r['pred'][3] = 110.01
        self.assertEqual(c.size_state(r, True), 'FA')

    def test_duplicate_and_test_rejected(self):
        with self.assertRaises(ValueError): c.summarize([row(0), row(0)], set())
        with self.assertRaises(ValueError): c.summarize([row(0, split='test')], set())

    def test_original_sim_train_split_preserved(self):
        r = row(0, split='train_sim'); r['reliability_role'] = 'train'
        self.assertEqual(c.summarize([r], set())['states']['FR'], 1)
        r['reliability_role'] = 'val'
        with self.assertRaises(ValueError): c.summarize([r], set())

    def test_flag_switches_exclude_gaps(self):
        rows = [row(0), row(1), row(3)]
        s = c.summarize(rows, {'seq_0', 'seq_3'})
        self.assertEqual(s['adjacent_observed_pairs'], 1)
        self.assertEqual(s['flag_switches'], 1)

    def test_same_count_ties_do_not_select_by_GT(self):
        rows = [row(0, bad=True), row(1)]
        risks = {n: {r['image']: .5 for r in rows} for n in ('full_simple', 'score_only')}
        s = c.compare(rows, {'seq_1'}, risks)
        self.assertEqual(s['same_count_controls']['score_only']['states']['FA'], 1)
        self.assertTrue(s['same_count_controls']['score_only']['partial_boundary_tie']['continuity_order_sensitive'])

    def test_invalid_edges_and_unknown_acceptance(self):
        r = row(0); r['pred'][2] = float('nan')
        with self.assertRaises(ValueError): c.size_state(r, True)
        with self.assertRaises(ValueError): c.summarize([row(0)], {'unknown'})

    def test_runner_refuses_existing_result(self):
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            runner.run(runner.ROOT/'work_dirs', runner.ROOT/'tests')

    def test_runner_enforces_project_directory(self):
        with self.assertRaisesRegex(ValueError, 'work directory'):
            runner.run(Path('/tmp/outside_evidence'), runner.ROOT/'work_dirs/new_review')

    def test_runner_refuses_tampered_receipt_before_output(self):
        with tempfile.TemporaryDirectory(dir=str(runner.ROOT/'work_dirs')) as directory:
            parent = Path(directory); source = parent/'fit'; source.mkdir()
            for name in ('fit_report.json', 'candidate_policies.json'):
                (source/name).write_text(json.dumps(dict(selected_arm=None, test_read=False)))
            (source/'scored_TRAIN_VAL.jsonl.gz').write_bytes(b'not opened before hash check')
            receipt = dict(protocol='port_reliability_feature_ablation_v1', selected_arm=None,
                           test_read=False, artifacts={n: '0'*64 for n in
                           ('fit_report.json', 'candidate_policies.json', 'scored_TRAIN_VAL.jsonl.gz')})
            (source/'completion.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                runner.run(source, parent/'out')
            self.assertFalse((parent/'out').exists())


if __name__ == '__main__': unittest.main()
