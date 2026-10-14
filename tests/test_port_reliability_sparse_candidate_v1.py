import copy
import math
import unittest
from crane_project.utils import port_reliability_sparse_candidate_v1 as core


def row(image='real_seq01_00000', pred=None):
    p = pred or [0., 0., 100., 20., 0., .8]
    n = dict(index=0, box=[0., 0., 110., 22., 0., .7])
    r = dict(image=image, pred=p, gt=[0., 0., 100., 20., 0.], image_size=[1000, 1000],
        sequence='real_seq01', domain='real', frame_id=0, split='train', reliability_role='train',
        temporal=dict(available=True, primary=.01), original_simple_decision=dict(size_accepted=True),
        risks=dict(partial_auc4=.2, crossfit_fusion=.3, full_simple=.2, fusion4=.2, score_only=.2),
        candidate_pool=[n['box'], p], candidate_chosen=1, candidate_evidence=dict(neighbors=[n]))
    old = copy.deepcopy(r)
    old['fallback_decision'] = True
    return r, old


class SparseAuditTests(unittest.TestCase):
    cuts = dict(supported=.5317310050227115, fallback=.4094748402105834)

    def test_signed_edges_and_axis_swap(self):
        r, _ = row()
        a, _ = core.local_features(r['pred'], r['candidate_evidence']['neighbors'])
        r['pred'][2:4] = [20., 100.]
        b, _ = core.local_features(r['pred'], r['candidate_evidence']['neighbors'])
        self.assertEqual(a, b)
        self.assertAlmostEqual(a['neighbor1_log_L'], math.log(1.1))

    def test_opposing_neighbors_not_collapsed(self):
        r, _ = row()
        n = r['candidate_evidence']['neighbors'] + [dict(index=2, box=[0., 0., 90., 18., 0., .6])]
        x, a = core.local_features(r['pred'], n)
        self.assertGreater(x['neighbor1_log_L'], 0)
        self.assertLess(x['neighbor2_log_L'], 0)
        self.assertEqual(a, ['mixed_or_zero'] * 2)

    def test_GT_never_changes_features_and_no_mutation(self):
        r, old = row()
        original = copy.deepcopy((r, old))
        a = core.join([r], [old], self.cuts)
        self.assertEqual((r, old), original)
        r['gt'][2] = old['gt'][2] = 200.
        b = core.join([r], [old], self.cuts)
        self.assertEqual(a[0]['features'], b[0]['features'])
        self.assertNotEqual(a[0]['states'], b[0]['states'])

    def test_fallback_uses_own_cutoff(self):
        r, old = row()
        for v in (r, old):
            v['temporal']['available'] = False
            v['risks']['partial_auc4'] = v['risks']['crossfit_fusion'] = .45
        old['fallback_decision'] = False
        a = core.join([r], [old], self.cuts)
        self.assertEqual(a[0]['states']['fallback_decision'], 'FR')

    def test_role_and_duplicate_rejected(self):
        r, old = row()
        with self.assertRaises(ValueError): core.join([r, r], [old, old], self.cuts)
        r['reliability_role'] = 'val'
        with self.assertRaises(ValueError): core.join([r], [old], self.cuts)

    def test_identity_and_self_rejected(self):
        r, old = row()
        r['candidate_chosen'] = 0
        with self.assertRaises(ValueError): core.join([r], [old], self.cuts)
        r, old = row()
        old['pred'][0] += .1
        with self.assertRaises(ValueError): core.join([r], [old], self.cuts)

    def test_null_empty_class_and_ties(self):
        self.assertIsNone(core.pairs([], [1])['AUC_higher'])
        self.assertEqual(core.pairs([1], [1])['AUC_higher'], .5)
        self.assertEqual(core.pairs([2], [1])['AUC_lower'], 0.)
        self.assertIsNone(core.distribution([])['median'])

    def test_ten_percent_boundary_matches_original_label(self):
        r, _ = row(pred=[0., 0., 110., 18., 0., .8])
        self.assertFalse(core.wrong(r))
        r['pred'][2] = 110.000001
        self.assertTrue(core.wrong(r))

    def test_full_rows_all_four_states(self):
        rows = []
        for i, (bad, accept) in enumerate(((True, True), (True, False), (False, True), (False, False))):
            r, old = row('real_seq01_%05d' % i)
            r['gt'][2] = old['gt'][2] = 200. if bad else 100.
            for v in (r, old): v['risks']['crossfit_fusion'] = .2 if accept else .8
            old['fallback_decision'] = accept
            rows += core.join([r], [old], self.cuts)
        a = core.analyze(rows)
        self.assertEqual(a['groups']['all']['fallback_decision']['any'], dict(FA=1, FR=1, ED=1, CR=1, MISSING=0))
        self.assertEqual(len(a['remaining_FA']), 1)

    def test_missing_has_separate_state(self):
        r, old = row()
        for v in (r, old):
            v['pred'] = None
            v['risks'] = {k: None for k in v['risks']}
            v['candidate_pool'] = []
            v['candidate_evidence']['neighbors'] = []
            v['candidate_chosen'] = None
        old['fallback_decision'] = False
        x = core.join([r], [old], self.cuts)
        self.assertTrue(all(s == 'MISSING' for s in x[0]['states'].values()))

    def test_no_neighbors_empty_signals(self):
        r, old = row()
        r['candidate_evidence']['neighbors'] = []
        a = core.analyze(core.join([r], [old], self.cuts))
        f = a['signals']['fallback_decision']['all']['features']['neighbor1_log_L']
        self.assertEqual(f['conditional_pairs'], 0)
        self.assertIsNone(f['conditional_AUC_higher'])


if __name__ == '__main__': unittest.main()
