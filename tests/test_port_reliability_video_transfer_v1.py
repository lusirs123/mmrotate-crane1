import unittest
from copy import deepcopy
import numpy as np
from crane_project.utils import port_reliability_video_transfer_v1 as c


def row(i, seq='v0', error=False, risk=.3, available=True):
    # Both classes and nonconstant geometry/score; labels are genuine box errors.
    p = [0., 0., 112. if error else 100., 40., 0., .6+i*.001]
    return dict(image='%s_%05d' % (seq, i), sequence=seq, frame_id=i, domain='real',
        split='train', reliability_role='train', image_size=[1000, 600], pred=p,
        gt=[0., 0., 100., 40., 0.], original_simple_decision={'size_accepted': True},
        temporal=dict(available=available, primary=.01*i if available else None, acquisition_boundary=False),
        risks=dict(static3=risk, fusion4=risk, full_simple=risk, score_only=1-p[5]))


class TestVideoTransfer(unittest.TestCase):
    def test_four_states_missing_and_boundary(self):
        rows = [row(i, error=i in (0, 2)) for i in range(5)]
        rows[4]['pred'] = None
        value = c.counts(rows, {rows[0]['image'], rows[1]['image']})
        self.assertEqual(value['states'], dict(FA=1, FR=1, ED=1, CR=1, MISSING=1))
        r = row(0); r['pred'][2] = 110.
        self.assertFalse(c.wrong(r))
        r['pred'][2:4] = [40., 110.]
        self.assertFalse(c.wrong(r))

    def test_missing_cannot_accept(self):
        r = row(0); r['pred'] = None
        with self.assertRaises(ValueError): c.counts([r], {r['image']})

    def test_fr_resets_at_join_gap_and_video(self):
        rows = [row(0), row(1), row(2), row(4), row(0, 'v1')]
        rows[2]['temporal']['acquisition_boundary'] = True
        self.assertEqual(c.counts(rows, set())['longest_correct_FR'], 2)

    def test_exact_count_ties_not_label_selected(self):
        rows = [row(0, error=True), row(1)]
        p = c.point(rows, 'static3', count=1)
        self.assertEqual(p['states']['FA'], 1)
        self.assertEqual(p['tie_bounds'], dict(FA_min=0, FA_max=1, partial_tie=True))

    def test_correct_target_uses_whole_tie(self):
        rows = [row(0), row(1), row(2, error=True)]
        p = c.point(rows, 'static3', correct=1)
        self.assertFalse(p['exact_CR']); self.assertEqual(p['states']['CR'], 2)
        self.assertEqual(c.point(rows, 'static3', correct=0)['accepted'], 0)

    def test_one_global_threshold_protects_fit_videos(self):
        fit = [row(0, 'a', risk=.2), row(1, 'a', error=True), row(0, 'b', risk=.8)]
        t = c.cutoff(fit, 'static3')
        self.assertEqual(t['risk_le'], .8)
        self.assertEqual(set(t['fit_videos']), {'a', 'b'})

    def test_held_labels_do_not_change_models_or_cutoff(self):
        fit = [row(i, error=i % 3 == 0, available=i > 2) for i in range(12)]
        held = [row(i, 'held', available=i > 2) for i in range(10)]
        a = c.fit_score(fit, held)
        altered = deepcopy(held)
        for r in altered: r['gt'][2] = 200.
        b = c.fit_score(fit, altered)
        self.assertEqual(a[0], b[0]); self.assertEqual(a[1], b[1])

    def test_fit_only_normalizer_and_static_fallback(self):
        fit = [row(i, error=i % 3 == 0, available=i > 2) for i in range(12)]
        held = [row(100, 'held', available=False)]
        held[0]['pred'][2] = 1e7
        models, thresholds, scored, value = c.fit_score(fit, held)
        x = np.array([c.fusion.simple.descriptor(r['pred'], r['image_size']) for r in fit])
        np.testing.assert_allclose(models['fallback3']['mean'], x.mean(0), atol=1e-12)
        self.assertEqual(value[0]['risks']['static3'], value[0]['risks']['fusion4'])
        self.assertNotIn('held', thresholds['fusion4']['fit_videos'])

    def test_train_only_and_no_mutation(self):
        rows = [row(i, error=i % 3 == 0, available=i > 2) for i in range(12)]
        saved = deepcopy(rows); c.fit_score(rows, [])
        self.assertEqual(rows, saved)
        rows[0]['split'] = 'val'
        with self.assertRaises(ValueError): c.audit(rows, {})

    def test_single_class_is_not_silently_fitted(self):
        with self.assertRaises(ValueError): c.fit_score([row(i) for i in range(6)], [])


if __name__ == '__main__': unittest.main()
