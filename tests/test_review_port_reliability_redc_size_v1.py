import unittest
from crane_project.tools import review_port_reliability_redc_size_v1 as r


def row(i,bad=False,seq='real_seq01'):
    return dict(image=seq+str(i),sequence=seq,frame_id=i,
        pred=[0,0,12 if bad else 10,4,0,.8],gt=[0,0,10,4,0],domain='real',
        experiment_risks={'m':.9 if bad else .1})


class ScalarReview(unittest.TestCase):
    def test_independent_edges_ignore_equivalent_swap(self):
        a=row(0);a['pred'][2:4]=[4,10]
        self.assertFalse(r.bad(a));a['pred'][2]=5;self.assertTrue(r.bad(a))

    def test_missing_excluded(self):
        a=row(0);a['pred']=None
        c,l=r.counts([a],set());self.assertEqual(c['MISSING'],1)
        self.assertEqual(sum(c[k] for k in ('FA','FR','ED','CR')),0)

    def test_FR_resets_at_gap_and_ED(self):
        rows=[row(0),row(1),row(2,True),row(4),row(5)]
        c,l=r.counts(rows,set());self.assertEqual(c['FR'],4);self.assertEqual(c['ED'],1)
        self.assertEqual(l,2)

    def test_FA_and_CR_not_FR(self):
        rows=[row(0,True),row(1)]
        c,l=r.counts(rows,{x['image'] for x in rows})
        self.assertEqual(c['FA'],1);self.assertEqual(c['CR'],1);self.assertEqual(l,0)

    def test_pairwise_AUC_ties(self):
        rows=[row(0),row(1,True)]
        self.assertEqual(r.auc(rows,'m'),1.)
        rows[1]['experiment_risks']['m']=.1
        self.assertEqual(r.auc(rows,'m'),.5)

    def test_single_class_AUC_not_defined(self):
        self.assertIsNone(r.auc([row(0)],'m'))

    def test_grouping_preserves_all_frames(self):
        rows=[row(0),row(1,seq='real_seq02')]
        g=r.grouped(rows);self.assertEqual(len(g['all']),2)
        self.assertEqual(len(g['domain:real']),2)
        self.assertEqual(len(g['sequence:real_seq02']),1)


if __name__=='__main__':unittest.main()
