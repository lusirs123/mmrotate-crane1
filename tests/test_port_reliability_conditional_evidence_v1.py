import copy
import json
import unittest
import numpy as np
from crane_project.utils import port_reliability_conditional_evidence_v1 as c


def row(i=1,bad=False,d=.02,score=.7,seq='a',role='train'):
    return dict(image=seq+str(i),sequence=seq,domain='real',frame_id=i,
        reliability_role=role,split='train' if role=='train' else role,
        pred=[0,0,80 if bad else 100,40 if bad else 50,0,score],
        gt=[0,0,100,50,0],image_size=[1000,1000],risks={'fusion4':.2},
        temporal=dict(available=d is not None,primary=d,history_frames=[]))


class Tests(unittest.TestCase):
    def test_missing_not_error_or_zero(self):
        r=row();r['pred']=None
        self.assertIsNone(c.bad(r));self.assertEqual(c.regime(r),'missing')
        self.assertEqual(c.counts([r])['missing'],1)

    def test_label_edge_exchange(self):
        r=row(bad=True);r['pred'][2:4]=r['pred'][2:4][::-1];r['pred'][4]+=np.pi/2
        self.assertTrue(c.bad(r))

    def test_threshold_inclusive(self):
        self.assertEqual(c.regime(row(d=c.SETTINGS['level_cutoff'])),'low')
        self.assertEqual(c.regime(row(d=None)),'unsupported')

    def test_auc_ties_and_direction(self):
        rows=[row(1,bad=True,score=.2),row(2,score=.8)]
        self.assertEqual(c.auc(rows,0),0);self.assertEqual(c.auc(rows,0,-1),1)
        rows[1]['pred'][5]=.2;self.assertEqual(c.auc(rows,0),.5)

    def test_train_only_and_test_rejected(self):
        with self.assertRaises(ValueError):c.freeze_train([row(role='val')])
        with self.assertRaises(ValueError):c.validate([row(role='test')],'TEST')

    def test_rules_no_gt_quantiles_or_val(self):
        rows=[row(1,True),row(2)];a=c.freeze_train(rows)
        rows[0]['gt'][2:4]=[80,40]
        self.assertEqual(a['quantiles'],c.freeze_train(rows)['quantiles'])

    def test_no_cross_video_size_matching(self):
        rows=[row(1,True,seq='a'),row(2,seq='b')]
        f=c.freeze_train(rows)
        self.assertEqual(c.size_matched(rows,f)['matched_errors'],0)

    def test_tolerance_and_nearest(self):
        rows=[row(1,True),row(2),row(3)];rows[2]['pred'][2:4]=[81,41];rows[2]['gt'][2:4]=[81,41]
        f=c.freeze_train(rows);p=c.size_matched(rows,f)
        self.assertEqual(p['matched_errors'],1);self.assertEqual(p['pairs'][0]['good_image'],'a3')

    def test_unsupported_not_interaction_low(self):
        rows=[row(1,True,d=None),row(2)];f=c.freeze_train(rows)
        g=c.analyze(rows,f,.5)['groups']['all']
        self.assertEqual(g['regimes']['unsupported']['bad'],1)
        for interaction in g['interactions'].values():
            self.assertEqual(sum(x['frames'] for x in interaction['cells'].values()),1)

    def test_no_mutation(self):
        rows=[row(1,True),row(2)];before=copy.deepcopy(rows);f=c.freeze_train(rows)
        c.analyze(rows,f,.5);self.assertEqual(rows,before)

    def test_small_support_cannot_qualify(self):
        rows=[row(1,True),row(2)];f=c.freeze_train(rows);a=c.analyze(rows,f,.5)
        s=c.prerequisite_screen({'TRAIN':a,'VAL':a})
        self.assertEqual(s['status'],'NO_SUFFICIENT_INTERACTION_EVIDENCE')
        self.assertFalse(s['automatic_training'])

    def test_json_roundtrip_screen(self):
        rows=[row(1,True),row(2)];f=c.freeze_train(rows);a=c.analyze(rows,f,.5)
        screen=c.prerequisite_screen({'TRAIN':a,'VAL':a})
        self.assertEqual(screen,json.loads(json.dumps(screen,sort_keys=True)))

    def test_recovery_history_offline(self):
        rows=[row(1,True),row(2)];rows[1]['risks']['fusion4']=.9
        rows[1]['temporal']['history_frames']=[1]
        f=c.freeze_train(rows);a=c.analyze(rows,f,.5)
        self.assertEqual(a['groups']['all']['FR_with_bad_history'],1)
        self.assertEqual(a['details'][1]['state'],'FR')


if __name__=='__main__':unittest.main()
