import unittest
from crane_project.utils import port_pilot_size_support_v1 as c

class PilotTests(unittest.TestCase):
    def test_historical_split_and_overlap(self):
        s=dict(sequence='real_seq11',k0=1.9,target_geometry='top_beam_only',train_stems=['real_seq11_%06d'%i for i in range(203)],aux_val_stems=['real_seq11_%06d'%i for i in range(203,251)])
        self.assertEqual(len(c.train_stems(s)),203)
        s['aux_val_stems'][0]=s['train_stems'][0]
        with self.assertRaises(ValueError):c.train_stems(s)
    def test_canonical_labels_and_types(self):
        gt=[0,0,100,50,0]
        d=c.detail([0,0,90,45,0,.7],gt);self.assertFalse(d['wrong'])
        d=c.detail([0,0,89,50,0,.7],gt);self.assertEqual((d['axis'],d['direction'],d['severity']),('L_only','small','10_15'))
        d=c.detail([0,0,89,65,0,.7],gt);self.assertEqual((d['axis'],d['direction'],d['severity']),('both','mixed','over25'))
        self.assertFalse(c.detail([0,0,50,100,-1.5707963267948966,.7],gt)['wrong'])
        self.assertIsNone(c.detail(None,gt))
    def test_original_frame_runs(self):
        rows=[dict(image=str(i),frame_id=i) for i in (1,2,4,5,6,50)]
        runs=c.intervals(rows,lambda r:r['frame_id']!=5)
        self.assertEqual([(x['start'],x['end'],x['length']) for x in runs],[(1,2,2),(4,4,1),(6,6,1),(50,50,1)])
    def test_missing_separate_and_center_no_filter(self):
        rows=[]
        for i,(p,keep) in enumerate([([0,0,10,5,0,.8],True),([0,0,20,5,0,.8],True),([0,0,10,5,0,.8],False),([0,0,20,5,0,.8],False),(None,False)]):
            gt=[100,100,10,5,0];d=c.detail(p,gt)
            state='MISSING' if d is None else ('FA' if d['wrong'] else 'CR') if keep else ('ED' if d['wrong'] else 'FR')
            rows.append(dict(image=str(i),frame_id=i,sequence='real_seq11',pred=p,gt=gt,detail=d,size_state=state,decision=dict(size_accepted=keep),score_decision=dict(size_accepted=keep),sampling_origin='original59'))
        r=c.report(rows);self.assertEqual(r['methods']['simple']['states'],dict(FA=1,FR=1,ED=1,CR=1,MISSING=1));self.assertEqual(r['bad_center_ge15'],2)
    def test_invalid_sizes(self):
        with self.assertRaises(ValueError):c.detail([0,0,0,5,0,.8],[0,0,10,5,0])
if __name__=='__main__':unittest.main()
