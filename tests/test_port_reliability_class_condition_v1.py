import unittest
from copy import deepcopy
from crane_project.tools import analyze_port_reliability_class_condition_v1 as a
from test_port_reliability_video_transfer_v1 import row
class TestClassCondition(unittest.TestCase):
 def material(self):
  rows=[row(i,'a' if i<24 else 'b',error=i%2==0) for i in range(48)]
  for r in rows:r['domain']='real'
  return rows
 def model(self):return dict(weights=[0.,1.,-1.,1.],mean=[0.,0.,0.],scale=[1.,1.,1.])
 def test_true_labels_not_acceptance(self):
  rows=self.material()
  for r in rows:r['original_simple_decision']={'size_accepted':True}
  z=a.analyze(rows,self.model());self.assertEqual((z['wrong'],z['correct']),(24,24))
 def test_no_mutation(self):
  rows=self.material();saved=deepcopy(rows);a.analyze(rows,self.model());self.assertEqual(rows,saved)
 def test_ties_and_joint_support(self):
  rows=self.material()
  for r in rows:r['pred']=[0.,0.,100.,40.,0.,.8];r['gt'][2]=80. if r['frame_id']%2==0 else 100.
  z=a.analyze(rows,self.model());p=z['pairs'][0]
  self.assertEqual(p['both_labels_both_videos_joint_cells'],1)
  self.assertEqual(p['classes']['wrong']['joint_total_variation'],0.)
  self.assertEqual(z['videos']['a']['contributions'][0]['contribution_AUC'],.5)
  self.assertEqual(a.review(rows,self.model(),z)['rows'],48)
 def test_contribution_sign_fixed(self):
  self.assertEqual(a.auc(a.np.array([0.]),a.np.array([1.])),1.)
  self.assertEqual(a.auc(a.np.array([0.]),a.np.array([-1.])),0.)
 def test_boundary_breaks_segment(self):
  rows=[row(i) for i in range(4)];rows[2]['temporal']['acquisition_boundary']=True
  self.assertEqual(a.segments(rows,[True]*4),2)
 def test_sparse_video_not_supported(self):
  z=a.analyze(self.material()[:30],self.model());self.assertFalse(z['videos']['b']['comparison_supported'])
 def test_corruption_detected(self):
  rows=self.material();z=a.analyze(rows,self.model());z['pairs'][0]['classes']['wrong']['shared_joint_cells']+=1
  with self.assertRaises(ValueError):a.review(rows,self.model(),z)
 def test_duplicate_or_missing_fails(self):
  rows=self.material();rows[1]['image']=rows[0]['image']
  with self.assertRaises(ValueError):a.analyze(rows,self.model())
  rows=self.material();rows[0]['pred']=None
  with self.assertRaises(ValueError):a.analyze(rows,self.model())
if __name__=='__main__':unittest.main()
