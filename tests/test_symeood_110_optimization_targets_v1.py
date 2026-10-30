"""Meaningful boundaries of the prespecified no-regression VAL target."""
import importlib.util
from pathlib import Path
import unittest

P = Path(__file__).resolve().parents[1]/'crane_project/utils/symeood_110_optimization_targets_v1.py'
spec = importlib.util.spec_from_file_location('goals', P)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


class GoalTests(unittest.TestCase):
    def setUp(self):
        self.old = dict(frames=887,AP75=79.30305,mean_RIoU=.848678,
            center_MAE_px=3.616224,long_MAE_pct=6.124563,short_MAE_pct=6.515371,
            Recall_pct=100*883/887,center_correct=880,MCML_max_native=3,joint10=632)
        self.sim = dict(frames=512,pure_angle_RMSE_deg=1.659260)

    def test_local_gain_cannot_hide_any_regression(self):
        new = dict(self.old,AP75=81.)
        self.assertTrue(g.assess(new,self.sim,self.old,self.sim)['eligible'])
        for key,value in [('center_MAE_px',4.),('long_MAE_pct',6.3),
                          ('short_MAE_pct',6.7),('Recall_pct',100*882/887),
                          ('center_correct',879),('joint10',631),('MCML_max_native',4)]:
            with self.subTest(key=key):
                self.assertFalse(g.assess(dict(new,**{key:value}),self.sim,self.old,self.sim)['eligible'])
        self.assertFalse(g.assess(new,dict(self.sim,pure_angle_RMSE_deg=2.),self.old,self.sim)['eligible'])

    def test_small_gain_does_not_pass_minimum_effect(self):
        self.assertFalse(g.assess(dict(self.old,AP75=79.8),self.sim,self.old,self.sim)['eligible'])

    def test_all_failed_means_no_selection(self):
        metrics=[dict(self.old,side='S',epoch='epoch_20',domain='all',selected=True),
                 dict(self.sim,side='S',epoch='epoch_20',domain='sim',selected=True)]
        for ep in (16,18,20,22,24):
            metrics.extend([dict(self.old,side='R4',epoch='epoch_'+str(ep),domain='all',selected=False),
                            dict(self.sim,side='R4',epoch='epoch_'+str(ep),domain='sim',selected=False)])
        self.assertIsNone(g.select(metrics)['selected'])
        metrics[2]['AP75']=81.
        self.assertEqual(g.select(metrics)['selected'],'epoch_16')


if __name__=='__main__':
    unittest.main()
