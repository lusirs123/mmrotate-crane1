import unittest
from crane_project.tools.test_port_oof_momentum_restart_short_v1 import UPDATES,metrics,compare


class ShortComparisonTests(unittest.TestCase):
    def report(self,value=1.,reg=1.):
        records = [dict(index=i,losses=dict(main_cls=5.235 if i==1 else value,main_reg=1. if i==1 else reg),
                        original_guard_alarms=['main_cls'] if i==1 else []) for i in range(1,34)]
        return dict(optimizer_updates=32,records=records,failure=None,original_snapshot_unchanged=True,
                    snapshot_sha256='same_snapshot',source_sha256='same_source')

    def test_identical_initial_alarm_excluded_from_recovery_metrics(self):
        self.assertEqual(UPDATES,32)
        m = metrics(self.report()['records'])
        self.assertEqual(m['mean_post_cls'],1.);self.assertEqual(m['loss_alarm_forwards'],1)

    def test_local_support_never_becomes_prevention_or_adoption(self):
        result = compare(self.report(2.),self.report(1.))
        self.assertTrue(result['local_recovery_increment_supported'])
        self.assertFalse(result['prevention_passed']);self.assertFalse(result['full_training_stability_proven'])
        self.assertFalse(result['adopted'])

    def test_regression_damage_cannot_pass(self):
        result = compare(self.report(2.,1.),self.report(1.,3.))
        self.assertFalse(result['local_recovery_increment_supported'])
        self.assertFalse(result['criteria']['final_reg_not_worse'])

    def test_incomplete_and_nonincremental_do_not_pass(self):
        r = self.report();r['optimizer_updates'] = 31
        self.assertFalse(compare(self.report(2.),r)['local_recovery_increment_supported'])
        self.assertFalse(compare(self.report(),self.report())['local_recovery_increment_supported'])


if __name__=='__main__':unittest.main()
