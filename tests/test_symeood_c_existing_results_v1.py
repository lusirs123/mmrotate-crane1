"""Boundary/conservation checks for the saved-output audit (no torch/MMCV)."""
import importlib.util
import math
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1]/'crane_project/tools/audit_symeood_c_existing_results_v1.py'
SPEC = importlib.util.spec_from_file_location('c_audit', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def row(fid, hit=True, output=True, center=0., angle=1., diag=10., seq='s'):
    return dict(domain='sim', seq=seq, fid=fid, frame=seq+'_'+str(fid),
                output=output, center_export=center if output else float('inf'),
                riou_export=.8 if hit and output else .1 if output else 0.,
                diag_export=diag if output else None,
                angle_export=math.radians(angle) if output else None,
                angle_error_deg=angle if output else None,
                gated_angle_sq=angle**2 if output and center < 10 else 8100.)


class AuditChecks(unittest.TestCase):
    def test_sequence_and_gap_break_windows_and_runs(self):
        rows=[row(1,False),row(2,False),row(4,False),row(1,False,seq='t')]
        result=audit.aggregate(rows)
        self.assertEqual(result['MCML_max'],2)
        self.assertEqual(result['TDR_windows'],0)
        self.assertIsNone(result['MRF'])
        self.assertEqual(len(result['pairs']),1)

    def test_missing_breaks_pairs_but_is_failure(self):
        result=audit.aggregate([row(1),row(2,output=False),row(3)])
        self.assertEqual(len(result['pairs']),0)
        self.assertEqual(result['hits'],2)
        self.assertEqual(result['MCML_max'],1)
        self.assertEqual(result['MRF'],1)

    def test_terminal_failure_is_not_recovery(self):
        result=audit.aggregate([row(1,False),row(2),row(3,False),row(4,False)])
        self.assertEqual(result['MRF'],1)
        self.assertEqual(result['MCML_max'],2)

    def test_window_size_ten_and_endpoint(self):
        result=audit.aggregate([row(i,hit=i==10) for i in range(1,12)])
        self.assertEqual(result['TDR_windows'],2)
        self.assertEqual(result['TDR_hits'],2)

    def test_center_threshold_is_strict(self):
        result=audit.aggregate([row(1,center=15.),row(2,center=14.99)])
        self.assertEqual(result['center'],1)

    def test_gate_penalty_and_angle_terms_have_opposite_signs(self):
        result=audit.compare([row(1,angle=3),row(2,angle=3)],
                             [row(1,center=10,angle=1),row(2,angle=1)])
        angle=result['angle']
        self.assertEqual(angle['gate_lost'],1)
        self.assertGreater(angle['penalty_delta_MSE'],0)
        self.assertLess(angle['pass_angle_delta_MSE'],0)
        self.assertAlmostEqual(angle['gated_RMSE_on']**2-angle['gated_RMSE_off']**2,
                               angle['penalty_delta_MSE']+angle['pass_angle_delta_MSE'])

    def test_common_pairs_intersection_and_identity(self):
        off=[row(1),row(2,diag=12),row(3,diag=14),row(4,diag=15)]
        on=[row(1),row(2,diag=12),row(3,output=False),row(4,diag=15)]
        result=audit.compare(off,on)
        self.assertEqual(result['common_pairs']['count'],1)
        self.assertAlmostEqual(result['common_pairs']['delta_DFR'],0)
        self.assertAlmostEqual(result['common_pairs']['delta_ACI'],0)
        identity=audit.compare(off,off)
        self.assertEqual(identity['gains']['hit'],[])
        self.assertEqual(identity['losses']['hit'],[])

    def test_transition_reversal(self):
        off=[row(1,False),row(2),row(3)]
        on=[row(1),row(2,False),row(3,False)]
        a=audit.compare(off,on);b=audit.compare(on,off)
        self.assertEqual(a['gains']['hit'],b['losses']['hit'])
        self.assertEqual(a['losses']['hit'],b['gains']['hit'])
        self.assertEqual(a['wholly_new_on_events'],1)
        self.assertEqual(a['fully_resolved_off_events'],1)


if __name__=='__main__':
    unittest.main()
