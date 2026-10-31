"""Fresh frontend provenance, no TEST and VAL selection gates; CPU stdlib."""
import ast
from copy import deepcopy
import json
import sys
import unittest
from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
from crane_project.tools import train_symeood_r4_lvsr_pair_v1 as t
from crane_project.tools import run_symeood_r4_lvsr_pair_v1 as runner


def example():
    baseline=dict(AP75=88.,TP50=100,mean_RIoU=.85,joint10=80,long_MAE=.05,short_MAE=.06)
    return {d:dict(frozen_identity=True,b=deepcopy(baseline),midpoint=deepcopy(baseline),
        edge_residual=dict(baseline,joint10=81,long_MAE=.049,short_MAE=.059)) for d in ('overall','real','sim')}


class LVSRChecks(unittest.TestCase):
    def test_checked_sources_fix_R4_M20_and_unchanged_size_head(self):
        before='torch' in sys.modules;identity=p.checked_sources()
        self.assertEqual(before,'torch' in sys.modules)
        self.assertEqual(identity['midpoint_sha256'],p.MIDPOINT_SHA)
        self.assertEqual(identity['detector_sha256'],p.parent.R4_SHA)
        self.assertEqual(identity['settings']['parameters'],2480)
        self.assertFalse(identity['settings']['continuation'])

    def test_stages_and_same_root(self):
        self.assertEqual(t.parser().parse_args([]).stage,'check')
        choices=t.parser()._option_string_actions['--stage'].choices
        self.assertEqual(set(choices),{'check','prepare','inputs','verify','train'})
        self.assertIn(p.BASE,p.paths()['train'].parents)
        self.assertNotEqual(p.paths()['inputs'],p.paths('attempt_002')['inputs'])
        with self.assertRaises(ValueError):p.paths('../escape')
        for stage in ('inputs','verify','train'):
            self.assertIn('--gpu',runner.command(stage,'attempt_001',2))
        with self.assertRaises(ValueError):runner.command('test','attempt_001',2)

    def test_VAL_AP_guard_cannot_be_replaced_by_size_gain(self):
        metrics=example();self.assertTrue(p.assess(metrics)['pass_all'])
        metrics['real']['edge_residual']['AP75']-=.01
        result=p.assess(metrics)
        self.assertFalse(result['pass_all']);self.assertIn('real/AP75_vs_M_and_R4',result['failed'])

    def test_AP_must_protect_R4_as_well_as_M20(self):
        metrics=example();metrics['real']['b']['AP75']=89.
        self.assertFalse(p.assess(metrics)['pass_all'])
        metrics['real']['edge_residual']['AP75']=89.
        self.assertTrue(p.assess(metrics)['pass_all'])

    def test_exact_identity_and_domain_size_protection(self):
        metrics=example();metrics['sim']['frozen_identity']=False
        self.assertFalse(p.assess(metrics)['pass_all'])
        metrics=example();metrics['real']['edge_residual']['short_MAE']=.061
        self.assertFalse(p.assess(metrics)['pass_all'])
        metrics=example();metrics['overall']['edge_residual']['joint10']=80
        self.assertFalse(p.assess(metrics)['pass_all'])

    def test_no_eligible_keeps_null_and_does_not_promote_representative(self):
        m=example();good=dict(epoch=1,metrics=m,eligibility=p.assess(m),selection_key=p.selection_key(m,1))
        bad=deepcopy(good);bad['epoch']=2;bad['eligibility']['pass_all']=False;bad['selection_key'][0]=-999
        selected=p.choose([good,bad]);self.assertEqual(selected['selected_epoch'],1)
        selected=p.choose([bad]);self.assertIsNone(selected['selected_epoch']);self.assertEqual(selected['representative_epoch'],2)
        self.assertFalse(selected['automatic_adoption']);self.assertFalse(selected['selection_on_test'])

    def test_runtime_uses_saved_R4_features_without_detector_build_or_old_head(self):
        source=p.ROOT/'crane_project/tools/train_symeood_r4_lvsr_pair_v1.py'
        tree=ast.parse(source.read_text())
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for name in ('create_cache','build_detector','fixed_test_inputs','simple_test','replay_fit'):
            self.assertNotIn(name,calls)
        self.assertIn('load_cache',calls);self.assertIn('load_state_dict',calls)
        self.assertIn('full_VAL_M20_replay',source.read_text())


if __name__=='__main__':unittest.main()
