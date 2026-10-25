"""Selector edge cases and rejection of corrupted S-contract receipts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'crane_project/tools'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


PERF=load('audit_symeood_cross_entry_performance_v1')
SUMMARY=load('summarize_symeood_cross_entry_review_v1')


class ContractTests(unittest.TestCase):
    def test_s_must_reject_gradient_leak_despite_passed_flag(self):
        row=dict(level_count=5,single_batch_alpha=True,alpha_requires_grad=False,
                 G_to_actual_gradient_none=True,per_level_formula_equal=True,
                 all_parameter_gradients_equal=True,other_losses_equal={'bbox':True,'aux':True},
                 counter_equal=True,classification_G_S_relative=2**-24,passed=True)
        self.assertTrue(SUMMARY.verify_s(row))
        for key,value in [('alpha_requires_grad',True),('G_to_actual_gradient_none',False),
                          ('all_parameter_gradients_equal',False),('counter_equal',False),('single_batch_alpha',False)]:
            bad=dict(row);bad[key]=value
            self.assertFalse(SUMMARY.verify_s(bad),key)

    def selection(self,epochs,expected,limit=5):
        config=dict(weight_real=.3,weight_sim=.7,soft_w_tdr=.35,soft_w_r_center=.25,
                    soft_w_aci=.2,soft_w_angle=.2,mcml_limit=limit,r_center_delta=.005)
        candidates={}
        for ep,center,mc,angle in epochs:
            candidates['epoch_'+str(ep)]=dict(metrics={'real/R_center(%)':center,'sim/R_center(%)':center,
                'real/TDR_w10(%)':100,'sim/TDR_w10(%)':100,'real/MCML_max(frames)':mc,
                'sim/MCML_max(frames)':mc,'sim/A-RMSE(deg)':angle,'sim/ACI':.9})
        data=dict(selection_config=config,candidate_epochs=[16,18,20,22,24],center_thresh_px=15,
                  all_checkpoints=candidates,selected_checkpoint='epoch_'+str(expected))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sweep.json';p.write_text(json.dumps(data));return PERF.selector(p)

    def test_exact_score_tie_preserves_original_order(self):
        result=self.selection([(e,99,1,3) for e in (16,18,20,22,24)],16)
        self.assertEqual(result['ties'],5)

    def test_center_guard_excludes_better_angle_outside_tolerance(self):
        result=self.selection([(16,97,1,0),(18,99,1,10),(20,98,1,0),(22,98,1,0),(24,98,1,0)],18)
        self.assertEqual(result['selected'],'epoch_18')

    def test_all_mcml_fail_uses_original_fallback(self):
        result=self.selection([(16,99,8,1),(18,98,6,3),(20,99,6,4),(22,99,6,2),(24,99,7,0)],22)
        self.assertEqual(result['selected'],'epoch_22')


if __name__=='__main__':unittest.main()
