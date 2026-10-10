import ast
from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np
from crane_project.utils import port_reliability_oof_v1 as core
from crane_project.utils import port_reliability_complementarity_v1 as metrics


def rows():
    result=[]
    for seq,n in core.COUNTS.items():
        for i in range(n):
            result.append(dict(image=seq+'_'+str(i).zfill(5),sequence=seq,domain=seq.split('_')[0],
                split='train_sim' if seq.startswith('sim') else 'train',frame_id=i,
                gt=[50.,50.,20.,10.,0.],pred=[50.,50.,20. if i else 24.,10.,0.,.5],
                image_size=[100,100]))
    return result


class ContractTests(unittest.TestCase):
    def test_two_folds_exclude_full_sequences(self):
        p=core.plan(rows());self.assertEqual(len(p['A']['predict_images']),899)
        self.assertEqual(len(p['B']['predict_images']),911)
        for v in p.values():
            self.assertFalse(set(v['train_images']) & set(v['predict_images']))
            self.assertEqual(v['sim_train'],748)

    def test_no_new_external_roles(self):
        r=rows();r[0]['split']='test'
        with self.assertRaises(ValueError):core.plan(r)

    def test_labels_follow_auxiliary_not_formal(self):
        r=rows();aux=[]
        for x in r:
            if x['domain']=='real':
                v=deepcopy(x);v['pred'][2]=30.
                v['excluded_fold']=next(g for g,s in core.GROUPS.items() if x['sequence'] in s)
                aux.append(v)
        out=core.materialize_training(r,aux)
        self.assertTrue(metrics.bad(out[1]));self.assertFalse(metrics.bad(r[1]))
        self.assertEqual([x for x in out if x['domain']=='sim'][1]['pred'],[x for x in r if x['domain']=='sim'][1]['pred'])

    def test_missing_is_not_a_quality_error(self):
        r=rows();aux=[deepcopy(x) for x in r if x['domain']=='real']
        for x in aux:x['excluded_fold']=next(g for g,s in core.GROUPS.items() if x['sequence'] in s)
        aux[0]['pred']=None
        out=core.materialize_training(r,aux)
        self.assertIsNone(metrics.bad(out[0]))
        self.assertEqual(core.support(out)['all']['missing'],1)

    def test_wrong_fold_rejected(self):
        r=rows();aux=[deepcopy(x) for x in r if x['domain']=='real']
        for x in aux:x['excluded_fold']='B'
        with self.assertRaises(ValueError):core.materialize_training(r,aux)

    def test_gt_identity_cannot_change(self):
        r=rows();aux=[deepcopy(x) for x in r if x['domain']=='real']
        for x in aux:x['excluded_fold']=next(g for g,s in core.GROUPS.items() if x['sequence'] in s)
        aux[0]['gt'][2]=25
        with self.assertRaises(ValueError):core.materialize_training(r,aux)

    def test_fit_disallows_val(self):
        r=rows()[:10];r[0]['split']='val'
        with self.assertRaises(ValueError):core.fit(r)

    def test_fit_same_data_exact_control(self):
        r=rows()[:20]
        self.assertEqual(core.fit(r),core.fit(deepcopy(r)))

    def test_bad_auxiliary_detector_stops(self):
        r=rows()
        for x in r:
            if x['domain']=='real':x['pred']=None
        self.assertFalse(core.source_gate(r)['passed'])

    def test_original_rows_preserved(self):
        r=rows();before=deepcopy(r);aux=[deepcopy(x) for x in r if x['domain']=='real']
        for x in aux:x['excluded_fold']=next(g for g,s in core.GROUPS.items() if x['sequence'] in s)
        core.materialize_training(r,aux);self.assertEqual(r,before)

    def test_runner_global_references_resolve(self):
        path=Path(__file__).parents[1]/'crane_project/tools/run_port_reliability_oof_v1.py'
        import symtable
        table=symtable.symtable(path.read_text(),str(path),'exec')
        import builtins
        available=set(table.get_identifiers())|set(dir(builtins))
        def visit(t):
            for s in t.get_symbols():
                if s.is_referenced() and s.is_global():self.assertIn(s.get_name(),available)
            for child in t.get_children():visit(child)
        visit(table)

    def test_preflight_runs_in_disposable_subprocess(self):
        path=Path(__file__).parents[1]/'crane_project/tools/run_port_reliability_oof_v1.py'
        tree=ast.parse(path.read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        self.assertIn("'--mode','preflight'",ast.get_source_segment(path.read_text(),main))

    def test_actual_control_reproduces_formal_size_model(self):
        from crane_project.tools import run_port_reliability_oof_v1 as runner
        actual,policy=runner.original.load_rows()
        model=core.fit([r for r in actual if r['reliability_role']=='train'])
        for key in ('weights','mean','scale'):
            np.testing.assert_allclose(model[key],policy['simple_policy']['models']['size'][key],atol=1e-10,rtol=1e-10)

    def test_deployable_wrapper_keeps_angle_and_frontend(self):
        from crane_project.tools import run_port_reliability_oof_v1 as runner
        from crane_project.utils.port_midpoint_sigma15_reliability_v1 import Sigma15Reliability
        actual,policy=runner.original.load_rows();candidate=deepcopy(policy)
        candidate['simple_policy']['models']['size']['weights']=[0.,0.,0.,0.]
        candidate['simple_policy']['cutoffs']['simple']['size']['risk_le']=.49
        online=Sigma15Reliability(candidate,candidate['front_end'])
        old=Sigma15Reliability(policy,policy['front_end'])
        for r in actual[:10]:
            a,b=online.decide(r['pred'],r['image_size']),old.decide(r['pred'],r['image_size'])
            self.assertEqual(a['angle_accepted'],b['angle_accepted'])
            self.assertEqual(a['center_accepted'],b['center_accepted'])
            self.assertEqual(a['final_box_original'],b['final_box_original'])

    def test_actual_mmrotate_loader_compatibility(self):
        try:
            from mmcv import Config
            from mmrotate.utils import compat_cfg
        except ImportError:
            self.skipTest('Server MMRotate runtime required')
        cfg=Config.fromfile(str(Path(__file__).parents[1]/'crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py'))
        cfg.data.samples_per_gpu=2;cfg.data.workers_per_gpu=2
        cfg.data.pop('train_dataloader',None)
        cfg=compat_cfg(cfg)
        self.assertEqual(cfg.data.train_dataloader.samples_per_gpu,2)


if __name__=='__main__':unittest.main()
