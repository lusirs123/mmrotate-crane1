"""Finite TRAIN forensics, loss-neutral instrumentation and fixed VAL selection."""
from copy import deepcopy
import ast
import math
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from crane_project.utils import port_reliability_oof_followup_v1 as follow
from crane_project.utils import port_reliability_oof_training_v2 as repair

ROOT=Path(__file__).resolve().parents[1]


class Attr(dict):
    def __getattr__(self,key):
        try: return self[key]
        except KeyError: raise AttributeError(key)
    __setattr__=dict.__setitem__


def config():
    pipeline=[dict(type='PortIsotropicShrink',prob=.5,scale_range=(.5,1.)),
        dict(type='RRandomFlip',flip_ratio=[.25]*3),
        dict(type='Normalize',mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375],to_rgb=True),
        dict(type='Pad',size=(1024,1024),pad_val=dict(img=(114.,)*3))]
    cfg=Attr(data=Attr(train=[dict(pipeline=deepcopy(pipeline)) for _ in range(2)],val={},test={}),
        model=Attr(bbox_head=Attr(loss_cls=Attr(),loss_bbox=Attr()),train_cfg=Attr(assigner=Attr())),
        custom_imports=dict(imports=[]),optimizer_config=Attr(grad_clip=dict(max_norm=10,norm_type=2)),
        optimizer=Attr(type='SGD',lr=.0025,momentum=.9,weight_decay=.0001),
        lr_config=Attr(warmup_iters=1000,step=[16,22]),runner=Attr(max_epochs=24),
        evaluation=None,load_from=None,resume_from=None,auto_resume=False)
    return repair.repair_config(cfg)


def val_rows(count=20):
    return [dict(image=d+'_seq_000%02d'%i,sequence=d+'_seq',domain=d,
        split='val',frame_id=i,image_size=[100,100],gt=[50.,50.,20.,10.,0.],
        pred=[50.,50.,20.,10.,0.,.5]) for d in ('real','sim') for i in range(count)]


class ContractTests(unittest.TestCase):
    def setUp(self):
        # Execute the real main-selector pure definitions without importing its
        # OpenCV inference/evaluator dependencies, absent from the Mac runtime.
        path=ROOT/'crane_project/tools/ckpt_sweep.py'; tree=ast.parse(path.read_text())
        nodes=[node for node in tree.body if
            isinstance(node,ast.FunctionDef) and node.name in ('extract_metrics','select_best_checkpoint') or
            isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SELECTION_CONFIG' for t in node.targets)]
        namespace=dict(math=math)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
        self.assertEqual(namespace['SELECTION_CONFIG'],follow.SELECTION['selection_config'])
        patcher=patch.object(follow,'main_selector',return_value=(namespace['select_best_checkpoint'],namespace['SELECTION_CONFIG']))
        patcher.start(); self.addCleanup(patcher.stop)
    def test_diagnosis_retains_optimizer_schedule_augmentation_and_budget(self):
        cfg=config(); before=deepcopy(cfg)
        follow.configure(cfg,'diagnose','work_dirs/new/A/trace')
        for field in ('optimizer','optimizer_config','lr_config','runner','model','data'):
            self.assertEqual(cfg[field],before[field])
        self.assertIsNone(cfg.checkpoint_config)
        self.assertEqual([h['priority'] for h in cfg.custom_hooks],['VERY_HIGH','LOW'])
        self.assertEqual(follow.TRACE['max_updates'],7000)

    def test_selection_preserves_training_only_changes_saved_candidates(self):
        cfg=config(); before=deepcopy(cfg)
        follow.configure(cfg,'selected')
        for field in ('optimizer','optimizer_config','lr_config','runner','model','data'):
            self.assertEqual(cfg[field],before[field])
        self.assertEqual(cfg.checkpoint_config,dict(interval=2,max_keep_ckpts=12))

    def test_trainer_rejects_val_or_resume(self):
        for field in ('val','test'):
            cfg=config(); cfg.data[field]=dict(ann_file=field+'/annfiles')
            with self.assertRaises(ValueError): follow.configure(cfg,'selected')
        cfg=config(); cfg.resume_from='old.pth'
        with self.assertRaises(ValueError): follow.configure(cfg,'diagnose','x')

    def test_large_finite_spike_detected_instead_of_only_nan(self):
        self.assertEqual(follow.anomaly(dict(main_cls=.3,main_reg=.3),19.),[])
        self.assertEqual(follow.anomaly(dict(main_cls=18.5,main_reg=2.25),12575.),
                         ['main_cls','gradient_before_clip'])
        self.assertEqual(follow.anomaly(dict(main_cls=np.nan,main_reg=.3),19.),['nonfinite'])

    def test_nonfinite_trace_evidence_is_serializable_without_hiding_errors(self):
        value=follow.json_safe(dict(norm=float('nan'),loss=np.array([1.,np.inf])))
        self.assertEqual(value,dict(norm='nan',loss=[1.,'inf']))
        json.dumps(value,allow_nan=False)

    def test_missing_boxes_do_not_become_wrong_size_labels_or_fake_coverage(self):
        identities=val_rows(); rows=deepcopy(identities)
        rows[0]['pred']=None
        result=follow.selection_metrics(rows,identities)
        real=result['groups']['domain:real']
        self.assertEqual(real['outputs'],19)
        self.assertEqual(real['size_correct_full_frame_coverage'],.95)
        self.assertEqual(real['center_hit_rate_on_outputs'],1.)
        self.assertEqual(real['center_correct_full_frame_coverage'],.95)
        self.assertEqual(real['penalized_edge_MAE'],.05)

    def test_full_val_required_no_subsets_duplicate_or_test(self):
        identities=val_rows()
        for rows in (identities[:-1],identities+[identities[0]],
                     [dict(r,split='test') for r in identities],
                     [dict(r,gt=[50,50,30,10,0]) for r in identities]):
            with self.assertRaises(ValueError): follow.selection_metrics(rows,identities)

    def test_bad_video_still_reported_without_adding_selection_gates(self):
        ids=val_rows(100); rows=deepcopy(ids)
        for r in rows[:11]: r['pred'][0]=80.
        result=follow.selection_metrics(rows,ids)
        self.assertAlmostEqual(result['groups']['sequence:real_seq']['center_hit_rate_on_outputs'],.89)
        self.assertFalse(result['used_for_checkpoint_choice'])

    def test_scale_and_edge_swap_metrics_are_canonical(self):
        ids=val_rows(); rows=deepcopy(ids)
        for r in rows: r['pred']=[50.,50.,10.,20.,np.pi/2,.5]
        result=follow.selection_metrics(rows,ids)
        self.assertEqual(result['groups']['all']['size_correct_full_frame_coverage'],1.)
        self.assertEqual(result['groups']['all']['penalized_edge_MAE'],0.)
        for r in rows: r['pred'][2:4]=[8.,16.]
        result=follow.selection_metrics(rows,ids)
        self.assertEqual(result['groups']['all']['size_correct_full_frame_coverage'],0.)
        self.assertAlmostEqual(result['groups']['all']['penalized_edge_MAE'],.2)

    def candidates(self,kind='midpoint'):
        return [dict(epoch=e,offline_metrics={'real/TDR_w10(%)':88.,'sim/TDR_w10(%)':88.,
            'real/R_center(%)':85.,'sim/R_center(%)':85.,'sim/ACI':.9,
            'real/MCML_max(frames)':0,'sim/MCML_max(frames)':0,'sim/A-RMSE(deg)':5.})
                for e in follow.SELECTION[kind+'_epochs']]

    def test_fixed_full_scan_no_cherrypicking_survivors(self):
        with self.assertRaises(ValueError): follow.select(self.candidates()[:-1],'midpoint')
        c=self.candidates(); c[-1]['epoch']=1
        with self.assertRaises(ValueError): follow.select(c,'midpoint')

    def test_exact_main_selector_center_constraint_soft_score_and_tie(self):
        c=self.candidates(); c[1]['offline_metrics']['sim/ACI']=1.
        self.assertEqual(follow.select(c,'midpoint')['epoch'],2)
        c[0]['offline_metrics']['sim/R_center(%)']=95.
        self.assertEqual(follow.select(c,'midpoint')['epoch'],1)
        self.assertEqual(follow.select(list(reversed(self.candidates())),'midpoint')['epoch'],1)

    def test_main_fallback_retained_without_added_95_90_gates(self):
        c=self.candidates()
        self.assertEqual(follow.select(c,'midpoint')['epoch'],1)
        for item,mcml in zip(c,(8,7,9)): item['offline_metrics']['sim/MCML_max(frames)']=mcml
        selected=follow.select(c,'midpoint')
        self.assertEqual(selected['epoch'],2)
        self.assertEqual(selected['selection_info']['selection'],'fallback_mcml_min')

    def test_val_pipeline_repairs_both_checks_but_builds_once(self):
        cfg=config()
        template=dict(pipeline=[dict(type='LoadImageFromFile'),dict(type='MultiScaleFlipAug',
            transforms=[dict(type='Normalize',mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375],to_rgb=True),
                dict(type='Pad',size=(1024,1024),pad_val=dict(img=(114.,)*3))])])
        before=deepcopy(template)
        specs=repair.inference_specs([template,deepcopy(template)],cfg)
        self.assertEqual(template,before)
        self.assertEqual(specs[0]['pipeline'][1]['transforms'][1]['pad_val'],cfg.data.train[0]['pipeline'][-1]['pad_val'])

    def test_selected_source_assessment_keeps_baseline_and_new_identity(self):
        # Replayed real saved rows with synthetic OOF == formal predictions:
        # exercises output provenance/policy contract, not performance evidence.
        from crane_project.tools import run_port_reliability_oof_followup_v1 as runner
        from crane_project.tools import run_port_reliability_oof_v1 as legacy
        actual,policy=legacy.original.load_rows()
        train=[r for r in actual if r['reliability_role']=='train']
        with tempfile.TemporaryDirectory(dir=ROOT/'work_dirs') as tmp:
            out=Path(tmp)
            for fold,sequences in legacy.core.GROUPS.items():
                folder=out/fold; folder.mkdir()
                weights=folder/'synthetic.weights'; weights.write_bytes(b'synthetic_not_a_model')
                selected=dict(path=str(weights),sha256=legacy.sha(weights),epoch=2)
                legacy.write(folder/'selected_source_completion.json',dict(source_gate=dict(passed=True),
                    detector=selected,midpoint=selected))
                aux=[dict(r,excluded_fold=fold,selected_detector_sha256=selected['sha256'],
                    selected_midpoint_sha256=selected['sha256']) for r in train if r['sequence'] in sequences]
                legacy.dump_rows(folder/'oof_predictions.jsonl.gz',aux)
            with patch.object(legacy,'checked',return_value=(actual,policy,None,None,{})):
                runner.assess(out)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['protocol'],follow.VERSION)
            self.assertFalse(report['independent_VAL_claim']); self.assertFalse(report['TEST_read'])
            candidate=json.loads((out/'candidate_policy.json').read_text())
            self.assertEqual(candidate['simple_policy']['models']['angle'],policy['simple_policy']['models']['angle'])
            self.assertEqual(candidate['candidate_identity']['protocol'],follow.VERSION)
            self.assertFalse(candidate['candidate_identity']['auxiliary_models_required_online'])


class ReplayComparisonTests(unittest.TestCase):
    def test_finite_loss_and_gradient_preserve_separate_tolerances(self):
        original=dict(total_loss=3.,gradient_before_clip=4.)
        close=follow.compare_replay_trace(3.+1e-6,4.+5e-5,original)
        self.assertTrue(close['passed'])
        self.assertAlmostEqual(close['loss_difference'],1e-6)
        self.assertAlmostEqual(close['gradient_difference'],5e-5)
        self.assertFalse(follow.compare_replay_trace(3.+5e-5,4.,original)['passed'])
        self.assertFalse(follow.compare_replay_trace(3.,4.+2e-4,original)['passed'])
        zero=dict(total_loss=0.,gradient_before_clip=0.)
        self.assertTrue(follow.compare_replay_trace(9e-6,9e-5,zero)['passed'])
        self.assertFalse(follow.compare_replay_trace(2e-5,0.,zero)['passed'])

    def test_nonfinite_json_roundtrip_reproduces_only_same_anomaly(self):
        values=(float('nan'),float('inf'),float('-inf'))
        for expected in values:
            saved=json.loads(json.dumps(follow.json_safe(dict(
                total_loss=expected,gradient_before_clip=expected)),allow_nan=False))
            for actual in values:
                with self.subTest(expected=saved['total_loss'],actual=repr(actual)):
                    result=follow.compare_replay_trace(actual,actual,saved)
                    matched=(math.isnan(actual) and math.isnan(expected)) or actual==expected
                    self.assertEqual(result['passed'],matched)
                    self.assertIsNone(result['loss_difference'])
                    self.assertIsNone(result['gradient_difference'])
                    self.assertTrue(result['loss_comparison']['nonfinite'])
                    report=json.loads(json.dumps(follow.json_safe(result),allow_nan=False))
                    self.assertEqual(report['loss_comparison']['expected'],saved['total_loss'])

    def test_finite_nonfinite_mismatch_rejected_in_either_direction(self):
        for value in (float('nan'),float('inf'),float('-inf')):
            for actual,expected in ((value,1.),(1.,value)):
                for key in ('loss','gradient'):
                    with self.subTest(actual=repr(actual),expected=repr(expected),key=key):
                        original=follow.json_safe(dict(total_loss=expected if key=='loss' else 1.,
                            gradient_before_clip=expected if key=='gradient' else 1.))
                        result=follow.compare_replay_trace(actual if key=='loss' else 1.,
                            actual if key=='gradient' else 1.,original)
                        self.assertFalse(result['passed'])
                        self.assertIsNone(result[key+'_difference'])

    def test_matching_nonfinite_gradient_with_finite_loss_preserves_evidence(self):
        result=follow.compare_replay_trace(2.,float('nan'),
            dict(total_loss=2.,gradient_before_clip='nan'))
        self.assertTrue(result['passed'])
        self.assertEqual(result['loss_difference'],0.)
        self.assertIsNone(result['gradient_difference'])
        self.assertFalse(result['loss_comparison']['nonfinite'])
        self.assertTrue(result['gradient_comparison']['nonfinite'])

    def test_malformed_saved_value_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            follow.compare_replay_trace(1.,1.,dict(total_loss='corrupt',gradient_before_clip=1.))


try:
    import torch
    from mmrotate.core.hooks.port_oof_trace_hook import (PortOOFTraceHook,PortOOFTraceCommitHook,
        TraceStopped,cpu_tree,rng_state,restore_rng,runtime_state,restore_runtime)
    HAS_RUNTIME=True
except ImportError:
    HAS_RUNTIME=False


@unittest.skipUnless(HAS_RUNTIME,'Torch/MMCV required; run on server before training')
class RuntimeTests(unittest.TestCase):
    def model(self):
        model=torch.nn.Module(); model.backbone=torch.nn.Linear(1,1)
        model.neck=torch.nn.Linear(1,1); model.bbox_head=torch.nn.Module()
        model.bbox_head.retina_cls=torch.nn.Linear(1,1)
        model.bbox_head.retina_reg=torch.nn.Conv2d(1,15,1)
        model.bbox_head.assigner=SimpleNamespace(_local_call_count=0)
        model.bbox_head.loss_cls=torch.nn.Module()
        model.bbox_head.loss_cls.register_buffer('_local_iter',torch.tensor(0))
        return model

    def trace_runner(self,tmp):
        model=self.model(); optimizer=torch.optim.SGD(model.parameters(),lr=.0025,momentum=.9)
        runner=SimpleNamespace(model=model,optimizer=optimizer,iter=0,epoch=0)
        trace=PortOOFTraceHook(tmp); trace.model=model; trace.runner=runner
        trace.log=(Path(tmp)/'updates.jsonl').open('x'); trace.before=trace.snapshot(runner)
        trace.batch=dict(img_metas=[]); trace.metas=[]
        return trace,runner

    def set_losses(self,trace,runner,large=False):
        model=runner.model
        cls=model.bbox_head.retina_cls(torch.ones(1,1)).square().mean()
        reg=model.bbox_head.retina_reg(torch.ones(1,1,1,1)).square().mean()
        if large: cls=cls+6.
        trace.losses=dict(loss_cls=[cls],loss_bbox=[reg])
        runner.outputs=dict(loss=cls+reg)

    def test_quarantine_before_optimizer_and_independent_cpu_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace,runner=self.trace_runner(tmp); self.set_losses(trace,runner,large=True)
            before=cpu_tree(runner.model.state_dict())
            with self.assertRaises(TraceStopped): trace.after_train_iter(runner)
            self.assertTrue(all(torch.equal(v,runner.model.state_dict()[k]) for k,v in before.items()))
            self.assertEqual(len(runner.optimizer.state),0)
            snapshot=torch.load(str(Path(tmp)/'before_anomaly.pt'),map_location='cpu')
            self.assertTrue(all(t.device.type=='cpu' for t in snapshot['model'].values()))
            self.assertFalse(json.loads((Path(tmp)/'updates.jsonl').read_text())['optimizer_updated'])

    def test_probe_does_not_accumulate_or_change_actual_backward_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace,runner=self.trace_runner(tmp); self.set_losses(trace,runner)
            named=[p for p in runner.model.parameters() if p.requires_grad]
            expected=torch.autograd.grad(runner.outputs['loss'],named,retain_graph=True,allow_unused=True)
            trace.after_train_iter(runner)
            self.assertTrue(all(p.grad is None for p in named))
            runner.outputs['loss'].backward()
            for p,g in zip(named,expected):
                self.assertTrue((p.grad is None and g is None) or torch.equal(p.grad,g))
            torch.nn.utils.clip_grad_norm_(named,10.); runner.optimizer.step()
            runner._port_oof_trace=trace; PortOOFTraceCommitHook().after_train_iter(runner)
            trace.log.close()
            self.assertTrue(json.loads((Path(tmp)/'updates.jsonl').read_text())['optimizer_updated'])

    def test_restore_rng_and_non_state_dict_assigner(self):
        model=self.model(); state=runtime_state(model); rng=rng_state()
        reference=torch.rand(5); model.bbox_head.assigner._local_call_count=6412
        model.eval(); restore_rng(rng); restore_runtime(model,state)
        self.assertTrue(torch.equal(torch.rand(5),reference))
        self.assertTrue(model.training)
        self.assertEqual(model.bbox_head.assigner._local_call_count,0)

    def test_actual_loss_instrumentation_preserves_nfl_value_gradient_and_calls(self):
        from mmrotate.models.losses.sym_nfl_loss import SymNFLLoss
        from mmcv.runner import HOOKS,EpochBasedRunner
        # Real SymNFL, a minimal decoder and a single head level. Two independent
        # GTs verify image isolation, while wrappers see the real method signature.
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.num_classes=1
                self.loss_cls=SymNFLLoss(isolate_images=True,strict_finite=True)
                self.bbox_coder=SimpleNamespace(decode=lambda anchors,targets:targets)
            def loss_single(self,cls_score,bbox_pred,anchors,labels,label_weights,bbox_targets,
                            bbox_weights,num_total_samples,decode_max_size=None):
                logits=cls_score.permute(0,2,3,1).reshape(-1,1)
                boxes=bbox_pred.permute(0,2,3,1).reshape(-1,5)
                ids=torch.arange(2).repeat_interleave(1)
                return self.loss_cls(logits,labels.reshape(-1),boxes,bbox_targets.reshape(-1,5),
                    weight=label_weights.reshape(-1),avg_factor=num_total_samples,
                    pred_image_ids=ids,gt_image_ids=ids)
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.bbox_head=Head()
            def forward_train(self,img,img_metas,gt_bboxes,gt_labels,**kwargs):
                gt=torch.stack(gt_bboxes); labels=torch.zeros(2,1,dtype=torch.long)
                value=self.bbox_head.loss_single(img,gt.reshape(2,5,1,1),gt,labels,torch.ones(2,1),gt,
                    torch.ones_like(gt),2.,decode_max_size=1024.)
                return dict(loss_cls=[value],loss_bbox=[img.sum()*0.])
            def train_step(self,data,optimizer):
                losses=self.forward_train(**data)
                return dict(loss=sum(losses['loss_cls'])+sum(losses['loss_bbox']),num_samples=2,log_vars={})
        metadata=[dict(filename='real_00001.jpg'),dict(filename='sim_00001.jpg')]
        gts=[torch.tensor([[20.,20.,12.,4.,0.]]),torch.tensor([[40.,40.,16.,6.,0.]])]
        x=torch.full((2,1,1,1),-2.,requires_grad=True)
        baseline=Model(); expected=baseline.forward_train(x,metadata,gts,[torch.tensor([0])]*2)['loss_cls'][0]
        expected_gradient=torch.autograd.grad(expected,x)[0]
        model=Model(); trace=PortOOFTraceHook('unused'); trace.install(model,capture_failure=False)
        actual=model.forward_train(x,metadata,gts,[torch.tensor([0])]*2)['loss_cls'][0]
        self.assertTrue(torch.equal(expected,actual))
        self.assertTrue(torch.equal(expected_gradient,torch.autograd.grad(actual,x)[0]))
        self.assertEqual(int(model.bbox_head.loss_cls._local_iter),1)
        self.assertEqual(len(trace.targets),2)
        self.assertAlmostEqual(trace.classification[0]['positive']+trace.classification[0]['negative'],float(actual),places=6)
        self.assertEqual(set(trace.batch),{'img','img_metas','gt_bboxes','gt_labels'})
        # Registration order is exercised against MMCV's real priority machinery.
        with tempfile.TemporaryDirectory() as tmp:
            import logging
            runner=EpochBasedRunner(model,optimizer=torch.optim.SGD([x],lr=.1),
                work_dir=tmp,logger=logging.getLogger('trace-test'),max_epochs=1)
            runner.register_hook(PortOOFTraceCommitHook(),priority='LOW')
            from mmrotate.core.hooks.port_strict_optimizer_hook import PortStrictOptimizerHook
            runner.register_hook(PortStrictOptimizerHook(grad_clip=dict(max_norm=10,norm_type=2)),priority='ABOVE_NORMAL')
            runner.register_hook(PortOOFTraceHook(tmp+'/trace'),priority='VERY_HIGH')
            self.assertEqual([type(h).__name__ for h in runner.hooks],
                ['PortOOFTraceHook','PortStrictOptimizerHook','PortOOFTraceCommitHook'])


if __name__=='__main__': unittest.main()
