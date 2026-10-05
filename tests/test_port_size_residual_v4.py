"""Residual semantics, real autograd/IO, fresh paired budget and flag isolation."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import inspect
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from crane_project.utils import port_size_residual_v4 as core
from crane_project.utils import port_size_residual_comparison_v4 as comparison
from crane_project.tools import run_port_size_residual_v4 as entry

HAS_TORCH=importlib.util.find_spec('torch') is not None
STANDARDIZER=dict(mean=[0.,0.,0.],scale=[1.,1.,1.])


def box():return [128.,128.,80.,40.,.37]


class Runtime:
    def decide(self,p,size,method='simple'):
        return dict(final_box_original=deepcopy(p),center_accepted=p is not None,
            size_accepted=p is not None and p[2]>60.,angle_accepted=p is not None,
            risks={'size':.2 if p is None or p[2]>60 else .7,'angle':.2})


def rows():
    records=[]
    for role in comparison.ROLES:
        for domain in ('real','sim'):
            for i in range(4):
                pred=box()+[.8] if i!=3 else None
                if i==1:pred[2]=50.
                source=dict(image=role+domain+str(i),domain=domain,sequence=domain+'_fixture',frame_id=i,
                    split='val' if role=='val' else ('train' if domain=='real' else 'train_sim'),
                    image_size=[256,256],gt=box(),pred=pred)
                raw={a:[-3.,-3.] if pred is not None else None for a in core.ARMS}
                if i==2:raw['a1']=None
                evidence=core.online_evidence(raw,pred,source['image_size'],Runtime())
                records.append(comparison.annotate(source,evidence,role))
    return records


def completed_report(path,status,contract,**extra):
    path.parent.mkdir(parents=True,exist_ok=True)
    entry.base.write_new(path,dict(status=status,contract=contract,**extra))
    entry.base.write_new(path.parent/'completion.json',dict(status=status,contract=contract,
        contract_sha256=entry.simple.fingerprint(contract),artifacts={path.name:entry.base.sha(path)}))


class ResidualTests(unittest.TestCase):
    def test_signed_targets_and_size_event(self):
        gt=box();pred=box()+[.8];pred[2]*=.85;pred[3]*=1.15
        np.testing.assert_allclose(core.residual_target(pred,gt),np.log([.85,1.15]),atol=1e-14)
        self.assertFalse(core.target_is_good(core.residual_target(pred,gt)))
        for l,s in ((.9,1.1),(.90001,1.09999),(1.,1.)):
            self.assertTrue(core.target_is_good(np.log([l,s])))
        self.assertFalse(core.target_is_good(np.log([.899,1.])));self.assertFalse(core.target_is_good(np.log([1.,1.101])))
        self.assertIsNone(core.residual_target(None,gt))

    def test_equivalent_OBB_and_isotropic_coordinates(self):
        g=box();p=box()+[.8];p[2]*=1.15;expected=core.residual_target(p,g)
        for q in (p,[*p[:2],p[3],p[2],p[4]+math.pi/2,p[5]],[*p[:4],p[4]+math.pi,p[5]]):
            np.testing.assert_allclose(core.residual_target(q,g),expected,atol=1e-14)
            scaled=q.copy();scaled[:4]=[x*.5 for x in scaled[:4]]
            gt=g.copy();gt[:4]=[x*.5 for x in gt[:4]]
            np.testing.assert_allclose(core.residual_target(scaled,gt),expected,atol=1e-14)
        q=p.copy();q[0]+=100;q[4]+=.8
        np.testing.assert_allclose(core.residual_target(q,g),expected)

    def test_NLL_independent_derivatives_and_optimum(self):
        self.assertTrue(core.numerical_probe([-.13,.24])['passed'])
        sigma=np.array([.07,.2]);raw=np.log(np.expm1(sigma-.001))
        terms=core.nll_numpy(raw,sigma)
        np.testing.assert_allclose(terms['derivative'],0.,atol=1e-13)
        a=core.nll_numpy(raw,[.07,-.2]);self.assertAlmostEqual(a['total'],terms['total'])
        self.assertLess(core.nll_numpy([-3.,-3.],[0.,0.])['total'],0.)

    def test_model_mass_uses_asymmetric_relative_error_interval(self):
        raw=np.log(np.expm1(np.array([.1,.2])-.001));value=core.distribution_reading(raw)
        expected=np.prod([.5*(math.erf(math.log(1.1)/(s*math.sqrt(2)))-math.erf(math.log(.9)/(s*math.sqrt(2)))) for s in (.1,.2)])
        self.assertAlmostEqual(value['model_good_mass'],expected,14)
        self.assertFalse(value['calibrated_error_probability'])
        risks=[core.distribution_reading([z,z])['risk'] for z in (-8.,-4.,-2.,0.,10.)]
        self.assertEqual(risks,sorted(risks))
        self.assertTrue(np.all(core.sigma_from_raw([-1000.,1000.])>0))
        self.assertFalse(core.distribution_reading(None)['defined'])

    def test_online_API_and_three_original_flags_remain_GT_free(self):
        for function in (core.online_evidence,core.distribution_reading,entry.inputs):
            self.assertFalse(set(inspect.signature(function).parameters)&{'gt','domain','sequence','row','history'})
        p=box()+[.8];before=deepcopy(p);value=core.online_evidence({'a0':[-3.,-3.],'a1':None},p,[256,256],Runtime())
        self.assertEqual(p,before);self.assertTrue(value['frozen']['center_accepted'])
        self.assertTrue(value['frozen']['size_accepted']);self.assertTrue(value['frozen']['angle_accepted'])
        self.assertFalse(value['candidate_flags_created']);self.assertFalse(value['GT_online'])
        self.assertFalse(core.online_evidence({'a0':None,'a1':None},None,[256,256],Runtime())['frozen']['center_accepted'])

    def test_standardizer_fits_only_output_descriptors_and_not_GT(self):
        fit=[dict(split='train',pred=box()+[.8],image_size=[256,256],gt=box())]
        standard=core.fit_standardizer(fit);other=deepcopy(fit);other[0]['gt'][2]*=10
        self.assertEqual(standard,core.fit_standardizer(other))
        self.assertTrue(all(x>0 for x in standard['scale']))
        with self.assertRaises(ValueError):core.fit_standardizer([dict(fit[0],split='val')])
        with self.assertRaises(ValueError):core.fit_standardizer([dict(fit[0],pred=None)])

    def test_malformed_geometry_dispersion_and_supervision_fail(self):
        for value in ([float('nan'),0.],[0.],[0.,float('inf')]):
            with self.assertRaises(ValueError):core.distribution_reading(value)
        with self.assertRaises(ValueError):core.residual_target([128,128,-1,40,0,.8],box())
        with self.assertRaises(ValueError):core.nll_numpy([-3.,-3.],[float('nan'),0.])

    def test_missing_outputs_and_unavailable_good_evidence_keep_denominators(self):
        summary=comparison.summarize(rows());g=summary['summary']['val']['domain:real']
        self.assertEqual((g['frames'],g['outputs'],g['missing_outputs']),(4,3,1))
        self.assertEqual(g['output_coverage'],.75);self.assertEqual(g['center_hit_rate_on_outputs'],1.)
        self.assertEqual(g['full_frame_center_correct_coverage'],.75)
        self.assertEqual(g['availability']['a1']['unavailable_good'],1)
        self.assertEqual(g['primary_same_count']['a1']['FR'],1)
        self.assertFalse(summary['final_deployment_policy_created'])
        self.assertFalse(summary['prespecified_continuation_review']['eligible_to_design_separate_constrained_workpoint'])

    def test_same_count_confusion_identity_and_boundary_tie_bounds(self):
        data=[r for r in rows() if r['pred'] is not None][:3]
        for r in data:r['arms']['a1']=core.distribution_reading([-3.,-3.])
        matched=comparison.matched(data,1,4)['a1']
        self.assertEqual(matched['accepted'],1)
        self.assertEqual(matched['tie_bounds']['bad_min_over_tie'],0)
        self.assertEqual(matched['tie_bounds']['bad_max_over_tie'],1)
        self.assertEqual(matched['FR_tie_bounds'],{'min':1,'max':2})
        self.assertEqual(matched['FA']+sum(not r['size_bad'] for r in data)-1,matched['FR'])
        with self.assertRaises(ValueError):comparison.matched(data,4,4)

    def test_comparison_threshold_label_isolation_and_replay(self):
        records=rows();cutoffs=comparison.comparison_cutoffs(records);changed=deepcopy(records)
        for r in changed:
            if r['size_bad'] is not None:r['size_bad']=not r['size_bad']
        self.assertEqual(cutoffs,comparison.comparison_cutoffs(changed))
        result=comparison.summarize(records)
        self.assertEqual(result,comparison.summarize(json.loads(json.dumps(records))))
        self.assertFalse(result['calibration_parameters_fitted'])

    def test_no_TEST_duplicate_or_missing_role(self):
        records=rows()
        with self.assertRaises(ValueError):comparison.summarize(records+[records[0]])
        with self.assertRaises(ValueError):comparison.summarize(records[:8])
        with self.assertRaises(ValueError):comparison.annotate(dict(records[0],split='test'),
            {k:records[0][k] for k in ('arms','frozen','frozen_score_only')},'val')

    def test_distribution_quality_and_ranking_are_separate_gates(self):
        groups=comparison.summarize(rows())['summary']
        for role,values in groups.items():
            for name,g in values.items():
                for arm in core.ARMS:g['common_distribution'][arm]['mean_nll']=1. if arm=='a0' else .5
                if role=='val':
                    g['availability']['a1']['defined']=g['outputs']
                    for method,p in g['primary_same_count'].items():
                        n=4 if method=='a1' else 10;p['reachable']=True
                        p['tie_bounds'].update(bad_min_over_tie=n,bad_max_over_tie=n)
                        p['FR_tie_bounds']={'min':n,'max':n}
                    for m in comparison.METHODS:g['ranks']['common']['methods'][m]['error_auroc']=.8 if m=='a1' else .7
        for d,k in core.CONTINUATION['primary_accept_counts'].items():groups['val']['domain:'+d]['frozen_simple_count']=k
        review=comparison.continuation_review(groups)
        self.assertTrue(review['D']['passed']);self.assertTrue(review['Q1']['passed'])
        self.assertTrue(review['eligible_to_design_separate_constrained_workpoint'])
        groups['val']['sequence:real_fixture']['primary_same_count']['a1']['tie_bounds']['bad_max_over_tie']=11
        review=comparison.continuation_review(groups)
        self.assertTrue(review['D']['passed']);self.assertFalse(review['Q1']['passed'])
        groups['val']['domain:real']['common_distribution']['a1']['mean_nll']=2.
        self.assertFalse(comparison.continuation_review(groups)['D']['passed'])

    def test_stages_verify_nested_hashes_failure_and_experiment_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'check'/'check_report.json';contract={'fixed':True}
            completed_report(path,'PASS',contract);entry.stage(path,'PASS',contract,root)
            with self.assertRaises(ValueError):entry.stage(path,'PASS',{'fixed':False},root)
            with self.assertRaises(ValueError):entry.stage(path,'PASS',contract,root/'another')
            artifact=path.parent/'a0'/'marker.json';artifact.parent.mkdir();artifact.write_text('{}')
            proof=json.loads((path.parent/'completion.json').read_text());proof['artifacts']['a0/marker.json']=entry.base.sha(artifact)
            (path.parent/'completion.json').write_text(json.dumps(proof));entry.stage(path,'PASS',contract,root)
            artifact.write_text('{"changed":true}')
            with self.assertRaises(ValueError):entry.stage(path,'PASS',contract,root)
            (path.parent/'failure.json').write_text('{}')
            with self.assertRaises(ValueError):entry.stage(path,'PASS',contract,root)

    def test_new_outputs_cannot_overwrite_baseline_or_scatter_stages(self):
        root=entry.ROOT/'work_dirs'
        args=SimpleNamespace(mode='train',out_dir=root/'port_midpoint_reliability_v1_fit')
        with self.assertRaises(ValueError):entry.validate_output(args)
        args.out_dir=root/'port_size_residual_v4_fixture_fresh'/'train';entry.validate_output(args)
        args.out_dir=root/'port_size_residual_v4_fixture_fresh'/'assess'
        with self.assertRaises(ValueError):entry.validate_output(args)

    def test_exact_source_protocol_parent_pins_and_Python38(self):
        protocol,sources=entry.checked_sources();self.assertEqual(protocol,core.protocol_document())
        self.assertFalse(protocol['final_deployment_policy_created']);self.assertFalse(protocol['test_read'])
        for name in sources['sources']:
            if name.endswith('.py'):ast.parse((entry.ROOT/name).read_text(),feature_version=(3,8))
        self.assertNotIn('torch',entry.__dict__)


@unittest.skipUnless(HAS_TORCH,'Torch required for real candidate autograd/IO tests')
class TorchTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch;torch.set_num_threads(1)
        self.devices=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])

    def fixture(self,device='cpu'):
        from crane_project.utils.port_size_residual_v4_torch import SizeDispersion
        t=self.torch;t.manual_seed(1701)
        model=SizeDispersion(STANDARDIZER).to(device)
        roi=t.randn(1,256,9,9,device=device);support=t.ones(1,1,9,9,device=device)
        descriptor=t.tensor([[1.,-.3,.7]],device=device)
        return model,(roi,support,descriptor)

    def test_real_NLL_autograd_float64_oracle_both_signs_and_negative_loss(self):
        from crane_project.utils.port_size_residual_v4_torch import verify_autograd
        for device in self.devices:
            for e in ([0.,0.],[-.3,.2],[.0001,-.0001]):
                self.assertTrue(verify_autograd([-3.,-2.],e,device)['passed'])

    def test_same_init_a0_image_invariance_a1_live_image_gradients(self):
        from crane_project.utils.port_size_residual_v4_torch import SizeDispersion,loss_terms,precision
        for device in self.devices:
            model,example=self.fixture(device);other=SizeDispersion(STANDARDIZER).to(device)
            other.load_state_dict(model.state_dict());self.assertEqual(entry.base.state_digest(model),entry.base.state_digest(other))
            with precision():
                a=model(*example,'a0');b=model(example[0]*3,example[1],example[2],'a0')
                self.assertTrue(self.torch.equal(a,b))
                loss_terms(model(*example,'a1'),self.torch.tensor([[.12,-.07]],device=device))['total'].backward()
            self.assertGreater(float(model.stem.weight.grad.abs().sum()),0.)
            self.assertIsNone(example[0].grad)

    def test_actual_update_clipping_groups_and_error_stops_before_optimizer(self):
        from crane_project.utils.port_size_residual_v4_torch import update,precision
        for device in self.devices:
            model,example=self.fixture(device);t=self.torch;optimizer=t.optim.Adam(model.parameters(),lr=.001)
            target=t.tensor([[.12,-.09]],device=device)
            with precision():raw=model(*example,'a1')
            record=update(raw,target,model,optimizer,'a1',True)
            self.assertGreater(record['parameter_groups_preclip']['stem'],0.)
            self.assertGreater(record['parameter_groups_preclip']['output'],0.)
            self.assertLessEqual(record['postclip_norm'],10.00001)
            state=entry.base.state_digest(model)
            with precision():raw=model(*example,'a1')
            original=core.nll_numpy
            def wrong(*args):
                value=original(*args);value['derivative']=[1000.,1000.];return value
            with patch.object(core,'nll_numpy',side_effect=wrong),patch.object(optimizer,'step',wraps=optimizer.step) as step:
                with self.assertRaises(ValueError):update(raw,target,model,optimizer,'a1')
                self.assertEqual(step.call_count,0)
            self.assertEqual(state,entry.base.state_digest(model))

    def test_A0_stem_stays_fixed_and_all_inputs_are_detached(self):
        from crane_project.utils.port_size_residual_v4_torch import update,precision
        model,example=self.fixture();optimizer=self.torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
        before={k:v.clone() for k,v in model.stem.state_dict().items()}
        with precision():raw=model(*example,'a0')
        record=update(raw,self.torch.tensor([[.1,-.03]]),model,optimizer,'a0')
        self.assertEqual(record['parameter_groups_preclip']['stem'],0.)
        self.assertTrue(all(self.torch.equal(v,model.stem.state_dict()[k]) for k,v in before.items()))
        with self.assertRaises(ValueError):model(example[0].requires_grad_(),example[1],example[2],'a1')

    def test_no_support_A1_unavailable_A0_still_uses_only_geometry(self):
        model,example=self.fixture();empty=self.torch.zeros_like(example[1])
        self.assertTrue(bool(self.torch.isfinite(model(example[0],empty,example[2],'a0')).all()))
        with self.assertRaises(ValueError):model(example[0],empty,example[2],'a1')

    def test_candidate_precision_restores_baseline_backend_flags_after_failure(self):
        from crane_project.utils.port_size_residual_v4_torch import precision
        t=self.torch;before=(t.backends.cudnn.deterministic,t.backends.cudnn.benchmark)
        flags=[(owner,owner.allow_tf32) for owner in (t.backends.cuda.matmul,t.backends.cudnn) if hasattr(owner,'allow_tf32')]
        with self.assertRaises(RuntimeError):
            with precision():
                self.assertTrue(t.backends.cudnn.deterministic);self.assertFalse(t.backends.cudnn.benchmark)
                for owner,_ in flags:self.assertFalse(owner.allow_tf32)
                raise RuntimeError('injected')
        self.assertEqual(before,(t.backends.cudnn.deterministic,t.backends.cudnn.benchmark))
        self.assertTrue(all(owner.allow_tf32==value for owner,value in flags))

    def test_final_box_sampling_mapping_canonical_swap_and_padding(self):
        from crane_project.utils import port_geometry_refine_g_v1 as geometry
        t=self.torch;meta=dict(img_shape=[256,240,3],ori_shape=[256,240,3],pad_shape=[256,256,3],
            scale_factor=[1.]*4,flip=False)
        yy,xx=t.meshgrid(t.arange(32.),t.arange(32.))
        features=[((xx+2*yy)/64)[None,None].expand(1,256,32,32).contiguous()];p=box()+[.8]
        formal=SimpleNamespace(g=geometry)
        a=entry.inputs(features,meta,p,[240,256],formal,t)
        q=[*p[:2],p[3],p[2],p[4]+math.pi/2,p[5]]
        b=entry.inputs(features,meta,q,[240,256],formal,t)
        self.assertTrue(t.allclose(a[0],b[0],atol=1e-5));self.assertTrue(t.allclose(a[2],b[2]))
        self.assertTrue(bool(((a[1]>=0)&(a[1]<=1.00001)).all()))
        with self.assertRaises(ValueError):entry.inputs(features,dict(meta,scale_factor=[.5,.8,.5,.8]),p,[240,256],formal,t)

    def test_actual_save_reload_marker_arm_role_and_no_overwrite(self):
        model,example=self.fixture();t=self.torch;optimizer=t.optim.Adam(model.parameters(),lr=.001)
        contract={'fixture':True};initial=entry.base.state_digest(model)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);item=entry.save_reload(root/'a1',model,optimizer,contract,'a1',4,1536,initial,
                {'fixed':True},example,t)
            self.assertTrue(item['save_reload_exact']);p=root/item['path']
            entry.load_checkpoint(p,contract,'a1','paired_size_residual_train',t)
            with self.assertRaises(ValueError):entry.load_checkpoint(p,contract,'a0','paired_size_residual_train',t)
            with self.assertRaises(FileExistsError):entry.save_reload(root/'a1',model,optimizer,contract,'a1',4,1536,
                initial,{'fixed':True},example,t)

    def test_real_3072_updates_fresh_pair_same_order_budget_and_saved_final_epoch(self):
        # Synthetic scheduling/optimizer integration; no project images or model
        # performance inference. Actual compact heads and checkpoint IO run.
        from crane_project.utils.port_size_residual_v4_torch import SizeDispersion
        t=self.torch;a0,example=self.fixture();a1=SizeDispersion(STANDARDIZER)
        a1.load_state_dict(a0.state_dict());arms={'a0':a0,'a1':a1};initial=entry.base.state_digest(a0)
        detector=t.nn.Linear(1,1);head=t.nn.Linear(1,1);frozen=entry.snapshot(detector,head)
        fit=[dict(image='fit_%03d'%i,domain='real' if i<256 else 'sim',split='train' if i<256 else 'train_sim',
            gt=box(),pred=box()+[.8],image_size=[256,256]) for i in range(384)]
        for i,r in enumerate(fit):r['pred'][2]*=1.02+(i%5)*.01
        contract={'fixture':True};prepared=dict(contract=contract,split={'fit':fit})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);check=root/'check'/'check_report.json';smoke=root/'smoke'/'smoke_report.json'
            completed_report(check,entry.STATUSES['check'],contract,standardizer=STANDARDIZER)
            completed_report(smoke,entry.STATUSES['smoke'],contract,initial_sha256=initial,frozen=frozen,
                discarded=True,check_report_sha256=entry.base.sha(check))
            out=root/'train';out.mkdir();args=SimpleNamespace(check_report=check,smoke_report=smoke,out_dir=out,gpu=0)
            def view(*args):return [example[0]],{}, {},dict(b=0.,midpoint=0.)
            with patch.object(entry,'models',return_value=(None,t,detector,head,None,arms,initial)),\
                 patch.object(entry,'paired_view',side_effect=view),patch.object(entry,'inputs',return_value=example),\
                 patch.object(entry,'unchanged') as unchanged,patch.object(t.cuda,'max_memory_allocated',return_value=0),\
                 redirect_stdout(io.StringIO()):
                entry.paired_train(args,prepared)
            report=json.loads((out/'train_report.json').read_text())
            events=[json.loads(s) for s in (out/'train_steps.jsonl').read_text().splitlines()]
            self.assertEqual(report['steps_per_arm'],1536);self.assertEqual(report['total_optimizer_updates'],3072)
            self.assertEqual(unchanged.call_count,1536)
            self.assertEqual(len(events),3072)
            for epoch in range(1,5):
                expected=[fit[i]['image'] for i in np.random.RandomState(1701+epoch).permutation(384)]
                for arm in core.ARMS:self.assertEqual([r['image'] for r in events if r['epoch']==epoch and r['arm']==arm],expected)
            self.assertTrue(all(v['path'].endswith('epoch_04.pth') for v in report['checkpoints'].values()))
            self.assertFalse(report['smoke_reused']);self.assertFalse(report['performance_PASS'])
            self.assertEqual(report['frozen'],entry.snapshot(detector,head))

    def test_real_smoke_14_initial_views_updates_discarded_and_assessment_replays(self):
        # Complete synthetic smoke/assess orchestration with real CPU heads and
        # file-stream checkpoint IO. The frozen detector adapter is isolated.
        from crane_project.utils.port_size_residual_v4_torch import SizeDispersion
        t=self.torch;a0,example=self.fixture();a1=SizeDispersion(STANDARDIZER)
        a1.load_state_dict(a0.state_dict());arms={'a0':a0,'a1':a1};initial=entry.base.state_digest(a0)
        detector=t.nn.Linear(1,1);head=t.nn.Linear(1,1);frozen=entry.snapshot(detector,head)
        fit=[dict(image='fit_%03d'%i,domain='real' if i<256 else 'sim',split='train' if i<256 else 'train_sim',
            gt=box(),pred=box()+[.8],image_size=[256,256]) for i in range(384)]
        for r in fit:r['pred'][2]*=1.03
        selected=fit[:10]+fit[256:260]
        prepared=dict(contract={'fixture':True},split={'fit':fit})
        def view(*args):return [example[0]],{}, {},dict(b=0.,midpoint=0.)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);check=root/'check'/'check_report.json'
            completed_report(check,entry.STATUSES['check'],prepared['contract'],standardizer=STANDARDIZER)
            smoke_dir=root/'smoke';smoke_dir.mkdir()
            args=SimpleNamespace(check_report=check,out_dir=smoke_dir,gpu=0)
            with patch.object(entry,'models',return_value=(None,t,detector,head,None,arms,initial)),\
                 patch.object(entry.reference,'ideal_selection',return_value=selected),\
                 patch.object(entry,'paired_view',side_effect=view),patch.object(entry,'inputs',return_value=example),\
                 patch.object(entry,'unchanged'),redirect_stdout(io.StringIO()):
                entry.smoke(args,prepared)
            smoke_path=smoke_dir/'smoke_report.json';smoked=json.loads(smoke_path.read_text())
            self.assertEqual(len(smoked['initial_measurements']),28);self.assertEqual(len(smoked['updates']),8)
            self.assertTrue(smoked['discarded']);self.assertEqual(smoked['initial_sha256'],initial)
            self.assertTrue(all(r['autograd']['passed'] for r in smoked['initial_measurements']))
            entry.base.write_new(smoke_dir/'completion.json',dict(status=entry.STATUSES['smoke'],contract=prepared['contract'],
                contract_sha256=entry.simple.fingerprint(prepared['contract']),artifacts=entry.artifact_index(smoke_dir)))
            train_dir=root/'train';train_dir.mkdir();items={}
            for arm,model in arms.items():
                items[arm]=entry.save_reload(train_dir/arm,model,t.optim.Adam(model.parameters(),lr=.001),prepared['contract'],
                    arm,4,1536,initial,frozen,example,t)
            train_path=train_dir/'train_report.json'
            completed_report(train_path,entry.STATUSES['train'],prepared['contract'],standardizer=STANDARDIZER,
                check_report_sha256=entry.base.sha(check),smoke_report_sha256=entry.base.sha(smoke_path),
                initial_sha256=initial,steps_per_arm=1536,frozen=frozen,checkpoints=items)
            # Assessment must reload the saved pair, not use its supplied init.
            fresh0,_=self.fixture();fresh1=SizeDispersion(STANDARDIZER);fresh1.load_state_dict(fresh0.state_dict())
            fresh={'a0':fresh0,'a1':fresh1}
            plan=[(r['assessment_role'],{k:v for k,v in r.items() if k in
                ('image','domain','sequence','frame_id','split','image_size','gt','pred')}) for r in rows()]
            prepared.update(plan=plan,policy={},previous=[None,None,None,None,None,{'front_end':{}}])
            assess_dir=root/'assess';assess_dir.mkdir();args=SimpleNamespace(train_report=train_path,out_dir=assess_dir,gpu=0)
            def online_inputs(features,meta,p,size,formal,torch):return example if p is not None else None
            def assessment_view(*args):return [example[0]],dict(img_shape=[256,256,3],ori_shape=[256,256,3],
                pad_shape=[256,256,3],scale_factor=[1.]*4),{},dict(b=0.,midpoint=0.)
            with patch.object(entry,'models',return_value=(None,t,detector,head,None,fresh,initial)),\
                 patch.object(entry,'paired_view',side_effect=assessment_view),patch.object(entry,'inputs',side_effect=online_inputs),\
                 patch.object(entry,'unchanged'),patch.object(entry.migration.new,'MidpointReliability',return_value=Runtime()),\
                 patch.object(t.cuda,'max_memory_allocated',return_value=0),redirect_stdout(io.StringIO()):
                entry.assess(args,prepared)
            report=json.loads((assess_dir/'assessment.json').read_text())
            self.assertEqual(report['state_before'],report['state_after']);self.assertFalse(report['GT_online'])
            self.assertFalse(report['probability_calibrated']);self.assertFalse(report['final_deployment_policy_created'])
            self.assertEqual(report['summary']['val']['all']['frames'],8)
            for arm in core.ARMS:self.assertEqual(entry.base.state_digest(fresh[arm]),entry.base.state_digest(arms[arm]))

    def test_train_rejects_smoke_weights_as_initialization_before_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);contract={'fixture':True};check=root/'check'/'check_report.json';smoke=root/'smoke'/'smoke_report.json'
            completed_report(check,entry.STATUSES['check'],contract,standardizer=STANDARDIZER)
            model,_=self.fixture();frozen={'b':'b','midpoint':'midpoint'}
            completed_report(smoke,entry.STATUSES['smoke'],contract,initial_sha256='smoke_updated_state',frozen=frozen,
                discarded=True,check_report_sha256=entry.base.sha(check))
            out=root/'train';out.mkdir();args=SimpleNamespace(check_report=check,smoke_report=smoke,out_dir=out,gpu=0)
            with patch.object(entry,'models',return_value=(None,self.torch,None,None,None,{},entry.base.state_digest(model))),\
                 patch.object(entry,'snapshot',return_value=frozen):
                with self.assertRaises(ValueError):entry.paired_train(args,{'contract':contract})
            self.assertFalse((out/'train_steps.jsonl').exists())


if __name__=='__main__':unittest.main()
