"""Range physics, real autograd, stage isolation, fixed paired budget and metrics."""
import ast
from contextlib import nullcontext
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from crane_project.utils import port_size_halfpeak_range_v3 as core
from crane_project.utils import port_size_halfpeak_comparison_v3 as comparison
from crane_project.tools import run_port_size_halfpeak_range_v3 as entry


def meta(scale=1.,flip=None):
    n=int(256*scale)
    return dict(img_shape=[n,n,3],ori_shape=[256,256,3],pad_shape=[n,n,3],scale_factor=[scale]*4,
        flip=flip is not None,flip_direction=flip)


def box():return [128.,128.,80.,40.,.37]


def records():
    rows=[]
    for role in ('reference_holdout_train','val'):
        for domain in ('real','sim'):
            for i in range(4):
                pred=box()+[.8] if i!=3 else None
                if i==1:pred[2]=50.
                source=dict(image=role+domain+str(i),domain=domain,sequence=domain+'_fixture',frame_id=i,
                    assessment_role=role,split='val' if role=='val' else ('train' if domain=='real' else 'train_sim'),
                    image_size=[256,256],gt=box(),pred=pred)
                arms={}
                for arm in core.ARMS:
                    defined=i<2
                    frozen=dict(final_box_original=pred,center_accepted=pred is not None,
                        size_accepted=i<2,angle_accepted=pred is not None,risks={'size':.1+i*.2,'angle':.2})
                    arms[arm]=dict(reference=dict(defined=defined,reason='defined' if defined else 'weak_evidence',
                        long_original_px=80.,short_original_px=40.),
                        reading=dict(risk=.1+i*.3 if defined else None),frozen=frozen,frozen_score_only=deepcopy(frozen),
                        offline_range_proxy=dict(eligible=True,inner=.1,outer=.2,
                            mean_absolute_profile_error=.15,contrast_floor_active=False))
                rows.append(comparison.annotate(source,arms,role))
    return rows


class RangeGeometryTests(unittest.TestCase):
    def test_half_peak_targets_ideal_map_and_both_sides_of_each_edge(self):
        constants=core.toy_constants(box(),meta());report=core.numerical_probe(constants)
        self.assertTrue(report['passed'],report)
        self.assertEqual(len(report['cases']),4)
        self.assertEqual(constants['indices'].shape,(37,4))
        self.assertEqual((len(constants['inner']),len(constants['outer'])),(24,24))
        self.assertEqual(len(np.intersect1d(constants['inner'],constants['outer'])),12)
        np.testing.assert_allclose(constants['weights'].sum(axis=1),1.,atol=1e-14)

    def test_isotropic_reflections_periodicity_and_width_swap_preserve_physics(self):
        b=box();variants=[b,[*b[:2],b[3],b[2],b[4]+math.pi/2],[*b[:4],b[4]+math.pi]]
        for scale in (1.,.5):
            for flip in (None,'horizontal','vertical','diagonal'):
                for v in variants:
                    constants=core.toy_constants(v,meta(scale,flip))
                    self.assertTrue(constants['eligible'])
                    self.assertTrue(core.numerical_probe(constants)['passed'])

    def test_contrast_normalization_is_amplitude_background_invariant_above_floor(self):
        constants=core.toy_constants(box(),meta())
        a=core.numpy_terms(core.synthetic_probability(constants,(1.15,.85),.8,.01),constants)
        b=core.numpy_terms(core.synthetic_probability(constants,(1.15,.85),.5,.25),constants)
        np.testing.assert_allclose(a['ratios'],b['ratios'],atol=1e-14)
        self.assertFalse(a['contrast_floor_active']);self.assertFalse(b['contrast_floor_active'])

    def test_range_toy_scales_match_the_actual_unchanged_halfpeak_reader(self):
        from crane_project.utils.port_midpoint_reliability_v1 import template_reference
        settings=json.loads((entry.ROOT/'crane_project/tools/port_midpoint_reliability_v1_protocol.json').read_text())
        constants=core.toy_constants(box(),meta())
        for axis in (0,1):
            for factor in (.85,1.15):
                factors=[1.,1.];factors[axis]=factor
                p=core.synthetic_probability(constants,factors)
                reading=template_reference(p,box(),meta(),settings)
                self.assertTrue(reading['defined'],reading)
                np.testing.assert_allclose([reading['long_original_px'],reading['short_original_px']],
                    np.array(box()[2:4])*factors,atol=1e-6,rtol=1e-7)

    def test_geometry_skip_protects_padding_resolution_clipping_and_background(self):
        tiny=box();tiny[3]=8.
        self.assertEqual(core.projection(tiny,meta())['reason'],'unresolved_GT_short')
        edge=box();edge[0]=2.
        self.assertEqual(core.projection(edge,meta())['reason'],'GT_sample_footprint_clipped')
        padded=meta();padded['img_shape']=[200,200,3];padded['ori_shape']=[200,200,3]
        value=core.projection(box(),padded);_,valid=core.geometry.grid(padded)
        self.assertTrue(value['eligible'])
        self.assertFalse(np.any(value['background_mask'] & ~valid))
        self.assertTrue(valid.ravel()[value['indices']].all())
        huge=[128.,128.,1000.,1000.,0.]
        self.assertEqual(core.projection(huge,meta())['reason'],'insufficient_GT_background')
        with self.assertRaises(ValueError):core.projection(box(),dict(meta(),scale_factor=[.5,.8,.5,.8]))

    def test_signed_below_background_responses_are_not_clamped_away(self):
        constants=core.projection(box(),meta());p=np.full(constants['shape'],.2)
        p.ravel()[constants['indices'].ravel()]=.1
        terms=core.numpy_terms(p,constants)
        self.assertTrue(terms['contrast_floor_active']);self.assertTrue((terms['ratios']<0).all())
        self.assertGreater(terms['inner'],0.);self.assertGreater(terms['outer'],0.)
        with self.assertRaises(ValueError):core.numpy_terms(p[:2],constants)
        p[0,0]=float('nan')
        with self.assertRaises(ValueError):core.numpy_terms(p,constants)

    def test_online_reader_remains_GT_free_and_preserves_center_and_final_box(self):
        self.assertFalse(set(inspect.signature(core.online_reading).parameters)&{'gt','domain','sequence','history'})
        class Runtime:
            def decide(self,p,size,method='simple'):
                return dict(final_box_original=p,center_accepted=p is not None,size_accepted=False,
                    angle_accepted=p is not None,risks={'size':.3,'angle':.2})
        pred=box()+[.8];before=deepcopy(pred)
        with patch.object(entry.migration.new,'template_reference',return_value=dict(
                defined=True,long_original_px=80.,short_original_px=40.)):
            r=core.online_reading(np.zeros((128,128)),pred,[256,256],meta(),Runtime(),{'risk_scale':.1})
        self.assertEqual(pred,before);self.assertTrue(r['frozen']['center_accepted'])
        self.assertFalse(r['frozen']['size_accepted'])
        self.assertFalse(core.online_reading(None,None,[256,256],meta(),Runtime(),{'risk_scale':.1})['reference']['defined'])

    def test_common_full_support_unavailable_costs_and_center_denominators(self):
        rows=records();summary=comparison.summarize(rows);g=summary['summary']['val']['domain:real']
        self.assertEqual((g['frames'],g['outputs'],g['missing_outputs']),(4,3,1))
        self.assertEqual(g['output_coverage'],.75);self.assertEqual(g['center_hit_rate_on_outputs'],1.)
        self.assertEqual(g['full_frame_center_correct_coverage'],.75)
        self.assertEqual(g['references']['a1']['unavailable_good'],1)
        self.assertEqual(g['primary_same_count']['a1']['FA'],1)
        self.assertFalse(summary['final_deployment_policy_created'])
        self.assertFalse(summary['prespecified_continuation_review']['eligible_to_design_separate_constrained_final_workpoint'])
        self.assertFalse(summary['range_and_signed_diagnostics']['val']['domain:real']['a1']['GT_proxy_used_for_online_flags'])

    def test_ties_quantile_label_isolation_JSON_replay_and_95pct_bound(self):
        rows=records();before=comparison.original.diagnostic_cutoffs(rows)
        changed=deepcopy(rows)
        for r in changed:
            if r['size_bad'] is not None:r['size_bad']=not r['size_bad']
        self.assertEqual(before,comparison.original.diagnostic_cutoffs(changed))
        self.assertEqual(comparison.summarize(rows),comparison.summarize(json.loads(json.dumps(rows))))
        population=[r for r in rows if r['pred'] is not None][:3]
        for r in population:
            r['arms']['a1']['reference']['defined']=True;r['arms']['a1']['reading']['risk']=.5
        point=comparison.original.matched(population,1,4)['a1']
        self.assertEqual(point['accepted'],1)
        self.assertEqual(point['tie_bounds']['bad_min_over_tie'],0)
        self.assertEqual(point['tie_bounds']['bad_max_over_tie'],1)
        self.assertEqual(core.acceptance_bound(887,886,642,843)['minimum_bad_accepted'],201)

    def test_no_TEST_duplicates_or_missing_assessment_role(self):
        rows=records()
        with self.assertRaises(ValueError):comparison.summarize(rows[:8])
        with self.assertRaises(ValueError):comparison.summarize(rows+[rows[0]])
        with self.assertRaises(ValueError):comparison.annotate(dict(rows[0],split='test'),rows[0]['arms'],'val')

    def test_preset_review_does_not_reward_reference_or_ranking_only(self):
        groups=comparison.summarize(records())['summary']
        # Populate genuine-gain scalar fixture: tests the decision policy, not a model.
        for role,values in groups.items():
            for name,g in values.items():
                for arm in core.ARMS:
                    g['references'][arm].update(defined=3,all_output_reference_correct_coverage=.2 if arm=='a0' else .5)
                    g['common_reference'][arm].update(mean=.2 if arm=='a0' else .1,p90=.3 if arm=='a0' else .2,
                        within_10_fraction=.2 if arm=='a0' else .5,long_both_fraction=.6,short_both_fraction=.6)
                for method,p in g['primary_same_count'].items():
                    p['reachable']=True;p['tie_bounds'].update(bad_min_over_tie=10,bad_max_over_tie=10 if method!='a1' else 4)
                if role=='val':
                    g['ranks']['common']['rank']['score']['error_auroc']=.7
                    g['ranks']['common']['rank']['simple']['error_auroc']=.6
                    g['ranks']['common']['rank']['a1']['error_auroc']=.75
        for domain,count in (('real',333),('sim',510)):groups['val']['domain:'+domain]['frozen_simple_count']=count
        result=comparison.continuation_review(groups)
        self.assertTrue(result['R']['passed']);self.assertTrue(result['Q1']['passed'])
        self.assertTrue(result['eligible_to_design_separate_constrained_final_workpoint'])
        self.assertFalse(result['automatic_deployment_or_TEST'])
        groups['val']['domain:real']['ranks']['common']['rank']['a1']['error_auroc']=.7
        self.assertFalse(comparison.continuation_review(groups)['Q1']['passed'])
        groups['val']['sequence:real_fixture']['references']['a1']['all_output_reference_correct_coverage']=.1
        self.assertFalse(comparison.continuation_review(groups)['R']['passed'])

    def test_stage_requires_new_status_contract_exact_SHA_and_no_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'report.json';contract={'fixed':True}
            entry.base.write_new(p,dict(status='PASS',contract=contract))
            entry.base.write_new(root/'completion.json',dict(status='PASS',contract=contract,
                contract_sha256=entry.simple.fingerprint(contract),artifacts={'report.json':entry.base.sha(p)}))
            entry.stage(p,'PASS',contract)
            with self.assertRaises(ValueError):entry.stage(p,entry.STATUSES['smoke'],contract)
            with self.assertRaises(ValueError):entry.stage(p,'PASS',{'fixed':False})
            (root/'failure.json').write_text('{}')
            with self.assertRaises(ValueError):entry.stage(p,'PASS',contract)

    def test_gradient_guard_requires_each_term_in_both_parameter_groups(self):
        bounds={'median_new_to_original_min':.01,'median_new_to_original_max':10.,'median_clip_retention_min':.1}
        rows=[dict(domain=d,arm='a1',projection={'eligible':True},gradient=dict(new_to_original=.5,
            new_gradient_norm=1.,common_clip_scale=1.,groups={g:dict(new_norm=1.,raw_norms=dict(v1=1.,inner=1.,outer=1.))
                for g in ('stem','output')})) for d in ('real','sim')]
        self.assertTrue(entry.strength_summary(rows,bounds)['passed'])
        rows[0]['gradient']['groups']['stem']['raw_norms']['outer']=0.
        self.assertFalse(entry.strength_summary(rows,bounds)['passed'])
        rows[0]['gradient']['groups']['stem']['raw_norms']['outer']=1.
        rows[0]['gradient']['common_clip_scale']=.01
        self.assertFalse(entry.strength_summary(rows,bounds)['passed'])

    def test_new_source_contract_closed_v2_and_Python38_no_eager_Torch(self):
        protocol,sources=entry.checked_sources()
        self.assertEqual(protocol['coefficients'],core.COEFFICIENTS)
        self.assertTrue(protocol['proxy_is_not_online_IRLS'])
        self.assertNotIn('replay',entry.STATUSES)
        for name in sources['sources']:
            if name.endswith('.py'):
                ast.parse((entry.ROOT/name).read_text(),feature_version=(3,8))
        self.assertNotIn('torch',entry.__dict__)

    def test_fresh_pair_runs_all_1536_steps_same_order_and_no_replay_or_resume(self):
        class Scalar:
            dtype='float';device='fake'
            def __init__(self,v):self.value=float(v)
            def __float__(self):return self.value
            def __getitem__(self,k):return self
            def __mul__(self,v):return Scalar(self.value*v)
            def __add__(self,v):return Scalar(self.value+float(v))
            __radd__=__add__
            def detach(self):return self
            def backward(self):pass
            def double(self):return self
            def square(self):return Scalar(self.value**2)
            def sum(self):return self
        class Model:
            def __init__(self):self.updates=0;self.parameter=SimpleNamespace(grad=Scalar(1.))
            def train(self):pass
            def __call__(self,x):return Scalar(1.)
            def parameters(self):return [self.parameter]
        class Optimizer:
            def __init__(self,model):self.model=model
            def zero_grad(self):pass
            def step(self):self.model.updates+=1
        arms={a:Model() for a in core.ARMS};detector=object();head=object()
        iterator=iter(Optimizer(arms[a]) for a in core.ARMS)
        torch=SimpleNamespace(as_tensor=lambda *a,**k:Scalar(1.),optim=SimpleNamespace(Adam=lambda *a,**k:next(iterator)),
            nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *a:1.)),
            cuda=SimpleNamespace(max_memory_allocated=lambda *a:0,max_memory_reserved=lambda *a:0))
        backend=SimpleNamespace(loss_terms=lambda *a:dict(v1=Scalar(1.),inner=Scalar(2.),outer=Scalar(3.)),
            coefficients=lambda arm:dict(v1=1.,inner=.125 if arm=='a1' else 0.,outer=.125 if arm=='a1' else 0.),
            update=lambda terms,model,opt,arm,logits:(opt.step() or {'fixture':True}),reference_precision=nullcontext)
        inputs=[None]*10;inputs[6]={'fit':[dict(image='fit%04d'%i,domain='real',gt=box()) for i in range(384)]};inputs[7]={}
        previous=[None]*6;previous[1]=inputs;old=[None,previous]
        frozen={'b':'b_sha','midpoint':'h_sha'}
        def digest(m):return 'b_sha' if m is detector else 'h_sha' if m is head else 'ref_'+str(m.updates)
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(out_dir=Path(d),check_report=Path('check'),smoke_report=Path(d)/'smoke',gpu=0)
            args.smoke_report.write_text('{}')
            constants=dict(eligible=False,reason='fixture',cells=36,short_model_px=8.,background_cells=20)
            with patch.dict('sys.modules',{'crane_project.utils.port_size_halfpeak_range_v3_torch':backend}),\
                    patch.object(entry,'stage',return_value=dict(initial_sha256='init',frozen=frozen)),\
                    patch.object(entry,'models',return_value=(None,torch,detector,head,None,arms,'init')),\
                    patch.object(entry.reference,'view',return_value=([Scalar(1)],{},None)),\
                    patch.object(core,'projection',return_value=constants),\
                    patch.object(entry.reference.size,'target_map',return_value=(np.ones((1,1)),np.ones((1,1)))),\
                    patch.object(entry,'save_reload',return_value={'fixture':True}),\
                    patch.object(entry.base,'state_digest',side_effect=digest),patch('builtins.print'):
                entry.paired_train(args,(old,{}))
            self.assertEqual({a:m.updates for a,m in arms.items()},dict(a0=1536,a1=1536))
            events=[json.loads(x) for x in (Path(d)/'train_steps.jsonl').read_text().splitlines()]
            self.assertEqual(len(events),3072)
            self.assertTrue(all(events[i]['image']==events[i+1]['image'] for i in range(0,3072,2)))
            report=json.loads((Path(d)/'train_report.json').read_text())
            self.assertFalse(report['trained_VAL']);self.assertFalse(report['trained_holdout'])
            self.assertEqual(report['fixed_final_epoch'],4)


class ActualTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.threads=torch.get_num_threads();torch.set_num_threads(min(2,cls.threads))

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.threads)

    def test_actual_numpy_derivative_in_both_floor_branches_and_padding_support(self):
        from crane_project.utils.port_size_halfpeak_range_v3_torch import verify_auxiliary_autograd
        for scale in (1.,.5):
            result=verify_auxiliary_autograd(core.projection(box(),meta(scale)),'cpu')
            self.assertTrue(result['passed']);self.assertEqual(len(result['cases']),4)
            self.assertEqual({r['contrast_floor_active'] for r in result['cases']},{False,True})

    def test_real_Torch_scale_derivative_corrects_plus_and_minus_for_both_edges(self):
        import torch
        from crane_project.utils.port_size_halfpeak_range_v3_torch import loss_terms
        constants=core.toy_constants(box(),meta());xx,yy=constants['_toy_target']
        x=torch.tensor(xx,dtype=torch.float64);y=torch.tensor(yy,dtype=torch.float64)
        for axis in (0,1):
            for factor in (.85,1.15):
                f=torch.tensor(factor,dtype=torch.float64,requires_grad=True)
                u=x/f if axis==0 else x;v=y/f if axis==1 else y
                p=.01+.8*torch.exp(-.5*(u*u+v*v))*((u.abs()<=2)&(v.abs()<=2))
                logits=(p.log()-torch.log1p(-p))[None,None]
                terms=loss_terms(logits,torch.ones_like(logits)*.5,torch.ones_like(logits),constants)
                derivative=float(torch.autograd.grad(.125*(terms['inner']+terms['outer']),f)[0])
                self.assertGreater(derivative*(factor-1),0.)

    def test_center_and_background_remain_live_and_below_background_receives_gradient(self):
        import torch
        from crane_project.utils.port_size_halfpeak_range_v3_torch import loss_terms
        constants=core.projection(box(),meta());n=np.prod(constants['shape'])
        p=np.linspace(.1,.11,n).reshape(constants['shape']);p.ravel()[constants['indices'].ravel()]=.04
        p.ravel()[constants['indices'][-1]]=.8
        logits=torch.tensor(np.log(p)-np.log1p(-p),dtype=torch.float64)[None,None].requires_grad_()
        terms=loss_terms(logits,torch.ones_like(logits)*.5,torch.ones_like(logits),constants)
        gradient=torch.autograd.grad(terms['inner']+terms['outer'],logits)[0][0,0].numpy()
        self.assertGreater(np.abs(gradient[constants['background_mask']]).sum(),0.)
        self.assertGreater(np.abs(gradient.ravel()[constants['indices'][-1]]).sum(),0.)
        self.assertGreater(np.abs(gradient.ravel()[constants['indices'][:-1].ravel()]).sum(),0.)

    def test_same_original_pixel_gradients_for_A0_and_skipped_auxiliary(self):
        import torch
        from crane_project.utils import port_size_halfpeak_range_v3_torch as backend
        from crane_project.utils.port_size_reference_v1_torch import reference_loss
        torch.manual_seed(1701);z=torch.randn(1,1,128,128,requires_grad=True)
        target,valid=core.geometry.target_map(box(),meta());t=torch.tensor(target)[None,None];v=torch.tensor(valid)[None,None]
        expected=torch.autograd.grad(reference_loss(z,t,v),z,retain_graph=True)[0]
        terms=backend.loss_terms(z,t,v,core.projection(box(),meta()))
        weights=backend.coefficients('a0');actual=torch.autograd.grad(sum(terms[k]*weights[k] for k in weights),z,retain_graph=True)[0]
        self.assertTrue(torch.equal(expected,actual))
        skipped=backend.loss_terms(z,t,v,dict(eligible=False))
        self.assertEqual(float(skipped['inner'].detach()),0.);self.assertEqual(float(skipped['outer'].detach()),0.)
        actual=torch.autograd.grad(skipped['v1']+.125*(skipped['inner']+skipped['outer']),z)[0]
        self.assertTrue(torch.equal(expected,actual))

    def test_actual_reference_three_gradients_update_and_save_reload_exact(self):
        import torch
        from crane_project.utils import port_size_halfpeak_range_v3_torch as backend
        torch.manual_seed(1701);model=backend.SizeReference();features=torch.randn(1,256,32,32)
        target,valid=core.geometry.target_map(box(),meta());t=torch.tensor(target)[None,None];v=torch.tensor(valid)[None,None]
        z=model(features);terms=backend.loss_terms(z,t,v,core.projection(box(),meta()))
        optimizer=torch.optim.Adam(model.parameters(),lr=.001);before=entry.base.state_digest(model)
        measured=backend.update(terms,model,optimizer,'a1',z)
        self.assertTrue(measured['gradient_consistency']['passed'])
        for group in ('stem','output'):
            self.assertTrue(all(measured['groups'][group]['raw_norms'][k]>0 for k in ('v1','inner','outer')))
        self.assertNotEqual(before,entry.base.state_digest(model))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);report=entry.save_reload(root/'a1',model,optimizer,{'fixture':True},'a1',0,1,
                'smoke_discarded',before,{'b':'b_sha'},[features],torch)
            self.assertTrue(report['save_reload_exact'])
            payload=entry.load(root/report['path'],{'fixture':True},'smoke_discarded','a1',torch)
            self.assertEqual(payload['protocol'],core.VERSION)
            with self.assertRaises(ValueError):entry.load(root/report['path'],{'different':True},'smoke_discarded','a1',torch)

    def test_incorrect_shared_seed_or_VJP_stops_optimizer(self):
        import torch
        from crane_project.utils import port_size_halfpeak_range_v3_torch as backend
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.stem=torch.nn.Linear(4,3);self.output=torch.nn.Linear(3,1)
            def forward(self,x):return self.output(self.stem(x))
        for mode in ('seed','VJP'):
            torch.manual_seed(1701);model=Model();z=model(torch.randn(8,4))
            terms=dict(v1=z.square().mean(),inner=z.mean(),outer=z.square().mean())
            opt=torch.optim.Adam(model.parameters());original=torch.autograd.grad
            def corrupt(*args,**kwargs):
                out=original(*args,**kwargs)
                if mode=='seed' and args[0] is terms['inner'] and args[1] is z:return tuple(-g for g in out)
                if mode=='VJP' and args[0] is z and 'grad_outputs' in kwargs:return tuple(g+.1 for g in out)
                return out
            with patch.object(torch.autograd,'grad',side_effect=corrupt),patch.object(opt,'step') as step:
                with self.assertRaisesRegex(ValueError,'Gradient verification failed'):
                    backend.update(terms,model,opt,'a1',z)
                step.assert_not_called()


if __name__=='__main__':unittest.main()
