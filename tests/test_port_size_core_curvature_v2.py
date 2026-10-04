"""New core supervision, actual Torch gradient decomposition, paired-stage safety."""
import ast
from contextlib import nullcontext
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from crane_project.utils import port_size_core_curvature_v2 as core
from crane_project.utils import port_size_core_comparison_v2 as comparison
from crane_project.tools import run_port_size_core_curvature_v2 as entry


def meta(scale=1., flip=False):
    n=int(256*scale)
    return dict(img_shape=[n,n,3],ori_shape=[256,256,3],pad_shape=[n,n,3],
                scale_factor=[scale]*4,flip=flip,flip_direction='horizontal' if flip else None)


def box(): return [128.,128.,80.,40.,.37]


def evidence(defined=True,risk=.2,bad=False):
    ref=dict(defined=defined,reason='defined' if defined else 'weak_evidence',long_original_px=80.,short_original_px=40.)
    frozen=dict(final_box_original=None,center_accepted=True,size_accepted=True,angle_accepted=True,risks={'size':.3,'angle':.2})
    return dict(reference=ref,reading=dict(defined=defined,risk=risk if defined else None),
                frozen=frozen,frozen_score_only=deepcopy(frozen))


def rows_fixture():
    result=[]
    for role in ('reference_holdout_train','val'):
        for i in range(4):
            pred=[128.,128.,80. if i!=1 else 50.,40.,.37,.8] if i!=3 else None
            source=dict(image=role+str(i),split='val' if role=='val' else 'train',domain='real',
                sequence='r',frame_id=i,image_size=[256,256],gt=box(),pred=pred)
            arms={a:evidence(i<2,.1+i*.3) for a in core.ARMS}
            for item in arms.values():
                item['frozen']['size_accepted']=i<2
                item['frozen']['final_box_original']=pred
                item['frozen_score_only']['final_box_original']=pred
            result.append(comparison.annotate(source,arms,role))
    return result


class CoreGeometryTests(unittest.TestCase):
    def test_ideal_and_both_scale_directions_have_correct_curvature_and_derivative(self):
        result=core.numerical_probe(core.projection(box(),meta()))
        self.assertTrue(result['passed'],result)
        self.assertLess(result['ideal_curvature_error'],1e-10)
        self.assertEqual(len(result['cases']),4)

    def test_new_loss_is_periodic_swap_and_isotropic_transform_invariant(self):
        b=box()
        variants=[b,[*b[:2],b[3],b[2],b[4]+math.pi/2],[*b[:4],b[4]+math.pi]]
        for transform in (meta(),meta(.5),meta(.5,True)):
            for value in variants:
                projection=core.projection(value,transform)
                self.assertTrue(projection['eligible'])
                self.assertTrue(core.numerical_probe(projection)['passed'])

    def test_padding_excluded_and_unresolved_GT_keeps_original_pixel_loss_role(self):
        transform=meta();transform['img_shape']=[200,200,3];transform['ori_shape']=[200,200,3]
        projection=core.projection(box(),transform)
        _,valid=core.geometry.grid(transform)
        self.assertFalse(np.any(projection['mask'] & ~valid))
        value=box();value[3]=8.
        skipped=core.projection(value,meta())
        self.assertFalse(skipped['eligible']);self.assertEqual(skipped['reason'],'unresolved_GT_short')
        with self.assertRaises(ValueError):core.projection(box(),dict(meta(),scale_factor=[.5,.8,.5,.8]))

    def test_residual_gradient_has_independent_directional_finite_difference(self):
        constants=core.projection(box(),meta());x,y=constants['coordinates'].T
        z=-.5*(x*x+y*y)+.15*np.sin(4*x+3*y)
        result=core.numpy_terms(z,constants,True)
        direction=np.cos(np.arange(len(z))*.43);step=1e-6
        def objective(a):
            v=core.numpy_terms(a,constants);return .25*v['curv']+.05*v['quad']
        finite=(objective(z+step*direction)-objective(z-step*direction))/(2*step)
        self.assertAlmostEqual(finite,float(result['gradient_y'].dot(direction)),places=8)
        self.assertGreater(result['quad'],0.)

    def test_95pct_coverage_has_unavoidable_bad_and_finite_gain_bound(self):
        value=core.acceptance_bound(887,886,642,843)
        self.assertEqual(value['minimum_bad_accepted'],201)
        self.assertEqual(212-value['minimum_bad_accepted'],11)

    def test_online_API_is_GT_domain_sequence_free_and_preserves_center(self):
        import inspect
        self.assertFalse(set(inspect.signature(core.online_reading).parameters) & {'gt','domain','sequence','history'})
        class Runtime:
            def decide(self,p,size,method='simple'):
                return dict(final_box_original=p,center_accepted=p is not None,
                    size_accepted=True,angle_accepted=True,risks={'size':.2,'angle':.3})
        pred=box()+[.8];before=deepcopy(pred)
        with patch.object(entry.migration.new,'template_reference',return_value=evidence()['reference']):
            value=core.online_reading(np.zeros((128,128)),pred,[256,256],meta(),Runtime(),{'risk_scale':.1})
        self.assertEqual(pred,before);self.assertTrue(value['frozen']['center_accepted'])
        missing=core.online_reading(None,None,[256,256],meta(),Runtime(),{'risk_scale':.1})
        self.assertFalse(missing['reference']['defined']);self.assertFalse(missing['frozen']['center_accepted'])

    def test_common_vs_full_output_unavailable_cost_and_missing_denominator(self):
        rows=rows_fixture();result=comparison.summarize(rows)
        report=result['summary']['val']['all']
        self.assertEqual((report['frames'],report['outputs'],report['missing_outputs']),(4,3,1))
        self.assertEqual(report['references']['a1']['unavailable_good'],1)
        self.assertEqual(report['output_coverage'],.75)
        self.assertEqual(report['center_hit_rate_on_outputs'],1.)
        self.assertEqual(report['full_frame_center_correct_coverage'],.75)
        point=report['primary_same_count']['a1']
        self.assertEqual((point['FA'],point['FR'],point['CR'],point['ED']),(1,1,1,0))
        self.assertFalse(result['comparison_workpoints']['a1']['reachable'])
        self.assertFalse(result['final_deployment_policy_created'])

    def test_ties_are_label_free_with_bounds_and_exact_counts(self):
        rows=rows_fixture()[:3]
        for row in rows:
            for arm in core.ARMS:
                row['arms'][arm]['reading']['risk']=.5
                row['arms'][arm]['reference']['defined']=True
        point=comparison.matched(rows,1,4)['a1']
        self.assertEqual(point['accepted'],1)
        self.assertEqual(point['tie_bounds']['bad_min_over_tie'],0)
        self.assertEqual(point['tie_bounds']['bad_max_over_tie'],1)
        self.assertEqual(point['FR'],1)

    def test_comparison_quantile_is_label_free_and_not_a_final_workpoint(self):
        rows=rows_fixture();before=comparison.diagnostic_cutoffs(rows)
        changed=deepcopy(rows)
        for row in changed:
            if row['size_bad'] is not None:row['size_bad']=not row['size_bad']
        self.assertEqual(before,comparison.diagnostic_cutoffs(changed))
        self.assertEqual(comparison.state([rows[0],rows[1]],[True,True],4)['minimum_FA_at_same_count'],1)

    def test_no_TEST_or_single_missing_role_duplicate_identity(self):
        rows=rows_fixture()
        with self.assertRaises(ValueError):comparison.summarize(rows[:4])
        with self.assertRaises(ValueError):comparison.summarize(rows+[rows[0]])
        source=dict(rows[0],split='test')
        with self.assertRaises(ValueError):comparison.annotate(source,rows[0]['arms'],'val')

    def test_skipped_or_zero_strength_and_extreme_clipping_cannot_release_training(self):
        bounds={'median_new_to_original_min':.01,'median_new_to_original_max':10.,'median_clip_retention_min':.1}
        rows=[]
        for d in ('real','sim'):
            rows.append(dict(domain=d,arm='a1',projection={'eligible':True},gradient=dict(
                new_to_original=.5,new_gradient_norm=1.,common_clip_scale=1.,groups={g:{'new_norm':1.} for g in ('stem','output')})))
        self.assertTrue(entry.strength_summary(rows,bounds)['passed'])
        rows[0]['gradient']['new_to_original']=0.
        self.assertFalse(entry.strength_summary(rows,bounds)['passed'])
        rows[0]['gradient']['new_to_original']=.5;rows[0]['gradient']['common_clip_scale']=.01
        self.assertFalse(entry.strength_summary(rows,bounds)['passed'])

    def test_decomposition_bounds_uncancelled_operands_per_tensor_not_result(self):
        row=dict(name='stem.weight',dtype_epsilon=float(np.finfo(np.float32).eps),
            uncancelled_norm=20.,uncancelled_max=4.,error_norm=2e-6,error_max=1e-6)
        result=core.gradient_consistency([row])
        self.assertTrue(result['passed'])
        # A genuine missing/sign/wrong-scale gradient remains far outside the
        # roundoff bound, even when a different tensor has a huge norm.
        bad=dict(row,name='output.bias',error_norm=.01,error_max=.01)
        large=dict(row,name='stem.large',uncancelled_norm=1e9,uncancelled_max=1e8)
        self.assertFalse(core.gradient_consistency([large,bad])['passed'])
        zero=dict(row,uncancelled_norm=0.,uncancelled_max=0.,error_norm=0.,error_max=0.)
        self.assertTrue(core.gradient_consistency([zero])['passed'])
        self.assertFalse(core.gradient_consistency([dict(zero,error_norm=1e-6,error_max=1e-6)])['passed'])
        with self.assertRaises(ValueError):core.gradient_consistency([dict(row,error_norm=float('nan'))])

    def test_stage_requires_success_contract_all_SHA_and_no_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'report.json';contract={'fixed':True}
            entry.base.write_new(p,dict(status='PASS',contract=contract))
            entry.base.write_new(root/'completion.json',dict(status='PASS',contract=contract,
                contract_sha256=entry.simple.fingerprint(contract),artifacts={'report.json':entry.base.sha(p)}))
            self.assertEqual(entry.stage(p,'PASS',contract)['status'],'PASS')
            with self.assertRaises(ValueError):entry.stage(p,'PASS',{'fixed':False})
            (root/'failure.json').write_text('{}')
            with self.assertRaises(ValueError):entry.stage(p,'PASS',contract)

    def test_paired_numeric_records_JSON_replay_and_theoretical_bounds_are_identical(self):
        rows=rows_fixture()
        # Add a second domain/video so pooled and per-domain ranking allocation differ.
        second=deepcopy(rows)
        for row in second:
            row['image']='sim_'+row['image'];row['domain']='sim';row['sequence']='s'
            for arm in core.ARMS:
                if row['arms'][arm]['reading']['risk'] is not None:
                    row['arms'][arm]['reading']['risk']+=.05
        rows+=second
        report=comparison.summarize(rows)
        restored=json.loads(json.dumps(rows,allow_nan=False))
        self.assertEqual(report,comparison.summarize(restored))
        self.assertEqual(report['summary']['val']['all']['outputs'],6)
        self.assertEqual(report['summary']['val']['domain:sim']['outputs'],3)
        self.assertEqual(report['summary']['val']['all']['common_reference']['a1']['all_output_denominator'],6)
        self.assertEqual(report['summary']['val']['all']['common_reference']['a1']['all_output_reference_correct_coverage'],4/6)

    def test_new_source_contract_and_no_eager_Torch_and_Python38(self):
        protocol,sources=entry.checked_sources()
        self.assertTrue(protocol['comparison_only']);self.assertFalse(protocol['test_read'])
        for name in sources['sources']:
            if name.endswith('.py'):
                tree=ast.parse((entry.ROOT/name).read_text(),feature_version=(3,8))
                if 'torch' not in name:
                    imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
                    self.assertFalse(any(isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names) for n in imports))

    def test_all_1536_slots_update_both_arms_including_non_diagnostic_steps(self):
        # Fake backend checks scheduling/optimizer contract, not learning quality.
        class Scalar:
            dtype='float';device='fake'
            def __init__(self,value):self.value=float(value)
            def __float__(self):return self.value
            def __getitem__(self,key):return self
            def __mul__(self,other):return Scalar(self.value*other)
            def __add__(self,other):return Scalar(self.value+float(other))
            __radd__=__add__
            def detach(self):return self
            def backward(self):pass
            def double(self):return self
            def square(self):return Scalar(self.value**2)
            def sum(self):return self
        class Model:
            def __init__(self):self.updates=0;self.parameter=SimpleNamespace(grad=Scalar(1))
            def train(self):pass
            def __call__(self,feature):return Scalar(1)
            def parameters(self):return [self.parameter]
        class Optimizer:
            def __init__(self,model):self.model=model
            def zero_grad(self):pass
            def step(self):self.model.updates+=1
        arms={a:Model() for a in core.ARMS};detector=object();head=object()
        optimizer_iter=iter([Optimizer(arms[a]) for a in core.ARMS])
        torch=SimpleNamespace(as_tensor=lambda *a,**k:Scalar(1),
            optim=SimpleNamespace(Adam=lambda *a,**k:next(optimizer_iter)),
            nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *a:1.)),
            cuda=SimpleNamespace(max_memory_allocated=lambda *a:0,max_memory_reserved=lambda *a:0))
        def update(terms,model,optimizer,arm):optimizer.step();return {'fixture':True}
        backend=SimpleNamespace(loss_terms=lambda *a:{'v1':Scalar(1),'curv':Scalar(2),'quad':Scalar(3)},
            coefficients=lambda arm:{'v1':1,'curv':.25 if arm=='a1' else 0,'quad':.05 if arm=='a1' else 0},
            update=update,reference_precision=nullcontext)
        split={'fit':[dict(image='fit%04d'%i,domain='real',gt=box()) for i in range(384)]}
        original=[None]*10;original[6]=split;original[7]={}
        previous=[None]*6;previous[1]=original
        old=[None,previous];frozen={'b':'b_sha','midpoint':'h_sha'}
        checked={'initial_sha256':'init','frozen':frozen}
        def digest(module):
            if module is detector:return 'b_sha'
            if module is head:return 'h_sha'
            return 'ref_'+str(module.updates)
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(check_report=Path('check'),smoke_report=Path(directory)/'smoke',
                                 out_dir=Path(directory),gpu=0)
            args.smoke_report.write_text('{}')
            def saved(*a):return dict(path='epoch04',sha256='sha',marker_sha256='marker')
            with patch.dict('sys.modules',{'crane_project.utils.port_size_core_curvature_v2_torch':backend}), \
                    patch.object(entry,'stage',return_value=checked), \
                    patch.object(entry,'models',return_value=(None,torch,detector,head,None,arms,'init')), \
                    patch.object(entry.reference,'view',return_value=([Scalar(1)],{},None)), \
                    patch.object(core,'projection',return_value={'eligible':False,'reason':'fixture','cells':0,'short_model_px':0,'condition':None}), \
                    patch.object(entry.reference.size,'target_map',return_value=(np.zeros((1,1)),np.ones((1,1)))), \
                    patch.object(entry,'save_reload',side_effect=saved), \
                    patch.object(entry.base,'state_digest',side_effect=digest),patch('builtins.print'):
                entry.paired_train(args,(old,{}))
                args.out_dir=Path(directory)/'failed';args.out_dir.mkdir()
                optimizer_iter=iter([Optimizer(arms[a]) for a in core.ARMS])
                def failed_update(*unused):
                    error=ValueError('decomposition fixture');error.details={'fixture':True};raise error
                backend.update=failed_update
                with self.assertRaisesRegex(ValueError,'epoch=1 slot=1 image=') as failure:
                    entry.paired_train(args,(old,{}))
                self.assertTrue(failure.exception.details['fixture'])
                self.assertEqual(failure.exception.details['train_context']['arm'],'a0')
                self.assertEqual(failure.exception.details['train_context']['slot'],1)
            self.assertEqual({a:m.updates for a,m in arms.items()},{'a0':1536,'a1':1536})
            events=[json.loads(line) for line in (Path(directory)/'train_steps.jsonl').read_text().splitlines()]
            self.assertEqual(len(events),3072)
            for step in range(1536):
                self.assertEqual(events[2*step]['image'],events[2*step+1]['image'])
                self.assertEqual(events[2*step]['step'],step+1)


@unittest.skipUnless(importlib.util.find_spec('torch'),'Actual Torch autograd/save-reload must run before server training')
class ActualTorchTests(unittest.TestCase):
    def test_cancellation_roundoff_passes_but_wrong_decomposition_stops_update(self):
        import torch
        from crane_project.utils import port_size_core_curvature_v2_torch as backend
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.stem=torch.nn.Linear(16,8);self.output=torch.nn.Linear(8,1)
            def forward(self,x):return self.output(self.stem(x))
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            torch.manual_seed(1701);model=Model().to(device)
            z=model(torch.randn(64,16,device=device));direction=torch.randn_like(z)
            terms=dict(v1=(z*direction).sum(),
                curv=(z*(-4*direction+1e-5*torch.randn_like(z))).sum(),quad=z.sum()*0)
            _,expected=backend.gradient_measurement(terms,model,'a1')
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            measured=backend.update(terms,model,optimizer,'a1')
            maximum=max(float(g.abs().max()) for g in expected)
            if device == 'cpu':
                self.assertGreater(measured['component_sum_backward_max_error'],
                                   1e-5*max(maximum,1e-7)+1e-9)
            self.assertTrue(measured['gradient_consistency']['passed'])
            self.assertAlmostEqual(measured['actual_clip_retention'],
                measured['actual_postclip_norm']/measured['actual_preclip_norm'])
            # Inject a real decomposition defect in a non-cancelled graph;
            # stopping the optimizer is essential, not just reporting failure.
            z=model(torch.randn(64,16,device=device))
            terms=dict(v1=z.square().mean(),curv=z.sum()*0,quad=z.sum()*0)
            measurement,expected=backend.gradient_measurement(terms,model,'a1')
            corrupted=list(expected);corrupted[0]=expected[0]+.01*measurement['uncancelled_tensor_scales'][0]['uncancelled_max']
            with patch.object(backend,'gradient_measurement',return_value=(measurement,corrupted)), \
                    patch.object(optimizer,'step') as step:
                with self.assertRaisesRegex(ValueError,'arm=a1 parameter=') as failure:
                    backend.update(terms,model,optimizer,'a1')
                self.assertFalse(failure.exception.details['gradient_consistency']['passed'])
                step.assert_not_called()

    def test_reference_precision_restores_flags_even_on_failure(self):
        import torch
        from crane_project.utils.port_size_core_curvature_v2_torch import reference_precision
        original=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
            with self.assertRaisesRegex(ValueError,'fixture'):
                with reference_precision():
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertFalse(torch.backends.cudnn.allow_tf32)
                    with reference_precision():
                        self.assertFalse(torch.backends.cudnn.allow_tf32)
                    raise ValueError('fixture')
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=original

    def test_reference_forward_precision_preserves_architecture_and_initial_state(self):
        import torch
        from crane_project.utils import port_size_core_curvature_v2_torch as backend
        original=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
            for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
                torch.manual_seed(1701);baseline=backend.OriginalSizeReference()
                torch.manual_seed(1701);model=backend.SizeReference()
                self.assertEqual(entry.base.state_digest(model),entry.base.state_digest(baseline))
                self.assertEqual([(n,tuple(p.shape)) for n,p in model.named_parameters()],
                                 [(n,tuple(p.shape)) for n,p in baseline.named_parameters()])
                model=model.to(device);seen=[]
                def hook(module,inputs,output):
                    seen.append(module)
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertFalse(torch.backends.cudnn.allow_tf32)
                handles=[m.register_forward_hook(hook) for m in model.modules() if isinstance(m,torch.nn.Conv2d)]
                logits=model(torch.randn(1,256,8,8,device=device))
                for handle in handles:handle.remove()
                self.assertEqual(len(seen),3)
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                self.assertTrue(torch.backends.cudnn.allow_tf32)
                # Actual convolution graph update, also exercises CUDA when
                # present. Forward hooks catch saved TF32-flag regressions.
                measured=backend.update(dict(v1=logits.square().mean(),
                    curv=logits.sum()*0,quad=logits.sum()*0),model,
                    torch.optim.Adam(model.parameters(),lr=.001),'a0')
                self.assertTrue(measured['gradient_consistency']['passed'])
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=original

    def test_Torch_auxiliary_gradients_match_independent_numpy_and_padding_zero(self):
        import torch
        from crane_project.utils.port_size_core_curvature_v2_torch import loss_terms
        constants=core.projection(box(),meta());x,y=constants['coordinates'].T
        response=math.log(.8)-.5*((x/1.15)**2+y*y)+.05*np.sin(3*x)
        probabilities=np.exp(response)
        logits=np.full(constants['mask'].shape,-4.)
        logits[constants['mask']]=np.log(probabilities)-np.log1p(-probabilities)
        z=torch.tensor(logits,dtype=torch.float64)[None,None].requires_grad_()
        t,v=core.geometry.target_map(box(),meta())
        terms=loss_terms(z,torch.tensor(t)[None,None],torch.tensor(v,dtype=torch.float64)[None,None],constants)
        actual=torch.autograd.grad(.25*terms['curv']+.05*terms['quad'],z)[0][0,0].numpy()
        expected=core.numpy_terms(response,constants,True)['gradient_y']*(1-probabilities)
        np.testing.assert_allclose(actual[constants['mask']],expected,rtol=1e-7,atol=1e-10)
        self.assertEqual(float(np.abs(actual[~constants['mask']]).sum()),0.)

    def test_real_module_gradient_decomposition_clipping_and_discarded_save_reload(self):
        import torch
        from crane_project.utils.port_size_core_curvature_v2_torch import SizeReference,loss_terms,update
        torch.manual_seed(1701);model=SizeReference()
        transform=dict(img_shape=[64,64,3],ori_shape=[64,64,3],pad_shape=[64,64,3],scale_factor=[1.]*4,flip=False)
        gt=[32.,32.,32.,16.,.3];constants=core.projection(gt,transform)
        features=torch.randn(1,256,8,8)
        target,valid=core.geometry.target_map(gt,transform)
        terms=loss_terms(model(features),torch.tensor(target)[None,None],torch.tensor(valid,dtype=torch.float32)[None,None],constants)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        measured=update(terms,model,optimizer,'a1')
        self.assertGreater(measured['new_gradient_norm'],0.)
        self.assertGreater(measured['groups']['stem']['new_norm'],0.)
        self.assertGreater(measured['groups']['output']['new_norm'],0.)
        self.assertLessEqual(measured['actual_postclip_norm'],10.00001)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'head.pth';contract={'test':True}
            entry.reference.save_checkpoint(path,dict(protocol=core.VERSION,contract=contract,
                role='smoke_discarded',arm='a1',epoch=0,state=model.state_dict()))
            payload=entry.load(path,contract,'smoke_discarded','a1',torch)
            fresh=SizeReference();fresh.load_state_dict(payload['state'])
            self.assertTrue(torch.equal(model(features),fresh(features)))
            with self.assertRaises(ValueError):entry.load(path,contract,'paired_reference_train','a1',torch)
            with self.assertRaises(FileExistsError):entry.reference.save_checkpoint(path,payload)


if __name__=='__main__':unittest.main()
