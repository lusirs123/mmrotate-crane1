"""Boundary candidate contracts; synthetic geometry, no data/model/GPU access.

Native torch tests are required on the server and never read real ROI files or
checkpoints. A CPU test pass is not a CUDA smoke or a measured size improvement.
"""
import ast
from contextlib import redirect_stderr
from copy import deepcopy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from crane_project.tools import run_port_geometry_size_boundary_continuous_v1 as tool


class ProtocolTests(unittest.TestCase):
    def test_source_closure_preserves_frozen_parent_and_new_scope(self):
        identity=tool.checked_sources()
        self.assertEqual(identity['parent_identity'],tool.parent.checked_sources())
        self.assertEqual(identity['parent_identity']['parent_identity']['parent_identity'],tool.sealed.checked_sources())
        document=tool.protocol_document()
        self.assertEqual(document['arms'],['boundary_continuous'])
        self.assertEqual(document['parameter_count'],2760)
        self.assertEqual(document['settings'],tool.design.SETTINGS)
        self.assertEqual(document['fixed_checkpoint'],tool.sealed.FIXED)
        self.assertFalse(document['scope']['test_access'])
        self.assertFalse(document['scope']['depth_formula_modified'])
        self.assertFalse(document['scope']['policy_modified'])
        self.assertFalse(document['scope']['automatic_promotion'])

    def test_no_test_tuning_resume_or_budget_overrides(self):
        for flags in (['--stage','test'],['--stage','finite','--steps','300'],
                      ['--stage','finite','--resume','old.pth'],['--stage','finite','--ratio-weight','1'],
                      ['--stage','finite','--seed','1']):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): tool.parser().parse_args(flags)

    def test_online_interface_has_no_gt_domain_sequence_depth(self):
        tree=ast.parse((tool.ROOT/'crane_project/utils/port_geometry_size_boundary_continuous_v1.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        self.assertEqual([a.arg for a in forward.args.args],
            ['self','roi','support','boxes_original','boxes_model','scale_xy','midpoint_original'])

    def test_shared_output_parent_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            valid=root/'work_dirs'/'port_geometry_size_boundary_v1'/(tool.VERSION+'_synthetic')/'check'
            self.assertEqual(tool.checked_output_directory(valid,root),valid.resolve())
            valid.mkdir(parents=True)
            with self.assertRaises(FileExistsError): tool.checked_output_directory(valid,root)
            for bad in (root/'docs'/'check',root/'work_dirs'/'old'/'finite',valid/'nested',
                        root/'work_dirs'/'port_geometry_size_boundary_v1'/'wrong'/'check'):
                with self.assertRaises(ValueError): tool.checked_output_directory(bad,root)

    def test_static_check_does_not_load_collection_or_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'check'
            args=tool.parser().parse_args(['--stage','check','--out-dir',str(out)])
            with patch.object(tool,'checked_output_directory',return_value=out), \
                    patch.object(tool,'current_plan',side_effect=AssertionError('must not read data')):
                tool.run(args)
            report=tool.sealed.read(out/'completion.json')
            self.assertEqual(report['status'],'SIZE_CONTINUOUS_STATIC_CONTRACT_PASS')
            self.assertFalse(report['model_loaded'])
            self.assertFalse(report['val_tensor_access'])
            self.assertEqual(report['parameter_updates'],0)
            self.assertEqual(tool.sealed.read(out/'artifacts.json')['files']['completion.json'],
                             tool.sealed.sha(out/'completion.json'))

    def test_no_failed_or_unrelated_smoke_unlocks_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            tool.sealed.write(p/'completion.json',dict(protocol='old',status='SIZE_CONTINUOUS_SMOKE_COMPLETE'))
            with self.assertRaises(ValueError): tool.checked_smoke(p,{}, {})

    def test_missing_smoke_engineering_proofs_cannot_unlock_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);arm=root/'boundary_continuous';arm.mkdir()
            report=dict(protocol=tool.VERSION,stage='smoke',arm='boundary_continuous',proof={},updates=2,engineering={})
            tool.sealed.write(arm/'completion.json',report)
            (arm/'experimental_head.pth').write_bytes(b'synthetic, never loaded')
            tool.sealed.write(arm/'artifacts.json',dict(protocol=tool.VERSION,files={
                p.name:tool.sealed.sha(p) for p in arm.iterdir() if p.is_file()}))
            tool.sealed.write(root/'completion.json',dict(protocol=tool.VERSION,status='SIZE_CONTINUOUS_SMOKE_COMPLETE',
                source_identity={},proof={},val_tensor_access=False,arms=['boundary_continuous']))
            tool.sealed.write(root/'artifacts.json',dict(protocol=tool.VERSION,files={
                'completion.json':tool.sealed.sha(root/'completion.json')}))
            with self.assertRaisesRegex(ValueError,'engineering'): tool.checked_smoke(root,{}, {})

    def test_shell_stops_on_failure_and_archives_shared_run_at_workdir_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'crane_project/tools').mkdir(parents=True);(root/'bin').mkdir()
            script=root/'crane_project/tools/run_port_geometry_size_boundary_continuous_v1.sh'
            script.write_text((tool.ROOT/'crane_project/tools/run_port_geometry_size_boundary_continuous_v1.sh').read_text())
            fake=root/'bin/python'
            fake.write_text('#!/bin/sh\necho synthetic-failure\nexit 19\n');fake.chmod(0o755)
            checksum=root/'bin/sha256sum'
            checksum.write_text('#!/bin/sh\necho synthetic-checksum\n');checksum.chmod(0o755)
            result=subprocess.run(['bash',str(script)],env=dict(os.environ,PATH=str(root/'bin')+':'+os.environ['PATH']),
                                  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            self.assertEqual(result.returncode,19,result.stdout)
            packages=list((root/'work_dirs').glob('*.tar.gz'));self.assertEqual(len(packages),1)
            runs=list((root/'work_dirs'/'port_geometry_size_boundary_v1').iterdir());self.assertEqual(len(runs),1)
            self.assertTrue((runs[0]/'run.log').is_file())
            with tarfile.open(packages[0]) as archive:
                names=archive.getnames()
                self.assertFalse(any('/prepare/' in n or '/finite/' in n for n in names))
                member=next(n for n in names if n.endswith('/run_exit_code.txt'))
                self.assertEqual(archive.extractfile(member).read(),b'19\n')

    def test_relocated_head_and_collect_use_fixed_hash_not_newest_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            moved=root/'work_dirs'/'port_results'/'geometry'/'fixed_head'
            moved.mkdir(parents=True)
            weight=moved/tool.sealed.FIXED['path'];weight.write_bytes(b'synthetic fixed identity')
            digest=tool.sealed.sha(weight)
            with patch.dict(tool.sealed.FIXED,{'sha256':digest}):
                self.assertEqual(tool.resolve_head(root=root),moved.resolve())
                other=root/'work_dirs'/'duplicate';other.mkdir()
                (other/tool.sealed.FIXED['path']).write_bytes(weight.read_bytes())
                with self.assertRaises(ValueError): tool.resolve_head(root=root)
                self.assertEqual(tool.resolve_head(moved,root),moved.resolve())
            collect=root/'work_dirs'/'relocated'/'collect';collect.mkdir(parents=True)
            (collect/'completion.json').write_text('{}')
            with patch.object(tool.sealed,'sha',return_value=tool.support.INPUT_FILES['collect/completion.json']):
                self.assertEqual(tool.resolve_collection(root=root),collect.parent.resolve())

    def test_new_gates_reject_bias_or_ratio_tradeoff(self):
        def group(mae=.049,bias=-.03,ratio=.02):
            return dict(midpoint=dict(edges={'short':dict(mae=.05,signed_log_mean=-.04)},
                            continuous_sizes=dict(log_ratio_mae=.03,log_ratio_p95=.05)),
                        edge_residual=dict(edges={'short':dict(mae=mae,signed_log_mean=bias)},
                            continuous_sizes=dict(log_ratio_mae=ratio,log_ratio_p95=.04)))
        summary={'val_s1':{'all':{k:group() for k in ('real','sim','real_seq07','real_seq14','sim_seq10')}}}
        parent_gate=dict(checks={'old_protection':True})
        with patch.object(tool.old.metrics,'finite_gates',return_value=parent_gate), \
                patch.object(tool.parent,'extra_probe_gates',side_effect=lambda s,g:g):
            self.assertTrue(tool.boundary_gates(summary,{})['finite_joint_gate_pass'])
            summary['val_s1']['all']['real_seq07']=group(bias=-.08)
            summary['val_s1']['all']['sim']=group(ratio=.04)
            failed=tool.boundary_gates(summary,{})['failed_checks']
            self.assertIn('val/real_seq07/abs_short_log_bias',failed)
            self.assertIn('val/sim/log_ratio_mae',failed)

    def test_failed_train_gate_cannot_load_val_shard(self):
        with patch.object(tool,'load_records',side_effect=AssertionError('VAL loader must not run')) as loader:
            with self.assertRaisesRegex(ValueError,'TRAIN'):
                tool.load_val_after_gate({'train_gate_pass':False},None,None,None,None,None)
            loader.assert_not_called()
        with patch.object(tool,'load_records',return_value=([],['synthetic-val'],123)) as loader:
            self.assertEqual(tool.load_val_after_gate({'train_gate_pass':True},1,2,3,4,5),(['synthetic-val'],123))
            loader.assert_called_once_with(1,2,3,4,5,True,include_train=False)

    def test_train_gate_requires_actual_delivered_gain_and_protects_both_scales(self):
        def method(mae=.04,short_bias=-.02,ratio=.02,joint=10):
            return dict(frames=10,output_coverage={'numerator':10,'denominator':10,'fraction':1.},
                joint_size10_full_frame={'numerator':joint},
                edges={e:dict(mae=mae,p95=.06,signed_log_mean=short_bias) for e in ('long','short')},
                continuous_sizes=dict(log_ratio_mae=ratio,log_ratio_p95=.04))
        group=dict(midpoint=method(),edge_residual=method(mae=.039),
                   paired={'frozen_identity_preserved':True})
        summary={shard:{role:{domain:deepcopy(group) for domain in ('real','sim')}
            for role in ('fit','probe')} for shard in ('train_s1','train_s05')}
        self.assertTrue(tool.train_gates(summary)['train_gate_pass'])
        summary['train_s05']['probe']['real']['edge_residual']['edges']['short']['p95']=.09
        gate=tool.train_gates(summary)
        self.assertFalse(gate['train_gate_pass'])
        self.assertIn('train_s05/probe/real/short_p95',gate['failed_checks'])
        summary['train_s1']['fit']['sim']['edge_residual']=method()
        self.assertIn('train_s1/fit/sim/pair_mae_strict_gain',tool.train_gates(summary)['failed_checks'])

    def test_finite_initial_load_is_train_only_and_val_requires_gate(self):
        source=(tool.ROOT/'crane_project/tools/run_port_geometry_size_boundary_continuous_v1.py').read_text()
        tree=ast.parse(source)
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        loads=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
               and n.func.id=='load_records']
        self.assertEqual(len(loads),1)
        self.assertIs(loads[0].args[-1].value,False)
        branches=[n for n in ast.walk(run) if isinstance(n,ast.If) and
            isinstance(n.test,ast.Subscript) and isinstance(n.test.value,ast.Name) and n.test.value.id=='train_gate']
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_val_after_gate'
                            for b in branches for n in ast.walk(b)))


@unittest.skipUnless(importlib.util.find_spec('torch'),'native torch unavailable')
class NativeTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from crane_project.utils import port_geometry_size_boundary_continuous_v1 as model
        cls.t,cls.model=torch,model
        torch.set_num_threads(1)

    def inputs(self,n=8):
        t=self.t
        b=t.tensor([[100.,100.,100.,50.,0.,.9]]).repeat(n,1)
        gt=b[:,:5].clone();gt[:,2:4]*=1.04
        return [t.randn(n,256,9,9),t.ones(n,1,9,9),b,b[:,:5].clone(),t.ones(n,2),b.clone(),gt]

    def test_neutral_and_empty_preserve_exact_identity(self):
        t=self.t;head=self.model.ContinuousBoundarySizeHead()
        self.assertEqual(sum(p.numel() for p in head.parameters()),2760)
        for n in (0,1,8):
            x=self.inputs(n);result=head(*x[:-1])
            self.assertTrue(t.equal(result['boxes_original'],x[-2]))
            self.assertTrue(t.equal(result['delta_roi'],t.zeros(n,2)))
            self.assertTrue(bool((result['bucket_index']==4).all()))
            self.assertEqual(tuple(result['border_support'].shape),(n,2,9,2))

    def test_rotated_original_border_midpoints_and_symmetry(self):
        t=self.t;x=self.inputs(1);x[5][:,4]=.63
        points=self.model.border_points(x[5],x[3])
        m=x[5][0];u=t.tensor([math.cos(.63),math.sin(.63)])
        v=t.tensor([-math.sin(.63),math.cos(.63)])
        self.assertTrue(t.allclose(points[0,0,0,4,4],m[:2]+50*u,atol=1e-5))
        self.assertTrue(t.allclose(points[0,1,0,4,4],m[:2]+25*v,atol=1e-5))
        self.assertTrue(t.allclose(points[0,0,0,4,0],m[:2]+50*u-25*v,atol=1e-5))
        self.assertTrue(t.allclose(points[0,:,0,:,4]+points[0,:,1,:,4],2*m[:2],atol=1e-5))

    def test_actual_sx_sy_and_different_original_model_swap(self):
        t=self.t;x=self.inputs(1)
        x[2]=t.tensor([[23.,41.,100.,100.1,.72,.7]])
        x[4]=t.tensor([[1.002,1.]])
        x[3]=x[2][:,:5].clone();x[3][:,:4]*=x[4][:,[0,1,0,1]]
        x[5]=x[2].clone()
        self.assertTrue(bool(x[2][0,2]<x[2][0,3]))
        self.assertFalse(bool(x[3][0,2]<x[3][0,3]))
        b,cb,xy,sides,rotation,m=self.model.residual.checked_inputs(*x[:-1])
        points=self.model.border_points(m,x[3])
        _,_,local=self.model.pool_borders(t.ones(1,8,9,9),x[1],points,cb,xy,sides,rotation)
        q=points[0,0,0,4,4]
        dx=float(q[0]*xy[0,0]-cb[0,0]);dy=float(q[1]*xy[0,1]-cb[0,1]);a=float(cb[0,4])
        expected=t.tensor([(dx*math.cos(a)+dy*math.sin(a))/float(sides[0,0]),
                           (-dx*math.sin(a)+dy*math.cos(a))/float(sides[0,1])])
        self.assertTrue(t.allclose(local[0,0,0,4,4],expected,atol=1e-6))
        self.assertTrue(t.equal(self.model.ContinuousBoundarySizeHead()(*x[:-1])['boxes_original'],m))

    def test_border_pool_retains_side_evidence_and_masks_support(self):
        t=self.t;x=self.inputs(1)
        b,cb,xy,sides,rotation,m=self.model.residual.checked_inputs(*x[:-1])
        points=self.model.border_points(m,x[3])
        features=t.arange(9,dtype=t.float32)[None,None,None,:].expand(1,8,9,9)
        paired,coverage,_=self.model.pool_borders(features,x[1],points,cb,xy,sides,rotation)
        self.assertAlmostEqual(float(paired[0,0,4,0]),6+2/3,places=5)
        self.assertAlmostEqual(float(paired[0,0,4,8]),1+1/3,places=5)
        self.assertGreater(float(paired[0,0,4,0]),float(paired[0,0,4,8]))
        empty,none,_=self.model.pool_borders(features,t.zeros_like(x[1]),points,cb,xy,sides,rotation)
        self.assertTrue(bool((empty==0).all()));self.assertTrue(bool((none==0).all()))
        # Entire borders outside the cached ROI are not extrapolated as evidence.
        far=points+10000
        empty,none,_=self.model.pool_borders(features,x[1],far,cb,xy,sides,rotation)
        self.assertTrue(bool((empty==0).all()));self.assertTrue(bool((none==0).all()))

    def test_gt_center_angle_and_representation_do_not_change_size_target(self):
        t=self.t;x=self.inputs(1)
        a=self.model.boundary_targets(x[6],x[5],x[3])
        gt=x[6].clone();gt[:,:2]+=1000;gt[:,4]+=.8
        b=self.model.boundary_targets(gt,x[5],x[3])
        self.assertTrue(t.equal(a['log_residual'],b['log_residual']))
        gt[:,2:4]=gt[:,2:4].flip(-1);gt[:,4]+=math.pi/2
        c=self.model.boundary_targets(gt,x[5],x[3])
        self.assertTrue(t.equal(a['log_residual'],c['log_residual']))

    def test_unclipped_targets_outside_grid_and_common_scale_supervision(self):
        t=self.t;x=self.inputs(1)
        x[6][:,2:4]*=math.exp(1.)
        target=self.model.boundary_targets(x[6],x[5],x[3])
        self.assertTrue(bool(target['outside_grid'].all()))
        self.assertTrue(bool((target['bucket_index']==8).all()))
        self.assertGreater(float(target['fine_log'].min()),.5)
        result=self.model.ContinuousBoundarySizeHead()(*x[:-1])
        result['fine_log'].retain_grad()
        terms=self.model.continuous_loss(result,x[6],x[5],x[3]);terms['total'].backward()
        self.assertTrue(bool((result['fine_log'].grad<0).all()))
        self.assertTrue(t.equal(terms['total'],terms['decoded']))

    def test_training_updates_only_new_head_and_second_stem_receives_gradients(self):
        t=self.t;t.manual_seed(1703);x=self.inputs()
        x[0].requires_grad_(True);x[1].requires_grad_(True)
        head=self.model.ContinuousBoundarySizeHead();frozen=t.nn.Linear(1,1)
        for p in frozen.parameters():p.requires_grad_(False)
        optimizer=t.optim.Adam(head.parameters(),lr=.001)
        records=[]
        for i in range(8):
            records.append(dict(sample_role='fit',eligible=True,domain='real' if i<4 else 'sim',
                scale=1. if i%4<2 else .5,**{k:v[i:i+1] for k,v in zip(
                tool.sealed.TENSOR_KEYS[:-1]+('midpoint_original','gt_original'),x)}))
        first=tool.update(head,optimizer,frozen,records,list(range(8)),t.device('cpu'),t,'boundary_continuous')
        later=tool.update(head,optimizer,frozen,records,list(range(8)),t.device('cpu'),t,'boundary_continuous')
        self.assertEqual(first['stem_grad_norm'],0.)
        self.assertGreater(first['terminal_grad_norm'],0.)
        self.assertGreater(later['stem_grad_norm'],0.)
        self.assertLessEqual(later['total_grad_norm_after_clip'],10.0001)
        self.assertGreater(first['decoded_fine_grad_norm'],0.)
        self.assertGreater(first['decoded_logit_grad_norm'],0.)
        self.assertIsNone(x[0].grad);self.assertIsNone(x[1].grad)
        self.assertTrue(all(p.grad is None for p in frozen.parameters()))
        # Invalid roles and optimizer ownership never enter an update.
        records[0]['sample_role']='probe'
        with self.assertRaises(ValueError):tool.update(head,optimizer,frozen,records,list(range(8)),t.device('cpu'),t,'boundary_continuous')
        bad=t.optim.Adam(list(head.parameters())+list(frozen.parameters()),lr=.001)
        with self.assertRaises(ValueError):tool.update(head,bad,frozen,records,list(range(8)),t.device('cpu'),t,'boundary_continuous')

    def test_canonical_direction_order_and_whole_pair_fallback(self):
        t=self.t;x=self.inputs(1);head=self.model.ContinuousBoundarySizeHead()
        with t.no_grad():head.u.bias[1]=.02;head.v.bias[1]=-.02
        result=head(*x[:-1]);self.assertTrue(bool(result['accepted'][0]))
        self.assertTrue(t.equal(result['boxes_original'][:,[0,1,4,5]],x[5][:,[0,1,4,5]]))
        with t.no_grad():head.u.bias[1]=2.
        result=head(*x[:-1]);self.assertFalse(bool(result['accepted'][0]))
        self.assertTrue(t.equal(result['boxes_original'],x[5]))
        with t.no_grad():head.u.weight[0,0]=float('nan')
        self.assertTrue(t.equal(head(*x[:-1])['boxes_original'],x[5]))
        # Near-square crossings reject the whole pair; exact squares bypass.
        x[2][:,3]=99.9;x[3]=x[2][:,:5].clone();x[5]=x[2].clone()
        head=self.model.ContinuousBoundarySizeHead()
        with t.no_grad():head.v.bias[1]=.01
        result=head(*x[:-1]);self.assertFalse(bool(result['checks']['m_raw_order'][0]))
        self.assertTrue(t.equal(result['boxes_original'],x[5]))
        x[2][:,3]=100.;x[3]=x[2][:,:5].clone();x[5]=x[2].clone()
        result=head(*x[:-1]);self.assertTrue(bool(result['square_bypass'][0]))

    def test_state_serialization_prediction_replay_and_diagnostic_rows(self):
        t=self.t;x=self.inputs(1);head=self.model.ContinuousBoundarySizeHead()
        with t.no_grad():head.u.bias[1]=.01
        before=head(*x[:-1])['boxes_original']
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'head.pth';t.save(head.state_dict(),p)
            restored=self.model.ContinuousBoundarySizeHead()
            restored.load_state_dict(tool.sealed.torch_load(t,p),strict=True)
            self.assertTrue(t.equal(before,restored(*x[:-1])['boxes_original']))
        row=dict(image='synthetic',sequence='real_seq07',domain='real',frame_id=1,
                 scale=1.,shard='val_s1',split='val',sample_role='val',eligible=True,
                 frozen_m_accepted=True,target_diagnostic=None,
                 **{k:v for k,v in zip(tool.sealed.TENSOR_KEYS[:-1]+('midpoint_original','gt_original'),x)})
        rows=tool.evaluate_rows(head,[row],t.device('cpu'),t)
        self.assertEqual(rows[0]['boundary']['predicted_bucket'],[4,4])
        self.assertEqual(rows[0]['midpoint'],x[5][0].tolist())
        json.dumps(rows,allow_nan=False)

    def test_continuous_summary_uses_outputs_and_native_long_short(self):
        row=dict(image='one',sequence='real_seq07',domain='real',frame_id=1,
            shard='val_s1',sample_role='val',eligible_train=True,gt=[0.,0.,100.,50.,0.],
            midpoint=[0.,0.,100.,50.,0.,.9],edge_residual=[0.,0.,104.,48.,0.,.9],
            size_delivery='accepted',failed_checks=[])
        missing=deepcopy(row);missing.update(image='missing',frame_id=2,midpoint=None,edge_residual=None)
        rows=[row,missing]
        for r in rows:
            r['metrics']={m:tool.old.metrics.row_metrics(r['gt'],r[m],.8 if r[m] is not None else 0.)
                          for m in tool.old.metrics.METHODS}
        with patch.object(tool.old.metrics,'evaluate',side_effect=tool.old.metrics.summarize_shards):
            summary=tool.evaluate(rows)
        candidate=summary['val_s1']['all']['real']['edge_residual']
        self.assertEqual(candidate['output_coverage']['denominator'],2)
        self.assertEqual(candidate['center_correct_conditional']['denominator'],1)
        self.assertAlmostEqual(candidate['continuous_sizes']['short_relative_bias'],-.04)
        self.assertAlmostEqual(candidate['continuous_sizes']['log_ratio_mae'],abs(math.log(1.04/.96)))

    def test_identical_capacity_and_initial_parameter_state_as_hard_head(self):
        t=self.t
        t.manual_seed(1703);hard=self.model.boundary.BoundarySizeHead()
        t.manual_seed(1703);soft=self.model.ContinuousBoundarySizeHead()
        self.assertEqual(tool.sealed.state_digest(hard),tool.sealed.state_digest(soft))
        self.assertEqual(set(hard.state_dict()),set(soft.state_dict()))

    def test_decoded_size_gradient_reaches_probabilities_and_all_fine_buckets(self):
        t=self.t;x=self.inputs(2);head=self.model.ContinuousBoundarySizeHead()
        out=head(*x[:-1]);terms=self.model.continuous_loss(out,x[6],x[5],x[3])
        logits_grad,fine_grad=t.autograd.grad(terms['total'],(out['bucket_logits'],out['fine_log']))
        self.assertGreater(float(logits_grad.abs().sum()),0.)
        self.assertTrue(bool((fine_grad!=0).all()))
        self.assertTrue(t.allclose(logits_grad.sum(-1),t.zeros(2,2),atol=1e-7))
        # Positive target: positive-grid mass is encouraged, negative mass discouraged.
        self.assertTrue(bool((logits_grad[:,:,8]<0).all()))
        self.assertTrue(bool((logits_grad[:,:,0]>0).all()))

    def test_continuous_mean_gradient_matches_finite_difference(self):
        t=self.t;grid=t.linspace(-.5,.5,9,dtype=t.float64);prior=-.5*(grid/.125)**2
        logits=(prior+.2*t.randn(1,2,9,dtype=t.float64)).requires_grad_()
        fine=(.01*t.randn(1,2,9,dtype=t.float64)).requires_grad_()
        delta=self.model.continuous_decode(logits,fine,grid,prior)['delta_roi']
        grad=t.autograd.grad(delta.sum(),logits)[0]
        epsilon=1e-6;plus=logits.detach().clone();minus=plus.clone()
        plus[0,0,6]+=epsilon;minus[0,0,6]-=epsilon
        difference=(self.model.continuous_decode(plus,fine.detach(),grid,prior)['delta_roi'].sum()-
                    self.model.continuous_decode(minus,fine.detach(),grid,prior)['delta_roi'].sum())/(2*epsilon)
        self.assertAlmostEqual(float(grad[0,0,6]),float(difference),places=8)

    def test_same_mode_can_have_different_continuous_outputs(self):
        t=self.t;grid=t.linspace(-.5,.5,9);prior=-.5*(grid/.125)**2
        base=prior.expand(1,2,9).clone();fine=t.zeros_like(base)
        changed=base.clone();changed[:,:,5]+=.1
        self.assertTrue(t.equal(base.argmax(-1),changed.argmax(-1)))
        a=self.model.continuous_decode(base,fine,grid,prior)['delta_roi']
        b=self.model.continuous_decode(changed,fine,grid,prior)['delta_roi']
        self.assertTrue(t.equal(a,t.zeros(1,2)))
        self.assertTrue(bool((b>a).all()))

    def test_direct_size_loss_uses_candidate_before_fallback(self):
        t=self.t;x=self.inputs(1);head=self.model.ContinuousBoundarySizeHead()
        with t.no_grad():head.u.bias[1]=2.
        out=head(*x[:-1]);self.assertFalse(bool(out['accepted'][0]))
        self.assertTrue(t.equal(out['boxes_original'],x[5]))
        loss=self.model.continuous_loss(out,x[6],x[5],x[3])['total']
        self.assertGreater(float(loss),.9)
        grad=t.autograd.grad(loss,out['fine_log'])[0]
        self.assertGreater(float(grad.abs().sum()),0.)


if __name__=='__main__':unittest.main()
