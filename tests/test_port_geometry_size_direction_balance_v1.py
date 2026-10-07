"""Single-factor contribution contracts; native tests use synthetic CPU tensors."""
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from crane_project.utils import port_geometry_size_direction_balance_v1 as d
from crane_project.tools import run_port_geometry_size_direction_balance_v1 as tool


def direction_fixture():
    rows=[]
    for shard,scale in [('train_s1',1.),('train_s05',.5)]:
        for role in ['fit','probe']:
            for domain in ['real','sim']:
                for label,l,s in [('need_enlarge',94.,47.),('need_shrink',106.,53.),('near_correct',100.5,50.25),('mixed',106.,47.)]:
                    pred=[0.,0.,l,s,0.,.8];corrected=[0.,0.,(l+100)/2,(s+50)/2,0.,.8]
                    rows.append(dict(image=domain+'_'+role+'_'+label,shard=shard,scale=scale,sample_role=role,
                        eligible_train=True,domain=domain,gt=[0.,0.,100.,50.,0.],midpoint=pred,edge_residual=corrected))
    return rows


class ContractTests(unittest.TestCase):
    def test_groups_are_disjoint_canonical_and_keep_near_correct(self):
        gt=[0.,0.,100.,50.,0.]
        for label,l,s in [('need_enlarge',94.,49.),('need_shrink',106.,51.),('near_correct',101.,49.),('mixed',106.,47.)]:
            self.assertEqual(d.group(gt,[0.,0.,l,s,0.,.8]),label)
            self.assertEqual(d.group(gt,[0.,0.,s,l,math.pi/2,.8]),label)
        self.assertEqual(d.group(gt,None),'missing')
        self.assertEqual(d.group(gt,[0.,0.,100.,55.,0.,.8]),'mixed')

    def test_capped_balance_uses_actual_exposure_and_preserves_total_mass(self):
        n=dict(need_enlarge=16,need_shrink=127,near_correct=218,mixed=39)
        w=d.capped_weights(n)
        self.assertEqual(w['need_enlarge'],4.)
        self.assertAlmostEqual(w['near_correct'],112/218)
        self.assertAlmostEqual(sum(n[g]*w[g] for g in n),400)
        self.assertTrue(all(0<x<=4 for x in w.values()))
        with self.assertRaises(ValueError):d.capped_weights(dict(n,need_enlarge=0))

    def test_pinned_contract_has_no_role_leak_and_keeps_both_scales(self):
        x=tool.sealed.read(tool.EXPOSURE)
        self.assertEqual(x['slots'],1600)
        self.assertFalse(x['test_or_val_used'])
        self.assertEqual(set(x['cells']),{'real/1.0','real/0.5','sim/1.0','sim/0.5'})
        self.assertTrue(all(r['role']=='fit' and r['eligible'] for r in x['scheduled_views']))
        for cell in x['cells'].values():
            self.assertEqual(sum(cell['slots'].values()),400)
            self.assertAlmostEqual(sum(cell['weighted_slots'].values()),400)

    def test_direction_gates_reject_wrong_direction_despite_other_improvements(self):
        summary=d.direction_summary(direction_fixture())
        self.assertTrue(d.direction_gates(summary)['direction_gate_pass'])
        bad=deepcopy(summary);key='train_s1/fit/sim/need_enlarge'
        bad[key]['corrected']['common_log_mae']=bad[key]['midpoint']['common_log_mae']+.001
        self.assertFalse(d.direction_gates(bad)['direction_gate_pass'])
        bad=deepcopy(summary);key='train_s1/probe/real/near_correct'
        bad[key]['new_size10_errors_from_correct']=1
        self.assertFalse(d.direction_gates(bad)['direction_gate_pass'])

    def test_inadequate_direction_support_is_explicit_not_automatic_pass(self):
        x=d.direction_summary(direction_fixture());del x['train_s1/probe/sim/need_enlarge']
        gate=d.direction_gates(x)
        self.assertIn('train_s1/probe/sim/need_enlarge/support_present',gate['failed_checks'])

    def test_train_control_failure_blocks_val_without_io(self):
        with patch.object(tool.control,'load_val_after_gate',side_effect=AssertionError('VAL touched')):
            with self.assertRaises(ValueError):tool.load_val_after_gate(dict(train_gate_pass=False))
        gates=dict(direction_balanced=dict(checks=dict(direction=True)))
        contrast=dict(checks=dict(control=False))
        self.assertFalse(tool.train_gate_for_val(gates,contrast)['train_gate_pass'])

    def test_sources_and_old_model_contract_are_unchanged(self):
        x=tool.checked_sources()
        self.assertEqual(x['control_identity'],tool.control.checked_sources())
        self.assertEqual(tool.cache_identity(x),tool.sealed.checked_sources())
        p=tool.protocol_document()
        self.assertEqual(p['arms'],['uniform_control','direction_balanced'])
        self.assertEqual(p['parameter_count'],2760)
        self.assertFalse(p['scope']['test_access']);self.assertFalse(p['scope']['automatic_promotion'])
        self.assertEqual(p['model_settings'],tool.control.MODEL_SETTINGS)

    def test_static_check_does_not_touch_collection_model_or_gpu(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'check';args=tool.parser().parse_args(['--stage','check','--out-dir',str(out)])
            with patch.object(tool,'checked_output_directory',return_value=out),patch.object(tool,'current_plan',side_effect=AssertionError('collection')):
                tool.run(args)
            r=tool.sealed.read(out/'completion.json')
            self.assertEqual(r['status'],'SIZE_DIRECTION_STATIC_CONTRACT_PASS')
            self.assertFalse(r['model_loaded']);self.assertFalse(r['val_tensor_access'])
            self.assertEqual(r['parameter_updates'],0)

    def test_missing_or_old_smoke_cannot_unlock_finite(self):
        with self.assertRaises(ValueError):tool.checked_smoke(None,{}, {})
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);tool.sealed.write(p/'completion.json',dict(protocol='old'))
            tool.sealed.write(p/'artifacts.json',dict(files={}))
            with self.assertRaises(ValueError):tool.checked_smoke(p,{}, {})

    def test_output_never_overwrites_old_or_git_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'work_dirs/port_geometry_size_boundary_v1'/(tool.VERSION+'_synthetic')/'finite'
            self.assertEqual(tool.checked_output_directory(p,root),p.resolve())
            p.mkdir(parents=True)
            with self.assertRaises(FileExistsError):tool.checked_output_directory(p,root)
            with self.assertRaises(ValueError):tool.checked_output_directory(root/'docs/finite',root)

    def test_shell_failure_stops_and_archives_without_weights_at_workdirs_root(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'crane_project/tools').mkdir(parents=True);(root/'bin').mkdir()
            p=root/'crane_project/tools'/('run_'+tool.VERSION+'.sh')
            p.write_text((tool.ROOT/'crane_project/tools'/p.name).read_text())
            fake=root/'bin/python';fake.write_text('#!/bin/sh\nexit 19\n');fake.chmod(0o755)
            sha=root/'bin/sha256sum';sha.write_text('#!/bin/sh\necho synthetic-sha\n');sha.chmod(0o755)
            result=subprocess.run(['bash',str(p)],env=dict(os.environ,PATH=str(root/'bin')+':'+os.environ['PATH']),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            self.assertEqual(result.returncode,19,result.stdout)
            files=list((root/'work_dirs').glob('*.tar.gz'));self.assertEqual(len(files),1)
            with tarfile.open(files[0]) as archive:
                names=archive.getnames()
                self.assertFalse(any(n.endswith(('.pt','.pth','.pkl')) for n in names))
                member=next(n for n in names if Path(n).name=='run_exit_code.txt')
                self.assertEqual(archive.extractfile(member).read(),b'19\n')


@unittest.skipUnless(importlib.util.find_spec('torch'),'native torch unavailable')
class NativeTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from crane_project.utils import port_geometry_size_boundary_continuous_v1 as model
        cls.t,cls.model=torch,model;torch.set_num_threads(1)

    def inputs(self,n=8):
        t=self.t;b=t.tensor([[100.,100.,100.,50.,0.,.9]]).repeat(n,1)
        gt=b[:,:5].clone();gt[:,2:4]*=1.04
        return [t.randn(n,256,9,9),t.ones(n,1,9,9),b,b[:,:5].clone(),t.ones(n,2),b.clone(),gt]

    def records(self,x):
        return [dict(sample_role='fit',eligible=True,domain='real' if i<4 else 'sim',scale=1. if i%4<2 else .5,
            direction_group='need_enlarge',direction_weight=2.,**{k:v[i:i+1] for k,v in zip(tool.sealed.TENSOR_KEYS[:-1]+('midpoint_original','gt_original'),x)}) for i in range(8)]

    def test_weighting_changes_only_per_sample_gradient_multiplier(self):
        t=self.t;x=self.inputs(4);delta=t.zeros(4,2,requires_grad=True);out=dict(delta_roi=delta)
        weights=[4.,.5,2.,.75]
        base=d.loss(out,x[6],x[5],x[3],[1.]*4,'uniform_control')
        expected=self.model.continuous_loss(out,x[6],x[5],x[3])
        self.assertTrue(t.equal(base['total'],expected['total']))
        g0=t.autograd.grad(base['total'],delta,retain_graph=True)[0]
        weighted=d.loss(out,x[6],x[5],x[3],weights,'direction_balanced')
        g1=t.autograd.grad(weighted['total'],delta)[0]
        self.assertTrue(t.allclose(g1,g0*t.tensor(weights)[:,None],atol=1e-7,rtol=1e-6))
        with self.assertRaises(ValueError):d.loss(out,x[6],x[5],x[3],[5.]*4,'direction_balanced')
        with self.assertRaises(ValueError):d.loss(out,x[6],x[5],x[3],[2.]*4,'uniform_control')

    def test_uniform_two_steps_match_original_updates_bitwise(self):
        t=self.t;t.manual_seed(1703);x=self.inputs();records=self.records(x)
        h0=self.model.ContinuousBoundarySizeHead();h1=self.model.ContinuousBoundarySizeHead();h1.load_state_dict(h0.state_dict())
        frozen=t.nn.Linear(1,1)
        for p in frozen.parameters():p.requires_grad_(False)
        o0=t.optim.Adam(h0.parameters(),lr=.001);o1=t.optim.Adam(h1.parameters(),lr=.001)
        for _ in range(2):
            tool.control.update(h0,o0,frozen,records,list(range(8)),t.device('cpu'),t,'boundary_continuous')
            event=tool.update(h1,o1,frozen,records,list(range(8)),t.device('cpu'),t,'uniform_control')
            self.assertTrue(all(t.equal(a,b) for a,b in zip(h0.parameters(),h1.parameters())))
            self.assertAlmostEqual(sum(v['weighted_loss_contribution'] for v in event['direction_contribution'].values()),event['loss'],places=7)
        self.assertTrue(tool.old.tree_equal(o0.state_dict(),o1.state_dict(),t))

    def test_balanced_update_owns_only_new_head_and_keeps_frozen_inputs(self):
        t=self.t;t.manual_seed(1703);x=self.inputs();x[0].requires_grad_(True);x[1].requires_grad_(True)
        head=self.model.ContinuousBoundarySizeHead();frozen=t.nn.Linear(1,1)
        for p in frozen.parameters():p.requires_grad_(False)
        optimizer=t.optim.Adam(head.parameters(),lr=.001);records=self.records(x)
        first=tool.update(head,optimizer,frozen,records,list(range(8)),t.device('cpu'),t,'direction_balanced')
        later=tool.update(head,optimizer,frozen,records,list(range(8)),t.device('cpu'),t,'direction_balanced')
        self.assertEqual(first['stem_grad_norm'],0);self.assertGreater(later['stem_grad_norm'],0)
        self.assertGreater(first['decoded_logit_grad_norm'],0);self.assertGreater(first['decoded_fine_grad_norm'],0)
        self.assertIsNone(x[0].grad);self.assertIsNone(x[1].grad);self.assertTrue(all(p.grad is None for p in frozen.parameters()))
        records[0]['sample_role']='probe'
        with self.assertRaises(ValueError):tool.update(head,optimizer,frozen,records,list(range(8)),t.device('cpu'),t,'direction_balanced')

    def test_inference_stays_gt_free_neutral_and_direction_preserving(self):
        t=self.t;x=self.inputs();head=self.model.ContinuousBoundarySizeHead()
        self.assertEqual(sum(p.numel() for p in head.parameters()),2760)
        a=head(*x[:-1]);self.assertTrue(t.equal(a['boxes_original'],x[5]))
        x[-1][:,2:4]*=2;b=head(*x[:-1]);self.assertTrue(t.equal(a['boxes_original'],b['boxes_original']))
        with t.no_grad():head.u.bias[1]=.02;head.v.bias[1]=-.02
        c=head(*x[:-1]);self.assertTrue(t.equal(c['boxes_original'][:,[0,1,4,5]],x[5][:,[0,1,4,5]]))


if __name__=='__main__':unittest.main()
