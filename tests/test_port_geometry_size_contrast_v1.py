"""New sampler isolation, fixed paired protocol, loss and native gradient tests."""
from collections import Counter
from copy import deepcopy
from contextlib import redirect_stderr
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

from crane_project.tools import run_port_geometry_size_contrast_v1 as tool
from crane_project.utils import port_geometry_size_contrast_v1 as design


def fixture():
    rows=[]
    for seq,n in design.diagnostic.COUNTS['train'].items():
        for i in range(n):
            rows.append(dict(image=seq+'_'+str(i).zfill(5),sequence=seq,domain=seq.split('_')[0],
                split='train_sim' if seq.startswith('sim') else 'train',frame_id=i,
                reliability_role='train',gt=[0.,0.,100.,50.,0.],
                pred=[0.,0.,85. if i%20==0 else 99.,42. if i%20==0 else 49.,0.,.9],
                b_original=[0.,0.,100.,50.,0.,.9]))
    legacy={r['image']:'probe' if i<16 else 'fit' for i,r in enumerate(rows[:64])}
    return rows,legacy


def views(plan):
    return [dict(r,scale=scale,eligible=True) for scale in (1.,.5) for r in plan['records']]


class PlanningTests(unittest.TestCase):
    def test_source_closure_and_budget(self):
        identity=tool.checked_sources()
        self.assertIn('sources',identity)
        self.assertEqual(tool.protocol_document()['settings']['steps'],200)
        self.assertEqual(tool.protocol_document()['arms'],['dual_log','dual_log_ratio'])
        self.assertFalse(tool.protocol_document()['scope']['automatic_promotion'])

    def test_val_cannot_enter_plan(self):
        rows,legacy=fixture(); rows[-1]['reliability_role']='val'
        with self.assertRaises(ValueError): design.build_plan(rows,legacy,{})

    def test_roles_ignore_predictions_and_preserve_input(self):
        rows,legacy=fixture(); original=deepcopy(rows)
        a=design.build_plan(rows,legacy,{})
        self.assertEqual(rows,original)
        for r in rows: r['pred']=[0.,0.,110.,55.,0.,.9]
        b=design.build_plan(rows,legacy,{})
        self.assertEqual([(r['image'],r['sample_role']) for r in a['records']],
                         [(r['image'],r['sample_role']) for r in b['records']])

    def test_legacy_and_purge_excluded_and_source_metadata_retained(self):
        rows,legacy=fixture()
        sampling={rows[100]['image']:dict(source_video='x',segment='train',source_frame_index=3)}
        plan=design.build_plan(rows,legacy,sampling)
        self.assertEqual(plan['role_counts']['legacy'],64)
        self.assertEqual(sum(plan['role_counts'].values()),2558)
        seq=[r for r in plan['records'] if r['sequence']=='sim_seq08']
        self.assertEqual(Counter(r['sample_role'] for r in seq),Counter(fit=589,probe=149,purged=10))
        self.assertEqual(next(r for r in plan['records'] if r['image']==rows[100]['image'])['sampling'],sampling[rows[100]['image']])
        self.assertFalse(plan['independent_holdout'])
        self.assertEqual(plan['support_sufficient'],'NOT_AUTOMATICALLY_DETERMINED')

    def test_schedule_fit_only_deterministic_and_balanced(self):
        rows,legacy=fixture(); records=views(design.build_plan(rows,legacy,{}))
        a=design.schedule(records); b=design.schedule(records)
        self.assertEqual(a,b); self.assertEqual(len(a),200)
        for batch in a:
            self.assertEqual(len(set(batch)),8)
            self.assertEqual(Counter((records[i]['domain'],records[i]['scale']) for i in batch),
                             Counter({(d,s):2 for d in ('real','sim') for s in (1.,.5)}))
            self.assertTrue(all(records[i]['sample_role']=='fit' and records[i]['eligible'] for i in batch))
        report=design.schedule_report(records,a)
        self.assertEqual(report['slots'],1600)
        for r in report['exposures']:
            if r['stratum'] not in ('correct','missing'): self.assertLessEqual(r['times'],4)
        for scale in (1.,.5):
            counts=[report['by_sequence_scale'][s+'/'+str(scale)] for s in design.diagnostic.COUNTS['train'] if s.startswith('real')]
            self.assertLessEqual(max(counts)-min(counts),1)

    def test_schedule_never_replaces_ineligible_or_missing_video(self):
        rows,legacy=fixture(); records=views(design.build_plan(rows,legacy,{}))
        for r in records:
            if r['sequence']=='real_seq12': r['eligible']=False
        with self.assertRaises(ValueError): design.schedule(records)

    def test_tuning_and_test_cli_are_absent(self):
        for args in (['--stage','test'],['--stage','finite','--steps','300'],
                     ['--stage','finite','--ratio-weight','1'],['--stage','finite','--resume','x']):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): tool.parser().parse_args(args)

    def test_prepared_plan_tamper_rejected(self):
        rows,legacy=fixture(); plan=design.build_plan(rows,legacy,{})
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); proof={'fixed':True}; identity={'same':True}
            for name,value in (('plan.json',plan),('input_identity.json',proof),('source_identity.json',identity),
                ('protocol.json',tool.protocol_document()),('completion.json',{'status':'SIZE_CONTRAST_PREPARED'})):
                tool.sealed.write(p/name,value)
            tool.sealed.write(p/'artifacts.json',dict(protocol=design.VERSION,files={
                q.name:tool.sealed.sha(q) for q in p.iterdir()}))
            tool.checked_plan(p/'plan.json',plan,proof,identity)
            (p/'plan.json').write_text(json.dumps(dict(plan,role_counts={})))
            with self.assertRaises(ValueError): tool.checked_plan(p/'plan.json',plan,proof,identity)

    def test_probe_continuous_degradation_cannot_pass(self):
        summary={s:dict(probe={d:{method:dict(edges={edge:dict(mae=.03,p95=.08)
                    for edge in ('long','short')}) for method in ('midpoint','edge_residual')}
                    for d in ('real','sim')}) for s in ('train_s1','train_s05')}
        gate=dict(checks={'inherited_failure':False})
        value=tool.extra_probe_gates(summary,gate)
        self.assertFalse(value['finite_joint_gate_pass'])
        summary['train_s1']['probe']['real']['edge_residual']['edges']['short']['mae']=.04
        value=tool.extra_probe_gates(summary,dict(checks={}))
        self.assertIn('train_s1/real/probe_short_mae',value['failed_checks'])

    def test_shell_failure_stops_and_packages_at_workdir_root(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); (p/'crane_project/tools').mkdir(parents=True); (p/'bin').mkdir()
            script=p/'crane_project/tools/run_port_geometry_size_contrast_v1.sh'
            script.write_text((tool.ROOT/'crane_project/tools/run_port_geometry_size_contrast_v1.sh').read_text())
            fake=p/'bin/python'
            fake.write_text('#!/bin/sh\necho expected-failure\nexit 19\n'); fake.chmod(0o755)
            checksum=p/'bin/sha256sum'
            checksum.write_text('#!/bin/sh\necho synthetic-checksum\n'); checksum.chmod(0o755)
            result=subprocess.run(['bash',str(script)],env=dict(os.environ,PATH=str(p/'bin')+':'+os.environ['PATH']),
                                  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            self.assertEqual(result.returncode,19,result.stdout)
            packages=list((p/'work_dirs').glob('*.tar.gz')); self.assertEqual(len(packages),1)
            with tarfile.open(packages[0]) as archive:
                names=archive.getnames()
                self.assertFalse(any('/prepare/' in n or '/finite/' in n for n in names))
                member=next(n for n in names if n.endswith('/run_exit_code.txt'))
                self.assertEqual(archive.extractfile(member).read(),b'19\n')


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Native torch requires server mmrotljj')
class NativeTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from crane_project.utils import port_geometry_midpoint_edge_residual_v1 as model
        cls.t=torch; cls.model=model

    def inputs(self):
        t=self.t
        b=t.tensor([[100.,100.,100.,50.,0.,.9]]).repeat(8,1)
        bm=b[:,:5].clone(); m=b.clone()
        gt=b[:,:5].clone(); gt[:,2]=104.; gt[:,3]=52.
        return [t.randn(8,256,9,9),t.ones(8,1,9,9),b,bm,t.ones(8,2),m,gt]

    def test_common_scale_error_not_hidden_by_ratio(self):
        t=self.t; x=self.inputs(); delta=t.zeros(8,2,requires_grad=True)
        terms=design.losses(delta,x[-1],x[-2],x[3],'dual_log_ratio')
        self.assertGreater(float(terms['dual']),0.)
        self.assertLess(float(terms['ratio']),1e-10)
        terms['total'].backward()
        self.assertTrue(bool((delta.grad<0).all()))

    def test_base_loss_exact_and_ratio_swap_invariant(self):
        t=self.t; x=self.inputs(); x[-1][:,3]=48.
        delta=t.tensor([[.02,-.01]]).repeat(8,1).requires_grad_(True)
        terms=design.losses(delta,x[-1],x[-2],x[3],'dual_log')
        self.assertTrue(t.equal(terms['total'],self.model.size_loss(delta,x[-1],x[-2],x[3])))
        self.assertEqual(terms['ratio_weight'],0.)
        bm=x[3].clone(); bm[:,2:4]=bm[:,2:4].flip(-1)
        swapped=design.losses(delta.flip(-1),x[-1],x[-2],bm,'dual_log_ratio')
        normal=design.losses(delta,x[-1],x[-2],x[3],'dual_log_ratio')
        self.assertTrue(t.equal(swapped['total'],normal['total']))

    def test_update_gradients_clip_and_frozen_inputs(self):
        t=self.t; x=self.inputs()
        for arm in design.ARMS:
            t.manual_seed(1703); head=self.model.IndependentEdgeResidualHead()
            frozen=t.nn.Linear(1,1)
            for p in frozen.parameters(): p.requires_grad_(False)
            optimizer=t.optim.Adam(head.parameters(),lr=.001)
            records=[]
            for i in range(8):
                records.append(dict(sample_role='fit',eligible=True,domain='real' if i<4 else 'sim',
                    scale=1. if i%4<2 else .5,**{k:v[i:i+1].clone() for k,v in zip(
                        tool.sealed.TENSOR_KEYS[:-1]+('midpoint_original','gt_original'),x)}))
            first=tool.update(head,optimizer,frozen,records,list(range(8)),t.device('cpu'),t,arm)
            later=tool.update(head,optimizer,frozen,records,list(range(8)),t.device('cpu'),t,arm)
            self.assertEqual(first['stem_grad_norm'],0.)
            self.assertGreater(first['terminal_grad_norm'],0.)
            self.assertGreater(later['stem_grad_norm'],0.)
            self.assertLessEqual(later['total_grad_norm_after_clip'],10.0001)
            self.assertTrue(all(p.grad is None for p in frozen.parameters()))
            self.assertTrue(all(r['midpoint_original'].grad is None for r in records))

    def test_native_serialization_exact_and_direction_preserved(self):
        t=self.t; x=self.inputs(); head=self.model.IndependentEdgeResidualHead()
        output=head(*x[:-1])
        self.assertTrue(t.equal(output['boxes_original'],x[-2]))
        optimizer=t.optim.Adam(head.parameters(),lr=.001)
        # GPU RNG roundtrip is mandatory in real smoke, not emulated here.
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'head.pth'
            with p.open('xb') as f: t.save(head.state_dict(),f)
            restored=tool.sealed.torch_load(t,p)
            self.assertTrue(tool.old.tree_equal(restored,head.state_dict(),t))
        delivered=self.model.deliver_sizes(t.tensor([[.02,-.02]]).repeat(8,1),x[2],x[3],x[4],x[5])
        self.assertTrue(t.equal(delivered['boxes_original'][:,[0,1,4,5]],x[5][:,[0,1,4,5]]))


if __name__=='__main__': unittest.main()
