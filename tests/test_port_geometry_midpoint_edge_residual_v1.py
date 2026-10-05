"""Finite learner contracts, data isolation, scalar gates and native geometry.

Native torch/evaluator tests must run in mmrotljj; no synthetic test reads real
weights, ROI files, images, annotations or TEST. Server smoke is a separate gate.
"""
import ast
from collections import Counter
from contextlib import redirect_stderr
from copy import deepcopy
import importlib.util
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crane_project.tools import preflight_port_geometry_midpoint_edge_residual_v1 as tool
from crane_project.utils import port_geometry_midpoint_edge_residual_metrics_v1 as metrics


def selected_fixture():
    roles = tool.sealed.sample_roles()
    records = [dict(image=name, sequence=s['sequence'], domain=s['domain'], split=s['split'],
        frame_id=int(name.rsplit('_', 1)[1]), scale=scale, role='train', eligible=True,
        shard='train_s1' if scale == 1. else 'train_s05')
        for scale in (1., .5) for name, s in roles.items()]
    return tool.select_views(records, roles), roles


def gate_fixture():
    """A hypothetical passing result, never a claimed measured result."""
    summary = {'val_s1': {'all': {}}}
    def pair(name, frames, size_correct, gain):
        domain = name.split('_')[0]
        base = dict(frames=frames, output_coverage=metrics.rate(frames, frames),
            center_correct_conditional=metrics.rate(frames, frames), center_correct_full_frame=metrics.rate(frames, frames),
            joint_size10_full_frame=metrics.rate(size_correct, frames),
            center_and_size10_full_frame=metrics.rate(size_correct, frames), mean_riou_full_frame=.8,
            edges={k: dict(mae=.05, p95=.09, maximum=.15) for k in ('long', 'short')},
            longest_failure_run=dict(riou=2),
            temporal={domain: dict(dfr_percent_per_frame=.1, dfr_adjacent_pairs=frames-1,
                aci=.99, aci_adjacent_pairs=frames-1, a_rmse_deg=2. if domain == 'sim' else None,
                angle_penalty_frames=frames if domain == 'sim' else 0, tdr_w10=metrics.rate(frames-9, frames-9),
                mcml_max=2, mcml_mean=2., mcml_segments=1, mcml_pass_limit5=1, mrf_mean=2., mrf_events=1)})
        after = deepcopy(base)
        after['joint_size10_full_frame'] = metrics.rate(size_correct+gain, frames)
        after['center_and_size10_full_frame'] = metrics.rate(size_correct+gain, frames)
        return dict(midpoint=base, edge_residual=after,
                    paired=dict(frozen_identity_preserved=True, new_riou_below_0_5=0))
    for name, frames, correct, gain in (('real',375,199,1), ('sim',512,464,1),
            ('real_seq07',226,63,1), ('real_seq14',149,136,0), ('sim_seq10',512,464,1)):
        summary['val_s1']['all'][name] = pair(name, frames, correct, gain)
    for shard in ('train_s1', 'train_s05'):
        summary[shard] = dict(probe={d: pair(d,8,7,0) for d in ('real','sim')},
            all=dict(overall=dict(paired=dict(frozen_identity_preserved=True))))
    return summary, {key: True for key in metrics.ENGINEERING_REQUIRED}


class SourceAndScheduleTests(unittest.TestCase):
    def test_fixed_source_identity_protocol_and_online_interface(self):
        identity = tool.checked_sources()
        self.assertEqual(identity['parent_identity'], tool.sealed.checked_sources())
        document = tool.protocol_document()
        self.assertEqual(document['stages'], ['check','smoke','finite'])
        self.assertEqual(document['settings']['steps'], 200)
        self.assertEqual(document['parameter_count'], 2818)
        self.assertFalse(document['scope']['test_access'])
        self.assertFalse(document['scope']['automatic_promotion'])
        source = tool.ROOT/'crane_project/utils/port_geometry_midpoint_edge_residual_v1.py'
        tree = ast.parse(source.read_text())
        head = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        forward = next(n for n in head.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        self.assertEqual([a.arg for a in forward.args.args],
            ['self','roi','support','boxes_original','boxes_model','scale_xy','midpoint_original'])

    def test_test_tuning_and_resume_cli_are_absent(self):
        for arguments in (['--stage','test'], ['--stage','train'], ['--stage','finite','--steps','300'],
                          ['--stage','finite','--seed','1'], ['--stage','finite','--resume','old.pth']):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                tool.parser().parse_args(arguments)

    def test_fixed64_role_counts_and_missing_views_not_replaced(self):
        selected, roles = selected_fixture()
        self.assertEqual(len({r['image'] for r in selected}), 64)
        self.assertEqual(Counter(r['sample_role'] for r in selected), {'fit':96,'probe':32})
        with self.assertRaisesRegex(ValueError, 'Missing/duplicate'):
            tool.select_views(selected[:-1], roles)
        bad = deepcopy(selected)
        bad[0]['domain'] = 'sim' if bad[0]['domain'] == 'real' else 'real'
        with self.assertRaisesRegex(ValueError, 'identity'):
            tool.select_views(bad, roles)

    def test_schedule_budget_balance_roles_and_determinism(self):
        selected, _ = selected_fixture()
        batches = tool.schedule(selected)
        self.assertEqual(batches, tool.schedule(selected))
        self.assertEqual((len(batches), sum(map(len,batches))), (200,1600))
        used = set()
        for indices in batches:
            rows = [selected[i] for i in indices]
            self.assertTrue(all(r['sample_role'] == 'fit' and r['eligible'] for r in rows))
            self.assertEqual(Counter((r['domain'],r['scale']) for r in rows),
                {('real',1.):2, ('real',.5):2, ('sim',1.):2, ('sim',.5):2})
            self.assertEqual(len(set(indices)), 8)
            used.update(indices)
        self.assertEqual(len(used), 96)

    def test_schedule_has_no_gt_error_selection_or_probe_leakage(self):
        selected, _ = selected_fixture()
        expected = tool.schedule(selected)
        for r in selected:
            r['gt_original'] = 'unreadable synthetic GT sentinel'
            r['size_error'] = float('inf')
        self.assertEqual(tool.schedule(selected), expected)
        target = next(i for i,r in enumerate(selected) if r['sample_role'] == 'fit')
        selected[target]['eligible'] = False
        self.assertNotIn(target, {i for batch in tool.schedule(selected) for i in batch})
        for r in selected:
            if r['domain'] == 'sim' and r['scale'] == .5:
                r['eligible'] = False
        with self.assertRaisesRegex(ValueError, 'no resampling'):
            tool.schedule(selected)

    def test_replay_covers_each_domain_scale(self):
        selected, _ = selected_fixture()
        replay = tool.replay_views(selected)
        self.assertEqual(Counter((r['domain'], r['scale']) for r in replay),
            {('real',1.):2, ('real',.5):2, ('sim',1.):2, ('sim',.5):2})
        self.assertTrue(all(r['sample_role'] == 'fit' and r['eligible'] for r in replay))

    def test_failed_run_is_indexed_without_overwriting_prior_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'fresh'
            args = tool.parser().parse_args(['--stage','smoke','--out-dir',str(out)])
            with patch.object(tool, 'checked_sources', side_effect=ValueError('synthetic source mismatch')):
                with self.assertRaisesRegex(ValueError, 'synthetic source mismatch'):
                    tool.run(args)
            report = tool.sealed.read(out/'completion.json')
            self.assertEqual(report['status'], 'EDGE_RESIDUAL_FAILED')
            self.assertEqual(report['size_head_updates'], 0)
            digest = tool.sealed.sha(out/'completion.json')
            self.assertEqual(tool.sealed.read(out/'artifacts.json')['files']['completion.json'], digest)
            with self.assertRaises(FileExistsError):
                tool.run(args)
            self.assertEqual(tool.sealed.sha(out/'completion.json'), digest)

    def test_unrelated_or_failed_smoke_cannot_unlock_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'completion.json'
            tool.sealed.write(path, dict(status='EDGE_RESIDUAL_FAILED'))
            with self.assertRaisesRegex(ValueError, 'successful|Successful'):
                tool.checked_smoke(path, {}, {}, {})
            with self.assertRaisesRegex(ValueError, '--smoke-report'):
                tool.checked_smoke(None, {}, {}, {})


class ScalarReportingAndGateTests(unittest.TestCase):
    def test_center_output_full_frame_and_joint_denominators(self):
        gt = [0.,0.,100.,50.,0.]
        pred = gt+[.8]
        rows = []
        for i, value in enumerate((pred, [0.,0.,120.,50.,0.,.8], None)):
            rows.append(dict(sequence='real_seq07',frame_id=i,domain='real',gt=gt,
                midpoint=value,edge_residual=value,metrics={k:metrics.row_metrics(gt,value,.8 if value else 0.) for k in metrics.METHODS}))
        result = metrics.method_summary(rows,'midpoint')
        self.assertEqual(result['output_coverage'], metrics.rate(2,3))
        self.assertEqual(result['center_correct_conditional'], metrics.rate(2,2))
        self.assertEqual(result['center_correct_full_frame'], metrics.rate(2,3))
        self.assertEqual(result['joint_size10_conditional'], metrics.rate(1,2))
        self.assertEqual(result['joint_size10_full_frame'], metrics.rate(1,3))
        self.assertEqual(result['edges']['long']['n'], 2)
        empty = metrics.method_summary([], 'midpoint')
        self.assertIsNone(empty['center_correct_conditional']['fraction'])

    def test_raw_theta_copy_alone_cannot_pass_axis_identity(self):
        row = dict(midpoint=[0.,0.,100.,95.,.3,.8], edge_residual=[0.,0.,92.,103.,.3,.8])
        self.assertFalse(metrics.frozen_identity(row))
        row['edge_residual'] = [0.,0.,103.,92.,.3,.8]
        self.assertTrue(metrics.frozen_identity(row))
        row['edge_residual'][5] += 1e-8
        self.assertFalse(metrics.frozen_identity(row))
        self.assertTrue(metrics.frozen_identity(dict(midpoint=None,edge_residual=None)))
        self.assertFalse(metrics.frozen_identity(dict(midpoint=None,edge_residual=row['edge_residual'])))

    def test_canonical_equivalence_and_exact_10percent_metric(self):
        a = [0.,0.,100.,50.,.2]
        b = [0.,0.,50.,100.,.2+math.pi/2]
        for x,y in zip(metrics.canonical(a),metrics.canonical(b)):
            self.assertAlmostEqual(x,y)
        self.assertTrue(metrics.row_metrics(a,[0.,0.,109.,50.,.2,.8],.8)['joint_size10'])
        self.assertFalse(metrics.row_metrics(a,[0.,0.,110.00001,50.,.2,.8],.8)['joint_size10'])
        self.assertAlmostEqual(metrics.percentile([0.,1.,2.]),1.9)

    def test_failure_intervals_split_gaps_and_sequences(self):
        rows = [dict(sequence=s,frame_id=i,metrics=dict(midpoint=dict(output=True,center_hit=True,riou=.4)))
                for s,i in (('real_seq07',1),('real_seq07',2),('real_seq07',6),('real_seq07',7),('real_seq14',8))]
        self.assertEqual([r['length'] for r in metrics.failure_intervals(rows,'midpoint','riou')],[2,2,1])

    def test_joint_gate_requires_both_domains_and_all_engineering_proofs(self):
        summary, proof = gate_fixture()
        self.assertTrue(metrics.finite_gates(summary,proof)['finite_joint_gate_pass'])
        bad = deepcopy(summary)
        group = bad['val_s1']['all']['sim']
        group['edge_residual']['joint_size10_full_frame'] = deepcopy(group['midpoint']['joint_size10_full_frame'])
        self.assertIn('val/sim/joint_size10',metrics.finite_gates(bad,proof)['failed_checks'])
        self.assertIn('engineering/all_required_proofs',metrics.finite_gates(summary,{'neutral_exact':True})['failed_checks'])

    def test_one_sequence_tail_cannot_be_hidden_by_domain_average(self):
        summary, proof = gate_fixture()
        summary['val_s1']['all']['real_seq14']['edge_residual']['edges']['short']['p95'] += .001
        report = metrics.finite_gates(summary,proof)
        self.assertFalse(report['finite_joint_gate_pass'])
        self.assertIn('val/real_seq14/short/p95',report['failed_checks'])
        self.assertFalse(report['automatic_promotion'])

    def test_riou_threshold_new_failure_and_mean_tolerance(self):
        summary, proof = gate_fixture()
        group = summary['val_s1']['all']['real']
        group['edge_residual']['mean_riou_full_frame'] -= 5e-6
        self.assertTrue(metrics.finite_gates(summary,proof)['finite_joint_gate_pass'])
        group['paired']['new_riou_below_0_5'] = 1
        self.assertIn('val/real/no_new_riou_failures',metrics.finite_gates(summary,proof)['failed_checks'])

    def test_aci_is_invariance_check_and_pair_counts_must_match(self):
        summary, proof = gate_fixture()
        values = summary['val_s1']['all']['sim']['edge_residual']['temporal']['sim']
        values['aci'] += 1e-4
        self.assertIn('val/sim/aci',metrics.finite_gates(summary,proof)['failed_checks'])
        values['aci'] -= 1e-4
        values['aci_adjacent_pairs'] -= 1
        self.assertIn('val/sim/aci_adjacent_pairs',metrics.finite_gates(summary,proof)['failed_checks'])

    def test_mrf_undefined_is_not_zero_and_tdr_denominators_must_match(self):
        summary, proof = gate_fixture()
        group = summary['val_s1']['all']['real_seq07']
        for method in metrics.METHODS:
            group[method]['temporal']['real'].update(mrf_mean=None,mrf_events=0)
        self.assertTrue(metrics.finite_gates(summary,proof)['finite_joint_gate_pass'])
        group['edge_residual']['temporal']['real']['mrf_mean'] = 0.
        self.assertIn('val/real_seq07/mrf_mean',metrics.finite_gates(summary,proof)['failed_checks'])
        group['edge_residual']['temporal']['real']['tdr_w10'] = metrics.rate(216,216)
        self.assertIn('val/real_seq07/tdr_w10',metrics.finite_gates(summary,proof)['failed_checks'])

    def test_probe_regression_fails_even_with_val_gain(self):
        summary, proof = gate_fixture()
        summary['train_s05']['probe']['real']['edge_residual']['joint_size10_full_frame']['numerator'] -= 1
        self.assertIn('train_s05/real/probe_joint_size10',metrics.finite_gates(summary,proof)['failed_checks'])

    def test_temporal_protection_allows_improvement_without_denominator_change(self):
        summary, proof = gate_fixture()
        group = summary['val_s1']['all']['real_seq07']
        before, after = (group[method]['temporal']['real'] for method in metrics.METHODS)
        before.update(tdr_w10=metrics.rate(215,217),mcml_pass_limit5=0,mcml_max=6,mcml_mean=6.)
        after.update(tdr_w10=metrics.rate(216,217),mcml_pass_limit5=1,mcml_max=5,mcml_mean=5.)
        self.assertTrue(metrics.finite_gates(summary,proof)['finite_joint_gate_pass'])
        after['tdr_w10'] = metrics.rate(214,217)
        self.assertIn('val/real_seq07/tdr_w10',metrics.finite_gates(summary,proof)['failed_checks'])


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'torch unavailable: native checks required in mmrotljj')
class NativeTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from crane_project.utils import port_geometry_midpoint_edge_residual_v1 as model
        from crane_project.utils import port_geometry_midpoint_sigma_v1 as frozen
        cls.torch, cls.model, cls.original, cls.frozen = torch, model, model.original, frozen

    def tensors(self, raw=None, xy=(1.,1.), m=None, n=1):
        t = self.torch
        b = t.tensor([raw or [500.,400.,100.,50.,.2,.75]], dtype=t.float32).repeat(n,1)
        scale = b.new_tensor([xy]).repeat(n,1)
        bm = b[:,:5].clone()
        bm[:,:4] *= scale[:,[0,1,0,1]]
        middle = b.clone() if m is None else b.new_tensor([m]).repeat(n,1)
        return t.zeros(n,256,9,9),t.ones(n,1,9,9),b,bm,scale,middle

    def test_zero_head_is_bitwise_m_not_b_and_has_2818_parameters(self):
        t = self.torch
        inputs = self.tensors(m=[502.,399.,104.,49.,.22,.75])
        head = self.model.IndependentEdgeResidualHead()
        result = head(*inputs)
        self.assertEqual(sum(p.numel() for p in head.parameters()),2818)
        self.assertTrue(t.equal(result['boxes_original'],inputs[-1]))
        self.assertFalse(t.equal(result['boxes_original'],inputs[2]))
        self.assertTrue(bool(result['neutral_bypass'].all()))

    def test_m_original_model_swaps_are_distinct_target_assignment(self):
        t = self.torch
        inputs = self.tensors(raw=[500.,400.,100.,100.05,.3,.75],xy=(1.01,.99),
                              m=[500.,400.,100.01,100.,.3,.75])
        _,_,b,bm,_,m = inputs
        self.assertTrue(bool((b[:,2] < b[:,3])[0]))
        self.assertFalse(bool((bm[:,2] < bm[:,3])[0]))
        goal = b.new_tensor([[500.,400.,50.,110.,.3]])
        target = self.model.size_targets(goal,m,bm)
        self.assertTrue(t.equal(target['raw_edges'],b.new_tensor([[110.,50.]])))
        self.assertTrue(t.equal(target['roi_edges'],target['raw_edges']))
        m2 = m.clone(); m2[:,2:4] = b.new_tensor([[100.,100.01]])
        self.assertTrue(t.equal(self.model.size_targets(goal,m2,bm)['raw_edges'],b.new_tensor([[50.,110.]])))

    def test_raw_swap_equivalence_and_pair_involution(self):
        t = self.torch
        inputs = self.tensors(xy=(.503,.498))
        values = inputs[2].new_tensor([[.04,-.03]])
        output = self.model.deliver_sizes(values,*inputs[2:])['boxes_original']
        swapped = [v.clone() for v in inputs]
        for index in (2,3,5):
            swapped[index][:,2:4] = swapped[index][:,[3,2]]
            swapped[index][:,4] += math.pi/2
        # Unequal sx/sy raw restoration must be rebuilt after raw-axis exchange.
        swapped[3] = swapped[2][:,:5].clone()
        swapped[3][:,:4] *= swapped[4][:,[0,1,0,1]]
        # Unequal resize makes this new representation's model frame differ;
        # test involution separately and physical equivalence with exact scale.
        pair = self.model.raw_roi_pair(values,swapped[3])
        self.assertTrue(t.equal(self.model.raw_roi_pair(pair,swapped[3]),values))
        self.assertTrue(t.equal(output[:,[0,1,4,5]],inputs[-1][:,[0,1,4,5]]))
        a = self.tensors()
        b = self.tensors(raw=[500.,400.,50.,100.,.2+math.pi/2,.75])
        first = self.model.deliver_sizes(values,*a[2:])['boxes_original']
        second = self.model.deliver_sizes(values,*b[2:])['boxes_original']
        self.assertTrue(t.allclose(self.original.canonical_boxes(first[:,:5]),
                                   self.original.canonical_boxes(second[:,:5]),atol=2e-5,rtol=1e-6))

    def test_original_b_bound_prevents_two_stage_compounding(self):
        inputs = self.tensors(m=[500.,400.,124.,50.,.2,.75])
        result = self.model.deliver_sizes(inputs[2].new_tensor([[math.log(1.1),0.]]),*inputs[2:])
        self.assertFalse(bool(result['checks']['original_b_edge_bound'][0]))
        self.assertTrue(self.torch.equal(result['boxes_original'],inputs[-1]))

    def test_crossing_rejects_entire_pair_and_canonical_90deg_change(self):
        inputs = self.tensors(raw=[500.,400.,100.,95.,.2,.75])
        result = self.model.deliver_sizes(inputs[2].new_tensor([[math.log(.92),math.log(1.08)]]),*inputs[2:])
        self.assertFalse(bool(result['checks']['m_raw_order'][0]))
        self.assertFalse(bool(result['checks']['m_canonical_direction'][0]))
        self.assertTrue(self.torch.equal(result['boxes_original'],inputs[-1]))

    def test_roi_failure_and_neutral_bypass_even_if_rebuilt_m_outside(self):
        inputs = self.tensors(raw=[0.,0.,100.,50.,0.,.75],m=[14.,0.,120.,50.,0.,.75])
        delta = inputs[2].new_tensor([[math.log(124./120.),0.]])
        result = self.model.deliver_sizes(delta,*inputs[2:])
        self.assertTrue(bool(result['checks']['original_b_edge_bound'][0]))
        self.assertFalse(bool(result['checks']['roi_midpoints'][0]))
        self.assertTrue(self.torch.equal(result['boxes_original'],inputs[-1]))
        inputs[-1][:,2] = 125.
        neutral = self.model.deliver_sizes(delta.new_zeros(1,2),*inputs[2:])
        self.assertFalse(bool(neutral['checks']['roi_midpoints'][0]))
        self.assertTrue(self.torch.equal(neutral['boxes_original'],inputs[-1]))

    def test_four_midpoints_not_stricter_four_corners(self):
        inputs = self.tensors(raw=[0.,0.,100.,50.,0.,.75],m=[0.,0.,100.,50.,math.radians(9.),.75])
        result = self.model.deliver_sizes(inputs[2].new_tensor([[math.log(1.24),math.log(1.24)]]),*inputs[2:])
        self.assertTrue(bool(result['checks']['roi_midpoints'][0]))
        self.assertTrue(bool(result['accepted'][0]))
        # A corner extends beyond the short ROI limit; it is not the contract.
        w,h,theta = (float(result['boxes_original'][0,i]) for i in (2,3,4))
        self.assertGreater((w*.5*math.sin(theta)+h*.5*math.cos(theta))/75.,.5+1e-5)

    def test_nonfinite_overflow_nonpositive_whole_pair_fallback(self):
        inputs = self.tensors()
        for value in (float('nan'),float('inf'),float('-inf'),1000.,-1000.):
            with self.subTest(value=value):
                result = self.model.deliver_sizes(inputs[2].new_tensor([[value,.01]]),*inputs[2:])
                self.assertFalse(bool(result['accepted'][0]))
                self.assertTrue(self.torch.equal(result['boxes_original'],inputs[-1]))

    def test_exact_square_and_empty_outputs(self):
        inputs = self.tensors(raw=[500.,400.,100.,100.,.2,.75])
        result = self.model.deliver_sizes(inputs[2].new_tensor([[.01,-.01]]),*inputs[2:])
        self.assertTrue(bool(result['square_bypass'][0]))
        self.assertTrue(self.torch.equal(result['boxes_original'],inputs[-1]))
        empty = [v[:0] for v in inputs]
        output = self.model.IndependentEdgeResidualHead()(*empty)
        self.assertEqual(tuple(output['boxes_original'].shape),(0,6))
        self.assertEqual(tuple(output['delta_roi'].shape),(0,2))

    def test_loss_before_neutral_fallback_and_no_target_clipping(self):
        t = self.torch
        inputs = self.tensors()
        gt = inputs[2].new_tensor([[500.,400.,150.,60.,.2]])
        delta = inputs[2].new_zeros(1,2).requires_grad_()
        self.assertTrue(t.equal(self.model.deliver_sizes(delta,*inputs[2:])['boxes_original'],inputs[-1]))
        target = self.model.size_targets(gt,inputs[-1],inputs[3])['log_residual']
        self.assertGreater(float(target[0,0]),math.log(1.25))
        loss = self.model.size_loss(delta,gt,inputs[-1],inputs[3]); loss.backward()
        self.assertGreater(float(delta.grad.norm()),0.)
        self.assertIsNone(gt.grad)

    def test_corrupt_m_score_and_inherited_bounds_are_rejected_before_delivery(self):
        inputs = self.tensors()
        delta = inputs[2].new_zeros(1,2)
        for index,value in ((5,.5),(0,550.),(4,.5),(2,130.)):
            bad = [v.clone() for v in inputs]
            bad[-1][:,index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.model.deliver_sizes(delta,*bad[2:])

    def test_only_new_parameters_and_later_stem_receive_gradients(self):
        t = self.torch
        t.manual_seed(1703)
        head = self.model.IndependentEdgeResidualHead()
        frozen = self.frozen.SigmaMidpointHead(1.5).eval()
        frozen.requires_grad_(False)
        before = tool.sealed.state_digest(frozen)
        inputs = self.tensors(n=8)
        inputs[0] = t.randn(8,256,9,9)
        # Intentionally marked grad inputs still must be detached by the head.
        for value in inputs:
            value.requires_grad_()
        gt = inputs[2].detach().clone()[:,:5]; gt[:,2:4] *= gt.new_tensor([1.08,.85])
        gt.requires_grad_()
        optimizer = t.optim.Adam(head.parameters(),lr=.001)
        norms = []
        for _ in range(2):
            optimizer.zero_grad()
            out = head(*inputs)
            loss = self.model.size_loss(out['delta_roi'],gt,inputs[-1],inputs[3]); loss.backward()
            norms.append((sum(float(p.grad.square().sum()) for p in head.stem.parameters()),
                          sum(float(p.grad.square().sum()) for p in list(head.u.parameters())+list(head.v.parameters()))))
            optimizer.step()
        self.assertEqual(norms[0][0],0.)
        self.assertGreater(norms[0][1],0.)
        self.assertGreater(norms[1][0],0.)
        self.assertTrue(all(value.grad is None for value in inputs+[gt]))
        self.assertTrue(all(p.grad is None for p in frozen.parameters()))
        self.assertEqual(tool.sealed.state_digest(frozen),before)

    def test_native_checkpoint_optimizer_rng_and_prediction_replay(self):
        t = self.torch
        t.manual_seed(1703)
        records, _ = selected_fixture()
        prepared = []
        for r in tool.replay_views(records):
            roi,support,b,bm,xy,m = self.tensors()
            roi = t.randn_like(roi)
            prepared.append(dict(r,roi=roi,support=support,boxes_original=b,boxes_model=bm,scale_xy=xy,
                midpoint_original=m,gt_original=b[:,:5].clone(),target_diagnostic=None,frozen_m_accepted=True))
        t.manual_seed(1703)
        head = self.model.IndependentEdgeResidualHead()
        optimizer = t.optim.Adam(head.parameters(),lr=.001)
        frozen = self.frozen.SigmaMidpointHead(1.5).eval(); frozen.requires_grad_(False)
        for r in prepared:
            r['gt_original'][:,2:4] *= r['gt_original'].new_tensor([1.08,.85])
        for _ in range(2):
            tool.update(head,optimizer,frozen,prepared,list(range(8)),t.device('cpu'),t,self.model)
        with tempfile.TemporaryDirectory() as tmp:
            loaded,opt,proof = tool.save_reload(Path(tmp),'smoke',2,head,optimizer,prepared,
                {'fixture':'source'},{'fixture':'context'},t.device('cpu'),t,self.model)
            self.assertTrue(proof['save_reload_exact'])
            self.assertEqual(proof['prediction_replay_views'],8)
            self.assertEqual(tool.sealed.state_digest(head),tool.sealed.state_digest(loaded))
            self.assertTrue(tool.tree_equal(optimizer.state_dict(),opt.state_dict(),t))


@unittest.skipUnless(importlib.util.find_spec('numpy') is not None and importlib.util.find_spec('cv2') is not None,
                     'numpy/OpenCV unavailable: native evaluator check required in mmrotljj')
class NativeEvaluatorTests(unittest.TestCase):
    def test_size_changes_dfr_and_riou_but_freezes_aci_and_angle_penalty(self):
        rows = []
        for i, width in enumerate((100.,102.,105.,100.,99.,101.,100.,103.,100.,102.,100.,101.)):
            rows.append(dict(shard='val_s1',sample_role='val',eligible_train=False,
                domain='sim',sequence='sim_seq10',frame_id=i,image='sim_seq10_%05d'%i,
                gt=[0.,0.,100.,50.,.1],midpoint=[0.,0.,width,50.,.1,.75],
                edge_residual=[0.,0.,100.,50.,.1,.75],size_delivery='accepted',failed_checks=[]))
        value = metrics.evaluate(rows)['val_s1']['all']['sim']
        a,b = (value[method] for method in metrics.METHODS)
        self.assertEqual(a['temporal']['sim']['aci'],b['temporal']['sim']['aci'])
        self.assertEqual(a['temporal']['sim']['a_rmse_deg'],b['temporal']['sim']['a_rmse_deg'])
        self.assertEqual(a['temporal']['sim']['aci_adjacent_pairs'],11)
        self.assertGreater(a['temporal']['sim']['dfr_percent_per_frame'],b['temporal']['sim']['dfr_percent_per_frame'])
        self.assertGreater(b['mean_riou_full_frame'],a['mean_riou_full_frame'])
        self.assertIsNone(a['temporal']['sim']['mrf_mean'])
        self.assertEqual(a['temporal']['sim']['mrf_events'],0)


if __name__ == '__main__':
    unittest.main()
