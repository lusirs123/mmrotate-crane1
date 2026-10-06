"""Migration identity, lossless component flags and offline stage contracts."""
from argparse import Namespace
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from crane_project.tools import run_port_midpoint_sigma15_reliability_v1 as entry
from crane_project.utils import port_midpoint_sigma15_reliability_v1 as new
from crane_project.utils import port_simple_component_reliability_v1 as simple

SETTINGS = json.loads(entry.base.PROTOCOL.read_text())


def frontend():
    return dict(name='B24_midpoint_sigma1p5_epoch03', frozen_b=SETTINGS['frozen_b'],
        midpoint_checkpoint=dict(path='head_epoch_03.pth', epoch=3, sigma_cells=1.5,
            updates=2706, sha256=new.HEAD_SHA), selection_sha256='sealed')


def fixture(full=False):
    result = {}
    for role in ('train','val'):
        counts = SETTINGS['counts'][role] if full else dict(real_seq07=30, sim_seq10=30)
        values = []
        for seq, count in counts.items():
            for i in range(count):
                gt = [120.,90.,80.,35.,.27]
                b = gt+[.11+.8*((i*13)%31)/31]
                if i%3: b[2] *= 1.2
                if i%4: b[4] += .13
                pred = deepcopy(b)
                pred[3] *= 1.01
                if role == 'val' and i == 0 and seq.startswith('real'):
                    b = pred = None
                row = dict(image=seq+'_%05d'%i, sequence=seq, domain=seq.split('_')[0],
                    frame_id=i, split='train_sim' if role=='train' and seq.startswith('sim') else role,
                    image_size=[640,480], gt=gt, pred=pred, b_original=b,
                    midpoint_accepted=True if pred else None,
                    midpoint_candidate=deepcopy(pred), train_angle_eligible=role=='train',
                    angle_axis_well_defined=True)
                values.append(row)
        result[role] = values
    return result['train'], result['val']


def make_policy(train, val):
    return dict(protocol=new.VERSION, front_end=frontend(), contract={'frozen':'contract'},
        simple_policy=simple.create_policy(train, val, SETTINGS))


def predictions(values):
    return [dict(image=r['image'],sequence=r['sequence'],domain=r['domain'],frame_id=r['frame_id'],
        gt=r['gt'],b=r['b_original'],midpoint=r['pred'],candidate=r['midpoint_candidate'],
        accepted=r['midpoint_accepted']) for r in values]


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.val = fixture()
        cls.policy = make_policy(cls.train, cls.val)

    def test_three_flags_rejection_keeps_center_and_exact_raw_box(self):
        policy = deepcopy(self.policy)
        for component in ('size','angle'):
            policy['simple_policy']['cutoffs']['simple'][component]['risk_le'] = 0.
        runtime = new.Sigma15Reliability(policy, frontend())
        box = [120,90,80,35,.27,.8]; before = deepcopy(box)
        d = runtime.decide(box,[640,480])
        self.assertTrue(d['center_accepted'])
        self.assertFalse(d['size_accepted']); self.assertFalse(d['angle_accepted'])
        self.assertEqual(d['final_box_original'], before); self.assertEqual(box, before)
        self.assertNotIn('raw_b_output',d)
        d = runtime.decide(None,[640,480])
        self.assertIsNone(d['final_box_original'])
        self.assertTrue(all(d[c+'_accepted'] is False for c in simple.COMPONENTS))

    def test_wrong_sigma_epoch_b_weight_and_old_protocol_refused(self):
        for key, value in (('epoch',23),('sigma_cells',1.),('updates',200),('sha256','old')):
            front = frontend(); front['midpoint_checkpoint'][key] = value
            with self.assertRaises(ValueError): new.Sigma15Reliability(dict(self.policy,front_end=front),front)
        front = frontend(); front['frozen_b'] = dict(front['frozen_b'],checkpoint_sha256='wrong')
        with self.assertRaises(ValueError): new.Sigma15Reliability(dict(self.policy,front_end=front),front)
        with self.assertRaises(ValueError): new.Sigma15Reliability(dict(self.policy,protocol='port_midpoint_reliability_v1'),frontend())
        with self.assertRaises(ValueError): new.Sigma15Reliability(self.policy,dict(frontend(),selection_sha256='changed'))

    def test_swapped_edges_pi_and_isotropic_resize_preserve_flags_and_risks(self):
        runtime = new.Sigma15Reliability(self.policy,frontend())
        box = [120,90,80,35,.27,.8]
        result = runtime.decide(box,[640,480])
        variants = [(box[:2]+[35,80,.27+math.pi/2,.8],[640,480]),
            (box[:4]+[.27+math.pi,.8],[640,480]),
            ([60,45,40,17.5,.27,.8],[320,240])]
        for pred, size in variants:
            value = runtime.decide(pred,size)
            for component in simple.COMPONENTS:
                self.assertEqual(value[component+'_accepted'],result[component+'_accepted'])
            np.testing.assert_allclose(list(value['risks'].values()),list(result['risks'].values()),rtol=1e-12)
            self.assertEqual(value['final_box_original'],pred)

    def test_api_has_no_GT_domain_sequence_or_history(self):
        self.assertEqual(list(inspect.signature(new.Sigma15Reliability.decide).parameters),
            ['self','final_box_original','image_size','method'])
        runtime = new.Sigma15Reliability(self.policy,frontend())
        with self.assertRaises(TypeError): runtime.decide(self.val[1]['pred'],[640,480],gt=self.val[1]['gt'])

    def test_calibration_does_not_depend_on_VAL_GT(self):
        changed = deepcopy(self.val)
        for row in changed: row['gt'] = [20,30,35,15,-.8]
        self.assertEqual(simple.create_policy(self.train,self.val,SETTINGS),
                         simple.create_policy(self.train,changed,SETTINGS))

    def test_current_TRAIN_geometry_drives_refit_not_old_weights(self):
        changed = deepcopy(self.train)
        for i, row in enumerate(changed):
            if i%2: row['pred'][2] *= .7
        self.assertNotEqual(simple.create_policy(changed,self.val,SETTINGS)['models'],self.policy['simple_policy']['models'])

    def test_domain_video_states_coverage_and_matched_counts(self):
        decisions, groups = new.evaluate(self.val,self.policy,self.policy['simple_policy'])
        self.assertEqual(set(groups),{'all','domain:real','domain:sim','sequence:real_seq07','sequence:sim_seq10'})
        summary = groups['domain:real']
        self.assertEqual(summary['output_frames'],29); self.assertEqual(summary['frames'],30)
        self.assertEqual(summary['output_coverage'],29/30)
        self.assertEqual(summary['center_hit_rate_on_outputs'],1.)
        self.assertEqual(summary['all_frame_center_correct_coverage'],29/30)
        for group in groups.values():
            for component in simple.COMPONENTS:
                stats = group['components'][component]
                item = stats['simple']
                self.assertEqual(item['output_frames'],sum(item[k] for k in ('false_accept','false_reject','error_detected','correct_retained')))
                for method in ('matched_score_diagnostic','matched_old_simple_diagnostic'):
                    self.assertEqual(stats[method]['accepted_frames'],item['accepted_frames'])
                self.assertLessEqual(stats['coverage_false_accept_lower_bound'],item['false_accept'])
        self.assertTrue(all(d['methods']['simple']['final_box_original']==r['pred'] for d,r in zip(decisions,self.val)))

    def test_angle_ineligible_acceptance_has_separate_all_frame_denominator(self):
        changed = deepcopy(self.val)
        for row in changed[:10]: row['angle_axis_well_defined'] = False
        _, stats = new.evaluate(changed,self.policy,self.policy['simple_policy'])
        item = stats['all']['components']['angle']['raw']
        self.assertEqual(item['eligible_frames'],50)
        self.assertEqual(item['online_accepted_frames_all'],59)
        self.assertEqual(item['online_accepted_coverage_all_frames'],59/60)
        self.assertEqual(item['unassessed_accepted'],9)

    def test_invalid_prediction_and_method_refused(self):
        runtime = new.Sigma15Reliability(self.policy,frontend())
        for pred in ([120,90,-80,35,0.,.8],[120,90,80,35,0.,.05],[120,90,80,35,float('nan'),.8]):
            with self.assertRaises(ValueError): runtime.decide(pred,[640,480])
        with self.assertRaises(ValueError): runtime.decide(self.val[1]['pred'],[640,480],'ranked')


class StageTests(unittest.TestCase):
    def test_prepare_binds_sigma_selection_and_exact_parent_cache_identity(self):
        train, val = fixture()
        original_train = [dict(r,pred=r['b_original']) for r in train]
        original_val = [dict(r,pred=r['b_original']) for r in val]
        for values in (original_train,original_val):
            for row in values:
                for key in ('b_original','midpoint_accepted','midpoint_candidate'): row.pop(key)
        selected = frontend()['midpoint_checkpoint']
        proof = dict(training_proof=dict(identity=dict(parent_identity={'sealed':'parent'})),
                     training_runtime={'sealed':'runtime'})
        cache = dict(manifest_sha256=entry.CACHE_SHA,runtime=proof['training_runtime'])
        fake = SimpleNamespace(checked_sources=lambda:{'sealed':'sources'},
            checked_selection=lambda *a:(dict(selected_checkpoint=selected),proof),
            t=SimpleNamespace(checked_cache_metadata=lambda path,parent:
                cache if parent == {'sealed':'parent'} else (_ for _ in ()).throw(AssertionError('Wrong nested parent identity'))))
        old = dict(protocol=entry.binding.VERSION,front_end=dict(midpoint_checkpoint={'epoch':23},
            frozen_b=SETTINGS['frozen_b']),simple_policy=simple.create_policy(train,val,SETTINGS))
        from crane_project import tools as package
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); selection = root/'selection.json'; selection.write_text('{}')
            with (root/'val_epoch_03.rows.jsonl').open('w') as stream:
                for row in predictions(val): stream.write(json.dumps(row)+'\n')
            legacy = root/'old_policy.json'; legacy.write_text(json.dumps(old))
            bpath = root/'B.pth'; bpath.write_text('synthetic')
            args = Namespace(selection=selection,formal_cache=root,b_checkpoint=bpath,old_policy=legacy)
            real_sha = entry.base.sha
            def sha(path):
                if Path(path)==legacy: return entry.OLD_POLICY_SHA
                if Path(path)==bpath: return new.B_SHA
                return real_sha(path)
            with patch.object(package,'eval_port_geometry_midpoint_sigma15_v1',fake,create=True), \
                    patch.object(entry,'checked_sources',return_value=({'protocol':new.VERSION},{'sealed':True})), \
                    patch.object(entry.base,'checked_inputs',return_value=(original_train,original_val,{'sealed':True})), \
                    patch.object(entry.base,'sha',side_effect=sha):
                prepared = entry.prepare(args)
                self.assertEqual(prepared['final'],val)
                self.assertEqual(prepared['contract']['front_end']['midpoint_checkpoint'],selected)
                self.assertEqual(prepared['cache'],cache)

    def test_actual_sigma_head_identity_forward_including_empty_output(self):
        import torch
        from crane_project.utils.port_geometry_midpoint_sigma_v1 import SigmaMidpointHead
        head = SigmaMidpointHead(1.5).eval().requires_grad_(False)
        for count in (0,1):
            b = torch.tensor([[120.,90.,80.,35.,.27,.8]])[:count]
            before = deepcopy(head.state_dict())
            with torch.no_grad():
                output = head(torch.zeros(count,256,9,9),torch.ones(count,1,9,9),
                    b,b[:,:5],torch.ones(count,2))
            self.assertTrue(torch.equal(output['boxes_original'],b))
            self.assertTrue(all(torch.equal(before[k],v) for k,v in head.state_dict().items()))
            self.assertFalse(any(p.requires_grad for p in head.parameters()))

    def test_sources_are_sealed_without_loading_checkpoint_or_detector(self):
        p,_ = entry.checked_sources()
        self.assertEqual(p['midpoint_sha256'],new.HEAD_SHA)
        self.assertFalse(p['test_read'])

    def test_replay_rejects_score_missing_metadata_and_box_changes(self):
        _, val = fixture()
        entry.verify_replay(val,deepcopy(val))
        for key, value in (('pred',None),('gt',[1,2,3,4,0]),('midpoint_accepted',False)):
            changed = deepcopy(val); changed[1][key] = value
            with self.assertRaises(ValueError): entry.verify_replay(changed,val)
        changed = deepcopy(val); changed[1]['pred'][5] += .001
        with self.assertRaises(ValueError): entry.verify_replay(changed,val)
        changed = deepcopy(val); changed[1]['pred'][2] += .5
        with self.assertRaises(ValueError): entry.verify_replay(changed,val)
        with self.assertRaises(ValueError): entry.verify_replay(val[:-1],val)

    def test_original_B_pair_is_verified_even_when_midpoint_scores_match(self):
        _, val = fixture()
        sources = [dict(r,pred=r['b_original']) for r in val]
        entry.validate_b_pair(sources,val)
        changed = deepcopy(val); changed[1]['b_original'][0] += 3
        with self.assertRaises(ValueError): entry.validate_b_pair(sources,changed)
        changed = deepcopy(val); changed[1]['b_original'] = None
        with self.assertRaises(ValueError): entry.validate_b_pair(sources,changed)

    def test_completion_detects_mutation_failure_and_wrong_contract(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            entry.base.write_new(out/'check_report.json',{'pass':True})
            entry.finish(out,'check',{'frozen':1})
            entry.completed(out,'check',{'frozen':1})
            with self.assertRaises(ValueError): entry.completed(out,'check',{'frozen':2})
            (out/'check_report.json').write_text('{}')
            with self.assertRaises(ValueError): entry.completed(out,'check',{'frozen':1})
            (out/'failure.json').write_text('{}')
            with self.assertRaises(ValueError): entry.completed(out,'check',{'frozen':1})

    def test_full_synthetic_collection_fit_save_reload_and_tamper(self):
        train, val = fixture(full=True)
        sources_train = [dict(r,pred=r['b_original']) for r in train]
        sources_val = [dict(r,pred=r['b_original']) for r in val]
        for values in (sources_train,sources_val):
            for row in values:
                for key in ('b_original','midpoint_accepted','midpoint_candidate'): row.pop(key)
        contract = dict(final_VAL_fingerprint=simple.fingerprint(val), front_end=frontend())
        prepared = dict(train=sources_train,val=sources_val,final=val,contract=contract,
            old_protocol=SETTINGS,old_simple=simple.create_policy(train,val,SETTINGS))
        with tempfile.TemporaryDirectory() as d:
            run = Path(d); collect = run/'collect'; collect.mkdir()
            with (collect/'final_rows.jsonl').open('x') as stream:
                for role, values in (('train',train),('val',val)):
                    for row in values: stream.write(json.dumps(dict(row,reliability_role=role))+'\n')
            entry.finish(collect,'collect',contract)
            args = Namespace(run_dir=run)
            a,b = entry.read_collection(args,prepared)
            self.assertEqual(a,train); self.assertEqual(b,val)
            out = run/'fit'; out.mkdir()
            entry.fit(args,prepared,out); entry.finish(out,'fit',contract)
            entry.completed(out,'fit',contract)
            p = entry.read(out/'policy.json')
            self.assertEqual(p['front_end'],frontend())
            self.assertFalse(p['reference_branch_used'])
            self.assertEqual(len(entry.rows(out/'val_decisions.jsonl')),887)
            report = entry.read(out/'fit_report.json')
            self.assertIn('sequence:real_seq07',report['VAL_calibration_descriptive'])
            self.assertEqual(sum(m['train_rows'] for m in report['models'].values()),5116)
            (collect/'final_rows.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError): entry.read_collection(args,prepared)

    def test_no_overwrite_and_no_TEST_route(self):
        text = inspect.getsource(entry.main)
        self.assertIn('out.exists()',text)
        self.assertEqual(set(entry.STATUSES),{'check','collect','fit','verify'})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'policy.json'
            entry.base.write_new(path,{'old':1})
            with self.assertRaises(FileExistsError): entry.base.write_new(path,{'new':2})
            self.assertEqual(entry.read(path),{'old':1})


if __name__ == '__main__': unittest.main()
