import inspect
import math
import unittest
from copy import deepcopy
import numpy as np
from crane_project.utils import port_reliability_temporal_fusion_v1 as core
from test_port_reliability_temporal_evidence_v1 import row


def toy():
    return core.level.enrich([row(i, size=80 if i % 3 == 0 else 100+i/20) for i in range(30)], {})


class FusionTests(unittest.TestCase):
    def test_paired_training_and_all_output_fallback(self):
        rows = toy(); models = core.fit(rows)
        self.assertEqual(models['static3']['train_images'], models['fusion4']['train_images'])
        self.assertEqual(models['fallback3']['train_rows'], 30)
        self.assertEqual(models['fusion4']['train_rows'], 27)
        self.assertGreater(models['fusion4']['bad_rows'], 0)

    def test_three_dimensional_matches_existing_solver(self):
        rows = toy(); model = core.fit(rows)['static3']
        part = [r for r in rows if r['temporal']['available']]
        x = [core.features(r['pred'],r['image_size'],r['temporal'],3) for r in part]
        reference = core.simple.fit_linear_risk(x, [int(core.level.metrics.bad(r)) for r in part], core.FITTING)
        np.testing.assert_allclose(model['weights'], reference['weights'], atol=1e-12)

    def test_fit_rejects_VAL_and_TEST(self):
        for split in ('val','test'):
            rows = toy(); rows[0]['split'] = split; rows[0]['reliability_role'] = split
            with self.assertRaises(ValueError): core.fit(rows)

    def test_both_labels_required(self):
        with self.assertRaises(ValueError): core.fit_matrix([[1,2,3],[4,5,6]],[0,0],3)

    def test_shared_fallback_without_fake_temporal_value(self):
        rows = toy(); models = core.fit(rows)
        r = rows[0]; risks = core.online_risks(r['pred'], r['image_size'], r['temporal'], models)
        self.assertEqual(risks['static3'], risks['fusion4'])
        with self.assertRaises(ValueError): core.features(r['pred'],r['image_size'],r['temporal'],4)

    def test_GT_and_future_do_not_change_scores(self):
        raw = [row(i, size=80 if i % 3 == 0 else 100) for i in range(30)]
        enriched = core.level.enrich(raw, {}); models = core.fit(enriched)
        expected = core.score(enriched, models)
        changed = deepcopy(raw)
        for r in changed: r['gt'] = [0,0,999,333,0]
        changed[-1]['pred'][2] = 777
        actual = core.score(core.level.enrich(changed, {}), models)
        self.assertEqual([r['risks'] for r in expected[:-1]], [r['risks'] for r in actual[:-1]])
        self.assertNotIn('gt', inspect.signature(core.CausalFusion.decide).parameters)
        self.assertNotIn('gt', inspect.signature(core.online_risks).parameters)

    def test_missing_resets_history_and_has_no_risk(self):
        raw = [row(i) for i in range(8)]; raw[4] = row(4,None)
        rows = core.level.enrich(raw, {})
        models = core.fit(toy()); scored = core.score(rows, models)
        self.assertIsNone(scored[4]['risks']['fusion4'])
        self.assertFalse(scored[7]['temporal']['available'])

    def test_isotropic_swap_and_pi_invariance(self):
        rows = toy(); models = core.fit(rows); original = core.score(rows, models)
        raw = deepcopy(rows)
        for r in raw:
            r['pred'][2:4] = [2*r['pred'][3], 2*r['pred'][2]]
            r['pred'][4] += math.pi/2+math.pi
            r['image_size'] = [2000,2000]
        altered = core.score(core.level.enrich(raw, {}), models)
        np.testing.assert_allclose([r['risks']['fusion4'] for r in original], [r['risks']['fusion4'] for r in altered], atol=1e-12)

    def test_online_preserves_other_components_and_output(self):
        models = core.fit(toy())
        import json
        from pathlib import Path
        path = Path(__file__).resolve().parents[1]/'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit/policy.json'
        policy = json.loads(path.read_text())['simple_policy']
        api = core.CausalFusion(models,.5,policy); original = core.simple.SimpleComponentReliability(policy)
        for i in range(6):
            pred = None if i == 4 else [0,0,100,50,0,.8]
            before = deepcopy(pred); new = api.decide(pred,[1000,1000],i); old = original.decide(pred,[1000,1000])
            for k in ('raw_b_output','center_accepted','angle_accepted','risks'): self.assertEqual(new[k],old[k])
            self.assertEqual(pred,before)
            if pred is None: self.assertFalse(new['size_accepted'])

    def test_static_refit_not_original_simple_veto(self):
        rows = toy(); models = core.fit(rows)
        a = core.score(rows,models)
        for r in rows: r['original_simple_decision']['size_accepted'] = False
        self.assertEqual([r['risks'] for r in a], [r['risks'] for r in core.score(rows,models)])

    def test_VAL_global_whole_tie_and_group_protection(self):
        rows = toy()
        for r in rows:
            r['split']='val';r['reliability_role']='val';r['risks']['fusion4']=.8 if r['frame_id']==29 else .2
        cutoff = core.select_cutoff(rows,'fusion4')
        for part in core.level.metrics.grouped(rows).values():
            good = [r for r in part if core.level.metrics.bad(r) is False]
            kept = [r for r in good if r['risks']['fusion4']<=cutoff['risk_le']]
            self.assertGreaterEqual(len(kept),math.ceil(.95*len(good)))
        rows[0]['split']='train';rows[0]['reliability_role']='train'
        with self.assertRaises(ValueError):core.select_cutoff(rows,'fusion4')

    def test_evaluate_missing_and_old_shared_target(self):
        rows = toy(); rows = core.score(rows, core.fit(rows))
        cutoffs = {a:dict(risk_le=.8) for a in core.ARMS}
        result = core.evaluate(rows,cutoffs)
        candidate = result['groups']['all']['candidate']
        self.assertEqual(sum(candidate['states'].values()),len(rows))
        self.assertIn('passed',core.gate(result))

    def test_score_does_not_mutate_rows(self):
        rows = toy(); saved = deepcopy(rows); core.score(rows,core.fit(rows))
        self.assertEqual(rows,saved)

    def test_independent_old_match_helper_signature_and_selection(self):
        from crane_project.tools.review_port_reliability_temporal_fusion_v1 import matched_veto_ids
        rows = core.score(toy(),core.fit(toy()))
        ids = matched_veto_ids(rows,5,'CR')
        expected = core.trend.matched_veto(rows,correct=5)
        self.assertEqual(core.level.summarize(rows,ids)['states'],expected['states'])
        self.assertIsNone(matched_veto_ids(rows,len(rows)+1,'count'))

    def test_review_gate_survives_sorted_JSON_roundtrip(self):
        import json
        from crane_project.tools.review_port_reliability_temporal_fusion_v1 import equivalent_gate
        rows=core.score(toy(),core.fit(toy()))
        report=core.evaluate(rows,{a:dict(risk_le=.8) for a in core.ARMS})
        before=core.gate(report)
        after=core.gate(json.loads(json.dumps(report,sort_keys=True)))
        self.assertTrue(equivalent_gate(before,after))
        broken=deepcopy(after);broken['failures'].append(dict(check='unexpected'))
        self.assertFalse(equivalent_gate(before,broken))


if __name__ == '__main__': unittest.main()
