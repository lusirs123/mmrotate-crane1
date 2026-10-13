import inspect
import json
import math
import gzip
import tempfile
from unittest.mock import patch
import unittest
from copy import deepcopy
from pathlib import Path
import numpy as np
from crane_project.utils import port_reliability_fallback_decision_v1 as c
from test_port_reliability_image_temporal_crossfit_v1 import fixture
from test_port_reliability_temporal_fusion_v1 import toy


class Tests(unittest.TestCase):
    def test_branch_cutoff_and_inclusive_boundary(self):
        self.assertTrue(c.decide_size(.5, True, .5, .4))
        self.assertFalse(c.decide_size(.5, False, .5, .4))
        self.assertTrue(c.decide_size(.4, False, .5, .4))
        self.assertTrue(c.decide_size(.4, True, .3, .5) is False)

    def test_missing_and_invalid_risk(self):
        self.assertFalse(c.decide_size(None, False, .5, .4))
        for x in (float('nan'), float('inf'), -.1, 1.1):
            with self.assertRaises(ValueError): c.decide_size(x, False, .5, .4)
        for x in (float('nan'), 1.1, -.1):
            with self.assertRaises(ValueError): c.decide_size(None, False, .5, x)
        with self.assertRaises(ValueError): c.decide_size(.3, 1, .5, .4)

    def rows(self):
        rows = fixture()
        for r in rows:
            r['risks'] = dict(partial_auc4=.45, crossfit_fusion=.45, in_sample_fusion=.45, fusion4=.5, full_simple=.6, score_only=.7)
        return rows

    def test_no_GT_read_and_no_mutation(self):
        rows = self.rows(); saved = deepcopy(rows); scored = c.score(rows, .5, .4)
        self.assertEqual(rows, saved)
        for r in rows: r.pop('gt')
        other = c.score(rows, .5, .4)
        self.assertEqual([r[c.METHOD] for r in other], [r[c.METHOD] for r in scored])
        for r, old in zip(scored, saved):
            self.assertEqual({k:v for k,v in r.items() if k != c.METHOD}, old)

    def test_supported_unchanged_and_fallback_exact(self):
        rows = self.rows(); scored = c.score(rows, .5, .4)
        for r in scored:
            self.assertEqual(r[c.METHOD], r['temporal']['available'])

    def test_equal_cutoffs_reproduce_old_all_flags(self):
        for r in c.score(self.rows(), .5, .5): self.assertTrue(r[c.METHOD])

    def test_mismatched_fallback_rejected(self):
        rows = self.rows(); rows[0]['temporal']['available'] = False
        rows[0]['risks']['partial_auc4'] = .3
        with self.assertRaises(ValueError): c.score(rows, .5, .4)

    def test_missing_row_never_accepted(self):
        rows = self.rows()[:1]; rows[0]['pred'] = None
        rows[0]['temporal']['available'] = False
        rows[0]['risks']['partial_auc4'] = rows[0]['risks']['crossfit_fusion'] = None
        self.assertFalse(c.score(rows, .5, .4)[0][c.METHOD])

    def test_online_identity_gaps_reset_and_resolution(self):
        prior = c.prior; base = prior.temporal.base.fit(toy()); partial = base['fusion4']
        pca = dict(mean=np.zeros(2304).tolist(), components=np.eye(256,2304).tolist())
        head = {'network.0.weight':np.zeros((16,258)).tolist(),'network.0.bias':np.zeros(16).tolist(),
            'network.2.weight':np.zeros((8,16)).tolist(),'network.2.bias':np.zeros(8).tolist(),
            'network.4.weight':np.zeros((1,8)).tolist(),'network.4.bias':[0.],'beta':[0.]}
        norm = dict(mean=np.zeros(259).tolist(),scale=np.ones(259).tolist())
        fusion = dict(weights=[0,0,0],normalizer=dict(mean=[0,0],scale=[1,1]))
        policy = json.loads((Path(__file__).resolve().parents[1]/'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit/policy.json').read_text())['simple_policy']
        args = (base,partial,pca,head,norm,fusion,.8,policy)
        new = c.CausalFallbackDecision(*args,fallback_cutoff=.2)
        old = prior.CausalImageTemporal(*args)
        for frame,pred,size,reset in [(i,[0,0,100,50,0,.8],[1000,1000],False) for i in range(5)]+[(6,None,[1000,1000],False),(7,[0,0,100,50,0,.8],[1000,1000],False),(8,[0,0,100,50,0,.8],[800,800],False),(0,[0,0,100,50,0,.8],[800,800],True)]:
            saved = deepcopy(pred); a = new.decide(pred,size,frame,np.zeros(2304),reset); b = old.decide(pred,size,frame,np.zeros(2304),reset)
            self.assertEqual(pred,saved)
            for key,value in b.items():
                if key != 'size_accepted': self.assertEqual(a[key],value)
            supported = b['size_image_temporal_history_available'];risk = b['size_image_temporal_risk']
            self.assertEqual(a['size_accepted'],risk is not None and risk <= (.8 if supported else .2))
        self.assertNotIn('gt',inspect.signature(new.decide).parameters)

    def test_evaluation_and_gate_contract(self):
        rows = c.score(self.rows(), .5, .4)
        result = c.evaluate(rows, .5, .4)
        self.assertIn('in_sample_fusion', result['groups']['all']['controls'])
        self.assertIn('passed', c.gate(result))
        self.assertEqual(result['groups']['all']['candidate']['states']['MISSING'], 0)

    def test_independent_review_serialization_and_tamper_detection(self):
        from crane_project.tools import review_port_reliability_fallback_decision_v1 as audit
        rows = self.rows(); scored = c.score(rows, .5, .4)
        roles = {role:c.evaluate(scored, .5, .4) for role in ('TRAIN', 'VAL')}
        gate = c.gate(roles['VAL'])
        report = dict(protocol=c.VERSION, status='EVIDENCE_PASS_NO_ADOPTION' if gate['passed'] else 'EVIDENCE_FAILED_STOP',
            roles=roles, gate=gate, input_pins={}, frozen_cutoffs=dict(supported=.5,fallback=.4),
            VAL_previously_exposed=True, design_informed_by_exposed_VAL=True,
            fits=0, quality_updates=0, calibration_calls=0, detector_inferences=0, detector_updates=0,
            TEST_opened=False, formal_policy_changed=False, candidate_adopted=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source'; source.mkdir(); result = root/'result'; result.mkdir()
            for role in ('TRAIN','VAL'):
                for path,values in ((source,rows),(result,scored)):
                    with gzip.open(path/('scored_'+role+'.jsonl.gz'),'wt') as f:
                        for r in values: f.write(json.dumps(r)+'\n')
            save = lambda: (result/'report.json').write_text(json.dumps(report,sort_keys=True))
            save()
            with patch.object(audit,'ROOT',root), patch.object(audit.runner,'load',return_value=(dict(crossfit_result='source'),{},[],dict(supported=.5,fallback=.4))):
                self.assertTrue(audit.review(result)['passed'])
                report['roles']['VAL']['groups']['all']['candidate']['states']['FA'] += 1; save()
                with self.assertRaises(ValueError): audit.review(result)

    def test_frozen_contract_check(self):
        from crane_project.tools import run_port_reliability_fallback_decision_v1 as runner
        p,pins,absent,cuts = runner.load(True)
        self.assertEqual(cuts,p['frozen_cutoffs'])
        self.assertTrue(pins)


if __name__ == '__main__': unittest.main()
