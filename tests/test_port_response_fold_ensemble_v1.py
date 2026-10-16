import copy
import inspect
import unittest
from unittest import mock
import numpy as np
from crane_project.utils import port_response_fold_ensemble_v1 as core
from crane_project.utils import port_reliability_response_stats_v1 as response
from crane_project.utils import port_simple_component_reliability_v1 as simple


def fixtures():
    network = dict(normalizer=dict(mean=[0.]*19, scale=[1.]*19), weights={
        'network.0.weight': [[0.]*19 for _ in range(8)], 'network.0.bias': [0.]*8,
        'network.2.weight': [[0.]*8], 'network.2.bias': [0.]})
    meta = dict(normalizer=dict(mean=[0.]*5, scale=[1.]*5), weights=[.4, 1., 0., 0., 0., 0.])
    folds = {}
    for s, value in zip(core.SEQUENCES, [-8., -2., 0., 1., 3., 4.]):
        models = {n: copy.deepcopy(network) for n in core.NAMES[1:]}
        models['formal'] = dict(mean=[0.]*3, scale=[1.]*3, weights=[value, 0., 0., 0.])
        folds[s] = dict(models=models, mapping=dict(mean=[0.]*5, scale=[1.]*5))
    return meta, folds


class EnsembleTest(unittest.TestCase):
    def test_logit_average_not_probability_average_and_all_folds(self):
        meta, folds = fixtures(); raw = np.ones((2, 19))
        risks, proof = core.ensemble(raw, folds, meta)
        expected = simple.sigmoid(.4+np.mean([-8., -2., 0., 1., 3., 4.]))
        np.testing.assert_allclose(risks['fold_fusion'], expected, rtol=0, atol=1e-15)
        self.assertGreater(abs(float(expected)-float(simple.sigmoid(proof['fold_fusion_logits']).mean())), .01)
        np.testing.assert_array_equal(risks['fold_simple'], core.fold_simple(raw, folds))
        del folds[core.SEQUENCES[-1]]
        with self.assertRaises(ValueError):
            core.ensemble(raw, folds, meta)

    def test_each_fold_mapping_and_meta_mapping_are_used(self):
        meta, folds = fixtures(); raw = np.zeros((3, 19))
        folds['real_seq01']['mapping'] = dict(mean=[1.]*5, scale=[2.]*5)
        meta['normalizer'] = dict(mean=[.5]*5, scale=[3.]*5)
        values, proof = core.ensemble(raw, folds, meta)
        expected = []
        for j, s in enumerate(core.SEQUENCES):
            z = proof['fold_base_logits'][j, 0, 0]
            mapped = (z-folds[s]['mapping']['mean'][0])/folds[s]['mapping']['scale'][0]
            expected.append(.4+(mapped-.5)/3)
        np.testing.assert_allclose(values['fold_fusion'], simple.sigmoid(np.mean(expected)), rtol=0, atol=1e-15)

    def test_TRAIN_uses_only_saved_held_scores(self):
        meta, folds = fixtures()
        z = np.arange(15, dtype=float).reshape(3, 5); mapped = z/10
        with mock.patch.object(core, 'ensemble', side_effect=AssertionError('Do not average TRAIN')):
            value = core.held_training(z, mapped, meta)
        np.testing.assert_allclose(value['fold_simple'], simple.sigmoid(z[:, 0]), rtol=0, atol=1e-15)
        np.testing.assert_allclose(value['fold_fusion'], simple.sigmoid(.4+mapped[:, 0]), rtol=0, atol=1e-15)

    def test_online_contract_and_component_preservation(self):
        self.assertEqual(list(inspect.signature(core.ensemble).parameters), ['raw', 'folds', 'meta'])
        meta, folds = fixtures()
        with self.assertRaises(ValueError):
            core.ensemble(np.zeros((1, 20)), folds, meta)
        with self.assertRaises(ValueError):
            core.ensemble(np.full((1, 19), np.nan), folds, meta)
        old = dict(final_box_original=[1, 2, 10, 5, 0, .5], center_accepted=True, angle_accepted=False, size_accepted=True)
        value = response.size_decision(old, .9, .5)
        self.assertFalse(value['size_accepted']); value['size_accepted'] = True
        self.assertEqual(value, old)


if __name__ == '__main__':
    unittest.main()
