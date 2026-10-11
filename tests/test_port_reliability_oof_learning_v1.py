from copy import deepcopy
from types import SimpleNamespace
import unittest

from crane_project.tools import diagnose_port_reliability_oof_learning_v1 as diag


class LearningContractTests(unittest.TestCase):
    def test_selection_cannot_depend_on_prediction_quality_or_excluded_identity(self):
        rows=[dict(image=f'{d}_{seq}_{i}',domain=d,sequence=f'{d}_{seq}',pred=[i])
              for d in ('real','sim') for seq in ('a','b') for i in range(5)]
        allowed={r['image'] for r in rows if r['sequence'].endswith('b')}
        expected=['real_b_2','sim_b_2']
        self.assertEqual(diag.selected_names(rows,allowed),expected)
        for r in rows:r['pred']=None
        self.assertEqual(diag.selected_names(rows[::-1],allowed),expected)

    def test_disabling_augmentation_preserves_source_and_geometry_pipeline(self):
        cfg=SimpleNamespace(data=SimpleNamespace(train=[dict(pipeline=[dict(type='RResize'),
            dict(type='PortIsotropicShrink',prob=.5),dict(type='RRandomFlip',flip_ratio=[.25]*3),
            dict(type='Normalize'),dict(type='Pad')])]))
        before=deepcopy(cfg.data.train)
        result=diag.deterministic_specs(cfg)
        self.assertEqual(cfg.data.train,before)
        self.assertEqual([o['type'] for o in result[0]['pipeline']], [o['type'] for o in before[0]['pipeline']])
        self.assertEqual(result[0]['pipeline'][1]['prob'],0.)
        self.assertEqual(result[0]['pipeline'][2]['flip_ratio'],0.)

    def test_lr_matches_original_linear_warmup_not_constant_full_lr(self):
        cfg=SimpleNamespace(lr_config=dict(warmup='linear',warmup_iters=1000,warmup_ratio=.001),
                            optimizer=SimpleNamespace(lr=.0025))
        self.assertAlmostEqual(diag.learning_rate(cfg,0),.0000025)
        self.assertAlmostEqual(diag.learning_rate(cfg,500),.0025*.5005)
        self.assertAlmostEqual(diag.learning_rate(cfg,1000),.0025)
        cfg.lr_config['warmup']='exp'
        with self.assertRaises(ValueError):diag.learning_rate(cfg,0)


if __name__=='__main__':unittest.main()
