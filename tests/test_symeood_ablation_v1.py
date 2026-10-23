"""Static checks for the S4x three-factor ablation (no mmcv required)."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'crane_project/configs'


def _src(p):
    return p.read_text(encoding='utf-8')


def test_switches_default_to_historical_behaviour():
    text_pola = _src(ROOT / 'mmrotate/core/bbox/assigners/sym_pola.py')
    text_nfl = _src(ROOT / 'mmrotate/models/losses/sym_nfl_loss.py')
    assert "cost_reg_mode: str = 'kld'" in text_pola
    assert 'use_geometry_weight=True' in text_nfl
    # the source must be syntactically valid after the surgical edit
    ast.parse(text_pola)
    ast.parse(text_nfl)


def test_group_files_exist_and_declare_their_switches():
    for g in ('000', '100', '010', '001', '110', '101', '011', '111'):
        body = _src(CFG / f'crane_symeood_k1_ablation_{g}_v1.py')
        assert f"work_dirs/symeood_ablation_v1/{g}" in body
        assert body.count("cost_reg_mode='iou'") == (1 if g[0] == '0' else 0)
        assert body.count('use_geometry_weight=False') == (1 if g[2] == '0' else 0)
        assert body.count("type='RotatedIoULoss'") == (1 if g[1] == '0' else 0)
        assert body.count("type='L1Loss'") == 0


def test_111_adds_no_override():
    body = _src(CFG / 'crane_symeood_k1_ablation_111_v1.py')
    assert 'model = dict(' not in body


def test_no_config_touches_budget_or_selection():
    for p in CFG.glob('crane_symeood_k1_ablation_*_v1.py'):
        body = _src(p)
        for forbidden in ('lr=', 'max_epochs', 'step=[', 'evaluation', 'interval=',
                          'loss_weight=0.25', 'alpha=', 'gamma='):
            if p.name.endswith('base.py'):
                continue
            assert forbidden not in body, (p.name, forbidden)


def test_c_off_takes_precedence_over_isolate_images():
    body = _src(ROOT / 'mmrotate/models/losses/sym_nfl_loss.py')
    off = body.index('if not self.use_geometry_weight:')
    iso = body.index('elif self.isolate_images:')
    geo = body.index('elif gt_bboxes is not None')
    assert off < iso < geo
    assert 'mu_sym = pred_logits.new_ones' in body


def test_b_groups_are_111_vs_101():
    a = _src(CFG / 'crane_symeood_k1_ablation_111_v1.py')
    b = _src(CFG / 'crane_symeood_k1_ablation_101_v1.py')
    assert 'model = dict(' not in a
    assert "type='RotatedIoULoss'" in b
    assert "type='L1Loss'" not in b


def test_helper_semantics_without_mmcv():
    torch = __import__('pytest').importorskip('torch')
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'chk', str(ROOT / 'crane_project/tools/check_symeood_ablation_v1.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    a = {'p': torch.zeros(2), 'q': torch.zeros(2), 'r': None}
    b = {'p': torch.zeros(2), 'q': torch.ones(2), 'r': None}
    mism, both_none = m.compare_grads(a, b, torch)
    assert both_none == 1 and mism == ['q']
    assert m.grad_activity({'x': None, 'y': torch.zeros(3), 'z': torch.ones(3)}) == (2, 1)
    assert m.grad_activity({'x': None, 'y': torch.zeros(3)})[1] == 0
    assert m.flatten({'a': {'b': 1}, 'c': 2}) == {'a.b': 1, 'c': 2}
