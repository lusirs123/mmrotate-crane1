"""Focused CPU/stdlib checks; no datasets, GPU, inference cache or fitting."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest
import torch

from crane_project.tools import run_detector_module_ablation_v1 as runner
from crane_project.tools import train_symeood_110_midpoint_v1 as original_m
from crane_project.tools import run_symeood_110_lvsr_v1 as original_l
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
from crane_project.utils import port_size_qalign_v1 as vector

torch.set_num_threads(1)


def fixture(empty=False):
    b = torch.tensor([[128., 128., 100., 60., .2, .9]])
    row = dict(image='real_seq01_00001', role='train', split='train', scale=1.,
        domain='real', sequence='real_seq01', frame_id=1, eligible=True,
        roi=torch.randn(1, 256, 9, 9), support=torch.ones(1, 1, 9, 9),
        boxes_original=b, boxes_model=b[:, :5].clone(), scale_xy=torch.ones(1, 2),
        gt_original=b[:, :5].clone())
    if empty:
        for k in ('roi', 'support', 'boxes_original', 'boxes_model', 'scale_xy'):
            row[k] = row[k][:0]
    return row


def test_default_check_is_stdlib_only_readonly_subprocess(tmp_path):
    code = """
import sys
from unittest.mock import patch
from crane_project.tools import run_detector_module_ablation_v1 as r
with patch.object(r,'runtime_imports', side_effect=AssertionError('runtime called')):
    r.main(r.parser().parse_args([]))
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
"""
    output = subprocess.check_output(['/usr/bin/python3', '-c', code], cwd=str(runner.ROOT), text=True)
    result = json.loads(output)
    assert not result['training_started'] and not result['cache_created']
    out = tmp_path/'unused'
    with patch.object(runner, 'output_root', return_value=out):
        with patch.object(runner, 'runtime_imports', side_effect=AssertionError('runtime called')):
            runner.main(runner.parser().parse_args([]))
    assert not out.exists()


@pytest.mark.parametrize('stage', ['cache', 'midpoint', 'lvsr'])
def test_runtime_requires_explicit_execute_before_import(stage):
    with patch.object(runner, 'runtime_imports', side_effect=AssertionError('runtime called')):
        with pytest.raises(ValueError, match='explicit --execute'):
            runner.main(runner.parser().parse_args(['--stage', stage]))


def test_contract_reuses_stage_specific_original_hyperparameters():
    p = runner.checked_contract()
    assert set(p['arms']) == set(runner.ARMS)
    assert p['midpoint']['lr'] == original_m.SETTINGS['lr']
    assert p['midpoint']['batch_size'] == original_m.SETTINGS['batch_size']
    assert p['midpoint']['train_center_match_px'] == original_m.SETTINGS['train_center_match_px']
    for stage in ('midpoint', 'lvsr'):
        assert p[stage]['seed'] == 1703 and p[stage]['epochs'] == 24
        assert p[stage]['adam'] == dict(betas=[.9, .999], eps=1e-8)
    assert p['arms']['eood_ml']['midpoint']
    assert not p['arms']['110_l']['midpoint']
    assert p['frontends']['eood']['epoch'] == 24
    assert 'aug_b' in p['frontends']['eood']['checkpoint']


def test_source_or_setting_drift_rejected(tmp_path):
    p = runner.checked_contract()
    p['lvsr']['lr'] = .0001
    path = tmp_path/'changed.json'
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError, match='hyperparameters'):
        runner.checked_contract(path)
    p = runner.read(runner.CONTRACT)
    p['arms']['110_l']['midpoint'] = True
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError, match='protocol'):
        runner.checked_contract(path)
    with patch.object(runner, 'sha', return_value='0'*64):
        with pytest.raises(ValueError, match='Source identity'):
            runner.checked_contract()


def test_missing_arm_and_frontend_cache_binding():
    p = runner.checked_contract()
    assert runner.cache_path(p, 'eood') == runner.output_root(p)/'eood_cache'
    assert runner.cache_path(p, '110') == runner.ROOT/p['frontends']['110']['cache']
    with pytest.raises(ValueError):
        runner.arm_spec(p, '111_l')
    assert 'test' not in runner.parser()._option_string_actions['--stage'].choices


def test_wrong_stage_combination_stops_before_runtime():
    with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0', 'WORLD_SIZE': '1'}):
        with patch.object(runner, 'runtime_imports', side_effect=AssertionError('runtime called')):
            with pytest.raises(ValueError, match='has no LVSR'):
                runner.main(runner.parser().parse_args(['--stage', 'lvsr', '--arm', 'eood_m', '--execute']))
            with pytest.raises(ValueError, match='reuses the sealed'):
                runner.main(runner.parser().parse_args(['--stage', 'cache', '--arm', '110_l', '--execute']))


def test_top1_preserves_native_scores_no_added_threshold_and_stable_ties():
    a = np.array([[1, 2, 3, 4, .1, .01], [5, 6, 7, 8, .2, .04], [9, 0, 1, 2, .3, .04]], dtype=np.float32)
    saved = a.copy()
    assert np.array_equal(runner.top1([[a]], np)[0][0], a[1:2])
    assert np.array_equal(a, saved)
    assert runner.top1([[a[:0]]], np)[0][0].shape == (0, 6)
    bad = a.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        runner.top1([[bad]], np)


def test_eood_adapter_matches_native_single_stage_call_chain():
    # Test only the changed adapter, without loading MMRotate CUDA extensions.
    class Head:
        num_classes = 1
        def __call__(self, features):
            assert features == ('same_P3',)
            return ('scores', 'boxes')
        def get_bboxes(self, a, b, metas, rescale=False):
            assert (a, b, metas) == ('scores', 'boxes', ['same_meta'])
            return [(np.array([[1, 2, 3, 4, .1, .6], [5, 6, 7, 8, .2, .8]]), [0, 0])]
    module = types.ModuleType('mmrotate.core')
    module.rbbox2result = lambda b, labels, count: [b]
    model = types.SimpleNamespace(bbox_head=Head())
    with patch.dict(sys.modules, {'mmrotate.core': module}):
        adapter = runner.NativeFrontendAdapter(model, 'eood', np)
        result = adapter.simple_test_from_features(('same_P3',), ['same_meta'], rescale=True)
    assert result[0][0].shape == (1, 6)
    assert result[0][0][0, 5] == .8


@pytest.mark.parametrize('empty', [False, True])
def test_l_only_q_equals_b_without_midpoint_and_without_source_mutation(empty):
    source = fixture(empty)
    before = {k: v.clone() for k, v in source.items() if isinstance(v, torch.Tensor)}
    out = runner.attach_reference([source], None, 'cpu', torch)[0]
    assert torch.equal(out['midpoint_original'], source['boxes_original'])
    assert 'midpoint_original' not in source
    for k, v in before.items():
        assert torch.equal(source[k], v)
    assert out['midpoint_original'].data_ptr() != source['boxes_original'].data_ptr() or empty
    x = [out[k] for k in ('roi', 'support', 'boxes_original', 'boxes_model', 'scale_xy', 'midpoint_original')]
    h = PairSpecificSizeHead()
    result = h(*x)
    assert torch.equal(result['boxes_original'], source['boxes_original'])
    assert sum(p.numel() for p in h.parameters()) == 2480
    assert not any(p.grad is not None for p in h.parameters())


def test_m_reference_is_used_but_not_updated_or_replaced_by_gt():
    source = fixture()
    class FrozenM:
        def __call__(self, *x):
            assert len(x) == 5  # no GT enters forward
            q = x[2].clone()
            q[:, :2] += .5
            q[:, 2:4] *= torch.tensor([[.99, 1.01]])
            return dict(boxes_original=q)
    out = runner.attach_reference([source], FrozenM(), 'cpu', torch)[0]
    assert not torch.equal(out['midpoint_original'], source['boxes_original'])
    assert out['midpoint_original'][0, 5] == source['boxes_original'][0, 5]
    x = [out[k] for k in ('roi', 'support', 'boxes_original', 'boxes_model', 'scale_xy', 'midpoint_original')]
    assert torch.equal(PairSpecificSizeHead()(*x)['boxes_original'], out['midpoint_original'])


def test_l_only_uses_original_size_loss_and_frozen_features_gradients_no_update():
    source = fixture()
    source['roi'].requires_grad_()
    row = runner.attach_reference([source], None, 'cpu', torch)[0]
    x = [row[k] for k in ('roi', 'support', 'boxes_original', 'boxes_model', 'scale_xy', 'midpoint_original')]
    h = PairSpecificSizeHead()
    before = deepcopy(h.state_dict())
    target = row['gt_original'].clone()
    target[:, 2:4] *= torch.tensor([[.97, 1.02]])
    output = h(*x)
    loss = vector.size_loss(output, target, x[-1], x[3])
    assert torch.equal(loss, vector.residual.size_loss(output['delta_roi'], target, x[-1], x[3]))
    loss.backward()
    assert source['roi'].grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters())
    assert all(torch.equal(v, h.state_dict()[k]) for k, v in before.items())


def test_save_fresh_head_optimizer_identity_rng_no_overwrite(tmp_path):
    original_m.runtime_imports()
    h = PairSpecificSizeHead()
    opt = torch.optim.Adam(h.parameters(), lr=.001)
    identity = dict(frontend='eood', reference='B', detector_sha256='frozen')
    path = tmp_path/'head_epoch_01.pth'
    saved = runner.save_head(path, h, opt, 1, 0, identity, original_m, torch)
    payload = torch.load(str(path), map_location='cpu')
    assert payload['identity'] == identity and payload['updates'] == 0
    assert payload['parent_checkpoint'] is None and not payload['optimizer_state']['state']
    clone = PairSpecificSizeHead()
    clone.load_state_dict(payload['head_state'], strict=True)
    assert all(torch.equal(v, clone.state_dict()[k]) for k, v in h.state_dict().items())
    assert saved['sha256'] == runner.sha(path)
    with pytest.raises(FileExistsError):
        runner.save_head(path, h, opt, 1, 0, identity, original_m, torch)


def summary(joint=100, long=.04):
    return dict(overall=dict(edge_residual=dict(sizes=dict(joint10=joint))),
        **{d: dict(edge_residual=dict(sizes=dict(edges=dict(long=dict(mae=long), short=dict(mae=.05)))))
           for d in ('real', 'sim')})


def test_joint_first_common_selection_all_epochs_and_ties():
    events = [dict(epoch=e, checkpoint=dict(epoch=e), selection_key=runner.lvsr_selection_key(s, e))
              for e, s in ((1, summary(100, .01)), (2, summary(101, .06)), (3, summary(101, .06)))]
    out = runner.select_lvsr(events)
    assert out['selected_epoch'] == 2
    assert out['fixed_epoch24'] == 'head_epoch_24.pth'
    assert out['original_adoption_gate_passed'] is None


def test_existing_lvsr_order_reused_and_no_fixed_639_budget():
    assert original_l.schedule_indices(17, 2) == original_l.schedule_indices(17, 2)
    assert sorted(original_l.schedule_indices(17, 2)) == list(range(17))
    assert original_l.schedule_indices(17, 2) != original_l.schedule_indices(17, 3)
    source = Path(runner.__file__).read_text()
    assert '639' not in source
    assert 'base.update(h, opt, base.batch(rr, torch)' in source
    assert 'f.update(h, opt, train, indices' in source


def test_source_writes_and_training_are_not_performed_on_import():
    # An existing smoke receipt is allowed; importing never creates a fit.
    assert not (runner.output_root(runner.checked_contract())/'eood_m').exists()


def test_smoke_is_bounded_and_never_selects_test_or_by_gt(tmp_path):
    records = [dict(id=d+'_seq01_'+str(i), sequence=d+'_seq01', split=s)
        for s in ('train', 'val', 'test') for d in ('real', 'sim') for i in range(5)]
    (tmp_path/'manifest.json').write_text(json.dumps(dict(records=records)))
    f = types.SimpleNamespace(g=types.SimpleNamespace(ready=types.SimpleNamespace(DATA=tmp_path)))
    selected = runner.smoke_views(f)
    assert len(selected) == 12
    assert sum(role == 'train' for _, role, _ in selected) == 8
    assert sum(role == 'val' for _, role, _ in selected) == 4
    assert not any(r['split'] == 'test' for r, _, _ in selected)
