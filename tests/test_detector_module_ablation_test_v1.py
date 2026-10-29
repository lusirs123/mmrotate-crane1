"""Only new fixed-head identities, reference routing and safeguards; CPU only."""
from copy import deepcopy
import inspect
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest
import torch

from crane_project.tools import eval_detector_module_ablation_test_v1 as e
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead

torch.set_num_threads(1)


def spec():
    return dict(selected_checkpoint=dict(epoch=17, updates=10880), identity=dict(arm='eood_l', reference='B'))


def test_only_frozen_original_VAL_selected_head():
    s = spec()
    p = dict(protocol='detector_module_ablation_v1', epoch=17, updates=10880,
        identity=s['identity'], parent_checkpoint=None)
    e.validate_payload(p, s)
    for changes in (dict(epoch=24), dict(updates=0), dict(identity=dict(arm='eood_ml')),
                    dict(parent_checkpoint='continued'), dict(protocol='old_shared')):
        with pytest.raises(ValueError):
            e.validate_payload(dict(p, **changes), s)


def test_box_comparison_keeps_presence_score_and_geometry():
    a = dict(b=[1, 2, 100, 60, .1, .9])
    assert e.compare_boxes(a, a, ['b'], np) == dict(b=0.)
    for b in (None, [1, 2, 100, 60, .1, .8], [1, 2, 101, 60, .1, .9]):
        with pytest.raises(ValueError):
            e.compare_boxes(a, dict(b=b), ['b'], np)
    assert e.compare_boxes(dict(b=None), dict(b=None), ['b'], np) == dict(b=0.)


def test_historical_score_roundoff_only_when_explicitly_enabled():
    a = dict(b=[818.7040405273438, 274.3970642089844, 121.99850463867188,
        46.7526741027832, .3391265869140625, .5435750484466553])
    b = dict(b=[818.7040405273438, 274.3970642089844, 121.99848937988281,
        46.7526741027832, .3391265869140625, .5435755848884583])
    with pytest.raises(ValueError):
        e.compare_boxes(a, b, ['b'], np)
    assert e.compare_boxes(a, b, ['b'], np, score_atol=1e-6, score_rtol=1e-6)['b'] < 2e-5
    # The next historical frame also differs by low-micro score roundoff.
    c = dict(b=[850.1607666015625, 460.2402038574219, 120.45423889160156,
        45.117923736572266, .2799566984176636, .47955015301704407])
    d = dict(b=[850.1607666015625, 460.2402038574219, 120.45427703857422,
        45.11792755126953, .2799569368362427, .4795518219470978])
    assert e.compare_boxes(c, d, ['b'], np, score_atol=1e-5, score_rtol=1e-5)['b'] < 4e-5
    for changed in (dict(b=None), dict(b=b['b'][:5]+[.5436]),
            dict(b=b['b'][:2]+[122.1]+b['b'][3:])):
        with pytest.raises(ValueError):
            e.compare_boxes(a, changed, ['b'], np, score_atol=1e-6, score_rtol=1e-6)
        with pytest.raises(ValueError):
            e.compare_boxes(a, changed, ['b'], np, score_atol=1e-5, score_rtol=1e-5)


def test_completed_VAL_gate_requires_identical_model_and_inference(tmp_path):
    import json
    old = dict(heads={'fixed': True}, training_contract_sha256='same', artifacts={},
        sources={'model.py': 'same', str(Path(e.__file__).relative_to(e.ROOT)): 'old'})
    prior = tmp_path/'frozen.json';prior.write_text(json.dumps(old))
    receipt = tmp_path/'gate.json';receipt.write_text(json.dumps(dict(passed=True, frames=887)))
    reuse = dict(frozen_contract=str(prior), receipt=str(receipt),
        files={str(p):e.training.sha(p) for p in (prior, receipt)})
    lock = deepcopy(old);lock['groups']={'eood':dict(reuse_cached_VAL_gate=reuse)}
    lock['sources'][str(Path(e.__file__).relative_to(e.ROOT))]='new'
    assert e.cached_VAL_gate(lock, 'eood', None, None, None)['cached_head_replay_frames_this_attempt'] == 0
    for field in ('heads', 'artifacts', 'sources'):
        bad=deepcopy(lock);bad[field]['unexpected']='changed'
        with pytest.raises(ValueError):
            e.cached_VAL_gate(bad, 'eood', None, None, None)
    receipt.write_text(json.dumps(dict(passed=False, frames=887)))
    with pytest.raises(ValueError):
        e.cached_VAL_gate(lock, 'eood', None, None, None)


def fixture(empty=False):
    b = torch.tensor([[128., 128., 100., 60., .2, .9]])
    x = dict(roi=torch.randn(1, 256, 9, 9), support=torch.ones(1, 1, 9, 9),
        boxes_original=b, boxes_model=b[:, :5].clone(), scale_xy=torch.ones(1, 2),
        gt_original='MUST_NOT_BE_READ')
    return {k: v[:0] if empty and isinstance(v, torch.Tensor) else v for k, v in x.items()}


@pytest.mark.parametrize('empty', [False, True])
def test_110_size_only_path_has_no_midpoint_or_GT(empty):
    x = fixture(empty)
    h = PairSpecificSizeHead().eval().requires_grad_(False)
    with patch.object(torch.Tensor, 'cuda', lambda self: self):
        got = e.forward_heads(x, {'110_l': h}, '110')
    expected = None if empty else x['boxes_original'][0].tolist()
    assert got == {'b': expected, '110_l': expected}
    assert all(p.grad is None for p in h.parameters())
    assert 'gt' not in inspect.signature(e.forward_heads).parameters


def test_eood_single_L_uses_B_and_combined_L_uses_M():
    class M:
        def __call__(self, roi, support, b, bm, xy):
            q = b.clone();q[:, 0] += .5;q[:, 2:4] *= torch.tensor([[1.01, .99]])
            return dict(boxes_original=q)
    heads = dict(eood_m=M(), eood_l=PairSpecificSizeHead(), eood_ml=PairSpecificSizeHead())
    x = fixture()
    with torch.no_grad(), patch.object(torch.Tensor, 'cuda', lambda self: self):
        got = e.forward_heads(x, heads, 'eood')
    assert got['eood_l'] == got['b']
    assert got['eood_ml'] == got['eood_m']
    assert got['eood_m'] != got['b']


def test_reference_pool_matches_frozen_selected_rows():
    lock = e.checked()
    assert len(e.references(lock, 'eood')) == len(e.references(lock, '110')) == 887
    for arm, epoch in [('eood_m', 23), ('eood_l', 17), ('eood_ml', 17), ('110_l', 2)]:
        assert lock['heads'][arm]['selected_checkpoint']['epoch'] == epoch
    assert lock['groups']['eood']['physical_gpu'] == 0
    assert lock['groups']['110']['physical_gpu'] == 1


def test_summary_reference_routes_not_mislabeled():
    seen = []
    module = types.ModuleType('crane_project.utils.port_geometry_size_distribution_metrics_v1')
    def fake(rows, cache):
        seen.append(rows[0]['midpoint'])
        return {'overall': dict(b={'base': True}, midpoint={'M': True}, edge_residual={'L': True}, paired={})}
    module.summary = fake
    row = dict(b='B', eood_m='M', eood_l='L', eood_ml='ML')
    import crane_project.utils as package
    with patch.dict(sys.modules, {module.__name__: module}), patch.object(package, 'port_geometry_size_distribution_metrics_v1', module, create=True):
        out = e.summarize([row], 'eood')
    assert seen == ['B', 'M']
    assert set(out['metrics']['overall']) == {'eood', 'eood_m', 'eood_l', 'eood_ml', 'eood_l_paired_vs_reference', 'eood_ml_paired_vs_reference'}
    assert not out['TEST_used_for_selection']


def test_backup_preserves_existing_different_files(tmp_path):
    source = tmp_path/'weight.pth';source.write_bytes(b'fixed bytes')
    digest = e.training.sha(source)
    lock = dict(groups={'110': dict(arms=['110_l'])},
        heads={'110_l': dict(checkpoint=str(source), selected_checkpoint=dict(sha256=digest))},
        backup_directories=[str(tmp_path/'one'), str(tmp_path/'two')])
    report = e.backup_heads(lock, '110')
    assert len(report[0]['copies']) == 2
    assert e.backup_heads(lock, '110') == report
    p = Path(report[0]['copies'][0]['path']);p.write_bytes(b'other')
    with pytest.raises(ValueError):
        e.backup_heads(lock, '110')
    assert p.read_bytes() == b'other'


def test_default_parser_cannot_start_test():
    args = e.parser().parse_args(['--family', '110'])
    assert not args.execute
