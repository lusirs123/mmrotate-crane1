"""CPU synthetic unit fixtures, not TRAIN/VAL experimental results."""
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path

import pytest
import torch

from crane_project.tools import train_port_geometry_midpoint_readout_v1 as t

f, m = t.f, t.model


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def synthetic_row(seq, i=0, role='train', scale=1., missing=False):
    b = torch.tensor([[64., 48., 32., 16., .2, .8]])
    if missing:
        b = b[:0]
    xy = torch.tensor([[scale, scale]]).expand(len(b), -1)
    bm = b[:, :5].clone(); bm[:, :4] *= scale
    return dict(image='%s_%05d'%(seq, i), sequence=seq, frame_id=i,
        domain=seq.split('_')[0], role=role,
        split='val' if role == 'val' else ('train_sim' if seq.startswith('sim_') else 'train'),
        scale=scale, eligible=role == 'train' and not missing,
        boxes_original=b, boxes_model=bm, scale_xy=xy,
        gt_original=torch.tensor([[64.7, 47.6, 32.9, 15.5, .22]]),
        roi=torch.randn(len(b), 256, 9, 9), support=torch.ones(len(b), 1, 9, 9))


def synthetic_fixture():
    torch.manual_seed(25)
    train = [synthetic_row(seq, i) for seq in ('real_seq01', 'sim_seq08') for i in range(8)]
    val = [synthetic_row(seq, i, 'val', missing=seq == 'real_seq07' and i == 1)
        for seq in ('real_seq07', 'real_seq14', 'sim_seq10') for i in range(12)]
    return train, val


def args_of(row):
    return [row[k] for k in ('roi', 'support', 'boxes_original', 'boxes_model', 'scale_xy')]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('swapped', [False, True])
def test_exact_zero_identity_raw_axes_angle_boundary_actual_xy_empty(dtype, swapped):
    row = synthetic_row('real_seq01', scale=.5)
    b = row['boxes_original'].to(dtype); b[:, 4] = math.pi/2-1e-6
    if swapped:
        b[:, [2, 3]] = b[:, [3, 2]]
    xy = b.new_tensor([[.497, .499]])
    bm = b[:, :5].clone(); bm[:, :4] *= xy[:, [0, 1, 0, 1]]
    roi, support = row['roi'].to(dtype), row['support'].to(dtype)
    head = m.UniformMidpointHead().to(dtype)
    output = head(roi, support, b, bm, xy)
    assert torch.equal(output['boxes_original'], b)
    assert torch.equal(output['point_delta_roi'], torch.zeros_like(output['point_delta_roi']))
    assert output['accepted'].all()
    assert head(roi[:0], support[:0], b[:0], bm[:0], xy[:0])['boxes_original'].shape == (0, 6)
    assert torch.allclose(output['probabilities'], torch.full_like(output['probabilities'], 1./81))


def test_same_initial_capacity_state_and_nonzero_head_gradient_frozen_inputs():
    before = deepcopy(f.m.SETTINGS); row = synthetic_row('sim_seq08')
    f.seed_all(); old = t.t.model.SigmaMidpointHead(1.5)
    f.seed_all(); new = m.UniformMidpointHead()
    assert f.g.state_digest(old) == f.g.state_digest(new)
    assert sum(p.numel() for p in new.parameters()) == 17696
    roi, support, b, bm, xy = args_of(row)
    roi.requires_grad_(); support.requires_grad_()
    output = new(roi, support, b, bm, xy)
    loss, _ = f.m.point_loss(output['points_roi'], f.m.target_points(row['gt_original'], b, bm, xy))
    loss.backward()
    assert new.output.weight.grad.norm() > 0 and torch.isfinite(new.output.weight.grad).all()
    # Zero final weights give zero stem gradient on the first backward.
    assert all(p.grad is not None and torch.equal(p.grad, torch.zeros_like(p.grad)) for p in new.stem.parameters())
    assert roi.grad is None and support.grad is None and f.m.SETTINGS == before
    assert 'gt' not in inspect.signature(new.forward).parameters
    assert not torch.equal(old(*args_of(row))['probabilities'], output['probabilities'])


def test_uniform_neutral_is_actual_grid_expectation_and_prior_independent_of_anchor():
    a = torch.tensor([[[.33, 0.], [0., .33], [-.33, 0.], [0., -.33]]])
    grid, prior, neutral = m.uniform_prior(a)
    _, shifted_prior, shifted_neutral = m.uniform_prior(a+.1)
    assert torch.equal(prior, shifted_prior) and torch.equal(neutral, shifted_neutral)
    assert torch.equal(neutral, torch.matmul(prior.softmax(-1), grid))
    _, gaussian, gaussian_neutral = t.t.model.point_prior(a, 1.5)
    assert not torch.equal(prior, gaussian) and not torch.equal(neutral, gaussian_neutral)
    assert not torch.allclose(gaussian_neutral, a, atol=1e-5)


@pytest.mark.parametrize('bad', ['support', 'roi', 'scale'])
def test_invalid_geometry_and_support_rejected(bad):
    row = synthetic_row('real_seq01')
    if bad == 'support':
        row['support'].fill_(1.1)
    elif bad == 'roi':
        row['roi'][0, 0, 0, 0] = float('nan')
    else:
        row['scale_xy'].fill_(.5)
    with pytest.raises(ValueError):
        m.UniformMidpointHead()(*args_of(row))


def test_nonzero_readout_preserves_raw_score_count_and_whole_box_fallback():
    row = synthetic_row('real_seq01'); head = m.UniformMidpointHead()
    with torch.no_grad():
        head.output.weight.normal_(0, 3.)
        output = head(*args_of(row))
    assert torch.equal(output['boxes_original'][:, 5], row['boxes_original'][:, 5])
    assert output['boxes_original'].shape == row['boxes_original'].shape
    rejected = ~output['accepted']
    assert torch.equal(output['boxes_original'][rejected], row['boxes_original'][rejected])


def test_new_checkpoint_identity_exact_reload_no_overwrite_and_wrong_loader(tmp_path):
    _, val = synthetic_fixture(); head = m.UniformMidpointHead()
    proof = {'synthetic': True}; cache = {'manifest_sha256': 'synthetic'}
    path = tmp_path/'synthetic_zero.pth'
    ckpt = t.save_checkpoint(path, head, None, 0, 0, proof, cache)
    replay, payload = t.load_head(path, proof, cache, 'cpu')
    assert payload['optimizer_state'] is None and ckpt['updates'] == 0
    assert f.evaluate(head, val, 'cpu') == f.evaluate(replay, val, 'cpu')
    with pytest.raises(ValueError, match='identity/source/cache'):
        t.load_head(path, {'wrong': True}, cache, 'cpu')
    with pytest.raises(ValueError, match='parameter/source/cache'):
        t.t.load_head(path, 1.5, proof, cache, 'cpu')
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        t.save_checkpoint(path, head, None, 0, 0, proof, cache)
    assert path.read_bytes() == before
    payload['readout'] = 'gaussian'; torch.save(payload, tmp_path/'synthetic_wrong.pth')
    with pytest.raises(ValueError, match='identity/source/cache'):
        t.load_head(tmp_path/'synthetic_wrong.pth', proof, cache, 'cpu')
    with pytest.raises(ValueError, match='uniform head class'):
        t.save_checkpoint(tmp_path/'wrong_class.pth', f.m.SpatialMidpointHead(), None, 0, 0, proof, cache)


def test_preflight_backward_save_reload_has_zero_optimizer_updates(tmp_path, monkeypatch):
    train, val = synthetic_fixture(); f.seed_all(); initial_head = m.UniformMidpointHead()
    initial = deepcopy(initial_head.state_dict()); digest = f.g.state_digest(initial_head)
    reference = {'rows': f.evaluate(initial_head, val, 'cpu')}
    def forbidden(*args, **kwargs):
        raise AssertionError('No optimizer update or detector access in preflight')
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    monkeypatch.setattr(f, 'build_detector', forbidden)
    monkeypatch.setattr(f, 'create_cache', forbidden)
    value = t.preflight(train, val, initial, {'initial_state': digest}, reference,
        {'synthetic': True}, {'manifest_sha256': 'synthetic'}, 'cpu', tmp_path)
    assert value['head_updates'] == 0 and value['fitted_weight'] is False
    assert len(value['gradient_batches']) == 2 and value['initial_full_val_exact_b']
    assert value['checkpoint']['head_digest'] == digest


def test_real_source_contract_keeps_historical_identities_and_exposes_detector_drift():
    result = t.checked_sources(); reference = result['reference_identity']
    assert reference['training_identity']['parent_identity']['sources'] == t.t.read(f.SOURCES)['sources']
    assert set(result['detector_only_source_differences']) <= t.DETECTOR_ONLY
    for name, detail in result['detector_only_source_differences'].items():
        assert detail['historical_sha256'] == reference['sources'][name]
        assert detail['current_sha256'] == result['sources'][name]
        assert detail['instantiated_or_called_by_this_tool'] is False
    protocol = t.protocol_document()
    assert protocol['scope']['new_formal_conditions'] == 1
    assert protocol['scope']['formal_updates'] == 21648 and protocol['scope']['preflight_updates'] == 0
    assert protocol['selection_config'] == f.SELECTION_CONFIG and protocol['reference']['epoch'] == 3
    assert t.parser()._option_string_actions['--stage'].choices == ('check', 'preflight', 'train')


def test_executed_historical_source_drift_is_not_exempt(monkeypatch):
    original_sha = t.t.sha
    path = t.ROOT/'crane_project/utils/port_geometry_midpoint_v1.py'
    monkeypatch.setattr(t.t, 'sha', lambda p: 'wrong' if Path(p) == path else original_sha(p))
    with pytest.raises(ValueError, match='Executed historical dependency'):
        t.recorded_reference_identity()


def test_check_stage_no_tensor_load_gpu_update_or_detector_and_failure_leaves_no_directory(tmp_path, monkeypatch):
    proof = {'synthetic': True}
    monkeypatch.setattr(t, 'checked_contract', lambda *a: ({}, {}, {}, proof))
    def forbidden(*a, **kw):
        raise AssertionError('Static must not access tensor/GPU/detector')
    monkeypatch.setattr(torch, 'load', forbidden); monkeypatch.setattr(f, 'runtime', forbidden)
    monkeypatch.setattr(f, 'build_detector', forbidden); monkeypatch.setattr(f, 'update', forbidden)
    args = t.parser().parse_args(['--stage', 'check', '--out-dir', str(tmp_path/'synthetic_check')])
    report = t.run(args)
    assert report['head_updates_total'] == report['cache_tensor_loads'] == 0
    assert report['status'].startswith('STATIC_READOUT_CONTRACT_PASS')
    assert t.t.read(tmp_path/'synthetic_check/artifacts.json')['files']['completion.json'] == t.t.sha(tmp_path/'synthetic_check/completion.json')
    def missing(*a):
        raise FileNotFoundError('synthetic missing input')
    monkeypatch.setattr(t, 'checked_contract', missing)
    args.out_dir = str(tmp_path/'synthetic_failure')
    with pytest.raises(FileNotFoundError):
        t.run(args)
    assert not Path(args.out_dir).exists()


def test_output_overlap_existing_directory_train_without_gate_and_ddp_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='disjoint'):
        t.safe_output(tmp_path/'inputs/result', [tmp_path/'inputs'])
    with pytest.raises(ValueError, match='disjoint'):
        t.safe_output(tmp_path, [tmp_path/'inputs'])
    with pytest.raises(FileExistsError):
        t.safe_output(tmp_path, [tmp_path.parent/'unrelated'])
    args = t.parser().parse_args(['--stage', 'train', '--out-dir', str(tmp_path/'synthetic_train')])
    with pytest.raises(ValueError, match='preflight-report'):
        t.run(args)
    monkeypatch.setenv('WORLD_SIZE', '2'); args.stage = 'check'
    with pytest.raises(ValueError, match='one logical GPU'):
        t.run(args)


def test_comparison_table_three_denominators_and_fallback_counts():
    _, val = synthetic_fixture(); head = m.UniformMidpointHead()
    rows = f.evaluate(head, val, 'cpu'); value = f.summaries(rows)
    reference = {'rows': rows, 'value': value}
    result = dict(rows=rows, value=value, selection={'selected_checkpoint': {'epoch': 7}})
    table = t.comparison_rows(result, reference)
    assert len(table) == 6
    real = next(r for r in table if r['condition'] == m.READOUT and r['domain'] == 'real')
    assert real['frames'] == 24 and real['output_frames'] == 23
    assert real['output_coverage_pct'] == pytest.approx(100.*23/24)
    assert real['accepted_frames']+real['fallback_frames'] == real['output_frames']
    assert real['selected_epoch'] == 7 and real['pure_angle_rmse_deg'] is None


@pytest.mark.parametrize('tamper', [None, 'updates', 'gradient', 'weight'])
def test_preflight_gate_binds_zero_updates_gradients_and_checkpoint(tmp_path, tamper):
    proof = {'synthetic': True}; head = m.UniformMidpointHead(); cache = {'manifest_sha256': 'synthetic'}
    ckpt = t.save_checkpoint(tmp_path/'synthetic_zero.pth', head, None, 0, 0, proof, cache)
    schedule = {'initial_state': f.g.state_digest(head), 'epoch_steps': [902]*24}
    gradient = dict(loss=.01, output_grad_norm_after_clip=.1,
        grad_norm_before_clip=1., grad_norm_after_clip=1., optimizer_updates=0)
    report = dict(protocol=t.VERSION, stage='preflight', status=t.PREFLIGHT_STATUS,
        proof=proof, schedule=schedule, readout=m.READOUT, head_updates_total=0,
        test_access=False, detector_updates=0, detector_forward_calls=0,
        feature_extractions=0, cache_tensor_loads=3, reference_replay={'exact_cached_val_replay': True},
        preflight=dict(checkpoint=ckpt, fitted_weight=False, head_updates=0, save_reload_exact=True,
            initial_full_val_exact_b=True, gradient_batches=[deepcopy(gradient), deepcopy(gradient)]))
    if tamper == 'updates':
        report['head_updates_total'] = 1
    elif tamper == 'gradient':
        report['preflight']['gradient_batches'][0]['output_grad_norm_after_clip'] = 0.
    elif tamper == 'weight':
        report['preflight']['checkpoint']['sha256'] = 'wrong'
    path = tmp_path/'completion.json'; t.t.write(path, report); t.publish(tmp_path)
    if tamper:
        with pytest.raises(ValueError):
            t.checked_preflight(path, proof, schedule)
    else:
        assert t.checked_preflight(path, proof, schedule) == report


def test_formal_control_flow_all24_val_same_rule_fresh_state_without_any_training(tmp_path, monkeypatch):
    """Simulate events/selection only; no optimizer step or real data fitting."""
    train, val = synthetic_fixture(); f.seed_all(); head = m.UniformMidpointHead()
    initial = deepcopy(head.state_dict()); schedule = dict(initial_state=f.g.state_digest(head), epoch_steps=[2]*24)
    rows = f.evaluate(head, val, 'cpu'); reference = dict(rows=rows, value=f.summaries(rows))
    def fake_update(head, optimizer, records, indices, device, progress, context):
        progress(dict(context, stage='optimizer_step'))
        return dict(loss=.01, point_parts=[.01]*4, grad_norm_before_clip=0.,
            grad_norm_after_clip=0., stem_grad_norm=0., output_grad_norm=0.)
    calls = []
    def select(entries, config):
        assert len(entries) == 24 and config == f.SELECTION_CONFIG
        assert all(v['checkpoint']['updates'] == v['checkpoint']['epoch']*2 for v in entries.values())
        calls.append(sorted(entries)); return 'head_epoch_07.pth', None, {'synthetic': True}
    def forbidden(*a, **kw):
        raise AssertionError('No real optimization or detector use in flow fixture')
    monkeypatch.setattr(t.t, 'tracked_update', fake_update)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    monkeypatch.setattr(f, 'select_best_checkpoint', select)
    monkeypatch.setattr(f, 'build_detector', forbidden)
    events = []; result = t.train_condition(train, val, initial, {'synthetic': True},
        {'manifest_sha256': 'synthetic'}, reference, schedule, 'cpu', tmp_path, events.append)
    assert len(calls) == 1 and result['updates'] == 48
    assert sum(r['stage'] == 'optimizer_step' for r in events) == 48
    assert result['selection']['selected_checkpoint']['epoch'] == 7
    assert result['selection']['automatic_promotion'] is False and result['selection']['test_access'] is False
    assert len(result['all_epoch_values']) == 24
    assert result['selection']['selected_checkpoint']['head_digest'] == schedule['initial_state']
