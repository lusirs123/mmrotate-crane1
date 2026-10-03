"""Synthetic CPU regression fixtures; no trained-performance evidence."""
from argparse import Namespace
from copy import deepcopy
import json
import math

import numpy as np
import pytest
import torch

from crane_project.tools import diagnose_port_reliability_mechanism_v1 as tool
from crane_project.utils import port_reliability_mechanism_v1 as m
from crane_project.utils.port_structure_reliability_v1 import (
    component_probes_original, quality_targets_original, response_targets)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def meta():
    return dict(scale_factor=[1., 1., 1., 1.], ori_shape=(64, 64, 3), img_shape=(64, 64, 3),
        pad_shape=(64, 64, 3), flip=False, flip_direction=None,
        img_norm_cfg=dict(mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=False))


def values():
    gt = torch.tensor([32., 32., 32., 16., 0.])
    probes, _ = component_probes_original(gt)
    boxes = torch.cat((gt[None], probes))
    target, mask, errors = quality_targets_original(boxes, gt)
    axis = torch.tensor([[16., 32.], [48., 32.]])
    response, valid = response_targets(axis, (8, 8), meta())
    return (torch.randn(1, 256, 8, 8), boxes, torch.full((len(boxes),), .7), meta(),
            target, mask, 1, response, valid), gt, axis, errors


def test_tied_ranks_auc_polarity_and_undefined_support():
    assert m.ranks([3, 1, 1, 2]).tolist() == [4., 1.5, 1.5, 3.]
    assert m.spearman([1, 1], [2, 3]) is None
    errors = np.array([[1., .05, 1.], [20., .2, 5.]])
    qualities = np.array([[.9, .8, .7], [.1, .2, .3]])
    result = m.quality_profile(errors, qualities, 'center')
    assert result['correct_vs_incorrect_auroc'] == 1.
    assert result['spearman_quality_vs_negative_error'] == pytest.approx(1.)
    assert m.quality_profile(errors, qualities[::-1], 'center')['correct_vs_incorrect_auroc'] == 0.
    assert m.quality_profile(errors, qualities, 'angle', [True, False])['correct_vs_incorrect_auroc'] is None
    with pytest.raises(ValueError):
        m.quality_profile(errors, qualities*2, 'center')


def test_component_boundaries_and_direction_mask_are_preserved():
    errors = np.array([[15., .1, 3.], [14., .11, 3.1]])
    q = np.full_like(errors, .5)
    assert m.quality_profile(errors, q, 'center')['incorrect_outputs'] == 1
    assert m.quality_profile(errors, q, 'size')['correct_outputs'] == 1
    assert m.quality_profile(errors, q, 'angle', [True, False])['assessed_outputs'] == 1
    empty = m.quality_profile(np.empty((0, 3)), np.empty((0, 3)), 'center')
    assert empty['quality_target_mae'] is None and not empty['auroc_defined']


def test_train_case_selection_ignores_other_epochs_and_merges_duplicate_views():
    records = []
    for domain in ('real', 'sim'):
        for i, errors in enumerate(([1, .1, 1], [20, .5, 30])):
            records.append(dict(epoch=8, genuine_count=1, image=domain+str(i), domain=domain,
                dataset_index=i, sequence=domain, optimizer_step=len(records)+1, slot=i,
                view_seed=i, view_image_sha256='fixture', genuine_errors_original=[errors]))
    other = dict(records[0], epoch=1, genuine_errors_original=[[999, 999, 999]])
    cases = tool.choose_cases(records+[other])
    assert len(cases) == 4
    assert next(r for r in cases if r['image'] == 'real1')['selection_roles'] == [
        'max_center_error', 'max_size_error', 'max_angle_error']


def test_response_zero_restores_state_and_restores_flag_after_error():
    args, _, _, _ = values()
    arm = tool.branch.make_arms()['structure'].eval()
    before = tool.prior.state_digest(arm)
    with torch.no_grad():
        full = arm(*args[:4])['qualities'].cpu().tolist()
    zero = m.zero_response_qualities(arm, *args[:4])
    assert np.abs(np.array(full)-np.array(zero)).max() > 0
    assert arm.use_structure and before == tool.prior.state_digest(arm)
    assert all(p.grad is None for p in arm.parameters())
    class Fails:
        use_structure, training = True, False
        def __call__(self, *unused):
            raise ValueError('fixture')
    bad = Fails()
    with pytest.raises(ValueError, match='fixture'):
        m.zero_response_qualities(bad, *args[:4])
    assert bad.use_structure


def test_existing_loss_gradient_decomposition_is_finite_and_read_only():
    args, _, _, _ = values()
    arm = tool.branch.make_arms()['structure'].eval()
    before = tool.prior.state_digest(arm)
    result = m.structure_gradients(arm, *args)
    assert sum(result['quality_component_losses'].values()) == pytest.approx(result['loss_quality'])
    assert result['shared_stem_quality_vs_structure']['first_norm'] > 0
    assert result['shared_stem_quality_vs_structure']['second_norm'] > 0
    assert before == tool.prior.state_digest(arm)
    assert all(p.grad is None for p in arm.parameters()) and args[0].grad is None
    inverse = m.gradient_relation(torch.tensor([1.]), torch.tensor([-2.]))
    assert inverse['cosine'] == -1 and inverse['summed_norm'] == 1
    assert m.gradient_relation(torch.zeros(1), torch.ones(1))['cosine'] is None


def test_annotation_response_evidence_has_correct_stride_and_periodicity():
    args, gt, axis, _ = values()
    response, valid = args[-2:]
    result = m.response_evidence(response[0].numpy(), valid[0, 0].numpy(), axis.numpy(), gt[None].numpy())
    assert result['global_peaks'][0]['xy_model_px'] == [32., 32.]
    assert result['B_context_line_moment']['direction_defined']
    assert result['B_context_line_moment']['angle_error_to_annotation_axis_deg'] < 1e-5
    assert m.angular_difference(0, math.pi) == pytest.approx(0.)
    flat = m.response_evidence(np.full((2, 8, 8), .5), np.ones((8, 8)), axis.numpy(), gt[None].numpy())
    assert not flat['B_context_line_moment']['direction_defined']
    assert flat['B_context_line_moment']['angle_error_to_annotation_axis_deg'] is None


def test_preview_panels_preserve_geometry_and_can_be_read(tmp_path):
    import cv2
    args, gt, axis, _ = values()
    path = tmp_path/'preview.png'
    image = torch.full((1, 3, 64, 64), 60.)
    m.preview(path, image, meta(), args[-2][0].numpy(), axis.numpy(), gt[None].numpy())
    pixels = cv2.imread(str(path))
    assert pixels.shape == (64, 192, 3)


def test_statistic_roundoff_does_not_allow_changed_counts_or_nonfinite():
    tool.assert_same_statistics({'n': 4, 'value': .5}, {'n': 4, 'value': .5+1e-12})
    with pytest.raises(ValueError):
        tool.assert_same_statistics({'n': 4}, {'n': 5})
    with pytest.raises(ValueError):
        tool.assert_same_statistics(float('nan'), float('nan'))


def evidence_fixture(tmp_path):
    folders = [tmp_path/'train', tmp_path/'val']
    for folder in folders:
        folder.mkdir()
    args = Namespace(train_dir=folders[0], val_dir=folders[1], out_dir=tmp_path/'out')
    for name, path in tool.evidence_paths(args).items():
        path.write_text(json.dumps(dict(synthetic=name)))
    proof = {n: tool.branch.sha(p) for n, p in tool.evidence_paths(args).items()}
    return args, proof


def test_changed_evidence_is_rejected_before_inference(tmp_path):
    args, proof = evidence_fixture(tmp_path)
    proof['train_steps.jsonl'] = 'wrong'
    with pytest.raises(ValueError, match='exact reviewed'):
        tool.checked_evidence(args, {}, dict(evidence_sha256=proof), {})


def test_full_cpu_review_preserves_missing_denominators_and_does_not_run_model(tmp_path, monkeypatch):
    args, proof = evidence_fixture(tmp_path)
    args.input_snapshot = tmp_path/'snapshot.json'; args.input_snapshot.write_text('{}')
    monkeypatch.setattr(tool.train, 'SNAPSHOT_SHA', tool.branch.sha(args.input_snapshot))
    sources, protocol = dict(synthetic=True), dict(max_gpu_views=8)
    monkeypatch.setattr(tool, 'checked_sources', lambda: (sources, protocol, {}))
    logs, inputs = [], []
    gt = [32., 32., 32., 16., 0.]
    for domain in ('real', 'sim'):
        inputs.append(dict(image_size=[64, 64]))
        for ep in range(1, 9):
            update = dict(loss_quality=.1, loss_genuine=.05, loss_probes=.15, loss_structure=0.,
                          genuine_quality_before_update=[[.8, .7, .6]])
            logs.append(dict(epoch=ep, image=domain, domain=domain, sequence=domain,
                dataset_index=len(inputs)-1, slot=0, view_seed=0, optimizer_step=len(logs)+1,
                view_image_sha256='synthetic', genuine_count=1,
                genuine_b_original=[gt+[.7]], genuine_errors_original=[[1., .05, 1.]],
                scale_factor=[1., 1., 1., 1.], img_shape=[64, 64, 3], pad_shape=[64, 64, 3],
                flip=False, flip_direction=None, qualification=dict(evaluation_angle_eligible=True),
                train_angle_mask=1., arms={arm: deepcopy(update) for arm in tool.branch.ARMS}))
    rows = [dict(image='missing', domain='real', sequence='real', pred=None, errors=None,
                 angle_axis_well_defined=True, qualities={arm: None for arm in tool.branch.ARMS}, cached_b_model=[]),
            dict(image='sim', domain='sim', sequence='sim', pred=gt+[.7],
                 errors=dict(center_px=1., size_max_relative=.05, angle_deg=1.),
                 angle_axis_well_defined=True, qualities={arm: [.8, .7, .6] for arm in tool.branch.ARMS},
                 cached_b_model=[gt])]
    result = tool.cpu_review(args, logs, rows, inputs, proof, dict(checkpoint_sha256='fixture'), sources, protocol)
    assert result['cached_val_genuine']['domain:real']['output_coverage'] == 0
    assert result['cached_val_genuine']['domain:real']['all_frame_center_correct_coverage'] == 0
    assert result['cached_val_genuine']['domain:real']['center_hit_rate_on_outputs'] is None
    assert not result['model_inference'] and not result['test_read']
    assert result['training_stream_role'].startswith('logged_online_pre_update')


def test_small_gpu_orchestration_with_cpu_fixture_never_updates_weights(tmp_path, monkeypatch):
    """CPU fixture substitutes CUDA/I/O only; it is not server performance."""
    args, proof = evidence_fixture(tmp_path)
    args.gpu = 0; args.b_checkpoint = tmp_path/'B.pth'; args.train_cache = tmp_path/'cache'
    args.structure_report = tmp_path/'structure.json'
    args.input_snapshot = tmp_path/'snapshot.json'; args.input_snapshot.write_text('{}')
    monkeypatch.setattr(tool.train, 'SNAPSHOT_SHA', tool.branch.sha(args.input_snapshot))
    values0, gt, axis, errors = values()
    image = torch.full((1, 3, 64, 64), 60.)
    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
            self.eval()
    detector = Detector(); frozen = tool.prior.state_digest(detector)
    case = dict(image='real_fixture', domain='real', sequence='real_fixture', dataset_index=0,
        slot=0, view_seed=0, optimizer_step=1, logged_view_image_sha256=tool.prior.tensor_sha(image),
        selection_roles=['first_dataset_index'])
    replay = dict(view_image_sha256=case['logged_view_image_sha256'], input_sha256='fixture',
        scale_factor=[1., 1., 1., 1.], img_shape=[64, 64, 3], pad_shape=[64, 64, 3],
        flip=False, flip_direction=None, genuine_b_original=[gt.tolist()+[.7]],
        genuine_errors_original=errors[:1].tolist())
    logs = [dict(replay, optimizer_step=1, epoch=8)]
    inputs = [dict(gt_original=gt.tolist(), direction_qualification=dict(structure_source='synthetic_axis'))]
    path = args.train_dir/'epoch_08.pth'; path.write_bytes(b'synthetic CPU fixture only')
    sources = dict(training_sources=dict(synthetic=True))
    protocol = dict(branch_checkpoint_sha256=tool.branch.sha(path), max_gpu_views=8)
    args.review_report = tmp_path/'review.json'
    tool.branch.write_new(args.review_report, dict(status='CPU_GENUINE_SUPPORT_REVIEW_COMPLETE',
        sources=sources, evidence_sha256=proof, protocol_contract=protocol, gpu_cases=[case]))
    original_make = tool.branch.make_arms
    original = original_make()
    payload = dict(frozen_b_state=frozen, contract=dict(architecture=tool.branch.architecture(original)),
                   arms={n: a.state_dict() for n, a in original.items()})
    monkeypatch.setattr(tool, 'choose_cases', lambda *unused: [case])
    monkeypatch.setattr(tool.previous.base, 'fixed_bundle', lambda *unused: payload)
    monkeypatch.setattr(tool.prior, 'check_cfg', lambda: Namespace(data=Namespace(train=[])))
    monkeypatch.setattr(tool.train, 'checked_inputs', lambda *unused: (None, dict(b_cache_state=frozen)))
    monkeypatch.setattr(tool.train, 'build_runtime', lambda *unused: (detector, dict(synthetic_cpu=True)))
    monkeypatch.setattr(tool.branch, 'make_arms', lambda *unused: original_make('cpu'))
    monkeypatch.setattr(tool.train, 'datasets_for', lambda *unused: ['fixture'])
    monkeypatch.setattr(tool.train, 'training_view', lambda *unused: (values0, deepcopy(replay), None))
    monkeypatch.setattr(tool.train, 'axis_for', lambda *unused: axis.numpy())
    monkeypatch.setattr(tool, 'replay_image', lambda *unused: (image, meta()))
    monkeypatch.setattr(tool, 'checked_sources', lambda: (sources, protocol, {}))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *unused: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *unused: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda *unused: 0)
    tool.gpu_probe(args, logs, inputs, proof, dict(b_state=frozen), sources, protocol, {})
    report = json.loads((args.out_dir/'probe.json').read_text())
    records = [json.loads(l) for l in (args.out_dir/'cases.jsonl').open()]
    assert report['heads_before'] == report['heads_after']
    assert report['detector_updates'] == report['head_updates'] == 0
    assert not report['optimizer_created'] and not report['test_read']
    assert records[0]['genuine_count'] == 1 and len(records[0]['probe_names']) == 13
    assert len(records[0]['qualities']['structure']) == 14
    assert report['cases'] == 1


@pytest.mark.parametrize('missing_path', [False, True])
def test_prepare_review_builds_and_reuses(tmp_path, monkeypatch, missing_path):
    args = Namespace(out_dir=tmp_path/'probe',
        review_report=tmp_path/'missing.json' if missing_path else None)
    cases = [dict(image='train_only')]
    monkeypatch.setattr(tool, 'choose_cases', lambda logs: cases)
    calls = []
    def build(a, logs, rows, inputs, proof, completion, sources, protocol):
        calls.append(a.out_dir)
        a.out_dir.mkdir()
        (a.out_dir/'review.json').write_text(json.dumps(dict(
            status='CPU_GENUINE_SUPPORT_REVIEW_COMPLETE', sources=sources,
            evidence_sha256=proof, protocol_contract=protocol, gpu_cases=cases)))
    monkeypatch.setattr(tool, 'cpu_review', build)
    tool.prepare_review(args, [], [], {}, {}, {}, {}, {})
    assert args.review_report == tmp_path/'probe_review/review.json'
    assert len(calls) == 1 and not args.out_dir.exists()
    args.review_report = None
    tool.prepare_review(args, [], [], {}, {}, {}, {}, {})
    assert len(calls) == 1
    tool.prepare_review(args, [], [], {}, {}, {}, {}, {})
    assert len(calls) == 1  # explicit existing report is validated too
    with pytest.raises(ValueError):
        tool.prepare_review(args, [], [], {}, {}, {}, {'changed': True}, {})
    assert len(calls) == 1


def test_prepare_review_preserves_invalid_evidence(tmp_path, monkeypatch):
    args = Namespace(out_dir=tmp_path/'probe', review_report=tmp_path/'invalid.json')
    args.review_report.write_text('{broken')
    monkeypatch.setattr(tool, 'cpu_review', lambda *a: pytest.fail('must not replace evidence'))
    with pytest.raises(json.JSONDecodeError):
        tool.prepare_review(args, [], [], {}, {}, {}, {}, {})
    assert args.review_report.read_text() == '{broken'
    args.review_report = None
    (tmp_path/'probe_review').mkdir()
    with pytest.raises(FileExistsError, match='Preserve incomplete'):
        tool.prepare_review(args, [], [], {}, {}, {}, {}, {})
    assert not args.out_dir.exists()
