"""CPU contract/flow tests. Synthetic fixtures are not TEST performance evidence."""
from argparse import Namespace
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import pytest
import torch

from crane_project.tools import eval_port_reliability_branches_v1_test as entry
from crane_project.utils.port_reliability_train_policy_v1 import POLICY


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def meta():
    return dict(scale_factor=[1., 1., 1., 1.], ori_shape=(1024, 1024, 3),
                img_shape=(1024, 1024, 3), pad_shape=(1024, 1024, 3), flip=False)


def sources():
    gt = [32., 30., 24., 12., .1]
    return [dict(image=name+'_00001', sequence=name, domain=name.split('_')[0],
                 frame_id=1, gt=gt, angle_axis_well_defined=True, split='test',
                 image_size=[1024, 1024], image_sha256='synthetic')
            for name in ('real_seq03', 'sim_seq09')]


class FakeDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.eval()

    def extract_feat(self, image):
        return [torch.zeros(1, 256, 128, 128)]

    def simple_test_from_features(self, features, metas, rescale):
        return [[np.array([[33., 30., 24., 12., .1, .7]], dtype=np.float32)]]


def bundle(tmp_path, frozen):
    training, protocol, _ = entry.cached.checked_sources()
    arms = entry.branch.make_arms()
    contract = dict(role='formal_train', sources=training, protocol=protocol,
        direction_policy=POLICY, frozen_b=entry.ready.FROZEN_B, frozen_b_state=frozen,
        numeric_input_snapshot_sha256=entry.train.SNAPSHOT_SHA,
        train_input_manifest_sha256='2094bb4cde19dd97909bcd02b36db19c09c6d5c348c0b246e9b159a68c1b4022',
        architecture=entry.branch.architecture(arms))
    path = tmp_path/'epoch_08.pth'
    entry.branch.save_checkpoint(path, arms, entry.branch.make_optimizers(arms), contract,
        8, 20464, frozen, 'synthetic_fixture_not_real_training')
    log = tmp_path/'train_steps.jsonl'
    log.write_text('synthetic CPU fixture\n')
    entry.branch.write_new(tmp_path/'completion.json', dict(
        status='PAIRED_FORMAL_TRAIN_COMPLETE_FIXED_EPOCH8', sources=training, protocol=protocol,
        checkpoint_sha256=entry.branch.sha(path), optimizer_steps_per_arm=20464,
        save_reload_quality_exact=True, b_raw_unchanged=True, detector_optimizer_steps=0,
        b_state=frozen, train_log_sha256=entry.branch.sha(log)))
    return path, training, protocol


def test_completed_original_bundle_required_even_with_a_valid_serialization(tmp_path):
    path, training, protocol = bundle(tmp_path, dict(synthetic=True))
    contract = json.loads(entry.PROTOCOL.read_text())
    with pytest.raises(ValueError, match='original completed'):
        entry.fixed_bundle(path, training, protocol, contract)
    contract['branch_checkpoint_sha256'] = entry.branch.sha(path)  # fixture identity only
    assert entry.fixed_bundle(path, training, protocol, contract)['epoch'] == 8
    (tmp_path/'completion.json').unlink()
    with pytest.raises(FileNotFoundError):
        entry.fixed_bundle(path, training, protocol, contract)


def cache_fixture(tmp_path, monkeypatch, predictions=None):
    rows = sources()
    checkpoint = tmp_path/'b_epoch_24.pth'
    checkpoint.write_bytes(b'synthetic B, not actual weights')
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps(dict(evidence_role='source_val_checkpoint_selection',
        selected_checkpoint='epoch_24', config_sha256=entry.ready.FROZEN_B['config_sha256'],
        selected_path=str(checkpoint))))
    frozen = dict(entry.ready.FROZEN_B, checkpoint_sha256=entry.branch.sha(checkpoint),
                  selection_sha256=entry.branch.sha(selection))
    monkeypatch.setattr(entry.ready, 'FROZEN_B', frozen)
    pkl = tmp_path/'results.pkl'
    if predictions is None:
        predictions = [[np.empty((0, 6), dtype=np.float32)],
                       [np.array([rows[1]['gt']+[.71]], dtype=np.float32)]]
    pkl.write_bytes(pickle.dumps(predictions))
    report = tmp_path/'final_test_metrics_v2.json'
    identity = dict(annotations_sha256='synthetic-annotations')
    report.write_text(json.dumps(dict(protocol='crane_ckpt_sweep_final_test_v2',
        metric_protocol_version=2, evidence_role='fixed_test_after_source_val_selection',
        config=str(entry.ready.B_CONFIG), config_sha256=frozen['config_sha256'],
        checkpoint=str(checkpoint), checkpoint_sha256=frozen['checkpoint_sha256'],
        results_pkl=str(pkl), results_pkl_sha256=entry.branch.sha(pkl),
        gt_dir=str(entry.ready.DATA/'test/annfiles'), gt_annotations_sha256=identity['annotations_sha256'],
        frame_count=2, center_thresh_px=15.)))
    args = Namespace(b_checkpoint=checkpoint, b_selection=selection, b_test_pkl=pkl, b_test_report=report)
    return args, rows, identity


def test_cache_requires_generation_proof_preserves_empty_frames_and_raw_scores(tmp_path, monkeypatch):
    args, rows, identity = cache_fixture(tmp_path, monkeypatch)
    result, proof = entry.native_cache(args, rows, identity, require=True)
    assert result[0]['pred'] is None and result[1]['pred'][5] == pytest.approx(.71)
    assert proof['status'] == 'GENERATION_BOUND_NATIVE_TEST_CACHE_VERIFIED'
    args.b_test_report.unlink()
    assert entry.native_cache(args, rows, identity, require=False)[0] is None
    with pytest.raises(FileNotFoundError, match='proof unavailable'):
        entry.native_cache(args, rows, identity, require=True)


@pytest.mark.parametrize('tamper', ['checkpoint', 'pkl', 'report', 'selection'])
def test_bad_cache_identity_is_rejected_before_unpickling(tmp_path, monkeypatch, tamper):
    args, rows, identity = cache_fixture(tmp_path, monkeypatch)
    target = dict(checkpoint=args.b_checkpoint, pkl=args.b_test_pkl,
                  report=args.b_test_report, selection=args.b_selection)[tamper]
    if tamper == 'report':
        r = json.loads(target.read_text()); r['checkpoint_sha256'] = 'wrong'; target.write_text(json.dumps(r))
    else:
        target.write_bytes(target.read_bytes()+b'changed')
    monkeypatch.setattr(entry.pickle, 'load', lambda *a: pytest.fail('Unpickled before identity check'))
    with pytest.raises(ValueError):
        entry.native_cache(args, rows, identity, require=True)


@pytest.mark.parametrize('predictions', [
    [[np.zeros((0, 6))]],  # missing frame
    [[np.zeros((0, 6)), np.zeros((0, 6))], [np.zeros((0, 6))]],  # multiple classes
    [[np.zeros((0, 5))], [np.zeros((0, 6))]],  # geometry-only / surrogate scores
    [[np.array([[1., 1., 2., 1., 0., .05]])], [np.zeros((0, 6))]],
    [[np.array([[1., 1., 2., 1., 0., .8]]*2)], [np.zeros((0, 6))]],
])
def test_provenance_does_not_override_invalid_prediction_contract(tmp_path, monkeypatch, predictions):
    args, rows, identity = cache_fixture(tmp_path, monkeypatch, predictions)
    with pytest.raises(ValueError):
        entry.native_cache(args, rows, identity, require=True)


def test_sidecar_alone_is_valid_and_two_proofs_must_agree(tmp_path, monkeypatch):
    args, rows, identity = cache_fixture(tmp_path, monkeypatch)
    report = json.loads(args.b_test_report.read_text())
    proof = {k: report[k] for k in ('config', 'config_sha256', 'checkpoint', 'checkpoint_sha256',
                                  'results_pkl', 'results_pkl_sha256')}
    proof.update(protocol='crane_prediction_provenance_v1', split='fixed_test',
                 annotations_sha256=identity['annotations_sha256'])
    sidecar = Path(str(args.b_test_pkl)+'.provenance.json')
    sidecar.write_text(json.dumps(proof))
    assert set(entry.native_cache(args, rows, identity, True)[1]['proof_sha256']) == {'report', 'sidecar'}
    args.b_test_report.unlink()
    assert set(entry.native_cache(args, rows, identity, True)[1]['proof_sha256']) == {'sidecar'}
    proof['split'] = 'source_val'; sidecar.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='sidecar differs'):
        entry.native_cache(args, rows, identity, True)


def test_test_inputs_checks_actual_bytes_and_does_not_resplit(tmp_path, monkeypatch):
    monkeypatch.setattr(entry.ready, 'DATA', tmp_path)
    records = []
    identity = hashlib.sha256()
    for name in ('real_seq03_00001', 'sim_seq09_00001'):
        image = tmp_path/'test/images'/(name+'.jpg'); image.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (64, 64), 'gray').save(image)
        ann = tmp_path/'test/annfiles'/(name+'.txt'); ann.parent.mkdir(parents=True, exist_ok=True)
        ann.write_text('20 20 44 20 44 32 20 32 grab 0\n')
        digest = entry.branch.sha(image)
        identity.update(image.name.encode()); identity.update(b'\0'); identity.update(bytes.fromhex(digest))
        records.append(dict(id=name, sequence=name.rsplit('_', 1)[0], split='test',
            images=dict(path='test/images/'+name+'.jpg', sha256=digest),
            annfiles=dict(path='test/annfiles/'+name+'.txt', sha256=entry.branch.sha(ann))))
    manifest = tmp_path/'manifest.json'; manifest.write_text(json.dumps(dict(records=records)))
    contract = dict(test_counts=dict(real_seq03=1, sim_seq09=1),
        test_annotations_sha256=entry.ready.set_sha(entry.ready.files(tmp_path/'test/annfiles', '.txt')),
        test_image_identity_sha256=identity.hexdigest())
    rows, proof = entry.test_inputs(contract)
    assert len(rows) == 2 and proof['frames'] == 2 and rows[0]['image_size'] == [64, 64]
    image.write_bytes(image.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='image bytes'):
        entry.test_inputs(contract)
    records[-1]['split'] = 'val'; manifest.write_text(json.dumps(dict(records=records)))
    with pytest.raises(ValueError, match='no resplitting'):
        entry.test_inputs(contract)


def test_gt_is_absent_from_quality_inputs_and_side_output_changes_fail(monkeypatch):
    detector, arms = FakeDetector(), entry.branch.make_arms()
    source = dict(sources()[0], pred=[32., 30., 24., 12., .1, .71])
    image = torch.zeros(1, 3, 32, 32)
    result = entry.capture(detector, arms, image, [meta()], source, 'cache')
    changed_gt = dict(source, gt=[100., 200., 48., 24., 1.], domain='synthetic-other-domain')
    other = entry.capture(detector, arms, image, [meta()], changed_gt, 'cache')
    assert result['qualities'] == other['qualities'] and result['errors'] != other['errors']
    assert result['pred'][5] == .71 and not result['runtime_vs_native_cache']['all_six_close']
    original = entry.branch.online_qualities
    def interfering(*args):
        values = original(*args)
        detector.simple_test_from_features = lambda *a, **k: [[np.array([[34., 30., 24., 12., .1, .7]])]]
        return values
    monkeypatch.setattr(entry.branch, 'online_qualities', interfering)
    with pytest.raises(ValueError, match='changed original B'):
        entry.capture(detector, arms, image, [meta()], source, 'cache')


def test_capture_restores_original_coordinates_without_reordering_raw_width_height():
    class ResizedDetector(FakeDetector):
        def simple_test_from_features(self, features, metas, rescale):
            box = [40., 60., 24., 48., -.3, .7] if rescale else [20., 30., 12., 24., -.3, .7]
            return [[np.array([box], dtype=np.float32)]]
    m = meta(); m.update(scale_factor=[.5]*4, img_shape=(512, 512, 3))
    source = dict(sources()[0], pred=[40., 60., 24., 48., -.3, .7])
    result = entry.capture(ResizedDetector(), entry.branch.make_arms(),
                           torch.zeros(1, 3, 32, 32), [m], source, 'cache')
    assert result['model_boxes'][0] == pytest.approx([20., 30., 12., 24., -.3])
    assert result['runtime_vs_native_cache']['all_six_close']
    m['scale_factor'] = [.5, .8, .5, .8]
    with pytest.raises(ValueError, match='Anisotropic'):
        entry.capture(ResizedDetector(), entry.branch.make_arms(),
                      torch.zeros(1, 3, 32, 32), [m], source, 'cache')


@pytest.mark.parametrize('prediction_source', ['cache', 'online'])
def test_full_scoring_flow_retains_mismatches_and_all_predefined_comparisons(tmp_path, monkeypatch, prediction_source):
    detector = FakeDetector()
    checkpoint = tmp_path/'b.pth'; checkpoint.write_bytes(b'synthetic')
    monkeypatch.setattr(entry.ready, 'FROZEN_B', dict(entry.ready.FROZEN_B, checkpoint_sha256=entry.branch.sha(checkpoint)))
    path, training, training_protocol = bundle(tmp_path, entry.prior.state_digest(detector))
    _, _, contract, evaluation = entry.checked_sources()
    contract = dict(contract, branch_checkpoint_sha256=entry.branch.sha(path))  # fixture only
    rows, data_identity, cache_identity = sources(), dict(synthetic=True), dict(synthetic=True)
    if prediction_source == 'cache':
        rows = [dict(rows[0], pred=None), dict(rows[1], pred=rows[1]['gt']+[.71])]
    monkeypatch.setattr(entry.train, 'build_runtime', lambda *a: (detector, dict(synthetic_cpu=True)))
    monkeypatch.setattr(entry, 'test_dataset', lambda *a: object())
    monkeypatch.setattr(entry, 'test_view', lambda *a: (torch.zeros(1, 3, 32, 32), [meta()]))
    monkeypatch.setattr(entry, 'test_inputs', lambda *a: (deepcopy(rows), data_identity))
    monkeypatch.setattr(entry, 'native_cache', lambda *a, **kw: (deepcopy(rows), cache_identity))
    monkeypatch.setattr(entry, 'checked_sources', lambda: (training, training_protocol, contract, evaluation))
    original_make = entry.branch.make_arms
    monkeypatch.setattr(entry.branch, 'make_arms', lambda *a: original_make('cpu'))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *a: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda *a: 0)
    # Runtime must never construct an optimizer; the serialized TRAIN states only authenticate provenance.
    monkeypatch.setattr(entry.branch, 'make_optimizers', lambda *a: pytest.fail('Optimizer created in TEST'))
    args = Namespace(branch_checkpoint=path, b_checkpoint=checkpoint, gpu=0,
        out_dir=tmp_path/'evaluation', prediction_source=prediction_source)
    before = entry.branch.sha(path)
    entry.run(args, rows, data_identity, cache_identity, training, training_protocol, contract, evaluation, None)
    report = json.loads((args.out_dir/'test_compare.json').read_text())
    saved = [json.loads(r) for r in (args.out_dir/'test_qualities.jsonl').read_text().splitlines()]
    assert len(saved) == 2 and report['test_repeatedly_exposed'] and not report['test_used_for_selection']
    assert report['detector_optimizer_steps'] == report['branch_optimizer_steps'] == 0
    assert entry.branch.sha(path) == before and detector.weight.grad is None
    if prediction_source == 'cache':
        assert saved[0]['pred'] is None and saved[1]['pred'] == rows[1]['pred']
        assert saved[0]['genuine_online_qualities'] == {name: [] for name in entry.branch.ARMS}
        assert report['runtime_cache_parity']['compared_frames'] == 2
        assert not report['fresh_runtime_B_reproduces_all_cached_predictions']
        assert report['strata']['domain:real']['output_coverage'] == 0
        assert report['strata']['domain:real']['center_correct_output_frames'] == 0
        assert report['strata']['domain:real']['missing_output_frames'] == 1
    else:
        assert saved[0]['pred'][0] == 33 and saved[0]['pred'][5] == pytest.approx(.7)
        assert report['runtime_cache_parity']['compared_frames'] == 0
        assert report['cached_B_predictions_scores_exactly_preserved'] is None
    for group in report['strata'].values():
        for curves in group['component_curves'].values():
            assert len(curves) == 6
            for curve in curves:
                assert set(curve['methods']) == {'score', 'geometry', 'roi', 'structure'}
                assert len({m['accepted_frames'] for m in curve['methods'].values()}) == 1
    complete = json.loads((args.out_dir/'completion.json').read_text())
    assert complete['frames'] == 2 and complete['report_sha256'] == entry.branch.sha(args.out_dir/'test_compare.json')


def test_report_publication_is_exclusive_and_nan_cannot_publish(tmp_path):
    path = tmp_path/'completion.json'
    with pytest.raises(ValueError):
        entry.write_json(path, dict(value=float('nan')))
    assert not path.exists()
    entry.write_json(path, dict(value=1))
    with pytest.raises(FileExistsError):
        entry.write_json(path, dict(value=2))
    assert json.loads(path.read_text()) == dict(value=1)
    assert len(list(tmp_path.iterdir())) == 1
