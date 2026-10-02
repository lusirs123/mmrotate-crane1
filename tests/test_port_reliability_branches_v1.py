"""Necessary CPU checks of paired controls, persistent updates, reload and metrics.

Synthetic features test logic, not B performance. Actual B CUDA integration is
the four-view server smoke; never present these fixtures as new TRAIN evidence.
"""
from copy import deepcopy
import inspect
import json
import math

import numpy as np
import pytest
import torch

from crane_project.utils import port_reliability_branches_v1 as branch
from crane_project.utils import port_structure_reliability_v1 as core
from crane_project.utils.port_reliability_train_policy_v1 import direction_qualification, train_quality_targets_original
from crane_project.tools import train_port_reliability_branches_v1 as train
from crane_project.tools import eval_port_reliability_branches_v1 as evaluate


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture(genuine=True, native_square=False):
    meta = dict(scale_factor=[1., 1., 1., 1.], ori_shape=(64, 64, 3),
        img_shape=(64, 64, 3), pad_shape=(64, 64, 3), flip=False)
    gt = torch.tensor([[32., 30., 24., 21. if native_square else 12., .1]])
    axis = np.asarray([[32., 30.], [32., 30.]])+np.array([-1., 1.])[:, None]*12*np.array([math.cos(.1), math.sin(.1)])
    qualification = direction_qualification(gt, 'real' if native_square else 'sim',
        axis if native_square else None, 24./21 if native_square else None)
    probes, _ = core.component_probes_original(gt)
    pred = gt.clone()
    pred[:, 0] += 7
    boxes = torch.cat((pred, probes)) if genuine else probes
    target, mask, _ = train_quality_targets_original(boxes, gt, qualification)
    p3 = torch.arange(256*8*8, dtype=torch.float32).reshape(1, 256, 8, 8)/16384
    response, valid = core.response_targets(torch.tensor(axis, dtype=torch.float32), (8, 8), meta)
    return p3, boxes, torch.full((len(boxes),), .7 if genuine else .5), meta, target, mask, int(genuine), response, valid


def test_paired_init_and_disabled_roi_structure_paths():
    arms = branch.make_arms()
    assert branch.architecture(arms)['roi']['trainable_parameters'] == 52227
    assert branch.architecture(arms)['structure']['parameters'] == 52293
    assert branch.architecture(arms)['geometry']['parameters'] == 771
    for key, value in arms['roi'].state_dict().items():
        assert torch.equal(value, arms['structure'].state_dict()[key])
    values = fixture()
    before = deepcopy(arms['roi'].response.state_dict())
    log = branch.update_arms(arms, branch.make_optimizers(arms), *values)
    assert log['roi']['loss_structure'] == log['geometry']['loss_structure'] == 0
    assert log['structure']['loss_structure'] > 0
    assert all(v > 0 for v in log['structure']['response_task_gradient_norms'])
    assert all(v > 0 for v in log['structure']['quality_task_gradient_norms'])
    assert all(torch.equal(value, arms['roi'].response.state_dict()[key]) for key, value in before.items())
    assert all(p.grad is None for p in arms['roi'].response.parameters())


def test_no_output_training_and_native_square_angle_supervision():
    values = fixture(genuine=False, native_square=True)
    assert len(values[1]) == 13 and values[6] == 0 and values[5][:, 2].sum() == 13
    arms = branch.make_arms()
    log = branch.update_arms(arms, branch.make_optimizers(arms), *values)
    assert all(r['loss_genuine'] == 0 for r in log.values())
    assert all(r['quality_task_gradient_norms'][2] > 0 for r in log.values())


def test_disconnected_detector_and_no_gt_online_inputs():
    detector = torch.nn.Sequential(torch.nn.Conv2d(3, 256, 1), torch.nn.BatchNorm2d(256))
    core.freeze_detector(detector)
    state = train.prior.state_digest(detector)
    values = list(fixture())
    with torch.no_grad():
        values[0] = detector(torch.ones(1, 3, 8, 8))
    arms = branch.make_arms()
    branch.update_arms(arms, branch.make_optimizers(arms), *values)
    core.assert_detector_frozen(detector)
    assert train.prior.state_digest(detector) == state
    for cls in (branch.GeometryQuality, core.StructureComponentReliability):
        assert set(inspect.signature(cls.forward).parameters) == ({'self', 'p3', 'boxes', 'scores', 'meta'} if cls == branch.GeometryQuality else {'self', 'p3', 'boxes_model', 'scores', 'meta'})
    values[0] = values[0].detach().requires_grad_(True)
    with pytest.raises(ValueError, match='graph'):
        branch.update_arms(arms, branch.make_optimizers(arms), *values)


def test_save_resume_matches_uninterrupted_updates_and_rng(tmp_path):
    values = fixture()
    arms = branch.make_arms()
    optimizers = branch.make_optimizers(arms)
    contract = dict(role='discarded_smoke', fixture='same_views')
    for _ in range(2):
        branch.update_arms(arms, optimizers, *values)
    path = tmp_path/'step_02.pth'
    branch.save_checkpoint(path, arms, optimizers, contract, 0, 2, dict(frozen=True), 'chain')
    random_expected = (np.random.rand(), torch.rand(1))
    for _ in range(2):
        branch.update_arms(arms, optimizers, *values)
    expected = branch.online_qualities(arms, *values[:4])
    restored = branch.make_arms()
    restored_optimizers = branch.make_optimizers(restored)
    payload = branch.read_checkpoint(path, contract)
    branch.load_bundle(payload, restored, restored_optimizers, restore_random=True)
    assert np.random.rand() == random_expected[0]
    assert torch.equal(torch.rand(1), random_expected[1])
    for _ in range(2):
        branch.update_arms(restored, restored_optimizers, *values)
    assert branch.online_qualities(restored, *values[:4]) == expected
    for name in branch.ARMS:
        assert all(torch.equal(v, restored[name].state_dict()[k]) for k, v in arms[name].state_dict().items())
    with pytest.raises(FileExistsError):
        branch.save_checkpoint(path, arms, optimizers, contract, 0, 2, {}, 'chain')
    with pytest.raises(ValueError, match='identical'):
        branch.read_checkpoint(path, dict(contract, fixture='different'))
    with pytest.raises(ValueError, match='epoch8'):
        branch.read_checkpoint(path, for_val=True)


def test_fixed_epoch_and_checkpoint_corruption_rejected(tmp_path):
    arms = branch.make_arms()
    optimizer = branch.make_optimizers(arms)
    for epoch in (0, 7, 8):
        path = tmp_path/('epoch_%02d.pth' % epoch)
        branch.save_checkpoint(path, arms, optimizer, dict(role='formal_train'), epoch, epoch*2558, {}, 'chain')
        if epoch == 8:
            assert branch.read_checkpoint(path, for_val=True)['optimizer_steps_per_arm'] == 20464
        else:
            with pytest.raises(ValueError, match='epoch8'):
                branch.read_checkpoint(path, for_val=True)
    with path.open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='SHA'):
        branch.read_checkpoint(path, for_val=True)


def test_empty_online_outputs_and_equivalent_geometry():
    arms = branch.make_arms()
    p3, boxes, scores, meta = fixture()[:4]
    assert branch.online_qualities(arms, p3, boxes[:0], scores[:0], meta) == {n: [] for n in branch.ARMS}
    exchanged = boxes[:, [0, 1, 3, 2, 4]].clone()
    exchanged[:, 4] += math.pi/2
    assert torch.allclose(branch.geometry_descriptor(boxes, scores, meta),
                          branch.geometry_descriptor(exchanged, scores, meta), atol=1e-6)


def row(index, output=True, center=0., size=0., angle=0., eligible=True):
    return dict(image='real_seq07_%05d' % index, frame_id=index, sequence='real_seq07', domain='real',
        pred=[1., 1., 2., 1., 0., .7] if output else None,
        angle_axis_well_defined=eligible,
        errors=dict(center_px=center, size_max_relative=size, angle_deg=angle) if output else None,
        qualities={name: [.9, .9, .9] if index == 1 else [.1, .1, .1] for name in branch.ARMS} if output else {name: None for name in branch.ARMS})


def test_matched_counts_output_denominators_boundaries_and_rejection_cost():
    rows = [row(1, center=15., size=.10, angle=3.), row(2, size=.11, angle=4.),
            row(3, output=False), row(4, eligible=False)]
    result = branch.compare_component_rankings(rows, (1., .5))['domain:real']
    assert result['output_coverage'] == .75
    assert result['center_hit_rate_on_outputs'] == pytest.approx(2/3)
    assert result['all_frame_center_correct_coverage'] == .5
    center = result['component_curves']['center'][1]
    assert center['matched_actual_accept_count'] == 2
    assert {v['accepted_frames'] for v in center['methods'].values()} == {2}
    assert center['methods']['roi']['incorrect_accepted'] == 1
    assert center['methods']['roi']['correct_rejected'] == 1
    angle = result['component_curves']['angle'][0]['methods']['score']
    assert angle['eligible_frames'] == 3 and angle['output_frames'] == 2
    assert angle['incorrect_accepted'] == 1 and angle['missing_outputs'] == 1
    size = result['component_curves']['size'][0]['methods']['score']
    assert size['incorrect_accepted'] == 1  # exactly 10% is good; exactly 15px is bad
    assert center['methods']['score']['total_at_cutoff'] == 3
    assert center['methods']['score']['accepted_at_cutoff'] == 2
    assert result['component_curves']['center'][0]['methods']['score']['longest_unaccepted_consecutive_frames'] == 1


def test_val_no_probes_training_or_test_path_options_and_protocol_fixed():
    source = inspect.getsource(evaluate)
    assert 'component_probes_original' not in source and 'optimizer.step' not in source
    assert "cfg.data.test" not in source and "--split" not in source
    protocol = json.loads(train.PROTOCOL.read_text())
    assert protocol['epochs'] == 8 and protocol['steps_per_arm'] == 20464
    assert protocol['selection'] == 'fixed_final_epoch_8_no_VAL_checkpoint_selection'
    assert protocol['coverages'] == list(branch.COVERAGES)


def test_actual_augmented_train_pipeline_new_runner_original_targets():
    """Real data pipeline with a fake detector, never real B inference/evidence."""
    from mmcv.utils import import_modules_from_strings
    from mmrotate.datasets import build_dataset
    cfg = train.prior.check_cfg()
    import_modules_from_strings(**cfg.custom_imports)
    specs = train.prior.fixed_specs(cfg, .5, True)
    for spec in specs:
        part = build_dataset(spec)
        index = next(i for i, r in enumerate(part.data_infos) if r['filename'] == (
            'real_seq05_00000.jpg' if spec['ann_file'].startswith('train/') else 'sim_seq08_00374.jpg'))
        name = part.data_infos[index]['filename'][:-4]
        gt = core.canonical_boxes(torch.tensor(part.get_ann_info(index)['bboxes']))[0]
        image_path = train.ready.DATA/('train' if name.startswith('real') else 'train_sim')/'images'/(name+'.jpg')
        from PIL import Image
        with Image.open(image_path) as im:
            size = list(im.size)
        source = dict(image=name, sequence=name.rsplit('_', 1)[0],
            domain=name.split('_', 1)[0], image_size=size, gt_original=gt.tolist())
        axis = train.axis_for(source)
        qualification = direction_qualification(gt, source['domain'], axis if source['domain'] == 'real' else None,
            train.ready.NATIVE_K[source['sequence']] if source['domain'] == 'real' else None)
        source['direction_qualification'] = qualification

        class FakeDetector:
            def extract_feat(self, image):
                return [torch.zeros(1, 256, 128, 128)]

            def simple_test_from_features(self, features, metas, rescale):
                # Pretend CURRENT augmented inference differs from clean B;
                # this constant is a fixture, never a training source/cache.
                pred = gt.clone().reshape(1, 5)
                pred[0, 0] += 20
                model = core.map_boxes(pred, metas[0])
                result = pred if rescale else model
                return [[np.concatenate((result.numpy(), [[.7]]), axis=1)]]

        values, record, _ = train.training_view(FakeDetector(), (part, index), source, -1, 1733)
        assert record['flip'] and values[6] == 1 and record['probes'] == 13
        assert record['genuine_errors_original'][0][0] == pytest.approx(20., abs=1e-3)
        assert record['genuine_targets'][0][0] == pytest.approx(math.exp(-20/15), abs=1e-5)
        assert record['train_angle_mask'] == 1
        assert not values[0].requires_grad and not values[4].requires_grad
