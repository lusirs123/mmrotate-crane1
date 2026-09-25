"""Focused checks for the proposed source relation probe."""

import math

import numpy as np
import torch

from crane_project.tools.probe_k1_dino_object_background_source_v5 import (
    assess_evidence, balanced_relation_loss, heldout_relation_stats,
    heldout_teacher_gap, object_background_masks, relation_measure,
    select_source_records)


def test_rotated_object_and_ring_stay_inside_image():
    boxes = torch.tensor([[8., 8., 10., 2., math.pi / 2]])
    meta = dict(img_shape=(16, 12, 3), pad_shape=(16, 16, 3))
    obj, bg = object_background_masks(boxes, meta, 16, 16, 'cpu')
    assert obj[4, 8]
    assert not obj[8, 4]
    assert not torch.any(obj & bg)
    assert not torch.any(bg[:, 12:])


def test_heldout_teacher_gap_does_not_use_its_own_object_token():
    obj = torch.zeros((5, 5), dtype=torch.bool)
    obj[2, 1] = obj[2, 2] = True
    bg = torch.zeros_like(obj)
    bg[2, 4] = True
    feat = torch.zeros((2, 5, 5))
    feat[0, 2, 1] = 1
    feat[0, 2, 2] = 1
    feat[1, 2, 4] = 1
    assert heldout_teacher_gap(feat, obj, bg) > 0.9
    assert heldout_relation_stats(feat, obj, bg)['auc'] == 1.0
    feat[0, 2, 2] = 0
    feat[1, 2, 2] = 1
    assert heldout_teacher_gap(feat, obj, bg) == 0.0
    assert heldout_relation_stats(feat, obj, bg)['auc'] == 0.5


def test_relation_loss_reaches_adapter_without_fpn_gradient():
    base = torch.randn((1, 4, 5, 5))
    adapter = torch.nn.Conv2d(4, 4, 1, bias=False)
    torch.nn.init.zeros_(adapter.weight)
    obj = torch.zeros((5, 5), dtype=torch.bool)
    obj[1:3, 1:3] = True
    bg = torch.zeros_like(obj)
    bg[4, 1:4] = True
    relation, _ = relation_measure(
        (base.detach() + adapter(base.detach()))[0], obj, bg)
    loss = relation[obj | bg].square().mean()
    grad = torch.autograd.grad(loss, adapter.weight)[0]
    assert torch.count_nonzero(grad) > 0
    assert base.grad is None


def _assessment_row(domain='real', lost_center=False):
    baseline_detection = dict(output=True, center_hit=True, riou_hit=True)
    control_detection = dict(output=True, center_hit=True, riou_hit=True)
    relation_detection = dict(output=True, center_hit=not lost_center, riou_hit=True)
    return dict(
        role='heldout', domain=domain,
        teacher_heldout=dict(auc=.8, gap=.1),
        control=dict(
            relation_mse=1.0,
            student_heldout=dict(auc=.6, gap=.02),
            classification=dict(score_gap=0., score_object=.5),
            detection=control_detection),
        relation=dict(
            relation_mse=.9,
            student_heldout=dict(auc=.6, gap=.02),
            classification=dict(score_gap=.01, score_object=.5),
            detection=relation_detection),
        baseline=dict(detection=baseline_detection))


def test_assessment_requires_all_domains_and_keeps_training_manual():
    rows = [_assessment_row('real') for _ in range(2)]
    rows += [_assessment_row('sim') for _ in range(2)]
    result = assess_evidence(rows)
    assert result['status'] == 'supports_next_controlled_experiment'
    assert result['automatic_training_authorized'] is False
    assert assess_evidence(rows[:1])['automatic_training_authorized'] is False


def test_assessment_rejects_lost_center_hit():
    rows = [_assessment_row('real') for _ in range(2)]
    rows += [_assessment_row('sim', lost_center=True) for _ in range(2)]
    assert assess_evidence(rows)['status'] == 'no_support_in_this_probe'


def test_relation_loss_balances_object_and_background_counts():
    student = torch.zeros((1, 3))
    teacher = torch.ones_like(student)
    teacher[:, 1:] = 2.
    obj = torch.tensor([[True, False, False]])
    bg = torch.tensor([[False, True, True]])
    assert torch.isclose(
        balanced_relation_loss(student, teacher, obj, bg),
        torch.tensor(2.5))


def test_single_sim_sequence_uses_disclosed_frame_block_fallback():
    infos = []
    for frame in range(8):
        infos.append(dict(
            filename='sim_seq08_{:05d}.jpg'.format(frame),
            ann=dict(bboxes=np.array([[10., 10., 8., 6., 0.]], dtype=np.float32))))
    selected = select_source_records(infos, 'sim', 0)
    assert {row['selection_mode'] for row in selected} == {
        'single_sequence_frame_block_split'}
    assert {row['role'] for row in selected} == {'fit', 'heldout'}
    assert {row['frame_number'] for row in selected if row['role'] == 'fit'} <= set(range(4))
    assert {row['frame_number'] for row in selected if row['role'] == 'heldout'} <= set(range(4, 8))
