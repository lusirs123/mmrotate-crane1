"""Behavioural tests for the actual top-1 selection decision."""

import importlib.util
from pathlib import Path

import torch


MODULE = Path(__file__).resolve().parents[1] / (
    'mmrotate/models/losses/candidate_selection.py')
spec = importlib.util.spec_from_file_location('candidate_selection', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
candidate_selection_loss = module.candidate_selection_loss


def test_threshold_failure_has_gradient_toward_good_candidate():
    logits = torch.tensor([-4.0, -6.0], requires_grad=True)
    ious = torch.tensor([0.8, 0.1])
    loss, good, bad = candidate_selection_loss(logits, ious)
    assert (good, bad) == (0, 1)
    loss.backward()
    assert logits.grad[0] < 0
    assert logits.grad[1] == 0


def test_wrong_top1_pushes_good_up_and_bad_down():
    logits = torch.tensor([0.0, 1.0], requires_grad=True)
    ious = torch.tensor([0.9, 0.2], requires_grad=True)
    loss, good, bad = candidate_selection_loss(logits, ious)
    assert (good, bad) == (0, 1)
    loss.backward()
    assert logits.grad[0] < 0 < logits.grad[1]
    assert ious.grad is None


def test_success_and_missing_good_need_no_added_gradient():
    logits = torch.tensor([2.0, -1.0], requires_grad=True)
    success, _, _ = candidate_selection_loss(logits, torch.tensor([0.8, 0.1]))
    missing, good, bad = candidate_selection_loss(logits, torch.tensor([0.2, 0.1]))
    assert success.item() == 0
    assert missing.item() == 0 and good is None and bad is None
    (success + missing).backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_empty_candidate_is_safe_and_only_good_competes_with_threshold():
    empty = torch.empty(0, requires_grad=True)
    loss, good, bad = candidate_selection_loss(empty, torch.empty(0))
    assert (good, bad) == (None, None)
    loss.backward()
    logits = torch.tensor([-4.0], requires_grad=True)
    loss, good, bad = candidate_selection_loss(logits, torch.tensor([0.7]))
    assert (good, bad) == (0, None) and loss.item() > 0
