#!/usr/bin/env python3
"""Locked source-only lightweight two-frame S7 ranker training."""

import os
import sys
from typing import List

PROJ_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import dino_teacher_rotated_labeller as labeller
from crane_project.tools import (
    dino_teacher_s7_selective_promotion_train as selective_v1)


TRAINING_NAME = 'DINO S7 Lightweight Two-Frame Small-Object Ranker V2'


def parse_args():
    return selective_v1.parse_args()


def validate_args(args):
    # Reuse the phase-2 source-gate and exact selected-checkpoint provenance
    # audit.  The JSON is evidence only and never supplies target examples.
    selective_v1.validate_args(args)


def _replace_value(argv: List[str], option: str, value: str):
    argv[argv.index(option) + 1] = str(value)


def build_locked_labeller_argv(args) -> List[str]:
    argv = selective_v1.build_locked_labeller_argv(args)
    argv.insert(argv.index('--s7-selective-promotion') + 1,
                '--s7-selective-two-frame')
    _replace_value(argv, '--s7-selective-hidden', '16')
    _replace_value(argv, '--s7-selective-max-candidates', '20')
    return argv


def main():
    args = parse_args()
    validate_args(args)
    os.makedirs(args.feature_cache_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    original_argv = sys.argv
    try:
        sys.argv = build_locked_labeller_argv(args)
        labeller.main()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
