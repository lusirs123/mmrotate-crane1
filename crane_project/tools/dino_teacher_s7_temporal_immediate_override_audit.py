#!/usr/bin/env python3
"""Strict recursive source-only immediate-override audit.

The source attribution audit authorized one bounded confirmation-rule check.
This wrapper reuses its locked checkpoint/model validation and changes only the
read-only audit mode to the labeller's recursive immediate-override path.
Target data, training, checkpoint selection, and tunable model knobs remain
blocked by the delegated locked wrapper.
"""

import os
import sys

PROJ_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import dino_teacher_rotated_labeller as labeller
from crane_project.tools import (
    dino_teacher_s7_temporal_source_attribution_audit as attribution)


AUDIT_NAME = 'DINO S7 Temporal Recursive Immediate-Override Audit V1'


def build_locked_immediate_argv(args):
    argv = attribution.build_locked_labeller_argv(args)
    attribution_index = argv.index('--source-temporal-attribution-audit')
    argv[attribution_index] = '--source-temporal-immediate-override-audit'
    return argv


def main():
    args = attribution.parse_args(description=AUDIT_NAME)
    attribution.validate_args(args)
    os.makedirs(os.path.abspath(os.path.dirname(args.out_json)), exist_ok=True)
    os.makedirs(args.feature_cache_dir, exist_ok=True)
    locked_argv = build_locked_immediate_argv(args)
    original_argv = sys.argv
    try:
        sys.argv = locked_argv
        labeller.main()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
