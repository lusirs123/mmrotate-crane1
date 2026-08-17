#!/usr/bin/env python3
"""Re-gate an existing smooth-geometry audit without another model forward.

The protocol-27 artifact already contains the complete frozen source summaries
and frame rows.  Protocol-28 changes only the full/small coverage floors, so
this tool re-evaluates that policy offline and records that no new forward or
parameter update was performed.
"""

import argparse
import copy
import json
import os
import sys
import tempfile


PROJ_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Offline protocol-28 smooth-geometry source re-gate')
    parser.add_argument('--input-json', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--min-gain-domains', type=int, default=2)
    parser.add_argument('--min-gain-sequences', type=int, default=2)
    parser.add_argument('--small-min-gain-domains', type=int, default=1)
    parser.add_argument('--small-min-gain-sequences', type=int, default=1)
    return parser.parse_args()


def validate_args(args):
    if not os.path.isfile(args.input_json):
        raise ValueError('input-json does not exist: {}'.format(args.input_json))
    if os.path.exists(args.out_json):
        raise ValueError('Refusing to overwrite result: {}'.format(args.out_json))
    if any(value <= 0 for value in (
            args.min_gain_domains, args.min_gain_sequences,
            args.small_min_gain_domains, args.small_min_gain_sequences)):
        raise ValueError('All coverage floors must be positive')


def source_support_gate(summary, args, subset):
    if subset == 'small':
        min_domains = int(args.small_min_gain_domains)
        min_sequences = int(args.small_min_gain_sequences)
    elif subset == 'full':
        min_domains = int(args.min_gain_domains)
        min_sequences = int(args.min_gain_sequences)
    else:
        raise ValueError('subset must be full or small')
    domains = list(summary.get('gain_domains') or [])
    sequences = list(summary.get('gain_sequences') or [])
    checks = dict(
        candidate_gain_pair_exists=(
            int(summary.get('native_wrong_s7_correct_pair_count', 0)) > 0),
        minimum_gain_domains=len(domains) >= min_domains,
        minimum_gain_sequences=len(sequences) >= min_sequences)
    return dict(
        passed=bool(all(checks.values())), checks=checks, subset=subset,
        min_gain_domains=min_domains, min_gain_sequences=min_sequences,
        observed_gain_domains=len(domains),
        observed_gain_sequences=len(sequences), gain_domains=domains,
        gain_sequences=sequences,
        coverage_limited=(
            len(domains) < int(args.min_gain_domains)
            or len(sequences) < int(args.min_gain_sequences)))


def build_regated_result(payload, args):
    if int(payload.get('protocol_version', -1)) != 27:
        raise ValueError('Offline re-gate requires protocol-27 input')
    if payload.get('target_dev') is not None:
        raise ValueError('Cannot re-gate an artifact that read target data')
    if int(payload.get('parameter_update_count', -1)) != 0:
        raise ValueError('Cannot re-gate an artifact with parameter updates')
    source = payload.get('source') or {}
    full_summary = source.get('full') or {}
    small_summary = source.get('small') or {}
    full_support = source_support_gate(full_summary, args, subset='full')
    small_support = source_support_gate(small_summary, args, subset='small')
    quality_support = []
    for name in ('sym_kld', 'gwd', 'normalized_gwd'):
        full_metric = (full_summary.get('metrics') or {}).get(name) or {}
        small_metric = (small_summary.get('metrics') or {}).get(name) or {}
        if (int(full_metric.get('net_top1_gain', 0)) > 0
                and int(small_metric.get('net_top1_gain', 0)) > 0
                and int(small_metric.get('top1_gains', 0)) > 0):
            quality_support.append(name)
    support_passed = bool(full_support['passed'] and small_support['passed'])
    quality_passed = bool(quality_support)
    allowed = bool(support_passed and quality_passed)
    coverage_limited = bool(small_support['coverage_limited'])
    decision = (
        'SOURCE_ONLY_SMOOTH_GEOMETRY_RANK_SUPPORT_PASS_COVERAGE_LIMITED_'
        'TARGET_NOT_READ'
        if allowed and coverage_limited else
        'SOURCE_ONLY_SMOOTH_GEOMETRY_RANK_SUPPORT_PASS_TARGET_NOT_READ'
        if allowed else
        'SOURCE_ONLY_SMOOTH_GEOMETRY_RANK_SUPPORT_INSUFFICIENT_TARGET_NOT_READ')

    result = copy.deepcopy(payload)
    result['protocol_version'] = 28
    result['audit_name'] = (
        'Source-only Smooth-Geometry Rank-Support Audit '
        '(offline coverage re-gate)')
    result['protocol'].update(
        protocol_version=28,
        coverage_regate_only=True,
        new_model_forward_count=0,
        feasibility_gate=(
            'source candidate support in full and small subsets with '
            'separate coverage-aware domain/sequence floors, then positive '
            'net top1 gain from at least one smooth metric'))
    result['isolation'].update(
        read_only_evaluation=True,
        parameter_updates_performed=False,
        target_used_for_training=False,
        target_used_for_checkpoint_selection=False,
        target_used_for_threshold_tuning=False)
    result['source']['support_gate'] = dict(
        passed=support_passed, full=full_support, small=small_support,
        coverage_limited=coverage_limited,
        coverage_note=(
            'The source-small split has fewer domains/sequences than the '
            'full-source coverage floor; this permits source-only research '
            'training but not a multi-domain small-object claim.'
            if coverage_limited else None))
    result['source']['quality_gate'] = dict(
        passed=quality_passed, supported_metrics=quality_support,
        requirement='positive net top1 gain in both full and source-small '
                    'source subsets')
    result['source']['training_feasibility'] = dict(
        allowed=allowed,
        reason=(
            'A source-only geometry-guided quality head may be trained; '
            'small coverage is limited and cannot support a multi-domain '
            'generalization claim.' if allowed and coverage_limited else
            'A source-only geometry-guided quality head may be trained.'
            if allowed else
            'Do not start geometry-quality training; source support or the '
            'ranking signal is insufficient.'))
    result['candidate_forward_count'] = int(
        payload.get('candidate_forward_count', 0))
    result['new_model_forward_count'] = 0
    result['parameter_update_count'] = 0
    result['target_dev'] = None
    result['eligible_for_training'] = allowed
    result['eligible_for_deployment'] = False
    result['eligible_for_full_test'] = False
    result['decision'] = decision
    return result


def write_json_atomic(path, payload):
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    fd, temporary = tempfile.mkstemp(
        prefix='.smooth_geometry_regate.', suffix='.tmp', dir=directory)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2,
                      allow_nan=False)
            handle.write('\n')
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return 0


def main():
    args = parse_args()
    validate_args(args)
    with open(args.input_json, 'r') as handle:
        payload = json.load(handle)
    result = build_regated_result(payload, args)
    replacements = write_json_atomic(args.out_json, result)
    print('[smooth-geometry-regate] {}'.format(result['decision']))
    print('[smooth-geometry-regate] new_model_forward_count=0')
    print('[json] nonfinite_replacements={}'.format(replacements))


if __name__ == '__main__':
    main()
