"""Source-only diagnostics of a saved Base V3 score baseline; no fitting."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from crane_project.tools.base_v3_obb_reliability_baseline import (
    PROTOCOL, _identity, _risk_summary, _write_exact)


THRESHOLDS = dict(center_error_px=5.0, long_relative_error=0.1,
                  short_relative_error=0.1, angle_error_deg=3.0)
COVERAGES = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)


def portable_digest(payload):
    """Remove only recorded input locations; retain hashes and all results."""
    portable = copy.deepcopy(payload)
    for identity in portable['inputs'].values():
        if isinstance(identity, dict):
            identity.pop('path', None)
    raw = json.dumps(portable, sort_keys=True, separators=(',', ':'),
                     allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def validate(payload, contract):
    if (payload.get('protocol') != PROTOCOL
            or payload.get('fixed_test_read') is not False
            or payload.get('target_data_read') is not False
            or payload.get('evidence_boundary') !=
            'official_source_val_exploratory_only'):
        raise ValueError('Expected official source-val baseline')
    for role in ('promoted_refiner', 'promotion_report', 'source_gate',
                 'base_v3_results', 'k1_checkpoint', 'k1_results',
                 'source_val_audit'):
        if payload['inputs'][role]['sha256'] != contract['expected_inputs'][
                role + '_sha256']:
            raise ValueError('Input identity mismatch: ' + role)
    rows = payload['records']
    if (len(rows) != contract['source_val_scope']['frame_count']
            or len({r['frame_key'] for r in rows}) != len(rows)):
        raise ValueError('Invalid frame count or duplicate frame keys')
    for domain, count in contract['source_val_scope']['domain_counts'].items():
        if sum(r['domain'] == domain for r in rows) != count:
            raise ValueError('Invalid domain count')
    # Reject NaN/Infinity rather than silently emitting invalid diagnostics.
    json.dumps(payload, allow_nan=False)


def diagnose(rows):
    groups = dict(all=rows)
    groups.update({d: [r for r in rows if r['domain'] == d]
                   for d in ('real', 'sim')})
    groups.update({s: [r for r in rows if r['anchor_source'] == s]
                   for s in ('k1', 'dino_fallback')})
    output = {}
    for name, group in groups.items():
        if not group:
            output[name] = dict(frame_count=0, curves=[])
            continue
        ordered = sorted(group, key=lambda r: (-r['anchor_score'], r['frame_key']))
        curves = []
        for coverage in COVERAGES:
            kept = ordered[:max(1, round(len(ordered) * coverage))]
            curves.append(dict(
                target_coverage=coverage, realized_coverage=len(kept)/len(group),
                summary=_risk_summary(kept),
                domain_coverage={d: sum(r['domain'] == d for r in kept) /
                                 sum(r['domain'] == d for r in group)
                                 for d in ('real', 'sim')
                                 if any(r['domain'] == d for r in group)},
                component_failure_rates={k: sum(r['errors'][k] > t for r in kept)
                                         / len(kept) for k, t in THRESHOLDS.items()}))
        high = ordered[:max(1, round(len(ordered)*0.5))]
        cases = {}
        for key, threshold in THRESHOLDS.items():
            failures = [r for r in high if r['errors'][key] > threshold]
            cases[key] = dict(count=len(failures), cases=[dict(
                frame_key=r['frame_key'], anchor_score=r['anchor_score'],
                anchor_source=r['anchor_source'], errors=r['errors'])
                for r in sorted(failures,
                                key=lambda r: (-r['errors'][key], r['frame_key']))])
        center_good_geometry_bad = [r['frame_key'] for r in group
            if r['errors']['center_error_px'] <= THRESHOLDS['center_error_px']
            and any(r['errors'][k] > t for k, t in THRESHOLDS.items()
                    if k != 'center_error_px')]
        output[name] = dict(frame_count=len(group), curves=curves,
                            high_score_top_half_failures=cases,
                            center_good_geometry_bad=center_good_geometry_bad)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--reference-baseline')
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text())
    contract = json.loads(Path(args.contract).read_text())
    validate(baseline, contract)
    digest = portable_digest(baseline)
    comparison = dict(status='REFERENCE_NOT_SUPPLIED')
    if args.reference_baseline:
        reference = json.loads(Path(args.reference_baseline).read_text())
        validate(reference, contract)
        reference_digest = portable_digest(reference)
        comparison = dict(status='MATCH' if digest == reference_digest else 'MISMATCH',
                          reference=_identity(args.reference_baseline),
                          reference_portable_sha256=reference_digest)
        if digest != reference_digest:
            raise ValueError('Baseline content differs after removing input paths')
    result = dict(protocol='base_v3_reliability_failure_audit_v1',
                  evidence_boundary='official_source_val_exploratory_only',
                  fixed_test_read=False, target_data_read=False,
                  input=_identity(args.baseline), contract=_identity(args.contract),
                  portable_baseline_sha256=digest, comparison=comparison,
                  thresholds=THRESHOLDS,
                  high_score_definition='top_half_within_each_reported_group',
                  interpretation='Thresholds are diagnostic, not deployment gates. '
                  'Domain and GT are audit-only. No calibration or model fitting.',
                  groups=diagnose(baseline['records']))
    identity = _write_exact(args.out_json, result)
    print(json.dumps(dict(output=identity, portable_baseline_sha256=digest,
                          comparison=comparison), indent=2))


if __name__ == '__main__':
    main()
