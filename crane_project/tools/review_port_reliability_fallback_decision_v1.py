#!/usr/bin/env python3
"""Independent frozen-decision, counts, matching and continuity audit."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_reliability_fallback_decision_v1 as runner
from crane_project.tools.review_port_reliability_temporal_fusion_v1 import check_summary, rank_ids, bad
from crane_project.tools.review_port_reliability_image_temporal_crossfit_v1 import equivalent
core = runner.core


def review(result, allow_absent=False):
    p, pins, absent, cuts = runner.load(allow_absent)
    report = json.loads((result / 'report.json').read_text()); checks = total = 0
    if report['input_pins'] != pins or report['frozen_cutoffs'] != cuts:
        raise ValueError('Report contract differs')
    for role in ('TRAIN', 'VAL'):
        saved = runner.io.read(result / ('scored_' + role + '.jsonl.gz'))
        original = runner.io.read(ROOT / p['crossfit_result'] / ('scored_' + role + '.jsonl.gz'))
        if len(saved) != len(original):
            raise ValueError('Frame coverage changed')
        ids = set()
        for r, old in zip(saved, original):
            if {k: v for k, v in r.items() if k != core.METHOD} != old:
                raise ValueError('Frozen frame/risk/geometry changed')
            available = old['temporal']['available']
            method = 'crossfit_fusion' if available else 'partial_auc4'
            threshold = cuts['supported'] if available else cuts['fallback']
            expected = old['pred'] is not None and old['risks'][method] <= threshold
            if r[core.METHOD] != expected:
                raise ValueError('Branch decision differs')
            if expected: ids.add(r['image'])
            total += 1; checks += 1
        for group, part in core.prior.temporal.base.level.metrics.grouped(saved).items():
            entry = report['roles'][role]['groups'][group]
            selected = ids & {r['image'] for r in part}
            check_summary(part, selected, entry['candidate']); checks += 1
            for method, matches in entry['controls'].items():
                for kind, data in matches.items():
                    ranked = rank_ids(part, method, count=len(selected)) if kind == 'same_count' else rank_ids(part, method, correct=entry['candidate']['states']['CR'])
                    check_summary(part, ranked, data); checks += 1
            for method, summary in entry['frozen_actual'].items():
                cutoff = cuts['supported'] if method == 'crossfit_fusion' else cuts['fallback']
                accepted = {r['image'] for r in part if r['pred'] is not None and r['risks'][method] <= cutoff}
                check_summary(part, accepted, summary); checks += 1
            for method, changes in entry['changes'].items():
                cutoff = cuts['supported'] if method == 'crossfit_fusion' else cuts['fallback']
                expected_changes = [dict(image=r['image'], history_available=r['temporal']['available'], wrong=bad(r),
                    before=r['pred'] is not None and r['risks'][method] <= cutoff, after=r['image'] in selected)
                    for r in part if (r['pred'] is not None and r['risks'][method] <= cutoff) != (r['image'] in selected)]
                if changes != expected_changes: raise ValueError('Change accounting differs')
                checks += 1
        if not equivalent(report['roles'][role], core.evaluate(saved, cuts['supported'], cuts['fallback'])):
            raise ValueError('Full metrics/center/regime/ties differ')
    normalize = lambda g: dict(g, failures=sorted(g['failures'], key=lambda x: json.dumps(x, sort_keys=True)))
    if not equivalent(normalize(report['gate']), normalize(core.gate(report['roles']['VAL']))):
        raise ValueError('Gate differs')
    expected_status = 'EVIDENCE_PASS_NO_ADOPTION' if report['gate']['passed'] else 'EVIDENCE_FAILED_STOP'
    if report['protocol'] != core.VERSION or report['status'] != expected_status or not report['VAL_previously_exposed'] or not report['design_informed_by_exposed_VAL']:
        raise ValueError('Version/status/exposed-data disclosure differs')
    for key in ('fits', 'quality_updates', 'calibration_calls', 'detector_inferences', 'detector_updates', 'TEST_opened', 'formal_policy_changed', 'candidate_adopted'):
        if report[key]: raise ValueError('Unauthorized scope: ' + key)
    return dict(passed=True, rows=total, independent_checks=checks,
        absent_frontend=absent, report_sha256=runner.io.sha(result / 'report.json'),
        supported_decision_identity=True, unsupported_partial_decision_identity=True, TEST_opened=False)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--result', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--allow-absent-frontend', action='store_true'); a = p.parse_args()
    runner.io.write(a.out, review(a.result, a.allow_absent_frontend)); print('Independent fallback decision review passed')


if __name__ == '__main__': main()
