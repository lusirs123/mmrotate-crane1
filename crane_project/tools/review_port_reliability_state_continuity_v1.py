#!/usr/bin/env python3
"""Supplement existing TRAIN/VAL ablation evidence without fitting or selection."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_reliability_state_continuity_v1 as review

PROTOCOL = ROOT/'crane_project/tools/port_reliability_state_continuity_v1_protocol.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def run(input_dir, out):
    for path in (input_dir, out):
        try:
            path.resolve().relative_to(ROOT)
        except ValueError:
            raise ValueError('Inputs and new results must remain in the project work directory')
    if out.exists():
        raise ValueError('Refuse to overwrite supplemental evidence')
    names = ('completion.json', 'fit_report.json', 'candidate_policies.json', 'scored_TRAIN_VAL.jsonl.gz')
    pins = {name: sha(input_dir/name) for name in names}
    receipt = read(input_dir/'completion.json'); report = read(input_dir/'fit_report.json')
    if (receipt['selected_arm'] is not None or report['selected_arm'] is not None
            or receipt['test_read'] or report['test_read']
            or receipt['protocol'] != 'port_reliability_feature_ablation_v1'):
        raise ValueError('Expected sealed, unadopted TRAIN/VAL-only ablation')
    for name in names[1:]:
        if receipt['artifacts'][name] != pins[name]:
            raise ValueError('Sealed artifact hash differs: '+name)
    policies = read(input_dir/'candidate_policies.json')
    with gzip.open(input_dir/'scored_TRAIN_VAL.jsonl.gz', 'rt') as f:
        rows = [json.loads(line) for line in f]
    review.checked_order(rows)
    methods = ('original_simple', 'full_simple_retention95', 'score_retention95')+tuple(sorted(policies))
    results = {}; protections = {}
    for role in ('train', 'val'):
        subset = [r for r in rows if r['reliability_role'] == role]
        if len(subset) != (2558 if role == 'train' else 887):
            raise ValueError('Frozen TRAIN/VAL identity count differs')
        review.checked_order(subset)
        for row in subset:
            base = row['original_simple_decision']
            if (base['final_box_original'] != row['pred'] or
                    base['center_accepted'] != (row['pred'] is not None) or
                    (row['pred'] is None and
                     (base['size_accepted'] or base['angle_accepted'] or
                      any(v is not None for v in base['risks'].values())))):
                raise ValueError('Frozen original box changed')
            for name in policies:
                d = row['candidate_decisions'][name]
                if (d['final_box_original'] != row['pred'] or
                        any(d[k] != base[k] for k in ('center_accepted', 'angle_accepted')) or
                        d['risks']['angle'] != base['risks']['angle']):
                    raise ValueError('Frozen geometry/center/angle changed')
                expected = (row['pred'] is not None and
                            row['size_risks'][name] <= policies[name]['size_cutoff']['risk_le'])
                if d['size_accepted'] != expected:
                    raise ValueError('Saved candidate decision differs from frozen cutoff')
        output = [r for r in subset if r['pred'] is not None]
        hits = sum(sum((r['pred'][j]-r['gt'][j])**2 for j in (0, 1)) < 225 for r in output)
        protections[role] = dict(frames=len(subset), output_frames=len(output), center_hits=hits,
            output_coverage=len(output)/len(subset), center_hit_rate_on_outputs=hits/len(output),
            all_frame_center_correct_coverage=hits/len(subset), candidate_geometry_center_angle_preserved=True)
        groups = {'all': subset}
        for field in ('domain', 'sequence'):
            for value in sorted({r[field] for r in subset}):
                groups[field+':'+value] = [r for r in subset if r[field] == value]
        results[role] = {}
        for group, values in groups.items():
            risks = {name: {r['image']: r['size_risks'][name] for r in values}
                     for name in ('full_simple', 'score_only')}
            decisions = {'original_simple': {r['image'] for r in values
                          if r['original_simple_decision']['size_accepted']}}
            for name, alias in (('full_simple_retention95', 'full_simple'), ('score_retention95', 'score_only')):
                cutoff = report['retention95_VAL_controls'][alias]['all']['global_risk_le']
                decisions[name] = {r['image'] for r in values if r['pred'] is not None
                                   and risks[alias][r['image']] <= cutoff}
            for name in policies:
                decisions[name] = {r['image'] for r in values if r['candidate_decisions'][name]['size_accepted']}
            results[role][group] = {name: review.compare(values, decisions[name], risks) for name in methods}
            # Reproduce all historical VAL confusion, coverage and unavailable counts.
            if role == 'val':
                expected_stats = {'original_simple': report['frozen_legacy_VAL']['simple'][group]}
                expected_stats.update({n: report['candidates'][n]['stats'][group] for n in policies})
                expected_stats.update({n: report['retention95_VAL_controls'][a][group] for n, a in
                                       (('full_simple_retention95', 'full_simple'), ('score_retention95', 'score_only'))})
                for name, old in expected_stats.items():
                    actual = results[role][group][name]['actual']
                    for state, key in (('FA','incorrect_accepted'), ('FR','correct_rejected'),
                                       ('ED','incorrect_rejected'), ('CR','correct_accepted')):
                        if actual['states'][state] != old[key]:
                            raise ValueError('Historical confusion changed')
                    if actual['runs']['unavailable']['longest'] != old['longest_flag_unavailable']:
                        raise ValueError('Historical total-unavailable run changed')
                    for ctrl, old_ctrl in old['matched_controls'].items():
                        current = results[role][group][name]['same_count_controls'][ctrl]
                        if (current['states']['FA'] != old_ctrl['incorrect_accepted'] or
                                current['runs']['unavailable']['longest'] != old_ctrl['longest_flag_unavailable']):
                            raise ValueError('Historical matched control changed')
    if pins != {name: sha(input_dir/name) for name in names}:
        raise ValueError('Source evidence mutated')
    protocol = read(PROTOCOL)
    source_names = ['crane_project/utils/port_reliability_state_continuity_v1.py',
                    str(Path(__file__).resolve().relative_to(ROOT)), str(PROTOCOL.relative_to(ROOT)),
                    'tests/test_port_reliability_state_continuity_v1.py']
    result = dict(protocol=protocol, input_paths={n: str((input_dir/n).resolve()) for n in names},
                  input_sha256=pins, source_sha256={n: sha(ROOT/n) for n in source_names},
                  results=results, protections=protections, historical_VAL_metrics_reproduced=True,
                  historical_selected_arm=None, new_selection_performed=False, test_read=False,
                  training_updates=0, detector_inferences=0, policy_updates=0,
                  limitations=['TRAIN is in-sample; VAL repeatedly developed/calibrated',
                               'Run adjacency uses observed frame indices, not elapsed time',
                               'State labels are offline GT only; no online GT decision',
                               'Same-count controls use image tie order; not deployable thresholds'])
    out.mkdir(parents=True, exist_ok=False)
    write(out/'report.json', result)
    write(out/'completion.json', dict(status='SUPPLEMENTAL_REVIEW_COMPLETE_NO_SELECTION',
        input_sha256=pins, artifacts={'report.json': sha(out/'report.json')}, test_read=False,
        historical_gate_unchanged=True, policy_updates=0))
    for method, value in results['val']['all'].items():
        a = value['actual']
        print(method, a['states'], 'longest FA/FR/ED/missing/unavailable',
              [a['runs'][k]['longest'] for k in ('incorrect_acceptance', 'correct_rejection',
                                               'error_detection', 'missing_output', 'unavailable')])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=ROOT/'work_dirs/port_reliability_feature_ablation_v1/20261008_size_features_v1/fit')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); run(a.input_dir, a.out)


if __name__ == '__main__':
    main()
