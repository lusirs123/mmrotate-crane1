#!/usr/bin/env python3
"""CPU-only fixed sigma1.5 M/simple TRAIN/VAL tradeoff, rejecting TEST."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_reliability_tradeoff_v1 as trade
from crane_project.utils import port_midpoint_sigma15_reliability_v1 as binding
from crane_project.utils import port_simple_component_reliability_v1 as simple

PROTOCOL = ROOT/'crane_project/tools/port_reliability_tradeoff_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_reliability_tradeoff_v1_sources.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda v:
        (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+v)))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def checked_sources():
    p, m = read(PROTOCOL), read(SOURCES)
    parent = ROOT/'crane_project/tools/port_midpoint_sigma15_reliability_v1_sources.json'
    required = set(read(parent)['sources']) | {
        str(parent.relative_to(ROOT)), str(PROTOCOL.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        'crane_project/utils/port_reliability_tradeoff_v1.py',
        'crane_project/utils/port_reliability_separability_v1.py',
        'crane_project/utils/port_midpoint_sigma15_reliability_v1.py',
        'tests/test_port_reliability_tradeoff_v1.py', 'tools/run_port_reliability_tradeoff_v1.sh'}
    actual = {name: sha(ROOT/name) for name in m['sources']}
    if (m['protocol'] != trade.VERSION or p['protocol'] != trade.VERSION
            or set(actual) != required or actual != m['sources'] or m['protocol_sha256'] != sha(PROTOCOL)
            or p['primary_good_retention'] != .95 or p['targets'] != [.90, .95, .99]
            or p['test_read'] is not False or p['parameter_updates'] != 0):
        raise ValueError('Sealed source/protocol mismatch')
    return p, dict(manifest_sha256=sha(SOURCES), sources=actual)


def validate_rows(rows):
    if len({r['image'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image identity')
    if len({(r['sequence'], r['frame_id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate sequence/frame identity')
    counts = {'train': 0, 'val': 0}
    for r in rows:
        role = r['reliability_role']
        if (role not in counts or r['split'] not in
                (('train', 'train_sim') if role == 'train' else ('val',))):
            raise ValueError('TEST or mixed role forbidden')
        if type(r['frame_id']) is not int or r['frame_id'] < 0 or r['domain'] not in ('real', 'sim'):
            raise ValueError('Invalid sequence/domain metadata')
        counts[role] += 1
    if counts != {'train': 2558, 'val': 887}:
        raise ValueError('Requires exact frozen TRAIN2558/VAL887')
    return counts


def checked_inputs(directory, p):
    paths = {n: directory/n for n in p['inputs']}
    actual = {n: sha(path) for n, path in paths.items()}
    if actual != p['inputs']:
        raise ValueError('Frozen input SHA mismatch (do not substitute another cache/policy)')
    collect, fit, policy = [read(paths[n]) for n in
        ('collect/completion.json', 'fit/completion.json', 'fit/policy.json')]
    for receipt, stage, expected in (
            (collect, 'collect', 'SIGMA15_RELIABILITY_COLLECTION_COMPLETE'),
            (fit, 'fit', 'SIGMA15_SIMPLE_FIT_COMPLETE_REVIEW_REQUIRED')):
        if receipt['status'] != expected:
            raise ValueError('Parent stage incomplete')
        if (receipt['protocol'] != binding.VERSION or receipt['test_read'] is not False
                or any(receipt[k] != 0 for k in ('detector_updates','midpoint_updates','reference_updates'))):
            raise ValueError('Parent stage identity mismatch')
        for name, digest in receipt['artifacts'].items():
            relative = stage+'/'+name
            if relative in actual and actual[relative] != digest:
                raise ValueError('Parent artifact receipt mismatch')
    if collect['contract'] != fit['contract'] or policy['contract'] != fit['contract']:
        raise ValueError('Parent contract mismatch')
    parent = ROOT/'crane_project/tools/port_midpoint_sigma15_reliability_v1_sources.json'
    if (collect['contract']['sources']['manifest_sha256'] != sha(parent)
            or collect['contract']['sources']['sources'] != read(parent)['sources']):
        raise ValueError('Parent receipt/source manifest mismatch')
    binding.checked_frontend(policy['front_end'])
    simple.validate_policy(policy['simple_policy'])
    if (policy['protocol'] != binding.VERSION or policy['test_read'] is not False
            or policy['front_end'] != collect['contract']['front_end']):
        raise ValueError('Policy/frontend mismatch')
    rows = lines(paths['collect/final_rows.jsonl'])
    counts = validate_rows(rows)
    stripped_val = [{k: v for k, v in r.items() if k != 'reliability_role'}
                    for r in rows if r['reliability_role'] == 'val']
    if simple.fingerprint(stripped_val) != collect['contract']['final_VAL_fingerprint']:
        raise ValueError('Frozen VAL row fingerprint differs')
    return rows, policy, dict(input_sha256=actual, counts=counts,
        front_end=policy['front_end'], input_paths={n: str(path.resolve()) for n, path in paths.items()}), paths


def verify_frozen_report(report, old, role):
    key = 'TRAIN_in_sample_descriptive' if role == 'train' else 'VAL_calibration_descriptive'
    for name, group in report['strata'].items():
        for component in ('size', 'angle'):
            for method in ('simple', 'score_only'):
                a = group['components'][component]['methods'][method]['fixed_workpoint']
                b = old[key][name]['components'][component][method]
                for metric in ('correct_accepted','incorrect_accepted','correct_rejected','incorrect_rejected'):
                    if a[metric] != b[metric]:
                        raise ValueError('Original report confusion mismatch: '+name+'/'+metric)
                if (a['continuity']['longest_flag_unavailable_observed_consecutive_frames'] !=
                        b['longest_flag_unavailable_all_frames']):
                    raise ValueError('Original report continuity mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    p, sources = checked_sources()
    rows, policy, proof, paths = checked_inputs(args.input_dir, p)
    args.out.mkdir(parents=True, exist_ok=False)
    reports, all_curves = {}, []
    old = read(paths['fit/fit_report.json'])
    for role in ('train', 'val'):
        role_rows = [r for r in rows if r['reliability_role'] == role]
        report, curves, prepared = trade.analyze(role_rows, policy['simple_policy'], role, p['targets'])
        if role == 'val':
            proof['VAL_replay'] = trade.replay_val(prepared, lines(paths['fit/val_decisions.jsonl']))
        verify_frozen_report(report, old, role)
        reports[role] = report
        all_curves.extend(curves)
    if proof['input_sha256'] != {n: sha(path) for n, path in paths.items()}:
        raise ValueError('Input file changed during analysis')
    result = dict(protocol=p, proof=proof, source_proof=sources, reports=reports,
        limitations=p['limitations'], test_read=False, parameter_updates=0, policy_created=False)
    write(args.out/'report.json', result)
    with gzip.open(args.out/'complete_curves.jsonl.gz', 'xt') as f:
        for row in all_curves:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=str(ROOT), text=True).strip()
    write(args.out/'completion.json', dict(protocol=trade.VERSION,
        status='FROZEN_M_TRADEOFF_COMPLETE_REVIEW_REQUIRED', git_commit=commit,
        artifacts={n: sha(args.out/n) for n in ('report.json','complete_curves.jsonl.gz')},
        input_sha256=proof['input_sha256'], source_manifest_sha256=sha(SOURCES),
        parameter_updates=0, detector_inferences=0, test_read=False, policy_created=False,
        curve_points=len(all_curves), frozen_VAL_replay=proof['VAL_replay']))
    for a in reports['val']['retention_assessments']:
        if a['component'] == 'size' and a['target'] == p['primary_good_retention']:
            g = a['groups']['all']
            print(a['method'], a['constraint_scope'], 'FA/FR/ED/CR',
                  [g[k] for k in ('incorrect_accepted','correct_rejected','incorrect_rejected','correct_accepted')],
                  'accepted',g['accepted_assessed_outputs'], flush=True)
    print('FROZEN_M_TRADEOFF_COMPLETE_REVIEW_REQUIRED', args.out, flush=True)


if __name__ == '__main__':
    main()
