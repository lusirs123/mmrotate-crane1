"""Read-only frame complementarity audit for frozen DINO ranking results."""

import argparse
import hashlib
import json
from pathlib import Path


PROTOCOL = 'dino_quality_ranking_complementarity_audit_v1'
CONTRACT_PROTOCOL = '{}_contract'.format(PROTOCOL)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_key(value):
    if isinstance(value, str):
        parts = value.split('|')
        if len(parts) >= 2:
            return '{}|{}'.format(parts[-2], int(parts[-1]))
        raise ValueError('Frame key must contain sequence and frame: {!r}'.format(value))
    return '{}|{}'.format(value['seq'], int(value['frame']))


def select_node(payload, selector):
    kind = selector['kind']
    if kind == 'candidate_alpha':
        rows = payload['source']['candidates']
        matches = [row for row in rows
                   if abs(float(row['alpha']) - float(selector['value'])) <= 1e-12]
    elif kind == 'history_epoch':
        rows = payload['source']['history']
        matches = [row for row in rows
                   if int(row['epoch']) == int(selector['value'])]
    elif kind == 'best_epoch':
        rows = payload['source']['history']
        best = int(payload['source']['best_epoch'])
        matches = [row for row in rows if int(row['epoch']) == best]
    else:
        raise ValueError('Unsupported selector kind: {}'.format(kind))
    if len(matches) != 1:
        raise ValueError('Selector {} matched {} nodes'.format(selector, len(matches)))
    return matches[0]


def recursive_dicts(value, path='selected'):
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from recursive_dicts(child, '{}.{}'.format(path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from recursive_dicts(child, '{}[{}]'.format(path, index))


def find_frame_outcomes(node):
    found = []
    for path, value in recursive_dicts(node):
        if not value:
            continue
        for key in ('frame_outcomes', 'outcomes'):
            rows = value.get(key)
            if (isinstance(rows, list) and rows
                    and isinstance(rows[0], dict)
                    and {'seq', 'frame'}.issubset(rows[0])
                    and ('top1_hit' in rows[0] or 'top1' in rows[0])):
                found.append(('{}.{}'.format(path, key), rows))
    if len(found) > 1:
        raise ValueError('Multiple frame outcome lists found: {}'.format(
            [path for path, _ in found]))
    return found[0] if found else (None, None)


def find_transition(node):
    found = []
    for path, value in recursive_dicts(node):
        lost_key = next((key for key in (
            'lost_frame_keys', 'newly_incorrect_frame_keys') if key in value), None)
        gained_key = next((key for key in (
            'gained_frame_keys', 'newly_correct_frame_keys') if key in value), None)
        if lost_key and gained_key:
            found.append((path, value[lost_key], value[gained_key], value))
    if len(found) > 1:
        exact = [item for item in found if item[0].endswith('source_exact_retention')]
        found = exact if len(exact) == 1 else found
    if len(found) != 1:
        raise ValueError('Expected one lost/gained transition record, found {}'.format(
            [item[0] for item in found]))
    return found[0]


def normalize_frame_rows(rows):
    result = {}
    for row in rows:
        key = canonical_key(row)
        if key in result:
            raise ValueError('Duplicate frame outcome: {}'.format(key))
        hit = row.get('top1_hit', row.get('top1'))
        result[key] = dict(
            hit=bool(hit), seq=str(row['seq']), frame=int(row['frame']),
            riou=(None if row.get('top1_riou') is None
                   else float(row['top1_riou'])),
            score=(None if row.get('top1_score') is None
                    else float(row['top1_score'])),
            best_usable_rank=row.get('best_usable_rank'))
    return result


def validate_transition_counts(name, transition, baseline_hit_count,
                               gained_count, lost_count):
    aliases = (
        (('baseline_correct_count',), baseline_hit_count),
        (('retained_correct_count',), baseline_hit_count - lost_count),
        (('lost_correct_count',), lost_count),
        (('gained_correct_count', 'newly_correct_count'), gained_count),
        (('candidate_correct_count',),
         baseline_hit_count - lost_count + gained_count))
    for keys, expected in aliases:
        present = [key for key in keys if key in transition]
        if len(present) > 1:
            values = {int(transition[key]) for key in present}
            if len(values) != 1:
                raise ValueError(
                    '{} transition has conflicting count aliases {}'.format(
                        name, present))
        for key in present:
            actual = int(transition[key])
            if actual != expected:
                raise ValueError(
                    '{} transition {}={} but reconstructed value is {}'.format(
                        name, key, actual, expected))


def input_uses_target(payload):
    if payload.get('target_dev') is not None:
        return True
    protocol = payload.get('protocol', {})
    isolation = payload.get('isolation', {})
    flags = []
    for container in (protocol, isolation):
        if isinstance(container, dict):
            flags.extend(container.get(key) for key in (
                'target_used_for_training',
                'target_used_for_checkpoint_selection') if key in container)
    return any(value is True for value in flags)


def load_entry(spec, repo_root, baseline=None):
    path = Path(spec['path'])
    if not path.is_absolute():
        path = Path(repo_root) / path
    if not path.is_file():
        candidates = sorted(
            candidate.resolve() for candidate in
            Path(repo_root).glob('work_dirs/**/{}'.format(path.name))
            if candidate.is_file())
        hint = ''
        if candidates:
            hint = ' Existing files with the same name: {}'.format(
                ', '.join(str(candidate) for candidate in candidates[:10]))
        raise FileNotFoundError(
            'Contract input for {} does not exist: {}.{}'.format(
                spec['name'], path, hint))
    payload = json.loads(path.read_text(encoding='utf-8'))
    if input_uses_target(payload):
        raise ValueError('{} contains target-dependent evidence'.format(spec['name']))
    node = select_node(payload, spec['selector'])
    outcome_path, outcome_rows = find_frame_outcomes(node)
    record = dict(
        name=spec['name'], path=str(path.resolve()), sha256=file_sha256(path),
        selector=spec['selector'], source_protocol=payload.get('protocol'),
        extraction=None, rows=None, gains=set(), losses=set())
    if outcome_rows is not None:
        rows = normalize_frame_rows(outcome_rows)
        record['extraction'] = outcome_path
        record['rows'] = rows
        if baseline is not None:
            if set(rows) != set(baseline['rows']):
                raise ValueError('{} frame set differs from baseline'.format(spec['name']))
            record['gains'] = {key for key in rows
                               if rows[key]['hit'] and not baseline['rows'][key]['hit']}
            record['losses'] = {key for key in rows
                                if baseline['rows'][key]['hit'] and not rows[key]['hit']}
    else:
        if baseline is None:
            raise ValueError('Baseline must provide full frame outcomes')
        path_name, lost, gained, transition = find_transition(node)
        record['extraction'] = path_name
        gained_keys = [canonical_key(key) for key in gained]
        lost_keys = [canonical_key(key) for key in lost]
        if len(gained_keys) != len(set(gained_keys)):
            raise ValueError('{} transition repeats gained frames'.format(
                spec['name']))
        if len(lost_keys) != len(set(lost_keys)):
            raise ValueError('{} transition repeats lost frames'.format(
                spec['name']))
        record['gains'] = set(gained_keys)
        record['losses'] = set(lost_keys)
        if record['gains'] & record['losses']:
            raise ValueError('{} transition both gains and loses a frame'.format(
                spec['name']))
        unknown = (record['gains'] | record['losses']) - set(baseline['rows'])
        if unknown:
            raise ValueError('{} transition has unknown frames: {}'.format(
                spec['name'], sorted(unknown)[:5]))
        invalid_gain = {key for key in record['gains'] if baseline['rows'][key]['hit']}
        invalid_loss = {key for key in record['losses'] if not baseline['rows'][key]['hit']}
        if invalid_gain or invalid_loss:
            raise ValueError('{} transition disagrees with baseline'.format(spec['name']))
        baseline_hits = sum(row['hit'] for row in baseline['rows'].values())
        validate_transition_counts(
            spec['name'], transition, baseline_hits,
            len(record['gains']), len(record['losses']))
        record['transition_record'] = transition
    return record


def by_sequence(keys):
    counts = {}
    for key in keys:
        seq = key.rsplit('|', 1)[0]
        counts[seq] = counts.get(seq, 0) + 1
    return dict(sorted(counts.items()))


def public_method(record, baseline_hits, frame_count):
    hit_count = baseline_hits + len(record['gains']) - len(record['losses'])
    return dict(
        name=record['name'], input=dict(
            path=record['path'], sha256=record['sha256'],
            selector=record['selector'], extraction=record['extraction']),
        frame_count=frame_count, top1_hit_count=hit_count,
        gained_count=len(record['gains']), lost_count=len(record['losses']),
        net_gain=hit_count - baseline_hits,
        gained_frame_keys=sorted(record['gains']),
        lost_frame_keys=sorted(record['losses']),
        gained_by_sequence=by_sequence(record['gains']),
        lost_by_sequence=by_sequence(record['losses']))


def pairwise(methods):
    result = []
    for index, left in enumerate(methods):
        for right in methods[index + 1:]:
            union = left['gains'] | right['gains']
            intersection = left['gains'] & right['gains']
            result.append(dict(
                left=left['name'], right=right['name'],
                shared_gain_count=len(intersection),
                left_only_gain_count=len(left['gains'] - right['gains']),
                right_only_gain_count=len(right['gains'] - left['gains']),
                gain_union_count=len(union),
                gain_jaccard=(float(len(intersection) / len(union))
                              if union else 1.0),
                shared_gain_frame_keys=sorted(intersection),
                left_only_gain_frame_keys=sorted(left['gains'] - right['gains']),
                right_only_gain_frame_keys=sorted(right['gains'] - left['gains'])))
    return result


def has_unique_gains(methods):
    for row in methods:
        other_gains = set().union(*(
            other['gains'] for other in methods if other is not row))
        if row['gains'] - other_gains:
            return True
    return False


def build_report(contract, repo_root):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected contract protocol')
    if contract.get('evidence_role') != 'source_validation_only':
        raise ValueError('Complementarity audit is source-validation only')
    baseline = load_entry(contract['baseline'], repo_root)
    baseline_hits = sum(row['hit'] for row in baseline['rows'].values())
    expected = contract.get('expected_frame_count')
    if expected is not None and len(baseline['rows']) != int(expected):
        raise ValueError('Baseline frame count does not match contract')
    method_specs = contract.get('methods')
    if not isinstance(method_specs, list) or len(method_specs) < 2:
        raise ValueError('Complementarity audit requires at least two methods')
    methods = [load_entry(spec, repo_root, baseline)
               for spec in method_specs]
    names = [row['name'] for row in methods]
    if len(names) != len(set(names)):
        raise ValueError('Method names must be unique')
    if baseline['name'] in names:
        raise ValueError('Method names must differ from baseline name')
    union_gains = set().union(*(row['gains'] for row in methods))
    union_losses = set().union(*(row['losses'] for row in methods))
    baseline_public = public_method(
        baseline, baseline_hits, len(baseline['rows']))
    baseline_public.update(gained_count=0, lost_count=0, net_gain=0,
                           gained_frame_keys=[], lost_frame_keys=[],
                           gained_by_sequence={}, lost_by_sequence={})
    return dict(
        protocol=PROTOCOL, protocol_version=1,
        evidence_boundary=dict(
            role='source_validation_only', exposed_target_read=False,
            optimizer_steps=0, checkpoint_writes=0,
            oracle_is_gt_dependent=True,
            oracle_authorizes_inference_rule=False,
            threshold_or_checkpoint_selection_performed=False),
        baseline=baseline_public,
        methods=[public_method(row, baseline_hits, len(baseline['rows']))
                 for row in methods],
        pairwise_gain_complementarity=pairwise(methods),
        oracle_diagnostic=dict(
            definition='baseline_or_any_method_top1_hit_using_source_GT',
            baseline_hit_count=baseline_hits,
            union_gain_count=len(union_gains),
            oracle_hit_count=baseline_hits + len(union_gains),
            oracle_hit_rate=float(
                (baseline_hits + len(union_gains)) / len(baseline['rows'])),
            any_method_loss_frame_count=len(union_losses),
            union_gain_frame_keys=sorted(union_gains),
            any_method_loss_frame_keys=sorted(union_losses)),
        selector_evidence=dict(
            status='OUTCOME_COMPLEMENTARITY_ONLY',
            gt_free_selector_validated=False,
            reason=(
                'Gain/loss overlap can show complementarity, but outcome '
                'labels alone cannot define a deployable method selector.')),
        decision=(
            'COMPLEMENTARY_GAINS_PRESENT_SELECTOR_NOT_AUTHORIZED'
            if has_unique_gains(methods) else
            'NO_UNIQUE_COMPLEMENTARY_GAINS_SELECTOR_NOT_AUTHORIZED'))


def markdown(report):
    lines = [
        '# DINO quality-ranking frame complementarity audit', '',
        '- Protocol: `{}`'.format(report['protocol']),
        '- Decision: `{}`'.format(report['decision']),
        '- Evidence: source validation only; no training or target read.', '',
        '| Method | Top-1 | Gain | Loss | Net |',
        '| --- | ---: | ---: | ---: | ---: |']
    baseline = report['baseline']
    lines.append('| {} | {} | 0 | 0 | 0 |'.format(
        baseline['name'], baseline['top1_hit_count']))
    for row in report['methods']:
        lines.append('| {} | {} | {} | {} | {} |'.format(
            row['name'], row['top1_hit_count'], row['gained_count'],
            row['lost_count'], row['net_gain']))
    lines += ['', '## Pairwise gain complementarity', '',
              '| Left | Right | Shared | Left only | Right only | Jaccard |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in report['pairwise_gain_complementarity']:
        lines.append('| {} | {} | {} | {} | {} | {:.6f} |'.format(
            row['left'], row['right'], row['shared_gain_count'],
            row['left_only_gain_count'], row['right_only_gain_count'],
            row['gain_jaccard']))
    lines += ['', '## Transitions by sequence', '']
    for row in report['methods']:
        lines += [
            '### {}'.format(row['name']), '',
            '- Gains: `{}`'.format(json.dumps(
                row['gained_by_sequence'], ensure_ascii=False,
                sort_keys=True)),
            '- Losses: `{}`'.format(json.dumps(
                row['lost_by_sequence'], ensure_ascii=False,
                sort_keys=True)), '']
    oracle = report['oracle_diagnostic']
    lines += ['', '## GT-dependent diagnostic upper bound', '',
              '- Baseline: `{}`'.format(oracle['baseline_hit_count']),
              '- Union gains: `{}`'.format(oracle['union_gain_count']),
              '- Oracle Top-1: `{}`'.format(oracle['oracle_hit_count']), '',
              'This upper bound uses source GT and does not authorize a runtime selector.', '']
    return '\n'.join(lines)


def write_exact_outputs(outputs):
    for path, text in outputs:
        path = Path(path)
        if path.exists() and path.read_text(encoding='utf-8') != text:
            raise RuntimeError(
                'Refusing to overwrite different output: {}'.format(path))
    for path, text in outputs:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', required=True)
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--out-md', required=True)
    args = parser.parse_args()
    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    report = build_report(contract, Path(args.repo_root).resolve())
    report['contract'] = dict(
        path=str(contract_path.resolve()), sha256=file_sha256(contract_path))
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    json_text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    markdown_text = markdown(report)
    write_exact_outputs(((out_json, json_text), (out_md, markdown_text)))
    print(json.dumps(dict(
        decision=report['decision'], out_json=str(out_json.resolve()),
        out_md=str(out_md.resolve()),
        out_json_sha256=file_sha256(out_json),
        out_md_sha256=file_sha256(out_md),
        oracle_hit_count=report['oracle_diagnostic']['oracle_hit_count']),
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
