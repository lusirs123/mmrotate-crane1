import json
import sys

import pytest

from crane_project.tools import dino_quality_ranking_complementarity_audit_v1 as audit


def write(path, payload):
    path.write_text(json.dumps(payload), encoding='utf-8')


def baseline_payload():
    return dict(protocol=dict(target_used_for_training=False), source=dict(
        candidates=[dict(alpha=0.5, frame_outcomes=[
            dict(seq='real_a', frame=1, top1_hit=True, top1_riou=.7),
            dict(seq='real_a', frame=2, top1_hit=False, top1_riou=.2),
            dict(seq='sim_b', frame=3, top1_hit=False, top1_riou=.1)])]))


def method_payload(epoch, lost, gained):
    return dict(target_dev=None, source=dict(best_epoch=epoch, history=[dict(
        epoch=epoch, source_exact_retention=dict(
            baseline_correct_count=1,
            retained_correct_count=1 - len(lost),
            lost_correct_count=len(lost),
            gained_correct_count=len(gained),
            candidate_correct_count=1 - len(lost) + len(gained),
            lost_frame_keys=lost, gained_frame_keys=gained))]))


def contract(baseline, first, second):
    return dict(
        protocol=audit.CONTRACT_PROTOCOL,
        evidence_role='source_validation_only', expected_frame_count=3,
        baseline=dict(
            name='native', path=str(baseline),
            selector=dict(kind='candidate_alpha', value=.5)),
        methods=[
            dict(name='m1', path=str(first),
                 selector=dict(kind='history_epoch', value=1)),
            dict(name='m2', path=str(second),
                 selector=dict(kind='best_epoch'))])


def test_build_report_counts_unique_and_shared_gains(tmp_path):
    base = tmp_path / 'base.json'
    first = tmp_path / 'first.json'
    second = tmp_path / 'second.json'
    write(base, baseline_payload())
    write(first, method_payload(1, [], ['val|real_a|2']))
    write(second, method_payload(2, [], ['val|sim_b|3']))
    report = audit.build_report(contract(base, first, second), tmp_path)
    assert report['baseline']['top1_hit_count'] == 1
    assert [row['top1_hit_count'] for row in report['methods']] == [2, 2]
    pair = report['pairwise_gain_complementarity'][0]
    assert pair['shared_gain_count'] == 0
    assert pair['left_only_gain_count'] == 1
    assert pair['right_only_gain_count'] == 1
    assert report['oracle_diagnostic']['oracle_hit_count'] == 3
    assert report['decision'] == (
        'COMPLEMENTARY_GAINS_PRESENT_SELECTOR_NOT_AUTHORIZED')


def test_rejects_transition_that_disagrees_with_baseline(tmp_path):
    base = tmp_path / 'base.json'
    first = tmp_path / 'first.json'
    second = tmp_path / 'second.json'
    write(base, baseline_payload())
    write(first, method_payload(1, [], ['val|real_a|1']))
    write(second, method_payload(2, [], ['val|sim_b|3']))
    with pytest.raises(ValueError, match='disagrees with baseline'):
        audit.build_report(contract(base, first, second), tmp_path)


def test_rejects_target_dependent_result(tmp_path):
    base = tmp_path / 'base.json'
    first = tmp_path / 'first.json'
    second = tmp_path / 'second.json'
    write(base, baseline_payload())
    payload = method_payload(1, [], ['val|real_a|2'])
    payload['target_dev'] = {'summary': {}}
    write(first, payload)
    write(second, method_payload(2, [], ['val|sim_b|3']))
    with pytest.raises(ValueError, match='target-dependent'):
        audit.build_report(contract(base, first, second), tmp_path)


def test_missing_input_reports_same_named_candidate(tmp_path):
    base = tmp_path / 'base.json'
    first = tmp_path / 'expected' / 'method.json'
    actual = tmp_path / 'work_dirs' / 'actual' / 'method.json'
    second = tmp_path / 'second.json'
    actual.parent.mkdir(parents=True)
    write(base, baseline_payload())
    write(actual, method_payload(1, [], ['val|real_a|2']))
    write(second, method_payload(2, [], ['val|sim_b|3']))
    with pytest.raises(FileNotFoundError, match='work_dirs/actual/method.json'):
        audit.build_report(contract(base, first, second), tmp_path)


def test_full_frame_outcomes_must_align(tmp_path):
    base = tmp_path / 'base.json'
    first = tmp_path / 'first.json'
    second = tmp_path / 'second.json'
    write(base, baseline_payload())
    payload = baseline_payload()
    payload['source']['history'] = [dict(
        epoch=1, frame_outcomes=payload['source']['candidates'][0][
            'frame_outcomes'][:-1])]
    write(first, payload)
    write(second, method_payload(2, [], ['val|sim_b|3']))
    spec = contract(base, first, second)
    spec['methods'][0]['selector'] = dict(kind='history_epoch', value=1)
    with pytest.raises(ValueError, match='frame set differs'):
        audit.build_report(spec, tmp_path)


def test_rejects_transition_count_that_disagrees_with_frame_keys(tmp_path):
    base = tmp_path / 'base.json'
    first = tmp_path / 'first.json'
    second = tmp_path / 'second.json'
    write(base, baseline_payload())
    payload = method_payload(1, [], ['val|real_a|2'])
    payload['source']['history'][0]['source_exact_retention'][
        'gained_correct_count'] = 2
    write(first, payload)
    write(second, method_payload(2, [], ['val|sim_b|3']))
    with pytest.raises(ValueError, match='reconstructed value is 1'):
        audit.build_report(contract(base, first, second), tmp_path)


def test_requires_two_distinct_method_names(tmp_path):
    base = tmp_path / 'base.json'
    first = tmp_path / 'first.json'
    second = tmp_path / 'second.json'
    write(base, baseline_payload())
    write(first, method_payload(1, [], ['val|real_a|2']))
    write(second, method_payload(2, [], ['val|sim_b|3']))
    spec = contract(base, first, second)
    spec['methods'][1]['name'] = 'm1'
    with pytest.raises(ValueError, match='must be unique'):
        audit.build_report(spec, tmp_path)
    spec['methods'] = spec['methods'][:1]
    with pytest.raises(ValueError, match='at least two methods'):
        audit.build_report(spec, tmp_path)


def test_cli_writes_bound_json_and_markdown(tmp_path, monkeypatch):
    base = tmp_path / 'base.json'
    first = tmp_path / 'first.json'
    second = tmp_path / 'second.json'
    contract_path = tmp_path / 'contract.json'
    out_json = tmp_path / 'out' / 'audit.json'
    out_md = tmp_path / 'out' / 'audit.md'
    write(base, baseline_payload())
    write(first, method_payload(1, [], ['val|real_a|2']))
    write(second, method_payload(2, [], ['val|sim_b|3']))
    write(contract_path, contract(base, first, second))
    monkeypatch.setattr(sys, 'argv', [
        'audit', '--contract', str(contract_path),
        '--repo-root', str(tmp_path), '--out-json', str(out_json),
        '--out-md', str(out_md)])
    audit.main()
    report = json.loads(out_json.read_text(encoding='utf-8'))
    assert report['contract']['sha256'] == audit.file_sha256(contract_path)
    assert report['oracle_diagnostic']['oracle_hit_count'] == 3
    rendered = out_md.read_text(encoding='utf-8')
    assert 'Transitions by sequence' in rendered
    assert 'runtime selector' in rendered
    out_json.write_text('{}\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='Refusing to overwrite'):
        audit.main()
