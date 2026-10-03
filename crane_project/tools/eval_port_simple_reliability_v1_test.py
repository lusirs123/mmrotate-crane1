#!/usr/bin/env python3
"""CPU-only fixed simple-v1 TEST closure from reviewed server numeric records.

No detector/head imports, inference, fitting, threshold/weight selection or GT
reconstruction. Historical ROI/structure qualities are discarded. Both the
policy and cache must match the previously reviewed exact bytes.
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_simple_reliability_v1 as parent
from crane_project.utils import port_simple_component_reliability_v1 as simple

VERSION = 'port_simple_component_reliability_v1_fixed_test'
PROTOCOL = ROOT/'crane_project/tools/port_simple_reliability_v1_test_protocol.json'
MANIFEST = ROOT/'crane_project/tools/port_simple_reliability_v1_test_sources.json'
FIELDS = ('image', 'sequence', 'frame_id', 'domain', 'split', 'gt', 'pred',
          'image_size', 'angle_axis_well_defined', 'image_sha256', 'annotation_sha256')


def checked_sources():
    original, original_sources = parent.checked_sources()
    contract = json.loads(PROTOCOL.read_text())
    manifest = json.loads(MANIFEST.read_text())
    current = {name: parent.sha(ROOT/name) for name in manifest['sources']}
    if (manifest['protocol'] != VERSION or contract['protocol'] != VERSION
            or current != manifest['sources'] or contract['parent_protocol'] != original
            or contract['parent_manifest_sha256'] != original_sources['manifest_sha256']
            or contract['test_used_for_selection'] is not False
            or contract['test_repeatedly_exposed'] is not True
            or contract['methods'] != ['raw', 'score_only', 'simple']):
        raise ValueError('Fixed TEST or unchanged simple-v1 source contract differs')
    return contract, dict(manifest_sha256=parent.sha(MANIFEST), sources=current,
                         parent=original_sources)


def exact_bundle(directory, hashes):
    directory = Path(directory)
    if (directory/'failure.json').exists():
        raise ValueError('Failed bundle cannot be used: '+str(directory))
    actual = {name: parent.sha(directory/name) for name in hashes}
    if actual != hashes:
        raise ValueError('Requires exact reviewed artifact bytes: '+str(
            [name for name in hashes if actual[name] != hashes[name]]))
    return actual


def validate_rows(rows, counts):
    images = [r['image'] for r in rows]
    if (not rows or len(set(images)) != len(rows) or images != sorted(images)
            or Counter(r['sequence'] for r in rows) != Counter(counts)):
        raise ValueError('Wrong fixed TEST order/counts or duplicate frames')
    for r in rows:
        if (r['split'] != 'test' or r['sequence'] != r['image'].rsplit('_', 1)[0]
                or r['domain'] != r['image'].split('_')[0]
                or r['frame_id'] != int(r['image'].rsplit('_', 1)[1])
                or type(r['angle_axis_well_defined']) is not bool):
            raise ValueError('Wrong fixed TEST row role: '+r['image'])
        g = simple.canonical(r['gt'])
        if r['angle_axis_well_defined'] != bool(g[2]/g[3] >= 1.2):
            raise ValueError('Saved offline angle qualification differs: '+r['image'])
        size = np.asarray(r['image_size'], dtype=float)
        if size.shape != (2,) or not np.isfinite(size).all() or min(size) <= 0:
            raise ValueError('Invalid original image size')
        simple.prediction(r['pred'])
        if r['pred'] is not None:
            simple.descriptor(r['pred'], r['image_size'])
        for key in ('image_sha256', 'annotation_sha256'):
            if len(r[key]) != 64 or len(bytes.fromhex(r[key])) != 32:
                raise ValueError('Invalid saved byte identity')
    return rows


def checked_inputs(args, contract, sources):
    policy_hashes = exact_bundle(args.policy.parent, contract['reviewed_policy_files'])
    if args.policy.name != 'policy.json':
        raise ValueError('Use the original policy.json, not an alternate policy name')
    policy = parent.load_policy(args.policy, sources['parent'], contract['parent_protocol'])
    cache_hashes = exact_bundle(args.test_cache_dir, contract['reviewed_test_files'])
    report = json.loads((args.test_cache_dir/'test_compare.json').read_text())
    completion = json.loads((args.test_cache_dir/'completion.json').read_text())
    check = json.loads((args.test_cache_dir/'input_check.json').read_text())
    data, native = report['data_identity'], report['native_cache']
    if (report['protocol'] != 'port_reliability_branches_v1_fixed_test'
            or report['status'] != 'FIXED_B24_QUALITY8_TEST_SCORED_REVIEW_REQUIRED'
            or report['prediction_source'] != 'cache'
            or report['cached_B_predictions_scores_exactly_preserved'] is not True
            or report['detector_optimizer_steps'] != 0 or report['branch_optimizer_steps'] != 0
            or report['test_used_for_selection'] is not False
            or report['test_repeatedly_exposed'] is not True
            or native['status'] != 'GENERATION_BOUND_NATIVE_TEST_CACHE_VERIFIED'
            or native['checkpoint_sha256'] != policy['frozen_b']['checkpoint_sha256']
            or data != contract['reviewed_test_data_identity']
            or check['data_identity'] != data or check['native_cache'] != native
            or completion['status'] != 'FIXED_TEST_SCORING_COMPLETE_REVIEW_REQUIRED'
            or completion['report_sha256'] != cache_hashes['test_compare.json']
            or completion['rows_sha256'] != cache_hashes['test_qualities.jsonl']
            or report['rows_sha256'] != cache_hashes['test_qualities.jsonl']
            or completion['frames'] != data['frames']):
        raise ValueError('Reviewed native TEST/cache completion identity differs')
    # No old qualities, errors, runtime replacement, or feature maps enter scoring.
    rows = [{k: row[k] for k in FIELDS} for row in (
        json.loads(line) for line in (args.test_cache_dir/'test_qualities.jsonl').read_text().splitlines())]
    validate_rows(rows, data['sequence_counts'])
    image_identity = hashlib.sha256()
    for r in rows:
        image_identity.update((r['image']+'.jpg').encode())
        image_identity.update(b'\0'); image_identity.update(bytes.fromhex(r['image_sha256']))
    if len(rows) != data['frames'] or image_identity.hexdigest() != data['image_identity_sha256']:
        raise ValueError('Saved full-frame/image identity differs')
    proof = dict(policy_files_sha256=policy_hashes, test_files_sha256=cache_hashes,
        data_identity=data, native_cache=native, numeric_rows_sha256=simple.fingerprint(rows),
        original_numeric_GT_preserved=True, old_quality_scores_used=False,
        current_dataset_bytes_rehashed=False, fresh_B_inference=False,
        scope='exact reviewed server numeric cache, not a fresh online or dataset rebuild test')
    return policy, rows, proof


def decision_metrics(stats):
    """Binary state errors among eligible output frames; missing is separate."""
    result = deepcopy(stats)
    good = stats['correct_accepted']+stats['correct_rejected']
    bad = stats['incorrect_accepted']+stats['incorrect_rejected']
    errors = stats['incorrect_accepted']+stats['correct_rejected']
    result.update(state_errors_on_outputs=errors,
        state_accuracy_on_outputs=1-errors/stats['output_frames'] if stats['output_frames'] else None,
        incorrect_acceptance_rate_on_bad=stats['incorrect_accepted']/bad if bad else None,
        incorrect_rejection_rate_on_good=stats['correct_rejected']/good if good else None,
        error_detection_rate=stats['incorrect_rejected']/bad if bad else None,
        good_retention_rate=stats['correct_accepted']/good if good else None,
        balanced_state_accuracy=.5*(stats['correct_accepted']/good+stats['incorrect_rejected']/bad)
                                if good and bad else None)
    return result


def score(rows, policy):
    before = simple.fingerprint([rows, policy])
    records, strata = simple.evaluate(rows, policy)
    originals = {r['image']: r for r in rows}
    for r in records:
        source = originals[r['image']]
        r.update({k: source[k] for k in FIELDS if k != 'image'})
        r['offline_errors'] = simple.geometry_errors(source['gt'], source['pred']) if source['pred'] else None
        for method in ('raw', 'score_only', 'simple'):
            if r['methods'][method]['raw_b_output'] != source['pred']:
                raise ValueError('Original B width/height/angle/score changed')
    for group, summary in strata.items():
        values = [r for r in records if group == 'all' or
                  (r[group.split(':', 1)[0]] == group.split(':', 1)[1])]
        for component in simple.COMPONENTS:
            c = summary['components'][component]
            for method in ('raw', 'score_only', 'simple', 'matched_score_diagnostic'):
                c[method] = decision_metrics(c[method])
                if method != 'matched_score_diagnostic':
                    count = sum(r['methods'][method][component+'_accepted'] for r in values)
                    c[method].update(flag_accepted_all_frames=count, flag_coverage_all_frames=count/len(values))
    if simple.fingerprint([rows, policy]) != before:
        raise ValueError('Frozen policy or original cached inputs changed')
    return records, strata


def table_text(title, strata):
    text = ['## '+title, '', '| 范围 | 方法 | 分量 | 接受/可评帧 | 错误接受 | 正确误拒 | 接受后正确率 | 全帧正确覆盖 |',
            '|---|---|---|---:|---:|---:|---:|---:|']
    def pct(x):
        return 'NA' if x is None else '%.2f%%' % (100*x)
    for group in ('domain:real', 'domain:sim'):
        for method in ('raw', 'score_only', 'simple'):
            for component in ('size', 'angle'):
                s = strata[group]['components'][component][method]
                text.append('| %s | %s | %s | %d/%d | %d | %d | %s | %s |' % (
                    group, method, component, s['accepted_frames'], s['eligible_frames'],
                    s['incorrect_accepted'], s['correct_rejected'], pct(s['correctness_on_accepted']),
                    pct(s['full_frame_correct_coverage'])))
    return '\n'.join(text)


def paper_text(report, val):
    s = report['strata']
    lines = ['# 固定B＋simple v1：论文流程与证据收尾', '',
        '本页由冻结policy和已核验数值缓存生成；方法实现已完成，收益按表判断，不自动宣称有效新方法。', '',
        '当前RGB → SymEOOD＋等比例尺度增强B（VAL选epoch24）→ 原图OBB/score → 三个独立使用标志 → 分量质量及覆盖评价。', '',
        '中心标志保留有效B输出，未训练独立中心正确性预测器。尺寸/方向使用三个当前框特征的独立线性风险及冻结VAL门限。',
        '在线API只接收当前B原框和原图尺寸；GT只用于离线评价，尺寸/方向拒绝不删除中心，不修正原框或补回漏检。', '',
        'TEST中心：', '', '| 域 | 输出/总帧 | 输出帧中心命中率 | 全帧中心正确覆盖 |', '|---|---:|---:|---:|']
    for domain in ('real', 'sim'):
        a = s['domain:'+domain]
        conditional = 'NA' if a['center_hit_rate_on_outputs'] is None else '%.4f%%' % (100*a['center_hit_rate_on_outputs'])
        lines.append('| %s | %d/%d | %s | %.4f%% |' % (domain, a['output_frames'], a['frames'],
                     conditional, 100*a['all_frame_center_correct_coverage']))
    lines.extend(['', table_text('VAL：开发/工作门限校准，非独立确认', val), '',
                  table_text('TEST：固定方案在已多次暴露划分上的报告', s), '',
        '尺寸误差为规范化长短边最大相对误差≤10%，方向为π周期长边角误差≤3°（GT aspect≥1.2仅用于离线可评资格），中心<15px。',
        '表内尺寸/方向覆盖分母是各自可评帧；test_compare.json同时保存所有帧标志覆盖、不可评接受数量、完整OBB和逐序列统计。',
        '固定score-only使用VAL门限；同接受数score是离线排序诊断，不能当作从TEST产生的部署门限。', '',
        '论文可写：分量质量并不等同统一检测置信度；建立独立分量接口并报告错误接受、正确误拒和覆盖代价。',
        '论文暂不能写：保证三分量正确、稳定/显著优于score、改善原B几何或深度、已实现物理安全报警。',
        '本版保留为简单基线。局部计数改善不证明跨视频稳定优势；如无增量，负结果应保留。',
        'VAL已用于开发/校准，TEST已多次暴露；本轮不据TEST重新选特征、门限、前端或权重，不称未接触独立确认。',
        '这是缓存评分收尾，不是全量新在线验证或性能改进。后续优化限定TRAIN/VAL，先查评分可分性，再另立有限新版本。', ''])
    return '\n'.join(lines)


def run(args, contract, sources, policy, rows, proof):
    records, strata = score(rows, policy)
    with (args.out_dir/'test_decisions.jsonl').open('x') as stream:
        for r in records:
            stream.write(json.dumps(r, allow_nan=False)+'\n')
    fields = ['group', 'component', 'method', 'eligible_frames', 'output_frames', 'accepted_frames',
        'incorrect_accepted', 'correct_rejected', 'incorrect_rejected', 'correct_accepted',
        'accepted_coverage', 'correctness_on_accepted', 'full_frame_correct_coverage',
        'state_errors_on_outputs', 'state_accuracy_on_outputs', 'balanced_state_accuracy',
        'flag_accepted_all_frames', 'flag_coverage_all_frames', 'unassessed_accepted']
    with (args.out_dir/'component_metrics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for group, summary in strata.items():
            for c in simple.COMPONENTS:
                for method in ('raw', 'score_only', 'simple', 'matched_score_diagnostic'):
                    stat = summary['components'][c][method]
                    writer.writerow(dict(group=group, component=c, method=method,
                                         **{k: stat.get(k) for k in fields[3:]}))
    report = dict(protocol=VERSION, status='FIXED_SIMPLE_V1_TEST_SCORED_REVIEW_REQUIRED',
        evidence_role='frozen_report_on_repeatedly_exposed_TEST_not_untouched_confirmation',
        sources=sources, contract=contract, input_proof=proof, strata=strata,
        policy_sha256=parent.sha(args.policy), rows_sha256=parent.sha(args.out_dir/'test_decisions.jsonl'),
        numeric_result_sha256=simple.fingerprint(strata), cached_B_inputs_unchanged=True,
        detector_updates=0, reliability_updates=0, detector_inferences=0, GPU_used=False,
        test_repeatedly_exposed=True, test_used_for_selection=False,
        deployment_threshold_selected_on_TEST=False, old_quality_scores_used=False,
        whole_TEST_runtime_parity_verified=False)
    parent.write_new(args.out_dir/'test_compare.json', report)
    val = json.loads((args.policy.parent/'fit_report.json').read_text())['val_calibration_descriptive']
    with (args.out_dir/'paper_flow_and_results.md').open('x') as stream:
        stream.write(paper_text(report, val))
    # Recheck exact source inputs before publishing COMPLETE.
    _, _, after = checked_inputs(args, contract, sources)
    _, current_sources = checked_sources()
    if after != proof or current_sources != sources:
        raise ValueError('Inputs or source bytes changed during scoring')
    names = ('input_check.json', 'test_compare.json', 'test_decisions.jsonl',
             'component_metrics.csv', 'paper_flow_and_results.md')
    parent.write_new(args.out_dir/'completion.json', dict(
        status='FIXED_SIMPLE_V1_TEST_CLOSURE_COMPLETE_REVIEW_REQUIRED', frames=len(rows),
        files_sha256={name: parent.sha(args.out_dir/name) for name in names},
        policy_sha256=report['policy_sha256'], numeric_result_sha256=report['numeric_result_sha256'],
        cached_B_inputs_unchanged=True, detector_updates=0, reliability_updates=0,
        test_used_for_selection=False))
    for domain in ('real', 'sim'):
        a = strata['domain:'+domain]
        print(domain, 'B outputs', a['output_frames'], '/', a['frames'],
              'center/output', a['center_hit_rate_on_outputs'],
              'full-frame center', a['all_frame_center_correct_coverage'])
        for c in ('size', 'angle'):
            v = a['components'][c]
            print(' ', c, 'simple accepted/bad', v['simple']['accepted_frames'],
                  v['simple']['incorrect_accepted'], 'same-count score bad',
                  v['matched_score_diagnostic']['incorrect_accepted'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=('check', 'run'), required=True)
    ap.add_argument('--policy', type=Path, default=Path('work_dirs/port_simple_reliability_v1_fit/policy.json'))
    ap.add_argument('--test-cache-dir', type=Path, default=Path('work_dirs/port_reliability_branches_v1_test_cached_v1'))
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args(); os.chdir(ROOT)
    if args.out_dir.exists():
        raise FileExistsError('Preserve evidence; choose a NEW --out-dir: '+str(args.out_dir))
    contract, sources = checked_sources()
    policy, rows, proof = checked_inputs(args, contract, sources)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    try:
        parent.write_new(args.out_dir/'input_check.json', dict(
            status='FIXED_SIMPLE_V1_TEST_INPUT_CHECK_PASS_NO_FIT_NO_INFERENCE',
            sources=sources, proof=proof, metrics_computed=False, GPU_used=False))
        if args.mode == 'run':
            run(args, contract, sources, policy, rows, proof)
    except Exception as error:
        parent.write_new(args.out_dir/'failure.json', dict(status='FAILED_PRESERVE_OUTPUTS', error=str(error)))
        raise
    print('Saved', args.out_dir, 'mode', args.mode, 'CPU only; no fitting/inference')


if __name__ == '__main__':
    main()
