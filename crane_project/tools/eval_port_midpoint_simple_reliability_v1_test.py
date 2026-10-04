#!/usr/bin/env python3
"""CPU-only TEST scoring of frozen B24 + formal midpoint23 + simple flags.

Reuses the already reviewed formal TEST outputs. The original numeric TEST
cache supplies only image sizes and data identities, not predictions/qualities.
No image/weight loading, detector inference, fitting, selection or template use.
"""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import eval_port_simple_reliability_v1_test as previous
from crane_project.tools import run_port_midpoint_reliability_v1 as migration
from crane_project.utils import port_reliability_separability_v1 as rank

base = previous.parent
simple = previous.simple
VERSION = 'port_midpoint_simple_reliability_v1_fixed_test'
PROTOCOL = ROOT/'crane_project/tools/port_midpoint_simple_reliability_v1_test_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_midpoint_simple_reliability_v1_test_sources.json'
METHODS = ('raw', 'score_only', 'simple')


def checked_sources():
    old_protocol, old_sources = previous.checked_sources()
    midpoint_protocol, midpoint_sources = migration.checked_sources()
    protocol = json.loads(PROTOCOL.read_text())
    manifest = json.loads(SOURCES.read_text())
    actual = {p:base.sha(ROOT/p) for p in manifest['sources']}
    parents = {k:base.sha(ROOT/p) for k,p in protocol['source_manifests'].items()}
    formal = json.loads((ROOT/protocol['source_manifests']['formal_eval']).read_text())
    formal_sources = {p:base.sha(ROOT/p) for p in formal['sources']}
    if (manifest['protocol'] != VERSION or protocol['protocol'] != VERSION
            or manifest['sources'] != actual or manifest['protocol_sha256'] != base.sha(PROTOCOL)
            or manifest['parent_manifests_sha256'] != parents
            or formal['sources'] != formal_sources
            or parents['formal_eval'] != protocol['formal_evaluation_sources_sha256']
            or formal['protocol'] != protocol['formal_evaluation_protocol']
            or protocol['front_end']['midpoint_checkpoint']['epoch'] != midpoint_protocol['midpoint_epoch']
            or protocol['front_end']['midpoint_checkpoint']['sha256'] != midpoint_protocol['midpoint_sha256']
            or protocol['front_end']['frozen_b'] != old_protocol['parent_protocol']['frozen_b']
            or protocol['methods'] != list(METHODS)
            or protocol['test_counts'] != {'real_seq03':200,'real_seq04':668,'sim_seq09':572}
            or protocol['error_limits'] != old_protocol['parent_protocol']['error_limits']
            or protocol['test_used_for_selection'] or not protocol['test_repeatedly_exposed']
            or protocol['GPU_used'] or protocol['GT_online']
            or protocol['reference_or_ROI_or_structure_scores_used']
            or any(protocol[k] for k in ('detector_updates', 'midpoint_updates',
                'reliability_updates', 'threshold_updates', 'detector_inferences'))):
        raise ValueError('Fixed midpoint/simple TEST source or scope contract differs')
    return protocol, dict(sources=actual, manifest_sha256=base.sha(SOURCES),
        parent_manifests_sha256=parents, old_simple_test=old_sources,
        midpoint_sources=midpoint_sources, formal_sources=formal_sources), old_protocol


def load_midpoint_policy(path, protocol):
    path = Path(path)
    if path.name != 'policy.json':
        raise ValueError('Use the reviewed midpoint policy.json')
    pins = protocol['reviewed_midpoint_policy']
    files = dict(pins['artifacts'], **{'completion.json':pins['completion_sha256']})
    hashes = previous.exact_bundle(path.parent, files)
    complete = json.loads((path.parent/'completion.json').read_text())
    policy = json.loads(path.read_text())
    if (complete['status'] != 'MIDPOINT_SIMPLE_POLICY_FIT_COMPLETE'
            or complete['artifacts'] != pins['artifacts']
            or simple.fingerprint(complete['contract']) != pins['contract_sha256']
            or policy['contract'] != complete['contract']
            or complete['contract']['front_end'] != protocol['front_end']
            or policy.get('test_read') is not False or complete.get('test_read') is not False
            or policy['contract'].get('test_read') is not False
            or any(complete[k] for k in ('detector_updates', 'midpoint_updates', 'reference_updates'))):
        raise ValueError('Wrong policy generation/front end; TEST cannot refit it')
    migration.new.MidpointReliability(policy, protocol['front_end'])
    return policy, hashes


def bind_rows(metadata, formal_rows, counts):
    """Use FORMAL same-run B/final boxes and FORMAL GT precision.

    The earlier cache carries image sizes/byte identities absent in formal rows.
    Its B predictions and old qualities must not replace either formal output.
    """
    previous.validate_rows(metadata, counts)
    if [r['image'] for r in formal_rows] != [r['image'] for r in metadata]:
        raise ValueError('Formal TEST identity/order/count differs from reviewed metadata')
    sources = []
    for saved, native in zip(metadata, formal_rows):
        if (any(native[k] != saved[k] for k in ('sequence', 'frame_id', 'domain'))
                or native['scale'] != 1.
                or not np.allclose(native['gt'], saved['gt'], atol=1e-4, rtol=1e-6)
                or native.get('original_b_raw_exact_before_after') is not True):
            raise ValueError('Formal TEST GT/role/native audit differs: '+saved['image'])
        # Only numerical precision of the reviewed formal GT is retained here;
        # annotation/image SHA and image dimensions still come from reviewed data.
        gt = simple.canonical(native['gt'])
        eligible = bool(gt[2]/gt[3] >= 1.2)
        if eligible != saved['angle_axis_well_defined']:
            raise ValueError('Formal direction qualification differs from reviewed metadata')
        source = {k:deepcopy(saved[k]) for k in previous.FIELDS}
        source.update(gt=deepcopy(native['gt']), pred=None, angle_axis_well_defined=eligible)
        sources.append(source)
    paired = migration.new.paired_rows(sources, formal_rows)
    previous.validate_rows(paired, counts)
    return paired


def validate_formal_bundle(report, artifacts, hashes, policy, protocol):
    proof = report['selection']
    expected = protocol['reviewed_formal_selection']
    data = report['data_identity']
    if (report['protocol'] != protocol['formal_evaluation_protocol'] or report['split'] != 'test'
            or report['status'] != 'FROZEN_FORMAL_MIDPOINT_TEST_COMPLETE_REVIEW_REQUIRED'
            or proof != expected
            or proof['selected_checkpoint'] != policy['front_end']['midpoint_checkpoint']
            or proof['selection_sha256'] != policy['front_end']['selection_sha256']
            or report['identity']['training_identity']['sources_sha256'] !=
                policy['front_end']['formal_sources_sha256']
            or report['identity']['sources_sha256'] != protocol['formal_evaluation_sources_sha256']
            or any(report['identity']['training_identity']['frozen_b'][k] !=
                policy['front_end']['frozen_b'][k] for k in ('checkpoint_sha256','config_sha256'))
            or data != protocol['test_data_identity'] or report['frames'] != sum(protocol['test_counts'].values())
            or data['sequence_counts'] != protocol['test_counts']
            or report['state_before'] != report['state_after']
            or report['state_before']['head'] != policy['front_end']['midpoint_checkpoint']['head_digest']
            or report['detector_updates'] != 0 or report['head_updates'] != 0
            or report['selection_on_test'] or report['automatic_promotion']
            or not report['test_access'] or not report['test_repeatedly_exposed']
            or report['feature_extractions'] != report['frames']
            or report['native_head_calls'] != 3*report['frames']
            or artifacts['files'] != {k:v for k,v in hashes.items() if k != 'artifacts.json'}):
        raise ValueError('Wrong/unfinished formal TEST or selected midpoint identity')


def checked_inputs(args, protocol, sources, old_protocol):
    # This checked legacy reader deliberately discards old ROI/structure scores.
    old_args = argparse.Namespace(policy=args.old_policy, test_cache_dir=args.metadata_dir)
    old_policy, metadata, metadata_proof = previous.checked_inputs(
        old_args, old_protocol, sources['old_simple_test'])
    policy, policy_hashes = load_midpoint_policy(args.policy, protocol)
    hashes = previous.exact_bundle(args.formal_test_dir, protocol['reviewed_formal_test_files'])
    complete = json.loads((args.formal_test_dir/'completion.json').read_text())
    artifacts = json.loads((args.formal_test_dir/'artifacts.json').read_text())
    validate_formal_bundle(complete, artifacts, hashes, policy, protocol)
    if complete['identity']['sources'] != sources['formal_sources']:
        raise ValueError('Formal TEST was not generated by the verified source version')
    if any(complete['data_identity'][k] != metadata_proof['data_identity'][k]
           for k in ('frames', 'manifest_sha256', 'image_identity_sha256', 'sequence_counts')):
        raise ValueError('Formal TEST and dimension/identity metadata use different data')
    if complete['data_identity']['annotation_sha256'] != metadata_proof['data_identity']['annotations_sha256']:
        raise ValueError('Formal TEST annotations differ from reviewed metadata')
    formal = [json.loads(line) for line in (args.formal_test_dir/'test_rows.jsonl').read_text().splitlines()]
    rows = bind_rows(metadata, formal, protocol['test_counts'])
    proof = dict(midpoint_policy_files_sha256=policy_hashes, formal_test_files_sha256=hashes,
        frozen_front_end=protocol['front_end'], formal_selection=complete['selection'],
        metadata_source=metadata_proof, final_rows_sha256=simple.fingerprint(rows),
        metadata_cache_predictions_or_qualities_used=False,
        GT_precision='reviewed_formal_TEST_numeric_GT_no_reconstruction',
        current_dataset_bytes_rehashed=False, weight_bytes_loaded=False,
        fresh_detector_inference=False, same_run_formal_B_and_midpoint=True)
    return policy, old_policy, rows, proof


def score(rows, policy, old_policy):
    before = simple.fingerprint([rows, policy, old_policy])
    records, strata = previous.score(rows, policy['simple_policy'])
    b_rows = [dict(r, pred=deepcopy(r['b_original'])) for r in rows]
    b_records, b_strata = previous.score(b_rows, old_policy)
    api = migration.new.MidpointReliability(policy, policy['front_end'])
    for record, b_record, source in zip(records, b_records, rows):
        for method in METHODS:
            record['methods'][method]['final_box_original'] = record['methods'][method].pop('raw_b_output')
            if (record['methods'][method] != api.decide(source['pred'], source['image_size'], method)
                    or record['methods'][method]['center_accepted'] != (source['pred'] is not None)):
                raise ValueError('Scoring changed final box or frozen three flags')
        record.update(b_original=deepcopy(source['b_original']),
            midpoint_accepted=source['midpoint_accepted'],
            paired_B_baseline_methods=b_record['methods'])
    for collection, values in ((strata, rows), (b_strata, b_rows)):
        for group, summary in collection.items():
            members = [r for r in values if group == 'all' or
                r[group.split(':',1)[0]] == group.split(':',1)[1]]
            for component, methods in summary['components'].items():
                for name in METHODS+('matched_score_diagnostic',):
                    s = methods[name]
                    s.update(bad_accepted=s['incorrect_accepted'],
                        good_false_rejected=s['correct_rejected'],
                        bad_correctly_rejected=s['incorrect_rejected'])
                eligible = [r for r in members if r['pred'] is not None and
                    (component != 'angle' or r['angle_axis_well_defined'])]
                field, limit = {'center':('center_px',15.), 'size':('size_max_relative',.1),
                                'angle':('angle_deg',3.)}[component]
                bad = [(simple.geometry_errors(r['gt'],r['pred'])[field] >= limit if component == 'center'
                        else simple.geometry_errors(r['gt'],r['pred'])[field] > limit) for r in eligible]
                count = methods['simple']['accepted_frames']
                methods['matched_score_diagnostic']['tie_bounds'] = rank.same_count_reference(
                    [1-r['pred'][5] for r in eligible], bad, [r['image'] for r in eligible], [count])[count]
    if before != simple.fingerprint([rows, policy, old_policy]):
        raise ValueError('Fixed inputs/policies changed during scoring')
    return records, strata, b_strata


def summary_text(strata, b_strata):
    lines = ['# 固定B＋正式midpoint＋普通三分量判断：TEST报告', '',
        '前端B epoch24＋正式midpoint epoch23；使用已冻结的新前端simple参数及VAL门限。',
        '复用已有正式TEST原图坐标框，CPU评分；无新推理、训练、门限/权重选择或模板/ROI/结构分支。',
        '中心有输出即保留，尚不是中心正确性预测；尺寸/方向是建议使用标志，不是已知真值标签。',
        'GT仅离线评价：中心<15px、规范长短边最大相对误差≤10%、π周期长边角误差≤3°。',
        '方向GT aspect≥1.2仅用于离线资格；无输出时三个标志均false，尺寸/方向拒绝不删除中心。', '',
        '| 域 | 输出/总帧 | 输出覆盖 | 输出帧中心命中率 | 全帧正确中心覆盖 |',
        '|---|---:|---:|---:|---:|']
    pct = lambda x:'NA' if x is None else '%.4f%%'%(100*x)
    for domain in ('real', 'sim'):
        s = strata['domain:'+domain]
        lines.append('| %s | %d/%d | %s | %s | %s |'%(domain,s['output_frames'],s['frames'],
            pct(s['output_coverage']),pct(s['center_hit_rate_on_outputs']),pct(s['all_frame_center_correct_coverage'])))
    lines.extend(['', previous.table_text('当前midpoint：raw/score-only/simple三个固定方法',strata), '',
        '| 域 | 分量 | B＋旧simple接受/错接/误拒 | midpoint＋新simple接受/错接/误拒 |',
        '|---|---|---:|---:|'])
    for domain in ('real','sim'):
        for c in ('size','angle'):
            old,new=[s['domain:'+domain]['components'][c]['simple'] for s in (b_strata,strata)]
            show=lambda x:'%d/%d/%d'%(x['accepted_frames'],x['bad_accepted'],x['good_false_rejected'])
            lines.append('| %s | %s | %s | %s |'%(domain,c,show(old),show(new)))
    lines.extend(['', '上述旧基线使用同次正式评价的B框与原B simple policy；不冒用历史缓存分数。',
        '迁移比较同时改变最终框和拟合参数，不能把差异单独归因于可靠性方法创新。',
        '同接受数score为离线诊断；原部署门限保持VAL冻结，不由TEST产生。',
        '完整OBB和分量可用性分开，拒绝不修复检测几何；接口保持不代表深度精度保证。',
        'TEST已多次暴露，仅报告冻结流程，不用于调参/选权或称为未接触独立确认。',
        '工程完成不代表simple稳定优于score；效果必须依据错误接受、正确误拒及覆盖判断。',''])
    return '\n'.join(lines)


def write_result(args, prepared):
    protocol, sources, _, policy, old_policy, rows, proof = prepared
    records, strata, b_strata = score(rows, policy, old_policy)
    path = args.out_dir/'test_decisions.jsonl'
    with path.open('x') as stream:
        for row in records:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    restored = [json.loads(line) for line in path.read_text().splitlines()]
    if restored != records:
        raise ValueError('Saved TEST decisions changed on reload')
    columns = ('group','component','method','eligible_frames','output_frames','accepted_frames',
        'bad_accepted','good_false_rejected','bad_correctly_rejected','correct_accepted',
        'accepted_coverage','correctness_on_accepted','full_frame_correct_coverage',
        'state_accuracy_on_outputs','error_detection_rate','good_retention_rate',
        'flag_accepted_all_frames','flag_coverage_all_frames','unassessed_accepted')
    with (args.out_dir/'component_metrics.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns); writer.writeheader()
        for group,value in strata.items():
            for c in simple.COMPONENTS:
                for m in METHODS+('matched_score_diagnostic',):
                    writer.writerow(dict(group=group,component=c,method=m,
                        **{k:value['components'][c][m].get(k) for k in columns[3:]}))
    report = dict(protocol=VERSION,status='FIXED_MIDPOINT_SIMPLE_TEST_SCORED_REVIEW_REQUIRED',
        contract=protocol,sources=sources,input_proof=proof,strata=strata,
        paired_B_old_simple_strata=b_strata,
        policy_sha256=base.sha(args.policy),old_policy_sha256=base.sha(args.old_policy),
        rows_sha256=base.sha(path),numeric_result_sha256=simple.fingerprint([strata,b_strata]),
        final_boxes_scores_counts_and_missing_outputs_preserved=True,
        detector_updates=0,midpoint_updates=0,reliability_updates=0,threshold_updates=0,
        detector_inferences=0,GPU_used=False,GT_online=False,
        reference_or_ROI_or_structure_scores_used=False,
        current_dataset_bytes_rehashed=False,whole_TEST_runtime_parity_rechecked=False,
        evidence_role=protocol['evidence_role'],test_repeatedly_exposed=True,test_used_for_selection=False)
    base.write_new(args.out_dir/'test_compare.json',report)
    with (args.out_dir/'test_summary.md').open('x') as stream:
        stream.write(summary_text(strata,b_strata))
    for domain in ('real','sim'):
        s = strata['domain:'+domain]
        print(domain,'midpoint outputs',s['output_frames'],'/',s['frames'],
            'center/output',s['center_hit_rate_on_outputs'],
            'full-frame center',s['all_frame_center_correct_coverage'],flush=True)
        for c in ('size','angle'):
            z=s['components'][c]
            print(' ',c,'simple accepted/bad/false-reject',z['simple']['accepted_frames'],
                z['simple']['bad_accepted'],z['simple']['good_false_rejected'],
                'same-count score bad',z['matched_score_diagnostic']['bad_accepted'],flush=True)
    return 'FIXED_MIDPOINT_SIMPLE_TEST_COMPLETE_REVIEW_REQUIRED'


def prepare(args):
    protocol,sources,old_protocol = checked_sources()
    policy,old_policy,rows,proof = checked_inputs(args,protocol,sources,old_protocol)
    return protocol,sources,old_protocol,policy,old_policy,rows,proof


def finish(args,prepared,status):
    current = prepare(args)
    if current != prepared:
        raise ValueError('Sources/frozen inputs changed during TEST scoring')
    artifacts = {p.name:base.sha(p) for p in sorted(args.out_dir.iterdir()) if p.is_file()}
    base.write_new(args.out_dir/'completion.json',dict(status=status,mode=args.mode,
        frames=len(prepared[5]),sources=prepared[1],input_proof=prepared[6],artifacts=artifacts,
        detector_updates=0,midpoint_updates=0,reliability_updates=0,threshold_updates=0,
        detector_inferences=0,GPU_used=False,test_repeatedly_exposed=True,test_used_for_selection=False))
    print('Saved',args.out_dir,status,'CPU only; no fitting/inference',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('check','run'),required=True)
    parser.add_argument('--policy',type=Path,default=Path('work_dirs/port_midpoint_reliability_v1_fit/policy.json'))
    parser.add_argument('--old-policy',type=Path,default=Path('work_dirs/port_simple_reliability_v1_fit/policy.json'))
    parser.add_argument('--formal-test-dir',type=Path,default=Path('work_dirs/port_geometry_midpoint_formal_v1_test_eval'))
    parser.add_argument('--metadata-dir',type=Path,default=Path('work_dirs/port_reliability_branches_v1_test_cached_v1'))
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError('Preserve evidence; choose a NEW --out-dir: '+str(args.out_dir))
    prepared=prepare(args)
    args.out_dir.mkdir(parents=True,exist_ok=False)
    base.write_new(args.out_dir/'input_check.json',dict(status='FIXED_MIDPOINT_SIMPLE_TEST_INPUTS_PASS',
        contract=prepared[0],sources=prepared[1],proof=prepared[6],CPU_only=True,
        metrics_computed=False,mode=args.mode))
    try:
        status='FIXED_MIDPOINT_SIMPLE_TEST_INPUTS_PASS' if args.mode=='check' else write_result(args,prepared)
        finish(args,prepared,status)
    except Exception as error:
        base.write_new(args.out_dir/'failure.json',dict(status='FAILED_PRESERVE_OUTPUTS',
            type=type(error).__name__,error=str(error)))
        raise


if __name__=='__main__':
    main()
