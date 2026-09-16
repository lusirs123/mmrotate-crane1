"""Verify the downloaded run and supplement existing custom detector metrics."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from crane_project.tools.eval_crane_offline import CraneOfflineEvaluator, parse_dota_txt, compute_riou, angle_diff, obb_diag
from crane_project.tools.base_v3_obb_paper_final_report_v1 import _reconstruct_obb
from crane_project.tools.base_v3_obb_component_reliability_continuous import _error_from_value
from crane_project.tools.base_v3_obb_reliability_baseline import _write_exact

EXPECTED = {
    'fixed_test_focused_paper_model_pipeline_v1.json': 'c90011407e83d14704dd81b27e7cb390bdb66b609df89129c4850f488f6b3ca2',
    'fixed_test_focused_paper_model_pipeline_v1.md': '436bd6755bcf82e1520ca49b393243d875126218004f6f8caf7ebcda2f439f16',
    'fixed_test_focused_paper_reliability_v1.json': '3886713a021c7041b76cf38b969cb54b7b8ee3417403e76778c7213ca3d9614b',
    'fixed_test_focused_paper_reliability_v1.md': '5122959000cea0ebbf6560887fc1f0382b2aeb6b7baa7505a17cbc22d2a5e560',
    'unified_reliability_diagnostic_v31_r1.json': '093de7db915308eb41170b3fcdc3f8f5f61d83f51dcca1df0e668988116ddb3d',
    'unified_true_online_finalization_v2.json': '1d848597ac1488a4bb70d290b69bdb8ea1405cb8e7f1134f505fe4b9fa0ae88b',
    'unified_true_online_observations_v2.csv': '8a9665c220a1138520a4bf34008f016b15f2b549d734d9819acbc5f8930704d3',
    'unified_true_online_paper_metrics_v3.json': '9d4860d82fcdbc8d92cc75425da98ec2acd72294b33ecf8c4930890d5a46130d',
    'unified_true_online_pipeline_v1.json': 'ef5487b9b20cc8f7723fd06f5004b330060f101637b3f68ebdbec23ac1a158ac',
}

METHODS = ('raw', 'score_rejection_only', 'v51_hybrid')

def frozen_metrics(records):
    """Protocol-2 aggregation, using bound server errors instead of re-parsed GT.

    Every record is a positive-GT frame. Partial OBB output is missing for
    these whole-box metrics. Frame gaps and sequence changes break windows.
    """
    evaluator = CraneOfflineEvaluator()
    groups = defaultdict(list)
    for row in records:
        groups[(row['domain'], row['seq_id'])].append(row)
    buckets = defaultdict(lambda: defaultdict(list))
    for (domain, _), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r['frame_id'])
        segments = []
        for r in rows:
            if segments and r['frame_id'] <= segments[-1][-1]['frame_id']:
                raise ValueError('Duplicate frame')
            if not segments or r['frame_id'] != segments[-1][-1]['frame_id'] + 1:
                segments.append([])
            segments[-1].append(r)
        for segment in segments:
            b = buckets[domain]
            hits, previous, misses, maximum = [], None, 0, 0
            for r in segment:
                box = r['pred_box']
                present = box is not None
                iou = r['server_iou'] if present else 0.0
                if present and (iou is None or any(r['server_errors'][c] is None for c in ('center','angle'))):
                    raise ValueError('Complete box has missing server metrics')
                hit = iou >= evaluator.iou_thresh
                hits.append(hit)
                b['riou_vals'].append(iou)
                distance = r['server_errors']['center'] if present else float('inf')
                b['center_hits'].append(float(distance < evaluator.center_thresh_px))
                if domain == 'sim':
                    b['angle_errors'].append(math.radians(r['server_errors']['angle'])
                        if present and distance < evaluator.sim_angle_center_thresh_px else math.pi/2)
                if present and previous is not None:
                    diagonal = obb_diag(previous)
                    if diagonal > 1e-6:
                        b['dfr_vals'].append(abs(obb_diag(box)-diagonal)/diagonal)
                    delta = abs(float(angle_diff(np.array([box[4]]), np.array([previous[4]]))[0]))
                    b['aci_vals'].append(float(np.clip(1-delta/(evaluator.angle_limit_rad+1e-9),0,1)))
                previous = box
                if hit:
                    if misses:
                        b['mrf_vals'].append(misses)
                    misses = 0
                else:
                    misses += 1
                    maximum = max(maximum, misses)
            b['mcml_list'].append(maximum)
            w = evaluator.ekf_window
            b['tdr_hits'].extend(any(hits[i:i+w]) for i in range(max(0,len(hits)-w+1)))
    metrics = {}
    for domain, bucket in sorted(buckets.items()):
        evaluator._aggregate_test(metrics,bucket,domain)
    return metrics


def build_custom_metrics(online, final, paper_metrics):
    """Build protocol-2 custom metrics from one bound evaluation bundle.

    Static center/angle errors and RIoU come from the finalization report;
    temporal DFR/ACI use the exact online boxes.  No annotation is read here.
    """
    offline = {row['frame_key']: row for row in final.get('records', [])}
    online_records = online.get('records', [])
    if len(offline) != len(online_records):
        raise ValueError('Online and finalization frame counts differ')
    records = {method: [] for method in METHODS}
    for row in online_records:
        key = row['frame_key']
        if key not in offline:
            raise ValueError('Finalization is missing online frame: ' + key)
        detector = row['detector_components']
        reconstruction_row = dict(
            frame_key=key, base_v3_box=detector['base_v3_box'])
        for method in METHODS:
            observations = row['observations'][method]
            box = _reconstruct_obb(reconstruction_row, observations)
            records[method].append(dict(
                domain=row['domain'], seq_id=row['sequence'],
                frame_id=int(row['frame']), pred_box=box,
                server_errors=offline[key]['offline_errors'][method],
                server_iou=offline[key]['offline_riou'][method]))
    custom = {method: frozen_metrics(records[method]) for method in METHODS}
    for table in paper_metrics['paper_tables']['complete_obb_metrics']:
        if table['group'] in ('real', 'sim'):
            expected = round(
                table['available_obb_coverage'] *
                table['mean_available_riou'], 4)
            actual = custom[table['method']][table['group'] + '/mean_RIoU']
            if actual != expected:
                raise RuntimeError(
                    'Custom mean RIoU disagrees with bound paper table')
    return dict(
        protocol='base_v3_obb_custom_temporal_metrics_v1',
        source='bound_finalization_errors_and_riou_plus_online_boxes',
        evaluator_protocol_version=2,
        settings=dict(
            center_threshold_px=15,
            sim_angle_center_threshold_px=10,
            angle_normalization_deg=35,
            riou_hit_threshold=0.5,
            window_frames=10,
            mcml_limit=5),
        values=custom,
        notes=[
            'Custom metrics use complete OBBs; partial components count as missing.',
            'R_center is all-positive-GT-frame recall at 15 px.',
            'mean_RIoU includes zero for missing output.',
            'DFR and ACI use consecutive available pairs and reset at gaps or missing frames.',
            'A-RMSE is sim only with a 90-degree penalty for missing output or center error >=10 px.',
            'MCML_mean averages the maximum miss run of each contiguous clip.',
            'MRF excludes terminal miss runs that never recover.',
            'DEP is unavailable because this pure-vision run has no physical depth reference.'])


def main():
    root = Path('work_dirs/base_v3_obb_reliability_baseline_v1/unified_full_run_v1')
    for name, digest in EXPECTED.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Download hash mismatch: '+name)
    load = lambda name: json.loads((root/name).read_text())
    online = load('unified_true_online_pipeline_v1.json')
    final = load('unified_true_online_finalization_v2.json')
    diag = load('unified_reliability_diagnostic_v31_r1.json')
    metrics = load('unified_true_online_paper_metrics_v3.json')
    offline = {r['frame_key']:r for r in final['records']}
    methods = METHODS
    records = {m:[] for m in methods}
    annotations = {}
    max_delta = 0.0
    audit = {}
    for r in online['records']:
        key = r['frame_key']
        p = Path('crane_project/data/crane_grab/test/annfiles')/(key+'.txt')
        annotations[key] = hashlib.sha256(p.read_bytes()).hexdigest()
        boxes = parse_dota_txt(str(p))
        if len(boxes) != 1:
            raise RuntimeError('Expected one GT box: '+key)
        row = dict(frame_key=key, gt_box=boxes[0], **r['detector_components'])
        for m in methods:
            obs = r['observations'][m]
            for c in ('center','scale','angle'):
                value = obs[c]['value']
                actual = None if value is None else _error_from_value(value,row,c)
                expected = offline[key]['offline_errors'][m][c]
                if (actual is None) != (expected is None):
                    raise RuntimeError('Error presence differs: '+key)
                if actual is not None:
                    max_delta = max(max_delta,abs(actual-expected))
                    if abs(actual-expected) > audit.get(c, {}).get('max_abs_delta', -1):
                        audit[c] = dict(max_abs_delta=abs(actual-expected),frame_key=key,method=m)
            box = _reconstruct_obb(row,obs)
            actual_iou = None if box is None else compute_riou(box,boxes[0])
            expected_iou = offline[key]['offline_riou'][m]
            if (actual_iou is None) != (expected_iou is None):
                raise RuntimeError('IoU presence differs: '+key)
            if actual_iou is not None:
                max_delta = max(max_delta,abs(actual_iou-expected_iou))
                if abs(actual_iou-expected_iou) > audit.get('riou', {}).get('max_abs_delta', -1):
                    audit['riou'] = dict(max_abs_delta=abs(actual_iou-expected_iou),frame_key=key,method=m)
            records[m].append(dict(domain=r['domain'],seq_id=r['sequence'],
                                   frame_id=r['frame'],pred_box=box,
                                   server_errors=offline[key]['offline_errors'][m],server_iou=expected_iou))
    custom_report = build_custom_metrics(online, final, metrics)
    custom = custom_report['values']
    report = dict(protocol='unified_full_run_verified_custom_metrics_v1',
        verified_input_sha256=EXPECTED, annotation_sha256=annotations,
        local_gt_server_metric_max_abs_delta=max_delta,
        local_gt_audit=dict(status='UNRESOLVED_RECOMPUTATION_DIFFERENCE',by_metric=audit,
            used_for_custom_metrics=False),
        custom_metric_source='bound_server_per_frame_errors_and_riou_plus_online_boxes',
        model_report=load('fixed_test_focused_paper_model_pipeline_v1.json'),
        evaluator_sha256=hashlib.sha256(Path('crane_project/tools/eval_crane_offline.py').read_bytes()).hexdigest(),
        settings=custom_report['settings'],
        custom_metrics=custom, paper_tables=metrics['paper_tables'],
        reliability=dict(component_metrics=final['component_metrics'],
            joint=diag['joint_component_availability'],scale=diag['scale_matched_coverage'],
            angle=diag['angle_hold_pair_audit']),runtime=final['runtime_measurement'],
        notes=custom_report['notes'])
    _write_exact(root/'verified_custom_metrics_v1.json',report)
    fields = sorted(set(k for v in custom.values() for k in v))
    lines = ['# 本次完整运行：自定义指标补充核验','',
             '9 份文件 SHA256 一致。自定义指标使用服务器逐帧误差/RIoU 和在线框。',
             '本地 GT 重算存在未解释差异，未用于正式补算：'+json.dumps(audit,ensure_ascii=False),'',
             '| 指标 | raw | score rejection | V5.1 |','| --- | --- | --- | --- |']
    for k in fields:
        lines.append('| '+k+' | '+' | '.join(str(custom[m].get(k,'未定义')) for m in methods)+' |')
    lines += ['', '## 统计口径','']+['- '+s for s in report['notes']]
    lines += ['', '## 结果解释与使用范围','',
        '本次 992 帧（real 420、sim 572）可以作为冻结实现的固定 TEST 结果。图像→检测→分量判定→带有效性标志的输出链已经运行；这不等于小论文全部比较、消融与独立泛化验证已完成。',
        'raw 指 Base V3 前端输出，包含已有模型处理，不是未经处理的原始图像或基础检测器。',
        'R_center 使用完整框中心与 GT 中心距离 <15px 的全正样本召回率。可靠性中心正确率采用另一冻结阈值，不可混用。',
        'A-RMSE 仅在 sim 定义：缺框或中心误差 ≥10px 的帧记 90° 惩罚。拒绝方法的 22.9423° 主要体现该口径中的缺测代价，不能解释为有效方向的平均误差。',
        'DFR 衡量相邻有效框对角线相对变化；ACI 衡量相邻有效框方向一致性。二者包含目标真实运动，且拒绝后参与统计的帧对改变，不能独立证明几何更准确。',
        'TDR_w10 表示连续 10 帧窗口内至少存在一次 RIoU≥0.5 命中的窗口比例，不表示每帧都跟踪成功。窗口不跨越序列或帧号缺口。',
        'MCML 是 RIoU<0.5 或无完整框的连续长度，所以可大于连续无完整框长度；MRF 只统计最终恢复的失败段，末尾未恢复段不计。没有时间戳，长度只报告帧数。',
        '尺度误差是长短边相对误差中的较大值。有效分量错误率以有效输出数为分母，正确覆盖率以全部帧为分母。',
        'V5.1 增加部分尺度/方向输出，但完整 OBB 覆盖率和联合正确覆盖率低于 score rejection only。尺度排序在 all/real/sim 的非满覆盖率点均未优于分数排序；fallback 子组局部胜出不能推广到整体。',
        'fallback 的联合正确覆盖率为零，即没有框的三个分量同时满足阈值；不等于每个框、每个分量都错误。',
        '角度保持的本次平均误差变化取本次在线诊断。历史报告及本地 GT 重算的差异仍待核对，不推测原因。',
        '运行耗时包含模型初始化、图像 I/O、检测和报告；不能作为纯推理速度或可靠性模块单独开销。',
        '尚缺：本地与服务器 GT 解析/标注一致性审计、检测方法比较与保留改进消融的完整证据汇总、独立未知序列验证及单独模块计时。DEP 没有物理深度真值，不适用于当前纯视觉实验。',
        '', '## 本次服务器完整框、分量和来源分组表','',
        (root/'fixed_test_focused_paper_model_pipeline_v1.md').read_text(),
        '', '## 独立可靠性专项结果','',
        (root/'fixed_test_focused_paper_reliability_v1.md').read_text()]
    p=root/'verified_custom_metrics_v1.md'
    text='\n'.join(lines)+'\n'
    if p.exists() and p.read_text()!=text:
        raise RuntimeError('Refusing different supplement overwrite')
    p.write_text(text)
    print(json.dumps(dict(hashes_ok=9,max_delta=max_delta,custom=custom,
        runtime=report['runtime'],scale_all=diag['scale_matched_coverage']['all']['risk_wins_both_count'],
        angle_all={k:v for k,v in diag['angle_hold_pair_audit']['all'].items() if k not in ('prediction_cases','by_full_rejection_run_length')}),ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
