"""CPU grouped readback after the two fixed terminal M diagnostics complete.

Server-only per-view rows stay on the server. No model/GPU/optimizer or TEST.
AP uses the same float32 score sort and VOC07 ap11 as the current LVSR trainer.
"""
import csv
import gzip
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import diagnose_symeood_r4_midpoint_leavevideo_v1 as d


def rows(path):
    with gzip.open(path, 'rt') as f:
        result = list(map(json.loads, f))
    if len(result) != 6003 or len({(r['image'], r['scale']) for r in result}) != 6003:
        raise ValueError('Expected all6003 unique original TRAIN/VAL views')
    return result


def ap(rows_, method, np, ap11):
    present = [r for r in rows_ if r[method] is not None]
    score = np.asarray([r[method][5] for r in present], dtype=np.float32)
    order = np.argsort(-score)
    result = dict(frames=len(rows_), outputs=len(present))
    for threshold in (.5, .75):
        hits = np.asarray([r['metrics'][method]['riou'] >= threshold for r in present], dtype=np.float32)[order]
        tp, fp = hits.cumsum(), (1-hits).cumsum()
        result['AP'+str(round(threshold*100))] = 100*ap11(tp.astype(np.float64)/len(rows_),
            tp/np.maximum(tp+fp, np.finfo(np.float32).eps))
        result['TP'+str(round(threshold*100))] = int(hits.sum())
    return result


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    from crane_project.tools.eval_port_geometry_ap_v1 import ap11
    spec = d.checked_config()
    out = ROOT/spec['output']
    complete = d.read(out/'completion.json')
    if (complete['status'] != 'TWO_FIXED_BUDGET_M_DIAGNOSTICS_COMPLETE'
            or complete['formal_diagnostic_updates'] != 35960 or complete['TEST']
            or not complete['all_pins_and_cache_unchanged']):
        raise ValueError('Two complete fixed terminal diagnostics required')
    core = d.read(out/'core_manifest.json')
    if any(d.sha(out/name) != digest for name, digest in core.items()):
        raise ValueError('Original core manifest differs')
    reference = rows(ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/attempt_001/L5_all.rows.jsonl.gz')
    reference_map = {(r['image'], r['scale']): r for r in reference}
    protocol = d.read(out/'protocol.json')
    ap_groups = {}
    csv_rows = []
    report = ['# 两份临时 M 留视频诊断', '',
        '固定 R4，只训练排除 seq01／seq13 双尺度视图的两份临时 M。各17980次更新，固定终点，无VAL选权或TEST；正式M20、L5与数据划分保持。R4见过留出视频，且训练组成改变，本轮不能隔离纯成员身份因果，也不是LVSR候选性能。', '',
        'AP75沿用当前LVSR评估的float32分数排序与VOC07十一点AP；尺寸MAE分母为各组有输出视图。双边达标分母为该组全视图，无输出单列。TRAIN尺度分开统计；VAL真实／仿真及逐视频完整列于CSV。', '']
    for omitted in spec['arms']:
        summary = d.read(out/(omitted+'_summary.json'))
        plan = protocol['schedules'][omitted]
        if (summary['updates'] != 17980
                or summary['actual_ordered_view_sha256'] != plan['ordered_view_sha256']
                or summary['actual_video_scale_exposure'] != plan['video_scale_exposure']):
            raise ValueError('Actual optimizer count/order/exposure differs')
        candidate = rows(out/(omitted+'_terminal.rows.jsonl.gz'))
        old = [reference_map[r['image'], r['scale']] for r in candidate]
        for a, b in zip(old, candidate):
            if any(a[k] != b[k] for k in ('image', 'sequence', 'frame_id', 'domain', 'scale', 'gt', 'b')):
                raise ValueError('Reference/terminal row identity differs')
        ap_groups[omitted] = {}
        for split, groups in summary['comparison'].items():
            ap_groups[omitted][split] = {}
            for group, values in groups.items():
                indices = [i for i, r in enumerate(candidate)
                    if ((r['sequence'] in ('real_seq07', 'real_seq14', 'sim_seq10')) == (split == 'val'))
                    and (split == 'val' or r['scale'] == (1. if split == 'train_s1' else .5))
                    and (group == 'overall' or (group == 'remaining_TRAIN' and r['sequence'] != omitted)
                         or r['domain'] == group or r['sequence'] == group)]
                a, b = [old[i] for i in indices], [candidate[i] for i in indices]
                table = dict(R4=ap(b, 'b', np, ap11), M20=ap(a, 'midpoint', np, ap11),
                             temporary_M=ap(b, 'midpoint', np, ap11))
                ap_groups[omitted][split][group] = table
                for method in ('R4', 'M20', 'temporary_M'):
                    metric = values[method]
                    if table[method]['frames'] != metric['views'] or table[method]['outputs'] != metric['outputs']:
                        raise ValueError('AP/size metric denominator differs')
                    csv_rows.append(dict(omitted=omitted, split=split, group=group, method=method,
                        views=metric['views'], outputs=metric['outputs'], long_MAE_pct=100*metric['long_MAE'],
                        short_MAE_pct=100*metric['short_MAE'], joint10=metric['joint10'],
                        RIoU=metric['mean_RIoU'], AP75=table[method]['AP75'], TP75=table[method]['TP75'],
                        center_MAE_px=metric['center_MAE_px'], pure_angle_MAE_deg=metric['pure_angle_MAE_deg']))
        report += ['## 排除 '+omitted, '',
            '实际更新17980步／143840样本槽；排除%d视图、其中%d具原M训练资格。初始状态与原训练一致，实际顺序及视频×尺度曝光和预声明SHA一致。耗时%.2f秒。' %
            (plan['excluded_views'], plan['excluded_eligible'], summary['elapsed_seconds']), '',
            '| 分组 | 方法 | 长MAE | 短MAE | 双边达标 | RIoU | AP75 |',
            '|---|---|---:|---:|---:|---:|---:|']
        for split, group, label in [('train_s1', omitted, '留出原尺度'), ('train_s05', omitted, '留出半尺度'),
                                     ('val', 'overall', '完整VAL'), ('val', 'real', 'Real VAL'), ('val', 'sim', 'Sim VAL')]:
            for method in ('M20', 'temporary_M'):
                m = summary['comparison'][split][group][method]
                report.append('| %s | %s | %.4f%% | %.4f%% | %d/%d | %.6f | %.4f%% |' %
                    (label, method, 100*m['long_MAE'], 100*m['short_MAE'], m['joint10'], m['views'],
                     m['mean_RIoU'], ap_groups[omitted][split][group][method]['AP75']))
        s = summary['diagnostic_screen']
        report += ['', '固定参照为 '+s['reference_VAL_video']+' 的正式M20残差。原尺度长／短有符号残差W1：M20 '+
            ', '.join('%.6f'%v for v in s['M20_native_residual_W1'])+'；临时M '+
            ', '.join('%.6f'%v for v in s['temporary_native_residual_W1'])+'。双尺度新增失标%d视图，可达%d、未定%d；全部失标%d视图，可达%d、未定%d。' %
            (s['newly_bad_views'], s['newly_bad_verified_reachable'], s['newly_bad_unresolved'],
             s['all_bad_views'], s['all_bad_verified_reachable'], s['all_bad_unresolved']), '',
            '预定诊断筛查：'+json.dumps(s['checks'], ensure_ascii=False)+'。', '']
        for split in ('train_s1', 'train_s05'):
            cell = summary['comparison'][split][omitted]['versus_M20']
            report.append('%s：相对M20修回%d／新增失标%d；准确长边受损%d/%d、准确短边受损%d/%d（原误差≤3%%，损伤阈值1e−6）。' %
                (split, cell['repaired_joint10'], cell['damaged_joint10'],
                 cell['long_accurate_le3pct']['damaged'], cell['long_accurate_le3pct']['n'],
                 cell['short_accurate_le3pct']['damaged'], cell['short_accurate_le3pct']['n']))
        report.append('')
    report += ['## 判断', '',
        '两组均通过预定诊断筛查：'+str(complete['both_diagnostic_screens_passed'])+'。'+complete['recommendation'], '',
        '通过也只允许形成下一份LVSR训练输入生成提案，不能当作LVSR改善或自动采用。未通过则停止这一方向，保留L5及历史局部收益，不继续扩大搜索。相关帧高度相关，不作独立帧显著性；M训练视频组成和曝光变化随组完整保留。', '',
        '正式R4/M20/L5均0更新，缓存与来源身份前后不变；工程4次更新已丢弃。初次相对路径启动错误在缓存读取／模型训练前停止，失败日志与源码原样留存。仅核心分组结果回传，不下载权重／缓存／逐帧文件；论文正文未改。', '']
    d.save(out/'ap75_groups.json', dict(groups=ap_groups, GPU=False, optimizer_updates=0, TEST=False,
        source_sha256=d.sha(Path(__file__).resolve()), AP_source_sha256=d.sha(ROOT/'crane_project/tools/eval_port_geometry_ap_v1.py')))
    with (out/'comparison.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    with (out/'analysis.md').open('x') as f:
        f.write('\n'.join(report))
    d.save(out/'analysis_manifest.json', {name: d.sha(out/name) for name in
        ('core_manifest.json', 'ap75_groups.json', 'comparison.csv', 'analysis.md')})
    print(json.dumps(dict(status='CORE_ANALYSIS_COMPLETE', CSV_rows=len(csv_rows), GPU=False, TEST=False)))


if __name__ == '__main__':
    main()
