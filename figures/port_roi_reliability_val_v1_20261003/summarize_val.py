"""Export existing frozen cached-VAL evidence; no fitting or detector inference.

Run from the project root with a Python environment containing matplotlib.
The six existing coverage settings are descriptive matched-count evaluations.
No score cutoff is exported as a deployment threshold.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'work_dirs/port_reliability_branches_v1_server_review_20261003'
METHODS = ('score', 'geometry', 'roi', 'structure')
MAIN = METHODS[:3]
COVERAGES = (1., .99, .98, .95, .90, .80)
COMPONENTS = ('center', 'size', 'angle')
COLORS = dict(score='#0077BB', geometry='#EE7733', roi='#009988')
LABELS = dict(score='Detection score', geometry='Geometry', roi='Ordinary ROI')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_tables(report, predictions):
    baseline, points = [], []
    for group, summary in report['strata'].items():
        subset = [r for r in predictions if group == 'all' or
                  r[group.split(':')[0]] == group.split(':')[1]]
        outputs = [r for r in subset if r['pred'] is not None]
        hits = sum(r['errors']['center_px'] < 15 for r in outputs)
        assert len(subset) == summary['frames']
        assert len(outputs) == summary['output_frames']
        assert hits / len(outputs) == summary['center_hit_rate_on_outputs']
        assert hits / len(subset) == summary['all_frame_center_correct_coverage']
        baseline.append(dict(
            stratum=group, frames=len(subset), outputs=len(outputs),
            center_correct_outputs=hits, missing_outputs=len(subset)-len(outputs),
            output_coverage=summary['output_coverage'],
            center_hit_rate_on_outputs=summary['center_hit_rate_on_outputs'],
            all_frame_center_correct_coverage=summary['all_frame_center_correct_coverage']))
        for component, curves in summary['component_curves'].items():
            assert tuple(c['requested_eligible_frame_coverage'] for c in curves) == COVERAGES
            for curve in curves:
                counts = set()
                for method in METHODS:
                    stats = curve['methods'][method]
                    accepted = stats['accepted_frames']
                    counts.add(accepted)
                    assert accepted == stats['correct_accepted'] + stats['incorrect_accepted']
                    assert accepted == curve['matched_actual_accept_count']
                    assert stats['component_accepted_coverage'] == accepted / stats['eligible_frames']
                    assert stats['correctness_on_accepted'] == stats['correct_accepted'] / accepted
                    assert stats['all_eligible_frame_correct_component_coverage'] == stats['correct_accepted'] / stats['eligible_frames']
                    row = dict(stratum=group, component=component, method=method,
                               requested_eligible_frame_coverage=curve['requested_eligible_frame_coverage'])
                    row.update({k: v for k, v in stats.items() if k not in
                                ('cutoff_score_descriptive_only', 'total_at_cutoff', 'accepted_at_cutoff')})
                    row['incorrect_fraction_on_accepted'] = stats['incorrect_accepted'] / accepted
                    points.append(row)
                assert len(counts) == 1
    assert len(baseline) == 6 and len(points) == 432
    csv_write('baseline_coverage.csv', baseline)
    csv_write('component_coverage_all_methods.csv', points)
    return baseline, points


def write_markdown(baseline, points):
    lines = ['# 固定 ordinary ROI 分支的已有 VAL 结果', '',
             '来源：固定B原生VAL缓存及已完成epoch_08质量评分；单seed、已暴露VAL，属于探索性结果。',
             '所有数值复用原报告，无新增训练、推理或TEST读取。主对照为检测score、geometry与ordinary ROI；structure结果完整保留在CSV。', '',
             '## 原检测覆盖：侧分支不改变B框', '',
             '| 分组 | 总帧 | 输出 | 正确中心 | 输出覆盖 | 输出帧中心命中 | 全帧中心正确覆盖 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in baseline:
        lines.append('| {stratum} | {frames} | {outputs} | {center_correct_outputs} | {a:.4f}% | {b:.4f}% | {c:.4f}% |'.format(
            **row, a=100*row['output_coverage'], b=100*row['center_hit_rate_on_outputs'],
            c=100*row['all_frame_center_correct_coverage']))
    lines += ['', '中心正确：原图距离<15px；尺寸正确：长短边最大相对误差≤10%；方向正确：π周期角误差≤3°且GT方向满足原资格。',
              '各分量独立接受；不能将不同分量的接受集合拼成完整OBB。质量是连续误差映射的预测，不能直接称校准概率。', '',
              '## 全部预定覆盖档的主对照', '',
              '表内数字为错误接受数，越少越好。每一行方法接受数完全相同。ceil取整及缺输出使实际接受覆盖可能不等于名义档位。',
              '100%档是所有原输出，不是强制产生缺失框。各档为离线同数量排序评价，不是固定部署阈值。']
    for coverage in COVERAGES:
        lines += ['', '### 预定 {:.0f}% 档'.format(100*coverage), '',
                  '| 分组 | 分量 | 实际接受/资格帧 | score错误 | geometry错误 | ROI错误 | ROI误拒正确分量 | ROI正确接受全帧覆盖 | ROI最长未接受段/采样帧 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for group in ('domain:real', 'domain:sim'):
            for component in COMPONENTS:
                rows = {r['method']: r for r in points if r['stratum'] == group and
                        r['component'] == component and r['requested_eligible_frame_coverage'] == coverage}
                a = rows['roi']
                lines.append('| {} | {} | {}/{} | {} | {} | {} | {} | {:.4f}% | {} |'.format(
                    group, component, a['accepted_frames'], a['eligible_frames'],
                    rows['score']['incorrect_accepted'], rows['geometry']['incorrect_accepted'],
                    a['incorrect_accepted'], a['correct_rejected'],
                    100*a['all_eligible_frame_correct_component_coverage'],
                    a['longest_unaccepted_consecutive_frames']))
    lines += ['', '## 普通ROI相对score的跨档结果', '',
              '下表统计99/98/95/90/80%五档；100%档所有方法接受相同原框。各档并非独立重复实验，不作显著性计数。', '',
              '| 分组 | 分量 | 错误接受更少/相同/更多的档数 |', '|---|---|---:|']
    for group in ('domain:real', 'domain:sim'):
        for component in COMPONENTS:
            changes = []
            for coverage in COVERAGES[1:]:
                a = {r['method']: r for r in points if r['stratum'] == group and
                     r['component'] == component and r['requested_eligible_frame_coverage'] == coverage}
                changes.append(a['roi']['incorrect_accepted']-a['score']['incorrect_accepted'])
            lines.append('| {} | {} | {}/{}/{} |'.format(group, component,
                sum(x<0 for x in changes), sum(x==0 for x in changes), sum(x>0 for x in changes)))
    lines += ['', '## 当前证据可支持的结论', '',
              '- real中心：普通ROI五档错误接受均少于score，但拒绝会损失正确中心的全帧可用覆盖；当前流程保留原B中心并附加质量。',
              '- real尺寸：五档没有减少错误接受；95/90/80%更差。real方向同样没有错误接受增益。',
              '- sim尺寸/方向：部分档位错误接受减少，仍有无收益或退化档；sim方向95%档最长未接受段由score的2帧增至ROI的13帧。',
              '- sim中心：原输出无中心坏框，不能凭条件精度100%宣称中心可靠性判别有效，拒绝只会减少正确接受覆盖。',
              '- 序列分层及structure负结果保留在完整CSV；不以两域合并均值掩盖局限。',
              '- 曲线连线仅帮助读图，不代表在档位之间评价了连续阈值；最长段为原排序规则下的采样帧数，非物理秒数或新增时序方法。',
              '- 缓存评分未确认当前运行环境在线B全887帧数值一致；已知单帧极小差异与旧缓存身份分别保留。',
              '- 当前TEST已多次暴露，本轮仅整理VAL。', '']
    (OUT/'val_tables.md').write_text('\n'.join(lines), encoding='utf-8')


def plot(points, field, name, ylabel, title):
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.4), sharex=True)
    for di, domain in enumerate(('real', 'sim')):
        for ci, component in enumerate(COMPONENTS):
            ax = axes[di, ci]
            for method in MAIN:
                rows = sorted((r for r in points if r['stratum'] == 'domain:'+domain and
                               r['component'] == component and r['method'] == method),
                              key=lambda r: r['component_accepted_coverage'])
                ax.plot([100*r['component_accepted_coverage'] for r in rows],
                        [100*r[field] for r in rows], color=COLORS[method],
                        marker=dict(score='o', geometry='s', roi='^')[method],
                        markersize=4, linewidth=1.5, label=LABELS[method])
            ax.set_title('{} / {}'.format(domain.upper(), component))
            ax.set_xlim(79, 101)
            ax.set_xticks((80, 90, 95, 100))
            ax.grid(alpha=.22)
            ax.set_ylabel(ylabel)
            if di == 1:
                ax.set_xlabel('Accepted / eligible frames (%)')
            if field == 'incorrect_fraction_on_accepted':
                ax.set_ylim(bottom=0)
                if domain == 'sim' and component == 'center':
                    ax.set_ylim(0, 1)
                    ax.text(.05, .78, 'No bad-center outputs', transform=ax.transAxes, fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, .035))
    fig.suptitle(title, fontsize=12, y=.98)
    fig.text(.5, .007, 'Frozen cached VAL; epoch 8; six predefined points; one seed; exploratory',
             ha='center', fontsize=8)
    fig.subplots_adjust(left=.075, right=.985, top=.90, bottom=.17, wspace=.34, hspace=.35)
    fig.savefig(OUT/(name+'.png'), dpi=450)
    fig.savefig(OUT/(name+'.svg'))
    plt.close(fig)


def main():
    report = json.loads((SOURCE/'val_compare.json').read_text())
    completion = json.loads((SOURCE/'completion.json').read_text())
    predictions = [json.loads(line) for line in (SOURCE/'val_qualities.jsonl').read_text().splitlines()]
    assert report['status'] == 'FIXED_CACHED_B_EPOCH8_VAL_SCORED_REVIEW_REQUIRED'
    assert sha(SOURCE/'val_qualities.jsonl') == report['rows_sha256']
    assert report['branch_checkpoint_sha256'] == completion['checkpoint_sha256']
    assert report['frozen_b_state'] == completion['b_state']
    assert report['fixed_epoch'] == completion['completed_epoch'] == 8
    assert report['cached_b_outputs_scores_exactly_preserved'] and not report['test_read']
    assert len(predictions) == 887 and len({r['image'] for r in predictions}) == 887
    assert all(r['split'] == 'val' for r in predictions)
    baseline, points = export_tables(report, predictions)
    write_markdown(baseline, points)
    plot(points, 'incorrect_fraction_on_accepted', 'fig1_component_risk_coverage',
         'Incorrect / accepted (%)', 'Component error versus acceptance coverage')
    plot(points, 'all_eligible_frame_correct_component_coverage', 'fig2_correct_component_coverage',
         'Correct accepted / eligible (%)', 'Availability of correctly accepted components')
    inputs = {str((SOURCE/name).relative_to(ROOT)): sha(SOURCE/name)
              for name in ('val_compare.json', 'val_qualities.jsonl', 'completion.json')}
    outputs = {p.name: sha(p) for p in sorted(OUT.iterdir())
               if p.suffix in ('.csv', '.png', '.svg', '.md') and p.name != 'data-manifest.md'}
    manifest = dict(status='EXISTING_VAL_TABLES_AND_FIGURES_EXPORTED', real_data=True,
                    inputs=inputs, script_sha256=sha(Path(__file__)), outputs=outputs,
                    csv_records=dict(baseline=6, all_method_component_points=432),
                    detector_epoch=24, branch_epoch=8,
                    branch_checkpoint_sha256=report['branch_checkpoint_sha256'],
                    new_fitting=False, new_inference=False, test_read=False,
                    test_repeatedly_exposed=True, deployment_threshold_selected=False)
    (OUT/'data-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    (OUT/'data-manifest.md').write_text(
        '# VAL图表数据清单\n\n真实服务器回传数据；固定B ep24、质量分支ep08，单seed探索性VAL。\n\n'
        '| Figure | Data file | Real/mock | Source | Script | Outputs |\n'
        '|---|---|---|---|---|---|\n'
        '| Component risk–coverage | component_coverage_all_methods.csv | Real | 原val_compare.json（SHA见JSON） | summarize_val.py | fig1_component_risk_coverage.png / .svg |\n'
        '| Correct component availability | component_coverage_all_methods.csv | Real | 同上 | summarize_val.py | fig2_correct_component_coverage.png / .svg |\n\n'
        '全表含四方法与六个分组；主图仅score/geometry/ROI的两域结果。无mock数据，无新模型运行，无TEST评价或阈值选择。\n',
        encoding='utf-8')
    print('Exported 6 baseline rows, 432 component points, full VAL tables and two PNG/SVG figures:', OUT)


if __name__ == '__main__':
    main()
