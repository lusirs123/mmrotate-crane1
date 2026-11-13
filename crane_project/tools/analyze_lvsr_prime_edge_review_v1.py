"""Verify grouped CPU review receipts and write the local core analysis."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'work_dirs/lvsr_diagnostic_20261112_v1/prime_001'


def read(name):
    return json.loads((OUT / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    manifest = read('edge_review_manifest.json')
    assert len(manifest) == 3
    assert all(sha(OUT / name) == value for name, value in manifest.items())
    summary = read('edge_review_summary.json')
    completion = read('edge_review_completion.json')
    protocol = read('edge_review_protocol.json')
    assert summary['spec'] == protocol['spec']
    assert completion['passed'] and completion['CPU_only']
    assert not completion['CUDA_initialized'] and not completion['optimizer_created']
    assert completion['updates'] == 0 and not completion['TEST']
    assert completion['forward_views'] == 11998 and completion['elapsed_seconds'] < 600
    assert completion['parents_inputs_selection_unchanged']
    assert completion['decode_reconstruction_exact']
    assert not summary['inference_modified'] and not summary['performance_evidence']
    assert summary['decode_reconstruction_max_abs'] == 0
    assert all(v <= 1e-4 for v in summary['replay_max_abs'].values())
    assert completion['source_SHA'] == summary['source_SHA'] == protocol['source_SHA']
    assert all(sha(ROOT / name) == value for name, value in completion['source_SHA'].items())
    old_pins = read('core_summary.json')['upstream_SHA']
    assert all(completion['pinned_SHA'][k] == v for k, v in old_pins.items())
    assert completion['pinned_SHA']['work_dirs/lvsr_diagnostic_20261112_v1/prime_001/head_epoch_05.pth'] == '9590597e713d562c3fa953cd96866657d1d17de14d00c822155592e0a2d77301'
    rows = summary['groups']
    index = {(g['head'], g['split'], g['group'], g['edge'], g['subset']): g for g in rows}
    assert len(index) == len(rows) == 570
    for g in rows:
        for metric in g['metrics'].values():
            assert metric['n'] == g['n'] and metric['n'] > 0
            assert all(math.isfinite(metric[k]) for k in ('mean', 'median', 'min', 'max'))
            assert metric['min'] <= metric['median'] <= metric['max']
    core = []
    for head in ('L5', 'PRIME5'):
        for split, count in (('train_s1', 2558), ('train_s05', 2554), ('val', 887)):
            for edge in ('long', 'short'):
                get = lambda group, subset: index.get((head, split, group, edge, subset), {'n': 0})['n']
                assert get('overall', 'all') == count
                assert sum(get('overall', k) for k in ('shrink', 'near', 'enlarge')) == count
                assert sum(get(k, 'all') for k in ('real', 'sim')) == count
                assert sum(get(k, 'all') for k in {r['group'] for r in rows if '_seq' in r['group']}) == count
            g = index[head, split, 'overall', 'short', 'all']
            m = g['metrics']
            core.append(dict(head=head, split=split, n=count,
                energy0_mean=m['energy0']['mean'], energy0_median=m['energy0']['median'],
                mixed_views=round(m['mixed']['mean'] * count),
                zero_branch0=round(m['zero0']['mean'] * count),
                aux_energy0_mean=m.get('aux_energy0', {}).get('mean'),
                balanced_aux_energy0_mean=m.get('aux_balanced_energy0', {}).get('mean'),
                original_balanced_gradient_cosine_mean=m.get('norm_change_cosine', {}).get('mean')))
    assert all(index['PRIME5', 'val', 'overall', e, 'accurate']['n'] == n for e, n in [('long', 346), ('short', 418)])
    parameters = summary['parameter_gradients']
    assert [v['batch'] for v in parameters] == list(range(1, 17))
    ratio = statistics.median(v['aux_norm'] / v['size_norm'] for v in parameters)
    negatives = sum(v['cosine'] < 0 for v in parameters)
    with (OUT / 'edge_review_comparison.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(core[0]))
        writer.writeheader(); writer.writerows(core)
    save('edge_review_local_verification.json', dict(passed=True, manifest_files=3,
        grouped_rows=570, split_domain_video_direction_accounting=True,
        script_SHA_verified=True, upstream_receipts_match=True,
        views_per_head_present=5999, total_forwards=11998,
        coverage_note='6003 cached views per head; 4 missing-output TRAIN views were not forwarded. No new neutral-initialization run.',
        replay_max_abs=summary['replay_max_abs'], decode_max_abs=0,
        stem_aux_size_norm_ratio_median=ratio, stem_negative_batches=negatives,
        parameter_batches=16, GPU=False, updates=0, TEST=False))
    save('edge_review_decision.json', dict(
        status='do_not_start_static_branch_to_long_short_training',
        implementation_bug_confirmed=False, causal_root_cause_confirmed=False,
        observed='unequal branch feature energy and TRAIN per-location role mixing',
        blocker='Static branch labels do not follow the unchanged per-location size decoder.',
        counterfactual='Equal-block normalization changes feature gradients; no candidate was trained.',
        next_step='If continued, first define a geometry-consistent auxiliary representation and assess its method cost; no automatic training.',
        inference_changed=False, adoption=False, paper_body_changed=False))
    report = '''# 按边代理约束 CPU 审查

结论：不启动“分支0固定长边、分支1固定短边”的24轮训练。存在幅值不均，但它不是已证实的性能根因；原分支与最终长短边在TRAIN不完全对应。本轮只实现独立诊断入口，没有修改正式头／loss／trainer。

## 覆盖与核验

单线程CPU用时{seconds:.2f}秒，CUDA未初始化、无优化器、0更新、未访问TEST。L5／PRIME ep5各回放5999个有输出视图：TRAIN原尺度2558、半尺度2554、VAL887，总前向11998。协议中的6003是每头缓存总视图数，另4个无输出TRAIN视图未进入此审查；本轮没有重做中性初始化验证。仅返回分组JSON／CSV，不下载权重、缓存、逐帧行或梯度数组。

R4／M20／L5／L24、候选ep5、输入与选权SHA检查保持；PRIME ep5 SHA为`9590597e713d562c3fa953cd96866657d1d17de14d00c822155592e0a2d77301`。候选权重仅留服务器。旧输出CPU回放最大差L5为7.6294e−5、PRIME为9.1553e−5，均在预定容差内。原图坐标恢复→逐位置范数→排序→W聚合重建差0，中心／方向／分数继承通过；头／代理／缓存未更新。原上游中性输出验收沿用已完成记录，不重复冒称本轮验证。

## 主要证据

| 头／集合 | 分支0特征能量占比均值 | 逐位置角色混合视图 |
|---|---:|---:|
{table}

能量按共同支持平方根加权特征的平方范数计算。两分支是对边组；VAL这两个快照均为分支0对应长边，但不能把该观察外推为TRAIN固定身份。PRIME原尺度部分视图中，长边输出对分支1的Jacobian平方范数占比最高57.82%，短边输出对分支0最高72.14%。这不是解码错误，是保留逐位置排序后的实际依赖。

PRIME已有归一化并非令分支0完全失去信号：原尺度／半尺度辅助特征梯度中分支0能量占比均值31.91%／33.80%，两分支无零描述。仅将同一已训练代理与样本改为分块等幅归一化，仍用原二维目标时，该占比变为81.01%／82.77%，与原梯度平均余弦为0.7765／0.7783。它会重新分配梯度，而非自动得到无损平衡；该反事实也不是拟议的两套一维代理训练验证。

原epoch6首16自然批，0.001对齐项与尺寸项的实际stem梯度范数比中位数{ratio:.4f}%，7/16批夹角为负。该局部统计既不说明辅助项始终主导，也不足以认定冲突导致退化；未计算训练全过程的Adam有效更新归因。VAL梯度仅作冻结诊断，无反向更新或选权。

## 判断与下一步

静态按分支拆成两套长／短边代理未满足预定映射条件，本候选审查后停止，不开正式训练、不续训、不做新TEST。简单按位置交换隐藏特征也尚未获验证：独立分支的隐藏通道没有预设共同语义，排序取决于当前预测，不能仅凭尺寸相同就视为等价映射。

若继续，下一步先明确训练辅助表征如何沿原几何解码对应最终两边，证明该映射不改原支持／解码／交付，再决定唯一结构候选。该步骤会涉及训练表征／辅助目标的表述，不能称推理完全不变就无需说明方法变化。当前正式L5及selected=null沿革、PRIME ep5相对M的收益与相对L5的取舍均保留；本次没有新增性能结论，也不将单一归一化假设升级为核心因果结论。论文正文未修改。

服务器复用：`/media/omnisky/personal_files/ljj/symEOOD/work_dirs/lvsr_diagnostic_20261112_v1/prime_001/edge_review_*`。原始核心三文件SHA、本地570个分组及域／视频／方向计数守恒通过，详见`edge_review_local_verification.json`；简表见`edge_review_comparison.csv`。
'''.format(seconds=completion['elapsed_seconds'], ratio=ratio*100,
        table='\n'.join('| {head} / {split} | {energy0_mean:.2%} | {mixed_views}/{n} |'.format(**row) for row in core))
    with (OUT / 'edge_review_analysis.md').open('x') as f:
        f.write(report)
    print(json.dumps(dict(passed=True, groups=len(rows), forwards=11998, decision='do_not_train_static_mapping')))


if __name__ == '__main__':
    main()
