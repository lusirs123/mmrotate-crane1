"""Independent core-receipt and group accounting for the frozen mapping audit."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/prime_001'


def read(name): return json.loads((OUT/name).read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name, value):
    with (OUT/name).open('x') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    manifest = read('edge_mapping_manifest.json')
    assert len(manifest)==3 and all(sha(OUT/n)==v for n,v in manifest.items())
    s = read('edge_mapping_summary.json'); c = read('edge_mapping_completion.json')
    protocol = read('edge_mapping_protocol.json')
    assert c['passed'] and c['CPU_only'] and c['parameters_inputs_unchanged'] and c['source_checks_passed']
    assert not c['CUDA_initialized'] and not c['optimizer_created'] and c['updates']==0 and not c['TEST']
    assert c['forward_views']==18001 and c['elapsed_seconds']<600
    assert c['initial_neutral_present']==5999 and c['initial_neutral_missing']==4 and c['mathematical_checks']==7
    assert s['max_decode_abs']==s['max_jacobian_outside_mask']==s['max_aux_gradient_outside_mask']==0
    assert max(s['replay_max_abs'].values())<=1e-4 and not s['performance_evidence']
    assert s['source_SHA']==c['sources_SHA']==protocol['source_SHA']
    assert all(sha(ROOT/n)==v for n,v in s['source_SHA'].items())
    assert c['pinned_SHA']==read('edge_review_completion.json')['pinned_SHA']
    assert all(v['unchanged'] for v in s['state_SHA'].values())
    rows=s['groups']; index={(g['head'],g['split'],g['group'],g['edge'],g['subset']):g for g in rows}
    assert len(rows)==len(index)==855
    for g in rows:
        for m in g['metrics'].values():
            assert m['n']==g['n'] and m['n']>0
            assert all(math.isfinite(m[k]) for k in ('mean','min','max','median'))
            assert m['min']<=m['median']<=m['max']
    core=[]; param=[]; old={(g['head'],g['split'],g['group'],g['edge'],g['subset']):g for g in read('edge_review_summary.json')['groups']}
    for head in ('initial','L5','PRIME5'):
        for split,n in [('train_s1',2558),('train_s05',2554),('val',887)]:
            for edge in ('long','short'):
                key=(head,split,'overall',edge,'all');g=index[key];m=g['metrics'];assert g['n']==n
                count=lambda group,subset:index.get((head,split,group,edge,subset),{'n':0})['n']
                assert sum(count('overall',k) for k in ('shrink','near','enlarge'))==n
                assert sum(count(k,'all') for k in ('real','sim'))==n
                assert sum(count(k,'all') for k in {r['group'] for r in rows if '_seq' in r['group']})==n
                if head!='initial':
                    assert m['mixed']['mean']==old[key]['metrics']['mixed']['mean']
                    assert abs(m['near_01_mass']['mean']-old[key]['metrics']['tie_mass']['mean'])<1e-8
                core.append(dict(head=head,split=split,edge=edge,n=n,
                    mixed=round(m['mixed']['mean']*n),zero_sample=round(m['zero_sample']['mean']*n),
                    sample_norm_min=m['sample_norm']['min'],sample_norm_median=m['sample_norm']['median'],
                    zero_proxy=round(m['zero_proxy']['mean']*n),zero_aux_grad=round(m['zero_aux_grad']['mean']*n),
                    aux_grad_norm_max=m['aux_grad_norm']['max'],
                    near_001_mass_mean=m['near_001_mass']['mean'],near_01_mass_mean=m['near_01_mass']['mean'],
                    exact_tie_mass_max=m['exact_tie_mass']['max'],
                    total_to_joint_grad_norm_median=m['total_to_joint_grad_norm']['median'],
                    total_to_joint_grad_norm_max=m['total_to_joint_grad_norm']['max']))
        p=[v for v in s['parameter_gradients'] if v['head']==head]
        assert [v['batch'] for v in p]==list(range(1,17))
        ratios=[v['mapped_aux_norm']/v['size_norm'] for v in p if v['size_norm']>0]
        param.append(dict(head=head,batches=16,zero_size_stem_batches=sum(v['size_norm']==0 for v in p),
            zero_mapped_stem_batches=sum(v['mapped_aux_norm']==0 for v in p),
            mapped_aux_size_ratio_median=statistics.median(ratios) if ratios else None,
            mapped_negative_batches=sum(v['mapped_cosine'] is not None and v['mapped_cosine']<0 for v in p)))
    with (OUT/'edge_mapping_comparison.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(core[0]));writer.writeheader();writer.writerows(core)
    initial_zeros=sum(v['zero_sample'] for v in core if v['head']=='initial')
    later_zeros=sum(v['zero_sample'] for v in core if v['head']!='initial')
    status='mapping_passed_but_training_blocked_by_initial_zero_descriptors' if initial_zeros else 'mapping_passed_training_stability_not_proven'
    save('edge_mapping_decision.json',dict(status=status,geometric_mapping_passed=True,
        initial_zero_edge_views=initial_zeros,later_zero_edge_views=later_zeros,
        training_started=False,automatic_training_authorized=False,adoption=False,
        implementation_bug_confirmed=False,performance_evidence=False,
        note='No new epsilon, exclusion, fallback or gating was searched. Nonzero descriptors and mapping correctness alone do not prove performance.'))
    save('edge_mapping_local_verification.json',dict(passed=True,core_manifest_files=3,grouped_rows=855,
        forward_views=18001,initial_neutral_views=6003,mathematical_checks=7,
        domain_video_direction_accounting=True,previous_mixing_and_near_tie_metrics_match=True,
        local_source_SHA_verified=True,prior_pinned_receipts_match=True,
        decoder_reconstruction_max_abs=0,jacobian_and_aux_gradient_outside_mask_max=0,
        parameter_gradient_summary=param,updates=0,GPU=False,TEST=False))
    print(json.dumps(dict(passed=True,seconds=c['elapsed_seconds'],status=status,initial_zero_edge_views=initial_zeros,
        later_zero_edge_views=later_zeros,comparison=core,parameter_summary=param),ensure_ascii=False))


def finalize():
    verified=read('edge_mapping_local_verification.json')
    assert verified['passed']
    manifest=read('edge_mapping_zero_chain_manifest.json')
    assert len(manifest)==2 and all(sha(OUT/n)==v for n,v in manifest.items())
    z=read('edge_mapping_zero_chain_summary.json');zp=read('edge_mapping_zero_chain_protocol.json')
    assert z['passed'] and z['CPU_only'] and z['parameters_inputs_unchanged']
    assert z['updates']==0 and not z['TEST'] and not z['CUDA_initialized'] and not z['optimizer_created']
    assert z['source_SHA']==zp['source_SHA']==sha(ROOT/'crane_project/tools/check_lvsr_edge_mapping_zero_chain_v1.py')
    assert z['elapsed_seconds']<120 and z['forward_views']==5999 and z['zero_edge_views']==13
    actual={(r['split'],r['edge']):r['n'] for r in z['groups']}
    assert actual=={('train_s1','long'):4,('train_s05','long'):3,('val','long'):6}
    assert all(r['sample_norm_max']==r['stem_grad_norm_max']==0 for r in z['groups'])
    assert all(math.isfinite(r['feature_grad_norm_max']) and math.isfinite(r['proxy_grad_norm_max']) for r in z['groups'])
    main_completion=read('edge_mapping_completion.json')
    total_seconds=main_completion['elapsed_seconds']+z['elapsed_seconds']
    save('edge_mapping_zero_chain_local_verification.json',dict(passed=True,manifest_files=2,
        zero_cases=13,counts_match_main_review=True,all_zero_cases_stem_gradient_zero=True,
        source_SHA_verified=True,combined_CPU_seconds=total_seconds,total_forward_views=24000,updates=0,TEST=False,GPU=False))
    save('edge_mapping_final_decision.json',dict(status='engineering_feasible_one_fresh_control_worth_considering',
        mapping_exact=True,initialization_no_zero_descriptors=True,performance_proven=False,
        original_L5_gradient_explosion_proven=False,final_adoption=False,
        training_started=False,TEST=False,
        next_configuration='Same bank20x648, per-location detached decoder-role masks retaining original channels, per-edge1D targets and conditional arrangement; mean of edges.',
        next_budget='One fresh24-epoch run, original data/order/LR; no continuation or parameter search.',
        next_safety='Stop on nonfinite/identity failure or TRAIN selected descriptor/proxy norm<=originalEPS; no dropping views, epsilon tuning or inference fallback change.',
        ranking='Original full24 representative ranking; gates reported without prefiltering; no automatic TEST.',
        paper_impact='Auxiliary descriptor, detached role assignment,1D associations and conditional proxy-arrangement formulas need description; inference remains2480 parameters.'))
    core=[]
    with (OUT/'edge_mapping_comparison.csv').open() as f:
        for row in csv.DictReader(f):
            if row['edge']=='long': core.append(row)
    table='\n'.join('| {head} / {split} | {mixed}/{n} | {zero_sample}/{n} |'.format(**r) for r in core)
    text='''# 原通道保留、按解码归属选择的辅助表征审查

结论：解决了上轮静态分支标签的映射障碍，工程上可以作为一个独立对照候选；尚未训练，也没有尺寸收益证据。初始化和PRIME ep5无零描述，L5的13个零描述案例经链式梯度核查并非参数梯度爆炸。仅新增review模块和检查入口，正式head／loss／trainer未改。

## 唯一审查定义与方法影响

沿原图坐标恢复、逐位置两对边范数和原sort索引取得长／短角色。保留原648维的分支、通道、位置索引，不交换两分支隐藏通道。对边e的辅助选择为：

```text
m[n,e,b,c,i] = sqrt(W[n,i]) * 1[original_sort_index[n,i,e] == b]
z[n,e] = normalize(flatten(f[n]) * m[n,e])
p[n,e,k] = normalize(proxy[k] * m[n,e])
```

选择索引和W均detach，仅用于训练辅助项，不参与推理支持／尺寸聚合／交付，不读取GT。代理和样本使用同一选择；仍20×648个代理参数，沿用原TRAIN二维中心在长／短边上的坐标投影，不聚类或扩代理。对齐使用对应一维目标；代理排列也需按同一逐样本选择条件化，原全局代理loss不能直接照搬。两边固定取平均，拟保留原两辅助系数各0.001、tau／alpha，避免因拆分加倍。

因此它修改的是训练表征、目标关联和代理排列，不是原PRIME公式不变的纯推理结构对照。若采用，需补上述选择／detach、一维关联及平均loss定义；推理2480参数、局部向量、参考、W、逐位置解码及整框回退保持。代理仍12960参数，训练总15440，不新增推理门控或置信度头。这里证明的是特征位置的依赖对应；同一分支的stem权重仍在空间位置间共享，不能保证一次参数更新后另一边不变或准确边不受损。论文正文未修改。

## 已完成的核验

两端7项数学检查通过，覆盖角色混合／原通道保留、选择分割、未选择位置隔离、GT目标维度隔离、detach梯度、整体分支重命名不变性和零描述检测。原初始化、L5、PRIME ep5各5999个有输出TRAIN／VAL视图，另查初始化4个无输出视图，共18001前向；主审查{main_seconds:.2f}秒。6003视图中性输出精确重现M；原解码重建差0，辅助选择以外的实际尺寸Jacobian和辅助特征梯度均为0。L5／PRIME CPU回放差≤9.16e−5，中心／方向／分数及输出继承通过。原模型／父权重／输入／选权与源码SHA保持，0更新／无CUDA／优化器／TEST。

| 快照／集合 | 角色混合视图 | 长边辅助描述为零 |
|---|---:|---:|
{table}

短边描述在三个快照均无零；所有选后代理范数非零。PRIME的320／139个混合视图已经纳入正确映射，而非丢弃。近等长仍有硬选择敏感性：PRIME原／半尺度，0.1%长度差以内的位置支持质量占比均值0.0427%／0.0198%，1%以内为0.3582%／0.1623%，单视图1%以内最高16.05%／9.88%。初始化另有极少量近方形TRAIN视图在1%以内质量占比达100%，不能说硬选择在所有视图稳定；本轮三快照均无精确等长支持位置，不引入平滑或新阈值。

原epoch6首16自然批，初始化尺寸loss的stem梯度为0（零终端的预期行为），新辅助项16批均给stem非零梯度。冻结PRIME上，加权新辅助／尺寸stem梯度范数比中位0.7466%，负夹角4/16；冻结L5为5.0654%、12/16。该读数使用未按新目标训练的代理（初始化为原随机代理，L5／PRIME为旧PRIME ep5代理），不能据夹角或loss大小预测候选优劣。

## 零描述补查与边界

新增CPU{zero_seconds:.2f}秒补查，仅核对已发现的13个L5长边零描述（TRAIN原尺度4、半尺度3、VAL6），扫描5999有输出视图、不做更新。逐视图半权重对齐的特征梯度范数最高约1.49×10^6，但实际stem参数梯度全部为0；末端ReLU阻断了该路径。对齐项代理梯度最大约1.12×10^−7，属于接近零的数值残差。这是冻结L5施加新辅助定义后的观察，不是L5原训练的梯度爆炸证据。它提示死激活／零描述可能令辅助信号失效，而非证明它是尺寸损伤的根因；初始化和PRIME ep5未出现此问题。

合计CPU约{total_seconds:.2f}秒、24000前向。主审查855个分组与域／视频／方向计数、上一轮混合／近等长统计以及13例链式梯度核心均独立核验通过。只回传核心JSON／CSV，未下载权重、逐帧数据、特征或梯度数组，未新建服务器目录。

## 下一步建议

可以保留这个唯一候选，下一步才实现一个从原初始化开始的24轮独立对照；不要从PRIME ep5续训，也不同时改归一化、卷积、学习率或数据。正式接入前须检查新头／代理共同状态恢复、零辅助退化为原训练、推理不引入选择计算，并锁定原5112视图顺序与曝光。使用原全部24轮预定代表排序，门槛只报告不预筛，M增量、L5改进和R4整链分开；两边MAE／达标／RIoU／AP75、准确边、域／视频全部报告，不自动TEST。

建议预设停止条件：新训练出现非有限、身份变化或任一TRAIN选后描述／代理范数≤原EPS即停止该配置；不跳过视图、不扫epsilon、不加门控／新回退。若完成后仍是边／视频间误差转移，保留失败并结束这个方向。本次止于审查，未建立正式候选配置或启动训练，不提前填写VAL改善。原L5及selected=null、PRIME ep5相对M增量和已暴露TEST代价保持。

服务器复用`/media/omnisky/personal_files/ljj/symEOOD/work_dirs/lvsr_diagnostic_20261112_v1/prime_001/edge_mapping_*`。候选ep5身份仍为`9590597e713d562c3fa953cd96866657d1d17de14d00c822155592e0a2d77301`，本轮无新权重。
'''.format(main_seconds=main_completion['elapsed_seconds'],zero_seconds=z['elapsed_seconds'],total_seconds=total_seconds,table=table)
    with (OUT/'edge_mapping_analysis.md').open('x') as f:f.write(text)
    print(json.dumps(dict(passed=True,combined_CPU_seconds=total_seconds,decision='one_fresh_control_worth_considering',training_started=False)))


if __name__=='__main__':
    if '--finalize' in sys.argv: finalize()
    else: main()
