"""Genuine within-video error ranking on a frozen simple anchor; no online GT."""
from copy import deepcopy
import math
import numpy as np
from crane_project.utils import port_reliability_complementarity_v1 as metrics
from crane_project.utils import port_reliability_state_continuity_v1 as states

VERSION = 'port_reliability_within_video_rank_v1'
ARMS = ('bce_control', 'within_video_rank')
METHODS = ARMS + ('full_simple', 'score_only')
SETTINGS = dict(seed=1701, epochs=100, batch_size=256, lr=.001,
    weight_decay=.0001, clip_norm=5., residual_bound=.5, dtype='float64',
    hidden=[16,8], parameter_count=4290, correct_retention=.95,
    feature_dimensions=259, residual_input_dimensions=258, rank_weight=.25)


def normalize(features, normalizer):
    x = np.asarray(features, dtype=np.float64)
    m, s = (np.asarray(normalizer[k], dtype=np.float64) for k in ('mean','scale'))
    if (x.ndim != 2 or x.shape[1] != 259 or m.shape != (259,) or s.shape != (259,)
            or not np.isfinite(x).all() or not np.isfinite(m).all()
            or not np.isfinite(s).all() or np.any(s <= 0)):
        raise ValueError('Invalid frozen native feature normalization')
    return (x-m)/s


def anchors(rows, arm):
    if arm not in ARMS or any(r['pred'] is None for r in rows):
        raise ValueError('Anchors require a named arm and actual outputs')
    return checked_anchor([r['size_risks']['full_simple'] for r in rows])


def build_pairs(rows):
    """Offline TRAIN-only indices for every genuine same-video bad/good pair."""
    if (not rows or any(r.get('reliability_role') != 'train'
                       or r['split'] not in ('train','train_sim')
                       or r['pred'] is None for r in rows)):
        raise ValueError('Pairs require actual complete TRAIN outputs only')
    ids = [r['image'] for r in rows]
    if len(set(ids)) != len(ids) or ids != sorted(ids):
        raise ValueError('Pair identities must be unique and in frozen image order')
    groups = {}
    for i,r in enumerate(rows):
        if r['domain'] != r['sequence'].split('_')[0]:
            raise ValueError('Video/domain mismatch')
        group = groups.setdefault(r['sequence'],dict(bad=[],good=[]))
        group['bad' if metrics.bad(r) else 'good'].append(i)
    if any(not v['bad'] or not v['good'] for v in groups.values()):
        raise ValueError('Each contracted TRAIN video needs both genuine labels')
    return {name:dict(**v,pairs=len(v['bad'])*len(v['good']))
            for name,v in sorted(groups.items())}


def pair_counts(plan):
    return {name:dict(bad=len(v['bad']),good=len(v['good']),pairs=v['pairs'])
            for name,v in plan.items()}


def numpy_rank_loss(logits, plan):
    s = np.asarray(logits,dtype=np.float64)
    if s.ndim != 1 or not np.isfinite(s).all() or not plan:
        raise ValueError('Invalid ranking inputs')
    terms = {}
    for name,v in plan.items():
        b,g = np.asarray(v['bad'],dtype=int),np.asarray(v['good'],dtype=int)
        if not len(b) or not len(g) or v['pairs'] != len(b)*len(g):
            raise ValueError('Invalid Cartesian pair plan')
        terms[name] = float(np.logaddexp(0.,s[g][None,:]-s[b][:,None]).mean())
    return float(np.mean(list(terms.values()))),terms


def ranking_summary(rows, method):
    videos = {name:v for name,v in metrics.grouped(rows).items()
              if name.startswith('sequence:')}
    values = {name:dict(bad=sum(metrics.bad(r) is True for r in v),
                       good=sum(metrics.bad(r) is False for r in v),
                       error_AUROC=metrics.auc(v,method)) for name,v in videos.items()}
    eligible = [v for v in values.values() if v['error_AUROC'] is not None]
    total = sum(v['bad']*v['good'] for v in eligible)
    return dict(videos=values,eligible_videos=len(eligible),genuine_pair_count=total,
        within_video_macro_AUROC=float(np.mean([v['error_AUROC'] for v in eligible])) if eligible else None,
        within_video_pair_weighted_AUROC=sum(v['error_AUROC']*v['bad']*v['good'] for v in eligible)/total if total else None,
        pooled_AUROC=metrics.auc(rows,method),pairs_are_not_independent_samples=True)


def checked_anchor(value):
    r = np.asarray(value, dtype=np.float64)
    if r.ndim != 1 or not np.isfinite(r).all() or np.any((r <= 0) | (r >= 1)):
        raise ValueError('Frozen anchors must lie strictly inside (0,1); no clipping')
    return r


def readout(anchor, delta):
    """Equivalent to sigmoid(logit(anchor)+delta), exact at zero, differentiable."""
    r = checked_anchor(anchor); d = np.asarray(delta, dtype=np.float64)
    if d.shape != r.shape or not np.isfinite(d).all() or np.any(np.abs(d) > .5):
        raise ValueError('Invalid bounded log-odds residual')
    e = np.expm1(d)
    return r*(1+e)/(1+r*e)


def numpy_forward(model, features, normalizer, anchor):
    h = normalize(features, normalizer)[:,1:]
    for n in (0,2,4):
        weight = np.asarray(model['network.%d.weight'%n], dtype=np.float64)
        bias = np.asarray(model['network.%d.bias'%n], dtype=np.float64)
        expected = {0:(16,258), 2:(8,16), 4:(1,8)}[n]
        if weight.shape != expected or bias.shape != (expected[0],):
            raise ValueError('Head dimensions differ from contract')
        h = h @ weight.T + bias
        if n != 4:
            h = np.maximum(h, 0.)
    delta = .5*np.tanh(h[:,0]+float(model['beta'][0]))
    return readout(anchor, delta), delta


def class_weights(labels):
    y = np.asarray(labels, dtype=np.float64)
    if y.ndim != 1 or not np.isin(y,[0.,1.]).all() or not 0 < y.sum() < len(y):
        raise ValueError('Both genuine TRAIN classes required')
    return np.where(y == 1, len(y)/(2*y.sum()), len(y)/(2*(len(y)-y.sum())))


def decide(original, risk, cutoff):
    """Only size flag/risk changes. Missing is not classified; no GT argument."""
    d = deepcopy(original)
    present = d['final_box_original'] is not None
    if d['center_accepted'] != present:
        raise ValueError('Center output identity differs')
    if not present:
        if risk is not None or d['size_accepted'] or d['angle_accepted']:
            raise ValueError('Missing output cannot be classified')
        return d
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (risk,cutoff)):
        raise ValueError('Invalid risk or cutoff')
    d['risks']['size'] = risk
    d['size_accepted'] = risk <= cutoff
    return d


def calibrate(rows, method):
    """One VAL cutoff; whole ties; each video/domain >=95% correct retention."""
    if not rows or any(r['split'] != 'val' for r in rows):
        raise ValueError('Cutoff fitting is complete VAL only, never TRAIN/TEST')
    requirements = {}
    for name, group in metrics.grouped(rows).items():
        good = sorted(r['risks'][method] for r in group if metrics.bad(r) is False)
        required = math.ceil(.95*len(good))
        requirements[name] = dict(good=len(good), required=required,
                                  first_whole_tie_risk_le=good[required-1] if required else None)
    cutoff = max(v['first_whole_tie_risk_le'] for v in requirements.values() if v['good'])
    return dict(risk_le=cutoff, requirements=requirements, uses_VAL_GT=True,
                single_global_cutoff=True, calibration_not_independent_validation=True,
                tie_policy='accept_entire_cutoff_tie')


def statistics(rows, cutoffs):
    result = {}
    for method in METHODS:
        result[method] = {}
        for name, group in metrics.grouped(rows).items():
            actual = states.summarize(group,metrics.accepted(group,method,cutoffs[method]['risk_le']))
            result[method][name] = dict(actual=actual, error_AUROC=metrics.auc(group,method),
                controls={m:metrics.matched(group,m,actual) for m in METHODS if m != method})
    return result


def gate(statistics):
    failures = []; real_video_gain = False
    for group, value in statistics['within_video_rank'].items():
        actual = value['actual']; count = actual['states']
        if count['CR'] < math.ceil(.95*(count['CR']+count['FR'])):
            failures.append(dict(group=group, check='correct_retention95'))
        for method, comparison in value['controls'].items():
            same_cr = comparison['same_CR']
            conditions = dict(same_count_FA_nonincrease=count['FA'] <= comparison['same_count_tie_bounds']['bad_min'],
                same_CR_exact=comparison['same_CR_exact'],
                same_CR_FA_nonincrease=count['FA'] <= same_cr['states']['FA'],
                longest_correct_FR_nonincrease=actual['runs']['correct_rejection']['longest'] <= same_cr['runs']['correct_rejection']['longest'])
            if group == 'all':
                conditions.update(overall_strict_same_count_FA_gain=count['FA'] < comparison['same_count_tie_bounds']['bad_min'],
                                  overall_strict_same_CR_FA_gain=count['FA'] < same_cr['states']['FA'])
            for check, passed in conditions.items():
                if not passed:
                    failures.append(dict(group=group, control=method, check=check))
            if (group.startswith('sequence:real_') and method == 'full_simple'
                    and comparison['same_CR_exact'] and count['FA'] < same_cr['states']['FA']):
                auc = value['error_AUROC']
                controls = [statistics[m][group]['error_AUROC'] for m in ('bce_control','full_simple')]
                if auc is not None and all(v is not None and auc > v for v in controls):
                    real_video_gain = True
    if not real_video_gain:
        failures.append(dict(group='Real_videos', check='at_least_one_same_CR_FA_and_within_video_AUROC_gain'))
    return dict(passed=not failures, failures=failures, selected_arm='within_video_rank' if not failures else None,
                no_control_reselection=True, TEST_allowed=not failures)


def markdown(report):
    lines = ['# 同视频真实尺寸错误排序：BCE与BCE＋排序监督', '', '**状态：%s**。'%report['status'],
        '完整TRAIN拟合，完整VAL标定并评价；VAL已反复开发暴露。未读取TEST。',
        '仅尺寸risk/flag变化，原M、中心、方向、score、框及输出身份保持。', '',
        '| VAL工作点 | 接受数 | FA | FR | ED | CR | 错误AUROC |', '|---|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        v = report['statistics']['VAL'][m]['all']; c = v['actual']['states']
        lines.append('|%s|%d|%d|%d|%d|%d|%.6f|'%(m,v['actual']['accepted_outputs'],c['FA'],c['FR'],c['ED'],c['CR'],v['error_AUROC']))
    lines += ['', '同正确保留为各组独立离线比较，不可相加；同接受数采用图像顺序处理并列，同时报告并列FA上下限。', '',
        '| 组 | 候选CR | 候选FA | 同CR BCE控制 FA | 同CR simple FA | 同CR score FA |', '|---|---:|---:|---:|---:|---:|']
    for group,v in report['statistics']['VAL']['within_video_rank'].items():
        c = v['actual']['states']; controls = v['controls']
        lines.append('|%s|%d|%d|%d|%d|%d|'%(group,c['CR'],c['FA'],*(controls[m]['same_CR']['states']['FA'] for m in ('bce_control','full_simple','score_only'))))
    lines += ['', '原正式policy另报，不与不同接受数的候选直接作为排序证明。',
        'TRAIN拟合及工程验证不代表泛化；概率值为平衡损失风险评分，不是校准正确概率。',
        '两臂各1000更新，同初始化、同BCE曝光、同全TRAIN头部前向、同FP64；唯一实验差别为0.25排序项参与更新。',
        '固定0.5约束log-odds残差；均以正式simple为锚点，视频身份仅用于TRAIN配对，不按域/视频在线切换。',
        '未通过门槛即停止，不反选A、不追加预算或重选权重；正式M/simple保持。', '']
    lines += ['', '32,437个同视频配对来自71个真实错误帧，不是独立错误事件；Sim只有1个TRAIN错误。',
              '固定0.25排序系数；A排序项仅记录，B参与反传；同更新数不声称同反向成本。', '',
              '| 全VAL排序 | pooled AUROC | 视频内宏平均AUROC | 视频内配对加权AUROC |',
              '|---|---:|---:|---:|']
    for method in METHODS:
        v=report['ranking']['VAL'][method]
        lines.append('|%s|%.6f|%.6f|%.6f|'%(method,v['pooled_AUROC'],v['within_video_macro_AUROC'],v['within_video_pair_weighted_AUROC']))
    return '\n'.join(lines)
