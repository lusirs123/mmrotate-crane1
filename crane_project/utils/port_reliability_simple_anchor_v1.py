"""Bounded image residuals on frozen score/simple anchors; no online GT."""
from copy import deepcopy
import math
import numpy as np
from crane_project.utils import port_reliability_complementarity_v1 as metrics
from crane_project.utils import port_reliability_state_continuity_v1 as states

VERSION = 'port_reliability_simple_anchor_v1'
ARMS = ('score_anchor', 'simple_anchor')
METHODS = ARMS + ('full_simple', 'score_only')
SETTINGS = dict(seed=1701, epochs=100, batch_size=256, lr=.001,
    weight_decay=.0001, clip_norm=5., residual_bound=.5, dtype='float64',
    hidden=[16,8], parameter_count=4290, correct_retention=.95,
    feature_dimensions=259, residual_input_dimensions=258)


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
    return checked_anchor([1-r['pred'][5] if arm == 'score_anchor'
                           else r['size_risks']['full_simple'] for r in rows])


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
    for group, value in statistics['simple_anchor'].items():
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
                real_video_gain = True
    if not real_video_gain:
        failures.append(dict(group='Real_videos', check='at_least_one_strict_same_CR_simple_gain'))
    return dict(passed=not failures, failures=failures, selected_arm='simple_anchor' if not failures else None,
                no_control_reselection=True, TEST_allowed=not failures)


def markdown(report):
    lines = ['# 受限图像残差：score与simple锚点对照', '', '**状态：%s**。'%report['status'],
        '完整TRAIN拟合，完整VAL标定并评价；VAL已反复开发暴露。未读取TEST。',
        '仅尺寸risk/flag变化，原M、中心、方向、score、框及输出身份保持。', '',
        '| VAL工作点 | 接受数 | FA | FR | ED | CR | 错误AUROC |', '|---|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        v = report['statistics']['VAL'][m]['all']; c = v['actual']['states']
        lines.append('|%s|%d|%d|%d|%d|%d|%.6f|'%(m,v['actual']['accepted_outputs'],c['FA'],c['FR'],c['ED'],c['CR'],v['error_AUROC']))
    lines += ['', '同正确保留为各组独立离线比较，不可相加；同接受数采用图像顺序处理并列，同时报告并列FA上下限。', '',
        '| 组 | 候选CR | 候选FA | 同CR A FA | 同CR simple FA | 同CR score FA |', '|---|---:|---:|---:|---:|---:|']
    for group,v in report['statistics']['VAL']['simple_anchor'].items():
        c = v['actual']['states']; controls = v['controls']
        lines.append('|%s|%d|%d|%d|%d|%d|'%(group,c['CR'],c['FA'],*(controls[m]['same_CR']['states']['FA'] for m in ('score_anchor','full_simple','score_only'))))
    lines += ['', '原正式policy另报，不与不同接受数的候选直接作为排序证明。',
        'TRAIN拟合及工程验证不代表泛化；概率值为平衡损失风险评分，不是校准正确概率。',
        '两臂各1000更新，同初始化、同曝光、同FP64；唯一实验差别为锚点评分。',
        '固定0.5约束log-odds残差，不重新使用历史神经头评分，不按域/视频切换。',
        '未通过门槛即停止，不反选A、不追加预算或重选权重；正式M/simple保持。', '']
    return '\n'.join(lines)
