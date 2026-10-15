#!/usr/bin/env python3
"""One saved-result audit, never trains, calibrates, extracts, or opens TEST."""
import argparse
from bisect import bisect_right
import csv
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_reliability_paired_ranking_audit_v1 as core

VERSION = 'port_reliability_paired_ranking_audit_v1'
MANIFEST = ROOT/'crane_project/tools/port_reliability_paired_ranking_audit_v1_manifest.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def source_pins(main_file):
    # Python3.8 preserves relative __file__ when invoked as `python path.py`.
    paths = (Path(main_file).resolve(),ROOT/'crane_project/utils'/(VERSION+'.py'),
             MANIFEST,ROOT/'tests'/('test_'+VERSION+'.py'),
             ROOT/'tools'/('run_'+VERSION+'.sh'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def load_rows(path):
    # This entry point intentionally has no TEST role or arbitrary input glob.
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def pin_check(root, manifest):
    missing = []
    for name, expected in manifest['pins'].items():
        p = root/name
        if not p.is_file():
            missing.append(name)
        elif sha(p) != expected:
            raise ValueError('Changed pinned input: ' + name)
    return missing


def load_experiment(root, exp, policy, references):
    directory = root/exp['directory']
    report = read(directory/'report.json')
    receipt = read(directory/'completion.json')
    # Validate consumed artifact bytes against the original completion, without
    # opening weights, arrays, or TEST. Old source drift is reported separately.
    for name in ('report.json', 'models.json', 'cutoffs.json',
                 'scored_TRAIN.jsonl.gz', 'scored_VAL.jsonl.gz'):
        if sha(directory/name) != receipt['artifacts'][name]:
            raise ValueError('Original completion mismatch: ' + name)
    if report['parameter_counts'] != exp['parameter_counts']:
        raise ValueError('Architecture metadata changed')
    cuts = report.get('VAL_cutoffs', report.get('cutoffs'))
    if cuts != read(directory/'cutoffs.json'):
        raise ValueError('Original cutoffs disagree')
    methods = [exp['id']+'/'+a for a in exp['arms']]
    result = {}
    for role, expected in (('TRAIN', 2558), ('VAL', 887)):
        rows = load_rows(directory/('scored_'+role+'.jsonl.gz'))
        by_id = core.validate_rows(rows, role, expected)
        if role in references:
            core.pair_identity(references[role], by_id)
        else:
            references[role] = deepcopy(by_id)
        for r in rows:
            source = r[exp['risk_field']]
            r['audit_risks'] = {m: source[a] for m, a in zip(methods, exp['arms'])}
            r['audit_risks'].update(simple=source['full_simple'], score=source['score_only'])
            out = r['pred'] is not None
            for m in r['audit_risks']:
                risk = r['audit_risks'][m]
                if out != (risk is not None):
                    raise ValueError('Risk/output support changed')
                if out and (not math.isfinite(risk) or not 0 <= risk <= 1):
                    raise ValueError('Invalid risk')
            original = r['original_simple_decision']
            formal_cut = policy['cutoffs']['simple']['size']['risk_le']
            if out and (source['full_simple'] != original['risks']['size'] or
                        (source['full_simple'] <= formal_cut) != original['size_accepted'] or
                        source['score_only'] != 1-r['pred'][5]):
                raise ValueError('Formal score/flag mismatch')
            for arm in exp['arms']:
                decision = deepcopy(original)
                if out:
                    decision['risks']['size'] = source[arm]
                    decision['size_accepted'] = source[arm] <= cuts[arm]['risk_le']
                if decision != r['candidate_decisions'][arm]:
                    raise ValueError('Historical delivery mismatch')
            # Every saved simple score must also agree across experiments.
            ref = references[role][r['image']]
            if ref['original_simple_decision'] != original:
                raise ValueError('Original flags differ across runs')
        result[role] = rows
    train_ids = {r['image'] for r in result['TRAIN']}
    if train_ids & {r['image'] for r in result['VAL']}:
        raise ValueError('Overlapping roles')
    return result, {exp['id']+'/'+a: cuts[a]['risk_le'] for a in exp['arms']}, report


def expected_states(report, role, arm, group):
    if 'statistics' in report:
        return report['statistics'][role][arm][group]['actual']['states']
    key = 'train_statistics' if role == 'TRAIN' else 'VAL_statistics'
    return report[key][arm][group]['actual']['states']


def comparison(a, b, rows, a_delivery, b_delivery):
    # Both historical endpoints are anchors: avoid choosing a flattering one.
    anchors = {}
    for name, d in (('candidate_frozen', a_delivery), ('control_frozen', b_delivery)):
        n, cr = d['accepted'], d['states']['CR']
        anchors[name] = dict(same_count=dict(candidate=a.same_count(n), control=b.same_count(n)),
                             same_CR=dict(candidate=a.same_cr(cr), control=b.same_cr(cr)))
    region = core.compare_high_region(a, b)
    grid = {str(x): dict(candidate=a.same_cr(math.ceil(x*a.good)),
                        control=b.same_cr(math.ceil(x*b.good)))
            for x in (.95, .975, .99, 1.)}
    # High-CR target point independent check, including whole cutoff ties.
    for pair in grid.values():
        for rank, point in ((a, pair['candidate']), (b, pair['control'])):
            cut = point['risk_le']
            kept = [r for r in rows if r['pred'] is not None and cut is not None
                    and r['audit_risks'][rank.method] <= cut]
            if (sum(core.bad(r) for r in kept), sum(not core.bad(r) for r in kept)) != (point['FA'], point['CR']):
                raise ValueError('Independent same-CR mismatch')
    return dict(anchors=anchors, diagnostic_retention_grid=grid,
                high_correct_retention=region, offline_only=True)


def audit(root, manifest):
    missing = pin_check(root, manifest)
    policy = read(root/manifest['policy'])['simple_policy']
    references, datasets, cutoffs, reports, availability = {}, {}, {}, {}, []
    for exp in manifest['experiments']:
        required = [exp['directory']+'/'+n for n in ('report.json', 'models.json', 'completion.json',
                    'cutoffs.json', 'scored_TRAIN.jsonl.gz', 'scored_VAL.jsonl.gz')]
        absent = [p for p in required if p in missing]
        if absent:
            availability.append(dict(id=exp['id'], status='UNAVAILABLE_NO_RERUN', missing=absent))
            continue
        data, cuts, report = load_experiment(root, exp, policy, references)
        datasets[exp['id']] = data
        cutoffs.update(cuts)
        reports[exp['id']] = report
        availability.append(dict(id=exp['id'], status='SAVED_SCORES_AVAILABLE',
                                 frames={r: len(v) for r, v in data.items()}))
    if not datasets:
        raise ValueError('No saved scores available')
    cutoffs.update(simple=policy['cutoffs']['simple']['size']['risk_le'],
                   score=policy['cutoffs']['score_only']['size']['risk_le'])
    roles, curves, checks = {}, [], []
    for role in ('TRAIN', 'VAL'):
        joined = deepcopy(list(references[role].values()))
        for r in joined:
            d = r['original_simple_decision']
            r['audit_risks'] = dict(simple=d['risks']['size'],
                                    score=None if r['pred'] is None else 1-r['pred'][5])
        by_id = {r['image']: r for r in joined}
        for data in datasets.values():
            for r in data[role]:
                by_id[r['image']]['audit_risks'].update(r['audit_risks'])
        roles[role] = {}
        for group, rows in core.groups(joined).items():
            ranks = {m: core.Ranking(rows, m) for m in cutoffs}
            stats = {}
            for m, rank in ranks.items():
                kept = {r['image'] for r in rank.ordered if r['audit_risks'][m] <= cutoffs[m]}
                d = core.summarize(rows, kept)
                check = core.independent_check(rank, d, cutoffs[m])
                checks.append(dict(role=role, group=group, method=m, **check))
                if '/' in m:
                    expid, arm = m.split('/')
                    if d['states'] != expected_states(reports[expid], role, arm, group):
                        raise ValueError('Historical confusion differs: '+role+'/'+group+'/'+m)
                stats[m] = dict(frozen_delivery=d, frozen_risk_le=cutoffs[m], AUROC=rank.auc(),
                                good=rank.good, bad=rank.wrong,
                                small_error_support=rank.wrong < 10,
                                diagnostic_correct_retention={str(x): rank.same_cr(math.ceil(x*rank.good))
                                    for x in manifest['diagnostic_retention_grid']})
                curves.extend(dict(role=role, group=group, method=m, **p) for p in rank.curve())
            pairs = {}
            for m in cutoffs:
                if m in ('simple', 'score'):
                    continue
                expid, arm = m.split('/')
                controls = ['simple', 'score'] + [other for other in cutoffs
                            if other.startswith(expid+'/') and other != m]
                pairs[m] = {c: comparison(ranks[m], ranks[c], rows,
                            stats[m]['frozen_delivery'], stats[c]['frozen_delivery']) for c in controls}
            roles[role][group] = dict(methods=stats, comparisons=pairs)
    return dict(version=VERSION, inventory=manifest['experiments'], excluded=manifest['excluded'],
                availability=availability, missing_pinned_files=missing,
                strict_capacity_conclusion=manifest['strict_capacity_conclusion'],
                historical_source_drift=manifest['historical_source_drift'],
                original_contracts={name: value['contract'] for name, value in reports.items()},
                roles=roles, scope=manifest['scope'],
                status='SAVED_RESULT_AUDIT_COMPLETE_NO_AUTOMATIC_ADVANCEMENT',
                qualifications=['TRAIN scores are in-sample; VAL is repeatedly used development data.',
                 'Same-count splits ties by identity and reports FA bounds; same-CR uses whole ties.',
                 'All matching cutoffs are offline GT diagnostics, never online policies.',
                 '95%-100% is a descriptive region, not a new protection contract.',
                 'Small/correlated video support and correlated curve targets are not independent evidence.',
                 'No inference/feature/model replay; only saved scores and cached identities verified.',
                 'Historical TEST conclusions remain in prior records; no TEST files opened.',
                 'Source drift is disclosed; original artifacts and receipts are unchanged.']), curves, checks


def render(report):
    lines = ['# 已有判断器配对排序核对', '',
             '本轮只复用保存评分；零训练、零检测推理、零特征提取、零TEST访问、零policy更新。', '',
             '## 可比性与归因', '',
             '未找到可隔离纯容量作用的严格配对。五组完整TRAIN/VAL的框、GT、身份和原三标志逐帧一致；输入、读出或目标的差异分别披露。', '',
             '|实验|参数|可回答的问题与限制|', '|---|---|---|']
    for e in report['inventory']:
        lines.append('|'+e['id']+'|'+str(e['parameter_counts'])+'|'+e['attribution']+'|')
    lines += ['', '旧B geometry/ROI/structure MLP前端不同，排除当前M的成绩拼接。', '',
              '## 总体冻结点与高正确保留排序', '',
              'FA/FR/ED/CR来自原冻结门限（统一套用TRAIN与VAL），不是重新标定。', '',
              '|方法|TRAIN FA/FR/ED/CR|VAL FA/FR/ED/CR|TRAIN/VAL AUROC|TRAIN CR95 FA|VAL CR95 FA|',
              '|---|---|---|---|---|---|']
    tr = report['roles']['TRAIN']['all']['methods']
    va = report['roles']['VAL']['all']['methods']
    def state(x):
        return '/'.join(str(x['frozen_delivery']['states'][s]) for s in ('FA','FR','ED','CR'))
    for m in tr:
        a, b = tr[m], va[m]
        auc = lambda v: 'NA' if v is None else '%.6f'%v
        lines.append('|%s|%s|%s|%s / %s|%s|%s|' % (m,state(a),state(b),auc(a['AUROC']),auc(b['AUROC']),
            a['diagnostic_correct_retention']['0.95']['FA'],b['diagnostic_correct_retention']['0.95']['FA']))
    lines += ['', '## 与正式simple比较：同CR及同接受数', '',
              '下表同时报告原simple点和候选点作为离线锚点，完整各域/逐视频见report.json和CSV。', '',
              '|方法|数据|simple原点CR：候选/simple FA|候选原点CR：候选/simple FA|候选原点接受数：候选/simple FA|95%-100% CR区间：优/平/劣/不确定|',
              '|---|---|---|---|---|---|']
    for m in tr:
        if m in ('simple', 'score'):
            continue
        for role in ('TRAIN','VAL'):
            p = report['roles'][role]['all']['comparisons'][m]['simple']
            def pair(x):
                return '%s/%s'%(x['candidate']['FA'],x['control']['FA'])
            h=p['high_correct_retention']
            lines.append('|%s|%s|%s|%s|%s|%s|'%(m,role,
                pair(p['anchors']['control_frozen']['same_CR']),
                pair(p['anchors']['candidate_frozen']['same_CR']),
                pair(p['anchors']['candidate_frozen']['same_count']),
                '/'.join(str(h[k]) for k in ('better','equal','worse','tie_inconclusive'))))
    lines += ['', '区间计数是同一批相关样本上的相关工作点，不是独立重复实验；不以某一个有利点选候选。', '',
              '## 分域逐视频的支持与曲线', '',
              '|数据/组|正确/错误/缺失|simple AUROC|', '|---|---|---|']
    for role, groups in report['roles'].items():
        for group, result in groups.items():
            x=result['methods']['simple'];s=x['frozen_delivery']['states']
            lines.append('|%s/%s|%s/%s/%s|%s|'%(role,group,x['good'],x['bad'],s['MISSING'],x['AUROC']))
    lines += ['', '错误少于10个的组仅描述；单类AUROC为null。分组参照独立匹配，不相加充当全局策略。', '',
              '完整冻结点：metrics.csv；两类锚点匹配：matched_points.csv；每个整tie风险—覆盖/正确保留点：curves.jsonl.gz；高保留区间全体整数CR比较：report.json。', '',
              '## 结论：收益在哪一环节消失', '',
              '|配对/候选|TRAIN→VAL观察|判断|', '|---|---|---|',
              ]
    conclusions = {
        'redc': '|ReDC vs同输入linear|TRAIN高保留区间反而较差，VAL高保留区间优于linear；相对simple的VAL优势局部且逐视频不一致|不是单纯容量增益，也不是只修门限即可解决|',
        'readout': '|signed residual vs temperature|TRAIN高保留区间全优，VAL同一区间全劣|本次读出增量在迁移时反转；不继续扩大网络|',
        'native': '|native vs midpoint来源|TRAIN两者排序已完美；native在VAL总体仍有局部同CR/同数增益，但不是逐视频一致|部分排序信息确实迁移，不能写成VAL完全无增益；Sim保护仍受损|',
        'rank': '|within-video rank vs BCE|TRAIN高保留区间没有新增优势；VAL总体多数点改善，但seq07及Sim高保留区域仍有退化|目标对照存在局部增量，不是容量结论；不足以直接采用或归为纯校准问题|',
        'spatial': '|spatial axis vs spatial BCE|TRAIN辅助目标有增量；VAL部分点保留，但整体相对simple仍弱|有限增量未形成正式升级；不能用AUROC小升或loss下降放行|'}
    for entry in report['availability']:
        if entry['status'] == 'SAVED_SCORES_AVAILABLE':
            lines.append(conclusions[entry['id']])
        else:
            lines.append('|%s|缺少既有评分/产物|不补跑，无法作当前配对结论|'%entry['id'])
    lines += ['',
              '### VAL各视频：simple原工作点的同CR比较', '',
              '每组独立离线匹配，目标CR未必都在95%-100%区间。下表不相加为可部署全局策略。', '',
              '|方法|视频|目标CR|候选/simple FA|95%-100%区间优/平/劣/不确定|',
              '|---|---|---|---|---|']
    for m in ('redc/redc','readout/residual','native/native','rank/within_video_rank','spatial/spatial_axis'):
        if m not in va:
            continue
        for group, value in report['roles']['VAL'].items():
            if not group.startswith('sequence:'):
                continue
            p=value['comparisons'][m]['simple']
            x=p['anchors']['control_frozen']['same_CR'];h=p['high_correct_retention']
            lines.append('|%s|%s|%s|%s/%s|%s|'%(m,group,x['control']['CR'],x['candidate']['FA'],x['control']['FA'],
                '/'.join(str(h[k]) for k in ('better','equal','worse','tie_inconclusive'))))
    lines += ['',
              '纯容量结论：证据不足。当前有VAL局部排序优势（尤其native和视频内排序），但没有证实一个兼顾逐视频正确保护的稳定升级；不启动校准搜索、不采用候选、不新增TEST。', '',
              '原各域/逐视频95%正确保留的单一全局工作点及晋级要求已经检验失败，不换成校准名义重复搜索。若未来重议工作点，须先明确区别于既有合同的实际使用要求及其代价，另行讨论有限设计；本轮没有新候选进入执行。单调校准不能改变排序或创造不存在的可行工作点。', '',
              '历史源码漂移：'+json.dumps(report['historical_source_drift'],ensure_ascii=False), '',
              '源码漂移仅描述当前文件与旧运行pin的不同；本轮不用该旧审查器重新生成评分，也不重写旧manifest。']
    return '\n'.join(lines)+'\n'


def export_csv(out, report):
    metrics = []
    matches = []
    for role, groups in report['roles'].items():
        for group, result in groups.items():
            for m, v in result['methods'].items():
                d=v['frozen_delivery']
                metrics.append(dict(role=role, group=group, method=m, AUROC=v['AUROC'],
                    **d['states'], **{k:x for k,x in d.items() if k != 'states'}))
            for m, controls in result['comparisons'].items():
                for c,p in controls.items():
                    for anchor, kinds in p['anchors'].items():
                        for kind, values in kinds.items():
                            a,b=values['candidate'],values['control']
                            matches.append(dict(role=role,group=group,method=m,control=c,anchor=anchor,
                                matching=kind,candidate_FA=a['FA'],control_FA=b['FA'],
                                candidate_CR=a['CR'],control_CR=b['CR'],
                                candidate_FR=a['FR'],control_FR=b['FR'],
                                candidate_accepted=a['accepted'],control_accepted=b['accepted'],
                                candidate_tie=json.dumps(a.get('tie_bounds',a.get('exact_CR'))),
                                control_tie=json.dumps(b.get('tie_bounds',b.get('exact_CR')))))
    for name, rows in (('metrics.csv', metrics), ('matched_points.csv', matches)):
        with (out/name).open('x', newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(out):
    if out.is_symlink() or out.name != 'result' or out.resolve().parent.parent != ROOT/'work_dirs'/VERSION:
        raise ValueError('Use work_dirs/version/RUN_ID/result')
    if out.exists():
        raise FileExistsError(out)
    manifest=read(MANIFEST)
    report,curves,checks=audit(ROOT,manifest)
    if pin_check(ROOT,manifest) != report['missing_pinned_files']:
        raise ValueError('Input availability changed while reading')
    out.mkdir(parents=True,exist_ok=False)
    report['input_pins']=manifest['pins']
    report['manifest_sha256']=sha(MANIFEST)
    report['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    report['source_pins']=source_pins(__file__)
    write(out/'report.json',report)
    export_csv(out,report)
    with gzip.open(out/'curves.jsonl.gz','xt') as f:
        for p in curves:
            f.write(json.dumps(p,sort_keys=True,allow_nan=False)+'\n')
    (out/'analysis.md').write_text(render(report))
    write(out/'independent_review.json',dict(passed=True,checks=checks,
        group_methods=len(checks),curve_points=len(curves),
        pair_comparisons=sum(x['pair_comparisons'] for x in checks),
        historical_confusions_reproduced=True,identity_and_flags_unchanged=True))
    write(out/'completion.json',dict(status=report['status'],scope=manifest['scope'],
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(status=report['status'],experiments=len(report['availability']),
                          checks=len(checks),curves=len(curves))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
