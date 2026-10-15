#!/usr/bin/env python3
"""Read-only saved TRAIN/VAL disagreement supplement; no fitting or TEST API."""
import argparse
from bisect import bisect_right
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_reliability_paired_ranking_audit_v1 as rank

VERSION = 'port_midpoint_simple_disagreement_v1'
MANIFEST = ROOT/'crane_project/tools'/ (VERSION+'_manifest.json')
OLD40 = 'work_dirs/port_reliability_complementarity_v1/20261009_complementarity_v1/result'
OLD72 = 'work_dirs/port_reliability_paired_ranking_audit_v1/20261015_saved_pairs_server_v1_fix1/result'
SOURCE = 'work_dirs/port_reliability_feature_source_v1/20261009_source_v1/result'
CELLS = ('both_accept', 'simple_only', 'mlp_only', 'both_reject')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def rows(p):
    with gzip.open(p, 'rt') as f:
        return [json.loads(line) for line in f]


def signed_errors(r):
    """GT only in this offline error description, never in score features."""
    if r['pred'] is None:
        return None
    p = sorted(r['pred'][2:4], reverse=True)
    g = sorted(r['gt'][2:4], reverse=True)
    if any(not math.isfinite(x) or x <= 0 for x in p+g):
        raise ValueError('Invalid dimensions')
    return [(a-b)/b for a, b in zip(p, g)]


def quantile(values, q):
    if not values:
        return None
    v = sorted(values)
    x = (len(v)-1)*q
    i = math.floor(x)
    j = math.ceil(x)
    return v[i]+(v[j]-v[i])*(x-i)


def error_description(part):
    result = dict(axis=Counter(), direction=Counter(), severity=Counter())
    values = []
    for r in part:
        e = signed_errors(r)
        if e is None or max(map(abs, e)) <= .1:
            continue
        flags = [abs(x) > .1 for x in e]
        axis = 'both' if all(flags) else ('L_only' if flags[0] else 'S_only')
        failing = [x for x in e if abs(x) > .1]
        direction = 'small' if all(x < 0 for x in failing) else ('large' if all(x > 0 for x in failing) else 'mixed')
        maximum = max(map(abs, e))
        severity = '10_15' if maximum <= .15 else ('15_25' if maximum <= .25 else 'over25')
        result['axis'][axis] += 1
        result['direction'][direction] += 1
        result['severity'][severity] += 1
        values.append(maximum)
    result = {k: dict(v) for k, v in result.items()}
    result.update(wrong=len(values), maximum_relative_error=dict(median=quantile(values,.5),
                    p95=quantile(values,.95), maximum=max(values) if values else None))
    return result


def cell_name(simple, mlp):
    return ('both_accept' if mlp else 'simple_only') if simple else ('mlp_only' if mlp else 'both_reject')


def table(part, simple_ids, mlp_ids):
    cells = {k: [] for k in CELLS}
    missing = []
    for r in part:
        if r['pred'] is None:
            if r['image'] in simple_ids or r['image'] in mlp_ids:
                raise ValueError('Missing accepted')
            missing.append(r['image'])
        else:
            cells[cell_name(r['image'] in simple_ids, r['image'] in mlp_ids)].append(r)
    result = {}
    for name, group in cells.items():
        errors = sum(rank.bad(r) for r in group)
        result[name] = dict(outputs=len(group), correct=len(group)-errors, wrong=errors,
                            error_fraction=errors/len(group) if group else None,
                            images=[r['image'] for r in group], error_types=error_description(group))
    change = dict(rescued_FA=result['simple_only']['wrong'], new_FR=result['simple_only']['correct'],
                  restored_CR=result['mlp_only']['correct'], new_FA=result['mlp_only']['wrong'])
    simple = rank.summarize(part, simple_ids)
    mlp = rank.summarize(part, mlp_ids)
    assert mlp['states']['FA']-simple['states']['FA'] == change['new_FA']-change['rescued_FA']
    assert mlp['states']['FR']-simple['states']['FR'] == change['new_FR']-change['restored_CR']
    return dict(cells=result, MISSING=len(missing), missing_images=missing, changes=change,
                simple=simple, mlp=mlp, oracle_only=dict(
                potentially_removable_FA=change['rescued_FA'],potentially_restorable_CR=change['restored_CR'],
                GT_selects_changes=True, executable_policy=False))


def accepted(ranking, point, kind):
    if kind == 'same_count':
        return {r['image'] for r in ranking.ordered[:point['accepted']]}
    cut = point['risk_le']
    return {r['image'] for r in ranking.ordered if cut is not None and r['audit_risks'][ranking.method] <= cut}


def scalar_auc(part, feature):
    good = [feature(r) for r in part if rank.bad(r) is False]
    errors = [feature(r) for r in part if rank.bad(r) is True]
    if not good or not errors:
        return None
    return sum((e > g)+.5*(e == g) for e in errors for g in good)/(len(errors)*len(good))


def score_evidence(part):
    """Only two existing scores, fixed risk direction; no fit, grid, routing or rule search."""
    features = {'simple': lambda r:r['audit_risks']['simple'],
                'mlp': lambda r:r['audit_risks']['mlp']}
    good = [r for r in part if rank.bad(r) is False]
    errors = [r for r in part if rank.bad(r) is True]
    dominance = Counter(dict(error_higher_both=0,error_lower_both=0,conflicting=0,identical=0))
    for e in errors:
        for g in good:
            ds = e['audit_risks']['simple']-g['audit_risks']['simple']
            dm = e['audit_risks']['mlp']-g['audit_risks']['mlp']
            if ds == 0 and dm == 0:
                key = 'identical'
            elif ds >= 0 and dm >= 0:
                key = 'error_higher_both'
            elif ds <= 0 and dm <= 0:
                key = 'error_lower_both'
            else:
                key = 'conflicting'
            dominance[key] += 1
    pairs = len(good)*len(errors)
    return dict(correct=len(good),wrong=len(errors),pairs=pairs,
                AUROC={name:scalar_auc(part, f) for name,f in features.items()},
                fixed_direction='higher_saved_risk_means_error; no sign selection',
                pair_dominance=dict(dominance),
                pair_fractions={k:v/pairs if pairs else None for k,v in dominance.items()},
                small_error_support=len(errors)<10,
                limitations='Conditional, correlated descriptive pairs. Conflicting scores do not establish a learnable fusion rule. No classifier fitted.')


def independent_table_check(part, result):
    """Reconstruct cell counts from marginal confusion plus transitions, separate from table()."""
    s, m = result['simple']['states'], result['mlp']['states']
    a = result['changes'];c = result['cells']
    expected = dict(both_accept=(s['CR']-a['new_FR'],s['FA']-a['rescued_FA']),
                    simple_only=(a['new_FR'],a['rescued_FA']),
                    mlp_only=(a['restored_CR'],a['new_FA']),
                    both_reject=(s['FR']-a['restored_CR'],s['ED']-a['new_FA']))
    for name,(good,bad) in expected.items():
        assert (c[name]['correct'],c[name]['wrong']) == (good,bad)
        assert bad == c[name]['error_types']['wrong']
        assert all(sum(c[name]['error_types'][k].values())==bad for k in ('axis','direction','severity'))
    assert sum(c[n]['outputs'] for n in CELLS)+result['MISSING']==len(part)
    assert m['FA']==c['both_accept']['wrong']+c['mlp_only']['wrong']
    assert m['CR']==c['both_accept']['correct']+c['mlp_only']['correct']


def analyze(root, manifest):
    for p,h in manifest['pins'].items():
        if sha(root/p) != h:
            raise ValueError('Pinned input changed: '+p)
    old40=read(root/OLD40/'report.json')
    old72=read(root/OLD72/'report.json')
    source=read(root/SOURCE/'report.json')
    for directory, consumed in ((OLD40, ('report.json','joined_TRAIN.jsonl.gz','joined_VAL.jsonl.gz')),
                                (OLD72, ('report.json',)),
                                (SOURCE, ('report.json','scored_TRAIN.jsonl.gz','scored_VAL.jsonl.gz'))):
        receipt=read(root/directory/'completion.json')
        for name in consumed:
            assert sha(root/directory/name)==receipt['artifacts'][name]
    policy=read(root/manifest['policy'])['simple_policy']
    cuts=dict(simple=policy['cutoffs']['simple']['size']['risk_le'],
              mlp=source['VAL_cutoffs']['midpoint']['risk_le'])
    assert cuts['simple']==old40['cutoffs']['formal_simple']
    assert cuts['mlp']==old40['cutoffs']['source_midpoint']
    output={};checks=0
    for role, n in (('TRAIN',2558),('VAL',887)):
        joined=rows(root/OLD40/('joined_'+role+'.jsonl.gz'))
        saved=rows(root/SOURCE/('scored_'+role+'.jsonl.gz'))
        ref=rank.validate_rows(joined,role,n);current=rank.validate_rows(saved,role,n)
        rank.pair_identity(ref,current)
        for r in joined:
            original=current[r['image']]
            assert r['risks']['source_midpoint']==original['experiment_risks']['midpoint']
            assert r['risks']['full_simple']==original['experiment_risks']['full_simple']
            r['audit_risks']=dict(simple=r['risks']['full_simple'],mlp=r['risks']['source_midpoint'])
            if r['pred'] is not None:
                assert (r['audit_risks']['simple']<=cuts['simple'])==r['original_simple_decision']['size_accepted']
                assert (r['audit_risks']['mlp']<=cuts['mlp'])==original['candidate_decisions']['midpoint']['size_accepted']
        output[role]={}
        for name,part in rank.groups(joined).items():
            ranks={m:rank.Ranking(part,m) for m in cuts}
            keep={m:{r['image'] for r in ranks[m].ordered if r['audit_risks'][m]<=cuts[m]} for m in cuts}
            frozen=table(part,keep['simple'],keep['mlp']);independent_table_check(part,frozen)
            prior=old40['results'][role]['references']['formal_simple']['methods']['source_midpoint'][name]
            assert frozen['changes']==prior['changes']
            assert frozen['simple']['states']==prior['baseline']['states']
            assert frozen['mlp']['states']==prior['candidate']['states']
            for key, cell in (('rescued_FA','simple_only'),('new_FR','simple_only'),('restored_CR','mlp_only'),('new_FA','mlp_only')):
                assert len(prior['change_images'][key])==frozen['changes'][key]
                wrong_change=key in ('rescued_FA','new_FA')
                expected_ids={r['image'] for r in part if r['image'] in frozen['cells'][cell]['images']
                              and rank.bad(r) is wrong_change}
                assert set(prior['change_images'][key])==expected_ids
            matched={}
            for anchor, summary in (('simple_frozen',frozen['simple']),('mlp_frozen',frozen['mlp'])):
                matched[anchor]={}
                for kind in ('same_count','same_CR'):
                    pts={m:(r.same_count(summary['accepted']) if kind=='same_count' else r.same_cr(summary['states']['CR'])) for m,r in ranks.items()}
                    result=table(part,accepted(ranks['simple'],pts['simple'],kind),accepted(ranks['mlp'],pts['mlp'],kind))
                    independent_table_check(part,result)
                    old_anchor='control_frozen' if anchor=='simple_frozen' else 'candidate_frozen'
                    previous=old72['roles'][role][name]['comparisons']['native/midpoint']['simple']['anchors'][old_anchor][kind]
                    for m,oldkey in (('simple','control'),('mlp','candidate')):
                        assert pts[m]['FA']==previous[oldkey]['FA']
                        assert pts[m]['CR']==previous[oldkey]['CR']
                    matched[anchor][kind]=dict(points=pts,disagreements=result,
                        offline_GT_diagnostic_only=True,prior72_reproduced=True)
                    checks+=1
            conditions={}
            for condition in (*CELLS,'all_disagreement'):
                ids=set(frozen['cells'][condition]['images']) if condition in CELLS else set(frozen['cells']['simple_only']['images']+frozen['cells']['mlp_only']['images'])
                conditions[condition]=score_evidence([r for r in part if r['image'] in ids])
            # Old conditional-on-simple-accept was already evaluated. Reuse it,
            # do not create a second historical identity or recalculate a gate.
            output[role][name]=dict(frozen=frozen,matched=matched,
                score_only_evidence=conditions,
                reused_conditional_simple_accept=prior['conditional_on_simple_accept'],
                reused_AND=prior['reject_only_AND'],reused_AND_comparison=prior['AND_comparison'])
            checks+=1
    return dict(version=VERSION,scope=manifest['scope'],cutoffs=cuts,roles=output,
                reuse=dict(section40='formal_simple/source_midpoint full transition and conditional evidence reused and replay-checked',
                           section72='both frozen anchors same-count/same-CR points reused and replay-checked',
                           added='four cell layout, signed axis/type/severity, matched cell decomposition, fixed two-score conditional evidence'),
                verified_group_scenarios=checks,training_updates=0,policy_updates=0,
                status='DISAGREEMENT_AUDIT_COMPLETE_FUSION_NOT_AUTHORIZED_BY_THIS_REPORT',
                decision=dict(minimal_fusion_evidence_sufficient=False,
                  reason='TRAIN complement exists; VAL newly detected errors are confined to seq07 at frozen cutoffs, pooled same-count/same-CR net ranking is worse. TRAIN disagreements lack harmful counterexamples; conditional VAL score signals are descriptive and do not establish a transferable fusion rule.',
                  not_a_theoretical_impossibility=True,no_training_started=True),
                limitations=['TRAIN in-sample, VAL repeatedly exposed; no independent confirmation.',
                 'No requirement of all metrics improving in all videos is introduced.',
                 'Conditional AUROC/dominance does not establish a trained or deployable fusion model.',
                 'Error types/GT/video identity only describe results, never enter score evidence.',
                 'TEST not opened, inferred, or used to choose any rule.'])


def markdown(report):
    lines=['# midpoint MLP与正式simple：分歧互补核对','',
      '复用第40节正式simple参考和第72节匹配点；没有拟合、门限搜索、推理、特征提取或TEST访问。', '',
      '## 原冻结门限四组', '',
      '|数据/组|共同接受 正确/错误|simple接受MLP拒绝 正确/错误|simple拒绝MLP接受 正确/错误|共同拒绝 正确/错误|缺失|',
      '|---|---|---|---|---|---|']
    for role,groups in report['roles'].items():
        for name,value in groups.items():
            c=value['frozen']['cells']
            lines.append('|%s/%s|%s|%s|'%(role,name,'|'.join('%s/%s'%(c[k]['correct'],c[k]['wrong']) for k in CELLS),value['frozen']['MISSING']))
    lines += ['', '## 匹配工作点：总体互补分解','',
      '|数据|锚点/匹配|新增检出错误|新增误拒正确|恢复正确|重新接受错误|MLP/simple FA|MLP/simple CR|',
      '|---|---|---|---|---|---|---|---|']
    for role in report['roles']:
        for anchor,kinds in report['roles'][role]['all']['matched'].items():
            for kind,v in kinds.items():
                a=v['disagreements']['changes'];p=v['points']
                lines.append('|%s|%s/%s|%s|%s|%s|%s|%s/%s|%s/%s|'%(role,anchor,kind,
                  a['rescued_FA'],a['new_FR'],a['restored_CR'],a['new_FA'],p['mlp']['FA'],p['simple']['FA'],p['mlp']['CR'],p['simple']['CR']))
    lines += ['', '同CR接受整个cutoff tie；同数按身份拆tie，FA上下界见report。各视频离线独立匹配，不能相加为部署策略。','',
      '## 原门限分歧中的两分数证据','',
      '|数据/组|分歧方向|正确/错误|simple错误AUROC|MLP错误AUROC|', '|---|---|---|---|---|']
    for role,groups in report['roles'].items():
        for group,v in groups.items():
            for cell in ('simple_only','mlp_only'):
                x=v['score_only_evidence'][cell]
                fmt=lambda z:'NA' if z is None else '%.6f'%z
                lines.append('|%s/%s|%s|%s/%s|%s|%s|'%(role,group,cell,x['correct'],x['wrong'],fmt(x['AUROC']['simple']),fmt(x['AUROC']['mlp'])))
    lines += ['', '高风险固定表示错误，不根据VAL翻转方向；单类AUROC为空。联合两分数的同向/冲突配对仅作描述，完整计数在report.json。它不等于证明可训练融合器。', '',
      '## 继续条件的证据边界','',
      '只有TRAIN真实互补、VAL非单视频同方向收益，以及不依赖GT/视频身份的可用证据同时获得支持，才值得另行讨论最小融合。新增检出计数或GT oracle单独不足以放行。原门限与匹配点分开解释，不用原TEST微小净增益选择规则。', '',
      '### 本轮结论：证据不足以支持最小融合训练','',
      'TRAIN确有互补：MLP新增检出12错误、无新增误拒，恢复1037正确、无重新接受错误。但两种分歧方向分别为纯错误和纯正确，缺少误伤/放回错误反例，不能证明两分数能学到可信的冲突选择。', '',
      'VAL原门限新增检出9错误全部来自seq07，同时新增误拒3正确；恢复10正确却重新接受25错误。以simple点匹配同CR653时仍有16个补充检出，但同时放回21个错误，净FA195对190；同843接受FA193对190。不是完全无互补，而是净收益没有保留且新增检出集中。', '',
      'VAL的MLP-only组，MLP评分条件AUROC .892提供一个有限排序线索；但TRAIN对应组没有错误反例，且seq07仅2个正确、seq14仅1个错误、Sim无正确对照。不能据此挑门限或称跨视频稳定。simple-only组只有3正确/9错误，两分数AUROC .4815/.4444，没有确认可迁移的纠错/误伤区分依据。', '',
      '因此停在分析，不启动融合训练；这不证明所有两分数融合都不可能。没有新增逐视频全面胜出的硬门槛，也未用原TEST选择规则。分轴误差、偏小/偏大/混合、10%-15%/15%-25%/>25%见error_types.csv，中位/P95/最大值见report；均为离线GT描述。']
    return '\n'.join(lines)+'\n'


def export(out,report):
    four=[];errors=[];matches=[]
    for role,groups in report['roles'].items():
        for group,v in groups.items():
            for cell,x in v['frozen']['cells'].items():
                four.append(dict(role=role,group=group,cell=cell,correct=x['correct'],wrong=x['wrong'],outputs=x['outputs'],error_fraction=x['error_fraction']))
                e=x['error_types']
                for category in ('axis','direction','severity'):
                    for name,n in e[category].items():
                        errors.append(dict(role=role,group=group,cell=cell,category=category,type=name,count=n))
            for anchor,kinds in v['matched'].items():
                for kind,x in kinds.items():
                    a=x['disagreements']['changes'];p=x['points']
                    matches.append(dict(role=role,group=group,anchor=anchor,matching=kind,**a,
                        simple_FA=p['simple']['FA'],mlp_FA=p['mlp']['FA'],simple_CR=p['simple']['CR'],mlp_CR=p['mlp']['CR'],
                        simple_tie=json.dumps(p['simple'].get('tie_bounds',p['simple'].get('exact_CR'))),
                        mlp_tie=json.dumps(p['mlp'].get('tie_bounds',p['mlp'].get('exact_CR')))))
    for name,records in (('four_groups.csv',four),('matched_complementarity.csv',matches),('error_types.csv',errors)):
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def run(out):
    if out.is_symlink() or out.exists() or out.name!='result' or out.resolve().parent.parent!=ROOT/'work_dirs'/VERSION:
        raise ValueError('Use a new work_dirs/version/RUN_ID/result')
    manifest=read(MANIFEST)
    report=analyze(ROOT,manifest)
    for p,h in manifest['pins'].items():
        assert sha(ROOT/p)==h
    out.mkdir(parents=True,exist_ok=False)
    report['input_pins']=manifest['pins']
    report['source_pins']={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__).resolve(),MANIFEST,
        ROOT/'tests'/('test_'+VERSION+'.py'),ROOT/'tools'/('run_'+VERSION+'.sh'))}
    report['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    write(out/'report.json',report);export(out,report)
    (out/'analysis.md').write_text(markdown(report))
    write(out/'completion.json',dict(status=report['status'],scope=manifest['scope'],
         artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(status=report['status'],scenarios=report['verified_group_scenarios'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
