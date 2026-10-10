#!/usr/bin/env python3
"""CPU independent reconstruction of every TRAIN label, fit and VAL decision."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_oof_v1 as runner
from crane_project.utils import port_reliability_oof_v1 as core
from crane_project.utils import port_reliability_within_video_rank_v1 as ranking


def read(path):
    with gzip.open(path,'rt') as f:return [json.loads(l) for l in f]


def compare(a,b):
    if isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('Report keys differ')
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError('Report list length differs')
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,float):
        if not np.isclose(a,b,atol=1e-10,rtol=1e-10):raise ValueError('Report scalar differs')
    elif a!=b:raise ValueError('Report state differs')


def review(out,receipt):
    completion=json.loads((out/'completion.json').read_text())
    if completion.get('failure_stage')=='midpoint_training_support':
        support=json.loads((out/completion['support_report']).read_text())
        empty=[d for d,scales in support.items() if not sum(v['eligible'] for v in scales.values())]
        compare(sorted(empty),sorted(completion['empty_domains']))
        if not empty or completion['status']!='AUXILIARY_SOURCE_FAILED_STOP':
            raise ValueError('Invalid empty-support stop')
        runner.write(receipt,dict(passed=True,stage='midpoint_support_failed_stop',
            saved_support_consistency_checked=True, GPU_predictions_independently_replayed=False,
            VAL_scored=False));return
    if completion.get('failure_stage')=='fold_oof_source':
        folder=out/completion['fold']
        gate=core.source_gate(read(folder/'oof_predictions.jsonl.gz'), require_two_classes=False)
        compare(gate,json.loads((folder/'oof_source_gate.json').read_text()))
        if gate['passed'] or completion['status']!='AUXILIARY_SOURCE_FAILED_STOP':
            raise ValueError('Invalid fold source stop')
        runner.write(receipt,dict(passed=True,stage='fold_source_failed_stop',
            source_gate_recomputed=True,VAL_scored=False));return
    rows,policy=runner.original.load_rows()
    train=[r for r in rows if r['reliability_role']=='train']
    aux=read(out/'A/oof_predictions.jsonl.gz')+read(out/'B/oof_predictions.jsonl.gz')
    oof=core.materialize_training(train,aux)
    compare(oof,read(out/'OOF_TRAIN.jsonl.gz'))
    material=json.loads((out/'training_material.json').read_text())
    compare(material,dict(source_gate=core.source_gate(oof),original=core.support(train),oof=core.support(oof)))
    if not material['source_gate']['passed']:
        runner.write(receipt,dict(passed=True,stage='source_failed_stop',TRAIN_rows=len(oof),VAL_scored=False));return
    common={r['image'] for r in oof if r['pred'] is not None}&{r['image'] for r in train if r['pred'] is not None}
    models=dict(in_sample=core.fit(train),oof=core.fit(oof),
        common_in_sample=core.fit([r for r in train if r['image'] in common]),
        common_oof=core.fit([r for r in oof if r['image'] in common]))
    compare(models,json.loads((out/'quality_models.json').read_text()))
    val=core.score([r for r in rows if r['reliability_role']=='val'],models)
    compare(val,read(out/'scored_VAL.jsonl.gz'))
    report=json.loads((out/'report.json').read_text())
    cutoffs={m:ranking.calibrate(val,m) for m in core.METHODS}
    stats=core.statistics(val,cutoffs)
    compare(cutoffs,report['cutoffs']);compare(stats,report['statistics']);compare(core.gate(stats),report['gate'])
    candidate=json.loads((out/'candidate_policy.json').read_text())
    from crane_project.utils.port_midpoint_sigma15_reliability_v1 import Sigma15Reliability
    online=Sigma15Reliability(candidate,policy['front_end'])
    compare(candidate['simple_policy']['models']['size'],models['oof'])
    compare(candidate['simple_policy']['cutoffs']['simple']['size'],cutoffs['oof'])
    compare(candidate['simple_policy']['models']['angle'],policy['simple_policy']['models']['angle'])
    if 'full_VAL_GT_correct_retention95' not in candidate['simple_policy']['calibration_role']:
        raise ValueError('Incorrect candidate calibration provenance')
    for row in val:
        expected=ranking.decide(row['original_simple_decision'],row['risks']['oof'],cutoffs['oof']['risk_le'])
        compare(online.decide(row['pred'],row['image_size']),expected)
    from crane_project.utils import port_reliability_state_continuity_v1 as states
    from crane_project.utils import port_reliability_complementarity_v1 as metrics
    compare({g:states.summarize(part,{r['image'] for r in part if r['original_simple_decision']['size_accepted']})
             for g,part in metrics.grouped(val).items()},report['formal_policy_point'])
    runner.write(receipt,dict(passed=True,TRAIN_rows=len(oof),VAL_rows=len(val),
        all_labels_models_scores_states_cutoffs_and_gate_recomputed=True,
        exported_policy_online_decisions_replayed=True,
        status=report['status'],report_sha256=runner.sha(out/'report.json'),TEST_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--receipt',required=True)
    args=p.parse_args();review(Path(args.out),Path(args.receipt))
