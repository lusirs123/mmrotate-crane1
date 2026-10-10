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
    from crane_project.utils import port_reliability_state_continuity_v1 as states
    from crane_project.utils import port_reliability_complementarity_v1 as metrics
    compare({g:states.summarize(part,{r['image'] for r in part if r['original_simple_decision']['size_accepted']})
             for g,part in metrics.grouped(val).items()},report['formal_policy_point'])
    runner.write(receipt,dict(passed=True,TRAIN_rows=len(oof),VAL_rows=len(val),
        all_labels_models_scores_states_cutoffs_and_gate_recomputed=True,
        status=report['status'],report_sha256=runner.sha(out/'report.json'),TEST_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--receipt',required=True)
    args=p.parse_args();review(Path(args.out),Path(args.receipt))
