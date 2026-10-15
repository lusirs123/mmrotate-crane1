"""One preselected seed, frozen VAL thresholds, exposed TEST supplement; no fitting."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from crane_project.tools import run_port_reliability_response_warm_v1 as warm
from crane_project.tools import review_port_reliability_response_warm_v1 as audit
from crane_project.utils import port_reliability_response_stats_v1 as response
from crane_project.utils.port_reliability_paired_ranking_audit_v1 import Ranking, groups, independent_check, identity

ROOT = warm.ROOT
SOURCE = ROOT/'work_dirs/port_reliability_response_warm_v1/20261015_response_warm_gpu_v1/result'
VERSION = 'port_reliability_response_warm_test_v1'
SEED = 1701
METHODS = ('formal', 'score', 'parent_1701', 'bce_1701', 'hard_1701', 'soft_1701')
REPORT_SHA = '0c0f7f87d495f7699f75b3dbd683d35e9c1beb3ec26fac395e3c29391a8866b1'


def selected(models, cuts):
    names = METHODS[2:]
    return {k: models[k] for k in names}, {k: cuts[k] for k in METHODS}


def source_check(offline=False):
    if warm.sha(SOURCE/'report.json') != REPORT_SHA:
        raise ValueError('Frozen parent report changed')
    complete = json.loads((SOURCE/'completion.json').read_text())
    warm.pins({str(SOURCE/k): v for k,v in complete['artifacts'].items()})
    manifest = json.loads((SOURCE/'protocol.json').read_text())
    warm.pins(manifest['source_pins'])
    runtime_pins={**manifest['development_pins'], **manifest['val_pins']}
    warm.pins({p:v for p,v in runtime_pins.items() if not offline or (ROOT/p).exists()})
    freeze = json.loads((SOURCE/'freeze.json').read_text())
    if warm.sha(SOURCE/'models.json') != freeze['models_sha256']:
        raise ValueError('Frozen weights changed')
    return manifest, freeze


def log_diagnostic():
    records = [json.loads(s) for s in (SOURCE/'train_log.jsonl').read_text().splitlines()]
    names = ('bce_1701', 'hard_1701', 'soft_1701')
    return dict(records=[r for r in records if r['method'] in names],
        checkpoint_weights_saved_during_training=False,
        limitation='Logs contain loss/ESS/spread, not intermediate weights or high-retention ranking; cannot infer a causal time link or select a checkpoint.')


def run(out):
    if out.exists() or Path.cwd().resolve() != ROOT or out.parent.parent != ROOT/'work_dirs'/VERSION:
        raise ValueError('New result path required from project root')
    manifest, freeze = source_check()
    models, cuts = selected(json.loads((SOURCE/'models.json').read_text()), freeze['cutoffs'])
    out.mkdir(parents=True)
    warm.write(out/'protocol.json', dict(seed=SEED, methods=METHODS, source_report_sha256=REPORT_SHA,
        source_freeze_sha256=warm.sha(SOURCE/'freeze.json'), frozen_cutoffs=cuts,
        scorer_fits=0, TEST_repeatedly_exposed=True, TEST_used_for_selection=False,
        old_VAL_gate_overridden_for_reporting_by_user=True,
        source_pins={str(p.relative_to(ROOT)):warm.sha(p) for p in [Path(__file__).resolve(), SOURCE/'models.json', SOURCE/'train_log.jsonl']}))
    warm.write(out/'models.json', models)
    # Save evaluation contract before opening any TEST row/cache. Never select seed/model/cut from TEST.
    warm.write(out/'freeze.json', dict(cutoffs=cuts, models_sha256=warm.sha(out/'models.json'), seed=SEED, TEST_read=False))
    policy=json.loads((ROOT/warm.previous.history.POLICY).read_text())
    results={}
    for role in ('TRAIN', 'VAL', 'TEST'):
        if role != 'TEST':
            with gzip.open(SOURCE/('scored_'+role+'.jsonl.gz'),'rt') as f:
                rows=[json.loads(s) for s in f]
            shutil.copyfile(SOURCE/(role.lower()+'_responses.npz'), out/(role.lower()+'_responses.npz'))
            with np.load(out/(role.lower()+'_responses.npz')) as data:
                raw, ids=data['features'],data['images'].tolist()
        else:
            warm.pins(manifest['test_pins']); rows=warm.load_rows(role,manifest)
            from crane_project.utils.port_geometry_midpoint_sigma_v1 import SigmaMidpointHead
            torch.set_num_threads(1); torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
            torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
            payload=torch.load(str(ROOT/warm.previous.history.HEAD),map_location='cpu')
            if (payload['epoch'],payload['sigma_cells'],payload['updates']) != (3,1.5,2706):
                raise ValueError('Formal M changed')
            head=SigmaMidpointHead(1.5).to('cuda:0').eval().requires_grad_(False)
            head.load_state_dict(payload['head_state'],strict=True)
            raw,ids,proof=warm.previous.collect(rows,role,head,'cuda:0',out,manifest)
            warm.write(out/'TEST_replay_proof.json',proof)
        warm.previous.add_scores(rows,raw,ids,models,policy)
        for row in rows:
            row['response_decisions']={name:response.size_decision(row['original_simple_decision'],row['audit_risks'][name],cuts[name]) for name in models}
        with gzip.open(out/('scored_'+role+'.jsonl.gz'),'xt') as f:
            for row in rows: f.write(json.dumps(row,allow_nan=False)+'\n')
        results[role]=warm.metrics(rows,METHODS,cuts)
        print('Evaluated',role,flush=True)
    warm.write(out/'training_log_diagnostic.json',log_diagnostic())
    report=dict(status='FROZEN_1701_EXPOSED_TEST_SUPPLEMENT',results=results,cutoffs=cuts,
        decision=dict(new_selection=False,formal_policy_changed=False,old_VAL_gate_passed=False),
        scorer_fits=0,seed=SEED,TEST_read=True,TEST_repeatedly_exposed=True,TEST_used_for_selection=False)
    warm.dump_report(out,report)
    source_check()
    warm.write(out/'completion.json',dict(artifacts={p.name:warm.sha(p) for p in out.iterdir() if p.is_file()}))


def stable_identity(row, role):
    # Historical TEST lacks this optional metadata; preserve its TEST identity.
    return identity(dict(row,reliability_role=row.get('reliability_role',role.lower())))


def review(out, receipt, offline=False):
    manifest,freeze=source_check(offline)
    complete=json.loads((out/'completion.json').read_text())
    warm.pins({str(out/k):v for k,v in complete['artifacts'].items()})
    report=json.loads((out/'report.json').read_text())
    models=json.loads((out/'models.json').read_text())
    expected,cuts=selected(json.loads((SOURCE/'models.json').read_text()),freeze['cutoffs'])
    if models!=expected or report['cutoffs']!=cuts or tuple(report['results']['TEST'])!=METHODS:
        raise ValueError('Frozen contract changed')
    checks=[]; error=0.; staterror=0.;descerror=0.
    for role,results in report['results'].items():
        with gzip.open(out/('scored_'+role+'.jsonl.gz'),'rt') as f: rows=[json.loads(s) for s in f]
        original={r['image']:r for r in warm.load_rows(role,manifest)}
        if len(rows)!=len(original) or {r['image'] for r in rows}!=set(original): raise ValueError('Membership differs')
        for r in rows:
            if stable_identity(r,role)!=stable_identity(original[r['image']],role): raise ValueError('Output/GT identity changed')
            for name,d in r['response_decisions'].items():
                if d!=response.size_decision(r['original_simple_decision'],r['audit_risks'][name],cuts[name]): raise ValueError('Component flag changed')
        with np.load(out/(role.lower()+'_responses.npz')) as data:
            raw,ids,p,q=data['features'],data['images'].tolist(),data['probability'],data['prior']
        lookup={r['image']:r for r in rows if r['pred'] is not None}
        if set(ids)!=set(lookup) or len(ids)!=len(lookup): raise ValueError('Feature membership differs')
        for i,name in enumerate(ids):
            descerror=max(descerror,float(np.max(np.abs(raw[i,:3]-warm.simple.descriptor(lookup[name]['pred'],lookup[name]['image_size'])))))
            stat=np.array([a-b for pm,qm in zip(p[i],q[i]) for a,b in zip(audit.scalar_statistics(pm),audit.scalar_statistics(qm))])
            staterror=max(staterror,float(np.max(np.abs(raw[i,3:]-stat))))
        for name,m in models.items():
            values,_=audit.independent_model(m,raw)
            error=max(error,max(abs(v-lookup[i]['audit_risks'][name]) for i,v in zip(ids,values)))
        if warm.metrics(rows,METHODS,cuts)!=results: raise ValueError('Metric replay differs')
        for name,v in results.items():
            for group,part in groups(rows).items(): checks.append(independent_check(Ranking(part,name),v['at_frozen_VAL_cut'][group],cuts[name]))
        if role!='TEST':
            parent=json.loads((SOURCE/'report.json').read_text())['results'][role]
            if results!={k:parent[k] for k in METHODS}: raise ValueError('TRAIN/VAL changed')
    if error>1e-11 or max(staterror,descerror)>1e-12: raise ValueError('Independent scoring/statistics failed')
    omitted=[p for p in {**manifest['development_pins'],**manifest['val_pins']} if not (ROOT/p).exists()] if offline else []
    if omitted:
        server=json.loads((out/'independent_review_server.json').read_text())
        if server['status']!='INDEPENDENT_REVIEW_PASS' or server['report_sha256']!=warm.sha(out/'report.json'):
            raise ValueError('Missing full server review')
    warm.write(out/receipt,dict(status='INDEPENDENT_REVIEW_PASS',groups=len(checks),
        omitted_local_runtime_inputs_previously_verified_on_server=omitted,
        pairs=sum(c['pair_comparisons'] for c in checks),curve_points=sum(c['curve_points'] for c in checks),
        model_risk_max_abs=error,statistics_max_abs=staterror,descriptor_max_abs=descerror,report_sha256=warm.sha(out/'report.json')))
    print('INDEPENDENT_REVIEW_PASS',len(checks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--offline-review',action='store_true');parser.add_argument('--review',action='store_true');parser.add_argument('--receipt',default='independent_review.json')
    args=parser.parse_args()
    if args.review: review(args.out.resolve(),args.receipt,args.offline_review)
    else: run(args.out.resolve())
