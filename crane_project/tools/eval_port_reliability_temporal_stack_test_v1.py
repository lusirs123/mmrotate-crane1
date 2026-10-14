#!/usr/bin/env python3
"""Frozen six-method VAL/TEST supplement; cached native features, CPU only."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.utils import port_reliability_temporal_stack_test_v1 as core
from crane_project.tools import run_port_reliability_conditional_evidence_v1 as io


def checked(allow_absent=False):
    p=json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_protocol.json')).read_text())
    s=json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_sources.json')).read_text())
    if p['methods']!=list(core.METHODS) or p['ranking_controls']!=list(core.CONTROLS) or p['counts']!=core.test.COUNTS:
        raise ValueError('Frozen scope changed')
    pins=dict(p['input_pins'],**p['frontend_pins'],**s['sources']);absent=[]
    for n,h in pins.items():
        if allow_absent and n in p['frontend_pins'] and not (ROOT/n).exists():absent.append(n);continue
        if io.sha(ROOT/n)!=h:raise ValueError('Input/source changed: '+n)
    paths={k:ROOT/v for k,v in p['paths'].items()}
    load=lambda k,n:json.loads((paths[k]/n).read_text())
    reports={k:load(k,'report.json') for k in ('temporal','pauc','crossfit','fallback','veto')}
    for k,r in reports.items():
        review=load(k,'server_review.json')
        if not review['passed'] or review['report_sha256']!=io.sha(paths[k]/'report.json') or r['TEST_opened'] or r['candidate_adopted']:
            raise ValueError('Frozen reviewed source differs')
    c=p['frozen_cutoffs']
    if (load('temporal','calibration.json')['fusion4']['risk_le']!=c['fusion4'] or
        load('pauc','calibration.json')['risk_le']!=c['partial_auc4'] or
        load('crossfit','calibration.json')['crossfit_fusion']['risk_le']!=c['crossfit_fusion'] or
        load('veto','train_cutoff.json')['risk_le']!=c['temporal_veto'] or
        reports['fallback']['frozen_cutoffs']!=dict(supported=c['crossfit_fusion'],fallback=c['partial_auc4'])):
        raise ValueError('Frozen working point changed')
    models=load('temporal','models.json');pm=load('pauc','model.json')
    if models!=reports['temporal']['models'] or models!=reports['pauc']['base_models'] or pm!=reports['pauc']['model']:
        raise ValueError('Model/report mismatch')
    bundle=dict(models=models,pauc=pm,fusion=load('crossfit','fusion_models.json')['crossfit_fusion'],
        head=load('crossfit','quality_heads.json')['full'],policy=json.loads(paths['policy'].read_text())['simple_policy'],cutoffs=c)
    previous=load('native_test','completion.json');review=load('native_test','mac_review.json')
    if not previous['TEST_evaluated'] or not review['passed']:raise ValueError('Unreviewed native TEST cache')
    return p,pins,absent,paths,bundle


def test_rows(paths,bundle):
    records=[json.loads(l) for l in (ROOT/'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/frozen_test_20261007_163625_1260566/test_decisions.jsonl').read_text().splitlines()]
    native=[json.loads(l) for l in (ROOT/'work_dirs/port_geometry_size_boundary_v1/size_candidate_downstream_20261007_191454_1852169/collect/port_test/predictions.jsonl').read_text().splitlines()]
    rows=core.test.bind_rows(records,native)
    for r in rows:r['reliability_role']='test'
    with np.load(paths['native_test']/'test_native_projected.npz',allow_pickle=False) as a:
        ids=a['images'].tolist();x=np.array(a['features'],dtype=float)
    if ids!=[r['image'] for r in rows if r['pred'] is not None] or x.shape!=(1431,259) or not np.isfinite(x).all():
        raise ValueError('Complete cached native feature identity differs')
    descriptors=np.array([core.base.simple.descriptor(r['pred'],r['image_size']) for r in rows if r['pred'] is not None])
    np.testing.assert_allclose(x[:,:3],descriptors,atol=1e-12,rtol=0)
    head=bundle['head'];ri=core.base.simple.sigmoid(core.fusion.image.numpy_logits(head['model'],x,head['normalizer']))
    values={r['image']:None for r in rows};values.update(dict(zip(ids,map(float,ri))))
    return rows,values


def run(args):
    p,pins,absent,paths,bundle=checked(args.allow_absent_frontend)
    if args.check_only:print('Frozen six-method contract verified; no TEST scoring, fit or calibration');return
    if absent:raise ValueError('Server run requires frontend SHA verification')
    out=args.out
    if out is None or out.is_symlink() or out.name!='result' or out.resolve().parent.parent!=(ROOT/'work_dirs'/core.VERSION).resolve():raise ValueError('Use unified work_dirs/version/RUN_ID/result')
    out.mkdir(parents=True,exist_ok=False)
    io.write(out/'frozen_bundle.json',bundle)  # Immutable before opening TEST rows/features.
    val=io.read(paths['fallback']/'scored_VAL.jsonl.gz')
    if len(val)!=887 or any(r['split']!='val' for r in val):raise ValueError('Full frozen VAL required')
    replayed=core.replay(val,{r['image']:r['image_crossfit_risk'] for r in val},bundle)
    for old,new in zip(sorted(val,key=lambda r:r['image']),sorted(replayed,key=lambda r:r['image'])):
        if old['temporal']!=new['temporal'] or old['fallback_decision']!=new['flags']['fallback_margin']:
            raise ValueError('Previous VAL decision/history identity differs')
        for m in ('fusion4','partial_auc4','crossfit_fusion'):
            a,z=old['risks'][m],new['risks'][m]
            if (a is None)!=(z is None) or (a is not None and abs(a-z)>1e-12):raise ValueError('Previous VAL risk identity differs')
    rows,ri=test_rows(paths,bundle)
    if {r['image'] for r in val}&{r['image'] for r in rows}:raise ValueError('VAL/TEST overlap')
    scored={'VAL':replayed,'TEST':core.replay(rows,ri,bundle)}
    roles={k:core.evaluate(v) for k,v in scored.items()}
    for role,part in scored.items():
        with gzip.open(out/('scored_'+role+'.jsonl.gz'),'xt') as f:
            for r in part:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    report=dict(protocol=core.VERSION,status='FROZEN_EXPOSED_TEST_REPORT_COMPLETE',input_pins=pins,frozen_cutoffs=p['frozen_cutoffs'],
        methods=p['methods'],roles=roles,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        fits=0,calibration_calls=0,detector_inferences=0,detector_updates=0,quality_updates=0,threshold_updates=0,
        models_frozen_before_TEST=True,TEST_used_for_selection=False,TEST_repeatedly_exposed=True,VAL_failures_retracted=False,
        automatic_promotion=False,formal_policy_changed=False,GT_online=False,GPU_used=False,
        limitations=['VAL-selected thresholds have no TEST retention guarantee.','TEST repeatedly exposed, not untouched independent confirmation.',
            'Same-count and same-CR are offline comparisons only.','Fallback-margin ordering uses frozen branch cutoffs, not calibrated probabilities.',
            'Cached native features, no new inference or deployment latency measurement.'])
    for n,h in pins.items():
        if io.sha(ROOT/n)!=h:raise ValueError('Frozen input/source changed')
    io.write(out/'report.json',report);io.write(out/'completion.json',dict(status=report['status'],frames=2327,report_sha256=io.sha(out/'report.json')))
    for role in roles:
        print(role,{m:v['states'] for m,v in roles[role]['groups']['all']['actual'].items()},flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path);a.add_argument('--check-only',action='store_true');a.add_argument('--allow-absent-frontend',action='store_true');run(a.parse_args())
