#!/usr/bin/env python3
"""CPU-only bounded temporal evidence check on complete saved TRAIN and VAL."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.utils import port_reliability_temporal_history_gate_v1 as core
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_midpoint_sigma15_reliability_v1 as binding

PROTOCOL=ROOT/'crane_project/tools/port_reliability_temporal_history_gate_v1_protocol.json'
SOURCES=ROOT/'crane_project/tools/port_reliability_temporal_history_gate_v1_sources.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def load(verify_frontend=True):
    protocol=json.loads(PROTOCOL.read_text());sources=json.loads(SOURCES.read_text())
    if protocol['settings']!=core.SETTINGS:raise ValueError('Settings changed')
    pins=dict(protocol['input_pins'],**sources['sources'])
    if verify_frontend:pins.update(protocol['frozen_frontend_identity'])
    for p,h in pins.items():
        if sha(ROOT/p)!=h:raise ValueError('Input/source differs: '+p)
    wrapped=json.loads((ROOT/protocol['policy']).read_text())
    binding.checked_frontend(wrapped['front_end'])
    if wrapped['protocol']!=binding.VERSION:raise ValueError('Wrong frontend policy')
    policy=wrapped['simple_policy'];simple.validate_policy(policy)
    with gzip.open(ROOT/protocol['rows'],'rt') as f:raw=[json.loads(l) for l in f]
    fields=['image','sequence','domain','split','frame_id','image_size','gt','pred',
        'reliability_role','size_risks','original_simple_decision']
    rows=[{k:r[k] for k in fields} for r in raw]
    if len(rows)!=3445:raise ValueError('Complete TRAIN/VAL required')
    manifest=json.loads((ROOT/protocol['data_manifest']).read_text())
    identities={r['id']:(r['split'],r['sequence']) for r in manifest['records']
        if r['split'] in ('train','train_sim','val')}
    if set(identities)!={r['image'] for r in rows}:raise ValueError('Manifest identities differ')
    for r in rows:
        if identities[r['image']]!=(r['split'],r['sequence']):raise ValueError('Role mismatch')
        d=r['original_simple_decision']
        if d['final_box_original']!=r['pred']:raise ValueError('Saved final M identity differs')
        if r['pred'] is None:
            if d['size_accepted'] or d['center_accepted'] or d['angle_accepted']:raise ValueError('Missing flag differs')
        else:
            x=simple.descriptor(r['pred'],r['image_size'])[None,:]
            for c in ('size','angle'):
                value=float(simple.linear_risk(policy['models'][c],x)[0])
                if abs(value-d['risks'][c])>1e-12:raise ValueError('Policy replay differs')
                if bool(value<=policy['cutoffs']['simple'][c]['risk_le'])!=d[c+'_accepted']:
                    raise ValueError('Original flag differs')
    parts={role:[r for r in rows if r['reliability_role']==role.lower()] for role in ('TRAIN','VAL')}
    if [len(parts[k]) for k in parts]!=[2558,887]:raise ValueError('Roles changed')
    sampling=json.loads((ROOT/protocol['sampling_manifest']).read_text())
    join=sampling['sequences']['real_seq12']['first_second_segment_sample_index']
    if join!=122:raise ValueError('Known acquisition boundary changed')
    return protocol,sources,parts


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.is_symlink() or a.out.resolve().parent.parent!=(ROOT/'work_dirs'/core.VERSION).resolve():
        raise ValueError('Use work_dirs/version/RUN_ID/result')
    protocol,sources,parts=load();a.out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();before={p:sha(ROOT/p) for p in dict(protocol['input_pins'],**protocol['frozen_frontend_identity'])}
    train=core.enrich(parts['TRAIN'],protocol['boundaries'])
    history_gate=core.fit_history_gate(train)
    write(a.out/'history_gate_cutoff.json',history_gate)
    # Frozen constants are verified against the original TRAIN procedure, never reselected.
    level_train=core.level_view(train)
    old_threshold=core.previous.fit_cutoff(level_train)
    trend_threshold=core.trend.fit_cutoff(core.trend_view(train))
    if abs(old_threshold['risk_le']-core.LEVEL_CUTOFF)>1e-15 or abs(trend_threshold['risk_le']-core.TREND_CUTOFF)>1e-15:
        raise ValueError('Frozen temporal cutoff changed')
    write(a.out/'level_train_cutoff.json',old_threshold)
    write(a.out/'trend_train_cutoff.json',trend_threshold)
    threshold=dict(risk_le=1.,fixed_normalization_only=True,not_a_fitted_new_rejection_threshold=True)
    train=core.apply_gate(train,history_gate['q_le'])
    result={};old_result={};trend_result={};comparison={};branches={};enriched={'TRAIN':train}
    # Cutoff is complete and saved before any VAL evidence is computed.
    for role in ('TRAIN','VAL'):
        if role not in enriched:enriched[role]=core.apply_gate(core.enrich(parts[role],protocol['boundaries']),history_gate['q_le'])
        rows=enriched[role]
        result[role]=core.evaluate(rows,threshold['risk_le'])
        old_result[role]=core.previous.evaluate(core.level_view(rows),old_threshold['risk_le'])
        trend_result[role]=core.trend.evaluate(core.trend_view(rows),core.TREND_CUTOFF)
        branches[role]=core.branch_diagnostics(rows,history_gate['q_le'])
        comparison[role]=core.comparison(rows,old_threshold['risk_le'],threshold['risk_le'])
        with gzip.open(a.out/('scored_'+role+'.jsonl.gz'),'xt') as f:
            for r in rows:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
        print(role,'frames',len(rows),'temporal coverage',result[role]['supported_evidence']['all']['support_coverage'],flush=True)
    if before!={p:sha(ROOT/p) for p in before}:raise ValueError('Input changed')
    final_gate=core.gate(result['VAL'],comparison['VAL'])
    status='EVIDENCE_PASS_NO_ADOPTION' if final_gate['passed'] else 'EVIDENCE_FAILED_STOP'
    report=dict(protocol=core.VERSION,status=status,cutoff=threshold,level_cutoff=old_threshold,roles=result,
        level_roles=old_result,trend_roles=trend_result,trend_vs_level=comparison,gate=final_gate,
        history_gate_cutoff=history_gate,trend_cutoff=trend_threshold,branch_diagnostics=branches,
        candidate_kind='history_gated_level_or_trend',normalized_score_is_not_error_probability=True,
        input_pins=before,sources=sources,elapsed_seconds=time.monotonic()-started,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        training_updates=0,detector_inferences=0,TEST_opened=False,policy_changed=False,
        formal_center_size_angle_flags_unchanged=True,candidate_adopted=False,
        temporal_veto_is_offline_diagnostic=True,
        time_units='sampled observation index; no source-FPS or elapsed-second claim')
    write(a.out/'report.json',report)
    write(a.out/'completion.json',dict(status=status,frames=3445,TEST_opened=False,training_updates=0))
    print(status,flush=True)


if __name__=='__main__':main()
