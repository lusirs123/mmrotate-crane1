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
from crane_project.utils import port_reliability_temporal_evidence_v1 as core
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_midpoint_sigma15_reliability_v1 as binding

PROTOCOL=ROOT/'crane_project/tools/port_reliability_temporal_evidence_v1_protocol.json'
SOURCES=ROOT/'crane_project/tools/port_reliability_temporal_evidence_v1_sources.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def load():
    protocol=json.loads(PROTOCOL.read_text());sources=json.loads(SOURCES.read_text())
    if protocol['settings']!=core.SETTINGS:raise ValueError('Settings changed')
    for p,h in dict(protocol['input_pins'],**sources['sources']).items():
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
    started=time.monotonic();before={p:sha(ROOT/p) for p in protocol['input_pins']}
    train=core.enrich(parts['TRAIN'],protocol['boundaries'])
    threshold=core.fit_cutoff(train);write(a.out/'train_cutoff.json',threshold)
    result={};enriched={'TRAIN':train}
    # Cutoff is complete and saved before any VAL evidence is computed.
    for role in ('TRAIN','VAL'):
        if role not in enriched:enriched[role]=core.enrich(parts[role],protocol['boundaries'])
        rows=enriched[role]
        result[role]=core.evaluate(rows,threshold['risk_le'])
        with gzip.open(a.out/('scored_'+role+'.jsonl.gz'),'xt') as f:
            for r in rows:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
        print(role,'frames',len(rows),'temporal coverage',result[role]['supported_evidence']['all']['support_coverage'],flush=True)
    if before!={p:sha(ROOT/p) for p in before}:raise ValueError('Input changed')
    status='EVIDENCE_PASS_NO_ADOPTION' if result['VAL']['gate']['passed'] else 'EVIDENCE_FAILED_STOP'
    report=dict(protocol=core.VERSION,status=status,cutoff=threshold,roles=result,
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
