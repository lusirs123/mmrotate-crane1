#!/usr/bin/env python3
"""CPU paired TRAIN fit, save models, then extract/calibrate full exposed VAL."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.utils import port_reliability_pixel_pauc_v1 as core
from crane_project.tools import run_port_reliability_conditional_evidence_v1 as io


def load(allow_absent=False):
    p=json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_protocol.json')).read_text())
    s=json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_sources.json')).read_text())
    if p['settings']!=core.SETTINGS or p['pixel_settings']!=core.pixels.SETTINGS or p['fallback_cutoff']!=core.FALLBACK_CUTOFF or p['protocol']!=core.VERSION:
        raise ValueError('Prespecified contract changed')
    pins=dict(p['input_pins'],**p['frontend_pins'],**s['sources']);absent=[]
    for name,digest in pins.items():
        if allow_absent and name in p['frontend_pins'] and not (ROOT/name).exists():
            absent.append(name);continue
        if io.sha(ROOT/name)!=digest:raise ValueError('Source/input differs: '+name)
    previous={}
    for name in ('pauc_result','reference_result','pixel_result'):
        base=ROOT/p[name];report=json.loads((base/'report.json').read_text())
        review=json.loads((base/'server_review.json').read_text())
        if not review['passed'] or review['report_sha256']!=io.sha(base/'report.json') or report['TEST_opened']:
            raise ValueError('Reviewed parent identity differs')
        if name=='pixel_result':
            if not report['result']['feasibility_passed'] or report['fits'] or report['decisions_changed'] or report['VAL_opened']:
                raise ValueError('Pixel prerequisite identity differs')
        elif report['candidate_adopted'] or report['formal_policy_changed']:
            raise ValueError('Formal parent identity differs')
        previous[name]=report
    return p,pins,absent,previous['pauc_result']


def prepare(p,previous):
    train=io.read(ROOT/p['reference_result']/'scored_TRAIN.jsonl.gz')
    pix=json.loads((ROOT/p['pixel_result']/'rows.json').read_text())
    joined=core.attach(train,pix,'TRAIN')
    prepared=core.prepare(joined,previous['base_models'])
    return joined,prepared


def extract_val(rows,p,data_root):
    if len(rows)!=887 or len({r['image'] for r in rows})!=887 or any(r['split']!='val' or r['reliability_role']!='val' for r in rows):
        raise ValueError('Complete unique VAL only')
    manifest=json.loads((ROOT/p['manifest']).read_text())
    specs={r['id']:r for r in manifest['records'] if r['split']=='val'}
    if set(specs)!={r['image'] for r in rows}:raise ValueError('VAL manifest membership differs')
    output=[];data_root=data_root.resolve()
    for i,r in enumerate(rows):
        spec=specs[r['image']];path=(data_root/spec['images']['path']).resolve()
        if data_root not in path.parents or io.sha(path)!=spec['images']['sha256']:
            raise ValueError('Native VAL image identity differs')
        with Image.open(path) as image:
            if list(image.size)!=r['image_size']:raise ValueError('Native VAL image dimensions differ')
            gray=np.asarray(image.convert('L'),dtype=float)
        output.append(dict(image=r['image'],pred=r['pred'],domain=r['domain'],sequence=r['sequence'],
            evidence=core.pixels.evidence(gray,r['pred']),image_sha256=spec['images']['sha256'],
            wrong=core.level.metrics.bad(r),accepted=r['fallback_decision'],
            low_temporal=r['temporal']['available'] and r['temporal']['primary']<=core.parent.base.LEVEL_CUTOFF))
        if i%128==0:print('Native VAL pixels',i+1,len(rows),flush=True)
    return output


def run(args):
    p,pins,absent,previous=load(args.allow_absent_frontend)
    train,prepared=prepare(p,previous)
    if args.check_only:
        print('Local contract/TRAIN cache verified; no fits or VAL parsing;',prepared['active_rows'],'active,',len(prepared['wrong_images']),'wrong;',len(absent),'absent frontend weights explicitly allowed')
        return
    if absent:raise ValueError('Actual run requires frontend SHA verification')
    out=args.out
    if out is None or out.is_symlink() or out.name!='result' or out.resolve().parent.parent!=(ROOT/'work_dirs'/core.VERSION).resolve():
        raise ValueError('Use work_dirs/version/RUN_ID/result')
    out.mkdir(parents=True,exist_ok=False)
    io.write(out/'TRAIN_prepare.json',prepared)
    models={core.CONTROL:core.fit(train,prepared,4),core.METHOD:core.fit(train,prepared,6)}
    io.write(out/'models.json',models)  # Before reading or extracting VAL pixels.
    val=io.read(ROOT/p['reference_result']/'scored_VAL.jsonl.gz')
    if {r['image'] for r in train}&{r['image'] for r in val}:raise ValueError('TRAIN/VAL overlap')
    val_pixels=extract_val(val,p,args.data_root)
    io.write(out/'pixels_VAL.json',val_pixels)
    val=core.attach(val,val_pixels,'VAL')
    scored={'TRAIN':core.score(train,models),'VAL':core.score(val,models)}
    cutoffs={name:core.calibrate(scored['VAL'],name) for name in models}
    io.write(out/'calibration.json',cutoffs)
    roles={role:core.evaluate(rows,cutoffs) for role,rows in scored.items()}
    for role,rows in scored.items():
        with gzip.open(out/('scored_'+role+'.jsonl.gz'),'xt') as f:
            for r in rows:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    gate=core.gate(roles['VAL'])
    for name,digest in pins.items():
        if io.sha(ROOT/name)!=digest:raise ValueError('Frozen source/input changed')
    report=dict(protocol=core.VERSION,settings=core.SETTINGS,pixel_settings=core.pixels.SETTINGS,input_pins=pins,
        base_models=previous['base_models'],original_partial_model=previous['model'],models=models,prepared=prepared,
        calibration=cutoffs,roles=roles,gate=gate,status='EVIDENCE_PASS_NO_ADOPTION' if gate['passed'] else 'EVIDENCE_FAILED_STOP',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        model_saved_before_VAL=True,fit_calls=2,TRAIN_previously_exposed=True,VAL_previously_exposed=True,
        TEST_opened=False,formal_policy_changed=False,candidate_adopted=False,
        detector_inferences=0,detector_updates=0,neural_training_updates=0,
        limitations=['Peak image gradients may be background/texture, not true target edges.',
            'Uniform original smooth pAUC surrogate, no hard-error weighting or coefficient sign search.',
            'Matched thresholds with GT are offline only; global same-CR matching may violate a video95 condition.',
            'Paired control fits the same pixel-supported subset, not claimed to reproduce the original 2528-row fit.',
            'Two different predictors/thresholds in active/fallback branches are not one calibrated probability.',
            'No new data, independent TEST, safety guarantee or verified deployment latency.'])
    io.write(out/'report.json',report)
    io.write(out/'completion.json',dict(status=report['status'],rows=3445,TEST_opened=False))
    print(report['status'],'failed checks',len(gate['failures']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path)
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--allow-absent-frontend',action='store_true')
    parser.add_argument('--data-root',type=Path,default=ROOT/'crane_project/data/crane_grab_port_day2night_v1')
    run(parser.parse_args())
