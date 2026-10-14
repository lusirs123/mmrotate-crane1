#!/usr/bin/env python3
"""Native image extraction and offline labels in separate call sites, TRAIN only."""
import argparse,gzip,hashlib,json,sys,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.utils import port_reliability_pixel_border_v1 as core

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--data-root',type=Path,default=ROOT/'crane_project/data/crane_grab_port_day2night_v1');args=ap.parse_args()
    protocol=json.loads((ROOT/'crane_project/tools/port_reliability_pixel_border_v1_protocol.json').read_text())
    if protocol['version']!=core.VERSION or protocol['scope']!='TRAIN_ONLY_NO_FIT':raise ValueError('Scope differs')
    if protocol['settings']!=core.SETTINGS:raise ValueError('Settings differ')
    for path,digest in protocol['pins'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Source identity differs '+path)
    parent=ROOT/protocol['rows'];parent=parent.parent
    reviewed=json.loads((parent/'server_review.json').read_text())
    if not reviewed['passed'] or reviewed['report_sha256']!=sha(parent/'report.json'):raise ValueError('Parent review differs')
    out=args.out.resolve()
    if out.name!='result' or out.parent.parent!=ROOT/'work_dirs'/core.VERSION:raise ValueError('Run folder outside version')
    out.mkdir(parents=True,exist_ok=False)
    src=ROOT/protocol['rows'];rows=[json.loads(s) for s in gzip.open(src,'rt')]
    manifest=json.loads((ROOT/protocol['manifest']).read_text())
    records={r['id']:r for r in manifest['records'] if r['split'] in ('train','train_sim')}
    if len(rows)!=2558 or len({r['image'] for r in rows})!=2558 or set(records)!={r['image'] for r in rows}:raise ValueError('Complete unique TRAIN required')
    output=[]
    for i,r in enumerate(rows):
        if r['reliability_role']!='train' or r['split'] not in ('train','train_sim'):raise ValueError('Not TRAIN')
        spec=records[r['image']];path=(args.data_root/spec['images']['path']).resolve()
        if args.data_root.resolve() not in path.parents or sha(path)!=spec['images']['sha256']:raise ValueError('Image source differs')
        with Image.open(path) as im:
            if list(im.size)!=r['image_size']:raise ValueError('Native image dimensions differ')
            gray=np.asarray(im.convert('L'),dtype=float)
        risk=r['risks']['crossfit_fusion'] if r['temporal']['available'] else r['risks']['partial_auc4']
        cutoff=.5317310050227115 if r['temporal']['available'] else .4094748402105834
        if r['fallback_decision']!=(r['pred'] is not None and risk<=cutoff):raise ValueError('Frozen flag differs')
        ev=core.evidence(gray,r['pred'])
        bad=None;kind='missing';scale=None
        if r['pred'] is not None:
            p=sorted(r['pred'][2:4],reverse=True);g=sorted(r['gt'][2:4],reverse=True)
            errors=[abs(a-b)/b>.1 for a,b in zip(p,g)];bad=any(errors)
            kind='both' if all(errors) else 'L_only' if errors[0] else 'S_only' if errors[1] else 'correct'
            scale=float(np.sqrt(p[0]*p[1]))
        output.append(dict(image=r['image'],domain=r['domain'],sequence=r['sequence'],
            pred=r['pred'],image_sha256=spec['images']['sha256'],evidence=ev,wrong=bad,error_type=kind,scale=scale,
            low_temporal=r['temporal']['available'] and r['temporal']['primary']<=core.SETTINGS['low_temporal'],accepted=r['fallback_decision']))
        if i%256==0:print('TRAIN',i+1,len(rows),flush=True)
    write(out/'rows.json',output)
    write(out/'report.json',dict(version=core.VERSION,settings=core.SETTINGS,rows=len(rows),result=core.analyze(output),
         VAL_opened=False,TEST_opened=False,fits=0,decisions_changed=0,detector_inferences=0,
         source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_pins=protocol['pins']))
    for path,digest in protocol['pins'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Source changed during run')
    print('TRAIN_SCREEN_ONLY_NO_FIT',flush=True)
if __name__=='__main__':main()
