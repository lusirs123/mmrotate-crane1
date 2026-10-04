#!/usr/bin/env python3
"""Frozen full-TRAIN midpoint head: optional fresh VAL replay or fixed TEST.

Never fits/replays a short-fit head or selects on TEST. Training sources intact.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import train_port_geometry_midpoint_formal_v1 as f

old,g=f.old,f.g
VERSION='port_geometry_midpoint_formal_v1_frozen_evaluation'
PROTOCOL=ROOT/'crane_project/tools/port_geometry_midpoint_formal_v1_eval_protocol.json'
SOURCES=ROOT/'crane_project/tools/port_geometry_midpoint_formal_v1_eval_sources.json'


def protocol_document():
    return dict(protocol=VERSION,training_protocol=f.VERSION,
        counts=dict(val=g.ready.VAL_COUNTS,test=old.TEST_COUNTS),
        test_annotation_sha256=old.TEST_ANN_SHA,test_image_identity_sha256=old.TEST_IMAGE_SHA,
        selection='Verify complete24-epoch formal training and recompute original full-VAL selection from saved metrics. Load only that formal head. No short-fit replay, new updates, averaging or TEST selection.',
        static='Source/protocol, completion/selection artifacts and selected weight SHA only. No torch.load, dataset/image/annotation reads, cache load or CUDA.',
        inference='One frozen B and one frozen formal midpoint head. One extraction and three native B head calls per frame for raw/original/raw-after auditing. No GT/domain/sequence online input. Original transform, score/count and fallback preserved.',
        val='Optional fresh887-frame VAL inference; compare all delivered B/midpoint boxes, raw candidates, decisions and metrics with saved selected-epoch rows at original numerical tolerances. No reselection.',
        test='All1440 fixed exposed TEST frames. Paired same-run B/midpoint. No new training, threshold, seed, checkpoint or method choice.',
        reporting='Two overall console tables only; JSON retains real/sim/sequence/overall, candidates, pure and penalty angles, three center denominators, RIoU and temporal/failure intervals.',
        scope=dict(head_updates=0,detector_updates=0,selection_on_test=False,
            automatic_promotion=False,test_repeatedly_exposed=True))


def checked_sources():
    parent=f.checked_sources(); fixed=json.loads(SOURCES.read_text())
    required=set(parent['sources'])|{str(f.SOURCES.relative_to(ROOT)),str(PROTOCOL.relative_to(ROOT)),
        'crane_project/tools/eval_port_geometry_midpoint_formal_v1.py',
        'tests/test_port_geometry_midpoint_formal_v1_eval.py'}
    if (fixed.get('protocol')!=VERSION or set(fixed['sources'])!=required or
            json.loads(PROTOCOL.read_text())!=protocol_document()):
        raise ValueError('Formal evaluation source/protocol contract differs')
    actual={p:g.ready.sha(ROOT/p) for p in required}
    if actual!=fixed['sources']:
        raise ValueError('Formal evaluation source SHA differs')
    return dict(training_identity=parent,sources=actual,protocol_sha256=g.ready.sha(PROTOCOL),
        sources_sha256=g.ready.sha(SOURCES))


def checked_selection(selection_path, identity):
    path=Path(selection_path); selection=json.loads(path.read_text())
    completion_path=path.parent/'completion.json'
    completion=json.loads(completion_path.read_text())
    artifacts=json.loads((path.parent/'artifacts.json').read_text())['files']
    expected={('head_epoch_%02d.pth'%i) for i in range(1,25)}
    if (selection['protocol']!=f.VERSION or selection['identity']!=identity['training_identity'] or
            selection['split']!='val' or selection['test_access'] is not False or
            selection['selection_on_test'] is not False or selection['selection_config']!=f.SELECTION_CONFIG or
            set(selection['all_checkpoints'])!=expected or
            completion['status']!='FORMAL_MIDPOINT_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED' or
            completion['epochs_completed']!=24 or completion['detector_updates']!=0 or
            completion['test_access'] is not False or completion['selection']!=selection):
        raise ValueError('Requires completed formal training and unchanged full-VAL selection')
    for name in (path.name,'completion.json','selected_val_compare.json'):
        if artifacts.get(name)!=g.ready.sha(path.parent/name):
            raise ValueError('Formal training artifact SHA differs: '+name)
    chosen,_,info=f.select_best_checkpoint(selection['all_checkpoints'],f.SELECTION_CONFIG)
    selected=selection['selected_checkpoint']; name=selected['path']
    if (chosen!=name or info!=selection['selection_info'] or Path(name).name!=name or
            selection['all_checkpoints'][name]['checkpoint']!=selected or
            selected['epoch']!=int(name[len('head_epoch_'):-4]) or
            g.ready.sha(path.parent/name)!=selected['sha256'] or artifacts.get(name)!=selected['sha256']):
        raise ValueError('Formal head does not match original full-VAL selection')
    return selection,dict(selection_sha256=g.ready.sha(path),completion_sha256=g.ready.sha(completion_path),
        selected_checkpoint=selected,cache_manifest_sha256=selection['cache_manifest_sha256'])


def val_inputs():
    manifest=json.loads((g.ready.DATA/'manifest.json').read_text())
    selected=[r for r in manifest['records'] if r['split']=='val']
    expected={r['id']:r for r in selected}
    anns=g.ready.files(g.ready.DATA/'val/annfiles','.txt')
    images=g.ready.files(g.ready.DATA/'val/images','.jpg')
    if (len(expected)!=887 or len(selected)!=887 or len(anns)!=887 or len(images)!=887 or
            {p.stem for p in anns}!=set(expected) or {p.stem for p in images}!=set(expected) or
            Counter(r['sequence'] for r in selected)!=Counter(g.ready.VAL_COUNTS) or
            g.ready.set_sha(anns)!=g.ready.FROZEN_B['annotation_sha256']):
        raise ValueError('Fixed VAL identities/counts/annotations differ')
    rows=[]; digest=hashlib.sha256()
    for ann in anns:
        name=ann.stem; source=expected[name]; image=g.ready.DATA/'val/images'/(name+'.jpg')
        sha=g.ready.sha(image)
        if (source['images']['sha256']!=sha or source['annfiles']['sha256']!=g.ready.sha(ann) or
                source['images']['path']!='val/images/'+image.name or
                source['annfiles']['path']!='val/annfiles/'+ann.name):
            raise ValueError('VAL frame bytes differ: '+name)
        g.ready.polygon_box(ann); gt=old.parse_dota_txt(str(ann))
        if len(gt)!=1:
            raise ValueError('One grab GT required')
        with Image.open(image) as im:
            size=list(im.size)
        seq,number=name.rsplit('_',1)
        if source['sequence']!=seq:
            raise ValueError('VAL sequence differs')
        rows.append(dict(image=name,sequence=seq,frame_id=int(number),domain=seq.split('_')[0],
            gt=gt[0].tolist(),image_size=size))
        digest.update(image.name.encode()); digest.update(b'\0'); digest.update(bytes.fromhex(sha))
    return rows,dict(frames=887,annotation_sha256=g.ready.FROZEN_B['annotation_sha256'],
        image_identity_sha256=digest.hexdigest())


def prediction_row(source, prediction):
    # Match formal cached evaluator's float32 GT conversion, including angles.
    gt=np.asarray(source['gt'],dtype=np.float32).tolist()
    result=dict(image=source['image'],sequence=source['sequence'],frame_id=source['frame_id'],
        domain=source['domain'],scale=1.,gt=gt,**prediction)
    result['metrics']={name:g.decompose(np.asarray(gt),np.asarray(result[name][:5]) if result[name] is not None else None)
        for name in ('b','midpoint','candidate')}
    return result


def verify_val_rows(current, path):
    saved=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    indexed={r['image']:r for r in saved}
    if len(saved)!=len(indexed) or len(current)!=len(saved) or {r['image'] for r in current}!=set(indexed):
        raise ValueError('Fresh VAL identity set differs from selected epoch')
    for r in current:
        old_row=indexed[r['image']]
        if any(r[k]!=old_row[k] for k in ('sequence','frame_id','domain','gt','accepted','candidate_valid','failed_checks')):
            raise ValueError('Fresh VAL metadata/decision differs: '+r['image'])
        for method in ('b','midpoint','candidate'):
            a,b=r[method],old_row[method]
            if (a is None)!=(b is None) or (a is not None and
                    (a[5]!=b[5] or not np.allclose(a[:5],b[:5],atol=1e-4,rtol=1e-6))):
                raise ValueError('Fresh VAL box differs: '+r['image'])
            now,previous=r['metrics'][method],old_row['metrics'][method]
            for key in ('output','center_hit','angle_penalty_reason'):
                if now.get(key)!=previous.get(key):
                    raise ValueError('Fresh VAL correctness/penalty differs')
            if abs(now['riou']-previous['riou'])>1e-5:
                raise ValueError('Fresh VAL RIoU differs')
    return dict(frames=len(current),selected_epoch_box_decision_replay_pass=True)


def run(args):
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=False); started=time.perf_counter()
    report=dict(protocol=VERSION,split=args.split,status='RUNNING',head_updates=0,detector_updates=0,
        feature_extractions=0,native_head_calls=0,test_access=False,selection_on_test=False,
        test_repeatedly_exposed=True,automatic_promotion=False)
    try:
        identity=checked_sources(); selection,proof=checked_selection(args.selection,identity)
        report.update(identity=identity,selection=proof)
        if args.check_only:
            report['status']='STATIC_FORMAL_MIDPOINT_EVAL_PASS_NO_DATA_GPU_UPDATES'
        else:
            rt=f.runtime(args.gpu); f.seed_all()
            # Selected formal head, no call to old.replay_fit/preflight.fit_arm.
            detector,head,cfg=f.load_selected_pipeline(args.selection,args.gpu)
            payload=torch.load(str(Path(args.selection).parent/selection['selected_checkpoint']['path']),map_location='cpu')
            if rt!=payload['context']['runtime']:
                raise ValueError('Evaluation runtime/GPU differs from formal extraction')
            context=payload['context']; del payload
            before=dict(b=g.state_digest(detector),head=g.state_digest(head))
            if args.split=='test':
                report['test_access']=True
                sources,data=old.fixed_test_inputs(); dataset=old.build_test_dataset(cfg,sources)
            else:
                sources,data=val_inputs(); dataset=f.dataset_for(cfg,'val',1.)
                if any(data[k]!=context['data_identity']['val'][k] for k in data):
                    raise ValueError('VAL data differs from formal cached evaluation')
                expected=json.loads((Path(args.selection).parent/'selected_val_compare.json').read_text())
                names=[Path(info['filename']).stem for info in f.infos(dataset)]
                if names!=[s['image'] for s in sources]:
                    raise ValueError('VAL loader order differs')
            rows=[]
            def audit(event):
                if event['stage']=='test_feature_extracted':
                    report['feature_extractions']+=1
                elif event['stage']=='test_native_head_called':
                    report['native_head_calls']+=1
                else:
                    raise ValueError('Unexpected native inference audit event')
            with (out/(args.split+'_rows.jsonl')).open('x') as stream, (out/'progress.jsonl').open('x') as progress:
                for index,source in enumerate(sources):
                    image,metas=old.load_view(dataset,index,source,args.gpu)
                    prediction=old.capture(detector,head,image,metas,audit=audit)
                    row=prediction_row(source,prediction); rows.append(row)
                    stream.write(json.dumps(g.json_native(row),ensure_ascii=False,allow_nan=False)+'\n')
                    progress.write(json.dumps(dict(stage='frozen_evaluation',image=source['image'],done=index+1,total=len(sources)))+'\n')
                    if index%100==0 or index==len(sources)-1:
                        print('FORMAL frozen',args.split,index+1,'/',len(sources),flush=True)
            after=dict(b=g.state_digest(detector),head=g.state_digest(head))
            if before!=after:
                raise ValueError('Frozen B/head state changed during evaluation')
            if report['feature_extractions']!=len(rows) or report['native_head_calls']!=3*len(rows):
                raise ValueError('Frozen native inference call counts differ')
            value=f.summaries(rows); value['split']=args.split
            if args.split=='val':
                row_path=Path(args.selection).parent/('val_epoch_%02d.rows.jsonl'%selection['selected_checkpoint']['epoch'])
                artifacts=json.loads((Path(args.selection).parent/'artifacts.json').read_text())['files']
                if artifacts.get(row_path.name)!=g.ready.sha(row_path):
                    raise ValueError('Saved selected VAL rows SHA differs')
                report['val_replay']=verify_val_rows(rows,row_path)
                if value!=expected:
                    raise ValueError('Fresh full-VAL summary differs from selected epoch')
            g.write_new(out/(args.split+'_compare.json'),value)
            report.update(status='FROZEN_FORMAL_MIDPOINT_'+args.split.upper()+'_COMPLETE_REVIEW_REQUIRED',
                data_identity=data,runtime=rt,state_before=before,state_after=after,
                frames=len(rows),feature_extractions=len(rows),native_head_calls=3*len(rows),
                gpu_peak_mib=dict(allocated=torch.cuda.max_memory_allocated(args.gpu)/2**20,
                    reserved=torch.cuda.max_memory_reserved(args.gpu)/2**20))
            for method in ('b','midpoint'):
                print('OVERALL',args.split.upper(),'method='+method,flush=True)
                metrics=value['groups']['overall'][method]['metric_protocol_v2']
                logger=old.CraneOfflineEvaluator(mode=args.split)
                logger._log_metrics(metrics,sorted({r['domain'] for r in rows}))
        report['elapsed_seconds']=time.perf_counter()-started
        g.write_new(out/'completion.json',report)
        g.write_new(out/'artifacts.json',dict(files={p.name:g.ready.sha(p) for p in out.iterdir() if p.is_file()}))
        print('Saved',out/'completion.json','status',report['status'],flush=True)
    except Exception as error:
        report.update(status='FAILED',error=type(error).__name__+': '+str(error))
        g.write_new(out/'failure.json',report)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=('val','test'),required=True)
    p.add_argument('--selection',required=True)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--out-dir',required=True)
    return p


if __name__=='__main__':
    if Path.cwd().resolve()!=ROOT:
        raise SystemExit('Run from symEOOD project root')
    run(parser().parse_args())
