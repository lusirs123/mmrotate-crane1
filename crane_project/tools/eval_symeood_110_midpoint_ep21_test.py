#!/usr/bin/env python3
"""Fixed new110+sigma1.5 ep21: eight saved VAL checks -> one native TEST.

Reuse the historical sigma15 evaluator's capture, row verification, dataset
and metric functions. No full VAL replay, cache generation or optimization.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
VERSION='symeood_110_midpoint_ep21_fixed_test'
CONTRACT=ROOT/'crane_project/data_contracts/symeood_110_midpoint_ep21_test.json'
FORMAL=ROOT/'work_dirs/symeood_110_retrain_seed0_20261026_v1/downstream/midpoint/formal'
DETECTOR_SHA='7f35dacc308943949a8fd2e6aaad312d14651564903ec87c8c8dcec0552ec054'
HEAD_SHA='7aaaacc6c666116e466c9a99ac59d71acf7f77312dc86de224fcf965569fd9ac'
VAL_NAMES=('real_seq07_00001','real_seq07_00167','real_seq07_00169','real_seq07_00181',
    'real_seq14_00001','sim_seq10_00000','sim_seq10_00211','sim_seq10_00410')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def checked_identity():
    from crane_project.tools import train_symeood_110_midpoint_v1 as f
    lock=read(CONTRACT)
    if (lock['protocol']!=VERSION or lock['detector_sha256']!=DETECTOR_SHA or
            lock['head_sha256']!=HEAD_SHA or lock['val_names']!=list(VAL_NAMES) or
            lock['test_frames']!=1440 or lock['full_VAL_replay'] is not False or lock['physical_gpu']!=0):
        raise ValueError('Fixed TEST contract differs')
    for name,h in lock['sources'].items():
        if sha(ROOT/name)!=h:raise ValueError('Evaluation source changed: '+name)
    training_identity=f.checked_sources()
    if training_identity['frozen_b']['checkpoint_sha256']!=DETECTOR_SHA:
        raise ValueError('Wrong detector identity')
    for name,h in lock['training_artifacts'].items():
        if sha(FORMAL/name)!=h:raise ValueError('Fixed training artifact changed: '+name)
    selection=read(FORMAL/'train/selection.json')
    cp=selection['selected_checkpoint']
    if (cp['epoch']!=21 or cp['path']!='head_epoch_21.pth' or cp['sha256']!=HEAD_SHA or
            cp['updates']!=18900 or selection['selection_on_test'] or selection['test_access'] or
            selection['identity']!=training_identity):
        raise ValueError('Requires original VAL-selected ep21; no reselection')
    return f,lock,selection


def representative_rows(rows):
    indexed={r['image']:r for r in rows}
    if len(indexed)!=887:raise ValueError('Saved ep21 VAL rows incomplete')
    selected=[indexed[name] for name in VAL_NAMES]
    if ({r['domain'] for r in selected}!={'real','sim'} or
            {r['image'] for r in selected if r['b'] is None}!={'real_seq07_00167','real_seq07_00169'}):
        raise ValueError('Fixed representatives must cover both domains and saved missing outputs')
    return selected


def run(args):
    f,lock,selection=checked_identity()
    out=Path(args.out_dir).resolve()
    expected=ROOT/lock['output_directory']
    if out!=expected or out.exists():raise ValueError('Use the fixed new TEST output directory')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or args.gpu!=0:
        raise ValueError('Fixed evaluation uses physical GPU0/logical0')
    f.runtime_imports()
    # Reuse the historical sigma-aware evaluation imports without calling its
    # historical B/ep03 identity checks or its full-VAL gate.
    from crane_project.tools import eval_port_geometry_midpoint_sigma15_v1 as legacy
    e,old,g,torch=legacy.e,f.old,f.g,f.torch
    from PIL import Image
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    report=dict(protocol=VERSION,status='RUNNING',detector_sha256=DETECTOR_SHA,head_sha256=HEAD_SHA,
        selected_epoch=21,head_updates=0,detector_updates=0,full_VAL_replay=False,
        cache_rebuilt=False,test_access=False,selection_on_test=False,LVSR=False,depth=False,
        test_repeatedly_exposed=True,automatic_promotion=False,feature_extractions=0,native_head_calls=0,
        physical_gpu=0)
    def status(stage,**details):
        temporary=out/'status.tmp'
        temporary.write_text(json.dumps(dict(stage=stage,time=time.time(),pid=os.getpid(),**details),indent=2))
        temporary.replace(out/'status.json')
    try:
        g.write_new(out/'frozen_contract.json',lock)
        rt=f.runtime(args.gpu);f.seed_all()
        detector,head,cfg=f.load_selected_pipeline(FORMAL/'train/selection.json',args.gpu)
        cache=read(FORMAL/'roi_cache/cache_manifest.json')
        if rt!=cache['runtime']:raise ValueError('Runtime differs from ep21 extraction/training')
        if sha(Path(read(f.BASE_CONTRACT)['checkpoint']))!=DETECTOR_SHA or sha(FORMAL/'train/head_epoch_21.pth')!=HEAD_SHA:
            raise ValueError('Loaded model file SHA differs')
        before=dict(detector=g.state_digest(detector),head=g.state_digest(head))
        status('sample_VAL_gate')
        saved=legacy.read_rows(FORMAL/'train/val_epoch_21.rows.jsonl')
        samples=representative_rows(saved)
        manifest=read(g.ready.DATA/'manifest.json')
        meta={r['id']:r for r in manifest['records'] if r['split']=='val'}
        dataset=f.dataset_for(cfg,'val',1.)
        lookup={Path(v['filename']).stem:i for i,v in enumerate(f.infos(dataset))}
        fresh=[]
        with (out/'sample_VAL_reference.rows.jsonl').open('x') as stream:
            for row in samples:stream.write(json.dumps(row,allow_nan=False)+'\n')
        with torch.no_grad():
            for reference in samples:
                name=reference['image'];source=meta[name]
                image_path=g.ready.DATA/source['images']['path']
                ann=g.ready.DATA/source['annfiles']['path']
                if sha(image_path)!=source['images']['sha256'] or sha(ann)!=source['annfiles']['sha256']:
                    raise ValueError('Representative VAL file differs')
                with Image.open(image_path) as image: size=list(image.size)
                frame=dict(reference,image_size=size)
                image,metas=old.load_view(dataset,lookup[name],frame,args.gpu)
                fresh.append(e.prediction_row(frame,old.capture(detector,head,image,metas)))
                f.resource_check(args.gpu)
        replay=e.verify_val_rows(fresh,out/'sample_VAL_reference.rows.jsonl')
        if dict(detector=g.state_digest(detector),head=g.state_digest(head))!=before:
            raise ValueError('Representative VAL changed model state')
        g.write_new(out/'sample_VAL_gate.json',dict(passed=True,names=list(VAL_NAMES),**replay,
            full_887_VAL_replayed=False,feature_extractions=len(fresh),updates=0))
        with (out/'sample_VAL_online.rows.jsonl').open('x') as stream:
            for row in fresh:stream.write(json.dumps(g.json_native(row),allow_nan=False)+'\n')
        del dataset,saved,samples,fresh
        # No TEST image/annotation/dataset access occurs before the gate passes.
        status('TEST_identity')
        report['test_access']=True
        sources,data=old.fixed_test_inputs();dataset=old.build_test_dataset(cfg,sources)
        def audit(event):
            if event['stage']=='test_feature_extracted':report['feature_extractions']+=1
            elif event['stage']=='test_native_head_called':report['native_head_calls']+=1
            else:raise ValueError('Unknown native capture event')
        rows=[]
        with torch.no_grad(),(out/'test_rows.jsonl').open('x') as stream:
            for index,source in enumerate(sources):
                image,metas=old.load_view(dataset,index,source,args.gpu)
                row=e.prediction_row(source,old.capture(detector,head,image,metas,audit=audit))
                rows.append(row)
                stream.write(json.dumps(g.json_native(row),ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                f.resource_check(args.gpu)
                if index%50==0 or index==len(sources)-1:
                    status('TEST',done=index+1,total=1440)
                    print('FIXED ep21 TEST',index+1,'/1440',flush=True)
        legacy.checked_rows(rows,'test')
        after=dict(detector=g.state_digest(detector),head=g.state_digest(head))
        if after!=before or report['feature_extractions']!=1440 or report['native_head_calls']!=4320:
            raise ValueError('Frozen identity or one-pass call count differs')
        value=dict(split='test',groups=old.test_summaries(rows))
        g.write_new(out/'test_compare.json',value)
        report.update(status='FIXED_EP21_TEST_COMPLETE',frames=1440,domain_counts=dict(Counter(r['domain'] for r in rows)),
            runtime=rt,data_identity=data,state_before=before,state_after=after,
            sample_VAL_frames=8,elapsed_seconds=time.perf_counter()-started,resource=f.resource_check(args.gpu))
        g.write_new(out/'completion.json',report)
        status('complete',frames=1440,updates=0)
        g.write_new(out/'artifacts.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
        print('FIXED_EP21_TEST_COMPLETE',flush=True)
    except BaseException as error:
        report.update(status='FAILED',error=repr(error),elapsed_seconds=time.perf_counter()-started)
        g.write_new(out/'failure.json',report)
        status('stopped_on_error',error=repr(error),retry=False)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--out-dir',required=True)
    return p


if __name__=='__main__':run(parser().parse_args())
