"""One user-authorized TEST of the all-epoch-ranked visual ROI RMS representative.

Original L5 training selection stays null; candidate ranking is frozen before TEST. No epoch scan, training, threshold
change or automatic adoption. Reuse sealed L5 predictions as the comparison.
"""
import argparse
from collections import Counter
from datetime import datetime,timedelta,timezone
import csv
import gzip
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import train_lvsr_roi_rms_v1 as training
from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p

VERSION='roi_rms_ranked_fixed_test_20261112_v1'
L5_ROWS_SHA='9a11d0b625f0a05e45d7f7bacd7e41ff60def9bcd1f31561d30bd412b64ad240'
RUN=ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/roi_rms_001'
EXTRA=('crane_project/tools/eval_lvsr_roi_rms_fixed_test_v1.py','tests/test_lvsr_roi_rms_fixed_test_v1.py')


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(v) for v in Path(path).read_text().splitlines() if v.strip()]
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def validate_training(summary,complete):
    s=summary['selection']; expected=s['representative_checkpoint']; epoch=s['representative_epoch']
    if (not complete['passed'] or complete['epochs_completed']!=24 or complete['formal_optimizer_updates']!=15336
            or not 1<=epoch<=24 or expected['epoch']!=epoch or expected['path']!='head_epoch_%02d.pth'%epoch
            or len(expected['sha256'])!=64 or any(v not in '0123456789abcdef' for v in expected['sha256'])
            or complete['checkpoint']!=expected or training.choose(summary['epochs'])!=s
            or len(summary['epochs'])!=24 or [v['epoch'] for v in summary['epochs']]!=list(range(1,25))
            or summary['representative']['checkpoint']!=expected
            or summary['representative']['epoch']!=epoch or summary['TEST'] or complete['TEST']):
        raise ValueError('Requires completed training and unchanged all-epoch-ranked representative')
    return expected


def identity(require_weights=False):
    parent=p.checked_sources();s=read(RUN/'core_summary.json');c=read(RUN/'core_completion.json')
    checkpoint=validate_training(s,c)
    if s['source_SHA']!=training.source_hashes() or c['source_SHA']!=s['source_SHA']:raise ValueError('Training source drift')
    if require_weights:
        for path,expected in ((RUN/checkpoint['path'],checkpoint['sha256']),(ROOT/p.parent.R4_CHECKPOINT,p.parent.R4_SHA),
                (p.midpoint_directory()/'train/head_epoch_20.pth',p.MIDPOINT_SHA)):
            if p.sha(path)!=expected:raise ValueError('Frozen checkpoint SHA differs: '+str(path))
    return dict(protocol=VERSION,epoch=checkpoint['epoch'],checkpoint_sha256=checkpoint['sha256'],parent_identity=parent,training_sources=s['source_SHA'],
        evaluation_sources={n:p.sha(ROOT/n) for n in EXTRA},
        training_receipts={n:p.sha(RUN/n) for n in ('core_summary.json','core_completion.json','selection.json')},
        original_training_selected_epoch=None,user_fixed_representative=True,ranking='unchanged predeclared VAL ranking',
        TEST_frames=1440,physical_gpu=2,TEST_exposed=True,training_updates=0,checkpoint_scan=False,selection_on_test=False,
        automatic_adoption=False,download='core analysis only; no weights or prediction rows')


def metric(rows,key):
    import numpy as np
    from crane_project.tools.eval_crane_offline import compute_riou
    from crane_project.tools.eval_port_geometry_ap_v1 import evaluate_images
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import canonical,row_metrics
    values=[];ds=[];gs=[]
    for r in rows:
        pred=r[key];iou=compute_riou(np.asarray(canonical(pred)),np.asarray(canonical(r['gt']))) if pred else 0.
        values.append(row_metrics(r['gt'],pred,iou));ds.append([canonical(pred)+[pred[5]]] if pred else []);gs.append([canonical(r['gt'])])
    present=[v for v in values if v['output']]
    result=dict(frames=len(rows),outputs=len(present),long_MAE_pct=100*sum(v['long_relative_error'] for v in present)/len(present),
        short_MAE_pct=100*sum(v['short_relative_error'] for v in present)/len(present),joint10=sum(v['joint_size10'] for v in values),
        RIoU=sum(v['riou'] for v in values)/len(rows),TP50=sum(v['riou']>=.5 for v in values),TP75=sum(v['riou']>=.75 for v in values),
        center_MAE_px=sum(v['center_error_px'] for v in present)/len(present),
        center_RMSE_px=math.sqrt(sum(v['center_error_px']**2 for v in present)/len(present)))
    sim=[v for r,v in zip(rows,values) if r['domain']=='sim' and v['output']]
    result['Sim_angle_RMSE_deg']=math.sqrt(sum(v['angle_error_deg']**2 for v in sim)/len(sim)) if sim else None
    run=maximum=0;last=None
    for r,v in sorted(zip(rows,values),key=lambda x:(x[0]['sequence'],x[0]['frame_id'])):
        if last is None or last['sequence']!=r['sequence'] or last['frame_id']+1!=r['frame_id']:run=0
        run=run+1 if v['riou']<.5 else 0;maximum=max(maximum,run);last=r
    result['MCML_max']=maximum
    ap={}
    for t in (.5,.75):
        a=evaluate_images(ds,gs,t);result['AP'+str(round(t*100))]=a['ap_percent'];ap[str(t)]=a
        if a['tp']!=result['TP'+str(round(t*100))]:raise ValueError('AP/RIoU hit mismatch')
    return result,values,ds,gs,ap


def analyze(current,l5):
    from crane_project.tools.eval_port_geometry_ap_v1 import verify_native
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity
    if len(current)!=len(l5) or len(current)!=1440:raise ValueError('Full1440 paired rows required')
    merged=[]
    for a,b in zip(current,l5):
        if (a['image'],a['gt'])!=(b['image'],b['gt']) or not frozen_identity(a):raise ValueError('Frame/GT/frozen identity differs')
        merged.append(dict(a,L5=b['edge_residual']))
    groups={'overall':merged};groups.update({d:[r for r in merged if r['domain']==d] for d in ('real','sim')})
    groups.update({q:[r for r in merged if r['sequence']==q] for q in sorted({r['sequence'] for r in merged})})
    metrics={};paired={};accurate=[];table=[];native={}
    specs={'R4':'b','M20':'midpoint','L5':'L5','ROIRMS':'edge_residual'}
    for group,rr in groups.items():
        metrics[group]={};details={}
        for name,key in specs.items():
            m,v,ds,gs,ap=metric(rr,key);metrics[group][name]=m;details[name]=v;table.append(dict(group=group,method=name,**m))
            if group=='overall':
                for t in (.5,.75):native[name+'/'+str(t)]=verify_native(ds,gs,t,ap[str(t)])
        after=details['ROIRMS'];paired[group]={}
        for name in ('R4','M20','L5'):
            before=details[name]
            paired[group][name]=dict(fixed_joint=sum(not a['joint_size10'] and b['joint_size10'] for a,b in zip(before,after)),
                broken_joint=sum(a['joint_size10'] and not b['joint_size10'] for a,b in zip(before,after)),
                recovered75=sum(a['riou']<.75<=b['riou'] for a,b in zip(before,after)),lost75=sum(b['riou']<.75<=a['riou'] for a,b in zip(before,after)))
        for edge in ('long','short'):
            key=edge+'_relative_error';indices=[i for i,v in enumerate(details['M20']) if v['output'] and v[key]<=.03]
            entry=dict(group=group,edge=edge,fixed_M_count=len(indices))
            for name in ('M20','L5','ROIRMS'):
                entry[name+'_MAE_pct']=100*sum(details[name][i][key] for i in indices)/len(indices) if indices else None
                entry[name+'_harmed_vs_M']=sum(details[name][i][key]>details['M20'][i][key]+1e-6 for i in indices)
            accurate.append(entry)
    return dict(metrics=metrics,paired=paired,native_AP=native,frames=1440,TEST_exposed=True,selection_on_test=False),table,accurate


def infer():
    lock=read(RUN/'test_fixed_lock.json');now=identity(True)
    epoch=now['epoch']; checkpoint=read(RUN/'selection.json')['representative_checkpoint']
    if lock['identity']!=now or not lock['user_authorized']:raise ValueError('TEST lock drift')
    if any((RUN/n).exists() for n in ('test_fixed_rows.jsonl','test_fixed_completion.json','test_fixed_failure.json')):
        raise ValueError('Fresh fixed TEST only; no automatic retry')
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']='2'
    from crane_project.tools import train_symeood_r4_midpoint_v1 as adapted
    from crane_project.tools import eval_symeood_r4_midpoint_fixed_test_v1 as fixed
    from crane_project.tools import eval_symeood_r4_lvsr_ep5_fixed_test_v1 as previous
    from crane_project.utils.port_size_pair_roi_rms_v1 import PairROIRMSSizeHead
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity
    from crane_project.tools.eval_port_geometry_ap_v1 import validate_saved_gt
    adapted.resource_check(2,before=True);engine=adapted.make_engine(2);torch,old,g=engine.torch,engine.old,engine.g
    from PIL import Image
    import numpy as np
    started=time.monotonic();original_optimizer=torch.optim.Optimizer.__init__;handle=None;resources=[]
    def forbidden(*a,**kw):raise RuntimeError('Frozen TEST forbids optimizer')
    torch.optim.Optimizer.__init__=forbidden
    def status(stage,**kw):print(json.dumps(dict(stage=stage,time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),**kw)),flush=True)
    def resource():
        resources.append(engine.resource_check(0))
        import subprocess
        text=subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        if any(int(v.strip())!=os.getpid() for v in text.splitlines() if v.strip()):raise RuntimeError('GPU2 contention')
    try:
        runtime=engine.runtime(0);engine.seed_all();torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        detector,mid,cfg=engine.load_selected_pipeline(fixed.training_directory()/'train/selection.json',0)
        if runtime!=read(fixed.training_directory()/'roi_cache/cache_manifest.json')['runtime']:raise ValueError('Training runtime differs')
        payload=torch.load(str(RUN/checkpoint['path']),map_location='cpu')
        if (payload['epoch']!=epoch or payload['head_type']!='PairROIRMSSizeHead' or payload['parameters']!=2480
                or payload['updates']!=epoch*639 or payload['parent_checkpoint'] is not None):raise ValueError('Wrong candidate payload')
        head=PairROIRMSSizeHead().cuda();head.load_state_dict(payload['head_state'],strict=True)
        for model in (detector,mid,head):model.eval().requires_grad_(False)
        def states():return {n:g.state_digest(m) for n,m in (('R4',detector),('M20',mid),('LVSR_ROIRMS',head))}
        before=states();captured={};counts=dict(features=0,native_heads=0,size_heads=0)
        def hook(module,args,output):captured.update(inputs=args,midpoint=output['boxes_original'])
        handle=mid.register_forward_hook(hook)
        def audit(event):
            if event['stage']=='test_feature_extracted':counts['features']+=1
            elif event['stage']=='test_native_head_called':counts['native_heads']+=1
            else:raise ValueError('Unknown native audit event')
        def predict(image,metas):
            captured.clear();result=old.capture(detector,mid,image,metas,audit=audit)
            m=captured['midpoint'];o=head(*captured['inputs'],m);counts['size_heads']+=1
            result.update(edge_residual=o['boxes_original'][0].cpu().tolist() if len(m) else None,
                size_delivery='no_output' if not len(m) else 'neutral' if bool(o['neutral_bypass'][0]) else 'square' if bool(o['square_bypass'][0]) else 'accepted' if bool(o['accepted'][0]) else 'fallback',
                size_failed_checks=[k for k,v in o['checks'].items() if len(m) and not bool(v[0])],
                vector_valid=bool(o['vector_valid'][0]) if len(m) else None)
            if not frozen_identity(result):raise ValueError('M inheritance failed')
            return result
        status('sample_VAL_replay')
        with gzip.open(RUN/('epoch_%02d_VAL.rows.jsonl.gz'%epoch),'rt') as f:reference={r['image']:r for r in map(json.loads,f)}
        manifest=read(g.ready.DATA/'manifest.json');metadata={r['id']:r for r in manifest['records'] if r['split']=='val'}
        dataset=engine.dataset_for(cfg,'val',1.);lookup={Path(r['filename']).stem:i for i,r in enumerate(engine.infos(dataset))};val_proof=[]
        with torch.no_grad():
            for name in fixed.VAL_NAMES:
                ref=reference[name];meta=metadata[name]
                for k in ('images','annfiles'):
                    if p.sha(g.ready.DATA/meta[k]['path'])!=meta[k]['sha256']:raise ValueError('VAL bytes differ')
                with Image.open(g.ready.DATA/meta['images']['path']) as im:source=dict(ref,image_size=list(im.size))
                image,metas=old.load_view(dataset,lookup[name],source,0);a=predict(image,metas)
                maxima=previous.compare_boxes(dict(a,image=name),ref,('b','midpoint','edge_residual'))
                previous.compare_decisions(dict(a,failed_checks=a['size_failed_checks']),ref)
                val_proof.append(dict(image=name,max_abs=maxima));resource()
        save(RUN/'test_fixed_VAL_replay.json',dict(passed=True,frames=8,rows=val_proof))
        if states()!=before:raise ValueError('VAL replay changed model state')
        status('fixed_TEST_identity')
        sources,data_identity=old.fixed_test_inputs()
        if p.sha(ROOT/fixed.TEST_REFERENCE)!=fixed.TEST_REFERENCE_SHA:raise ValueError('Sealed GT reference differs')
        if p.sha(previous.OUT/'test_rows.jsonl')!=L5_ROWS_SHA:raise ValueError('Sealed L5 rows differ')
        l5=rows(previous.OUT/'test_rows.jsonl');sealed=rows(ROOT/fixed.TEST_REFERENCE)
        if len(sources)!=1440 or len(l5)!=1440 or len(sealed)!=1440:raise ValueError('Full TEST required')
        for source,ref,oldrow in zip(sources,sealed,l5):
            if source['image']!=ref['image'] or source['image']!=oldrow['image'] or ref['gt']!=oldrow['gt']:raise ValueError('Same TEST identity/GT required')
            ann=g.ready.DATA/'test/annfiles'/(source['image']+'.txt');poly=np.asarray([float(v) for v in ann.read_text().split()[:8]]).reshape(4,2)
            validate_saved_gt(ref['gt'],poly,np.asarray(source['gt']));source['gt']=ref['gt']
        dataset=old.build_test_dataset(cfg,sources);current=[];maxima={'b':0.,'midpoint':0.}
        with torch.no_grad(),(RUN/'test_fixed_rows.jsonl').open('x') as stream:
            for i,source in enumerate(sources):
                image,metas=old.load_view(dataset,i,source,0);a=predict(image,metas)
                row=dict(image=source['image'],sequence=source['sequence'],domain=source['domain'],frame_id=source['frame_id'],gt=source['gt'],scale=1.,split='test',**a)
                for key,value in previous.compare_boxes(row,l5[i],('b','midpoint')).items():maxima[key]=max(maxima[key],value)
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();current.append(row)
                if i%50==0 or i==1439:resource();status('TEST',done=i+1,total=1440)
        if counts!=dict(features=1448,native_heads=4344,size_heads=1448):raise ValueError('Unexpected forward counts')
        if states()!=before or identity(True)!=now:raise ValueError('Final weight/source/selection drift')
        status('TEST_core_analysis');report,table,accurate=analyze(current,l5)
        report.update(protocol=VERSION,checkpoint=checkpoint,
            original_L5_VAL_selected_epoch=None,candidate_VAL_selected_epoch=read(RUN/'core_summary.json')['selection']['selected_epoch'],user_fixed_representative=True,automatic_adoption=False,training_updates=0,
            delivery_counts=dict(Counter(r['size_delivery'] for r in current)),comparison_L5_rows_SHA=L5_ROWS_SHA)
        save(RUN/'test_fixed_summary.json',report)
        for name,rr in (('test_fixed_metrics.csv',table),('test_fixed_accurate_edges.csv',accurate)):
            with (RUN/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
        save(RUN/'test_fixed_completion.json',dict(passed=True,frames=1440,user_authorized=True,epoch=epoch,checkpoint_sha=checkpoint['sha256'],
            state_before=before,state_after=states(),counts=counts,baseline_max_abs=maxima,source_identity=now,data_identity=data_identity,
            training_updates=0,selection_on_test=False,original_VAL_selection_unchanged=True,TEST_exposed=True,automatic_adoption=False,
            elapsed_seconds=time.monotonic()-started,resources=resources,runtime=runtime))
        names=['test_fixed_summary.json','test_fixed_metrics.csv','test_fixed_accurate_edges.csv','test_fixed_completion.json']
        save(RUN/'test_fixed_core_manifest.json',{n:p.sha(RUN/n) for n in names})
        status('TEST_COMPLETE',frames=1440,epoch=epoch)
    except BaseException as error:
        save(RUN/'test_fixed_failure.json',dict(error=repr(error),training_updates=0,retry=False));raise
    finally:
        if handle is not None:handle.remove()
        torch.optim.Optimizer.__init__=original_optimizer


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('check','prepare','infer'),default='check');args=parser.parse_args()
    if args.stage=='check':print(json.dumps(identity(False),ensure_ascii=False))
    elif args.stage=='prepare':
        save(RUN/'test_fixed_lock.json',dict(identity=identity(True),user_authorized=True,
            authorized_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat()));print('RANKED_ROI_RMS_TEST_LOCKED')
    else:infer()
