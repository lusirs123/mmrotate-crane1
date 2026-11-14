"""User-authorized D-only locked TEST and full-path efficiency. No V rerun."""
import argparse
from collections import Counter
import csv
import gc
import gzip
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import train_lvsr_vd_control_v1 as training
from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p

CONFIG=ROOT/'crane_project/configs/crane_lvsr_vd_fixed_test_efficiency_v1.json'
RUN=ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/vd_control_001'
L5_ROWS_SHA='9a11d0b625f0a05e45d7f7bacd7e41ff60def9bcd1f31561d30bd412b64ad240'
EXTRA=['crane_project/configs/crane_lvsr_vd_fixed_test_efficiency_v1.json',
 'crane_project/tools/eval_lvsr_vd_fixed_test_efficiency_v1.py','tests/test_lvsr_vd_fixed_test_efficiency_v1.py',
 'crane_project/tools/eval_lvsr_partial_visual_norm_fixed_test_v1.py',
 'crane_project/tools/benchmark_crane_final_models_v1.py','crane_project/tools/benchmark_port_detection_runtime_v1.py',
 'crane_project/utils/port_detection_runtime_v1.py','configs/crane_stage_efficiency_v1.json']
read,save=training.read,training.save


def rows(path):
    with Path(path).open() as f:return [json.loads(s) for s in f if s.strip()]


def checked_config():
    c=read(CONFIG)
    if (c['physical_gpu']!=3 or c['D_epoch']!=2 or c['TEST_frames']!=1440 or c['efficiency_arms']!=['D']
        or c['warmup_frames']!=50 or c['repeats']!=3 or c['batch_size']!=1 or c['precision']!='FP32'):
        raise ValueError('D-only fixed TEST/efficiency scope differs')
    return c


def validate_training(s,c,selection):
    cfg=checked_config()
    cp=dict(path='D_epoch_02.pth',sha256=cfg['D_sha256'],epoch=2)
    if (c['status']!='DIRECT_CONTROL_TRAIN_VAL_COMPLETE' or c['D_updates']!=15336 or c['D_epochs']!=24
        or c['D_checkpoint']!=cp or selection['representative_checkpoint']!=cp
        or selection['representative_epoch']!=2 or selection['selected_epoch'] is not None
        or len(s['D_all24'])!=24 or [e['epoch'] for e in s['D_all24']]!=list(range(1,25))
        or s['representative_D']['checkpoint']!=cp
        or training.pick(s['D_all24'],p)['epoch']!=2 or c['TEST'] or s['TEST']):
        raise ValueError('Completed predeclared representative D ep2 required')
    return cp


def identity():
    from crane_project.tools import eval_symeood_r4_lvsr_ep5_fixed_test_v1 as previous
    cfg=checked_config();parent=p.checked_sources()
    s=read(RUN/'core_summary.json');c=read(RUN/'completion.json');selection=read(RUN/'selection.json')
    cp=validate_training(s,c,selection)
    protocol=read(RUN/'protocol.json')
    if protocol['identity']['sources']!=training.source_hashes() or protocol['identity']['parent']!=parent:
        raise ValueError('Original D training sources/parents changed')
    for name,h in read(RUN/'core_manifest.json').items():
        if p.sha(RUN/name)!=h:raise ValueError('D training receipt changed: '+name)
    weights={str(RUN/cp['path']):cp['sha256'],str(ROOT/p.parent.R4_CHECKPOINT):p.parent.R4_SHA,
             str(p.midpoint_directory()/'train/head_epoch_20.pth'):p.MIDPOINT_SHA}
    for name,h in weights.items():
        if p.sha(name)!=h:raise ValueError('Frozen weight SHA differs')
    if p.sha(previous.OUT/'test_rows.jsonl')!=L5_ROWS_SHA:raise ValueError('Sealed V/R4/M TEST changed')
    return dict(settings=cfg,checkpoint=cp,weights=weights,parent=parent,
        training_receipts={n:p.sha(RUN/n) for n in ('core_manifest.json','core_summary.json','completion.json','selection.json')},
        sources={n:p.sha(ROOT/n) for n in EXTRA},L5_rows_SHA=L5_ROWS_SHA,
        historical_VAL_selected_epoch=None,TEST_exposed=True,training_updates=0,selection_on_test=False,
        automatic_adoption=False)


def rename_D(value):
    if isinstance(value,dict):return {k.replace('PartialVisualNorm','D'):rename_D(v) for k,v in value.items()}
    if isinstance(value,list):return [rename_D(v) for v in value]
    if isinstance(value,tuple):return tuple(rename_D(v) for v in value)
    return value.replace('PartialVisualNorm','D') if isinstance(value,str) else value


class Monitor:
    """Selected-GPU2-second driver samples; no changes to other processes."""
    def __init__(self):
        self.done=threading.Event();self.error=None;self.samples=[];self.stage='startup'
    def sample(self):
        from crane_project.tools.smoke_faa_gpu_v1 import gpu_rows,processes
        cfg=checked_config();g=next(x for x in gpu_rows() if x['uuid']==cfg['gpu_uuid'])
        apps=[x for x in processes() if x['uuid']==cfg['gpu_uuid']]
        own=sum(x['used_mib'] for x in apps if x['pid']==os.getpid())
        if any(x['pid']!=os.getpid() for x in apps):raise RuntimeError('GPU3 contention; stop')
        if own>=cfg['memory']['process_stop_mib'] or g['free_mib']<cfg['memory']['minimum_free_mib']:
            raise RuntimeError('GPU3 memory limit; stop')
        value=dict(unix_time=time.time(),stage=self.stage,gpu=g,own_process_mib=own)
        self.samples.append(value)
        with (RUN/'test_D_efficiency_resources.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    def loop(self):
        while not self.done.is_set():
            try:self.sample()
            except BaseException as e:self.error=repr(e);return
            self.done.wait(2)
    def check(self):
        if self.error:raise RuntimeError(self.error)
    def start(self):
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def stop(self):
        self.done.set();self.thread.join(timeout=15);self.check()


def efficiency(engine,detector,mid,head,cfg,identity_,references,monitor):
    import cv2
    from crane_project.tools import benchmark_crane_final_models_v1 as b
    from crane_project.tools.benchmark_port_detection_runtime_v1 import transform,prepare,host_info
    from crane_project.utils import port_geometry_refine_g_v1 as geometry
    torch=engine.torch;monitor.stage='D_full_chain_efficiency';c=identity_['settings']
    image_profile=read(ROOT/'configs/crane_stage_efficiency_v1.json')
    ids=[r['image'] for r in references]
    paths=b.images(image_profile,ids,verify_bytes=True)
    pipeline=SimpleNamespace(torch=torch,geometry=geometry,detector=detector,midpoint=mid,lvsr=head)
    predict=b.FinalForward(pipeline,3);compose=transform(cfg)
    models=(detector,mid,head);before=[engine.g.state_digest(m) for m in models]
    calls=[0];handle=detector.backbone.register_forward_pre_hook(lambda *a:calls.__setitem__(0,calls[0]+1))
    measurements=[];maximum=0.;first=[]
    try:
        with torch.no_grad():
            for i in range(c['warmup_frames']):
                frame=cv2.imread(str(paths[i]))
                if frame is None:raise ValueError('Warmup decode failed')
                image,metas=prepare(compose,frame,0);actual=b.top1(predict(image,metas))
                b.compare_top1(actual,references[i]['edge_residual'],score_atol=0.);predict.verify_invariants()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(0)
            with (RUN/'test_D_efficiency_times.jsonl').open('x') as f:
                for repeat in range(c['repeats']):
                    for i,path in enumerate(paths):
                        torch.cuda.synchronize();start=time.perf_counter();frame=cv2.imread(str(path))
                        if frame is None:raise ValueError('Measured decode failed')
                        app=time.perf_counter();image,metas=prepare(compose,frame,0)
                        torch.cuda.synchronize();model=time.perf_counter()
                        result=predict(image,metas);actual=b.top1(result);torch.cuda.synchronize();finish=time.perf_counter()
                        maximum=max(maximum,b.compare_top1(actual,references[i]['edge_residual'],score_atol=0.))
                        predict.verify_invariants()
                        if repeat==0:first.append(actual)
                        else:b.compare_top1(actual,first[i],score_atol=0.)
                        record=dict(repeat=repeat+1,image=ids[i],decoded_frame_to_obb=finish-app,
                            model_and_postprocess=finish-model,file_to_obb=finish-start,preprocessing_h2d=model-app)
                        measurements.append(record);f.write(json.dumps(record)+'\n')
                        if i%50==0:monitor.check();training.status(RUN,'D_efficiency',repeat=repeat+1,done=i+1,total=1440)
                    f.flush()
    finally:handle.remove()
    expected=50+1440*3;b.verify_single_pass(predict.calls,expected,3)
    if calls[0]!=expected or [engine.g.state_digest(m) for m in models]!=before:
        raise ValueError('Timed stages or frozen state differs')
    groups={domain:{window:b.timing_summary([r[window] for r in measurements if domain=='all' or r['image'].startswith(domain+'_')])
        for window in b.WINDOWS} for domain in ('all','real','sim')}
    value=dict(passed=True,arms=['D'],groups=groups,
        per_repeat={str(i):b.timing_summary([r['decoded_frame_to_obb'] for r in measurements if r['repeat']==i]) for i in (1,2,3)},
        params=b.parameter_counts('final',models,detector),active_head_parameters=2450,
        peak_mib=dict(allocated=torch.cuda.max_memory_allocated(0)/2**20,reserved=torch.cuda.max_memory_reserved(0)/2**20),
        timed_frames=4320,warmup_frames=50,counts=predict.calls,backbone_calls=calls[0],
        output_max_component_error=maximum,states_unchanged=True,
        host=host_info(),settings=c,historical_M_V_remeasured=False,
        comparison_boundary='Separate-batch historical context only; no same-batch speedup/increment claim',
        accuracy_reselected=False,training_updates=0)
    save(RUN/'test_D_efficiency_summary.json',value)
    with (RUN/'test_D_efficiency.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['domain','window','mean_ms','p50_ms','p95_ms','fps']);w.writeheader()
        for domain,windows in groups.items():
            for window,m in windows.items():w.writerow(dict(domain=domain,window=window,**{k:m[k] for k in ('mean_ms','p50_ms','p95_ms','fps')}))
    return value


def execute():
    locked=identity()
    for name in ('test_D_lock.json','test_D_rows.jsonl','test_D_completion.json','test_D_failure.json'):
        if (RUN/name).exists():raise ValueError('Preserve existing TEST/efficiency; no retry: '+name)
    save(RUN/'test_D_lock.json',dict(identity=locked,user_authorized=True,efficiency_authorized=True,previous_models_remeasure=False))
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']='3'
    from crane_project.tools import train_symeood_r4_midpoint_v1 as adapted
    from crane_project.tools import eval_symeood_r4_midpoint_fixed_test_v1 as fixed
    from crane_project.tools import eval_symeood_r4_lvsr_ep5_fixed_test_v1 as previous
    from crane_project.tools import eval_lvsr_partial_visual_norm_fixed_test_v1 as metrics
    from crane_project.utils.port_size_pair_direct_control_v1 import PairDirectSizeControlHead
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity
    from crane_project.tools.eval_port_geometry_ap_v1 import validate_saved_gt
    from PIL import Image
    import numpy as np
    adapted.resource_check(3,before=True)
    engine=adapted.make_engine(3);torch=engine.torch;runtime=engine.runtime(0);engine.seed_all();torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original_optimizer=torch.optim.Optimizer.__init__
    def forbidden(*a,**k):raise RuntimeError('No optimizer allowed in frozen TEST/efficiency')
    torch.optim.Optimizer.__init__=forbidden
    monitor=Monitor();monitor.start();handle=None;started=time.monotonic()
    try:
        detector,mid,cfg=engine.load_selected_pipeline(fixed.training_directory()/'train/selection.json',0)
        if runtime!=read(fixed.training_directory()/'roi_cache/cache_manifest.json')['runtime']:raise ValueError('Runtime differs')
        payload=torch.load(str(RUN/locked['checkpoint']['path']),map_location='cpu')
        if payload['epoch']!=2 or payload['updates']!=1278 or payload['parameters']!=2450 or payload['head_type']!='PairDirectSizeControlHead':
            raise ValueError('D checkpoint payload differs')
        head=PairDirectSizeControlHead().cuda();head.load_state_dict(payload['head_state'],strict=True)
        for m in (detector,mid,head):m.eval().requires_grad_(False)
        from crane_project.tools.smoke_faa_gpu_v1 import processes
        if {a['uuid'] for a in processes() if a['pid']==os.getpid()}!={locked['settings']['gpu_uuid']}:
            raise ValueError('Runtime physical GPU mapping differs')
        def states():return {name:engine.g.state_digest(m) for name,m in [('R4',detector),('M20',mid),('D',head)]}
        before=states();captured={};counts=dict(features=0,native_heads=0,D_heads=0)
        def hook(module,args,out):captured.update(inputs=args,midpoint=out['boxes_original'])
        handle=mid.register_forward_hook(hook)
        def audit(event):
            if event['stage']=='test_feature_extracted':counts['features']+=1
            elif event['stage']=='test_native_head_called':counts['native_heads']+=1
            else:raise ValueError('Unknown audit event')
        def predict(image,metas):
            captured.clear();out=engine.old.capture(detector,mid,image,metas,audit=audit);m=captured['midpoint']
            d=head(*captured['inputs'],m);counts['D_heads']+=1
            out.update(edge_residual=d['boxes_original'][0].cpu().tolist() if len(m) else None,
                vector_valid=bool(d['vector_valid'][0]) if len(m) else None,
                size_delivery='no_output' if not len(m) else 'neutral' if bool(d['neutral_bypass'][0]) else 'square' if bool(d['square_bypass'][0]) else 'accepted' if bool(d['accepted'][0]) else 'fallback',
                size_failed_checks=[k for k,v in d['checks'].items() if len(m) and not bool(v[0])])
            if not frozen_identity(out):raise ValueError('D changed inherited M fields')
            return out
        monitor.stage='sample_VAL_replay'
        training.status(RUN,'D_TEST_VAL_replay')
        with gzip.open(RUN/'D_epoch_02_VAL.rows.jsonl.gz','rt') as f:reference={r['image']:r for r in map(json.loads,f)}
        data=engine.g.ready.DATA;manifest=read(data/'manifest.json');metadata={r['id']:r for r in manifest['records'] if r['split']=='val'}
        dataset=engine.dataset_for(cfg,'val',1.);lookup={Path(v['filename']).stem:i for i,v in enumerate(engine.infos(dataset))};proof=[]
        with torch.no_grad():
            for name in fixed.VAL_NAMES:
                ref=reference[name];meta=metadata[name]
                for k in ('images','annfiles'):
                    if p.sha(data/meta[k]['path'])!=meta[k]['sha256']:raise ValueError('VAL bytes differ')
                with Image.open(data/meta['images']['path']) as im:source=dict(ref,image_size=list(im.size))
                image,metas=engine.old.load_view(dataset,lookup[name],source,0);actual=predict(image,metas)
                error=previous.compare_boxes(dict(actual,image=name),ref,('b','midpoint','edge_residual'))
                previous.compare_decisions(dict(actual,failed_checks=actual['size_failed_checks']),ref)
                proof.append(dict(image=name,max_abs=error))
        save(RUN/'test_D_VAL_replay.json',dict(passed=True,frames=8,proof=proof))
        training.status(RUN,'D_TEST_identity')
        sources,data_identity=engine.old.fixed_test_inputs()
        if p.sha(ROOT/fixed.TEST_REFERENCE)!=fixed.TEST_REFERENCE_SHA:raise ValueError('GT reference changed')
        l5=rows(previous.OUT/'test_rows.jsonl');sealed=rows(ROOT/fixed.TEST_REFERENCE)
        if len(sources)!=1440 or len(l5)!=1440 or len(sealed)!=1440:raise ValueError('1440 TEST frames required')
        for source,ref,v in zip(sources,sealed,l5):
            if source['image']!=ref['image'] or source['image']!=v['image'] or ref['gt']!=v['gt']:raise ValueError('TEST identity differs')
            poly=np.asarray([float(x) for x in (data/'test/annfiles'/(source['image']+'.txt')).read_text().split()[:8]]).reshape(4,2)
            validate_saved_gt(ref['gt'],poly,np.asarray(source['gt']));source['gt']=ref['gt']
        dataset=engine.old.build_test_dataset(cfg,sources);current=[];maxima=dict(b=0.,midpoint=0.)
        monitor.stage='D_TEST'
        with torch.no_grad(),(RUN/'test_D_rows.jsonl').open('x') as f:
            for i,source in enumerate(sources):
                image,metas=engine.old.load_view(dataset,i,source,0);value=predict(image,metas)
                row=dict(image=source['image'],sequence=source['sequence'],domain=source['domain'],frame_id=source['frame_id'],gt=source['gt'],scale=1.,split='test',**value)
                for k,e in previous.compare_boxes(row,l5[i],('b','midpoint')).items():maxima[k]=max(maxima[k],e)
                current.append(row);f.write(json.dumps(row,allow_nan=False)+'\n')
                if i%50==0:monitor.check();training.status(RUN,'D_TEST',done=i+1,total=1440);f.flush()
        if counts!=dict(features=1448,native_heads=4344,D_heads=1448):raise ValueError('Audited inference counts differ')
        if states()!=before or identity()!=locked:raise ValueError('Frozen source/state changed')
        training.status(RUN,'D_TEST_analysis')
        report,table,accurate=rename_D(metrics.analyze(current,l5))
        report.update(protocol=locked['settings']['protocol'],checkpoint=locked['checkpoint'],training_updates=0,
            automatic_adoption=False,delivery_counts=dict(Counter(r['size_delivery'] for r in current)))
        save(RUN/'test_D_summary.json',report)
        for name,rr in [('test_D_metrics.csv',table),('test_D_accurate_edges.csv',accurate)]:
            with (RUN/name).open('x',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
        save(RUN/'test_D_completion.json',dict(passed=True,frames=1440,checkpoint=locked['checkpoint'],state_before=before,
            state_after=states(),audit_counts=counts,baseline_max_abs=maxima,data_identity=data_identity,
            runtime=runtime,training_updates=0,selection_on_test=False,TEST_exposed=True,
            rows_SHA=p.sha(RUN/'test_D_rows.jsonl'),elapsed_seconds=time.monotonic()-started))
        handle.remove();handle=None;captured.clear()
        del image,metas,value,dataset
        gc.collect();torch.cuda.empty_cache()
        training.status(RUN,'D_full_chain_efficiency')
        timed=efficiency(engine,detector,mid,head,cfg,locked,current,monitor)
        if states()!=before or identity()!=locked:raise ValueError('Final source/state changed')
        monitor.stop()
        samples=monitor.samples
        save(RUN/'test_D_efficiency_completion.json',dict(passed=True,training_updates=0,selected_D_epoch=2,
            historical_M_V_remeasured=False,weight_state_unchanged=True,driver_sample_seconds=2,
            driver_samples=len(samples),driver_peak_mib=max(r['own_process_mib'] for r in samples),
            efficiency_driver_peak_mib=max(r['own_process_mib'] for r in samples if r['stage']=='D_full_chain_efficiency'),
            minimum_free_mib=min(r['gpu']['free_mib'] for r in samples),elapsed_seconds=time.monotonic()-started,
            sources=locked['sources'],D_rows_SHA=p.sha(RUN/'test_D_rows.jsonl')))
        names=['test_D_lock.json','test_D_VAL_replay.json','test_D_summary.json','test_D_metrics.csv',
            'test_D_accurate_edges.csv','test_D_completion.json','test_D_efficiency_summary.json',
            'test_D_efficiency.csv','test_D_efficiency_completion.json']
        save(RUN/'test_D_core_manifest.json',{n:p.sha(RUN/n) for n in names})
        training.status(RUN,'D_TEST_EFFICIENCY_COMPLETE',D_epoch=2)
    except BaseException as error:
        save(RUN/'test_D_failure.json',dict(error=repr(error),training_updates=0,retry=False))
        training.status(RUN,'D_TEST_EFFICIENCY_FAILED',error=repr(error));raise
    finally:
        if handle is not None:handle.remove()
        monitor.done.set()
        torch.optim.Optimizer.__init__=original_optimizer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:print(json.dumps(checked_config(),indent=2));return
    def timeout(*unused):raise RuntimeError('7200-second fixed TEST/efficiency budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,checked_config()['timeout_seconds'])
    try:execute()
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
