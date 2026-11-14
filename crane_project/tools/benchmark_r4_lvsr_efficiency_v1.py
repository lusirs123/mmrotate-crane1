"""Authorized LB-only full-chain timing; sealed accuracy outputs verify identity."""
import argparse
import csv
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import eval_r4_lvsr_fixed_test_v1 as fixed
from crane_project.tools import benchmark_crane_final_models_v1 as b
read,save=fixed.read,fixed.save
RUN=fixed.RUN
CONFIG=ROOT/'crane_project/configs/crane_r4_lvsr_efficiency_v1.json'
EXTRA=['crane_project/configs/crane_r4_lvsr_efficiency_v1.json',
 'crane_project/tools/benchmark_r4_lvsr_efficiency_v1.py','tests/test_r4_lvsr_efficiency_v1.py',
 'crane_project/tools/benchmark_crane_final_models_v1.py',
 'crane_project/tools/benchmark_port_detection_runtime_v1.py',
 'crane_project/utils/port_detection_runtime_v1.py','configs/crane_stage_efficiency_v1.json']


def checked_config():
    c=read(CONFIG)
    if (c['physical_gpu']!=3 or c['epoch']!=3 or c['checkpoint_sha256']!=fixed.checked_config()['checkpoint_sha256']
        or c['frames']!=1440 or c['warmup_frames']!=50 or c['repeats']!=3 or c['batch_size']!=1
        or c['precision']!='FP32' or c['CPU_threads']!=1 or c['input_shape']!=[1,3,1024,1024]
        or c['accuracy_evaluation'] or c['old_models_remeasure'] or c['selection_on_test'] or c['training_updates']):
        raise ValueError('Only locked LB ep3 full-chain efficiency is authorized')
    return c


def validate_TEST(done,summary,terminal):
    cp=dict(path='head_epoch_03.pth',epoch=3,sha256=checked_config()['checkpoint_sha256'])
    if (not done['passed'] or done['frames']!=1440 or done['checkpoint']!=cp
        or done['state_before']!=done['state_after'] or done['training_updates']!=0
        or done['midpoint_model_built'] or done['midpoint_forwards']!=0 or done['selection_on_test']
        or summary['frames']!=1440 or summary['checkpoint']!=cp or summary['automatic_adoption']
        or not terminal['passed'] or terminal['frames']!=1440 or terminal['checkpoint']!=cp):
        raise ValueError('Completed and independently verified fixed LB TEST required')
    return cp


def identity():
    done=read(RUN/'test_LB_completion.json');summary=read(RUN/'test_LB_summary.json')
    terminal=read(RUN/'test_LB_terminal_verification.json');cp=validate_TEST(done,summary,terminal)
    for n,h in read(RUN/'test_LB_core_manifest.json').items():
        if fixed.p.sha(RUN/n)!=h:raise ValueError('Sealed TEST core drift: '+n)
    accuracy=fixed.identity()
    if accuracy!=read(RUN/'test_LB_lock.json')['identity']:raise ValueError('Accuracy identity changed')
    if fixed.p.sha(RUN/'test_LB_rows.jsonl')!=done['rows_SHA']:raise ValueError('Sealed LB predictions changed')
    return dict(settings=checked_config(),checkpoint=cp,accuracy_identity=accuracy,
        sources={n:fixed.p.sha(ROOT/n) for n in EXTRA},
        accuracy_receipts={n:fixed.p.sha(RUN/n) for n in ('test_LB_core_manifest.json','test_LB_completion.json',
            'test_LB_summary.json','test_LB_terminal_verification.json','test_LB_rows.jsonl')},
        training_updates=0,accuracy_evaluation=False,old_models_remeasured=False,midpoint_model_built=False)


def status(stage,**details):
    v=dict(stage=stage,pid=os.getpid(),time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        efficiency=True,accuracy_evaluation=False,training_updates=0,midpoint_forwards=0,**details)
    tmp=RUN/'efficiency_LB_status.tmp';tmp.write_text(json.dumps(v,indent=2));tmp.replace(RUN/'efficiency_LB_status.json')


class LBForward:
    """One feature extraction, native detection, ROI and LB call per image."""
    def __init__(self,pipeline):
        self.pipeline=pipeline
        self.calls=dict(features=0,detector_head=0,sampling=0,midpoint=0,lvsr=0)
    def __call__(self,image,metas):
        import numpy as np
        p=self.pipeline
        features=p.detector.extract_feat(image);self.calls['features']+=1
        native=p.detector.simple_test_from_features(features,metas,rescale=False);self.calls['detector_head']+=1
        raw_np=np.asarray(native[0][0])
        if raw_np.ndim!=2 or raw_np.shape[1]!=6 or len(raw_np)>1:raise ValueError('Top1 native R4 output required')
        raw=image.new_tensor(raw_np)
        original=p.torch.cat((p.geometry.map_boxes(raw[:,:5],metas[0],inverse=True),raw[:,5:]),1)
        roi,support,_=p.geometry.sample_local(features[0],raw[:,:5],metas[0],'aligned');self.calls['sampling']+=1
        xy=original.new_tensor(np.asarray(metas[0]['scale_factor']).reshape(-1)[:2]).reshape(1,2).expand(len(original),-1)
        size=p.lvsr(roi,support,original,raw[:,:5],xy,original);self.calls['lvsr']+=1
        self.last=(original,size['boxes_original'])
        return size['boxes_original'].cpu().numpy()
    def verify_invariants(self):
        # Outside timers, with no model forward or diagnostic native-head replay.
        torch=self.pipeline.torch;original,output=self.last
        if len(original)!=len(output) or not torch.equal(original[:,[0,1,4,5]],output[:,[0,1,4,5]]):
            raise ValueError('LB changed B presence/center/angle/score')
        if not torch.equal(original[:,2]<original[:,3],output[:,2]<output[:,3]):raise ValueError('LB changed axis order')


def verify_counts(counts,expected):
    if counts!=dict(features=expected,detector_head=expected,sampling=expected,midpoint=0,lvsr=expected):
        raise ValueError('LB timing repeated/skipped a stage or ran M: '+str(counts))


class Monitor(fixed.Monitor):
    def sample(self):
        from crane_project.tools.smoke_faa_gpu_v1 import gpu_rows,processes
        g=next(v for v in gpu_rows() if v['uuid']==self.c['gpu_uuid'])
        apps=[v for v in processes() if v['uuid']==self.c['gpu_uuid']]
        own=sum(v['used_mib'] for v in apps if v['pid']==os.getpid())
        if (any(v['pid']!=os.getpid() for v in apps) or own>=self.c['memory']['process_stop_mib']
            or g['free_mib']<self.c['memory']['minimum_free_mib']):raise RuntimeError('GPU3 contention/memory limit')
        v=dict(unix_time=time.time(),gpu=g,own_process_mib=own);self.samples.append(v)
        with (RUN/'efficiency_LB_resources.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')


def execute():
    for n in ('efficiency_LB_lock.json','efficiency_LB_times.jsonl','efficiency_LB_completion.json','efficiency_LB_failure.json'):
        if (RUN/n).exists():raise ValueError('Preserve previous efficiency, no automatic retry: '+n)
    locked=identity();c=locked['settings']
    save(RUN/'efficiency_LB_lock.json',dict(identity=locked,user_authorized=True,old_models_remeasure=False))
    status('LB_EFFICIENCY_startup');started=time.monotonic()
    from crane_project.tools import train_symeood_r4_midpoint_v1 as adapted
    from crane_project.utils import port_geometry_refine_g_v1 as geometry
    from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
    from crane_project.tools.benchmark_port_detection_runtime_v1 import transform,prepare,host_info
    from crane_project.tools.smoke_faa_gpu_v1 import processes
    import cv2
    adapted.resource_check(3,before=True)
    engine=adapted.make_engine(3);torch=engine.torch;runtime=engine.runtime(0);engine.seed_all();torch.set_num_threads(1)
    cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original_optimizer=torch.optim.Optimizer.__init__
    def forbidden(*args,**kwargs):raise RuntimeError('No optimizer allowed during timing')
    torch.optim.Optimizer.__init__=forbidden
    monitor=Monitor(c);monitor.start();handle=None
    try:
        cfg=engine.check_cfg();detector=engine.build_detector(cfg,0)
        if runtime!=read(RUN/'protocol.json')['runtime']:raise ValueError('Frozen training runtime differs')
        payload=torch.load(str(RUN/locked['checkpoint']['path']),map_location='cpu')
        if (payload['epoch']!=3 or payload['updates']!=1917 or payload['condition']!='B'
            or payload['identity']!=locked['accuracy_identity']['training_identity']
            or payload['parameters']!=2480 or payload['midpoint_model_used']):raise ValueError('Wrong LB payload')
        head=PairSpecificSizeHead().cuda();head.load_state_dict(payload['head_state'],strict=True);del payload
        for m in (detector,head):m.eval().requires_grad_(False)
        if {v['uuid'] for v in processes() if v['pid']==os.getpid()}!={c['gpu_uuid']}:raise ValueError('Physical GPU mapping differs')
        models=(detector,head)
        def states():return {n:engine.g.state_digest(m) for n,m in [('R4',detector),('LB',head)]}
        before=states()
        if before!=read(RUN/'test_LB_completion.json')['state_after']:raise ValueError('Timed model state differs from accuracy model')
        # Discard GT/metric/decision fields immediately; only predictions identify outputs.
        refs=[dict(image=r['image'],box=r['edge_residual'],B=r['b']) for r in fixed.rows(RUN/'test_LB_rows.jsonl')]
        ids=[r['image'] for r in refs];profile=read(ROOT/'configs/crane_stage_efficiency_v1.json')
        if len(refs)!=1440 or len(set(ids))!=1440 or ids!=sorted(ids):raise ValueError('Full sealed TEST order required')
        paths=b.images(profile,ids,verify_bytes=True)
        predict=LBForward(SimpleNamespace(torch=torch,geometry=geometry,detector=detector,lvsr=head));compose=transform(cfg)
        backbone=[0];handle=detector.backbone.register_forward_pre_hook(lambda *a:backbone.__setitem__(0,backbone[0]+1))
        maximum=0.;B_maximum=0.;measurements=[];first=[]
        def verify(actual,index):
            nonlocal maximum,B_maximum
            predict.verify_invariants()
            maximum=max(maximum,b.compare_top1(actual,refs[index]['box'],score_atol=0.))
            B_maximum=max(B_maximum,b.compare_top1(b.top1(predict.last[0].cpu().numpy()),refs[index]['B'],score_atol=0.))
        status('LB_EFFICIENCY_warmup',total=50)
        with torch.no_grad():
            for i in range(c['warmup_frames']):
                frame=cv2.imread(str(paths[i]))
                if frame is None:raise ValueError('Warmup decode failed')
                image,metas=prepare(compose,frame,0);actual=b.top1(predict(image,metas));verify(actual,i)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(0)
            save(RUN/'efficiency_LB_warmup.json',dict(passed=True,frames=50,max_abs=maximum,B_max_abs=B_maximum,
                counts=predict.calls,backbone_calls=backbone[0],midpoint_forwards=0))
            with (RUN/'efficiency_LB_times.jsonl').open('x') as f:
                for repeat in range(c['repeats']):
                    for i,path in enumerate(paths):
                        torch.cuda.synchronize();start=time.perf_counter();frame=cv2.imread(str(path))
                        if frame is None:raise ValueError('Measured decode failed')
                        app=time.perf_counter();image,metas=prepare(compose,frame,0)
                        torch.cuda.synchronize();model=time.perf_counter()
                        result=predict(image,metas);actual=b.top1(result);torch.cuda.synchronize();finish=time.perf_counter()
                        verify(actual,i)
                        if repeat==0:first.append(actual)
                        else:b.compare_top1(actual,first[i],score_atol=0.)
                        row=dict(repeat=repeat+1,image=ids[i],decoded_frame_to_obb=finish-app,
                            model_and_postprocess=finish-model,file_to_obb=finish-start,preprocessing_h2d=model-app)
                        measurements.append(row);f.write(json.dumps(row)+'\n')
                        if i%50==0:monitor.check();engine.resource_check(0);status('LB_EFFICIENCY',repeat=repeat+1,done=i+1,total=1440);f.flush()
        expected=50+1440*3;verify_counts(predict.calls,expected)
        if backbone[0]!=expected or states()!=before or identity()!=locked:raise ValueError('Timed counts/frozen identity differs')
        monitor.stop()
        groups={d:{w:b.timing_summary([r[w] for r in measurements if d=='all' or r['image'].startswith(d+'_')])
            for w in b.WINDOWS} for d in ('all','real','sim')}
        report=dict(passed=True,arm='R4_LB',checkpoint=locked['checkpoint'],settings=c,groups=groups,
            per_repeat={str(i):b.timing_summary([r['decoded_frame_to_obb'] for r in measurements if r['repeat']==i]) for i in (1,2,3)},
            params=b.parameter_counts('r4',models,detector),active_head_parameters=2480,
            peak_mib=dict(allocated=torch.cuda.max_memory_allocated(0)/2**20,reserved=torch.cuda.max_memory_reserved(0)/2**20),
            timed_frames=4320,warmup_frames=50,counts=predict.calls,backbone_calls=backbone[0],
            output_max_component_error=maximum,B_output_max_component_error=B_maximum,states_unchanged=True,
            host=host_info(),runtime=runtime,
            numerical_settings=dict(cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32),
            accuracy_evaluation=False,accuracy_reselected=False,training_updates=0,
            historical_models_remeasured=False,midpoint_model_built=False,midpoint_forwards=0,
            comparison_boundary=c['historical_comparison'])
        save(RUN/'efficiency_LB_summary.json',report)
        with (RUN/'efficiency_LB_metrics.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['domain','window','mean_ms','p50_ms','p95_ms','fps']);w.writeheader()
            for d,windows in groups.items():
                for window,v in windows.items():w.writerow(dict(domain=d,window=window,**{k:v[k] for k in ('mean_ms','p50_ms','p95_ms','fps')}))
        save(RUN/'efficiency_LB_completion.json',dict(passed=True,checkpoint=locked['checkpoint'],timed_frames=4320,
            state_before=before,state_after=states(),training_updates=0,accuracy_evaluation=False,midpoint_forwards=0,
            driver_sample_seconds=2,driver_samples=len(monitor.samples),
            driver_peak_mib=max(v['own_process_mib'] for v in monitor.samples),
            minimum_free_mib=min(v['gpu']['free_mib'] for v in monitor.samples),
            times_SHA=fixed.p.sha(RUN/'efficiency_LB_times.jsonl'),resources_SHA=fixed.p.sha(RUN/'efficiency_LB_resources.jsonl'),
            elapsed_seconds=time.monotonic()-started,historical_models_remeasured=False))
        names=['efficiency_LB_lock.json','efficiency_LB_warmup.json','efficiency_LB_summary.json',
            'efficiency_LB_metrics.csv','efficiency_LB_completion.json']
        save(RUN/'efficiency_LB_core_manifest.json',{n:fixed.p.sha(RUN/n) for n in names})
        status('LB_EFFICIENCY_COMPLETE',timed_frames=4320)
    except BaseException as e:
        save(RUN/'efficiency_LB_failure.json',dict(error=repr(e),training_updates=0,retry=False));status('LB_EFFICIENCY_FAILED',error=repr(e));raise
    finally:
        if handle is not None:handle.remove()
        monitor.done.set();torch.optim.Optimizer.__init__=original_optimizer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:print(json.dumps(checked_config(),indent=2));return
    def timeout(*unused):raise RuntimeError('Fixed3600-second LB timing budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,checked_config()['timeout_seconds'])
    try:execute()
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
