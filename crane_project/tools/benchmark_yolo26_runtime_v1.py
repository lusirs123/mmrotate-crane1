"""Native YOLO timing under the shared decoded-frame-to-OBB TEST protocol.

Uses official YOLO.predict on decoded BGR arrays; no replacement preprocessing,
decoding or postprocessing. References are checked outside all timers.
"""
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import benchmark_crane_final_models_v1 as common
from crane_project.tools.prepare_external_obb_v1 import YOLO_CONFIG
from crane_project.utils.external_obb_io_v1 import yolo_to_common


def run_worker(config_path,out):
    cfg,weights=common.contract(config_path,require_weights=True)
    refs=common.references(cfg)['yolo26'];ids=[r['image'] for r in refs]
    paths=common.images(cfg,ids,verify_bytes=True)
    os.environ.update(YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',YOLO_CONFIG_DIR=str(out/'settings'))
    sys.path.insert(0,str(ROOT/'third_party/ultralytics'))
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from crane_project.tools.smoke_faa_gpu_v1 import processes
    from crane_project.tools.benchmark_port_detection_runtime_v1 import host_info
    if Path(ultralytics.__file__).resolve().parent!=ROOT/'third_party/ultralytics/ultralytics':
        raise ValueError('Wrong native YOLO source')
    expected_runtime=common.read(ROOT/'configs/yolo26/crane_yolo26m_obb_resume_v1.json')['runtime']
    if dict(python=sys.version.split()[0],torch=torch.__version__,ultralytics=ultralytics.__version__)!=expected_runtime:
        raise ValueError('Use the same native YOLO runtime as training/export')
    torch.set_num_threads(1);cv2.setNumThreads(1)
    if torch.cuda.device_count()!=1:raise ValueError('Single GPU timing only')
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(cfg['memory']['allocator_mib']*2**20/torch.cuda.get_device_properties(0).total_memory,0)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    probe=torch.empty(1,device='cuda:0')
    expected_gpu=common.read(out/'launch_protocol.json')['gpu']['uuid']
    if {r['uuid'] for r in processes() if r['pid']==os.getpid()}!={expected_gpu}:raise ValueError('GPU identity differs')
    del probe
    def forbidden(*a,**kw):raise RuntimeError('Optimizer construction forbidden during timing')
    torch.optim.Optimizer.__init__=forbidden
    (out/'active_model.json').write_text(json.dumps(dict(model='yolo26')))
    model=YOLO(weights['yolo26'][0],task='obb',verbose=False)
    if model.model.model[-1].__class__.__name__!='OBB26':raise ValueError('Wrong native OBB head')
    model.model.eval().requires_grad_(False)
    registered=sum(p.numel() for p in model.model.parameters())
    head=model.model.model[-1]
    aux={id(p):p for m in head.one2many.values() if m is not None for p in m.parameters()}
    aux_count=sum(p.numel() for p in aux.values())
    options=common.read(ROOT/YOLO_CONFIG)['prediction']
    def predict(frame):
        result=model.predict(source=frame,device='cuda:0',**options)[0]
        obb=result.obb
        boxes=yolo_to_common(obb.xywhr.detach().cpu().numpy(),obb.conf.detach().cpu().numpy(),obb.cls.detach().cpu().numpy())[0]
        return boxes,common.top1(boxes)
    # Native predictor initialization/fusion/warmup is outside the 50 shared
    # warmup frames and all measured frames; record this extra setup explicitly.
    frame=cv2.imread(str(paths[0]))
    with torch.inference_mode():
        _,value=predict(frame);common.compare_top1(value,refs[0]['box'])
    live=model.predictor.model.backend.model
    if not model.predictor.model.end2end or model.predictor.model.fp16:raise ValueError('Need one-to-one FP32 inference')
    if live.training or any(p.requires_grad for p in live.parameters()):raise ValueError('Unfrozen predictor')
    before=common.frozen.state_digest(live)
    boundary=[None];shape=[None];calls=[0]
    original=model.predictor.preprocess
    def preprocess(images):
        value=original(images);torch.cuda.synchronize(0)
        boundary[0]=time.perf_counter();shape[0]=tuple(value.shape)
        if value.dtype!=torch.float32 or shape[0]!=(1,3,1024,1024):raise ValueError('Wrong timing precision/shape')
        return value
    model.predictor.preprocess=preprocess
    handle=live.model[0].register_forward_pre_hook(lambda *a:calls.__setitem__(0,calls[0]+1))
    records=[];first=[];maximum=0.
    with torch.inference_mode(),(out/'yolo26.timings.jsonl').open('x') as log:
        for index in range(cfg['warmup_frames']):
            frame=cv2.imread(str(paths[index]))
            if frame is None:raise ValueError('Image decode failed')
            _,value=predict(frame);common.compare_top1(value,refs[index]['box'])
        torch.cuda.synchronize(0);torch.cuda.reset_peak_memory_stats(0)
        for repeat in range(cfg['repeats']):
            for index,path in enumerate(paths):
                torch.cuda.synchronize(0);file_start=time.perf_counter()
                frame=cv2.imread(str(path))
                if frame is None:raise ValueError('Image decode failed')
                app_start=time.perf_counter();boundary[0]=None
                boxes,value=predict(frame)
                torch.cuda.synchronize(0);finish=time.perf_counter()
                if boundary[0] is None:raise ValueError('Native preprocessing boundary missing')
                maximum=max(maximum,common.compare_top1(value,refs[index]['box']))
                if repeat==0:first.append(value)
                else:common.compare_top1(value,first[index],score_atol=0.)
                row=dict(arm='yolo26',repeat=repeat+1,image=ids[index],native_candidates=len(boxes),
                    decoded_frame_to_obb=finish-app_start,model_and_postprocess=finish-boundary[0],
                    file_to_obb=finish-file_start,preprocessing_h2d=boundary[0]-app_start)
                records.append(row);log.write(json.dumps(row)+'\n')
            log.flush();common.save(out/('yolo26.repeat_%d.json'%(repeat+1)),dict(completed_frames=len(paths)))
            print('yolo26 repeat',repeat+1,'complete',flush=True)
    handle.remove()
    expected=cfg['warmup_frames']+cfg['repeats']*len(paths)
    if calls[0]!=expected or common.frozen.state_digest(live)!=before:raise ValueError('Repeated forward or changed model state')
    common.contract(config_path,require_weights=True);common.references(cfg);common.images(cfg,ids,verify_bytes=True)
    groups={domain:{key:common.timing_summary([r[key] for r in records if domain=='all' or r['image'].startswith(domain+'_')]) for key in common.WINDOWS} for domain in ('all','real','sim')}
    params=dict(registered_including_training_aux=registered,excluding_declared_training_aux=registered-aux_count,
        declared_training_aux=aux_count,native_fused_inference_parameters=sum(p.numel() for p in live.parameters()),
        note='Exclude only native one-to-many cv2/cv3/cv4; fused inference count is reported separately.')
    arm=dict(groups=groups,parameters=params,frozen_states=[before],frozen_states_unchanged=True,
        forward_frames=calls[0],initialization_frame_outside_timing=True,final_stage_calls=None,
        reference_max_component_error=maximum,verified_timed_frames=len(records),
        per_repeat={str(i):common.timing_summary([r['decoded_frame_to_obb'] for r in records if r['repeat']==i]) for i in range(1,cfg['repeats']+1)},
        gpu_peak_mib=dict(allocated=torch.cuda.max_memory_allocated(0)/2**20,reserved=torch.cuda.max_memory_reserved(0)/2**20))
    report=dict(passed=True,protocol=cfg['schema'],runtime=dict(native=expected_runtime,cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),opencv=cv2.__version__,gpu=torch.cuda.get_device_name(0)),host=host_info(),
        split='test',frames=len(ids),repeats=cfg['repeats'],warmup=cfg['warmup_frames'],batch_size=1,precision='FP32',
        training_updates=0,accuracy_evaluated=False,checkpoint_selection=False,modules_benchmarked=False,
        input_pipeline=options,output_verification_tolerances=dict(geometry_atol=1e-4,geometry_rtol=1e-6,historical_score_atol=1e-5,same_runtime_repeat_score_atol=0.),
        runtime_comparison_note='YOLO requires Torch2.5/Ultralytics; MMRotate comparators used Torch1.13. Same hardware/timing scope, different framework runtime.',arms=dict(yolo26=arm))
    common.save(out/'yolo26.completed.json',arm);common.save(out/'runtime_compare.json',report)
    with (out/'runtime_compare.csv').open('x') as stream:
        fields=['model','mean_ms','p50_ms','p95_ms','fps','allocated_peak_mib','reserved_peak_mib','parameters_excluding_declared_aux','parameters_registered']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        t=groups['all']['decoded_frame_to_obb'];m=arm['gpu_peak_mib']
        writer.writerow(dict(model='yolo26',**{k:t[k] for k in ('mean_ms','p50_ms','p95_ms','fps')},allocated_peak_mib=m['allocated'],reserved_peak_mib=m['reserved'],parameters_excluding_declared_aux=registered-aux_count,parameters_registered=registered))
