"""Single-GPU native R3Det task training with accumulation and memory caps.

Smoke uses four disposable TRAIN microbatches/two updates. Formal training
starts afresh; no TEST, automatic restart, or checkpoint selection is launched.
"""
import argparse
from datetime import datetime, timezone, timedelta
from functools import partial
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.check_r3det_comparison_v1 import CONFIG, EPOCHS, check_config, source_identity
from crane_project.tools.prepare_external_obb_v1 import DATA, MANIFEST_SHA, dataset_manifest, read, sha, write_new
from crane_project.tools.run_oriented_rcnn_comparison_v1 import assert_sources, RESNET_SHA
from crane_project.tools.smoke_faa_gpu_v1 import stop_group
from crane_project.utils import external_training_monitor_io_v1 as io


def idle(gpu):
    all_rows = io.gpu_rows()
    row = next(r for r in all_rows if r['index']==gpu)
    if row['free_mib'] < 7168 or any(p['uuid']==row['uuid'] for p in io.processes()):
        raise RuntimeError('Selected GPU needs 7168MiB free and no compute process')
    ordinal = next(i for i,r in enumerate(sorted(all_rows,key=lambda r:r['pci'])) if r['uuid']==row['uuid'])
    return row, ordinal


def runtime_check():
    from mmcv import Config
    cfg, _ = check_config()
    if dict(Config.fromfile(str(ROOT/CONFIG),import_custom_modules=False).items()) != cfg:
        raise ValueError('MMCV/static configuration resolution differs')
    return cfg


def geometry_check():
    import numpy as np
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmrotate.core import poly2obb_np
    from mmrotate.datasets import build_dataset
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import canonical
    from crane_project.utils.port_geometry_midpoint_edge_feasibility_v1 import wrap_pi
    cfg = Config.fromfile(str(ROOT/CONFIG))
    import_modules_from_strings(**cfg.custom_imports)
    counts, maxima = {}, dict(center_px=0.,relative_edge=0.,angle_rad=0.)
    specs = cfg.data.train+[cfg.data.val]
    for split,spec in zip(('train','train_sim','val'),specs):
        dataset = build_dataset(spec); counts[split] = len(dataset)
        for info in dataset.data_infos:
            lines=(DATA/split/'annfiles'/(info['img_id']+'.txt')).read_text().splitlines()
            polygons=[np.asarray(line.split()[:8],dtype=np.float32) for line in lines if len(line.split())>=9 and line.split()[8]=='grab']
            if len(polygons)!=len(info['ann']['bboxes']):
                raise ValueError('Object count changed during angle adaptation')
            for poly,oc in zip(polygons,info['ann']['bboxes']):
                if not 0<float(oc[4])<=math.pi/2+1e-6:
                    raise ValueError('Invalid oc angle')
                expected=canonical(poly2obb_np(poly,version='le90'));actual=canonical(oc)
                delta=[max(abs(actual[i]-expected[i]) for i in (0,1)),
                       max(abs(actual[i]/expected[i]-1) for i in (2,3)),
                       abs(wrap_pi(actual[4]-expected[4]))]
                if delta[0]>.002 or delta[1]>1e-4 or delta[2]>1e-5:
                    raise ValueError('Angle conversion changed polygon geometry: '+info['img_id'])
                for key,value in zip(maxima,delta):maxima[key]=max(maxima[key],value)
    if counts != dict(train=1810,train_sim=748,val=887):
        raise ValueError('Dataset counts differ')
    return dict(passed=True,counts=counts,maximum_geometry_difference=maxima,TEST_read=False)


def prepare(out,gpu,smoke=None):
    cfg=runtime_check();sources=source_identity();before,ordinal=idle(gpu)
    verified=0
    for row in dataset_manifest():
        if row['split']=='test':continue
        for kind in ('images','annfiles'):
            if sha(DATA/row[kind]['path'])!=row[kind]['sha256']:raise ValueError('Data drift: '+row['id'])
            verified+=1
    if verified!=6890:raise ValueError('Incomplete TRAIN/VAL byte audit')
    resnet=Path.home()/'.cache/torch/hub/checkpoints/resnet50-0676ba61.pth'
    if sha(resnet)!=RESNET_SHA:raise ValueError('ImageNet cache differs')
    if smoke:
        trial=read(smoke/'protocol.json');result=read(smoke/'result.json')
        if not result['passed'] or trial['sources']!=sources or trial['physical_gpu']!=gpu:
            raise ValueError('Passing same-source same-GPU smoke required')
        geometry=trial['geometry_check']
    else:geometry=geometry_check()
    from crane_project.utils.crane_frozen_pipeline_v1 import read_constants
    selection=read_constants(ROOT/'crane_project/tools/ckpt_sweep.py',['SELECTION_CONFIG'])['SELECTION_CONFIG']
    protocol=dict(model='r3det',config=CONFIG,reference='MMRotate v0.3.4 standard R3Det, not exact TensorFlow paper reproduction',
        created_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),sources=sources,
        physical_gpu=gpu,pci_cuda_ordinal=ordinal,before=before,initialization={str(resnet):RESNET_SHA},
        manifest_sha256=MANIFEST_SHA,verified_train_val_files=verified,geometry_check=geometry,
        epochs=12,train_frames=2558,val_frames=887,microbatch=1,accumulation=2,effective_batch=2,
        expected_microbatches=30696,expected_optimizer_updates=15348,
        optimizer=cfg['optimizer'],lr_config=cfg['lr_config'],dtype='FP32',seed=0,
        allocator_mib=6144,driver_mib=6656,minimum_free_mib=1536,
        own_scale_augmentation=False,formal_training=bool(smoke),smoke_directory=str(smoke) if smoke else None,
        checkpoint_selection=dict(split='val',candidates=EPOCHS,rule=selection),
        auto_resume=False,auto_restart=False,auto_TEST=False)
    out.mkdir(parents=True,exist_ok=False);write_new(out/'protocol.json',protocol)
    return protocol


def guarded_run(out,protocol,smoke=False):
    assert_sources(protocol);idle(protocol['physical_gpu'])
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(protocol['pci_cuda_ordinal']),
        LOCAL_RANK='0',EXTERNAL_RUN_DIR=str(out),PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    command=[sys.executable,str(Path(__file__).resolve()),'--stage','smoke-worker' if smoke else 'worker','--out',str(out)]
    peak=dict(process_mib=0,minimum_free_mib=8192);reason=None;start=time.monotonic();last=protocol['before']
    with (out/'native_train.log').open('x') as log,(out/'gpu_samples.jsonl').open('x') as samples:
        child=subprocess.Popen(command,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_new(out/'launcher.json',dict(pid=child.pid,command=command))
        try:
            while child.poll() is None:
                last=next(r for r in io.gpu_rows() if r['index']==protocol['physical_gpu'])
                apps=io.processes();owned=[p for p in apps if p['pid']==child.pid]
                used=sum(p['used_mib'] for p in owned if p['uuid']==last['uuid'])
                peak['process_mib']=max(peak['process_mib'],used);peak['minimum_free_mib']=min(peak['minimum_free_mib'],last['free_mib'])
                if any(p['uuid']!=last['uuid'] for p in owned):reason='WRONG_GPU'
                if any(p['uuid']==last['uuid'] and p['pid']!=child.pid for p in apps):reason='GPU_CONTENTION'
                if used>protocol['driver_mib'] or last['free_mib']<protocol['minimum_free_mib']:reason='MEMORY_GUARD'
                if smoke and time.monotonic()-start>900:reason='SMOKE_TIMEOUT'
                item=dict(seconds=time.monotonic()-start,gpu=last,owned_processes=owned)
                samples.write(json.dumps(item)+'\n');samples.flush()
                heartbeat=io.read_json(out/'rank_0_heartbeat.json') if (out/'rank_0_heartbeat.json').exists() else None
                io.atomic_json(out/'status.json',dict(state='RUNNING',model='r3det',peak=peak,latest=item,heartbeat=heartbeat,TEST=False))
                if reason:stop_group(child,out);break
                time.sleep(.5 if smoke else 2)
        except BaseException as error:
            reason='MONITOR_ERROR:'+repr(error);traceback.print_exc();stop_group(child,out)
        code=child.wait()
    report_path=out/('rank_0_result.json' if smoke else 'rank_0_complete.json')
    report=read(report_path) if report_path.exists() else None
    passed=code==0 and reason is None and report is not None and report['passed']
    if passed:
        try:
            assert_sources(protocol)
            if smoke:
                if list(out.rglob('*.pth')):raise ValueError('Unexpected trial checkpoint')
            elif (report['optimizer_updates']!=15348 or report['completed_epochs']!=12
                  or not all((out/'train'/('epoch_%d.pth'%e)).exists() for e in EPOCHS)):
                raise ValueError('Incomplete native training or candidate retention')
        except BaseException as error:passed=False;reason=repr(error)
    result=dict(passed=passed,state='COMPLETED' if passed else 'STOPPED_OR_FAILED',exit_code=code,reason=reason,
        elapsed_seconds=time.monotonic()-start,sampled_peak=peak,worker=report,final_observed_gpu=last,
        formal_training=not smoke,auto_restart=False,TEST=False)
    write_new(out/('result.json' if smoke else 'completion.json'),result);io.atomic_json(out/'status.json',result)
    return passed


def smoke_worker(out,torch):
    import copy,logging
    import numpy as np
    from torch.utils.data import DataLoader,Subset
    from mmcv import Config
    from mmcv.parallel import MMDataParallel,collate,scatter
    from mmcv.runner import EpochBasedRunner,Hook,build_optimizer
    from mmcv.utils import import_modules_from_strings
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    cfg=Config.fromfile(str(ROOT/CONFIG));import_modules_from_strings(**cfg.custom_imports)
    model=build_detector(cfg.model);model.init_weights();model.cuda(0)
    for head in (model.bbox_head,model.refine_head[0]):
        if len(head.cls_convs)!=4 or len(head.reg_convs)!=4:
            raise ValueError('Expected native four-layer classification/regression towers')
    wrapped=MMDataParallel(model,device_ids=[0]);optimizer=build_optimizer(wrapped,cfg.optimizer)
    updates=[0];step=optimizer.step;observations=[]
    def counted(*a,**kw):
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        if updates[0]>=2 or not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError('Invalid gradient or exceeded trial updates')
        result=step(*a,**kw);updates[0]+=1;return result
    optimizer.step=counted
    dataset=build_dataset(cfg.data.train)
    loader=DataLoader(Subset(dataset,[0,1810,1,1811]),batch_size=1,shuffle=False,num_workers=0,
        collate_fn=partial(collate,samples_per_gpu=1))
    logger=logging.getLogger('r3det_trial');logger.setLevel(logging.INFO);logger.addHandler(logging.StreamHandler())
    runner=EpochBasedRunner(wrapped,optimizer=optimizer,work_dir=str(out),logger=logger,max_epochs=1)
    runner.register_training_hooks(cfg.lr_config,cfg.optimizer_config,checkpoint_config=None,log_config=None)
    class Observe(Hook):
        def after_train_iter(self,runner):
            loss=float(runner.outputs['loss'])
            if not math.isfinite(loss) or runner.iter>=4:raise ValueError('Trial loss/bound failed')
            observations.append(dict(iteration=int(runner.iter),updates=updates[0],loss=loss,terms=runner.outputs['log_vars']))
    runner.register_hook(Observe(),priority='LOWEST')
    params={'initial_head':model.bbox_head.retina_reg.weight,'refine_head':model.refine_head[0].retina_reg.weight,
            'feature_refine':model.feat_refine_module[0].conv_1_1.weight}
    initial={name:p.detach().clone() for name,p in params.items()}
    runner.run([loader],[('train',1)])
    changed={name:not torch.equal(initial[name],p) for name,p in params.items()}
    if [o['updates'] for o in observations]!=[0,1,1,2] or not all(changed.values()):
        raise ValueError('Native accumulation/FRM/heads did not update correctly: '+repr(changed))
    if any(m.training for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)):
        raise ValueError('Native norm_eval must freeze batch statistics')
    optimizer.zero_grad(set_to_none=True);model.eval();boxes=[]
    for spec in cfg.data.train:
        spec=copy.deepcopy(spec);spec.update(pipeline=cfg.test_pipeline,test_mode=True)
        ds=build_dataset(spec);item=scatter(collate([ds[0]],samples_per_gpu=1),[0])[0]
        if list(item['img'][0].shape)!=[1,3,1024,1024]:raise ValueError('Input shape differs')
        with torch.no_grad():value=np.asarray(model(return_loss=False,rescale=True,**item)[0][0])
        if value.ndim!=2 or value.shape[1]!=6 or not np.isfinite(value).all():raise ValueError('Invalid native TRAIN output')
        boxes.append(len(value))
    return dict(passed=True,optimizer_updates=updates[0],microbatches=observations,parameters_changed=changed,
        native_TRAIN_boxes=boxes,parameters_discarded=True,VAL_inference=False,TEST=False)


def worker(out,smoke):
    protocol=read(out/'protocol.json');assert_sources(protocol)
    write_new(out/'rank_0_identity.json',dict(pid=os.getpid(),rank=0,physical_gpu=protocol['physical_gpu']))
    import torch
    torch.set_num_threads(1);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(protocol['allocator_mib']*2**20/torch.cuda.get_device_properties(0).total_memory,0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(0);torch.cuda.reset_peak_memory_stats(0)
    try:
        if smoke:
            result=smoke_worker(out,torch)
            result.update(peak_allocated_mib=torch.cuda.max_memory_allocated(0)/2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved(0)/2**20)
            write_new(out/'rank_0_result.json',result)
        else:
            sys.argv=['tools/train.py',str(ROOT/CONFIG),'--seed','0','--deterministic','--work-dir',str(out/'train')]
            runpy.run_path(str(ROOT/'tools/train.py'),run_name='__main__')
    except BaseException as error:
        io.atomic_json(out/'rank_0_failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('check','check-runtime','smoke','launch','queue','worker','smoke-worker'),required=True)
    parser.add_argument('--gpu',type=int,choices=(0,1,2,3),default=0)
    parser.add_argument('--out',type=Path);parser.add_argument('--smoke',type=Path)
    args=parser.parse_args();os.chdir(ROOT)
    if args.stage=='check':
        cfg,_=check_config();print(json.dumps(dict(passed=True,sources=len(source_identity()),optimizer=cfg['optimizer'],lr_config=cfg['lr_config']),indent=2));return
    if args.stage=='check-runtime':
        runtime_check();print(json.dumps(geometry_check(),indent=2));return
    if args.out is None:parser.error('--out required')
    out=args.out.resolve()
    if args.stage in ('worker','smoke-worker'):return worker(out,args.stage=='smoke-worker')
    if args.stage=='queue':return 0 if guarded_run(out,read(out/'protocol.json')) else 1
    if args.stage=='smoke':return 0 if guarded_run(out,prepare(out,args.gpu),smoke=True) else 1
    if args.smoke is None:parser.error('--smoke required for formal launch')
    prepare(out,args.gpu,args.smoke.resolve())
    with (out/'supervisor.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--stage','queue','--out',str(out)],
            cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pid=child.pid,physical_gpu=args.gpu,out=str(out),epochs=12,auto_TEST=False)
    write_new(out/'launch.json',receipt);print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':sys.exit(main())
