"""Frozen FAA VAL selection -> one TEST; reuse native inference/evaluators.

No training, threshold tuning, checkpoint expansion or automatic retry. All
configuration remains in configs. This file only binds stages and resources.
"""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TRAIN = ROOT/'work_dirs/faa_gpu23_20261101_v1/attempt_001'
CONFIG = 'configs/faa/crane_faa_oriented_rcnn_r50_fpn_24e_port_day2night_2gpu_v1.py'
EPOCHS = (16, 18, 20, 22, 24)
WEIGHTS = dict(zip(EPOCHS, (
    '548f00dbbadb1da723c535373011332373aab6cc1a0eccc23529d745742ef156',
    '981cfa98004a74073be32bf5b146f40706cb9b396ef58f9ff247e032c4d297cc',
    '2883485cd98d44f0680323c210e181d16ad5a1331e692f091923921b7a9c30d0',
    '3487caa87bfb82971f8fb936192f3feeefc93cec21bfd81a305518ebd9d3d257',
    '69a8c233a02a79834f5bdcba1f42de9784b28ffc3e6f80a8224a4f33046927cb')))
REF = 'work_dirs/smallpaper_readiness_20261028_v1/saved_predictions_unified.rows.jsonl'
REF_SHA = '1abae6afe6ece14c356c7b5608a5ceb692316172605d970444c41a2a12fa8481'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def status(path, value):
    path = Path(path); tmp = path.with_name(path.name+'.%d.tmp'%os.getpid())
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(str(tmp), str(path))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def checkpoint(epoch):
    return TRAIN/'train'/('epoch_%d.pth'%epoch)


def validate_selection(record, protocol):
    """Pure identity gate: TEST cannot substitute another epoch/rule/config."""
    names = ['epoch_%d'%e for e in EPOCHS]
    if record['candidate_epochs'] != list(EPOCHS) or list(record['all_checkpoints']) != names:
        raise ValueError('Incomplete or expanded VAL candidate set')
    if record['config_sha256'] != protocol['sources'][CONFIG]:
        raise ValueError('VAL configuration identity differs')
    if record['selection_config'] != protocol['selection_config']:
        raise ValueError('VAL selection rule differs')
    if record['center_thresh_px'] != 15.0 or record['evidence_role'] != 'source_val_checkpoint_selection':
        raise ValueError('Not the frozen source-VAL selection')
    chosen = record['selected_checkpoint']
    if chosen not in names: raise ValueError('Invalid selected epoch')
    for e in EPOCHS:
        r = record['all_checkpoints']['epoch_%d'%e]
        if r['checkpoint_sha256'] != WEIGHTS[e] or Path(r['checkpoint']) != checkpoint(e):
            raise ValueError('Candidate weight identity differs')
    if record['selected_path'] != record['all_checkpoints'][chosen]['checkpoint']:
        raise ValueError('Selected path differs')
    return int(chosen.split('_')[-1])


def verify(protocol):
    for name, digest in protocol['sources'].items():
        if sha(ROOT/name) != digest: raise ValueError('Source drift: '+name)


def prepare(out):
    from crane_project.tools import ckpt_sweep as s
    from crane_project.tools.smoke_faa_gpu_v1 import gpu_rows
    assert read(TRAIN/'completion.json')['passed']
    train = read(TRAIN/'protocol.json')
    assert train['checkpoint_selection']['candidates'] == list(EPOCHS)
    assert train['checkpoint_selection']['selection_config'] == s.SELECTION_CONFIG
    for name, digest in train['source_sha256'].items(): assert sha(ROOT/name) == digest, name
    for e in EPOCHS: assert sha(checkpoint(e)) == WEIGHTS[e]
    sources = dict(train['source_sha256'])
    for name in ('tools/test.py','crane_project/tools/ckpt_sweep.py',
                 'crane_project/tools/eval_crane_offline.py',
                 'crane_project/tools/eval_port_geometry_ap_v1.py',
                 'crane_project/tools/analyze_symeood_ablation_first4_v1.py',
                 'crane_project/utils/port_geometry_midpoint_edge_residual_metrics_v1.py',
                 'crane_project/tools/eval_faa_val_test_v1.py',
                 'crane_project/tools/smoke_faa_gpu_v1.py'):
        sources[name] = sha(ROOT/name)
    sources['tests/test_faa_val_test_v1.py'] = sha(ROOT/'tests/test_faa_val_test_v1.py')
    assert sha(ROOT/REF) == REF_SHA
    gpus = {r['index']:r for r in gpu_rows() if r['index'] in (2,3)}
    assert len(gpus) == 2 and all(r['free_mib'] >= 7168 for r in gpus.values())
    all_gpus = sorted(gpu_rows(), key=lambda r:r['pci'])
    ordinals = [next(i for i,r in enumerate(all_gpus) if r['uuid']==gpus[g]['uuid']) for g in (2,3)]
    ann, images = s.dataset_paths(str(ROOT/CONFIG),'val')
    assert len(s.get_val_img_ids(ann)) == 887
    data = ROOT/'crane_project/data/crane_grab_port_day2night_v1'
    manifest = read(data/'manifest.json')
    for row in manifest['records']:
        if row['split'] != 'val': continue
        for kind in ('images','annfiles'): assert sha(data/row[kind]['path']) == row[kind]['sha256']
    out.mkdir(parents=True, exist_ok=False)
    save(out/'protocol.json', dict(protocol='faa_fixed_VAL_then_TEST_v1',
        created_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        training_protocol_sha256=sha(TRAIN/'protocol.json'),
        config=CONFIG, sources=sources, candidates=list(EPOCHS), weights=WEIGHTS,
        selection_config=s.SELECTION_CONFIG, val_annotations=ann, val_images=images,
        val_annotations_sha256=s.annotation_set_sha256(ann),
        manifest_sha256=sha(data/'manifest.json'),
        gpus=gpus, ordinals=ordinals, allocator_mib=6144, driver_mib=6656,
        minimum_free_mib=1536, no_updates=True, threshold_changes=False,
        TEST_authorized=True, TEST_after_VAL_lock=True, TEST_reselection=False,
        auto_retry=False, sealed_TEST_reference=REF, sealed_TEST_reference_sha256=REF_SHA))


def worker(task):
    """Wrap the existing multi_gpu_test only to audit immutability/resources."""
    rank = int(os.environ['LOCAL_RANK']); directory = task.parent
    save(directory/('rank_%d_identity.json'%rank), dict(pid=os.getpid(),rank=rank))
    spec = read(task); protocol = read(Path(spec['pipeline'])/'protocol.json')
    verify(protocol)
    assert sha(spec['checkpoint']) == WEIGHTS[spec['epoch']]
    import torch
    import mmdet.apis
    import mmcv.runner
    from mmdet.apis import set_random_seed
    from crane_project.tools.smoke_faa_gpu_v1 import processes
    assert torch.cuda.device_count()==2
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(6144*2**20/torch.cuda.get_device_properties(rank).total_memory,rank)
    torch.set_num_threads(1); set_random_seed(0, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    probe=torch.empty(1,device='cuda:%d'%rank)
    owned={r['uuid'] for r in processes() if r['pid']==os.getpid()}
    assert owned == {protocol['gpus'][str((2,3)[rank])]['uuid']}; del probe
    original=mmdet.apis.multi_gpu_test; calls=[]
    def digest(model):
        h=hashlib.sha256()
        for name,value in model.state_dict().items():
            h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def guarded(model,loader,*args,**kwargs):
        assert len(loader.dataset)==spec['frames']
        assert Path(loader.dataset.ann_file).resolve()==Path(spec['annotations']).resolve()
        assert model.module.neck.__class__.__name__=='FAAFusionFPN'
        assert model.module.roi_head.bbox_head.__class__.__name__=='FAAHead'
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        before=digest(model); count=[0]
        def audit(module,inputs):
            assert not torch.is_grad_enabled() and not module.training
            count[0]+=1
            if count[0]%25==0:
                status(directory/('rank_%d_progress.json'%rank),dict(rank=rank,forwarded=count[0],local_total=len(loader)))
        handle=model.register_forward_pre_hook(audit); torch.cuda.reset_peak_memory_stats(rank)
        try:values=original(model,loader,*args,**kwargs)
        finally:handle.remove()
        assert count[0]==len(loader)==math.ceil(spec['frames']/2)
        after=digest(model); assert before==after
        save(directory/('rank_%d_frozen.json'%rank),dict(passed=True,rank=rank,
            before=before,after=after,forwarded=count[0],frames=spec['frames'],
            DDP_padding_frames=2*count[0]-spec['frames'],training_updates=0,
            peak_allocated_mib=torch.cuda.max_memory_allocated(rank)/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(rank)/2**20))
        calls.append(True);return values
    def forbidden(*a,**k):raise RuntimeError('Optimizer construction forbidden in evaluation')
    torch.optim.Optimizer.__init__=forbidden
    original_load=mmcv.runner.load_checkpoint
    def strict_load(*args,**kwargs):
        kwargs['strict']=True
        return original_load(*args,**kwargs)
    mmcv.runner.load_checkpoint=strict_load
    mmdet.apis.multi_gpu_test=guarded
    sys.argv=['tools/test.py',str(ROOT/CONFIG),spec['checkpoint'],'--launcher','pytorch',
        '--work-dir',str(directory/'native'),'--out',str(directory/'preds/results.pkl'),
        '--tmpdir',str(directory/'collect'),
        '--result-identity-out',str(directory/'runtime_identity.json'),
        '--runtime-audit-out',str(directory/'runtime_audit.json'),'--cfg-options',
        'data.test.ann_file='+spec['annotations'],'data.test.img_prefix='+spec['images']]
    runpy.run_path(str(ROOT/'tools/test.py'),run_name='__main__')
    assert calls==[True] and sha(spec['checkpoint'])==WEIGHTS[spec['epoch']]


def infer(out, split, epoch, protocol):
    from crane_project.tools import ckpt_sweep as s
    from crane_project.tools.smoke_faa_gpu_v1 import gpu_rows,processes,stop_group,worker_pids
    directory=out/split/('epoch_%d'%epoch); directory.mkdir(parents=True,exist_ok=False)
    (directory/'preds').mkdir()
    ann,images=s.dataset_paths(str(ROOT/CONFIG),'val' if split=='val' else 'test')
    frames=887 if split=='val' else 1440
    if split=='test':
        locked=read(out/'selected_identity.json')
        assert locked['epoch']==epoch and locked['selection_sha256']==sha(out/'selection/sweep_results.json')
        data=ROOT/'crane_project/data/crane_grab_port_day2night_v1'
        assert sha(data/'manifest.json')==protocol['manifest_sha256']
        for r in read(data/'manifest.json')['records']:
            if r['split']=='test':
                for k in ('images','annfiles'):assert sha(data/r[k]['path'])==r[k]['sha256']
    spec=dict(pipeline=str(out),split=split,epoch=epoch,checkpoint=str(checkpoint(epoch)),
              annotations=ann,images=images,frames=frames)
    save(directory/'task.json',spec)
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=','.join(map(str,protocol['ordinals'])),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(ROOT),
        PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
        str(Path(__file__).resolve()),'--stage','worker','--out',str(directory/'task.json')]
    peak={str(g):dict(process_mib=0,minimum_free_mib=8192) for g in (2,3)}
    reason=None;started=time.monotonic()
    with (directory/'native.log').open('x') as log,(directory/'gpu_samples.jsonl').open('x') as samples:
        child=subprocess.Popen(command,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(directory/'launcher.json',dict(pid=child.pid,command=command))
        try:
            while child.poll() is None:
                gpus={r['index']:r for r in gpu_rows() if r['index'] in (2,3)}
                pids=worker_pids(directory);apps=[r for r in processes() if r['pid'] in pids]
                assert all(r['uuid'] in {v['uuid'] for v in gpus.values()} for r in apps)
                for g,row in gpus.items():
                    own=sum(a['used_mib'] for a in apps if a['uuid']==row['uuid'])
                    peak[str(g)]['process_mib']=max(peak[str(g)]['process_mib'],own)
                    peak[str(g)]['minimum_free_mib']=min(peak[str(g)]['minimum_free_mib'],row['free_mib'])
                    if own>protocol['driver_mib'] or row['free_mib']<protocol['minimum_free_mib']:reason='GPU_MEMORY_GUARD'
                if time.monotonic()-started>2400:reason='STAGE_TIMEOUT'
                sample=dict(unix_time=time.time(),gpus=gpus,apps=apps)
                samples.write(json.dumps(sample)+'\n');samples.flush()
                status(out/'status.json',dict(state='RUNNING',stage=split,epoch=epoch,
                    stage_directory=str(directory),elapsed_seconds=time.monotonic()-started,peaks=peak,
                    progress=[read(p) for p in directory.glob('rank_*_progress.json')]))
                if reason:stop_group(child,directory);break
                time.sleep(2)
        except BaseException:
            stop_group(child,directory);raise
        code=child.wait()
    assert code==0 and reason is None,(code,reason,str(directory/'native.log'))
    assert all(read(directory/('rank_%d_frozen.json'%r))['passed'] for r in (0,1))
    identity=read(directory/'runtime_identity.json'); ids=s.get_val_img_ids(ann)
    assert [r['frame_key'] for r in identity['runtime_dataset_order']]==ids and len(ids)==frames
    assert identity['checkpoint_sha256']==WEIGHTS[epoch]
    assert identity['results_sha256']==sha(directory/'preds/results.pkl')
    s.check_or_record_prediction(str(ROOT/CONFIG),str(checkpoint(epoch)),str(directory/'preds/results.pkl'),
                                 ann,'source_val' if split=='val' else 'fixed_test',generated=True)
    with (directory/'preds/results.pkl').open('rb') as stream:values=pickle.load(stream)
    import numpy as np
    assert len(values)==frames
    for row in values:
        assert len(row)==1 and row[0].ndim==2 and row[0].shape[1]==6 and np.isfinite(row[0]).all()
        if len(row[0]):assert int(np.argmax(row[0][:,5]))==0,'Legacy Top1 and score-argmax differ'
    s.pkl_to_dota(str(directory/'preds/results.pkl'),ids,str(directory/'preds'))
    save(directory/'completion.json',dict(passed=True,frames=frames,epoch=epoch,split=split,
        checkpoint_sha256=WEIGHTS[epoch],peaks=peak,training_updates=0))
    return directory


def paper_metrics(directory, split):
    """Same raw-float Top1 geometry/AP code as the fixed EOOD/SGAR/R4 reports."""
    import numpy as np
    from crane_project.tools.ckpt_sweep import get_val_img_ids
    from crane_project.tools.eval_crane_offline import parse_seq_frame,compute_riou
    from crane_project.tools.eval_port_geometry_ap_v1 import evaluate_images,verify_native,validate_saved_gt
    from crane_project.tools.analyze_symeood_ablation_first4_v1 import txtbox
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import canonical,row_metrics
    spec=read(directory/'task.json');ids=get_val_img_ids(spec['annotations'])
    with (directory/'preds/results.pkl').open('rb') as stream:preds=pickle.load(stream)
    reference=None
    if split=='test':
        assert sha(ROOT/REF)==REF_SHA
        reference=[json.loads(l) for l in (ROOT/REF).read_text().splitlines()]
        assert [r['image'] for r in reference]==ids
    rows=[]
    native_by_image={image:raw[0] for image,raw in zip(ids,preds)}
    for i,(image,raw) in enumerate(zip(ids,preds)):
        ann=Path(spec['annotations'])/(image+'.txt');gt=txtbox(ann).tolist()
        if reference is not None:
            poly=np.asarray([float(v) for v in ann.read_text().split()[:8]]).reshape(4,2)
            validate_saved_gt(reference[i]['gt'],poly,gt);gt=reference[i]['gt']
        domain,seq,frame=parse_seq_frame(image)
        pred=raw[0][raw[0][:,5].argmax()].tolist() if len(raw[0]) else None
        riou=compute_riou(np.asarray(canonical(pred)),np.asarray(canonical(gt))) if pred else 0.
        rows.append(dict(image=image,domain=domain,sequence=seq,frame_id=frame,gt=gt,prediction=pred,
                         score=pred[5] if pred else 0.,**row_metrics(gt,pred,riou)))
    metrics=[];checks=[]
    for domain in ('all','real','sim'):
        group=[r for r in rows if domain=='all' or r['domain']==domain]
        valid=[r for r in group if r['output']];sim=[r for r in valid if r['domain']=='sim'];n=len(valid)
        value=dict(domain=domain,frames=len(group),outputs=n,
            Recall_pct=100*sum(r['riou']>=.5 for r in group)/len(group),
            mean_RIoU=sum(r['riou'] for r in group)/len(group),
            center_MAE_px=sum(r['center_error_px'] for r in valid)/n if n else None,
            center_RMSE_px=math.sqrt(sum(r['center_error_px']**2 for r in valid)/n) if n else None,
            Sim_angle_RMSE_deg=math.sqrt(sum(r['angle_error_deg']**2 for r in sim)/len(sim)) if sim else None,
            long_MAE_pct=100*sum(r['long_relative_error'] for r in valid)/n if n else None,
            short_MAE_pct=100*sum(r['short_relative_error'] for r in valid)/n if n else None,
            joint10=sum(r['joint_size10'] for r in group),
            joint10_pct=100*sum(r['joint_size10'] for r in group)/len(group))
        gt=[[canonical(r['gt'])] for r in group]
        dets=[[canonical(r['prediction'])+[r['score']]] if r['output'] else [] for r in group]
        for t in (.5,.75):
            v=evaluate_images(dets,gt,t);value['AP%d'%round(t*100)]=v['ap_percent']
            checks.append(dict(domain=domain,threshold=t,**verify_native(dets,gt,t,v)))
            if split=='test':
                complete=[[canonical(b)+[float(b[5])] for b in native_by_image[r['image']]] for r in group]
                full=evaluate_images(complete,gt,t)
                value['native_complete_AP%d'%round(t*100)]=full['ap_percent']
                checks.append(dict(domain=domain,threshold=t,scope='native_complete',
                                   **verify_native(complete,gt,t,full)))
        metrics.append(value)
    with (directory/'per_frame.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    save(directory/'paper_metrics.json',dict(metrics=metrics,native_checks=checks,
        AP_scope='Top1 OBB VOC07 11-point',split=split,training_updates=0))


def controller(out):
    protocol=read(out/'protocol.json');verify(protocol)
    try:
        for epoch in EPOCHS:
            directory=infer(out,'val',epoch,protocol)
            paper_metrics(directory,'val')
        verify(protocol)
        command=[sys.executable,str(ROOT/'crane_project/tools/ckpt_sweep.py'),
            '--config',str(ROOT/CONFIG),'--work-dir',str(TRAIN/'train'),
            '--sweep-dir',str(out/'selection'),'--reuse-preds-from',str(out/'val'),
            '--epochs',*map(str,EPOCHS),'--gpu','2']
        env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (out/'selection.log').open('x') as log:
            subprocess.run(command,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        selected=read(out/'selection/sweep_results.json');epoch=validate_selection(selected,protocol)
        assert sha(checkpoint(epoch))==WEIGHTS[epoch]
        save(out/'selected_identity.json',dict(epoch=epoch,checkpoint=str(checkpoint(epoch)),sha256=WEIGHTS[epoch],
            selection_sha256=sha(out/'selection/sweep_results.json'),
            locked_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),TEST_used_for_selection=False))
        directory=infer(out,'test',epoch,protocol)
        paper_metrics(directory,'test');verify(protocol)
        result=dict(passed=True,state='COMPLETED',selected_epoch=epoch,checkpoint_sha256=WEIGHTS[epoch],
            selection=str(out/'selection/sweep_results.json'),TEST_metrics=str(directory/'paper_metrics.json'),
            TEST_runs=1,TEST_reselection=False,training_updates=0)
        save(out/'completion.json',result);status(out/'status.json',result)
    except BaseException as error:
        status(out/'status.json',dict(state='FAILED',error=repr(error),auto_retry=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('launch','controller','worker'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();os.chdir(ROOT);out=args.out.resolve()
    if args.stage=='worker':return worker(out)
    if args.stage=='controller':return controller(out)
    prepare(out)
    with (out/'pipeline.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--stage','controller','--out',str(out)],
            cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(out/'launch.json',dict(pid=child.pid,out=str(out),auto_TEST_after_VAL=True))
    print(json.dumps(read(out/'launch.json'),indent=2),flush=True)


if __name__=='__main__':main()
