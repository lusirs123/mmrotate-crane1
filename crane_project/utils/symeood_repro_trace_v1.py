"""Observation-only state/input capture for the one canonical SymEOOD entry.

No model/loss/assignment changes, probes, optimizer changes or automatic resume.
"""
import base64
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import random

import numpy as np
import torch
from mmcv.runner import HOOKS,Hook


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def tensor_digest(values):
    h=hashlib.sha256()
    for name,t in values:
        t=t.detach().cpu().contiguous();h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()


def jsonable(x):
    if isinstance(x,dict):return {k:jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    return x.tolist() if hasattr(x,'tolist') else x


def rng_payload():
    state=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
    return base64.b64encode(pickle.dumps(state,protocol=4)).decode()


def rng_digest():
    n=np.random.get_state()
    h=hashlib.sha256(json.dumps([random.getstate(),[n[0],n[1].tolist(),n[2],n[3],n[4]]]).encode())
    h.update(torch.get_rng_state().numpy().tobytes());h.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    return h.hexdigest()


def input_record(data):
    tensors=[('img',data['img'].data[0])]
    for k in ('gt_bboxes','gt_labels'):tensors.extend((k+str(i),v) for i,v in enumerate(data[k].data[0]))
    meta=jsonable(data['img_metas'].data[0])
    return dict(tensors_sha256=tensor_digest(tensors),metadata=meta,
                combined_sha256=hashlib.sha256((tensor_digest(tensors)+json.dumps(meta,sort_keys=True)).encode()).hexdigest())


def append(path,row):
    with Path(path).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')


LIBRARY_MARKERS=('libcuda','libcudnn','libcublas','libnccl','libnvrtc','libtorch','libc10','mmcv/_ext')


def mapped_libraries():
    paths=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        item=line.split()[-1]
        if item.startswith('/') and any(k in item for k in LIBRARY_MARKERS):paths.add(item)
    return sorted(paths)


def runtime():
    import mmcv,mmdet
    return dict(python=__import__('sys').version,torch=torch.__version__,mmcv=mmcv.__version__,mmdet=mmdet.__version__,
        torch_cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        flags=dict(cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32),
        environment={k:os.environ.get(k) for k in ('CUDA_HOME','LD_LIBRARY_PATH','CONDA_PREFIX','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        libraries={p:sha(p) for p in mapped_libraries()})


def assert_runtime_lock(path):
    """Validate loaded subset AND bytes of every frozen expected library."""
    lock=json.loads(Path(path).read_text());current=runtime();expected=lock['runtime']
    assert lock['state']=='NATIVE_GPU_ACCEPTED','CPU preparation is not native GPU acceptance'
    for key in ('python','torch','mmcv','mmdet','torch_cuda','cudnn'):assert current[key]==expected[key],('runtime mismatch',key)
    assert current['flags']==lock['formal_flags'],'formal flags changed'
    assert current['environment']==expected['environment'],'environment changed'
    assert set(current['libraries'])<=set(expected['libraries']),'unexpected loaded CUDA/Torch library'
    for p,h in expected['libraries'].items():assert sha(p)==h,('runtime library bytes changed',p)
    for p,h in lock['source_sha256'].items():assert sha(p)==h,('source changed',p)
    return current


@HOOKS.register_module()
class SymEOODReproTraceV1(Hook):
    def __init__(self,runtime_lock=None,engineering=False):
        assert runtime_lock is not None or engineering,'A formal run requires a runtime lock'
        self.runtime_lock=runtime_lock;self.engineering=engineering

    def before_run(self,runner):
        import torch.distributed as dist
        self.rank=dist.get_rank();self.world=dist.get_world_size();assert self.world==2
        assert runner.iter==0,'Resume is not supported by this version; states are saved, never silently resumed'
        self.model=runner.model.module;self.work=Path(runner.work_dir)
        if self.runtime_lock:assert_runtime_lock(self.runtime_lock)
        digest=rng_digest();state=rng_payload();self.initial_state_sha=tensor_digest(self.model.state_dict().items())
        classes={n:dict(module=type(v).__module__,name=type(v).__qualname__,source=inspect.getfile(type(v)),sha256=sha(inspect.getfile(type(v)))) for n,v in [('model',self.model),('head',self.model.bbox_head),('assigner',self.model.bbox_head.assigner),('classification',self.model.bbox_head.loss_cls),('regression',self.model.bbox_head.loss_bbox)]}
        record=dict(rank=self.rank,initial_state_sha256=self.initial_state_sha,rng=state,classes=classes)
        append(self.work/('repro_initial_rank%d.jsonl'%self.rank),record)
        if self.rank==0:
            path=self.work/'complete_initial_model.pth';assert not path.exists()
            torch.save(dict(state_dict=self.model.state_dict(),rank0_rng=state,classes=classes),path)
        assert digest==rng_digest() and self.initial_state_sha==tensor_digest(self.model.state_dict().items())

    def before_train_epoch(self,runner):
        assert runner.data_loader.num_workers==2 and not runner.data_loader.persistent_workers
        self.input_hash=hashlib.sha256()

    def before_train_iter(self,runner):
        record=input_record(runner.data_batch);self.input_hash.update(record['combined_sha256'].encode())
        append(self.work/('repro_inputs_rank%d.jsonl'%self.rank),dict(epoch=runner.epoch+1,iteration=runner.iter,**record))

    def after_train_epoch(self,runner):
        import torch.distributed as dist
        before_digest=rng_digest();before=rng_payload();params=tensor_digest(self.model.named_parameters());buffers=tensor_digest(self.model.named_buffers())
        local=dict(rank=self.rank,epoch=runner.epoch+1,iteration=runner.iter,sampler_epoch=runner.data_loader.sampler.epoch,
            assigner_calls=self.model.bbox_head.assigner._local_call_count,nfl_calls=int(self.model.bbox_head.loss_cls._local_iter),
            rng=before,input_sha256=self.input_hash.hexdigest(),batches=len(runner.data_loader))
        states=[None]*self.world;dist.all_gather_object(states,local)
        if self.rank==0:
            path=self.work/('repro_state_epoch%d.pth'%(runner.epoch+1));assert not path.exists()
            runner.save_checkpoint(str(self.work),filename_tmpl='repro_state_epoch{}.pth',create_symlink=False,
                meta=dict(reproducibility_v1=states,resume_supported=False))
        dist.barrier()
        checks=dict(rng=before_digest==rng_digest(),parameters=params==tensor_digest(self.model.named_parameters()),buffers=buffers==tensor_digest(self.model.named_buffers()),
            assigner=local['assigner_calls']==self.model.bbox_head.assigner._local_call_count,nfl=local['nfl_calls']==int(self.model.bbox_head.loss_cls._local_iter))
        append(self.work/('repro_conservation_rank%d.jsonl'%self.rank),dict(epoch=runner.epoch+1,checks=checks,passed=all(checks.values())))
        assert all(checks.values()),checks
        if self.runtime_lock:assert_runtime_lock(self.runtime_lock)

    def after_run(self,runner):
        append(self.work/('repro_runtime_rank%d.jsonl'%self.rank),runtime())
