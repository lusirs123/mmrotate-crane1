"""Explicit CPU-only identity/load audit. No data loader, forward or optimizer.

Can be streamed to the existing server with ``python - --root ... --out ...``.
Writes only a fresh audit directory; never modifies model/config/cache files.
"""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys

CHECKPOINT='work_dirs/symeood_c_pair_seed0_20261025_v2/110/epoch_20.pth'
CHECKPOINT_SHA='a4a53af6074387643cea9eb553895181efa3db9d62a7ee0fae95725d76b182f1'
CONFIG='work_dirs/symeood_c_pair_seed0_20261025_v2/configs/Coff_seed0.py'
CONFIG_SHA='0fd0b8c6870336490e4b965a3dd94d13ecb99654bd3d6a36da131680d69d2fff'
PROTOCOL='work_dirs/symeood_c_pair_seed0_20261025_v2/protocol.json'
PROTOCOL_SHA='6130b8886c9dfc679492c12106a65c8f6b5daa4daed6f81a94c17c6f85d615b0'
HISTORICAL={
 'B':('work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth',
      '8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23'),
 'midpoint':('work_dirs/crane_symeood_k1_port_day2night_midpoint_sigma_v1/sigma_1p5/head_epoch_03.pth',
      '16c2fb448ac4e1c53530b8086d547f6f6ccb8d6b0763a42391c34f9b337982d7'),
 'A20':('work_dirs/port_size_qalign_v1/train_20261019_v1/fixed_b/head_epoch_20.pth',
      '545e18459d3ead4a54c441d64de27ca44a8394bfdc20cc2caf4c2f7534326479'),
 'LVSR_C6':('work_dirs/port_size_a20_protection_train_v1/train_20261020_v1_gpu2/C/head_epoch_06.pth',
      '04de79a63963062d641fda9ce70e51f30dceb37e048c2b3429d0c2a0161dec52')}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024*1024),b''):digest.update(part)
    return digest.hexdigest()


def plain(value):
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,range):return list(value)
    return value


def audit(root,out):
    root=root.resolve();out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(root))
    checkpoint_present=(root/CHECKPOINT).is_file()
    required=[(CONFIG,CONFIG_SHA),(PROTOCOL,PROTOCOL_SHA)]
    if checkpoint_present:required.append((CHECKPOINT,CHECKPOINT_SHA))
    for name,expected in required:
        assert sha(root/name)==expected,('identity mismatch',name)
    lock=json.loads((root/PROTOCOL).read_text())
    mismatches=[name for name,h in lock['sources'].items()
                if not (root/name).exists() or sha(root/name)!=h]
    assert not mismatches,mismatches
    import torch,mmcv,mmdet,mmrotate
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    assert not torch.cuda.is_initialized(),'CPU audit must not initialize CUDA'
    cfg=Config.fromfile(str(root/CONFIG))
    import_modules_from_strings(**cfg.custom_imports)
    assert cfg.model.bbox_head.loss_cls.use_geometry_weight is False
    assert cfg.model.bbox_head.loss_bbox.type=='SymKLDLoss'
    assert cfg.model.train_cfg.assigner.get('cost_reg_mode','kld')=='kld'
    if not checkpoint_present:
        historical={}
        for label,(name,expected) in HISTORICAL.items():
            path=root/name;entry=dict(path=name,exists=path.is_file(),expected_sha256=expected)
            if path.is_file():entry.update(actual_sha256=sha(path),matched=sha(path)==expected)
            historical[label]=entry
        cache_meta={}
        for name in ('work_dirs/port_results/geometry/port_geometry_midpoint_formal_v1_roi_cache/cache_manifest.json',
                     'work_dirs/port_geometry_midpoint_formal_v1_roi_cache/cache_manifest.json'):
            path=root/name
            if path.exists():
                data=json.loads(path.read_text())
                cache_meta[name]=dict(manifest_sha256=sha(path),identity=data.get('identity'),
                    record_counts=data.get('record_counts'),tensor_bytes=data.get('tensor_bytes'),
                    files={n:dict(expected_sha256=h,exists=(path.parent/n).is_file(),
                        bytes=(path.parent/n).stat().st_size if (path.parent/n).is_file() else None)
                        for n,h in data['files'].items()})
        artifacts=[]
        for split,name,h in [
            ('val','110/ckpt_sweep/epoch_20/preds/results.pkl','2f404509ab4b38a4377ed8ac274bfe14d9a4281027a9f7c6ae3eb09d67409123'),
            ('test','110/ckpt_sweep/final_test/epoch_20/preds/results.pkl','aa31b9333286fcd2e566b104d2c4dc83b0c27a113611688f7af7518d43c4131e')]:
            path=root/'work_dirs/symeood_c_pair_seed0_20261025_v2'/name
            artifacts.append(dict(split=split,path=str(path),exists=path.exists(),expected_sha256=h,
                actual_sha256=sha(path) if path.exists() else None))
        old=root/'work_dirs/symeood_ablation_v1/110/epoch_20.pth'
        receipt=dict(passed=False,audit_completed=True,status='BLOCKED_FIXED_110_CHECKPOINT_MISSING',
            checkpoint=dict(path=CHECKPOINT,expected_sha256=CHECKPOINT_SHA,exists=False),
            strict_loading=dict(attempted=False,missing_keys=None,unexpected_keys=None,
                explanation='Unknown: no checkpoint bytes. Do not report zero incompatible keys.'),
            config=dict(path=CONFIG,sha256=CONFIG_SHA,actual_model=plain(cfg.model),
                train=plain(cfg.data.train),val=plain(cfg.data.val),test=plain(cfg.data.test)),
            source_lock=dict(path=PROTOCOL,sha256=PROTOCOL_SHA,entries=len(lock['sources']),mismatches=mismatches),
            runtime=dict(python=sys.version,torch=torch.__version__,mmcv=mmcv.__version__,mmdet=mmdet.__version__,
                mmrotate=mmrotate.__version__,cuda_initialized=torch.cuda.is_initialized()),
            historical_weights=historical,historical_cache_metadata=cache_meta,prediction_artifacts=artifacts,
            different_old110=dict(path=str(old),exists=old.exists(),sha256=sha(old) if old.exists() else None,
                role='different historical run, never silently substituted'),
            limits=dict(optimizer_updates=0,model_constructions=0,forward_calls=0,cache_shards_loaded=0,
                images_read=0,TEST_inference=0,old_head_transfer_validated=False))
        assert not torch.cuda.is_initialized()
        with (out/'cpu_identity_load.json').open('x') as stream:json.dump(receipt,stream,indent=2)
        print(json.dumps(dict(status=receipt['status'],source_count=len(lock['sources']),
            strict_load_attempted=False,receipt=str(out/'cpu_identity_load.json')),indent=2))
        return
    checkpoint=torch.load(str(root/CHECKPOINT),map_location='cpu')
    state=checkpoint['state_dict'];meta=checkpoint['meta']
    assert all(t.device.type=='cpu' for t in state.values())
    meta_cfg=Config.fromstring(meta['config'],'.py')
    config_checks={key:plain(cfg.get(key))==plain(meta_cfg.get(key))
        for key in ('model','data','optimizer','optimizer_config','lr_config','runner')}
    assert all(config_checks.values()),config_checks
    detector=build_detector(copy.deepcopy(cfg.model))
    result=detector.load_state_dict(state,strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    detector.eval();detector.requires_grad_(False)
    assert all(p.device.type=='cpu' and p.grad is None and not p.requires_grad for p in detector.parameters())
    assert not torch.cuda.is_initialized()
    modules={}
    for name,module in [('detector',detector),('backbone',detector.backbone),('neck',detector.neck),
        ('head',detector.bbox_head),('loss_cls',detector.bbox_head.loss_cls),
        ('loss_bbox',detector.bbox_head.loss_bbox),('assigner',detector.bbox_head.assigner)]:
        path=Path(inspect.getfile(type(module)))
        modules[name]=dict(type=type(module).__qualname__,module=type(module).__module__,
                           path=str(path),sha256=sha(path))
    historical={}
    for label,(name,expected) in HISTORICAL.items():
        path=root/name
        historical[label]=dict(path=name,exists=path.exists(),expected_sha256=expected)
        if path.exists():
            actual=sha(path);historical[label].update(actual_sha256=actual,matched=actual==expected)
            assert actual==expected,(label,actual)
    # Inspect existing metadata only; do not load tensor shards or run a head.
    cache_meta={}
    for name in ('work_dirs/port_results/geometry/port_geometry_midpoint_formal_v1_roi_cache/cache_manifest.json',
                 'work_dirs/port_geometry_midpoint_formal_v1_roi_cache/cache_manifest.json'):
        path=root/name
        if not path.exists():continue
        data=json.loads(path.read_text())
        cache_meta[name]=dict(manifest_sha256=sha(path),identity=data.get('identity'),
            record_counts=data.get('record_counts'),tensor_bytes=data.get('tensor_bytes'),
            files={n:dict(expected_sha256=h,exists=(path.parent/n).exists(),
                           bytes=(path.parent/n).stat().st_size if (path.parent/n).exists() else None)
                   for n,h in data['files'].items()})
    inference_artifacts=[]
    for split,pkl,h in [
        ('val','110/ckpt_sweep/epoch_20/preds/results.pkl','2f404509ab4b38a4377ed8ac274bfe14d9a4281027a9f7c6ae3eb09d67409123'),
        ('test','110/ckpt_sweep/final_test/epoch_20/preds/results.pkl','aa31b9333286fcd2e566b104d2c4dc83b0c27a113611688f7af7518d43c4131e')]:
        path=root/'work_dirs/symeood_c_pair_seed0_20261025_v2'/pkl
        assert sha(path)==h
        inference_artifacts.append(dict(split=split,path=str(path),sha256=h,
            role='hash existing artifact only; not re-inference or metric-based selection'))
    receipt=dict(passed=True,scope='CPU identity/strict checkpoint load and metadata audit only',
        checkpoint=dict(path=CHECKPOINT,sha256=CHECKPOINT_SHA,epoch=meta.get('epoch'),iteration=meta.get('iter')),
        config=dict(path=CONFIG,sha256=CONFIG_SHA,checkpoint_meta_fields_equal=config_checks),
        source_lock=dict(path=PROTOCOL,sha256=PROTOCOL_SHA,entries=len(lock['sources']),mismatches=mismatches),
        state=dict(tensor_keys=len(state),missing_keys=result.missing_keys,unexpected_keys=result.unexpected_keys,
                   parameters=sum(p.numel() for p in detector.parameters()),strict=True),
        modules=modules,model=plain(cfg.model),train_pipeline=plain(cfg.data.train),
        val_pipeline=plain(cfg.data.val),test_pipeline=plain(cfg.data.test),
        runtime=dict(python=sys.version,torch=torch.__version__,mmcv=mmcv.__version__,mmdet=mmdet.__version__,
                     mmrotate=mmrotate.__version__,cuda_initialized=torch.cuda.is_initialized()),
        historical_weights=historical,historical_cache_metadata=cache_meta,prediction_artifacts=inference_artifacts,
        limits=dict(optimizer_updates=0,forward_calls=0,cache_shards_loaded=0,images_read=0,
                    TEST_inference=0,performance_reproduction_established=False,old_head_transfer_validated=False))
    with (out/'cpu_identity_load.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(dict(passed=True,missing_keys=result.missing_keys,unexpected_keys=result.unexpected_keys,
        source_count=len(lock['sources']),tensor_keys=len(state),checkpoint_epoch=meta.get('epoch'),
        cuda_initialized=torch.cuda.is_initialized(),receipt=str(out/'cpu_identity_load.json')),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();audit(args.root,args.out)
