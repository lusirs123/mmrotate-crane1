#!/usr/bin/env python3
"""Full-TRAIN midpoint adaptation on frozen, VAL-selected SymEOOD+B.

Stages: check -> cache -> discarded save/reload smoke -> train + full VAL.
No TEST stage or TEST data access. This is head training, not detector retraining.
The old short-fit protocol and its failed probe gate remain unchanged.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import eval_port_geometry_midpoint_v1_test as old
from crane_project.tools.ckpt_sweep import SELECTION_CONFIG, select_best_checkpoint

g, m = old.g, old.m
VERSION = 'port_geometry_midpoint_formal_v1'
PROTOCOL = ROOT/'crane_project/tools/port_geometry_midpoint_formal_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_geometry_midpoint_formal_v1_sources.json'
SETTINGS = dict(epochs=24, batch_size=8, seed=1703, lr=.001,
    weight_decay=0., clip_norm=10., train_scales=[1., .5], val_scale=1.,
    max_cpu_tensor_bytes=1024*2**20, train_center_match_px=15.,
    balanced_batch_real=4, balanced_batch_sim=4)
SMOKE_STATUS = 'FORMAL_MIDPOINT_SMOKE_SAVE_RELOAD_PASS_DISCARDED'


def protocol_document():
    return dict(protocol=VERSION, settings=deepcopy(SETTINGS),
        midpoint_settings=deepcopy(m.SETTINGS), parameter_count=m.PARAMETER_COUNT,
        frozen_b=deepcopy(g.ready.FROZEN_B), train_counts=g.ready.TRAIN_COUNTS,
        val_counts=g.ready.VAL_COUNTS, selection_config=deepcopy(SELECTION_CONFIG),
        authorization='New user-authorized full-TRAIN head candidate; does not change the historical failed probe verdict.',
        initialization='New seed1703 zero-output midpoint head. Do not resume or initialize from the exposed-TEST short-fit head.',
        training='Frozen B eval/no_grad. Extract every TRAIN identity at1.0/.5 and VAL at1.0 once. Only aligned detached local ROI/support cached on CPU. Train only midpoint for24 epochs.',
        sampling='Every eligible TRAIN view visited each epoch. Independently shuffle each domain, cycle the smaller domain and pad the last batch to4+4. No VAL, GT-error or TEST sampling preference.',
        eligibility='All TRAIN views recorded; optimize only B-present views with original center error<15px, inherited from finite v1. Eligibility uses TRAIN GT offline only; report missing/mismatched counts by sequence/scale. All VAL frames evaluated without eligibility filtering.',
        augmentation='Deterministic1.0/.5 isotropic views, no new flips/photometry. This is the head adaptation protocol, not a rerun of B random0.5..1.0 augmentation.',
        objective='Unchanged normalized four-midpoint SmoothL1 beta.1. No added temporal, center, edge, angle, classification or reliability loss.',
        checkpoint='Save all24 head/optimizer/RNG snapshots with file-stream atomic no-overwrite IO and exact reload verification. Smoke updates discarded. Keep B checkpoint separate and immutable.',
        selection='Use existing ckpt_sweep SELECTION_CONFIG and select_best_checkpoint on ALL24 full-VAL results only; no averaging, TEST selection or subset search. Selection does not automatically approve replacement.',
        validation='Paired same-cache B/midpoint/candidate; all887 VAL frames, both domains and every sequence. Output/conditional center/all-frame coverage separately; pure periodic angle distinct from protocol penalty. Compute full-video temporal metrics on VAL via historical evaluator test-mode, with split explicitly VAL.',
        online='B native original/model boxes, scale_xy and detached aligned P3 only. Original midpoint architecture/point matching/decoder/fallback and score/count invariants unchanged. No GT/domain/sequence online inputs.',
        scope=dict(detector_updates=0, train_frames=2558, train_views=5116,
            val_frames=887, cache_feature_extractions=6003,
            full_detector_finetuning=False, test_access=False,
            formal_head_training=True, automatic_promotion=False,
            original_probe_gate_passed=False, test_repeatedly_exposed=True))


def checked_sources():
    fixed = json.loads(SOURCES.read_text())
    parent = json.loads(old.MANIFEST.read_text())
    required = set(parent['sources']) | {str(old.MANIFEST.relative_to(ROOT)),
        str(PROTOCOL.relative_to(ROOT)),
        'crane_project/tools/train_port_geometry_midpoint_formal_v1.py',
        'crane_project/tools/ckpt_sweep.py',
        'tests/test_port_geometry_midpoint_formal_v1.py'}
    if (fixed.get('protocol') != VERSION or set(fixed['sources']) != required or
            json.loads(PROTOCOL.read_text()) != protocol_document()):
        raise ValueError('Formal midpoint source/protocol contract differs')
    actual = {p:g.ready.sha(ROOT/p) for p in required}
    if actual != fixed['sources']:
        raise ValueError('Source SHA differs: '+', '.join(p for p in actual if actual[p] != fixed['sources'][p]))
    if any(actual[p] != sha for p, sha in parent['sources'].items()):
        raise ValueError('Historical midpoint runtime sources changed')
    return dict(protocol_sha256=g.ready.sha(PROTOCOL), sources_sha256=g.ready.sha(SOURCES),
        sources=actual, frozen_b=deepcopy(g.ready.FROZEN_B))


def seed_all():
    random.seed(SETTINGS['seed']); np.random.seed(SETTINGS['seed'])
    torch.manual_seed(SETTINGS['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SETTINGS['seed'])
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True


def atomic_save(path, value):
    """File stream avoids PyTorchFileWriter filename errors; never overwrite."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, str(path))
    finally:
        os.unlink(temporary)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    if value['cuda'] is not None:
        torch.cuda.set_rng_state_all(value['cuda'])


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys()==b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a)==type(b) and len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def save_checkpoint(path, head, optimizer, epoch, updates, identity, context=None):
    payload=dict(protocol=VERSION, identity=identity, epoch=epoch, updates=updates,
        head_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
        optimizer_state=deepcopy(optimizer.state_dict()), rng=rng_state(),
        head_digest=g.state_digest(head), frozen_b=g.ready.FROZEN_B,
        context=deepcopy(context))
    atomic_save(path, payload)
    reloaded=torch.load(str(path), map_location='cpu')
    if not tree_equal(payload, reloaded):
        raise ValueError('Checkpoint head/optimizer/RNG reload differs')
    return dict(path=Path(path).name, sha256=g.ready.sha(path), epoch=epoch,
        updates=updates, head_digest=payload['head_digest'], save_reload_exact=True)


def checked_data():
    """Only TRAIN/VAL folders are opened; TEST images/annotations stay closed."""
    manifest=json.loads((g.ready.DATA/'manifest.json').read_text())
    selected=[r for r in manifest['records'] if r['split'] in ('train','train_sim','val')]
    indexed={r['id']:r for r in selected}
    if len(selected)!=len(indexed):
        raise ValueError('Duplicate TRAIN/VAL manifest identity')
    fixture=json.loads(g.FIXTURE.read_text())
    result=[]; identities={}
    for split in ('train','train_sim','val'):
        anns=g.ready.files(g.ready.DATA/split/'annfiles','.txt')
        images=g.ready.files(g.ready.DATA/split/'images','.jpg')
        counts=(g.ready.VAL_COUNTS if split=='val' else
            {k:v for k,v in g.ready.TRAIN_COUNTS.items() if g.ready.TRAIN_SPLITS[k]==split})
        ann_sha=g.ready.set_sha(anns)
        expected=(g.ready.FROZEN_B['annotation_sha256'] if split=='val' else
            fixture['train_annotation_identities'][split+'_annotation_sha256'])
        if (Counter(p.stem.rsplit('_',1)[0] for p in anns)!=Counter(counts) or
                {p.stem for p in anns}!={p.stem for p in images} or ann_sha!=expected):
            raise ValueError('TRAIN/VAL names/counts/annotation bytes differ: '+split)
        image_identity=hashlib.sha256()
        for ann in anns:
            name=ann.stem; image=g.ready.DATA/split/'images'/(name+'.jpg')
            source=indexed.get(name); image_sha=g.ready.sha(image)
            if (source is None or source['split']!=split or
                    source['images']['path']!=split+'/images/'+image.name or
                    source['annfiles']['path']!=split+'/annfiles/'+ann.name or
                    source['images']['sha256']!=image_sha or source['annfiles']['sha256']!=g.ready.sha(ann)):
                raise ValueError('TRAIN/VAL manifest byte identity differs: '+name)
            g.ready.polygon_box(ann)
            gt=old.parse_dota_txt(str(ann))
            if len(gt)!=1:
                raise ValueError('Requires one fixed grab GT: '+name)
            with Image.open(image) as im:
                size=list(im.size)
            seq, number=name.rsplit('_',1)
            if source['sequence']!=seq:
                raise ValueError('Manifest sequence differs: '+name)
            result.append(dict(image=name,split=split,sequence=seq,domain=seq.split('_')[0],
                frame_id=int(number),gt=gt[0].tolist(),image_size=size,
                image_sha256=image_sha,annotation_sha256=g.ready.sha(ann)))
            image_identity.update(image.name.encode()); image_identity.update(b'\0')
            image_identity.update(bytes.fromhex(image_sha))
        identities[split]=dict(annotation_sha256=ann_sha,
            image_identity_sha256=image_identity.hexdigest(),frames=len(anns))
    if len(result)!=3445 or len(indexed)!=3445:
        raise ValueError('Full TRAIN2558/VAL887 identity count differs')
    return result,identities


def runtime(gpu):
    import cv2
    import mmcv
    import mmdet
    import mmrotate
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('Frozen-cache small-head training uses one GPU; do not use torchrun/DDP')
    if not torch.cuda.is_available() or not 0<=gpu<torch.cuda.device_count():
        raise ValueError('Valid logical CUDA device required')
    torch.cuda.set_device(gpu)
    torch.cuda.reset_peak_memory_stats(gpu)
    return dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        mmcv=mmcv.__version__,mmdet=mmdet.__version__,mmrotate=mmrotate.__version__,
        opencv=cv2.__version__,gpu=torch.cuda.get_device_name(gpu))


def build_detector(cfg, gpu):
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector as build
    from crane_project.tools.diagnose_port_shape_e_h_train_gradients_v1 import checkpoint_contract
    if g.ready.sha(g.CHECKPOINT)!=g.ready.FROZEN_B['checkpoint_sha256']:
        raise ValueError('Requires unchanged B epoch24 checkpoint')
    import_modules_from_strings(**cfg.custom_imports)
    spec=deepcopy(cfg.model); spec.pretrained=None; spec.train_cfg=None
    detector=build(spec)
    loaded=load_checkpoint(detector,str(g.CHECKPOINT),map_location='cpu',strict=True)
    checkpoint_contract(loaded['meta'],cfg,'b'); del loaded
    g.freeze_detector(detector.cuda(gpu))
    return detector


def capture_cache(detector, image, metas):
    """GT-free online boundary, retaining only detached CPU local tensors."""
    g.assert_detector_frozen(detector); meta=metas[0]; g.checked_meta(meta)
    with torch.no_grad():
        features=detector.extract_feat(image)
        if features[0].shape!=(1,256,128,128):
            raise ValueError('Frozen P3 shape differs')
        raw_np=g.flatten_prediction(detector.simple_test_from_features(features,metas,rescale=False))
        b_np=g.flatten_prediction(detector.simple_test_from_features(features,metas,rescale=True))
        raw,b=image.new_tensor(raw_np),image.new_tensor(b_np)
        if (raw.shape!=b.shape or not torch.equal(raw[:,5],b[:,5]) or
                not torch.allclose(g.map_boxes(raw[:,:5],meta,inverse=True),b[:,:5],atol=1e-4,rtol=1e-6)):
            raise ValueError('Native original coordinate restoration differs')
        roi,support,_=g.sample_local(features[0],raw[:,:5],meta,'aligned')
        after=g.flatten_prediction(detector.simple_test_from_features(features,metas,rescale=False))
        if not np.array_equal(raw_np,after):
            raise ValueError('Local sampling changed frozen B output')
        xy=image.new_tensor(np.asarray(meta['scale_factor']).reshape(-1)[:2]).reshape(1,2).expand(len(b),-1)
        m.frame(b,raw[:,:5],xy)
        return dict(roi=roi.cpu(),support=support.cpu(),boxes_original=b.cpu(),
            boxes_model=raw[:,:5].cpu(),scale_xy=xy.cpu())


def dataset_for(cfg, split, scale):
    from mmrotate.datasets import build_dataset
    from mmdet.datasets.dataset_wrappers import ConcatDataset
    if split=='train':
        return ConcatDataset([build_dataset(v) for v in g.fixed_specs(cfg,scale)])
    if split!='val' or scale!=1.:
        raise ValueError('Only full TRAIN and native-scale VAL allowed')
    spec=deepcopy(cfg.data.val); spec['test_mode']=True
    for key,folder in (('ann_file','annfiles'),('img_prefix','images')):
        if (ROOT/spec.get('data_root','')/spec[key]).resolve()!=(g.ready.DATA/'val'/folder).resolve():
            raise ValueError('Non-VAL loader path')
    return build_dataset(spec)


def infos(dataset):
    if hasattr(dataset,'datasets'):
        return [info for part in dataset.datasets for info in part.data_infos]
    return dataset.data_infos


def tensor_bytes(records):
    return sum(v.numel()*v.element_size() for r in records for v in r.values() if isinstance(v,torch.Tensor))


def create_cache(directory, cfg, gpu, identity, progress):
    from mmcv.parallel import collate,scatter
    directory.mkdir(parents=True,exist_ok=False)
    sources,data_identity=checked_data(); rt=runtime(gpu); seed_all()
    detector=build_detector(cfg,gpu); before=g.state_digest(detector)
    manifest=dict(protocol=VERSION,identity=identity,data_identity=data_identity,
        runtime=rt,detector_state=before,files={},record_counts={},support={},tensor_bytes=0,
        detector_updates=0,feature_extractions=0,native_head_calls=0,test_access=False)
    by_name={r['image']:r for r in sources}
    for split,scale,label in (('train',1.,'train_s1'),('train',.5,'train_s05'),('val',1.,'val_s1')):
        dataset=dataset_for(cfg,split,scale); names=[Path(info['filename']).stem for info in infos(dataset)]
        expected={r['image'] for r in sources if (r['split']=='val')==(split=='val')}
        if len(names)!=len(expected) or set(names)!=expected:
            raise ValueError('Full dataset loader names differ: '+label)
        records=[]
        for index,name in enumerate(names):
            source=by_name[name]
            value=scatter(collate([dataset[index]],samples_per_gpu=1),[gpu])[0]
            image,metas=value['img'][0],value['img_metas'][0]
            if (len(value['img'])!=1 or image.shape!=(1,3,1024,1024) or len(metas)!=1 or
                    Path(metas[0]['filename']).stem!=name or metas[0].get('flip',False) or
                    list(metas[0]['ori_shape'][:2][::-1])!=source['image_size']):
                raise ValueError('Deterministic image view differs: '+name)
            tensors=capture_cache(detector,image,metas)
            gt=torch.tensor(source['gt'],dtype=torch.float32).reshape(1,5)
            center=float((tensors['boxes_original'][0,:2]-gt[0,:2]).norm()) if len(tensors['boxes_original']) else None
            row=dict(source,scale=scale,role='train' if split=='train' else 'val',
                eligible=split=='train' and center is not None and center<SETTINGS['train_center_match_px'],
                center_error_px=center,gt_original=gt,**tensors)
            if len(tensors['boxes_original']):
                m.target_points(gt,tensors['boxes_original'],tensors['boxes_model'],tensors['scale_xy'])
            records.append(row)
            manifest['feature_extractions']+=1; manifest['native_head_calls']+=3
            if index%50==0 or index==len(names)-1:
                progress(dict(stage='cache',shard=label,done=index+1,total=len(names)))
            del value,image,metas,tensors
        size=tensor_bytes(records); manifest['tensor_bytes']+=size
        if manifest['tensor_bytes']>SETTINGS['max_cpu_tensor_bytes']:
            raise ValueError('CPU local tensor cache exceeds1GiB limit')
        path=directory/(label+'.pt'); atomic_save(path,dict(records=records,identity=identity,detector_state=before))
        manifest['files'][path.name]=g.ready.sha(path); manifest['record_counts'][label]=len(records)
        manifest['support'][label]=dict(total=len(records),
            eligible_for_training=sum(r['eligible'] for r in records),
            no_output=sum(not len(r['boxes_original']) for r in records),
            center_mismatch=sum(r['role']=='train' and len(r['boxes_original']) and not r['eligible'] for r in records))
        del dataset,records
    if g.state_digest(detector)!=before:
        raise ValueError('Frozen B state changed during extraction')
    del detector; torch.cuda.empty_cache()
    manifest['status']='COMPLETE_FROZEN_B_TRAIN_VAL_CACHE'
    g.write_new(directory/'cache_manifest.json',manifest)
    return manifest


def load_cache(directory, identity):
    directory=Path(directory); manifest=json.loads((directory/'cache_manifest.json').read_text())
    if (manifest['status']!='COMPLETE_FROZEN_B_TRAIN_VAL_CACHE' or manifest['identity']!=identity or
            manifest['record_counts']!={'train_s1':2558,'train_s05':2558,'val_s1':887} or
            manifest['feature_extractions']!=6003 or manifest['native_head_calls']!=18009 or
            manifest['detector_updates']!=0 or manifest['test_access'] is not False or
            set(manifest['files'])!={'train_s1.pt','train_s05.pt','val_s1.pt'}):
        raise ValueError('Complete formal TRAIN/VAL cache contract differs')
    shards={}
    for name,digest in manifest['files'].items():
        path=directory/name
        if g.ready.sha(path)!=digest:
            raise ValueError('Local ROI cache file SHA differs: '+name)
        payload=torch.load(str(path),map_location='cpu')
        if payload['identity']!=identity or payload['detector_state']!=manifest['detector_state']:
            raise ValueError('Cache shard B/source identity differs')
        rows=payload['records']
        if len(rows)!=manifest['record_counts'][Path(name).stem]:
            raise ValueError('Cache shard count differs')
        shards[Path(name).stem]=rows
    train=shards['train_s1']+shards['train_s05']; val=shards['val_s1']
    validate_records(train,val)
    if tensor_bytes(train+val)!=manifest['tensor_bytes'] or manifest['tensor_bytes']>SETTINGS['max_cpu_tensor_bytes']:
        raise ValueError('Cache tensor budget differs')
    return train,val,manifest


def validate_records(train, val):
    for rows,role,expected in ((train,'train',g.ready.TRAIN_COUNTS),(val,'val',g.ready.VAL_COUNTS)):
        keys=[(r['image'],r['scale']) for r in rows]
        if len(keys)!=len(set(keys)):
            raise ValueError('Duplicate cache image/view')
        scales=SETTINGS['train_scales'] if role=='train' else [1.]
        for scale in scales:
            if Counter(r['sequence'] for r in rows if r['scale']==scale)!=Counter(expected):
                raise ValueError('Cache full sequence/view counts differ')
        for r in rows:
            if (r['role']!=role or r['scale'] not in scales or
                    r['split']!=('val' if role=='val' else g.ready.TRAIN_SPLITS[r['sequence']]) or
                    r['domain']!=r['sequence'].split('_')[0] or r['image']!=r['sequence']+'_'+str(r['frame_id']).zfill(5)):
                raise ValueError('TRAIN/VAL role or identity leakage')
            tensors=[r[k] for k in ('roi','support','boxes_original','boxes_model','scale_xy','gt_original')]
            if any(t.device.type!='cpu' or t.requires_grad or t.grad_fn is not None or not bool(torch.isfinite(t).all()) for t in tensors):
                raise ValueError('Cache must contain finite detached CPU tensors')
            n=len(r['boxes_original'])
            if (n>1 or r['roi'].shape!=(n,256,9,9) or r['support'].shape!=(n,1,9,9) or r['gt_original'].shape!=(1,5)):
                raise ValueError('Cache tensor shape differs')
            m.frame(r['boxes_original'],r['boxes_model'],r['scale_xy'])
            center=float((r['boxes_original'][0,:2]-r['gt_original'][0,:2]).norm()) if n else None
            if r['eligible']!=(role=='train' and center is not None and center<SETTINGS['train_center_match_px']):
                raise ValueError('Cached offline eligibility differs')


def epoch_batches(records, epoch):
    pools={d:[i for i,r in enumerate(records) if r['role']=='train' and r['eligible'] and r['domain']==d]
        for d in ('real','sim')}
    if any(not p for p in pools.values()):
        raise ValueError('No eligible TRAIN views in one domain')
    rng=np.random.RandomState(SETTINGS['seed']+epoch)
    count=math.ceil(max(map(len,pools.values()))/4)
    order={d:rng.permutation(p).tolist() for d,p in pools.items()}
    # Oversample only to complete domain-balanced epochs; every original view visited.
    return [[order[d][(step*4+j)%len(order[d])] for d in ('real','sim') for j in range(4)]
        for step in range(count)]


def batch(records, indices, device):
    rows=[records[i] for i in indices]
    if any(r['role']!='train' or not r['eligible'] for r in rows):
        raise ValueError('Only eligible TRAIN may enter optimization')
    return [torch.cat([r[k] for r in rows]).to(device) for k in
        ('roi','support','boxes_original','boxes_model','scale_xy','gt_original')]


def update(head, optimizer, records, indices, device):
    head.train(); roi,support,b,bm,xy,gt=batch(records,indices,device)
    optimizer.zero_grad(); out=head(roi,support,b,bm,xy)
    target=m.target_points(gt,b,bm,xy); loss,parts=m.point_loss(out['points_roi'],target)
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite four-midpoint loss')
    loss.backward(); params=list(head.parameters())
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
        raise ValueError('Missing/nonfinite head gradient')
    stem_norm=sum(float(p.grad.double().square().sum()) for p in head.stem.parameters())**.5
    output_norm=float(head.output.weight.grad.double().norm())
    norm=float(torch.nn.utils.clip_grad_norm_(params,SETTINGS['clip_norm']))
    after=sum(float(p.grad.double().square().sum()) for p in params)**.5
    if not math.isfinite(norm) or after>SETTINGS['clip_norm']+1e-4:
        raise ValueError('Invalid head gradient clipping')
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in params):
        raise ValueError('Nonfinite updated head parameters')
    return dict(loss=float(loss.detach()),point_parts=parts.detach().cpu().tolist(),
        grad_norm_before_clip=norm,grad_norm_after_clip=after,
        stem_grad_norm=stem_norm,output_grad_norm=output_norm)


def evaluate(head, records, device):
    head.eval(); rows=[]; before=g.state_digest(head)
    with torch.no_grad():
        for r in records:
            roi,support,b,bm,xy=[r[k].to(device) for k in ('roi','support','boxes_original','boxes_model','scale_xy')]
            output=head(roi,support,b,bm,xy); present=bool(len(b))
            pred=output['boxes_original'][0].cpu().tolist() if present else None
            baseline=b[0].cpu().tolist() if present else None
            candidate=output['candidate_original'][0].cpu().tolist() if present and bool(output['candidate_valid'][0]) else None
            if len(output['boxes_original'])!=len(b) or not torch.equal(output['boxes_original'][:,5],b[:,5]):
                raise ValueError('Inference changed frozen B score/count')
            gt=r['gt_original'][0].tolist()
            rows.append(dict(image=r['image'],sequence=r['sequence'],domain=r['domain'],
                frame_id=r['frame_id'],scale=r['scale'],gt=gt,b=baseline,midpoint=pred,
                candidate=candidate,candidate_valid=bool(output['candidate_valid'][0]) if present else None,
                accepted=bool(output['accepted'][0]) if present else None,
                failed_checks=[k for k,v in output['checks'].items() if present and not bool(v[0])],
                metrics={k:g.decompose(np.asarray(gt),np.asarray(v[:5]) if v is not None else None)
                    for k,v in (('b',baseline),('midpoint',pred),('candidate',candidate))}))
    if g.state_digest(head)!=before:
        raise ValueError('Validation changed midpoint parameters/buffers')
    return rows


def summaries(rows):
    groups=dict(overall=rows,real=[r for r in rows if r['domain']=='real'],sim=[r for r in rows if r['domain']=='sim'])
    groups.update({seq:[r for r in rows if r['sequence']==seq] for seq in sorted({r['sequence'] for r in rows})})
    result={}
    logger=logging.getLogger(old.CraneOfflineEvaluator.__module__)
    disabled=logger.disabled; logger.disabled=True
    try:
        for name,subset in groups.items():
            result[name]={method:old.summary([dict(r,pred=r[method],metrics=r['metrics'][method]) for r in subset],continuous=method!='candidate')
                for method in ('b','midpoint','candidate')}
            result[name]['fallbacks']=sum(r['accepted'] is False for r in subset)
            result[name]['paired']=dict(
                new_center_failures=sum(r['metrics']['b']['center_hit'] and not r['metrics']['midpoint']['center_hit'] for r in subset),
                recovered_centers=sum(not r['metrics']['b']['center_hit'] and r['metrics']['midpoint']['center_hit'] for r in subset),
                new_riou_below_0_5=sum(r['metrics']['b']['riou']>=.5 and r['metrics']['midpoint']['riou']<.5 for r in subset))
    finally:
        logger.disabled=disabled
    # Evaluator mode=test requests temporal metrics only; all supplied records are VAL.
    return dict(split='val',metric_mode='full_video_protocol_v2',groups=result)


def comparison_checks(value):
    """Review flags, NOT a changed checkpoint-selection rule or automatic promotion."""
    checks={}; groups=value['groups']
    for domain in ('real','sim'):
        old_value,new=groups[domain]['b'],groups[domain]['midpoint']
        old_fraction=old_value['all_frame_center_correct_fraction']; new_fraction=new['all_frame_center_correct_fraction']
        checks[domain+'/all_frame_center_preserved']=(new_fraction['denominator']==old_fraction['denominator'] and new_fraction['numerator']>=old_fraction['numerator'])
        checks[domain+'/no_new_center_failures']=groups[domain]['paired']['new_center_failures']==0
        checks[domain+'/output_coverage_preserved']=new['output_coverage_fraction']==old_value['output_coverage_fraction']
        checks[domain+'/mean_riou_nonregressing']=new['all_frame_mean_riou']>=old_value['all_frame_mean_riou']-1e-5
        checks[domain+'/center_continuity_preserved']=new['longest_failure_run_frames']['center']<=old_value['longest_failure_run_frames']['center']
        for field in ('long_edge_relative_error','short_edge_relative_error'):
            checks[domain+'/'+field]=new[field]['mean']<=old_value[field]['mean']+1e-6
        metrics_old,metrics_new=old_value['metric_protocol_v2'],new['metric_protocol_v2']
        for metric,direction in (('DFR(%/frame)',-1),('ACI',1),('MCML_max(frames)',-1)):
            key=domain+'/'+metric
            checks[key]=direction*(metrics_new[key]-metrics_old[key])>=-1e-6
    real=groups['real']; sim=groups['sim']
    checks['real/center_rmse_nonregressing']=real['midpoint']['center_error_px']['rmse']<=real['b']['center_error_px']['rmse']+1e-6
    checks['sim/pure_angle_rmse_nonregressing']=sim['midpoint']['angle_error_deg']['rmse']<=sim['b']['angle_error_deg']['rmse']+1e-5
    for seq in g.ready.VAL_COUNTS:
        if seq in groups:
            checks[seq+'/riou_continuity_preserved']=groups[seq]['midpoint']['longest_failure_run_frames']['riou']<=groups[seq]['b']['longest_failure_run_frames']['riou']
    return dict(checks=checks,all_checks_passed=all(checks.values()),automatic_promotion=False)


def support_report(records):
    return {seq+'/'+str(scale):dict(total=len(subset),eligible=sum(r['eligible'] for r in subset),
        no_output=sum(not len(r['boxes_original']) for r in subset),
        center_mismatch=sum(len(r['boxes_original']) and not r['eligible'] for r in subset))
        for seq in sorted(g.ready.TRAIN_COUNTS) for scale in SETTINGS['train_scales']
        for subset in [[r for r in records if r['sequence']==seq and r['scale']==scale]]}


def optimizer_for(head):
    return torch.optim.Adam(head.parameters(),lr=SETTINGS['lr'],weight_decay=SETTINGS['weight_decay'])


def checkpoint_context(cache):
    return {k:deepcopy(cache[k]) for k in ('manifest_sha256','detector_state','runtime','data_identity') if k in cache}


def load_selected_pipeline(selection_path, gpu):
    """Load the complete frozen B + formal head route; opens no dataset or TEST.

    A caller prepares the unchanged B inference view, then invokes old.capture
    with these two frozen modules. tools/test.py cannot load a head-only .pth.
    """
    selection_path=Path(selection_path); selection=json.loads(selection_path.read_text())
    identity=checked_sources(); selected=selection['selected_checkpoint']
    name=selected['path']
    if (selection['protocol']!=VERSION or selection['split']!='val' or selection['identity']!=identity or
            selection['selection_config']!=SELECTION_CONFIG or selection['selection_on_test'] is not False or
            Path(name).name!=name):
        raise ValueError('Requires an unchanged full-VAL formal selection')
    path=selection_path.parent/name
    if g.ready.sha(path)!=selected['sha256']:
        raise ValueError('Selected formal head checkpoint SHA differs')
    payload=torch.load(str(path),map_location='cpu')
    if (payload['protocol']!=VERSION or payload['identity']!=identity or
            payload['epoch']!=selected['epoch'] or payload['updates']!=selected['updates'] or
            payload['frozen_b']!=g.ready.FROZEN_B or
            payload['context']['manifest_sha256']!=selection['cache_manifest_sha256']):
        raise ValueError('Selected formal checkpoint metadata differs')
    cfg=g.check_cfg(); detector=build_detector(cfg,gpu)
    if g.state_digest(detector)!=payload['context']['detector_state']:
        raise ValueError('Inference B state differs from formal training')
    head=m.SpatialMidpointHead().to('cuda:'+str(gpu)); head.load_state_dict(payload['head_state'],strict=True)
    if g.state_digest(head)!=selected['head_digest'] or g.state_digest(head)!=payload['head_digest']:
        raise ValueError('Selected formal head state differs')
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return detector,head,cfg


def smoke_views(val):
    return [r for seq in sorted({r['sequence'] for r in val})
        for r in ([v for v in val if v['sequence']==seq][:1]+[v for v in val if v['sequence']==seq][-1:])]


def online_smoke(cfg, gpu, head, val, cache):
    """Six actual VAL forwards; GT never enters old.capture or the head."""
    detector=build_detector(cfg,gpu); before=g.state_digest(detector)
    if before!=cache['detector_state']:
        raise ValueError('Online frozen B state differs from cache')
    frozen=m.SpatialMidpointHead().to('cuda:'+str(gpu))
    frozen.load_state_dict(head.state_dict(),strict=True)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    dataset=dataset_for(cfg,'val',1.)
    lookup={Path(v['filename']).stem:i for i,v in enumerate(infos(dataset))}
    selected=smoke_views(val); expected=evaluate(head,selected,'cuda:'+str(gpu))
    for r,reference in zip(selected,expected):
        image,metas=old.load_view(dataset,lookup[r['image']],r,gpu)
        current=old.capture(detector,frozen,image,metas)
        for method in ('b','midpoint'):
            a,b=current[method],reference[method]
            if (a is None)!=(b is None) or (a is not None and
                    (a[5]!=b[5] or not np.allclose(a[:5],b[:5],atol=1e-4,rtol=1e-6))):
                raise ValueError('Online/cache prediction differs: '+r['image'])
    if g.state_digest(detector)!=before:
        raise ValueError('Online midpoint mutated frozen B')
    del detector,frozen,dataset; torch.cuda.empty_cache()
    return dict(frames=len(selected),feature_extractions=len(selected),native_head_calls=3*len(selected),
        b_state_unchanged=True,online_cache_agreement=True)


def smoke(train, val, cache, identity, device, out, cfg=None, gpu=None):
    seed_all(); head=m.SpatialMidpointHead().to(device); optimizer=optimizer_for(head)
    initial=g.state_digest(head); batches=epoch_batches(train,1)
    logs=[update(head,optimizer,train,batches[i%len(batches)],device) for i in range(2)]
    if logs[0]['output_grad_norm']<=0 or logs[1]['stem_grad_norm']<=0:
        raise ValueError('Initial output/step2 stem gradients ineffective')
    proof=save_checkpoint(out/'discarded_smoke_head.pth',head,optimizer,0,2,identity,checkpoint_context(cache))
    payload=torch.load(str(out/proof['path']),map_location='cpu')
    replay=m.SpatialMidpointHead().to(device); replay.load_state_dict(payload['head_state'],strict=True)
    opt=optimizer_for(replay); opt.load_state_dict(payload['optimizer_state'])
    first=evaluate(head,smoke_views(val),device); second=evaluate(replay,smoke_views(val),device)
    if first!=second:
        raise ValueError('Reloaded inference differs')
    online=online_smoke(cfg,gpu,replay,val,cache) if cfg is not None else None
    restore_rng(payload['rng']); next_a=update(head,optimizer,train,batches[0],device)
    restore_rng(payload['rng']); next_b=update(replay,opt,train,batches[0],device)
    if next_a!=next_b or g.state_digest(head)!=g.state_digest(replay) or not tree_equal(optimizer.state_dict(),opt.state_dict()):
        raise ValueError('Reloaded optimizer continuation differs')
    return dict(status=SMOKE_STATUS,identity=identity,cache_manifest_sha256=cache['manifest_sha256'],
        initial_state=initial,checkpoint=proof,updates_before_save=2,continuation_updates_each=1,
        optimizer_continuation_exact=True,inference_reload_exact=True,discarded=True,
        online=online,detector_updates=0,test_access=False,logs=logs)


def train_formal(train, val, cache, identity, device, out, progress):
    seed_all(); head=m.SpatialMidpointHead().to(device); optimizer=optimizer_for(head)
    initial=g.state_digest(head); initial_rows=evaluate(head,val,device)
    if any(r['b']!=r['midpoint'] for r in initial_rows):
        raise ValueError('Zero-initial midpoint no longer returns exact B')
    g.write_new(out/'val_baseline.json',summaries(initial_rows))
    checkpoints={}; updates=0; epoch_summaries={}
    for epoch in range(1,SETTINGS['epochs']+1):
        batches=epoch_batches(train,epoch); losses=[]
        for step,indices in enumerate(batches,1):
            metrics=update(head,optimizer,train,indices,device); updates+=1; losses.append(metrics['loss'])
            progress(dict(stage='train',epoch=epoch,step=step,epoch_steps=len(batches),updates=updates,**metrics))
        ckpt=save_checkpoint(out/('head_epoch_%02d.pth'%epoch),head,optimizer,epoch,updates,identity,checkpoint_context(cache))
        rows=evaluate(head,val,device); value=summaries(rows)
        g.write_new(out/('val_epoch_%02d.json'%epoch),value)
        with (out/('val_epoch_%02d.rows.jsonl'%epoch)).open('x') as stream:
            for row in rows:
                stream.write(json.dumps(g.json_native(row),ensure_ascii=False,allow_nan=False)+'\n')
        name=ckpt['path']; checkpoints[name]=dict(metrics=value['groups']['overall']['midpoint']['metric_protocol_v2'],checkpoint=ckpt)
        epoch_summaries[name]=value
        progress(dict(stage='epoch_complete',epoch=epoch,updates=updates,
            loss_mean=float(np.mean(losses)),checkpoint=ckpt,selection_metrics=checkpoints[name]['metrics']))
        print('FORMAL frozen-B midpoint epoch',epoch,'updates',updates,'loss',float(np.mean(losses)),flush=True)
    chosen,_,selection_info=select_best_checkpoint(checkpoints,SELECTION_CONFIG)
    if chosen is None:
        raise ValueError('No valid full-VAL checkpoint')
    selected=checkpoints[chosen]['checkpoint']; payload=torch.load(str(out/chosen),map_location='cpu')
    selected_head=m.SpatialMidpointHead().to(device); selected_head.load_state_dict(payload['head_state'],strict=True)
    selected_rows=evaluate(selected_head,val,device)
    selected_value=summaries(selected_rows)
    if selected_value!=epoch_summaries[chosen]:
        raise ValueError('Selected head reload VAL differs')
    selection=dict(protocol=VERSION,split='val',identity=identity,cache_manifest_sha256=cache['manifest_sha256'],
        selected_checkpoint=selected,selection_info=selection_info,selection_config=deepcopy(SELECTION_CONFIG),
        all_checkpoints=checkpoints,comparison=comparison_checks(selected_value),
        test_access=False,selection_on_test=False,automatic_promotion=False)
    g.write_new(out/'selection.json',selection); g.write_new(out/'selected_val_compare.json',selected_value)
    return dict(status='FORMAL_MIDPOINT_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED',identity=identity,
        cache_manifest_sha256=cache['manifest_sha256'],initial_state=initial,
        epochs_completed=SETTINGS['epochs'],updates=updates,detector_updates=0,
        selection=selection,train_support=support_report(train),test_access=False,
        original_probe_gate_passed=False,automatic_promotion=False)


def run(args):
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=False)
    progress_path=out/'progress.jsonl'; started=time.perf_counter()
    report=dict(protocol=VERSION,stage=args.stage,status='RUNNING',detector_updates=0,test_access=False)
    def progress(value):
        with progress_path.open('a') as stream:
            stream.write(json.dumps(g.json_native(value),ensure_ascii=False,allow_nan=False)+'\n')
        if value.get('stage')=='cache':
            print('FROZEN B cache',value['shard'],value['done'],'/',value['total'],flush=True)
    try:
        identity=checked_sources(); cfg=g.check_cfg(); report['identity']=identity
        if args.stage=='check':
            report.update(status='STATIC_FORMAL_MIDPOINT_CONTRACT_PASS_NO_DATA_GPU_UPDATES',settings=SETTINGS)
        elif args.stage=='cache':
            if not args.cache_dir:
                raise ValueError('--cache-dir required')
            report['cache']=create_cache(Path(args.cache_dir),cfg,args.gpu,identity,progress)
            report['status']='FULL_TRAIN_VAL_FROZEN_B_CACHE_COMPLETE_NO_UPDATES'
        else:
            if not args.cache_dir:
                raise ValueError('--cache-dir required')
            rt=runtime(args.gpu); train,val,cache=load_cache(args.cache_dir,identity)
            cache['manifest_sha256']=g.ready.sha(Path(args.cache_dir)/'cache_manifest.json')
            if rt!=cache['runtime']:
                raise ValueError('Training runtime/GPU differs from frozen extraction')
            report['runtime']=rt; device='cuda:'+str(args.gpu)
            if args.stage=='smoke':
                report.update(smoke(train,val,cache,identity,device,out,cfg,args.gpu))
            else:
                if not args.smoke_report:
                    raise ValueError('--smoke-report required; smoke weights are never used for training')
                proof=json.loads(Path(args.smoke_report).read_text())
                if (proof['status']!=SMOKE_STATUS or proof['identity']!=identity or not proof['discarded'] or
                        proof['cache_manifest_sha256']!=cache['manifest_sha256'] or
                        not proof['optimizer_continuation_exact'] or not proof['inference_reload_exact'] or
                        not isinstance(proof.get('online'),dict) or proof['online'].get('frames')!=6 or not proof['online']['b_state_unchanged'] or
                        not proof['online']['online_cache_agreement']):
                    raise ValueError('Fresh successful discarded smoke proof required')
                path=Path(args.smoke_report).parent/proof['checkpoint']['path']
                if g.ready.sha(path)!=proof['checkpoint']['sha256']:
                    raise ValueError('Discarded smoke save/reload checkpoint differs')
                report['smoke_report_sha256']=g.ready.sha(args.smoke_report)
                report.update(train_formal(train,val,cache,identity,device,out,progress))
        report['elapsed_seconds']=time.perf_counter()-started
        if args.stage!='check':
            report['gpu_peak_mib']=dict(allocated=torch.cuda.max_memory_allocated(args.gpu)/2**20,
                reserved=torch.cuda.max_memory_reserved(args.gpu)/2**20)
        g.write_new(out/'completion.json',report)
        g.write_new(out/'artifacts.json',dict(protocol=VERSION,
            files={p.name:g.ready.sha(p) for p in out.iterdir() if p.is_file()}))
        print('Saved',out/'completion.json','status',report['status'],flush=True)
    except Exception as error:
        report.update(status='FAILED',error=type(error).__name__+': '+str(error),elapsed_seconds=time.perf_counter()-started)
        g.write_new(out/'failure.json',report)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('check','cache','smoke','train'),required=True)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--cache-dir')
    p.add_argument('--smoke-report')
    p.add_argument('--out-dir',required=True)
    return p


if __name__=='__main__':
    if Path.cwd().resolve()!=ROOT:
        raise SystemExit('Run from the symEOOD project root')
    print('Formal midpoint: all TRAIN identities, frozen SymEOOD+B epoch24, head-only24 epochs; VAL-only selection; no TEST.',flush=True)
    run(parser().parse_args())
