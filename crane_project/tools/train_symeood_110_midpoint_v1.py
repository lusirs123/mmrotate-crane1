#!/usr/bin/env python3
"""Copy of the formal midpoint trainer for the sealed new110 and fixed sigma1.5.

Stages include bounded trial and user-authorized fresh-cache formal midpoint.
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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Keep static check stdlib-only; numerical imports happen only on an explicit runtime stage.
VERSION = 'symeood_110_midpoint_v1'
SOURCES = ROOT/'crane_project/data_contracts/symeood_110_midpoint_v1.json'
BASE_CONTRACT = ROOT/'crane_project/data_contracts/symeood_110_detection_base_v2_20261026.json'
PARENTS = ('port_geometry_midpoint_sigma_v1_sources.json',
    'port_size_qalign_v1_sources.json','port_size_a20_protection_train_v1_sources.json')


def runtime_imports():
    global np, Image, torch, old, g, m, head_model, SELECTION_CONFIG, select_best_checkpoint
    import numpy as np
    from PIL import Image
    import torch
    from crane_project.tools import eval_port_geometry_midpoint_v1_test as old
    from crane_project.tools.ckpt_sweep import SELECTION_CONFIG, select_best_checkpoint
    from crane_project.utils import port_geometry_midpoint_sigma_v1 as head_model
    g, m = old.g, old.m


def new_head():
    return head_model.SigmaMidpointHead(1.5)


SETTINGS = dict(epochs=24, batch_size=8, seed=1703, lr=.001,
    weight_decay=0., clip_norm=10., train_scales=[1., .5], val_scale=1.,
    max_cpu_tensor_bytes=1024*2**20, train_center_match_px=15.,
    balanced_batch_real=4, balanced_batch_sim=4)
SMOKE_STATUS = 'NEW110_SIGMA1P5_SMOKE_SAVE_RELOAD_PASS_DISCARDED'


def protocol_document():
    return dict(protocol=VERSION, settings=deepcopy(SETTINGS),
        midpoint_settings=dict(m.SETTINGS, prior_sigma_cells=1.5), parameter_count=m.PARAMETER_COUNT,
        frozen_b=frontend_identity(), train_counts=g.ready.TRAIN_COUNTS,
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


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def frontend_identity():
    base=json.loads(BASE_CONTRACT.read_text())
    if (base['schema']!='symeood_detection_base_v2' or base['group']!='110' or
            base['status']!='USER_ADOPTED_ARCHIVED_DOWNSTREAM_NOT_VALIDATED' or
            base['factors']!={'A':True,'B':True,'C':False} or base['selection']['epoch']!=20 or
            base['selection']['reselected'] or base['selection']['basis']!='original_VAL_rule_before_TEST'):
        raise ValueError('Requires the sealed adopted new110; never substitute the lost old110')
    return dict(identity=base['identity'],selected_epoch='epoch_20',
        checkpoint_sha256=base['checkpoint_sha256'],config_sha256=base['config_sha256'],
        annotation_sha256='8bd2e2da86d52b555e2cac5e1098361bd01a4ca09f9df9cb0642e07accc1f6be')


def checked_sources():
    fixed=json.loads(SOURCES.read_text())
    required={str(BASE_CONTRACT.relative_to(ROOT)),str(Path(__file__).resolve().relative_to(ROOT)),
        'tests/test_symeood_110_midpoint_v1.py'}
    for parent in PARENTS:
        name='crane_project/tools/'+parent
        required.add(name);required.update(json.loads((ROOT/name).read_text())['sources'])
    # New110's custom imports include the trace hook absent from the historical head closure.
    required.add('crane_project/utils/symeood_repro_trace_v1.py')
    if (fixed['protocol']!=VERSION or fixed['settings']!=SETTINGS or fixed['sigma_cells']!=1.5 or
            fixed['frontend']!=frontend_identity() or fixed['stages']!=['check','cache','smoke','trial','formal'] or
            set(fixed['sources'])!=required):
        raise ValueError('New110 midpoint contract differs')
    for name,digest in fixed['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('New110 source SHA differs: '+name)
    base=json.loads(BASE_CONTRACT.read_text())
    if sha(ROOT/base['config'])!=base['config_sha256']:
        raise ValueError('Sealed new110 expanded config SHA differs')
    if sha(ROOT/base['source_lock'])!=base['evidence_sha256'][base['source_lock']]:
        raise ValueError('Sealed new110 training identity differs')
    if fixed['reuse_unchanged']!=unchanged_functions():
        raise ValueError('Copied mathematical/training functions changed')
    differences={name:dict(historical=digest,current=sha(ROOT/name)) for name,digest in
        json.loads((ROOT/'crane_project/tools'/PARENTS[0]).read_text())['sources'].items() if sha(ROOT/name)!=digest}
    if differences!=fixed['historical_midpoint_source_differences']:
        raise ValueError('Unreviewed historical/current detector-source difference')
    return dict(sources_sha256=sha(SOURCES),sources=fixed['sources'],
        frozen_b=frontend_identity(),sigma_cells=1.5,settings=deepcopy(SETTINGS))


def unchanged_functions():
    # AST comparison ignores whitespace, but forbids changing these reused calculations.
    import ast
    names=('seed_all','atomic_save','rng_state','restore_rng','tree_equal','checked_data','capture_cache',
        'dataset_for','infos','tensor_bytes','epoch_batches','batch','update','evaluate',
        'summaries','comparison_checks','support_report','optimizer_for','checkpoint_context','smoke_views')
    def functions(path):
        tree=ast.parse(path.read_text())
        return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    old_source=functions(ROOT/'crane_project/tools/train_port_geometry_midpoint_formal_v1.py')
    copied=functions(Path(__file__))
    # Only the VAL annotation identity reference changes in checked_data; the data are unchanged.
    copied['checked_data']=copied['checked_data'].replace(
        "Call(func=Name(id='frontend_identity', ctx=Load()), args=[], keywords=[])",
        "Attribute(value=Attribute(value=Name(id='g', ctx=Load()), attr='ready', ctx=Load()), attr='FROZEN_B', ctx=Load())")
    if any(old_source[n]!=copied[n] for n in names):
        raise ValueError('Unexpected copied function change')
    return list(names)


def actual_counts(train, val):
    eligible={domain:sum(r['eligible'] and r['domain']==domain for r in train) for domain in ('real','sim')}
    if not all(eligible.values()):raise ValueError('No eligible TRAIN views in one domain')
    present=sum(bool(len(r['boxes_original'])) for r in train)
    steps=math.ceil(max(eligible.values())/4)
    return dict(train_views=len(train),val_views=len(val),eligible_by_domain=eligible,
        missing_train=len(train)-present,missing_val=sum(not len(r['boxes_original']) for r in val),
        midpoint_steps_per_epoch=steps,midpoint_formal_budget=24*steps,
        lvsr_projected_present_views=present,lvsr_projected_steps_per_epoch=math.ceil(present/8),
        lvsr_actual_counts_pending_selected_midpoint=True,lvsr_parent_epoch=None,lvsr_parent_sha256=None,
        lvsr_arms=['fixed_b'],continuation_arms=['C'],continuation_order_offset=20)


def check_cfg():
    from mmcv import Config
    base=json.loads(BASE_CONTRACT.read_text());cfg=Config.fromfile(str(ROOT/base['config']))
    if (cfg.model.bbox_head.loss_cls.use_geometry_weight is not False or
            cfg.model.bbox_head.loss_bbox.type!='SymKLDLoss' or
            cfg.model.train_cfg.assigner.type!='SymPOLAAssigner' or cfg.load_from is not None or cfg.resume_from is not None):
        raise ValueError('New110 model/config differs')
    return cfg


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
        head_digest=g.state_digest(head), frozen_b=frontend_identity(), sigma_cells=1.5,
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
        expected=(frontend_identity()['annotation_sha256'] if split=='val' else
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
    torch.cuda.set_per_process_memory_fraction(min(1.,6000*2**20/torch.cuda.get_device_properties(gpu).total_memory),gpu)
    torch.cuda.reset_peak_memory_stats(gpu)
    return dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        mmcv=mmcv.__version__,mmdet=mmdet.__version__,mmrotate=mmrotate.__version__,
        opencv=cv2.__version__,gpu=torch.cuda.get_device_name(gpu))


def resource_check(gpu):
    """Same 6,000MiB allocator/6,800MiB process stop, explicit device only."""
    import subprocess
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'],text=True,timeout=10)
    own=sum(int(parts[1].strip()) for line in text.splitlines()
        for parts in [line.split(',')] if len(parts)==2 and parts[0].strip()==str(os.getpid()))
    allocated=torch.cuda.max_memory_allocated(gpu)/2**20
    if own>=6800 or allocated>6000:
        raise ValueError('RESOURCE_STOP: new110 midpoint memory cap')
    return dict(process_polled_mib=own,allocator_peak_mib=allocated,
        process_stop_mib=6800,allocator_cap_mib=6000)


def build_detector(cfg, gpu):
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector as build
    from crane_project.tools.diagnose_port_shape_e_h_train_gradients_v1 import config_value, plain
    import ast
    base=json.loads(BASE_CONTRACT.read_text());checkpoint=Path(base['checkpoint'])
    if sha(checkpoint)!=frontend_identity()['checkpoint_sha256']:
        raise ValueError('Requires the sealed new110 ep20 checkpoint')
    import_modules_from_strings(**cfg.custom_imports)
    spec=deepcopy(cfg.model);spec.pretrained=None;spec.train_cfg=None
    detector=build(spec);loaded=load_checkpoint(detector,str(checkpoint),map_location='cpu',strict=True)
    fields=('model','data','optimizer','optimizer_config','lr_config','runner','load_from','resume_from')
    parsed={}
    for node in ast.parse(loaded['meta'].get('config','')).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in fields:
            name=node.targets[0].id
            if name in parsed:raise ValueError('Duplicate checkpoint config field')
            parsed[name]=config_value(node.value)
    meta=loaded['meta']
    if (parsed!=plain({k:cfg.to_dict()[k] for k in fields}) or meta.get('seed')!=0 or
            meta.get('epoch')!=20 or meta.get('iter')!=12800):
        raise ValueError('New110 checkpoint metadata differs')
    if sha(checkpoint)!=frontend_identity()['checkpoint_sha256']:
        raise ValueError('New110 checkpoint changed during strict loading')
    detector.new110_load_receipt=dict(strict=True,missing_keys=[],unexpected_keys=[],
        checkpoint_sha256=sha(checkpoint),epoch=20,iteration=12800,seed=0)
    del loaded
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
    manifest['strict_load']=detector.new110_load_receipt
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
            resource_check(gpu)
            if len(tensors['boxes_original']) and not bool((tensors['support'].sum((1,2,3))>0).all()):
                raise ValueError('Natural output has no ROI support; do not filter by GT')
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
    manifest['status']='COMPLETE_FROZEN_110_TRAIN_VAL_CACHE'
    g.write_new(directory/'cache_manifest.json',manifest)
    return manifest


def load_cache(directory, identity):
    directory=Path(directory); manifest=json.loads((directory/'cache_manifest.json').read_text())
    if (manifest['protocol']!=VERSION or manifest['status']!='COMPLETE_FROZEN_110_TRAIN_VAL_CACHE' or manifest['identity']!=identity or
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
            if bool(((r['support']<0)|(r['support']>1)).any()) or (n and not bool((r['support'].sum((1,2,3))>0).all())):
                raise ValueError('Invalid or zero natural ROI support')
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
            payload['frozen_b']!=frontend_identity() or payload.get('sigma_cells')!=1.5 or
            payload['context']['manifest_sha256']!=selection['cache_manifest_sha256']):
        raise ValueError('Selected formal checkpoint metadata differs')
    cfg=check_cfg(); detector=build_detector(cfg,gpu)
    if g.state_digest(detector)!=payload['context']['detector_state']:
        raise ValueError('Inference B state differs from formal training')
    head=new_head().to('cuda:'+str(gpu)); head.load_state_dict(payload['head_state'],strict=True)
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
    frozen=new_head().to('cuda:'+str(gpu))
    frozen.load_state_dict(head.state_dict(),strict=True)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    dataset=dataset_for(cfg,'val',1.)
    lookup={Path(v['filename']).stem:i for i,v in enumerate(infos(dataset))}
    selected=smoke_views(val); expected=evaluate(head,selected,'cuda:'+str(gpu))
    for r,reference in zip(selected,expected):
        image,metas=old.load_view(dataset,lookup[r['image']],r,gpu)
        tensors=capture_cache(detector,image,metas)
        for key,tensor in tensors.items():
            if tensor.shape!=r[key].shape or not torch.allclose(tensor,r[key],atol=1e-4,rtol=1e-6):
                raise ValueError('Online/cache tensor differs: '+r['image']+'/'+key)
        with torch.no_grad():
            result=frozen(*[tensors[k].to('cuda:'+str(gpu)) for k in
                ('roi','support','boxes_original','boxes_model','scale_xy')])
        current={method:value[0].cpu().tolist() if len(value) else None for method,value in
            (('b',tensors['boxes_original']),('midpoint',result['boxes_original']))}
        resource_check(gpu)
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
    seed_all(); head=new_head().to(device); optimizer=optimizer_for(head)
    initial=g.state_digest(head); batches=epoch_batches(train,1)
    if any(r['b']!=r['midpoint'] for r in evaluate(head,train+val,device)):
        raise ValueError('Full-cache neutral sigma1.5 output differs from new110')
    logs=[update(head,optimizer,train,batches[i%len(batches)],device) for i in range(2)]
    if logs[0]['output_grad_norm']<=0 or logs[1]['stem_grad_norm']<=0:
        raise ValueError('Initial output/step2 stem gradients ineffective')
    proof=save_checkpoint(out/'discarded_smoke_head.pth',head,optimizer,0,2,identity,checkpoint_context(cache))
    payload=torch.load(str(out/proof['path']),map_location='cpu')
    replay=new_head().to(device); replay.load_state_dict(payload['head_state'],strict=True)
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
        online=online,neutral_views=len(train)+len(val),actual_optimizer_steps=4,
        detector_updates=0,test_access=False,logs=logs)


def train_formal(train, val, cache, identity, device, out, progress):
    seed_all(); head=new_head().to(device); optimizer=optimizer_for(head)
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
    selected_head=new_head().to(device); selected_head.load_state_dict(payload['head_state'],strict=True)
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


def finite_trial(args):
    """Fresh20-view trial only; reuse capture/update/smoke, never full cache."""
    if args.cache_dir:
        raise ValueError('Trial refuses existing cache input; extracts fresh real views')
    identity=checked_sources();runtime_imports()
    from mmcv.parallel import collate,scatter
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=False)
    report=dict(protocol=VERSION,stage='trial',formal_training=False,detector_updates=0,
        test_access=False,old_cache_used=False,formal_cache_reusable=False,
        actual_optimizer_steps=0,identity=identity,samples=[])
    try:
        cfg=check_cfg();report['runtime']=runtime(args.gpu);seed_all()
        detector=build_detector(cfg,args.gpu);before=g.state_digest(detector)
        report['strict_load']=detector.new110_load_receipt
        manifest=json.loads((g.ready.DATA/'manifest.json').read_text())
        selected={r['id']:r for r in manifest['records'] if r['split'] in ('train','train_sim','val')}
        train=[];val=[]
        for split,scale in (('train',1.),('train',.5),('val',1.)):
            dataset=dataset_for(cfg,split,scale);records=infos(dataset)
            chosen=[]
            for domain,seq,n in (('real','real_seq01' if split=='train' else 'real_seq07',4 if split=='train' else 2),
                                 ('sim','sim_seq08' if split=='train' else 'sim_seq10',4 if split=='train' else 2)):
                candidates=sorted([(Path(v['filename']).stem,i) for i,v in enumerate(records)
                    if Path(v['filename']).stem.startswith(seq+'_')])
                if len(candidates)<n:raise ValueError('Insufficient bounded trial samples')
                chosen.extend(candidates[:n])
            for name,index in chosen:
                source=selected[name];annotation=g.ready.DATA/source['annfiles']['path']
                image_path=g.ready.DATA/source['images']['path']
                if (sha(annotation)!=source['annfiles']['sha256'] or sha(image_path)!=source['images']['sha256']):
                    raise ValueError('Trial sample byte identity differs: '+name)
                gt=old.parse_dota_txt(str(annotation))
                if len(gt)!=1:raise ValueError('Expected one grab GT')
                value=scatter(collate([dataset[index]],samples_per_gpu=1),[args.gpu])[0]
                image,metas=value['img'][0],value['img_metas'][0]
                if image.shape!=(1,3,1024,1024) or len(metas)!=1 or Path(metas[0]['filename']).stem!=name or metas[0].get('flip',False):
                    raise ValueError('Trial view identity differs')
                tensors=capture_cache(detector,image,metas)
                truth=torch.tensor(gt[0],dtype=torch.float32).reshape(1,5)
                center=float((tensors['boxes_original'][0,:2]-truth[0,:2]).norm()) if len(tensors['boxes_original']) else None
                seq,fid=name.rsplit('_',1)
                row=dict(image=name,sequence=seq,domain=seq.split('_')[0],frame_id=int(fid),
                    role=split,split=source['split'],scale=scale,gt_original=truth,
                    eligible=split=='train' and center is not None and center<15.,**tensors)
                support=tensors['support']
                if bool(((support<0)|(support>1)).any()) or (len(support) and not bool((support.sum((1,2,3))>0).all())):
                    raise ValueError('Invalid or zero trial support')
                (train if split=='train' else val).append(row)
                report['samples'].append(dict(image=name,role=split,scale=scale,eligible=row['eligible'],
                    output=bool(len(tensors['boxes_original'])),center_error_px=center,
                    image_sha256=source['images']['sha256'],annotation_sha256=source['annfiles']['sha256']))
                resource_check(args.gpu)
            del dataset
        if len(train)!=16 or len(val)!=4:raise ValueError('Trial must remain bounded to20 fresh views')
        if g.state_digest(detector)!=before:raise ValueError('Trial mutated frozen detector')
        g.assert_detector_frozen(detector)
        cache_path=out/'trial_cache.pt'
        payload=dict(protocol='new110_bounded_trial_cache_only',identity=identity,train=train,val=val,
            detector_state=before,formal_cache_reusable=False)
        atomic_save(cache_path,payload)
        got=torch.load(str(cache_path),map_location='cpu')
        if not tree_equal(payload,got):raise ValueError('Trial cache save/reload differs')
        train,val=got['train'],got['val']
        cache=dict(manifest_sha256=sha(cache_path),detector_state=before,runtime=report['runtime'])
        # Keep the original four-step discarded smoke; only the supplied dataset is small.
        report['actual_counts']=actual_counts(train,val)
        original_optimizer=globals()['optimizer_for']
        def counted_optimizer(head):
            optimizer=original_optimizer(head);step=optimizer.step
            def counted_step(*a,**kw):
                result=step(*a,**kw);report['actual_optimizer_steps']+=1
                return result
            optimizer.step=counted_step
            return optimizer
        globals()['optimizer_for']=counted_optimizer
        try:
            outcome=smoke(train,val,cache,identity,'cuda:'+str(args.gpu),out)
        finally:
            globals()['optimizer_for']=original_optimizer
        if report['actual_optimizer_steps']!=4:raise ValueError('Unexpected finite trial update count')
        report.update(outcome)
        if g.state_digest(detector)!=before:raise ValueError('Head updates mutated detector')
        g.assert_detector_frozen(detector)
        report.update(status='BOUNDED_MIDPOINT_TRIAL_PASS',detector_frozen=True,trial_cache_sha256=sha(cache_path),
            feature_extractions=20,native_head_calls=60,formal_training=False,formal_cache_reusable=False,
            resource=resource_check(args.gpu),automatic_next_stage=False)
        g.write_new(out/'completion.json',report)
        print(json.dumps(dict(status=report['status'],views=20,discarded_optimizer_steps=4,
            detector_updates=0,formal_training=False)),flush=True)
        return report
    except BaseException as error:
        report.update(status='BOUNDED_MIDPOINT_TRIAL_FAILED',error=repr(error))
        g.write_new(out/'failure.json',report)
        raise


def checked_formal_trial():
    """Reuse the passed trial without replaying its updates or loading its cache."""
    contract=json.loads(SOURCES.read_text())['formal_run']
    path=ROOT/contract['trial_report']
    if sha(path)!=contract['trial_report_sha256']:
        raise ValueError('Passed bounded trial receipt changed')
    trial=json.loads(path.read_text())
    if (trial['status']!='BOUNDED_MIDPOINT_TRIAL_PASS' or trial['actual_optimizer_steps']!=4 or
            not trial['discarded'] or not trial['detector_frozen'] or trial['old_cache_used'] or
            trial['formal_cache_reusable'] or trial['identity']['frozen_b']!=frontend_identity() or
            not trial['optimizer_continuation_exact'] or not trial['inference_reload_exact']):
        raise ValueError('Requires passed, discarded new110 trial')
    # Only the execution wrapper/tests have changed since the successful trial.
    allowed={str(Path(__file__).resolve().relative_to(ROOT)),'tests/test_symeood_110_midpoint_v1.py'}
    for name,digest in trial['identity']['sources'].items():
        if name not in allowed and sha(ROOT/name)!=digest:
            raise ValueError('Numerical dependency changed after trial: '+name)
    return trial


def formal_run(args):
    """One fresh cache -> one24-epoch midpoint, then stop; no TEST/LVSR/retry."""
    identity=checked_sources();trial=checked_formal_trial()
    contract=json.loads(SOURCES.read_text())['formal_run']
    if args.cache_dir or Path(args.out_dir).resolve()!=ROOT/contract['directory']:
        raise ValueError('Formal requires its new empty directory; old/trial caches forbidden')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or args.gpu!=0 or int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('Formal midpoint requires only physical GPU2/logical0')
    out=Path(args.out_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    runtime_imports()
    def status(stage,**details):
        value=dict(protocol=VERSION,stage=stage,pid=os.getpid(),time=time.time(),physical_gpu=2,
            detector_updates=0,test_access=False,lvsr=False,automatic_retry=False,**details)
        temporary=out/'status.tmp'
        with temporary.open('w') as stream:json.dump(value,stream,indent=2)
        temporary.replace(out/'status.json')
    def progress(value):
        with (out/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(g.json_native(value),allow_nan=False)+'\n')
        if value['stage']=='cache':
            status('cache',shard=value['shard'],done=value['done'],total=value['total'])
            print('CACHE',value['shard'],value['done'],'/',value['total'],flush=True)
        elif value['stage']=='epoch_complete' or (value['stage']=='train' and value['step']%200==0):
            resource_check(args.gpu)
            status('training',epoch=value['epoch'],updates=value['updates'])
    try:
        g.write_new(out/'authorization.json',dict(contract=contract,identity=identity,
            trial_updates_discarded=True,old_caches_used=False,trial_cache_used=False,
            trial_weights_used=False,formal_epochs=24,automatic_next_stage=False))
        status('cache_starting')
        cfg=check_cfg()
        create_cache(out/'roi_cache',cfg,args.gpu,identity,progress)
        if checked_sources()!=identity:raise ValueError('Sources changed during cache generation')
        status('validating_fresh_cache')
        train,val,cache=load_cache(out/'roi_cache',identity)
        cache['manifest_sha256']=sha(out/'roi_cache/cache_manifest.json')
        if cache['runtime']!=trial['runtime']:raise ValueError('Runtime differs from passed GPU2 trial')
        counts=actual_counts(train,val)
        g.write_new(out/'counts.json',counts)
        seed_all();initial=new_head().to('cuda:0')
        if g.state_digest(initial)!=trial['initial_state']:
            raise ValueError('Fresh midpoint initialization differs from passed trial')
        del initial
        dest=out/'train';dest.mkdir()
        status('training',epoch=0,updates=0)
        result=train_formal(train,val,cache,identity,'cuda:0',dest,progress)
        if result['updates']!=counts['midpoint_formal_budget'] or result['epochs_completed']!=24:
            raise ValueError('Formal midpoint update budget differs')
        if checked_sources()!=identity:raise ValueError('Sources changed during training')
        selected=result['selection']['selected_checkpoint']
        source=dest/selected['path']
        if sha(source)!=selected['sha256']:raise ValueError('Selected checkpoint SHA differs')
        payload=torch.load(str(source),map_location='cpu');head=new_head()
        head.load_state_dict(payload['head_state'],strict=True)
        if g.state_digest(head)!=selected['head_digest']:raise ValueError('Selected strict load differs')
        import shutil
        destinations=[out/'selected_backup',Path('/home/omnisky/symeood_model_backups')/
            'symeood_110_retrain_seed0_20261026_v1'/'midpoint']
        copies=[]
        for folder in destinations:
            folder.mkdir(parents=True,exist_ok=False)
            target=folder/'selected_midpoint.pth'
            with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
            if sha(target)!=selected['sha256']:raise ValueError('Midpoint backup SHA differs')
            copies.append(dict(path=str(target),sha256=selected['sha256'],device=target.stat().st_dev))
        if len({x['device'] for x in copies})!=2:raise ValueError('Independent backup filesystems required')
        g.write_new(out/'backup_receipt.json',dict(selected=selected,copies=copies,
            strict_head_loading=True,missing_keys=[],unexpected_keys=[]))
        g.write_new(dest/'completion.json',result)
        g.write_new(out/'completion.json',dict(status='MIDPOINT_FORMAL_COMPLETE_REVIEW_REQUIRED',
            result=result,counts=counts,backups=copies,TEST_run=False,LVSR_run=False,automatic_adoption=False))
        status('complete',epochs=24,updates=result['updates'],selected=selected,
            backups_verified=True,review_required=True)
    except BaseException as error:
        status('stopped_on_error',error=repr(error),retry=False)
        raise


def run(args):
    if args.stage=='formal':
        return formal_run(args)
    if args.stage=='trial':
        return finite_trial(args)
    if args.stage not in ('check','cache','smoke'):
        raise ValueError('Only preparation stages are enabled; formal training awaits authorization')
    if args.stage=='check':
        identity=checked_sources()
        report=dict(protocol=VERSION,status='LOCAL_STATIC_CHECK_PASS_RUNTIME_NOT_RUN',
            identity=identity,settings=SETTINGS,server_access=False,cache_created=False,
            optimizer_updates=0,test_access=False,formal_training=False,
            pending=['GPU strict load','6003 real cached views','ROI/support/online agreement',
                     'discarded finite updates and exact restoration','resource measurement'])
        print(json.dumps(report,ensure_ascii=False,indent=2));return report
    runtime_imports()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=False)
    progress_path=out/'progress.jsonl'; started=time.perf_counter()
    report=dict(protocol=VERSION,stage=args.stage,status='RUNNING',detector_updates=0,test_access=False)
    def progress(value):
        with progress_path.open('a') as stream:
            stream.write(json.dumps(g.json_native(value),ensure_ascii=False,allow_nan=False)+'\n')
        if value.get('stage')=='cache':
            print('FROZEN B cache',value['shard'],value['done'],'/',value['total'],flush=True)
    try:
        identity=checked_sources(); cfg=check_cfg(); report['identity']=identity
        if args.stage=='cache':
            if not args.cache_dir:
                raise ValueError('--cache-dir required')
            report['cache']=create_cache(Path(args.cache_dir),cfg,args.gpu,identity,progress)
            report['status']='FULL_TRAIN_VAL_FROZEN_B_CACHE_COMPLETE_NO_UPDATES'
        else:
            if not args.cache_dir:
                raise ValueError('--cache-dir required')
            rt=runtime(args.gpu); train,val,cache=load_cache(args.cache_dir,identity)
            report['actual_counts']=actual_counts(train,val)
            cache['manifest_sha256']=g.ready.sha(Path(args.cache_dir)/'cache_manifest.json')
            if rt!=cache['runtime']:
                raise ValueError('Training runtime/GPU differs from frozen extraction')
            report['runtime']=rt; device='cuda:'+str(args.gpu)
            if args.stage=='smoke':
                report.update(smoke(train,val,cache,identity,device,out,cfg,args.gpu))
            report['resource']=resource_check(args.gpu)
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
    p.add_argument('--stage',choices=('check','cache','smoke','trial','formal'),required=True)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--cache-dir')
    p.add_argument('--out-dir',required=True)
    return p


if __name__=='__main__':
    if Path.cwd().resolve()!=ROOT:
        raise SystemExit('Run from the symEOOD project root')
    print('New110 sigma1.5 midpoint; formal generates fresh cache and trains only midpoint; no TEST/LVSR.',flush=True)
    run(parser().parse_args())
