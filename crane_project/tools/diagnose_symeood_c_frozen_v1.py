"""Frozen C diagnosis: TRAIN/VAL only, no optimizer, no checkpoint writes.

Independent observation hooks leave the model's training/inference methods
unchanged. Stages: prepare -> engineer -> freeze -> run. Raw evidence stays
under the requested output directory; existing files are never deleted.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GROUPS = ('100', '101', '110', '111')
SEED = 20261024


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def plain(x):
    import numpy as np
    import torch
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)):
        return [plain(v) for v in x]
    return x


def write(path, data):
    with Path(path).open('x') as f:
        json.dump(plain(data), f, indent=2, allow_nan=False)


def append(path, data):
    with Path(path).open('a') as f:
        f.write(json.dumps(plain(data), allow_nan=False)+'\n')


def imports():
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    import mmrotate.datasets.crane_custom_dota
    import mmrotate.datasets.pipelines.port_train_augment
    import mmrotate.models.detectors.sym_eood_detector
    import mmrotate.models.dense_heads.sym_eood_head


def config(group):
    from mmcv import Config
    return Config.fromfile(str(ROOT/'work_dirs/symeood_ablation_v1'/group/
                               ('crane_symeood_k1_ablation_'+group+'_v1.py')))


def dataset_infer(cfg, split):
    from mmdet.datasets import build_dataset
    if split == 'val':
        dc = copy.deepcopy(cfg.data.val)
    else:
        dc = copy.deepcopy(cfg.data.train)
        for item in dc:
            item['pipeline'] = copy.deepcopy(cfg.data.val.pipeline)
            item['test_mode'] = True
    return build_dataset(dc)


def dataset_items(ds):
    children = ds.datasets if hasattr(ds,'datasets') else [ds]
    items = []
    for child in children:
        for i,info in enumerate(child.data_infos):
            frame = Path(info['filename']).stem
            assert frame.startswith(('real_', 'sim_'))
            image = Path(child.img_prefix)/info['filename']
            ann = Path(child.ann_file)/(frame+'.txt')
            items.append((child,i,frame,image,ann))
    return items


def batch_to_device(data):
    import torch
    if torch.is_tensor(data):
        return data.cuda()
    if isinstance(data,list):
        return [batch_to_device(x) for x in data]
    if isinstance(data,tuple):
        return tuple(batch_to_device(x) for x in data)
    if isinstance(data,dict):
        return {k:(v if k=='img_metas' else batch_to_device(v)) for k,v in data.items()}
    return data


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_sha(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build(group):
    import torch
    from mmdet.models import build_detector
    from mmcv.runner import load_checkpoint
    cfg=config(group)
    model=build_detector(copy.deepcopy(cfg.model)).cuda()
    load_checkpoint(model,str(ROOT/'work_dirs/symeood_ablation_v1'/group/'epoch_24.pth'),map_location='cpu',strict=True)
    assert type(model.bbox_head).__name__=='SymEOODHead'
    assert not model.bbox_head.use_score_context_modulation
    assert not model.bbox_head.filter_padding_anchors
    assert not model.bbox_head.loss_cls.isolate_images
    assert not model.bbox_head.loss_cls.use_quality_target
    assert model.bbox_head.loss_cls.use_geometry_weight==(group[-1]=='1')
    assert model.bbox_head.loss_cls.loss_weight==.25
    assert all(p.grad is None for p in model.parameters())
    return model,cfg


def prepare(out):
    import torch
    import numpy as np
    from mmdet.datasets import build_dataset
    from mmcv.parallel import collate
    out.mkdir(parents=True,exist_ok=False)
    (out/'inputs').mkdir()
    cfg=config('100')
    sources={}
    manifests=[]
    for batch in ('first4','last4'):
        m=ROOT/'work_dirs/symeood_ablation_v1'/('launch_'+batch+'_20261023')/'manifest.json'
        manifests.append(json.loads(m.read_text())['sha256'])
    # Do not import or read TEST data/outputs: only source manifests and these
    # four explicit TRAIN/VAL config/weight identities are inputs.
    for name in manifests[0]:
        if name.startswith(('mmrotate/','configs/','crane_project/configs/')):
            assert manifests[0][name]==manifests[1][name]
            assert sha(ROOT/name)==manifests[0][name],name
            sources[name]=manifests[0][name]
    identity={}
    for g in GROUPS:
        wd=ROOT/'work_dirs/symeood_ablation_v1'/g
        sweep=json.loads((wd/'ckpt_sweep/sweep_results.json').read_text())
        e=sweep['all_checkpoints']['epoch_24']
        weight=wd/'epoch_24.pth'
        assert sha(weight)==e['checkpoint_sha256']
        assert sha(e['results_pkl'])==e['results_pkl_sha256']
        cfgpath=wd/('crane_symeood_k1_ablation_'+g+'_v1.py')
        identity[g]=dict(weight=str(weight),weight_sha256=sha(weight),config=str(cfgpath),config_sha256=sha(cfgpath),
                         val_pkl=e['results_pkl'],val_pkl_sha256=sha(e['results_pkl']))
    ids={}
    inventory=[]
    for split in ('train','val'):
        ds=dataset_infer(cfg,split)
        items=dataset_items(ds)
        assert len(items)==(2558 if split=='train' else 887)
        ids[split]=[r[2] for r in items]
        for child,i,frame,image,ann in items:
            assert len(child.get_ann_info(i)['bboxes'])==1
            inventory.append(dict(split=split,frame=frame,image=str(image),image_sha256=sha(image),annotation=str(ann),annotation_sha256=sha(ann)))
    ds=build_dataset(copy.deepcopy(cfg.data.train))
    items=dataset_items(ds)
    rng=random.Random(SEED)
    chosen=[]
    for domain in ('real','sim'):
        pool=[i for i,r in enumerate(items) if r[2].startswith(domain+'_')]
        chosen.extend(rng.sample(pool,min(128,len(pool))))
    rng.shuffle(chosen)
    assert len(chosen)==256 and len(set(chosen))==256
    views=[]
    for b in range(128):
        samples=[];records=[]
        for slot,index in enumerate(chosen[2*b:2*b+2]):
            child,i,frame,_,_=items[index]
            seed=SEED+index*17
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            trace=[]
            original=child.pipeline.transforms
            class Trace:
                def __init__(self,t):self.t=t
                def __call__(self,result):
                    value=self.t(result)
                    if value is not None:
                        trace.append(dict(transform=type(self.t).__name__,**{
                            k:plain(value[k]) for k in ('img_shape','pad_shape','scale_factor','flip','flip_direction','port_shrink_scale') if k in value}))
                    return value
            child.pipeline.transforms=[Trace(t) for t in original]
            try:sample=child[i]
            finally:child.pipeline.transforms=original
            assert Path(sample['img_metas'].data['filename']).stem==frame
            samples.append(sample)
            records.append(dict(frame=frame,index=index,slot=slot,seed=seed,trace=trace,
                                image_tensor_sha256=tensor_sha(sample['img'].data),
                                gt=plain(sample['gt_bboxes'].data),meta=plain(sample['img_metas'].data)))
        collated=collate(samples,samples_per_gpu=2)
        data={k:v.data[0] for k,v in collated.items()}
        assert data['img'].shape[0]==2
        path=out/'inputs'/('batch_%03d.pt'%b)
        torch.save(data,str(path))
        views.append(dict(batch=b,path=str(path),sha256=sha(path),shape=list(data['img'].shape),samples=records))
    write(out/'draft.json',dict(groups=identity,sources=sources,seed=SEED,data_inventory=inventory,
                                candidate_ids=ids,training_views=views,
                                semantics=dict(batch_per_device=2,historical_recovery=False,
                                               BN='all BatchNorm eval; original backbone norm_eval=True',
                                               mature_assign_count=8000,mature_nfl_count=5000,
                                               tie_kld_atol=1e-6,tie_kld_rtol=1e-5,
                                               tie_distance_atol=1e-4,tie_distance_rtol=1e-6,
                                               strength='one batch scalar alpha=stopgrad(L_G/L_U)',
                                               near_zero_loss=1e-12,clip_norm=10.,
                                               candidate_sequence=['score>0.05','global top1'],NMS=False,nms_pre=False)))
    print('PREPARED',out,flush=True)


def check_identity(out):
    d=json.loads((out/'draft.json').read_text())
    for p,h in d['sources'].items():assert sha(ROOT/p)==h,p
    for g,r in d['groups'].items():
        for p,h in (('weight','weight_sha256'),('config','config_sha256'),('val_pkl','val_pkl_sha256')):
            assert sha(r[p])==r[h],(g,p)
    return d


class CandidateReader:
    def __init__(self,model,destination):
        self.head=model.bbox_head
        self.original=self.head._get_bboxes_single
        self.destination=destination
        destination.mkdir(parents=True,exist_ok=False)
        self.context=None
        self.rows=[]
        self.last=None
        self.head._get_bboxes_single=types.MethodType(self.observe,self.head)

    def observe(self,head,cls,bbox,anchors,img_shape,scale,cfg,rescale=False,with_nms=False):
        import torch
        import numpy as np
        from mmrotate.core.bbox.iou_calculators import rbbox_overlaps
        result=self.original(cls,bbox,anchors,img_shape,scale,cfg,rescale,with_nms)
        assert rescale and not with_nms
        cfg=head.test_cfg if cfg is None else cfg
        assert cfg.score_thr==.05 and cfg.max_per_img==1
        frame,gt=self.context
        boxes=[];scores=[];levels=[];local=[]
        for level,(cs,bp,an) in enumerate(zip(cls,bbox,anchors)):
            score=cs.permute(1,2,0).reshape(-1,head.cls_out_channels).sigmoid().max(1)[0]
            delta=bp.permute(1,2,0).reshape(-1,5)
            box=head.bbox_coder.decode(an,delta,max_shape=img_shape)
            box[:,:4]/=box.new_tensor(scale)
            assert bool(torch.isfinite(box).all()) and bool(torch.isfinite(score).all())
            boxes.append(box);scores.append(score)
            levels.append(torch.full_like(score,level,dtype=torch.int16))
            local.append(torch.arange(len(score),device=score.device,dtype=torch.int32))
        boxes=torch.cat(boxes);scores=torch.cat(scores)
        gt=boxes.new_tensor(gt)
        iou=rbbox_overlaps(boxes,gt).max(1)[0]
        center=(boxes[:,:2]-gt[0,:2]).norm(dim=1)
        theta=boxes[:,4]+(boxes[:,2]<boxes[:,3]).to(boxes.dtype)*math.pi/2
        gt_theta=gt[0,4]+float(gt[0,2]<gt[0,3])*math.pi/2
        angle=((theta-gt_theta+math.pi/2)%math.pi-math.pi/2).abs()*180/math.pi
        passed=scores>.05
        qualified=iou>=.5
        ids=torch.nonzero(passed,as_tuple=False).flatten()
        selected=int(ids[scores[ids].topk(1)[1][0]]) if len(ids) else -1
        if selected>=0:
            expected=torch.cat([boxes[selected],scores[selected,None]])[None]
            assert torch.equal(expected,result[0]),('candidate native mismatch',frame,float((expected-result[0]).abs().max()))
        else:assert len(result[0])==0
        category=('no_qualified' if not bool(qualified.any()) else
                  'score_filter_lost' if not bool((qualified&passed).any()) else
                  'success' if selected>=0 and bool(qualified[selected]) else 'ranking_lost')
        np.savez_compressed(self.destination/(frame+'.npz'),box=boxes.cpu().numpy(),score=scores.cpu().numpy(),
                            level=torch.cat(levels).cpu().numpy(),anchor_id=torch.cat(local).cpu().numpy(),
                            iou=iou.cpu().numpy(),center=center.cpu().numpy(),angle=angle.cpu().numpy(),
                            passed=passed.cpu().numpy(),selected=np.int64(selected))
        row=dict(frame=frame,category=category,candidates=len(boxes),qualified=int(qualified.sum()),
                 passed=int(passed.sum()),qualified_passed=int((qualified&passed).sum()),
                 selected=selected,selected_iou=float(iou[selected]) if selected>=0 else None,
                 selected_center=float(center[selected]) if selected>=0 else None,
                 selected_angle=float(angle[selected]) if selected>=0 else None,
                 highest_iou=float(iou.max()),NMS=False,nms_pre=False,
                 candidate_file_sha256=sha(self.destination/(frame+'.npz')))
        self.last=result[0].detach().cpu().numpy()
        self.rows.append(row)
        append(self.destination.parent/(self.destination.name+'_summary.jsonl'),row)
        return result

    def close(self):self.head._get_bboxes_single=self.original


def candidate_run(model,cfg,out,split,group,limit=None):
    import numpy as np
    import torch
    from mmcv.parallel import collate,scatter
    ds=dataset_infer(cfg,split)
    items=dataset_items(ds)
    directory=out/('candidates_'+split)
    observer=CandidateReader(model,directory)
    gold=None
    if split=='val':
        sweep=json.loads((ROOT/'work_dirs/symeood_ablation_v1'/group/'ckpt_sweep/sweep_results.json').read_text())
        with open(sweep['all_checkpoints']['epoch_24']['results_pkl'],'rb') as f:gold=pickle.load(f)
        assert len(gold)==len(items)
    model.eval()
    input_hasher=hashlib.sha256()
    with torch.no_grad():
        for index,(child,i,frame,image,ann) in enumerate(items):
            if limit is not None and index>=limit:break
            data=scatter(collate([ds[index]],samples_per_gpu=1),[torch.cuda.current_device()])[0]
            input_hasher.update(frame.encode());input_hasher.update(tensor_sha(data['img'][0]).encode())
            observer.context=(frame,child.get_ann_info(i)['bboxes'])
            actual=model(return_loss=False,rescale=True,**data)[0][0]
            assert np.array_equal(actual,observer.last)
            if gold is not None:assert np.array_equal(actual,gold[index][0]),('VAL replay mismatch',frame,actual,gold[index][0])
            if (index+1)%100==0:print('CANDIDATES',out.name,split,index+1,'peakMiB',torch.cuda.max_memory_allocated()/1024**2,flush=True)
    observer.close()
    rec=dict(passed=True,frames=len(observer.rows),VAL_exact=split=='val',input_tensor_sequence_sha256=input_hasher.hexdigest())
    write(out/('candidates_'+split+'_receipt.json'),rec)
    return rec


def tie_category(own,other,atol,rtol):
    """0 own closer; 1 external strictly closer; 2 tie; 3 no own; 4 no other."""
    import torch
    finite_own=torch.isfinite(own);finite_other=torch.isfinite(other)
    tolerance=atol+rtol*torch.maximum(own.abs(),other.abs())
    result=torch.zeros_like(own,dtype=torch.int8)
    result[finite_own&finite_other&(other<own-tolerance)]=1
    result[finite_own&finite_other&((other-own).abs()<=tolerance)]=2
    result[~finite_own]=3
    result[finite_own&~finite_other]=4
    return result


class LossReader:
    def __init__(self,model,destination):
        self.model=model;self.destination=destination
        destination.mkdir(parents=True,exist_ok=False)
        self.loss=model.bbox_head.loss_cls
        self.original=self.loss.forward
        self.records=[];self.features=None;self.scores=None
        self.batch=None
        self.h1=model.neck.register_forward_hook(lambda m,i,o:setattr(self,'features',o))
        self.h2=model.bbox_head.register_forward_hook(lambda m,i,o:setattr(self,'scores',o[0]))
        self.loss.forward=types.MethodType(self.observe,self.loss)

    def observe(self,loss,logits,labels,pred_bboxes,gt_bboxes,weight=None,avg_factor=None,**kwargs):
        import torch
        import numpy as np
        import torch.nn.functional as F
        from mmdet.models.losses.utils import weight_reduce_loss
        native=self.original(logits,labels,pred_bboxes,gt_bboxes,weight,avg_factor,**kwargs)
        assert loss._get_current_tau()==1. and loss.training
        level=len(self.records)
        count=len(logits);per_image=count//2
        assert per_image*2==count and logits.shape[1]==1
        image_ids=torch.arange(2,device=logits.device).repeat_interleave(per_image)
        pos=(labels>=0)&(labels<1)
        gt_images=image_ids[pos]
        assert len(gt_images)==len(gt_bboxes)
        mu=logits.new_ones(count)
        minimum=logits.new_full((count,),float('nan'))
        ref=torch.full((count,),-1,device=logits.device,dtype=torch.int32)
        cross=torch.full((count,),-1,device=logits.device,dtype=torch.int8)
        cross_distance=cross.clone()
        top=torch.zeros(count,device=logits.device,dtype=torch.bool)
        own_mu=mu.clone()
        reselected_mu=mu.clone()
        with torch.no_grad():
            if len(gt_bboxes):
                dist=((pred_bboxes[:,None,:2]-gt_bboxes[None,:,:2])**2).sum(-1)
                ids=dist.min(1)[0].topk(min(loss.spatial_topk,count),largest=False)[1]
                from mmrotate.models.losses.sym_kld_calculator import sym_kld
                kld=sym_kld(pred_bboxes[ids,None,:].detach(),gt_bboxes[None].detach(),eps=loss.eps)
                _,indices=kld.min(1)
                mins=loss._min_sym_kld(pred_bboxes[ids].detach(),gt_bboxes.detach())
                torch.testing.assert_close(kld.min(1)[0],mins,rtol=1e-6,atol=1e-7)
                assert bool(torch.isfinite(kld).all())
                mu[ids]=1+torch.exp(-mins.clamp(0,20))
                minimum[ids]=mins;ref[ids]=indices.to(torch.int32);top[ids]=True
                owner=image_ids[ids,None]==gt_images[None]
                inf=torch.full_like(kld,float('inf'))
                self_kld=torch.where(owner,kld,inf).min(1)[0]
                other_kld=torch.where(~owner,kld,inf).min(1)[0]
                cross[ids]=tie_category(self_kld,other_kld,1e-6,1e-5)
                own_mu[ids]=torch.where(torch.isfinite(self_kld),1+torch.exp(-self_kld.clamp(0,20)),torch.ones_like(self_kld))
                own_dist=torch.where(owner,dist[ids],inf).min(1)[0]
                other_dist=torch.where(~owner,dist[ids],inf).min(1)[0]
                cross_distance[ids]=tie_category(own_dist,other_dist,1e-4,1e-6)
                # Offline own-GT reference: reselect top-K within each image,
                # not a repaired model and not used for native losses/output.
                for image_id in (0,1):
                    own=gt_bboxes[gt_images==image_id]
                    ii=torch.nonzero(image_ids==image_id,as_tuple=False).flatten()
                    if not len(own):continue
                    dd=((pred_bboxes[ii,None,:2]-own[None,:,:2])**2).sum(-1).min(1)[0]
                    chosen=ii[dd.topk(min(loss.spatial_topk,len(ii)),largest=False)[1]]
                    vv=loss._min_sym_kld(pred_bboxes[chosen].detach(),own.detach())
                    reselected_mu[chosen]=1+torch.exp(-vv.clamp(0,20))
        sigmoid=logits.sigmoid()
        target=F.one_hot(labels,num_classes=2)[:,:-1].float()
        pt=(1-sigmoid)*target+sigmoid*(1-target)
        focal_weight=(loss.alpha*target+(1-loss.alpha)*(1-target))*pt.pow(loss.gamma)
        focal=F.binary_cross_entropy_with_logits(logits,target,reduction='none')*focal_weight
        shaped_weight=weight.view(-1,1) if weight is not None else None
        U=weight_reduce_loss(focal,shaped_weight,loss.reduction,avg_factor)*loss.loss_weight
        G=weight_reduce_loss(focal*mu[:,None],shaped_weight,loss.reduction,avg_factor)*loss.loss_weight
        reference=G if loss.use_geometry_weight else U
        assert torch.equal(reference,native),('native loss replication',level,float(reference),float(native))
        if loss.use_geometry_weight:G=native
        else:U=native
        valid=weight>0 if weight is not None else torch.ones_like(labels,dtype=torch.bool)
        category=torch.where(valid,torch.where(pos,0,1),2)
        contribution=focal[:,0]*(weight if weight is not None else 1)*loss.loss_weight/avg_factor
        rec=dict(level=level,U=U,G=G,category=category,image_ids=image_ids,
                 summary=dict(level=level,candidates=count,positive=int((category==0).sum()),negative=int((category==1).sum()),ignored=int((category==2).sum()),
                              gt_count=len(gt_bboxes),empty_gt=not len(gt_bboxes),tau=loss._get_current_tau(),avg_factor=avg_factor,
                              mu_own_mean_abs_delta=float((mu-own_mu).abs().mean()),mu_own_changed=int(((mu-own_mu).abs()>1e-6).sum()),
                              mu_mean=float(mu.mean()),mu_min=float(mu.min()),mu_max=float(mu.max())))
        for name,c in [('positive',0),('negative',1),('ignored',2)]:
            mask=category==c
            rec['summary'][name+'_topk']=int((mask&top).sum())
            rec['summary'][name+'_loss_U']=float(contribution[mask].sum().detach())
            rec['summary'][name+'_loss_G']=float((contribution[mask]*mu[mask]).sum().detach())
        for name,value in [('kld',cross),('distance',cross_distance)]:
            for cat in range(5):rec['summary'][name+'_reference_'+str(cat)]=int((value==cat).sum())
        top_mu=mu[top]
        rec['summary']['mu_top_near1']=int((top_mu<1.1).sum())
        rec['summary']['mu_top_near2']=int((top_mu>=1.9).sum())
        rec['summary']['mu_top_middle']=int(((top_mu>=1.1)&(top_mu<1.9)).sum())
        np.savez_compressed(self.destination/('batch_%03d_level%d.npz'%(self.batch,level)),
                            labels=labels.cpu().numpy(),label_weight=weight.detach().cpu().numpy(),
                            image_ids=image_ids.cpu().numpy(),anchor_id=np.tile(np.arange(per_image,dtype=np.int32),2),
                            topk=top.cpu().numpy(),mu=mu.cpu().numpy(),min_kld=minimum.cpu().numpy(),
                            reference_gt_index=ref.cpu().numpy(),reference_gt_image=gt_images.cpu().numpy(),
                            assigned_gt=np.where(pos.cpu().numpy(),0,-1).astype(np.int8),
                            cross_kld=cross.cpu().numpy(),cross_distance=cross_distance.cpu().numpy(),
                            own_reference_mu_same_topk=own_mu.cpu().numpy(),perimage_reselected_mu=reselected_mu.cpu().numpy(),
                            focal_raw=focal.detach().cpu().numpy(),loss_U=contribution.detach().cpu().numpy(),
                            loss_G=(contribution*mu).detach().cpu().numpy())
        self.records.append(rec)
        return native

    def close(self):
        self.loss.forward=self.original
        self.h1.remove();self.h2.remove()


def metrics(vectors_a,vectors_b):
    import torch
    aa=bb=ab=0.
    for a,b in zip(vectors_a,vectors_b):
        if a is not None:aa+=float(a.double().square().sum())
        if b is not None:bb+=float(b.double().square().sum())
        if a is not None and b is not None:ab+=float((a.double()*b.double()).sum())
    return dict(norm_a=math.sqrt(aa),norm_b=math.sqrt(bb),ratio=math.sqrt(aa/bb) if bb>1e-24 else None,
                cosine=max(-1.,min(1.,ab/math.sqrt(aa*bb))) if aa*bb>1e-24 else None)


def grad_run(model,cfg,out,views,limit=None):
    import torch
    reader=LossReader(model,out/'loss_samples')
    model.train()
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    initial_buffers={n:b.detach().clone() for n,b in model.named_buffers()}
    initial_counter=model.bbox_head.assigner._local_call_count
    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    for view in views[:limit]:
        b=view['batch']
        assert sha(view['path'])==view['sha256']
        data=batch_to_device(torch.load(view['path'],map_location='cpu'))
        reader.batch=b;reader.records=[];reader.features=None;reader.scores=None
        model.bbox_head.assigner._local_call_count=8000
        model.bbox_head.loss_cls._local_iter.fill_(5000)
        losses=model.forward_train(**data)
        assert len(reader.records)==5
        U=sum(r['U'] for r in reader.records);G=sum(r['G'] for r in reader.records)
        if float(U)<=1e-12:
            append(out/'gradient_summary.jsonl',dict(batch=b,frames=[r['frame'] for r in view['samples']],
                    U=float(U.detach()),G=float(G.detach()),S=None,alpha=None,strength_degenerate=True,
                    levels=[r['summary'] for r in reader.records]))
            with torch.no_grad():
                for name,buf in model.named_buffers():buf.copy_(initial_buffers[name])
            model.bbox_head.assigner._local_call_count=initial_counter
            reader.records=[];reader.features=None;reader.scores=None
            del losses,U,G,data
            continue
        alpha=(G/U).detach();S=alpha*U
        assert 1-1e-6<=float(alpha)<=2+1e-6
        torch.testing.assert_close(S.detach(),G.detach(),rtol=1e-6,atol=1e-8)
        def total_value(v):return sum(v) if isinstance(v,(list,tuple)) else v
        other=sum(total_value(v) for key,v in losses.items() if key!='loss_cls' and 'loss' in key)
        targets=[p for _,p in parameters]+list(reader.features)+list(reader.scores)
        options=dict(retain_graph=True,allow_unused=True)
        gu=torch.autograd.grad(U,targets,**options)
        gg=torch.autograd.grad(G,targets,**options)
        gs=torch.autograd.grad(S,targets,**options)
        if limit is not None:
            gn=torch.autograd.grad(sum(losses['loss_cls']),targets,**options)
            expected=gg if model.bbox_head.loss_cls.use_geometry_weight else gu
            # Native objective and observed logits gradients must be exact.
            # Repeated cuDNN convolution backwards are not bit deterministic
            # under the historical flags; measure rather than mislabel them.
            npar=len(parameters);nfeat=len(reader.features)
            for a,z in zip(gn[npar+nfeat:],expected[npar+nfeat:]):
                assert (a is None)==(z is None)
                if a is not None:torch.testing.assert_close(a,z,rtol=0,atol=0)
            repeat=metrics(gn[:npar],expected[:npar])
            repeat['max_abs_difference']=max(float((a-z).abs().max()) for a,z in zip(gn[:npar],expected[:npar]) if a is not None)
            del gn,expected
        go=torch.autograd.grad(other,[p for _,p in parameters],retain_graph=False,allow_unused=True)
        assert all(g is None or bool(torch.isfinite(g).all()) for g in gu+gg+gs+go)
        n=len(parameters);nf=len(reader.features)
        import numpy as np
        for level in range(5):
            np.savez_compressed(out/'loss_samples'/('batch_%03d_level%d_logits_grad.npz'%(b,level)),
                                U=gu[n+nf+level].cpu().numpy(),G=gg[n+nf+level].cpu().numpy(),S=gs[n+nf+level].cpu().numpy())
        spaces=dict(logits=list(range(n+nf,len(targets))),FPN_features=list(range(n,n+nf)),
                    cls_head=[i for i,(name,_) in enumerate(parameters) if name.startswith('bbox_head.retina_cls.')],
                    FPN_parameters=[i for i,(name,_) in enumerate(parameters) if name.startswith('neck.')],all_parameters=list(range(n)))
        summary=dict(batch=b,frames=[r['frame'] for r in view['samples']],U=float(U.detach()),G=float(G.detach()),S=float(S.detach()),
                     alpha=float(alpha),strength_degenerate=False,native_is='G' if model.bbox_head.loss_cls.use_geometry_weight else 'U',
                     levels=[r['summary'] for r in reader.records],gradient={},total={},label_partition={})
        if limit is not None:summary['native_backward_repeat']=repeat
        for space,ids in spaces.items():
            summary['gradient'][space]=dict(G_vs_S=metrics([gg[i] for i in ids],[gs[i] for i in ids]),
                                            G_vs_U=metrics([gg[i] for i in ids],[gu[i] for i in ids]))
        for c,name in [(0,'positive'),(1,'negative'),(2,'ignored')]:
            norms={}
            for label,values in [('U',gu),('G',gg),('S',gs)]:
                squared=0.
                for level,rec in enumerate(reader.records):
                    g=values[n+nf+level]
                    if g is not None:
                        flat=g.permute(0,2,3,1).reshape(-1,1)
                        squared+=float(flat[rec['category']==c].double().square().sum())
                norms[label]=math.sqrt(squared)
            summary['label_partition'][name]=norms
        for name,values in [('U',gu),('G',gg),('S',gs)]:
            combined=[]
            for a,other_g in zip(values[:n],go):
                combined.append(other_g if a is None else a if other_g is None else a+other_g)
            norm=metrics(combined,combined)['norm_a']
            summary['total'][name]=dict(preclip_norm=norm,clip_coefficient=min(1.,10/(norm+1e-6)))
        # autograd.grad does not accumulate p.grad. Only diagnostics/counters
        # may have moved; restore all buffers and the nonpersistent counter.
        assert all(p.grad is None for p in model.parameters())
        with torch.no_grad():
            for name,buf in model.named_buffers():buf.copy_(initial_buffers[name])
        model.bbox_head.assigner._local_call_count=initial_counter
        assert all(torch.equal(buf,initial_buffers[name]) for name,buf in model.named_buffers())
        append(out/'gradient_summary.jsonl',summary)
        reader.records=[];reader.features=None;reader.scores=None
        del losses,U,G,S,other,gu,gg,gs,go,targets,data,combined,values,rec
        if (b+1)%8==0:print('GRADIENT',out.name,b+1,'peakMiB',torch.cuda.max_memory_allocated()/1024**2,flush=True)
    reader.close()


def engineer(out,attempt):
    import torch
    draft=check_identity(out)
    result=[]
    for group in GROUPS:
        model,cfg=build(group)
        before=model_sha(model)
        dest=out/('engineering_'+group+'_attempt%d'%attempt)
        dest.mkdir(exist_ok=False)
        candidate_run(model,cfg,dest,'val',group,limit=1)
        grad_run(model,cfg,dest,draft['training_views'],limit=1)
        assert model_sha(model)==before
        result.append(dict(group=group,passed=True,model_state_sha256=before,
                           grad_none=all(p.grad is None for p in model.parameters())))
        del model;torch.cuda.empty_cache()
    write(out/'engineering.json',dict(passed=True,groups=result,attempt=attempt,script_sha256=sha(__file__)))


def freeze(out):
    draft=check_identity(out)
    engineering=json.loads((out/'engineering.json').read_text());assert engineering['passed']
    assert engineering['script_sha256']==sha(__file__)
    for view in draft['training_views']:assert sha(view['path'])==view['sha256']
    write(out/'protocol.json',dict(status='FROZEN',draft_sha256=sha(out/'draft.json'),
                                  engineering_sha256=sha(out/'engineering.json'),script_sha256=sha(__file__),
                                  operation='TRAIN/VAL candidate and gradient observation; zero optimizer updates',
                                  candidate_inputs='complete TRAIN 2558 + VAL 887, exact original inference pipeline',
                                  training_inputs='128 fixed batch2 diagnostic batches; 128 Real + 128 Sim; not historical recovery',
                                  semantics=draft['semantics'],created=time.strftime('%Y-%m-%d %H:%M:%S')))


def run(out):
    import torch
    draft=check_identity(out)
    protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['status']=='FROZEN' and protocol['script_sha256']==sha(__file__)
    assert protocol['draft_sha256']==sha(out/'draft.json')
    inputs={}
    # All four full VAL replays are a hard gate before any TRAIN observation.
    for phase in ('val','train','gradient'):
        for group in GROUPS:
            model,cfg=build(group)
            before=model_sha(model)
            dest=out/group
            dest.mkdir(exist_ok=True)
            print('START',phase,group,flush=True)
            if phase in ('val','train'):
                rec=candidate_run(model,cfg,dest,phase,group)
                if phase in inputs:assert inputs[phase]==rec['input_tensor_sequence_sha256']
                inputs[phase]=rec['input_tensor_sequence_sha256']
            else:grad_run(model,cfg,dest,draft['training_views'])
            assert model_sha(model)==before,'model/buffer changed'
            append(out/'state_checks.jsonl',dict(group=group,phase=phase,before=before,after=model_sha(model),passed=True,
                                                peak_torch_MiB=torch.cuda.max_memory_allocated()/1024**2))
            del model;torch.cuda.empty_cache()
    check_identity(out)
    write(out/'complete.json',dict(passed=True,groups=GROUPS,VAL_replay_frames=3548,candidate_frames=13780,
                                  gradient_batches=512,updates=0,input_digests=inputs,time=time.strftime('%Y-%m-%d %H:%M:%S')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','engineer','freeze','run'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args()
    os.chdir(ROOT)
    imports()
    try:
        if args.stage=='engineer':engineer(args.out.resolve(),args.attempt)
        else:globals()[args.stage](args.out.resolve())
    except Exception as exc:
        if args.out.is_dir():append(args.out/'failures.jsonl',dict(stage=args.stage,error=repr(exc),time=time.strftime('%Y-%m-%d %H:%M:%S')))
        raise
