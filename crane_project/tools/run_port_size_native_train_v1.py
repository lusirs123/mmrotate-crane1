#!/usr/bin/env python3
"""S031: one paired native/cached P3 run; no TEST/depth or module search.

cache: full fixed TRAIN/VAL, three float32 memmap pairs + shared tensors.
train: mmap batches, discarded Adam/RNG replay, 24 epochs/arm, full VAL each.
Never retains full P3 on disk or a full point dataset on GPU.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
VERSION='port_size_native_train_v1'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def rows_write(p,rows):
    with gzip.open(str(p),'xt') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')


def protocol():
    from crane_project.utils.port_size_native_p3_v1 import protocol as parent
    p=deepcopy(parent());p.update(protocol=VERSION,status='AUTHORIZED_ONE_PAIRED24_EPOCH_RUN',
        parent_protocol='port_size_native_p3_v1',
        qualifications='Real/near +0.001 allows0.1pp cost, not strict nondegradation and not S025 passing. RIoU>1e-5 is numeric threshold only; practical benefit needs size/magnitude/video evidence.',
        cache='All6003 fixed views, no GT filtering,3 mmap float32 point pairs plus shared masks/coords/geometry/modelpoints/inside, exact input provenance; stream batch8.',
        engineering='11+ unit checks, full neutral replay, cache/native nonneutral equivalence,2 discarded Adam updates/arm; exact state/optimizer/RNG and next-update continuation. No TRAIN gate.',
        efficiency_revision='S030 nominal bool is NOT verified. Paired randomized100 A/B repetitions after20 warmups, same frozen view, two blocks; report jitter/CI. If shared load prevents stable inference, mark efficiency unconfirmed, never claim passed.',
        execution='cache then fresh training after functional engineering; resource/nonfinite/mapping failure stops. Unconfirmed shared-GPU latency is not a fabricated engineering pass; efficiency remains separate adoption requirement.',
        limits=dict(epochs_per_arm=24,detector_updates=0,midpoint_updates=0,test_reads=0,depth_reads=0,automatic_promotion=False))
    return p


def checked_sources():
    p=ROOT/'crane_project/tools/port_size_native_train_v1_sources.json';v=read(p)
    if read(ROOT/'crane_project/tools/port_size_native_train_v1_protocol.json')!=protocol():raise ValueError('Protocol drift')
    for n,s in v['sources'].items():
        if sha(ROOT/n)!=s:raise ValueError('Source drift '+n)
    from crane_project.tools import eval_port_size_proposal_downstream_v1 as parent
    return dict(current=v,parent=parent.checked_sources(),native=parent.checked_depth_sources())


def seed(torch):
    random.seed(1703);torch.manual_seed(1703);torch.cuda.manual_seed_all(1703)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def benchmark(n,head,roi,support,b,bm,xy,m,p3,meta,torch):
    """Same warmed input, random A/B order, paired and block uncertainty."""
    import numpy as np
    def f(arm):return head(n.read_points(roi,support,b,bm,xy,m,arm,p3,meta),b,bm,xy,m)
    with torch.no_grad():
        for _ in range(20):
            for arm in n.ARMS:f(arm)
        pairs=[];rng=random.Random(1703)
        for i in range(100):
            order=list(n.ARMS);rng.shuffle(order);v={}
            for arm in order:
                a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();f(arm);z.record();torch.cuda.synchronize();v[arm]=a.elapsed_time(z)
            pairs.append(dict(index=i,order=order,**v,delta_ms=v[n.ARMS[1]]-v[n.ARMS[0]]))
    delta=np.array([x['delta_ms'] for x in pairs]);mu=float(delta.mean());sem=float(delta.std(ddof=1)/10)
    blocks=[float(delta[i:i+50].mean()) for i in (0,50)];ci=[mu-1.984*sem,mu+1.984*sem]
    stable=ci[1]-ci[0]<=5 and abs(blocks[0]-blocks[1])<=5
    return dict(pairs=pairs,mean_delta_ms=mu,median_delta_ms=float(np.median(delta)),p95_delta_ms=float(np.percentile(delta,95)),
        mean_ci95_ms=ci,block_means_ms=blocks,stability_rule='CI width<=5ms and two50-pair block means differ<=5ms',
        sufficiently_stable=stable,verified_latency_budget_pass=bool(stable and ci[1]<=5),
        status='CONFIRMED' if stable else 'UNCONFIRMED_SHARED_LOAD',
        scope='CUDA events read+head, batch1,20 warmups and100 randomized paired repeats; shared GTX1080, not whole-pipeline/deployment throughput')


def build_cache(args,out,torch,proof):
    import numpy as np
    from crane_project.utils import port_size_native_p3_v1 as n
    from crane_project.utils import port_geometry_refine_g_v1 as g
    from crane_project.tools import run_port_geometry_size_distribution_v1 as prior
    from tools.depth.audit_port_midpoint_depth_v1 import load_native
    from crane_project.tools.preflight_port_geometry_g_v1 import fixed_specs
    from mmcv.parallel import collate,scatter
    from mmrotate.datasets import build_dataset
    from mmdet.datasets.dataset_wrappers import ConcatDataset
    cm=args.cache_dir/'cache_manifest.json'
    if sha(cm)!=prior.sealed.CACHE_SHA:raise ValueError('Original cache manifest SHA')
    cache=read(cm);input_sha={str(cm):sha(cm)}
    for p in (args.b_weight,args.m_weight):input_sha[str(p)]=sha(p)
    detector,midpoint,_,helpers,runtime=load_native(args.b_weight,args.m_weight,0)
    cfg=helpers.g.check_cfg();bd=helpers.g.state_digest(detector);md=helpers.g.state_digest(midpoint)
    seed(torch);head=n.CommonPointSizeHead().cuda().eval()
    # Nonzero fixed engineering readout, no optimizer/fitting.
    with torch.no_grad():head.u.weight.fill_(.01);head.v.weight.fill_(.02)
    report=dict(protocol=VERSION,status='BUILDING',runtime=runtime,source_proof=proof,input_sha256=input_sha,
        feature_extractions=0,neutral_outputs_checked=0,missing=0,read_max_abs=0.,logits_reload_max_abs=0.,files={},shards=[],benchmark=None,
        identity_tolerance=dict(box_atol=1e-4,box_rtol=1e-6,roi_atol=1e-4,roi_rtol=1e-6),image_sha_stream=hashlib.sha256())
    for label,split,scale in [('train_s1','train',1.),('train_s05','train',.5),('val_s1','val',1.)]:
        path=args.cache_dir/(label+'.pt');expect=cache['files'][path.name]
        if sha(path)!=expect:raise ValueError('Immutable old shard SHA '+label)
        input_sha[str(path)]=expect;raw=torch.load(str(path),map_location='cpu');records=raw['records']
        if raw['identity']!=cache['identity'] or raw['detector_state']!=cache['detector_state']:raise ValueError('Original shard provenance')
        count=2558 if split=='train' else 887
        if len(records)!=count or len({r['image'] for r in records})!=count:raise ValueError('Shard count')
        if split=='train':dataset=ConcatDataset([build_dataset(s) for s in fixed_specs(cfg,scale)])
        else:
            spec=deepcopy(cfg.data.val);spec.test_mode=True;dataset=build_dataset(spec)
        parts=dataset.datasets if split=='train' else [dataset]
        offset=0;lookup={}
        for part in parts:
            for i,info in enumerate(part.data_infos):lookup[Path(info['filename']).stem]=offset+i
            offset+=len(part)
        if set(lookup)!={r['image'] for r in records}:raise ValueError('Dataset identity substitution')
        mappings={a:np.lib.format.open_memmap(str(out/(label+'_'+a+'.npy')),mode='w+',dtype=np.float32,shape=(count,256,132,9)) for a in n.ARMS}
        shared=dict(masks=torch.zeros(count,1,132,9),local=torch.zeros(count,2,132,9),geometry=torch.zeros(count,4),
                    model_points=torch.zeros(count,1188,2),inside=torch.zeros(count,1,132,9,dtype=torch.bool))
        meta_rows=[];replay_max=0.;start=time.monotonic()
        for i,rec in enumerate(records):
            batch=scatter(collate([dataset[lookup[rec['image']]]],samples_per_gpu=1),[0])[0]
            image,metas=batch['img'][0],batch['img_metas'][0];meta=metas[0]
            if (tuple(image.shape)!=(1,3,1024,1024) or meta.get('flip',False) or Path(meta['filename']).stem!=rec['image'] or list(meta['ori_shape'][:2][::-1])!=rec['image_size']):raise ValueError('Native view substitution')
            ish=sha(meta['filename'])
            if ish!=rec['image_sha256']:raise ValueError('Image SHA changed')
            report['image_sha_stream'].update((rec['image']+str(scale)+ish).encode())
            with torch.no_grad():
                fs=detector.extract_feat(image);report['feature_extractions']+=1
                if any(x.requires_grad or x.grad_fn is not None for x in fs):raise ValueError('Detector graph retained')
                raw_b=image.new_tensor(helpers.g.flatten_prediction(detector.simple_test_from_features(fs,metas,rescale=False)))
                b=image.new_tensor(helpers.g.flatten_prediction(detector.simple_test_from_features(fs,metas,rescale=True)));bm=raw_b[:,:5]
                xy=b.new_tensor(meta['scale_factor'][:2]).reshape(1,2).expand(len(b),-1)
                for actual,ref in ((b,rec['boxes_original']),(bm,rec['boxes_model'])):
                    if actual.shape!=ref.shape or not torch.allclose(actual.cpu(),ref,atol=1e-4,rtol=1e-6):raise ValueError('Native B cache replay '+rec['image'])
                if not torch.equal(b[:,5].cpu(),rec['boxes_original'][:,5]) or not torch.equal(xy.cpu(),rec['scale_xy']):raise ValueError('Score/resize identity')
                roi,sup,_=g.sample_local(fs[0],bm,meta,'aligned')
                for actual,ref in ((roi,rec['roi']),(sup,rec['support'])):
                    if not torch.allclose(actual.cpu(),ref,atol=1e-4,rtol=1e-6):raise ValueError('ROI replay '+rec['image'])
                if len(b):replay_max=max(replay_max,float((roi.cpu()-rec['roi']).abs().max()))
                # Use the ORIGINAL frozen ROI/M exactly, then validate fresh
                # native geometry against it before storing either point arm.
                oldroi,oldsup=rec['roi'].cuda(),rec['support'].cuda()
                oldb,oldbm,oldxy=[rec[k].cuda() for k in ('boxes_original','boxes_model','scale_xy')]
                m=midpoint(oldroi,oldsup,oldb,oldbm,oldxy)['boxes_original']
                fresh_m=midpoint(roi,sup,b,bm,xy)['boxes_original']
                if not torch.allclose(fresh_m,m,atol=1e-4,rtol=1e-6):raise ValueError('Midpoint original-cache identity')
                reads={a:n.read_points(oldroi,oldsup,oldb,oldbm,oldxy,m,a,fs[0],meta) for a in n.ARMS}
                for key in shared:
                    if not torch.equal(reads[n.ARMS[0]][key],reads[n.ARMS[1]][key]):raise ValueError('Paired common tensors differ')
                present=bool(len(b))
                row={k:rec[k] for k in ('image','sequence','domain','frame_id','scale','split','image_size','image_sha256','annotation_sha256')}
                row.update(shard=label,offset=i,present=present,gt=rec['gt_original'][0].tolist(),
                    b=oldb[0].cpu().tolist() if present else None,bm=oldbm[0].cpu().tolist() if present else None,
                    midpoint=m[0].cpu().tolist() if present else None,scale_xy=oldxy[0].cpu().tolist() if present else [float(v) for v in meta['scale_factor'][:2]],
                    img_shape=list(meta['img_shape']),ori_shape=list(meta['ori_shape']))
                for arm in n.ARMS:
                    if present:mappings[arm][i]=reads[arm]['samples'][0].cpu().numpy()
                    else:mappings[arm][i]=0
                if present:
                    for key in shared:shared[key][i]=reads[n.ARMS[0]][key][0].cpu()
                    # Cache roundtrip vs actual native forward, nonzero weights.
                    restored={k:v.detach().cpu().cuda() for k,v in reads['native_p3'].items()}
                    restored['samples']=torch.from_numpy(np.array(mappings['native_p3'][i:i+1],copy=True)).cuda()
                    a=head(reads['native_p3'],oldb,oldbm,oldxy,m);z=head(restored,oldb,oldbm,oldxy,m)
                    if not torch.equal(a['bucket_logits'],z['bucket_logits']) or not torch.equal(a['boxes_original'],z['boxes_original']):raise ValueError('Point-cache/native replay differs')
                    report['neutral_outputs_checked']+=1
                    if report['benchmark'] is None:
                        report['benchmark']=benchmark(n,head,oldroi,oldsup,oldb,oldbm,oldxy,m,fs[0],meta,torch)
                        write(out/'efficiency.json',report['benchmark'])
                else:report['missing']+=1
                meta_rows.append(row)
            if (i+1)%100==0 or i==0 or i+1==count:
                print(json.dumps(dict(stage='cache',shard=label,done=i+1,total=count,elapsed_s=time.monotonic()-start)),flush=True)
            del batch,image,metas,fs,raw_b,b,bm,xy,roi,sup,oldroi,oldsup,oldb,oldbm,oldxy,m,fresh_m,reads
        for arm,mapping in mappings.items():
            mapping.flush();report['files'][label+'_'+arm+'.npy']=sha(out/(label+'_'+arm+'.npy'))
        del mappings
        torch.save(shared,str(out/(label+'_shared.pt')));write(out/(label+'_records.json'),meta_rows)
        for suffix in ('_shared.pt','_records.json'):report['files'][label+suffix]=sha(out/(label+suffix))
        report['shards'].append(dict(label=label,count=count,present=sum(r['present'] for r in meta_rows),roi_replay_max_abs=replay_max))
        del raw,records,shared,meta_rows,dataset,lookup;gc.collect()
    if report['feature_extractions']!=6003 or report['neutral_outputs_checked']!=5998 or report['missing']!=5:raise ValueError('Full cache counts differ')
    if bd!=helpers.g.state_digest(detector) or md!=helpers.g.state_digest(midpoint):raise ValueError('Frozen model mutated')
    if any(sha(p)!=s for p,s in input_sha.items()):raise ValueError('Input drift')
    report['image_sha_stream']=report['image_sha_stream'].hexdigest();report['status']='COMPLETE'
    report['disk_bytes']=sum((out/p).stat().st_size for p in report['files'])
    report['allocator_peak_mib']=torch.cuda.max_memory_allocated()/2**20
    write(out/'cache_manifest.json',report)
    del detector,midpoint,head;gc.collect();torch.cuda.empty_cache()


class PointStore:
    def __init__(self,path,torch,verify=True):
        import numpy as np
        self.path=Path(path);self.torch=torch;self.meta=read(self.path/'cache_manifest.json')
        if self.meta['status']!='COMPLETE':raise ValueError('Incomplete point cache')
        if verify and any(sha(self.path/n)!=s for n,s in self.meta['files'].items()):raise ValueError('Point cache SHA drift')
        self.shared={};self.maps={};self.rows=[]
        for label in ('train_s1','train_s05','val_s1'):
            self.shared[label]=torch.load(str(self.path/(label+'_shared.pt')),map_location='cpu')
            for arm in ('cached_p3','native_p3'):self.maps[label,arm]=np.load(str(self.path/(label+'_'+arm+'.npy')),mmap_mode='r')
            self.rows.extend(read(self.path/(label+'_records.json')))
        self.train=sorted([r for r in self.rows if r['split']=='train'],key=lambda r:(r['scale'],r['image']))
        self.val=sorted([r for r in self.rows if r['split']=='val'],key=lambda r:r['image'])
        if len(self.train)!=5116 or len(self.val)!=887:raise ValueError('Point row split identity')

    def batch(self,rows,arm,device):
        import numpy as np
        torch=self.torch
        if any(not r['present'] for r in rows):raise ValueError('Missing view cannot enter loss')
        samples=np.stack([self.maps[r['shard'],arm][r['offset']] for r in rows])
        values={'samples':torch.from_numpy(samples).to(device)}
        for key in ('masks','local','geometry'):
            values[key]=torch.stack([self.shared[r['shard']][key][r['offset']] for r in rows]).to(device)
        b,bm,xy,m,gt=[torch.tensor([r[k] for r in rows],dtype=torch.float32,device=device) for k in ('b','bm','scale_xy','midpoint','gt')]
        return values,b,bm,xy,m,gt


def near_joint(rows):
    from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
    groups=dict(overall=rows)
    groups.update({d:[r for r in rows if r['domain']==d] for d in ('real','sim')})
    groups.update({s:[r for r in rows if r['sequence']==s] for s in sorted({r['sequence'] for r in rows})})
    if rows and rows[0]['split']=='train':
        groups.update({d+'/'+str(sc):[r for r in rows if r['domain']==d and r['scale']==sc] for d in ('real','sim') for sc in (1.,.5)})
    out={}
    for key,rs in groups.items():
        pp=[r for r in rs if r['midpoint'] is not None]
        errors=lambda r,k:[sorted(r[k][2:4],reverse=True)[j]/sorted(r['gt'][2:4],reverse=True)[j]-1 for j in (0,1)]
        count=dict(both_improved=0,long_only=0,short_only=0,both_worsened=0,tied_or_mixed=0,wrong_edge_fixed_near_other_harmed=0)
        for r in pp:
            a,c=errors(r,'midpoint'),errors(r,'edge_residual');change=[abs(c[j])-abs(a[j]) for j in (0,1)]
            label='both_improved' if max(change)<-1e-12 else 'both_worsened' if min(change)>1e-12 else 'long_only' if change[0]<-1e-12 and change[1]>1e-12 else 'short_only' if change[1]<-1e-12 and change[0]>1e-12 else 'tied_or_mixed'
            count[label]+=1
            count['wrong_edge_fixed_near_other_harmed']+=int(any(abs(a[j])>.03 and change[j]<-1e-12 and abs(a[1-j])<=.03 and change[1-j]>1e-12 for j in (0,1)))
        item=dict(patterns=count,near={})
        for j,e in enumerate(('long','short')):
            subset=[r for r in pp if abs(errors(r,'midpoint')[j])<=.03]
            before=[abs(errors(r,'midpoint')[j]) for r in subset];after=[abs(errors(r,'edge_residual')[j]) for r in subset]
            item['near'][e]=dict(count=len(subset),before_mae=sum(before)/len(before) if before else None,
                after_mae=sum(after)/len(after) if after else None,harmed=sum(b>a+1e-12 for a,b in zip(before,after)),improved=sum(b<a-1e-12 for a,b in zip(before,after)))
        item['loss_by_fixed_M_single_edge_group']={}
        for j,e in enumerate(('long','short')):
            for name in ('near','need_enlarge','need_shrink'):
                subset=[r for r in pp if ('near' if abs(errors(r,'midpoint')[j])<=.03 else 'need_enlarge' if errors(r,'midpoint')[j]<0 else 'need_shrink')==name and r.get('loss_edges_canonical') is not None]
                item['loss_by_fixed_M_single_edge_group'][e+'/'+name]=dict(count=len(subset),
                    unweighted_mean_loss=sum(r['loss_edges_canonical'][j] for r in subset)/len(subset) if subset else None,weight=1.)
        both=[r for r in pp if max(map(abs,errors(r,'midpoint')))<=.03]
        item['both_near']=dict(count=len(both),methods={k:metrics.size_summary(both,k) for k in metrics.METHODS})
        out[key]=item
    return out


def evaluate(head,store,rows,arm,torch):
    from crane_project.utils import port_size_native_p3_v1 as n
    head.eval();output=[]
    with torch.no_grad():
        present=[r for r in rows if r['present']];pred={}
        for start in range(0,len(present),8):
            rr=present[start:start+8];ts=store.batch(rr,arm,'cuda:0');value=head(*ts[:-1])
            target=n.old.residual.size_targets(ts[-1],ts[-2],ts[2])['log_residual']
            losses=torch.nn.functional.smooth_l1_loss(value['delta_roi'],target,beta=.1,reduction='none')
            swap=(ts[2][:,2]<ts[2][:,3]) ^ (ts[-2][:,2]<ts[-2][:,3])
            losses=torch.where(swap[:,None],losses.flip(-1),losses)
            for j,r in enumerate(rr):
                pred[r['image'],r['scale']]=dict(edge_residual=value['boxes_original'][j].cpu().tolist(),
                    delta_roi=value['delta_roi'][j].cpu().tolist(),
                    loss_edges_canonical=losses[j].cpu().tolist(),
                    size_delivery='neutral' if bool(value['neutral_bypass'][j]) else 'square' if bool(value['square_bypass'][j]) else 'accepted' if bool(value['accepted'][j]) else 'fallback',
                    failed_checks=[k for k,v in value['checks'].items() if not bool(v[j])])
        for r in rows:
            v={k:r[k] for k in ('image','sequence','domain','frame_id','scale','split','shard','gt','b','midpoint')}
            v.update(pred.get((r['image'],r['scale']),dict(edge_residual=None,delta_roi=None,loss_edges_canonical=None,size_delivery='no_output',failed_checks=[])))
            from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity
            if not frozen_identity(v):raise ValueError('M output identity changed')
            output.append(v)
    return output


def update(head,opt,batch,torch):
    from crane_project.utils import port_size_native_p3_v1 as n
    head.train();opt.zero_grad();value=head(*batch[:-1])
    loss=n.old.residual.size_loss(value['delta_roi'],batch[-1],batch[-2],batch[2])
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in head.parameters()):raise ValueError('Missing/nonfinite gradient')
    stem=math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in head.stem.parameters()))
    terminal=math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in list(head.u.parameters())+list(head.v.parameters())))
    norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),10.));opt.step()
    if any(not bool(torch.isfinite(p).all()) for p in head.parameters()):raise ValueError('Nonfinite parameter')
    return dict(loss=float(loss.detach()),stem_grad_norm=stem,terminal_grad_norm=terminal,grad_norm=norm)


def rng_get(torch):return dict(python=random.getstate(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def rng_set(x,torch):
    random.setstate(x['python']);torch.set_rng_state(x['cpu'])
    if x['cuda']:torch.cuda.set_rng_state_all(x['cuda'])
def equal(a,b,torch):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k],torch) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y,torch) for x,y in zip(a,b))
    return a==b


def smoke(store,out,torch):
    from crane_project.utils import port_size_native_p3_v1 as n
    results={};present=[r for r in store.train if r['present']]
    for arm in n.ARMS:
        seed(torch);h=n.CommonPointSizeHead().cuda();opt=torch.optim.Adam(h.parameters(),lr=.001)
        initial=deepcopy(h.state_dict());batch=store.batch(present[:8],arm,'cuda:0')
        with torch.no_grad():assert torch.equal(h(*batch[:-1])['boxes_original'],batch[-2])
        first=update(h,opt,batch,torch);assert first['stem_grad_norm']==0 and first['terminal_grad_norm']>0
        snap=dict(head=deepcopy(h.state_dict()),opt=deepcopy(opt.state_dict()),rng=rng_get(torch))
        stream=io.BytesIO();torch.save(snap,stream);stream.seek(0);got=torch.load(stream)
        clone=n.CommonPointSizeHead().cuda();clone.load_state_dict(got['head']);op=torch.optim.Adam(clone.parameters(),lr=.001);op.load_state_dict(got['opt'])
        assert equal(snap['head'],clone.state_dict(),torch) and equal(snap['opt'],op.state_dict(),torch) and equal(snap['rng'],got['rng'],torch)
        second=store.batch(present[8:16],arm,'cuda:0');rng_set(snap['rng'],torch);a=update(h,opt,second,torch)
        state=rng_get(torch);rng_set(got['rng'],torch);b=update(clone,op,second,torch)
        assert a['stem_grad_norm']>0 and equal(h.state_dict(),clone.state_dict(),torch) and equal(opt.state_dict(),op.state_dict(),torch) and equal(state,rng_get(torch),torch)
        with torch.no_grad():assert torch.equal(h(*second[:-1])['bucket_logits'],clone(*second[:-1])['bucket_logits'])
        results[arm]=dict(first=first,second=a,loss_gradient_adam_rng_next_update_exact=True,discarded=True)
        del h,opt,clone,op,batch,second,got,snap
    seed(torch);torch.cuda.empty_cache();write(out/'smoke.json',results)
    return results


def eligibility(summary,diag):
    checks={}
    for d in ('real','sim'):
        m,c=[summary[d][k] for k in ('midpoint','edge_residual')]
        for e in ('long','short'):
            a,b=c['sizes']['edges'][e]['mae'],m['sizes']['edges'][e]['mae']
            checks[d+'/'+e]=a<b-1e-6 if d=='sim' else a<=b+.001
            v=diag[d]['near'][e];checks[d+'/near_'+e]=v['after_mae']<=v['before_mae']+.001
        checks[d+'/riou']=c['mean_riou_full_frame']>=m['mean_riou_full_frame']-.001
        checks[d+'/identity']=summary[d]['paired']['frozen_identity'] and summary[d]['paired']['new_output']==summary[d]['paired']['lost_output']==0
        a,b=m['temporal'][d],c['temporal'][d]
        checks[d+'/dfr']=b['dfr_percent_per_frame']<=a['dfr_percent_per_frame']
        checks[d+'/mcml']=b['mcml_max']<=a['mcml_max'] and b['mcml_mean']<=a['mcml_mean']
        checks[d+'/tdr']=b['tdr_w10']['denominator']==a['tdr_w10']['denominator'] and b['tdr_w10']['numerator']>=a['tdr_w10']['numerator']
        checks[d+'/aci']=b['aci']==a['aci'] and b['aci_adjacent_pairs']==a['aci_adjacent_pairs']
    for k in ('a','r'):checks['sim/'+k]=summary['sim']['edge_residual']['sizes'][k]['abs_mae']<=summary['sim']['midpoint']['sizes'][k]['abs_mae']+1e-6
    checks['overall/riou']=summary['overall']['edge_residual']['mean_riou_full_frame']>=summary['overall']['midpoint']['mean_riou_full_frame']-1e-5
    return dict(pass_all=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v])


def ranking(value,epoch):
    return [value[d]['edge_residual']['sizes']['edges'][e]['mae'] for d in ('sim','real') for e in ('long','short')]+[epoch]


def checkpoint(path,h,opt,epoch,arm,proof,torch):
    saved_rng=rng_get(torch)
    value=dict(protocol=VERSION,arm=arm,epoch=epoch,updates=epoch*639,proof=proof,
        head_state={k:v.detach().cpu().clone() for k,v in h.state_dict().items()},optimizer_state=deepcopy(opt.state_dict()),rng=saved_rng)
    with path.open('xb') as f:torch.save(value,f)
    got=torch.load(str(path),map_location='cpu')
    if not equal(value,got,torch):raise ValueError('Checkpoint exact replay failed')
    rng_set(saved_rng,torch)
    return sha(path)


def train(args,out,torch,proof):
    from crane_project.utils import port_size_native_p3_v1 as n
    from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
    from crane_project.tools.run_port_geometry_size_condition_v1 import components
    store=PointStore(args.point_cache,torch)
    if store.meta['source_proof']['current']!=proof['current']:raise ValueError('Cache/training source identity differs')
    smoke(store,out,torch)
    report=dict(protocol=VERSION,status='RUNNING',arms={},point_cache_sha256=sha(args.point_cache/'cache_manifest.json'),
        formal_head_updates=0,test_reads=0,depth_reads=0,source_proof=proof,efficiency=store.meta['benchmark'])
    for arm in n.ARMS:
        dest=out/arm;dest.mkdir();seed(torch);head=n.CommonPointSizeHead().cuda();opt=torch.optim.Adam(head.parameters(),lr=.001,weight_decay=0.)
        init_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in head.state_dict().values())).hexdigest()
        train_rows=[r for r in store.train if r['present']];events=[]
        zero_val=evaluate(head,store,store.val,arm,torch)
        if any(r['edge_residual']!=r['midpoint'] for r in zero_val):raise ValueError('Full VAL zero identity')
        zero_train=evaluate(head,store,store.train,arm,torch)
        if any(r['edge_residual']!=r['midpoint'] for r in zero_train):raise ValueError('Full TRAIN zero identity')
        if arm==n.ARMS[0]:
            rows_write(out/'baseline_val.rows.jsonl.gz',zero_val);rows_write(out/'baseline_train.rows.jsonl.gz',zero_train)
            baseline=metrics.summary(zero_val)
            for d,expected in [('real',(374,199,.082202,.084058)),('sim',(512,464,.043216,.043450))]:
                s=baseline[d]['midpoint']['sizes']
                if s['outputs']!=expected[0] or s['joint10']!=expected[1] or any(abs(s['edges'][e]['mae']-v)>1e-6 for e,v in zip(('long','short'),expected[2:])):raise ValueError('Fixed formal M VAL reference not reproduced')
            write(out/'baseline_val.summary.json',dict(metrics=baseline,components=components(zero_val),diagnostics=near_joint(zero_val)))
        for epoch in range(1,25):
            began=time.monotonic();ids=list(range(len(train_rows)));random.Random(1703+epoch).shuffle(ids)
            schedule_sha=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest();loss_sum=0.;count=0
            for start in range(0,len(ids),8):
                rr=[train_rows[j] for j in ids[start:start+8]];batch=store.batch(rr,arm,'cuda:0')
                step=update(head,opt,batch,torch);loss_sum+=step['loss']*len(rr);count+=len(rr);report['formal_head_updates']+=1
                if start%800==0:print(json.dumps(dict(stage='train',arm=arm,epoch=epoch,view=start,total=len(ids),loss=step['loss'])),flush=True)
            val=evaluate(head,store,store.val,arm,torch);summary=metrics.summary(val);diag=near_joint(val)
            comp=components(val);gate=eligibility(summary,diag)
            # TRAIN is explanatory and never controls full VAL evaluation.
            tr=evaluate(head,store,store.train,arm,torch);train_diag=dict(size_groups=metrics.train_report(tr),joint=near_joint(tr))
            ep=dict(epoch=epoch,selection_key=ranking(summary,epoch),eligibility=gate,
                metrics=summary,components=comp,diagnostics=diag,training_mean_loss=loss_sum/count,
                schedule_sha256=schedule_sha,epoch_seconds=time.monotonic()-began)
            ck=dest/('head_epoch_%02d.pth'%epoch);ep['checkpoint_sha256']=checkpoint(ck,head,opt,epoch,arm,dict(point_cache_sha256=report['point_cache_sha256'],source_commit=args.source_commit),torch)
            rows_write(dest/('epoch_%02d_val.rows.jsonl.gz'%epoch),val);write(dest/('epoch_%02d_val.json'%epoch),ep)
            rows_write(dest/('epoch_%02d_train.rows.jsonl.gz'%epoch),tr);write(dest/('epoch_%02d_train.json'%epoch),train_diag)
            events.append(ep)
            print(json.dumps(dict(stage='epoch_finished',arm=arm,epoch=epoch,eligible=gate['pass_all'],failed=gate['failed'],selection_key=ep['selection_key'],elapsed_s=ep['epoch_seconds'])),flush=True)
            del batch,val,tr,summary,diag,comp,train_diag,zero_val,zero_train
            zero_val=zero_train=None
        selected=[e for e in events if e['eligibility']['pass_all']]
        result=dict(initial_state_digest=init_hash,representative=min(events,key=lambda e:e['selection_key'])['epoch'],
            selected=min(selected,key=lambda e:e['selection_key'])['epoch'] if selected else None,
            eligible_epochs=[e['epoch'] for e in selected],epochs=events)
        report['arms'][arm]=result;write(dest/'selection.json',result)
        del head,opt;torch.cuda.empty_cache()
    if report['formal_head_updates']!=30672:raise ValueError('Formal update budget differs')
    if report['arms'][n.ARMS[0]]['initial_state_digest']!=report['arms'][n.ARMS[1]]['initial_state_digest']:raise ValueError('Paired initialization differs')
    if [e['schedule_sha256'] for e in report['arms'][n.ARMS[0]]['epochs']]!=[e['schedule_sha256'] for e in report['arms'][n.ARMS[1]]['epochs']]:raise ValueError('Paired schedules differ')
    report['status']='COMPLETE';write(out/'training.json',report)


def main(args):
    out=args.out_dir.resolve()
    if out.parent!=(ROOT/'work_dirs'/VERSION).resolve() or out.exists():raise ValueError('Fresh original-root result directory required')
    out.mkdir(parents=True);stop=None;report=dict(protocol=VERSION,status='FAILED',stage=args.stage,started_utc=datetime.now(timezone.utc).isoformat())
    try:
        proof=checked_sources();write(out/'source_identity.json',proof);write(out/'protocol.json',protocol())
        from crane_project.tools.audit_port_size_native_p3_v1 import setup_gpu
        resource,stop,thread=setup_gpu(out);report['resource']=resource
        import torch
        torch.cuda.set_device(0);torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(1350*2**20/torch.cuda.get_device_properties(0).total_memory,0);seed(torch)
        if args.stage=='cache':build_cache(args,out,torch,proof)
        else:
            if args.point_cache is None:raise ValueError('Point cache required')
            train(args,out,torch,proof)
        if checked_sources()!=proof:raise ValueError('End source drift')
        report.update(status='COMPLETE',allocator_peak_mib=torch.cuda.max_memory_allocated()/2**20,reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20)
    except Exception as e:
        report['error']=type(e).__name__+': '+str(e);raise
    finally:
        if stop is not None:stop.set();thread.join(timeout=11)
        report['ended_utc']=datetime.now(timezone.utc).isoformat();write(out/'finished.json',report)
        # Large point files are already SHA-bound in cache manifest; do not
        # hash13GiB twice here. Result-only artifacts exclude maps by extension.
        write(out/'artifacts.json',dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.npy'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('cache','train'),required=True)
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--point-cache',type=Path)
    p.add_argument('--cache-dir',type=Path,default=ROOT/'work_dirs/port_results/geometry/port_geometry_midpoint_formal_v1_roi_cache')
    p.add_argument('--b-weight',type=Path,default=ROOT/'work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth')
    p.add_argument('--m-weight',type=Path,default=ROOT/'work_dirs/crane_symeood_k1_port_day2night_midpoint_sigma_v1/sigma_1p5/head_epoch_03.pth')
    main(p.parse_args())
