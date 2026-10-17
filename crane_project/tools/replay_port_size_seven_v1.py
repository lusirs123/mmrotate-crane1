#!/usr/bin/env python3
"""Frozen C20/P07, original complete TRAIN seven-slot replay. No optimizer.

Only train_s1/train_s05 are opened. B/M and images are never inferred, and
VAL/TEST/depth rows are never opened. Natural replay is an engineering gate.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
VERSION='port_size_seven_replay_v1'
PINS=dict(c20='fefbe410d84a3728137d0603ff8b0713bc0dd145fbc3727c4bb4aa114a70e974',
          p07='d1fa5587fb42fe30db6994a14eb2e345cd00c552b1718ba45e5c3f2e8b5ec6fd')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f]


def canonical(p):
    if p is None:return None
    v=list(p[:5])
    if not all(math.isfinite(x) for x in v) or min(v[2:4])<=0:raise ValueError('Invalid box')
    if v[2]<v[3]:v[2],v[3],v[4]=v[3],v[2],v[4]+math.pi/2
    v[4]=(v[4]+math.pi/2)%math.pi-math.pi/2
    return v


def fixed_edges(box,m):
    return [box[3],box[2]] if m[2]<m[3] else list(box[2:4])


def errors(edges,gt):
    g=sorted(gt[2:4],reverse=True)
    e=[v/w-1 for v,w in zip(edges,g)];z=[math.log(v/w) for v,w in zip(edges,g)]
    return dict(relative=e,log_relative=z,a=sum(z)/2,r=z[0]-z[1])


def delivery_status(value,j):
    return 'square' if value['square_bypass'][j] else 'neutral' if value['neutral_bypass'][j] else 'accepted' if value['accepted'][j] else 'fallback'


def compare_saved(actual,reference,label):
    import numpy as np
    if (actual is None)!=(reference is None):raise ValueError('Natural presence mismatch '+label)
    if actual is None:return 0.
    if len(actual)!=len(reference):raise ValueError('Natural shape mismatch '+label)
    atol,rtol=(1e-6,1e-5) if 'delta' in label else (1e-4,1e-6)
    if not np.allclose(actual,reference,atol=atol,rtol=rtol):raise ValueError('Natural numeric mismatch '+label)
    return max(abs(a-b) for a,b in zip(actual,reference))


def load_train(args,t,torch,used):
    cache=read(args.cache_dir/'cache_manifest.json')
    if sha(args.cache_dir/'cache_manifest.json')!=t.sealed.CACHE_SHA:raise ValueError('Cache identity')
    used['cache_manifest']=t.sealed.CACHE_SHA
    baseline=t.old.read_rows(t.checked_file(args.i_run,'baseline_train.rows.jsonl.gz',used))
    if len(baseline)!=5116:raise ValueError('Full TRAIN baseline missing')
    keyed={(r['image'],r['scale']):r for r in baseline}
    if len(keyed)!=5116:raise ValueError('Duplicate TRAIN baseline')
    result=[]
    for label in ('train_s1','train_s05'):
        path=args.cache_dir/(label+'.pt')
        if sha(path)!=cache['files'][path.name]:raise ValueError('TRAIN cache bytes changed')
        used['cache/'+path.name]=cache['files'][path.name]
        raw=t.sealed.torch_load(torch,path)
        if raw['identity']!=cache['identity'] or raw['detector_state']!=cache['detector_state'] or len(raw['records'])!=2558:raise ValueError('TRAIN cache provenance')
        if Counter(r['sequence'] for r in raw['records'])!=Counter(t.sealed.TRAIN_COUNTS):raise ValueError('TRAIN video counts')
        for r in raw['records']:
            src=keyed[(r['image'],r['scale'])];n=len(r['boxes_original'])
            if (r['boxes_original'][0].tolist() if n else None)!=src['b'] or r['gt_original'][0].tolist()!=src['gt']:raise ValueError('Cache B/GT mismatch')
            if any(r[k]!=src[k] for k in ('image','sequence','domain','frame_id','split')):raise ValueError('Cache image identity')
            m=torch.tensor([src['midpoint']],dtype=r['boxes_original'].dtype) if n else r['boxes_original'].clone()
            result.append(dict(r,shard=label,midpoint_original=m))
        del raw
    return sorted(result,key=lambda r:(r['scale'],r['image'])),cache


def serialize_view(rec,m,gt,b,bm,xy,q,delta_roi,delivered,j,slots):
    # delta_roi has model-B ROI order, q has ORIGINAL raw order. Natural M
    # canonical long/short identity stays fixed even when Q/pre-candidate cross.
    result=dict(image=rec['image'],sequence=rec['sequence'],domain=rec['domain'],frame_id=rec['frame_id'],
        source_split=rec['split'],scale=rec['scale'],shard=rec['shard'],gt=gt,b=b,b_model=bm,scale_xy=xy,midpoint=m,
        natural_raw_swap=m[2]<m[3],model_b_swap=bm[2]<bm[3],slots=[])
    for k,name in enumerate(slots):
        pos=j*7+k;d=list(delta_roi[j][k]);raw=list(reversed(d)) if bm[2]<bm[3] else d
        direct=list(q[j][k]);direct[2:4]=[direct[z+2]*math.exp(raw[z]) for z in (0,1)]
        candidate=delivered['candidate_original'][pos];final=delivered['boxes_original'][pos]
        status=delivery_status(delivered,pos)
        if any(final[z]!=m[z] for z in (0,1,4,5)) or (final[2]<final[3])!=(m[2]<m[3]):raise ValueError('Delivery changed protected M identity')
        qs=q[j][k];fixed_q=fixed_edges(qs,m);fixed_pre=fixed_edges(candidate,m)
        result['slots'].append(dict(slot=name,input_raw=qs,input_fixed_edges=fixed_q,input_errors=errors(fixed_q,gt),
            delta_roi=d,delta_raw=raw,delta_fixed=list(reversed(raw)) if m[2]<m[3] else raw,
            pre_direct_raw=direct,pre_delivery_raw=candidate,pre_fixed_edges=fixed_pre,pre_fixed_errors=errors(fixed_pre,gt),
            final_raw=final,final_canonical=canonical(final),final_errors=errors(sorted(final[2:4],reverse=True),gt),
            input_axis_cross=(qs[2]<qs[3])!=(m[2]<m[3]),pre_axis_cross=(candidate[2]<candidate[3])!=(m[2]<m[3]),
            final_axis_cross=(final[2]<final[3])!=(m[2]<m[3]),status=status,whole_pair_fallback=status=='fallback',
            accepted=bool(delivered['accepted'][pos]),neutral_bypass=bool(delivered['neutral_bypass'][pos]),square_bypass=bool(delivered['square_bypass'][pos]),
            checks={n:bool(v[pos]) for n,v in delivered['checks'].items()}))
    return result


def run(args):
    import torch
    from crane_project.tools import run_port_geometry_size_proposal_v1 as t
    out=args.out;out.mkdir(parents=True,exist_ok=False);started=time.monotonic();used={}
    report=dict(protocol=VERSION,status='FAILED',training_updates=0,optimizer_created=False,detector_loaded=False,
        feature_extractions=0,val_rows_read=False,test_read=False,depth_read=False,original_h=.0625,slots=list(t.model.SLOTS))
    samples=[]
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='3' or os.environ.get('WORLD_SIZE','1')!='1':raise ValueError('Fourth physical card only')
        manifest=read(ROOT/'crane_project/tools/port_size_seven_replay_v1_sources.json')
        if any(sha(ROOT/n)!=h for n,h in manifest['sources'].items()):raise ValueError('Replay source drift')
        report['source_manifest_sha256']=sha(ROOT/'crane_project/tools/port_size_seven_replay_v1_sources.json');report['training_sources']=t.check_sources()
        proof=read(t.checked_file(args.proposal_run,'input_runtime_proof.json',used));runtime=proof['runtime']
        torch.set_num_threads(1);torch.cuda.set_device(0);t.old.seed_all(torch)
        torch.backends.cudnn.allow_tf32=runtime['tf32_cudnn'];torch.backends.cuda.matmul.allow_tf32=runtime['tf32_matmul']
        torch.cuda.set_per_process_memory_fraction(1100/(torch.cuda.get_device_properties(0).total_memory/2**20),0)
        samples.append(t.memory())
        current=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(0),tf32_cudnn=torch.backends.cudnn.allow_tf32,tf32_matmul=torch.backends.cuda.matmul.allow_tf32)
        if any(current[k]!=runtime[k] for k in current):raise ValueError('Original training runtime mismatch '+str(current))
        report['runtime']=current
        train,cache=load_train(args,t,torch,used);packed=t.old.pack(train,torch.device('cuda:0'),torch)
        if len(train)!=5116 or len(packed['indices'])!=5112:raise ValueError('Full TRAIN missing')
        report['cache_identity']=cache['identity'];report['views']=len(train);report['present_views']=len(packed['indices'])
        heads={};refs={};maxima={};initial={}
        for public,arm,epoch in [('c20','repeat7',20),('p07','proposal7',7)]:
            name=arm+'/head_epoch_%02d.pth'%epoch;path=t.checked_file(args.proposal_run,name,used)
            if sha(path)!=PINS[public]:raise ValueError('Weight SHA')
            payload=t.sealed.torch_load(torch,path)
            if any(payload[k]!=v for k,v in dict(protocol=t.VERSION,arm=arm,epoch=epoch,updates=epoch*639).items()):raise ValueError('Weight metadata')
            h=t.model.GeometryConditionSizeHead().cuda().eval().requires_grad_(False);h.load_state_dict(payload['head_state'],strict=True)
            if t.sealed.state_digest(h)!=payload['head_digest']:raise ValueError('Head tensor digest')
            heads[public]=h;initial[public]=payload['head_digest'];del payload
            ref=t.old.read_rows(t.checked_file(args.proposal_run,arm+'/train_epoch_%02d.rows.jsonl.gz'%epoch,used))
            if len(ref)!=5116:raise ValueError('Natural reference incomplete')
            for r,x in zip(train,ref):
                if any(r[k]!=x[k] for k in ('image','scale','domain','sequence','frame_id','split')) or r['gt_original'][0].tolist()!=x['gt']:raise ValueError('Reference source order')
                actual=r['midpoint_original'][0].tolist() if len(r['midpoint_original']) else None
                if actual!=x['midpoint']:raise ValueError('Reference M differs')
            refs[public]=ref
            # Complete natural replay BEFORE seven-slot behavior is examined.
            natural=t.old.evaluate_rows(h,train,packed,torch,batch_size=64);mx=dict(box=0.,delta=0.)
            for a,b in zip(natural,ref):
                mx['box']=max(mx['box'],compare_saved(a['edge_residual'],b['edge_residual'],public))
                mx['delta']=max(mx['delta'],compare_saved(a['delta_roi'],b['delta_roi'],public+'/delta'))
                if a['size_delivery']!=b['size_delivery'] or a['failed_checks']!=b['failed_checks']:raise ValueError('Natural protection differs')
            maxima[public]=mx;samples.append(t.memory());print('NATURAL_VERIFIED',public,mx,flush=True)
        report['natural_reference_replay']=maxima;report['weights']=PINS
        seven={}
        for public,h in heads.items():
            values={};mx=dict(box=0.,delta=0.);counters=Counter()
            with torch.no_grad():
                for start in range(0,len(packed['indices']),32):
                    xs=[x[start:start+32] for x in packed['tensors']];q,target=t.model.proposals(xs[-2],xs[-1],xs[3],'proposal7')
                    value=t.model.readout(h,*xs[:-1],q);delivery=t.model.delivery(value,*xs[2:-1],q)
                    # Exact named edge target is zero under fixed natural axis.
                    long_roi=((xs[-2][:,2]<xs[-2][:,3]) ^ (xs[3][:,2]<xs[3][:,3])).long();ids=torch.arange(len(q),device=q.device)
                    if not torch.equal(target[ids,5,long_roi],torch.zeros(len(q),device=q.device)) or not torch.equal(target[ids,6,1-long_roi],torch.zeros(len(q),device=q.device)):raise ValueError('Exact-slot target drift')
                    if not bool(torch.isfinite(value['delta_roi']).all()):raise ValueError('Nonfinite frozen readout')
                    ds={k:delivery[k].detach().cpu().tolist() for k in ('boxes_original','candidate_original','accepted','neutral_bypass','square_bypass')}
                    ds['checks']={k:v.detach().cpu().tolist() for k,v in delivery['checks'].items()}
                    cpu=[x.cpu().tolist() for x in xs];qs=q.cpu().tolist();deltas=value['delta_roi'].cpu().tolist()
                    for j,index in enumerate(packed['indices'][start:start+32]):
                        r=train[index];ref=refs[public][index]
                        item=serialize_view(r,cpu[-2][j],cpu[-1][j],cpu[2][j],cpu[3][j],cpu[4][j],qs,deltas,ds,j,t.model.SLOTS)
                        first=item['slots'][0];mx['box']=max(mx['box'],compare_saved(first['final_raw'],ref['edge_residual'],public+'/7'))
                        mx['delta']=max(mx['delta'],compare_saved(first['delta_roi'],ref['delta_roi'],public+'/7delta'))
                        if first['status']!=ref['size_delivery'] or [k for k,v in first['checks'].items() if not v]!=ref['failed_checks']:raise ValueError('Seven natural delivery mismatch')
                        for v in item['slots']:counters[v['status']]+=1;counters['q_axis_cross']+=int(v['input_axis_cross']);counters['pre_axis_cross']+=int(v['pre_axis_cross'])
                        values[index]=item
                    if start%512==0:samples.append(t.memory());print('SEVEN',public,start,'/',5112,flush=True)
            with gzip.open(out/(public+'.rows.jsonl.gz'),'xt') as f:
                for i,r in enumerate(train):
                    item=values.get(i)
                    if item is None:item=dict(image=r['image'],sequence=r['sequence'],domain=r['domain'],frame_id=r['frame_id'],source_split=r['split'],scale=r['scale'],shard=r['shard'],gt=r['gt_original'][0].tolist(),b=None,b_model=None,scale_xy=None,midpoint=None,slots=[dict(slot=s,status='no_output',input_raw=None,delta_roi=None,pre_delivery_raw=None,final_raw=None,final_canonical=None) for s in t.model.SLOTS])
                    f.write(json.dumps(item,allow_nan=False)+'\n')
            seven[public]=dict(natural_max_difference=mx,counters=dict(counters),slot_count=5116*7,missing_slots=4*7)
            if t.sealed.state_digest(h)!=initial[public] or any(p.grad is not None for p in h.parameters()):raise ValueError('Frozen head changed')
            del values
        # Inputs are immutable through replay, including cache bytes and heads.
        for label in ('train_s1.pt','train_s05.pt'):
            if sha(args.cache_dir/label)!=cache['files'][label]:raise ValueError('Cache changed after replay')
        for public,arm,ep in [('c20','repeat7',20),('p07','proposal7',7)]:
            if sha(args.proposal_run/arm/('head_epoch_%02d.pth'%ep))!=PINS[public]:raise ValueError('Weight changed after replay')
        if any(sha(ROOT/n)!=h for n,h in manifest['sources'].items()) or t.check_sources()!=report['training_sources']:raise ValueError('Replay sources changed during run')
        samples.append(t.memory());report.update(status='FROZEN_TRAIN_SEVEN_REPLAY_COMPLETE',seven=seven,input_files=used,
            state_before=initial,state_after={k:t.sealed.state_digest(h) for k,h in heads.items()},memory_samples=samples,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    except Exception as e:
        report['error']=repr(e);report['status']='RESOURCE_STOP' if 'RESOURCE_STOP' in str(e) else 'ENGINEERING_FAILED';raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;write(out/'completion.json',report)
        write(out/'artifacts.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('cache-dir','i-run','proposal-run','out'):p.add_argument('--'+n,required=True,type=Path)
    run(p.parse_args())
