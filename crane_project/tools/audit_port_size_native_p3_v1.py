#!/usr/bin/env python3
"""Bounded TRAIN-only engineering audit. No optimizer, fitting or VAL stage.

Choose physical GPU2/3 before Torch import. Allocator capped at1350MiB;
independent nvidia-smi watchdog terminates own process at1900MiB (<2GiB).
Only first sorted identity per TRAIN video, both original fixed scales.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
VERSION='port_size_native_p3_v1'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def smi(fields,kind='gpu'):
    s=subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True,timeout=10)
    return [[c.strip() for c in v.split(',')] for v in s.splitlines() if v.strip()]


def setup_gpu(out):
    cards=[r for r in smi('index,uuid,memory.free') if int(r[0]) in (2,3)]
    requested=os.environ.get('CUDA_VISIBLE_DEVICES')
    if requested is not None:
        if requested not in ('2','3'):raise ValueError('Only physical GPU2/3 authorized')
        cards=[r for r in cards if r[0]==requested]
    if not cards:raise RuntimeError('RESOURCE_STOP no authorized GPU')
    card=max(cards,key=lambda r:int(r[2]))
    if int(card[2])<3072:raise RuntimeError('RESOURCE_STOP free<3072MiB')
    os.environ['CUDA_VISIBLE_DEVICES']=card[0]
    state=dict(physical_gpu=int(card[0]),uuid=card[1],free_before_mib=int(card[2]),
               process_peak_polled_mib=0,process_stop_mib=1900,allocator_cap_mib=1350,user_limit_mib=2048)
    stop=threading.Event()
    def watch():
        while not stop.is_set():
            try:
                apps=smi('pid,gpu_uuid,used_memory','compute-apps')
                own=sum(int(r[2]) for r in apps if r[0]==str(os.getpid()) and r[1]==card[1])
                state['process_peak_polled_mib']=max(own,state['process_peak_polled_mib'])
                if own>=1900:
                    write(out/'RESOURCE_STOP.json',dict(state,reason='Own process reached1900MiB safety stop'))
                    os._exit(75)
            except Exception as e:
                # A blind monitor cannot promise the user cap: stop fail closed.
                write(out/'RESOURCE_STOP.json',dict(state,reason='GPU watchdog failed: '+str(e)))
                os._exit(75)
            stop.wait(.2)
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    return state,stop,thread


def footprint(points,h,w):
    cells=set()
    for x,y in points.reshape(-1,2).tolist():
        a,b=math.floor(x/8),math.floor(y/8)
        for dx in (0,1):
            for dy in (0,1):
                if 0<=a+dx<w and 0<=b+dy<h:cells.add((a+dx,b+dy))
    return cells


def linear_evidence(oldpoints,points,h,w,img_hw):
    """Sufficient nullspace witness even when both readers touch same cells.

    Sampling is a fixed linear map per channel. A point source can differ in
    coefficients without adding information if the old81 samples span all
    relevant source cells. Report that case explicitly, not as new evidence.
    """
    import numpy as np
    def weights(pts):
        rows=[]
        for x,y in pts.reshape(-1,2).cpu().tolist():
            x,y=x/8,y/8;a,b=math.floor(x),math.floor(y);u,v=x-a,y-b
            row={}
            for dx,dy,weight in ((0,0,(1-u)*(1-v)),(1,0,u*(1-v)),(0,1,(1-u)*v),(1,1,u*v)):
                ix,iy=a+dx,b+dy
                if 0<=ix<w and 0<=iy<h and ix*8<img_hw[1] and iy*8<img_hw[0] and weight>0:
                    row[(ix,iy)]=weight
            rows.append(row)
        return rows
    left,right=weights(oldpoints),weights(points)
    cells=sorted(set(k for row in left+right for k in row));lookup={v:i for i,v in enumerate(cells)}
    def matrix(rows):
        value=np.zeros((len(rows),len(cells)),dtype=np.float64)
        for i,row in enumerate(rows):
            for k,v in row.items():value[i,lookup[k]]=v
        return value
    a,c=matrix(left),matrix(right)
    _,s,vh=np.linalg.svd(a,full_matrices=False)
    rank=int((s>1e-9).sum());basis=vh[:rank]
    remainder=c-(c@basis.T)@basis
    norm=np.linalg.norm(remainder,axis=1);j=int(norm.argmax())
    vector=remainder[j]/max(float(norm[j]),1e-30)
    invisible=float(np.abs(a@vector).max());response=float(np.abs(c@vector).max())
    return dict(old_rank=rank,union_source_cells=len(cells),relative_singular_tolerance=1e-9,
                old_null_response=invisible,native_response=response,
                witness=bool(invisible<1e-6 and response>1e-3),
                max_outside_old_row_span=float(norm[j]),
                note='Coordinate-linear witness only, not semantic usefulness; no new information if old span is complete.')


def main(args):
    out=args.out_dir.resolve()
    if out.parent!=(ROOT/'work_dirs'/VERSION).resolve() or out.exists():raise ValueError('Fresh result-only directory required')
    out.mkdir(parents=True)
    report=dict(protocol=VERSION,status='FAILED',started_utc=datetime.now(timezone.utc).isoformat(),
        training_updates=0,optimizer_instances=0,val_images=0,test_images=0,depth_images=0,
        feature_extractions=0,checks=[],views=[])
    stop=None
    try:
        source=json.loads((ROOT/'crane_project/tools/port_size_native_p3_v1_sources.json').read_text())
        if any(sha(ROOT/p)!=s for p,s in source['sources'].items()):raise ValueError('Source SHA drift')
        from crane_project.tools import eval_port_size_proposal_downstream_v1 as parent
        provenance=parent.checked_sources(); native_provenance=parent.checked_depth_sources()
        write(out/'source_identity.json',dict(current=source,parent=provenance,native=native_provenance,
            local_source_commit=args.source_commit,server_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip()))
        resource,stop,thread=setup_gpu(out);report['resource']=resource
        import torch
        from crane_project.utils import port_size_native_p3_v1 as n
        from crane_project.utils import port_geometry_refine_g_v1 as g
        from tools.depth.audit_port_midpoint_depth_v1 import load_native
        from crane_project.tools.preflight_port_geometry_g_v1 import fixed_specs
        from mmcv.parallel import collate,scatter
        from mmrotate.datasets import build_dataset
        from mmdet.datasets.dataset_wrappers import ConcatDataset
        torch.cuda.set_per_process_memory_fraction(1350*2**20/torch.cuda.get_device_properties(0).total_memory,0)
        torch.cuda.reset_peak_memory_stats();torch.set_num_threads(1)
        write(out/'protocol.json',n.protocol())
        if json.loads((ROOT/'crane_project/tools/port_size_native_p3_v1_protocol.json').read_text())!=n.protocol():raise ValueError('Contract drift')
        cm=args.cache_dir/'cache_manifest.json'
        from crane_project.tools import run_port_geometry_size_distribution_v1 as prior
        if sha(cm)!=prior.sealed.CACHE_SHA:raise ValueError('Cache identity')
        cache=json.loads(cm.read_text());inputs={str(cm):sha(cm)}
        selected={}
        for label in ('train_s1','train_s05'):
            path=args.cache_dir/(label+'.pt');expected=cache['files'][path.name]
            if sha(path)!=expected:raise ValueError('TRAIN shard changed')
            raw=torch.load(str(path),map_location='cpu');inputs[str(path)]=expected
            if raw['identity']!=cache['identity'] or len(raw['records'])!=2558:raise ValueError('TRAIN identity/count')
            representatives={}
            for r in sorted(raw['records'],key=lambda r:r['image']):representatives.setdefault(r['sequence'],r)
            if set(representatives)!=set(prior.sealed.TRAIN_COUNTS):raise ValueError('TRAIN videos changed')
            selected[label]=[deepcopy(representatives[s]) for s in sorted(representatives)]
            del raw
        for p in (args.b_weight,args.m_weight):inputs[str(p)]=sha(p)
        detector,midpoint,_,helpers,runtime=load_native(args.b_weight,args.m_weight,0)
        report['runtime']=runtime
        cfg=helpers.g.check_cfg()
        detector_before=helpers.g.state_digest(detector);midpoint_before=helpers.g.state_digest(midpoint)
        torch.manual_seed(1703);head=n.CommonPointSizeHead().cuda().eval()
        initial=deepcopy(head.state_dict())
        backbone_shapes=[]
        handle=detector.backbone.register_forward_hook(lambda module,inputs,outputs:backbone_shapes.append([list(x.shape) for x in outputs]))
        def timed(f,repeats=5):
            for _ in range(2):f()
            torch.cuda.synchronize();durations=[]
            for _ in range(repeats):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();v=f();b.record();torch.cuda.synchronize();durations.append(a.elapsed_time(b))
            return v,dict(mean_ms=statistics.mean(durations),min_ms=min(durations),max_ms=max(durations),
                          repeats=repeats,warmups=2,timing='CUDA events, batch1 read+common head; excludes detector/M')
        for label,scale in [('train_s1',1.),('train_s05',.5)]:
            dataset=ConcatDataset([build_dataset(s) for s in fixed_specs(cfg,scale)])
            lookup={Path(info['filename']).stem:offset+i for part,offset in zip(dataset.datasets,(0,len(dataset.datasets[0]))) for i,info in enumerate(part.data_infos)}
            for rec in selected[label]:
                batch=scatter(collate([dataset[lookup[rec['image']]]],samples_per_gpu=1),[0])[0]
                image,metas=batch['img'][0],batch['img_metas'][0];meta=metas[0]
                if sha(meta['filename'])!=rec['image_sha256']:raise ValueError('TRAIN image bytes changed')
                if tuple(image.shape)!=(1,3,1024,1024) or meta.get('flip',False):raise ValueError('View substitution')
                with torch.no_grad():
                    features=detector.extract_feat(image);report['feature_extractions']+=1
                    if any(f.requires_grad or f.grad_fn is not None for f in features):raise ValueError('Frozen graph retained')
                    raw6=image.new_tensor(helpers.g.flatten_prediction(detector.simple_test_from_features(features,metas,rescale=False)));bm=raw6[:,:5]
                    b=image.new_tensor(helpers.g.flatten_prediction(detector.simple_test_from_features(features,metas,rescale=True)))
                    xy=b.new_tensor(meta['scale_factor'][:2]).reshape(1,2).expand(len(b),-1)
                    if not torch.equal(raw6[:,5],b[:,5]):raise ValueError('Native score restoration mismatch')
                    if not torch.allclose(g.map_boxes(bm,meta,inverse=True),b[:,:5],atol=1e-4,rtol=1e-6):raise ValueError('Native coordinate restoration')
                    if len(b)!=len(rec['boxes_original']) or not torch.allclose(b.cpu(),rec['boxes_original'],atol=1e-4,rtol=1e-6):raise ValueError('B cache replay differs')
                    if not torch.equal(b[:,5].cpu(),rec['boxes_original'][:,5]) or not torch.allclose(bm.cpu(),rec['boxes_model'],atol=1e-4,rtol=1e-6) or not torch.equal(xy.cpu(),rec['scale_xy']):raise ValueError('Cached raw axes/score/resize differs')
                    roi,support,oldpoints=g.sample_local(features[0],bm,meta,'aligned')
                    roi_error=float((roi.cpu()-rec['roi']).abs().max()) if len(b) else 0.
                    if not torch.allclose(roi.cpu(),rec['roi'],atol=1e-4,rtol=1e-6) or not torch.allclose(support.cpu(),rec['support'],atol=1e-6,rtol=1e-6):raise ValueError('Original ROI cache replay differs')
                    m=midpoint(roi,support,b,bm,xy)['boxes_original']
                    if not len(m):raise ValueError('Predetermined engineering representative missing output; report, do not substitute')
                    reads={a:n.read_points(roi,support,b,bm,xy,m,a,features[0],meta) for a in n.ARMS}
                    for k in ('masks','local','geometry','model_points','inside'):
                        if not torch.equal(reads[n.ARMS[0]][k],reads[n.ARMS[1]][k]):raise ValueError('Common input mismatch '+k)
                    times={}
                    for arm in n.ARMS:
                        report['whole_run_allocated_peak_mib']=max(report.get('whole_run_allocated_peak_mib',0.),torch.cuda.max_memory_allocated()/2**20)
                        torch.cuda.reset_peak_memory_stats();base_mem=torch.cuda.memory_allocated()
                        def run():return head(n.read_points(roi,support,b,bm,xy,m,arm,features[0],meta),b,bm,xy,m)
                        value,cost=timed(run)
                        cost['incremental_allocator_peak_mib']=(torch.cuda.max_memory_allocated()-base_mem)/2**20
                        times[arm]=cost
                        if not torch.equal(value['boxes_original'],m):raise ValueError('Neutral identity differs')
                    a,c=[reads[k] for k in n.ARMS];inside=c['inside'].expand_as(c['samples'])
                    delta=(c['samples']-a['samples'])[inside]
                    h,w=features[0].shape[-2:];oldcells=footprint(oldpoints,h,w)
                    pts=c['model_points'][c['inside'].reshape(1,-1)].reshape(-1,2)
                    newcells=footprint(pts,h,w);extra=newcells-oldcells
                    # Construct a nullspace witness on the ACTUAL coordinate grid:
                    # perturb a native cell unseen by all old9x9 samples.
                    witness=False;witness_magnitude=0.;pulse=direct=oldpulse=None
                    for cell in sorted(extra):
                        pulse=torch.zeros_like(features[0]);pulse[0,0,cell[1],cell[0]]=1.
                        oldpulse,_,_=g.sample_local(pulse,bm,meta,'aligned')
                        direct=n.read_points(torch.zeros_like(roi),support,b,bm,xy,m,'native_p3',pulse,meta)
                        magnitude=float(direct['samples'].abs().max())
                        if torch.equal(oldpulse,torch.zeros_like(oldpulse)) and magnitude>1e-6:
                            witness=True;witness_magnitude=magnitude;break
                    linear=linear_evidence(oldpoints,pts,h,w,meta['img_shape'][:2])
                    view=dict(image=rec['image'],sequence=rec['sequence'],domain=rec['domain'],scale=scale,
                        original_hw=list(meta['ori_shape'][:2]),model_hw=list(image.shape[-2:]),img_hw=list(meta['img_shape'][:2]),
                        actual_sxsy=xy.cpu().tolist(),b=b.cpu().tolist(),m=m.cpu().tolist(),bm=bm.cpu().tolist(),
                        backbone_shapes=backbone_shapes[-1],fpn_shapes=[list(f.shape) for f in features],
                        roi_cache_max_abs=roi_error,common_masks_exact=True,neutral_M_exact=True,
                        positions_total=c['inside'].numel(),positions_inside=int(c['inside'].sum()),
                        old_roi_source_cells=len(oldcells),native_point_source_cells=len(newcells),extra_source_cells=len(extra),
                        nullspace_witness=witness,witness_response=witness_magnitude,
                        linear_basis=linear,
                        feature_difference_mean_abs=float(delta.abs().mean()),feature_difference_max_abs=float(delta.abs().max()),timings=times,
                        raw_point_bytes=c['samples'].numel()*c['samples'].element_size())
                    report['views'].append(view)
                # Connectivity check uses arbitrary deterministic readout weights,
                # NOT a fitted update. Full initial state restored immediately.
                with torch.no_grad():head.u.weight.fill_(.01);head.v.weight.fill_(.02)
                value=head(reads['native_p3'],b,bm,xy,m)
                grads=torch.autograd.grad(value['delta_roi'].sum(),list(head.parameters()))
                if not all(bool(torch.isfinite(v).all()) for v in grads) or not sum(float(v.abs().sum()) for v in grads[:4])>0:raise ValueError('Size gradient path missing')
                replay=n.CommonPointSizeHead().cuda().eval();replay.load_state_dict(head.state_dict())
                with torch.no_grad():
                    again=replay(reads['native_p3'],b,bm,xy,m)
                    if not torch.equal(again['bucket_logits'],value['bucket_logits']):raise ValueError('Fixed-input nonneutral reload mismatch')
                del replay,again
                head.load_state_dict(initial)
                if any(p.grad is not None for p in detector.parameters()) or any(p.grad is not None for p in midpoint.parameters()):raise ValueError('Frozen weights received gradient')
                if not torch.equal(head(reads['native_p3'],b,bm,xy,m)['boxes_original'],m):raise ValueError('Restored neutral differs')
                print(json.dumps(dict(image=rec['image'],scale=scale,extra_cells=len(extra),witness=witness,linear_witness=linear['witness'],resource=resource)),flush=True)
                del batch,image,metas,features,roi,support,reads,a,c,delta,value,grads,pulse,direct,oldpulse
            del dataset,lookup
        handle.remove()
        if report['feature_extractions']!=12:raise ValueError('Wrong bounded extraction count')
        if detector_before!=helpers.g.state_digest(detector) or midpoint_before!=helpers.g.state_digest(midpoint):raise ValueError('Frozen model mutated')
        for p,s in inputs.items():
            if sha(p)!=s:raise ValueError('Input changed during run')
        # Save only small engineering state/reload proof; no trained checkpoint.
        import io
        stream=io.BytesIO();torch.save(head.state_dict(),stream);stream.seek(0)
        reloaded=n.CommonPointSizeHead().cuda();reloaded.load_state_dict(torch.load(stream,map_location='cuda:0'))
        if any(not torch.equal(v,reloaded.state_dict()[k]) for k,v in head.state_dict().items()):raise ValueError('Reload mismatch')
        report['checks']=['native/cache B and ROI replay','point/raw-axis/sxsy mapping','same coverage/masks/geometry',
            'zero residual exact M','detached B/M, connected size gradients, no updates','state reload exact','frozen before/after digest','input SHA before/after']
        report['input_sha256']=inputs
        report['whole_run_allocated_peak_mib']=max(report.get('whole_run_allocated_peak_mib',0.),torch.cuda.max_memory_allocated()/2**20)
        report['allocator_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
        report['views_with_new_information']=sum(v['linear_basis']['witness'] for v in report['views'])
        report['all_real_views_have_new_evidence']=report['views_with_new_information']==len(report['views'])
        report['evidence_decision']='SUBSET_HAS_NEW_INFORMATION' if report['views_with_new_information'] else 'NO_NEW_INFORMATION_WITNESS_STOP_DESIGN'
        report['status']='ENGINEERING_PASS'
    except Exception as e:
        report['error']=type(e).__name__+': '+str(e)
        raise
    finally:
        if stop is not None:stop.set();thread.join(timeout=11)
        report['ended_utc']=datetime.now(timezone.utc).isoformat();write(out/'engineering.json',report)
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
        write(out/'artifacts.json',dict(protocol=VERSION,files=files))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('out-dir','cache-dir','b-weight','m-weight'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    main(p.parse_args())
