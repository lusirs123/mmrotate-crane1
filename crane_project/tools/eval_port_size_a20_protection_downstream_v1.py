#!/usr/bin/env python3
"""Frozen B/M/A20/C6/R6/P5 native validation, TEST, Unknown-02. Explicit --run only.

No training/selection/refit and no legacy endpoint modification. Import/--help
never construct detector, read images or open a CUDA context.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.utils.port_size_a20_protection_downstream_v1 import (
    VERSION,ARMS,HEADS,PINS,SEQUENCE,protocol,depth,identity,head_metadata,replay_inputs,
    compare_boxes,require_val_proof,detection_report,depth_report,delivery_label)
from crane_project.tools.eval_port_size_proposal_downstream_v1 import (
    write,write_rows,load,rows,native_view,memory_snapshot,checked_depth_sources)


def check_memory(value):
    if value['process_mib']>=6800:raise RuntimeError('RESOURCE_STOP own process >=6800MiB')


def checked_sources():
    path=ROOT/'crane_project/tools/port_size_a20_protection_downstream_v1_sources.json';proof=load(path)
    if proof['protocol']!=VERSION or load(ROOT/'crane_project/tools/port_size_a20_protection_downstream_v1_protocol.json')!=protocol():raise ValueError('Frozen downstream contract drift')
    if any(depth.sha(ROOT/n)!=h for n,h in proof['sources'].items()):raise ValueError('Downstream source drift')
    return dict(manifest_sha256=depth.sha(path),**proof)


def checked_references(a20_folder,protection_folder,cache_dir,load_cache=True):
    import gzip
    import torch
    from crane_project.tools import run_port_geometry_size_distribution_v1 as parent
    sealed=parent.sealed
    a20_folder,protection_folder,cache_dir=map(Path,(a20_folder,protection_folder,cache_dir))
    saved={};proof={}
    for folder,tag in [(a20_folder,'a20'),(protection_folder,'continuation')]:
        index=load(folder/'artifacts.json')['files'];training=load(folder/'training.json');finished=load(folder/'finished.json')
        if training['status']!=finished['status'] or finished['status']!='COMPLETE':raise ValueError('Incomplete fixed fit')
        if tag=='a20':
            if training['protocol']!='port_size_qalign_v1' or training['formal_head_updates']!=30672:raise ValueError('S039 fit identity')
            members=['a20']
        else:
            if training['protocol']!='port_size_a20_protection_train_v1' or training['formal_updates']!=11502:raise ValueError('S043 fit identity')
            members=['c6','r6','p5']
            for public in members:
                arm=HEADS[public]['arm'];info=training['arms'][arm]
                if info['representative']!=HEADS[public]['epoch'] or info['selected'] is not None or info['eligible_epochs']:
                    raise ValueError('Original unadopted representative identity differs')
        wanted=['training.json','finished.json','protocol.json']
        for public in members:
            spec=HEADS[public];arm=spec['arm'];epoch=spec['epoch']
            names=[arm+'/head_epoch_%02d.pth'%epoch,arm+'/epoch_%02d_val.rows.jsonl.gz'%epoch];wanted+=names
            if depth.sha(folder/names[0])!=PINS[public]:raise ValueError('Fixed checkpoint SHA '+public)
            with gzip.open(folder/names[1],'rt') as stream:saved[public]=[json.loads(line) for line in stream]
        for name in wanted:
            if depth.sha(folder/name)!=index[name]:raise ValueError('Indexed reference drift '+name)
        proof[tag]=dict(index_sha256=depth.sha(folder/'artifacts.json'),files={n:index[n] for n in wanted})
    if any(len(v)!=887 for v in saved.values()):raise ValueError('Full saved VAL required')
    refs={}
    for i,a in enumerate(saved['a20']):
        ref=dict(a,delivery={},failed_checks={},vector_valid={})
        for public in HEADS:
            row=saved[public][i]
            if any(a[k]!=row[k] for k in ('image','gt','b','midpoint','scale','sequence','domain','frame_id')):raise ValueError('Paired frozen VAL identity')
            ref[public]=row['edge_residual'];ref['delivery'][public]=row['size_delivery'];ref['failed_checks'][public]=row['failed_checks'];ref['vector_valid'][public]=row.get('vector_valid')
        if a['image'] in refs:raise ValueError('Duplicate frozen VAL frame')
        refs[a['image']]=ref
    cache=load(cache_dir/'cache_manifest.json')
    if depth.sha(cache_dir/'cache_manifest.json')!=sealed.CACHE_SHA or depth.sha(cache_dir/'val_s1.pt')!=cache['files']['val_s1.pt']:raise ValueError('Original VAL cache SHA')
    proof.update(cache_manifest_sha256=sealed.CACHE_SHA,val_shard_sha256=cache['files']['val_s1.pt'])
    records={}
    if load_cache:
        payload=sealed.torch_load(torch,cache_dir/'val_s1.pt')
        if payload['identity']!=cache['identity'] or payload['detector_state']!=cache['detector_state'] or len(payload['records'])!=887:raise ValueError('VAL cache payload identity')
        from crane_project.utils.port_size_qalign_v1 import residual
        for row in payload['records']:
            sealed.checked_record(row,'val_s1',torch,residual.original)
            if row['image'] in records:raise ValueError('Duplicate cache frame')
            ref=refs[row['image']]
            if row['gt_original'][0].tolist()!=ref['gt'] or (row['boxes_original'][0].tolist() if len(row['boxes_original']) else None)!=ref['b']:raise ValueError('Frozen row/cache binding')
            records[row['image']]=row
        if records.keys()!=refs.keys():raise ValueError('VAL cache/reference frames differ')
    return refs,records,proof


def run(args):
    if not args.run:raise ValueError('Explicit --run required for native dataset evaluation')
    # Dataset paths are explicit, source stays outside the result root.
    import torch
    import numpy as np
    from tools.depth import audit_port_midpoint_depth_v1 as native
    from tools import eval_port_frozen_downstream_v1 as legacy
    from crane_project.tools import eval_port_geometry_midpoint_formal_v1 as formal
    from crane_project.tools import eval_port_geometry_midpoint_v1_test as test
    from crane_project.tools import run_port_geometry_size_condition_v1 as condition
    from crane_project.utils.port_size_qalign_v1 import AlignedSizeHead,read_pair
    out=Path(args.out_dir).resolve()
    if out.parent!=(ROOT/'work_dirs'/VERSION).resolve():raise ValueError('Use a fresh result-only directory under work_dirs/'+VERSION)
    out.mkdir(parents=True,exist_ok=False)
    report=dict(protocol=VERSION,status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),
        weights={},training_updates=0,selection=False,formula_refit=False,reliability_filter=False,
        a20_remains_unadopted=True,continuation_remains_unadopted=True,automatic_adoption=False,test_selection=False,depth_selection=False)
    tick=time.monotonic();stop=None;watcher=None
    try:
        if os.environ.get('WORLD_SIZE','1')!='1' or os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3'):
            raise ValueError('One explicit physical GPU required')
        from crane_project.tools.audit_port_size_native_p3_v1 import setup_gpu
        resource,stop,watcher=setup_gpu(out,7000,6800,6000);report['resource']=resource
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(6000*2**20/torch.cuda.get_device_properties(0).total_memory,0)
        torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        for a in ARMS:
            p=Path(getattr(args,a+'_weight')).resolve()
            if depth.sha(p)!=PINS[a]:raise ValueError('Frozen weight SHA differs '+a)
            report['weights'][a]=dict(path=str(p),sha256=PINS[a])
        if depth.sha(ROOT/depth.CALIBRATION)!=depth.CAL_SHA:raise ValueError('Raw-opt SHA drift')
        calibration=load(ROOT/depth.CALIBRATION)
        report['source_identity']=checked_sources()
        from crane_project.tools.run_port_size_qalign_v1 import checked_sources as training_sources
        report['size_source_identity']=training_sources()
        references,cache_records,reference_proof=checked_references(args.a20_run,args.protection_run,args.cache_dir)
        report['frozen_reference']=reference_proof
        report['depth_source_identity']=checked_depth_sources()
        report['calibration_sha256']=depth.CAL_SHA
        report['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip()
        report['script_sha256']=depth.sha(__file__)
        detector,mid,pipeline,helpers,runtime=native.load_native(Path(args.b_weight),Path(args.midpoint_weight),0)
        report['runtime']=runtime
        heads={}
        for a,spec in HEADS.items():
            payload=torch.load(getattr(args,a+'_weight'),map_location='cpu');head_metadata(payload,a)
            h=AlignedSizeHead('fixed_b').cuda(0);h.load_state_dict(payload['head_state'],strict=True);h.eval().requires_grad_(False)
            heads[a]=h
        states=lambda:dict(b=helpers.g.state_digest(detector),midpoint=helpers.g.state_digest(mid),**{a:helpers.g.state_digest(h) for a,h in heads.items()})
        initial=states(); report['state_before']=initial
        observed=[];captured={};intermediates=[];counters=dict(feature_extractions=0,native_head_calls=0,size_head_calls=0)
        def audit(event):
            key={'test_feature_extracted':'feature_extractions','test_native_head_called':'native_head_calls'}[event['stage']]
            counters[key]+=1
        def hook(module,inputs,output):captured.update(inputs=inputs,midpoint=output['boxes_original'])
        handle=mid.register_forward_hook(hook)
        def predict(source):
            image,metas=native_view(pipeline,source,helpers)
            captured.clear()
            result=helpers.capture(detector,mid,image,metas,audit)
            roi,support,b,bm,xy=captured['inputs'];m=captured['midpoint']
            if source.get('split')=='val':
                ref=cache_records[source['image']]
                ref_inputs=[ref[k].to(roi.device) for k in ('roi','support','boxes_original','boxes_model','scale_xy')]
                saved_m=references[source['image']]['midpoint']
                ref_inputs.append(m.new_tensor([saved_m]) if saved_m is not None else m.new_empty(0,6))
                intermediates.append(dict(image=source['image'],**replay_inputs([roi,support,b,bm,xy,m],ref_inputs)))
            result['delivery']={};result['failed_checks']={};result['pre_guard']={};result['vector_valid']={}
            result['native_meta']=dict(original_hw=list(metas[0]['ori_shape'][:2]),image_hw=list(metas[0]['img_shape'][:2]),pad_hw=list(metas[0]['pad_shape'][:2]),scale_xy=[float(v) for v in metas[0]['scale_factor'][:2]])
            pair=read_pair(roi,support,b,bm,xy,m)
            result['support']=dict(common_sum=float(pair['support'].sum()),common_active=int((pair['support']>0).sum()),zero_common=bool(len(m) and not (pair['support']>0).any()),source_sums={a:float(v.sum()) for a,v in pair['source_support'].items()},outside={a:int((~v).sum()) for a,v in pair['inside'].items()})
            result['support']['cache_coordinate_ranges']={a:dict(min_xy=g[0].amin(0).cpu().tolist(),max_xy=g[0].amax(0).cpu().tolist()) if len(m) else None for a,g in pair['source_local'].items()}
            with torch.no_grad():
                for a,h in heads.items():
                    value=h(roi,support,b,bm,xy,m)
                    counters['size_head_calls']+=1
                    result[a]=value['boxes_original'][0].cpu().tolist() if len(m) else None
                    result['pre_guard'][a]=value['candidate_original'][0].cpu().tolist() if len(m) and bool(value['candidate_valid'][0]) else None
                    result['vector_valid'][a]=bool(value['vector_valid'][0]) if len(m) else None
                    result['failed_checks'][a]=[k for k,v in value['checks'].items() if not bool(v[0])] if len(m) else []
                    result['delivery'][a]=delivery_label(value)
                    identity(result['midpoint'],result[a])
            return result
        def stream(sources,directory,is_detection):
            directory.mkdir()
            predictions=[]
            with (directory/'predictions.jsonl').open('x') as f:
                for i,s in enumerate(sources):
                    if i%10==0:
                        mem=memory_snapshot();observed.append(mem);check_memory(mem)
                    if depth.sha(s['image_path'])!=s['sha256']:raise ValueError('Image SHA changed')
                    p=predict(s);p.update(frame_id=s['frame_id'],image_sha256=s['sha256'])
                    if is_detection:p.update(image=s['image'],sequence=s['sequence'],domain=s['domain'],scale=1.,split=s['split'])
                    f.write(json.dumps(p,allow_nan=False)+'\n');f.flush();predictions.append(p)
                    if i==0:
                        mem=memory_snapshot();observed.append(mem);check_memory(mem)
                        print('FIRST_FRAME_MEMORY',mem,flush=True)
                    if i==0 or (i+1)%100==0 or i+1==len(sources):print(directory.name,i+1,'/',len(sources),flush=True)
            return predictions
        def detection_sources(split):
            if split=='val':sources,proof=formal.val_inputs()
            else:sources,proof=test.fixed_test_inputs()
            for s in sources:
                p=helpers.g.ready.DATA/split/'images'/(s['image']+'.jpg')
                s.update(image_path=str(p.resolve()),sha256=depth.sha(p),split=split)
            return sources,proof
        sources,data=detection_sources('val')
        valpred=stream(sources,out/'val',True)
        refs=[references[p['image']] for p in valpred]
        report['val_replay']=compare_boxes(valpred,refs)
        report['val_replay']['intermediate_frames']=len(intermediates)
        require_val_proof(report['val_replay'])
        write_rows(out/'val/intermediate_replay.jsonl',intermediates)
        write(out/'VAL_NATIVE_CHAIN_VERIFIED.json',report['val_replay'])
        for p,s in zip(valpred,sources):p['gt']=np.asarray(s['gt'],dtype=np.float32).tolist()
        write_rows(out/'val/evaluation_rows.jsonl',valpred)
        write(out/'val/summary.json',detection_report(valpred));write(out/'val/input_identity.json',data)
        print('VAL_NATIVE_CHAIN_VERIFIED',flush=True)
        # TEST access only after complete VAL replay succeeded.
        require_val_proof(report['val_replay'])
        sources,data=detection_sources('test')
        ps=stream(sources,out/'test',True)
        for p,s in zip(ps,sources):p['gt']=np.asarray(s['gt'],dtype=np.float32).tolist()
        write_rows(out/'test/evaluation_rows.jsonl',ps)
        write(out/'test/summary.json',detection_report(ps));write(out/'test/input_identity.json',data)
        depth_contract=load(ROOT/'tools/port_frozen_downstream_v1_protocol.json')
        for name in [SEQUENCE]:
            pin=depth_contract['sequences'][name]
            manifest,tp,images=legacy.checked_dataset(args.data_root,name,pin)
            proof=dict(pin,role='exposed_TEST_frozen_description',calibration_sha256=depth.CAL_SHA,camera=manifest['camera'],physical_reference=manifest['obb_reference_geometry'])
            predictions=stream(images,out/name,False)
            # Finished all six prediction streams before parsing metric truth.
            truths=rows(tp)
            legacy.validate_truth(truths,manifest,depth_contract['sequences'][name])
            evaluated,summary=depth_report(predictions,truths,manifest,calibration)
            with (out/name/'depth_rows.jsonl').open('x') as f:
                for r in evaluated:f.write(json.dumps(r,allow_nan=False)+'\n')
            write(out/name/'summary.json',summary);write(out/name/'input_identity.json',proof)
        handle.remove();report['state_after']=states()
        if checked_references(args.a20_run,args.protection_run,args.cache_dir,load_cache=False)[2]!=reference_proof:raise ValueError('Frozen training reference changed')
        report['source_after']=checked_sources()
        if report['source_identity']!=report['source_after']:raise ValueError('Source changed during run')
        if report['depth_source_identity']!=checked_depth_sources():raise ValueError('Depth dependency changed during run')
        if report['size_source_identity']!=training_sources():raise ValueError('Size dependency changed during run')
        for arm,pin in PINS.items():
            if depth.sha(getattr(args,arm+'_weight'))!=pin:raise ValueError('Weight changed during run')
        if depth.sha(ROOT/depth.CALIBRATION)!=depth.CAL_SHA:raise ValueError('Calibration changed during run')
        if initial!=report['state_after']:raise ValueError('Frozen model state changed')
        if counters!=dict(feature_extractions=3424,native_head_calls=10272,size_head_calls=13696):raise ValueError('Native work count differs')
        report.update(status='FROZEN_NATIVE_VAL_TEST_DEPTH_COMPLETE',memory_samples=observed,counters=counters,
            peak_torch_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_torch_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    except Exception as e:
        report.update(status='RESOURCE_STOP' if 'RESOURCE_STOP' in str(e) or isinstance(e,torch.cuda.OutOfMemoryError) else 'FAILED',error=repr(e))
        raise
    finally:
        if stop is not None:stop.set();watcher.join(timeout=11)
        report['elapsed_seconds']=time.monotonic()-tick
        write(out/'completion.json',report)
        files={str(p.relative_to(out)):depth.sha(p) for p in out.rglob('*') if p.is_file()}
        write(out/'artifacts.json',dict(files=files))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true',help='Explicit fixed six-version native VAL -> TEST -> Unknown-02 execution')
    for arm in ARMS:p.add_argument('--'+arm+'-weight',required=True,type=Path)
    for name in ('a20-run','protection-run','cache-dir','out-dir','data-root'):p.add_argument('--'+name,required=True,type=Path)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if not args.run:raise SystemExit('No evaluation executed: explicit --run is required')
    run(args)
