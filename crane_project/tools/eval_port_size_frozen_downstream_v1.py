#!/usr/bin/env python3
"""Fixed B/M/I07/axis15 evaluation; no fitting, selection or depth filtering."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_midpoint_depth_v1 as depth

VERSION = 'port_size_frozen_downstream_v1'
ARMS = ('b', 'midpoint', 'i07', 'axis15')
PINS = dict(b=depth.B_SHA, midpoint=depth.HEAD_SHA,
    i07='affe4795b73cae325f878b15c2d81f942461bd965bae0b3c3ca6a7a944eb0464',
    axis15='7af0022adf214750789dbf71490ec9a07de69e14d89b8167c3d0e62cdc3950db')


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_rows(path, values):
    with Path(path).open('x') as f:
        for r in values:f.write(json.dumps(r,allow_nan=False)+'\n')


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def identity(m, candidate):
    if (m is None) != (candidate is None):
        raise ValueError('Size head changed output presence')
    if m is not None and any(m[k] != candidate[k] for k in (0, 1, 4, 5)):
        raise ValueError('Size head changed center/angle/score')


def pair_depth(rs, before, after):
    common = [r for r in rs if all(r[k]['depth']['z_m'] is not None for k in (before, after))]
    changes = [abs(r[after]['depth']['z_m']-r['truth_z_m'])-
               abs(r[before]['depth']['z_m']-r['truth_z_m']) for r in common]
    return dict(common=len(common), improved=sum(v < -1e-12 for v in changes),
        worsened=sum(v > 1e-12 for v in changes), tied=sum(abs(v) <= 1e-12 for v in changes),
        mean_abs_error_delta_m=math.fsum(changes)/len(changes) if changes else None,
        new_numeric=sum(r[before]['depth']['z_m'] is None and r[after]['depth']['z_m'] is not None for r in rs),
        lost_numeric=sum(r[before]['depth']['z_m'] is not None and r[after]['depth']['z_m'] is None for r in rs))


def memory_snapshot():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    physical = int(os.environ['CUDA_VISIBLE_DEVICES'])
    values = [s.strip().split(', ') for s in raw.splitlines()]
    gpu = next(v for v in values if int(v[0]) == physical)
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits'], text=True)
    own = [s.strip().split(', ') for s in raw.splitlines() if s.strip()]
    usage = sum(int(v[2]) for v in own if int(v[0]) == os.getpid() and v[1] == gpu[1])
    return dict(physical_gpu=physical, free_mib=int(gpu[2]), process_mib=usage)


def check_memory(value):
    if value['process_mib'] > 1536 or value['free_mib'] < 2048:
        raise RuntimeError('RESOURCE_STOP: process>1536MiB or free<2048MiB: '+str(value))


def compare_boxes(current, reference):
    import numpy as np
    if len(current) != len(reference): raise ValueError('VAL length differs')
    maxima = {a:0. for a in ARMS}
    for r, ref in zip(current, reference):
        if r['image'] != ref['image']: raise ValueError('VAL order differs')
        for a in ARMS:
            x, y = r[a], ref[a]
            if (x is None) != (y is None): raise ValueError('VAL presence differs '+r['image'])
            if x is not None:
                maxima[a] = max(maxima[a], max(abs(v-w) for v,w in zip(x[:5],y[:5])))
                if x[5] != y[5] or not np.allclose(x[:5],y[:5],rtol=1e-6,atol=1e-4):
                    raise ValueError('VAL original tolerance mismatch '+r['image']+'/'+a)
        for a in ('i07','axis15'):
            if r['delivery'][a] != ref['delivery'][a]: raise ValueError('VAL delivery decision differs')
    return dict(pass_check=True, frames=len(current), max_box_component_abs_diff=maxima,
                atol=1e-4,rtol=1e-6,score_presence_decisions_exact=True)


def detection_report(rs):
    import numpy as np
    from crane_project.tools import run_port_geometry_size_condition_v1 as condition
    from crane_project.tools import eval_port_geometry_ap_v1 as ap
    result = {}
    for arm in ('i07','axis15'):
        adapted = [dict(r,edge_residual=r[arm]) for r in rs]
        summary = condition.metrics.summary(adapted)
        components = condition.components(adapted)
        for label, cell in summary.items():
            target = result.setdefault(label,{})
            for public, key in [('b','b'),('midpoint','midpoint'),(arm,'edge_residual')]:
                if public in target: continue
                target[public] = cell[key]
                target[public]['components'] = components[label][key]
                subset = sorted([r for r in adapted if label=='overall' or r['domain']==label or r['sequence']==label],key=lambda r:r['image'])
                dets = [[condition.metrics.old.canonical(r[key])+[r[key][5]]] if r[key] is not None else [] for r in subset]
                gt = [[condition.metrics.old.canonical(r['gt'])] for r in subset]
                native = {}
                for threshold in (.5,.75):
                    ref=ap.evaluate_images(dets,gt,threshold)
                    native[str(threshold)] = ap.verify_native(dets,gt,threshold,ref)
                target[public]['native_ap11'] = native
            result[label][arm+'_paired_vs_M']=cell['paired']
    return result


def depth_report(predictions, truths, manifest, calibration):
    rs=[]
    intr,geom,params=manifest['camera']['intrinsics'],manifest['obb_reference_geometry'],calibration['parameters']
    if len(predictions)!=len(truths):raise ValueError('Depth frame count mismatch')
    for i,(p,t) in enumerate(zip(predictions,truths)):
        if p['frame_id']!='frame_%05d'%i:raise ValueError('Depth prediction order differs')
        gt=depth.truth_box(t); oracle=depth.depth(gt,intr,geom,params)
        r=dict(frame_id=p['frame_id'],frame_index=i,truth_z_m=t['camera_geometry']['z_cg_opt_m'],
               theta_total_deg=t['pivot_relative']['theta_total_deg'],gt_box=gt)
        for arm,box in [(a,p[a]) for a in ARMS]+[('gt_obb',gt)]:
            comp=depth.depth(box,intr,geom,params) if box is not None else dict(z_m=None,q_signed=None,status='missing_output')
            q=comp['q_signed']
            r[arm]=dict(box=box,depth=comp,q_in_fit_support=q is not None and params['q_signed_min']<=q<=params['q_signed_max'],
                geometry=depth.geometry_row(box,gt,comp,oracle) if box is not None else None)
        rs.append(r)
    summary=dict(sequence=manifest['sequence_id'],split=manifest['split'],
        methods={a:depth.summarize(rs,a) for a in (*ARMS,'gt_obb')},
        paired={before+'->'+after:pair_depth(rs,before,after) for before,after in
                [('b','midpoint'),('b','i07'),('b','axis15'),('midpoint','i07'),('midpoint','axis15'),('i07','axis15')]},
        propagation={a:None for a in ARMS},by_truth_depth={},by_theta={})
    from crane_project.utils.port_depth_four_arm_v1 import propagation
    summary['propagation']={a:propagation(rs,calibration,a) for a in ARMS}
    for label,lo,hi in [('around12m',0,14),('around16m',14,18),('around20m',18,22),('around24m',22,math.inf)]:
        sub=[r for r in rs if lo<=r['truth_z_m']<hi]
        summary['by_truth_depth'][label]={a:depth.summarize(sub,a) for a in ARMS} if sub else None
    for label,lo,hi in [('0_3',0,3),('3_5',3,5),('5_8',5,8),('8_inf',8,math.inf)]:
        sub=[r for r in rs if lo<=r['theta_total_deg']<hi]
        summary['by_theta'][label]={a:depth.summarize(sub,a) for a in ARMS} if sub else None
    return rs,summary


def run(args):
    # Dataset paths are explicit, source stays outside the result root.
    import torch
    import numpy as np
    from tools.depth import audit_port_midpoint_depth_v1 as native
    from tools import eval_port_frozen_downstream_v1 as legacy
    from crane_project.tools import eval_port_geometry_midpoint_formal_v1 as formal
    from crane_project.tools import eval_port_geometry_midpoint_v1_test as test
    from crane_project.tools import run_port_geometry_size_condition_v1 as condition
    from crane_project.utils.port_geometry_size_condition_v1 import GeometryConditionSizeHead
    out=Path(args.out_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(protocol=VERSION,status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),
        weights={},training_updates=0,selection=False,formula_refit=False,reliability_filter=False,
        i07_identity_unchanged=True,axis15_remains_unadopted=True,memory_budget=dict(process_mib=1536,min_free_mib=2048,allocator_mib=1024))
    try:
        if os.environ.get('WORLD_SIZE','1')!='1' or ',' in os.environ.get('CUDA_VISIBLE_DEVICES',''):
            raise ValueError('One explicit physical GPU required')
        check_memory(memory_snapshot())
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(1024/(torch.cuda.get_device_properties(0).total_memory/2**20),0)
        for a in ARMS:
            p=Path(getattr(args,a+'_weight')).resolve()
            if depth.sha(p)!=PINS[a]:raise ValueError('Frozen weight SHA differs '+a)
            report['weights'][a]=dict(path=str(p),sha256=PINS[a])
        if depth.sha(ROOT/depth.CALIBRATION)!=depth.CAL_SHA:raise ValueError('Raw-opt SHA drift')
        calibration=load(ROOT/depth.CALIBRATION)
        report['size_source_identity']=condition.checked_sources()
        _,_,report['depth_source_identity']=legacy.checked_sources()
        report['calibration_sha256']=depth.CAL_SHA
        report['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip()
        report['script_sha256']=depth.sha(__file__)
        detector,mid,pipeline,helpers,runtime=native.load_native(Path(args.b_weight),Path(args.midpoint_weight),0)
        report['runtime']=runtime
        heads={}
        for a,epoch,protocol in [('i07',7,condition.VERSION),('axis15',15,'port_geometry_size_axis_balance_v1')]:
            payload=torch.load(getattr(args,a+'_weight'),map_location='cpu')
            if payload['protocol']!=protocol or payload['epoch']!=epoch or payload['updates']!=epoch*639:raise ValueError('Size head metadata differs')
            h=GeometryConditionSizeHead().cuda(0);h.load_state_dict(payload['head_state'],strict=True);h.eval().requires_grad_(False)
            if helpers.g.state_digest(h)!=payload['head_digest']:raise ValueError('Size head tensor digest differs')
            heads[a]=h
        states=lambda:dict(b=helpers.g.state_digest(detector),midpoint=helpers.g.state_digest(mid),**{a:helpers.g.state_digest(h) for a,h in heads.items()})
        initial=states(); report['state_before']=initial
        observed=[];captured={};counters=dict(feature_extractions=0,native_head_calls=0)
        def audit(event):
            key={'test_feature_extracted':'feature_extractions','test_native_head_called':'native_head_calls'}[event['stage']]
            counters[key]+=1
        def hook(module,inputs,output):captured.update(inputs=inputs,midpoint=output['boxes_original'])
        handle=mid.register_forward_hook(hook)
        def predict(source):
            image,metas=native.native_view(pipeline,source,0,helpers)
            captured.clear()
            result=helpers.capture(detector,mid,image,metas,audit)
            roi,support,b,bm,xy=captured['inputs'];m=captured['midpoint']
            result['delivery']={}
            with torch.no_grad():
                for a,h in heads.items():
                    value=h(roi,support,b,bm,xy,m)
                    result[a]=value['boxes_original'][0].cpu().tolist() if len(m) else None
                    result['delivery'][a]=('no_output' if not len(m) else 'square' if value['square_bypass'][0] else 'neutral' if value['neutral_bypass'][0] else 'accepted' if value['accepted'][0] else 'fallback')
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
        for folder,filename in [(args.uniform_run,'candidate/val_epoch_07.rows.jsonl.gz'),(args.axis_run,'candidate/val_epoch_15.rows.jsonl.gz')]:
            folder=Path(folder)
            if depth.sha(folder/filename)!=load(folder/'artifacts.json')['files'][filename]:raise ValueError('Saved VAL rows SHA drift')
        original=condition.old.read_rows(Path(args.uniform_run)/'candidate/val_epoch_07.rows.jsonl.gz')
        weighted=condition.old.read_rows(Path(args.axis_run)/'candidate/val_epoch_15.rows.jsonl.gz')
        refs=[]
        for i,(a,b) in enumerate(zip(original,weighted)):
            if a['image']!=b['image']:raise ValueError('Saved VAL identities differ')
            refs.append(dict(a,i07=a['edge_residual'],axis15=b['edge_residual'],delivery=dict(i07=a['size_delivery'],axis15=b['size_delivery'])))
        report['val_replay']=compare_boxes(valpred,refs)
        for p,s in zip(valpred,sources):p['gt']=np.asarray(s['gt'],dtype=np.float32).tolist()
        write_rows(out/'val/evaluation_rows.jsonl',valpred)
        write(out/'val/summary.json',detection_report(valpred));write(out/'val/input_identity.json',data)
        print('VAL_NATIVE_CHAIN_VERIFIED',flush=True)
        # TEST access only after complete VAL replay succeeded.
        sources,data=detection_sources('test')
        ps=stream(sources,out/'test',True)
        for p,s in zip(ps,sources):p['gt']=np.asarray(s['gt'],dtype=np.float32).tolist()
        write_rows(out/'test/evaluation_rows.jsonl',ps)
        write(out/'test/summary.json',detection_report(ps));write(out/'test/input_identity.json',data)
        depth_contract=load(ROOT/'tools/port_frozen_downstream_v1_protocol.json')
        for name in [depth.SEQUENCE,'fixed_dev_01_unseen_depth_swing','unknown_test_02_coverage_retry']:
            if name==depth.SEQUENCE:
                manifest,tp,images,proof=native.checked_dataset(args.data_root)
            else:
                pin=depth_contract['sequences'][name]
                manifest,tp,images=legacy.checked_dataset(args.data_root,name,pin)
                proof=pin
            predictions=stream(images,out/name,False)
            # Finished all four prediction streams before parsing metric truth.
            truths=rows(tp)
            if name==depth.SEQUENCE:depth.validate_truth(truths,manifest)
            else:legacy.validate_truth(truths,manifest,depth_contract['sequences'][name])
            evaluated,summary=depth_report(predictions,truths,manifest,calibration)
            with (out/name/'depth_rows.jsonl').open('x') as f:
                for r in evaluated:f.write(json.dumps(r,allow_nan=False)+'\n')
            write(out/name/'summary.json',summary);write(out/name/'input_identity.json',proof)
        handle.remove();report['state_after']=states()
        if initial!=report['state_after']:raise ValueError('Frozen model state changed')
        if counters!=dict(feature_extractions=5298,native_head_calls=15894):raise ValueError('Native work count differs')
        report.update(status='FROZEN_NATIVE_VAL_TEST_DEPTH_COMPLETE',memory_samples=observed,counters=counters,
            peak_torch_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_torch_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    except Exception as e:
        report.update(status='RESOURCE_STOP' if 'RESOURCE_STOP' in str(e) or isinstance(e,torch.cuda.OutOfMemoryError) else 'FAILED',error=repr(e))
        raise
    finally:
        write(out/'completion.json',report)
        files={str(p.relative_to(out)):depth.sha(p) for p in out.rglob('*') if p.is_file()}
        write(out/'artifacts.json',dict(files=files))


def parser():
    p=argparse.ArgumentParser()
    for a in ARMS:p.add_argument('--'+a.replace('_','-')+'-weight',required=True)
    for n in ('uniform-run','axis-run','out-dir','data-root'):p.add_argument('--'+n,required=True,type=Path)
    return p


if __name__=='__main__':run(parser().parse_args())
