#!/usr/bin/env python3
"""Frozen new110/M21/ep7/C3/C6: eight VAL links, TEST then unchanged Raw-opt."""
from copy import deepcopy
from datetime import datetime,timezone
import gzip
import itertools
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import run_symeood_110_lvsr_c_ep7 as training
from crane_project.tools import eval_port_size_proposal_downstream_v1 as old
from crane_project.utils.port_size_a20_protection_downstream_v1 import depth,identity,delivery_label,geometry_summary
from crane_project.tools.run_port_size_native_train_v1 import read,write,sha
from crane_project.tools.eval_port_size_proposal_downstream_v1 import write_rows,rows,native_view
from crane_project.tools import eval_symeood_110_midpoint_ep21_test as previous
VERSION='symeood_110_lvsr_frozen_downstream'
CONTRACT=ROOT/'crane_project/data_contracts/symeood_110_lvsr_downstream.json'
COMMON=training.COMMON
FORMAL=training.base.FORMAL
OUT=COMMON/'frozen_evaluation'
SEQUENCE='unknown_test_02_coverage_retry'
ARMS=('b','midpoint','ep7','c3','c6')
HEADS=dict(ep7=dict(path=COMMON/'fixed_b/head_epoch_07.pth',epoch=7,arm='fixed_b',sha256=training.PARENT_SHA),
 c3=dict(path=COMMON/'continuation_ep7/C/head_epoch_03.pth',epoch=3,arm='C',sha256='990066f7c04dc387a49b3cbd5d0adbde567b5a8c915cacf898cb75b980b7640d'),
 c6=dict(path=COMMON/'continuation_ep7/C/head_epoch_06.pth',epoch=6,arm='C',sha256='454a132c5bc4383121791c2b35a9f696672b13bc19c38463b694a8ec1a1fc6ac'))


def checked_sources():
    lock=read(CONTRACT);training.checked_sources();previous.checked_identity()
    if lock['versions']!=list(ARMS) or lock['val_names']!=list(previous.VAL_NAMES) or lock['physical_gpu']!=0:
        raise ValueError('Fixed evaluation contract differs')
    for name,h in lock['sources'].items():
        if sha(ROOT/name)!=h:raise ValueError('Evaluation source differs: '+name)
    for name,h in lock['artifacts'].items():
        if sha(ROOT/name)!=h:raise ValueError('Fixed evidence differs: '+name)
    complete=read(COMMON/'continuation_ep7/completion.json')
    if complete['versions']!=dict(selected=None,representative=3,fixed_final=6,eligible_epochs=[]):
        raise ValueError('Fixed C3 representative / C6 auxiliary identity differs')
    if sha(ROOT/depth.CALIBRATION)!=depth.CAL_SHA:raise ValueError('Frozen calibration differs')
    return lock


def compare_frame(prediction,reference,arms):
    import numpy as np
    maxima={}
    for arm in arms:
        x,y=prediction[arm],reference[arm]
        if (x is None)!=(y is None):raise ValueError('Presence differs '+arm)
        if x is not None and (x[5]!=y[5] or not np.allclose(x[:5],y[:5],atol=1e-4,rtol=1e-6)):
            raise ValueError('Frozen native box differs '+arm)
        maxima[arm]=max(abs(a-b) for a,b in zip(x[:5],y[:5])) if x is not None else 0.
    return maxima


def val_references():
    result={}
    for arm,spec in HEADS.items():
        folder=COMMON/'fixed_b' if arm=='ep7' else COMMON/'continuation_ep7/C'
        with gzip.open(folder/('epoch_%02d_val.rows.jsonl.gz'%spec['epoch']),'rt') as stream:
            for row in map(json.loads,stream):
                value=result.setdefault(row['image'],dict(row,delivery={},failed_checks={},vector_valid={}))
                value[arm]=row['edge_residual'];value['delivery'][arm]=row['size_delivery']
                value['failed_checks'][arm]=row['failed_checks'];value['vector_valid'][arm]=row.get('vector_valid')
    if len(result)!=887:raise ValueError('Saved VAL identity differs')
    return [result[n] for n in previous.VAL_NAMES]


def status(stage,**extra):
    tmp=OUT/'status.tmp';tmp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),**extra),indent=2));tmp.replace(OUT/'status.json')


def run():
    proof=checked_sources()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('WORLD_SIZE','1')!='1':raise ValueError('Single physical GPU0 required')
    if OUT.exists():raise ValueError('Existing frozen evaluation preserved; no automatic retry')
    OUT.mkdir()
    started=time.monotonic();report=dict(protocol=VERSION,status='RUNNING',training_updates=0,selection=False,formula_refit=False,TEST=True,depth=True,
        versions=list(ARMS),representative='c3',auxiliary='c6',full_VAL_replay=False,TRAIN_cache_rebuilt=False)
    try:
        from crane_project.tools import train_symeood_110_midpoint_v1 as f
        f.runtime_imports();torch=f.torch
        from crane_project.tools import eval_port_geometry_midpoint_v1_test as helper
        from crane_project.utils.port_size_qalign_v1 import AlignedSizeHead
        from mmdet.datasets.pipelines import Compose
        f.runtime(0);f.seed_all();torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        detector,mid,cfg=f.load_selected_pipeline(FORMAL/'train/selection.json',0)
        pipeline=Compose(deepcopy(cfg.data.val.pipeline))
        heads={}
        for arm,spec in HEADS.items():
            if sha(spec['path'])!=spec['sha256']:raise ValueError('Weight differs '+arm)
            payload=torch.load(str(spec['path']),map_location='cpu')
            if payload['epoch']!=spec['epoch'] or payload['arm']!=spec['arm']:raise ValueError('Weight metadata differs')
            if arm!='ep7' and (payload['parent_epoch']!=7 or payload['parent_sha256']!=training.PARENT_SHA):raise ValueError('C parent differs')
            h=AlignedSizeHead('fixed_b').cuda();keys=h.load_state_dict(payload['head_state'],strict=True);h.eval().requires_grad_(False);heads[arm]=h
        states=lambda:dict(b=f.g.state_digest(detector),midpoint=f.g.state_digest(mid),**{a:f.g.state_digest(h) for a,h in heads.items()})
        before=states();counters=dict(feature_extractions=0,native_head_calls=0,size_head_calls=0)
        captured={}
        def hook(module,inputs,output):captured.update(inputs=inputs,midpoint=output['boxes_original'])
        handle=mid.register_forward_hook(hook)
        def audit(event):counters['feature_extractions' if event['stage']=='test_feature_extracted' else 'native_head_calls']+=1
        def predict(source):
            image,metas=native_view(pipeline,source,helper);captured.clear()
            result=helper.capture(detector,mid,image,metas,audit)
            roi,support,b,bm,xy=captured['inputs'];m=captured['midpoint']
            result.update(delivery={},failed_checks={},pre_guard={},vector_valid={})
            for arm,h in heads.items():
                value=h(roi,support,b,bm,xy,m);counters['size_head_calls']+=1
                result[arm]=value['boxes_original'][0].cpu().tolist() if len(m) else None
                result['pre_guard'][arm]=value['candidate_original'][0].cpu().tolist() if len(m) and bool(value['candidate_valid'][0]) else None
                result['vector_valid'][arm]=bool(value['vector_valid'][0]) if len(m) else None
                result['failed_checks'][arm]=[k for k,v in value['checks'].items() if not bool(v[0])] if len(m) else []
                result['delivery'][arm]=delivery_label(value);identity(result['midpoint'],result[arm])
            return result
        # This user-authorized eight-frame interface gate replaces the old full887 gate.
        status('sample_VAL_gate');references=val_references();sample=[]
        with torch.no_grad():
            for ref in references:
                source=dict(ref,image_path=str(f.g.ready.DATA/'val/images'/(ref['image']+'.jpg')))
                from PIL import Image
                with Image.open(source['image_path']) as image:source['image_size']=list(image.size)
                p=predict(source);p['image']=ref['image'];maximum=compare_frame(p,ref,ARMS)
                for arm in HEADS:
                    for k in ('delivery','failed_checks','vector_valid'):
                        if p[k][arm]!=ref[k][arm]:raise ValueError('VAL decision differs '+k+'/'+arm)
                sample.append(dict(image=ref['image'],max_box_abs=maximum))
        write(OUT/'sample_VAL_gate.json',dict(passed=True,frames=8,full_VAL_replay=False,rows=sample))
        # Existing B/M predictions are checked as the three size heads share required fresh ROI extraction.
        status('TEST');sources,data=helper.fixed_test_inputs();saved={r['image']:r for r in rows(FORMAL.parent/'test_ep21/test_rows.jsonl')}
        predictions=[];maxima=dict(b=0.,midpoint=0.)
        with torch.no_grad(),(OUT/'test_rows.jsonl').open('x') as stream:
            for i,s in enumerate(sources):
                s.update(image_path=str(f.g.ready.DATA/'test/images'/(s['image']+'.jpg')))
                p=predict(s);p.update(**{k:s[k] for k in ('image','sequence','domain','frame_id')},scale=1.,split='test',gt=f.np.asarray(s['gt'],dtype=f.np.float32).tolist())
                compared=compare_frame(p,saved[s['image']],('b','midpoint'))
                for arm,v in compared.items():maxima[arm]=max(maxima[arm],v)
                predictions.append(p);stream.write(json.dumps(p,allow_nan=False)+'\n');stream.flush()
                if i%100==0 or i==len(sources)-1:
                    status('TEST',done=i+1,total=len(sources));f.resource_check(0);print('TEST',i+1,'/',len(sources),flush=True)
        write(OUT/'test_summary.json',detection_report(predictions));write(OUT/'test_input_identity.json',data)
        write(OUT/'existing_TEST_baseline_replay.json',dict(frames=1440,maximum_box_difference=maxima,separate_baseline_forward=False,features_required_for_LVSR=True))
        from tools import eval_port_frozen_downstream_v1 as legacy
        frozen=read(ROOT/'tools/port_frozen_downstream_v1_protocol.json');pin=frozen['sequences'][SEQUENCE]
        manifest,tp,images=legacy.checked_dataset(ROOT/'crane_project/data/webots_depth',SEQUENCE,pin)
        calibration=read(ROOT/depth.CALIBRATION);predictions=[]
        with torch.no_grad(),(OUT/'depth_predictions.jsonl').open('x') as stream:
            for i,s in enumerate(images):
                if depth.sha(s['image_path'])!=s['sha256']:raise ValueError('Depth image changed')
                p=predict(s);p.update(frame_id=s['frame_id'],image_sha256=s['sha256'])
                predictions.append(p);stream.write(json.dumps(p,allow_nan=False)+'\n');stream.flush()
                if i%100==0 or i==len(images)-1:
                    status('Unknown02',done=i+1,total=len(images));f.resource_check(0);print('Unknown02',i+1,'/',len(images),flush=True)
        # Truth enters metrics only after the full prediction stream is sealed.
        truths=rows(tp);legacy.validate_truth(truths,manifest,pin)
        evaluated,summary=depth_report(predictions,truths,manifest,calibration)
        write_rows(OUT/'depth_rows.jsonl',evaluated);write(OUT/'depth_summary.json',summary)
        write(OUT/'depth_input_identity.json',dict(pin=pin,camera=manifest['camera'],physical_reference=manifest['obb_reference_geometry'],calibration_sha256=depth.CAL_SHA))
        handle.remove()
        if states()!=before:raise ValueError('Frozen parameters/buffers changed')
        if checked_sources()!=proof:raise ValueError('Source or input artifacts changed')
        for arm,spec in HEADS.items():
            if sha(spec['path'])!=spec['sha256']:raise ValueError('Weight changed')
        expected=8+1440+1097
        if counters!=dict(feature_extractions=expected,native_head_calls=3*expected,size_head_calls=3*expected):raise ValueError('Inference work count differs')
        report.update(status='FROZEN_TEST_DEPTH_COMPLETE',counters=counters,state_before=before,state_after=states(),elapsed_seconds=time.monotonic()-started,resource=f.resource_check(0),source_identity=proof)
        write(OUT/'completion.json',report);status('complete',frames_TEST=1440,frames_depth=1097,updates=0)
        write(OUT/'artifacts.json',dict(files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
    except BaseException as error:
        report.update(status='FAILED',error=repr(error));write(OUT/'failure.json',report);status('stopped_on_error',error=repr(error));raise




def detection_report(rows,verify_native=True):
    from crane_project.tools import run_port_geometry_size_condition_v1 as condition
    from crane_project.tools import eval_port_geometry_ap_v1 as ap
    from crane_project.tools.run_port_size_native_train_v1 import near_joint
    from crane_project.utils.port_size_qalign_v1 import cross_edge
    result={};diagnostics={}
    for arm in HEADS:
        adapted=[dict(r,edge_residual=r[arm]) for r in rows]
        for r in adapted:identity(r['midpoint'],r['edge_residual'])
        summary=condition.metrics.summary(adapted);components=condition.components(adapted)
        near=near_joint(adapted)
        for label,cell in summary.items():
            target=result.setdefault(label,{})
            sub=sorted([r for r in adapted if label=='overall' or r['domain']==label or r['sequence']==label],key=lambda r:r['image'])
            for public,key in [('b','b'),('midpoint','midpoint'),(arm,'edge_residual')]:
                if public in target:continue
                target[public]=dict(cell[key],components=components[label][key])
                if verify_native:
                    ds=[[condition.metrics.old.canonical(r[key])+[r[key][5]]] if r[key] is not None else [] for r in sub]
                    gt=[[condition.metrics.old.canonical(r['gt'])] for r in sub]
                    target[public]['native_ap11']={str(t):ap.verify_native(ds,gt,t,ap.evaluate_images(ds,gt,t)) for t in (.5,.75)}
            target[arm+'_paired_vs_M']=cell['paired']
            diagnostics.setdefault(label,{})[arm]=dict(near_and_joint=near[label],cross_edge=cross_edge(sub)['overall'])
    return dict(metrics=result,diagnostics=diagnostics,native_ap_verified=verify_native)

def depth_report(predictions,truths,manifest,calibration):
    if len(predictions)!=len(truths) or not truths:raise ValueError('Full nonempty depth sequence required')
    intr,geom=manifest['camera']['intrinsics'],manifest['obb_reference_geometry'];params=calibration['parameters']
    result=[]
    for i,(p,t) in enumerate(zip(predictions,truths)):
        if p['frame_id']!='frame_%05d'%i:raise ValueError('Depth frame identity/order')
        gt=depth.truth_box(t);oracle=depth.depth(gt,intr,geom,params)
        if oracle['status']!='finite_formula':raise ValueError('Frozen GT formula invalid')
        row=dict(frame_id=p['frame_id'],frame_index=i,truth_z_m=t['camera_geometry']['z_cg_opt_m'],theta_total_deg=t['pivot_relative']['theta_total_deg'],gt_box=gt)
        for arm in HEADS:identity(p['midpoint'],p[arm])
        for arm,box in [(a,p[a]) for a in ARMS]+[('gt_obb',gt)]:
            comp=depth.depth(box,intr,geom,params) if box is not None else dict(z_m=None,q_signed=None,status='missing_output')
            q=comp['q_signed']
            row[arm]=dict(box=box,depth=comp,q_in_fit_support=q is not None and params['q_signed_min']<=q<=params['q_signed_max'],
                geometry=depth.geometry_row(box,gt,comp,oracle) if box is not None else None)
        result.append(row)
    from crane_project.utils.port_depth_four_arm_v1 import propagation
    pairs={}
    for before,after in itertools.combinations(ARMS,2):
        pair=old.pair_depth(result,before,after);events={k:[] for k in ('improved','worsened','tied')}
        for row in result:
            a,b=row[before]['depth']['z_m'],row[after]['depth']['z_m']
            if a is None or b is None:continue
            change=abs(b-row['truth_z_m'])-abs(a-row['truth_z_m'])
            events['improved' if change < -1e-12 else 'worsened' if change>1e-12 else 'tied'].append(row['frame_id'])
        pair['frame_ids']=events;pair['total_frames']=len(result);pair['common_coverage']=pair['common']/len(result)
        pair['new_numeric_frame_ids']=[r['frame_id'] for r in result if r[before]['depth']['z_m'] is None and r[after]['depth']['z_m'] is not None]
        pair['lost_numeric_frame_ids']=[r['frame_id'] for r in result if r[before]['depth']['z_m'] is not None and r[after]['depth']['z_m'] is None]
        pairs[before+'->'+after]=pair
    common=[r for r in result if all(r[a]['depth']['z_m'] is not None for a in ARMS)]
    summary=dict(sequence=manifest['sequence_id'],split=manifest['split'],role='exposed_TEST_frozen_description_not_selection',
        frame_count=len(result),methods={a:depth.summarize(result,a) for a in (*ARMS,'gt_obb')},paired=pairs,
        size_geometry={a:geometry_summary(result,a) for a in ARMS},propagation={a:propagation(result,calibration,a) for a in ARMS},
        all_versions_common_valid=dict(count=len(common),total_frames=len(result),coverage=len(common)/len(result),frame_ids=[r['frame_id'] for r in common],methods={a:depth.error_metrics([r[a]['depth']['z_m'] for r in common],[r['truth_z_m'] for r in common]) for a in ARMS}),
        boundaries='Same frozen Raw-opt and numerical validity; no q/size/reliability filtering, compensation or refit. Signed propagation identity, not additive MAE attribution. Sequence stays separate; no independent-generalization claim.')
    return result,summary

if __name__=='__main__':run()
