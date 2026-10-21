"""Frozen S049 ten-version two-stage workflow reporting. No model changes, fitting or selection."""
import itertools
import math
import os
import threading
from crane_project.tools.audit_port_size_native_p3_v1 import smi, write
from crane_project.utils import port_midpoint_depth_v1 as depth
from crane_project.tools import eval_port_size_proposal_downstream_v1 as old
from crane_project.utils.port_size_qalign_downstream_v1 import (
    identity, delivery_label, replay_inputs, require_val_proof, geometry_summary)
from crane_project.utils import port_size_qalign_downstream_v1 as previous

VERSION='port_size_direct_continuation_downstream_v1'
ARMS=('b','midpoint','d02','a20','c6','r6','dc6','dr6','dc1','dr1')
MAIN=('b','midpoint','d02','c6','r6','dc6','dr6')
AUXILIARY=('a20','dc1','dr1')
from crane_project.utils import port_size_structure_downstream_v1 as sealed_previous
HEADS={k:dict(v) for k,v in sealed_previous.HEADS.items() if k!='v20'}
HEADS.update(dc6=dict(arm='C',epoch=6,sha256='c4f8c00dd5b53412ee42d15462baf9fb0e4804d4490a1702b740d6b701654f3b'),
    dr6=dict(arm='R',epoch=6,sha256='918c04224b051a50329ed28b2581f1e9fcfb14e1a800075f03575f3e04bceac1'),
    dc1=dict(arm='C',epoch=1,sha256='ffdbe77bc40906c85d200c720b40e24f8cbc5c9f3544f98cfb892613599f8905'),
    dr1=dict(arm='R',epoch=1,sha256='d7d11a47bd55682e7c4093277553ecf6ae2ddd7ee76097690c48ac4b40683e25'))
DIRECT=('d02','dc6','dr6','dc1','dr1')
PINS=dict(b=depth.B_SHA,midpoint=depth.HEAD_SHA,**{k:v['sha256'] for k,v in HEADS.items()})
SEQUENCE='unknown_test_02_coverage_retry'


def protocol():
    p=previous.protocol()
    p.update(protocol=VERSION,versions=list(ARMS),main_versions=list(MAIN),auxiliary_versions=list(AUXILIARY),size_heads=HEADS,weight_sha256=PINS,
        inference='All ten actual same-run frozen versions. Original fixed B-P3/read_pair/W, original D_direct or AlignedSizeHead forward, size-only delivery. No output aliases, filters, fitting or epoch competition.',
        stages=['full source and checkpoint identity','complete887 native VAL including sealed original tensors/W/axis/resize and cached final/pre_guard/score/presence/validity replay; fail blocks TEST/depth','complete1440 TEST Real868 Sim572','complete1097 Unknown02 frozen Raw-opt only'],
        depth_metrics=['mae_m','rmse_m','bias_m','p95_abs_error_m','max_abs_error_m','abs_rel','numeric_depth_count','frame_count','numeric_depth_coverage'],
        fixed_roles='Main fixed-budget epoch6 D-C6,D-R6 versus LVSR-C6,LVSR-R6 and B/M/D02. D-C1,D-R1 retain original descriptive representative identity but auxiliary only. A20 is auxiliary LVSR start. All fixed before dataset access; no selecting between epoch1 and6.',
        comparisons='D02->DC6 ordinary continuation; DC6->DR6 ratio increment; LVSR_C6<->DC6 and LVSR_R6<->DR6 full two-stage workflows, NOT equal-stage pure structural attribution. All pairwise depth and size transitions plus M/start references.',
        historical='All old baselines rerun, exact per-frame compared to S047 indexed results (B/M/D02/A20/C6/R6). Original source/VAL adoption failures remain; no new adoption or independent generalization from exposed TEST/Unknown.')
    p['depth_reporting']=p['depth_reporting'].replace('all-four','all-ten')
    p['resources']['physical_gpu_choices']=[2,3]
    p['resources']['choice']='Maximum free on physical GPU2/3, prefer3 on tie; single card, own7GB cap, do not touch other processes'
    return p




def make_head(public):
    if public not in HEADS:raise ValueError('Unknown frozen size head')
    from crane_project.utils import port_size_structure_v1 as structure
    from crane_project.utils.port_size_qalign_v1 import AlignedSizeHead
    return structure.head('D_direct') if public in DIRECT else AlignedSizeHead('fixed_b')


def head_metadata(payload,public):
    if public not in ('dc6','dr6','dc1','dr1'):
        sealed_previous.head_metadata(payload,public)
        return
    spec=HEADS[public]
    if payload.get('protocol')!='port_size_direct_continuation_v1' or payload.get('arm')!=spec['arm'] or payload.get('epoch')!=spec['epoch'] or payload.get('continuation_updates')!=spec['epoch']*639 or payload.get('source_D02_sha256')!=PINS['d02']:
        raise ValueError('Requires exact frozen S048 checkpoint, no re-selection')
    pins=payload['input_identity']['frontend']
    if pins['b_sha256']!=PINS['b'] or pins['m_sha256']!=PINS['midpoint']:raise ValueError('D continuation frontend differs')



def compare_boxes(current,reference):
    import numpy as np
    if len(current)!=887 or len(reference)!=887:raise ValueError('Complete887 native VAL required')
    maxima={a:0. for a in ARMS}
    if len({r['image'] for r in current})!=887:raise ValueError('Duplicate VAL identity')
    for r,ref in zip(current,reference):
        if r['image']!=ref['image']:raise ValueError('VAL order differs')
        for arm in ARMS:
            a,b=r[arm],ref[arm]
            if (a is None)!=(b is None):raise ValueError('VAL output presence differs')
            if a is not None:
                if a[5]!=b[5] or not np.allclose(a[:5],b[:5],atol=1e-4,rtol=1e-6):raise ValueError('Native/cached box differs '+arm)
                maxima[arm]=max(maxima[arm],max(abs(x-y) for x,y in zip(a[:5],b[:5])))
        for arm in HEADS:
            if r['delivery'][arm]!=ref['delivery'][arm]:raise ValueError('Delivery decision differs '+arm)
            if r['failed_checks'][arm]!=ref['failed_checks'][arm]:raise ValueError('Delivery checks differ '+arm)
            if r['representation_valid'][arm]!=ref['representation_valid'][arm]:raise ValueError('Representation validity differs '+arm)
            a,b=r['pre_guard'][arm],ref['pre_guard'][arm]
            if (a is None)!=(b is None) or (a is not None and (a[5]!=b[5] or not np.allclose(a[:5],b[:5],atol=1e-4,rtol=1e-6))):raise ValueError('Pre-guard replay differs '+arm)
    return dict(pass_check=True,frames=887,max_box_component_abs_diff=maxima,score_presence_delivery_exact=True)



def detection_report(rows,verify_native=True):
    from crane_project.tools import run_port_geometry_size_condition_v1 as condition
    from crane_project.tools import eval_port_geometry_ap_v1 as ap
    from crane_project.tools.run_port_size_structure_v1 import near_report as near_joint
    from crane_project.utils.port_size_qalign_v1 import cross_edge
    result={};diagnostics={}
    for arm in HEADS:
        adapted=[dict(r,edge_residual=r[arm],size_delivery=r.get('delivery',{}).get(arm,'not_recorded'),representation_valid=r.get('representation_valid',{}).get(arm,True)) for r in rows]
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
            from collections import Counter
            diagnostics.setdefault(label,{})[arm]=dict(near_and_joint=near[label],cross_edge=cross_edge(sub)['overall'],
                pre_guard_sizes=condition.metrics.size_summary([dict(r,edge_residual=r.get('pre_guard',{}).get(arm)) for r in sub],'edge_residual'),
                delivery_counts=dict(Counter(r.get('delivery',{}).get(arm,'not_recorded') for r in sub)),
                failed_checks=dict(Counter(k for r in sub for k in r.get('failed_checks',{}).get(arm,[]))))

    from crane_project.utils.port_geometry_size_distribution_metrics_v1 import paired
    transitions={}
    groups={'overall':rows,**{d:[r for r in rows if r['domain']==d] for d in ('real','sim')},**{v:[r for r in rows if r['sequence']==v] for v in sorted({r['sequence'] for r in rows})}}
    for group,rr in groups.items():
        transitions[group]={}
        for before,after in itertools.combinations(ARMS,2):
            transitions[group][before+'->'+after]=paired([dict(r,midpoint=r[before],edge_residual=r[after]) for r in rr])
    return dict(metrics=result,diagnostics=diagnostics,workflow_transitions=transitions,native_ap_verified=verify_native)



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


def setup_gpu(out,user_limit_mib=7000,process_stop_mib=6800,allocator_cap_mib=6000):
    if not 0<allocator_cap_mib<process_stop_mib<user_limit_mib:raise ValueError("Invalid GPU memory limits")
    cards=[r for r in smi('index,uuid,memory.free') if int(r[0]) in (2,3)]
    requested=os.environ.get('CUDA_VISIBLE_DEVICES')
    if requested is not None:
        if requested not in ('2','3'):raise ValueError('Only physical GPU2/3 authorized by current user request')
        cards=[r for r in cards if r[0]==requested]
    if not cards:raise RuntimeError('RESOURCE_STOP no authorized GPU')
    card=max(cards,key=lambda r:int(r[2]))
    if int(card[2])<3072:raise RuntimeError('RESOURCE_STOP free<3072MiB')
    os.environ['CUDA_VISIBLE_DEVICES']=card[0]
    state=dict(physical_gpu=int(card[0]),uuid=card[1],free_before_mib=int(card[2]),
               process_peak_polled_mib=0,process_stop_mib=process_stop_mib,allocator_cap_mib=allocator_cap_mib,user_limit_mib=user_limit_mib)
    stop=threading.Event()
    def watch():
        while not stop.is_set():
            try:
                apps=smi('pid,gpu_uuid,used_memory','compute-apps')
                own=sum(int(r[2]) for r in apps if r[0]==str(os.getpid()) and r[1]==card[1])
                state['process_peak_polled_mib']=max(own,state['process_peak_polled_mib'])
                if own>=process_stop_mib:
                    write(out/'RESOURCE_STOP.json',dict(state,reason='Own process reached%dMiB safety stop'%process_stop_mib))
                    os._exit(75)
            except Exception as e:
                # A blind monitor cannot promise the user cap: stop fail closed.
                write(out/'RESOURCE_STOP.json',dict(state,reason='GPU watchdog failed: '+str(e)))
                os._exit(75)
            stop.wait(.2)
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    return state,stop,thread


