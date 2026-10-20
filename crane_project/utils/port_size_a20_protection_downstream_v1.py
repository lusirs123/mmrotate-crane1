"""Frozen S043 six-version reporting. No model changes, fitting or selection."""
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

VERSION='port_size_a20_protection_downstream_v1'
ARMS=('b','midpoint','a20','c6','r6','p5')
HEADS=dict(a20=dict(arm='fixed_b',epoch=20,sha256='545e18459d3ead4a54c441d64de27ca44a8394bfdc20cc2caf4c2f7534326479'),
    c6=dict(arm='C',epoch=6,sha256='04de79a63963062d641fda9ce70e51f30dceb37e048c2b3429d0c2a0161dec52'),
    r6=dict(arm='R',epoch=6,sha256='c433497f1fd769773ec1d10909b1072e4e576a6eb3b453b8b43d31ae96a7447a'),
    p5=dict(arm='P',epoch=5,sha256='e9d0108204c79bdac22311a288560ec7fffc43c86f1091a474e16adbb9272c81'))
PINS=dict(b=depth.B_SHA,midpoint=depth.HEAD_SHA,**{k:v['sha256'] for k,v in HEADS.items()})
SEQUENCE='unknown_test_02_coverage_retry'


def protocol():
    p=previous.protocol()
    p['resources']['physical_gpu_choices']=[1]
    p.update(protocol=VERSION,versions=list(ARMS),size_heads=HEADS,weight_sha256=PINS,
        inference='Same-run native B/M; four frozen fixed-B2800params size heads, same B-P3 cache/read_pair commonW, decoder and pair fallback. Single natural M input; no filters, iteration, NMS or compensation.',
        stages=['complete887 native VAL: sealed cache/resize/coordinates/W and all six frozen output/score/delivery replay; failure blocks TEST/depth',
                'complete1440 detection TEST, six frozen versions, native AP11 and existing geometry/custom metrics',
                'complete1097 Unknown-02 only; fixed Raw-opt/calibration/physical reference, exposed TEST description'],
        depth_reporting='metreMAE/RMSE/bias/P95/coverage; all-six and pairwise common-valid paired errors/counts/frameIDs incl M/A20 and C6->R6; own short/a/r geometry and signed scale/ratio/interaction propagation, NOT additive MAE attribution.',
        historical='All C/R/P epochs failed S043 VAL eligibility; C6/R6/P5 fixed descriptive representatives, not re-selected. A20 remains unadopted, M formal. P5 vs R6 confounds epoch; use existing same-epoch VAL for mechanism. Unknown/TEST descriptive only, no independent generalization claim.')
    return p


def head_metadata(payload,public):
    if public=='a20':
        previous.head_metadata(payload,public)
        return
    spec=HEADS[public]
    if (payload.get('protocol')!='port_size_a20_protection_train_v1' or payload.get('arm')!=spec['arm'] or
        payload.get('epoch')!=spec['epoch'] or payload.get('continuation_updates')!=spec['epoch']*639 or
        payload.get('source_A20_sha256')!=PINS['a20']):
        raise ValueError('Requires fixed S043 representative; no re-selection')
    pins=payload['input_identity']['frontend']
    if pins['b_sha256']!=PINS['b'] or pins['m_sha256']!=PINS['midpoint']:
        raise ValueError('Continuation frontend identity differs')


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
            if r['vector_valid'][arm]!=ref['vector_valid'][arm]:raise ValueError('Vector validity differs '+arm)
    return dict(pass_check=True,frames=887,max_box_component_abs_diff=maxima,score_presence_delivery_exact=True)



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


def setup_gpu(out,user_limit_mib=7000,process_stop_mib=6800,allocator_cap_mib=6000):
    if not 0<allocator_cap_mib<process_stop_mib<user_limit_mib:raise ValueError("Invalid GPU memory limits")
    cards=[r for r in smi('index,uuid,memory.free') if int(r[0]) in (1,)]
    requested=os.environ.get('CUDA_VISIBLE_DEVICES')
    if requested is not None:
        if requested not in ('1',):raise ValueError('Only physical GPU1 authorized by current user request')
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


