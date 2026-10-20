"""Frozen S039 reporting and replay contracts. No training or epoch selection."""
import itertools
import math
from crane_project.utils import port_midpoint_depth_v1 as depth
from crane_project.tools import eval_port_size_proposal_downstream_v1 as old

VERSION='port_size_qalign_downstream_v1'
ARMS=('b','midpoint','a20','q20')
HEADS=dict(a20=dict(arm='fixed_b',epoch=20,sha256='545e18459d3ead4a54c441d64de27ca44a8394bfdc20cc2caf4c2f7534326479'),
           q20=dict(arm='q_aligned',epoch=20,sha256='dac187b4f010fe1b18585022a95ca4afb6a08b692e2b594f93f3b0e5945507d7'))
PINS=dict(b=depth.B_SHA,midpoint=depth.HEAD_SHA,**{k:v['sha256'] for k,v in HEADS.items()})
SEQUENCE='unknown_test_02_coverage_retry'


def protocol():
    return dict(protocol=VERSION,versions=list(ARMS),size_heads=HEADS,weight_sha256=PINS,
        inference='Same-run native B/M, existing B-P3 cache construction, identical S038 read_pair W=min(S_A,S_Q), frozen2800params heads; no Q02/other epochs, no iteration/NMS/reliability/size filter.',
        stages=['complete887 native VAL: source cache tensors/resize/coordinates/W and saved ep20 output/delivery replay; failure blocks TEST/depth',
                'complete1440 frozen detection TEST, four versions, existing custom metrics/geometry and native AP11',
                'complete1097 Unknown-02 only; exposed TEST descriptive, same frozen Raw-opt/camera/top-beam reference and numerical validity'],
        datasets=dict(val=dict(frames=887,role='native_chain_replay_no_selection'),
                      detection_test=dict(frames=1440,role='exposed_TEST_frozen_description'),
                      unknown02=dict(sequence=SEQUENCE,frames=1097,role='exposed_TEST_frozen_description')),
        geometry=dict(center_px_strict=15,size_relative_max_inclusive=.1,angle_period_deg=180,angle_max_deg_inclusive=3,
                      angle_gt_aspect_min=1.2,near_reference='fixed_M',near_max=.03),
        depth_metrics=['mae_m','rmse_m','bias_m','p95_abs_error_m','numeric_depth_count','frame_count','numeric_depth_coverage'],
        calibration_sha256=depth.CAL_SHA,
        replay=dict(box_atol=1e-4,box_rtol=1e-6,roi_atol=1e-4,roi_rtol=1e-6,support_atol=1e-6,coordinate_atol=1e-5,
                    score_presence_delivery_checks_exact=True),
        reporting='overall/domain/video; long-shortMAE/P95/bias,a/r,joint10 transitions; near fixedM<=3%,cross intersection/ALL positive harm/both-harmed/effective correction; center<15px/output,sizeboth<=10%/output,angle180period<=3deg/GTaspect>=1.2 output; missing/output/full-frame coverage; fullRIoU,DFR,ACI,MCML,TDR,AP50/75.',
        depth_reporting='metreMAE,RMSE,bias,P95,numeric count/frames,all-four and pairwise common-valid metrics,new/lost valid and frameIDs of improved/worsened/tied; short/a/r geometry; signed short/ratio/interaction propagation NOT additive MAE decomposition.',
        limits=dict(training_updates=0,selection=False,formula_refit=False,automatic_adoption=False,train03=False,fixed_dev=False,reliability_filter=False,size_accept_filter=False),
        resources=dict(physical_gpu_choices=[2,3],single_gpu=True,user_limit_mib=7000,stop_mib=6800,allocator_cap_mib=6000),
        historical='A20/Q20 remain descriptive, S039 VAL failures retained; M formal. Unknown cannot select or compensate. New independent depth generalization evidence still missing.')


def identity(m,c):
    old.identity(m,c)
    if m is not None and (m[2]<m[3])!=(c[2]<c[3]):raise ValueError('Size head changed raw axis identity')


def head_metadata(payload,public):
    spec=HEADS[public]
    if (payload.get('protocol')!='port_size_qalign_v1' or payload.get('arm')!=spec['arm'] or
            payload.get('epoch')!=20 or payload.get('updates')!=12780):
        raise ValueError('Requires fixed S039 ep20; no re-selection')
    pins=payload['input_identity']
    if pins['b_sha256']!=PINS['b'] or pins['m_sha256']!=PINS['midpoint']:
        raise ValueError('Head frontend identity differs')


def delivery_label(output,index=0):
    # Match S039 evaluate() precedence, including neutral square boxes.
    if not len(output['boxes_original']):return 'no_output'
    if bool(output['neutral_bypass'][index]):return 'neutral'
    if bool(output['square_bypass'][index]):return 'square'
    return 'accepted' if bool(output['accepted'][index]) else 'fallback'


def replay_inputs(actual,reference):
    """Native hook vs sealed VAL tensors, then compare common W and coordinates."""
    import torch
    from crane_project.utils.port_size_qalign_v1 import read_pair
    names=('roi','support','b','bm','xy','m');result={}
    for i,(a,b) in enumerate(zip(actual,reference)):
        a,b=a.detach().cpu(),b.detach().cpu()
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Native/cache tensor identity '+names[i])
        atol=1e-6 if i==1 else 1e-4
        if not torch.allclose(a,b,atol=atol,rtol=1e-6):raise ValueError('Native/cache values '+names[i])
        result[names[i]+'_max_abs']=float((a-b).abs().max()) if a.numel() else 0.
    if len(actual[2]) and not torch.equal(actual[2][:,5].cpu(),reference[2][:,5].cpu()):raise ValueError('Native B score changed')
    pa,pb=read_pair(*actual),read_pair(*reference)
    if not torch.allclose(pa['support'].cpu(),pb['support'].cpu(),atol=1e-6,rtol=1e-6):raise ValueError('Common W differs')
    result['W_max_abs']=float((pa['support'].cpu()-pb['support'].cpu()).abs().max()) if len(actual[2]) else 0.
    for arm in ('fixed_b','q_aligned'):
        if not torch.equal(pa['inside'][arm].cpu(),pb['inside'][arm].cpu()):raise ValueError('Native/cache inside decision differs')
        if not torch.allclose(pa['source_local'][arm].cpu(),pb['source_local'][arm].cpu(),atol=1e-5,rtol=1e-6):raise ValueError('Read coordinates differ')
    return result


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


def require_val_proof(proof):
    if proof.get('pass_check') is not True or proof.get('frames')!=887 or proof.get('intermediate_frames')!=887:
        raise ValueError('Complete native VAL/intermediate replay must pass before TEST/depth')


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


def geometry_summary(rows,arm):
    values={k:[] for k in ('long','short','a','r')}
    for row in rows:
        box=row[arm]['box']
        if box is None:continue
        p,g=depth.canonical(box),depth.canonical(row['gt_box']);logs=[]
        for j,e in ((2,'long'),(3,'short')):
            values[e].append(p[j]/g[j]-1);logs.append(math.log(p[j]/g[j]))
        values['a'].append(sum(logs)/2);values['r'].append(logs[0]-logs[1])
    return {k:dict(count=len(v),mae=math.fsum(map(abs,v))/len(v) if v else None,
                   bias=math.fsum(v)/len(v) if v else None,p95=depth.percentile(list(map(abs,v)),.95)) for k,v in values.items()}


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
        all_four_common_valid=dict(count=len(common),total_frames=len(result),coverage=len(common)/len(result),frame_ids=[r['frame_id'] for r in common],methods={a:depth.error_metrics([r[a]['depth']['z_m'] for r in common],[r['truth_z_m'] for r in common]) for a in ARMS}),
        boundaries='Same frozen Raw-opt and numerical validity; no q/size/reliability filtering, compensation or refit. Signed propagation identity, not additive MAE attribution. Sequence stays separate; no independent-generalization claim.')
    return result,summary
