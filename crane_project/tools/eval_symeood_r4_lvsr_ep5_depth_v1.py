"""Frozen R4/M20/L5 Unknown-02 depth; unchanged Raw-opt, no EOOD or training."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import eval_symeood_r4_lvsr_ep5_fixed_test_v1 as fixed
from crane_project.tools import eval_symeood_r4_midpoint_depth_v1 as prior
from crane_project.utils import port_midpoint_depth_v1 as depth

VERSION='symeood_r4_m20_l5_unknown02_frozen_depth_v1'
BASE=fixed.p.BASE
DEPTH_ROOT=BASE/'lvsr/depth'
PROTOCOL=DEPTH_ROOT/'protocol.json'
OUT=DEPTH_ROOT/'attempt_001'
ARMS=('R4','R4_midpoint','R4_midpoint_L5')
BASELINE_PINS={
    'attempt_001/pair_completion.json':'4cd1904888bdf3152d4dcc140eaa3a8f8bfb5289fefc4495041e5d749f499b32',
    'attempt_001/depth_pair_summary.json':'ae33bacf10dc3512197f858b4d0b335d978025e9de2f6faf554f8fc119a22989',
    'attempt_001/depth_pair_rows.jsonl':'e57c5270231cd0abfd099e0a841f7f9cae1fd13755ae898fb91ec28a3540b0d6',
    'attempt_001/R4_midpoint.predictions.jsonl':'b4fb7f714bfbbefb3192e7fc0aa1812d1aafbb121ca492a80d5366c6a9964001',
}
TEST_PINS={
    'completion.json':'b70e2ae583f36cf33ae8973604946a62430b17be9e988cdb770795d8a77a44c0',
    'sample_VAL_gate.json':'cc6092ae1111749db478000163a1560ca32046c19a89152cff11c0aa87751a52',
}
EXTRA_SOURCES=(
    'crane_project/tools/eval_symeood_r4_lvsr_ep5_depth_v1.py',
    'tests/test_symeood_r4_lvsr_ep5_depth_v1.py',
    'crane_project/tools/eval_symeood_r4_midpoint_depth_v1.py',
    depth.CALIBRATION,
)
read,save=fixed.read,fixed.save


def baseline_directory():
    for path in (BASE/'depth',BASE/'readbacks/fixed_test_depth/depth'):
        if (path/'attempt_001/pair_completion.json').exists():
            for name,digest in BASELINE_PINS.items():
                if depth.sha(path/name)!=digest:raise ValueError('Fixed R4/M depth artifact differs: '+name)
            receipt=read(path/'attempt_001/pair_completion.json')
            if (receipt['status']!='FIXED_R4_M_UNKNOWN02_DEPTH_PAIR_COMPLETE' or receipt['frames']!=1097
                    or receipt['active_arms']!=['R4','R4_midpoint'] or receipt['training_updates']
                    or receipt['formula_refit'] or receipt['depth_for_small_paper']):
                raise ValueError('Completed requested R4/M pair required')
            return path
    raise ValueError('Previous complete R4/M depth pair required')


def completed_test_directory():
    for path in (fixed.OUT,BASE/'readbacks/lvsr_fixed_test/lvsr/test/attempt_002'):
        if (path/'completion.json').exists():
            for name,digest in TEST_PINS.items():
                if depth.sha(path/name)!=digest:raise ValueError('Fixed L5 TEST artifact differs: '+name)
            c=read(path/'completion.json')
            if (c['status']!='FIXED_R4_M20_L5_TEST_COMPLETE' or c['frames']!=1440
                    or c['training_updates'] or c['lvsr_sha256']!=fixed.HEAD_SHA
                    or c['state_before']!=c['state_after'] or not read(path/'sample_VAL_gate.json')['passed']):
                raise ValueError('Same-weight completed L5 online VAL/TEST proof required')
            return path
    raise ValueError('Fixed L5 TEST completion required')


def static_identity():
    parent=fixed.static_identity();baseline_directory();completed_test_directory()
    closure=read(ROOT/'tools/port_frozen_downstream_v1_sources.json')
    for name in prior.LEGACY_SOURCES:
        if depth.sha(ROOT/name)!=closure['sources'][name]:raise ValueError('Depth source differs: '+name)
    if depth.sha(ROOT/depth.CALIBRATION)!=depth.CAL_SHA:raise ValueError('Original calibration differs')
    pin=read(ROOT/'tools/port_frozen_downstream_v1_protocol.json')['sequences'][prior.SEQUENCE]
    if pin['count']!=1097 or pin['split']!='unknown_test':raise ValueError('Full Unknown-02 required')
    return dict(protocol=VERSION,parent=parent,sequence=prior.SEQUENCE,pin=pin,physical_gpu=2,
        arms=list(ARMS),baseline_artifacts=BASELINE_PINS,completed_TEST_artifacts=TEST_PINS,
        sources={n:depth.sha(ROOT/n) for n in prior.LEGACY_SOURCES+EXTRA_SOURCES},
        calibration_sha256=depth.CAL_SHA,target='z_cg_opt_m',coordinate_contract='raw_opt_v1',
        formula_refit=False,training_updates=0,selection_on_test=False,q_support_filter=False,
        reliability_filter=False,EOOD=False,depth_for_small_paper=False,automatic_adoption=False,
        reused_same_weight_online_VAL_gate=True,reused_R4_M_depth=True,exposed_TEST=True)


def score(predictions,truths,manifest,calibration):
    """CPU metrics only, after the full GT-free prediction stream has closed."""
    from crane_project.utils.port_depth_four_arm_v1 import direct_errors,propagation
    if len(predictions)!=1097 or len(truths)!=1097:raise ValueError('All1097 frames required')
    intr=manifest['camera']['intrinsics'];geom=manifest['obb_reference_geometry'];params=calibration['parameters']
    evaluated=[]
    for index,(pred,truth) in enumerate(zip(predictions,truths)):
        if pred['frame_id']!='frame_%05d'%index:raise ValueError('Frame identity/order differs')
        gt=depth.truth_box(truth);oracle=depth.depth(gt,intr,geom,params)
        if oracle['status']!='finite_formula':raise ValueError('GT formula diagnostic not computable')
        row=dict(frame_id=pred['frame_id'],frame_index=index,gt_box=gt,
            truth_z_m=truth['camera_geometry']['z_cg_opt_m'],theta_total_deg=truth['pivot_relative']['theta_total_deg'])
        for arm,box in [(a,pred[a]) for a in ARMS]+[('gt_obb',gt)]:
            value=depth.depth(box,intr,geom,params) if box is not None else dict(z_m=None,q_signed=None,status='missing_output')
            q=value['q_signed']
            row[arm]=dict(box=box,depth=value,
                q_in_fit_support=q is not None and params['q_signed_min']<=q<=params['q_signed_max'],
                geometry=depth.geometry_row(box,gt,value,oracle) if box is not None else None)
        evaluated.append(row)
    paired={}
    for before in ('R4','R4_midpoint'):
        valid=[r for r in evaluated if r[before]['depth']['z_m'] is not None and r[ARMS[-1]]['depth']['z_m'] is not None]
        changes=[abs(r[ARMS[-1]]['depth']['z_m']-r['truth_z_m'])-abs(r[before]['depth']['z_m']-r['truth_z_m']) for r in valid]
        paired[before+'->'+ARMS[-1]]=dict(common=len(valid),
            improved=sum(v < -1e-12 for v in changes),worsened=sum(v > 1e-12 for v in changes),
            tied=sum(abs(v)<=1e-12 for v in changes),
            mean_absolute_error_change_m=math.fsum(changes)/len(changes) if changes else None)
    common=[r for r in evaluated if all(r[a]['depth']['z_m'] is not None for a in ARMS)]
    summary=dict(protocol=VERSION,sequence=prior.SEQUENCE,split='unknown_test',frames=1097,exposed_TEST=True,
        arms=list(ARMS),target='z_cg_opt_m',calibration_sha256=depth.CAL_SHA,
        methods={a:depth.summarize(evaluated,a) for a in (*ARMS,'gt_obb')},
        direct_depth_errors={a:direct_errors(evaluated,a) for a in (*ARMS,'gt_obb')},
        signed_propagation={a:propagation(evaluated,calibration,a) for a in ARMS},paired=paired,
        common_valid=dict(count=len(common),total=1097,methods={a:depth.error_metrics(
            [r[a]['depth']['z_m'] for r in common],[r['truth_z_m'] for r in common]) for a in ARMS}),
        formula_refit=False,training_updates=0,selection_on_test=False,q_support_filter=False,
        reliability_filter=False,depth_for_small_paper=False,automatic_adoption=False,
        metric_truth_read_after_predictions_closed=True)
    return evaluated,summary


def prepare():
    identity=static_identity();DEPTH_ROOT.mkdir(parents=True,exist_ok=False)
    save(PROTOCOL,dict(identity,user_authorized=True,
        authorized_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat()))
    print('FIXED_L5_DEPTH_PREPARED_NO_GPU_OR_TRUTH_ACCESS',flush=True)


def infer():
    lock=read(PROTOCOL);identity=static_identity()
    if ({k:lock[k] for k in identity}!=identity or not lock['user_authorized'] or OUT.exists()
            or os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or os.environ.get('WORLD_SIZE','1')!='1'):
        raise ValueError('Fresh output and single authorized GPU2 required')
    from crane_project.tools import train_symeood_r4_midpoint_v1 as adapted
    adapted.resource_check(2,before=True);engine=adapted.make_engine(2)
    from tools import eval_port_frozen_downstream_v1 as legacy
    from tools.depth.audit_port_midpoint_depth_v1 import native_view
    from mmdet.datasets.pipelines import Compose
    from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity
    torch,g,old=engine.torch,engine.g,engine.old
    original_init=torch.optim.Optimizer.__init__
    def forbidden(*args,**kwargs):raise AssertionError('No optimizer during frozen depth')
    torch.optim.Optimizer.__init__=forbidden
    OUT.mkdir(parents=True,exist_ok=False);started=time.perf_counter();handle=None;resources=[]
    report=dict(protocol=VERSION,training_updates=0,formula_refit=False,selection_on_test=False,
        target='z_cg_opt_m',physical_gpu=2,metric_truth_read=False,depth_for_small_paper=False,
        lvsr_sha256=fixed.HEAD_SHA,detector_sha256=fixed.p.parent.R4_SHA,midpoint_sha256=fixed.p.MIDPOINT_SHA)
    def status(stage,**extra):
        tmp=OUT/'status.tmp';tmp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),
            physical_gpu=2,time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),**extra),indent=2));tmp.replace(OUT/'status.json')
    def resource():
        r=engine.resource_check(0);resources.append(r)
        with (OUT/'resources.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    try:
        save(OUT/'frozen_contract.json',lock)
        runtime=engine.runtime(0);engine.seed_all();torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        status('load_frozen_models')
        detector,mid,cfg=engine.load_selected_pipeline(fixed.fixed.training_directory()/'train/selection.json',0)
        payload=torch.load(str(fixed.training_directory()/'head_epoch_05.pth'),map_location='cpu')
        fixed.validate_payload(payload,identity['parent']['training_identity'])
        head=PairSpecificSizeHead().cuda();head.load_state_dict(payload['head_state'],strict=True)
        for model in (detector,mid,head):model.eval().requires_grad_(False)
        def states():return {n:g.state_digest(m) for n,m in [('detector',detector),('midpoint',mid),('lvsr',head)]}
        before=states();test=read(completed_test_directory()/'completion.json')
        if before!=test['state_after'] or runtime!=test['runtime']:
            raise ValueError('Same-weight online VAL/TEST state/runtime proof differs')
        save(OUT/'reused_online_VAL_gate.json',dict(passed=True,new_model_updates=0,
            existing_gate_sha256=TEST_PINS['sample_VAL_gate.json'],same_states=True,
            same_runtime=True,full_VAL_repeated=False,existing_TEST_sha256=TEST_PINS['completion.json']))
        baseline=[json.loads(v) for v in (baseline_directory()/'attempt_001/R4_midpoint.predictions.jsonl').read_text().splitlines()]
        manifest,truth_path,images=legacy.checked_dataset(ROOT/'crane_project/data/webots_depth',prior.SEQUENCE,lock['pin'])
        if len(images)!=1097 or len(baseline)!=1097:raise ValueError('Full baseline/images required')
        captured={};counts=dict(feature_extractions=0,native_head_calls=0,size_head_calls=0)
        def hook(module,inputs,output):captured.update(inputs=inputs,midpoint=output['boxes_original'])
        handle=mid.register_forward_hook(hook)
        def audit(event):
            if event['stage']=='test_feature_extracted':counts['feature_extractions']+=1
            elif event['stage']=='test_native_head_called':counts['native_head_calls']+=1
            else:raise ValueError('Unknown capture event')
        def predict(image,metas):
            captured.clear();pred=old.capture(detector,mid,image,metas,audit=audit)
            m=captured['midpoint'];value=head(*captured['inputs'],m);counts['size_head_calls']+=1
            pred.update(edge_residual=value['boxes_original'][0].cpu().tolist() if len(m) else None,
                size_delivery='no_output' if not len(m) else 'neutral' if bool(value['neutral_bypass'][0]) else
                    'square' if bool(value['square_bypass'][0]) else 'accepted' if bool(value['accepted'][0]) else 'fallback',
                size_failed_checks=[k for k,v in value['checks'].items() if len(m) and not bool(v[0])])
            if not frozen_identity(pred):raise ValueError('L5 changed center/angle/score/presence/axis')
            return pred
        pipeline=Compose(deepcopy(cfg.data.val.pipeline));maxima=dict(b=0.,midpoint=0.)
        status('Unknown02',done=0,total=1097);resource()
        with torch.no_grad(),(OUT/'depth_predictions.jsonl').open('x') as stream:
            for index,source in enumerate(images):
                ref=baseline[index]
                if (source['frame_id']!=ref['frame_id'] or source['sha256']!=ref['image_sha256']
                        or depth.sha(source['image_path'])!=source['sha256']):
                    raise ValueError('Original depth frame/image SHA differs')
                image,metas=native_view(pipeline,source,0,old);pred=predict(image,metas)
                for arm,value in fixed.compare_boxes(pred,ref,('b','midpoint')).items():maxima[arm]=max(maxima[arm],value)
                pred.update(frame_id=source['frame_id'],image_sha256=source['sha256'])
                stream.write(json.dumps(g.json_native(pred),allow_nan=False)+'\n');stream.flush()
                if index%25==0 or index==1096:
                    resource();status('Unknown02',done=index+1,total=1097)
                    print('FIXED L5 DEPTH',index+1,'/1097',flush=True)
        handle.remove();handle=None
        if states()!=before or counts!=dict(feature_extractions=1097,native_head_calls=3291,size_head_calls=1097):
            raise ValueError('Frozen model state or single-pass count differs')
        save(OUT/'prediction_completion.json',dict(status='FIXED_L5_UNKNOWN02_PREDICTIONS_COMPLETE',frames=1097,
            training_updates=0,state_before=before,state_after=states(),counters=counts,
            baseline_max_box_abs=maxima,baseline_scores_exact=True,
            model_metric_truth_read=False,runtime=runtime,lvsr_sha256=fixed.HEAD_SHA))
        # Free GPU before any metric truth is parsed. Baseline models ran only
        # within the one feature pass needed by L5; no separate baseline job.
        del detector,mid,head,pipeline,captured,payload;torch.cuda.empty_cache()
        status('CPU_depth_metrics');truths=legacy.read_rows(truth_path)
        legacy.validate_truth(truths,manifest,lock['pin'])
        calibration=read(ROOT/depth.CALIBRATION)
        predictions=[]
        for line in (OUT/'depth_predictions.jsonl').read_text().splitlines():
            row=json.loads(line);predictions.append(dict(frame_id=row['frame_id'],
                R4=row['b'],R4_midpoint=row['midpoint'],R4_midpoint_L5=row['edge_residual']))
        evaluated,summary=score(predictions,truths,manifest,calibration)
        previous=read(baseline_directory()/'attempt_001/depth_pair_summary.json')
        for arm in ('R4','R4_midpoint','gt_obb'):
            if (summary['methods'][arm]!=previous['methods'][arm]
                    or summary['direct_depth_errors'][arm]!=previous['direct_depth_errors'][arm]):
                raise ValueError('Original R4/M depth metrics not reproduced: '+arm)
        with (OUT/'depth_rows.jsonl').open('x') as f:
            for row in evaluated:f.write(json.dumps(row,allow_nan=False)+'\n')
        save(OUT/'depth_summary.json',summary)
        if static_identity()!=identity or depth.sha(ROOT/fixed.p.parent.R4_CHECKPOINT)!=fixed.p.parent.R4_SHA:
            raise ValueError('Frozen source/weight/baseline changed')
        report.update(status='FIXED_R4_M20_L5_UNKNOWN02_DEPTH_COMPLETE',frames=1097,arms=list(ARMS),
            metric_truth_read_after_predictions_closed=True,baseline_metrics_exact=True,runtime=runtime,
            calibration_sha256=depth.CAL_SHA,counters=counts,state_before=before,state_after=before,
            elapsed_seconds=time.perf_counter()-started,
            resource=dict(own_peak_mib=max(r['own_process_mib'] for r in resources),
                minimum_free_mib=min(r['free_mib'] for r in resources),allocator_peak_mib=torch.cuda.max_memory_allocated(0)/2**20))
        save(OUT/'completion.json',report);status('complete',frames=1097,updates=0)
        save(OUT/'artifacts.json',dict(files={f.name:depth.sha(f) for f in OUT.iterdir() if f.is_file()}))
        print('FIXED_R4_M20_L5_UNKNOWN02_DEPTH_COMPLETE',flush=True)
    except BaseException as error:
        report.update(status='FAILED',error=repr(error));save(OUT/'failure.json',report)
        status('stopped_on_error',error=repr(error),retry=False);raise
    finally:
        if handle is not None:handle.remove()
        torch.optim.Optimizer.__init__=original_init


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('check','prepare','infer'),default='check')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.stage=='check':print(json.dumps(static_identity(),indent=2))
    else:globals()[args.stage]()
