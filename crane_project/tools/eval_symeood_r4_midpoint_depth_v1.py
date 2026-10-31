"""Frozen matched EOOD / R4 / R4+M Unknown-02 depth; no formula/model fitting."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import eval_symeood_r4_midpoint_fixed_test_v1 as fixed
from crane_project.utils import port_midpoint_depth_v1 as depth

VERSION='symeood_r4_m_ep20_unknown02_frozen_depth_v1'
BASE=fixed.BASE
DEPTH_ROOT=BASE/'depth'
OUT=DEPTH_ROOT/'attempt_001'
SEQUENCE='unknown_test_02_coverage_retry'
E_CHECKPOINT='work_dirs/crane_eood_k1_port_day2night_aug_b_v1/epoch_24.pth'
E_SHA='95da2d648cfef95e1fcd00f2cdfc258d9cf75ed038135bbfc00059f8f6f2a4db'
E_CONFIG=fixed.p.E_CONFIG
ARMS=('EOOD','R4','R4_midpoint')
LEGACY_SOURCES=('tools/eval_port_frozen_downstream_v1.py',
    'tools/depth/audit_port_midpoint_depth_v1.py',
    'crane_project/utils/port_midpoint_depth_v1.py',
    'crane_project/utils/port_depth_four_arm_v1.py',
    'tools/port_frozen_downstream_v1_protocol.json',
    'docs/data/webots_depth_formal_frame_index_v1.manifest.json')
EXTRA_SOURCES=('crane_project/tools/eval_symeood_r4_midpoint_depth_v1.py',
    'tests/test_symeood_r4_midpoint_depth_v1.py', depth.CALIBRATION)


def read(path):return fixed.read(path)
def save(path,value):return fixed.save(path,value)


def static_identity():
    parent=fixed.static_identity()
    index=read(ROOT/'tools/port_frozen_downstream_v1_sources.json')
    for name in LEGACY_SOURCES:
        if depth.sha(ROOT/name)!=index['sources'][name]:
            raise ValueError('Original depth source changed: '+name)
    if depth.sha(ROOT/depth.CALIBRATION)!=depth.CAL_SHA:
        raise ValueError('Original Raw-opt calibration differs')
    frozen=read(ROOT/'tools/port_frozen_downstream_v1_protocol.json')
    pin=frozen['sequences'][SEQUENCE]
    if pin['count']!=1097 or pin['split']!='unknown_test':
        raise ValueError('Full original Unknown-02 TEST required')
    return dict(protocol=VERSION,parent=parent,physical_gpu=2,sequence=SEQUENCE,pin=pin,
        sources={n:depth.sha(ROOT/n) for n in LEGACY_SOURCES+EXTRA_SOURCES},
        arms=list(ARMS),EOOD_config=E_CONFIG,EOOD_checkpoint=E_CHECKPOINT,EOOD_checkpoint_sha256=E_SHA,
        calibration_sha256=depth.CAL_SHA,target='z_cg_opt_m',coordinate_contract='raw_opt_v1',
        formula_refit=False,training_updates=0,selection_on_test=False,q_support_filter=False,
        reliability_filter=False,LVSR=False,exposed_TEST=True,automatic_adoption=False)


def prepare():
    identity=static_identity()
    DEPTH_ROOT.mkdir(parents=True,exist_ok=False)
    save(DEPTH_ROOT/'protocol.json',identity)
    print('R4_M_DEPTH_PREPARED_NO_GPU_OR_TRUTH_READ',flush=True)


def score(predictions,truths,manifest,calibration):
    """Metric-only path, after all model prediction streams have closed."""
    from crane_project.utils.port_depth_four_arm_v1 import direct_errors,propagation
    if len(predictions)!=len(truths) or len(truths)!=1097:
        raise ValueError('Full1097 predictions/truth required')
    rows=[]; intr=manifest['camera']['intrinsics']; geom=manifest['obb_reference_geometry']
    params=calibration['parameters']
    for index,(pred,truth) in enumerate(zip(predictions,truths)):
        if pred['frame_id']!='frame_%05d'%index:
            raise ValueError('Depth prediction identity/order differs')
        gt=depth.truth_box(truth); oracle=depth.depth(gt,intr,geom,params)
        if oracle['status']!='finite_formula':raise ValueError('GT formula diagnostic failed')
        row=dict(frame_id=pred['frame_id'],frame_index=index,gt_box=gt,
            truth_z_m=truth['camera_geometry']['z_cg_opt_m'],
            theta_total_deg=truth['pivot_relative']['theta_total_deg'])
        for method,box in [(a,pred[a]) for a in ARMS]+[('gt_obb',gt)]:
            value=depth.depth(box,intr,geom,params) if box is not None else dict(z_m=None,q_signed=None,status='missing_output')
            q=value['q_signed']
            row[method]=dict(box=box,depth=value,
                q_in_fit_support=q is not None and params['q_signed_min']<=q<=params['q_signed_max'],
                geometry=depth.geometry_row(box,gt,value,oracle) if box is not None else None)
        rows.append(row)
    common=[r for r in rows if all(r[a]['depth']['z_m'] is not None for a in ARMS)]
    pairs={}
    for reference in ('EOOD','R4'):
        valid=[r for r in rows if r[reference]['depth']['z_m'] is not None and r['R4_midpoint']['depth']['z_m'] is not None]
        changes=[abs(r['R4_midpoint']['depth']['z_m']-r['truth_z_m'])-abs(r[reference]['depth']['z_m']-r['truth_z_m']) for r in valid]
        pairs[reference+'->R4_midpoint']=dict(common=len(valid),
            improved=sum(v < -1e-12 for v in changes),worsened=sum(v > 1e-12 for v in changes),
            tied=sum(abs(v)<=1e-12 for v in changes),
            mean_absolute_error_change_m=math.fsum(changes)/len(changes) if changes else None)
    summary=dict(sequence=SEQUENCE,split='unknown_test',exposed_TEST=True,frames=1097,
        calibration_sha256=depth.CAL_SHA,target='z_cg_opt_m',
        methods={a:depth.summarize(rows,a) for a in (*ARMS,'gt_obb')},
        direct_depth_errors={a:direct_errors(rows,a) for a in (*ARMS,'gt_obb')},
        signed_propagation={a:propagation(rows,calibration,a) for a in ARMS},paired=pairs,
        common_valid=dict(count=len(common),total=1097,methods={a:depth.error_metrics(
            [r[a]['depth']['z_m'] for r in common],[r['truth_z_m'] for r in common]) for a in ARMS}),
        formula_refit=False,selection_on_test=False,q_support_filter=False,reliability_filter=False,
        model_metric_truth_read=False,metric_truth_read_after_all_predictions=True)
    return rows,summary


def infer():
    lock=read(DEPTH_ROOT/'protocol.json'); identity=static_identity()
    if lock!=identity or OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='2':
        raise ValueError('Fresh output, unchanged protocol and authorized GPU2 required')
    completed=read(fixed.OUT/'completion.json');gate=read(fixed.OUT/'sample_VAL_gate.json')
    if (completed['status']!='FIXED_R4_M_EP20_TEST_COMPLETE' or not gate['passed']
            or completed['head_sha256']!=fixed.HEAD_SHA or completed['detector_sha256']!=fixed.p.R4_SHA):
        raise ValueError('Same-weight successful online gate and completed detector TEST required')
    from crane_project.tools import train_symeood_r4_midpoint_v1 as adapted
    adapted.resource_check(2,before=True)
    engine=adapted.make_engine(2)
    from tools import eval_port_frozen_downstream_v1 as legacy
    from tools.depth.audit_port_midpoint_depth_v1 import native_view
    from crane_project.utils.port_depth_four_arm_v1 import checkpoint_meta
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from mmdet.datasets.pipelines import Compose
    import numpy as np
    torch,g,old=engine.torch,engine.g,engine.old
    OUT.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    report=dict(protocol=VERSION,training_updates=0,formula_refit=False,selection_on_test=False,
        physical_gpu=2,metric_truth_read=False,LVSR=False,exposed_TEST=True)
    resources=[]
    def status(stage,**extra):
        tmp=OUT/'status.tmp';tmp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),time=time.time(),**extra),indent=2));tmp.replace(OUT/'status.json')
    def resource():
        r=engine.resource_check(0);resources.append(r)
        with (OUT/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),**r))+'\n')
    original_optimizer_init=torch.optim.Optimizer.__init__
    def forbidden(*args,**kwargs):raise AssertionError('No optimizer in frozen depth TEST')
    torch.optim.Optimizer.__init__=forbidden
    try:
        save(OUT/'frozen_contract.json',lock)
        runtime=engine.runtime(0);engine.seed_all();torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        status('input_identity')
        manifest,truth_path,images=legacy.checked_dataset(ROOT/'crane_project/data/webots_depth',SEQUENCE,lock['pin'])
        calibration=read(ROOT/depth.CALIBRATION)
        detector,head,cfg=engine.load_selected_pipeline(fixed.training_directory()/'train/selection.json',0)
        before=dict(detector=g.state_digest(detector),head=g.state_digest(head))
        if runtime!=read(fixed.training_directory()/'roi_cache/cache_manifest.json')['runtime']:
            raise ValueError('Frozen runtime differs')
        pipeline=Compose(deepcopy(cfg.data.val.pipeline));counts=dict(feature_extractions=0,native_head_calls=0)
        def audit(event):counts['feature_extractions' if event['stage']=='test_feature_extracted' else 'native_head_calls']+=1
        predictions=[]
        with torch.no_grad(),(OUT/'R4_midpoint.predictions.jsonl').open('x') as stream:
            for index,source in enumerate(images):
                if depth.sha(source['image_path'])!=source['sha256']:raise ValueError('Depth image changed')
                image,metas=native_view(pipeline,source,0,old)
                pred=old.capture(detector,head,image,metas,audit=audit)
                pred.update(frame_id=source['frame_id'],image_sha256=source['sha256'])
                predictions.append(dict(frame_id=source['frame_id'],R4=pred['b'],R4_midpoint=pred['midpoint']))
                stream.write(json.dumps(g.json_native(pred),allow_nan=False)+'\n');stream.flush();resource()
                if index%50==0 or index==1096:
                    status('R4_midpoint',done=index+1,total=1097);print('R4+M DEPTH',index+1,'/1097',flush=True)
        after=dict(detector=g.state_digest(detector),head=g.state_digest(head))
        if after!=before or counts!=dict(feature_extractions=1097,native_head_calls=3291):
            raise ValueError('Frozen R4/M state or one-pass call counts differ')
        save(OUT/'R4_midpoint.proof.json',dict(before=before,after=after,counts=counts,updates=0,
            model_metric_truth_read=False,output_count_and_scores_preserved=True))
        del detector,head,pipeline;torch.cuda.empty_cache()
        # Native matched-augmentation EOOD, retaining its own original candidate pipeline.
        status('EOOD_load')
        if depth.sha(ROOT/E_CHECKPOINT)!=E_SHA:raise ValueError('Matched EOOD checkpoint differs')
        ecfg=Config.fromfile(str(ROOT/E_CONFIG))
        if ecfg.data.test.pipeline!=cfg.data.test.pipeline:raise ValueError('EOOD/R4 inference view differs')
        import_modules_from_strings(**ecfg.custom_imports)
        spec=deepcopy(ecfg.model);spec.pretrained=None;spec.backbone.init_cfg=None
        baseline=build_detector(spec)
        loaded=load_checkpoint(baseline,str(ROOT/E_CHECKPOINT),map_location='cpu',strict=True)
        meta=checkpoint_meta(loaded['meta'],ecfg.to_dict(),24);del loaded
        g.freeze_detector(baseline.cuda(0));ebefore=g.state_digest(baseline)
        pipeline=Compose(deepcopy(ecfg.data.test.pipeline));ecount=[0]
        hook=baseline.neck.register_forward_hook(lambda *args:ecount.__setitem__(0,ecount[0]+1))
        try:
            with torch.no_grad(),(OUT/'EOOD.predictions.jsonl').open('x') as stream:
                for index,source in enumerate(images):
                    if depth.sha(source['image_path'])!=source['sha256']:raise ValueError('Depth image changed')
                    image,metas=native_view(pipeline,source,0,old)
                    value=baseline.simple_test(image,metas,rescale=True)
                    if len(value)!=1 or len(value[0])!=1:raise ValueError('One image/class native EOOD required')
                    boxes=np.asarray(value[0][0])
                    if boxes.ndim!=2 or boxes.shape[1]!=6 or not np.isfinite(boxes).all():raise ValueError('Finite Nx6 EOOD required')
                    chosen=int(boxes[:,5].argmax()) if len(boxes) else None
                    box=boxes[chosen].tolist() if chosen is not None else None
                    predictions[index]['EOOD']=box
                    stream.write(json.dumps(dict(frame_id=source['frame_id'],box=box,
                        native_candidates=boxes.tolist(),selected_native_index=chosen),allow_nan=False)+'\n');stream.flush();resource()
                    if index%50==0 or index==1096:
                        status('EOOD',done=index+1,total=1097);print('EOOD DEPTH',index+1,'/1097',flush=True)
        finally:hook.remove()
        eafter=g.state_digest(baseline)
        if ebefore!=eafter or ecount[0]!=1097:raise ValueError('Frozen EOOD state/call count differs')
        save(OUT/'EOOD.proof.json',dict(before=ebefore,after=eafter,feature_extractions=ecount[0],
            checkpoint_metadata=meta,checkpoint_sha256=E_SHA,updates=0,model_metric_truth_read=False))
        del baseline,pipeline;torch.cuda.empty_cache()
        # All streams are closed before parsing metric truth.
        status('depth_metrics');truths=legacy.read_rows(truth_path)
        legacy.validate_truth(truths,manifest,lock['pin'])
        rows,summary=score(predictions,truths,manifest,calibration)
        with (OUT/'depth_rows.jsonl').open('x') as stream:
            for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
        save(OUT/'depth_summary.json',summary)
        if (static_identity()!=identity or depth.sha(ROOT/E_CHECKPOINT)!=E_SHA
                or depth.sha(ROOT/fixed.p.R4_CHECKPOINT)!=fixed.p.R4_SHA):
            raise ValueError('Frozen source/weight changed')
        report.update(status='FIXED_R4_M_EOOD_UNKNOWN02_DEPTH_COMPLETE',frames=1097,
            runtime=runtime,arms=list(ARMS),metric_truth_read_after_all_predictions=True,
            input_identity=lock['pin'],calibration_sha256=depth.CAL_SHA,
            elapsed_seconds=time.perf_counter()-started,
            resource=dict(own_peak_mib=max(r['own_process_mib'] for r in resources),
                minimum_free_mib=min(r['free_mib'] for r in resources)))
        save(OUT/'completion.json',report);status('complete',frames=1097,updates=0)
        save(OUT/'artifacts.json',dict(files={f.name:depth.sha(f) for f in OUT.iterdir() if f.is_file()}))
        print('FIXED_R4_M_EOOD_UNKNOWN02_DEPTH_COMPLETE',flush=True)
    except BaseException as error:
        report.update(status='FAILED',error=repr(error));save(OUT/'failure.json',report)
        status('stopped_on_error',error=repr(error),retry=False);raise
    finally:torch.optim.Optimizer.__init__=original_optimizer_init


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--stage',choices=('check','prepare','infer'),default='check')
    return result


if __name__=='__main__':
    args=parser().parse_args()
    if args.stage=='check':print(json.dumps(static_identity(),indent=2))
    else:globals()[args.stage]()
