"""Fixed Train-03 depth comparisons and offline error propagation; no training."""
import ast
from copy import deepcopy
import hashlib
import math
from pathlib import Path

from crane_project.utils import port_midpoint_depth_v1 as original

VERSION = 'port_depth_four_arm_v1_train03'
ARMS = ('eood', 'symeood', 'symeood_b', 'symeood_b_midpoint')
REFERENCE_SHA = 'ec24e38ad9c9efc1ca07887e1efd53897766fae358ea17fabc6620513811e07f'
ARCHIVE = 'crane_project/data/crane_grab_port_day2night_v1/provenance/config_retirement_20260929'
BASELINES = {
    'eood': dict(config='crane_project/configs/crane_eood_k1_port_day2night_v1.py',
        stem='crane_eood_k1', epoch=24,
        checkpoint_sha256='ee277d72cdf27d76e3216da4d17256b453b539d69d692cdd71960fadf7ebd2bf',
        selection_sha256='a0bf581181890dc65cc381812cff44e93dd5c6e5658857443c08bfebba39dc8a',
        historical_config_sha256='4265a9362b5e2dc99f273591a98342c9ba8deba3bbca8ae1059bf4a209a1a3c0'),
    'symeood': dict(config='crane_project/configs/crane_symeood_k1_port_day2night_v1.py',
        stem='crane_symeood_k1', epoch=20,
        checkpoint_sha256='780a5de1a17b32041175bdf408a209872100f776de811075c55204bf333a92fe',
        selection_sha256='7f1439a451442572c9e35131a3b29312f0c37900a5414a59f8e98c3769ff502d',
        historical_config_sha256='ca8b950c8bba2775b20a075e619a24915bcee443c1a1e0bd1858a908d966ea45'),
}


def protocol_document():
    return dict(protocol=VERSION, arms=list(ARMS), sequence=original.SEQUENCE, frame_count=981,
        baselines=deepcopy(BASELINES), frozen_b_sha256=original.B_SHA,
        frozen_midpoint_sha256=original.HEAD_SHA, reused_reference_archive_sha256=REFERENCE_SHA,
        calibration_sha256=original.CAL_SHA,
        inference='Only existing EOOD ep24 and unaugmented SymEOOD ep20 infer all981 Train-03 frames. Original native pipeline/NMS retained. EOOD highest-score native candidate, stable first index for ties; do not change max_per_img. Reuse previously sealed B/M predictions, do not rerun them.',
        numerical=dict(cudnn_benchmark=False, cudnn_deterministic=True,
                       matmul_allow_tf32=False, cudnn_allow_tf32=False, batch=1, input_dtype='float32'),
        evidence='Compare original fixed frontends through unchanged Raw-opt. Prediction streams close before new metric truth parsing. No detector/head update, refit, selection, DINO, reliability filtering, TEST, Fixed-dev or Unknown reads.',
        reporting='Output/numeric coverage, absolute depth difference/MAE/median/P95/bias/full-frame descriptive coverage; center has output-conditional and all-frame denominators. Same q support, actual sx/sy and raw w/h-angle. Offline GT substitution is an error-propagation diagnostic, not deployable output.',
        limitation='Train-03 calibration diagnostic, not independently held-out final or real-port metric depth precision. Existing B/M are historical sealed outputs, not same-run inference with the new baselines.')


def literal(node):
    # Match the reviewed checkpoint-meta contract; never execute saved config.
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict' and not node.args:
        if any(k.arg is None for k in node.keywords):
            raise ValueError('Saved config expansion forbidden')
        return {k.arg:literal(k.value) for k in node.keywords}
    if isinstance(node, ast.Dict):
        return {literal(k):literal(v) for k,v in zip(node.keys,node.values)}
    if isinstance(node,(ast.List,ast.Tuple)):
        return [literal(v) for v in node.elts]
    return ast.literal_eval(node)


def plain(value):
    if isinstance(value,dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [plain(v) for v in value]
    return value


def checkpoint_meta(meta,cfg,epoch):
    fields=('model','data','optimizer','optimizer_config','lr_config','runner','load_from','resume_from')
    saved={}
    for node in ast.parse(meta.get('config','')).body:
        if (isinstance(node,ast.Assign) and len(node.targets)==1
                and isinstance(node.targets[0],ast.Name) and node.targets[0].id in fields):
            key=node.targets[0].id
            if key in saved: raise ValueError('Duplicate saved config field')
            saved[key]=literal(node.value)
    if saved!=plain({k:cfg[k] for k in fields}) or meta.get('epoch')!=epoch or meta.get('seed')!=0:
        raise ValueError('Requires original selected baseline and resolved checkpoint config')
    return dict(epoch=epoch,seed=0,config_text_sha256=hashlib.sha256(meta['config'].encode()).hexdigest())


def assignments(path):
    result={}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node,ast.Expr) and isinstance(node.value,(ast.Str,ast.Constant)):
            if isinstance(ast.literal_eval(node.value),str): continue
        if (not isinstance(node,ast.Assign) or len(node.targets)!=1
                or not isinstance(node.targets[0],ast.Name) or node.targets[0].id in result):
            raise ValueError('Unexpected archived config statement')
        result[node.targets[0].id]=node.value
    return result


def selection_contract(root,arm,selection):
    spec=BASELINES[arm]; key='epoch_%d'%spec['epoch']; stem=spec['stem']
    if (selection.get('evidence_role')!='source_val_checkpoint_selection'
            or selection.get('selected_checkpoint')!=key
            or selection.get('config_sha256')!=spec['historical_config_sha256']
            or selection['all_checkpoints'][key]['checkpoint_sha256']!=spec['checkpoint_sha256']):
        raise ValueError('Original source-VAL selection differs; never reselect')
    child=assignments(Path(root)/ARCHIVE/(stem+'_port_day2night_seq06_v1.py.txt'))
    previous=assignments(Path(root)/ARCHIVE/(stem+'_port_day2night_v1.py.txt'))
    current=assignments(Path(root)/spec['config']); fields={'_base_','data_root','data'}
    if (set(child)!={'_base_','work_dir'} or literal(child['_base_'])!=['./'+stem+'_port_day2night_v1.py']
            or literal(child['work_dir'])!='work_dirs/'+stem+'_port_day2night_seq06_v1'
            or set(previous)!=fields or set(current)!=fields|{'work_dir'}
            or any(ast.dump(previous[k])!=ast.dump(current[k]) for k in fields)):
        raise ValueError('Reviewed archived/current config equivalence differs')
    return dict(selected_epoch=spec['epoch'],selection_on_new_data=False,archived_overrides_equivalent=True)


def top1(candidates,arm):
    """Use native outputs only, without geometry gating or new score filtering."""
    if arm not in BASELINES: raise ValueError('Unknown new baseline')
    for box in candidates:
        if (len(box)!=6 or not all(math.isfinite(v) for v in box)
                or min(box[2:4])<=0 or not 0<=box[5]<=1):
            raise ValueError('Invalid native OBB output')
    if arm=='symeood' and len(candidates)>1:
        raise ValueError('Original SymEOOD top1 output changed')
    if not candidates: return None,None
    index=max(range(len(candidates)),key=lambda i:candidates[i][5])
    return list(candidates[index]),index


def evaluate_one(predictions,truths,manifest,calibration):
    paired=[dict(frame_id=p['frame_id'],b=p['box'],midpoint=p['box'],accepted=None) for p in predictions]
    rows,summary=original.evaluate(paired,truths,manifest,calibration)
    return rows,summary['groups']['b']


def direct_errors(rows,key='b'):
    finite=[r for r in rows if r[key]['depth']['z_m'] is not None]
    errors=[r[key]['depth']['z_m']-r['truth_z_m'] for r in finite]
    absolute=[abs(v) for v in errors]
    n=len(rows)
    return dict(frame_count=n,numeric_count=len(finite),median_abs_error_m=original.percentile(absolute,.5),
        p90_abs_error_m=original.percentile(absolute,.9),
        overestimate_count=sum(v>0 for v in errors),underestimate_count=sum(v<0 for v in errors),
        abs_error_coverage={str(t):dict(count=sum(v<=t for v in absolute),denominator=n,
            fraction=sum(v<=t for v in absolute)/n) for t in (.5,1.,2.)},
        relative_error_le_10_percent_count=sum(abs(r[key]['depth']['z_m']-r['truth_z_m'])/r['truth_z_m']<=.1 for r in finite))


def propagation(rows,calibration,key='b'):
    """Exact signed-error algebra, not additive attribution of MAE or a model fix."""
    beta=calibration['parameters']['beta']; terms={k:[] for k in ('gt_formula','short_scale','ratio_q','interaction')}
    short_z=[];ratio_z=[];truth=[];max_residual=0.;failures=0
    for r in rows:
        delivered=r[key]['depth'];gt=r['gt_obb']['depth']
        if delivered['z_m'] is None:
            failures+=1;continue
        zp,zg,q,qg=delivered['z_m'],gt['z_m'],delivered['q_signed'],gt['q_signed']
        try:
            zs=zg/(1+r[key]['geometry']['short_relative_error'])
            zq=zg*math.exp(beta*(q*q-qg*qg))
        except (OverflowError,ZeroDivisionError):
            failures+=1;continue
        if not all(math.isfinite(v) for v in (zs,zq)):
            failures+=1;continue
        values=(zg-r['truth_z_m'],zs-zg,zq-zg,zp-zs-zq+zg)
        for name,value in zip(terms,values):terms[name].append(value)
        max_residual=max(max_residual,abs(math.fsum(values)-(zp-r['truth_z_m'])))
        short_z.append(zs);ratio_z.append(zq);truth.append(r['truth_z_m'])
    n=len(truth)
    return dict(role='offline_GT_substitution_diagnostic_not_deployable',count=n,diagnostic_uncomputable_count=failures,
        signed_error_component_mean_m={name:math.fsum(v/n for v in values) if n else None for name,values in terms.items()},
        max_signed_identity_residual_m=max_residual if n else None,
        short_scale_only_diagnostic=original.error_metrics(short_z,truth),
        ratio_only_diagnostic=original.error_metrics(ratio_z,truth),
        note='Components add to signed bias on common diagnostic frames, not to MAE. GT substitutions never alter delivered model boxes or enter inference.')
