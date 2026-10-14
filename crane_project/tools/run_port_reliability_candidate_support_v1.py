#!/usr/bin/env python3
"""Full TRAIN candidate extraction -> prerequisite screen -> optional paired fit/VAL."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.utils import port_reliability_candidate_support_v1 as core
from crane_project.tools import run_port_reliability_conditional_evidence_v1 as io


def load(allow_absent=False):
    p=json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_protocol.json')).read_text())
    s=json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_sources.json')).read_text())
    if p['settings']!=core.SETTINGS or p['fit_settings']!=core.parent.SETTINGS:raise ValueError('Prespecified contract differs')
    pins=dict(p['input_pins'],**p['frontend_pins'],**s['sources']);absent=[]
    for name,digest in pins.items():
        if allow_absent and name in p['frontend_pins'] and not (ROOT/name).exists():absent.append(name);continue
        if io.sha(ROOT/name)!=digest:raise ValueError('Source/input changed: '+name)
    previous=ROOT/p['previous_result'];report=json.loads((previous/'report.json').read_text())
    review=json.loads((previous/'server_review.json').read_text())
    if not review['passed'] or review['report_sha256']!=io.sha(previous/'report.json'):raise ValueError('Previous results not reviewed')
    if report['TEST_opened'] or report['candidate_adopted'] or report['formal_policy_changed']:raise ValueError('Previous role differs')
    return p,pins,absent,report


def save_rows(path,rows):
    with gzip.open(path,'xt') as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')


def collect(rows,role,gpu,out,p):
    import torch
    from mmcv.parallel import collate,scatter
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as frozen
    from crane_project.utils.port_reliability_candidate_support_v1_torch import trace_candidates
    if role not in ('TRAIN','VAL'):raise ValueError('No TEST role')
    expected=2558 if role=='TRAIN' else 887
    if len(rows)!=expected or len({r['image'] for r in rows})!=expected:raise ValueError('Full unique role required')
    torch.set_num_threads(1);torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    cfg=frozen.g.check_cfg();detector=frozen.build_detector(cfg,gpu);state=frozen.g.state_digest(detector)
    cache=json.loads((ROOT/p['cache_manifest']).read_text())
    if state!=cache['detector_state']:raise ValueError('Frozen B state differs')
    dataset=frozen.dataset_for(cfg,role.lower(),1.)
    lookup={r['image']:r for r in rows};names=[Path(i['filename']).stem for i in frozen.infos(dataset)]
    if len(names)!=expected or set(names)!=set(lookup):raise ValueError('Dataset membership differs')
    refs={r['image']:r for r in io.read(ROOT/p['reference_rows'][role])}
    manifest=json.loads((ROOT/p['manifest']).read_text())
    sources={r['id']:r for r in manifest['records'] if r['split'] in (('train','train_sim') if role=='TRAIN' else ('val',))}
    if set(sources)!=set(lookup):raise ValueError('Manifest role mismatch')
    result=deepcopy(rows);dest={r['image']:r for r in result};maxerr=np.zeros(6);timings=[]
    for i,name in enumerate(names):
        row=lookup[name];ref=refs[name]
        for key in ('pred','gt','image_size','split','reliability_role'):
            if ref[key]!=row[key]:raise ValueError('Pinned M frame differs: '+name+' '+key)
        spec=sources[name];image_path=frozen.g.ready.DATA/spec['images']['path']
        if io.sha(image_path)!=spec['images']['sha256']:raise ValueError('Image bytes changed: '+name)
        if io.sha(frozen.g.ready.DATA/spec['annfiles']['path'])!=spec['annfiles']['sha256']:raise ValueError('Annotation identity changed: '+name)
        # Dataset uses deterministic test pipeline: no GT tensor enters forward.
        value=scatter(collate([dataset[i]],samples_per_gpu=1),[gpu])[0]
        image,metas=value['img'][0],value['img_metas'][0]
        if len(value['img'])!=1 or image.shape!=(1,3,1024,1024) or len(metas)!=1 or Path(metas[0]['filename']).stem!=name or metas[0].get('flip',False):raise ValueError('Single deterministic view required')
        if list(metas[0]['ori_shape'][:2][::-1])!=row['image_size']:raise ValueError('Original image size differs')
        frozen.g.checked_meta(metas[0]);frozen.g.assert_detector_frozen(detector)
        prediction,pool,chosen,trace=trace_candidates(detector,image,metas)
        if trace['score_thr']!=.05:raise ValueError('Frozen native score threshold changed')
        b=frozen.g.flatten_prediction(prediction)
        if len(b)!=int(row['pred'] is not None):raise ValueError('B/M missing identity differs')
        if len(b):
            expected_b=np.asarray(ref['b_original']);delta=np.abs(b[0]-expected_b)
            delta[4]=abs((b[0,4]-expected_b[4]+np.pi/2)%np.pi-np.pi/2);maxerr=np.maximum(maxerr,delta)
            if not np.allclose(b[0,:4],expected_b[:4],atol=5e-4,rtol=2e-5) or delta[4]>2e-6 or b[0,5]!=expected_b[5] or b[0,5]!=row['pred'][5]:raise ValueError('Original native B replay differs: '+name)
        start=time.monotonic();ev=core.evidence(row['pred'],pool,chosen);post=time.monotonic()-start
        # Store whole post-filter pool for independent source-index/geometric review.
        dest[name].update(candidate_evidence=ev,candidate_pool=pool.tolist(),candidate_chosen=chosen,
            candidate_trace=dict(trace,postprocess_seconds=post,image_sha256=spec['images']['sha256']),b_original=ref['b_original'])
        timings.append((trace['forward_seconds'],post))
        if i%64==0 or i==len(names)-1:print('Candidate source',role,i+1,'/',expected,'surviving',len(pool),'neighbors',len(ev['neighbors']),flush=True)
        del value,image,metas,prediction,b,pool
    if frozen.g.state_digest(detector)!=state:raise ValueError('Detector changed')
    del detector,dataset;torch.cuda.empty_cache()
    save_rows(out/('candidates_'+role+'.jsonl.gz'),result)
    return result,dict(frames=expected,detector_state=state,detector_inferences=expected,
        detector_updates=0,GT_online=False,M_reused_from_pinned_cache=True,max_B_difference=maxerr.tolist(),
        mean_forward_seconds=float(np.mean(timings,axis=0)[0]),mean_statistics_seconds=float(np.mean(timings,axis=0)[1]),
        timing_includes_python_trace=True,not_deployment_latency=True)


def run(args):
    p,pins,absent,previous=load(args.allow_absent_frontend)
    if args.check_only:print('Candidate support contract/source pins verified; no inference');return
    if absent:raise ValueError('Actual execution requires both pinned frontend checkpoints')
    if args.out is None or args.out.is_symlink() or args.out.resolve().parent.parent!=(ROOT/'work_dirs'/core.VERSION).resolve():raise ValueError('Use work_dirs/version/RUN_ID/result')
    args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    train=io.read(ROOT/p['previous_result']/'scored_TRAIN.jsonl.gz')
    if any(r['split'] not in ('train','train_sim') or r['reliability_role']!='train' for r in train):raise ValueError('TRAIN-only collection')
    train,proof=collect(train,'TRAIN',args.gpu,args.out,p)
    screen=core.screen(train);io.write(args.out/'TRAIN_screen.json',screen)
    report=dict(protocol=core.VERSION,settings=core.SETTINGS,input_pins=pins,absent_frontend=[],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        train_proof=proof,screen=screen,fit_calls=0,VAL_opened=False,TEST_opened=False,
        formal_policy_changed=False,candidate_adopted=False,detector_updates=0,
        limitations=['O2O alternative boxes are not independently trusted measurements.',
            'B candidates supervise final M correctness; no claim of independent detector errors.',
            'TRAIN directions/screen are development evidence, not statistical guarantees.',
            'Previously exposed VAL is not independent validation.'])
    if not screen['passed']:
        report['status']='TRAIN_EVIDENCE_FAILED_STOP'
    else:
        models={name:core.fit(train,dim,previous['base_models']['fusion4']['weights'][0]) for name,dim in ((core.CONTROL,4),(core.METHOD,6))}
        if models[core.CONTROL]['train_images']!=models[core.METHOD]['train_images']:raise ValueError('Paired fit identities differ')
        io.write(args.out/'models.json',models)  # Frozen before VAL rows/data loader.
        scored_train=core.score(train,models)
        val=io.read(ROOT/p['previous_result']/'scored_VAL.jsonl.gz')
        if set(r['image'] for r in train)&set(r['image'] for r in val):raise ValueError('Role overlap')
        val,vproof=collect(val,'VAL',args.gpu,args.out,p);val=core.score(val,models)
        cutoffs={name:core.calibrate(val,name,previous['calibration']['risk_le']) for name in models}
        io.write(args.out/'calibration.json',cutoffs)
        save_rows(args.out/'scored_TRAIN.jsonl.gz',scored_train);save_rows(args.out/'scored_VAL.jsonl.gz',val)
        evaluations={role:core.evaluate(rows,cutoffs) for role,rows in (('TRAIN',scored_train),('VAL',val))}
        gate=core.gate(evaluations['VAL'])
        report.update(models=models,calibration=cutoffs,roles=evaluations,gate=gate,val_proof=vproof,
            fit_calls=2,VAL_opened=True,models_saved_before_VAL=True,
            status='EVIDENCE_PASS_NO_ADOPTION' if gate['passed'] else 'VAL_EVIDENCE_FAILED_STOP')
    for name,digest in pins.items():
        if io.sha(ROOT/name)!=digest:raise ValueError('Frozen source/input changed')
    report['elapsed_seconds']=time.monotonic()-start
    io.write(args.out/'report.json',report);io.write(args.out/'completion.json',dict(status=report['status'],TEST_opened=False))
    print(report['status'],'TRAIN screen',screen['failures'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path);parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--allow-absent-frontend',action='store_true')
    run(parser.parse_args())
