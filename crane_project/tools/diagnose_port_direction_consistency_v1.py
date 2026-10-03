#!/usr/bin/env python3
"""Eight fixed TRAIN views; explicit direction evidence only, no detector update.

Paired angle-only MLP copies: zero-feature control / predicted line evidence /
annotation-template numerical reference. The third arm is OFFLINE ONLY.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import fit_port_reliability_candidates_v1 as parent
from crane_project.utils import port_direction_consistency_v1 as direction
from crane_project.utils.port_structure_reliability_v1 import (
    assert_detector_frozen,map_points,response_targets)
branch,train,prior=parent.branch,parent.train,parent.prior
VERSION='port_direction_consistency_v1'
PROTOCOL=ROOT/'crane_project/tools/port_direction_consistency_v1_protocol.json'
MANIFEST=ROOT/'crane_project/tools/port_direction_consistency_v1_sources.json'
ARMS=('raw_zero_control','predicted_relation','ideal_template_reference')


def checked_sources():
    old,oldprotocol,training_protocol=parent.checked_sources()
    p=json.loads(PROTOCOL.read_text());m=json.loads(MANIFEST.read_text())
    if (m['protocol']!=VERSION or p['protocol']!=VERSION
            or m['sources']!={name:branch.sha(ROOT/name) for name in m['sources']}
            or m['protocol_sha256']!=branch.sha(PROTOCOL)
            or m['parent_candidate_manifest_sha256']!=old['manifest_sha256']
            or p['feature_names']!=list(direction.FEATURE_NAMES)
            or p['cases']!=oldprotocol['cases'] or p['controls']!=list(ARMS)
            or p['steps']!=2000 or p['lr']!=.001 or p['val_or_test_read'] is not False):
        raise ValueError('Reviewed direction source/protocol contract differs')
    return dict(manifest_sha256=branch.sha(MANIFEST),parent_candidate=old),p,oldprotocol,training_protocol


def checked_evidence(args,sources,protocol,oldprotocol,require_cache=False):
    if branch.sha(args.candidate_report)!=protocol['candidate_report_sha256']:
        raise ValueError('Requires the exact reviewed fixed TRAIN candidate fit report')
    report=json.loads(args.candidate_report.read_text())
    if (report['sources']!=sources['parent_candidate'] or report['protocol']!=oldprotocol
            or report['status']!='FIXED_TRAIN_CANDIDATE_FIT_COMPLETE_REVIEW_REQUIRED'
            or report['detector_updates']!=0 or report['original_head_updates']!=0
            or report['val_or_test_read'] or report['deployable_weights_saved']):
        raise ValueError('Completed candidate evidence differs')
    probe,oldrows=parent.checked_cases(args,oldprotocol)
    cache=None
    if args.candidate_cache.is_file():
        cache=parent.load_cache(args.candidate_cache,sources['parent_candidate'],oldprotocol)
        if (cache['proof']!=report['proof'] or cache['frozen_b_state']!=report['frozen_b_state']
                or cache['heads_before']!=report['heads_before'] or cache['heads_after']!=report['heads_after']):
            raise ValueError('Candidate descriptor cache and report differ')
        rows=cache['arms']['structure']['rows']
        if len(rows)!=8 or len(cache['replay'])!=8:
            raise ValueError('Requires the same eight cached views')
        for row,old,before in zip(rows,oldrows,report['results']['structure']['before']['views']):
            if (row['image']!=old['case']['image'] or row['domain']!=old['case']['domain']
                    or row['genuine_count']!=1 or row['probe_names']!=old['probe_names']
                    or row['descriptor'].shape!=(14,539)
                    or not torch.isfinite(row['descriptor']).all()
                    or not torch.equal(row['target'],torch.tensor(before['targets']))
                    or not torch.equal(row['mask'],torch.tensor(before['masks']))
                    or not torch.allclose(row['initial_qualities'],torch.tensor(before['qualities']),atol=1e-6,rtol=0)):
                raise ValueError('Cached structure descriptor identity differs')
    elif require_cache:
        raise FileNotFoundError('Reuse the completed candidate descriptors.pt and its .sha.json: '+str(args.candidate_cache))
    return report,probe,oldrows,cache


def unsigned_error(a,b):
    return abs((a-b+math.pi/2)%math.pi-math.pi/2)*180/math.pi


def collect(args,sources,protocol,oldprotocol,training_protocol,report,probe,oldrows,oldcache):
    inputs,proof=train.checked_inputs(args.input_snapshot,args.train_cache,args.structure_report)
    if proof!=oldcache['proof']: raise ValueError('Original TRAIN proof differs')
    payload=parent.parent.previous.base.fixed_bundle(args.branch_checkpoint,
        sources['parent_candidate']['parent']['training_sources'],training_protocol)
    if branch.sha(args.branch_checkpoint)!=oldprotocol['checkpoint_sha256']:
        raise ValueError('Requires completed original epoch08')
    cfg=prior.check_cfg();detector,runtime=train.build_runtime(cfg,args.b_checkpoint,args.gpu)
    frozen=prior.state_digest(detector)
    if frozen!=oldcache['frozen_b_state'] or frozen!=payload['frozen_b_state'] or frozen!=proof['b_cache_state']:
        raise ValueError('Frozen B identity differs')
    arms=branch.make_arms('cuda:'+str(args.gpu))
    if branch.architecture(arms)!=payload['contract']['architecture']:
        raise ValueError('Architecture differs')
    branch.load_bundle(payload,arms);del payload
    states={n:prior.state_digest(a) for n,a in arms.items()}
    if states!=oldcache['heads_before']: raise ValueError('Original heads differ')
    for arm in arms.values():
        arm.eval()
        for p in arm.parameters():p.requires_grad_(False)
    indexed=train.datasets_for(deepcopy(cfg.data.train),inputs)
    cache=dict(sources=sources,protocol=protocol,old_candidate_cache_sha256=branch.sha(args.candidate_cache),
        candidate_report_sha256=branch.sha(args.candidate_report),initial=oldcache['arms']['structure']['initial'],
        rows={name:[] for name in ARMS},features=[],proof=proof,runtime=runtime,replay=[])
    torch.cuda.reset_peak_memory_stats(args.gpu)
    for old,row,original_replay in zip(oldrows,oldcache['arms']['structure']['rows'],oldcache['replay']):
        case=old['case'];source=inputs[case['dataset_index']]
        values,replay,raw_state=train.training_view(detector,indexed[case['dataset_index']],source,args.gpu,case['view_seed'])
        p3,boxes,scores,meta,target,mask,n,_,_=values
        for key in ('image','input_sha256','view_seed','view_image_sha256','scale_factor','img_shape','pad_shape','flip','flip_direction'):
            if replay[key]!=old['replay'][key]: raise ValueError('TRAIN view replay differs: '+key)
        bparity=parent.parent.previous.prediction_difference(replay['genuine_b_original'],old['replay']['genuine_b_original'])
        if not bparity['all_six_close'] or prior.tensor_sha(p3)!=original_replay['p3_sha256']:
            raise ValueError('B output/P3 differs from the fixed candidate cache')
        if n!=1 or not torch.equal(target.cpu(),row['target']) or not torch.equal(mask.cpu(),row['mask']):
            raise ValueError('Original targets/masks differ')
        descriptor,q,mlp,delta=parent.fitutil.capture(arms['structure'],p3,boxes,scores,meta)
        if (not torch.allclose(descriptor,row['descriptor'],atol=1e-6,rtol=0)
                or not torch.allclose(q,row['initial_qualities'],atol=1e-6,rtol=0)):
            raise ValueError('Original descriptor/quality replay differs')
        with torch.no_grad():
            probabilities=arms['structure'](p3,boxes,scores,meta)['response_logits'].sigmoid().cpu().numpy()[0]
        model_boxes=boxes.detach().cpu()
        # This online computation has NO annotation or label input.
        actual,actual_details=direction.direction_features(probabilities[1],model_boxes,meta)
        # Only this separate offline numerical reference reads TRAIN annotations.
        axis=map_points(torch.as_tensor(train.axis_for(source),dtype=p3.dtype,device=p3.device),meta)
        ideal,_=response_targets(axis,p3.shape[-2:],meta)
        reference,ideal_details=direction.direction_features(ideal.cpu().numpy()[0,1],model_boxes,meta)
        axis_np=axis.cpu().numpy(); theta=math.atan2(*(axis_np[1]-axis_np[0])[::-1])
        for name,extra in zip(ARMS,(torch.zeros_like(actual),actual,reference)):
            saved={key:row[key] for key in ('image','domain','target','mask','genuine_count','probe_names')}
            saved['descriptor']=torch.cat((row['descriptor'],extra),dim=1)
            cache['rows'][name].append(saved)
        def error(details):
            value=details[1]['reference_angle_rad']
            return None if value is None else unsigned_error(value,theta)
        cache['features'].append(dict(image=case['image'],domain=case['domain'],candidate_names=['genuine_B']+row['probe_names'],
            predicted_features=actual.tolist(),ideal_features=reference.tolist(),
            predicted_analytic_direction_scores=direction.analytic_direction_scores(actual),
            ideal_analytic_direction_scores=direction.analytic_direction_scores(reference),
            analytic_score_role='offline 3deg-reference diagnostic; not a deployment confidence or gate',
            predicted_details=actual_details,
            ideal_details=ideal_details,predicted_reference_error_at_GT_context_deg=error(actual_details),
            ideal_reference_error_at_GT_context_deg=error(ideal_details),
            annotation_role='native_axis' if source['domain']=='real' else 'Webots_OBB_axis_proxy',
            gt_comparison_role='offline diagnostic only, never an online feature'))
        feats,raw,metas=raw_state
        with torch.no_grad():
            repeated=prior.flatten_prediction(detector.simple_test_from_features(feats,metas,rescale=False))
        if not np.array_equal(raw,repeated): raise ValueError('Side computation changed raw B output')
        cache['replay'].append(dict(image=case['image'],b_parity=bparity,p3_sha256=prior.tensor_sha(p3),
            b_raw_exact_before_after=True,original_quality_cpu_parity_max_abs=delta))
        print('Direction evidence TRAIN',len(cache['replay']),'/ 8',case['image'],
              'reference error predicted/ideal',error(actual_details),error(ideal_details),flush=True)
        del values,raw_state,p3,boxes,scores,target,mask,mlp,probabilities,ideal,axis,feats,raw,metas
    assert_detector_frozen(detector)
    if (prior.state_digest(detector)!=frozen or {n:prior.state_digest(a) for n,a in arms.items()}!=states
            or any(p.grad is not None for a in arms.values() for p in a.parameters())):
        raise ValueError('Frozen detector/heads changed or received gradients')
    if train.checked_inputs(args.input_snapshot,args.train_cache,args.structure_report)[1]!=proof:
        raise ValueError('TRAIN evidence changed')
    cache.update(frozen_b_state=frozen,heads_before=states,heads_after=states,detector_updates=0,
        original_head_updates=0,max_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20,
        max_reserved_mib=torch.cuda.max_memory_reserved(args.gpu)/2**20)
    return cache


def save_cache(path,cache):
    direction.atomic_file(path,lambda stream:torch.save(cache,stream))
    direction.atomic_json(path.with_suffix('.sha.json'),dict(sha256=branch.sha(path),sources=cache['sources'],protocol=cache['protocol']))


def load_cache(path,sources,protocol):
    marker=json.loads(path.with_suffix('.sha.json').read_text())
    if marker!=dict(sha256=branch.sha(path),sources=sources,protocol=protocol):
        raise ValueError('Direction cache identity differs')
    value=torch.load(str(path),map_location='cpu')
    if value['sources']!=sources or value['protocol']!=protocol:
        raise ValueError('Direction cache source/protocol differs')
    if (tuple(value['rows'])!=ARMS or any(len(rows)!=8 for rows in value['rows'].values())
            or value['candidate_report_sha256']!=protocol['candidate_report_sha256']
            or value['detector_updates']!=0 or value['original_head_updates']!=0):
        raise ValueError('Direction cache view/evidence contract differs')
    return value


def fit_cache(cache,out_dir):
    torch.set_num_threads(1)
    original=branch.make_arms('cpu')['structure'].quality
    original.load_state_dict(cache['initial'],strict=True)
    before=prior.state_digest(original)
    initial=direction.AngleOnlyQuality(original)
    results={};start_scores=None
    with (out_dir/'progress.jsonl').open('x') as stream:
        for name in ARMS:
            rows=cache['rows'][name]
            with torch.no_grad():
                scores=[initial(r['descriptor']).sigmoid() for r in rows]
                originals=[original(r['descriptor'][:,:initial.raw_dim]).sigmoid() for r in rows]
            if any(not torch.allclose(a,b,atol=1e-6,rtol=0) for a,b in zip(scores,originals)):
                raise ValueError('Initial direction quality differs from original ep08')
            if start_scores is None:start_scores=scores
            elif any(not torch.equal(a,b) for a,b in zip(start_scores,scores)):
                raise ValueError('Paired controls do not start with identical quality')
            def notify(item):
                a=item['post_update'];p=a['pair_summary']['all:angle']
                record=dict(arm=name,step=item['step'],loss=a['loss'],direction_loss=a['direction_loss_contribution'],
                            angle_gap=p['mean_predicted_gap'],angle_correct=p['correct_sign'],angle_pairs=p['count'])
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                print(name,'step',item['step'],'angle gap',p['mean_predicted_gap'],'correct',str(p['correct_sign'])+'/'+str(p['count']),flush=True)
            result,fitted=direction.fit(initial,rows,steps=cache['protocol']['steps'],lr=cache['protocol']['lr'],
                                        milestones=cache['protocol']['milestones'],notify=notify)
            result.update(final_state=prior.state_digest(fitted),parameters=sum(p.numel() for p in initial.angle.parameters()),
                uses_GT_in_features=name=='ideal_template_reference',
                role='offline_numerical_reference' if name=='ideal_template_reference' else 'paired_TRAIN_diagnostic')
            results[name]=result
    if prior.state_digest(original)!=before:
        raise ValueError('Original CPU head changed')
    if checked_sources()[:2]!=(cache['sources'],cache['protocol']):
        raise ValueError('Sources changed during fit')
    report={key:value for key,value in cache.items() if key not in ('rows','initial')}
    report.update(status='FIXED_TRAIN_DIRECTION_CONSISTENCY_COMPLETE_REVIEW_REQUIRED',results=results,
        center_size_quality_exact=True,all_controls_start_scores_exact=True,deployable_weights_saved=False,
        val_or_test_read=False,test_repeatedly_exposed=True,formal_training=False,
        interpretation='Selected TRAIN only. Ideal arm reads annotations and is never deployable. Same structure descriptor/stem for paired controls; only explicit direction input differs. No generalization or unique root-cause claim.')
    direction.atomic_json(out_dir/'direction_report.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('check','run','fit-cache'),default='check')
    p.add_argument('--candidate-cache',type=Path,default=Path('work_dirs/port_reliability_candidate_fit_v1/descriptors.pt'))
    p.add_argument('--candidate-report',type=Path,default=Path('work_dirs/port_reliability_candidate_fit_v1/fit_report.json'))
    p.add_argument('--probe-dir',type=Path,default=Path('work_dirs/port_reliability_mechanism_v1_probe_path_fix_v1'))
    p.add_argument('--branch-checkpoint',type=Path,default=Path('work_dirs/port_reliability_branches_v1_train/epoch_08.pth'))
    p.add_argument('--input-snapshot',type=Path,default=Path('work_dirs/port_reliability_train_support_v1/train_input_snapshot.json'))
    p.add_argument('--train-cache',type=Path,default=Path('work_dirs/port_reliability_train_support_v1_cache'))
    p.add_argument('--structure-report',type=Path,default=Path('work_dirs/port_reliability_readiness_v1_structure_complete/train_structure_check.json'))
    p.add_argument('--b-checkpoint',type=Path,default=prior.CHECKPOINT)
    p.add_argument('--direction-cache',type=Path)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0)
    args=p.parse_args();os.chdir(ROOT)
    if args.out_dir.exists():raise FileExistsError('Preserve existing evidence; choose a new --out-dir: '+str(args.out_dir))
    sources,protocol,oldprotocol,training_protocol=checked_sources()
    if args.mode=='fit-cache':
        if args.direction_cache is None:p.error('fit-cache requires --direction-cache')
        cache=load_cache(args.direction_cache,sources,protocol)
        args.out_dir.mkdir(parents=True);fit_cache(cache,args.out_dir)
    else:
        report,probe,rows,oldcache=checked_evidence(args,sources,protocol,oldprotocol,require_cache=args.mode=='run')
        if args.mode=='check':
            direction.atomic_json(args.out_dir/'input_check.json',dict(sources=sources,protocol=protocol,
                status='DIRECTION_INPUT_CHECK_PASS' if oldcache is not None else 'DIRECTION_REPORT_ONLY_CHECK_PASS_CACHE_STILL_REQUIRED',
                candidate_cache_verified=oldcache is not None,inference=False,optimizer_steps=0,
                limitation='Frozen checkpoints, actual TRAIN bytes, numerical replay verified in run mode.'))
        else:
            cache=collect(args,sources,protocol,oldprotocol,training_protocol,report,probe,rows,oldcache)
            if checked_sources()!=(sources,protocol,oldprotocol,training_protocol):
                raise ValueError('Sources changed during collection')
            checked_evidence(args,sources,protocol,oldprotocol,require_cache=True)
            if branch.sha(args.candidate_cache)!=cache['old_candidate_cache_sha256']:
                raise ValueError('Original candidate cache bytes changed during collection')
            args.out_dir.mkdir(parents=True);path=args.out_dir/'direction_cache.pt'
            save_cache(path,cache);fit_cache(load_cache(path,sources,protocol),args.out_dir)
    print('Saved',args.out_dir,flush=True)


if __name__=='__main__':main()
