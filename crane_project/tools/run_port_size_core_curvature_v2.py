#!/usr/bin/env python3
"""check -> TRAIN smoke -> exact failed-frame replay -> paired epoch04 -> assess.

Only new reference heads learn. Existing B/midpoint/policy/readers stay pinned.
95% is a comparison point, never a final deployment-accuracy objective. No TEST.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import eval_port_midpoint_size_template_v1 as parent
from crane_project.utils import port_size_core_curvature_v2 as core

migration = parent.migration
reference = migration.reference
base = parent.base
simple = parent.simple
PROTOCOL = ROOT/'crane_project/tools/port_size_core_curvature_v2_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_size_core_curvature_v2_sources.json'
STATUSES = dict(check='CORE_CURVATURE_NUMERICAL_CHECK_PASS',
    smoke='CORE_CURVATURE_TRAIN_PRECHECK_SAVE_RELOAD_PASS_DISCARDED',
    replay='CORE_CURVATURE_FAILURE_FRAME_REPLAY_PASS_NO_OPTIMIZER_UPDATE',
    train='CORE_CURVATURE_PAIRED_EPOCH04_COMPLETE_REVIEW_REQUIRED',
    assess='CORE_CURVATURE_PAIRED_ASSESSMENT_COMPLETE_REVIEW_REQUIRED')


def checked_sources():
    protocol = json.loads(PROTOCOL.read_text()); manifest = json.loads(SOURCES.read_text())
    actual = {p:base.sha(ROOT/p) for p in manifest['sources']}
    if (actual != manifest['sources'] or manifest['protocol'] != core.VERSION
            or protocol['protocol'] != core.VERSION
            or manifest['protocol_sha256'] != base.sha(PROTOCOL)
            or manifest['parent_template_sources_sha256'] != base.sha(parent.SOURCES)
            or protocol['coefficients'] != core.COEFFICIENTS or protocol['steps_per_arm'] != 1536
            or protocol['epochs'] != 4 or protocol['test_read']
            or protocol['comparison_only'] is not True
            or (protocol['seed'],protocol['fit_frames'],protocol['holdout_frames'],protocol['guard_frames'],
                protocol['initial_gradient_views'],protocol['smoke_updates_per_arm'],
                protocol['component_gradient_every_train_slots']) != (1701,384,432,32,14,4,64)
            or protocol['optimizer'] != {'type':'Adam','lr':.001,'weight_decay':0.,'clip_norm':10.}
            or protocol.get('gradient_decomposition_check') != core.GRADIENT_CHECK
            or protocol.get('failure_frame_replay') != core.FAILURE_REPLAY
            or protocol['gradient_guard'] != {
                'median_new_to_original_min':.01,'median_new_to_original_max':10.,
                'median_clip_retention_min':.1,
                'role':'prespecified_broad_engineering_guard_not_optimality_no_coefficient_search'}
            or any(protocol[k] for k in ('detector_updates','midpoint_updates','baseline_policy_updates',
                                         'final_deployment_policy_created'))):
        raise ValueError('New source/protocol or frozen parent identity differs')
    return protocol, dict(manifest_sha256=base.sha(SOURCES), sources=actual)


def prepare(args):
    protocol, sources = checked_sources()
    old = parent.prepare(args)  # exact reviewed B24, midpoint23, simple, legacy ref04
    previous = old[1]
    split = previous[1][6]
    if len(split['fit']) != 384 or len(split['holdout']) != 432 or len(split['guard']) != 32:
        raise ValueError('Fixed reference TRAIN roles differ')
    contract = dict(protocol=protocol, sources=sources, frozen_parent=old[4],
        fit_images_sha256=simple.fingerprint([r['image'] for r in split['fit']]),
        test_read=False, GT_online=False, comparison_only=True,
        b_and_simple_have_seen_full_detector_TRAIN=True)
    return old, contract


def finish(args, prepared, status):
    if prepare(args)[1] != prepared[1]: raise ValueError('Inputs/sources changed during stage')
    artifacts = {p.name:base.sha(p) for p in sorted(args.out_dir.iterdir()) if p.is_file()}
    base.write_new(args.out_dir/'completion.json', dict(status=status, contract=prepared[1],
        contract_sha256=simple.fingerprint(prepared[1]), artifacts=artifacts,
        mode=args.mode, test_read=False, detector_updates=0, midpoint_updates=0, policy_updates=0))
    print('Saved', args.out_dir, status, flush=True)


def stage(path, status, contract):
    if path is None: raise ValueError('Missing preceding stage report')
    path = Path(path)
    if (path.parent/'failure.json').exists(): raise ValueError('Failed stage cannot release training')
    report = json.loads(path.read_text()); completion = json.loads((path.parent/'completion.json').read_text())
    if (report.get('status') != status or completion['status'] != status
            or report['contract'] != contract or completion['contract'] != contract
            or completion['contract_sha256'] != simple.fingerprint(contract)
            or completion['artifacts'].get(path.name) != base.sha(path)):
        raise ValueError('Stage identity/status differs')
    for name, sha in completion['artifacts'].items():
        if Path(name).name != name or base.sha(path.parent/name) != sha:
            raise ValueError('Stage artifact differs')
    return report


def check(args, prepared):
    old, contract = prepared; fit = old[1][1][6]['fit']
    selected = reference.ideal_selection(old[1][1][6])
    records = []
    for row in selected:
        value = core.projection(row['gt'], reference.numeric_meta(row))
        record = dict(image=row['image'], domain=row['domain'], projection=core.public_projection(value))
        record['numerical'] = core.numerical_probe(value) if value['eligible'] else None
        records.append(record)
    eligible = []
    for row in fit:
        value = core.projection(row['gt'], reference.numeric_meta(row))
        eligible.append(dict(image=row['image'], domain=row['domain'], sequence=row['sequence'],
                             **core.public_projection(value)))
    passed = (all(r['numerical'] is None or r['numerical']['passed'] for r in records)
              and all(any(r['domain'] == d and r['eligible'] for r in eligible) for d in ('real','sim')))
    report = dict(status=STATUSES['check'] if passed else 'CORE_CURVATURE_NUMERICAL_CHECK_FAILED',
        contract=contract, records=records, fit_eligibility=eligible,
        eligibility_reasons=dict(Counter(r['reason'] for r in eligible)),
        actual_Torch_gradient_strength_verified=False, actual_image_learning_verified=False,
        only_fit_GT_used_for_numeric_gate=True, test_read=False)
    base.write_new(args.out_dir/'check_report.json', report)
    if not passed: raise ValueError('Fixed numerical loss checks failed')
    return STATUSES['check']


def models(args, old):
    formal, torch, detector, head, pipeline, _ = migration.online_modules(args, old[1], False)
    from crane_project.utils.port_size_core_curvature_v2_torch import SizeReference
    torch.manual_seed(1701); torch.cuda.manual_seed_all(1701)
    prototype = SizeReference().cuda(args.gpu)
    initial = {k:v.detach().cpu().clone() for k,v in prototype.state_dict().items()}
    initial_sha = base.state_digest(prototype)
    arms = {'a0':prototype, 'a1':SizeReference().cuda(args.gpu)}
    for model in arms.values(): model.load_state_dict(initial, strict=True)
    if any(base.state_digest(model) != initial_sha for model in arms.values()):
        raise ValueError('Pair initialization differs')
    return formal, torch, detector, head, pipeline, arms, initial_sha


def load(path, contract, role, arm, torch):
    path = Path(path); tag = json.loads(Path(str(path)+'.sha.json').read_text())
    if tag['sha256'] != base.sha(path) or tag['contract_sha256'] != simple.fingerprint(contract):
        raise ValueError('Candidate checkpoint bytes/contract differ')
    payload = torch.load(str(path), map_location='cpu')
    if (payload['protocol'] != core.VERSION or payload['contract'] != contract
            or payload['role'] != role or tag['role'] != role or payload['arm'] != arm
            or tag['epoch'] != payload['epoch']):
        raise ValueError('Candidate checkpoint identity differs')
    return payload


def save_reload(directory, model, optimizer, contract, arm, epoch, steps, role,
                initial_sha, frozen, features, torch):
    directory.mkdir(exist_ok=True)
    path = directory/('epoch_%02d.pth' % epoch)
    model.eval()
    with torch.no_grad(): expected = model(features[0]).detach().cpu()
    payload = dict(protocol=core.VERSION, contract=contract, role=role, arm=arm,
        epoch=epoch, steps=steps, initial_sha256=initial_sha, frozen=frozen,
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}, optimizer=optimizer.state_dict())
    reference.save_checkpoint(path, payload)
    restored = load(path, contract, role, arm, torch)
    from crane_project.utils.port_size_core_curvature_v2_torch import SizeReference
    fresh = SizeReference().to(features[0].device).eval()
    fresh.load_state_dict(restored['state'], strict=True)
    with torch.no_grad(): actual = fresh(features[0]).detach().cpu()
    if not torch.equal(expected, actual): raise ValueError('Candidate save/reload changed logits')
    return dict(path=str(path.relative_to(directory.parent)), sha256=base.sha(path),
                marker_sha256=base.sha(Path(str(path)+'.sha.json')), save_reload_exact=True)


def strength_summary(records, bounds):
    """Broad engineering guard fixed before measurements; not optimality proof."""
    result = {}; passed = True
    for domain in ('real', 'sim'):
        rows = [r['gradient'] for r in records if r['domain'] == domain and r['arm'] == 'a1'
                and r['projection']['eligible']]
        checks = bool(rows) and all(r['new_to_original'] is not None and
            math.isfinite(r['new_to_original']) and r['new_gradient_norm'] > 0 and
            all(r['groups'][g]['new_norm'] > 0 for g in ('stem','output')) for r in rows)
        ratios = [r['new_to_original'] for r in rows if r['new_to_original'] is not None]
        retention = [r['common_clip_scale'] for r in rows]
        median = float(np.median(ratios)) if ratios else None
        keep = float(np.median(retention)) if retention else None
        accepted = bool(checks and bounds['median_new_to_original_min'] <= median <= bounds['median_new_to_original_max']
                        and keep >= bounds['median_clip_retention_min'])
        passed &= accepted
        result[domain] = dict(eligible_measurements=len(rows), median_new_to_original=median,
            min_new_to_original=min(ratios) if ratios else None,
            max_new_to_original=max(ratios) if ratios else None,
            median_common_clip_retention=keep, engineering_guard_passed=accepted)
    return dict(domains=result, passed=bool(passed), bounds=bounds,
                coefficients_optimal_or_learning_effective_proven=False)


def smoke(args, prepared):
    old, contract = prepared
    stage(args.check_report, STATUSES['check'], contract)
    formal, torch, detector, head, pipeline, arms, initial_sha = models(args, old)
    from crane_project.utils.port_size_core_curvature_v2_torch import loss_terms, gradient_measurement, update, verify_auxiliary_autograd
    frozen = dict(b=base.state_digest(detector), midpoint=base.state_digest(head))
    split = old[1][1][6]; raw_sources = old[1][1][7]
    collected = parent.reviewed_stage(args.collection_dir, 'collect', old[1][5],
                                     old[0]['reviewed_migration_stages'])
    final_fit = {r['image']:r for r in migration.read_collection(
        args.collection_dir, collected, old[1][1])[0]}
    runtime = migration.new.MidpointReliability(old[2], old[1][5]['front_end'])
    records = []
    # Actual gradient strength at SAME untouched initialization, all 14 fit views.
    for row in reference.ideal_selection(split):
        features, meta, _ = reference.view(row, raw_sources, detector, pipeline, args.gpu)
        before = migration.midpoint_from_features(formal, torch, detector, head, features, meta)
        parent.assessment.paired_prediction(final_fit[row['image']]['b_original'], before['b'], row['image'])
        parent.assessment.paired_prediction(final_fit[row['image']]['pred'], before['midpoint'], row['image'])
        constants = core.projection(row['gt'], meta)
        numerical = (verify_auxiliary_autograd(constants, features[0].device)
                     if constants['eligible'] else None)
        target, valid = reference.size.target_map(row['gt'], meta)
        t = torch.as_tensor(target, device=features[0].device)[None,None]
        v = torch.as_tensor(valid, device=features[0].device, dtype=t.dtype)[None,None]
        for arm, model in arms.items():
            terms = loss_terms(model(features[0]), t, v, constants)
            measured, _ = gradient_measurement(terms, model, arm)
            records.append(dict(image=row['image'], domain=row['domain'], arm=arm,
                phase='initial_no_update_14_fit_views', projection=core.public_projection(constants),
                auxiliary_autograd=numerical, gradient=measured))
        after = migration.midpoint_from_features(formal, torch, detector, head, features, meta)
        if before != after: raise ValueError('Initial gradient probe changed final output')
        flags = runtime.decide(before['midpoint'], row['image_size'])
        if flags['center_accepted'] != (before['midpoint'] is not None): raise ValueError('Center not retained')
        del features, t, v, terms
    strength = strength_summary(records, contract['protocol']['gradient_guard'])
    base.write_new(args.out_dir/'initial_gradient_report.json', dict(contract=contract,
        same_initial_sha256=initial_sha, measurements=records, summary=strength))
    if not strength['passed']: raise ValueError('Fixed actual-gradient guard failed; no coefficient search')
    # Four real/sim alternating updates PER ARM, then discard both.
    real = next(r for r in split['fit'] if r['domain']=='real')
    sim = next(r for r in split['fit'] if r['domain']=='sim')
    opts = {a:torch.optim.Adam(m.parameters(), lr=.001, weight_decay=0.) for a,m in arms.items()}
    updated = []
    for slot, row in enumerate((real,sim,real,sim), 1):
        features, meta, _ = reference.view(row, raw_sources, detector, pipeline, args.gpu)
        before = migration.midpoint_from_features(formal, torch, detector, head, features, meta)
        parent.assessment.paired_prediction(final_fit[row['image']]['b_original'], before['b'], row['image'])
        parent.assessment.paired_prediction(final_fit[row['image']]['pred'], before['midpoint'], row['image'])
        constants = core.projection(row['gt'], meta)
        target, valid = reference.size.target_map(row['gt'], meta)
        t = torch.as_tensor(target, device=features[0].device)[None,None]
        v = torch.as_tensor(valid, device=features[0].device, dtype=t.dtype)[None,None]
        for arm, model in arms.items():
            logits = model(features[0])
            measured = update(loss_terms(logits, t, v, constants), model, opts[arm], arm, logits)
            updated.append(dict(step=slot, image=row['image'], domain=row['domain'], arm=arm, gradient=measured))
        after = migration.midpoint_from_features(formal, torch, detector, head, features, meta)
        if before != after: raise ValueError('Smoke update changed B/midpoint output')
        del t, v
    checkpoints = {a:save_reload(args.out_dir/a,m,opts[a],contract,a,0,4,'smoke_discarded',
        initial_sha,frozen,features,torch) for a,m in arms.items()}
    if frozen != dict(b=base.state_digest(detector),midpoint=base.state_digest(head)):
        raise ValueError('TRAIN precheck changed frozen model state')
    report = dict(status=STATUSES['smoke'], contract=contract, initial_sha256=initial_sha,
        strength=strength, update_measurements=updated, checkpoints=checkpoints,
        discarded=True, no_holdout_or_VAL_image_used=True, final_output_before_after_exact=True,
        frozen=frozen, test_read=False)
    base.write_new(args.out_dir/'smoke_report.json',report)
    return STATUSES['smoke']


def checked_replay_contract(previous, current):
    pins = core.FAILURE_REPLAY
    if (previous['sources']['manifest_sha256'] != pins['prior_manifest_sha256']
            or simple.fingerprint(previous['sources']['sources']) != pins['prior_sources_fingerprint']
            or simple.fingerprint(previous['protocol']) != pins['prior_protocol_fingerprint']
            or {k:v for k,v in previous.items() if k not in ('sources','protocol')} !=
               {k:v for k,v in current.items() if k not in ('sources','protocol')}):
        raise ValueError('Failure replay requires exact gradfix1 source and identical frozen inputs')


def replay(args, prepared):
    """Read pinned epoch1, inspect exact TRAIN failure frame, never resume."""
    old,contract=prepared
    stage(args.check_report,STATUSES['check'],contract)
    checked=stage(args.smoke_report,STATUSES['smoke'],contract)
    if args.failed_train_dir is None:raise ValueError('Missing failed gradfix1 TRAIN directory')
    directory=args.failed_train_dir;settings=core.FAILURE_REPLAY
    previous=json.loads((directory/'input_check.json').read_text())
    failure=json.loads((directory/'failure.json').read_text())
    prior=previous['contract'];checked_replay_contract(prior,contract)
    context={k:settings[k] for k in ('epoch','slot','image','domain','arm')}
    if previous['mode'] != 'train' or (failure.get('details') or {}).get('train_context') != context:
        raise ValueError('Failure frame/arm/epoch identity differs')
    formal,torch,detector,head,pipeline,arms,initial=models(args,old)
    frozen=dict(b=base.state_digest(detector),midpoint=base.state_digest(head))
    if initial != checked['initial_sha256'] or frozen != checked['frozen']:
        raise ValueError('Replay initialization/front end differs')
    checkpoint=directory/settings['arm']/'epoch_01.pth'
    payload=load(checkpoint,prior,'paired_reference_train',settings['arm'],torch)
    if (payload['epoch']!=settings['checkpoint_epoch'] or payload['steps']!=settings['checkpoint_steps']
            or payload['initial_sha256']!=initial or payload['frozen']!=frozen):
        raise ValueError('Replay checkpoint is not pinned epoch1 pair')
    rows=old[1][1][6]['fit']
    order=np.random.RandomState(1701+settings['epoch']).permutation(384)
    row=rows[int(order[settings['slot']-1])]
    if row['image']!=settings['image'] or row['domain']!=settings['domain']:
        raise ValueError('Replay frame is outside exact original TRAIN schedule')
    from crane_project.utils.port_size_core_curvature_v2_torch import loss_terms,update
    model=arms[settings['arm']];model.load_state_dict(payload['state'],strict=True);model.train()
    before_state=base.state_digest(model)
    features,meta,_=reference.view(row,old[1][1][7],detector,pipeline,args.gpu)
    before=migration.midpoint_from_features(formal,torch,detector,head,features,meta)
    collected=parent.reviewed_stage(args.collection_dir,'collect',old[1][5],old[0]['reviewed_migration_stages'])
    final_fit={r['image']:r for r in migration.read_collection(args.collection_dir,collected,old[1][1])[0]}
    for key,field in (('b','b_original'),('midpoint','pred')):
        parent.assessment.paired_prediction(final_fit[row['image']][field],before[key],row['image'])
    constants=core.projection(row['gt'],meta);target,valid=reference.size.target_map(row['gt'],meta)
    t=torch.as_tensor(target,device=features[0].device)[None,None]
    v=torch.as_tensor(valid,device=features[0].device,dtype=t.dtype)[None,None]
    class InspectOnly:
        calls=0
        def zero_grad(self):model.zero_grad()
        def step(self):self.calls+=1  # deliberately no parameter/optimizer update
    inspector=InspectOnly();logits=model(features[0])
    measurement=update(loss_terms(logits,t,v,constants),model,inspector,settings['arm'],logits)
    after=migration.midpoint_from_features(formal,torch,detector,head,features,meta)
    if (inspector.calls!=1 or before_state!=base.state_digest(model) or before!=after
            or frozen!=dict(b=base.state_digest(detector),midpoint=base.state_digest(head))):
        raise ValueError('Inspect-only replay changed parameters or frozen output')
    base.write_new(args.out_dir/'replay_report.json',dict(status=STATUSES['replay'],contract=contract,
        initial_sha256=initial,frozen=frozen,frame=context,gradient=measurement,optimizer_updates=0,
        old_failure_sha256=base.sha(directory/'failure.json'),
        old_input_check_sha256=base.sha(directory/'input_check.json'),
        replay_checkpoint_sha256=base.sha(checkpoint),
        replay_checkpoint_marker_sha256=base.sha(str(checkpoint)+'.sha.json'),
        parameters_unchanged=True,trained_VAL=False,test_read=False,performance_PASS=False))
    return STATUSES['replay']


def paired_train(args, prepared):
    old, contract = prepared
    stage(args.check_report, STATUSES['check'], contract)
    checked = stage(args.smoke_report, STATUSES['smoke'], contract)
    replayed = stage(args.replay_report, STATUSES['replay'], contract)
    formal, torch, detector, head, pipeline, arms, initial_sha = models(args,old)
    if checked['initial_sha256'] != initial_sha: raise ValueError('Fresh pair initialization differs from precheck')
    if replayed['initial_sha256'] != initial_sha or replayed['optimizer_updates'] != 0:
        raise ValueError('Exact failure replay did not release fresh initialization')
    from crane_project.utils.port_size_core_curvature_v2_torch import (
        loss_terms, coefficients, update, reference_precision)
    frozen=dict(b=base.state_digest(detector),midpoint=base.state_digest(head))
    if frozen != checked['frozen']: raise ValueError('Precheck and training front end differ')
    optimizers={a:torch.optim.Adam(m.parameters(),lr=.001,weight_decay=0.) for a,m in arms.items()}
    split=old[1][1][6]; raw=old[1][1][7]; checkpoints={}; steps=0; start=time.monotonic()
    with (args.out_dir/'train_steps.jsonl').open('x') as stream:
        for epoch in range(1,5):
            order=np.random.RandomState(1701+epoch).permutation(384)
            for model in arms.values(): model.train()
            for slot,index in enumerate(order):
                row=split['fit'][int(index)]
                features,meta,_=reference.view(row,raw,detector,pipeline,args.gpu)
                constants=core.projection(row['gt'],meta)
                target,valid=reference.size.target_map(row['gt'],meta)
                t=torch.as_tensor(target,device=features[0].device)[None,None]
                v=torch.as_tensor(valid,device=features[0].device,dtype=t.dtype)[None,None]
                for arm,model in arms.items():
                    logits=model(features[0]);terms=loss_terms(logits,t,v,constants)
                    if slot % 64 == 0:
                        try:
                            record=update(terms,model,optimizers[arm],arm,logits)
                        except ValueError as error:
                            error.details=dict(getattr(error,'details',{}) or {},
                                train_context=dict(epoch=epoch,slot=slot+1,image=row['image'],
                                                   domain=row['domain'],arm=arm))
                            error.args=(str(error)+' epoch=%d slot=%d image=%s' %
                                        (epoch,slot+1,row['image']),)
                            raise
                    else:
                        weights=coefficients(arm); total=sum(terms[k]*weights[k] for k in weights)
                        optimizers[arm].zero_grad()
                        with reference_precision():
                            total.backward()
                        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.))
                        post=math.sqrt(sum(float(p.grad.double().square().sum()) for p in model.parameters()))
                        if not math.isfinite(norm) or norm<=0 or not 0<post<=10.00001:
                            raise ValueError('Invalid TRAIN gradient/clip')
                        record=dict(losses={k:float(v.detach()) for k,v in terms.items()},
                            total_preclip_norm=norm,actual_postclip_norm=post,
                            actual_clip_retention=post/norm,component_measurement=False,
                            backward_precision=core.GRADIENT_CHECK['backward_precision'])
                        optimizers[arm].step()
                    event=dict(epoch=epoch,step=steps+1,slot=slot,image=row['image'],domain=row['domain'],
                        arm=arm,projection=core.public_projection(constants),gradient=record)
                    stream.write(json.dumps(event,allow_nan=False)+'\n')
                steps+=1
                if slot%64==0:
                    stream.flush();print('Paired TRAIN',epoch,slot+1,'/384',row['image'],flush=True)
                del t,v,terms
            for arm,model in arms.items():
                item=save_reload(args.out_dir/arm,model,optimizers[arm],contract,arm,epoch,steps,
                    'paired_reference_train',initial_sha,frozen,features,torch)
                checkpoints[arm]=item
            del features
    if steps!=1536 or frozen!=dict(b=base.state_digest(detector),midpoint=base.state_digest(head)):
        raise ValueError('Fixed budget or frozen states differ')
    final_states={a:base.state_digest(m) for a,m in arms.items()}
    if any(sha == initial_sha for sha in final_states.values()):
        raise ValueError('A reference did not update from initial state')
    report=dict(status=STATUSES['train'],contract=contract,initial_sha256=initial_sha,
        steps_per_arm=steps,total_optimizer_updates=2*steps,checkpoints=checkpoints,
        frozen=frozen,smoke_sha256=base.sha(args.smoke_report),
        replay_sha256=base.sha(args.replay_report),
        elapsed_seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved(args.gpu)/2**20,
        trained_holdout=False,trained_VAL=False,test_read=False,
        final_reference_state_sha256=final_states,
        fixed_final_epoch=4,no_checkpoint_selection=True,performance_PASS=False)
    base.write_new(args.out_dir/'train_report.json',report)
    return STATUSES['train']


def assess(args, prepared):
    from crane_project.tools.eval_port_size_core_curvature_v2 import run
    return run(args, prepared)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=tuple(STATUSES),required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=0)
    for name in ('check_report','smoke_report','replay_report','train_report','failed_train_dir'):
        parser.add_argument('--'+name.replace('_','-'),type=Path)
    defaults=dict(selection='crane_symeood_k1_port_day2night_midpoint_formal_v1/selection.json',
        formal_cache='port_geometry_midpoint_formal_v1_roi_cache',collection_dir='port_midpoint_reliability_v1_collect',
        policy='port_midpoint_reliability_v1_fit/policy.json',reference_checkpoint='port_size_reference_v1_train/epoch_04.pth',
        train_cache='port_reliability_train_support_v1_cache',input_snapshot='port_reliability_train_support_v1/train_input_snapshot.json',
        val_dir='port_reliability_branches_v1_val_cached_v1',old_policy='port_simple_reliability_v1_fit/policy.json',
        b_checkpoint='crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth')
    for name,path in defaults.items():parser.add_argument('--'+name.replace('_','-'),type=Path,default=Path('work_dirs')/path)
    args=parser.parse_args();prepared=prepare(args)
    args.out_dir.mkdir(parents=True,exist_ok=False)
    base.write_new(args.out_dir/'input_check.json',dict(contract=prepared[1],mode=args.mode))
    try:
        status={'check':check,'smoke':smoke,'replay':replay,'train':paired_train,'assess':assess}[args.mode](args,prepared)
        finish(args,prepared,status)
    except Exception as error:
        base.write_new(args.out_dir/'failure.json',dict(type=type(error).__name__,error=str(error),
            details=getattr(error,'details',None),mode=args.mode,test_read=False))
        raise


if __name__=='__main__':main()
