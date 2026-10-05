#!/usr/bin/env python3
"""check -> discarded TRAIN smoke -> fresh paired epoch04 -> held TRAIN/VAL.

Only two tiny SIZE dispersion heads learn. No TEST, baseline update, reference
checkpoint dependency, calibration fitting, resume or checkpoint selection.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_midpoint_reliability_v1 as migration
from crane_project.tools import eval_port_midpoint_size_template_v1 as parent
from crane_project.utils import port_size_residual_v4 as core
from crane_project.utils import port_size_residual_comparison_v4 as comparison

base=migration.base
simple=migration.simple
reference=migration.reference
PROTOCOL=ROOT/'crane_project/tools/port_size_residual_v4_protocol.json'
SOURCES=ROOT/'crane_project/tools/port_size_residual_v4_sources.json'
STATUSES=dict(check='SIZE_RESIDUAL_V4_NUMERICAL_AND_FIT_SUPPORT_PASS',
    smoke='SIZE_RESIDUAL_V4_TRAIN_GRADIENT_SAVE_RELOAD_PASS_DISCARDED',
    train='SIZE_RESIDUAL_V4_PAIRED_EPOCH04_COMPLETE_REVIEW_REQUIRED',
    assess='SIZE_RESIDUAL_V4_FIXED_ASSESSMENT_COMPLETE_REVIEW_REQUIRED')
DIRECT_SOURCES=(
    'crane_project/tools/run_port_size_residual_v4.py',
    'crane_project/utils/port_size_residual_v4.py',
    'crane_project/utils/port_size_residual_v4_torch.py',
    'crane_project/utils/port_size_residual_comparison_v4.py',
    'crane_project/tools/port_size_residual_v4_protocol.json',
    'tests/test_port_size_residual_v4.py',
    'crane_project/utils/port_geometry_refine_g_v1.py',
    'crane_project/utils/port_structure_reliability_v1.py',
    'crane_project/utils/port_reliability_separability_v1.py')


def checked_sources():
    manifest=json.loads(SOURCES.read_text());protocol=json.loads(PROTOCOL.read_text())
    actual={p:base.sha(ROOT/p) for p in DIRECT_SOURCES}
    if (protocol!=core.protocol_document() or manifest['protocol']!=core.VERSION or
        manifest['sources']!=actual or manifest['protocol_sha256']!=base.sha(PROTOCOL) or
        manifest['parent_midpoint_sources_sha256']!=base.sha(migration.SOURCES) or
        manifest['reviewed_collection_helper_sources_sha256']!=base.sha(parent.SOURCES)):
        raise ValueError('Fixed residual source/protocol contract differs')
    migration.checked_sources();parent.checked_sources()
    return protocol,dict(manifest_sha256=base.sha(SOURCES),sources=actual)


def prepare(args):
    protocol,sources=checked_sources()
    previous=migration.prepare(args)
    parent_protocol,_=parent.checked_sources();pins=parent_protocol['reviewed_migration_stages']
    collected=parent.reviewed_stage(args.collection_dir,'collect',previous[5],pins)
    train,val=migration.read_collection(args.collection_dir,collected,previous[1])
    fitted=parent.reviewed_stage(args.policy.parent,'fit',previous[5],pins)
    policy=json.loads(args.policy.read_text())
    if args.policy.name!='policy.json' or policy['contract']!=fitted or policy['test_read']:
        raise ValueError('Use the exact reviewed frozen policy')
    migration.new.MidpointReliability(policy,previous[5]['front_end'])
    index={r['image']:r for r in train};old_split=previous[1][6]
    split={role:[deepcopy(index[r['image']]) for r in rows] for role,rows in old_split.items()}
    if {k:len(v) for k,v in split.items()}!={'fit':384,'holdout':432,'guard':32}:
        raise ValueError('Fixed TRAIN roles differ')
    plan=parent.evaluation_rows(train,val,old_split,{'reference_holdout_train':432,'val':887})
    plan=[('residual_holdout_train' if role=='reference_holdout_train' else role,row) for role,row in plan]
    contract=dict(protocol=protocol,sources=sources,frozen_parent=previous[5],
        frozen_policy_sha256=base.sha(args.policy),
        frozen_collection_sha256=base.sha(args.collection_dir/'predictions.jsonl'),
        fit_rows_sha256=simple.fingerprint(split['fit']),
        partition_sha256=simple.fingerprint({k:[r['image'] for r in v] for k,v in split.items()}),
        locked_evaluation_rows_sha256=simple.fingerprint(plan),
        baseline_saw_full_detector_TRAIN=True,residual_holdout_is_not_detector_or_simple_holdout=True,
        GT_online=False,test_read=False,legacy_reference_checkpoint_loaded=False)
    return dict(previous=previous,policy=policy,split=split,plan=plan,contract=contract,
                raw_sources=previous[1][7])


def artifact_index(directory):
    return {str(p.relative_to(directory)):base.sha(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p.name!='completion.json'}


def finish(args,prepared,status):
    if prepare(args)['contract']!=prepared['contract']:
        raise ValueError('Inputs/sources changed during this stage')
    base.write_new(args.out_dir/'completion.json',dict(status=status,contract=prepared['contract'],
        contract_sha256=simple.fingerprint(prepared['contract']),artifacts=artifact_index(args.out_dir),
        mode=args.mode,test_read=False,detector_updates=0,midpoint_updates=0,policy_updates=0))
    print('Saved',args.out_dir,status,flush=True)


def stage(path,status,contract,experiment_dir=None):
    if path is None:raise ValueError('Preceding stage report required')
    path=Path(path)
    mode=next((k for k,v in STATUSES.items() if v==status),None)
    if mode is not None and (path.parent.name!=mode or path.name!=('assessment.json' if mode=='assess' else mode+'_report.json')):
        raise ValueError('Report filename/directory does not match its stage')
    if experiment_dir is not None and path.parent.parent.resolve()!=Path(experiment_dir).resolve():
        raise ValueError('All stages must belong to this same experiment directory')
    if (path.parent/'failure.json').exists():raise ValueError('Failed stage cannot release the next stage')
    report=json.loads(path.read_text());completion=json.loads((path.parent/'completion.json').read_text())
    if (report.get('status')!=status or completion.get('status')!=status or
        report.get('contract')!=contract or completion.get('contract')!=contract or
        completion.get('contract_sha256')!=simple.fingerprint(contract) or
        completion['artifacts'].get(path.name)!=base.sha(path)):
        raise ValueError('Stage status/contract/SHA differs')
    for name,sha in completion['artifacts'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or base.sha(path.parent/p)!=sha:
            raise ValueError('Stage artifact differs or escapes its directory')
    return report


def check(args,prepared):
    fit=prepared['split']['fit'];records=[]
    for row in reference.ideal_selection(prepared['split']):
        target=core.residual_target(row['pred'],row['gt'])
        records.append(dict(image=row['image'],domain=row['domain'],target=None if target is None else target.tolist(),
            numerical=core.numerical_probe(target) if target is not None else None))
    standardizer=core.fit_standardizer(fit)
    passed=all(r['numerical'] is None or r['numerical']['passed'] for r in records)
    passed &= all(any(r['domain']==d and r['pred'] is not None for r in fit) for d in ('real','sim'))
    support={field+':'+key:core.support_summary([r for r in fit if r[field]==key])
             for field in ('domain','sequence') for key in sorted({r[field] for r in fit})}
    report=dict(status=STATUSES['check'] if passed else 'SIZE_RESIDUAL_V4_CHECK_FAILED',
        contract=prepared['contract'],standardizer=standardizer,fit_support=support,numerical=records,
        fit_outputs=standardizer['fit_outputs'],nominal_slots_per_arm=1536,
        maximum_updates_per_arm=4*standardizer['fit_outputs'],only_fit_GT_used_for_release=True,
        supervised_signed_residuals=True,rare_error_support_not_solved=True,
        actual_Torch_gradient_verified=False,trained_model_performance=False,test_read=False)
    base.write_new(args.out_dir/'check_report.json',report)
    if not passed:raise ValueError('TRAIN numerical/support check failed')
    return STATUSES['check']


def models(args,prepared,standardizer):
    formal,torch,detector,head,pipeline,_=migration.online_modules(args,prepared['previous'],False)
    from crane_project.utils.port_size_residual_v4_torch import SizeDispersion
    # Candidate seed is set AFTER loading the unchanged formal front end.
    torch.manual_seed(1701);torch.cuda.manual_seed_all(1701)
    first=SizeDispersion(standardizer).cuda(args.gpu)
    initial={k:v.detach().cpu().clone() for k,v in first.state_dict().items()}
    arms={'a0':first,'a1':SizeDispersion(standardizer).cuda(args.gpu)}
    for model in arms.values():model.load_state_dict(initial,strict=True)
    digest=base.state_digest(first)
    if any(base.state_digest(m)!=digest for m in arms.values()):raise ValueError('Pair initialization differs')
    return formal,torch,detector,head,pipeline,arms,digest


def inputs(features,meta,final_box,image_size,formal,torch):
    """Online inputs exclude the labeled row and every GT field."""
    if final_box is None:return None
    # Final midpoint box, not B's earlier box or GT, determines the image ROI.
    box=torch.as_tensor([final_box[:5]],device=features[0].device,dtype=features[0].dtype)
    model_box=formal.g.map_boxes(box,meta,inverse=False)
    roi,support,_=formal.g.sample_local(features[0],model_box,meta,'aligned')
    descriptor=torch.as_tensor(simple.descriptor(final_box,image_size),
                               device=features[0].device,dtype=features[0].dtype)[None]
    return roi,support,descriptor


def snapshot(detector,head):
    return dict(b=base.state_digest(detector),midpoint=base.state_digest(head))


def paired_view(row,prepared,formal,torch,detector,head,pipeline,gpu):
    features,meta,_=reference.view(row,prepared['raw_sources'],detector,pipeline,gpu)
    first=migration.midpoint_from_features(formal,torch,detector,head,features,meta)
    differences={key:parent.assessment.paired_prediction(row[field],first[key],row['image'])
                 for key,field in (('b','b_original'),('midpoint','pred'))}
    return features,meta,first,differences


def unchanged(row,first,features,meta,formal,torch,detector,head):
    if first!=migration.midpoint_from_features(formal,torch,detector,head,features,meta):
        raise ValueError('Candidate operation changed final output: '+row['image'])


def load_checkpoint(path,contract,arm,role,torch):
    path=Path(path);tag=json.loads(Path(str(path)+'.sha.json').read_text())
    if tag['sha256']!=base.sha(path) or tag['contract_sha256']!=simple.fingerprint(contract):
        raise ValueError('Checkpoint bytes/contract differ')
    payload=torch.load(str(path),map_location='cpu')
    if (payload['protocol']!=core.VERSION or payload['contract']!=contract or payload['arm']!=arm or
        payload['role']!=role or tag['role']!=role or tag['epoch']!=payload['epoch']):
        raise ValueError('Checkpoint arm/role differs')
    return payload


def save_reload(directory,model,optimizer,contract,arm,epoch,steps,initial,frozen,example,torch):
    from crane_project.utils.port_size_residual_v4_torch import SizeDispersion,precision
    directory.mkdir(exist_ok=True);path=directory/('epoch_%02d.pth'%epoch)
    role='smoke_discarded' if epoch==0 else 'paired_size_residual_train'
    model.eval()
    with torch.no_grad(),precision():expected=model(*example,arm).detach().cpu()
    standardizer=dict(mean=model.descriptor_mean.detach().cpu().tolist(),scale=model.descriptor_scale.detach().cpu().tolist())
    payload=dict(protocol=core.VERSION,contract=contract,arm=arm,role=role,epoch=epoch,steps=steps,
        initial_sha256=initial,frozen=frozen,standardizer=standardizer,
        state={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict())
    reference.save_checkpoint(path,payload)
    restored=load_checkpoint(path,contract,arm,role,torch)
    fresh=SizeDispersion(standardizer).to(example[0].device).eval();fresh.load_state_dict(restored['state'],strict=True)
    with torch.no_grad(),precision():actual=fresh(*example,arm).detach().cpu()
    if not torch.equal(expected,actual):raise ValueError('Candidate save/reload changed dispersion')
    return dict(path=str(path.relative_to(directory.parent)),sha256=base.sha(path),
                marker_sha256=base.sha(Path(str(path)+'.sha.json')),save_reload_exact=True)


def smoke(args,prepared):
    checked=stage(args.check_report,STATUSES['check'],prepared['contract'],args.out_dir.parent)
    formal,torch,detector,head,pipeline,arms,initial=models(args,prepared,checked['standardizer'])
    from crane_project.utils.port_size_residual_v4_torch import precision,update,verify_autograd
    frozen=snapshot(detector,head);opts={a:torch.optim.Adam(m.parameters(),lr=.001,weight_decay=0.) for a,m in arms.items()}
    records=[];examples=[]
    for row in reference.ideal_selection(prepared['split']):
        if row['pred'] is None:continue
        features,meta,first,_=paired_view(row,prepared,formal,torch,detector,head,pipeline,args.gpu)
        example=inputs(features,meta,row['pred'],row['image_size'],formal,torch)
        if float(example[1].sum())<=1e-6:raise ValueError('Initial fit image has no final-box support')
        target=torch.as_tensor(core.residual_target(row['pred'],row['gt']),device=features[0].device)[None]
        for arm,model in arms.items():
            with precision():raw=model(*example,arm)
            # Measurement does not step or mutate state at the common initialization.
            before=base.state_digest(model)
            try:
                verify=verify_autograd(raw.detach().cpu().numpy()[0].tolist(),target.cpu().numpy()[0].tolist(),raw.device)
            except ValueError as error:
                error.args=(str(error)+' image='+row['image']+' arm='+arm,)
                raise
            from crane_project.utils.port_size_residual_v4_torch import loss_terms,norm
            model.zero_grad()
            with precision():loss_terms(raw,target)['total'].backward()
            groups={k:norm(list(getattr(model,k).parameters())) for k in ('stem','output')}
            if before!=base.state_digest(model) or not math.isfinite(groups['output']) or groups['output']<=0:
                raise ValueError('Invalid untouched-initialization gradient: '+row['image']+' '+arm)
            if arm=='a1' and (not math.isfinite(groups['stem']) or groups['stem']<=0):
                raise ValueError('Image dispersion path has zero/nonfinite initial gradient: '+row['image']+' '+arm)
            records.append(dict(image=row['image'],domain=row['domain'],arm=arm,
                raw=raw.detach().cpu().numpy()[0].tolist(),NLL=core.nll_numpy(raw.detach().cpu().numpy()[0],target.cpu().numpy()[0]),
                parameter_gradient_norms=groups,autograd=verify,a0_stem_zero_expected=arm=='a0'))
        unchanged(row,first,features,meta,formal,torch,detector,head)
        del features,target,raw
    for domain in ('real','sim'):
        if not any(r['domain']==domain and r['arm']=='a1' for r in records):
            raise ValueError('Both domains must have initial image gradients')
    updated=[]
    selected=[next(r for r in prepared['split']['fit'] if r['domain']==d and r['pred'] is not None) for d in ('real','sim')]
    for slot,row in enumerate(selected*2,1):
        features,meta,first,_=paired_view(row,prepared,formal,torch,detector,head,pipeline,args.gpu)
        example=inputs(features,meta,row['pred'],row['image_size'],formal,torch);examples=example
        if float(example[1].sum())<=1e-6:raise ValueError('Smoke fit image has no usable support')
        target=torch.as_tensor(core.residual_target(row['pred'],row['gt']),device=features[0].device)[None]
        for arm,model in arms.items():
            with precision():raw=model(*example,arm)
            measured=update(raw,target,model,opts[arm],arm,True)
            updated.append(dict(slot=slot,image=row['image'],domain=row['domain'],arm=arm,gradient=measured))
        unchanged(row,first,features,meta,formal,torch,detector,head)
        del features,target,raw
    guard=all(float(np.median([r['gradient']['clip_retention'] for r in updated if r['domain']==d and r['arm']==a]))>=.1
              for d in ('real','sim') for a in core.ARMS)
    checkpoints={a:save_reload(args.out_dir/a,m,opts[a],prepared['contract'],a,0,4,initial,frozen,examples,torch) for a,m in arms.items()}
    if not guard or snapshot(detector,head)!=frozen:raise ValueError('Smoke clipping/frozen state guard failed')
    report=dict(status=STATUSES['smoke'],contract=prepared['contract'],initial_sha256=initial,
        check_report_sha256=base.sha(args.check_report),initial_measurements=records,updates=updated,
        checkpoints=checkpoints,frozen=frozen,discarded=True,no_holdout_or_VAL_images_used=True,
        clip_retention_median_min=.1,final_output_before_after_exact=True,test_read=False)
    base.write_new(args.out_dir/'smoke_report.json',report)
    return STATUSES['smoke']


def paired_train(args,prepared):
    checked=stage(args.check_report,STATUSES['check'],prepared['contract'],args.out_dir.parent)
    smoked=stage(args.smoke_report,STATUSES['smoke'],prepared['contract'],args.out_dir.parent)
    formal,torch,detector,head,pipeline,arms,initial=models(args,prepared,checked['standardizer'])
    from crane_project.utils.port_size_residual_v4_torch import precision,update
    frozen=snapshot(detector,head)
    if (initial!=smoked['initial_sha256'] or frozen!=smoked['frozen'] or not smoked['discarded'] or
        smoked['check_report_sha256']!=base.sha(args.check_report)):
        raise ValueError('Fresh train source/init/check differs from discarded smoke')
    opts={a:torch.optim.Adam(m.parameters(),lr=.001,weight_decay=0.) for a,m in arms.items()}
    steps=0;slots=0;checkpoints={};skipped=[];example=None;start=time.monotonic()
    with (args.out_dir/'train_steps.jsonl').open('x') as stream:
        for epoch in range(1,5):
            order=np.random.RandomState(1701+epoch).permutation(384)
            for model in arms.values():model.train()
            epoch_skips=[]
            for slot,index in enumerate(order,1):
                row=prepared['split']['fit'][int(index)];slots+=1
                if row['pred'] is None:
                    reason='no_frozen_output';features=None
                else:
                    features,meta,first,_=paired_view(row,prepared,formal,torch,detector,head,pipeline,args.gpu)
                    current=inputs(features,meta,row['pred'],row['image_size'],formal,torch)
                    reason='no_image_support' if float(current[1].sum())<=1e-6 else None
                if reason is not None:
                    epoch_skips.append(row['image']);event=dict(epoch=epoch,slot=slot,image=row['image'],domain=row['domain'],
                        arms_skipped=list(core.ARMS),reason=reason)
                    if features is not None:unchanged(row,first,features,meta,formal,torch,detector,head)
                    stream.write(json.dumps(event,allow_nan=False)+'\n');continue
                example=current;target=torch.as_tensor(core.residual_target(row['pred'],row['gt']),device=features[0].device)[None]
                for arm,model in arms.items():
                    with precision():raw=model(*example,arm)
                    try:
                        record=update(raw,target,model,opts[arm],arm,slot%64==1)
                    except Exception as error:
                        error.details=dict(epoch=epoch,slot=slot,image=row['image'],domain=row['domain'],arm=arm,
                            signed_residual=target.cpu().numpy()[0].tolist())
                        error.args=(str(error)+' epoch=%d slot=%d image=%s arm=%s'%(epoch,slot,row['image'],arm),)
                        raise
                    event=dict(epoch=epoch,slot=slot,step=steps+1,image=row['image'],domain=row['domain'],arm=arm,
                               signed_residual=target.cpu().numpy()[0].tolist(),gradient=record)
                    stream.write(json.dumps(event,allow_nan=False)+'\n')
                steps+=1
                # All fitting views retain frozen-output parity, not only sampled updates.
                unchanged(row,first,features,meta,formal,torch,detector,head)
                if slot%64==1:
                    stream.flush();print('Paired residual TRAIN',epoch,slot,'/384',row['image'],flush=True)
                del features,target,raw
            if epoch==1:skipped=sorted(epoch_skips)
            elif sorted(epoch_skips)!=skipped:raise ValueError('Fitting support changed across epochs')
            if example is None:raise ValueError('No valid paired updates')
            for arm,model in arms.items():
                checkpoints[arm]=save_reload(args.out_dir/arm,model,opts[arm],prepared['contract'],arm,epoch,steps,initial,frozen,example,torch)
    if slots!=1536 or steps!=4*(384-len(skipped)) or snapshot(detector,head)!=frozen:
        raise ValueError('Fixed paired budget or frozen states differ')
    final={a:base.state_digest(m) for a,m in arms.items()}
    if any(v==initial for v in final.values()):raise ValueError('A dispersion arm never updated')
    report=dict(status=STATUSES['train'],contract=prepared['contract'],standardizer=checked['standardizer'],
        check_report_sha256=base.sha(args.check_report),smoke_report_sha256=base.sha(args.smoke_report),
        initial_sha256=initial,slots_per_arm=slots,steps_per_arm=steps,total_optimizer_updates=2*steps,
        paired_skipped_fit_images=skipped,checkpoints=checkpoints,frozen=frozen,final_candidate_state_sha256=final,
        fixed_final_epoch=4,smoke_reused=False,no_checkpoint_selection=True,
        trained_holdout=False,trained_VAL=False,test_read=False,performance_PASS=False,
        elapsed_seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20)
    base.write_new(args.out_dir/'train_report.json',report)
    return STATUSES['train']


def assess(args,prepared):
    trained=stage(args.train_report,STATUSES['train'],prepared['contract'],args.out_dir.parent)
    experiment=args.out_dir.parent
    check_path=experiment/'check'/'check_report.json';smoke_path=experiment/'smoke'/'smoke_report.json'
    checked=stage(check_path,STATUSES['check'],prepared['contract'],experiment)
    smoked=stage(smoke_path,STATUSES['smoke'],prepared['contract'],experiment)
    if (trained['check_report_sha256']!=base.sha(check_path) or trained['smoke_report_sha256']!=base.sha(smoke_path) or
        smoked['check_report_sha256']!=base.sha(check_path) or not smoked['discarded'] or
        trained['standardizer']!=checked['standardizer'] or trained['initial_sha256']!=smoked['initial_sha256']):
        raise ValueError('Assessment stage chain/standardizer differs')
    formal,torch,detector,head,pipeline,arms,initial=models(args,prepared,trained['standardizer'])
    from crane_project.utils.port_size_residual_v4_torch import precision
    before=snapshot(detector,head)
    if before!=trained['frozen'] or initial!=trained['initial_sha256']:raise ValueError('Assessment front end/init differs')
    for arm,model in arms.items():
        item=trained['checkpoints'][arm];path=args.train_report.parent/item['path']
        if path.parent!=args.train_report.parent/arm or path.name!='epoch_04.pth' or base.sha(path)!=item['sha256'] or base.sha(str(path)+'.sha.json')!=item['marker_sha256']:
            raise ValueError('Requires indexed fixed epoch04 only')
        payload=load_checkpoint(path,prepared['contract'],arm,'paired_size_residual_train',torch)
        if payload['epoch']!=4 or payload['steps']!=trained['steps_per_arm'] or payload['initial_sha256']!=initial or payload['frozen']!=before:
            raise ValueError('Candidate budget/source differs')
        model.load_state_dict(payload['state'],strict=True);model.eval().requires_grad_(False)
    before.update({a:base.state_digest(m) for a,m in arms.items()})
    runtime=migration.new.MidpointReliability(prepared['policy'],prepared['previous'][5]['front_end'])
    records=[];delta=dict(b=0.,midpoint=0.);start=time.monotonic()
    with (args.out_dir/'assessment_rows.jsonl').open('x') as stream:
        for index,(role,row) in enumerate(prepared['plan'],1):
            features,meta,first,differences=paired_view(row,prepared,formal,torch,detector,head,pipeline,args.gpu)
            delta={k:max(delta[k],differences[k]) for k in delta};example=inputs(features,meta,row['pred'],row['image_size'],formal,torch)
            raw_by_arm={a:None for a in core.ARMS}
            if example is not None:
                for arm,model in arms.items():
                    if arm=='a1' and float(example[1].sum())<=1e-6:continue
                    with torch.no_grad(),precision():raw_by_arm[arm]=model(*example,arm).cpu().numpy()[0].tolist()
            evidence=core.online_evidence(raw_by_arm,row['pred'],row['image_size'],runtime)
            item=comparison.annotate(row,evidence,role)
            item['raw_dispersion_by_arm']=raw_by_arm
            unchanged(row,first,features,meta,formal,torch,detector,head)
            item['final_output_before_after_exact']=True
            item['transform_meta']={k:list(meta[k]) for k in ('img_shape','ori_shape','pad_shape','scale_factor')}
            item=json.loads(json.dumps(item,default=lambda v:v.item(),allow_nan=False))
            stream.write(json.dumps(item,allow_nan=False)+'\n');records.append(item)
            if index%100==0:
                stream.flush();print('Fixed residual assessment',role,index,'/',len(prepared['plan']),flush=True)
            del features
    after=snapshot(detector,head);after.update({a:base.state_digest(m) for a,m in arms.items()})
    if before!=after:raise ValueError('Assessment modified frozen model states')
    restored=[json.loads(v) for v in (args.out_dir/'assessment_rows.jsonl').read_text().splitlines()]
    stats=comparison.summarize(records)
    if restored!=records or stats!=comparison.summarize(restored):raise ValueError('Saved metrics do not replay')
    report=dict(status=STATUSES['assess'],contract=prepared['contract'],train_report_sha256=base.sha(args.train_report),
        **stats,state_before=before,state_after=after,max_absolute_formal_box_difference=delta,
        frozen_boxes_scores_output_count_center_and_angle_flags_preserved=True,
        GT_online=False,test_read=False,probability_calibrated=False,
        evidence_role='bounded_residual_holdout_and_repeatedly_exposed_VAL_not_independent_confirmation',
        elapsed_seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20)
    base.write_new(args.out_dir/'assessment.json',report)
    for domain in ('real','sim'):
        group=stats['summary']['val']['domain:'+domain]
        print(domain,'same-count FA/FR',{m:(p['FA'],p['FR']) for m,p in group['primary_same_count'].items()},flush=True)
    print('Fixed review',stats['prespecified_continuation_review'],flush=True)
    return STATUSES['assess']


def validate_output(args):
    target=args.out_dir.resolve();work=(ROOT/'work_dirs').resolve()
    if target.parent.parent!=work or not target.parent.name.startswith(core.VERSION) or target.name!=args.mode:
        raise ValueError('Use work_dirs/port_size_residual_v4[optional_suffix]/'+args.mode)
    if target.exists():raise ValueError('Preserve prior stage output; choose a fresh experiment directory')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=tuple(STATUSES),required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=0)
    for name in ('check_report','smoke_report','train_report'):
        parser.add_argument('--'+name.replace('_','-'),type=Path)
    defaults=dict(selection='crane_symeood_k1_port_day2night_midpoint_formal_v1/selection.json',
        formal_cache='port_geometry_midpoint_formal_v1_roi_cache',collection_dir='port_midpoint_reliability_v1_collect',
        policy='port_midpoint_reliability_v1_fit/policy.json',train_cache='port_reliability_train_support_v1_cache',
        input_snapshot='port_reliability_train_support_v1/train_input_snapshot.json',
        val_dir='port_reliability_branches_v1_val_cached_v1',old_policy='port_simple_reliability_v1_fit/policy.json',
        b_checkpoint='crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth')
    for name,path in defaults.items():parser.add_argument('--'+name.replace('_','-'),type=Path,default=Path('work_dirs')/path)
    args=parser.parse_args();validate_output(args);prepared=prepare(args)
    args.out_dir.mkdir(parents=True,exist_ok=False)
    base.write_new(args.out_dir/'input_check.json',dict(mode=args.mode,contract=prepared['contract']))
    try:
        status={'check':check,'smoke':smoke,'train':paired_train,'assess':assess}[args.mode](args,prepared)
        finish(args,prepared,status)
    except Exception as error:
        base.write_new(args.out_dir/'failure.json',dict(type=type(error).__name__,error=str(error),
            details=getattr(error,'details',None),mode=args.mode,test_read=False))
        raise


if __name__=='__main__':main()
