#!/usr/bin/env python3
"""TRAIN instability reproduction OR independently versioned VAL auxiliary selection.

No TEST stage; no formal detector/policy replacement. Run stages as subprocesses
to release detector/optimizer CUDA storage before offline checkpoint scans.
"""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_oof_v1 as old
from crane_project.utils import port_reliability_oof_followup_v1 as contract


def checked(out):
    saved=json.loads((out/'followup.json').read_text())
    if saved['protocol']!=contract.VERSION or saved['trace']!=contract.TRACE or saved['selection']!=contract.SELECTION:
        raise ValueError('Follow-up contract changed')
    for path,pin in saved['sources'].items():
        if old.sha(ROOT/path)!=pin: raise ValueError('Follow-up source changed: '+path)
    expected_folds=('A',) if saved['purpose']=='diagnose' else ('A','B')
    if set(saved['configuration_sha256'])!=set(expected_folds): raise ValueError('Prepared fold set differs')
    for fold,pin in saved['configuration_sha256'].items():
        if old.sha(out/fold/'detector_config.py')!=pin: raise ValueError('Prepared follow-up config changed')
    return saved


def prepare(out,purpose):
    from mmcv import Config
    old.prepare(out)
    files=['crane_project/tools/run_port_reliability_oof_followup_v1.py',
        'crane_project/utils/port_reliability_oof_followup_v1.py',
        'mmrotate/core/hooks/port_oof_trace_hook.py',
        'crane_project/tools/ckpt_sweep.py','crane_project/tools/eval_crane_offline.py',
        'crane_project/utils/port_geometry_size_support_v1.py',
        'tests/test_port_reliability_oof_followup_v1.py',
        'crane_project/tools/run_port_reliability_oof_followup_v1.sh']
    for fold in (('A',) if purpose=='diagnose' else ('A','B')):
        folder=out/fold; path=folder/'detector_config.py'
        cfg=Config.fromfile(str(path))
        contract.configure(cfg,purpose,folder/'trace' if purpose=='diagnose' else None)
        cfg.dump(str(path))
        membership=json.loads((folder/'membership.json').read_text())
        membership['resolved_config_sha256']=old.sha(path)
        (folder/'membership.json').write_text(json.dumps(membership,indent=2)+'\n')
    old.write(out/'followup.json',dict(protocol=contract.VERSION,purpose=purpose,
        trace=contract.TRACE,selection=contract.SELECTION,
        sources={p:old.sha(ROOT/p) for p in files},
        configuration_sha256={fold:old.sha(out/fold/'detector_config.py') for fold in
            (('A',) if purpose=='diagnose' else ('A','B'))},
        parent_protocol=old.core.VERSION,parent_checks_preserved=True,
        selection_VAL_used=purpose=='selected',full_VAL_independent_claim=False,
        diagnostic_initialization='fresh_ImageNet_same_seed_no_A24_or_formal_weights',
        detector_updates_contract='original_24_epochs; diagnosis bounded at7000',
        auxiliary_midpoint_updates='same3_epochs; checkpoints1/2/3',
        deployed_auxiliary_models=0,TEST_read=False))


def diagnose(out):
    if checked(out)['purpose']!='diagnose': raise ValueError('Wrong run purpose')
    old.check_fold(out,'A')
    proc=subprocess.run([sys.executable,str(ROOT/'tools/train.py'),str(out/'A/detector_config.py'),
        '--no-validate','--seed','1701','--deterministic'],cwd=str(ROOT))
    path=out/'A/trace/completion.json'
    if not path.exists(): raise RuntimeError('TRAIN trace failed unexpectedly; exit=%d'%proc.returncode)
    result=json.loads(path.read_text())
    if proc.returncode==0 or result['status'] not in (
            'ANOMALY_QUARANTINED_BEFORE_OPTIMIZER','WINDOW_LIMIT_REACHED_NOT_ROOT_CAUSE',
            'FORWARD_FAILURE_QUARANTINED_BEFORE_OPTIMIZER'):
        raise RuntimeError('Trace stop boundary inconsistent with subprocess exit')
    old.write(out/'completion.json',dict(result,training_subprocess_exit=proc.returncode,
        expected_bounded_stop=True,unique_root_cause_proven=False))


def replay(out,gpu,boundary='anomaly'):
    """One saved transformed TRAIN batch, no optimizer step or data loading."""
    import torch
    from types import SimpleNamespace
    from mmcv import Config
    from mmcv.runner import build_optimizer
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from mmrotate.core.hooks.port_oof_trace_hook import (PortOOFTraceHook,cpu_tree,
        restore_rng,restore_runtime,runtime_state,gradient_summary)
    if checked(out)['purpose']!='diagnose': raise ValueError('Replay requires diagnostic run')
    old.check_fold(out,'A')
    directory=out/'A/trace'/('replay_'+boundary); directory.mkdir(exist_ok=False)
    snapshot_path=out/'A/trace'/('before_anomaly.pt' if boundary=='anomaly' else 'before_previous_update.pt')
    pin=old.sha(snapshot_path)
    completion=json.loads((out/'A/trace/completion.json').read_text())
    if completion['snapshot_sha256'].get(snapshot_path.name)!=pin:
        raise ValueError('Saved diagnostic snapshot changed')
    snapshot=torch.load(str(snapshot_path),map_location='cpu')
    if snapshot['protocol']!=contract.VERSION or not snapshot['diagnostic_only']:
        raise ValueError('Only diagnostic pre-forward snapshots can be replayed')
    cfg=Config.fromfile(str(out/'A/detector_config.py')); old.training_repair.validate_config(cfg)
    import_modules_from_strings(**cfg.custom_imports)
    specification=deepcopy(cfg.model)
    specification.backbone.pop('init_cfg',None); specification.backbone.pretrained=None
    model=build_detector(specification,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
    torch.cuda.set_device(gpu); model.cuda(gpu); model.load_state_dict(snapshot['model'],strict=True)
    optimizer=build_optimizer(model,cfg.optimizer); optimizer.load_state_dict(snapshot['optimizer'])
    def device_tree(value):
        if isinstance(value,torch.Tensor): return value.cuda(gpu)
        if isinstance(value,dict): return {k:device_tree(v) for k,v in value.items()}
        if isinstance(value,list): return [device_tree(v) for v in value]
        if isinstance(value,tuple): return tuple(device_tree(v) for v in value)
        return value
    batch={k:(cpu_tree(v) if k=='img_metas' else device_tree(v)) for k,v in snapshot['batch'].items()}
    trace=PortOOFTraceHook(directory/'unused'); trace.model=model
    trace.runner=SimpleNamespace(model=model,optimizer=optimizer,iter=snapshot['iteration'])
    trace.before=snapshot; trace.install(model,capture_failure=False)
    restore_runtime(model,snapshot['runtime']); restore_rng(snapshot['rng'])
    if boundary=='anomaly' and completion['status']=='FORWARD_FAILURE_QUARANTINED_BEFORE_OPTIMIZER':
        expected=torch.load(str(out/'A/trace/after_forward_before_update.pt'),map_location='cpu')['forward_error']
        actual=None
        try: model.train_step(batch,optimizer)
        except Exception as exc: actual=repr(exc)
        old.write(directory/'report.json',dict(protocol=contract.VERSION,snapshot_sha256=pin,
            expected_forward_error=expected,actual_forward_error=actual,passed=actual==expected,
            optimizer_updates=0,VAL_read=False,TEST_read=False))
        if actual!=expected: raise ValueError('Saved forward failure did not reproduce')
        return
    value=model.train_step(batch,optimizer)
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    norms=gradient_summary(torch.autograd.grad(value['loss'],[p for _,p in named],allow_unused=True),named)
    original=(snapshot['trace'] if boundary=='previous' else
        json.loads((out/'A/trace/updates.jsonl').read_text().splitlines()[-1]))
    actual_logs={k:float(v) for k,v in value['log_vars'].items()}
    # Restoring same augmented input/RNG/state supports this single-batch replay,
    # not bitwise reproducibility of a fresh full run on different hardware.
    comparison=contract.compare_replay_trace(actual_logs['loss'],norms['all'],original)
    model.load_state_dict(snapshot['model'],strict=True); restore_runtime(model,snapshot['runtime'])
    if old.sha(snapshot_path)!=pin: raise ValueError('Diagnostic snapshot modified')
    old.write(directory/'report.json',contract.json_safe(dict(protocol=contract.VERSION,step=snapshot['iteration']+1,
        snapshot_sha256=pin,**comparison,
        gradient_modules=norms,logs=actual_logs,optimizer_updates=0,
        VAL_read=False,TEST_read=False,original_snapshot_unchanged=True)))
    if not comparison['passed']: raise ValueError('Same-batch diagnostic replay did not reproduce saved loss/gradient')


def build_selected_detector(cfg,path,gpu,epoch):
    import torch
    from mmcv.utils import import_modules_from_strings
    from mmcv.runner import load_checkpoint
    from mmrotate.models import build_detector
    from mmdet.apis import set_random_seed
    old.training_repair.validate_config(cfg)
    set_random_seed(1701,deterministic=True)
    import_modules_from_strings(**cfg.custom_imports)
    specification=deepcopy(cfg.model)
    specification.backbone.pop('init_cfg',None); specification.backbone.pretrained=None
    model=build_detector(specification,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
    payload=load_checkpoint(model,str(path),map_location='cpu',strict=True)
    if payload['meta']['epoch']!=epoch: raise ValueError('Candidate checkpoint epoch differs')
    del payload
    torch.cuda.set_device(gpu)
    return model.cuda(gpu).eval().requires_grad_(False)


def inference_records(out,fold,model,gpu,split):
    """Unfiltered VAL or held-out TRAIN. GT attached offline after GT-free forward."""
    import torch
    from mmcv import Config
    from mmcv.parallel import collate,scatter
    from mmrotate.datasets import build_dataset
    from mmdet.datasets.dataset_wrappers import ConcatDataset
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as frozen
    rows,specification,folder=old.check_fold(out,fold)
    cfg=Config.fromfile(str(old.CONFIG)); aux=Config.fromfile(str(folder/'detector_config.py'))
    if split=='val':
        template=deepcopy(cfg.data.val); template['test_mode']=True
        # The historical repair verifies both TRAIN pipelines. Duplicate only the
        # template for that check; build exactly one full VAL dataset afterward.
        specs=old.training_repair.inference_specs([template,deepcopy(template)],aux)
        dataset=build_dataset(specs[0]); requested={r['image'] for r in rows if r['split']=='val'}
    elif split=='oof':
        specs=old.training_repair.inference_specs(frozen.g.fixed_specs(cfg,1.),aux)
        dataset=ConcatDataset([build_dataset(s) for s in specs]); requested=set(specification['predict_images'])
    else: raise ValueError('Only VAL or excluded TRAIN inference allowed')
    lookup={r['image']:r for r in rows}; result=[]
    guard=old.training_repair.MemoryPlateau()
    before=frozen.g.state_digest(model)
    with torch.no_grad():
        for i,info in enumerate(frozen.infos(dataset)):
            name=Path(info['filename']).stem
            if name not in requested: continue
            value=scatter(collate([dataset[i]],samples_per_gpu=1),[gpu])[0]
            tensors=frozen.capture_cache(model,value['img'][0],value['img_metas'][0])
            source=lookup[name]
            row={k:deepcopy(source[k]) for k in ('image','sequence','domain','split','frame_id','image_size','gt')}
            row.update(role='val' if split=='val' else 'excluded_train',scale=1.,
                eligible=False,gt_original=torch.tensor(source['gt'],dtype=torch.float32).reshape(1,5),**tensors)
            result.append(row)
            torch.cuda.synchronize(gpu)
            guard.observe(torch.cuda.memory_allocated(gpu),torch.cuda.memory_reserved(gpu))
            if len(result)%100==0: old.emit(dict(stage=split+'_cache',fold=fold,done=len(result),total=len(requested)))
    if len(result)!=len(requested) or {r['image'] for r in result}!=requested:
        raise ValueError('Inference membership incomplete')
    if frozen.g.state_digest(model)!=before: raise ValueError('Frozen candidate detector mutated')
    if frozen.tensor_bytes(result)>frozen.SETTINGS['max_cpu_tensor_bytes']:
        raise ValueError('CPU inference cache exceeds inherited1GiB bound')
    return result


def base_rows(records):
    return [dict({k:deepcopy(r[k]) for k in ('image','domain','sequence','split','frame_id','image_size','gt')},
        pred=r['boxes_original'][0].tolist() if len(r['boxes_original']) else None) for r in records]


def select_detector(out,fold,gpu):
    import torch
    from mmcv import Config
    rows,_,folder=old.check_fold(out,fold); cfg=Config.fromfile(str(folder/'detector_config.py'))
    for epoch in contract.SELECTION['detector_epochs']:
        if not (folder/('detector/epoch_%d.pth'%epoch)).is_file():
            raise ValueError('Incomplete detector budget/checkpoint scan before VAL loading')
    candidates=[]
    for epoch in contract.SELECTION['detector_epochs']:
        path=folder/('detector/epoch_%d.pth'%epoch)
        pin=old.sha(path)
        model=build_selected_detector(cfg,path,gpu,epoch)
        cache=inference_records(out,fold,model,gpu,'val'); predictions=base_rows(cache)
        candidates.append(dict(epoch=epoch,path=str(path),sha256=pin,metrics=contract.selection_metrics(predictions,rows),
            offline_metrics=contract.offline_metrics(predictions,rows)))
        old.dump_rows(folder/('detector_VAL_epoch_%02d.jsonl.gz'%epoch),predictions)
        if old.sha(path)!=pin: raise ValueError('Checkpoint changed during selection')
        del cache,model; torch.cuda.empty_cache()
    selected=contract.select(candidates,'detector')
    old.write(folder/'detector_selection.json',dict(protocol=contract.VERSION,kind='detector',
        contract=contract.SELECTION,candidates=candidates,selected=selected,
        selection_role='original_full_VAL_shared_development',TEST_read=False))
    if selected is None: source_failed(out,fold,'detector_selection_no_feasible_checkpoint')


def source_failed(out,fold,stage):
    old.write(out/'completion.json',dict(protocol=contract.VERSION,status='AUXILIARY_SOURCE_FAILED_STOP',
        fold=fold,failure_stage=stage,adopted=False,VAL_used_for_auxiliary_selection=True,
        reliability_VAL_scored=False,TEST_read=False))


def midpoint(out,fold,gpu):
    import torch
    from mmcv import Config
    from mmdet.apis import set_random_seed
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as frozen
    from crane_project.utils.port_geometry_midpoint_sigma_v1 import SigmaMidpointHead
    rows,specification,folder=old.check_fold(out,fold)
    selected=json.loads((folder/'detector_selection.json').read_text())['selected']
    if selected is None: raise ValueError('No selected detector')
    path=Path(selected['path'])
    if old.sha(path)!=selected['sha256']: raise ValueError('Selected auxiliary detector changed')
    cfg=Config.fromfile(str(folder/'detector_config.py'))
    model=build_selected_detector(cfg,path,gpu,selected['epoch']); before=frozen.g.state_digest(model)
    lookup={r['image']:r for r in rows}
    train=old.cache_views(out,fold,model,gpu,[lookup[n] for n in specification['train_images']])
    val=inference_records(out,fold,model,gpu,'val')
    excluded=inference_records(out,fold,model,gpu,'oof')
    if frozen.g.state_digest(model)!=before: raise ValueError('Frozen auxiliary changed during extraction')
    if frozen.tensor_bytes(train+val+excluded)>frozen.SETTINGS['max_cpu_tensor_bytes']:
        raise ValueError('Combined CPU cache exceeds1GiB bound')
    del model; torch.cuda.empty_cache()
    if any(not any(r['domain']==d and r['eligible'] for r in train) for d in ('real','sim')):
        source_failed(out,fold,'midpoint_no_eligible_TRAIN'); return
    # Two-scale eligible TRAIN only. VAL and excluded TRAIN never enter update().
    set_random_seed(1703,deterministic=True)
    head=SigmaMidpointHead(1.5).cuda(gpu); optimizer=frozen.optimizer_for(head)
    guard=old.training_repair.MemoryPlateau(); updates=0; saved=[]
    with (folder/'selected_midpoint_train.jsonl').open('x') as log:
        for epoch in contract.SELECTION['midpoint_epochs']:
            for indices in frozen.epoch_batches(train,epoch):
                value=frozen.update(head,optimizer,train,indices,torch.device('cuda',gpu)); updates+=1
                torch.cuda.synchronize(gpu)
                value['memory']=guard.observe(torch.cuda.memory_allocated(gpu),torch.cuda.memory_reserved(gpu))
                log.write(json.dumps(dict(epoch=epoch,step=updates,**value))+'\n')
            head_path=folder/('midpoint_epoch_%02d.pth'%epoch)
            payload=dict(protocol=contract.VERSION,fold=fold,epoch=epoch,updates=updates,sigma_cells=1.5,
                detector_sha256=selected['sha256'],train_images=specification['train_images'],
                head_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()})
            frozen.atomic_save(head_path,payload)
            if not frozen.tree_equal(payload,torch.load(str(head_path),map_location='cpu')):
                raise ValueError('Auxiliary midpoint exact reload failed')
            saved.append(dict(epoch=epoch,path=str(head_path),sha256=old.sha(head_path)))
    # Release training optimizer/gradients before validation. No VAL allocator growth in training stage.
    del head,optimizer,train; torch.cuda.empty_cache()
    candidates=[]
    for item in saved:
        head=SigmaMidpointHead(1.5).cuda(gpu)
        payload=torch.load(item['path'],map_location='cpu'); head.load_state_dict(payload['head_state'],strict=True)
        predictions=read_head(head,val,gpu); del head,payload; torch.cuda.empty_cache()
        candidates.append(dict(item,metrics=contract.selection_metrics(predictions,rows),
            offline_metrics=contract.offline_metrics(predictions,rows)))
        old.dump_rows(folder/('midpoint_VAL_epoch_%02d.jsonl.gz'%item['epoch']),predictions)
    chosen=contract.select(candidates,'midpoint')
    old.write(folder/'midpoint_selection.json',dict(protocol=contract.VERSION,kind='midpoint',
        candidates=candidates,selected=chosen,detector_selected_once=selected,contract=contract.SELECTION,TEST_read=False))
    if chosen is None: source_failed(out,fold,'midpoint_selection_no_feasible_checkpoint'); return
    if old.sha(chosen['path'])!=chosen['sha256']: raise ValueError('Selected midpoint changed')
    head=SigmaMidpointHead(1.5).cuda(gpu)
    payload=torch.load(chosen['path'],map_location='cpu'); head.load_state_dict(payload['head_state'],strict=True)
    predictions=read_head(head,excluded,gpu)
    for prediction,record in zip(predictions,excluded):
        prediction.update(excluded_fold=fold,auxiliary_B=record['boxes_original'][0].tolist() if len(record['boxes_original']) else None,
            selected_detector_sha256=selected['sha256'],selected_midpoint_sha256=chosen['sha256'])
    old.dump_rows(folder/'oof_predictions.jsonl.gz',sorted(predictions,key=lambda r:r['image']))
    gate=old.core.source_gate(predictions,require_two_classes=False)
    old.write(folder/'oof_source_gate.json',gate)
    old.write(folder/'selected_source_completion.json',dict(protocol=contract.VERSION,
        detector=selected,midpoint=chosen,updates=updates,support=old.core.support(predictions),
        source_gate=gate,TRAIN_sequences_excluded=specification['excluded_sequences'],
        original_VAL_used_for_selection=True,TEST_read=False,GT_used_online=False))
    if not gate['passed']: source_failed(out,fold,'fold_oof_source')


def read_head(head,records,gpu):
    import torch
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as frozen
    head.eval().requires_grad_(False); before=frozen.g.state_digest(head)
    predictions=base_rows(records); guard=old.training_repair.MemoryPlateau()
    with torch.no_grad():
        for row,record in zip(predictions,records):
            inputs=[record[k].cuda(gpu) for k in ('roi','support','boxes_original','boxes_model','scale_xy')]
            output=head(*inputs)['boxes_original'].cpu()
            base=record['boxes_original']
            if len(output)!=len(base) or not torch.equal(output[:,5],base[:,5]):
                raise ValueError('Auxiliary midpoint score/count changed')
            row['pred']=output[0].tolist() if len(output) else None
            del inputs,output
            torch.cuda.synchronize(gpu)
            guard.observe(torch.cuda.memory_allocated(gpu),torch.cuda.memory_reserved(gpu))
    if frozen.g.state_digest(head)!=before: raise ValueError('Frozen head changed')
    return predictions


def assess(out):
    """Same judge/control contract, explicit new selected-source provenance."""
    rows,policy,plan,source,proof=old.checked(); auxiliary=[]; selection={}
    for fold in ('A','B'):
        selection[fold]=json.loads((out/fold/'selected_source_completion.json').read_text())
        if not selection[fold]['source_gate']['passed']: raise ValueError('Source gate failed')
        for key in ('detector','midpoint'):
            selected=selection[fold][key]
            if old.sha(selected['path'])!=selected['sha256']: raise ValueError('Selected weights changed')
        with gzip.open(out/fold/'oof_predictions.jsonl.gz','rt') as f:
            predictions=[json.loads(line) for line in f]
        for row in predictions:
            if (row['selected_detector_sha256']!=selection[fold]['detector']['sha256'] or
                    row['selected_midpoint_sha256']!=selection[fold]['midpoint']['sha256']):
                raise ValueError('OOF prediction selected-source identity changed')
        auxiliary+=predictions
    train=[r for r in rows if r['reliability_role']=='train']
    oof=old.core.materialize_training(train,auxiliary); source_gate=old.core.source_gate(oof)
    old.write(out/'training_material.json',dict(source_gate=source_gate,original=old.core.support(train),oof=old.core.support(oof)))
    old.dump_rows(out/'OOF_TRAIN.jsonl.gz',oof)
    if not source_gate['passed']: source_failed(out,None,'combined_source'); return
    common={r['image'] for r in oof if r['pred'] is not None}&{r['image'] for r in train if r['pred'] is not None}
    models=dict(in_sample=old.core.fit(train),oof=old.core.fit(oof),
        common_in_sample=old.core.fit([r for r in train if r['image'] in common]),
        common_oof=old.core.fit([r for r in oof if r['image'] in common]))
    import numpy as np
    for key in ('weights','mean','scale'):
        if not np.allclose(models['in_sample'][key],policy['simple_policy']['models']['size'][key],atol=1e-10,rtol=1e-10):
            raise ValueError('Same-TRAIN control did not reproduce')
    val=old.core.score([r for r in rows if r['reliability_role']=='val'],models)
    cutoffs={name:old.ranking.calibrate(val,name) for name in old.core.METHODS}
    stats=old.core.statistics(val,cutoffs); gate=old.core.gate(stats)
    for method in models:
        for row in val:
            decision=old.ranking.decide(row['original_simple_decision'],row['risks'][method],cutoffs[method]['risk_le'])
            if any(decision[k]!=row['original_simple_decision'][k] for k in ('final_box_original','center_accepted','angle_accepted')):
                raise ValueError('Formal box/center/angle changed')
    report=dict(protocol=contract.VERSION,status='VAL_PASS_FROZEN_TEST_PENDING' if gate['passed'] else 'VAL_FAILED_STOP',
        proof=proof,auxiliary_selection=selection,selection_contract=contract.SELECTION,
        VAL_role='shared_development_auxiliary_selection_and_final_judge_evaluation',
        independent_VAL_claim=False,models=models,cutoffs=cutoffs,statistics=stats,gate=gate,
        center=old.metrics.center(val),ranking={m:old.ranking.ranking_summary(val,m) for m in old.core.METHODS},
        protections=dict(box_score_count_center_angle_preserved=True),TEST_read=False,original_policy_replaced=False)
    old.write(out/'report.json',report); old.dump_rows(out/'scored_VAL.jsonl.gz',val)
    candidate=deepcopy(policy)
    candidate['simple_policy']['models']['size']=models['oof']
    candidate['simple_policy']['cutoffs']['simple']['size']=cutoffs['oof']
    candidate['simple_policy']['calibration_role']='size: shared_development_full_VAL_retention95_per_domain_video; angle: unchanged'
    candidate['candidate_identity']=dict(protocol=contract.VERSION,gate_passed=gate['passed'],automatically_deployed=False,
        auxiliary_models_required_online=False,report_sha256=old.sha(out/'report.json'))
    old.simple.validate_policy(candidate['simple_policy']); old.write(out/'candidate_policy.json',candidate)
    old.write(out/'completion.json',dict(protocol=contract.VERSION,status=report['status'],adopted=False,
        VAL_used_for_auxiliary_selection=True,TEST_read=False,report_sha256=old.sha(out/'report.json')))
    lines=['# VAL选权的辅助OOF来源对照','',
        '**状态：'+report['status']+'**','',
        'VAL同时用于辅助检测/midpoint选权与判断器开发评价；不是独立验证。TEST未读取。','',
        '| 方法 | 接受数 | FA | FR | ED | CR | AUROC |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method in old.core.METHODS:
        value=stats[method]['all']; counts=value['actual']['states']
        lines.append('|%s|%d|%d|%d|%d|%d|%s|'%(method,value['actual']['accepted_outputs'],
            counts['FA'],counts['FR'],counts['ED'],counts['CR'],str(value['error_AUROC'])))
    (out/'analysis.md').write_text('\n'.join(lines)+'\n')


def stage(out,mode,fold=None,gpu=0):
    command=[sys.executable,__file__,'--mode',mode,'--out',str(out),'--gpu',str(gpu)]
    if fold: command+=['--fold',fold]
    subprocess.run(command,cwd=str(ROOT),check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=('diagnose','selected','prepare-diagnose','prepare-selected',
        'trace','replay','detector','select-detector','midpoint','assess'))
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--fold',choices=('A','B'))
    parser.add_argument('--snapshot-boundary',choices=('anomaly','previous'),default='anomaly')
    args=parser.parse_args(); out=args.out.resolve()
    if ROOT/'work_dirs' not in out.parents: raise ValueError('Outputs must remain under project work_dirs')
    try:
        if args.mode.startswith('prepare-'): prepare(out,args.mode.split('-',1)[1]); return
        if args.mode in ('diagnose','selected'):
            prepare(out,args.mode)
            if args.mode=='diagnose': stage(out,'trace',gpu=args.gpu); return
            stage_legacy=[sys.executable,str(ROOT/'crane_project/tools/run_port_reliability_oof_v1.py'),
                '--mode','preflight','--out',str(out),'--gpu',str(args.gpu)]
            subprocess.run(stage_legacy,cwd=str(ROOT),check=True)
            for fold in ('A','B'):
                for mode in ('detector','select-detector','midpoint'):
                    stage(out,mode,fold,args.gpu)
                    if (out/'completion.json').exists(): return
            stage(out,'assess',gpu=args.gpu); return
        identity=checked(out)
        if args.mode=='trace': diagnose(out); return
        if args.mode=='replay': replay(out,args.gpu,args.snapshot_boundary); return
        if identity['purpose']!='selected': raise ValueError('Diagnostic weights cannot enter selected-source workflow')
        if args.mode in ('detector','select-detector','midpoint') and args.fold is None:
            raise ValueError('Fold required')
        if args.mode=='detector':
            old.check_fold(out,args.fold)
            if not json.loads((out/'resource_preflight.json').read_text())['passed']: raise ValueError('Missing resource preflight')
            subprocess.run([sys.executable,str(ROOT/'tools/train.py'),str(out/args.fold/'detector_config.py'),
                '--no-validate','--seed','1701','--deterministic'],cwd=str(ROOT),check=True)
        elif args.mode=='select-detector': select_detector(out,args.fold,args.gpu)
        elif args.mode=='midpoint': midpoint(out,args.fold,args.gpu)
        elif args.mode=='assess': assess(out)
    except Exception as exc:
        if out.exists() and not (out/'failure.json').exists():
            old.write(out/'failure.json',dict(mode=args.mode,error=repr(exc),traceback=traceback.format_exc(),TEST_read=False))
        raise


if __name__=='__main__': main()
