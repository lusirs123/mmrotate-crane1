#!/usr/bin/env python3
"""Replay the reviewed matched step200, then freeze it on exposed TEST.

The original probe gate failed. This is an authorized exploratory diagnosis,
not formal training, new model selection, or an untouched final confirmation.
No new loss/seed/step/threshold choices. Original midpoint sources stay intact.
"""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import preflight_port_geometry_midpoint_v1 as t
from crane_project.tools.eval_crane_offline import CraneOfflineEvaluator, parse_dota_txt

g, m, a = t.g, t.m, t.a
evaluate, batch = t.evaluate, t.batch
VERSION = 'port_geometry_midpoint_v1_exposed_test_diagnosis'
PROTOCOL = ROOT/'crane_project/tools/port_geometry_midpoint_v1_test_protocol.json'
MANIFEST = ROOT/'crane_project/tools/port_geometry_midpoint_v1_test_sources.json'
REVIEW_SHA = 'a3d980c8d88c978caed4c669cf409a50e5467daed3f191f7c817ba5d83414005'
FINAL_STATE = dict(parameters='6ac5486e6f63f1a086a068fdd901e09ff62b07ccd5ad3f6f70a666bb0f68fbfa',
    buffers='44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a',
    parameter_count=17696, buffer_count=0)
TEST_COUNTS = dict(real_seq03=200, real_seq04=668, sim_seq09=572)
TEST_ANN_SHA = 'e0dbb1bd8aea7209314d8ed60bc44e965550ed606135cc0016e1075d717de13e'
TEST_IMAGE_SHA = '6448e47b9a715f47f04eacaaabb54a32b522c72208bf6e82793284de2aad665d'


def protocol_document():
    return dict(protocol=VERSION, evidence_role='fixed_short_fit_head_on_repeatedly_exposed_TEST',
        original_train_completion_sha256=REVIEW_SHA, original_probe_gate_passed=False,
        settings=deepcopy(m.SETTINGS), final_matched_state=FINAL_STATE,
        test_counts=TEST_COUNTS, test_annotations_sha256=TEST_ANN_SHA,
        test_image_identity_sha256=TEST_IMAGE_SHA, frozen_b=g.ready.FROZEN_B,
        replay='Exactly original matched initial state, 200 fit-only minibatches and fit implementation. Require entire result including all logs/128 rows and final state bitwise equal before reading TEST. No shuffled replay or new fitting choices.',
        checkpoint='Diagnostic midpoint head only; save via file stream, reload/state-verify before TEST. Not a new detector checkpoint or formal-trained model.',
        prediction='Fresh same-run frozen B ep24, single original TEST pipeline, one feature extraction and three native head calls/frame (raw,original,raw-after). B and midpoint paired on identical runtime outputs. Historical TEST metrics never substituted.',
        online='Only detached aligned P3 ROI/support, B original/model boxes and transform scales. No GT/domain/sequence input, ranking, quality gating, temporal smoothing or new thresholds.',
        missing='Every fixed frame retained. Missing B stays missing; invalid/out-of-bound correction falls back to full B under original midpoint rules. Scores exactly B.',
        evaluation='B/delivered/raw candidate, real/sim/each sequence/overall; center three denominators, both edges, pure periodic angle and separate protocol penalty, strict RIoU and full-video protocol-v2 temporal metrics. Raw invalid candidates are diagnostic, not delivered output coverage.',
        scope=dict(replay_head_updates=200,test_head_updates=0,detector_updates=0,
            max_test_feature_extractions=1440,max_native_head_calls=4320,
            val_access=False,formal_training=False,formal_training_approved=False,
            test_repeatedly_exposed=True,selection_on_test=False),
        limitation='Failed repeated TRAIN probe remains failed regardless of TEST outcome. No claim of stable gain, independent confirmation, depth precision or final-model replacement.')


def replay_fit(records,batches,initial,condition,device,progress):
    # This body is mechanically compared with the unchanged original fit_arm.
    # Only retain the final head rather than discarding it after evaluation.
    head=m.SpatialMidpointHead().to(device); head.load_state_dict(initial,strict=True)
    before=g.state_digest(head)
    start,_=evaluate(head,records,condition,device)
    if any(row['pred']!=r['boxes_original'][0].tolist() or not row['accepted']
           for row,r in zip(start['rows'],records)):
        raise ValueError('Zero midpoint correction is not exact delivered B')
    optimizer=torch.optim.Adam(head.parameters(),lr=m.SETTINGS['lr'],weight_decay=m.SETTINGS['weight_decay'])
    logs=[]; head.train()
    for step,indices in enumerate(batches,1):
        roi,support,b,bm,scales,gt=batch(records,indices,condition,device)
        optimizer.zero_grad()
        output=head(roi,support,b,bm,scales)
        target=m.target_points(gt,b,bm,scales)
        loss,parts=m.point_loss(output['points_roi'],target)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite point loss')
        task_norms=([float(torch.autograd.grad(p/4.,head.output.weight,retain_graph=True)[0].double().norm()) for p in parts] if step==1 else None)
        if task_norms is not None and not all(x>0 and math.isfinite(x) for x in task_norms):
            raise ValueError('Initial four-point task gradient ineffective')
        loss.backward()
        params=list(head.parameters())
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
            raise ValueError('Missing/nonfinite midpoint gradient')
        stem_norm=sum(float(p.grad.double().square().sum()) for p in head.stem.parameters())**.5
        if step==2 and stem_norm<=0:
            raise ValueError('Stem gradient ineffective after zero-output step')
        norm=float(torch.nn.utils.clip_grad_norm_(params,m.SETTINGS['clip_norm']))
        after=sum(float(p.grad.double().square().sum()) for p in params)**.5
        if not math.isfinite(norm) or not math.isfinite(after) or after>m.SETTINGS['clip_norm']+1e-4:
            raise ValueError('Invalid midpoint gradient clipping')
        optimizer.step()
        progress(dict(stage='update',arm=condition,step=step))
        if any(not bool(torch.isfinite(p).all()) for p in params):
            raise ValueError('Nonfinite updated midpoint head')
        log=dict(stage='fit',arm=condition,step=step,loss=float(loss.detach()),
            point_parts=parts.detach().cpu().tolist(),grad_norm_before_clip=norm,
            grad_norm_after_clip=after,clip_multiplier=min(1.,m.SETTINGS['clip_norm']/(norm+1e-6)),
            stem_grad_norm=stem_norm,initial_point_task_grad_norms=task_norms)
        logs.append(log); progress(log)
    final,maps=evaluate(head,records,condition,device,keep_maps=True)
    result=dict(initial_state=before,final_state=g.state_digest(head),initial=start,final=final,logs=logs,
        completed_updates=len(logs),finite_gradients=True,
        initial_four_point_gradients_effective=all(x>0 for x in logs[0]['initial_point_task_grad_norms']),
        stem_step2_effective=logs[1]['stem_grad_norm']>0)
    del optimizer
    torch.cuda.empty_cache()
    return result,maps,head


def checked_replay_body():
    old=ast.parse(inspect.getsource(t.fit_arm)).body[0]
    new=ast.parse(inspect.getsource(replay_fit)).body[0]
    if (ast.dump(old.args)!=ast.dump(new.args) or
            ast.dump(ast.Module(body=old.body[:-3],type_ignores=[]))!=
            ast.dump(ast.Module(body=new.body[:-3],type_ignores=[]))):
        raise ValueError('Replay fitting differs from original matched implementation')
    if (ast.dump(old.body[-3])!=ast.dump(ast.parse('del head,optimizer').body[0]) or
            ast.dump(new.body[-3])!=ast.dump(ast.parse('del optimizer').body[0]) or
            ast.dump(old.body[-2])!=ast.dump(new.body[-2]) or
            ast.dump(old.body[-1])!=ast.dump(ast.parse('return result,maps').body[0]) or
            ast.dump(new.body[-1])!=ast.dump(ast.parse('return result,maps,head').body[0])):
        raise ValueError('Replay may only retain the evaluated final head')


def checked_contract(baseline_path,review_path):
    checked_replay_body()
    protocol,samples,identity,baseline=t.checked_contract(baseline_path)
    manifest=json.loads(MANIFEST.read_text())
    paths=set(identity['sources'])|{str(t.MANIFEST.relative_to(ROOT)),
        str(PROTOCOL.relative_to(ROOT)),
        'crane_project/tools/eval_port_geometry_midpoint_v1_test.py',
        'tests/test_port_geometry_midpoint_v1_test.py'}
    sources={p:g.ready.sha(ROOT/p) for p in paths}
    if (manifest.get('protocol')!=VERSION or manifest.get('sources')!=sources or
            manifest.get('parent_manifest_sha256')!=g.ready.sha(t.MANIFEST) or
            json.loads(PROTOCOL.read_text())!=protocol_document()):
        raise ValueError('Fixed TEST diagnostic source/protocol differs')
    if g.ready.sha(review_path)!=REVIEW_SHA:
        raise ValueError('Requires exact reviewed midpoint completion, not a new run')
    reviewed=json.loads(Path(review_path).read_text())
    if (reviewed['status']!='TRAIN_MIDPOINT_CHECK_COMPLETE_REVIEW_REQUIRED' or
            reviewed['identity']!=identity or reviewed['settings']!=m.SETTINGS or
            reviewed['head_updates_total']!=400 or reviewed['minibatches']!=baseline['minibatches'] or
            reviewed['arms']['matched']['final_state']!=FINAL_STATE or
            reviewed['review']['diagnostic_continuation_conditions_met'] or
            reviewed['formal_training_approved']):
        raise ValueError('Original failed-probe experiment identity differs')
    return samples,identity,baseline,reviewed,dict(protocol=VERSION,
        sources=sources,manifest_sha256=g.ready.sha(MANIFEST),
        protocol_sha256=g.ready.sha(PROTOCOL),train_review_sha256=REVIEW_SHA,
        prior_identity=identity)


def verify_replay(result,reviewed,head):
    if (g.json_native(result)!=reviewed['arms']['matched'] or
            g.state_digest(head)!=reviewed['arms']['matched']['final_state'] or
            result['completed_updates']!=m.SETTINGS['steps_per_arm']):
        raise ValueError('TRAIN matched replay is not exact; stop before TEST, do not change tolerance')


def verify_native_runtime(expected,current):
    fields=('torch','cuda','cudnn','mmcv','mmdet','mmrotate','opencv','gpu')
    if any(k not in expected or k not in current or expected[k]!=current[k] for k in fields):
        raise ValueError('Native detector libraries/GPU differ from original frozen ROI extraction')


def save_head(path,head,identity):
    """File-stream ZIP save avoids the known server hidden-temp-path failure."""
    state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    digest=g.state_digest(head)
    payload=dict(protocol=VERSION,identity=identity,settings=m.SETTINGS,
        formal_training=False,diagnostic_only=True,final_state=digest,state_dict=state)
    fd,tmp=tempfile.mkstemp(prefix='midpoint-head-',suffix='.pth',dir=str(path.parent))
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
        os.link(tmp,str(path))
    finally:
        os.unlink(tmp)
    loaded=torch.load(str(path),map_location='cpu')
    if (loaded['identity']!=identity or loaded['settings']!=m.SETTINGS or
            loaded['final_state']!=digest or set(loaded['state_dict'])!=set(state) or
            any(not torch.equal(state[k],loaded['state_dict'][k]) for k in state)):
        raise ValueError('Diagnostic head save/reload identity differs')
    head.load_state_dict(loaded['state_dict'],strict=True)
    if g.state_digest(head)!=digest:
        raise ValueError('Reloaded diagnostic head state differs')
    return dict(path=path.name,sha256=g.ready.sha(path),state=digest,
                save_reload_exact=True,diagnostic_only=True)


def fixed_test_inputs():
    """Called only after exact TRAIN replay and saved-head freeze verification."""
    manifest_path=g.ready.DATA/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    selected=[r for r in manifest['records'] if r['split']=='test']
    expected={r['id']:r for r in selected}
    anns=g.ready.files(g.ready.DATA/'test/annfiles','.txt')
    images=g.ready.files(g.ready.DATA/'test/images','.jpg')
    if (len(expected)!=1440 or len(selected)!=1440 or len(anns)!=1440 or len(images)!=1440 or
            {p.stem for p in anns}!=set(expected) or {p.stem for p in images}!=set(expected) or
            Counter(r['sequence'] for r in selected)!=Counter(TEST_COUNTS) or
            g.ready.set_sha(anns)!=TEST_ANN_SHA):
        raise ValueError('Fixed TEST names/counts/annotation bytes differ')
    rows=[]; image_digest=hashlib.sha256()
    for ann in anns:
        name=ann.stem; old=expected[name]; seq,number=name.rsplit('_',1)
        image=g.ready.DATA/'test/images'/(name+'.jpg')
        sha=g.ready.sha(image)
        if (old['sequence']!=seq or old['images']['path']!='test/images/'+image.name or
                old['annfiles']['path']!='test/annfiles/'+ann.name or
                old['images']['sha256']!=sha or old['annfiles']['sha256']!=g.ready.sha(ann)):
            raise ValueError('Fixed TEST frame provenance differs: '+name)
        image_digest.update(image.name.encode()); image_digest.update(b'\0'); image_digest.update(bytes.fromhex(sha))
        g.ready.polygon_box(ann)  # strict one-grab polygon format/geometry guard
        gt=parse_dota_txt(str(ann))
        if len(gt)!=1:
            raise ValueError('Fixed TEST requires exactly one GT: '+name)
        with Image.open(image) as im:
            size=list(im.size)
        rows.append(dict(image=name,sequence=seq,frame_id=int(number),domain=seq.split('_')[0],
            gt=gt[0].tolist(),image_size=size,image_sha256=sha,annotation_sha256=old['annfiles']['sha256']))
    if image_digest.hexdigest()!=TEST_IMAGE_SHA:
        raise ValueError('Fixed TEST image bytes differ')
    return rows,dict(frames=len(rows),manifest_sha256=g.ready.sha(manifest_path),
        annotation_sha256=TEST_ANN_SHA,image_identity_sha256=TEST_IMAGE_SHA,
        sequence_counts=dict(Counter(r['sequence'] for r in rows)))


def build_detector(cfg,gpu,expected_state):
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector as build
    from crane_project.tools.diagnose_port_shape_e_h_train_gradients_v1 import checkpoint_contract
    if g.ready.sha(g.CHECKPOINT)!=g.ready.FROZEN_B['checkpoint_sha256']:
        raise ValueError('Requires unchanged VAL-selected B epoch24')
    import_modules_from_strings(**cfg.custom_imports)
    model_cfg=deepcopy(cfg.model); model_cfg.pretrained=None; model_cfg.train_cfg=None
    detector=build(model_cfg)
    loaded=load_checkpoint(detector,str(g.CHECKPOINT),map_location='cpu',strict=True)
    checkpoint_contract(loaded['meta'],cfg,'b'); del loaded
    detector.cuda(gpu); g.freeze_detector(detector)
    if g.state_digest(detector)!=expected_state:
        raise ValueError('B state differs from original TRAIN cache')
    return detector


def build_test_dataset(cfg,sources):
    from mmrotate.datasets import build_dataset
    spec=deepcopy(cfg.data.test)
    if ((ROOT/spec.get('data_root','')/spec['ann_file']).resolve()!=(g.ready.DATA/'test/annfiles').resolve() or
            (ROOT/spec.get('data_root','')/spec['img_prefix']).resolve()!=(g.ready.DATA/'test/images').resolve() or
            spec['pipeline']!=cfg.data.val.pipeline):
        raise ValueError('Requires unchanged single-scale B TEST pipeline/paths')
    spec['test_mode']=True
    dataset=build_dataset(spec)
    if len(dataset)!=1440 or [Path(v['filename']).stem for v in dataset.data_infos]!=[s['image'] for s in sources]:
        raise ValueError('TEST loader order/count differs')
    return dataset


def load_view(dataset,index,source,gpu):
    from mmcv.parallel import collate,scatter
    value=scatter(collate([dataset[index]],samples_per_gpu=1),[gpu])[0]
    image,metas=value['img'][0],value['img_metas'][0]
    if (len(value['img'])!=1 or image.shape!=(1,3,1024,1024) or len(metas)!=1 or
            Path(metas[0]['filename']).stem!=source['image'] or
            list(metas[0]['ori_shape'][:2][::-1])!=source['image_size'] or metas[0].get('flip',False)):
        raise ValueError('TEST view identity/augmentation differs')
    g.checked_meta(metas[0])
    return image,metas


def capture(detector,head,image,metas,audit=None):
    """Online boundary: no GT, domain/sequence or error/eligibility inputs."""
    g.assert_detector_frozen(detector)
    if head.training or any(p.requires_grad for p in head.parameters()):
        raise ValueError('TEST midpoint head must be frozen')
    with torch.no_grad():
        features=detector.extract_feat(image)
        if audit is not None:
            audit(dict(stage='test_feature_extracted'))
        if features[0].shape!=(1,256,128,128) or any(f.requires_grad or f.grad_fn is not None for f in features):
            raise ValueError('Frozen TEST P3 shape/graph differs')
        def native(rescale):
            value=detector.simple_test_from_features(features,metas,rescale=rescale)
            if audit is not None:
                audit(dict(stage='test_native_head_called',rescale=rescale))
            return g.flatten_prediction(value)
        raw_np=native(False)
        original_np=native(True)
        raw,original=image.new_tensor(raw_np),image.new_tensor(original_np)
        restored=g.map_boxes(raw[:,:5],metas[0],inverse=True)
        if (raw.shape!=original.shape or not torch.allclose(restored,original[:,:5],atol=1e-4,rtol=1e-6) or
                not torch.equal(raw[:,5],original[:,5])):
            raise ValueError('Native TEST B original-coordinate restoration differs')
        roi,support,_=g.sample_local(features[0],raw[:,:5],metas[0],'aligned')
        scales=original.new_tensor([np.asarray(metas[0]['scale_factor']).reshape(-1)[:2]]).reshape(1,2)
        scales=scales.expand(len(original),-1)
        output=head(roi,support,original,raw[:,:5],scales)
        after=native(False)
        if not np.array_equal(raw_np,after) or not torch.equal(output['boxes_original'][:,5],original[:,5]):
            raise ValueError('Midpoint branch changed B raw output or score')
        if len(output['boxes_original'])!=len(original):
            raise ValueError('Midpoint may not add/drop B outputs')
        present=bool(len(original))
        result=dict(b=original[0].cpu().tolist() if present else None,
            midpoint=output['boxes_original'][0].cpu().tolist() if present else None,
            candidate=(output['candidate_original'][0].cpu().tolist() if present and bool(output['candidate_valid'][0]) else None),
            candidate_valid=bool(output['candidate_valid'][0]) if present else None,
            accepted=bool(output['accepted'][0]) if present else None,
            failed_checks=[k for k,v in output['checks'].items() if present and not bool(v[0])],
            predicted_points_original=output['points_original'][0].cpu().tolist() if present else None,
            rectangle_projection_rms_px=float(output['rectangle_projection_rms_px'][0]) if present else None,
            support_mean=float(support.mean()) if present else None,
            original_b_raw_exact_before_after=True)
    return result


def failure_intervals(rows,kind):
    intervals=[]; run=[]; previous=None
    def flush():
        if run:
            intervals.append(dict(sequence=run[0]['sequence'],start=run[0]['frame_id'],
                end=run[-1]['frame_id'],length=len(run)))
            run.clear()
    for row in sorted(rows,key=lambda r:(r['sequence'],r['frame_id'])):
        key=(row['sequence'],row['frame_id'])
        if previous is None or key[0]!=previous[0] or key[1]!=previous[1]+1:
            flush()
        metrics=row['metrics']
        fail=(not metrics['output'] if kind=='no_output' else
              not metrics['center_hit'] if kind=='center' else metrics['riou']<.5)
        if fail:
            run.append(row)
        else:
            flush()
        previous=key
    flush()
    return intervals


def summary(rows,continuous=True):
    value=g.summarize(rows,continuous=continuous)
    count=len(rows); out=sum(r['metrics']['output'] for r in rows)
    hits=sum(r['metrics']['center_hit'] for r in rows)
    riou=[r['metrics']['riou'] for r in rows if r['metrics']['output']]
    value.update(output_coverage_fraction=dict(numerator=out,denominator=count),
        conditional_center_correct_fraction=dict(numerator=hits,denominator=out),
        all_frame_center_correct_fraction=dict(numerator=hits,denominator=count),
        riou_p10=float(np.percentile(riou,10)) if riou else None,
        riou_min=min(riou) if riou else None)
    if continuous:
        # The historical DOTA evaluator receives canonical long-axis boxes.
        # Normalize ONLY this offline metric view; preserve raw online w/h,
        # angle association and all six delivered values in test_rows.jsonl.
        offline=[dict(domain=r['domain'],seq_id=r['sequence'].split('_',1)[1],
            frame_id=r['frame_id'],gt_box=g.ready.canonical(r['gt']),
            pred_box=g.ready.canonical(r['pred'][:5]) if r['pred'] is not None else None,
            score=r['pred'][5] if r['pred'] is not None else 0.,plc_rope=None,image=r['image']) for r in rows]
        value['metric_protocol_v2']=CraneOfflineEvaluator(mode='test').evaluate_records(offline)
        value['failure_intervals']={kind:failure_intervals(rows,kind) for kind in ('no_output','center','riou')}
        value['longest_failure_run_frames']={kind:max((r['length'] for r in runs),default=0)
            for kind,runs in value['failure_intervals'].items()}
    return value


def test_summaries(records):
    groups=dict(overall=records,real=[r for r in records if r['domain']=='real'],
        sim=[r for r in records if r['domain']=='sim'])
    groups.update({seq:[r for r in records if r['sequence']==seq] for seq in TEST_COUNTS})
    result={}
    for key,rows in groups.items():
        result[key]={}
        for name in ('b','midpoint','candidate'):
            effective=[dict(r,pred=r[name],metrics=r['metrics'][name]) for r in rows]
            if name!='candidate':
                print('FIXED exploratory TEST summary',key,'method='+name,flush=True)
            result[key][name]=summary(effective,continuous=name!='candidate')
        common=[r for r in rows if r['b'] is not None]
        changes=[]
        for r in rows:
            old,new=r['metrics']['b'],r['metrics']['midpoint']
            changes.append(dict(image=r['image'],
                new_center_failure=old['center_hit'] and not new['center_hit'],
                recovered_center=not old['center_hit'] and new['center_hit'],
                new_riou_below_0_5=old['riou']>=.5 and new['riou']<.5,
                recovered_riou=old['riou']<.5 and new['riou']>=.5,
                new_zero_overlap=old['riou']>0 and new['riou']<=0,
                center_delta_px=new.get('center_error_px',0)-old.get('center_error_px',0),
                riou_delta=new['riou']-old['riou']))
        result[key]['paired']=dict(common_output_frames=len(common),
            fallbacks=sum(r['accepted'] is False for r in common),
            invalid_candidates=sum(not r['candidate_valid'] for r in common),
            fallback_reasons=dict(Counter(reason for r in common for reason in r['failed_checks'])),
            changes=changes)
    return result


def publish(out,report):
    g.write_new(out/'artifacts.json',dict(protocol=VERSION,status=report['status'],
        files={p.name:g.ready.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='artifacts.json'}))


def run(args):
    os.chdir(ROOT)
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('Single logical GPU only; do not use torchrun/DDP')
    out=Path(args.out_dir).resolve(); out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    report=dict(protocol=VERSION,status='STARTED',formal_training=False,formal_training_approved=False,
        original_probe_gate_passed=False,test_repeatedly_exposed=True,selection_on_test=False,
        replay_head_updates=0,test_head_updates=0,detector_updates=0,test_frames=0,
        feature_extractions=0,native_head_calls=0,cache_loads=0,test_access=False)
    with (out/'progress.jsonl').open('x') as stream:
        def progress(value):
            if value['stage']=='update':
                report['replay_head_updates']+=1
            if value['stage']=='test_feature_extracted':
                report['feature_extractions']+=1
            if value['stage']=='test_native_head_called':
                report['native_head_calls']+=1
            stream.write(json.dumps(g.json_native(value),ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()
            if value['stage']=='fit' and (value['step'] in (1,2) or value['step']%25==0):
                print('TRAIN exact matched replay step',value['step'],'loss',value['loss'],flush=True)
        progress(dict(stage='begin'))
        try:
            samples,identity,baseline,reviewed,evaluation=checked_contract(
                Path(args.baseline_report).resolve(),Path(args.train_report).resolve())
            report['identity']=evaluation
            g.write_new(out/'protocol.json',protocol_document()); shutil.copyfile(str(MANIFEST),str(out/'sources.json'))
            if args.check_only:
                report['status']='STATIC_MIDPOINT_TEST_CONTRACT_COMPLETE_NO_TEST_CACHE_GPU_UPDATES'
            else:
                runtime=a.checked_runtime(baseline)
                if (any(runtime[k]!=reviewed['runtime'][k] for k in ('python','torch','cuda','cudnn','numpy','opencv')) or
                        not torch.cuda.is_available() or not 0<=args.gpu<torch.cuda.device_count()):
                    raise ValueError('Exact replay requires original runtime and valid logical GPU')
                torch.cuda.set_device(args.gpu); device=torch.device('cuda',args.gpu)
                runtime['gpu']=torch.cuda.get_device_name(args.gpu)
                if runtime['gpu']!=reviewed['runtime']['gpu']:
                    raise ValueError('Exact replay requires original GPU model')
                report['runtime']=runtime; cache_dir=Path(args.cache_dir).resolve()
                if g.ready.sha(cache_dir/'cache_manifest.json')!=a.CACHE_MANIFEST_SHA:
                    raise ValueError('Requires unchanged reviewed TRAIN ROI cache')
                report['cache_loads']+=1
                payload,cache=g.checked_cache(cache_dir,identity['prior_identity']['g_identity'],samples)
                records=payload['records']; g.validate_records(records,samples); a.check_cache_reference(records,cache,baseline)
                import mmcv,mmdet,mmrotate
                native_runtime=dict(torch=runtime['torch'],cuda=runtime['cuda'],cudnn=runtime['cudnn'],
                    opencv=runtime['opencv'],gpu=runtime['gpu'],mmcv=mmcv.__version__,
                    mmdet=mmdet.__version__,mmrotate=mmrotate.__version__)
                verify_native_runtime(payload['extraction_runtime'],native_runtime)
                report['native_runtime']=native_runtime
                prepared,_=t.prepare_records(records,samples,t.protocol_document(samples)['donor_mapping'])
                frozen_b=payload['detector_state_before']
                g.seed_all(); initial_head=m.SpatialMidpointHead(); initial=deepcopy(initial_head.state_dict()); del initial_head
                batches=g.schedule(prepared)
                if batches!=reviewed['minibatches']:
                    raise ValueError('Fixed original matched fit-only schedule differs')
                (result,maps,head),cost=g.measured(lambda:replay_fit(prepared,batches,initial,'matched',device,progress),args.gpu)
                verify_replay(result,reviewed,head)
                if report['replay_head_updates']!=200:
                    raise ValueError('Exact replay update budget differs')
                report['replay_exact']=True; report['replay_cost']=cost
                g.write_new(out/'matched_replay.json',dict(protocol=VERSION,result=result,cost=cost,exact=True))
                report['head_checkpoint']=save_head(out/'fixed_matched_head.pth',head,evaluation)
                head.eval(); head.requires_grad_(False)
                for p in head.parameters():
                    p.grad=None
                head_before=g.state_digest(head)
                del payload,records,prepared,initial,batches,result,maps
                torch.cuda.empty_cache()
                progress(dict(stage='TRAIN_REPLAY_EXACT_HEAD_SAVED_RELOADED_FROZEN_BEFORE_TEST',state=head_before))
                report['test_access']=True
                sources,data_identity=fixed_test_inputs(); report['test_data']=data_identity
                cfg=g.check_cfg(); dataset=build_test_dataset(cfg,sources)
                detector=build_detector(cfg,args.gpu,frozen_b)
                b_before=g.state_digest(detector); report['frozen_b_state']=b_before
                all_rows=[]; costs=[]
                with (out/'test_rows.jsonl').open('x') as row_stream:
                    for index,source in enumerate(sources):
                        image,metas=load_view(dataset,index,source,args.gpu)
                        prediction,cost=g.measured(lambda:capture(detector,head,image,metas,progress),args.gpu)
                        # TEST labels enter only after the online prediction has completed.
                        row=dict(source,**prediction)
                        row['metrics']={name:g.decompose(source['gt'],row[name][:5] if row[name] is not None else None)
                            for name in ('b','midpoint','candidate')}
                        row['gpu_cost']=cost
                        row_stream.write(json.dumps(g.json_native(row),ensure_ascii=False,allow_nan=False)+'\n'); row_stream.flush()
                        all_rows.append(row); costs.append(cost); report['test_frames']+=1
                        progress(dict(stage='test_frame',image=source['image'],index=index+1))
                        if (index+1)%100==0 or index==0:
                            print('FIXED exploratory TEST',index+1,'/1440',source['image'],flush=True)
                        del image,metas,prediction
                g.assert_detector_frozen(detector)
                if (g.state_digest(detector)!=b_before or g.state_digest(head)!=head_before or
                        any(p.grad is not None for p in detector.parameters()) or
                        any(p.grad is not None for p in head.parameters()) or
                        g.ready.sha(out/'fixed_matched_head.pth')!=report['head_checkpoint']['sha256'] or
                        report['test_frames']!=1440 or report['feature_extractions']!=1440 or report['native_head_calls']!=4320):
                    raise ValueError('Frozen TEST states/budget changed')
                if any((r['b'] is None)!=(r['midpoint'] is None) or
                        (r['b'] is not None and r['b'][5]!=r['midpoint'][5]) for r in all_rows):
                    raise ValueError('Midpoint changed B missing-output or score contract')
                report['summary']=test_summaries(all_rows)
                report['test_head_and_detector_states_unchanged']=True
                report['test_cost']=dict(total_frame_ms=sum(c['elapsed_ms'] for c in costs),
                    max_peak_allocated_mib=max(c['peak_allocated_mib'] for c in costs),
                    max_peak_reserved_mib=max(c['peak_reserved_mib'] for c in costs))
                report['status']='FIXED_MIDPOINT_EXPOSED_TEST_COMPLETE_REVIEW_REQUIRED'
            report['elapsed_seconds']=time.perf_counter()-started
            progress(dict(stage='complete',status=report['status']))
            g.write_new(out/'completion.json',report)
        except Exception as error:
            report.update(status='FAILED_REVIEW_REQUIRED',error=type(error).__name__+': '+str(error),
                elapsed_seconds=time.perf_counter()-started)
            progress(dict(stage='failed',error=report['error']))
            g.write_new(out/'completion.json',report); publish(out,report)
            raise
    publish(out,report)
    print('Saved',out/'completion.json','status',report['status'],flush=True)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true',help='TRAIN/report/source contract only; no TEST reads, cache load, GPU or updates.')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--baseline-report',default=str(a.BASELINE))
    p.add_argument('--train-report',default='work_dirs/port_geometry_midpoint_v1_train/completion.json')
    p.add_argument('--cache-dir',default='work_dirs/port_geometry_g_v1_roi_cache')
    p.add_argument('--out-dir',required=True)
    return p


if __name__=='__main__':
    run(parser().parse_args())
