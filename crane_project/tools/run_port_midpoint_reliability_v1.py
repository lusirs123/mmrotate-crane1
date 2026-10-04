#!/usr/bin/env python3
"""Migrate reliability to frozen B+formal-midpoint, without training geometry.

check -> collect cached TRAIN/VAL final boxes -> fit simple -> online smoke.
probe independently compares two readers on 14 fixed TRAIN images, using the
existing reference epoch04. There is no TEST route or reader threshold search.
"""
import argparse
from copy import copy
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_size_reference_v1 as reference
from crane_project.tools import run_port_simple_reliability_v1 as base
from crane_project.tools import analyze_port_geometry_midpoint_size_temporal_v1 as formal_check
from crane_project.utils import port_midpoint_reliability_v1 as new
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_separability_v1 as rank

PROTOCOL = ROOT/'crane_project/tools/port_midpoint_reliability_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_midpoint_reliability_v1_sources.json'


def checked_sources():
    protocol = json.loads(PROTOCOL.read_text()); manifest = json.loads(SOURCES.read_text())
    actual = {p: base.sha(ROOT/p) for p in manifest['sources']}
    if (protocol['protocol'] != new.VERSION or manifest['protocol'] != new.VERSION
            or actual != manifest['sources'] or manifest['protocol_sha256'] != base.sha(PROTOCOL)
            or manifest['parent_reference_manifest_sha256'] != base.sha(reference.MANIFEST)
            or manifest['parent_formal_manifest_sha256'] != base.sha(ROOT/'crane_project/tools/port_geometry_midpoint_formal_v1_sources.json')
            or protocol['test_read']
            or any(protocol[k] for k in ('detector_updates', 'midpoint_updates', 'reference_updates'))):
        raise ValueError('Midpoint reliability source/protocol contract differs')
    return protocol, dict(sources=actual, manifest_sha256=base.sha(SOURCES))


def prepare(args):
    protocol, sources = checked_sources()
    original_args = copy(args)
    original_args.policy = args.old_policy
    old = reference.prepare(original_args)
    selection = json.loads(args.selection.read_text())
    if args.selection.name != 'selection.json': raise ValueError('Use completed formal selection.json')
    def read(name):
        role, filename = name.split('/', 1)
        if role != 'training' or Path(filename).name != filename:
            raise ValueError('Only completed formal TRAIN/VAL files allowed')
        return (args.selection.parent/filename).read_bytes()
    val, _, proof = formal_check.validate_inputs(read)
    chosen = proof['selected_checkpoint']
    if (chosen['epoch'] != protocol['midpoint_epoch'] or chosen['sha256'] != protocol['midpoint_sha256']
            or base.sha(args.selection.parent/chosen['path']) != chosen['sha256']):
        raise ValueError('Requires reviewed formal midpoint epoch23, not latest file/short-fit head')
    if (selection['identity']['frozen_b']['checkpoint_sha256'] != old[8]['frozen_b']['checkpoint_sha256']
            or base.sha(args.b_checkpoint) != old[8]['frozen_b']['checkpoint_sha256']):
        raise ValueError('Reference and midpoint must share the exact frozen B')
    cache = json.loads((args.formal_cache/'cache_manifest.json').read_text())
    if (base.sha(args.formal_cache/'cache_manifest.json') != selection['cache_manifest_sha256']
            or cache['identity'] != selection['identity'] or cache['test_access']
            or cache['detector_updates'] != 0 or cache['status'] != 'COMPLETE_FROZEN_B_TRAIN_VAL_CACHE'
            or cache['record_counts'] != {'train_s1':2558, 'train_s05':2558, 'val_s1':887}):
        raise ValueError('Requires completed original formal TRAIN/VAL ROI cache')
    final = new.paired_rows(old[3], val)
    front_end = dict(name=protocol['front_end_name'], frozen_b=old[8]['frozen_b'],
        midpoint_checkpoint=chosen, selection_sha256=base.sha(args.selection),
        formal_sources_sha256=selection['identity']['sources_sha256'])
    contract = dict(protocol=protocol, sources=sources, front_end=front_end,
        original_reference_contract_sha256=simple.fingerprint(old[8]), formal_proof=proof,
        formal_cache_manifest_sha256=base.sha(args.formal_cache/'cache_manifest.json'),
        native_VAL_final_rows_sha256=simple.fingerprint(final), test_read=False)
    return protocol, old, selection, cache, final, contract


def finish(args, contract, status):
    if prepare(args)[5] != contract: raise ValueError('Inputs/sources changed during stage')
    artifacts = {p.name: base.sha(p) for p in sorted(args.out_dir.iterdir()) if p.is_file()}
    base.write_new(args.out_dir/'completion.json', dict(status=status, contract=contract,
        artifacts=artifacts, mode=args.mode, detector_updates=0, midpoint_updates=0,
        reference_updates=0, test_read=False))
    print('Saved', args.out_dir, status, flush=True)


def completed(directory, expected, contract):
    directory = Path(directory)
    if (directory/'failure.json').exists(): raise ValueError('Preserve failed output; use a completed stage')
    proof = json.loads((directory/'completion.json').read_text())
    if proof['status'] != expected or proof['contract'] != contract or proof['test_read']:
        raise ValueError('Wrong preceding stage or front-end contract')
    for name, sha in proof['artifacts'].items():
        if base.sha(directory/name) != sha: raise ValueError('Preceding-stage artifact differs: '+name)
    return proof


def load_midpoint(args, selection, cache, gpu):
    import torch
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as formal
    path = args.selection.parent/selection['selected_checkpoint']['path']
    payload = torch.load(str(path), map_location='cpu')
    if (payload['identity'] != selection['identity'] or payload['epoch'] != 23
            or payload['protocol'] != formal.VERSION
            or payload['updates'] != selection['selected_checkpoint']['updates']
            or payload['head_digest'] != selection['selected_checkpoint']['head_digest']
            or payload['context']['manifest_sha256'] != selection['cache_manifest_sha256']
            or payload['context']['detector_state'] != cache['detector_state']
            or payload['context']['runtime'] != cache['runtime']):
        raise ValueError('Formal midpoint checkpoint/cache metadata differs')
    device = 'cuda:'+str(gpu)
    formal.old.verify_native_runtime(cache['runtime'], formal.runtime(gpu))
    formal.seed_all()
    head = formal.m.SpatialMidpointHead().to(device)
    head.load_state_dict(payload['head_state'], strict=True); head.eval().requires_grad_(False)
    if formal.g.state_digest(head) != selection['selected_checkpoint']['head_digest']:
        raise ValueError('Formal midpoint state differs')
    return formal, torch, head, device


def collect(args, prepared):
    _, old, selection, cache, _, contract = prepared
    formal, torch, head, device = load_midpoint(args, selection, cache, args.gpu)
    before = formal.g.state_digest(head); counts = {}; val_max_box_difference = 0.
    with (args.out_dir/'predictions.jsonl').open('x') as stream:
        for shard, role, sources in (('train_s1', 'train', old[2]), ('val_s1', 'val', old[3])):
            path = args.formal_cache/(shard+'.pt')
            if base.sha(path) != cache['files'][path.name]: raise ValueError('Formal ROI shard SHA differs')
            payload = torch.load(str(path), map_location='cpu')
            if payload['identity'] != selection['identity'] or payload['detector_state'] != cache['detector_state']:
                raise ValueError('Formal ROI shard identity differs')
            records = payload['records']
            if len(records) != len(sources) or any(r['scale'] != 1. or r['role'] != role for r in records):
                raise ValueError('Only full native-scale formal TRAIN/VAL allowed')
            predictions = formal.evaluate(head, records, device)
            rows = new.paired_rows(sources, predictions)
            if role == 'val':
                saved = {r['image']:r for r in prepared[4]}
                for r in rows:
                    for key in ('pred', 'b_original', 'midpoint_accepted', 'midpoint_candidate'):
                        a, b = r[key], saved[r['image']][key]
                        if (a is None) != (b is None) or (isinstance(a, list) and
                                (a[5]!=b[5] or not np.allclose(a[:5], b[:5], atol=1e-4, rtol=1e-6))) or (isinstance(a, bool) and a != b):
                            raise ValueError('Cached formal head differs from selected VAL: '+r['image'])
                        if isinstance(a,list): val_max_box_difference=max(val_max_box_difference,float(np.max(np.abs(np.array(a[:5])-b[:5]))))
                # Scoring uses the already reviewed selected VAL, after replay agreement.
                # Do not silently replace locked geometry with numerical replay variants.
                rows = deepcopy(prepared[4])
            for r in rows: stream.write(json.dumps(dict(r, reliability_role=role), allow_nan=False)+'\n')
            counts[role] = len(rows); stream.flush()
            print('Frozen midpoint cached', role, len(rows), 'no detector inference/updates', flush=True)
            del records, payload, predictions, rows
    if formal.g.state_digest(head) != before: raise ValueError('Collection updated midpoint')
    base.write_new(args.out_dir/'collection_report.json', dict(status='MIDPOINT_TRAIN_VAL_FINAL_BOXES_COLLECTED',
        counts=counts, detector_inferences=0, image_or_P3_extractions=0,
        midpoint_state_before=before, midpoint_state_after=formal.g.state_digest(head),
        native_scale_only=True, cached_native_VAL_replay_close=True, test_read=False,
        final_VAL_records_from_reviewed_selection=True,VAL_replay_max_absolute_box_difference=val_max_box_difference,
        peak_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20))
    return 'MIDPOINT_TRAIN_VAL_COLLECTION_COMPLETE'


def read_collection(path, contract, original):
    path = Path(path)
    completed(path, 'MIDPOINT_TRAIN_VAL_COLLECTION_COMPLETE', contract)
    rows = [json.loads(x) for x in (path/'predictions.jsonl').read_text().splitlines()]
    if len(rows) != 3445 or any(r.get('reliability_role') not in ('train', 'val') for r in rows):
        raise ValueError('Unexpected collection role/count')
    roles = {r:[x for x in rows if x['reliability_role']==r] for r in ('train', 'val')}
    for role, previous in (('train', original[2]), ('val', original[3])):
        if len(roles[role]) != len(previous) or len({r['image'] for r in roles[role]}) != len(previous):
            raise ValueError('Incomplete midpoint TRAIN/VAL collection')
        indexed = {r['image']:r for r in previous}
        if set(indexed) != {r['image'] for r in roles[role]}: raise ValueError('Wrong collection role')
        for r in roles[role]:
            source = indexed[r['image']]
            for key in ('gt','image_size','sequence','domain','frame_id','split','train_angle_eligible','angle_axis_well_defined'):
                if r[key] != source[key]: raise ValueError('Collection source metadata changed')
            simple.prediction(r['pred']); simple.prediction(r['b_original'])
            if (r['pred'] is None) != (r['b_original'] is None) or (r['pred'] is not None and r['pred'][5] != r['b_original'][5]):
                raise ValueError('Collection changed output/score')
            new.paired_rows([source], [dict(image=r['image'], sequence=r['sequence'],
                domain=r['domain'], frame_id=r['frame_id'], gt=r['gt'],
                b=r['b_original'], midpoint=r['pred'], accepted=r['midpoint_accepted'],
                candidate=r['midpoint_candidate'])])
    # The stage adds only its role marker; compare the reviewed underlying rows.
    bare = [{k:v for k,v in r.items() if k != 'reliability_role'} for r in roles['val']]
    if simple.fingerprint(bare) != contract['native_VAL_final_rows_sha256']:
        raise ValueError('Collected VAL differs from selected formal VAL')
    return roles['train'], roles['val']


def ranked(rows, policy):
    result = {}
    groups = {d:[r for r in rows if r['domain']==d] for d in ('real','sim')}
    groups.update({s:[r for r in rows if r['sequence']==s] for s in sorted({r['sequence'] for r in rows})})
    for group, subset in groups.items():
        result[group] = {}
        for component, field, limit in (('size','size_max_relative',.1),('angle','angle_deg',3.)):
            out = [r for r in subset if r['pred'] is not None and
                   (component!='angle' or r['angle_axis_well_defined'])]
            bad = [simple.geometry_errors(r['gt'],r['pred'])[field]>limit for r in out]
            x = np.array([simple.descriptor(r['pred'],r['image_size']) for r in out]).reshape(-1,3)
            result[group][component] = {name:rank.rank_metrics(risks,bad) for name,risks in
                [('score',[1-r['pred'][5] for r in out]),('simple',simple.linear_risk(policy['models'][component],x))]}
    return result


def fit(args, prepared):
    protocol, old, _, _, _, contract = prepared
    train, val = read_collection(args.collection_dir, contract, old)
    before = simple.fingerprint([train,val])
    old_protocol, _ = base.checked_sources()
    internal = simple.create_policy(train, val, old_protocol)
    policy = dict(protocol=new.VERSION, front_end=contract['front_end'], contract=contract,
        simple_policy=internal, test_read=False,
        internal_legacy_center_policy_means_retain_FINAL_output=True)
    base.write_new(args.out_dir/'policy.json', policy)
    loaded = json.loads((args.out_dir/'policy.json').read_text())
    runtime = new.MidpointReliability(loaded, contract['front_end'])
    original_runtime = new.MidpointReliability(policy, contract['front_end'])
    _, current_stats = simple.evaluate(val, internal)
    _, previous_stats = simple.evaluate(val, old[4])
    b_rows = [dict(r,pred=r['b_original']) for r in val]
    _, b_stats = simple.evaluate(b_rows, old[4])
    with (args.out_dir/'val_decisions.jsonl').open('x') as stream:
        for r in val:
            decisions = {m:runtime.decide(r['pred'],r['image_size'],m) for m in ('raw','score_only','simple')}
            if decisions != {m:original_runtime.decide(r['pred'],r['image_size'],m) for m in decisions}:
                raise ValueError('Policy save/reload changed decisions')
            if any(d['final_box_original'] != r['pred'] for d in decisions.values()):
                raise ValueError('Reliability modified final detection box')
            stream.write(json.dumps(dict(image=r['image'],methods=decisions), allow_nan=False)+'\n')
    if policy != loaded or simple.fingerprint([train,val]) != before:
        raise ValueError('Save/reload or fitting modified detection records')
    support = {}
    for domain in ('real','sim'):
        out = [r for r in train if r['domain']==domain and r['pred'] is not None]
        support[domain] = dict(outputs=len(out), frames=sum(r['domain']==domain for r in train),
            center_bad=sum(simple.geometry_errors(r['gt'],r['pred'])['center_px']>=old_protocol['error_limits']['center_px'] for r in out),
            size_bad=sum(simple.geometry_errors(r['gt'],r['pred'])['size_max_relative']>.1 for r in out),
            angle_bad_eval=sum(simple.geometry_errors(r['gt'],r['pred'])['angle_deg']>3 for r in out if r['angle_axis_well_defined']),
            angle_bad_train=sum(simple.geometry_errors(r['gt'],r['pred'])['angle_deg']>3 for r in out if r['train_angle_eligible']))
    report = dict(status='MIDPOINT_SIMPLE_REFIT_COMPLETE_REVIEW_REQUIRED', front_end=contract['front_end'],
        models=internal['models'],cutoffs=internal['cutoffs'],genuine_TRAIN_error_support=support,
        error_limits=old_protocol['error_limits'],
        paired_formal_B_old_policy=b_stats,midpoint_old_policy=previous_stats,
        midpoint_refit_policy=current_stats,VAL_rank=ranked(val,internal),
        limitations=['TRAIN refit and VAL coverage calibration are descriptive, not independent validation.',
            'New Gaussian reader is an unqualified separate candidate; these flags use simple only.',
            'Same score/missing count does not imply the same geometry correctness labels.',
            'Internal reused helper policy retains its historical B field names; exported box is the final midpoint box.'],
        save_reload_policy_exact=True, detection_records_unchanged=True,test_read=False)
    base.write_new(args.out_dir/'fit_report.json',report)
    for domain in ('real','sim'):
        s=current_stats['domain:'+domain]
        print('midpoint simple',domain,'outputs',s['output_frames'],'/',s['frames'],
            'center/output',s['center_hit_rate_on_outputs'],'full-frame center',s['all_frame_center_correct_coverage'],flush=True)
    return 'MIDPOINT_SIMPLE_POLICY_FIT_COMPLETE'


def online_modules(args, prepared, with_reference):
    import torch
    from mmdet.datasets.pipelines import Compose
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as formal
    _, old, selection, cache, _, _ = prepared
    formal.old.verify_native_runtime(cache['runtime'],formal.runtime(args.gpu)); formal.seed_all()
    torch.cuda.reset_peak_memory_stats(args.gpu)
    detector, head, cfg = formal.load_selected_pipeline(args.selection,args.gpu)
    model = None
    if with_reference:
        reference.gate(args.reference_checkpoint.parent/'train_report.json',
            'FIXED_EPOCH4_SIZE_REFERENCE_TRAIN_COMPLETE_REVIEW_REQUIRED',old[8])
        payload = reference.load_checkpoint(args.reference_checkpoint,old[8],'reference_train')
        if (base.sha(args.reference_checkpoint) != prepared[0]['reference_sha256'] or payload['epoch'] != 4
                or payload['steps'] != 1536 or payload['b_state'] != base.state_digest(detector)):
            raise ValueError('Requires original reference epoch04 and identical frozen B features')
        from crane_project.utils.port_size_reference_v1_torch import SizeReference
        model=SizeReference().cuda(args.gpu).eval().requires_grad_(False)
        model.load_state_dict(payload['state'],strict=True)
    return formal,torch,detector,head,Compose(deepcopy(cfg.data.val.pipeline)),model


def midpoint_from_features(formal,torch,detector,head,features,meta):
    """GT-free adapter of the existing midpoint inference, on shared frozen P3."""
    if (detector.training or head.training or any(p.requires_grad for m in (detector,head) for p in m.parameters())
            or features[0].shape != (1,256,128,128)
            or any(f.requires_grad or f.grad_fn is not None for f in features)):
        raise ValueError('Only frozen evaluation modules and detached features allowed')
    with torch.no_grad():
        raw_np=formal.g.flatten_prediction(detector.simple_test_from_features(features,[meta],rescale=False))
        b_np=formal.g.flatten_prediction(detector.simple_test_from_features(features,[meta],rescale=True))
        raw=features[0].new_tensor(raw_np); b=features[0].new_tensor(b_np)
        if (len(b)>1 or raw.shape!=b.shape or not torch.equal(raw[:,5],b[:,5]) or
                not torch.allclose(formal.g.map_boxes(raw[:,:5],meta,inverse=True),b[:,:5],atol=1e-4,rtol=1e-6)):
            raise ValueError('Native coordinate restoration differs')
        roi,support,_=formal.g.sample_local(features[0],raw[:,:5],meta,'aligned')
        xy=b.new_tensor(np.asarray(meta['scale_factor'])[:2]).reshape(1,2).expand(len(b),-1)
        output=head(roi,support,b,raw[:,:5],xy)
        after=formal.g.flatten_prediction(detector.simple_test_from_features(features,[meta],rescale=False))
        if not np.array_equal(raw_np,after) or len(output['boxes_original'])!=len(b) or not torch.equal(output['boxes_original'][:,5],b[:,5]):
            raise ValueError('Side computation changed B output/score/count')
        return dict(b=b[0].cpu().tolist() if len(b) else None,
            midpoint=output['boxes_original'][0].cpu().tolist() if len(b) else None)


def probe_summary(records):
    groups = {'all':records}
    groups.update({role:[r for r in records if r['reference_role']==role] for role in ('fit','holdout')})
    result={}
    for role,rows in groups.items():
        result[role]={}
        common = [r for r in rows if all(r['references'][m]['defined'] for m in ('moments','template'))]
        result[role]['common_defined'] = dict(views=len(common),
            size_error_mean={m:float(np.mean([r['reference_errors'][m] for r in common])) if common else None
                for m in ('moments','template')})
        for method in ('moments','template'):
            defined=[r for r in rows if r['references'][method]['defined']]
            errors=[r['reference_errors'][method] for r in defined]
            result[role][method]=dict(views=len(rows),defined=len(defined),
                unavailable=[r['image'] for r in rows if not r['references'][method]['defined']],
                size_error_mean=float(np.mean(errors)) if errors else None,
                within_10_percent=sum(e<=.1 for e in errors),
                long_both_sides=sum(min(r['probes'][method]['size_probe_risk_gaps'][k] for k in ('long_minus','long_plus'))>1e-6 for r in defined),
                short_both_sides=sum(min(r['probes'][method]['size_probe_risk_gaps'][k] for k in ('short_minus','short_plus'))>1e-6 for r in defined))
    return result


def probe(args,prepared):
    protocol, old, _, _, _, contract=prepared
    selected=new.fixed_probe_rows(old[6]); numerical=[]
    for row in selected:
        if row['reference_role']!='fit': continue
        meta=reference.numeric_meta(row); target,_=reference.size.target_map(row['gt'],meta)
        fitted=new.template_reference(target,row['gt'],meta,protocol)
        g=simple.canonical(row['gt'])
        error=float(max(abs(fitted['long_original_px']/g[2]-1),abs(fitted['short_original_px']/g[3]-1))) if fitted['defined'] else None
        numerical.append(dict(image=row['image'],reference=fitted,error=error,
            passed=error is not None and error<=protocol['probe']['numerical_max_relative_error']))
    if not all(r['passed'] for r in numerical):
        base.write_new(args.out_dir/'probe_report.json',dict(status='TEMPLATE_NUMERICAL_GATE_FAILED',numerical=numerical,test_read=False))
        return 'TEMPLATE_NUMERICAL_GATE_FAILED'
    formal,torch,detector,head,pipeline,model=online_modules(args,prepared,True)
    before={name:base.state_digest(value) for name,value in [('b',detector),('midpoint',head),('reference',model)]}
    records=[]
    for row in selected:
        features,meta,_=reference.view(row,old[7],detector,pipeline,args.gpu)
        first=midpoint_from_features(formal,torch,detector,head,features,meta)
        with torch.no_grad(): probability=model(features[0]).sigmoid()[0,0].cpu().numpy()
        pred=first['midpoint']; refs={}; probes={}; errors={}
        for method in ('moments','template'):
            function=reference.size.reference_from_map if method=='moments' else new.template_reference
            refs[method]=function(probability,pred[:5],meta,protocol) if pred is not None else dict(defined=False,reason='no_detector_output')
            diagnostic=reference.probe_record(row,refs[method],protocol); probes[method]=diagnostic
            g=simple.canonical(row['gt'])
            errors[method]=float(max(abs(refs[method]['long_original_px']/g[2]-1),abs(refs[method]['short_original_px']/g[3]-1))) if refs[method]['defined'] else None
        after=midpoint_from_features(formal,torch,detector,head,features,meta)
        if first!=after: raise ValueError('Reference/readers changed final midpoint prediction')
        target,valid=reference.size.target_map(row['gt'],meta)
        np.savez_compressed(args.out_dir/(row['image']+'.npz'),probability=probability,target=target,
            valid=valid,metadata_json=np.array(json.dumps(meta,default=lambda value:np.asarray(value).tolist())))
        record=dict(image=row['image'],domain=row['domain'],reference_role=row['reference_role'],
            gt=row['gt'],final_box_original=pred,b_original=first['b'],references=refs,probes=probes,
            reference_errors=errors,final_midpoint_before_after_exact=True,
            offline_map_evidence=new.offline_map_evidence(probability,row['gt'],pred,meta,protocol) if pred is not None else None)
        new.save_map_preview(args.out_dir/(row['image']+'.png'),probability,target,valid)
        records.append(record); print('Fixed TRAIN readers',row['image'],errors,flush=True)
        del features,probability,target
    after={name:base.state_digest(value) for name,value in [('b',detector),('midpoint',head),('reference',model)]}
    if before!=after: raise ValueError('Probe changed frozen model state')
    with (args.out_dir/'probe_rows.jsonl').open('x') as stream:
        for row in records: stream.write(json.dumps(row,allow_nan=False)+'\n')
    base.write_new(args.out_dir/'probe_report.json',dict(status='FIXED_MIDPOINT_TRAIN_READER_COMPARISON_COMPLETE_REVIEW_REQUIRED',
        numerical_fit_only=numerical,summary=probe_summary(records),state_before=before,state_after=after,
        reference_checkpoint_sha256=base.sha(args.reference_checkpoint),new_reader_is_not_automatically_deployed=True,
        no_VAL_image_inference=True,test_read=False,synthetic_probes_are_not_genuine_error_accuracy=True,
        peak_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20))
    return 'FIXED_MIDPOINT_TRAIN_READER_COMPARISON_COMPLETE_REVIEW_REQUIRED'


def smoke(args,prepared):
    _,old,_,_,final,contract=prepared
    completed(args.policy.parent,'MIDPOINT_SIMPLE_POLICY_FIT_COMPLETE',contract)
    policy=json.loads(args.policy.read_text())
    if policy['contract']!=contract: raise ValueError('Wrong policy contract')
    runtime=new.MidpointReliability(policy,contract['front_end'])
    formal,torch,detector,head,pipeline,_=online_modules(args,prepared,False)
    before=dict(b=base.state_digest(detector),midpoint=base.state_digest(head))
    selected=[]
    for seq in sorted({r['sequence'] for r in final}):
        group=sorted((r for r in final if r['sequence']==seq),key=lambda r:r['frame_id'])
        selected.extend([group[0],group[-1]])
    rows=[]
    for row in selected:
        features,meta,_=reference.view(row,old[7],detector,pipeline,args.gpu)
        first=midpoint_from_features(formal,torch,detector,head,features,meta)
        decision=runtime.decide(first['midpoint'],row['image_size'])
        after=midpoint_from_features(formal,torch,detector,head,features,meta)
        expected=row['pred']; actual=first['midpoint']
        if first!=after or decision['final_box_original']!=actual:
            raise ValueError('Reliability changed final midpoint output')
        if (expected is None)!=(actual is None) or (actual is not None and
                (expected[5]!=actual[5] or not np.allclose(expected[:5],actual[:5],atol=1e-4,rtol=1e-6))):
            raise ValueError('Online formal midpoint differs from frozen VAL: '+row['image'])
        rows.append(dict(image=row['image'],**decision)); del features
    after=dict(b=base.state_digest(detector),midpoint=base.state_digest(head))
    if before!=after: raise ValueError('Reliability smoke changed detection state')
    base.write_new(args.out_dir/'smoke_report.json',dict(status='MIDPOINT_SIMPLE_ONLINE_SMOKE_PASS',
        frames=len(rows),decisions=rows,state_before=before,state_after=after,GT_online=False,
        final_boxes_scores_and_missing_outputs_preserved=True,reference_reader_used=False,test_read=False))
    return 'MIDPOINT_SIMPLE_ONLINE_SMOKE_PASS'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('check','collect','probe','fit','smoke'),required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--selection',type=Path,default=Path('work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1/selection.json'))
    parser.add_argument('--formal-cache',type=Path,default=Path('work_dirs/port_geometry_midpoint_formal_v1_roi_cache'))
    parser.add_argument('--collection-dir',type=Path,default=Path('work_dirs/port_midpoint_reliability_v1_collect'))
    parser.add_argument('--policy',type=Path,default=Path('work_dirs/port_midpoint_reliability_v1_fit/policy.json'))
    parser.add_argument('--reference-checkpoint',type=Path,default=Path('work_dirs/port_size_reference_v1_train/epoch_04.pth'))
    parser.add_argument('--train-cache',type=Path,default=Path('work_dirs/port_reliability_train_support_v1_cache'))
    parser.add_argument('--input-snapshot',type=Path,default=Path('work_dirs/port_reliability_train_support_v1/train_input_snapshot.json'))
    parser.add_argument('--val-dir',type=Path,default=Path('work_dirs/port_reliability_branches_v1_val_cached_v1'))
    parser.add_argument('--old-policy',dest='old_policy',type=Path,default=Path('work_dirs/port_simple_reliability_v1_fit/policy.json'))
    parser.add_argument('--b-checkpoint',type=Path,default=Path('work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth'))
    args=parser.parse_args()
    prepared=prepare(args); args.out_dir.mkdir(parents=True,exist_ok=False)
    base.write_new(args.out_dir/'input_check.json',dict(contract=prepared[5],mode=args.mode))
    try:
        if args.mode=='check': status='MIDPOINT_RELIABILITY_STATIC_INPUTS_PASS'
        elif args.mode=='collect': status=collect(args,prepared)
        elif args.mode=='fit': status=fit(args,prepared)
        elif args.mode=='probe': status=probe(args,prepared)
        else: status=smoke(args,prepared)
        finish(args,prepared[5],status)
    except Exception as error:
        base.write_new(args.out_dir/'failure.json',dict(mode=args.mode,type=type(error).__name__,error=str(error)))
        raise


if __name__=='__main__': main()
