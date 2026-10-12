#!/usr/bin/env python3
"""Fixed uniform epoch04/Gaussian1.5 epoch03: fresh paired VAL gate then TEST.

One frozen detector extraction feeds both heads. No optimization, reselection,
parameter overrides or cache rebuilding. Current detector bytes are bound and
must reproduce historical B/Gaussian outputs before TEST can be reported.
"""
import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import train_port_geometry_midpoint_readout_v1 as r

f, g, torch, e = r.f, r.g, r.torch, r.e
VERSION = 'port_geometry_midpoint_readout_v1_fixed_test'
PROTOCOL = ROOT/'crane_project/tools/port_geometry_midpoint_readout_test_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_geometry_midpoint_readout_test_v1_sources.json'
DEFAULT_TRAIN = 'work_dirs/port_geometry_midpoint_readout_v1/formal_20261012_gpu2_v1/train'
DEFAULT_REFERENCE = 'docs/detection/ap_evidence_20261010/test_source'
FIXED = dict(path='head_epoch_04.pth', epoch=4, updates=3608,
    sha256='d0ec9a60ee33b0843128d65e5a3127291d5948926e14128bfbe7b8d07acb0c05',
    readout=r.model.READOUT,
    head_digest=dict(parameters='e0ff50258f6e1d84ef78607de40b5a82aeb8f27f05d97609961bc43c236505df',
        buffers='44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a',
        parameter_count=17696, buffer_count=0), save_reload_exact=True)
VAL_STATUS = 'READOUT_FIXED_ONLINE_VAL_PASS_BEFORE_TEST'


def protocol_document():
    return dict(protocol=VERSION, fixed_uniform=deepcopy(FIXED), fixed_gaussian=deepcopy(e.FIXED),
        frozen_b=deepcopy(g.ready.FROZEN_B), selection_config=deepcopy(f.SELECTION_CONFIG),
        static='Current91-source hash closure plus original87 training closure, full24 selection and selected weight byte SHA. No tensor/dataset/GPU/TEST input loads.',
        source_boundary='Bind all current detector bytes. Historical5 detector differences remain recorded, not claimed unused here. Fresh full887 VAL must reproduce both saved selected-head rows/decisions and metrics before TEST. Frozen detector state must equal original extraction.',
        inference='One frozen B extraction and3 native calls per image, shared aligned P3 ROI/support/native original/model boxes/actual sx-sy to both frozen heads. No GT/domain/sequence in capture. Same isotropic pipeline, raw axes, scores/counts and whole-box fallback.',
        val_gate='887 fresh standard-scale VAL frames for both heads; unchanged box atol1e-4/rtol1e-6, exact decisions/GT/scores; exact aggregate summary. No new selection. TEST reads require indexed successful matching VAL report checked before TEST input access.',
        test='1440 previously exposed fixed TEST frames, fixed three conditions B/uniform04/Gaussian1.5-03. Historical B/Gaussian TEST per-frame outputs and summaries must reproduce; fail on drift, do not widen tolerances or replace old evidence.',
        reporting='Existing full_video_protocol_v2 metrics, separate output-frame center hit/output coverage/all-frame center coverage, pure/penalty sim angles, candidates, acceptance/fallback/invalid and sequences. No new AP in this run.',
        scope=dict(detector_updates=0, head_updates=0, selection_on_test=False, automatic_promotion=False,
            cache_rebuilt=False, test_repeatedly_exposed=True, depth_accuracy_verified=False),
        limitations=['One seed; TEST repeatedly exposed and Gaussian engineering adoption followed prior TEST.',
            'Contrast removes Gaussian prior while retaining neutral residual subtraction; no separate subtraction/root-cause claim.'])


def checked_sources():
    training = r.checked_sources()
    fixed = r.t.read(SOURCES)
    required = set(training['sources']) | {
        str(r.SOURCES.relative_to(ROOT)), str(PROTOCOL.relative_to(ROOT)),
        'crane_project/tools/eval_port_geometry_midpoint_readout_test_v1.py',
        'tests/test_port_geometry_midpoint_readout_test_v1.py'}
    if (fixed['protocol'] != VERSION or set(fixed['sources']) != required or
            fixed['training_sources_sha256'] != r.t.sha(r.SOURCES) or r.t.read(PROTOCOL) != protocol_document()):
        raise ValueError('Fixed TEST source/protocol scope differs')
    actual = {name:r.t.sha(ROOT/name) for name in required}
    if actual != fixed['sources']:
        raise ValueError('Fixed TEST current source SHA differs')
    return dict(training_identity=training, sources=actual,
        protocol_sha256=r.t.sha(PROTOCOL), sources_sha256=r.t.sha(SOURCES))


def indexed(directory, names, protocol):
    directory = Path(directory)
    index = r.t.read(directory/'artifacts.json')
    if index['protocol'] != protocol:
        raise ValueError('Artifact protocol differs')
    digests = {}
    for name in names:
        path = directory/name
        digest = r.t.sha(path)
        if index['files'].get(name) != digest:
            raise ValueError('Artifact SHA differs: '+name)
        digests[name] = digest
    return digests


def checked_training(args, identity):
    # Reuse original TRAIN/VAL contract; it reads hashes/text, never tensor shards.
    cache, baseline, gaussian, proof = r.checked_contract(args.baseline_dir, args.reference_selection, args.cache_dir)
    directory = Path(args.training_dir)
    names = {'completion.json','selection.json','selected_val_compare.json', 'val_readout_compare.json',
        FIXED['path'], 'val_epoch_04.rows.jsonl'} | {'val_epoch_%02d.json'%x for x in range(1,25)}
    digests = indexed(directory, names, r.VERSION)
    complete = r.t.read(directory/'completion.json'); selection = r.t.read(directory/'selection.json')
    table = r.t.read(directory/'val_readout_compare.json'); value = r.t.read(directory/'selected_val_compare.json')
    expected = {'head_epoch_%02d.pth'%x for x in range(1,25)}
    if (complete['status'] != 'READOUT_FULL_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED' or
            complete['stage'] != 'train' or complete['protocol'] != r.VERSION or
            complete['head_updates_total'] != 21648 or complete['epochs_completed'] != 24 or
            complete['schedule']['epoch_steps'] != [902]*24 or complete['proof'] != proof or
            complete['selection'] != selection or proof['identity'] != identity['training_identity'] or
            selection['selected_checkpoint'] != FIXED or selection['protocol'] != r.VERSION or
            selection['readout'] != r.model.READOUT or selection['split'] != 'val' or
            selection['selection_config'] != f.SELECTION_CONFIG or selection['proof'] != proof or
            set(selection['all_checkpoints']) != expected or set(table['all_epoch_values']) != expected or
            table['proof'] != proof or table['split'] != 'val' or
            complete['runtime'] != cache['runtime'] or selection['cache_manifest_sha256'] != cache['manifest_sha256'] or
            table['reference_selection'] != e.FIXED or
            table['groups']['gaussian_prior_sigma1p5'] != gaussian['value']['groups']):
        raise ValueError('Requires fixed completed full24 uniform training and original Gaussian selection')
    for doc in (complete, selection, table):
        if doc['test_access'] is not False or doc['automatic_promotion'] is not False:
            raise ValueError('TRAIN/VAL evidence touched TEST or promoted a model')
    if (complete['detector_updates'] or complete['detector_forward_calls'] or complete['feature_extractions'] or
            complete['selection_on_test'] is not False or selection['selection_on_test'] is not False or
            complete['reference_replay']['exact_cached_val_replay'] is not True):
        raise ValueError('Training frozen/source boundary differs')
    if digests[FIXED['path']] != FIXED['sha256']:
        raise ValueError('Fixed uniform weight bytes differ')
    for epoch in range(1,25):
        name = 'head_epoch_%02d.pth'%epoch; entry = selection['all_checkpoints'][name]
        ck = entry['checkpoint']; epoch_value = r.t.read(directory/('val_epoch_%02d.json'%epoch))
        if (ck['path'] != name or ck['epoch'] != epoch or ck['updates'] != epoch*902 or
                ck['readout'] != r.model.READOUT or ck['save_reload_exact'] is not True or
                r.t.read(directory/'artifacts.json')['files'].get(name) != ck['sha256'] or
                entry['metrics'] != epoch_value['groups']['overall']['midpoint']['metric_protocol_v2'] or
                table['all_epoch_values'][name] != epoch_value):
            raise ValueError('All24 uniform epoch identity/summary differs')
    chosen, _, info = f.select_best_checkpoint(selection['all_checkpoints'], f.SELECTION_CONFIG)
    if chosen != FIXED['path'] or info != selection['selection_info']:
        raise ValueError('Original full-VAL choice differs')
    rows = e.read_rows(directory/'val_epoch_04.rows.jsonl'); e.checked_rows(rows,'val',online=False)
    if f.summaries(rows) != value or table['groups'][r.model.READOUT] != value['groups']:
        raise ValueError('Selected uniform full VAL rows/summary differ')
    gate = dict(proof=proof, training_files=digests, uniform_checkpoint=deepcopy(FIXED),
        gaussian_checkpoint=deepcopy(e.FIXED), runtime=complete['runtime'])
    return cache, gaussian, gate, dict(rows=rows, value=value)


def checked_val_report(path, identity, gate):
    if not path:
        raise ValueError('TEST requires successful paired --val-report before any TEST access')
    p = Path(path)
    names = {'completion.json','protocol.json','sources.json','uniform_rows.jsonl','gaussian_rows.jsonl',
        'uniform_summary.json','gaussian_summary.json','compare.json'}
    digests = indexed(p.parent, names, VERSION)
    d = r.t.read(p)
    if (p.name != 'completion.json' or d['status'] != VAL_STATUS or d['stage'] != 'val' or
            d['identity'] != identity or d['gate'] != gate or d['frames'] != 887 or
            d['head_updates'] != 0 or d['detector_updates'] != 0 or d['test_access'] is not False or
            d['selection_on_test'] is not False or d['automatic_promotion'] is not False or
            d['feature_extractions'] != 887 or d['native_head_calls'] != 2661 or
            d['state_before'] != d['state_after'] or d['runtime'] != gate['runtime'] or
            d['val_replay'] != dict(uniform=True, gaussian=True) or
            r.t.read(p.parent/'protocol.json') != protocol_document() or
            r.t.read(p.parent/'sources.json') != r.t.read(SOURCES)):
        raise ValueError('Requires unchanged successful paired online VAL gate')
    return dict(files=digests, pass_online_val=True, frames=887)


def historical_test(directory, identity, gate):
    directory = Path(directory)
    names = {'completion.json','frozen_selection.json','protocol.json','test_rows.jsonl'}
    digests = indexed(directory, names, e.VERSION)
    d = r.t.read(directory/'completion.json')
    if (d['status'] != 'FROZEN_SIGMA15_TEST_COMPLETE_REVIEW_REQUIRED' or d['split'] != 'test' or
            d['identity'] != identity['training_identity']['reference_identity'] or
            d['fixed_checkpoint'] != e.FIXED or d['selection']['selected_checkpoint'] != e.FIXED or
            d['runtime'] != gate['runtime'] or d['frames'] != 1440 or
            d['head_updates'] != 0 or d['detector_updates'] != 0 or d['test_access'] is not True or
            d['selection_on_test'] is not False or d['automatic_promotion'] is not False or
            d['state_before'] != d['state_after'] or d['feature_extractions'] != 1440 or d['native_head_calls'] != 4320 or
            r.t.read(directory/'protocol.json') != e.protocol_document() or
            r.t.read(directory/'frozen_selection.json')['selected_checkpoint'] != e.FIXED):
        raise ValueError('Historical fixed TEST identity/runtime differs')
    rows = e.read_rows(directory/'test_rows.jsonl'); e.checked_rows(rows,'test')
    return rows, dict(files=digests, data_identity=d['data_identity'], state=d['state_after'])


def capture(detector, heads, image, metas, audit):
    """No GT/source/domain inputs; one shared frozen detector feature pass."""
    g.assert_detector_frozen(detector)
    if set(heads) != {'uniform','gaussian'} or any(h.training or any(p.requires_grad for p in h.parameters()) for h in heads.values()):
        raise ValueError('Both fixed heads must be frozen')
    with torch.no_grad():
        features = detector.extract_feat(image); audit('feature_extracted')
        if features[0].shape != (1,256,128,128) or any(x.requires_grad or x.grad_fn is not None for x in features):
            raise ValueError('Frozen P3 shape/graph differs')
        def native(rescale):
            x = g.flatten_prediction(detector.simple_test_from_features(features,metas,rescale=rescale))
            audit('native_head_called'); return x
        raw_np = native(False); orig_np = native(True)
        raw, b = image.new_tensor(raw_np), image.new_tensor(orig_np)
        if (len(b)>1 or raw.shape != b.shape or not torch.equal(raw[:,5],b[:,5]) or
                not torch.allclose(g.map_boxes(raw[:,:5],metas[0],inverse=True),b[:,:5],atol=1e-4,rtol=1e-6)):
            raise ValueError('Frozen B count/score/native restoration differs')
        roi, support, _ = g.sample_local(features[0],raw[:,:5],metas[0],'aligned')
        xy = b.new_tensor(np.asarray(metas[0]['scale_factor']).reshape(-1)[:2]).reshape(1,2).expand(len(b),-1)
        result = {}
        for name, head in heads.items():
            output = head(roi,support,b,raw[:,:5],xy); present = bool(len(b))
            if len(output['boxes_original']) != len(b) or not torch.equal(output['boxes_original'][:,5],b[:,5]):
                raise ValueError('Head changed B output count/score')
            result[name] = dict(b=b[0].cpu().tolist() if present else None,
                midpoint=output['boxes_original'][0].cpu().tolist() if present else None,
                candidate=output['candidate_original'][0].cpu().tolist() if present and bool(output['candidate_valid'][0]) else None,
                candidate_valid=bool(output['candidate_valid'][0]) if present else None,
                accepted=bool(output['accepted'][0]) if present else None,
                failed_checks=[k for k,v in output['checks'].items() if present and not bool(v[0])],
                predicted_points_original=output['points_original'][0].cpu().tolist() if present else None,
                original_b_raw_exact_before_after=True)
        if not np.array_equal(raw_np,native(False)):
            raise ValueError('Shared heads changed B raw output')
    return result


def load_pipeline(args, cache, gate):
    cfg = g.check_cfg(); detector = f.build_detector(cfg,args.gpu)
    if g.state_digest(detector) != cache['detector_state']:
        raise ValueError('Current B state differs from original frozen extraction')
    device = next(detector.parameters()).device
    uniform, payload = r.load_head(Path(args.training_dir)/FIXED['path'],gate['proof'],cache,device)
    if payload['epoch'] != FIXED['epoch'] or payload['updates'] != FIXED['updates'] or payload['head_digest'] != FIXED['head_digest']:
        raise ValueError('Fixed uniform checkpoint payload differs')
    gaussian_selection = r.t.read(args.reference_selection)
    gaussian, payload = r.t.load_head(Path(args.reference_selection).parent/e.FIXED['path'],1.5,gaussian_selection['proof'],cache,device)
    if payload['head_digest'] != e.FIXED['head_digest'] or payload['epoch'] != 3 or payload['updates'] != 2706:
        raise ValueError('Fixed Gaussian checkpoint payload differs')
    heads = {'uniform':uniform,'gaussian':gaussian}
    for head in heads.values(): head.eval().requires_grad_(False)
    return detector, heads, cfg


def states(detector, heads):
    return dict(b=g.state_digest(detector), **{k:g.state_digest(v) for k,v in heads.items()})


def publish(out):
    r.t.write(out/'artifacts.json',dict(protocol=VERSION,
        files={p.name:r.t.sha(p) for p in out.iterdir() if p.is_file() and p.name!='artifacts.json'}))


def run(args):
    if int(os.environ.get('WORLD_SIZE','1')) != 1 or args.gpu < 0:
        raise ValueError('One GPU without DDP required')
    inputs = [args.training_dir,args.baseline_dir,args.cache_dir,str(Path(args.reference_selection).parent.parent),args.reference_test_dir]
    if args.val_report: inputs.append(str(Path(args.val_report).parent))
    r.safe_output(args.out_dir,inputs)
    identity = checked_sources(); cache, gaussian, gate, uniform = checked_training(args,identity)
    # TEST gate precedes all historical TEST rows and TEST datasets/image/GT reads.
    val_gate = checked_val_report(args.val_report,identity,gate) if args.stage=='test' else None
    out = Path(args.out_dir); out.mkdir(parents=True,exist_ok=False)
    report = dict(protocol=VERSION,stage=args.stage,status='RUNNING',identity=identity,gate=gate,
        head_updates=0,detector_updates=0,feature_extractions=0,native_head_calls=0,
        test_access=False,selection_on_test=False,automatic_promotion=False,test_repeatedly_exposed=True)
    start = time.perf_counter()
    try:
        r.t.write(out/'protocol.json',protocol_document()); r.t.write(out/'sources.json',r.t.read(SOURCES))
        if args.stage=='check':
            report['status']='STATIC_READOUT_EVAL_PASS_NO_TEST_DATA_TENSOR_GPU_UPDATES'
        else:
            runtime = f.runtime(args.gpu)
            if runtime != gate['runtime']: raise ValueError('Runtime differs from original extraction')
            f.seed_all(); detector, heads, cfg = load_pipeline(args,cache,gate)
            before = states(detector,heads); report.update(runtime=runtime,state_before=before)
            if args.stage=='test':
                report['online_val_gate'] = val_gate; report['test_access'] = True
                historical, history = historical_test(args.reference_test_dir,identity,gate)
                sources, data = f.old.fixed_test_inputs()
                if data != history['data_identity']:raise ValueError('TEST data bytes differ from historical run')
                dataset = f.old.build_test_dataset(cfg,sources); report['historical_test']=history
            else:
                sources, data = e.e.val_inputs(); dataset = f.dataset_for(cfg,'val',1.)
                if any(cache['data_identity']['val'][k]!=v for k,v in data.items()):raise ValueError('VAL data bytes differ')
            if [Path(v['filename']).stem for v in f.infos(dataset)] != [s['image'] for s in sources]:
                raise ValueError('Loader order differs')
            rows = {'uniform':[],'gaussian':[]}
            def audit(event):
                if event=='feature_extracted':report['feature_extractions']+=1
                elif event=='native_head_called':report['native_head_calls']+=1
                else:raise ValueError('Unexpected audit event')
            streams = {k:(out/(k+'_rows.jsonl')).open('x') for k in rows}
            try:
                with (out/'progress.jsonl').open('x') as progress:
                    for i, source in enumerate(sources):
                        image, metas = f.old.load_view(dataset,i,source,args.gpu)
                        outputs = capture(detector,heads,image,metas,audit)
                        for name in rows:
                            row = e.e.prediction_row(source,outputs[name]); rows[name].append(row)
                            streams[name].write(json.dumps(g.json_native(row),ensure_ascii=False,allow_nan=False)+'\n');streams[name].flush()
                        progress.write(json.dumps(dict(stage=args.stage,done=i+1,total=len(sources),image=source['image']))+'\n');progress.flush()
                        if i%100==0 or i==len(sources)-1: print('FIXED paired',args.stage,i+1,'/',len(sources),flush=True)
            finally:
                for stream in streams.values():stream.close()
            after = states(detector,heads)
            if before!=after or report['feature_extractions']!=len(sources) or report['native_head_calls']!=3*len(sources):
                raise ValueError('Frozen state/call budget differs')
            values = {}
            for name in rows:
                e.checked_rows(rows[name],args.stage)
                values[name]=f.summaries(rows[name]);values[name]['split']=args.stage
            if args.stage=='val':
                e.e.verify_val_rows(rows['uniform'],Path(args.training_dir)/'val_epoch_04.rows.jsonl')
                e.e.verify_val_rows(rows['gaussian'],Path(args.reference_selection).parent/'val_epoch_03.rows.jsonl')
                if values['uniform']!=uniform['value'] or values['gaussian']!=gaussian['value']:
                    raise ValueError('Full paired online VAL summary drift; stop before TEST')
                report['val_replay']=dict(uniform=True,gaussian=True);report['status']=VAL_STATUS
            else:
                e.e.verify_val_rows(rows['gaussian'],Path(args.reference_test_dir)/'test_rows.jsonl')
                previous = f.summaries(historical);previous['split']='test'
                if values['gaussian']!=previous:raise ValueError('Historical TEST Gaussian/B full summary drift')
                report['historical_test_replay']=dict(frames=1440,boxes_decisions_and_summary_pass=True)
                report['status']='READOUT_FIXED_TEST_COMPLETE_REVIEW_REQUIRED'
            if [(x['image'],x['b']) for x in rows['uniform']]!=[(x['image'],x['b']) for x in rows['gaussian']]:
                raise ValueError('Heads do not share exact B rows')
            compare = r.comparison_rows(dict(value=values['uniform'],rows=rows['uniform'],selection={'selected_checkpoint':FIXED}),
                dict(value=values['gaussian'],rows=rows['gaussian']))
            for row in compare: row['split']=args.stage
            for name in rows:r.t.write(out/(name+'_summary.json'),values[name])
            r.t.write(out/'compare.json',dict(protocol=VERSION,split=args.stage,rows=compare,
                automatic_promotion=False,selection_on_test=False))
            with (out/'compare.csv').open('x',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(compare[0]));writer.writeheader();writer.writerows(compare)
            for row in compare:
                print(args.stage.upper(),row['condition'],row['domain'],'RIoU',row['mean_riou'],
                    'A-RMSE',row['penalty_a_rmse_deg'],'DFR',row['dfr_pct_per_frame'],'ACI',row['aci'],
                    'center(output)',row['output_frame_r_center_pct'],'coverage',row['output_coverage_pct'],
                    'center(all)',row['full_frame_center_correct_pct'],flush=True)
            report.update(frames=len(sources),data_identity=data,state_after=after,
                cuda_peak_bytes=dict(allocated=torch.cuda.max_memory_allocated(args.gpu),reserved=torch.cuda.max_memory_reserved(args.gpu)))
        report['elapsed_seconds']=time.perf_counter()-start;r.t.write(out/'completion.json',report)
    except Exception as error:
        report.update(status='FAILED_FIXED_READOUT_EVALUATION',error=type(error).__name__+': '+str(error),elapsed_seconds=time.perf_counter()-start)
        r.t.write(out/'failure.json',report);raise
    finally:publish(out)
    print('Saved',out/'completion.json','status',report['status'],flush=True)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('check','val','test'),required=True)
    p.add_argument('--training-dir',default=DEFAULT_TRAIN)
    p.add_argument('--baseline-dir',default='work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1')
    p.add_argument('--reference-selection',default=e.DEFAULT_SELECTION)
    p.add_argument('--cache-dir',default='work_dirs/port_geometry_midpoint_formal_v1_roi_cache')
    p.add_argument('--reference-test-dir',default=DEFAULT_REFERENCE)
    p.add_argument('--val-report');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--out-dir',required=True)
    return p


if __name__=='__main__':run(parser().parse_args())
