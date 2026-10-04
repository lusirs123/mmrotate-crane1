#!/usr/bin/env python3
"""Frozen sigma=1.5/epoch03 evaluation: fresh full VAL gate, then fixed TEST.

Adds a sigma-aware loader; reuses the reviewed native inference and metrics.
Never trains, rebuilds ROI caches, or selects anything on TEST.
"""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import time
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import eval_port_geometry_midpoint_formal_v1 as e
from crane_project.tools import train_port_geometry_midpoint_sigma_v1 as t

f, g, old, torch = t.f, t.g, t.f.old, t.torch
VERSION = 'port_geometry_midpoint_sigma15_v1_frozen_evaluation'
PROTOCOL = ROOT/'crane_project/tools/port_geometry_midpoint_sigma15_v1_eval_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_geometry_midpoint_sigma15_v1_eval_sources.json'
DEFAULT_SELECTION = 'work_dirs/crane_symeood_k1_port_day2night_midpoint_sigma_v1/sigma_1p5/selection.json'
CACHE_SHA = '046c5998dee0ba3703f1ae4e08fc6a02e9804d216ea12241357f69f2a4afd1e3'
FIXED = dict(path='head_epoch_03.pth', epoch=3, updates=2706, sigma_cells=1.5,
    sha256='16c2fb448ac4e1c53530b8086d547f6f6ccb8d6b0763a42391c34f9b337982d7',
    head_digest=dict(parameters='f84d99dac7cb0f2dcb86b5eaf33c21c1d360bdca5d0193b7b3da6c98f92bdc63',
        buffers='44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a',
        parameter_count=17696, buffer_count=0), save_reload_exact=True)


def protocol_document():
    return dict(protocol=VERSION, training_protocol=t.VERSION, fixed_checkpoint=deepcopy(FIXED),
        cache_manifest_sha256=CACHE_SHA, frozen_b=deepcopy(g.ready.FROZEN_B),
        counts=dict(val=g.ready.VAL_COUNTS, test=old.TEST_COUNTS),
        test_annotation_sha256=old.TEST_ANN_SHA, test_image_identity_sha256=old.TEST_IMAGE_SHA,
        selection='Keep original ALL24 full-VAL head selection. Verify all24 summaries and original scoring; only the fixed sigma1.5/epoch03 SHA is admissible. No CLI sigma/epoch/threshold overrides.',
        static='Only source/protocol and indexed training text/selected weight SHA. No torch.load, image/annotation/dataset reads, ROI cache tensors or GPU.',
        inference='Frozen SymEOOD+B ep24 -> aligned P3 ROI -> frozen SigmaMidpointHead(1.5). Reuse original capture/decoder, isotropic transforms and sx/sy/raw w-h-angle restoration, scores/output count and GT-free fallback. One extraction/three native head calls per frame.',
        val_gate='Fresh standard-scale full887-frame online VAL must reproduce indexed epoch03 rows, decisions and full summaries under original tolerances. TEST requires this successful indexed report and matching source/checkpoint/runtime. Gate checked before any TEST input access.',
        test='One fixed exposed1440-frame TEST, same-run B/midpoint comparison. No weight averaging, retraining, cache rebuilding or TEST parameter/checkpoint selection.',
        reporting='Two overall B/midpoint console tables. JSON retains domains/sequences, pure/penalty angle, geometry, intervals and separate output-frame center correctness/output coverage/full-frame center correctness. Historical metric R_center remains all-frame.',
        scope=dict(head_updates=0, detector_updates=0, selection_on_test=False,
            automatic_promotion=False, test_repeatedly_exposed=True, depth_accuracy_verified=False))


def checked_sources():
    training, evaluation = t.checked_sources(), e.checked_sources()
    required = set(training['sources']) | set(evaluation['sources']) | {
        str(t.SOURCES.relative_to(ROOT)), str(e.SOURCES.relative_to(ROOT)),
        str(PROTOCOL.relative_to(ROOT)), str(Path(__file__).resolve().relative_to(ROOT)),
        'tests/test_port_geometry_midpoint_sigma15_v1_eval.py'}
    fixed = t.read(SOURCES)
    if (fixed['protocol'] != VERSION or set(fixed['sources']) != required or
            fixed['training_sources_sha256'] != t.sha(t.SOURCES) or
            fixed['evaluation_sources_sha256'] != t.sha(e.SOURCES) or
            t.read(PROTOCOL) != protocol_document()):
        raise ValueError('Sigma15 evaluation protocol/source scope differs')
    actual = {name:t.sha(ROOT/name) for name in required}
    if actual != fixed['sources']:
        raise ValueError('Sigma15 evaluation source SHA differs')
    return dict(training_identity=training, evaluation_identity=evaluation, sources=actual,
        protocol_sha256=t.sha(PROTOCOL), sources_sha256=t.sha(SOURCES))


def indexed(directory, names):
    directory = Path(directory)
    index = t.read(directory/'artifacts.json')
    if index['protocol'] != t.VERSION:
        raise ValueError('Sigma training artifact protocol differs')
    digests = {}
    for name in sorted(names):
        digest = t.sha(directory/name)
        if index['files'].get(name) != digest:
            raise ValueError('Sigma training artifact SHA differs: '+name)
        digests[name] = digest
    return digests, index['files']


def checked_selection(path, identity):
    """Bounded text audit; no tensor/cache/dataset load or repeated training audit."""
    path = Path(path).resolve(); arm, root = path.parent, path.parent.parent
    selection = t.read(path); complete = t.read(arm/'completion.json')
    grid = t.read(root/'completion.json'); proof = selection['proof']
    names = {'head_epoch_%02d.pth'%epoch for epoch in range(1,25)}
    if (selection['protocol'] != t.VERSION or selection['sigma_cells'] != 1.5 or
            selection['split'] != 'val' or selection['test_access'] is not False or
            selection['selection_on_test'] is not False or selection['automatic_promotion'] is not False or
            proof['identity'] != identity['training_identity'] or
            proof['cache_manifest_sha256'] != CACHE_SHA or selection['cache_manifest_sha256'] != CACHE_SHA or
            selection['selection_config'] != f.SELECTION_CONFIG or set(selection['all_checkpoints']) != names or
            selection['selected_checkpoint'] != FIXED or
            complete['protocol'] != t.VERSION or complete['sigma_cells'] != 1.5 or
            complete['status'] != 'SIGMA_FULL_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED' or
            complete['epochs_completed'] != 24 or complete['updates'] != 21648 or
            complete['proof'] != proof or complete['selection'] != selection or
            complete['detector_updates'] != 0 or complete['test_access'] is not False or
            grid['protocol'] != t.VERSION or grid['stage'] != 'train' or
            grid['status'] != 'SIGMA_FULL_GRID_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED' or
            grid['proof'] != proof or grid['head_updates_total'] != 43296 or
            grid['reused_updates'] != 21648 or grid['represented_updates'] != 64944 or
            grid['detector_updates'] != 0 or grid['detector_forward_calls'] != 0 or
            grid['feature_extractions'] != 0 or grid['test_access'] is not False or
            grid['selection_on_test'] is not False or grid['selected_sigma'] is not None or
            complete['schedule'] != grid['schedule'] or grid['schedule']['epoch_steps'] != [902]*24 or
            grid['default_replay']['exact_cached_val_replay'] is not True):
        raise ValueError('Requires unchanged completed sigma grid and fixed VAL-selected epoch03')
    for key in ('sigma_0p5','sigma_1p5'):
        if grid['conditions'][key]['updates'] != 21648 or grid['conditions'][key]['epochs'] != 24:
            raise ValueError('Sigma full training budgets differ')
    if grid['conditions']['sigma_1p5']['selected_checkpoint'] != FIXED:
        raise ValueError('Grid fixed checkpoint differs')
    root_digests, _ = indexed(root, {'completion.json','protocol.json','sources.json','val_sigma_compare.json'})
    if t.read(root/'protocol.json') != t.protocol_document() or t.read(root/'sources.json') != t.read(t.SOURCES):
        raise ValueError('Recorded sigma training sources/protocol differ')
    row_name = 'val_epoch_03.rows.jsonl'
    required = {path.name,'completion.json','selected_val_compare.json',row_name,FIXED['path']}
    required.update('val_epoch_%02d.json'%epoch for epoch in range(1,25))
    arm_digests, index = indexed(arm, required)
    if arm_digests[FIXED['path']] != FIXED['sha256']:
        raise ValueError('Actual fixed epoch03 weight SHA differs')
    table = t.read(root/'val_sigma_compare.json')
    if (table['protocol'] != t.VERSION or table['proof'] != proof or table['split'] != 'val' or
            table['test_access'] is not False or table['selected_epochs']['sigma_1p5'] != 3):
        raise ValueError('Recorded selected sigma VAL table differs')
    selected_value = t.read(arm/'selected_val_compare.json')
    for epoch in range(1,25):
        name = 'head_epoch_%02d.pth'%epoch; entry = selection['all_checkpoints'][name]
        ckpt = entry['checkpoint']; value = t.read(arm/('val_epoch_%02d.json'%epoch))
        if (ckpt['path'] != name or ckpt['epoch'] != epoch or ckpt['updates'] != epoch*902 or
                ckpt['sigma_cells'] != 1.5 or ckpt['save_reload_exact'] is not True or
                index.get(name) != ckpt['sha256'] or value['split'] != 'val' or
                entry['metrics'] != value['groups']['overall']['midpoint']['metric_protocol_v2'] or
                table['all_epoch_values']['sigma_1p5'][name] != value):
            raise ValueError('Full24 epoch checkpoint/VAL table differs')
        if epoch == 3 and value != selected_value:
            raise ValueError('Selected epoch03 VAL summary differs')
    chosen, _, info = f.select_best_checkpoint(selection['all_checkpoints'], f.SELECTION_CONFIG)
    if (chosen != FIXED['path'] or info != selection['selection_info'] or
            selection['all_checkpoints'][chosen]['checkpoint'] != FIXED):
        raise ValueError('Original ALL24 VAL choice differs')
    rows = read_rows(arm/row_name); checked_rows(rows, 'val', online=False)
    if f.summaries(rows) != selected_value or table['groups']['sigma_1p5'] != selected_value['groups']:
        raise ValueError('Indexed epoch03 full VAL rows/summary differ')
    return selection, dict(training_proof=proof, selected_checkpoint=deepcopy(FIXED),
        root_artifact_sha256=root_digests, arm_artifact_sha256=arm_digests,
        cache_manifest_sha256=CACHE_SHA, training_runtime=grid['runtime'])


def read_rows(path):
    return [json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        for line in Path(path).read_text().splitlines() if line.strip()]


def checked_rows(rows, split, online=True):
    counts = g.ready.VAL_COUNTS if split == 'val' else old.TEST_COUNTS
    if (len(rows) != sum(counts.values()) or len({r['image'] for r in rows}) != len(rows) or
            Counter(r['sequence'] for r in rows) != Counter(counts) or
            any(r['scale'] != 1. or r['domain'] != r['sequence'].split('_')[0] or
                r['image'] != '%s_%05d'%(r['sequence'],r['frame_id']) or
                (online and r.get('original_b_raw_exact_before_after') is not True) for r in rows)):
        raise ValueError('Full standard-scale '+split.upper()+' identities/rows differ')
    for row in rows:
        b, result = row['b'], row['midpoint']
        if ((b is None) != (result is None) or
                (b is not None and b[5] != result[5]) or
                (row['accepted'] is False and b != result)):
            raise ValueError('Frozen output count/score/whole-box fallback differs')


def load_pipeline(selection, proof, gpu):
    """Load exactly the explicit sigma-aware head; never use the default loader."""
    checkpoint = Path(selection).parent/FIXED['path']
    if t.sha(checkpoint) != FIXED['sha256']:
        raise ValueError('Fixed weight changed before load')
    payload = torch.load(str(checkpoint), map_location='cpu')
    if (payload['protocol'] != t.VERSION or payload['proof'] != proof['training_proof'] or
            payload['sigma_cells'] != 1.5 or payload['epoch'] != 3 or payload['updates'] != 2706 or
            payload['head_digest'] != FIXED['head_digest'] or payload['frozen_b'] != g.ready.FROZEN_B or
            payload['context']['manifest_sha256'] != CACHE_SHA or
            payload['context']['runtime'] != proof['training_runtime']):
        raise ValueError('Fixed sigma15 checkpoint payload/context differs')
    cfg = g.check_cfg(); detector = f.build_detector(cfg, gpu)
    if g.state_digest(detector) != payload['context']['detector_state']:
        raise ValueError('Frozen B state differs from training extraction')
    head = t.model.SigmaMidpointHead(1.5).to(next(detector.parameters()).device)
    head.load_state_dict(payload['head_state'], strict=True)
    head.eval().requires_grad_(False)
    if g.state_digest(head) != FIXED['head_digest']:
        raise ValueError('Loaded sigma15 head state differs')
    return detector, head, cfg, payload['context']


def checked_val_report(path, identity, proof, selection):
    if not path:
        raise ValueError('TEST requires --val-report from successful fresh sigma15 VAL')
    path = Path(path); directory = path.parent; report = t.read(path)
    required = {path.name,'protocol.json','sources.json','val_compare.json','val_rows.jsonl','progress.jsonl'}
    index = t.read(directory/'artifacts.json')
    if index['protocol'] != VERSION or any(index['files'].get(name) != t.sha(directory/name) for name in required):
        raise ValueError('Fresh VAL report artifact SHA differs')
    if (report['protocol'] != VERSION or report['split'] != 'val' or
            report['status'] != 'FROZEN_SIGMA15_VAL_COMPLETE_REVIEW_REQUIRED' or
            report['identity'] != identity or report['selection'] != proof or
            report['head_updates'] != 0 or report['detector_updates'] != 0 or
            report['test_access'] is not False or report['selection_on_test'] is not False or
            report['runtime'] != proof['training_runtime'] or report['frames'] != 887 or
            report['feature_extractions'] != 887 or report['native_head_calls'] != 2661 or
            report['state_before'] != report['state_after'] or
            report['state_after']['head'] != FIXED['head_digest'] or
            report['val_replay'] != dict(frames=887,selected_epoch_box_decision_replay_pass=True) or
            t.read(directory/'protocol.json') != protocol_document() or t.read(directory/'sources.json') != t.read(SOURCES)):
        raise ValueError('Requires successful matching full online VAL before TEST')
    rows = read_rows(directory/'val_rows.jsonl'); checked_rows(rows,'val')
    selected_dir = Path(selection).parent
    e.verify_val_rows(rows, selected_dir/'val_epoch_03.rows.jsonl')
    value = f.summaries(rows)
    if value != t.read(directory/'val_compare.json') or value != t.read(selected_dir/'selected_val_compare.json'):
        raise ValueError('Fresh VAL gate full summary differs')
    return dict(completion_sha256=t.sha(path), val_rows_sha256=t.sha(directory/'val_rows.jsonl'),
        val_compare_sha256=t.sha(directory/'val_compare.json'), frames=887, pass_online_val=True)


def publish(out, report):
    g.write_new(out/'completion.json',report)
    g.write_new(out/'artifacts.json',dict(protocol=VERSION,
        files={p.name:t.sha(p) for p in out.iterdir() if p.is_file() and p.name != 'artifacts.json'}))


def run(args):
    out = Path(args.out_dir).resolve(); training = Path(args.selection).resolve().parent.parent
    if training == out or training in out.parents or (args.val_report and
            (Path(args.val_report).resolve().parent == out or Path(args.val_report).resolve().parent in out.parents)):
        raise ValueError('Use a new output directory outside immutable training/VAL inputs')
    if out.exists():
        raise FileExistsError('Output exists; preserve it and choose a new --out-dir: '+str(out))
    out.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    report = dict(protocol=VERSION, split=args.split, status='RUNNING', head_updates=0, detector_updates=0,
        feature_extractions=0, native_head_calls=0, test_access=False, selection_on_test=False,
        test_repeatedly_exposed=True, automatic_promotion=False, fixed_checkpoint=deepcopy(FIXED))
    try:
        identity = checked_sources(); selection, proof = checked_selection(args.selection,identity)
        report.update(identity=identity,selection=proof)
        g.write_new(out/'protocol.json',protocol_document()); g.write_new(out/'sources.json',t.read(SOURCES))
        g.write_new(out/'frozen_selection.json',selection)
        # Gate precedes runtime, detector loading and every TEST file/dataset access.
        if args.split == 'test':
            report['online_val_gate'] = checked_val_report(args.val_report,identity,proof,args.selection)
        if args.check_only:
            report['status'] = 'STATIC_SIGMA15_EVAL_PASS_NO_DATA_GPU_UPDATES'
        else:
            runtime = f.runtime(args.gpu)
            if runtime != proof['training_runtime']:
                raise ValueError('Evaluation libraries/GPU differ from formal training')
            f.seed_all(); detector, head, cfg, context = load_pipeline(args.selection,proof,args.gpu)
            before = dict(b=g.state_digest(detector),head=g.state_digest(head))
            if args.split == 'test':
                report['test_access'] = True
                sources, data = old.fixed_test_inputs(); dataset = old.build_test_dataset(cfg,sources)
            else:
                sources, data = e.val_inputs(); dataset = f.dataset_for(cfg,'val',1.)
                if any(context['data_identity']['val'][k] != data[k] for k in data):
                    raise ValueError('Fresh VAL data identity differs from cached VAL')
                if [Path(v['filename']).stem for v in f.infos(dataset)] != [s['image'] for s in sources]:
                    raise ValueError('Fresh VAL loader order differs')
            rows = []
            def audit(event):
                if event['stage'] == 'test_feature_extracted': report['feature_extractions'] += 1
                elif event['stage'] == 'test_native_head_called': report['native_head_calls'] += 1
                else: raise ValueError('Unexpected frozen inference audit event')
            with (out/(args.split+'_rows.jsonl')).open('x') as stream, (out/'progress.jsonl').open('x') as progress:
                for index, source in enumerate(sources):
                    image, metas = old.load_view(dataset,index,source,args.gpu)
                    row = e.prediction_row(source,old.capture(detector,head,image,metas,audit=audit))
                    rows.append(row)
                    stream.write(json.dumps(g.json_native(row),ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()
                    progress.write(json.dumps(dict(stage='frozen_sigma15',image=source['image'],done=index+1,total=len(sources)))+'\n'); progress.flush()
                    if index%100 == 0 or index == len(sources)-1:
                        print('SIGMA1.5 epoch03 frozen',args.split,index+1,'/',len(sources),flush=True)
            checked_rows(rows,args.split)
            after = dict(b=g.state_digest(detector),head=g.state_digest(head))
            if before != after or report['feature_extractions'] != len(rows) or report['native_head_calls'] != 3*len(rows):
                raise ValueError('Frozen state/native call budget differs')
            value = f.summaries(rows); value['split'] = args.split
            if args.split == 'val':
                report['val_replay'] = e.verify_val_rows(rows,Path(args.selection).parent/'val_epoch_03.rows.jsonl')
                if value != t.read(Path(args.selection).parent/'selected_val_compare.json'):
                    raise ValueError('Fresh full VAL summary differs; stop before TEST')
            g.write_new(out/(args.split+'_compare.json'),value)
            report.update(status='FROZEN_SIGMA15_'+args.split.upper()+'_COMPLETE_REVIEW_REQUIRED',
                frames=len(rows),runtime=runtime,data_identity=data,state_before=before,state_after=after,
                gpu_peak_mib=dict(allocated=torch.cuda.max_memory_allocated(args.gpu)/2**20,
                    reserved=torch.cuda.max_memory_reserved(args.gpu)/2**20))
            for method in ('b','midpoint'):
                print('OVERALL',args.split.upper(),'method='+method,'sigma='+('NA' if method=='b' else '1.5 epoch03'),flush=True)
                summary = value['groups']['overall'][method]
                old.CraneOfflineEvaluator(mode=args.split)._log_metrics(summary['metric_protocol_v2'],['real','sim'])
                for domain in ('real','sim'):
                    group = value['groups'][domain][method]
                    print(domain,'output-frame center=',group['conditional_center_correct_fraction'],
                        'output coverage=',group['output_coverage_pct'],
                        'full-frame center=',group['all_frame_center_hit_pct'],flush=True)
        report['elapsed_seconds'] = time.perf_counter()-started
        publish(out,report)
        print('Saved',out/'completion.json','status',report['status'],flush=True)
    except Exception as error:
        report.update(status='FAILED',error=type(error).__name__+': '+str(error),
            elapsed_seconds=time.perf_counter()-started)
        g.write_new(out/'failure.json',report)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=('val','test'),required=True)
    p.add_argument('--selection',default=DEFAULT_SELECTION)
    p.add_argument('--val-report',help='Successful fresh sigma15 VAL completion.json; required for TEST, including check-only')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--out-dir',required=True)
    return p


if __name__ == '__main__':
    if Path.cwd().resolve() != ROOT:
        raise SystemExit('Run from symEOOD project root')
    run(parser().parse_args())
