#!/usr/bin/env python3
"""check -> collect -> fit -> verify: simple three flags on sigma1.5/epoch03.

No detector/head training, reference branch, TEST route or threshold search.
All stages live below one new experiment directory and refuse overwrites.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_simple_reliability_v1 as base
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_midpoint_reliability_v1 as binding
from crane_project.utils import port_midpoint_sigma15_reliability_v1 as new

PROTOCOL = ROOT/'crane_project/tools/port_midpoint_sigma15_reliability_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_midpoint_sigma15_reliability_v1_sources.json'
OLD_POLICY_SHA = '1adac499369bade5ef1f79495260002697d2b698c80ebf5424d3d77438fae4af'
CACHE_SHA = '046c5998dee0ba3703f1ae4e08fc6a02e9804d216ea12241357f69f2a4afd1e3'
STATUSES = dict(check='SIGMA15_RELIABILITY_INPUTS_PASS',
    collect='SIGMA15_RELIABILITY_COLLECTION_COMPLETE', fit='SIGMA15_SIMPLE_FIT_COMPLETE_REVIEW_REQUIRED',
    verify='SIGMA15_THREE_FLAGS_FULL_VAL_VERIFY_PASS_REVIEW_REQUIRED')


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda v:
        (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+v)))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def checked_sources():
    protocol, manifest = read(PROTOCOL), read(SOURCES)
    parents = ('port_simple_reliability_v1_sources.json',
        'port_geometry_midpoint_sigma15_v1_eval_sources.json', 'port_midpoint_reliability_v1_sources.json')
    required = {str(PROTOCOL.relative_to(ROOT)), str(Path(__file__).resolve().relative_to(ROOT)),
        'crane_project/utils/port_midpoint_sigma15_reliability_v1.py',
        'tests/test_port_midpoint_sigma15_reliability_v1.py'}
    for name in parents:
        path = ROOT/'crane_project/tools'/name
        required.update(read(path)['sources'])
        required.add(str(path.relative_to(ROOT)))
    actual = {name: base.sha(ROOT/name) for name in manifest['sources']}
    if (set(actual) != required or manifest['protocol'] != new.VERSION or actual != manifest['sources'] or
            manifest['protocol_sha256'] != base.sha(PROTOCOL) or protocol['protocol'] != new.VERSION or
            protocol['midpoint_sha256'] != new.HEAD_SHA or protocol['test_read'] is not False):
        raise ValueError('Sigma15 reliability sources/protocol differ')
    base.checked_sources()
    return protocol, dict(sources=actual, manifest_sha256=base.sha(SOURCES))


def prepare(args):
    protocol, sources = checked_sources()
    # Lazy import: --help and numeric unit tests need no detection libraries/GPU.
    from crane_project.tools import eval_port_geometry_midpoint_sigma15_v1 as sigma
    identity = sigma.checked_sources()
    selection, proof = sigma.checked_selection(args.selection, identity)
    if selection['selected_checkpoint']['sha256'] != new.HEAD_SHA:
        raise ValueError('Wrong selected checkpoint')
    old_protocol, _ = base.checked_sources()
    train, val, numeric = base.checked_inputs(args, old_protocol)
    cache = sigma.t.checked_cache_metadata(args.formal_cache, proof['training_proof']['identity']['parent_identity'])
    if cache['manifest_sha256'] != CACHE_SHA or cache['runtime'] != proof['training_runtime']:
        raise ValueError('Requires original frozen TRAIN/VAL ROI cache/runtime')
    if base.sha(args.b_checkpoint) != new.B_SHA:
        raise ValueError('Wrong B24 checkpoint')
    if base.sha(args.old_policy) != OLD_POLICY_SHA:
        raise ValueError('Preserve exact old sigma1/epoch23 comparison policy')
    old = read(args.old_policy)
    if (old['protocol'] != binding.VERSION or
            old['front_end']['midpoint_checkpoint']['epoch'] != 23 or
            old['front_end']['frozen_b']['checkpoint_sha256'] != new.B_SHA):
        raise ValueError('Old policy comparison identity differs')
    simple.validate_policy(old['simple_policy'])
    selected_path = args.selection.parent/'val_epoch_03.rows.jsonl'
    selected = rows(selected_path)
    final = binding.paired_rows(val, selected)
    validate_b_pair(val, final)
    front = dict(name='B24_midpoint_sigma1p5_epoch03', frozen_b=old_protocol['frozen_b'],
        midpoint_checkpoint=selection['selected_checkpoint'], selection_sha256=base.sha(args.selection))
    new.checked_frontend(front)
    contract = dict(protocol=protocol, sources=sources, front_end=front, sigma_proof=proof,
        numeric_input_proof=numeric, cache_manifest_sha256=CACHE_SHA,
        final_VAL_fingerprint=simple.fingerprint(final), old_policy_sha256=OLD_POLICY_SHA,
        test_read=False, test_repeatedly_exposed=True)
    return dict(protocol=protocol, contract=contract, train=train, val=val, final=final,
        cache=cache, selection=selection, old_simple=old['simple_policy'], old_protocol=old_protocol)


def validate_b_pair(sources, paired):
    source = {r['image']: r for r in sources}
    if len(source) != len(sources) or len(paired) != len(sources) or {r['image'] for r in paired} != set(source):
        raise ValueError('Original B paired identity/count differs')
    for row in paired:
        b, original = row['b_original'], source[row['image']]['pred']
        if (b is None) != (original is None) or (b is not None and
                (b[5] != original[5] or not np.allclose(b[:5], original[:5], atol=1e-4, rtol=1e-6))):
            raise ValueError('Midpoint cache does not share reviewed B: '+row['image'])


def verify_replay(current, expected):
    indexed = {r['image']: r for r in expected}
    if len(current) != len(expected) or len(indexed) != len(expected) or {r['image'] for r in current} != set(indexed):
        raise ValueError('Replay identity/count differs')
    for row in current:
        old = indexed[row['image']]
        for key in ('image_size','gt','sequence','domain','frame_id','split',
                    'train_angle_eligible','angle_axis_well_defined','midpoint_accepted'):
            if row[key] != old[key]:
                raise ValueError('Replay metadata/decision differs: '+row['image'])
        for key in ('pred','b_original','midpoint_candidate'):
            a, b = row[key], old[key]
            if (a is None) != (b is None) or (a is not None and
                    (a[5] != b[5] or not np.allclose(a[:5], b[:5], atol=1e-4, rtol=1e-6))):
                raise ValueError('Replay box/score/count differs: '+row['image'])


def finish(out, stage, contract):
    base.write_new(out/'completion.json', dict(protocol=new.VERSION, status=STATUSES[stage],
        contract=contract, artifacts={p.name:base.sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        detector_updates=0, midpoint_updates=0, reference_updates=0, test_read=False))
    print('Saved', out, STATUSES[stage], flush=True)


def completed(directory, stage, contract):
    directory = Path(directory)
    if (directory/'failure.json').exists():
        raise ValueError('Failed stage; preserve results and use a new experiment directory')
    report = read(directory/'completion.json')
    if (report['protocol'] != new.VERSION or report['status'] != STATUSES[stage] or
            report['contract'] != contract or report['test_read'] is not False or
            any(report[k] != 0 for k in ('detector_updates','midpoint_updates','reference_updates'))):
        raise ValueError('Preceding stage/frontend contract differs')
    if any(Path(name).name != name or base.sha(directory/name) != digest
           for name, digest in report['artifacts'].items()):
        raise ValueError('Preceding stage artifact SHA differs')
    return report


def collect(args, prepared, out):
    from crane_project.tools import eval_port_geometry_midpoint_sigma15_v1 as sigma
    torch, formal = sigma.torch, sigma.f
    cache, proof = prepared['cache'], prepared['contract']['sigma_proof']
    runtime = formal.runtime(args.gpu)
    if runtime != cache['runtime']:
        raise ValueError('Native CUDA/runtime differs from original cache')
    formal.seed_all()
    checkpoint = args.selection.parent/'head_epoch_03.pth'
    head, payload = sigma.t.load_head(checkpoint, 1.5, proof['training_proof'], cache, 'cuda:'+str(args.gpu))
    if (payload['epoch'] != 3 or payload['updates'] != 2706 or payload['head_digest'] != sigma.FIXED['head_digest']):
        raise ValueError('Loaded sigma15 checkpoint differs')
    head.eval().requires_grad_(False)
    before = formal.g.state_digest(head)
    with (out/'final_rows.jsonl').open('x') as stream:
        for key, role in (('train_s1','train'),('val_s1','val')):
            path = args.formal_cache/(key+'.pt')
            if base.sha(path) != cache['files'][path.name]:
                raise ValueError('ROI shard SHA differs')
            shard = torch.load(str(path), map_location='cpu')
            records = shard['records']
            if (shard['identity'] != cache['identity'] or shard['detector_state'] != cache['detector_state'] or
                    len(records) != cache['record_counts'][key] or
                    any(r['role'] != role or r['scale'] != 1. for r in records)):
                raise ValueError('ROI shard role/identity/count differs')
            predictions = formal.evaluate(head, records, 'cuda:'+str(args.gpu))
            paired = binding.paired_rows(prepared[role], predictions)
            validate_b_pair(prepared[role], paired)
            if role == 'val':
                verify_replay(paired, prepared['final'])
                paired = prepared['final']  # Keep sealed selected-epoch VAL values.
            for row in paired:
                stream.write(json.dumps(dict(row, reliability_role=role), allow_nan=False)+'\n')
            print('Frozen sigma15', role, len(paired), flush=True)
            del records, shard, predictions, paired
    if formal.g.state_digest(head) != before:
        raise ValueError('Frozen collection changed head')
    base.write_new(out/'collect_report.json', dict(train_frames=2558, val_frames=887,
        head_state_before=before, head_state_after=before, detector_loaded=False,
        feature_extractions=0, head_updates=0, detector_updates=0, test_read=False))


def read_collection(args, prepared):
    completed(args.run_dir/'collect', 'collect', prepared['contract'])
    all_rows = rows(args.run_dir/'collect/final_rows.jsonl')
    if len(all_rows) != 3445 or any(r['reliability_role'] not in ('train','val') for r in all_rows):
        raise ValueError('Collected roles/count differ')
    result = {}
    for role in ('train','val'):
        values = [dict(r) for r in all_rows if r['reliability_role'] == role]
        for row in values:
            row.pop('reliability_role')
        sources = prepared[role]
        base._checked_rows(values, prepared['old_protocol']['counts'][role], role)
        indexed = {r['image']:r for r in sources}
        metadata = ('gt','image_size','sequence','domain','frame_id','split','train_angle_eligible','angle_axis_well_defined')
        if any(any(r[k] != indexed[r['image']][k] for k in metadata) for r in values):
            raise ValueError('Collected label/metadata changed')
        predictions = [dict(image=r['image'],sequence=r['sequence'],domain=r['domain'],frame_id=r['frame_id'],
            gt=r['gt'],b=r['b_original'],midpoint=r['pred'],candidate=r['midpoint_candidate'],
            accepted=r['midpoint_accepted']) for r in values]
        if binding.paired_rows(sources, predictions) != values:
            raise ValueError('Collected final-box binding differs')
        validate_b_pair(sources, values)
        result[role] = values
    if simple.fingerprint(result['val']) != prepared['contract']['final_VAL_fingerprint']:
        raise ValueError('Collected VAL differs from fixed selected epoch03')
    return result['train'], result['val']


def fit(args, prepared, out):
    train, val = read_collection(args, prepared)
    before = simple.fingerprint([train,val])
    internal = simple.create_policy(train, val, prepared['old_protocol'])
    policy = dict(protocol=new.VERSION, front_end=prepared['contract']['front_end'],
        contract=prepared['contract'], simple_policy=internal, test_read=False,
        center_policy='retain_every_valid_FINAL_output', reference_branch_used=False)
    base.write_new(out/'policy.json', policy)
    reload = read(out/'policy.json')
    first = new.Sigma15Reliability(policy, policy['front_end'])
    second = new.Sigma15Reliability(reload, policy['front_end'])
    for row in val:
        for method in ('raw','score_only','simple'):
            a = first.decide(row['pred'], row['image_size'], method)
            if a != second.decide(row['pred'], row['image_size'], method) or a['final_box_original'] != row['pred']:
                raise ValueError('Policy replay changed final box or flags')
    decisions, statistics = new.evaluate(val, policy, prepared['old_simple'])
    _, train_stats = new.evaluate(train, policy, prepared['old_simple'])
    with (out/'val_decisions.jsonl').open('x') as stream:
        for record in decisions:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
    if reload != policy or simple.fingerprint([train,val]) != before:
        raise ValueError('Fitting/save/reload changed source observations')
    base.write_new(out/'fit_report.json', dict(protocol=new.VERSION, front_end=policy['front_end'],
        VAL_calibration_descriptive=statistics, TRAIN_in_sample_descriptive=train_stats,
        models=internal['models'], cutoffs=internal['cutoffs'], save_reload_exact=True,
        boxes_scores_count_unchanged=True, test_read=False, test_repeatedly_exposed=True,
        limitations=['95% pooled full-frame coverage is a fixed comparison workpoint, not an accuracy optimum.',
            'VAL is used for label-free coverage calibration; reported VAL performance is descriptive.',
            'Center retains outputs; it is not a trained center-correctness classifier.',
            'Balanced sigmoid risks are not calibrated error probabilities.',
            'Old sigma1 policy transfer and same-count rankings are offline controls on the NEW boxes.',
            'No reference training, depth guarantee or TEST evaluation.']))
    for domain in ('real','sim'):
        summary = statistics['domain:'+domain]
        print(domain, 'outputs', summary['output_frames'], '/', summary['frames'],
            'center/output', summary['center_hit_rate_on_outputs'],
            'full-frame center', summary['all_frame_center_correct_coverage'], flush=True)
        for component in ('size','angle'):
            item = summary['components'][component]['simple']
            print(component, 'accepted', item['accepted_frames'], 'FA', item['false_accept'],
                'FR', item['false_reject'], 'ED', item['error_detected'], 'CR', item['correct_retained'], flush=True)


def verify(args, prepared, out):
    from crane_project.tools import eval_port_geometry_midpoint_sigma15_v1 as sigma
    completed(args.run_dir/'fit', 'fit', prepared['contract'])
    policy = read(args.run_dir/'fit/policy.json')
    if policy['contract'] != prepared['contract']:
        raise ValueError('Policy contract differs')
    runtime = new.Sigma15Reliability(policy, prepared['contract']['front_end'])
    native = sigma.f.runtime(args.gpu)
    proof = prepared['contract']['sigma_proof']
    if native != proof['training_runtime']:
        raise ValueError('Native runtime differs')
    sigma.f.seed_all()
    detector, head, cfg, context = sigma.load_pipeline(args.selection, proof, args.gpu)
    before = dict(b=sigma.g.state_digest(detector), head=sigma.g.state_digest(head))
    sources, data = sigma.e.val_inputs()
    if any(context['data_identity']['val'][k] != v for k,v in data.items()):
        raise ValueError('Native VAL identity differs')
    dataset = sigma.f.dataset_for(cfg, 'val', 1.)
    if [Path(v['filename']).stem for v in sigma.f.infos(dataset)] != [r['image'] for r in sources]:
        raise ValueError('Native VAL order differs')
    expected = {r['image']:r for r in prepared['final']}
    predictions, decisions = [], []
    counts = dict(feature_extractions=0, native_head_calls=0)
    def audit(event):
        if event['stage'] == 'test_feature_extracted': counts['feature_extractions'] += 1
        elif event['stage'] == 'test_native_head_called': counts['native_head_calls'] += 1
        else: raise ValueError('Unexpected native inference audit event')
    with (out/'val_decisions.jsonl').open('x') as stream:
        for index, source in enumerate(sources):
            image, metas = sigma.old.load_view(dataset, index, source, args.gpu)
            # The legacy function name/audit event uses "test" for all inference;
            # this dataset is exclusively VAL, and annotations never enter capture.
            prediction = sigma.e.prediction_row(source, sigma.old.capture(detector, head, image, metas, audit=audit))
            predictions.append(prediction)
            row = expected[source['image']]
            methods = {m:runtime.decide(prediction['midpoint'], source['image_size'], m)
                       for m in ('raw','score_only','simple')}
            for method, decision in methods.items():
                saved = runtime.decide(row['pred'], row['image_size'], method)
                if (decision['final_box_original'] != prediction['midpoint'] or
                        any(decision[c+'_accepted'] != saved[c+'_accepted'] for c in simple.COMPONENTS)):
                    raise ValueError('Native three flags differ from cached policy replay: '+source['image'])
            record = dict(image=source['image'], methods=methods)
            stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
            decisions.append(record)
            if index%100 == 0 or index == len(sources)-1:
                print('Sigma15 three flags VAL', index+1, '/', len(sources), flush=True)
    sigma.e.verify_val_rows(predictions, args.selection.parent/'val_epoch_03.rows.jsonl')
    paired = binding.paired_rows(prepared['val'], predictions)
    validate_b_pair(prepared['val'], paired); verify_replay(paired, prepared['final'])
    _, statistics = new.evaluate(paired, policy, prepared['old_simple'])
    _, expected_stats = new.evaluate(prepared['final'], policy, prepared['old_simple'])
    # Numeric residual summaries can differ within native replay tolerance;
    # correctness and all four confusion counts must remain exactly the same.
    for group, summary in statistics.items():
        previous = expected_stats[group]
        for key in ('frames','output_frames','center_hits'):
            if summary[key] != previous[key]: raise ValueError('Native center/output counts differ')
        for component in simple.COMPONENTS:
            for method in ('raw','score_only','simple','matched_score_diagnostic','matched_old_simple_diagnostic'):
                for key in ('accepted_frames','false_accept','false_reject','error_detected','correct_retained'):
                    if summary['components'][component][method][key] != previous['components'][component][method][key]:
                        raise ValueError('Native judgement counts differ: '+group+'/'+component+'/'+method+'/'+key)
    after = dict(b=sigma.g.state_digest(detector), head=sigma.g.state_digest(head))
    if before != after or counts != dict(feature_extractions=887, native_head_calls=2661):
        raise ValueError('Native state/call budget differs')
    base.write_new(out/'verify_report.json', dict(protocol=new.VERSION, frames=len(decisions),
        VAL_descriptive=statistics, runtime=native, state_before=before, state_after=after, **counts,
        GT_online=False, reference_branch_used=False, boxes_scores_count_unchanged=True,
        three_flag_cached_native_replay_exact=True, test_read=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=tuple(STATUSES), required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--run-dir', type=Path, default=Path('work_dirs/port_midpoint_sigma15_reliability_v1'))
    parser.add_argument('--selection', type=Path, default=Path('work_dirs/crane_symeood_k1_port_day2night_midpoint_sigma_v1/sigma_1p5/selection.json'))
    parser.add_argument('--formal-cache', type=Path, default=Path('work_dirs/port_geometry_midpoint_formal_v1_roi_cache'))
    parser.add_argument('--train-cache', type=Path, default=Path('work_dirs/port_reliability_train_support_v1_cache'))
    parser.add_argument('--input-snapshot', type=Path, default=Path('work_dirs/port_reliability_train_support_v1/train_input_snapshot.json'))
    parser.add_argument('--val-dir', type=Path, default=Path('work_dirs/port_reliability_branches_v1_val_cached_v1'))
    parser.add_argument('--old-policy', type=Path, default=Path('work_dirs/port_midpoint_reliability_v1_fit/policy.json'))
    parser.add_argument('--b-checkpoint', type=Path, default=Path('work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth'))
    args = parser.parse_args()
    # A new child directory is mandatory; no old stage/baseline may be overwritten.
    out = args.run_dir/args.mode
    if out.exists(): raise FileExistsError('Preserve existing stage; choose a new --run-dir: '+str(out))
    prepared = prepare(args)
    if args.mode != 'check': completed(args.run_dir/'check', 'check', prepared['contract'])
    if any(out.resolve() == p.resolve() or p.resolve() in out.resolve().parents
           for p in (args.selection.parent.parent, args.formal_cache, args.train_cache,
                     args.input_snapshot.parent, args.val_dir, args.old_policy.parent, args.b_checkpoint.parent)):
        raise ValueError('Output directory must be outside frozen inputs')
    out.mkdir(parents=True, exist_ok=False)
    base.write_new(out/'input_check.json', prepared['contract'])
    try:
        if args.mode == 'check':
            base.write_new(out/'check_report.json', dict(front_end=prepared['contract']['front_end'],
                train_frames=len(prepared['train']), val_frames=len(prepared['val']),
                static_only=True, tensor_loads=0, detector_updates=0, head_updates=0, test_read=False))
        else: globals()[args.mode](args, prepared, out)
        if prepare(args)['contract'] != prepared['contract']: raise ValueError('Inputs changed during stage')
        finish(out, args.mode, prepared['contract'])
    except Exception as error:
        base.write_new(out/'failure.json', dict(mode=args.mode, type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__': main()
