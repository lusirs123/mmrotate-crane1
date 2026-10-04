#!/usr/bin/env python3
"""Fixed epoch04 readers on 432 reference-held TRAIN + all 887 midpoint VAL.

check is CPU provenance checking. run streams one image at a time under no_grad.
This entry never fits a model/policy, updates detection, reads TEST, or deploys
new reader flags. Every invocation requires a fresh output directory.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_midpoint_reliability_v1 as migration
from crane_project.utils import port_midpoint_size_template_v1 as assessment

base = migration.base
simple = migration.simple
PROTOCOL = ROOT/'crane_project/tools/port_midpoint_size_template_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_midpoint_size_template_v1_sources.json'


def checked_sources():
    protocol = json.loads(PROTOCOL.read_text())
    manifest = json.loads(SOURCES.read_text())
    actual = {p: base.sha(ROOT/p) for p in manifest['sources']}
    if (protocol['protocol'] != assessment.VERSION or manifest['protocol'] != assessment.VERSION
            or manifest['sources'] != actual or manifest['protocol_sha256'] != base.sha(PROTOCOL)
            or manifest['parent_midpoint_manifest_sha256'] != base.sha(migration.SOURCES)
            or protocol['test_read'] or any(protocol[k] for k in
                ('detector_updates', 'midpoint_updates', 'reference_updates', 'policy_updates',
                 'reader_parameter_search', 'deployment_cutoff_created'))
            or protocol['frames'] != {'reference_holdout_train':432, 'val':887}
            or protocol['coverage_fractions'] != [.9, .95]):
        raise ValueError('Fixed full-assessment source/protocol contract differs')
    return protocol, dict(manifest_sha256=base.sha(SOURCES), sources=actual)


def reviewed_stage(directory, role, current_contract, pins):
    """Accept only the exact reviewed pre-probe-fix success, with byte pins.

    Historical sources are data provenance, not executable instructions. Only
    the sources field may differ from the currently verified migration contract.
    Front end, data, parent reference, formal proof and protocols must be equal.
    """
    directory = Path(directory)
    path = directory/'completion.json'
    if base.sha(path) != pins[role]['completion_sha256']:
        raise ValueError('Requires exact reviewed '+role+' completion; do not rewrite old evidence')
    completion = json.loads(path.read_text())
    saved = completion['contract']
    if simple.fingerprint(saved) != pins['contract_sha256']:
        raise ValueError('Historical source contract is not the reviewed migration')
    if ({k:v for k,v in saved.items() if k != 'sources'} !=
            {k:v for k,v in current_contract.items() if k != 'sources'}):
        raise ValueError('Historical front end/data/protocol differs from current fixed inputs')
    if completion['artifacts'] != pins[role]['artifacts']:
        raise ValueError('Historical artifact manifest differs')
    expected = {'collect':'MIDPOINT_TRAIN_VAL_COLLECTION_COMPLETE',
                'fit':'MIDPOINT_SIMPLE_POLICY_FIT_COMPLETE'}[role]
    migration.completed(directory, expected, saved)
    return saved


def evaluation_rows(train, val, partition, counts):
    if len(train) != 2558 or len(val) != 887:
        raise ValueError('Requires complete original 2558 TRAIN and 887 VAL')
    held = [r['image'] for r in partition['holdout']]
    if len(set(held)) != len(held) or any(set(held) & {r['image'] for r in partition[k]}
                                        for k in ('fit', 'guard')):
        raise ValueError('Reference-held TRAIN overlaps fit/guard')
    index = {r['image']:r for r in train}
    if len(index) != len(train) or not set(held) <= set(index):
        raise ValueError('Reference held images missing/duplicated in locked collection')
    plan = [(role, deepcopy(r)) for role, rows in
            (('reference_holdout_train', [index[k] for k in held]), ('val', val)) for r in rows]
    if ({k:sum(role == k for role, _ in plan) for k in counts} != counts
            or len({r['image'] for _,r in plan}) != sum(counts.values())):
        raise ValueError('Wrong fixed full-assessment plan')
    for role, row in plan:
        if row['split'] not in (('train', 'train_sim') if role != 'val' else ('val',)):
            raise ValueError('Role cannot contain TEST or another split')
    return plan


def prepare(args):
    protocol, sources = checked_sources()
    previous = migration.prepare(args)
    current = previous[5]
    if previous[0]['template_reader'] != protocol['template_parameters']:
        raise ValueError('Template parameters differ from the fixed reviewed reader')
    pins = protocol['reviewed_migration_stages']
    collected = reviewed_stage(args.collection_dir, 'collect', current, pins)
    train, val = migration.read_collection(args.collection_dir, collected, previous[1])
    fitted = reviewed_stage(args.policy.parent, 'fit', current, pins)
    if args.policy.name != 'policy.json':
        raise ValueError('Use exact reviewed midpoint-refit policy.json')
    policy = json.loads(args.policy.read_text())
    if policy['contract'] != fitted or policy.get('test_read') is not False:
        raise ValueError('Reviewed policy contract differs')
    migration.new.MidpointReliability(policy, current['front_end'])
    reference = migration.reference
    train_report = args.reference_checkpoint.parent/'train_report.json'
    report_sha = reference.gate(train_report,
        'FIXED_EPOCH4_SIZE_REFERENCE_TRAIN_COMPLETE_REVIEW_REQUIRED', previous[1][8])
    marker = Path(str(args.reference_checkpoint)+'.sha.json')
    tag = json.loads(marker.read_text())
    expected = previous[0]['reference_sha256']
    if (base.sha(args.reference_checkpoint) != expected or tag['sha256'] != expected
            or tag['contract_sha256'] != simple.fingerprint(previous[1][8])):
        raise ValueError('Requires exact frozen reference epoch04 bytes and contract')
    plan = evaluation_rows(train, val, previous[1][6], protocol['frames'])
    for role, expected_domains in (('reference_holdout_train', protocol['reference_holdout_domains']),
                                    ('val', protocol['VAL_domains'])):
        actual = {d:sum(k == role and r['domain'] == d for k,r in plan) for d in ('real','sim')}
        if actual != expected_domains:
            raise ValueError('Fixed role/domain support differs')
    identities = [[role, r['image']] for role, r in plan]
    contract = dict(protocol=protocol, sources=sources, current_migration_contract=current,
        reused_reviewed_stage_contract_sha256=pins['contract_sha256'],
        reference_train_report_sha256=report_sha, reference_marker_sha256=base.sha(marker),
        reference_checkpoint_sha256=expected,
        frozen_policy_sha256=base.sha(args.policy),
        frozen_collection_sha256=base.sha(args.collection_dir/'predictions.jsonl'),
        evaluation_plan_sha256=simple.fingerprint(identities),
        locked_evaluation_records_sha256=simple.fingerprint(plan),
        test_read=False, reference_holdout_is_not_detector_or_simple_holdout=True)
    return protocol, previous, policy, plan, contract


def run(args, prepared):
    protocol, previous, policy, plan, contract = prepared
    formal, torch, detector, head, pipeline, model = migration.online_modules(args, previous, True)
    modules = {'b':detector, 'midpoint':head, 'reference':model}
    before = {name:base.state_digest(module) for name,module in modules.items()}
    policy_before = simple.fingerprint(policy)
    runtime = migration.new.MidpointReliability(policy, previous[5]['front_end'])
    records, max_delta = [], {'b':0., 'midpoint':0.}
    settings = previous[0]
    with (args.out_dir/'assessment_rows.jsonl').open('x') as stream, \
            (args.out_dir/'progress.jsonl').open('x') as progress:
        for index, (role, row) in enumerate(plan, 1):
            features, meta, _ = migration.reference.view(row, previous[1][7], detector, pipeline, args.gpu)
            first = migration.midpoint_from_features(formal, torch, detector, head, features, meta)
            for key, field in (('b','b_original'), ('midpoint','pred')):
                delta = assessment.paired_prediction(row[field], first[key], row['image'])
                max_delta[key] = max(max_delta[key], delta)
            with torch.no_grad():
                probability = model(features[0]).sigmoid()[0,0].cpu().numpy()
            # Lock context and labels to the SAME reviewed final midpoint boxes.
            evidence = assessment.online_readings(probability, row['pred'], row['image_size'],
                                                  meta, runtime, settings)
            after = migration.midpoint_from_features(formal, torch, detector, head, features, meta)
            if first != after:
                raise ValueError('Reference side computation changed final midpoint: '+row['image'])
            item = assessment.record(row, evidence, role)
            item['online_formal_parity_passed'] = True
            item['final_output_before_after_exact'] = True
            item['transform_meta'] = {key:list(meta[key]) for key in
                ('img_shape', 'ori_shape', 'pad_shape', 'scale_factor')}
            # Metadata may contain NumPy scalars in the native OpenMMLab view.
            item['transform_meta'] = json.loads(json.dumps(item['transform_meta'],
                default=lambda value:value.item(), allow_nan=False))
            # JSON rows retain numeric evidence only; no per-image GPU tensors/maps.
            stream.write(json.dumps(item, allow_nan=False)+'\n')
            records.append(item)
            del features, probability, first, after
            if index % 100 == 0 or index in (432, len(plan)):
                stream.flush()
                event = dict(completed=index, total=len(plan), role=role, image=row['image'],
                    peak_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20)
                progress.write(json.dumps(event, allow_nan=False)+'\n'); progress.flush()
                print('Fixed readers', role, index, '/', len(plan), flush=True)
    after = {name:base.state_digest(module) for name,module in modules.items()}
    if before != after or simple.fingerprint(policy) != policy_before:
        raise ValueError('Assessment changed frozen model/policy state')
    stats = assessment.summarize(records, protocol['coverage_fractions'])
    flags = {}
    for role in protocol['frames']:
        _, flags[role] = simple.evaluate([r for r in records if r['assessment_role'] == role], policy['simple_policy'])
    # Read saved JSON rows back: serialization must retain all numeric statistics.
    restored = [json.loads(line) for line in (args.out_dir/'assessment_rows.jsonl').read_text().splitlines()]
    if records != restored or stats != assessment.summarize(restored, protocol['coverage_fractions']):
        raise ValueError('Saved assessment rows/statistics do not replay exactly')
    report = dict(status='FIXED_MIDPOINT_SIZE_TEMPLATE_FULL_ASSESSMENT_COMPLETE_REVIEW_REQUIRED',
        contract=contract, summary=stats, frozen_three_flag_stats=flags,
        state_before=before, state_after=after, policy_before_after_exact=True,
        online_formal_parity_passed=True, max_absolute_formal_box_difference=max_delta,
        frozen_context_comes_from_reviewed_final_midpoint=True,
        final_boxes_scores_counts_and_missing_outputs_preserved=True,
        all_images_checked_by_SHA=True, GT_online=False, new_reader_flags_created=False,
        no_model_or_threshold_selection=True, test_read=False,
        definitions=dict(error_positive_class='final_size_max_relative_error_GT_gt_0.1',
            rank='higher_risk_means_more_likely_bad_not_calibrated_error_probability',
            matched_points='offline_exact_count_risk_then_image_with_all_method_tie_bounds',
            unavailable='numeric_reference_undefined_not_a_correctness_label',
            common_support='both_readers_defined_score_simple_evaluated_on_the_same_outputs',
            all_output_same_count='reader_unavailable_is_rejected_score_simple_can_use_all_outputs',
            frozen_stats_correct_rejected='legacy_field_means_good_false_rejected',
            frozen_stats_incorrect_rejected='legacy_field_means_bad_correctly_rejected'),
        detector_updates=0, midpoint_updates=0, reference_updates=0, policy_updates=0,
        peak_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved(args.gpu)/2**20,
        evidence_role='reference_holdout_feasibility_and_exploratory_source_VAL_not_independent_confirmation')
    base.write_new(args.out_dir/'assessment.json', report)
    for role in protocol['frames']:
        for domain in ('real', 'sim'):
            group = stats[role]['domain:'+domain]
            print(role, domain, 'outputs', group['outputs'], '/', group['frames'],
                'center/output', group['center_hit_rate_on_outputs'],
                'full-frame center', group['all_frame_center_correct_coverage'], flush=True)
            print('  common', group['common_defined']['support_outputs'],
                'AUROC', {m:r['error_auroc'] for m,r in group['common_defined']['rank'].items()}, flush=True)
    return report['status']


def finish(args, prepared, status):
    if prepare(args)[4] != prepared[4]:
        raise ValueError('Frozen inputs/sources changed during assessment')
    artifacts = {p.name:base.sha(p) for p in sorted(args.out_dir.iterdir()) if p.is_file()}
    base.write_new(args.out_dir/'completion.json', dict(status=status, contract=prepared[4],
        artifacts=artifacts, mode=args.mode, test_read=False, detector_updates=0,
        midpoint_updates=0, reference_updates=0, policy_updates=0))
    print('Saved', args.out_dir, status, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'run'), required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out-dir', type=Path, required=True)
    defaults = dict(selection='crane_symeood_k1_port_day2night_midpoint_formal_v1/selection.json',
        formal_cache='port_geometry_midpoint_formal_v1_roi_cache',
        collection_dir='port_midpoint_reliability_v1_collect',
        policy='port_midpoint_reliability_v1_fit/policy.json',
        reference_checkpoint='port_size_reference_v1_train/epoch_04.pth',
        train_cache='port_reliability_train_support_v1_cache',
        input_snapshot='port_reliability_train_support_v1/train_input_snapshot.json',
        val_dir='port_reliability_branches_v1_val_cached_v1',
        old_policy='port_simple_reliability_v1_fit/policy.json',
        b_checkpoint='crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth')
    for name, path in defaults.items():
        parser.add_argument('--'+name.replace('_', '-'), type=Path, default=Path('work_dirs')/path)
    args = parser.parse_args()
    prepared = prepare(args)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    base.write_new(args.out_dir/'input_check.json', dict(contract=prepared[4], mode=args.mode,
        evaluation_frames=prepared[0]['frames'], CPU_only=args.mode == 'check'))
    try:
        status = 'FIXED_MIDPOINT_SIZE_TEMPLATE_INPUTS_PASS' if args.mode == 'check' else run(args, prepared)
        finish(args, prepared, status)
    except Exception as error:
        base.write_new(args.out_dir/'failure.json', dict(mode=args.mode,
            type=type(error).__name__, error=str(error), test_read=False))
        raise


if __name__ == '__main__':
    main()
