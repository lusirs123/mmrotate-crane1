#!/usr/bin/env python3
"""Fixed F-S contract and selected VAL cache comparison; no inference or TEST.

--check-only checks the reviewed sources/config/TRAIN annotations/preflight.
Default mode reads the original B ep24 and the original-rule F-S selection.
The extra ordinary-real tail gates never participate in checkpoint selection.
"""
import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.preflight_port_size_f_s_v1 import (
    CONTROL, check_configs as check_candidate, library_contract)
from crane_project.tools.check_port_size_f_s_v1 import EXPECTED, checked_sources
from crane_project.tools.diagnose_port_center_size_d_v1 import load_frozen_val, sha, val_analysis
from crane_project.tools.compare_port_shape_e_h_val_v1 import conditions as original_conditions, riou_crosscheck
from crane_project.tools.audit_port_train_val_geometry_v1 import summarize, write_new
from crane_project.tools.analyze_port_shape_e_h_train_logs_v1 import config_value
from crane_project.tools.diagnose_port_shape_e_h_train_gradients_v1 import plain
from crane_project.tools.ckpt_sweep import (
    annotation_set_sha256, check_or_record_prediction, METRIC_PROTOCOL_VERSION, SELECTION_CONFIG)

EXPERIMENT = ROOT/'crane_project/configs/crane_symeood_k1_port_day2night_size_f_s_v1_formal.py'
PROTOCOL = ROOT/'crane_project/tools/port_size_f_s_v1_formal_protocol.json'
EPOCHS = [16, 18, 20, 22, 24]


def read_protocol():
    fixed = json.loads(PROTOCOL.read_text())
    if (fixed.get('protocol') != 'port_size_f_s_v1_formal_protocol'
            or fixed.get('candidate_settings') != EXPECTED
            or fixed.get('candidate_epochs') != EPOCHS
            or fixed.get('selection_config') != SELECTION_CONFIG
            or fixed.get('metric_protocol_version') != METRIC_PROTOCOL_VERSION
            or fixed.get('center_thresh_px') != 15.
            or fixed.get('original_gate_count') != 15
            or fixed.get('additional_gate_count') != 2):
        raise ValueError('Frozen F-S formula/selection/evaluation protocol differs')
    if any(sha(ROOT/path) != digest for path, digest in fixed['sources'].items()):
        raise ValueError('Frozen formal sources differ; preserve previous evidence')
    checked_sources()  # The unchanged 34-member preflight contract remains valid.
    return fixed


def check_configs():
    from mmcv import Config
    b, candidate = check_candidate()
    formal = Config.fromfile(str(EXPERIMENT))
    if formal.to_dict() != candidate.to_dict():
        raise ValueError('Formal F-S must resolve exactly to the reviewed candidate')
    return b, formal


def check_formal_contract(preflight_path):
    """Read existing evidence; no model construction, weights, or GPU calls."""
    fixed = read_protocol()
    b, f = check_configs()
    path = Path(preflight_path).resolve()
    probe = json.loads(path.read_text())
    if (sha(path) != fixed['reviewed_preflight']['sha256']
            or probe.get('status') != 'TRAIN_PREFLIGHT_COMPLETE_REVIEW_REQUIRED'
            or probe.get('candidate_settings') != EXPECTED
            or probe.get('sources') != checked_sources()
            or probe.get('optimizer_steps') != 0
            or probe['prerequisite']['sha256'] != fixed['reviewed_math']['sha256']
            or probe['identity']['checkpoint_sha256'] != fixed['frozen_b']['checkpoint_sha256']):
        raise ValueError('Requires the reviewed zero-step F-S preflight')
    annotations = {
        split: annotation_set_sha256(str(ROOT/'crane_project/data/crane_grab_port_day2night_v1'/split/'annfiles'))
        for split in ('train', 'train_sim')}
    if annotations != probe['train']['annotation_sha256']:
        raise ValueError('TRAIN annotations changed after the reviewed preflight')
    current_library = library_contract()
    return b, f, fixed, dict(
        status='FIXED_FORMAL_CONTRACT_CHECK_COMPLETE',
        protocol_sha256=sha(PROTOCOL), reviewed_preflight_sha256=sha(path),
        config_sha256=dict(b=sha(CONTROL), f_s=sha(EXPERIMENT)),
        resolved_formal_equals_reviewed_candidate=True,
        annotation_sha256=annotations, seed=0, epochs=24, gpus=2,
        samples_per_gpu=2, total_batch=4, optimizer_steps=0,
        formal_training_executed=False, test_repeatedly_exposed=True,
        library_contract=current_library,
        library_matches_reviewed_preflight=current_library == probe['library_contract'])


def rename_arm(value, source='d'):
    if isinstance(value, dict):
        names = {source: 'f_s', source+'_output': 'f_s_output',
                 source+'_center_hit': 'f_s_center_hit', source+'_minus_b': 'f_s_minus_b'}
        return {names.get(k, k.replace('compensated_d', 'size_f_s')):
                rename_arm(v, source) for k, v in value.items()}
    if isinstance(value, list):
        return [rename_arm(v, source) for v in value]
    return value


def conditions(b, f, historical_severe):
    """Keep all 15 original gates; recompute both tail baselines on shared rows."""
    if ([r['image'] for r in b] != [r['image'] for r in f]
            or len({r['image'] for r in b}) != len(b)):
        raise ValueError('Ordered unique VAL pairing required')
    result = rename_arm(original_conditions(b, f), 'e_h')
    if len(result['checks']) != 15:
        raise ValueError('Original joint gate set changed')
    pairs = [(a, c) for a, c in zip(b, f)
             if a['domain'] == 'real' and a['image'] not in historical_severe
             and a['metrics']['output'] and c['metrics']['output']]
    summaries = dict(b=summarize([a for a, c in pairs]), f_s=summarize([c for a, c in pairs]))
    tail = {}
    for field in ('mean', 'rmse'):
        bv = summaries['b']['center_error_px'][field]
        fv = summaries['f_s']['center_error_px'][field]
        tail['real_ordinary_shared_center_'+field+'_not_worse'] = bool(pairs) and fv <= bv
    original_checks = deepcopy(result['checks'])
    result.update(original_checks=original_checks, additional_tail_checks=tail,
                  ordinary_real_shared=dict(n=len(pairs), images=[a['image'] for a, c in pairs],
                                            summaries=summaries, excluded_historical_b_frames=sorted(historical_severe)))
    result['checks'].update(tail)
    result['all_conditions_met'] = all(result['checks'].values())
    result['original_conditions_met'] = all(original_checks.values())
    result['note'] = '17 frozen VAL gates; tail gates do not affect selection. No automatic replacement or significance claim.'
    return result


def severe_changes(rows, historical_severe):
    """Use the original zero-RIoU-output definition, separate from missing output."""
    groups = {}
    for row in rows:
        m = row['metrics']
        if row['domain'] == 'real' and m['output'] and m['riou'] == 0:
            groups.setdefault(row['sequence'], []).append(dict(
                image=row['image'], center_error_px=m['center_error_px'],
                new_relative_to_historical_b=row['image'] not in historical_severe))
    return groups


def validate_selections(b_selection, f_selection, fixed):
    names = {'epoch_'+str(x) for x in EPOCHS}
    for arm, selected, config in (('b', b_selection, CONTROL), ('f_s', f_selection, EXPERIMENT)):
        if (selected.get('evidence_role') != 'source_val_checkpoint_selection'
                or selected.get('candidate_epochs') != EPOCHS
                or set(selected.get('all_checkpoints', {})) != names
                or selected.get('selected_checkpoint') not in names
                or selected.get('selection_config') != fixed['selection_config']
                or selected.get('metric_protocol_version') != fixed['metric_protocol_version']
                or selected.get('center_thresh_px') != 15.
                or selected.get('config_sha256') != sha(config)
                or selected.get('source_val_annotations_sha256') != fixed['frozen_b']['annotation_sha256']):
            raise ValueError('Original VAL selection/config protocol differs: '+arm)
    if b_selection['selected_checkpoint'] != 'epoch_24':
        raise ValueError('Keep original B ep24; no checkpoint reselection')


def checkpoint_contract(meta, cfg, epoch):
    """Check saved training identity using literal AST, without executing config."""
    fields = ('model', 'data', 'optimizer', 'optimizer_config', 'lr_config',
              'runner', 'load_from', 'resume_from')
    parsed = {}
    for node in ast.parse(meta.get('config', '')).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in fields):
            key = node.targets[0].id
            if key in parsed:
                raise ValueError('Duplicate saved config field: '+key)
            parsed[key] = config_value(node.value)
    if (parsed != plain({k: cfg.to_dict()[k] for k in fields})
            or meta.get('seed') != 0 or meta.get('epoch') != epoch
            or meta.get('iter') != 640*epoch):
        raise ValueError('Selected checkpoint training config/seed/epoch/iter differs')
    return dict(status='MATCH', checked_fields=list(fields), seed=0, epoch=epoch, iter=640*epoch)


def load_arm(cfg, config, sweep, epoch):
    import torch
    rows, identity = load_frozen_val(cfg, config, sweep, epoch)
    ann = ROOT/'crane_project/data/crane_grab_port_day2night_v1/val/annfiles'
    check_or_record_prediction(str(config), identity['checkpoint'], identity['pkl'], str(ann), 'source_val')
    saved = torch.load(identity['checkpoint'], map_location='cpu')
    identity['training_contract'] = checkpoint_contract(saved.get('meta', {}), cfg, int(epoch.split('_')[1]))
    del saved
    if (len(rows) != 887 or len({r['image'] for r in rows}) != 887
            or sum(r['domain'] == 'real' for r in rows) != 375
            or sum(r['domain'] == 'sim' for r in rows) != 512):
        raise ValueError('Expected 375 real + 512 sim unique VAL rows')
    return rows, identity


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--require-reviewed-library', action='store_true',
                    help='Reject a library contract change from the reviewed server preflight')
    ap.add_argument('--train-preflight', default='work_dirs/port_size_f_s_v1_train_preflight.json')
    ap.add_argument('--b-sweep', default='work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/val_sweep_port_v1')
    ap.add_argument('--f-sweep', default='work_dirs/crane_symeood_k1_port_day2night_size_f_s_v1/val_sweep_port_v1')
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()
    os.chdir(ROOT)
    out = Path(args.out_json).resolve()
    if out.exists():
        raise FileExistsError('Preserve existing report; choose a new output name')
    b, f, fixed, contract = check_formal_contract(args.train_preflight)
    if args.require_reviewed_library and not contract['library_matches_reviewed_preflight']:
        raise ValueError('Reviewed server library contract differs; check the environment before formal execution')
    if args.check_only:
        write_new(out, contract)
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return
    bs, fs = Path(args.b_sweep).resolve(), Path(args.f_sweep).resolve()
    sb, sf = [json.loads((s/'sweep_results.json').read_text()) for s in (bs, fs)]
    validate_selections(sb, sf, fixed)
    if sha(bs/'sweep_results.json') != fixed['frozen_b']['selection_sha256']:
        raise ValueError('Historical B selection bytes changed')
    rb, ib = load_arm(b, CONTROL, bs, 'epoch_24')
    rf, iff = load_arm(f, EXPERIMENT, fs, sf['selected_checkpoint'])
    for key in ('checkpoint_sha256', 'pkl_sha256', 'config_sha256', 'annotation_sha256'):
        if ib[key] != fixed['frozen_b'][key]:
            raise ValueError('Historical B artifact changed: '+key)
    analysis = rename_arm(val_analysis(rb, rf))
    if analysis['historical_severe_b_frames'] != fixed['historical_severe_b_frames']:
        raise ValueError('Historical B severe frame group changed')
    gates = conditions(rb, rf, fixed['historical_severe_b_frames'])
    if set(gates['checks']) != set(fixed['gate_names']):
        raise ValueError('Frozen 17-gate identities differ')
    cross = dict(b=riou_crosscheck(rb), f_s=riou_crosscheck(rf))
    report = dict(protocol='port_size_f_s_v1_val_compare',
        evidence_role='source_val_only_fixed_experiment_comparison', formal_contract=contract,
        identities=dict(b=ib, f_s=iff), selection_info=sf['selection_info'],
        geometry=analysis, pre_registered_conditions=gates, riou_crosscheck=cross,
        real_zero_riou_outputs_by_sequence=dict(
            b=severe_changes(rb, fixed['historical_severe_b_frames']),
            f_s=severe_changes(rf, fixed['historical_severe_b_frames'])),
        selected_original_val_metrics=dict(
            b=sb['all_checkpoints']['epoch_24'].get('metrics'),
            f_s=sf['all_checkpoints'][sf['selected_checkpoint']].get('metrics')),
        metric_consistency_review_required=any(x['above_1e_3'] or x['half_threshold_changes'] for x in cross.values()),
        limitations=['Historical single-seed B control, not simultaneous paired optimization.',
                     'Output center hit, output coverage and all-frame correct coverage use distinct denominators.',
                     'TEST has been exposed repeatedly and is not read or used here.',
                     'Shared features can change centers, angles, coverage and continuity; depth has no independent GT check.',
                     'A single run or small improvement does not establish stable/significant gains.'])
    write_new(out, report)
    print(json.dumps(gates, ensure_ascii=False, indent=2))
    print('VAL selection:', report['selection_info'])
    print('RIoU crosscheck:', cross)
    print('Saved', out)


if __name__ == '__main__':
    main()
