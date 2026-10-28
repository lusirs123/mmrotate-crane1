#!/usr/bin/env python3
"""Original C size-only continuation from common ep11; revised VAL selection.

The original model, C update/evaluate and six-epoch order are reused directly.
No TEST or depth evaluation is launched by this entry point.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_symeood_110_lvsr_c_ep7 as legacy
from crane_project.tools.run_port_size_native_train_v1 import read, write, sha

base, old, objective = legacy.base, legacy.old, legacy.objective
CONTRACT = ROOT/'crane_project/data_contracts/symeood_110_lvsr_c_ep11_joint_v1.json'
COMMON = legacy.COMMON
PARENT_SHA = 'cb0d65914a073306559a6e5ebcb5a4186676922cb4ae9cb23dff39e202cdb80a'
OUT = COMMON/'continuation_C_ep11_joint_v2'


def selection_key(summary, epoch):
    """Joint <=10% full-frame count, bilateral mean MAE, earlier epoch."""
    value = summary['overall']['edge_residual']
    pair = summary['overall']['paired']
    count = value['joint_size10_full_frame']
    if (value['frames'] != 887 or count['denominator'] != 887 or
            value['output_coverage']['numerator'] != 885 or
            not pair['frozen_identity'] or pair['new_output'] or pair['lost_output']):
        raise ValueError('Frozen VAL/output/center-angle-score identity differs')
    errors = [value['edges'][e]['mae'] for e in ('long', 'short')]
    if any(not math.isfinite(v) or v < 0 for v in errors):
        raise ValueError('Invalid VAL edge MAE')
    return [-count['numerator'], sum(errors)/2, epoch]


def choose(events):
    # Legacy eligibility remains in each epoch as a diagnostic, not a veto.
    winner = min(events, key=lambda e: e['selection_key'])
    return dict(selected=winner['epoch'], representative=winner['epoch'],
                selection_policy='joint10_full_frame_then_bilateral_MAE_then_earlier',
                legacy_eligibility_is_diagnostic_only=True)


def select_with_parent(events, parent_summary):
    parent = dict(epoch=0, selection_key=selection_key(parent_summary, 0))
    winner = min([parent]+events, key=lambda e: e['selection_key'])
    return dict(selected_pool_epoch=winner['epoch'], parent_common_epoch=11,
                best_continuation_epoch=choose(events)['selected'],
                replace_parent=winner['epoch'] != 0, fixed_final=6)


def checked():
    proof = read(CONTRACT)
    base.checked_sources()
    if (proof['parent_epoch'] != 11 or proof['parent_sha256'] != PARENT_SHA or
            proof['arms'] != ['C'] or proof['epochs'] != 6 or
            proof['settings'] != read(legacy.CONTRACT)['settings']):
        raise ValueError('Fixed ep11 C contract differs')
    for name, digest in proof['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('C ep11 source differs: '+name)
    for name, digest in proof['common_artifacts'].items():
        if sha(COMMON/name) != digest:
            raise ValueError('Common-stage evidence differs: '+name)
    scores = []
    for epoch in range(1, 25):
        event = read(COMMON/('fixed_b/epoch_%02d_val.json' % epoch))
        scores.append((selection_key(event['metrics'], epoch), epoch))
    if min(scores)[1] != 11 or min(scores)[0][0] != -677:
        raise ValueError('Revised joint-first parent is not original ep11')
    return proof


def load_inputs(torch):
    """Use the existing new110 ROI cache and M21 boxes saved at ep11."""
    from crane_project.tools import train_symeood_110_midpoint_v1 as f
    f.runtime_imports()
    train, val, _ = f.load_cache(base.FORMAL/'roi_cache', f.checked_sources())
    path = COMMON/'fixed_b/head_epoch_11.pth'
    if sha(path) != PARENT_SHA:
        raise ValueError('Requires original shared common-stage ep11 bytes')
    payload = torch.load(str(path), map_location='cpu')
    if (payload['epoch'] != 11 or payload['arm'] != 'fixed_b' or
            payload['parent_epoch'] is not None or
            payload['input_identity']['detector_sha256'] != base.DETECTOR_SHA or
            payload['input_identity']['midpoint_sha256'] != base.MIDPOINT_SHA):
        raise ValueError('Parent is not original shared ep11 on new110/M21')
    if sha(Path(read(f.BASE_CONTRACT)['checkpoint'])) != base.DETECTOR_SHA:
        raise ValueError('Detector weight differs')
    if sha(base.FORMAL/'train/head_epoch_21.pth') != base.MIDPOINT_SHA:
        raise ValueError('Midpoint weight differs')
    saved = {}
    for split, rows in (('train', train), ('val', val)):
        with gzip.open(COMMON/('fixed_b/epoch_11_'+split+'.rows.jsonl.gz'), 'rt') as stream:
            saved[split] = [json.loads(line) for line in stream]
        lookup = {objective.key(r): r for r in saved[split]}
        if len(lookup) != len(rows) or len(lookup) != len(saved[split]):
            raise ValueError('Parent/cache row inventory differs')
        for row in rows:
            reference = lookup[objective.key(row)]
            for key in ('image', 'scale', 'domain', 'sequence', 'frame_id', 'split'):
                if row[key] != reference[key]:
                    raise ValueError('Cache metadata differs: '+key)
            b = row['boxes_original'][0].tolist() if len(row['boxes_original']) else None
            if b != reference['b'] or row['gt_original'][0].tolist() != reference['gt']:
                raise ValueError('Cache B/GT differs from ep11 records')
            row['midpoint_original'] = (row['boxes_original'].new_tensor([reference['midpoint']])
                if reference['midpoint'] is not None else row['boxes_original'].clone())
            row['shard'] = reference['shard']
    train.sort(key=lambda r: (r['scale'], r['image']))
    val.sort(key=lambda r: (r['scale'], r['image']))
    if [len(train), len(val), sum(bool(len(r['midpoint_original'])) for r in train)] != [5116, 887, 5112]:
        raise ValueError('Fixed TRAIN/VAL counts differ')
    parent = dict(epoch=11, path=path, sha256=PARENT_SHA,
        summary=read(COMMON/'fixed_b/epoch_11_val.json')['metrics'],
        train_rows=saved['train'], val_rows=saved['val'])
    return train, val, parent, payload


def status(state, **details):
    # base.fit_stage is single-process; the physical GPU comes from this run.
    value = dict(state=state, time=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(), physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),
        TEST=False, depth=False, detector_updates=0, midpoint_updates=0,
        automatic_retry=False, parent_epoch=11, **details)
    temporary = OUT/'status.tmp'
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(OUT/'status.json')


@contextmanager
def revised_reporting():
    """Bind the new selector/status/backup names for the unchanged fit loop.

    C forward, update, objective, optimizer and evaluation are never rebound.
    Restore all report bindings even if the original loop raises an error.
    """
    names = ('ranking', 'choose', 'backup', 'update_status')
    originals = {name: getattr(base, name) for name in names}
    def scoped_backup(path, stage, out, torch, role):
        role = 'revised_joint_selected' if role == 'eligible_selected' else role
        return originals['backup'](path, 'C_ep11_joint_v1_'+stage, out, torch, role)
    def scoped_status(out, state, **details):
        if out != OUT:
            raise ValueError('Unexpected status output directory')
        return status(state, **details)
    try:
        base.ranking, base.choose = selection_key, choose
        base.backup, base.update_status = scoped_backup, scoped_status
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def run():
    proof = checked()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != str(proof['physical_gpu']) or
            os.environ.get('WORLD_SIZE', '1') != '1'):
        raise ValueError('Use the pinned single physical GPU')
    OUT.mkdir(exist_ok=False)
    try:
        from crane_project.tools import train_symeood_110_midpoint_v1 as f
        f.runtime_imports()
        torch = f.torch
        torch.set_num_threads(1)
        runtime = f.runtime(0)
        if runtime != read(base.FORMAL/'roi_cache/cache_manifest.json')['runtime']:
            raise ValueError('Runtime differs from original new-chain cache')
        write(OUT/'frozen_contract.json', proof)
        status('loading_existing_inputs')
        train, val, parent, payload = load_inputs(torch)
        write(OUT/'identity.json', dict(parent_epoch=11, parent_sha256=PARENT_SHA,
            input_identity=payload['input_identity'], runtime=runtime,
            optimizer='fresh Adam', source_sha256=sha(Path(__file__)),
            contract_sha256=sha(CONTRACT), detector_forwards=0, midpoint_forwards=0))
        # Same bounded three-update check as the existing C entry; discarded.
        status('bounded_parent_trial')
        write(OUT/'trial.json', legacy.trial(train, val, parent, payload['head_state'], torch))
        with revised_reporting():
            result, backup = base.fit_stage(train, val, OUT, 'C', payload['input_identity'], torch, parent)
        final_choice = select_with_parent(result['epochs'], parent['summary'])
        backups = dict(best_continuation=backup)
        if result['selected'] != 6:
            backups['fixed_C6'] = base.backup(OUT/'C/head_epoch_06.pth',
                'C_ep11_joint_v1_fixed_C6', OUT, torch, 'fixed_final')
        else:
            backups['fixed_C6'] = backup
        if checked() != proof:
            raise ValueError('Sources or common-stage evidence changed during run')
        write(OUT/'selection_with_C0.json', dict(**final_choice,
            policy=proof['selection'], parent_VAL_joint10=677,
            candidate_keys={str(e['epoch']): e['selection_key'] for e in result['epochs']},
            parent_key=selection_key(parent['summary'], 0)))
        write(OUT/'completion.json', dict(state='complete', parent_epoch=11,
            parent_sha256=PARENT_SHA, versions=final_choice,
            formal_updates=result['formal_updates'], backups=backups,
            original_eligibility_diagnostic_only=True, TEST=False, depth=False))
        status('complete', formal_updates=result['formal_updates'], **final_choice)
    except BaseException as error:
        status('stopped_on_error', error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        run()
    else:
        proof = checked()
        print(json.dumps(dict(state='static_check_pass', parent_epoch=11,
            epochs=proof['epochs'], training_started=False)))
