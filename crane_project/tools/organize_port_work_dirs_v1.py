#!/usr/bin/env python3
"""Organize server outputs without changing frozen experiment source or JSON bytes.

Default is read-only planning. --apply moves port_* outputs and removes ONLY the
closed names below after a verified evidence archive. Closed model archives
exclude .pth/.pt weights; they are evidence backups, not runnable model backups.
Python 3.8 standard library only. Run after experiment processes have stopped.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
from datetime import datetime
import uuid

ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    'crane_symeood_k1_port_day2night_aug_c_v1',
    'crane_symeood_k1_port_day2night_center_size_d_v1',
    'crane_symeood_k1_port_day2night_shape_e_h_v1',
    'crane_symeood_k1_port_day2night_size_f_s_v1',
)
# Same closed finite diagnostics as the previous cleanup; no reliability names.
DIAGNOSTICS = (
    'port_geometry_midpoint_weights_v1_static',
    'port_geometry_midpoint_weights_v1_train',
    'port_geometry_midpoint_motion_v1_static',
    'port_geometry_midpoint_motion_v1_train',
    'port_geometry_midpoint_size_v1_static',
    'port_geometry_midpoint_size_v1_static_path_fix_v1',
    'port_geometry_midpoint_size_v1_train_path_fix_v1',
    'port_geometry_midpoint_size_temporal_v1_static',
    'port_geometry_midpoint_size_temporal_v1_val',
    'port_geometry_midpoint_v1_static',
    'port_geometry_midpoint_v1_train',
    'port_geometry_midpoint_v1_test_static',
    'port_geometry_midpoint_v1_test_diagnosis',
)
# Inputs hardcoded/defaulted in current tools. Only these receive root aliases;
# all other outputs are accessed at their new path or through explicit CLI args.
ALIASES = frozenset((
    'port_geometry_midpoint_formal_v1_roi_cache',
    'port_geometry_midpoint_formal_v1_test_eval',
    'port_geometry_midpoint_sigma15_v1_val_eval',
    'port_midpoint_reliability_v1_collect',
    'port_midpoint_reliability_v1_fit',
    'port_size_reference_v1_train',
    'port_reliability_train_support_v1_cache',
    'port_reliability_train_support_v1',
    'port_reliability_branches_v1_val_cached_v1',
    'port_reliability_branches_v1_test_cached_v1',
    'port_simple_reliability_v1_fit',
    'port_shape_e_h_v1_val_compare.json',
))


def category(name):
    if name.endswith(('.tar.gz', '.tgz', '.sha256')):
        return 'packages'
    if any(s in name for s in ('reliability', 'size_core_', 'size_reference_',
                               'midpoint_size_template', 'direction_consistency')):
        return 'reliability'
    if any(s in name for s in ('geometry', 'shape_e_', 'size_f_', 'center_size_',
                               'train_val_geometry', 'train_geometry')):
        return 'geometry'
    return 'other'


def destination(work, name):
    return work / 'port_results' / category(name) / name


def plain_parents(path, root):
    """Never follow an output-directory symlink during writes/deletion."""
    for p in (root,) + tuple(reversed(path.relative_to(root).parents)):
        p = p if p == root else root / p
        if p.is_symlink():
            raise ValueError('Symlink parent refused: ' + str(p))
        if p.exists() and not p.is_dir():
            raise ValueError('Not a directory: ' + str(p))


def movable_links(base):
    # Relative links inside one payload survive a move. External/absolute links
    # need manual treatment; do not silently break them by changing depth.
    if not base.is_dir():
        return
    for folder, dirs, files in os.walk(str(base), followlinks=False):
        for name in dirs + files:
            p = Path(folder) / name
            if p.is_symlink():
                target = os.readlink(str(p))
                try:
                    p.resolve().relative_to(base.resolve())
                except (ValueError, RuntimeError):
                    raise ValueError('External nested link refused: ' + str(p))
                if Path(target).is_absolute() or not p.exists():
                    raise ValueError('Absolute/broken nested link refused: ' + str(p))


def plan(work):
    if work.is_symlink() or not work.is_dir():
        raise ValueError('work_dirs must be an existing real directory')
    for group in ('geometry', 'reliability', 'packages', 'other'):
        plain_parents(work / 'port_results' / group / 'placeholder', work)
    index = work / 'port_results' / 'INDEX.md'
    if index.is_symlink() or (index.exists() and
            not index.read_text().startswith('# Server output index\n')):
        raise ValueError('Existing foreign INDEX.md refused')
    deletions, moves = [], []
    closed = set(MODELS + DIAGNOSTICS)
    for p in sorted(work.iterdir()):
        if p.name in closed:
            if p.is_symlink() or not p.is_dir():
                raise ValueError('Closed target must be a real directory: ' + str(p))
            deletions.append(p.name)
        elif p.name.startswith('port_') and p.name != 'port_results':
            target = destination(work, p.name)
            plain_parents(target, work)
            if p.is_symlink():
                if (p.name not in ALIASES or not target.exists() or target.is_symlink()
                        or os.readlink(str(p)) != str(target.relative_to(work))):
                    raise ValueError('Unknown/broken root symlink refused: ' + str(p))
                continue
            if not (p.is_dir() or p.is_file()):
                raise ValueError('Special output refused: ' + str(p))
            movable_links(p)
            if target.exists() or target.is_symlink():
                raise ValueError('Destination collision refused: ' + str(target))
            moves.append(dict(name=p.name, to=str(target.relative_to(work)),
                              root_alias=p.name in ALIASES))
    plain_parents(work / '_geometry_archive' / 'placeholder', work)
    return dict(moves=moves, archive_then_remove=deletions,
                closed_model_archive_excludes=['*.pth', '*.pt'],
                weights_recoverable_from_archive=False,
                baseline_and_midpoint_model_dirs='untouched',
                historical_dino_model_dirs='untouched',
                reliability_payloads='move only; no deletion',
                existing_geometry_archives='untouched')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def snapshot(work, names):
    """Snapshot all entries; checkpoint symlinks recorded, never dereferenced."""
    result = {}
    for name in names:
        base = work / name
        if base.is_symlink() or not base.is_dir():
            raise ValueError('Closed directory changed: ' + str(base))
        for folder, dirs, files in os.walk(str(base), followlinks=False):
            for item in [Path(folder)] + [Path(folder) / n for n in dirs + files]:
                key = str(item.relative_to(work))
                if key in result:
                    continue
                mode = item.lstat().st_mode
                weight = name in MODELS and item.suffix.lower() in ('.pth', '.pt')
                if stat.S_ISLNK(mode):
                    if not weight:
                        raise ValueError('Non-weight symlink in closed directory: ' + key)
                    info = dict(kind='link', target=os.readlink(str(item)))
                elif stat.S_ISDIR(mode):
                    info = dict(kind='directory')
                    weight = False
                elif stat.S_ISREG(mode):
                    info = dict(kind='file', size=item.stat().st_size, sha256=sha(item))
                else:
                    raise ValueError('Special file refused: ' + key)
                info['omitted_weight'] = weight
                result[key] = info
    return result


def archive_closed(work, names):
    if not names:
        return None
    before = snapshot(work, names)
    archive_dir = work / '_geometry_archive'
    archive_dir.mkdir(exist_ok=True)
    label = 'closed_outputs_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:12]
    archive = archive_dir / (label + '.tar.gz')
    with tarfile.open(str(archive), 'x:gz', dereference=False) as tf:
        for key, info in sorted(before.items()):
            if not info['omitted_weight']:
                tf.add(str(work / key), arcname=key, recursive=False)
    expected = {k: v for k, v in before.items() if not v['omitted_weight']}
    seen = {}
    with tarfile.open(str(archive), 'r:gz') as tf:
        for member in tf.getmembers():
            if member.isdir():
                info = dict(kind='directory', omitted_weight=False)
            elif member.isfile():
                stream = tf.extractfile(member)
                h = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
                info = dict(kind='file', size=member.size, sha256=h.hexdigest(), omitted_weight=False)
            else:
                raise ValueError('Unexpected archive member: ' + member.name)
            if member.name in seen:
                raise ValueError('Duplicate archive member: ' + member.name)
            seen[member.name] = info
    if seen != expected or snapshot(work, names) != before:
        raise ValueError('Archive/source verification failed; originals kept')
    digest = sha(archive)
    (archive_dir / (label + '.tar.gz.sha256')).write_text(digest + '  ' + archive.name + '\n')
    ledger = dict(archive=archive.name, sha256=digest, originals=before,
                  note='Model .pth/.pt omitted and permanently removed. Evidence-only backup.')
    (archive_dir / (label + '.manifest.json')).write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + '\n')
    for name in names:
        shutil.rmtree(str(work / name))
    return str(archive.relative_to(work))


def active_jobs():
    """Linux server guard: all crane tool/training jobs, excluding this tool."""
    output = subprocess.check_output(['ps', '-eo', 'pid=,args='], universal_newlines=True)
    jobs = []
    for line in output.splitlines():
        if ('organize_port_work_dirs_v1.py' not in line
                and any(s in line for s in ('crane_project/tools/', 'tools/train.py',
                                           'tools/test.py', 'tools/dist_train.sh'))):
            jobs.append(line.strip())
    return jobs


def apply(work, reviewed_plan):
    # Check the full plan again before any mutation. Caller must stop active jobs.
    if plan(work) != reviewed_plan:
        raise ValueError('Directory plan changed; rerun check-only')
    archived = archive_closed(work, reviewed_plan['archive_then_remove'])
    for move in reviewed_plan['moves']:
        source = work / move['name']
        target = work / move['to']
        plain_parents(target, work)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise ValueError('Destination changed; stopping: ' + str(target))
        if source.is_symlink() or not (source.is_file() or source.is_dir()):
            raise ValueError('Source changed; stopping: ' + str(source))
        source.rename(target)  # Same work_dirs filesystem; bytes are not rewritten.
        if move['root_alias']:
            source.symlink_to(move['to'], target_is_directory=target.is_dir())
    result_dir = work / 'port_results'
    result_dir.mkdir(exist_ok=True)
    index = []
    for group in ('geometry', 'reliability', 'packages', 'other'):
        folder = result_dir / group
        if folder.exists():
            index.extend(str(p.relative_to(work)) for p in sorted(folder.iterdir()))
    text = ('# Server output index\n\n'
            'Original filenames, JSON bytes and payloads are unchanged.\n'
            'Only active input aliases remain at work_dirs root.\n'
            'Explicit old output paths should use the paths below.\n'
            'Closed C/D/E-H/F-S model archives exclude weights; they cannot restore models.\n\n'
            + '\n'.join('- ' + p for p in index) + '\n')
    (result_dir / 'INDEX.md').write_text(text)
    log = dict(plan=reviewed_plan, verified_archive=archived, outputs=index)
    name = 'organization_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8] + '.json'
    (result_dir / name).write_text(json.dumps(log, ensure_ascii=False, indent=2) + '\n')
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    work = ROOT / 'work_dirs'
    p = plan(work)
    print(json.dumps(p, ensure_ascii=False, indent=2), flush=True)
    if not args.apply:
        print('READ_ONLY_PLAN_NO_CHANGES')
        return
    jobs = active_jobs()
    if jobs:
        raise RuntimeError('Stop experiment processes before organizing:\n' + '\n'.join(jobs))
    log = apply(work, p)
    print('ORGANIZATION_COMPLETE', 'moved=', len(p['moves']),
          'removed=', len(p['archive_then_remove']), 'archive=', log['verified_archive'])


if __name__ == '__main__':
    main()
