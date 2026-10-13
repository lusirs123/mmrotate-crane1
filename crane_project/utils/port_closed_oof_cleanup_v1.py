"""Exact-manifest cleanup of abandoned auxiliary weights and duplicate Git diffs."""
import hashlib
import json
import os
from pathlib import Path

RUNS = (
    '20261010_real_oof_v1','20261010_real_oof_v1_fix1',
    '20261011_211408_A_instability_trace_v1_diagnose','20261011_fixed_TRAIN_learning_v1',
    '20261011_real_oof_v2_gpu3','20261011_real_oof_v2_gpu3_fix1',
    '20261011_training_instability_forensic_v1','20261012_192025_A_task_budget_gpu3_prevention_v1',
    '20261012_200133_A_task_memory_v2_gpu3_prevention_v1','20261012_A_budget10_cacheaware_gpu3_prevention_v1',
    '20261012_A_budget10_prevention_v1','20261012_A_budget10_scalar_gpu3_prevention_v1',
    '20261012_A_budget10_scalar_prevention_v1','20261012_A_task_budget_gpu3_prevention_v1',
    '20261012_gradient_chain_v1','training_repair_v2_validation')
TEMP_FILES = (
    'oof_classification_stop_cc6282a.bundle','oof_task_budget_entry_fix.bundle',
    'oof_task_budget_source.bundle','oof_task_memory_v2.bundle',
    'port_reliability_oof_prevention_v1_20261012_A_budget10_cacheaware_gpu3.tar.gz',
    'port_reliability_oof_task_budget_20261012_192025_A_task_budget_gpu3_prevention_v1.tar.gz',
    'port_reliability_oof_task_budget_prevention_v1_20261012_A_gpu3.tar.gz',
    'port_reliability_oof_v1_20261011_real_oof_v2_gpu3.tar.gz',
    'port_reliability_oof_v1_20261011_real_oof_v2_gpu3_fix1.tar.gz')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def allowed(root, relative):
    root=Path(root).resolve();p=root/relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Absolute/traversal deletion refused')
    if p.is_symlink() or any(a.is_symlink() for a in p.parents if a!=root and root in a.parents):
        raise ValueError('Symlink deletion path refused')
    if p.resolve()!=p or root not in p.parents:
        raise ValueError('Deletion escapes checkout')
    parts=Path(relative).parts
    if len(parts)==2 and parts[0]=='work_dirs' and parts[1] in TEMP_FILES:return p
    if (len(parts)>=4 and parts[:2]==('work_dirs','port_reliability_oof_v1')
            and parts[2] in RUNS and p.suffix in ('.pt','.pth','.diff')):return p
    raise ValueError('Not an explicitly closed auxiliary artifact')


def plan(root):
    root=Path(root).resolve();selected=[];diffs={};preserved=[]
    for run in RUNS:
        folder=root/'work_dirs/port_reliability_oof_v1'/run
        if not folder.exists():continue
        if folder.is_symlink():raise ValueError('Auxiliary run cannot be a symlink')
        for parent,dirs,names in os.walk(folder,followlinks=False):
            dirs[:]=sorted(d for d in dirs if not (Path(parent)/d).is_symlink())
            for name in sorted(names):
                p=Path(parent)/name
                if p.is_symlink():continue
                if p.suffix not in ('.pt','.pth','.diff'):continue
                rel=str(p.relative_to(root));allowed(root,rel)
                h=sha(p);entry=dict(path=rel,sha256=h,bytes=p.stat().st_size)
                if p.suffix=='.diff':
                    if h not in diffs:diffs[h]=entry;preserved.append(entry);continue
                    entry.update(reason='duplicate_git_diff',retained_copy=diffs[h]['path'])
                else:entry['reason']='abandoned_auxiliary_checkpoint'
                selected.append(entry)
    # Archive contents/bundle objects need a separate preservation receipt before deletion.
    for name in TEMP_FILES:
        p=root/'work_dirs'/name
        if not p.exists():continue
        allowed(root,str(p.relative_to(root)))
        selected.append(dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size,
            reason='redundant_sync_bundle' if name.endswith('.bundle') else 'archive_preserved_extracted'))
    return dict(version='port_closed_oof_cleanup_v1',root=str(root),files=selected,
        retained_unique_git_diffs=preserved,planned_bytes=sum(p['bytes'] for p in selected),
        preserve='All reports/logs/source/configs/membership, original datasets and formal M/policy',
        delete_directories=False)


def execute(root, manifest, archive_receipts):
    root=Path(root).resolve()
    if manifest['root']!=str(root) or manifest['version']!='port_closed_oof_cleanup_v1':
        raise ValueError('Manifest belongs to another checkout/version')
    checked=[]
    # Validate ALL candidates and backups before deleting ANY file.
    for item in manifest['files']:
        p=allowed(root,item['path'])
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise ValueError('Changed cleanup candidate '+item['path'])
        if item['reason']=='duplicate_git_diff':
            retained=allowed(root,item['retained_copy'])
            if sha(retained)!=item['sha256'] or item['retained_copy'] in {x['path'] for x in manifest['files']}:
                raise ValueError('Missing retained diff')
        if item['reason']=='archive_preserved_extracted':
            receipt=archive_receipts.get(item['path'])
            if not receipt or receipt['archive_sha256']!=item['sha256'] or not receipt['members_verified']:
                raise ValueError('Archive has not been safely preserved')
            for member in receipt['members']:
                saved=root/member['saved_path']
                if root not in saved.resolve().parents or not saved.is_file() or sha(saved)!=member['sha256']:
                    raise ValueError('Archive backup member differs')
        if item['reason']=='redundant_sync_bundle':
            receipt=archive_receipts.get(item['path'])
            if not receipt or receipt.get('archive_sha256')!=item['sha256'] or not receipt.get('git_objects_present'):
                raise ValueError('Sync bundle objects have not been verified in Git')
        checked.append((p,item))
    removed=[]
    for p,item in checked:
        # Recheck the immutable file immediately before unlinking it.
        if sha(p)!=item['sha256']:raise ValueError('Cleanup candidate changed during execution')
        p.unlink();removed.append(item)
    return dict(version=manifest['version'],root=str(root),removed=removed,
        bytes_removed=sum(x['bytes'] for x in removed),directories_removed=0,
        formal_artifacts_touched=False)
