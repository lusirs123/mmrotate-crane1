"""Delete only verified redundant archives, regenerable pyc, and empty dirs.

No model, tensor, prediction, log, manifest or active-run deletion is allowed.
All preserved archive members are rehashed immediately before execution.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[2]
NAMES=(
 'symeood_ablation_v1','historical_b_ablation111_audit_20261025_v1',
 'symeood_b111_canonical_20261025_v2','symeood_c_pair_seed0_20261025_v1',
 'symeood_c_pair_seed0_20261025_v2','symeood_usg_v1',
 'symeood_cross_entry_review_20261025_v1','symeood_mal_strength_seed0_20261026_v1',
 'symeood_110_retrain_preflight_20261026_v1',
 'symeood_110_midpoint_lvsr_linkage_audit_20261026_v1',
 'symeood_110_midpoint_lvsr_linkage_audit_20261026_v2')
ACTIVE='symeood_110_retrain_seed0_20261026_v1'
OUT='symeood_ablation_v1/cleanup_review_20261026_v1'


def sha(path):
    with Path(path).open('rb') as f:return stream_sha(f)


def stream_sha(f):
    h=hashlib.sha256()
    for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def save(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2);f.write('\n')


def allowed(root,relative):
    p=PurePosixPath(relative)
    assert not p.is_absolute() and '..' not in p.parts and p.parts and p.parts[0] in NAMES,'Out of cleanup scope'
    target=root/'work_dirs'/str(p)
    assert not target.is_symlink(),'Never unlink a symlink in this cleanup'
    assert root/'work_dirs' in target.resolve().parents,'Escaping symlink'
    assert not any(x.is_symlink() for x in target.parents if x!=root),'Symlink ancestor'
    return target


def member_name(name):
    p=PurePosixPath(name)
    assert not p.is_absolute() and '..' not in p.parts and '\\' not in name,'Unsafe member'
    return p


def archive_files(path):
    if path.suffix=='.zip':
        with zipfile.ZipFile(path) as z:
            seen=set()
            for m in z.infolist():
                member_name(m.filename)
                assert not stat.S_ISLNK(m.external_attr>>16),'Archive symlink'
                if m.is_dir():continue
                assert m.filename not in seen,'Duplicate member name';seen.add(m.filename)
                with z.open(m) as f:h=stream_sha(f)
                yield m.filename,m.file_size,h
    else:
        with tarfile.open(path,'r:*') as z:
            seen=set()
            for m in z:
                member_name(m.name)
                if m.isdir():continue
                assert m.isfile() and m.name not in seen,'Nonregular/duplicate member';seen.add(m.name)
                with z.extractfile(m) as f:h=stream_sha(f)
                yield m.name,m.size,h


def prepare(root):
    wd=root/'work_dirs';out=wd/OUT
    out.mkdir(exist_ok=False)
    files=[];empty=[];summary=[];models=[];index=collections.defaultdict(list)
    for name in NAMES:
        base=wd/name
        if not base.is_dir():continue
        count=0;total=0;types=collections.defaultdict(lambda:[0,0])
        for d,dirs,names in os.walk(base,followlinks=False):
            p=Path(d)
            if p==out:dirs[:]=[];continue
            if not dirs and not names:empty.append(str(p.relative_to(wd)))
            for n in names:
                p=Path(d)/n
                if p.is_symlink():continue
                size=p.stat().st_size;count+=1;total+=size
                types[p.suffix][0]+=1;types[p.suffix][1]+=size
                files.append(p);index[(p.name,size)].append(p)
                if p.suffix in ('.pth','.pt','.npz','.pkl'):
                    models.append(dict(path=str(p.relative_to(wd)),bytes=size))
        summary.append(dict(name=name,files=count,bytes=total,types=dict(types)))
    active=wd/ACTIVE
    config_guards={str(p.relative_to(wd)):sha(p) for p in
        [active/'protocol.json',active/'runtime_identity_gpus01.json',active/'configs/Coff_seed0.py',
         wd/'symeood_c_pair_seed0_20261025_v1/native_gate/native_runtime_lock.json',
         wd/'symeood_c_pair_seed0_20261025_v2/protocol.json']}
    save(out/'inventory.json',dict(inactive_roots=summary,protected_active_root=ACTIVE,
        protected_binary_files=models,config_guards=config_guards))
    cache={};deletions=[];preserved=[]
    def digest(p):
        if p not in cache:cache[p]=sha(p)
        return cache[p]
    for path in files:
        relative=str(path.relative_to(wd))
        if path.suffix=='.pyc':
            try:source=Path(importlib.util.source_from_cache(str(path)))
            except ValueError:continue
            if source.is_file():deletions.append(dict(path=relative,bytes=path.stat().st_size,sha256=digest(path),
                reason='regenerable_bytecode_source_exists',retained=[dict(path=str(source.relative_to(wd)),sha256=digest(source))]))
            continue
        if not path.name.endswith(('.zip','.tar.gz','.tgz','.tar')):continue
        retained=[];missing=[];before=(path.stat().st_size,path.stat().st_mtime_ns)
        try:
            for name,size,h in archive_files(path):
                member=member_name(name);candidates=[]
                for parent in (path.parent,path.parent/path.stem,path.parent/'snapshot',path.parent/'last4_snapshot',wd,root):
                    candidate=parent/str(member)
                    if candidate.is_file() and not candidate.is_symlink() and candidate!=path and candidate.stat().st_size==size:
                        candidates.append(candidate)
                candidates.extend(index.get((member.name,size),[]))
                match=next((p for p in dict.fromkeys(candidates) if p!=path and not p.name.endswith(('.zip','.tar.gz','.tgz','.tar')) and digest(p)==h),None)
                if match is None:missing.append(dict(member=name,bytes=size,sha256=h))
                else:retained.append(dict(member=name,path=str(match.relative_to(root)),sha256=h))
            assert before==(path.stat().st_size,path.stat().st_mtime_ns),'Archive changed during audit'
            if retained and not missing:
                deletions.append(dict(path=relative,bytes=before[0],sha256=digest(path),reason='archive_all_members_retained_by_sha256',retained=retained))
            else:preserved.append(dict(path=relative,bytes=before[0],why='Unmatched/unique archive content',unmatched_count=len(missing),unmatched_members=missing[:20]))
        except (AssertionError,ValueError,tarfile.TarError,zipfile.BadZipFile) as e:
            preserved.append(dict(path=relative,bytes=before[0],why=str(e)))
        print('ARCHIVE_REVIEWED',relative,'delete' if any(x['path']==relative for x in deletions) else 'keep',flush=True)
    plan=dict(scope='Only inactive listed roots; active run and all weights/raw tensors/results protected',
        delete_files=deletions,delete_empty_dirs=empty,preserve_archives=preserved,
        planned_bytes=sum(x['bytes'] for x in deletions),guard_inventory_sha256=sha(out/'inventory.json'))
    save(out/'plan.json',plan)
    print(json.dumps(dict(files=len(deletions),bytes=plan['planned_bytes'],archives_preserved=len(preserved),empty_dirs=len(empty))),flush=True)


def execute(root):
    wd=root/'work_dirs';out=wd/OUT;plan=json.loads((out/'plan.json').read_text());inventory=json.loads((out/'inventory.json').read_text())
    assert sha(out/'inventory.json')==plan['guard_inventory_sha256']
    assert not (out/'deletion_claim.json').exists()
    save(out/'deletion_claim.json',dict(pid=os.getpid(),exact_plan_sha256=sha(out/'plan.json')))
    for p,h in inventory['config_guards'].items():assert sha(wd/p)==h,('Protected config drift',p)
    # Revalidate the complete plan before removing any file.
    cache={}
    for item in plan['delete_files']:
        p=allowed(root,item['path'])
        assert p.name.endswith(('.zip','.tar.gz','.tgz','.tar','.pyc'))
        assert p.is_file() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
        for proof in item['retained']:
            retained=(wd/proof['path']) if item['reason']=='regenerable_bytecode_source_exists' else root/proof['path']
            assert retained.is_file() and not retained.is_symlink()
            if retained not in cache:cache[retained]=sha(retained)
            assert cache[retained]==proof['sha256'],('Preservation failed',retained)
    removed=[]
    for item in plan['delete_files']:
        p=allowed(root,item['path']);assert sha(p)==item['sha256'];p.unlink()
        removed.append({k:item[k] for k in ('path','bytes','sha256','reason')})
    dirs=[]
    candidates=set(plan['delete_empty_dirs'])
    candidates.update(str((wd/x['path']).parent.relative_to(wd)) for x in plan['delete_files'] if x['reason']=='regenerable_bytecode_source_exists')
    for relative in sorted(candidates,key=lambda x:-len(Path(x).parts)):
        p=allowed(root,relative)
        if p.is_dir() and not any(p.iterdir()):p.rmdir();dirs.append(relative)
    for x in inventory['protected_binary_files']:
        p=wd/x['path'];assert p.is_file() and p.stat().st_size==x['bytes'],('Protected binary changed',p)
    for p,h in inventory['config_guards'].items():assert sha(wd/p)==h
    save(out/'cleanup_receipt.json',dict(passed=True,removed=removed,removed_empty_dirs=dirs,
        bytes_removed=sum(x['bytes'] for x in removed),all_recorded_weights_raw_arrays_predictions_preserved=True,
        protected_binary_count=len(inventory['protected_binary_files']),active_run_untouched=ACTIVE,
        preserved_archives=plan['preserve_archives'],plan_sha256=sha(out/'plan.json')))
    print(json.dumps(dict(passed=True,files_removed=len(removed),empty_dirs_removed=len(dirs),bytes_removed=sum(x['bytes'] for x in removed))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','execute'],required=True)
    p.add_argument('--pass-number',type=int,choices=[1,2],default=1);a=p.parse_args()
    if a.pass_number==2:OUT='symeood_ablation_v1/cleanup_review_20261026_v2'
    if a.mode=='prepare':prepare(ROOT)
    else:execute(ROOT)
