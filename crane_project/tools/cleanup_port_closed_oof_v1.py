#!/usr/bin/env python3
"""Prepare exact hashes/backups, then unlink only authorized abandoned OOF files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.utils import port_closed_oof_cleanup_v1 as core


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','execute'],required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve()
    base=(ROOT/'work_dirs/port_reliability_temporal_evidence_v1').resolve()
    if base not in out.parents or a.out.is_symlink():raise ValueError('Keep cleanup receipt under this new temporal run')
    if a.mode=='prepare':
        out.mkdir(parents=True,exist_ok=False);manifest=core.plan(ROOT);receipts={};eligible=[];preserved=[]
        for item in manifest['files']:
            path=core.allowed(ROOT,item['path'])
            if item['reason']=='archive_preserved_extracted':
                backup=out/'preserved_archive_members'/path.name;backup.mkdir(parents=True)
                members=[]
                with tarfile.open(path,'r:gz') as archive:
                    seen=set()
                    for m in archive.getmembers():
                        name=PurePosixPath(m.name)
                        if name.is_absolute() or '..' in name.parts or m.issym() or m.islnk():
                            raise ValueError('Unsafe archive member '+m.name)
                        if m.isdir():continue
                        if not m.isfile() or m.name in seen:raise ValueError('Nonregular/duplicate archive member')
                        seen.add(m.name);dest=backup/str(name);dest.parent.mkdir(parents=True,exist_ok=True)
                        content=archive.extractfile(m).read()
                        with dest.open('xb') as f:f.write(content)
                        h=hashlib.sha256(content).hexdigest()
                        if core.sha(dest)!=h:raise ValueError('Archive backup differs')
                        members.append(dict(original_member=m.name,saved_path=str(dest.relative_to(ROOT)),sha256=h,bytes=len(content)))
                receipts[item['path']]=dict(archive_sha256=item['sha256'],members_verified=True,members=members)
            elif item['reason']=='redundant_sync_bundle':
                lines=subprocess.check_output(['git','bundle','list-heads',str(path)],cwd=ROOT,text=True).splitlines()
                hashes=[line.split()[0] for line in lines];present=bool(hashes)
                for h in hashes:
                    present=present and subprocess.run(['git','cat-file','-e',h+'^{commit}'],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
                if not present:
                    preserved.append(dict(**item,why_preserved='bundle commits not all verified in current Git object database'));continue
                receipts[item['path']]=dict(archive_sha256=item['sha256'],git_objects_present=True,commits=hashes)
            eligible.append(item)
        manifest['files']=eligible;manifest['planned_bytes']=sum(v['bytes'] for v in eligible)
        manifest['unverified_bundles_preserved']=preserved
        write(out/'manifest.json',manifest);write(out/'preservation_receipts.json',receipts)
        print(json.dumps(dict(phase='prepared_no_deletion',files=len(eligible),bytes=manifest['planned_bytes'],preserved_bundles=len(preserved)),sort_keys=True))
    else:
        manifest=json.loads((out/'manifest.json').read_text());receipts=json.loads((out/'preservation_receipts.json').read_text())
        result=core.execute(ROOT,manifest,receipts)
        result['manifest_sha256']=core.sha(out/'manifest.json')
        write(out/'cleanup_receipt.json',result)
        print(json.dumps(dict(phase='deleted_exact_manifest',files=len(result['removed']),bytes=result['bytes_removed']),sort_keys=True))


if __name__=='__main__':main()
