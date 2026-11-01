"""Verify this archive; explicitly restore original paths only with --restore."""
import argparse
import hashlib
import json
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parent
ROOT = ARCHIVE.parents[1]


def check(restore=False):
    manifest = json.loads((ARCHIVE/'manifest.json').read_text())
    moved = []
    for row in manifest['records']:
        source = ROOT/row['archived']
        target = ROOT/row['original']
        if ARCHIVE not in source.resolve().parents or ROOT not in target.resolve().parents:
            raise ValueError('Unsafe archive path')
        if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Archive SHA differs: ' + row['archived'])
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Existing different file is never overwritten: ' + row['original'])
        if row['action'] == 'moved':
            moved.append((source, target))
    restored = []
    if restore:
        for row in manifest['records']:
            source, target = ROOT/row['archived'], ROOT/row['original']
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(source.read_bytes())
                restored.append(row['original'])
    return dict(passed=True, files_verified=len(manifest['records']),
                archived_configs=len(moved), restore_requested=restore, restored=restored)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore', action='store_true')
    print(json.dumps(check(parser.parse_args().restore), indent=2))
