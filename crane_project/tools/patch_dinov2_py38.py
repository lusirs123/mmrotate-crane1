#!/usr/bin/env python3
"""Make current DINOv2 type annotations importable on Python 3.8/3.9.

Recent DINOv2 sources use PEP 604 annotations such as ``float | None``.
Python 3.8 parses that expression but evaluates it during class definition,
raising TypeError. Postponed annotation evaluation fixes the import without
changing model execution or checkpoint values.
"""

import argparse
import ast
import json
import os
import re
import sys
from typing import Dict, List, Tuple


FUTURE_IMPORT = 'from __future__ import annotations\n'
MODERN_ANNOTATION = re.compile(
    r'(?:'
    r'(?:^|[(:,]\s*)(?:[A-Za-z_][\w.]*|\[[^\n]+\])\s*\|\s*'
    r'(?:None|[A-Za-z_][\w.]*)'
    r'|\b(?:list|dict|tuple|set|frozenset|type)\s*\[[^\n]+\]'
    r')', re.MULTILINE)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Patch DINOv2 PEP 604 annotations for Python 3.8/3.9')
    parser.add_argument('--repo', required=True,
                        help='Local facebookresearch/dinov2 clone')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--out-json')
    return parser.parse_args()


def _insertion_line(source: str) -> int:
    """Return a zero-based line index valid for a future import."""
    lines = source.splitlines(keepends=True)
    header = 0
    if lines and lines[0].startswith('#!'):
        header = 1
    if header < len(lines) and re.match(
            r'^\s*#.*coding[:=]\s*[-\w.]+', lines[header]):
        header += 1
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return header
    if tree.body and isinstance(tree.body[0], ast.Expr):
        value = tree.body[0].value
        if (isinstance(value, ast.Constant)
                and isinstance(value.value, str)):
            return int(getattr(tree.body[0], 'end_lineno',
                               tree.body[0].lineno))
    return header


def patch_source(source: str) -> Tuple[str, bool]:
    if FUTURE_IMPORT.strip() in source:
        return source, False
    if MODERN_ANNOTATION.search(source) is None:
        return source, False
    lines = source.splitlines(keepends=True)
    index = _insertion_line(source)
    lines.insert(index, FUTURE_IMPORT)
    return ''.join(lines), True


def discover_python_files(repo: str) -> List[str]:
    package = os.path.join(os.path.abspath(repo), 'dinov2')
    hubconf = os.path.join(os.path.abspath(repo), 'hubconf.py')
    if not os.path.isdir(package) or not os.path.isfile(hubconf):
        raise RuntimeError(
            '--repo must contain hubconf.py and the dinov2 package')
    paths = [hubconf]
    for root, dirs, files in os.walk(package):
        dirs[:] = sorted(item for item in dirs if item != '__pycache__')
        paths.extend(os.path.join(root, name)
                     for name in sorted(files) if name.endswith('.py'))
    return paths


def patch_repo(repo: str, dry_run: bool = False) -> Dict:
    candidates = discover_python_files(repo)
    changed = []
    for path in candidates:
        with open(path, 'r', encoding='utf-8') as handle:
            source = handle.read()
        updated, needs_change = patch_source(source)
        if not needs_change:
            continue
        changed.append(os.path.relpath(path, os.path.abspath(repo)))
        if not dry_run:
            mode = os.stat(path).st_mode
            temporary = path + '.sym_py38_tmp'
            with open(temporary, 'w', encoding='utf-8', newline='') as handle:
                handle.write(updated)
            os.chmod(temporary, mode)
            os.replace(temporary, path)
    return dict(
        repo=os.path.abspath(repo), python='{}.{}.{}'.format(*sys.version_info[:3]),
        dry_run=bool(dry_run), scanned_files=len(candidates),
        changed_count=len(changed), changed_files=changed,
        operation='postpone_annotations_only')


def main():
    args = parse_args()
    result = patch_repo(args.repo, args.dry_run)
    print('[dinov2-py38] changed={}/{} dry_run={}'.format(
        result['changed_count'], result['scanned_files'], result['dry_run']))
    for path in result['changed_files']:
        print('[patched] {}'.format(path))
    if args.out_json:
        out_path = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
