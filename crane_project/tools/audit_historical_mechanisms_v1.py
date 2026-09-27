"""Inventory historical assignment, classification and quality-guidance evidence.

This is a read-only evidence inventory.  It does not build a model, run
inference, choose a checkpoint, or use TEST metrics to select a direction.
It is intended to run in the full server checkout, where work_dirs and logs
may contain experiments that are absent from the local source checkout.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


TEXT_SUFFIXES = {
    '.json', '.jsonl', '.log', '.txt', '.md', '.py', '.yaml', '.yml', '.csv',
}
SKIP_DIRS = {
    '.git', '__pycache__', 'images', 'image', 'feature_cache', 'checkpoints',
    'checkpoint', 'preds', 'vis', 'visualizations',
}
MAX_BYTES = 12 * 1024 * 1024
SNIPPET_LIMIT = 8

MECHANISMS = {
    'assignment': {
        'ATSS': r'\bATSS\b|ATSSAssigner|atss',
        'PAA': r'\bPAA\b|PAAAssigner|paa',
        'SimOTA': r'SimOTA|sim_ota|simota',
        'MaxIoU': r'MaxIoU|MaxConvexIoU|IoUAssigner|max_iou',
        'quality_lower_bound': (
            r'quality.{0,30}(?:lower|floor|bound|threshold)|'
            r'(?:riou|iou).{0,20}(?:>|>=|threshold|minimum|min_)'),
        'SymPOLA': r'SymPOLAAssigner|sym_pola|POLA',
    },
    'classification': {
        'Focal': r'FocalLoss|focal_loss|focal',
        'QFL': r'QualityFocal|quality_focal|\bQFL\b',
        'VFL': r'Varifocal|varifocal|\bVFL\b',
        'SymNFL': r'SymNFLLoss|sym_nfl|SymNFL',
        'quality_target': r'quality.{0,30}(?:target|label|score)',
    },
    'quality_guidance': {
        'IoU_aware': r'IoU.aware|iou_aware|iouaware|IoUAware',
        'centerness': r'centerness|CenterNess|center_ness',
        'RegQuality': r'RegQuality|reg_quality|quality_head',
        'PQA': r'\bPQA\b|pqa_head|pqa_',
        'score_modulation': r'score_context_modulation|score_modulation|quality.*score',
        'DINO_S7': r'\bS7\b|s7_|highres.*RPN|RPN.*ROI',
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def iter_evidence_files(root):
    for base, dirs, files in os.walk(str(root)):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        normalized_base = base.replace('\\', '/')
        if '/data/' in normalized_base or normalized_base.endswith('/data'):
            dirs[:] = []
            continue
        for name in files:
            path = Path(base) / name
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                if path.stat().st_size > MAX_BYTES:
                    continue
            except OSError:
                continue
            yield path


def read_lines(path):
    try:
        return path.read_text(encoding='utf-8', errors='replace').splitlines()
    except (OSError, UnicodeError):
        return []


def is_test_path(path):
    return bool(re.search(r'(?:^|[/_.-])test(?:$|[/_.-])', str(path), re.I))


def evidence_class(path):
    value = str(path).replace('\\', '/')
    if '/tests/' in '/' + value or value.startswith('tests/'):
        return 'test_or_reference'
    if (value.startswith('crane_project/configs/')
            or value.startswith('mmrotate/')):
        return 'implementation_or_config'
    if (value.startswith('work_dirs/') or value.startswith('logs/')
            or value.startswith('docs/') or '/work_dirs/' in value):
        return 'historical_record'
    return 'other_text'


def collect_matches(root, files):
    matches = {group: {name: [] for name in rules}
               for group, rules in MECHANISMS.items()}
    class_counts = {group: {name: {} for name in rules}
                    for group, rules in MECHANISMS.items()}
    file_count = 0
    for path in files:
        file_count += 1
        lines = read_lines(path)
        if not lines:
            continue
        rel = str(path.relative_to(root))
        for group, rules in MECHANISMS.items():
            for name, pattern in rules.items():
                compiled = re.compile(pattern, re.I)
                for line_no, line in enumerate(lines, 1):
                    if not compiled.search(line):
                        continue
                    category = evidence_class(Path(rel))
                    counts = class_counts[group][name]
                    counts[category] = counts.get(category, 0) + 1
                    if len(matches[group][name]) >= SNIPPET_LIMIT:
                        break
                    matches[group][name].append({
                        'path': rel,
                        'line': line_no,
                        'evidence_class': category,
                        'test_artifact_path': is_test_path(path),
                        'text': line.strip()[:300],
                    })
    return file_count, matches, class_counts


def current_contract(root):
    config = root / 'crane_project/configs/crane_symeood_k1.py'
    lines = read_lines(config)
    text = '\n'.join(lines)
    return {
        'config': str(config.relative_to(root)) if config.exists() else None,
        'config_sha256': sha256(config) if config.exists() else None,
        'assigner_mentions': sorted(set(re.findall(
            r"type\s*=\s*['\"]([^'\"]*Assigner)['\"]", text))),
        'classification_loss_mentions': sorted(set(re.findall(
            r"loss_cls\s*=\s*dict\(\s*type\s*=\s*['\"]([^'\"]+)", text))),
        'test_cfg_present': 'test_cfg=dict(' in text,
        'test_cfg_score_thr': re.findall(r'score_thr\s*=\s*([0-9.]+)', text),
        'test_cfg_max_per_img': re.findall(r'max_per_img\s*=\s*(\d+)', text),
    }


def experiment_dirs(root):
    result = []
    for base, dirs, _files in os.walk(str(root)):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in dirs:
            low = name.lower()
            if any(token in low for token in ('s7', 'v1', 'v2', 'v3', 'v4', 'v5',
                                              'qfl', 'vfl', 'atss', 'paa', 'simota',
                                              'quality', 'centerness', 'iou')):
                path = Path(base) / name
                result.append({
                    'path': str(path.relative_to(root)),
                    'test_artifact_path': is_test_path(path),
                })
    return sorted(result, key=lambda x: x['path'])[:500]


def status_for(entries, class_counts=None):
    counts = class_counts or {}
    if not entries and not counts:
        return 'no_matching_evidence_in_scanned_checkout'
    if counts.get('historical_record', 0):
        return 'historical_record_present_review_experiment_identity'
    if counts.get('implementation_or_config', 0):
        return 'implementation_or_config_present'
    if counts.get('other_text', 0):
        return 'reference_text_only_or_unclassified'
    return 'test_or_reference_only'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = Path(args.out_json)
    if output.exists():
        raise FileExistsError(str(output))
    files = list(iter_evidence_files(root))
    file_count, matches, class_counts = collect_matches(root, files)
    mechanism_status = {
        group: {name: {
            'status': status_for(entries, class_counts[group][name]),
            'match_count_capped': len(entries),
            'evidence_class_counts': class_counts[group][name],
            'evidence': entries,
        } for name, entries in names.items()}
        for group, names in matches.items()
    }
    report = {
        'protocol': 'historical_mechanism_inventory_v1',
        'evidence_role': 'read_only_inventory',
        'selection_or_training_performed': False,
        'test_metrics_used_for_selection': False,
        'root': str(root),
        'scanned_text_file_count': file_count,
        'current_k1_contract': current_contract(root),
        'mechanisms': mechanism_status,
        'experiment_directories_capped': experiment_dirs(root),
        'limitations': [
            'A text match proves that a mechanism or name is present in a file, not that it was trained successfully.',
            'A matching experiment must be verified with its config, checkpoint, log and source-VAL result together.',
            'TEST-named artifacts are indexed for provenance only and are not used to choose a direction.',
            'No model is built and no inference is run by this script.',
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    compact = {
        group: {name: value['status'] for name, value in names.items()}
        for group, names in mechanism_status.items()
    }
    print(json.dumps({'protocol': report['protocol'],
                      'scanned_text_file_count': file_count,
                      'status': compact}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
