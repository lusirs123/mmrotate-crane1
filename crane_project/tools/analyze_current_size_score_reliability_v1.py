"""Describe size correctness ranked by the inherited detector score.

Run from the repository root with python -m. The default input is the sealed
1440-frame TEST for R4 ep16 + M20 + dual-branch LVSR ep5. This diagnostic
reports every score boundary, without selecting an operating threshold.
"""
import argparse
from collections import defaultdict
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path

from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import (
    rate,
    row_metrics,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    'work_dirs/symeood_r4_adaptation_20261031/readbacks/lvsr_fixed_test/'
    'lvsr/test/attempt_002/test_rows.jsonl'
)
SOURCE_SHA = '9a11d0b625f0a05e45d7f7bacd7e41ff60def9bcd1f31561d30bd412b64ad240'
METHODS = {'SymEOOD': 'b', 'SymEOOD+M20+LVSR-ep5': 'edge_residual'}


def summarize_group(rows):
    """Compare methods on identical accepted frame sets, preserving score ties."""
    ranked = sorted((row for row in rows if row[0] is not None),
                    key=lambda row: row[0], reverse=True)
    total_correct = [sum(row[1][i] for row in ranked) for i in range(len(METHODS))]
    accepted = 0
    correct = [0] * len(METHODS)
    curve = []
    for score, tied in groupby(ranked, key=lambda row: row[0]):
        for _, labels in tied:
            accepted += 1
            for i, label in enumerate(labels):
                correct[i] += label
        curve.append(dict(
            score_min=score,
            output_retention=rate(accepted, len(ranked)),
            all_frame_acceptance=rate(accepted, len(rows)),
            methods={
                name: dict(
                    error_among_accepted=rate(accepted - correct[i], accepted),
                    correct_output_retention=rate(correct[i], total_correct[i]),
                    all_frame_correct_coverage=rate(correct[i], len(rows)),
                    correct_rejected=total_correct[i] - correct[i],
                )
                for i, name in enumerate(METHODS)
            },
        ))
    return dict(frames=len(rows), outputs=len(ranked), missing=len(rows) - len(ranked),
                accept_all=curve[-1] if curve else None, score_curve=curve)


def analyze(rows):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        image = row['image']
        if image in seen:
            raise ValueError(f'Duplicate frame: {image}')
        seen.add(image)
        predictions = [row[key] for key in METHODS.values()]
        if any((pred is None) != (predictions[0] is None) for pred in predictions):
            raise ValueError(f'Output identity differs between stages: {image}')
        score = None
        if predictions[0] is not None:
            for pred in predictions:
                if len(pred) != 6 or not math.isfinite(pred[5]):
                    raise ValueError(f'Invalid prediction score: {image}')
            score = predictions[0][5]
            if any(pred[5] != score for pred in predictions):
                raise ValueError(f'Detector score differs between stages: {image}')
        # Size correctness is independent of RIoU. Zero is a placeholder for
        # the unused RIoU argument, never an input to the ranking.
        labels = [row_metrics(row['gt'], pred, 0.)['joint_size10'] for pred in predictions]
        sample = (score, labels)
        groups['overall'].append(sample)
        groups[f"domain/{row['domain']}"].append(sample)
        groups[f"sequence/{row['sequence']}"].append(sample)
    if not seen:
        raise ValueError('No frames')
    return {key: summarize_group(samples) for key, samples in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New JSON report path')
    args = parser.parse_args()

    data = SOURCE.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCE_SHA:
        raise ValueError('Prediction file differs from the sealed TEST source')
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if len(rows) != 1440:
        raise ValueError('Expected the full 1440-frame TEST')
    report = dict(
        model='R4 ep16 + Midpoint sigma1.5 ep20 + dual-branch LVSR ep5',
        source=str(SOURCE.relative_to(ROOT)), source_sha256=digest,
        protocol='Descriptive fixed TEST; no fitting or threshold selection',
        label='Both absolute relative side errors <= 0.10 on canonical long/short sides',
        groups=analyze(rows),
    )
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
