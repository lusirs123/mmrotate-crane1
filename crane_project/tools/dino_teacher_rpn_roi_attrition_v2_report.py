#!/usr/bin/env python3
"""Create a corrected RPN-to-ROI attrition report without GPU inference."""

import argparse
import json
import os

from crane_project.tools import dino_teacher_common as common
from crane_project.tools import (
    dino_teacher_rpn_roi_attrition_latency_audit as audit)


AUDIT_NAME = 'DINO RPN-to-ROI Attrition and Latency Audit V2 Report'
PROTOCOL_VERSION = 2


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument('--input-json', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def corrected_rows(rows):
    corrected = []
    for source_row in rows:
        row = dict(source_row)
        objects = []
        for source_object in source_row['objects']:
            obj = dict(source_object)
            obj['raw_attrition_cause'] = obj.get('attrition_cause')
            obj['rpn_initial_miss_recovered'] = bool(
                obj['rpn']['best_usable_rank'] is None
                and obj['roi_regression']['decoded_usable_count'] > 0)
            obj['attrition_cause'] = audit.attrition_cause_from_object(obj)
            objects.append(obj)
        row['objects'] = objects
        corrected.append(row)
    return corrected


def corrected_group(group):
    rows = corrected_rows(group['rows'])
    summary = audit.summarize_attrition(rows)
    return dict(
        diagnosis=audit.diagnose(summary),
        summary=summary,
        latency=group.get('latency', audit.summarize_latency(rows)),
        peak_memory_mib=group.get('peak_memory_mib'),
        rows=rows)


def main():
    args = parse_args()
    if not os.path.isfile(args.input_json):
        raise ValueError('Input JSON does not exist: {}'.format(
            args.input_json))
    if os.path.exists(args.out_json):
        raise ValueError('Refusing to overwrite: {}'.format(args.out_json))
    with open(args.input_json, 'r', encoding='utf-8') as handle:
        source = json.load(handle)
    if source.get('audit') != audit.AUDIT_NAME:
        raise ValueError('Input is not the V1 attrition audit')
    source_control = corrected_group(source['source_roi_control'])
    targets = {
        name: corrected_group(group)
        for name, group in source['target_diagnoses'].items()}
    payload = dict(
        audit=AUDIT_NAME,
        protocol_version=PROTOCOL_VERSION,
        source_audit_json=os.path.abspath(args.input_json),
        source_audit_sha256=common.file_sha256(args.input_json),
        protocol=dict(
            correction=(
                'Terminal failure is classified after ROI recovery; an '
                'initial RPN IoU miss that decodes to a usable box is not '
                'counted as an RPN terminal failure.'),
            source_audit_protocol_version=source.get('protocol_version'),
            target_role=source.get('protocol', {}).get('target_role'),
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            riou_thr=source.get('protocol', {}).get('riou_thr'),
            reconstruction_atol=source.get(
                'protocol', {}).get('reconstruction_atol')),
        isolation=source.get('isolation', {}),
        parameter_counts=source.get('parameter_counts', {}),
        source_roi_control=source_control,
        target_diagnoses=targets)
    replacements = common.write_json_atomic(args.out_json, payload)
    print('[v2] source {}'.format(source_control['diagnosis']))
    for name, result in targets.items():
        summary = result['summary']
        print('[v2] {} {} terminal_failures={} causes={}'.format(
            name, result['diagnosis'], summary['terminal_failure_count'],
            summary['terminal_failure_causes']))
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
