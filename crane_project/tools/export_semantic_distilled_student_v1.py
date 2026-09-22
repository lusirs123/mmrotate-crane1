"""Export a student-only SymEOOD checkpoint from a distillation run."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


PROTOCOL = 'symeood_semantic_distilled_student_export_v1'
TRAINING_ONLY_PREFIXES = (
    'semantic_distillation.', 'module.semantic_distillation.')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def student_only_payload(payload):
    if not isinstance(payload, dict) or not isinstance(
            payload.get('state_dict'), dict):
        raise ValueError('checkpoint must contain a state_dict')
    state_dict = payload['state_dict']
    removed = sorted(key for key in state_dict if key.startswith(
        TRAINING_ONLY_PREFIXES))
    if not removed:
        raise ValueError('checkpoint contains no semantic distillation adapter')
    output = dict(payload)
    output['state_dict'] = {
        key: value for key, value in state_dict.items() if key not in removed}
    meta = dict(output.get('meta') or {})
    meta['student_only_export_protocol'] = PROTOCOL
    meta['training_only_parameters_removed'] = removed
    output['meta'] = meta
    # Inference artifacts do not carry optimizer moments, including moments
    # that belonged to the removed adapter.
    output.pop('optimizer', None)
    return output, removed


def write_exact_checkpoint(path, payload):
    path = Path(path)
    if path.exists():
        raise RuntimeError('Refusing to overwrite existing output: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + '.tmp')
    torch.save(payload, temporary)
    os.replace(str(temporary), str(path))


def write_exact_json(path, payload):
    path = Path(path)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') != encoded:
        raise RuntimeError('Refusing to overwrite different output: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-checkpoint', required=True)
    parser.add_argument('--out-checkpoint', required=True)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    source = os.path.abspath(args.input_checkpoint)
    output = os.path.abspath(args.out_checkpoint)
    payload, removed = student_only_payload(load_checkpoint(source))
    write_exact_checkpoint(output, payload)
    report = dict(
        protocol=PROTOCOL,
        source_checkpoint=dict(path=source, sha256=sha256(source)),
        student_checkpoint=dict(path=output, sha256=sha256(output)),
        inference_config=(
            'crane_project/configs/'
            'crane_symeood_k1_dino_semantic_student_v1.py'),
        removed_parameter_keys=removed,
        frozen_dino_required_at_inference=False,
        distillation_adapter_required_at_inference=False)
    write_exact_json(args.out_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
