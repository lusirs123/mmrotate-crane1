"""Convert a SHA-verified official ImageNet classifier to backbone init only.

No downloads, training, dataset access or GPU. Source weights remain untouched.
The detector itself is the unchanged official single-class OBB26 architecture.
"""
import argparse
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.prepare_external_obb_v1 import YOLO_CONFIG, check_configs, check_sources, read, sha, write_new
from crane_project.tools.check_yolo26_runtime_v1 import official_yolo
from crane_project.utils.yolo26_imagenet_v1 import backbone_source


def prepare(source_sha256):
    check_sources()
    check_configs()
    spec = read(ROOT / YOLO_CONFIG)
    source = ROOT / spec['initialization']['source_checkpoint']
    destination = ROOT / spec['initialization']['checkpoint']
    receipt = destination.with_suffix('.initialization.json')
    if len(source_sha256) != 64 or sha(source) != source_sha256:
        raise ValueError('Official classification checkpoint SHA differs')
    if destination.exists() or receipt.exists():
        raise FileExistsError('Initialization artifact already exists')
    official_yolo()
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    classifier = YOLO(str(source), task='classify', verbose=False)
    detector = YOLO(str(ROOT / spec['architecture']), task='obb', verbose=False)
    weights = backbone_source(classifier.model, detector.model).float().cpu()
    detector.load(weights)
    if not all(torch.equal(detector.model.state_dict()[k], v) for k, v in weights.state_dict().items()):
        raise ValueError('Native loader did not transfer the shared backbone')
    buffer = io.BytesIO()
    torch.save(dict(model=weights, epoch=-1, optimizer=None, train_args=dict(task='classify'),
                    crane_initialization=dict(source_sha256=source_sha256, blocks=list(range(9)),
                                              detection_pretraining=False)), buffer)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(buffer.getvalue())
    value = dict(protocol='yolo26m_imagenet_backbone_initialization_v1',
                 source=str(source.relative_to(ROOT)), source_sha256=source_sha256,
                 checkpoint=str(destination.relative_to(ROOT)), checkpoint_sha256=sha(destination),
                 architecture_sha256=sha(ROOT / spec['architecture']),
                 upstream_commit=read(ROOT / spec['upstream_manifest'])['commit'],
                 blocks=list(range(9)), state_tensors=len(weights.state_dict()),
                 training_updates=0, GPU_used=False, detection_pretraining=False)
    write_new(receipt, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_sha256), indent=2))


if __name__ == '__main__':
    main()
