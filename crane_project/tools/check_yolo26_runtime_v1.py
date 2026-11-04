"""CPU-only official YOLO architecture, transfer, and TRAIN/VAL reader checks.

No trained weights, CUDA, optimizer, backwards, training or TEST-image access.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.prepare_external_obb_v1 import (
    ROOT, YOLO_CONFIG, check_configs, check_sources, read, write_new)


def official_yolo():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(tempfile.gettempdir()) / 'crane_yolo26_cpu_settings'))
    sys.path.insert(0, str(ROOT / 'third_party/ultralytics'))
    import ultralytics
    if Path(ultralytics.__file__).resolve().parent != ROOT / 'third_party/ultralytics/ultralytics':
        raise RuntimeError('Wrong Ultralytics package imported')
    return ultralytics


def check_runtime():
    source = check_sources()
    check_configs()
    ultralytics = official_yolo()
    from ultralytics import YOLO
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.nn.tasks import OBBModel
    from ultralytics.utils import YAML
    from crane_project.utils.yolo26_imagenet_v1 import backbone_source
    import torch
    torch.set_num_threads(2)
    spec = read(ROOT / YOLO_CONFIG)
    native = YAML.load(str(ROOT / 'configs/yolo26/crane_yolo26m_obb_train_v1.yaml'))
    for key, value in spec['train_args'].items():
        expected = str(ROOT / spec['initialization']['checkpoint']) if key == 'pretrained' else value
        observed = str(ROOT / native[key]) if key == 'pretrained' else native[key]
        if observed != expected:
            raise ValueError('Native YOLO training YAML differs from protocol: ' + key)
    torch.manual_seed(0)
    model = YOLO(str(ROOT / spec['architecture']), task='obb', verbose=False)
    head = model.model.model[-1]
    if head.__class__.__name__ != 'OBB26' or head.nc != 1 or not hasattr(head, 'one2one_cv2'):
        raise ValueError('Not the official single-class end-to-end OBB26 model')
    # This pinned release defaults to the one-to-many inference branch until
    # nms=False selects one-to-one. This is an official head-mode property.
    model.model.end2end = True
    # A randomly initialized classifier proves the official transfer mechanics,
    # not ImageNet weight provenance or pretrained-model performance.
    torch.manual_seed(1)
    classifier = YOLO(str(ROOT / 'third_party/ultralytics/ultralytics/cfg/models/26/yolo26m-cls.yaml'),
                      task='classify', verbose=False)
    source_model = backbone_source(classifier.model, model.model)
    src, dst = source_model.state_dict(), model.model.state_dict()
    shared = {k for k in src if k in dst and src[k].shape == dst[k].shape}
    if not shared or any(int(k.split('.')[1]) >= 9 for k in shared):
        raise ValueError('Unexpected classification/detection transfer outside shared blocks 0..8')
    head_before = {k: v.clone() for k, v in dst.items() if int(k.split('.')[1]) >= 9}
    model.load(source_model)
    transferred = model.model.state_dict()
    if not all(torch.equal(transferred[k], src[k]) for k in shared):
        raise ValueError('Official transfer did not preserve shared tensors')
    if not all(torch.equal(transferred[k], v) for k, v in head_before.items()):
        raise ValueError('Classifier weights changed the OBB-specific layers')
    # Native OBBTrainer.get_model uses OBBModel(...).load(weights), including
    # when native DDP rebuilds the model. Exercise that operation without a
    # trainer/optimizer. Keep pretrained=True or a local path during training.
    rebuilt = OBBModel(str(ROOT / spec['architecture']), nc=1, verbose=False)
    rebuilt.load(model.model, verbose=False)
    if not all(torch.equal(rebuilt.state_dict()[k], v) for k, v in transferred.items()):
        raise ValueError('Native trainer model rebuild loses transferred tensors')
    for p in model.model.parameters():
        p.requires_grad_(False)
    model.model.eval()
    with torch.inference_mode():
        output = model.model(torch.zeros(1, 3, 128, 128))
    tensor = output[0] if isinstance(output, tuple) else output
    if not torch.isfinite(tensor).all():
        raise ValueError('Nonfinite CPU forward')
    data_yaml = ROOT / spec['data']
    data = YAML.load(str(data_yaml))
    if 'test' in data:
        raise ValueError('TEST must not be present in training YAML')
    reader = {}
    with tempfile.TemporaryDirectory(prefix='crane_yolo_reader_') as folder:
        from ultralytics.nn.tasks import load_checkpoint
        source_file = Path(folder) / 'imagenet_backbone_synthetic.pt'
        torch.save(dict(model=source_model, epoch=-1, optimizer=None,
                        train_args=dict(task='classify')), source_file)
        restored, _ = load_checkpoint(str(source_file), device='cpu', fuse=False)
        if not all(torch.equal(restored.state_dict()[k], v) for k, v in src.items()):
            raise ValueError('Native checkpoint loader loses backbone initialization')
        for split in ('train', 'val'):
            images = (data_yaml.parent / data[split]).resolve()
            image = next(iter(sorted(images.glob('*.jpg'))))
            # Keep single-image scan caches inside the temporary directory.
            sample = Path(folder) / split / 'images' / image.name
            sample.parent.mkdir(parents=True)
            sample.symlink_to(image)
            label = Path(folder) / split / 'labels' / (image.stem + '.txt')
            label.parent.mkdir()
            label.write_bytes((images.parent / 'labels' / label.name).read_bytes())
            ds = YOLODataset(img_path=str(sample.parent), imgsz=1024, batch_size=1,
                             augment=False, data=data, task='obb', cache=False)
            batch = ds[0]
            if tuple(batch['img'].shape) != (3, 1024, 1024) or batch['bboxes'].shape != (1, 5):
                raise ValueError('Native OBB image/label reader differs')
            if not (batch['cls'] == 0).all() or not torch.isfinite(batch['bboxes']).all():
                raise ValueError('Native OBB labels differ')
            reader[split] = dict(frame=image.stem, image_shape=list(batch['img'].shape),
                                 bbox_shape=list(batch['bboxes'].shape))
    parameters = sum(p.numel() for p in model.model.parameters())
    named = dict(model.model.named_parameters())
    transferred_parameters = sum(named[k].numel() for k in shared if k in named)
    return dict(passed=True, sources=source, package_version=ultralytics.__version__,
                device='cpu', official_obb_head='OBB26', end2end=True,
                classes=1, model_parameters=parameters,
                classifier_transfer=dict(shared_state_tensors=len(shared),
                    transferred_parameters=transferred_parameters,
                    parameter_coverage=transferred_parameters / parameters,
                    blocks='0..8', obb_layers_unchanged=True,
                    native_trainer_rebuild_preserves_weights=True,
                    native_initialization_checkpoint_roundtrip=True,
                    real_pretrained_checkpoint_verified=False),
                synthetic_forward_shape=list(tensor.shape), native_readers=reader,
                training_updates=0, backward_calls=0, TEST_images_read=0,
                GPU_runtime_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = check_runtime()
    if args.out:
        write_new(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
