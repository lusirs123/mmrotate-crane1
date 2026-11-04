"""CPU-only official GauCho model/head and TRAIN/VAL pipeline checks.

Requires the existing MMCV 1.x / MMDetection 2.x runtime. Never call init_weights,
download weights, create an optimizer, run backward, access TEST images or CUDA.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.prepare_external_obb_v1 import GAUCHO_CONFIG, check_configs, check_sources, write_new


def check_runtime():
    sources = check_sources()
    check_configs()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(ROOT))
    import mmrotate
    import torch
    from mmcv import Config
    from mmrotate.models import build_detector
    from mmrotate.datasets import build_dataset
    if Path(mmrotate.__file__).resolve().parent != ROOT / 'mmrotate':
        raise ValueError('Wrong mmrotate runtime')
    torch.set_num_threads(2)
    cfg = Config.fromfile(str(ROOT / GAUCHO_CONFIG))
    model = build_detector(cfg.model)
    model.eval()
    if model.bbox_head.__class__.__name__ != 'GaussianFCOSHead' or not model.bbox_head.gaucho_encoding:
        raise ValueError('Not official GauCho encoding/head')
    parent = model.bbox_head.__class__.__mro__[1]
    if (parent.__module__ != 'mmrotate.models.dense_heads.gaucho_rotated_fcos_head'
            or model.bbox_head.loss_bbox.__class__.__module__ != 'mmrotate.models.losses.gaucho_gaussian_dist_loss_v1'):
        raise ValueError('GauCho must use its official parent/loss rather than modified project namesakes')
    with torch.no_grad():
        outputs = model.bbox_head([torch.randn(1, 256, n, n) for n in (8, 4, 2, 1, 1)])
    if not all(torch.isfinite(t).all() for group in outputs for t in group):
        raise ValueError('Nonfinite GauCho head forward')
    reports = {}
    os.chdir(ROOT)
    for name, config, count in (('train_real', cfg.data.train[0], 1810),
                                ('train_sim', cfg.data.train[1], 748),
                                ('val', cfg.data.val, 887)):
        dataset = build_dataset(config)
        if len(dataset) != count or tuple(dataset.CLASSES) != ('grab',):
            raise ValueError('GauCho dataset identity differs')
        sample = dataset[0]
        image = sample['img']
        image = image[0] if isinstance(image, list) else image
        if tuple(image.data.shape) != (3, 1024, 1024):
            raise ValueError('GauCho pipeline shape differs')
        reports[name] = dict(frames=len(dataset), image_shape=list(image.data.shape))
    return dict(passed=True, sources=sources, device='cpu',
                package=str(Path(mmrotate.__file__).resolve()),
                head_source=inspect.getfile(model.bbox_head.__class__),
                parent_source=inspect.getfile(parent),
                loss_source=inspect.getfile(model.bbox_head.loss_bbox.__class__),
                classes=model.bbox_head.num_classes, gaucho_encoding=True,
                pipelines=reports, model_parameters=sum(p.numel() for p in model.parameters()),
                training_updates=0, backward_calls=0, weights_downloaded=False,
                TEST_images_read=0, GPU_runtime_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    value = check_runtime()
    if args.out:
        write_new(args.out, value)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
