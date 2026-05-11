import torch
import sys
sys.path.append('.')
from mmcv import Config
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmcv.runner import load_checkpoint
import mmcv

cfg = Config.fromfile('crane_project/configs/crane_baseline.py')
model = build_detector(cfg.model)
load_checkpoint(model, 'work_dirs/crane_baseline/epoch_24.pth,
                map_location='cpu')
model.eval()

# 钩取 bbox_head 的原始分类输出
raw_scores = []
def hook_fn(module, input, output):
    raw_scores.append(output.sigmoid().detach())

model.bbox_head.retina_cls.register_forward_hook(hook_fn)

# 分别测试一帧 train 图像和一帧 test 空载图像
ds_train = build_dataset(cfg.data.train[0])
ds_test  = build_dataset(cfg.data.test)

for label, ds, idx in [('TRAIN', ds_train, 0), ('TEST空载', ds_test, 14)]:
    raw_scores.clear()
    item = ds[idx]
    img = item['img']
    if isinstance(img, list): img = img[0]
    if hasattr(img, 'data'): img = img.data
    
    with torch.no_grad():
        model.forward_dummy(img.unsqueeze(0))
    
    all_scores = torch.cat([s.flatten() for s in raw_scores])
    print(f'\n[{label}]')
    print(f'  sigmoid 输出范围: [{all_scores.min():.6f}, {all_scores.max():.6f}]')
    print(f'  >0.05 的比例: {(all_scores > 0.05).float().mean():.6f}')
    print(f'  >0.001 的比例: {(all_scores > 0.001).float().mean():.6f}')
    print(f'  >0.0001 的比例: {(all_scores > 0.0001).float().mean():.6f}')
    print(f'  最大值: {all_scores.max():.8f}')