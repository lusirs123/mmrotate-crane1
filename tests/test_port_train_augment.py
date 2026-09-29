"""Geometry and exact experiment-isolation checks (CPU, no CUDA model needed)."""
import importlib.util
from pathlib import Path
import numpy as np
from mmcv import Config

ROOT = Path(__file__).resolve().parents[1]
import importlib
mod = importlib.import_module('mmrotate.datasets.pipelines.port_train_augment')


def sample():
    return dict(img=np.full((101, 203, 3), 120, dtype=np.uint8),
                img_shape=(101, 203, 3), ori_shape=(202, 406, 3),
                scale_factor=np.full(4, .5, dtype=np.float32),
                bbox_fields=['gt_bboxes', 'gt_bboxes_ignore'],
                gt_bboxes=np.array([[80, 50, 30, 12, .3]], dtype=np.float32),
                gt_bboxes_ignore=np.empty((0, 5), dtype=np.float32))


def test_shrink_roundtrip_and_pixel_location():
    data=sample(); original=data['gt_bboxes'].copy()
    data['img'][40, 80]=255
    result=mod.PortIsotropicShrink(prob=1, scale_range=(.5,.5))(data)
    assert result['img'].shape == (51,102,3)
    np.testing.assert_allclose(result['gt_bboxes'][:,:4]/.5, original[:,:4])
    np.testing.assert_allclose(result['gt_bboxes'][:,4], original[:,4])
    np.testing.assert_allclose(result['scale_factor'], .25)
    assert (result['img'][20,40]==255).all()
    assert result['gt_bboxes_ignore'].shape==(0,5)
    assert result['ori_shape']==(202,406,3)


def test_skip_is_identity():
    data=sample(); before=data['img'].copy()
    result=mod.PortIsotropicShrink(prob=0)(data)
    np.testing.assert_array_equal(result['img'],before)
    np.testing.assert_allclose(result['scale_factor'], .5)


def test_photo_preserves_geometry_and_darkens():
    data=sample(); boxes=data['gt_bboxes'].copy(); scale=data['scale_factor'].copy()
    result=mod.PortPhotometricAug(prob=1,gamma_range=(2,2),gain_range=(1,1),contrast_range=(1,1))(data)
    assert result['img'].dtype==np.uint8 and result['img'].mean()<120
    np.testing.assert_array_equal(result['gt_bboxes'],boxes)
    np.testing.assert_array_equal(result['scale_factor'],scale)


def test_resolved_configs_only_intended_differences():
    base=Config.fromfile(str(ROOT/'crane_project/configs/crane_symeood_k1_port_day2night_v1.py')).to_dict()
    for arm in ['b','c']:
        cfg=Config.fromfile(str(ROOT/f'crane_project/configs/crane_symeood_k1_port_day2night_aug_{arm}_v1.py')).to_dict()
        for i,entry in enumerate(cfg['data']['train']):
            if arm=='b' or i==0:
                aug=entry['pipeline'].pop(3)
                assert aug['type']==('PortIsotropicShrink' if arm=='b' else 'PortPhotometricAug')
            assert entry['pipeline']==base['data']['train'][i]['pipeline']
        cfg['custom_imports']['imports'].remove('mmrotate.datasets.pipelines.port_train_augment')
        cfg['work_dir']=base['work_dir']
        assert cfg==base


def test_eood_scale_control_matches_symeood_b():
    folder=ROOT/'crane_project/configs'
    base=Config.fromfile(str(folder/'crane_eood_k1_port_day2night_v1.py')).to_dict()
    cfg=Config.fromfile(str(folder/'crane_eood_k1_port_day2night_aug_b_v1.py')).to_dict()
    sym=Config.fromfile(str(folder/'crane_symeood_k1_port_day2night_aug_b_v1.py')).to_dict()
    for key in ['data','optimizer','optimizer_config','lr_config','runner',
                'load_from','resume_from','evaluation','checkpoint_config']:
        assert cfg.get(key)==sym.get(key),key
    for i,entry in enumerate(cfg['data']['train']):
        aug=entry['pipeline'].pop(3)
        assert aug==dict(type='PortIsotropicShrink',prob=.5,scale_range=(.5,1.))
    cfg['custom_imports']['imports'].remove('mmrotate.datasets.pipelines.port_train_augment')
    cfg['work_dir']=base['work_dir']
    assert cfg['checkpoint_config']['max_keep_ckpts']==24
    cfg['checkpoint_config']['max_keep_ckpts']=base['checkpoint_config']['max_keep_ckpts']
    assert cfg==base
