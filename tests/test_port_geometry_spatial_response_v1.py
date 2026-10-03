"""Analytical spatial signals, coordinates, replay guards and execution scope."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from crane_project.tools import check_port_geometry_spatial_response_v1 as r
s = r.s


def meta():
    return dict(ori_shape=(1024, 1024, 3), img_shape=(1024, 1024, 3),
        pad_shape=(1024, 1024, 3), scale_factor=[1., 1., 1., 1.], flip=False)


def plane():
    y, x = torch.meshgrid(torch.arange(128.), torch.arange(128.))
    return (8*x+16*y)[None, None].repeat(1, 256, 1, 1)


def box():
    return torch.tensor([[64.25, 72.75, 24., 16., .2]])


def test_analytical_plane_translation_half_cell_derivatives_and_no_mutation():
    p3 = plane(); b = box(); before = b.clone(); original = p3.clone()
    response, points = s.shift_responses(p3, b, meta())
    pooled, _, _ = r.g.sample_local(p3, b, meta(), 'ordinary')
    assert pooled[0, 0].numpy() == pytest.approx((points[0, :, :, 0]+2*points[0, :, :, 1]).numpy(), abs=1e-4)
    for key in ('x_minus', 'x_plus'):
        assert response['shifts'][key]['difference_rms'] == pytest.approx(1., abs=5e-5)
    for key in ('y_minus', 'y_plus'):
        assert response['shifts'][key]['difference_rms'] == pytest.approx(2., abs=5e-5)
    assert response['central_difference']['x']['rms_per_model_px'] == pytest.approx(1., abs=5e-5)
    assert response['central_difference']['y']['rms_per_model_px'] == pytest.approx(2., abs=5e-5)
    assert torch.equal(b, before) and torch.equal(p3, original)
    assert pooled.grad_fn is None and not pooled.requires_grad


def test_zero_constant_and_channel_variation_are_not_normalized_to_false_signal():
    b = box(); p3 = torch.zeros(1, 256, 128, 128)
    response, _ = s.shift_responses(p3, b, meta())
    stats = s.summarize_feature(p3[:, :, :9, :9])
    assert stats['total_rms'] == stats['spatial_demeaned_rms'] == 0.
    assert stats['spatial_over_total'] is None
    assert response['shifts']['x_plus']['difference_over_base_rms'] is None
    p3 = p3+torch.arange(256.)[None, :, None, None]
    roi, _, _ = r.g.sample_local(p3, b, meta(), 'ordinary')
    stats = s.summarize_feature(roi)
    assert stats['total_rms'] > 0 and stats['spatial_demeaned_rms'] < 1e-4
    changed = roi.clone(); changed[:, 137, :, 4] += 10.
    assert s.summarize_feature(changed)['spatial_demeaned_rms'] > .1
    assert s.ratio(1., 1e-13) is None


def test_shift_units_and_raw_size_angle_association_at_odd_resize():
    geometry = r.q.e.view_geometry([1031, 721], .5)
    row = dict(geometry=geometry, original_image_wh=[1031, 721],
        gt_original=[100., 100., 20., 40., .3],
        arms={k:dict(pred=[100., 100., 20., 40., .3, .8]) for k in ('center_only', 'joint')})
    m = r.fixed_meta(row); b = r.g.map_boxes(torch.tensor([[100., 100., 20., 40., .3]]), m)
    values = r.model_boxes(row, m, b)
    sx, sy = geometry['scale_factor'][:2]
    assert values['b'][2:4] == pytest.approx([20*sx, 40*sy])
    assert values['gt'][2:4] == pytest.approx(np.array([20., 40.])*np.sqrt(sx*sy))
    assert values['b'][4] == pytest.approx(.3)
    response, _ = s.shift_responses(plane(), b, m)
    assert response['shifts']['x_plus']['displacement_original_px'] == pytest.approx([1/sx, 0])
    assert response['central_difference']['x']['rms_per_original_px'] == pytest.approx(sx, abs=1e-4)
    assert torch.allclose(r.g.map_boxes(b, m, inverse=True), torch.tensor([[100., 100., 20., 40., .3]]))


@pytest.mark.parametrize('mode', ['ordinary', 'aligned'])
def test_roi_index_coordinates_are_inverse_of_existing_sampler(mode):
    b = box()
    _, points = s.local.sampling_grid(b, (128, 128), mode)
    xy = np.array(s.roi_coordinates(points, b, mode))[0]
    yy, xx = np.meshgrid(np.arange(9), np.arange(9), indexing='ij')
    assert xy[..., 0] == pytest.approx(xx, abs=1e-5)
    assert xy[..., 1] == pytest.approx(yy, abs=1e-5)
    assert s.roi_coordinates(b[:, :2], b, mode)[0] == pytest.approx([4., 4.])


def test_native_patch_preserves_actual_cells_origin_mask_and_budget():
    p3 = plane(); _, points = s.local.sampling_grid(box(), (128, 128), 'ordinary')
    stats = s.native_patch(p3, points, box()[:, :2], meta())
    x0, y0, x1, y1 = stats['index_bounds_inclusive']
    expected = np.array([[x*8+y*16 for x in range(x0, x1+1)] for y in range(y0, y1+1)])
    assert np.array(stats['maps']['channel_rms']) == pytest.approx(expected)
    assert np.all(np.array(stats['valid_mask']) == 1)
    m = meta(); m['img_shape'] = (80, 70, 3)
    stats = s.native_patch(p3, points, box()[:, :2], m)
    assert 0 < np.array(stats['valid_mask']).mean() < 1
    # Separate raw-cell energies stay unchanged by the valid mask.
    assert np.array(stats['maps']['channel_rms']) == pytest.approx(expected)
    with pytest.raises(ValueError, match='budget'):
        s.native_patch(p3, points, torch.tensor([[900., 900.]]), meta())


def test_grad_nonfinite_and_cache_replay_substitution_fail():
    with pytest.raises(ValueError, match='detached'):
        s.shift_responses(plane().requires_grad_(), box(), meta())
    invalid = plane(); invalid[0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        s.shift_responses(invalid, box(), meta())
    x = torch.ones(1, 256, 9, 9)
    assert s.agreement(x+1e-6, x)['passed']
    changed = x.clone(); changed[0, 137, 3, 3] += .01
    with pytest.raises(ValueError, match='reviewed cache'):
        s.agreement(changed, x)
    with pytest.raises(ValueError, match='shape'):
        s.agreement(x[:, :, :8], x)


def test_static_does_not_load_cache_checkpoint_initialize_cuda_or_extract(monkeypatch, tmp_path):
    monkeypatch.setattr(r, 'checked_inputs', lambda _: ([], {}, [], dict(cache={}), {}, {}))
    def forbidden(*_, **__): raise AssertionError('GPU/cache/forward prohibited')
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(torch.cuda, 'init', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(r, 'extract_views', forbidden)
    out = tmp_path/'static'
    report = r.run(r.parser().parse_args(['--check-only', '--out-dir', str(out)]))
    assert report['scope']['feature_cache_loads'] == report['scope']['feature_extractions'] == 0
    assert report['scope']['gpu_devices_used'] == [] and report['formal_training_approved'] is False
    assert not (out/'panels').exists() and not (out/'cached_roi.json').exists()
    files = json.loads((out/'artifacts.json').read_text())['files']
    assert files == {str(p.relative_to(out)):r.g.ready.sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'artifacts.json'}
    with pytest.raises(FileExistsError): r.run(r.parser().parse_args(['--check-only', '--out-dir', str(out)]))


def test_source_protocol_fixed_selection_and_drift_before_data(monkeypatch, tmp_path):
    assert len(r.checked_sources()['sources']) == 68
    assert len(r.IMAGES) == len(set(r.IMAGES)) == 12
    protocol = r.protocol_document()
    assert protocol['scope']['feature_extractions'] == 24 and protocol['scope']['head_inference_calls'] == 48
    assert protocol['scope']['detector_updates'] == protocol['scope']['head_updates'] == 0
    broken = deepcopy(json.loads(r.MANIFEST.read_text()))
    broken['sources']['crane_project/utils/port_geometry_spatial_response_v1.py'] = 'changed'
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(broken)); monkeypatch.setattr(r, 'MANIFEST', path)
    with pytest.raises(ValueError, match='source SHA'): r.checked_sources()


def test_failure_preserves_completed_view_and_truthful_scope(monkeypatch, tmp_path):
    monkeypatch.setattr(r, 'checked_inputs', lambda _: ([], {}, [], dict(cache={}), {}, {}))
    monkeypatch.setattr(r.g, 'checked_cache', lambda *_: (dict(records=[]), {}))
    monkeypatch.setattr(r.g, 'validate_records', lambda *_: None)
    monkeypatch.setattr(r.q.o.c.a, 'check_cache_reference', lambda *_: None)
    monkeypatch.setattr(r, 'cache_audit', lambda *_: [])
    def fail(args, samples, rows, payload, audited, baseline, scope, out, progress):
        (out/'views').mkdir(); (out/'views/first.json').write_text('{}')
        scope.update(feature_extractions=2, head_inference_calls=4, completed_views=1, gpu_devices_used=[0])
        raise RuntimeError('Replay mismatch second view')
    monkeypatch.setattr(r, 'extract_views', fail)
    out = tmp_path/'failed'
    with pytest.raises(RuntimeError, match='Replay mismatch'):
        r.run(r.parser().parse_args(['--out-dir', str(out)]))
    report = json.loads((out/'completion.json').read_text())
    assert report['status'] == 'FAILED' and report['scope']['completed_views'] == 1
    assert report['scope']['feature_extractions'] == 2 and report['scope']['detector_updates'] == 0
    assert (out/'cached_roi.json').exists()
    assert 'views/first.json' in json.loads((out/'artifacts.json').read_text())['files']


def test_spatial_panel_pixel_affine_and_discrete_maps_render(tmp_path):
    p3 = plane(); b = box(); m = meta(); responses, points = s.shift_responses(p3, b, m)
    roi, mask, _ = r.g.sample_local(p3, b, m, 'ordinary')
    boxes = {k:b[0].tolist() for k in ('b', 'gt', 'center_only', 'joint')}
    xy = dict(zip(boxes, s.roi_coordinates(b[:, :2].repeat(4, 1), b, 'ordinary')))
    row = dict(image='synthetic', role='probe', domain='sim', scale=1., boxes_model=boxes,
        p3_patch=s.native_patch(p3, points, b[:, :2], m), responses=responses,
        reference_roi_coordinates=dict(ordinary=xy, aligned=xy),
        cached_roi={k:s.summarize_feature(roi) for k in ('ordinary', 'aligned')})
    path = tmp_path/'panel.png'
    result = s.render_panel(row, Image.new('RGB', (1024, 1024), (50, 50, 50)), path)
    assert result['pixel_crop_affine']['uniform_scale'] and Image.open(path).size == (1600, 1160)
    assert 'Cross-view' in result['display_note']


def test_full_24_view_driver_with_controlled_cpu_backends(monkeypatch, tmp_path):
    """Exercise native entry wiring/budget without claiming a real B/GPU run."""
    import mmcv.parallel
    import mmcv.runner
    import mmcv.utils
    import mmrotate.datasets
    import mmrotate.models
    import mmdet.datasets.dataset_wrappers
    from types import SimpleNamespace
    from crane_project.tools import diagnose_port_shape_e_h_train_gradients_v1 as grad
    p3 = plane(); samples = []; rows = []; records = []
    original = torch.cat((box(), torch.tensor([[.8]])), dim=1)
    for image in r.IMAGES:
        samples.append(dict(image=image, split='train', image_sha256='image', annotation_sha256='ann',
            previous_b_predictions={str(scale):original[0].tolist() for scale in (1., .5)}))
    for scale in (1., .5):
        for image in r.IMAGES:
            geometry = r.q.e.view_geometry([1024, 1024], scale)
            row = dict(image=image, role='probe', domain=image.split('_')[0], sequence=image.rsplit('_', 1)[0],
                scale=scale, original_image_wh=[1024, 1024], geometry=geometry, gt_original=box()[0].tolist(),
                arms={arm:dict(pred=original[0].tolist(), metrics={}) for arm in ('b', 'center_only', 'joint')})
            bm = r.g.map_boxes(box(), r.fixed_meta(row)); local = {}
            for arm in r.g.ARMS:
                roi, mask, _ = r.g.sample_local(p3, bm, r.fixed_meta(row), arm)
                local[arm] = dict(roi=roi, support=mask)
            records.append(dict(image=image, scale=scale, boxes_original=original.clone(), boxes_model=bm, local=local))
            rows.append(row)
    selected = {(v['image'], v['scale']):v for v in rows}
    detector_calls = []; head_calls = []
    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.parameter = torch.nn.Parameter(torch.tensor([.37])); self.current = None
        def cuda(self, _): return self
        def extract_feat(self, image):
            assert not torch.is_grad_enabled()
            self.current = rows[int(image[0, 0, 0, 0])]
            detector_calls.append((self.current['image'], self.current['scale']))
            return (p3,)
        def simple_test_from_features(self, features, metas, rescale=False):
            assert not torch.is_grad_enabled() and not any('gt' in k for k in metas[0])
            head_calls.append(rescale)
            b = original.clone()
            if not rescale: b[:, :5] = r.g.map_boxes(box(), metas[0])
            return [[b.numpy()]]
    detector = Detector(); frozen_state = r.g.state_digest(detector)
    class Part:
        def __init__(self, names, scale):
            self.names = names; self.scale = scale
            self.data_infos = [dict(filename=image+'.jpg') for image in names]
        def __len__(self): return len(self.names)
        def __getitem__(self, i):
            image = self.names[i]; row = selected[(image, self.scale)]
            index = next(j for j,v in enumerate(rows) if v is row)
            m = r.fixed_meta(row); m['filename'] = image+'.jpg'
            return dict(img=[torch.full((1, 3, 1024, 1024), float(index))], img_metas=[[m]])
    class Concat:
        def __init__(self, parts): self.datasets = parts
        def __getitem__(self, i):
            return self.datasets[0][i] if i < len(self.datasets[0]) else self.datasets[1][i-len(self.datasets[0])]
    monkeypatch.setattr(mmrotate.datasets, 'build_dataset', lambda spec:Part(*spec))
    monkeypatch.setattr(mmdet.datasets.dataset_wrappers, 'ConcatDataset', Concat)
    monkeypatch.setattr(r.g, 'fixed_specs', lambda cfg,scale:[(r.IMAGES[:6], scale), (r.IMAGES[6:], scale)])
    monkeypatch.setattr(mmcv.parallel, 'collate', lambda items, **_:items[0])
    monkeypatch.setattr(mmcv.parallel, 'scatter', lambda batch, *_:[batch])
    monkeypatch.setattr(mmrotate.models, 'build_detector', lambda _:detector)
    monkeypatch.setattr(mmcv.runner, 'load_checkpoint', lambda *_args, **_kw:dict(meta={}))
    monkeypatch.setattr(mmcv.utils, 'import_modules_from_strings', lambda **_:None)
    monkeypatch.setattr(grad, 'checkpoint_contract', lambda *_:{})
    cfg = SimpleNamespace(model=SimpleNamespace(), custom_imports={})
    protocol = json.loads(r.g.PROTOCOL.read_text())
    monkeypatch.setattr(r.g, 'check_cfg', lambda:cfg)
    monkeypatch.setattr(r.g, 'checked_inputs', lambda:(cfg, protocol, samples, {}))
    monkeypatch.setattr(r.g.ready, 'TRAIN_COUNTS', dict(fixture=12))
    monkeypatch.setattr(r.g.ready, 'sha', lambda path:r.g.ready.FROZEN_B['checkpoint_sha256'] if path == r.g.CHECKPOINT else 'image' if Path(path).suffix == '.jpg' else 'ann')
    monkeypatch.setattr(r.q.o.c.a, 'checked_runtime', lambda _:dict())
    monkeypatch.setattr(r.g, 'seed_all', lambda:None)
    monkeypatch.setattr(r.g, 'measured', lambda operation,_:(operation(), {}))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda:True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda:1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _:'controlled CPU fixture')
    monkeypatch.setattr(torch.cuda, 'set_device', lambda _:None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda:None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda _:0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda _:0)
    monkeypatch.setattr(r.q.e, 'native_rgb', lambda _:Image.new('RGB', (1024, 1024)))
    monkeypatch.setattr(r.q.e, 'native_view', lambda rgb,scale:(rgb, r.q.e.view_geometry([1024, 1024], scale)))
    audited = r.cache_audit(records, rows); scope = dict(feature_extractions=0, head_inference_calls=0)
    payload = dict(records=records, detector_state_before=frozen_state, extraction_runtime={})
    out = tmp_path/'driver'; out.mkdir(); progress = []
    result, report = r.extract_views(SimpleNamespace(gpu=0), samples, rows, payload, audited, {}, scope, out, progress.append)
    assert len(result) == len(detector_calls) == len(progress) == scope['completed_views'] == 24
    assert len(head_calls) == scope['head_inference_calls'] == 48 and scope['feature_extractions'] == 24
    assert scope['gpu_devices_used'] == [0] and report['detector_state_unchanged']
    assert len(list((out/'views').glob('*.json'))) == len(list((out/'panels').glob('*.png'))) == 24
    assert all(v['b_original_output_agreement']['passed'] for v in result)
