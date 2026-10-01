"""Actual SymEOODHead graph + finite differences, frozen bounded TRAIN I/O."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import MethodType

import pytest
import torch

from crane_project.tools import diagnose_port_shape_e_h_train_gradients_v1 as probe
from mmrotate.models.losses.covariance_shape_loss import CovarianceShapeLoss


def cpu_model(cfg):
    from mmrotate.models import build_head
    spec = deepcopy(cfg.model.bbox_head)
    spec.train_cfg, spec.test_cfg = cfg.model.train_cfg, cfg.model.test_cfg
    head = build_head(spec)
    def anchors(self, sizes, metas, device='cpu'):
        grids = self.anchor_generator.grid_priors(sizes, device='cpu')
        flags = [self.anchor_generator.valid_flags(sizes, m['pad_shape'], device='cpu') for m in metas]
        return [grids for _ in metas], flags
    head.get_anchors = MethodType(anchors, head)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bbox_head = head
            self.neck = torch.nn.Conv2d(3, 256, 1)
        def forward(self, img, img_metas, gt_bboxes, gt_labels, return_loss=True):
            from torch.nn.functional import adaptive_avg_pool2d
            x = self.neck(img)
            feats = [adaptive_avg_pool2d(x, (n, n)) for n in (8, 4, 2, 1, 1)]
            losses = self.bbox_head.loss(*self.bbox_head(feats), gt_bboxes, gt_labels, img_metas)
            losses['aux0_loss_fixture'] = x.square().mean()*.01
            return losses
    return Model().train()


def test_log_edge_finite_difference_and_covariance_equivalence():
    loss = CovarianceShapeLoss(mode='hellinger', loss_weight=.05)
    p = torch.tensor([[10., 11., 24., 12., .1]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[30., 31., 32., 16., .1]], dtype=torch.float64)
    grad = torch.autograd.grad(loss(p, t), p)[0][0]
    response = probe.signed_response(p[0].detach(), t[0], grad)
    assert response['long_error_times_gradient'] > 0
    assert response['short_error_times_gradient'] > 0
    assert response['xy_gradient'] == [0., 0.]
    for index in (2, 3, 4):
        plus, minus = p.detach().clone(), p.detach().clone()
        delta = 1e-5
        if index in (2, 3):
            plus[0, index] *= math.exp(delta); minus[0, index] *= math.exp(-delta)
            expected = p[0, index].detach()*grad[index]
        else:
            plus[0, index] += delta; minus[0, index] -= delta
            expected = grad[index]
        difference = (loss(plus, t)-loss(minus, t))/(2*delta)
        assert float(difference) == pytest.approx(float(expected), abs=1e-8)
    equivalent = t.clone()
    equivalent[:, 2:4] = t[:, [3, 2]]
    equivalent[:, 4] += math.pi/2
    assert float(loss(equivalent, t)) == pytest.approx(0., abs=1e-10)
    square = torch.tensor([1., 2., 16., 16., .1])
    assert not probe.signed_response(square, square, torch.zeros(5))['angle_defined_for_both']
    rotated = t.clone().requires_grad_(True)
    with torch.no_grad():
        rotated[:, 4] += .2
    angular = torch.autograd.grad(loss(rotated, t), rotated)[0][0]
    assert probe.signed_response(rotated[0].detach(), t[0], angular)['angle_error_times_gradient'] > 0


def test_original_gt_restoration_respects_resize_rounding():
    factor = [.8001, .8, .8001, .8]
    original = [30., 40., 32., 16., .2]
    transformed = [original[0]*factor[0], original[1]*factor[1],
                   original[2]*math.sqrt(factor[0]*factor[1]),
                   original[3]*math.sqrt(factor[0]*factor[1]), original[4]]
    assert probe.original_box(transformed, factor) == pytest.approx(original)
    assert probe.original_box([v*.5 if i < 4 else v for i, v in enumerate(transformed)],
                              [v*.5 for v in factor]) == pytest.approx(original)
    with pytest.raises(ValueError):
        probe.original_box(original, [1., 0., 1., 0.])


def test_checkpoint_contract_safe_complete_and_frozen_identity(tmp_path, monkeypatch):
    b, e = probe.check_configs()
    fields = ('model', 'data', 'optimizer', 'optimizer_config', 'lr_config', 'runner', 'load_from', 'resume_from')
    cfgtext = '\n'.join(k+' = '+repr(e.to_dict()[k]) for k in fields)
    meta = dict(config=cfgtext, seed=0, epoch=22, iter=14080)
    assert probe.checkpoint_contract(meta, e, 'e_h')['status'] == 'MATCH'
    for change in (dict(config='unrelated = 1'), dict(epoch=24), dict(seed=1), dict(iter=1)):
        with pytest.raises(ValueError):
            probe.checkpoint_contract(dict(meta, **change), e, 'e_h')
    marker = tmp_path/'executed'
    with pytest.raises(ValueError):
        probe.checkpoint_contract(dict(meta, config='model = __import__("pathlib").Path('+repr(str(marker))+').touch()'), e, 'e_h')
    assert not marker.exists()
    weight = tmp_path/'epoch_22.pth'
    weight.write_bytes(b'frozen fixture')
    monkeypatch.setitem(probe.WEIGHT_SHAS, 'e_h', probe.sha(weight))
    selection = dict(evidence_role='source_val_checkpoint_selection', selected_checkpoint='epoch_22',
        config_sha256=probe.sha(probe.EXPERIMENT), selected_path=str(weight),
        all_checkpoints={'epoch_22': dict(checkpoint=str(weight), checkpoint_sha256=probe.sha(weight))})
    source = tmp_path/'sweep_results.json'
    source.write_text(json.dumps(selection))
    assert probe.frozen_identity(tmp_path, 'e_h', probe.EXPERIMENT)['selected_epoch'] == 'epoch_22'
    weight.write_bytes(b'changed fixture')
    with pytest.raises(ValueError):
        probe.frozen_identity(tmp_path, 'e_h', probe.EXPERIMENT)


def test_phase_and_observers_restore_on_error():
    _, e = probe.check_configs()
    model = cpu_model(e)
    head = model.bbox_head
    counter = head.assigner._local_call_count
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    original, extra = head.loss_single, head.shape_compensation.forward
    with pytest.raises(RuntimeError, match='fixture'):
        with probe.late_state(model), probe.observe_nodes(head):
            assert head.loss_cls._get_current_tau() == 1.
            raise RuntimeError('fixture')
    assert head.assigner._local_call_count == counter
    assert head.loss_single == original and head.shape_compensation.forward == extra
    assert all(torch.equal(v, buffers[k]) for k, v in model.named_buffers())
    model.register_buffer('fixture_buffer', torch.tensor(1.))
    with pytest.raises(RuntimeError, match='non-schedule'):
        with probe.late_state(model):
            model.fixture_buffer.add_(1.)
    assert float(model.fixture_buffer) == 1.
    model.bn = torch.nn.BatchNorm2d(1).train()
    with pytest.raises(ValueError, match='norm_eval'):
        with probe.late_state(model):
            pass


def test_bounded_runtime_actual_head_nodes_input_pairing_and_immutable_state(tmp_path, monkeypatch):
    import mmcv.parallel
    import mmcv.runner
    import mmrotate.datasets
    import mmrotate.models
    from mmdet.datasets.dataset_wrappers import ConcatDataset
    from crane_project.tools import ckpt_sweep
    b, e = probe.check_configs()
    monkeypatch.setattr(mmrotate.models, 'build_detector',
        lambda cfg: cpu_model(b if cfg.bbox_head.get('shape_compensation') is None else e))
    def load(model, path, **kwargs):
        arm = Path(path).stem
        cfg = b if arm == 'b' else e
        epoch = probe.EPOCHS[arm]
        return dict(meta=dict(config=cfg.pretty_text, epoch=epoch, iter=epoch*640, seed=0))
    monkeypatch.setattr(mmcv.runner, 'load_checkpoint', load)
    class Leaf:
        CLASSES = ('grab',)
        def __init__(self, count, domain, scale):
            self.scale = scale
            self.data_infos = [dict(filename=domain+'_fixture_'+str(i)+'.jpg') for i in range(count)]
            if domain == 'real':
                for i, name in ((0, probe.IMAGES[0][0]), (905, probe.IMAGES[1][0])):
                    self.data_infos[i]['filename'] = name+'.jpg'
            else:
                for i, name in ((0, probe.IMAGES[0][1]), (374, probe.IMAGES[1][1])):
                    self.data_infos[i]['filename'] = name+'.jpg'
        def __len__(self):
            return len(self.data_infos)
        def __getitem__(self, i):
            path = tmp_path/self.data_infos[i]['filename']
            if not path.exists():
                path.write_bytes(b'CPU fixture')
            gt = torch.tensor([[30., 30., 32., 16., .2]])
            gt[:, :4] *= self.scale
            return dict(img=torch.ones(3, 16, 16)*self.scale, gt_bboxes=gt,
                gt_labels=torch.zeros(1, dtype=torch.long), img_metas=dict(filename=str(path),
                img_shape=(64, 64, 3), pad_shape=(128, 128, 3), scale_factor=[self.scale]*4, flip=False))
    def dataset(spec):
        scale = next(step.scale_range[0] for step in spec[0].pipeline
                     if step.type == 'PortIsotropicShrink')
        return ConcatDataset([Leaf(1810, 'real', scale), Leaf(748, 'sim', scale)])
    def collate(items, samples_per_gpu=2):
        return dict(img=torch.stack([x['img'] for x in items]),
            **{k: [x[k] for x in items] for k in ('gt_bboxes', 'gt_labels', 'img_metas')})
    monkeypatch.setattr(mmrotate.datasets, 'build_dataset', dataset)
    monkeypatch.setattr(mmcv.parallel, 'collate', collate)
    monkeypatch.setattr(mmcv.parallel, 'scatter', lambda data, devices: [data])
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self, *a: self)
    for name in ('set_device', 'manual_seed_all', 'reset_peak_memory_stats', 'empty_cache'):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *a: 'CPU fixture; CUDA unverified')
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *a: 0)
    monkeypatch.setattr(ckpt_sweep, 'annotation_set_sha256', lambda *a: 'CPU fixture')
    progress = tmp_path/'progress.jsonl'
    result = probe.runtime_probe(dict(b=b, e_h=e),
        {arm: dict(checkpoint=str(tmp_path/(arm+'.pth'))) for arm in ('b', 'e_h')}, 0, progress)
    assert len(result['rows']) == 8 and result['optimizer_steps'] == 0
    assert len(progress.read_text().splitlines()) == 8
    assert all(x['parameters_and_buffers_unchanged'] for x in result['states'].values())
    assert all(x['inputs_identical'] for x in result['pairing'])
    for row in result['rows']:
        assert row['actual_node_reuse'] and row['per_image_positive_counts'] == [1, 1]
        assert row['classification_tau'] == 1.
        assert 'aux0_loss_fixture' in row['losses']
        assert row['gradients']['regression_conv']['norms']['shape_005'] > 0
        assert row['gradients']['full_fpn']['norms']['shape_005'] > 0
        assert row['gradients']['classification_conv']['norms']['shape_005'] == 0
        for positive in row['positives']:
            assert positive['direct']['shape_005']['xy_gradient'] == [0., 0.]
        assert row['shape_role'].startswith('counterfactual') == (row['arm'] == 'b')
    json.dumps(result, allow_nan=False)
    # Exercise successful CLI persistence using the already verified runtime result.
    import sys
    manifest = tmp_path/'sources.json'
    manifest.write_text(json.dumps(dict(sources={s: probe.sha(probe.ROOT/s) for s in probe.SOURCES})))
    monkeypatch.setattr(probe, 'MANIFEST', manifest)
    monkeypatch.setattr(probe, 'frozen_identity', lambda *a: dict(checkpoint='CPU fixture'))
    monkeypatch.setattr(probe, 'runtime_probe', lambda *a: result)
    out = tmp_path/'runtime.json'
    monkeypatch.setattr(sys, 'argv', ['probe', '--out-json', str(out)])
    probe.main()
    persisted = json.loads(out.read_text())
    assert persisted['status'] == 'TRAIN_GRADIENT_CHECK_COMPLETE_REVIEW_REQUIRED'
    assert len(persisted['train']['rows']) == 8
    assert out.with_suffix('.artifacts.json').exists()
    assert out.with_suffix('.progress.jsonl').exists()


def test_cli_config_only_manifest_failure_and_no_overwrite(tmp_path, monkeypatch):
    import sys
    out = tmp_path/'config.json'
    manifest = tmp_path/'sources.json'
    manifest.write_text(json.dumps(dict(sources={s: probe.sha(probe.ROOT/s) for s in probe.SOURCES})))
    monkeypatch.setattr(probe, 'MANIFEST', manifest)
    monkeypatch.setattr(sys, 'argv', ['probe', '--config-only', '--out-json', str(out)])
    probe.main()
    report = json.loads(out.read_text())
    assert report['status'] == 'CONFIG_ONLY_RUNTIME_UNVERIFIED'
    assert report['optimizer_steps'] == 0 and 'train' not in report
    with pytest.raises(FileExistsError):
        probe.main()
    out = tmp_path/'mismatch.json'
    manifest.write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['probe', '--config-only', '--out-json', str(out)])
    with pytest.raises(ValueError, match='manifest'):
        probe.main()
    assert json.loads(out.read_text())['status'] == 'CHECK_FAILED'
