"""F-S numerical properties and real-head I/O; CUDA paths use explicit fixtures."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import MethodType

import pytest
import torch

from crane_project.tools import check_port_size_f_s_v1 as formula
from crane_project.tools import preflight_port_size_f_s_v1 as probe
from mmrotate.models.losses.log_size_loss import LogSizeLoss


def test_sorted_log_derivatives_detach_equivalence_and_normalizer():
    loss = LogSizeLoss()
    p = torch.tensor([[10., 20., 76., 38., .3], [10., 20., 84., 42., .5]],
                     dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[10., 20., 80., 40., .2]]*2, dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([1., .3], dtype=torch.float64)
    value = loss(p, t, weight=weights, avg_factor=4.)
    gp, gt = torch.autograd.grad(value, (p, t), allow_unused=True)
    assert gt is None and torch.count_nonzero(gp[:, [0, 1, 4]]) == 0
    expected = .1*weights[:, None]*formula.reduction_multiplier(4., p.dtype)/2.*(torch.log(p[:, 2:4]/t[:, 2:4])/.1).clamp(-1, 1)
    assert torch.allclose(gp[:, 2:4]*p[:, 2:4], expected, atol=1e-12, rtol=0)
    swapped = p.detach().clone(); swapped[:, 2:4] = p.detach()[:, [3, 2]]
    swapped[:, 4] += math.pi/2
    assert float(loss(swapped, t, weight=weights, avg_factor=4.)) == pytest.approx(float(value), abs=1e-12)
    pp, tt = p.detach().clone(), t.detach().clone()
    pp[:, :4] *= .5; tt[:, :4] *= .5
    assert float(loss(pp, tt, weight=weights, avg_factor=4.)) == pytest.approx(float(value), abs=1e-12)
    assert float(loss(p, t, weight=weights[:, None].expand(-1, 5), avg_factor=4.)) == pytest.approx(float(value))
    assert float(loss(p, t, weight=torch.zeros(2), avg_factor=4.)) == 0.
    assert float(loss(p, t, reduction_override='none').sum()) == pytest.approx(float(loss(p, t, reduction_override='sum')))


def test_empty_guards_and_invalid_inputs():
    loss = LogSizeLoss()
    empty = torch.empty(0, 5, requires_grad=True)
    loss(empty, empty).backward()
    assert empty.grad is not None and empty.grad.shape == empty.shape
    assert loss(empty, empty, reduction_override='none').shape == (0,)
    p = torch.tensor([[1., 2., 1e-8, 2e-8, .3]], requires_grad=True)
    t = torch.tensor([[1., 2., 1., 2., .3]])
    grad = torch.autograd.grad(loss(p, t), p)[0]
    assert torch.count_nonzero(grad) == 0  # protection deliberately blocks sub-eps edges
    for value in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            LogSizeLoss(beta=value)
    for value in (0., -1., float('nan')):
        bad = t.clone(); bad[:, 2] = value
        with pytest.raises(ValueError):loss(bad, t)
    with pytest.raises(ValueError):loss(t, t, avg_factor=0.)
    with pytest.raises(ValueError):loss(t, t, weight=torch.tensor([-1.]))
    with pytest.raises(ValueError):loss(t, t, weight=torch.ones(1, 2))


def test_saved_boxes_and_synthetic_formula_checks_preserve_counterexample():
    result = formula.review_fixture()
    assert result['all_mathematical_checks_passed']
    assert len(result['train_replay']) == 16 and len(result['val_descriptive']) == 8
    assert len(result['mechanisms']['rows']) == 34
    assert result['optimizer_steps'] == result['detector_forwards'] == 0
    assert max(r['maximum_finite_difference_error'] for r in result['train_replay']) < 2e-6
    exception = next(r for r in result['train_replay'] if r['arm'] == 'e_h' and r['scale'] == .5
                     and r['image'] == 'sim_seq08_00374')
    assert exception['direct']['size']['log_short_gradient'] > 0
    assert exception['direct']['main_plus_size']['log_short_gradient'] < 0
    assert all(r['direct']['size']['xy_gradient'] == [0., 0.] for r in result['train_replay'])


def test_f_s_equals_existing_d_size_component_only():
    from crane_project.tools.diagnose_port_center_size_d_v1 import d_parts
    p = torch.tensor([[10., 12., 84., 37., .4]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[12., 12., 80., 40., .2]], dtype=torch.float64)
    _, old_size = d_parts(p, t, torch.ones(1), 2., beta=.1, coefficient=.1)
    assert torch.equal(LogSizeLoss()(p, t, weight=torch.ones(1), avg_factor=2.), old_size)


class Tiny(torch.nn.Module):
    """Actual head; tiny shared convolution replaces ResNet/FPN for CPU checks."""
    def __init__(self, cfg):
        from mmrotate.models import build_head
        super().__init__()
        spec = deepcopy(cfg.bbox_head)
        spec.train_cfg, spec.test_cfg = cfg.train_cfg, cfg.test_cfg
        self.bbox_head = build_head(spec)
        def anchors(head, sizes, metas, device='cpu'):
            grids = head.anchor_generator.grid_priors(sizes, device='cpu')
            flags = [head.anchor_generator.valid_flags(sizes, m['pad_shape'], device='cpu') for m in metas]
            return [grids for _ in metas], flags
        self.bbox_head.get_anchors = MethodType(anchors, self.bbox_head)
        self.neck = torch.nn.Conv2d(3, 256, 1)
    def init_weights(self):
        self.bbox_head.init_weights()
        torch.nn.init.normal_(self.neck.weight, std=.01)
    def forward(self, img, img_metas, gt_bboxes, gt_labels, return_loss=True):
        from torch.nn.functional import adaptive_avg_pool2d
        x = self.neck(img)
        features = [adaptive_avg_pool2d(x, (n, n)) for n in (8, 4, 2, 1, 1)]
        result = self.bbox_head.loss(*self.bbox_head(features), gt_bboxes, gt_labels, img_metas)
        # Exercise RNG replay: unchanged auxiliary stochastic branch.
        result['aux0_loss_fixture'] = x.square().mean()*.01*torch.rand(())
        return result


def fixed_batch(tmp_path, scale=1.):
    paths = [tmp_path/(name+'.jpg') for name in probe.IMAGES[0]]
    for path in paths:path.write_bytes(b'CPU fixture only')
    box = torch.tensor([[30., 30., 32., 16., .2]])
    box[:, :4] *= scale
    return dict(img=torch.ones(2, 3, 16, 16)*scale, gt_bboxes=[box.clone(), box.clone()],
        gt_labels=[torch.zeros(1, dtype=torch.long)]*2,
        img_metas=[dict(filename=str(path), img_shape=(64,64,3), pad_shape=(128,128,3),
                       scale_factor=[scale]*4, flip=False) for path in paths])


@pytest.mark.parametrize('stage', ['frozen_b', 'initialization'])
def test_actual_nodes_gradient_scope_clipping_and_replayed_b_losses(tmp_path, stage):
    _, f = probe.check_configs()
    torch.manual_seed(0); model = Tiny(f.model).train()
    before = probe.parameter_digest(model)
    buffers = {k:v.clone() for k,v in model.named_buffers()}
    counter = model.bbox_head.assigner._local_call_count
    result = probe.probe_batch(model, fixed_batch(tmp_path), f, stage, 1.)
    assert result['actual_node_reuse'] and result['base_losses_unchanged']
    assert result['normalizers'] == [sum(result['per_image_positive_counts'])]*5
    assert result['direct_coordinate_norms']['x'] == result['direct_coordinate_norms']['y'] == result['direct_coordinate_norms']['angle'] == 0
    assert result['gradients']['full_fpn']['norms']['size'] > 0
    assert result['gradients']['classification_conv']['norms']['size'] == 0
    assert result['gradients']['regression_conv']['size_xy_angle_output_rows_norm'] == 0
    assert result['clipping']['complete_f_s']['before'] > 0
    assert probe.parameter_digest(model) == before
    assert all(torch.equal(v, buffers[k]) for k,v in model.named_buffers())
    assert model.bbox_head.assigner._local_call_count == counter
    assert all(p.grad is None for p in model.parameters())


def test_inference_and_state_unchanged_when_enabling_size_loss():
    b, f = probe.check_configs()
    torch.manual_seed(0); heads = [Tiny(b.model).bbox_head, Tiny(f.model).bbox_head]
    heads[1].load_state_dict(heads[0].state_dict(), strict=True)
    assert list(heads[0].state_dict()) == list(heads[1].state_dict())
    features = [torch.randn(1,256,n,n) for n in (8,4,2,1,1)]
    predictions = []
    for head in heads:
        output = head(features)
        grids = head.anchor_generator.grid_priors([x.shape[-2:] for x in features], device='cpu')
        predictions.append(head._get_bboxes_single([x[0].detach() for x in output[0]],
            [x[0].detach() for x in output[1]], grids, (64,64,3), [1.,1.,1.,1.], head.test_cfg, rescale=True))
    assert all(torch.equal(a,b) for a,b in zip(*predictions))


def test_exception_restore_and_buffer_bn_rejection(tmp_path):
    _, f = probe.check_configs(); model = Tiny(f.model).train()
    model.register_buffer('fixture', torch.tensor(1.))
    before = {k:v.clone() for k,v in model.named_buffers()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='non-schedule'):
        with probe.phase_state(model, 'warmup_o2m'):
            torch.rand(1); model.fixture.add_(1)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(v,before[k]) for k,v in model.named_buffers())
    original, extra = model.bbox_head.loss_single, model.bbox_head.shape_compensation.forward
    with pytest.raises(RuntimeError, match='fixture'):
        with probe.phase_state(model,'late_o2o'), probe.observe_nodes(model.bbox_head):
            raise RuntimeError('fixture')
    assert model.bbox_head.loss_single == original and model.bbox_head.shape_compensation.forward == extra
    model.bn = torch.nn.BatchNorm2d(1).train()
    with pytest.raises(ValueError, match='norm_eval'):
        with probe.phase_state(model,'late_o2o'):pass


def test_bounded_runtime_initialization_inputs_state_and_progress(tmp_path, monkeypatch):
    import mmcv.parallel
    import mmcv.runner
    import mmrotate.datasets
    import mmrotate.models
    from mmdet.datasets.dataset_wrappers import ConcatDataset
    from crane_project.tools import ckpt_sweep
    b, f = probe.check_configs()
    monkeypatch.setattr(mmrotate.models,'build_detector',lambda cfg:Tiny(cfg))
    def load(model,path,**kwargs):
        model.init_weights()
        fields = ('model','data','optimizer','optimizer_config','lr_config','runner','load_from','resume_from')
        return dict(meta=dict(config='\n'.join(k+' = '+repr(b.to_dict()[k]) for k in fields),seed=0,epoch=24,iter=15360))
    monkeypatch.setattr(mmcv.runner,'load_checkpoint',load)
    class Leaf:
        CLASSES = ('grab',)
        def __init__(self,domain,count,scale):
            self.scale=scale
            self.data_infos=[dict(filename=domain+'_fixture_'+str(i)+'.jpg') for i in range(count)]
            names = {0:'real_seq01_00000',905:'real_seq06_00006'} if domain=='real' else {0:'sim_seq08_00000',374:'sim_seq08_00374'}
            for i,name in names.items():self.data_infos[i]['filename']=name+'.jpg'
        def __len__(self):return len(self.data_infos)
        def __getitem__(self,i):
            path=tmp_path/self.data_infos[i]['filename'];path.write_bytes(b'fixed CPU fixture')
            box=torch.tensor([[30.,30.,32.,16.,.2]]);box[:,:4]*=self.scale
            return dict(img=torch.ones(3,16,16)*self.scale,gt_bboxes=box,gt_labels=torch.zeros(1,dtype=torch.long),
                img_metas=dict(filename=str(path),img_shape=(64,64,3),pad_shape=(128,128,3),scale_factor=[self.scale]*4,flip=False))
    def dataset(specs):
        scale=next(s for s in specs[0].pipeline if s['type']=='PortIsotropicShrink')['scale_range'][0]
        return ConcatDataset([Leaf('real',1810,scale),Leaf('sim',748,scale)])
    monkeypatch.setattr(mmrotate.datasets,'build_dataset',dataset)
    monkeypatch.setattr(mmcv.parallel,'collate',lambda items,**kw:dict(img=torch.stack([x['img'] for x in items]),
        **{k:[x[k] for x in items] for k in ('gt_bboxes','gt_labels','img_metas')}))
    monkeypatch.setattr(mmcv.parallel,'scatter',lambda batch,devices:[batch])
    monkeypatch.setattr(torch.nn.Module,'cuda',lambda self,*a:self)
    for name in ('set_device','manual_seed_all','reset_peak_memory_stats','empty_cache'):
        monkeypatch.setattr(torch.cuda,name,lambda *a:None)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *a:'CPU fixture; actual CUDA unverified')
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *a:0)
    monkeypatch.setattr(ckpt_sweep,'annotation_set_sha256',lambda *a:'CPU fixture')
    import hashlib
    monkeypatch.setattr(probe,'training_identity',lambda:dict(
        annotation_sha256=dict(train='CPU fixture',train_sim='CPU fixture'),
        image_sha256={name:hashlib.sha256(b'fixed CPU fixture').hexdigest() for pair in probe.IMAGES for name in pair}))
    progress=tmp_path/'progress.jsonl'
    checkpoint=tmp_path/'frozen_fixture.pth';checkpoint.write_bytes(b'CPU checkpoint fixture')
    result=probe.runtime_probe(b,f,dict(checkpoint=str(checkpoint),checkpoint_sha256=probe.sha(checkpoint)),0,progress)
    assert len(result['rows']) == len(progress.read_text().splitlines()) == 8
    assert result['states']['initialization']['contract']['b_f_initialization_equal']
    assert result['inputs_identical_across_stages'] and result['optimizer_steps']==0
    assert all(s['parameters_and_buffers_unchanged'] for s in result['states'].values())
    assert all(r['actual_node_reuse'] and r['base_losses_unchanged'] for r in result['rows'])
    # Persist this actual-head CPU runtime result through CLI without rerunning.
    import sys
    manifest=tmp_path/'manifest.json'
    sources={s:formula.sha(formula.ROOT/s) for s in formula.SOURCES}
    manifest.write_text(json.dumps(dict(sources=sources)))
    monkeypatch.setattr(formula,'MANIFEST',manifest);monkeypatch.setattr(probe,'MANIFEST',manifest)
    math_out=tmp_path/'math.json'
    monkeypatch.setattr(sys,'argv',['math','--out-json',str(math_out)])
    formula.main()
    monkeypatch.setattr(probe,'frozen_identity',lambda *a:dict(checkpoint='CPU fixture'))
    monkeypatch.setattr(probe,'runtime_probe',lambda *a:result)
    out=tmp_path/'runtime.json'
    monkeypatch.setattr(sys,'argv',['probe','--math-report',str(math_out),'--out-json',str(out)])
    probe.main();persisted=json.loads(out.read_text())
    assert persisted['status']=='TRAIN_PREFLIGHT_COMPLETE_REVIEW_REQUIRED'
    assert len(persisted['train']['rows'])==8 and not persisted['formal_training_authorized']
    assert out.with_suffix('.artifacts.json').exists() and out.with_suffix('.progress.jsonl').exists()


def test_cli_sources_prerequisite_no_overwrite_and_failure(tmp_path, monkeypatch):
    import sys
    manifest=tmp_path/'manifest.json'
    sources={s:formula.sha(formula.ROOT/s) for s in formula.SOURCES}
    manifest.write_text(json.dumps(dict(sources=sources)))
    monkeypatch.setattr(formula,'MANIFEST',manifest);monkeypatch.setattr(probe,'MANIFEST',manifest)
    math_out=tmp_path/'math.json'
    monkeypatch.setattr(sys,'argv',['check','--out-json',str(math_out)])
    formula.main()
    assert probe.prerequisite(math_out,sources)['status']=='SAVED_BOX_FORMULA_CHECK_COMPLETE_REVIEW_REQUIRED'
    with pytest.raises(FileExistsError):formula.main()
    bad=json.loads(math_out.read_text());bad['candidate_settings']['loss_weight']=.2
    bad_out=tmp_path/'bad.json';bad_out.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='Prerequisite'):probe.prerequisite(bad_out,sources)
    out=tmp_path/'config.json'
    monkeypatch.setattr(sys,'argv',['probe','--config-only','--out-json',str(out)])
    probe.main();assert json.loads(out.read_text())['status']=='CONFIG_ONLY_RUNTIME_UNVERIFIED'
    with pytest.raises(FileExistsError):probe.main()
    manifest.write_text('{}')
    out=tmp_path/'failed.json'
    monkeypatch.setattr(sys,'argv',['probe','--config-only','--out-json',str(out)])
    with pytest.raises(ValueError,match='manifest'):probe.main()
    assert json.loads(out.read_text())['status']=='CHECK_FAILED'


def test_head_empty_level_and_mutated_candidate_config_rejected(monkeypatch):
    from mmcv import Config
    _, f=probe.check_configs();head=Tiny(f.model).bbox_head
    cls=torch.zeros(1,3,1,1,requires_grad=True)
    reg=torch.zeros(1,15,1,1,requires_grad=True)
    anchors=torch.tensor([[16.,16.,32.,16.,0.]]).repeat(3,1)
    result=head.loss_single(cls,reg,anchors,torch.ones(3,dtype=torch.long),torch.ones(3),
        torch.zeros(3,5),torch.zeros(3,5),num_total_samples=1)
    assert float(result[-1])==0
    result[-1].backward();assert reg.grad is not None and torch.count_nonzero(reg.grad)==0
    original=Config.fromfile
    def changed(path,*a,**kw):
        cfg=original(path,*a,**kw)
        if Path(path)==probe.EXPERIMENT:cfg.model.bbox_head.shape_compensation.loss_weight=.2
        return cfg
    monkeypatch.setattr(Config,'fromfile',changed)
    with pytest.raises(ValueError,match='candidate'):probe.check_configs()


def test_backward_failure_clears_gradients_and_restores_state(tmp_path,monkeypatch):
    _,f=probe.check_configs();model=Tiny(f.model).train()
    buffers={k:v.clone() for k,v in model.named_buffers()}
    counter=model.bbox_head.assigner._local_call_count
    original=model.bbox_head.loss_single
    def fail(*a,**kw):raise RuntimeError('fixture clip failure')
    monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',fail)
    with pytest.raises(RuntimeError,match='clip failure'):
        probe.probe_batch(model,fixed_batch(tmp_path),f,'frozen_b',1.)
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(v,buffers[k]) for k,v in model.named_buffers())
    assert model.bbox_head.assigner._local_call_count==counter
    assert model.bbox_head.loss_single==original
