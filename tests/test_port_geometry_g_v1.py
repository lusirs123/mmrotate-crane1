"""Frozen-B geometry refinement: CPU mechanism and interface regressions."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest
import torch

from crane_project.utils import port_geometry_refine_g_v1 as g
from crane_project.tools import preflight_port_geometry_g_v1 as tool


def box():
    return torch.tensor([[32., 24., 32., 16., .17, .8]])


def meta():
    return dict(img_shape=(64, 64, 3), pad_shape=(64, 64, 3),
                ori_shape=(64, 64, 3), scale_factor=[1., 1., 1., 1.], flip=False)


@pytest.mark.parametrize('angle', [.17, math.pi/2-1e-6, -math.pi/2+1e-6])
@pytest.mark.parametrize('swap', [False, True])
def test_zero_residual_is_bitwise_identity_and_carries_angle_gradient(angle, swap):
    b = box(); b[0, 4] = angle
    if swap:
        b[:, [2, 3]] = b[:, [3, 2]]
    residual = torch.zeros(1, 3, requires_grad=True)
    refined, crossing = g.apply_residual(b, residual)
    assert torch.equal(refined, b) and not crossing.any()
    refined[:, 4].sum().backward()
    assert residual.grad[0, 2] == 1


def test_width_exchange_and_periodic_direction_produce_equivalent_sampling_and_loss():
    b = box(); alternate = b.clone()
    alternate[:, [2, 3]] = b[:, [3, 2]]; alternate[:, 4] += math.pi/2
    a, _ = g.sampling_grid(b[:, :5], (8, 8), 'aligned')
    c, _ = g.sampling_grid(alternate[:, :5], (8, 8), 'aligned')
    assert torch.allclose(a, c, atol=1e-6)
    gt = b[:, :5].clone(); gt[:, 2:4] *= 1.04; gt[:, 4] -= .03
    assert torch.allclose(g.regression_loss(b, gt)[0], g.regression_loss(alternate, gt)[0], atol=1e-6)


def test_bilinear_coordinates_and_padding_mask_are_observable():
    p = torch.zeros(1, 256, 8, 8)
    yy, xx = torch.meshgrid(torch.arange(8.), torch.arange(8.))
    p[0, 0] = xx*8; p[0, 1] = yy*8
    b = box()[:, :5]; b[:, 4] = 0
    roi, support, points = g.sample_local(p, b, meta(), 'ordinary')
    assert torch.allclose(roi[0, :2, 4, 4], b[0, :2])
    assert support[0, 0, 4, 4] == 1
    m = meta(); m['img_shape'] = (32, 40, 3)
    p[:, :, 4:, :] = 1e6; p[:, :, :, 5:] = 1e6
    local, valid, _ = g.sample_local(p, b, m, 'ordinary')
    assert float(local.abs().max()) < 100 and float(valid.min()) == 0
    assert torch.equal(points[0, 4, 4], b[0, :2])


def test_rotation_changes_only_sampling_grid_orientation():
    b = box()[:, :5]
    ordinary, po = g.sampling_grid(b, (8, 8), 'ordinary')
    aligned, pa = g.sampling_grid(b, (8, 8), 'aligned')
    assert torch.equal(po[:, 4, 4], pa[:, 4, 4])
    assert torch.allclose((po-b[:, None, None, :2]).square().sum(-1),
                          (pa-b[:, None, None, :2]).square().sum(-1), atol=1e-4)
    assert not torch.allclose(ordinary, aligned)


def test_raw_width_height_restoration_precedes_canonicalization():
    m = dict(img_shape=(563, 1024, 3), pad_shape=(1024, 1024, 3),
             ori_shape=(1100, 2000, 3), scale_factor=[.512, 563/1100, .512, 563/1100], flip=False)
    b = torch.tensor([[100., 200., 16., 32., .2]])
    restored = g.map_boxes(b, m, inverse=True)
    assert torch.allclose(restored[0, 2:4], torch.tensor([16/.512, 32/(563/1100)]))
    assert torch.allclose(g.map_boxes(restored, m), b, atol=1e-4)
    bad = deepcopy(m); bad['scale_factor'] = [.4, .6, .4, .6]
    with pytest.raises(ValueError, match='Anisotropic'):
        g.checked_meta(bad)


@pytest.mark.parametrize('swap', [False, True])
def test_bounded_crossing_projects_to_square_without_angle_jump_or_center_score_change(swap):
    b = box(); b[:, 2:4] = torch.tensor([20., 19.])
    if swap:
        b[:, [2, 3]] = b[:, [3, 2]]; b[:, 4] -= math.pi/2
    r = torch.tensor([[-math.log(1.25), math.log(1.25), .04]])
    refined, crossing = g.apply_residual(b, r)
    assert crossing.all() and torch.equal(refined[:, 2], refined[:, 3])
    assert torch.equal(refined[:, [0, 1, 5]], b[:, [0, 1, 5]])
    assert torch.allclose(g.canonical_boxes(refined[:, :5])[:, 4], g.wrap_pi(g.canonical_boxes(b[:, :5])[:, 4]+.04), atol=1e-6)
    for edge in (2, 3):
        assert .8-1e-6 <= float(refined[0, edge]/b[0, edge]) <= 1.25+1e-6
    with pytest.raises(ValueError, match='fixed G bound'):
        g.apply_residual(b, r*2)


def test_two_updates_have_effective_three_task_and_stem_gradients_without_input_gradients():
    torch.manual_seed(1703)
    head = g.LocalGeometryRefiner()
    roi = torch.randn(1, 256, 9, 9, requires_grad=True)
    support = torch.ones(1, 1, 9, 9, requires_grad=True)
    b = box().requires_grad_(); bm = b[:, :5].detach().clone().requires_grad_()
    gt = b[:, :5].detach().clone(); gt[:, 2:4] *= 1.08; gt[:, 4] -= .04; gt.requires_grad_()
    optimizer = torch.optim.Adam(head.parameters(), lr=.001, weight_decay=0.)
    initial = head(roi, support, b, bm)['boxes_original']
    assert torch.equal(initial, b)
    for step in range(2):
        optimizer.zero_grad()
        output = head(roi, support, b, bm)
        loss, parts = g.regression_loss(output['boxes_original'], gt)
        if step == 0:
            norms = [torch.autograd.grad(p, head.output.weight, retain_graph=True)[0].norm() for p in parts]
            assert all(x > 0 for x in norms)
        loss.backward()
        stem = sum(float(p.grad.square().sum()) for p in head.stem.parameters())
        assert stem == 0 if step == 0 else stem > 0
        assert all(torch.isfinite(p.grad).all() for p in head.parameters())
        optimizer.step()
    assert all(t.grad is None for t in (roi, support, b, bm, gt))
    assert torch.equal(output['boxes_original'][:, [0, 1, 5]], b[:, [0, 1, 5]])


def test_empty_output_remains_empty_and_frozen_feature_graph_is_rejected():
    b = torch.empty(0, 6); p = torch.zeros(1, 256, 8, 8)
    roi, support, _ = g.sample_local(p, b[:, :5], meta(), 'aligned')
    assert g.LocalGeometryRefiner()(roi, support, b, b[:, :5])['boxes_original'].shape == (0, 6)
    with pytest.raises(ValueError, match='detector graph'):
        g.sample_local(p.requires_grad_(), box()[:, :5], meta(), 'aligned')


def test_schedule_is_identical_domain_balanced_and_excludes_probe_and_bad_center_views():
    records = [dict(domain=d, role=role, eligible=eligible) for d in ('real', 'sim')
               for role in ('fit', 'probe') for eligible in (True, False)]
    first = tool.schedule(records)
    assert first == tool.schedule(records) and len(first) == 200
    for batch in first:
        assert len(batch) == 8
        assert sum(records[i]['domain'] == 'real' for i in batch) == 4
        assert all(records[i]['role'] == 'fit' and records[i]['eligible'] for i in batch)


def test_detector_freeze_preserves_bn_buffers_and_excludes_optimizer_updates():
    detector = torch.nn.Sequential(torch.nn.BatchNorm2d(2), torch.nn.Conv2d(2, 2, 1))
    g.freeze_detector(detector); before = tool.state_digest(detector)
    with torch.no_grad():
        detector(torch.randn(1, 2, 4, 4))
    g.assert_detector_frozen(detector)
    assert before == tool.state_digest(detector)


def test_sparse_report_keeps_missing_and_incorrect_output_denominators():
    gt = box()[0, :5].numpy()
    rows = []
    for i, pred in enumerate((None, gt.copy(), gt.copy())):
        if i == 2:
            pred[0] += 50
        rows.append(dict(image='real_seq01_%05d' % (i*10), sequence='real_seq01', frame_id=i*10,
                         metrics=tool.decompose(gt, pred)))
    report = tool.summarize(rows, continuous=False)
    assert report['output_center_hit_pct'] == 50
    assert report['output_coverage_pct'] == pytest.approx(200/3)
    assert report['all_frame_center_hit_pct'] == pytest.approx(100/3)
    assert report['longest_riou_failure_run'] is None


def test_historical_b_baseline_replay_checks_presence_geometry_and_score():
    protocol = dict(baseline_tolerances=dict(center_and_edges_px=.1, angle_rad=.001, score=.001))
    b = box(); old = b[0].tolist()
    assert tool.check_previous(b, old, protocol)['max_center_edge_delta_px'] < 1e-6
    with pytest.raises(ValueError, match='presence'):
        tool.check_previous(torch.empty(0, 6), old, protocol)
    changed = b.clone(); changed[0, 5] -= .01
    with pytest.raises(ValueError, match='previous fixed TRAIN'):
        tool.check_previous(changed, old, protocol)


def test_fixture_has_only_predeclared_train_images_and_disjoint_image_roles():
    import json
    fixture = json.loads(tool.FIXTURE.read_text())
    names = {role: {r['image'] for r in fixture['samples'] if r['role'] == role} for role in ('fit', 'probe')}
    assert len(names['fit']) == 48 and len(names['probe']) == 16
    assert not names['fit'] & names['probe']
    assert {r['sequence'] for r in fixture['samples']} == set(tool.ready.TRAIN_COUNTS)
    assert all(r['split'] in ('train', 'train_sim') for r in fixture['samples'])


def test_real_pipeline_contract_rejects_non_train_path_before_instantiation():
    cfg = tool.check_cfg()
    specs = tool.fixed_specs(cfg, .5)
    assert all(s['test_mode'] for s in specs)
    assert all(s['pipeline'][1]['transforms'][1]['type'] == 'PortIsotropicShrink' for s in specs)
    cfg.data.train[0]['ann_file'] = 'val/annfiles/'
    with pytest.raises(ValueError, match='Non-TRAIN'):
        tool.fixed_specs(cfg, 1.)


def test_cpu_runner_integration_preserves_boxes_coverage_and_initialization(tmp_path):
    torch.manual_seed(1703)
    records, samples = [], []
    for scale in (1., .5):
        for domain in ('real', 'sim'):
            for role in ('fit', 'probe'):
                name = domain+'_seq01_'+('00000' if role == 'fit' else '00001')
                b = box(); bm = b[:, :5].clone(); bm[:, :4] *= scale
                gt = b[:, :5].clone(); gt[:, 2:4] *= 1.04; gt[:, 4] -= .03
                local = {a: dict(roi=torch.ones(1, 256, 9, 9)*.1, support=torch.ones(1, 1, 9, 9)) for a in g.ARMS}
                records.append(dict(image=name, domain=domain, sequence=domain+'_seq01', role=role,
                    frame_id=0 if role == 'fit' else 1, scale=scale, eligible=True,
                    boxes_original=b, boxes_model=bm, gt_original=gt, local=local,
                    gt_input_short_cells=float(gt[:, 2:4].min()*scale/8),
                    target_residual=[math.log(1.04), math.log(1.04), -.03]))
                if scale == 1.:
                    samples.append(dict(image=name, domain=domain, sequence=domain+'_seq01', role=role, gt=gt[0].tolist()))
    tool.validate_records(records, samples)
    batches = tool.schedule(records)[:2]  # Synthetic CPU integration, not real TRAIN short-fit.
    initial = g.LocalGeometryRefiner().state_dict(); progress = []
    a = tool.fit_arm(records, batches, initial, 'ordinary', torch.device('cpu'), progress.append)
    b = tool.fit_arm(records, batches, initial, 'aligned', torch.device('cpu'), progress.append)
    assert a['initial_state'] == b['initial_state']
    assert a['initial_all_task_gradients_effective'] and a['stem_gradient_after_zero_output_step_effective']
    assert all(x['center_error_px/mean'] == 0 for x in tool.geometry_deltas(a['initial'], a['final']).values())
    assert len([x for x in progress if x['stage'] == 'update']) == 4
    records[0]['target_residual'][0] = .3
    support = tool.support_report(records, dict(minimum_eligible_views={role: {d: 1 for d in ('real', 'sim')} for role in ('fit', 'probe')}))
    assert all(type(n) is int for v in support.values() for n in v['target_outside_bounds_by_component'])
    assert support['fit/real/1.0']['target_outside_bounds_by_component'] == [1, 0, 0]
    # Serialize the entire integration report, including the formerly failing support counts.
    report = dict(train_support=support, arms=dict(ordinary=a, aligned=b),
                  aligned_minus_ordinary_final=tool.geometry_deltas(a['final'], b['final']))
    json.dumps(report, allow_nan=False)
    path = tmp_path/'integration.json'; tool.write_new(path, report)
    restored = json.loads(path.read_text())
    assert restored['train_support'] == support
    assert restored['arms']['ordinary']['initial']['rows'][0]['pred'] == box()[0].tolist()
    records[0]['role'] = 'probe'
    with pytest.raises(ValueError, match='role/order'):
        tool.validate_records(records, samples)


@pytest.mark.parametrize('swap', [False, True])
def test_residual_loss_gradient_signs_match_geometry_and_finite_difference(swap):
    b = box().double()
    gt = b[:, :5].clone(); gt[:, 2] *= 1.1; gt[:, 3] *= .9; gt[:, 4] += .04
    if swap:
        b[:, [2, 3]] = b[:, [3, 2]]; b[:, 4] -= math.pi/2
    residual = torch.zeros(1, 3, dtype=torch.double, requires_grad=True)
    value = g.regression_loss(g.apply_residual(b, residual)[0], gt)[0]
    gradient = torch.autograd.grad(value, residual)[0]
    assert gradient[0, 0] < 0 and gradient[0, 1] > 0 and gradient[0, 2] < 0
    for j in range(3):
        plus = residual.detach().clone(); minus = residual.detach().clone()
        plus[0, j] += 1e-6; minus[0, j] -= 1e-6
        finite = (g.regression_loss(g.apply_residual(b, plus)[0], gt)[0]-
                  g.regression_loss(g.apply_residual(b, minus)[0], gt)[0])/2e-6
        assert float(finite) == pytest.approx(float(gradient[0, j]), abs=1e-6)


def test_changed_runtime_source_binding_fails_before_data_or_gpu(monkeypatch, tmp_path):
    import json
    manifest = json.loads(tool.MANIFEST.read_text())
    manifest['sources']['crane_project/utils/port_geometry_refine_g_v1.py'] = '0'*64
    path = tmp_path/'changed_manifest.json'; path.write_text(json.dumps(manifest))
    monkeypatch.setattr(tool, 'MANIFEST', path)
    with pytest.raises(ValueError, match='Reviewed source SHA differs'):
        tool.checked_inputs()


def test_numeric_report_publication_keeps_numbers_and_rejects_overwrite(tmp_path):
    path = tmp_path/'report.json'
    value = dict(count=np.int64(7), valid=np.bool_(True), loss=np.float32(.25),
                 values=np.asarray([1, 2], dtype=np.int64))
    tool.write_new(path, value)
    result = json.loads(path.read_text()); before = path.read_bytes()
    assert result == dict(count=7, valid=True, loss=.25, values=[1, 2])
    assert type(result['count']) is int and type(result['valid']) is bool
    with pytest.raises(FileExistsError):
        tool.write_new(path, dict(replacement=True))
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('value', [np.float64(float('nan')), np.float32(float('inf')), torch.tensor(1.)])
def test_invalid_report_leaves_no_partial_file(value, tmp_path):
    path = tmp_path/'bad.json'
    with pytest.raises((ValueError, TypeError)):
        tool.write_new(path, dict(value=value))
    assert not path.exists() and not list(tmp_path.iterdir())


def test_atomic_publication_failure_cleans_only_own_temporary(monkeypatch, tmp_path):
    keep = tmp_path/'keep.txt'; keep.write_text('unchanged')
    def fail(*args):
        raise OSError('Simulated publication failure')
    monkeypatch.setattr(tool.os, 'link', fail)
    with pytest.raises(OSError, match='Simulated'):
        tool.write_new(tmp_path/'bad.json', dict(count=np.int64(1)))
    assert list(tmp_path.iterdir()) == [keep] and keep.read_text() == 'unchanged'


def cache_identities():
    manifest = json.loads(tool.MANIFEST.read_text())
    protocol = json.loads(tool.PROTOCOL.read_text()); fixture = json.loads(tool.FIXTURE.read_text())
    current = dict(manifest_sha256=tool.ready.sha(tool.MANIFEST), protocol_sha256=tool.ready.sha(tool.PROTOCOL),
        fixture_sha256=tool.ready.sha(tool.FIXTURE), sources=manifest['sources'], frozen_b=protocol['frozen_b'],
        train_annotation_identities=fixture['train_annotation_identities'])
    old = deepcopy(current); old['manifest_sha256'] = tool.LEGACY_MANIFEST_SHA
    old['sources'] = json.loads(manifest['report_fix_compatible_previous_manifest_json'])['sources']
    return current, old, fixture['samples']


@pytest.mark.parametrize('changed', ['B', 'TRAIN', 'protocol', 'fixture', 'head_source', 'old_runner', 'manifest'])
def test_legacy_cache_rejects_changes_beyond_report_repair(changed):
    current, old, _ = cache_identities()
    assert tool.accepted_cache_identity(current, current) == 'reused_current_cache'
    assert tool.accepted_cache_identity(old, current) == 'reused_legacy_report_only_fix'
    if changed == 'B':
        old['frozen_b']['checkpoint_sha256'] = '0'*64
    elif changed == 'TRAIN':
        old['train_annotation_identities']['train_annotation_sha256'] = '0'*64
    elif changed in ('protocol', 'fixture', 'manifest'):
        old[changed+'_sha256'] = '0'*64
    else:
        path = tool.RUNNER_SOURCE if changed == 'old_runner' else 'crane_project/utils/port_geometry_refine_g_v1.py'
        old['sources'][path] = '0'*64
    with pytest.raises(ValueError, match='identity differs'):
        tool.accepted_cache_identity(old, current)


def test_legacy_compatibility_rejects_altered_embedded_manifest(monkeypatch, tmp_path):
    current, old, _ = cache_identities()
    manifest = json.loads(tool.MANIFEST.read_text())
    manifest['report_fix_compatible_previous_manifest_json'] += '\n'
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(manifest))
    monkeypatch.setattr(tool, 'MANIFEST', path)
    with pytest.raises(ValueError, match='reviewed exact version'):
        tool.accepted_cache_identity(old, current)


@pytest.mark.parametrize('legacy', [False, True])
def test_complete_cache_reuse_preserves_files_and_checks_payload_hashes(legacy, tmp_path):
    current, old, samples = cache_identities(); identity = old if legacy else current
    records = []
    for scale in g.SETTINGS['scales']:
        for sample in samples:
            records.append(dict(image=sample['image'], domain=sample['domain'], sequence=sample['sequence'],
                role=sample['role'], scale=scale, eligible=False, boxes_original=torch.empty(0, 6),
                boxes_model=torch.empty(0, 5), gt_original=g.canonical_boxes(torch.tensor(sample['gt']).reshape(1, 5)),
                local={a: dict(roi=torch.empty(0, 256, 9, 9), support=torch.empty(0, 1, 9, 9)) for a in g.ARMS}))
    payload = dict(identity=identity, records=records, detector_state_before='same', detector_state_after='same')
    path = tmp_path/'local_roi.pt'; torch.save(payload, str(path))
    tool.write_new(tmp_path/'cache_manifest.json', dict(status='COMPLETE', identity=identity, record_count=128,
                   cpu_roi_bytes=tool.cpu_roi_bytes(records), files={'local_roi.pt': tool.ready.sha(path)}))
    before = {p.name: tool.ready.sha(p) for p in tmp_path.iterdir()}
    restored, cache = tool.checked_cache(tmp_path, current, samples)
    tool.validate_records(restored['records'], samples)
    assert restored['identity'] == cache['identity'] == identity
    assert {p.name: tool.ready.sha(p) for p in tmp_path.iterdir()} == before
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='artifact identity'):
        tool.checked_cache(tmp_path, current, samples)


def test_final_serialization_failure_is_guarded_and_publishes_honest_failure(monkeypatch, tmp_path):
    cfg = tool.check_cfg()
    # Deliberately invalid future identity tests the final publication exception path.
    monkeypatch.setattr(tool, 'checked_inputs', lambda: (cfg, {}, [], dict(bad=np.float64(float('nan')))))
    out = tmp_path/'failed.json'
    monkeypatch.setattr(tool.sys, 'argv', ['preflight', '--check-only', '--out-json', str(out), '--cache-dir', str(tmp_path/'cache')])
    with pytest.raises(ValueError, match='JSON compliant'):
        tool.main()
    report = json.loads(out.read_text())
    assert report['status'] == 'FAILED_REVIEW_REQUIRED' and not report['full_report_available']
    assert report['head_updates_total'] == 0 and report['formal_training'] is False
    artifact = json.loads(out.with_suffix('.artifacts.json').read_text())
    assert artifact['status'] == report['status']
    assert artifact['files'][str(out)] == tool.ready.sha(out)
