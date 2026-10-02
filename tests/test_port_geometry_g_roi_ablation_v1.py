"""CPU regressions for donor isolation and the unchanged G fit/evaluation path."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from crane_project.tools import preflight_port_geometry_g_roi_ablation_v1 as a


def synthetic():
    samples = [dict(image=domain+'_seq01_'+str(j)+'_'+role, domain=domain, role=role,
                    sequence=domain+'_seq01', frame_id=j)
               for role in ('fit', 'probe') for domain in ('real', 'sim') for j in range(4)]
    records = []
    for scale in (1., .5):
        for i, sample in enumerate(samples):
            b = torch.tensor([[32.+i, 24., 32., 16., .17, .8]])
            bm = b[:, :5].clone(); bm[:, :4] *= scale
            gt = b[:, :5].clone(); gt[:, 2:4] *= 1.04; gt[:, 4] -= .03
            sample['gt'] = gt[0].tolist()
            local = {arm: dict(roi=torch.full((1, 256, 9, 9), .02+i*.003+scale*.001),
                               support=torch.ones(1, 1, 9, 9)) for arm in a.g.ARMS}
            records.append(dict(sample, scale=scale, eligible=True,
                boxes_original=b, boxes_model=bm, gt_original=gt, local=local,
                parsed_gt_original=gt[0].tolist(), reference_gt_absolute_delta=[0.]*5,
                gt_input_short_cells=float(gt[:, 2:4].min()*scale/8),
                target_residual=[.0392207, .0392207, -.03]))
    return records, samples


def test_fixed_fixture_derangement_is_bijective_and_role_domain_isolated():
    samples = json.loads(a.g.FIXTURE.read_text())['samples']
    first = a.donor_map(samples)
    assert first == a.donor_map(samples)
    assert first == json.loads(a.PROTOCOL.read_text())['donor_mapping']
    assert len(first) == 64 and len(set(first.values())) == 64
    lookup = {s['image']: s for s in samples}
    for recipient, donor in first.items():
        assert recipient != donor
        assert (lookup[recipient]['role'], lookup[recipient]['domain']) == (lookup[donor]['role'], lookup[donor]['domain'])


def test_adapter_moves_only_roi_and_mask_without_tensor_copy_or_metadata_change():
    records, samples = synthetic(); mapping = a.donor_map(samples)
    prepared, assignments = a.assemble_records(records, samples, mapping)
    lookup = {(r['image'], r['scale']): r for r in records}
    for old, current, assignment in zip(records, prepared, assignments):
        donor = lookup[(mapping[old['image']], old['scale'])]
        for k in old:
            if k != 'local':
                assert current[k] is old[k]
        for k in ('roi', 'support'):
            assert current['local']['matched'][k] is old['local']['ordinary'][k]
            assert current['local']['shuffled'][k] is donor['local']['ordinary'][k]
        assert assignment['donor'] == mapping[old['image']] and assignment['scale'] == donor['scale']
        roi, mask, b, bm, gt = a.g.batch_tensors(prepared, [records.index(old)], 'shuffled', torch.device('cpu'))
        assert torch.equal(roi, donor['local']['ordinary']['roi'])
        assert torch.equal(mask, donor['local']['ordinary']['support'])
        assert torch.equal(b, old['boxes_original']) and torch.equal(bm, old['boxes_model'])
        assert torch.equal(gt, old['gt_original'])
    assert all(set(r['local']) == set(a.g.ARMS) for r in records)
    donors = {(x['recipient'], x['scale']): x['donor'] for x in assignments}
    assert all(donors[(s['image'], 1.)] == donors[(s['image'], .5)] for s in samples)


@pytest.mark.parametrize('change', ['self', 'role', 'domain', 'missing', 'ineligible', 'duplicate_view'])
def test_donor_or_cache_substitution_is_rejected(change):
    records, samples = synthetic(); mapping = a.donor_map(samples)
    if change == 'self':
        mapping[samples[0]['image']] = samples[0]['image']
    elif change in ('role', 'domain'):
        mapping[samples[0]['image']] = next(s['image'] for s in samples if s[change] != samples[0][change])
    elif change == 'missing':
        records[0]['boxes_original'] = torch.empty(0, 6)
    elif change == 'ineligible':
        records[0]['eligible'] = False
    else:
        records[-1] = records[0]
    with pytest.raises(ValueError):
        a.assemble_records(records, samples, mapping)


def test_two_condition_cpu_fit_replays_reference_and_preserves_recipient_boxes(tmp_path):
    records, samples = synthetic()
    prepared, assignments = a.assemble_records(records, samples, a.donor_map(samples))
    a.g.seed_all(); initial = a.g.LocalGeometryRefiner().state_dict()
    batches = a.g.schedule(records)[:2]
    before = [r['boxes_original'].clone() for r in records]
    reference = a.g.fit_arm(records, batches, initial, 'ordinary', torch.device('cpu'), lambda _: None)
    progress = []
    matched = a.g.fit_arm(prepared, batches, initial, 'matched', torch.device('cpu'), progress.append)
    a.verify_matched_result(matched, dict(arms=dict(ordinary=reference)))
    shuffled = a.g.fit_arm(prepared, batches, initial, 'shuffled', torch.device('cpu'), progress.append)
    assert matched['initial'] == shuffled['initial']
    assert matched['initial_state'] == shuffled['initial_state']
    assert all(result['initial_all_task_gradients_effective'] and
               result['stem_gradient_after_zero_output_step_effective'] for result in (matched, shuffled))
    for result in (matched, shuffled):
        for old, current in zip(result['initial']['rows'], result['final']['rows']):
            assert [old['pred'][j] for j in (0, 1, 5)] == [current['pred'][j] for j in (0, 1, 5)]
    assert all(torch.equal(b, r['boxes_original']) for b, r in zip(before, records))
    assert len([v for v in progress if v['stage'] == 'update']) == 4
    delta = a.g.geometry_deltas(shuffled['final'], matched['final'])
    path = tmp_path/'report.json'
    a.g.write_new(path, dict(assignments=assignments, matched=matched, shuffled=shuffled, delta=delta))
    assert json.loads(path.read_text())['matched'] == matched
    changed = deepcopy(matched); changed['final']['rows'][0]['pred'][2] += .01
    with pytest.raises(ValueError, match='Matched replay differs'):
        a.verify_matched_result(changed, dict(arms=dict(ordinary=reference)))


def test_source_change_is_rejected_before_train_data_or_gpu(monkeypatch, tmp_path):
    manifest = json.loads(a.MANIFEST.read_text())
    manifest['sources']['crane_project/utils/port_geometry_refine_g_v1.py'] = '0'*64
    path = tmp_path/'bad_sources.json'; path.write_text(json.dumps(manifest))
    monkeypatch.setattr(a, 'MANIFEST', path)
    with pytest.raises(ValueError, match='source SHA differs'):
        a.checked_contract(Path('/nonexistent/baseline.json'))


def test_runtime_change_fails_before_cuda_fitting():
    runtime = dict(python=a.sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), numpy=a.np.__version__, opencv=a.g.cv2.__version__)
    assert a.checked_runtime(dict(runtime=runtime))['torch'] == runtime['torch']
    runtime['torch'] = 'different'
    with pytest.raises(ValueError, match='runtime differs'):
        a.checked_runtime(dict(runtime=runtime))


def test_static_cli_publishes_atomic_report_and_forbids_overwrite(monkeypatch, tmp_path):
    out = tmp_path/'static'
    protocol = a.protocol_document(json.loads(a.g.FIXTURE.read_text())['samples'])
    monkeypatch.setattr(a, 'checked_contract', lambda _: (protocol, [], dict(test=True), {}))
    monkeypatch.setattr(a.sys, 'argv', ['ablation', '--check-only', '--out-dir', str(out)])
    a.main()
    report = json.loads((out/'completion.json').read_text())
    assert report['status'] == 'STATIC_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES'
    assert report['head_updates_total'] == report['detector_updates'] == report['detector_forward_calls'] == 0
    assert not report['formal_training'] and not report['checkpoint_exported']
    artifact = json.loads((out/'artifacts.json').read_text())
    assert all(a.g.ready.sha(out/p) == h for p, h in artifact['files'].items())
    before = (out/'completion.json').read_bytes()
    with pytest.raises(FileExistsError):
        a.main()
    assert (out/'completion.json').read_bytes() == before


def test_report_serialization_failure_leaves_new_protocol_failed_record(monkeypatch, tmp_path):
    protocol = a.protocol_document(json.loads(a.g.FIXTURE.read_text())['samples'])
    monkeypatch.setattr(a, 'checked_contract', lambda _: (protocol, [], dict(bad=float('nan')), {}))
    out = tmp_path/'failed'
    monkeypatch.setattr(a.sys, 'argv', ['ablation', '--check-only', '--out-dir', str(out)])
    with pytest.raises(ValueError, match='JSON compliant'):
        a.main()
    report = json.loads((out/'completion.json').read_text())
    assert report['protocol'] == a.VERSION and report['status'] == 'FAILED_REVIEW_REQUIRED'
    assert not report['full_report_available'] and report['head_updates_total'] == 0


@pytest.mark.parametrize('replay_ok', [True, False])
def test_full_entry_cpu_simulation_archives_previews_and_stops_on_replay_failure(monkeypatch, tmp_path, replay_ok):
    # Exercise the complete entry with synthetic CPU records; CUDA orchestration
    # is mocked explicitly, and the simulated two-step budget is not real TRAIN.
    records, samples = synthetic()
    monkeypatch.setitem(a.g.SETTINGS, 'steps_per_arm', 2)
    a.g.seed_all(); initial = a.g.LocalGeometryRefiner().state_dict()
    batches = a.g.schedule(records)
    original_fit = a.g.fit_arm
    reference = original_fit(records, batches, initial, 'ordinary', torch.device('cpu'), lambda _: None)
    if not replay_ok:
        reference['final_state']['parameters'] = 'different'
    cache_dir = tmp_path/'cache'; (cache_dir/'previews').mkdir(parents=True)
    files = {}
    for i in range(12):
        path = cache_dir/'previews'/('support_%02d.png' % i)
        path.write_bytes(b'synthetic PNG placeholder '+str(i).encode())
        files[str(path.relative_to(cache_dir))] = a.g.ready.sha(path)
    cache = dict(status='COMPLETE', identity=dict(synthetic=True), record_count=len(records), files=files)
    a.g.write_new(cache_dir/'cache_manifest.json', cache)
    monkeypatch.setattr(a, 'CACHE_MANIFEST_SHA', a.g.ready.sha(cache_dir/'cache_manifest.json'))
    protocol = a.protocol_document(samples)
    baseline = dict(cache=cache, minibatches=batches, arms=dict(ordinary=reference))
    monkeypatch.setattr(a, 'checked_contract', lambda _: (protocol, samples, dict(g_identity={}), baseline))
    monkeypatch.setattr(a, 'checked_runtime', lambda _: dict(evidence='CPU_simulation_only'))
    monkeypatch.setattr(a.g, 'checked_cache', lambda *_: (dict(records=records), cache))
    monkeypatch.setattr(a.g, 'measured', lambda operation, _: (operation(), dict(evidence='CPU_simulation_only')))
    monkeypatch.setattr(a.g, 'fit_arm', lambda rec, bat, init, condition, _device, prog:
                        original_fit(rec, bat, init, condition, torch.device('cpu'), prog))
    monkeypatch.setattr(a.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(a.torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(a.torch.cuda, 'set_device', lambda _: None)
    monkeypatch.setattr(a.torch.cuda, 'manual_seed_all', lambda _: None)
    monkeypatch.setattr(a.torch.cuda, 'get_device_name', lambda _: 'CPU simulation only')
    out = tmp_path/'run'
    monkeypatch.setattr(a.sys, 'argv', ['ablation', '--cache-dir', str(cache_dir), '--out-dir', str(out)])
    if replay_ok:
        a.main()
    else:
        with pytest.raises(ValueError, match='Matched replay differs'):
            a.main()
    report = json.loads((out/'completion.json').read_text())
    assert report['head_updates_total'] == (4 if replay_ok else 2)
    assert report['status'] == ('TRAIN_ROI_ABLATION_COMPLETE_REVIEW_REQUIRED' if replay_ok else 'FAILED_REVIEW_REQUIRED')
    assert (out/'matched.json').exists() and (out/'shuffled.json').exists() == replay_ok
    assert len(list((out/'previews').glob('*.png'))) == 12
    assert all(a.g.ready.sha(out/p) == digest for p, digest in files.items())
    artifact = json.loads((out/'artifacts.json').read_text())
    assert all(a.g.ready.sha(out/p) == h for p, h in artifact['files'].items())
