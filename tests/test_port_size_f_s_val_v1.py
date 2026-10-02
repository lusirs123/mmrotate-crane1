"""F-S fixed identities, original VAL gates, and shared ordinary-real tail gates."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import pytest
import torch

from crane_project.tools import compare_port_size_f_s_val_v1 as compare
from crane_project.tools.audit_port_train_val_geometry_v1 import decompose


def row(name, center=0., output=True):
    sequence, number = name.rsplit('_', 1)
    gt = [0., 0., 40., 20., .2]
    pred = [center, 0., 40., 20., .2, .9] if output else None
    return dict(image=name, domain='sim' if sequence.startswith('sim') else 'real',
        sequence=sequence, frame_id=int(number), split='val', image_sha256='CPU fixture',
        gt=gt, pred=pred, input_gt_short_px=20., gt_aspect=2., gt_angle_deg=math.degrees(.2),
        metrics=decompose(gt, pred[:5] if pred else None))


def selection(config, fixed, epoch='epoch_24'):
    return dict(evidence_role='source_val_checkpoint_selection', candidate_epochs=compare.EPOCHS,
        selected_checkpoint=epoch, config_sha256=compare.sha(config),
        selection_config=fixed['selection_config'], metric_protocol_version=fixed['metric_protocol_version'],
        center_thresh_px=15., source_val_annotations_sha256=fixed['frozen_b']['annotation_sha256'],
        selection_info=dict(selection='fallback_fixture'),
        all_checkpoints={'epoch_'+str(x):dict(metrics=dict(fixture=True)) for x in compare.EPOCHS})


def test_formal_config_and_existing_preflight_sources_stay_identical():
    from crane_project.tools.preflight_port_size_f_s_v1 import check_configs
    _, candidate = check_configs()
    b, formal = compare.check_configs()
    assert candidate.to_dict() == formal.to_dict()
    assert b.load_from is b.resume_from is None
    fixed = compare.read_protocol()
    assert len(fixed['sources']) == 41 and len(fixed['gate_names']) == 17


def test_contract_cli_reuses_reviewed_report_without_weights_gpu_or_overwrite(tmp_path, monkeypatch):
    evidence = compare.ROOT/'work_dirs/port_size_f_s_v1_train_preflight_server_20261001.json'
    out = tmp_path/'contract.json'
    monkeypatch.setattr(sys, 'argv', ['compare', '--check-only', '--train-preflight', str(evidence), '--out-json', str(out)])
    monkeypatch.setattr(compare, 'load_arm', lambda *a: pytest.fail('No weight or VAL loading in contract check'))
    compare.main()
    result = json.loads(out.read_text())
    assert result['resolved_formal_equals_reviewed_candidate'] and result['total_batch'] == 4
    assert result['optimizer_steps'] == 0 and not result['formal_training_executed']
    with pytest.raises(FileExistsError): compare.main()
    changed = tmp_path/'changed.json'
    changed.write_text(evidence.read_text()+'\n')
    with pytest.raises(ValueError, match='reviewed zero-step'): compare.check_formal_contract(changed)
    monkeypatch.setattr(compare, 'library_contract', lambda: dict(fixture='changed environment'))
    monkeypatch.setattr(sys, 'argv', ['compare', '--check-only', '--require-reviewed-library',
        '--train-preflight', str(evidence), '--out-json', str(tmp_path/'strict.json')])
    with pytest.raises(ValueError, match='server library contract'): compare.main()
    assert not (tmp_path/'strict.json').exists()


def test_source_or_selection_design_mutation_is_rejected(tmp_path, monkeypatch):
    fixed = json.loads(compare.PROTOCOL.read_text())
    fixed['sources'][str(compare.EXPERIMENT.relative_to(compare.ROOT))] = 'wrong'
    path = tmp_path/'protocol.json'; path.write_text(json.dumps(fixed))
    monkeypatch.setattr(compare, 'PROTOCOL', path)
    with pytest.raises(ValueError, match='sources differ'): compare.read_protocol()
    fixed['selection_config']['mcml_limit'] = 6
    path.write_text(json.dumps(fixed))
    with pytest.raises(ValueError, match='protocol differs'): compare.read_protocol()


def test_denominators_and_tail_baselines_use_shared_outputs():
    b = [row('real_seq07_00000', 2.), row('real_seq07_00001', 100.), row('real_seq07_00002', output=False),
         row('sim_seq10_00000')]
    f = [row('real_seq07_00000', 3.), row('real_seq07_00001', output=False), row('real_seq07_00002', output=False),
         row('sim_seq10_00000')]
    result = compare.conditions(b, f, [])
    assert len(result['original_checks']) == 15 and len(result['checks']) == 17
    assert result['coverage']['b']['real']['output_center_hit_pct'] == 50.
    assert result['coverage']['b']['real']['output_coverage_pct'] == pytest.approx(200/3)
    assert result['coverage']['b']['real']['all_frame_center_correct_coverage_pct'] == pytest.approx(100/3)
    ordinary = result['ordinary_real_shared']
    assert ordinary['n'] == 1 and ordinary['summaries']['b']['center_error_px']['mean'] == 2.
    assert ordinary['summaries']['f_s']['center_error_px']['rmse'] == 3.
    assert not any(result['additional_tail_checks'].values())
    assert compare.conditions(b, f, ['real_seq07_00000'])['ordinary_real_shared']['n'] == 0


def test_original_gates_and_tail_replayed_on_historical_e_h_evidence():
    saved = json.loads((compare.ROOT/'work_dirs/port_shape_e_h_v1_val_compare_server_20261001.json').read_text())
    # Historical E-H is only a regression fixture, never a new F-S result.
    b, e = saved['geometry']['rows']['b'], saved['geometry']['rows']['e_h']
    result = compare.conditions(b, e, saved['geometry']['historical_severe_b_frames'])
    assert result['original_checks'] == saved['pre_registered_conditions']['checks']
    assert result['ordinary_real_shared']['n'] == 364
    assert not any(result['additional_tail_checks'].values())
    same = compare.conditions(b, b, saved['geometry']['historical_severe_b_frames'])
    assert all(same['additional_tail_checks'].values())
    assert same['ordinary_real_shared']['summaries']['b']['center_error_px']['rmse'] == pytest.approx(5.2923, abs=1e-4)
    with pytest.raises(ValueError, match='pairing'): compare.conditions(b, list(reversed(e)), [])


@pytest.mark.parametrize('change', ['epoch', 'coefficient_protocol', 'config', 'center', 'annotation', 'missing_candidate'])
def test_selection_rejects_reselection_protocol_and_identity_changes(change):
    fixed = compare.read_protocol()
    b, f = selection(compare.CONTROL, fixed), selection(compare.EXPERIMENT, fixed, 'epoch_22')
    compare.validate_selections(b, f, fixed)
    if change == 'epoch': b['selected_checkpoint'] = 'epoch_22'
    elif change == 'coefficient_protocol': f['selection_config'] = dict(f['selection_config'], mcml_limit=6)
    elif change == 'config': f['config_sha256'] = compare.sha(compare.CONTROL)
    elif change == 'center': f['center_thresh_px'] = 16.
    elif change == 'annotation': f['source_val_annotations_sha256'] = 'wrong'
    else: del f['all_checkpoints']['epoch_24']
    with pytest.raises(ValueError): compare.validate_selections(b, f, fixed)


def meta(cfg, epoch):
    fields = ('model', 'data', 'optimizer', 'optimizer_config', 'lr_config', 'runner', 'load_from', 'resume_from')
    return dict(config='\n'.join(k+' = '+repr(cfg.to_dict()[k]) for k in fields), seed=0, epoch=epoch, iter=640*epoch)


def test_safe_checkpoint_metadata_contract_rejects_resume_seed_and_duplicate():
    b, f = compare.check_configs()
    assert compare.checkpoint_contract(meta(f, 22), f, 22)['status'] == 'MATCH'
    with pytest.raises(ValueError): compare.checkpoint_contract(meta(b, 22), f, 22)
    bad = meta(f, 22); bad['seed'] = 1
    with pytest.raises(ValueError): compare.checkpoint_contract(bad, f, 22)
    bad = meta(f, 22); bad['config'] += '\nresume_from = "epoch_24.pth"'
    with pytest.raises(ValueError, match='Duplicate'): compare.checkpoint_contract(bad, f, 22)


def test_load_arm_checks_real_provenance_and_cpu_metadata(tmp_path, monkeypatch):
    from crane_project.tools.ckpt_sweep import check_or_record_prediction
    _, f = compare.check_configs()
    checkpoint = tmp_path/'epoch_22.pth'; torch.save(dict(meta=meta(f, 22)), checkpoint)
    pkl = tmp_path/'results.pkl'; pkl.write_bytes(b'Synthetic loader fixture')
    rows = [row('real_seq07_'+str(i)) for i in range(375)] + [row('sim_seq10_'+str(i)) for i in range(512)]
    identity = dict(checkpoint=str(checkpoint), pkl=str(pkl))
    monkeypatch.setattr(compare, 'load_frozen_val', lambda *a: (rows, deepcopy(identity)))
    with pytest.raises(RuntimeError, match='provenance'): compare.load_arm(f, compare.EXPERIMENT, tmp_path, 'epoch_22')
    ann = compare.ROOT/'crane_project/data/crane_grab_port_day2night_v1/val/annfiles'
    check_or_record_prediction(str(compare.EXPERIMENT), str(checkpoint), str(pkl), str(ann), 'source_val', generated=True)
    actual, got = compare.load_arm(f, compare.EXPERIMENT, tmp_path, 'epoch_22')
    assert len(actual) == 887 and got['training_contract']['iter'] == 14080


def test_actual_f_s_val_cache_loader_with_real_annotations_and_synthetic_predictions(tmp_path):
    import pickle
    import numpy as np
    from mmrotate.datasets import build_dataset
    from crane_project.tools.ckpt_sweep import (
        annotation_set_sha256, check_or_record_prediction, pkl_to_dota)
    _, f = compare.check_configs()
    spec = deepcopy(f.data.val); spec.test_mode = True
    dataset = build_dataset(spec)
    # GT predictions validate only plumbing, not model performance.
    predictions = [[np.column_stack((dataset.get_ann_info(i)['bboxes'], [.9]))]
                   for i in range(len(dataset))]
    pred_dir = tmp_path/'preds'; pred_dir.mkdir()
    pkl = pred_dir/'results.pkl'
    with pkl.open('wb') as stream: pickle.dump(predictions, stream)
    checkpoint = tmp_path/'epoch_22.pth'; torch.save(dict(meta=meta(f, 22)), checkpoint)
    ann = compare.ROOT/'crane_project/data/crane_grab_port_day2night_v1/val/annfiles'
    check_or_record_prediction(str(compare.EXPERIMENT), str(checkpoint), str(pkl), str(ann), 'source_val', generated=True)
    pkl_to_dota(str(pkl), [Path(info['filename']).stem for info in dataset.data_infos], str(pred_dir))
    selected = dict(evidence_role='source_val_checkpoint_selection', selected_checkpoint='epoch_22',
        config_sha256=compare.sha(compare.EXPERIMENT), selected_path=str(checkpoint),
        source_val_annotations_sha256=annotation_set_sha256(str(ann)),
        all_checkpoints=dict(epoch_22=dict(checkpoint=str(checkpoint),
            checkpoint_sha256=compare.sha(checkpoint), results_pkl=str(pkl), results_pkl_sha256=compare.sha(pkl))))
    (tmp_path/'sweep_results.json').write_text(json.dumps(selected))
    rows, identity = compare.load_arm(f, compare.EXPERIMENT, tmp_path, 'epoch_22')
    assert len(rows) == 887 and all(r['metrics']['riou'] > .999 for r in rows)
    assert identity['training_contract']['status'] == 'MATCH'
    text = next((pred_dir/'Task1_grab').glob('*.txt'))
    text.write_text('changed export')
    with pytest.raises((ValueError, RuntimeError)): compare.load_arm(f, compare.EXPERIMENT, tmp_path, 'epoch_22')


def test_comparison_cli_keeps_selection_and_report_names_without_inference(tmp_path, monkeypatch):
    b, f = compare.check_configs(); fixed = deepcopy(compare.read_protocol())
    fixed['historical_severe_b_frames'] = []
    bs, fs = tmp_path/'b', tmp_path/'f'; bs.mkdir(); fs.mkdir()
    for folder, config in ((bs, compare.CONTROL), (fs, compare.EXPERIMENT)):
        (folder/'sweep_results.json').write_text(json.dumps(selection(config, fixed)))
    fixed['frozen_b']['selection_sha256'] = compare.sha(bs/'sweep_results.json')
    rows = [row('real_seq07_'+str(i)) for i in range(375)] + [row('sim_seq10_'+str(i)) for i in range(512)]
    monkeypatch.setattr(compare, 'check_formal_contract', lambda *a: (b, f, fixed, dict(fixture=True)))
    monkeypatch.setattr(compare, 'load_arm', lambda *a: (deepcopy(rows), deepcopy(fixed['frozen_b'])))
    monkeypatch.setattr(compare, 'riou_crosscheck', lambda r: dict(above_1e_3=0, half_threshold_changes=0))
    out = tmp_path/'compare.json'
    monkeypatch.setattr(sys, 'argv', ['compare', '--b-sweep', str(bs), '--f-sweep', str(fs), '--out-json', str(out)])
    compare.main(); result = json.loads(out.read_text())
    assert result['selection_info']['selection'] == 'fallback_fixture'
    assert len(result['pre_registered_conditions']['checks']) == 17
    assert 'f_s' in result['geometry']['rows'] and 'd' not in result['geometry']['rows']
    assert result['pre_registered_conditions']['coverage']['f_s']['real']['frames'] == 375
    assert result['real_zero_riou_outputs_by_sequence']['f_s'] == {}
    with pytest.raises(FileExistsError): compare.main()
