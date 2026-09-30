import importlib.util
from pathlib import Path
import math
import numpy as np
import pytest
from copy import deepcopy

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('port_geom',ROOT/'crane_project/tools/audit_port_train_val_geometry_v1.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def row(name,gt,pred):
    return dict(image=name,image_sha256=name,gt=list(gt),domain='sim',sequence='sim_seq10',
        frame_id=int(name),split='val',input_gt_short_px=20.,gt_aspect=2.,gt_angle_deg=0.,
        metrics=m.decompose(gt,pred))


def test_equivalent_width_height_swap_and_pi_wrap():
    gt=[10,10,40,20,math.radians(89)]
    pred=[10,10,20,40,math.radians(-1)]
    result=m.decompose(gt,pred)
    assert result['angle_error_deg'] < 1e-8
    assert result['raw_representation_angle_error_deg'] > 89
    assert result['riou'] > .999
    assert result['long_edge_relative_error'] == 0
    assert m.angle_error(math.radians(89),math.radians(-89)) < 2.001


def test_center_penalty_not_mislabelled_angle_error():
    gt=[0,0,40,20,0]
    a=row('1',gt,[11,0,40,20,0])
    b=row('2',gt,None)
    c=row('3',gt,[0,0,40,20,0])
    summary=m.summarize([a,b,c])
    assert a['metrics']['angle_error_deg']==0
    assert a['metrics']['protocol_angle_error_deg']==90
    assert summary['center_penalty_count']==1
    assert summary['no_output_penalty_count']==1
    assert summary['protocol_angle_squared_error_penalty_fraction']==1
    assert summary['angle_center_valid']['n']==1
    assert abs(summary['protocol_angle']['rmse']-math.sqrt(5400))<1e-9


def test_counterfactual_localization_and_size():
    gt=[30,30,40,20,.2]
    result=m.decompose(gt,[40,30,40,20,.2])
    assert result['riou_gain_if_gt_center']>0
    assert abs(result['riou_gain_if_gt_size'])<1e-8
    assert abs(result['riou_gain_if_gt_angle'])<1e-8
    result=m.decompose(gt,[30,30,80,40,.2])
    assert result['riou_gain_if_gt_size']>.7


def test_pairing_conditions_on_same_outputs():
    gt=[0,0,40,20,0]
    a=[row('1',gt,None),row('2',gt,gt)]
    b=[row('1',gt,gt),row('2',gt,gt)]
    result=m.paired_report(a,b)
    assert result['common_outputs']['frames']==1
    assert result['output_churn']['symeood_only_output']==1
    assert result['eood']['output_coverage_pct']==50
    assert result['common_outputs']['eood']['riou']['mean']>.999


def test_disjoint_frame_ids_do_not_make_contiguous_run():
    gt=[0,0,40,20,0]
    assert m.longest_failure([row('1',gt,None),row('3',gt,None)])==1


def test_square_angle_is_marked_and_sampling_is_paired():
    assert not m.decompose([0,0,20,20,0],[0,0,20,20,.1])['angle_axis_well_defined']
    class Dataset:
        data_infos=[dict(filename=f'{domain}_{seq}_{i:05d}.jpg')
                    for domain,seq in [('real','seq01'),('real','seq05'),('sim','seq08')]
                    for i in range(10)]
    first=m.pick_indices(Dataset(),4,1701)
    assert first==m.pick_indices(Dataset(),4,1701)
    assert len(first)==8
    assert sum(i<20 for i in first)==4
    assert sum(i>=20 for i in first)==4


def test_protocol_formula_matches_project_evaluator():
    from crane_project.tools.eval_crane_offline import CraneOfflineEvaluator
    gt=np.array([0,0,40,20,0.])
    predictions=[np.array([0,0,40,20,.03]),np.array([11,0,40,20,0.]),None]
    rows=[row(str(i+1),gt,pred) for i,pred in enumerate(predictions)]
    records=[dict(domain='sim',seq_id='seq10',frame_id=i+1,gt_box=gt,
                  pred_box=pred,plc_rope=None) for i,pred in enumerate(predictions)]
    official=CraneOfflineEvaluator().evaluate_records(records)
    assert abs(m.summarize(rows)['protocol_angle']['rmse']-official['sim/A-RMSE(deg)'])<.0001


def test_real_merged_configs_resolve_actual_head_contract_without_mutation():
    mmcv = pytest.importorskip('mmcv')
    from mmrotate.models import build_detector
    for arm, expected_max in [('eood', 2000), ('symeood', 1)]:
        cfg = mmcv.Config.fromfile(str(ROOT / 'crane_project/configs' /
            ('crane_' + arm + '_k1_port_day2night_aug_b_v1.py')))
        before = deepcopy(cfg.model)
        contract = m.inference_contract(cfg.model)
        assert contract['max_per_img'] == expected_max
        assert cfg.model == before
        # Exercise the same constructors used on the server, without weights/GPU.
        model_cfg = deepcopy(cfg.model)
        model_cfg.pretrained = None
        model_cfg.train_cfg = None
        model = build_detector(model_cfg)
        head = model.bbox_head.predictors[0] if arm == 'eood' else model.bbox_head
        assert head.test_cfg.score_thr == .05
        assert head.test_cfg.max_per_img == expected_max
        del model
        invalid = deepcopy(before)
        target = invalid.bbox_head.predictors[0].test_cfg if arm == 'eood' else invalid.test_cfg
        target.score_thr = .01
        with pytest.raises(ValueError, match='frozen inference contract'):
            m.inference_contract(invalid)


def test_eood_multiple_outputs_keep_first_and_validate_all_export_rows(tmp_path):
    from crane_project.tools.ckpt_sweep import pkl_to_dota
    import pickle
    a = np.array([[30.,30.,40.,20.,.1,.9], [90.,80.,35.,15.,-.2,.6]])
    predictions = [[a], [np.empty((0,6))]]
    pkl = tmp_path / 'results.pkl'
    with pkl.open('wb') as stream:
        pickle.dump(predictions, stream)
    folder = Path(pkl_to_dota(str(pkl), ['real_seq07_00001','real_seq07_00002'], str(tmp_path)))
    exported = m.parse_dota_txt(str(folder / 'real_seq07_00001.txt'))
    assert m.raw_box([a], 2000) == a[0].tolist()
    m.validate_export([a], exported, 2000)
    m.validate_export(predictions[1], [], 2000)
    assert m.raw_box(predictions[1], 2000) is None
    with pytest.raises(ValueError, match='export mismatch'):
        m.validate_export([a], exported[::-1], 2000)
    with pytest.raises(ValueError, match='output limit'):
        m.raw_box([a], 1)
    with pytest.raises(ValueError, match='sorted by score'):
        m.raw_box([a[::-1]], 2000)
