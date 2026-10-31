"""R4 midpoint identity, artifact layout and ablation audit (stdlib only)."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import runpy

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'symeood_r4_midpoint_v1'
DEFAULT_ROOT = ROOT/'work_dirs/symeood_r4_adaptation_20261031'
CONTRACT = ROOT/'crane_project/data_contracts/symeood_r4_midpoint_v1.json'
R4_CONFIG = 'work_dirs/symeood_110_reg_tower_20261030_v1/config.py'
R4_PROTOCOL = 'work_dirs/symeood_110_reg_tower_20261030_v1/protocol.json'
R4_SELECTION = 'work_dirs/symeood_110_reg_tower_20261030_v1/R4/ckpt_sweep/sweep_results.json'
R4_SELECTION_MIRROR = 'work_dirs/symeood_110_reg_tower_20261030_v1/downloaded_result/R4/ckpt_sweep/sweep_results.json'
R4_SELECTION_SHA = '2defb0d6cb16c07d413b86c48a9e704338f7cc675b13d8653a837099b192ebb3'
R4_CHECKPOINT = 'work_dirs/symeood_110_reg_tower_20261030_v1/R4/epoch_16.pth'
R4_SHA = 'b5ee405108970552717b3b7c8f0e56c6813232e19aebccff82a5c7183511b13b'
R4_CONFIG_SHA = 'b5045f42a30c8876acb6a3f8510d4be3206c93e45c51491266ed9939232dac4a'
S_CONFIG = 'work_dirs/symeood_110_retrain_seed0_20261026_v1/configs/Coff_seed0.py'
E_CONFIG = 'work_dirs/crane_eood_k1_port_day2night_aug_b_v1/crane_eood_k1_port_day2night_aug_b_v1.py'
EA_CONFIG = 'work_dirs/eood_A_geometry_20261030_v1/candidate_config.py'
EB_CONFIG = 'work_dirs/eood_B_regression_20261031_v3/candidate_config.py'
META_FIELDS = ('model', 'data', 'optimizer', 'optimizer_config', 'lr_config', 'runner', 'load_from', 'resume_from')
MEMORY = dict(allocator_mib=6144, process_stop_mib=6656, startup_free_mib=6656,
              minimum_free_mib=512, custom_temperature_stop=None)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def config(path):
    return {k: v for k, v in runpy.run_path(str(ROOT/path)).items() if not k.startswith('__')}


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def selection_receipt_path():
    # Same bytes, two locations: original server path or verified local download.
    for name in (R4_SELECTION, R4_SELECTION_MIRROR):
        path = ROOT/name
        if path.exists():
            if sha(path) != R4_SELECTION_SHA:
                raise ValueError('R4 selection receipt SHA differs: '+name)
            return path
    raise ValueError('R4 original VAL selection receipt is missing')


def frontend_identity():
    protocol = json.loads((ROOT/R4_PROTOCOL).read_text())
    selected = json.loads(selection_receipt_path().read_text())
    cp = selected['all_checkpoints']['epoch_16']
    if (protocol['protocol'] != 'symeood_110_r4_single_structure_v1'
            or selected['selected_checkpoint'] != 'epoch_16'
            or cp['checkpoint_sha256'] != R4_SHA
            or protocol['config_sha256'] != R4_CONFIG_SHA
            or selected['config_sha256'] != R4_CONFIG_SHA
            or sha(ROOT/R4_CONFIG) != R4_CONFIG_SHA
            or protocol['selection'] != selected['selection_config']
            or selected['evidence_role'] != 'source_val_checkpoint_selection'):
        raise ValueError('R4 original VAL-selected ep16 identity differs')
    return dict(identity='SymEOOD-R4-seed0-ep16-20261030', selected_epoch='epoch_16',
        checkpoint_sha256=R4_SHA, config_sha256=R4_CONFIG_SHA,
        annotation_sha256=selected['source_val_annotations_sha256'],
        selection_basis='original_VAL_rule_before_TEST', status='DOWNSTREAM_CANDIDATE_NOT_ADOPTED')


def artifact_paths(root, attempt):
    root = Path(root).resolve()
    if ROOT/'work_dirs' not in root.parents or not re.fullmatch(r'attempt_[0-9]{3}', attempt):
        raise ValueError('Use one experiment root under work_dirs and attempt_NNN children')
    parent = root/'midpoint'/attempt
    return dict(root=root, preparation=root/'preparation', attempt=parent,
                cache=parent/'roi_cache', verify=parent/'verification', train=parent/'train')


def assert_r4_structure_control(reference, candidate):
    a, b = deepcopy(reference), deepcopy(candidate)
    if a['model']['bbox_head']['type'] != 'SymEOODHead' or b['model']['bbox_head']['type'] != 'SymEOODRegTowerHeadV1':
        raise ValueError('Expected old110 and R4 model heads')
    b['model']['bbox_head']['type'] = a['model']['bbox_head']['type']
    if a['model'] != b['model']:
        raise ValueError('R4 differs beyond regression-tower head')
    allowed = {'model', 'load_from', 'work_dir', 'custom_hooks', 'custom_imports'}
    for key in a.keys() | b.keys():
        if key not in allowed and a.get(key) != b.get(key):
            raise ValueError('Non-structural R4 configuration changed: '+key)


def signature(cfg):
    model = cfg['model']; head = model['bbox_head']; original = head['type'] == 'EoodHead'
    main = head['predictors'][0] if original else head
    auxiliary = head['predictors'][1] if original else model['aux_bbox_head'][0]
    return dict(detector=model['type'], main_head=main['type'],
        classification_tower_layers=4 if original else 0,
        regression_tower_layers=4 if original or main['type']=='SymEOODRegTowerHeadV1' else 0,
        auxiliary_feature_source='shared_post_tower_features' if original else 'FPN_features',
        main_classification=deepcopy(main['loss_cls']),
        main_regression=deepcopy(main.get('loss_iou', main['loss_bbox'])),
        auxiliary_classification=deepcopy(auxiliary['loss_cls']),
        auxiliary_regression=deepcopy(auxiliary['loss_bbox']),
        assignment=deepcopy(main.get('train_cfg', model.get('train_cfg'))),
        inference=deepcopy(main.get('test_cfg', model.get('test_cfg'))))


def ablation_audit():
    from crane_project.utils.eood_A_geometry_protocol_v1 import check_model_delta as check_a
    from crane_project.utils.eood_B_regression_protocol_v1 import check_model_delta as check_b
    configs = {name: config(path) for name, path in
               [('E', E_CONFIG), ('EA', EA_CONFIG), ('EB', EB_CONFIG), ('110', S_CONFIG), ('R4', R4_CONFIG)]}
    assert_r4_structure_control(configs['110'], configs['R4'])
    check_a(configs['E']['model'], configs['EA']['model'])
    check_b(configs['E']['model'], configs['EB']['model'])
    return dict(signatures={name: signature(cfg) for name, cfg in configs.items()},
        valid_comparisons={
            '110_vs_R4': 'Existing single-seed regression-tower structural control; each original VAL-selected snapshot.',
            'R4_vs_R4_plus_midpoint': 'Planned midpoint module increment on the SAME frozen R4 checkpoint.',
            'R4_M_vs_R4_M_L': 'Possible future size-module increment; not implemented or run here.'},
        invalid_factorial='E, E+A, E+B and R4 cannot form a single controlled A/B factorial.',
        confounders=['classification tower', 'auxiliary feature source', 'classification and auxiliary weights',
                    'auxiliary regression form', 'assignment cost and positive/temperature schedule', 'output processing'],
        old_internal_AB='Historical eight-arm A/B/C experiments remain valid for their original skeleton; not R4-matched A/B evidence.',
        future_AB_requirement='Fix the complete R4 skeleton, initialization/training and output protocol before toggling A/B. No control retraining authorized here.',
        communication='E+A used NCCL, E+B used Gloo; no complete backend control. Not evidence of a backend-caused metric change.',
        inference_note='Compare actual forwarding code as well as config. These signatures do not infer towers solely from stacked_convs.')


def checked_sources():
    # Parent closure and original detector files remain immutable; no old head/cache is reused.
    from crane_project.tools import train_symeood_110_midpoint_v1 as parent
    parent.checked_sources()
    fixed = json.loads(CONTRACT.read_text())
    if fixed['protocol'] != VERSION or fixed['frontend'] != frontend_identity() or fixed['settings'] != parent.SETTINGS:
        raise ValueError('R4 midpoint contract differs')
    for name, digest in fixed['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('R4 midpoint source SHA differs: '+name)
    return dict(protocol=VERSION, contract_sha256=sha(CONTRACT), sources=fixed['sources'],
                frozen_b=frontend_identity(), sigma_cells=1.5, settings=deepcopy(parent.SETTINGS))


def check_checkpoint_metadata(meta, cfg):
    """Parse logged config safely: literals/dict only, never execute checkpoint text."""
    from crane_project.tools.analyze_port_shape_e_h_train_logs_v1 import config_value
    parsed = {}
    for node in ast.parse(meta.get('config', '')).body:
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in META_FIELDS:
            key = node.targets[0].id
            if key in parsed:
                raise ValueError('Duplicate checkpoint config field')
            parsed[key] = config_value(node.value)
    if (parsed != plain({k: cfg[k] for k in META_FIELDS}) or meta.get('epoch') != 16
            or meta.get('iter') != 10240 or meta.get('seed') != 0):
        raise ValueError('R4 checkpoint metadata differs')


def verification_binding(report, identity, manifest_sha):
    if (report.get('status') != 'R4_MIDPOINT_ZERO_UPDATE_VERIFIED'
            or report.get('identity') != identity or report.get('cache_manifest_sha256') != manifest_sha
            or report.get('optimizer_updates') != 0 or report.get('detector_updates') != 0
            or report.get('test_access') is not False):
        raise ValueError('Matching zero-update R4/cache verification required')
