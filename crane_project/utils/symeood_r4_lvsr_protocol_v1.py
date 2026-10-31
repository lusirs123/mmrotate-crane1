"""Identity and predeclared VAL-only gates for fresh R4/M20 dual-branch LVSR."""
from copy import deepcopy
import json
from pathlib import Path
import re
from crane_project.utils import symeood_r4_midpoint_protocol_v1 as parent

ROOT=parent.ROOT
VERSION='symeood_r4_m20_lvsr_pair_v1'
BASE=parent.DEFAULT_ROOT
CONTRACT=ROOT/'crane_project/data_contracts/symeood_r4_lvsr_pair_v1.json'
MIDPOINT_SHA='d595ab01c3aea48aec66d6cf1e38460990ea272bb5ca962ca8ec8e17dd8d6dce'
PINS={'train/selection.json':'ade312204703c975b4c95f0eae3f9fe45377fd83140b0182d13746938bd2966a',
    'train/completion.json':'6540a2fbb6d2c6163c0335712751c580cc8d455841e69d1cfbce4d615facd240',
    'train/val_epoch_20.rows.jsonl':'f7762c38b798b72683452da9faf5a390025e176c2ba9ddc6c49bf1a6b056979f',
    'roi_cache/cache_manifest.json':'7034d4073c16538d5b1683753934a31c2a108f39e9183b217d850f6762eb8017'}
SETTINGS=dict(epochs=24,batch_size=8,seed=1703,lr=.001,weight_decay=0.,betas=[.9,.999],eps=1e-8,
    clip_norm=10.,parameters=2480,architecture='PairSpecificSizeHead',alignment='fixed_b',
    loss='unchanged port_size_qalign_v1.size_loss; SmoothL1 beta0.1 log-size residual',
    initialization='fresh zero terminal; no old LVSR or smoke state',
    TRAIN='all naturally present M outputs at scales0.5/1; no GT-center eligibility filtering',
    VAL='all887 frames native scale; no TEST/depth',continuation=False)
GATE=dict(AP75_tolerance_pp=.0001,size_MAE_tolerance=1e-6,RIoU_tolerance=1e-5,
    AP75_reference='max(frozen M20, frozen R4) per overall/real/sim',
    size_reference='frozen M20',joint='overall strict gain; per domain nondecrease',
    protection='per-domain AP75,TP50,mean RIoU and exact center/angle/score/presence',
    ranking='overall joint count descending; real long/short then sim long/short MAE ascending; epoch',
    no_eligible='selected=null; descriptive representative only, never automatic promotion')
sha=parent.sha


def midpoint_directory():
    for folder in (BASE/'midpoint/attempt_001',BASE/'readbacks/completed/midpoint/attempt_001'):
        if (folder/'train/selection.json').exists():
            for name,digest in PINS.items():
                if sha(folder/name)!=digest:raise ValueError('Fixed M20 artifact differs: '+name)
            return folder
    raise ValueError('Completed R4/M20 input artifacts required')


def paths(attempt='attempt_001'):
    if not re.fullmatch(r'attempt_[0-9]{3}',attempt):raise ValueError('attempt_NNN child required')
    folder=BASE/'lvsr'/attempt
    return dict(root=BASE,lvsr=BASE/'lvsr',preparation=BASE/'lvsr/preparation',attempt=folder,
                inputs=folder/'inputs',verify=folder/'verification',train=folder/'train')


def checked_sources():
    identity=parent.checked_sources();folder=midpoint_directory()
    selection=json.loads((folder/'train/selection.json').read_text())
    complete=json.loads((folder/'train/completion.json').read_text())
    cp=selection['selected_checkpoint']
    if (selection['identity']!=identity or cp['sha256']!=MIDPOINT_SHA or cp['epoch']!=20
            or cp['path']!='head_epoch_20.pth' or cp['updates']!=17980
            or selection['test_access'] is not False or selection['selection_on_test'] is not False
            or complete['epochs_completed']!=24 or complete['updates']!=21576
            or complete['selection']!=selection or complete['detector_updates']!=0):
        raise ValueError('Original full-VAL-selected R4/M20 required')
    contract=json.loads(CONTRACT.read_text())
    if contract['protocol']!=VERSION or contract['settings']!=SETTINGS or contract['gate']!=GATE:
        raise ValueError('LVSR protocol settings/gate differ')
    for name,digest in contract['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('LVSR source SHA differs: '+name)
    return dict(protocol=VERSION,contract_sha256=sha(CONTRACT),sources=contract['sources'],
        midpoint_training_identity=identity,detector_sha256=parent.R4_SHA,midpoint_sha256=MIDPOINT_SHA,
        input_artifacts=deepcopy(PINS),settings=deepcopy(SETTINGS),gate=deepcopy(GATE))


def assess(metrics):
    checks={}
    for domain in ('overall','real','sim'):
        cell=metrics[domain];m,r,c=cell['midpoint'],cell['b'],cell['edge_residual']
        checks[domain+'/identity']=cell['frozen_identity']
        checks[domain+'/AP75_vs_M_and_R4']=c['AP75']>=max(m['AP75'],r['AP75'])-GATE['AP75_tolerance_pp']
        checks[domain+'/TP50_vs_M']=c['TP50']>=m['TP50']
        checks[domain+'/RIoU_vs_M']=c['mean_RIoU']>=m['mean_RIoU']-GATE['RIoU_tolerance']
        checks[domain+'/joint_vs_M']=c['joint10']>m['joint10'] if domain=='overall' else c['joint10']>=m['joint10']
        for edge in ('long','short'):
            checks[domain+'/'+edge+'_MAE_vs_M']=c[edge+'_MAE']<=m[edge+'_MAE']+GATE['size_MAE_tolerance']
    return dict(pass_all=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v])


def selection_key(metrics,epoch):
    return [-metrics['overall']['edge_residual']['joint10']]+[
        metrics[d]['edge_residual'][e+'_MAE'] for d in ('real','sim') for e in ('long','short')]+[epoch]


def choose(events):
    eligible=[e for e in events if e['eligibility']['pass_all']]
    best=min(eligible or events,key=lambda e:e['selection_key'])
    return dict(selected_epoch=best['epoch'] if eligible else None,representative_epoch=best['epoch'],
        eligible_epochs=[e['epoch'] for e in eligible],selection_on_test=False,automatic_adoption=False)
