#!/usr/bin/env python3
"""Full TRAIN video crossfit of tiny quality heads, then one exposed VAL."""
import argparse
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.utils import port_reliability_image_temporal_crossfit_v1 as core
from crane_project.tools import run_port_reliability_conditional_evidence_v1 as io


def load(allow_absent=False):
    p=json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_protocol.json')).read_text())
    s=json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_sources.json')).read_text())
    if p['settings']!=core.SETTINGS:raise ValueError('Fixed fitting contract differs')
    pins=dict(p['input_pins'],**p['frontend_pins'],**s['sources']);absent=[]
    for name,digest in pins.items():
        if allow_absent and name in p['frontend_pins'] and not (ROOT/name).exists():absent.append(name);continue
        if io.sha(ROOT/name)!=digest:raise ValueError('Input/source changed: '+name)
    prior=ROOT/p['temporal_result'];r=json.loads((prior/'report.json').read_text())
    review=json.loads((prior/'server_review.json').read_text())
    if not review['passed'] or review['report_sha256']!=io.sha(prior/'report.json') or r['TEST_opened'] or r['candidate_adopted']:raise ValueError('Unreviewed temporal source')
    native=ROOT/p['native_result'];nr=json.loads((native/'report.json').read_text());nv=json.loads((native/'mac_review.json').read_text())
    if not nv['passed'] or nv['report_sha256']!=io.sha(native/'report.json') or nr['TEST_read']:raise ValueError('Unreviewed native cache source')
    return p,pins,absent,r


def feature_matrix(path,rows):
    with np.load(path,allow_pickle=False) as a:features=np.array(a['features'],dtype=float);ids=a['images'].tolist()
    if features.shape!=(len(ids),259) or not np.isfinite(features).all() or len(set(ids))!=len(ids):raise ValueError('Invalid frozen feature cache')
    outputs=[r for r in rows if r['pred'] is not None]
    if set(ids)!=set(r['image'] for r in outputs):raise ValueError('Complete feature/output identity mismatch')
    lookup=dict(zip(ids,features));matrix=np.array([lookup[r['image']] for r in outputs])
    for row,vector in zip(outputs,matrix):
        expected=core.temporal.base.simple.descriptor(row['pred'],row['image_size'])
        if not np.allclose(vector[:3],expected,atol=1e-12,rtol=0):raise ValueError('Frozen descriptors differ')
    return matrix,outputs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path);parser.add_argument('--device',default='cpu')
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--allow-absent-frontend',action='store_true');a=parser.parse_args()
    p,pins,absent,prior=load(a.allow_absent_frontend)
    if a.check_only:print('All source/cache pins verified; no fitting',flush=True);return
    if absent:raise ValueError('Real fitting requires verified frontend identity')
    if a.out is None or a.out.is_symlink() or a.out.resolve().parent.parent!=(ROOT/'work_dirs'/core.VERSION).resolve():raise ValueError('Use work_dirs/version/RUN_ID/result')
    import torch
    from crane_project.utils import port_reliability_image_temporal_crossfit_v1_torch as training
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if a.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    parent=ROOT/p['temporal_result'];native=ROOT/p['native_result'];train=io.read(parent/'scored_TRAIN.jsonl.gz')
    if len(train)!=2558:raise ValueError('Complete TRAIN required')
    matrix,train=feature_matrix(native/'train_native_projected.npz',train)
    splits=core.folds(train);oof=np.full(len(train),np.nan);fold_heads={}
    with (a.out/'quality_train_log.jsonl').open('x') as log:
        for fold in splits:
            tag=fold['heldout'];head=training.train_head(matrix,train,fold['fit'],tag,a.device,log)
            if tag in head['training_sequences']:raise ValueError('Heldout video label leakage')
            oof[fold['test']]=training.predict(head,matrix[fold['test']]);fold_heads[tag]=dict(head,heldout_images=[train[i]['image'] for i in fold['test']])
        full=training.train_head(matrix,train,list(range(len(train))),'FULL_TRAIN',a.device,log)
    if not np.isfinite(oof).all():raise ValueError('Each TRAIN output must have one OOF risk')
    insample=training.predict(full,matrix);supported=[i for i,r in enumerate(train) if r['temporal']['available']]
    rows=[train[i] for i in supported];rp=[r['risks']['partial_auc4'] for r in rows]
    fusion={arm:core.fit_fusion(rows,rp,(insample if arm==core.ARMS[0] else oof)[supported]) for arm in core.ARMS}
    heads=dict(full=full,folds=fold_heads,partition=splits);io.write(a.out/'quality_heads.json',heads);io.write(a.out/'fusion_models.json',fusion)
    io.write(a.out/'TRAIN_image_scores.json',dict(images=[r['image'] for r in train],crossfit=oof.tolist(),in_sample=insample.tolist()))
    # All heads and fusion weights are fixed before parsing VAL rows/features.
    val=io.read(parent/'scored_VAL.jsonl.gz')
    if len(val)!=887 or any(r['split']!='val' or r['reliability_role']!='val' for r in val):raise ValueError('Complete exposed VAL required')
    if set(r['image'] for r in train)&set(r['image'] for r in val):raise ValueError('TRAIN/VAL overlap')
    vmat,vrows=feature_matrix(native/'val_native_projected.npz',val);vrisk=training.predict(full,vmat)
    image_risks={r['image']:None for r in val};image_risks.update({r['image']:float(v) for r,v in zip(vrows,vrisk)})
    scored={'VAL':core.score(val,image_risks,fusion)}
    # TRAIN delivery uses the final full head; OOF used only for candidate fitting.
    scored['TRAIN']=core.score(train,dict(zip((r['image'] for r in train),insample)),fusion)
    cutoffs={arm:core.temporal.base.select_cutoff(scored['VAL'],arm) for arm in core.ARMS};io.write(a.out/'calibration.json',cutoffs)
    roles={role:core.evaluate(rows,cutoffs,prior['calibration']['risk_le']) for role,rows in scored.items()}
    for role,rows in scored.items():
        with gzip.open(a.out/('scored_'+role+'.jsonl.gz'),'xt') as f:
            for row in rows:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        print(role,roles[role][core.METHOD]['groups']['all']['candidate']['states'],flush=True)
    gate=core.gate(roles['VAL'])
    for name,digest in pins.items():
        if name not in absent and io.sha(ROOT/name)!=digest:raise ValueError('Frozen input modified')
    supported_oof=[dict(r,risks=dict(r['risks'],image_crossfit=float(oof[i]),image_in_sample=float(insample[i]))) for i,r in enumerate(train)]
    train_diagnostic=dict(quality_AUROC={m:core.temporal.base.level.metrics.auc(supported_oof,m) for m in ('image_crossfit','image_in_sample')},
        full_minus_crossfit_logit_quantiles=np.quantile(core.bounded_logit(insample)-core.bounded_logit(oof),[0,.05,.5,.95,1]).tolist(),
        full_score_uses_seen_TRAIN=True,OOF_label_exclusion_only=True,frontend_and_PCA_have_seen_TRAIN=True)
    report=dict(protocol=core.VERSION,status='EVIDENCE_PASS_NO_ADOPTION' if gate['passed'] else 'EVIDENCE_FAILED_STOP',
        roles=roles,gate=gate,calibration=cutoffs,train_diagnostic=train_diagnostic,input_pins=pins,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        quality_head_count=len(splits)+1,quality_updates=sum(h['updates'] for h in fold_heads.values())+full['updates'],
        quality_parameter_count=4290,fusion_parameters_per_arm=3,device=a.device,
        model_saved_before_VAL=True,fit_calls=len(splits)+3,detector_inferences=0,detector_updates=0,
        TEST_opened=False,VAL_previously_exposed=True,formal_policy_changed=False,candidate_adopted=False,elapsed_seconds=time.monotonic()-started)
    io.write(a.out/'report.json',report);io.write(a.out/'completion.json',dict(status=report['status'],frames=3445,TEST_opened=False))
    print(report['status'],report['quality_updates'],'quality updates',flush=True)


if __name__=='__main__':main()
