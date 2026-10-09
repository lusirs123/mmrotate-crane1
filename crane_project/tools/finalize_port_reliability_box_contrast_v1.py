#!/usr/bin/env python3
"""Recover only a known reporting error; immutable trained outputs, no update."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_box_contrast_v1 as run
from crane_project.utils import port_reliability_state_continuity_v1 as states
from crane_project.utils import port_reliability_feature_ablation_v1 as ab
from crane_project.utils import port_simple_component_reliability_v1 as simple


def finalize(directory,training_commit):
    import torch
    started=time.monotonic();directory=Path(directory)
    if directory.resolve().parent.parent!=(ROOT/'work_dirs'/run.core.VERSION).resolve():raise ValueError('Original result directory required')
    if (directory/'report.json').exists() or (directory/'completion.json').exists():raise FileExistsError('Never overwrite a finished report')
    if json.loads((directory/'failure.json').read_text())!={'error':'ValueError: Invalid fixed normalization'}:
        raise ValueError('Only the known post-TRAIN reporting failure may be recovered')
    if not re.fullmatch('[0-9a-f]{40}',training_commit):raise ValueError('Exact training commit required')
    protocol,current=run.checked_sources()
    manifest_path=str(run.SOURCES.relative_to(ROOT))
    source_bytes=subprocess.check_output(['git','show',training_commit+':'+manifest_path],cwd=str(ROOT))
    training_source=json.loads(source_bytes)
    old_protocol=json.loads(subprocess.check_output(['git','show',training_commit+':'+str(run.PROTOCOL.relative_to(ROOT))],cwd=str(ROOT)))
    if old_protocol!=protocol or set(training_source['sources'])!=set(run.SOURCE_FILES):raise ValueError('Training design/source identity differs')
    before={p.name:run.sha(p) for p in directory.iterdir() if p.is_file()}
    input_pins={p:run.sha(ROOT/p) for p in run.PINS}
    if input_pins!=run.PINS:raise ValueError('Frozen original inputs changed')
    rows,policy=run.load_rows();saved=json.loads((directory/'models.json').read_text())
    if saved['epoch']!=100 or saved['update_counts']!=dict(original=1000,contrast=1000):raise ValueError('Fixed final budget differs')
    checkpoint=run.prior.load_trusted(directory/'final_heads.pth')
    if (checkpoint['epoch']!=100 or checkpoint['normalizer']!=saved['normalizer']
            or any(v.cpu().tolist()!=saved['models'][arm][name] for arm,values in checkpoint['models'].items() for name,v in values.items())):
        raise ValueError('Actual saved weights do not match immutable model JSON')
    logs=[json.loads(s) for s in (directory/'train_log.jsonl').read_text().splitlines()]
    if len(logs)!=2000 or Counter(v['arm'] for v in logs)!=dict(original=1000,contrast=1000):raise ValueError('Actual update log incomplete')
    cutoffs=json.loads((directory/'cutoffs.json').read_text());parts={};scores={};statistics={};proofs={};diagnostics={}
    masks=labels=matrix=None;normalizer=saved['normalizer'];exported=saved['models']
    manifest=json.loads((directory/'frozen_cache_manifest.json').read_text())
    for role in ('TRAIN','VAL'):
        part=[json.loads(s) for s in gzip.open(directory/('scored_'+role+'.jsonl.gz'),'rt')];parts[role]=part
        original=sorted([r for r in rows if r['reliability_role']==role.lower()],key=lambda r:r['image'])
        if simple.fingerprint(original)!=simple.fingerprint([{k:r[k] for k in run.prior.FIELDS} for r in part]):raise ValueError('Delivered original rows changed')
        data=np.load(directory/(role.lower()+'_projected.npz'),allow_pickle=False)
        matrices=data['features'];ids=data['images'].tolist();m=matrices[:,0] if role=='TRAIN' else matrices
        scores[role]={a:{r['image']:r['experiment_risks'][a] for r in part} for a in (*run.core.ARMS,'full_simple','score_only')}
        replay=run.score(part,m,ids,exported,normalizer)
        for method,values in replay.items():
            for image,risk in values.items():
                stored=scores[role][method][image]
                if risk is None:
                    if stored is not None:raise ValueError('Missing output gained a risk')
                elif abs(risk-stored)>1e-12:raise ValueError('Saved final scores differ from replay')
        statistics[role]={a:run.core.describe(part,scores[role],a,cutoffs[a]['risk_le']) for a in run.core.ARMS}
        diagnostics[role]={a:run.core.hidden_diagnostic(part,ids,m,exported[a],normalizer) for a in run.core.ARMS}
        traces=[json.loads(s) for s in gzip.open(directory/(role.lower()+'_trace.jsonl.gz'),'rt')]
        contrasts=sum(len(t.get('boxes',[]))-1 for t in traces if t['present'])
        aux_bad=sum(sum(t['labels'][1:]) for t in traces if t['present'])
        paired=sum(0 in t['labels'] and 1 in t['labels'] for t in traces if t['present'])
        proof=dict(frames=len(part),outputs=len(ids),detector_state=manifest['detector_state'],
            valid_contrasts=contrasts,auxiliary_bad=int(aux_bad),same_image_both_class_frames=paired,
            maximum_B_replay_difference=None,maximum_M_replay_difference=None,
            original_collection_checked_B_M_replay_bounds=True,
            replay_bounds=dict(xy_atol=5e-4,xy_rtol=2e-5,angle_atol=2e-6,score_exact=True),
            original_M_never_replaced=True,detector_updates=0,midpoint_updates=0,GT_online=False,
            proof_reconstructed_from_complete_trace=True,not_new_detector_inference=True)
        if role=='TRAIN':
            raw=np.load(directory/'train_raw.npz',allow_pickle=False);labels=raw['labels'];masks=raw['mask']
            values=raw['features'];proof['changed_pooled_features']=sum(
                not np.array_equal(values[i,0],values[i,j]) for i in range(len(ids)) for j in range(1,13) if masks[i,j])
            matrix=matrices
        else:proof['changed_pooled_features']=0
        proofs[role]=proof
    for method,cutoff in cutoffs.items():
        if ab.calibrate(parts['VAL'],scores['VAL'][method],.95)!=cutoff:raise ValueError('Original frozen VAL cutoff differs')
    engineering=json.loads((directory/'smoke_report.json').read_text())
    for arm in run.core.ARMS:
        engineering[arm]['final_checkpoint_matches_model_JSON']=True
    order_hash=hashlib.sha256();rng=np.random.RandomState(1701)
    for _ in range(100):order_hash.update(rng.permutation(2558).astype('<i8').tobytes())
    mechanism={a:run.core.contrast_diagnostic(run.core.numpy_logits(exported[a],matrix.reshape(-1,259),normalizer).reshape(masks.shape),labels,masks) for a in run.core.ARMS}
    val=parts['VAL'];fixed={};formal={};centers={}
    for name,group in ab.groups(val).items():
        fixed[name]={a:states.summarize(group,{r['image'] for r in group if r['pred'] is not None and scores['VAL'][a][r['image']]<=cutoffs[a]['risk_le']}) for a in scores['VAL']}
        formal[name]=states.summarize(group,{r['image'] for r in group if r['original_simple_decision']['size_accepted']})
        outputs=sum(r['pred'] is not None for r in group)
        hits=sum(simple.geometry_errors(r['gt'],r['pred'])['center_px']<15 for r in group if r['pred'] is not None)
        centers[name]=dict(frames=len(group),outputs=outputs,hits=hits,output_coverage=outputs/len(group),
            hit_rate_on_outputs=hits/outputs if outputs else None,correct_coverage_all_frames=hits/len(group))
    gate=run.core.gate(statistics['VAL']['contrast']);status='VAL_PASS_FROZEN_TEST_PENDING' if gate['passed'] else 'VAL_FAILED_STOP'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip()
    recovery=dict(known_error='Old291-D hidden diagnostic used with259-D inputs',training_commit=training_commit,
        reporting_commit=commit,training_updates_added=0,detector_inferences_added=0,
        original_outputs_sha256=before,reporting_source_sha256=run.sha(Path(__file__)),
        exact_old_collect_maximum_differences_not_recoverable=True)
    report=dict(protocol=run.core.VERSION,status=status,contract=protocol,
        sources=dict(manifest_sha256=hashlib.sha256(source_bytes).hexdigest(),sources=training_source['sources']),
        reporting_sources=current,reporting_recovery=recovery,input_sha256=input_pins,
        train_proof=proofs['TRAIN'],val_proof=proofs['VAL'],train_statistics=statistics['TRAIN'],
        VAL_statistics=statistics['VAL'],VAL_cutoffs=cutoffs,VAL_fixed_points=fixed,gate=gate,
        original_policy_summary=formal,center=centers,engineering=engineering,hidden_diagnostic=diagnostics,
        TRAIN_contrast_diagnostic=mechanism,parameter_counts=dict(original=4290,contrast=4290),
        update_counts=dict(original=1000,contrast=1000),batch_order_sha256=order_hash.hexdigest(),
        train_rows=2558,train_bad=71,val_rows=887,fixed_final_epoch=100,smoke_reused=False,new_data_roles=False,
        GT_online=False,TEST_read=False,TEST_repeatedly_exposed=True,original_policy_changed=False,
        boxes_scores_output_center_angle_unchanged=True,synthetic_samples_are_not_new_real_error_scenes=True,
        pca_normalizer_fit_only_original_TRAIN=True,elapsed_seconds=None,
        reporting_recovery_seconds=time.monotonic()-started,git_commit=training_commit)
    if any(run.sha(directory/name)!=pin for name,pin in before.items()):raise ValueError('Report recovery changed original artifacts')
    run.write(directory/'reporting_recovery.json',recovery)
    run.write(directory/'report.json',report)
    (directory/'TEST_NOT_RUN.txt').write_text('Known reporting error repaired without retraining. Failed VAL candidate stops; TEST never selects.\n')
    run.write(directory/'completion.json',dict(protocol=run.core.VERSION,status=status,TEST_read=False,
        reporting_recovered=True,artifacts={p.name:run.sha(p) for p in directory.iterdir() if p.is_file()}))
    print(status,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--training-commit',required=True);args=parser.parse_args()
    finalize(args.directory,args.training_commit)
