#!/usr/bin/env python3
"""Fixed eight TRAIN views, copied quality MLP only; no VAL/TEST or deployable weights."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import diagnose_port_reliability_mechanism_v1 as parent
from crane_project.utils import port_reliability_candidate_fit_v1 as fitutil
from crane_project.utils.port_structure_reliability_v1 import assert_detector_frozen
branch, train, prior = parent.branch, parent.train, parent.prior
VERSION='port_reliability_candidate_fit_v1'
PROTOCOL=ROOT/'crane_project/tools/port_reliability_candidate_fit_v1_protocol.json'
MANIFEST=ROOT/'crane_project/tools/port_reliability_candidate_fit_v1_sources.json'


def checked_sources():
    old, _, training_protocol = parent.checked_sources()
    protocol=json.loads(PROTOCOL.read_text())
    manifest=json.loads(MANIFEST.read_text())
    if (manifest['sources'] != {p:branch.sha(ROOT/p) for p in manifest['sources']}
            or manifest['protocol_sha256'] != branch.sha(PROTOCOL)
            or manifest['parent_manifest_sha256'] != old['manifest_sha256']):
        raise ValueError('Candidate diagnostic source contract differs')
    return dict(manifest_sha256=branch.sha(MANIFEST), parent=old), protocol, training_protocol


def checked_cases(args, protocol):
    paths={'probe.json':args.probe_dir/'probe.json','cases.jsonl':args.probe_dir/'cases.jsonl'}
    if {k:branch.sha(p) for k,p in paths.items()} != protocol['evidence_sha256']:
        raise ValueError('Requires exact reviewed eight-view TRAIN probe artifacts')
    report=json.loads(paths['probe.json'].read_text())
    rows=[json.loads(line) for line in paths['cases.jsonl'].read_text().splitlines()]
    if (len(rows)!=8 or [r['case'] for r in rows]!=protocol['cases']
            or report['branch_checkpoint_sha256']!=protocol['checkpoint_sha256']):
        raise ValueError('Fixed cases or checkpoint differ')
    return report,rows


def load_cache(path,sources,protocol):
    marker=json.loads(path.with_suffix('.sha.json').read_text())
    if marker!=dict(sha256=branch.sha(path), sources=sources, protocol=protocol):
        raise ValueError('Descriptor cache identity differs')
    cache=torch.load(str(path),map_location='cpu')
    if cache['sources']!=sources or cache['protocol']!=protocol:
        raise ValueError('Descriptor cache contract differs')
    return cache


def collect(args,sources,protocol,training_protocol,report,oldrows):
    inputs,proof=train.checked_inputs(args.input_snapshot,args.train_cache,args.structure_report)
    payload=parent.previous.base.fixed_bundle(args.branch_checkpoint,sources['parent']['training_sources'],training_protocol)
    if branch.sha(args.branch_checkpoint)!=protocol['checkpoint_sha256']:
        raise ValueError('Requires unchanged completed epoch08')
    cfg=prior.check_cfg()
    detector,runtime=train.build_runtime(cfg,args.b_checkpoint,args.gpu)
    frozen=prior.state_digest(detector)
    if frozen!=payload['frozen_b_state'] or frozen!=proof['b_cache_state'] or frozen!=report['frozen_b_state']:
        raise ValueError('Frozen B identity differs')
    arms=branch.make_arms('cuda:'+str(args.gpu))
    if branch.architecture(arms)!=payload['contract']['architecture']:
        raise ValueError('Architecture differs')
    branch.load_bundle(payload,arms)
    del payload
    states={name:prior.state_digest(arm) for name,arm in arms.items()}
    if states!=report['heads_before']:
        raise ValueError('Original head state differs')
    for arm in arms.values():
        arm.eval()
        for p in arm.parameters():
            p.requires_grad_(False)
    indexed=train.datasets_for(deepcopy(cfg.data.train),inputs)
    cache=dict(sources=sources,protocol=protocol,arms={},proof=proof,runtime=runtime,replay=[])
    torch.cuda.reset_peak_memory_stats(args.gpu)
    for old in oldrows:
        case=old['case']; source=inputs[case['dataset_index']]
        values,replay,raw_state=train.training_view(detector,indexed[case['dataset_index']],source,args.gpu,case['view_seed'])
        p3,boxes,scores,meta,target,mask,n,_,_=values
        for key in ('image','input_sha256','view_seed','view_image_sha256','scale_factor','img_shape','pad_shape','flip','flip_direction'):
            if replay[key]!=old['replay'][key]:
                raise ValueError('TRAIN replay differs: '+key)
        difference=parent.previous.prediction_difference(replay['genuine_b_original'],old['replay']['genuine_b_original'])
        if not difference['all_six_close']:
            raise ValueError('Frozen B output differs from fixed TRAIN view')
        if (n!=1 or len(boxes)!=14 or not torch.allclose(target.cpu(),torch.tensor(old['continuous_targets']),atol=1e-6,rtol=0)
                or not torch.equal(mask.cpu(),torch.tensor(old['quality_masks']))):
            raise ValueError('Candidate targets/masks differ')
        for name,arm in arms.items():
            descriptor,q,mlp,delta=fitutil.capture(arm,p3,boxes,scores,meta)
            if not torch.allclose(q,torch.tensor(old['qualities'][name]),atol=1e-6,rtol=0):
                raise ValueError('Original quality replay differs: '+name)
            if name not in cache['arms']:
                cache['arms'][name]=dict(initial=mlp.state_dict(),rows=[])
            cache['arms'][name]['rows'].append(dict(image=case['image'],domain=case['domain'],
                descriptor=descriptor,target=target.detach().cpu(),mask=mask.detach().cpu(),genuine_count=n,
                probe_names=old['probe_names'],initial_qualities=q,cpu_parity_max_abs=delta))
        cache['replay'].append(dict(image=case['image'],b_parity=difference,p3_sha256=prior.tensor_sha(p3),
                                    original_p3_sha256=old['p3_sha256']))
        print('Fixed TRAIN descriptors',len(cache['replay']),'/ 8',case['image'],flush=True)
        del values,raw_state,p3,boxes,scores,target,mask,mlp
    assert_detector_frozen(detector)
    if prior.state_digest(detector)!=frozen or {n:prior.state_digest(a) for n,a in arms.items()}!=states:
        raise ValueError('Frozen models changed')
    if any(p.grad is not None for a in arms.values() for p in a.parameters()):
        raise ValueError('Original head received gradients')
    if train.checked_inputs(args.input_snapshot,args.train_cache,args.structure_report)[1]!=proof:
        raise ValueError('TRAIN evidence changed during collection')
    cache.update(frozen_b_state=frozen,heads_before=states,heads_after=states,detector_updates=0,
                 original_head_updates=0,max_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20,
                 max_reserved_mib=torch.cuda.max_memory_reserved(args.gpu)/2**20)
    return cache


def fit_cache(cache,out_dir):
    torch.set_num_threads(1)
    arms=branch.make_arms('cpu')
    results={}
    for name in branch.ARMS:
        initial=arms[name].quality
        initial.load_state_dict(cache['arms'][name]['initial'],strict=True)
        before=prior.state_digest(initial)
        result,fitted=fitutil.fit(initial,cache['arms'][name]['rows'],steps=cache['protocol']['steps'],
                                lr=cache['protocol']['lr'],milestones=cache['protocol']['milestones'])
        if prior.state_digest(initial)!=before:
            raise ValueError('Copied MLP fitting changed initial weights')
        result.update(initial_mlp_state=before,final_mlp_state=prior.state_digest(fitted))
        results[name]=result
        print(name,'loss',result['before']['loss'],'->',result['final']['loss'],flush=True)
    report={k:v for k,v in cache.items() if k!='arms'}
    report.update(status='FIXED_TRAIN_CANDIDATE_FIT_COMPLETE_REVIEW_REQUIRED',results=results,
                  val_or_test_read=False,test_repeatedly_exposed=True,deployable_weights_saved=False,
                  interpretation='Selected in-sample views only; no generalization or unique root-cause claim; final fixed step, no best-step selection.')
    if checked_sources()[:2] != (cache['sources'],cache['protocol']):
        raise ValueError('Sources changed during CPU fit')
    branch.write_new(out_dir/'fit_report.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('check','run','fit-cache'),default='check')
    p.add_argument('--probe-dir',type=Path,default=Path('work_dirs/port_reliability_mechanism_v1_probe_path_fix_v1'))
    p.add_argument('--branch-checkpoint',type=Path,default=Path('work_dirs/port_reliability_branches_v1_train/epoch_08.pth'))
    p.add_argument('--input-snapshot',type=Path,default=Path('work_dirs/port_reliability_train_support_v1/train_input_snapshot.json'))
    p.add_argument('--train-cache',type=Path,default=Path('work_dirs/port_reliability_train_support_v1_cache'))
    p.add_argument('--structure-report',type=Path,default=Path('work_dirs/port_reliability_readiness_v1_structure_complete/train_structure_check.json'))
    p.add_argument('--b-checkpoint',type=Path,default=prior.CHECKPOINT)
    p.add_argument('--descriptor-cache',type=Path)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0)
    args=p.parse_args(); os.chdir(ROOT)
    if args.out_dir.exists():
        raise FileExistsError('Preserve existing evidence; choose a new --out-dir: '+str(args.out_dir))
    sources,protocol,training_protocol=checked_sources()
    if args.mode=='fit-cache':
        if args.descriptor_cache is None:
            p.error('fit-cache requires --descriptor-cache')
        cache=load_cache(args.descriptor_cache,sources,protocol)
        args.out_dir.mkdir(parents=True)
        fit_cache(cache,args.out_dir)
    else:
        report,rows=checked_cases(args,protocol)
        if args.mode=='check':
            # Exact archived probe bytes already bind original TRAIN views and targets.
            branch.write_new(args.out_dir/'input_check.json',dict(sources=sources,protocol=protocol,
                status='FIXED_TRAIN_PROBE_INPUT_CHECK_PASS',inference=False,optimizer_steps=0,
                limitation='Runtime checkpoint, TRAIN files/cache and replay verified in run mode.'))
        else:
            cache=collect(args,sources,protocol,training_protocol,report,rows)
            if checked_sources()!=(sources,protocol,training_protocol):
                raise ValueError('Sources changed during collection')
            checked_cases(args,protocol)
            args.out_dir.mkdir(parents=True)
            path=args.out_dir/'descriptors.pt'
            with path.open('xb') as stream:
                torch.save(cache,stream)
            branch.write_new(path.with_suffix('.sha.json'),dict(sha256=branch.sha(path),sources=sources,protocol=protocol))
            fit_cache(load_cache(path,sources,protocol),args.out_dir)
    print('Saved',args.out_dir,flush=True)


if __name__=='__main__':
    main()
