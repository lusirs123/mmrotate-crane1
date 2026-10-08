#!/usr/bin/env python3
"""Fixed two-feature size fits -> supervised VAL freeze -> gated exposed TEST."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import analyze_port_reliability_tradeoff_v1 as old
from crane_project.utils import port_reliability_feature_ablation_v1 as ab
from crane_project.utils import port_simple_component_reliability_v1 as simple

PROTOCOL = ROOT/'crane_project/tools/port_reliability_feature_ablation_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_reliability_feature_ablation_v1_sources.json'


def checked_sources():
    p, m = old.read(PROTOCOL), old.read(SOURCES)
    parents = ['crane_project/tools/port_reliability_tradeoff_v1_sources.json',
               'tools/port_frozen_downstream_v1_sources.json']
    required = set(parents)
    for name in parents: required.update(old.read(ROOT/name)['sources'])
    required.update([str(PROTOCOL.relative_to(ROOT)),str(Path(__file__).resolve().relative_to(ROOT)),
        'crane_project/utils/port_reliability_feature_ablation_v1.py',
        'tests/test_port_reliability_feature_ablation_v1.py', 'tools/run_port_reliability_feature_ablation_v1.sh'])
    actual = {name: old.sha(ROOT/name) for name in m['sources']}
    if (set(actual) != required or actual != m['sources'] or m['protocol'] != ab.VERSION
            or p['protocol'] != ab.VERSION or m['protocol_sha256'] != old.sha(PROTOCOL)
            or p['feature_arms'] != {k:list(v) for k,v in ab.ARMS.items()}
            or p['fitting'] != ab.FIT or p['target_good_retention'] != .95
            or p['test_used_for_selection'] is not False):
        raise ValueError('Fixed ablation source/contract mismatch')
    old.checked_sources()
    if p['test_contract'] != old.read(ROOT/'tools/port_frozen_downstream_v1_protocol.json')['reliability']:
        raise ValueError('Historical TEST input pins changed')
    return p, dict(manifest_sha256=old.sha(SOURCES),sources=actual)


def save_rows(path, rows, risk_maps, policies, original):
    api = ab.binding.Sigma15Reliability(original,original['front_end'])
    with gzip.open(path, 'xt') as f:
        for row in rows:
            baseline = api.decide(row['pred'],row['image_size'])
            candidates = {}
            for name,policy in policies.items():
                d = ab.decide(policy,row['pred'],row['image_size'])
                if (d['final_box_original'] != baseline['final_box_original'] or
                        d['center_accepted'] != baseline['center_accepted'] or
                        d['angle_accepted'] != baseline['angle_accepted'] or
                        d['risks']['angle'] != baseline['risks']['angle'] or
                        d['risks']['size'] != risk_maps[name][row['image']]):
                    raise ValueError('Online candidate differs or changes frozen output/center/angle')
                candidates[name] = d
            saved = dict(row, size_risks={name:v[row['image']] for name,v in risk_maps.items()},
                original_simple_decision=baseline, candidate_decisions=candidates)
            f.write(json.dumps(saved,ensure_ascii=False,allow_nan=False)+'\n')


def finish(out, status, extra):
    old.write(out/'completion.json', dict(protocol=ab.VERSION,status=status,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),
        source_manifest_sha256=old.sha(SOURCES),
        artifacts={p.name:old.sha(p) for p in sorted(out.iterdir()) if p.is_file()}, **extra))


def run_fit(args):
    p, sources = checked_sources()
    old_protocol, _ = old.checked_sources()
    rows, original, proof, paths = old.checked_inputs(args.input_dir,old_protocol)
    before = simple.fingerprint([rows,original])
    train = [r for r in rows if r['reliability_role']=='train']
    val = [r for r in rows if r['reliability_role']=='val']
    outputs = [r for r in train if r['pred'] is not None]
    features = np.array([simple.descriptor(r['pred'],r['image_size']) for r in outputs])
    labels = [ab.size_bad(r) for r in outputs]
    models = {name:ab.fit(features,labels,indices) for name,indices in ab.ARMS.items()}
    # Preserve full_simple exactly; it is a frozen control, not a refitted arm.
    val_risks = ab.scores(val,original,models)
    controls = {name:val_risks[name] for name in ab.CONTROLS}
    cutoffs = {name:ab.calibrate(val,values,p['target_good_retention']) for name,values in val_risks.items()}
    results = {}; policies = {}
    for name,model in models.items():
        stats = ab.describe(val,val_risks[name],cutoffs[name]['risk_le'],controls)
        results[name] = dict(stats=stats,gate=ab.gate(stats,p['target_good_retention']),
                             calibration=cutoffs[name], model=model)
        policies[name] = dict(protocol=ab.VERSION,arm=name,size_model=model,size_cutoff=cutoffs[name],
            frozen_original_policy=deepcopy(original),original_policy_sha256=proof['input_sha256']['fit/policy.json'],
            feature_ablation_only=True,GT_online=False,test_used_for_selection=False,
            calibration_role='VAL_GT_supervised_all_domains_videos_not_independent',adopted=False)
    selected = ab.choose(results)
    train_risks = ab.scores(train,original,models)
    train_stats = {name:ab.describe(train,values,cutoffs[name]['risk_le'],
        {m:train_risks[m] for m in ab.CONTROLS}) for name,values in train_risks.items()}
    legacy = {name:ab.describe(val,val_risks[alias],original['simple_policy']['cutoffs'][name]['size']['risk_le'],controls)
              for name,alias in [('simple','full_simple'),('score_only','score_only')]}
    calibrated_controls = {name:ab.describe(val,val_risks[name],cutoffs[name]['risk_le'],controls)
                           for name in ab.CONTROLS}
    if before != simple.fingerprint([rows,original]) or proof['input_sha256'] != {n:old.sha(path) for n,path in paths.items()}:
        raise ValueError('Frozen inputs or original policy changed')
    args.out.mkdir(parents=True,exist_ok=False)
    report = dict(protocol=p,source_proof=sources,input_proof=proof,
        frozen_legacy_VAL=legacy,retention95_VAL_controls=calibrated_controls,
        candidates=results,TRAIN_in_sample=train_stats,selected_arm=selected,
        selection_rule=p['selection_rule'],VAL_three_flags=ab.component_report(val,original,policies),
        test_read=False,detector_inferences=0,
        geometry_updates=0,size_score_models_fitted=2,GT_online=False)
    old.write(args.out/'fit_report.json',report)
    old.write(args.out/'candidate_policies.json',policies)
    if selected is not None: old.write(args.out/'selected_policy.json',policies[selected])
    save_rows(args.out/'scored_TRAIN_VAL.jsonl.gz',rows,
              {m:dict(train_risks[m],**val_risks[m]) for m in val_risks},policies,original)
    finish(args.out,'FEATURE_ABLATION_VAL_COMPLETE_REVIEW_REQUIRED',dict(selected_arm=selected,
        test_read=False,test_used_for_selection=False,original_inputs_unchanged=True,
        input_sha256=proof['input_sha256'],detector_inferences=0,geometry_updates=0))
    for name,v in results.items():
        s=v['stats']['all'];print(name,'FA/FR/ED/CR', [s[k] for k in
            ('incorrect_accepted','correct_rejected','incorrect_rejected','correct_accepted')],
            'gate',v['gate']['passed'],'failed_checks',len(v['gate']['failures']),flush=True)
    print('FEATURE_ABLATION_VAL_COMPLETE selected_arm='+str(selected),flush=True)


def checked_frozen_selection(directory):
    receipt=old.read(directory/'completion.json')
    if (receipt['protocol'] != ab.VERSION or receipt['status']!='FEATURE_ABLATION_VAL_COMPLETE_REVIEW_REQUIRED'
            or receipt['test_read'] is not False or receipt['selected_arm'] not in ab.ARMS):
        raise ValueError('No frozen VAL-passing selection; TEST must not be opened')
    for name,digest in receipt['artifacts'].items():
        if old.sha(directory/name)!=digest: raise ValueError('Frozen fit artifact changed')
    report=old.read(directory/'fit_report.json');selected=receipt['selected_arm']
    if (report['selected_arm']!=selected or ab.choose(report['candidates'])!=selected
            or any(ab.gate(v['stats'])!=v['gate'] for v in report['candidates'].values())):
        raise ValueError('Frozen selection/gate cannot be reproduced')
    policy=old.read(directory/'selected_policy.json')
    if (policy!=old.read(directory/'candidate_policies.json')[selected]
            or policy['size_cutoff']!=report['candidates'][selected]['calibration']
            or policy['size_model']!=report['candidates'][selected]['model']):
        raise ValueError('Frozen policy differs from selected candidate')
    return report,receipt,policy


def run_test(args):
    # Crucial ordering: no TEST path, hash or metadata is accessed before this gate.
    report,receipt,policy=checked_frozen_selection(args.fit_dir)
    p,sources=checked_sources()
    if receipt['source_manifest_sha256']!=sources['manifest_sha256']:
        raise ValueError('Source identity differs from frozen fit')
    from tools import eval_port_frozen_downstream_v1 as sealed
    pins=p['test_contract']
    input_proof=dict(geometry=sealed.indexed_bundle(args.geometry_test_dir,pins['geometry_files']),
        metadata=sealed.indexed_bundle(args.metadata_dir,pins['metadata_files']))
    geo=old.read(args.geometry_test_dir/'completion.json');original=policy['frozen_original_policy']
    if (geo['status']!='FROZEN_SIGMA15_TEST_COMPLETE_REVIEW_REQUIRED' or geo['split']!='test'
            or geo['frames']!=1440 or geo['state_before']!=geo['state_after']
            or geo['fixed_checkpoint']!=original['front_end']['midpoint_checkpoint']
            or geo['head_updates'] or geo['detector_updates'] or geo['selection_on_test']):
        raise ValueError('Wrong frozen M TEST generation')
    metadata=[{k:r[k] for k in sealed.flags.FIELDS} for r in sealed.read_rows(args.metadata_dir/'test_qualities.jsonl')]
    rows=sealed.bind_reliability(metadata,sealed.read_rows(args.geometry_test_dir/'test_rows.jsonl'))
    before=simple.fingerprint([rows,policy])
    name=policy['arm'];models={name:policy['size_model']};risks=ab.scores(rows,original,models)
    controls={m:risks[m] for m in ab.CONTROLS}
    cutoffs={m:report['retention95_VAL_controls'][m]['all']['global_risk_le'] for m in ab.CONTROLS}
    cutoffs[name]=policy['size_cutoff']['risk_le']
    stats={m:ab.describe(rows,r,cutoffs[m],controls) for m,r in risks.items()}
    args.out.mkdir(parents=True,exist_ok=False)
    save_rows(args.out/'scored_TEST.jsonl.gz',rows,risks,{name:policy},original)
    result=dict(protocol=ab.VERSION,role='frozen_selected_size_score_on_previously_exposed_TEST',
        selected_arm=name,stats=stats,three_flags=ab.component_report(rows,original,{name:policy}),
        input_proof=input_proof,source_proof=sources,
        frozen_fit_completion_sha256=old.sha(args.fit_dir/'completion.json'),
        frozen_policy_sha256=old.sha(args.fit_dir/'selected_policy.json'),
        test_read=True,test_used_for_selection=False,test_repeatedly_exposed=True,
        fitting_updates=0,cutoff_updates=0,detector_inferences=0,GT_online=False,
        original_inputs_unchanged=before==simple.fingerprint([rows,policy]))
    if not result['original_inputs_unchanged']:raise ValueError('Frozen TEST inputs/policy changed')
    old.write(args.out/'test_report.json',result)
    finish(args.out,'FEATURE_ABLATION_FROZEN_TEST_COMPLETE_REVIEW_REQUIRED',dict(selected_arm=name,
        test_read=True,test_used_for_selection=False,threshold_updates=0,detector_inferences=0))
    print('FEATURE_ABLATION_FROZEN_TEST_COMPLETE',name,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['fit','test'])
    p.add_argument('--input-dir',type=Path,default=Path('work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--fit-dir',type=Path)
    p.add_argument('--geometry-test-dir',type=Path,default=Path('work_dirs/port_results/geometry/port_geometry_midpoint_sigma15_v1_test_eval'))
    p.add_argument('--metadata-dir',type=Path,default=Path('work_dirs/port_reliability_branches_v1_test_cached_v1'))
    args=p.parse_args()
    if args.stage=='fit':run_fit(args)
    else:
        if args.fit_dir is None:p.error('--fit-dir is required for frozen TEST reporting')
        run_test(args)


if __name__=='__main__':main()
