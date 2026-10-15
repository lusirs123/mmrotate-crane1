"""Saved response features: full TRAIN fit -> protected VAL -> gated frozen TEST."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import numpy as np
import torch
from crane_project.utils import port_reliability_response_warm_v1 as core
from crane_project.utils import port_reliability_response_stats_v1 as response
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_simple_pilot_increment_v1 as base
from crane_project.utils.port_reliability_paired_ranking_audit_v1 import Ranking, bad, groups, compare_high_region
from crane_project.tools import run_port_reliability_response_stats_v1 as previous

ROOT = previous.ROOT
MANIFEST = ROOT/'crane_project/tools/port_reliability_response_warm_v1_manifest.json'
sha, write, pins, load_rows = previous.sha, previous.write, previous.pins, previous.load_rows


def cached(rows, role, manifest, out):
    item = manifest['saved_responses'][role]
    pins({item['path']: item['sha256']})
    path = ROOT/item['path']
    with np.load(path, allow_pickle=False) as cache:
        ids, raw = cache['images'].tolist(), cache['features']
        p, q = cache['probability'], cache['prior']
    outputs = {r['image']: r for r in rows if r['pred'] is not None}
    if len(ids) != len(outputs) or set(ids) != set(outputs) or raw.shape != (len(ids), 19):
        raise ValueError('Complete saved material pairing differs')
    descriptor = np.asarray([simple.descriptor(outputs[i]['pred'], outputs[i]['image_size']) for i in ids])
    response_error = float(np.max(np.abs(raw[:, 3:]-response.response_features(p, q))))
    descriptor_error = float(np.max(np.abs(raw[:, :3]-descriptor)))
    if response_error > 1e-12 or descriptor_error > 1e-12:
        raise ValueError('Saved feature construction differs')
    shutil.copyfile(path, out/(role.lower()+'_responses.npz'))
    labels = np.asarray([bad(outputs[i]) for i in ids], dtype=float)
    return raw, ids, labels, dict(frames=len(rows), outputs=len(ids), errors=int(labels.sum()),
        saved_npz_sha256=item['sha256'], descriptor_max_abs=descriptor_error,
        response_max_abs=response_error, detection_inferences=0, frozen_M_replay_reused=True)


def metrics(rows, methods, cuts):
    result = {name: base.diagnostics(rows, name, cuts) for name in methods}
    for name, value in result.items():
        rank = Ranking(rows, name)
        value['high_retention_97_100_vs_formal'] = compare_high_region(rank, Ranking(rows, 'formal'), .97)
        value['group_matched'] = {g: dict(
            same_CR_as_formal=Ranking(part, name).same_cr(base.at_cut(part, 'formal', cuts['formal'])['all']['states']['CR']),
            same_count_as_formal=Ranking(part, name).same_count(base.at_cut(part, 'formal', cuts['formal'])['all']['accepted']))
            for g, part in groups(rows).items()}
        if name.startswith(('hard_', 'soft_')):
            control = 'bce_'+name.split('_')[1]
            value['high_retention_97_100_vs_paired_BCE'] = compare_high_region(rank, Ranking(rows, control), .97)
    return result


def dump_report(out, report):
    write(out/'report.json', report)
    columns = ['role', 'method', 'group', 'FA', 'FR', 'ED', 'CR', 'MISSING',
        'correct_retention', 'accepted_error_rate', 'acceptance_coverage', 'full_frame_correct_coverage',
        'longest_correct_FR', 'longest_FR_observations', 'longest_without_CR_observations']
    with (out/'metrics.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, columns); writer.writeheader()
        for role, values in report['results'].items():
            for method, value in values.items():
                for group, d in value['at_frozen_VAL_cut'].items():
                    line = dict(role=role, method=method, group=group, **d['states'])
                    line.update({k: d[k] for k in columns if k in d}); writer.writerow(line)
    lines = ['# BCE初始化后的硬尾部与KL-DRO平滑目标对照', '', report['status'], '',
        '|角色|方法|FA|FR|ED|CR|Real CR|同接受数FA上界|AUROC|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for role, values in report['results'].items():
        for method, value in values.items():
            d = value['at_frozen_VAL_cut']; s = d['all']['states']
            lines.append('|%s|%s|%d|%d|%d|%d|%d|%d|%.6f|' % (role, method,
                s['FA'], s['FR'], s['ED'], s['CR'], d['domain:real']['states']['CR'],
                value['same_count_as_formal']['tie_bounds']['FA_max'], value['group_AUROC']['all']))
    lines += ['', 'parent=冻结原19维BCE；bce/hard/soft=相同父模型继续BCE/硬top2%/KL-DRO平滑目标。',
        '统一GPU全批量Adam额外5000步、float64、L2=.1；共同BCE起点/归一化，重置优化器，仅最终模型。',
        'hard沿用上轮目标；soft按每个错误框分别对所有正确框log-mean-exp后平均，参照ICML2022 Eq10。',
        '一个全局门限同时保护VAL CR>=653、Real CR>=189；同CR/同接受数仅离线诊断。',
        '无驻点梯度/时延否决，无自动policy替换；TEST已多次暴露，不用于选择。',
        json.dumps(report['decision'], ensure_ascii=False)]
    (out/'analysis.md').write_text('\n'.join(lines)+'\n')


def run(out):
    if out.exists() or Path.cwd().resolve() != ROOT or out.parent.parent != ROOT/'work_dirs'/core.VERSION:
        raise ValueError('Use new unified experiment/RUN_ID/result from project root')
    manifest = json.loads(MANIFEST.read_text())
    if manifest['settings'] != core.SETTINGS or manifest['arms'] != list(core.ARMS):
        raise ValueError('Fixed experiment settings differ')
    pins(manifest['source_pins']); pins(manifest['development_pins'])
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    out.mkdir(parents=True); write(out/'protocol.json', manifest)
    write(out/'runtime.json', dict(cwd=str(ROOT), environment=os.environ.get('CONDA_DEFAULT_ENV'),
        python=platform.python_version(), torch=torch.__version__, training_device='cuda:0', GPU=torch.cuda.get_device_name(0), CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        HEAD=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()))
    train = load_rows('TRAIN', manifest)
    raw, ids, labels, proof_train = cached(train, 'TRAIN', manifest, out)
    if raw.shape != (2558, 19) or labels.sum() != 71:
        raise ValueError('Original complete TRAIN differs')
    parent_path=ROOT/manifest['parent_models']
    parent_all=json.loads(parent_path.read_text())
    parents=[parent_all['B_%d'%seed] for seed in core.SEEDS]
    temperature=core.temperature_fit(raw,labels,parents)
    write(out/'temperature_frozen.json',temperature)
    write(out/'parent_models.json',{'parent_%d'%p['seed']:p for p in parents})
    models={}
    with (out/'train_log.jsonl').open('x') as log:
        for parent in parents:
            seed=parent['seed']; states={}
            models['parent_%d'%seed]=parent
            for arm in core.ARMS:
                name='%s_%d'%(arm,seed)
                obj,model,records=core.fit(raw,labels,parent,arm,temperature['temperature'],'cuda:0')
                models[name]=model
                states[arm]={k:v.detach().cpu() for k,v in obj.state_dict().items()}
                for record in records:
                    log.write(json.dumps(dict(method=name,**record),allow_nan=False)+'\n')
                log.flush()
                print('TRAIN',name,'steps',model['steps'],'loss',model['objective_final'],flush=True)
            if len({models['%s_%d'%(a,seed)]['initial_state_sha256'] for a in core.ARMS})!=1:
                raise ValueError('Warm start differs across arms')
            path=out/('seed_%d.pth'%seed); torch.save(states,str(path))
            reload=torch.load(str(path),map_location='cpu')
            if any(not torch.equal(v,reload[a][k]) for a,state in states.items() for k,v in state.items()):
                raise ValueError('Checkpoint reload differs')
    write(out/'models.json', models)
    write(out/'TRAIN_frozen.json', dict(models_sha256=sha(out/'models.json'), protocol_sha256=sha(MANIFEST),
        all_training_complete_before_VAL=True, TEST_read=False))
    # No VAL rows/NPZ were opened before all nine final models were frozen.
    pins(manifest['val_pins']); val = load_rows('VAL', manifest)
    vraw, vids, _, proof_val = cached(val, 'VAL', manifest, out)
    policy = json.loads((ROOT/previous.history.POLICY).read_text())
    previous.add_scores(train, raw, ids, models, policy); previous.add_scores(val, vraw, vids, models, policy)
    cuts = dict(formal=manifest['formal_cutoff'])
    for name in ('score',)+tuple(models):
        cuts[name] = core.global_cut(val, name)
    methods = ('formal', 'score')+tuple(models)
    roles = dict(TRAIN=train, VAL=val)
    results = {role: metrics(rows, methods, cuts) for role, rows in roles.items()}
    decision = core.gate(val, cuts)
    write(out/'freeze.json', dict(decision=decision, cutoffs=cuts, models_sha256=sha(out/'models.json'),
        protocol_sha256=sha(MANIFEST), TEST_read=False, all_models_cutoffs_gate_frozen_before_TEST=True))
    proofs = dict(TRAIN=proof_train, VAL=proof_val)
    if decision['passed']:
        # TEST response maps exist only in the frozen M cache, collect once after gate.
        pins(manifest['test_pins']); test = load_rows('TEST', manifest)
        from crane_project.utils.port_geometry_midpoint_sigma_v1 import SigmaMidpointHead
        payload = torch.load(str(ROOT/previous.history.HEAD), map_location='cpu')
        if (payload['epoch'], payload['sigma_cells'], payload['updates']) != (3, 1.5, 2706):
            raise ValueError('Wrong formal M')
        device = 'cuda:0'; torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        head = SigmaMidpointHead(1.5).to(device).eval().requires_grad_(False)
        head.load_state_dict(payload['head_state'], strict=True)
        traw, tids, proofs['TEST'] = previous.collect(test, 'TEST', head, device, out, manifest)
        selected = {k: v for k, v in models.items() if k.endswith('_1701')}
        previous.add_scores(test, traw, tids, selected, policy); roles['TEST'] = test
        results['TEST'] = metrics(test, ('formal', 'score')+tuple(selected), cuts)
    else:
        (out/'TEST_NOT_RUN.txt').write_text('Fixed VAL performance gate failed; no new TEST rows/cache/hash read.\n')
    for role, rows in roles.items():
        with gzip.open(out/('scored_'+role+'.jsonl.gz'), 'xt') as f:
            for row in rows:
                row['response_decisions'] = {name: response.size_decision(row['original_simple_decision'],
                    row['audit_risks'][name], cuts[name]) for name in models if name in row['audit_risks']}
                f.write(json.dumps(row, allow_nan=False)+'\n')
    report = dict(status='VAL_PASS_FROZEN_TEST_REPORTED' if decision['passed'] else 'VAL_FAILED_STOP',
        results=results, cutoffs=cuts, decision=decision, proofs=proofs, temperature=temperature,
        TEST_read=decision['passed'], TEST_repeatedly_exposed=True, TEST_used_for_selection=False,
        detector_updates=0, midpoint_updates=0, scorer_fits=9, pilot_used=False, GT_online=False,
        gradient_gate=False, latency_gate=False, original_policy_changed=False,
        boxes_scores_output_center_angle_unchanged=True,
        source_manifest_sha256=sha(MANIFEST), models_sha256=sha(out/'models.json'))
    dump_report(out, report)
    pins(manifest['source_pins']); pins(manifest['development_pins']); pins(manifest['val_pins'])
    write(out/'completion.json', dict(status=report['status'], artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(report['status'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    try:
        run(args.out.resolve())
    except Exception as error:
        if args.out.is_dir() and not (args.out/'completion.json').exists():
            write(args.out/'failure.json', dict(status='ENGINEERING_FAILED', error=repr(error)))
        raise
