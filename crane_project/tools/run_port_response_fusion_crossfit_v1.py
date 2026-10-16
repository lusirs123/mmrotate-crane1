"""Six video folds, two fixed linear meta learners, VAL freeze then all TEST reports."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
from crane_project.utils import port_response_fusion_crossfit_v1 as core
from crane_project.tools import run_port_reliability_response_warm_v1 as helper
from crane_project.utils import port_reliability_response_tail_v1 as tail
from crane_project.utils import port_reliability_response_warm_v1 as warm
from crane_project.utils import port_reliability_response_stats_v1 as response
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils.port_reliability_paired_ranking_audit_v1 import bad, groups, Ranking, independent_check
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'crane_project/tools/port_response_fusion_crossfit_v1_manifest.json'
METHODS = ('formal', 'score', 'bce_1701', 'soft_1701', 'bce0_1701', 'soft0_1701', 'fusion_in', 'fusion_oof')
sha, write = helper.sha, helper.write


def pins(items):
    for name, digest in items.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Changed source: '+name)


def load_role(role, protocol):
    info = protocol['roles'][role]; pins(info['pins'])
    with gzip.open(ROOT/info['rows'], 'rt') as f:
        rows = [json.loads(s) for s in f]
    with np.load(ROOT/info['features'], allow_pickle=False) as f:
        raw, ids, p, q = f['features'], f['images'].tolist(), f['probability'], f['prior']
    lookup = {r['image']: r for r in rows if r['pred'] is not None}
    if set(lookup) != set(ids) or len(ids) != len(lookup):
        raise ValueError('Identity pairing changed')
    np.testing.assert_allclose(raw[:, 3:], response.response_features(p, q), rtol=0, atol=1e-12)
    np.testing.assert_allclose(raw[:, :3], [simple.descriptor(lookup[i]['pred'], lookup[i]['image_size']) for i in ids], rtol=0, atol=1e-12)
    y = np.asarray([bad(lookup[i]) for i in ids], dtype=float)
    if (len(rows), len(ids), int(y.sum())) != tuple(info['counts']):
        raise ValueError('Role counts changed')
    sequences = np.asarray([lookup[i]['sequence'] for i in ids])
    return rows, raw, ids, y, sequences


def attach(rows, raw, ids, models, mapping, meta):
    lookup = {r['image']: r for r in rows}
    z = core.logits(models, raw)
    mapped = response.normalize(z, mapping)
    fused = {n: core.fusion_risk(m, mapped) for n, m in meta.items()}
    for row in rows:
        row['audit_risks'] = {n: None for n in METHODS}
    for k, image in enumerate(ids):
        row = lookup[image]
        row['audit_risks']['score'] = float(1-row['pred'][5])
        for j, n in enumerate(core.NAMES):
            row['audit_risks'][n] = float(simple.sigmoid(z[k, j]))
        for n in meta:
            row['audit_risks'][n] = float(fused[n][k])
    return mapped


def decision(results):
    value = {}
    for n in ('fusion_in', 'fusion_oof'):
        d = results[n]['at_frozen_VAL_cut']; s = d['all']['states']
        checks = dict(protects_CR=s['CR'] >= 653, protects_real=d['domain:real']['states']['CR'] >= 189,
                      reduces_FA=s['FA'] < 190,
                      same_count_gain=results[n]['same_count_as_formal']['tie_bounds']['FA_max'] < 190)
        value[n] = dict(passed=all(checks.values()), checks=checks, FA_gain=190-s['FA'])
    return dict(VAL_only=value, automatic_promotion=False, TEST_used_for_selection=False,
                primary_candidate='fusion_oof', fusion_in_is_control=True)


def run(out):
    if out.exists() or out.parent.parent != ROOT/'work_dirs'/core.VERSION or Path.cwd().resolve() != ROOT:
        raise ValueError('New correct result path required')
    protocol = json.loads(MANIFEST.read_text())
    if protocol['settings'] != core.SETTINGS:
        raise ValueError('Settings changed')
    pins(protocol['source_pins']); pins(protocol['model_pins'])
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    out.mkdir(parents=True); write(out/'protocol.json', protocol)
    write(out/'runtime.json', dict(cwd=str(ROOT), environment=os.environ.get('CONDA_DEFAULT_ENV'),
        HEAD=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(), torch=torch.__version__,
        GPU=torch.cuda.get_device_name(0), CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))
    rows, raw, ids, y, sequences = load_role('TRAIN', protocol)
    policy = json.loads((ROOT/protocol['policy']).read_text())['simple_policy']
    saved = json.loads((ROOT/protocol['full_models']).read_text())
    full = {n: saved[n] for n in core.NAMES if n != 'formal'}
    full['formal'] = policy['models']['size']
    oof = np.full((len(raw), 5), np.nan); fold_logits = np.full_like(oof, np.nan)
    fit_counts = []; fold_bundles = {}; ownership = np.zeros(len(raw), dtype=int)
    for sequence, fit, held in core.folds(sequences):
        print('FOLD START', sequence, 'fit/held', len(fit), len(held), flush=True)
        parents = []
        for seed in (1701, 1702, 1703):
            _, parent, history = tail.fit(raw[fit], y[fit], seed, 'bce')
            parents.append(parent)
            print('PARENT', sequence, seed, parent['objective_final'], flush=True)
        temperature = warm.temperature_fit(raw[fit], y[fit], parents)
        models = {'formal': simple.fit_linear_risk(raw[fit, :3], y[fit], core.LINEAR)}
        for n, kind, l2 in [('bce_1701', 'bce', .1), ('soft_1701', 'soft', .1),
                             ('bce0_1701', 'bce', 0.), ('soft0_1701', 'soft', 0.)]:
            models[n] = core.continuation(raw[fit], y[fit], parents[0], kind, l2, temperature['temperature'], 'cuda:0')
            print('BRANCH', sequence, n, models[n]['objective_final'], flush=True)
        zfit = core.logits(models, raw[fit]); zheld = core.logits(models, raw[held])
        mapping = core.mapping_fit(zfit)
        oof[held] = response.normalize(zheld, mapping); fold_logits[held] = zheld; ownership[held] += 1
        cut = np.array([np.sort(zfit[y[fit] == 0, j])[int(np.ceil(.9849*sum(y[fit] == 0)))-1] for j in range(5)])
        accepted = (zheld <= cut).all(1)
        support = dict(held_all_accept_correct=int(sum(accepted & (y[held] == 0))), held_all_accept_wrong=int(sum(accepted & (y[held] == 1))),
                       cut_source='fold fitting goods 98.49% each; diagnostic only; not historical VAL-cut cells')
        bundle = dict(held_sequence=sequence, fit_ids=[ids[i] for i in fit], held_ids=[ids[i] for i in held],
            fit_good=int(sum(y[fit] == 0)), fit_bad=int(sum(y[fit] == 1)), held_good=int(sum(y[held] == 0)), held_bad=int(sum(y[held] == 1)),
            parents=parents, temperature=temperature, models=models, mapping=mapping, support=support,
            diagnostic_logit_cuts=cut.tolist(), held_sim_has_no_sim_in_fit=sequence == 'sim_seq08')
        write(out/('fold_'+sequence+'.json'), bundle)
        fold_bundles[sequence] = sha(out/('fold_'+sequence+'.json'))
        fit_counts.append(dict(sequence=sequence, CPU_parent_fits=3, CPU_simple_fits=1, GPU_branch_fits=4))
        print('FOLD COMPLETE', sequence, support, flush=True)
    if not np.isfinite(oof).all() or not np.all(ownership == 1):
        raise ValueError('Incomplete or overlapping OOF')
    full_z = core.logits(full, raw); mapping = core.mapping_fit(full_z)
    insample = response.normalize(full_z, mapping)
    meta = {'fusion_in': core.fit_linear(insample, y), 'fusion_oof': core.fit_linear(oof, y)}
    write(out/'models.json', dict(full=full, full_mapping=mapping, meta=meta, fold_sha256=fold_bundles))
    np.savez_compressed(out/'TRAIN_scores.npz', images=np.array(ids), labels=y, sequences=sequences,
                        OOF_mapped=oof, OOF_logits=fold_logits, full_mapped=insample, full_logits=full_z)
    write(out/'TRAIN_frozen.json', dict(models_sha256=sha(out/'models.json'), folds=fold_bundles,
        TRAIN_scores_sha256=sha(out/'TRAIN_scores.npz'), all_fits_complete_before_VAL=True,
        VAL_opened=False, TEST_opened=False, fits=fit_counts, CPU_meta_fits=2, total_new_fits=50,
        main_seed=1701, auxiliary_parent_seeds_only_for_original_temperature_recipe=[1702, 1703]))
    cuts = {n: json.loads((ROOT/protocol['full_freeze']).read_text())['cutoffs'][n] for n in METHODS if not n.startswith('fusion_')}
    evaluated = {}; mapped_moments = {}
    for role in ('TRAIN', 'VAL'):
        rr, xx, ii, yy, ss = (rows, raw, ids, y, sequences) if role == 'TRAIN' else load_role(role, protocol)
        mapped = attach(rr, xx, ii, full, mapping, meta)
        mapped_moments[role] = dict(mean=mapped.mean(0).tolist(), std=mapped.std(0).tolist())
        if role == 'VAL':
            for n in meta:
                cuts[n] = warm.global_cut(rr, n)
        evaluated[role] = rr
    results = {r: helper.metrics(rr, METHODS, cuts) for r, rr in evaluated.items()}
    choice = decision(results['VAL'])
    write(out/'freeze.json', dict(cutoffs=cuts, decision=choice, models_sha256=sha(out/'models.json'),
        TRAIN_frozen_sha256=sha(out/'TRAIN_frozen.json'), TEST_opened=False, report_all_prespecified_TEST=True))
    # TEST hashes and data are first checked only after every fit, VAL result and cutoff freeze.
    test, xx, ii, yy, ss = load_role('TEST', protocol)
    mapped = attach(test, xx, ii, full, mapping, meta)
    mapped_moments['TEST'] = dict(mean=mapped.mean(0).tolist(), std=mapped.std(0).tolist())
    evaluated['TEST'] = test; results['TEST'] = helper.metrics(test, METHODS, cuts)
    for role, rr in evaluated.items():
        with gzip.open(out/('scored_'+role+'.jsonl.gz'), 'xt') as f:
            for row in rr:
                row['fusion_decisions'] = {n: response.size_decision(row['original_simple_decision'], row['audit_risks'][n], cuts[n]) for n in meta}
                f.write(json.dumps(row, allow_nan=False)+'\n')
    diagnostics = dict(OOF=core.conditional_summary(oof, y, sequences), full=core.conditional_summary(insample, y, sequences),
        OOF_moments=dict(mean=oof.mean(0).tolist(), std=oof.std(0).tolist()), mapped_moments=mapped_moments,
        per_video_score_shift={s: dict(OOF_mean=oof[sequences == s].mean(0).tolist(), OOF_std=oof[sequences == s].std(0).tolist(),
                                  full_mean=insample[sequences == s].mean(0).tolist(), full_std=insample[sequences == s].std(0).tolist()) for s in core.SEQUENCES},
        OOF_meta_TRAIN_AUROC={n: auc_from_arrays(core.fusion_risk(m, oof), y) for n, m in meta.items()},
        limitation='Base OOF only; meta learner sees these labels. OOF-meta TRAIN AUROC is in-sample for meta, not nested held-out fusion performance. Frozen M saw detection TRAIN. Full versus fold score shift remains.')
    write(out/'score_diagnostics.json', diagnostics)
    report = dict(status='VIDEO_CROSSFIT_FUSION_COMPLETE_FROZEN_TEST_REPORTED', results=results, cutoffs=cuts,
        decision=choice, new_scorer_fits=50, main_seed=1701, auxiliary_temperature_parent_seeds=[1702, 1703],
        detector_updates=0, midpoint_updates=0, new_detector_inferences=0, TEST_used_for_selection=False,
        TEST_repeatedly_exposed=True, GT_online=False, original_policy_changed=False, policy_exported=False)
    helper.dump_report(out, report)
    p = out/'analysis.md'
    p.write_text(p.read_text().split('\nparent=')[0]+'\n\n六视频留出，五维线性融合；fusion_in为训练内评分对照，fusion_oof为留出评分拟合。\n只报告1701，1702/1703父模型仅复现折内温度计算。50次新拟合：18父模型＋6simple＋24继续分支＋2融合。\n所有TRAIN训练完成后完整VAL确定统一门限，TEST随后全报告。无检测推理或policy替换。\n'+json.dumps(choice, ensure_ascii=False)+'\n')
    pins(protocol['source_pins']); pins(protocol['model_pins'])
    write(out/'completion.json', dict(artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(report['status'], flush=True)


def auc_from_arrays(scores, y):
    # Independent tie-aware rank sum, avoids artificial GT rows.
    a = np.asarray(scores)[y == 0]; b = np.asarray(scores)[y == 1]
    if not len(a) or not len(b):
        return None
    a = np.sort(a)
    return float(sum(np.searchsorted(a, v, side='left')+.5*(np.searchsorted(a, v, side='right')-np.searchsorted(a, v, side='left')) for v in b)/(len(a)*len(b)))


def review(out, receipt):
    protocol = json.loads((out/'protocol.json').read_text()); source_proof = review_source_pins(protocol['source_pins']); pins(protocol['model_pins'])
    for n, h in json.loads((out/'completion.json').read_text())['artifacts'].items():
        if sha(out/n) != h:
            raise ValueError('Artifact changed '+n)
    bundle = json.loads((out/'models.json').read_text()); freeze = json.loads((out/'freeze.json').read_text())
    trainf = json.loads((out/'TRAIN_frozen.json').read_text()); report = json.loads((out/'report.json').read_text())
    if trainf['models_sha256'] != sha(out/'models.json') or freeze['models_sha256'] != sha(out/'models.json') or freeze['TRAIN_frozen_sha256'] != sha(out/'TRAIN_frozen.json') or freeze['decision'] != decision(report['results']['VAL']):
        raise ValueError('Frozen states mismatch')
    saved = json.loads((ROOT/protocol['full_models']).read_text())
    policy = json.loads((ROOT/protocol['policy']).read_text())['simple_policy']
    if bundle['full']['formal'] != policy['models']['size'] or any(bundle['full'][n] != saved[n] for n in core.NAMES[1:]):
        raise ValueError('Original full models changed')
    oldcuts = json.loads((ROOT/protocol['full_freeze']).read_text())['cutoffs']
    if any(report['cutoffs'][n] != oldcuts[n] for n in METHODS if not n.startswith('fusion_')) or freeze['cutoffs'] != report['cutoffs']:
        raise ValueError('Original controls cutoffs changed')
    rows, raw, ids, y, sequences = load_role('TRAIN', protocol)
    cache = np.load(out/'TRAIN_scores.npz', allow_pickle=False)
    np.testing.assert_array_equal(cache['images'], ids); np.testing.assert_array_equal(cache['labels'], y)
    max_error = 0.; checks = []; fold_support = {}; membership = np.zeros(len(ids), dtype=int)
    for sequence, fit, held in core.folds(sequences):
        path = out/('fold_'+sequence+'.json')
        if sha(path) != bundle['fold_sha256'][sequence]:
            raise ValueError('Fold changed')
        f = json.loads(path.read_text())
        if f['fit_ids'] != [ids[i] for i in fit] or f['held_ids'] != [ids[i] for i in held] or set(f['fit_ids']) & set(f['held_ids']):
            raise ValueError('OOF membership leakage')
        expected_norm = response.normalize_fit(raw[fit], 'TRAIN')
        for parent in f['parents']:
            if parent['normalizer'] != expected_norm:
                raise ValueError('Parent normalization leakage')
            obj = RiskReplay(parent)
            if tail.state_digest(obj) != parent['final_state_sha256'] or parent['steps'] != 5000:
                raise ValueError('Parent final weights mismatch')
        for n in core.NAMES[1:]:
            m = f['models'][n]
            if m['normalizer'] != expected_norm or m['initial_state_sha256'] != f['parents'][0]['final_state_sha256'] or m['steps'] != 5000:
                raise ValueError('Branch parent/settings mismatch')
        ref_simple = simple.fit_linear_risk(raw[fit, :3], y[fit], core.LINEAR)
        numeric_dict_equal(ref_simple, f['models']['formal'])
        ref_tau = warm.temperature_fit(raw[fit], y[fit], f['parents'])
        numeric_dict_equal(ref_tau, f['temperature'])
        zfit = scalar_logits(f['models'], raw[fit]); zheld = scalar_logits(f['models'], raw[held])
        for key in ('mean', 'scale'):
            np.testing.assert_allclose(core.mapping_fit(zfit)[key], f['mapping'][key], rtol=0, atol=1e-10)
        mapped = response.normalize(zheld, f['mapping'])
        np.testing.assert_allclose(mapped, cache['OOF_mapped'][held], rtol=0, atol=1e-10)
        np.testing.assert_allclose(zheld, cache['OOF_logits'][held], rtol=0, atol=1e-10)
        cut = np.array([np.sort(zfit[y[fit] == 0, j])[int(np.ceil(.9849*sum(y[fit] == 0)))-1] for j in range(5)])
        np.testing.assert_allclose(cut, f['diagnostic_logit_cuts'], rtol=0, atol=1e-10)
        accept = (zheld <= cut).all(1)
        fold_support[sequence] = dict(correct=int(sum(accept & (y[held] == 0))), wrong=int(sum(accept & (y[held] == 1))))
        if (fold_support[sequence]['correct'], fold_support[sequence]['wrong']) != (f['support']['held_all_accept_correct'], f['support']['held_all_accept_wrong']):
            raise ValueError('Fold support differs')
        membership[held] += 1
    if not np.all(membership == 1):
        raise ValueError('Not exactly one held score per frame')
    z = scalar_logits(bundle['full'], raw)
    np.testing.assert_allclose(z, cache['full_logits'], rtol=0, atol=1e-10)
    for key in ('mean', 'scale'):
        np.testing.assert_allclose(core.mapping_fit(z)[key], bundle['full_mapping'][key], rtol=0, atol=1e-10)
    np.testing.assert_allclose(response.normalize(z, bundle['full_mapping']), cache['full_mapped'], rtol=0, atol=1e-10)
    for n, x in [('fusion_in', cache['full_mapped']), ('fusion_oof', cache['OOF_mapped'])]:
        m = bundle['meta'][n]; norm = response.normalize_fit(x, 'TRAIN')
        if m['normalizer'] != norm:
            raise ValueError('Meta normalization mismatch')
        d = np.column_stack([np.ones(len(x)), response.normalize(x, norm)])
        w = np.where(y == 1, .5/y.sum(), .5/(len(y)-y.sum()))
        obj, grad, _ = simple.logistic_objective(np.array(m['weights']), d, y, w, .1)
        if abs(obj-m['objective']) > 1e-10 or abs(grad).max() > 1e-8:
            raise ValueError('Meta objective/gradient mismatch')
    for role in ('TRAIN', 'VAL', 'TEST'):
        original, xx, ii, yy, ss = load_role(role, protocol)
        with gzip.open(out/('scored_'+role+'.jsonl.gz'), 'rt') as f:
            rr = [json.loads(s) for s in f]
        orig = {r['image']: r for r in original}; lookup = {r['image']: r for r in rr}
        if len(rr) != len(original) or set(orig) != set(lookup):
            raise ValueError('Membership changed')
        zz = scalar_logits(bundle['full'], xx); mapped = response.normalize(zz, bundle['full_mapping'])
        for row in rr:
            if bad(row) != bad(orig[row['image']]):
                raise ValueError('Label mismatch')
            for key in ('image','sequence','domain','split','frame_id','image_size','pred','gt','b_original','original_simple_decision'):
                if row.get(key) != orig[row['image']].get(key):
                    raise ValueError('Identity changed '+key)
            for n in bundle['meta']:
                if row['fusion_decisions'][n] != response.size_decision(row['original_simple_decision'], row['audit_risks'][n], report['cutoffs'][n]):
                    raise ValueError('Flags changed')
        for k, image in enumerate(ii):
            values = {n: float(simple.sigmoid(zz[k, j])) for j, n in enumerate(core.NAMES)}
            values['score'] = 1-lookup[image]['pred'][5]
            for n, m in bundle['meta'].items():
                weights = np.array(m['weights']); u = (mapped[k]-m['normalizer']['mean'])/m['normalizer']['scale']
                values[n] = float(simple.sigmoid(weights[0]+sum(a*b for a, b in zip(u, weights[1:]))))
            max_error = max(max_error, max(abs(v-lookup[image]['audit_risks'][n]) for n, v in values.items()))
        if helper.metrics(rr, METHODS, report['cutoffs']) != report['results'][role]:
            raise ValueError('Metric mismatch')
        for n, v in report['results'][role].items():
            for g, part in groups(rr).items():
                checks.append(independent_check(Ranking(part, n), v['at_frozen_VAL_cut'][g], report['cutoffs'][n]))
        if role == 'VAL':
            for n in bundle['meta']:
                goods = sorted(r['audit_risks'][n] for r in rr if bad(r) is False)
                real = sorted(r['audit_risks'][n] for r in rr if bad(r) is False and r['domain'] == 'real')
                if report['cutoffs'][n] != max(goods[652], real[188]):
                    raise ValueError('VAL cut mismatch')
    if max_error > 1e-10:
        raise ValueError('Independent scalar risk differs')
    write(out/receipt, dict(status='INDEPENDENT_REVIEW_PASS', groups=len(checks),
        pairs=sum(c['pair_comparisons'] for c in checks), curve_points=sum(c['curve_points'] for c in checks),
        fold_count=6, held_rows=int(membership.sum()), fold_support=fold_support,
        scalar_risk_max_abs=max_error, report_sha256=sha(out/'report.json'),
        source_proof=source_proof, refit_comparison_atol=1e-10, refit_comparison_rtol=1e-10))
    print('INDEPENDENT_REVIEW_PASS', len(checks), flush=True)


def scalar_logits(models, raw):
    out = []
    for row in raw:
        values = []
        for name in core.NAMES:
            m = models[name]
            if name == 'formal':
                x = [(a-b)/c for a, b, c in zip(row[:3], m['mean'], m['scale'])]
                v = m['weights'][0]+sum(a*b for a, b in zip(x, m['weights'][1:]))
            else:
                x = [(a-b)/c for a, b, c in zip(row, m['normalizer']['mean'], m['normalizer']['scale'])]
                w = m['weights']; h = [max(0., b+sum(a*c for a, c in zip(x, ww))) for b, ww in zip(w['network.0.bias'], w['network.0.weight'])]
                v = w['network.2.bias'][0]+sum(a*b for a, b in zip(h, w['network.2.weight'][0]))
            values.append(v)
        out.append(values)
    return np.array(out)


def RiskReplay(model):
    from crane_project.utils.port_reliability_response_stats_v1_torch import Risk
    obj = Risk(model['seed'], 19)
    obj.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in model['weights'].items()})
    return obj


def numeric_dict_equal(expected, actual):
    """Cross-platform refits have roundoff; identities and structure stay exact."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise ValueError('Recomputed structure differs')
        for key in expected:
            numeric_dict_equal(expected[key], actual[key])
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError('Recomputed sequence differs')
        for a, b in zip(expected, actual):
            numeric_dict_equal(a, b)
    elif isinstance(expected, float):
        np.testing.assert_allclose(expected, actual, atol=1e-10, rtol=1e-10)
    elif expected != actual:
        raise ValueError('Recomputed identity differs')


def review_source_pins(items):
    """Validate archived training code and this bounded review-only repair."""
    runner = str(Path(__file__).resolve().relative_to(ROOT))
    mismatches = [n for n, h in items.items() if sha(ROOT/n) != h]
    if not mismatches:
        return dict(original_sources_match=True)
    fixpath = ROOT/'crane_project/tools/port_response_fusion_crossfit_v1_review_fix.json'
    fix = json.loads(fixpath.read_text())
    if set(mismatches) != set(fix['repaired_source_pins']):
        raise ValueError('Unexpected source changes')
    pins(fix['repaired_source_pins'])
    for name in mismatches:
        import hashlib
        archived = subprocess.check_output(['git', 'show', fix['original_commit']+':'+name], cwd=ROOT)
        if hashlib.sha256(archived).hexdigest() != items[name]:
            raise ValueError('Original archived source differs')
    return dict(original_sources_match=False, original_commit_verified=fix['original_commit'],
        archived_original_sources={n: items[n] for n in mismatches}, repair_manifest_sha256=sha(fixpath),
        current_sources=fix['repaired_source_pins'], scope=fix['scope'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--review', action='store_true'); p.add_argument('--receipt', default='independent_review.json')
    a = p.parse_args()
    if a.review:
        review(a.out.resolve(), a.receipt)
    else:
        try:
            run(a.out.resolve())
        except Exception as error:
            if a.out.is_dir():
                write(a.out/'failure.json', dict(error=repr(error)))
            raise
