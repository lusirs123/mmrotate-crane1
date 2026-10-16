"""One fixed inference-source comparison, no fit or detector forward call."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from crane_project.utils import port_response_fold_ensemble_v1 as core
from crane_project.tools import run_port_response_fusion_crossfit_v1 as previous
from crane_project.tools import run_port_reliability_response_warm_v1 as helper
from crane_project.utils import port_reliability_response_stats_v1 as response
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils.port_reliability_paired_ranking_audit_v1 import bad, groups, Ranking, independent_check

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'work_dirs/port_response_fusion_crossfit_v1/20261016_video_oof1701_v1/result'
MANIFEST = ROOT/'crane_project/tools/port_response_fold_ensemble_v1_manifest.json'
METHODS = ('formal', 'full_fusion', 'fold_simple', 'fold_fusion')
sha, write, pins = helper.sha, helper.write, previous.pins


def snapshots():
    bundle = json.loads((SOURCE/'models.json').read_text())
    folds = {s: json.loads((SOURCE/('fold_'+s+'.json')).read_text()) for s in core.SEQUENCES}
    for s, f in folds.items():
        if sha(SOURCE/('fold_'+s+'.json')) != bundle['fold_sha256'][s] or f['held_sequence'] != s or set(f['fit_ids']) & set(f['held_ids']):
            raise ValueError('Fold provenance differs')
    return bundle, folds


def gains(rows, cuts):
    result = {}
    for g, part in groups(rows).items():
        result[g] = {}
        for n in METHODS[1:]:
            c = dict(new_ED=0, new_FR=0, rescued_CR=0, released_FA=0, old_five_common_wrong_rejected=0)
            for r in part:
                b = bad(r)
                if b is None:
                    continue
                a = r['audit_risks']['formal'] <= cuts['formal']; k = r['audit_risks'][n] <= cuts[n]
                if a and not k:
                    c['new_ED' if b else 'new_FR'] += 1
                if k and not a:
                    c['released_FA' if b else 'rescued_CR'] += 1
                if b and r['old_five_common_accepted'] and not k:
                    c['old_five_common_wrong_rejected'] += 1
            result[g][n] = c
    return result


def score(rows, raw, ids, bundle, folds, oldcuts, role, held_cache=None):
    control = {r['image']: (r['original_simple_decision']['risks']['size'], r['audit_risks']['fusion_oof']) for r in rows}
    if role == 'TRAIN':
        if list(held_cache['images']) != ids:
            raise ValueError('TRAIN cache membership differs')
        values = core.held_training(held_cache['OOF_logits'], held_cache['OOF_mapped'], bundle['meta']['fusion_oof'])
        proof = dict(held_base_logits=held_cache['OOF_logits'], held_mapped=held_cache['OOF_mapped'])
    else:
        values, proof = core.ensemble(raw, folds, bundle['meta']['fusion_oof'])
    full = core.full_fusion(raw, bundle)
    lookup = {r['image']: r for r in rows}
    for r in rows:
        r['old_five_common_accepted'] = r['pred'] is not None and all(r['audit_risks'][n] <= oldcuts[n] for n in core.NAMES)
        r['audit_risks'] = dict(formal=control[r['image']][0], full_fusion=control[r['image']][1], fold_simple=None, fold_fusion=None)
    for k, image in enumerate(ids):
        if abs(full[k]-control[image][1]) > 1e-10:
            raise ValueError('Frozen full fusion replay differs')
        for n in ('fold_simple', 'fold_fusion'):
            lookup[image]['audit_risks'][n] = float(values[n][k])
    return proof


def decision(results):
    value = {}
    for n in ('fold_simple', 'fold_fusion'):
        v = results[n]; a = v['at_frozen_VAL_cut']; s = a['all']['states']
        checks = dict(protects_CR=s['CR'] >= 653, protects_Real=a['domain:real']['states']['CR'] >= 189,
            reduces_FA=s['FA'] < 190, same_count_gain=v['same_count_as_formal']['tie_bounds']['FA_max'] < 190)
        value[n] = dict(passed=all(checks.values()), checks=checks, FA_gain=190-s['FA'])
    fusion = results['fold_fusion']['at_frozen_VAL_cut']['all']['states']['FA']
    base = results['fold_simple']['at_frozen_VAL_cut']['all']['states']['FA']
    return dict(VAL_only=value, fusion_gain_vs_fold_simple=base-fusion,
        fusion_specific_gain_supported=value['fold_fusion']['passed'] and fusion < base,
        automatic_promotion=False, TEST_used_for_selection=False)


def benchmark(raw, ids, bundle, folds):
    wanted = ('real_seq01_00000', 'real_seq06_00381', 'sim_seq08_00747')
    functions = dict(formal=lambda x: simple.sigmoid(core.simple_logits(bundle['full']['formal'], x)),
        full_fusion=lambda x: core.full_fusion(x, bundle), fold_simple=lambda x: core.fold_simple(x, folds),
        fold_fusion=lambda x: core.ensemble(x, folds, bundle['meta']['fusion_oof'])[0]['fold_fusion'])
    records = []
    for image in wanted:
        x = raw[ids.index(image):ids.index(image)+1]
        for fn in functions.values():
            for _ in range(10):
                fn(x)
        timings = {n: [] for n in METHODS}
        for repeat in range(30):
            order = METHODS if repeat % 2 == 0 else METHODS[::-1]
            for n in order:
                start = time.perf_counter(); functions[n](x)
                timings[n].append((time.perf_counter()-start)*1000)
        records.append(dict(image=image, milliseconds=timings,
            median_ms={n: float(np.median(v)) for n, v in timings.items()}))
    return dict(scope='CPU cached19D scoring only, batch1, excludes detector, image IO and response extraction; not end-to-end latency',
        warmups=10, repeats=30, alternating_order=True, dtype='float64', records=records,
        aggregate_median_ms={n: float(np.median([v for r in records for v in r['milliseconds'][n]])) for n in METHODS})


def run(out):
    if out.exists() or Path.cwd().resolve() != ROOT or out.parent.parent != ROOT/'work_dirs'/core.VERSION:
        raise ValueError('New correct result path required')
    protocol = json.loads(MANIFEST.read_text())
    if protocol['settings'] != core.SETTINGS:
        raise ValueError('Fixed aggregation settings differ')
    pins(protocol['source_pins']); pins(protocol['model_pins'])
    bundle, folds = snapshots()
    oldfreeze = json.loads((SOURCE/'freeze.json').read_text()); oldcuts = oldfreeze['cutoffs']
    out.mkdir(parents=True); write(out/'protocol.json', protocol)
    write(out/'runtime.json', dict(cwd=str(ROOT), environment=os.environ.get('CONDA_DEFAULT_ENV'),
        HEAD=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(), python=platform.python_version(),
        numpy=np.__version__, CPU=platform.processor(), CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), new_fits=0))
    write(out/'inference_models.json', dict(bundle=bundle, folds=folds))
    train, raw, ids, y, seq = previous.load_role('TRAIN', protocol)
    with np.load(SOURCE/'TRAIN_scores.npz', allow_pickle=False) as f:
        held = {k: f[k] for k in f.files}
    proof = score(train, raw, ids, bundle, folds, oldcuts, 'TRAIN', held)
    np.savez_compressed(out/'TRAIN_score_proof.npz', images=np.array(ids), **proof)
    write(out/'cached_scoring_benchmark.json', benchmark(raw, ids, bundle, folds))
    write(out/'inference_frozen.json', dict(models_sha256=sha(out/'inference_models.json'),
        formula=core.SETTINGS, fits=0, VAL_opened=False, TEST_opened=False, no_model_selection=True))
    val, vx, vi, vy, vs = previous.load_role('VAL', protocol)
    proof = score(val, vx, vi, bundle, folds, oldcuts, 'VAL')
    np.savez_compressed(out/'VAL_score_proof.npz', images=np.array(vi), **proof)
    cuts = dict(formal=oldcuts['formal'], full_fusion=oldcuts['fusion_oof'])
    for n in ('fold_simple', 'fold_fusion'):
        cuts[n] = helper.core.global_cut(val, n)
    evaluated = dict(TRAIN=train, VAL=val)
    results = {role: helper.metrics(rr, METHODS, cuts) for role, rr in evaluated.items()}
    choice = decision(results['VAL'])
    write(out/'freeze.json', dict(cutoffs=cuts, decision=choice, models_sha256=sha(out/'inference_models.json'),
        inference_frozen_sha256=sha(out/'inference_frozen.json'), TEST_opened=False,
        report_all_prespecified_TEST=True, no_weight_or_aggregation_selection=True))
    test, tx, ti, ty, ts = previous.load_role('TEST', protocol)
    proof = score(test, tx, ti, bundle, folds, oldcuts, 'TEST')
    np.savez_compressed(out/'TEST_score_proof.npz', images=np.array(ti), **proof)
    evaluated['TEST'] = test; results['TEST'] = helper.metrics(test, METHODS, cuts)
    gain = {role: gains(rr, cuts) for role, rr in evaluated.items()}
    for role, rr in evaluated.items():
        with gzip.open(out/('scored_'+role+'.jsonl.gz'), 'xt') as f:
            for r in rr:
                r['ensemble_decisions'] = {n: response.size_decision(r['original_simple_decision'], r['audit_risks'][n], cuts[n]) for n in METHODS[1:]}
                f.write(json.dumps(r, allow_nan=False)+'\n')
    report = dict(status='FROZEN_FOLD_ENSEMBLE_COMPLETE_EXPOSED_TEST_REPORTED', results=results,
        cutoffs=cuts, decision=choice, gains=gain, new_scorer_fits=0, detector_updates=0, midpoint_updates=0,
        new_detector_inferences=0, feature_extractions=0, seed=1701, GT_online=False, policy_exported=False,
        TEST_used_for_selection=False, TEST_repeatedly_exposed=True, original_policy_changed=False,
        TRAIN_semantics=core.SETTINGS['TRAIN'], inference_comparison_confound='Fold averaging changes variance as well as model provenance; improvement would not isolate a unique cause.')
    helper.dump_report(out, report)
    p = out/'analysis.md'
    p.write_text(p.read_text().split('\nparent=')[0].replace('BCE初始化后的硬尾部与KL-DRO平滑目标对照', '冻结六折集成推理对照')+
        '\n\n0拟合/0新检测推理；固定六折等权平均最终logit，再sigmoid。TRAIN只沿用原单折留出评分，非六折平均，meta已见这些标签。\n'+json.dumps(choice, ensure_ascii=False)+'\n')
    pins(protocol['source_pins']); pins(protocol['model_pins'])
    write(out/'completion.json', dict(artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(report['status'], json.dumps(choice, ensure_ascii=False), flush=True)


def scalar_fusion(meta, mapped):
    return np.array([meta['weights'][0]+sum((a-b)/c*w for a, b, c, w in zip(row,
        meta['normalizer']['mean'], meta['normalizer']['scale'], meta['weights'][1:])) for row in mapped])


def review(out, receipt):
    protocol = json.loads((out/'protocol.json').read_text()); pins(protocol['source_pins']); pins(protocol['model_pins'])
    for n, h in json.loads((out/'completion.json').read_text())['artifacts'].items():
        if sha(out/n) != h:
            raise ValueError('Artifact changed '+n)
    snapshot = json.loads((out/'inference_models.json').read_text()); bundle, folds = snapshots()
    if snapshot != dict(bundle=bundle, folds=folds):
        raise ValueError('Frozen model snapshot changed')
    report = json.loads((out/'report.json').read_text()); freeze = json.loads((out/'freeze.json').read_text())
    frozen = json.loads((out/'inference_frozen.json').read_text())
    if frozen['models_sha256'] != sha(out/'inference_models.json') or freeze['models_sha256'] != frozen['models_sha256'] or freeze['inference_frozen_sha256'] != sha(out/'inference_frozen.json') or freeze['cutoffs'] != report['cutoffs'] or freeze['decision'] != decision(report['results']['VAL']):
        raise ValueError('Inference/VAL freeze differs')
    oldcuts = json.loads((SOURCE/'freeze.json').read_text())['cutoffs']
    if report['cutoffs']['formal'] != oldcuts['formal'] or report['cutoffs']['full_fusion'] != oldcuts['fusion_oof']:
        raise ValueError('Original cuts changed')
    checks = []; maximum = 0.; proof_maximum = 0.
    for role in ('TRAIN', 'VAL', 'TEST'):
        original, raw, ids, y, seq = previous.load_role(role, protocol)
        orig = {r['image']: r for r in original}
        with gzip.open(out/('scored_'+role+'.jsonl.gz'), 'rt') as f:
            rows = [json.loads(s) for s in f]
        lookup = {r['image']: r for r in rows}
        if set(lookup) != set(orig) or len(rows) != len(orig):
            raise ValueError('Row membership changed')
        proof = np.load(out/(role+'_score_proof.npz'), allow_pickle=False)
        np.testing.assert_array_equal(proof['images'], ids)
        meta = bundle['meta']['fusion_oof']
        if role == 'TRAIN':
            cache = np.load(SOURCE/'TRAIN_scores.npz', allow_pickle=False)
            for name, source in [('held_base_logits', 'OOF_logits'), ('held_mapped', 'OOF_mapped')]:
                np.testing.assert_array_equal(proof[name], cache[source])
            # Replay every TRAIN row with only its own held-video model, never the five fitting models.
            z = np.zeros((len(raw), 5)); mapped = np.zeros_like(z); owner = np.zeros(len(raw), dtype=int)
            for s in core.SEQUENCES:
                ix = np.flatnonzero(seq == s)
                if set(np.array(ids)[ix]) != set(folds[s]['held_ids']):
                    raise ValueError('Held membership changed')
                z[ix] = previous.scalar_logits(folds[s]['models'], raw[ix])
                mapped[ix] = (z[ix]-folds[s]['mapping']['mean'])/folds[s]['mapping']['scale']; owner[ix] += 1
            if not np.all(owner == 1):
                raise ValueError('Invalid OOF ownership')
            for a, b in [(z, proof['held_base_logits']), (mapped, proof['held_mapped'])]:
                proof_maximum = max(proof_maximum, float(abs(a-b).max()))
            expected = dict(fold_simple=simple.sigmoid(z[:, 0]), fold_fusion=simple.sigmoid(scalar_fusion(meta, mapped)))
        else:
            z = np.stack([previous.scalar_logits(folds[s]['models'], raw) for s in core.SEQUENCES])
            mapped = np.stack([(a-folds[s]['mapping']['mean'])/folds[s]['mapping']['scale'] for s, a in zip(core.SEQUENCES, z)])
            h = np.stack([scalar_fusion(meta, a) for a in mapped])
            for a, b in [(z, proof['fold_base_logits']), (mapped, proof['fold_mapped']), (h, proof['fold_fusion_logits'])]:
                proof_maximum = max(proof_maximum, float(abs(a-b).max()))
            # Explicit equal accumulation; no fold/label/video-dependent weights.
            expected = dict(fold_simple=simple.sigmoid(sum(z[k, :, 0] for k in range(6))/6),
                            fold_fusion=simple.sigmoid(sum(h[k] for k in range(6))/6))
        zfull = previous.scalar_logits(bundle['full'], raw)
        mfull = (zfull-bundle['full_mapping']['mean'])/bundle['full_mapping']['scale']
        expected['full_fusion'] = simple.sigmoid(scalar_fusion(meta, mfull))
        for k, image in enumerate(ids):
            maximum = max(maximum, max(abs(v[k]-lookup[image]['audit_risks'][n]) for n, v in expected.items()))
        for r in rows:
            old = orig[r['image']]
            for key in ('image','sequence','domain','split','frame_id','image_size','pred','gt','b_original','reliability_role','original_simple_decision'):
                if r.get(key) != old.get(key):
                    raise ValueError('Original identity changed '+key)
            if r['audit_risks']['formal'] != old['original_simple_decision']['risks']['size'] or r['audit_risks']['full_fusion'] != old['audit_risks']['fusion_oof']:
                raise ValueError('Control risks changed')
            if r['old_five_common_accepted'] != (r['pred'] is not None and all(old['audit_risks'][n] <= oldcuts[n] for n in core.NAMES)):
                raise ValueError('Original common group changed')
            for n in METHODS[1:]:
                if r['ensemble_decisions'][n] != response.size_decision(r['original_simple_decision'], r['audit_risks'][n], report['cutoffs'][n]):
                    raise ValueError('Other flags or size decision changed')
        if helper.metrics(rows, METHODS, report['cutoffs']) != report['results'][role] or gains(rows, report['cutoffs']) != report['gains'][role]:
            raise ValueError('Metrics or gains differ')
        for n, v in report['results'][role].items():
            for g, part in groups(rows).items():
                checks.append(independent_check(Ranking(part, n), v['at_frozen_VAL_cut'][g], report['cutoffs'][n]))
        if role == 'VAL':
            for n in ('fold_simple','fold_fusion'):
                good = sorted(r['audit_risks'][n] for r in rows if bad(r) is False)
                real = sorted(r['audit_risks'][n] for r in rows if bad(r) is False and r['domain'] == 'real')
                if report['cutoffs'][n] != max(good[652], real[188]):
                    raise ValueError('Global VAL protection cutoff differs')
    if maximum > 1e-10 or proof_maximum > 1e-10:
        raise ValueError('Independent scalar inference differs')
    write(out/receipt, dict(status='INDEPENDENT_REVIEW_PASS', groups=len(checks),
        pairs=sum(c['pair_comparisons'] for c in checks), curve_points=sum(c['curve_points'] for c in checks),
        scalar_risk_max_abs=maximum, scalar_proof_max_abs=proof_maximum, held_TRAIN_rows=2558,
        fits_performed_in_review=0, report_sha256=sha(out/'report.json')))
    print('INDEPENDENT_REVIEW_PASS', len(checks), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--review', action='store_true'); p.add_argument('--receipt', default='independent_review.json'); a = p.parse_args()
    if a.review:
        review(a.out.resolve(), a.receipt)
    else:
        try:
            run(a.out.resolve())
        except Exception as error:
            if a.out.is_dir():
                write(a.out/'failure.json', dict(error=repr(error)))
            raise
