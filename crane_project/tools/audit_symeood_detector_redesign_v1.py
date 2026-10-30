"""CPU/read-only audit of saved EOOD/SymEOOD sources and diagnostics.

No torch import, model loading, inference, training, or TEST analysis.
Writes only a new evidence directory; never changes an experiment source.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]
E_CFG = 'work_dirs/crane_eood_k1_port_day2night_aug_b_v1/crane_eood_k1_port_day2night_aug_b_v1.py'
S_CFG = 'work_dirs/symeood_110_retrain_seed0_20261026_v1/configs/Coff_seed0.py'
DIAG = 'work_dirs/symeood_110_candidate_diagnostic_20261030_v1'
AUDIT = 'work_dirs/symeood_b2_20261030_v1/downloaded_result'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sources = {}

    def read(relative):
        data = (ROOT / relative).read_bytes()
        sources[relative] = sha(data)
        return data

    def load(relative):
        return json.loads(read(relative))

    index = load('docs/detection/证据索引/SymEOOD历史训练文件索引_20261021.json')
    protocol = load('work_dirs/symeood_110_retrain_seed0_20261026_v1/protocol.json')
    e_cfg_sha, s_cfg_sha = sha(read(E_CFG)), sha(read(S_CFG))
    assert e_cfg_sha == index['captured_files'][E_CFG]['sha256']
    assert s_cfg_sha == protocol['config_sha256']
    # These sealed expanded configs consist of ordinary configuration assignments.
    e, s = runpy.run_path(str(ROOT / E_CFG)), runpy.run_path(str(ROOT / S_CFG))
    e_main = e['model']['bbox_head']['predictors'][0]
    e_aux = e['model']['bbox_head']['predictors'][1]
    s_main, s_aux = s['model']['bbox_head'], s['model']['aux_bbox_head'][0]
    matched_fields = ('optimizer', 'optimizer_config', 'lr_config', 'runner',
                      'data', 'train_pipeline', 'test_pipeline')
    matches = {key: e[key] == s[key] for key in matched_fields}
    matches.update(backbone=e['model']['backbone'] == s['model']['backbone'],
                   neck=e['model']['neck'] == s['model']['neck'],
                   main_anchors=e_main['anchor_generator'] == s_main['anchor_generator'],
                   main_coder=e_main['bbox_coder'] == s_main['bbox_coder'])
    assert all(matches.values())
    assert not s_main['loss_cls']['use_geometry_weight']

    e_revision = index['historical_sources']['a074048']['full_commit']
    e_files = ('mmrotate/models/dense_heads/eood_head.py',
               'mmrotate/models/dense_heads/rotated_eood_head.py',
               'mmrotate/models/detectors/eood.py',
               'mmrotate/core/bbox/assigners/pola.py',
               'mmrotate/models/dense_heads/rotated_anchor_head.py',
               'mmrotate/models/dense_heads/rotated_atss_head.py',
               'mmrotate/models/losses/rotated_iou_loss.py',
               'mmrotate/core/hooks/set_epoch_info_hook.py')
    historical = {}
    for relative in e_files:
        result = subprocess.run(['git', '-C', str(ROOT), 'show',
                                 e_revision + ':' + relative], capture_output=True, check=True)
        current = read(relative)
        historical[relative] = dict(git_sha256=sha(result.stdout),
                                    current_sha256=sha(current),
                                    equal=result.stdout == current)
    assert all(row['equal'] for row in historical.values())
    s_files = ('mmrotate/models/dense_heads/sym_eood_head.py',
               'mmrotate/models/detectors/sym_eood_detector.py',
               'mmrotate/core/bbox/assigners/sym_pola.py',
               'mmrotate/models/losses/sym_kld_calculator.py',
               'mmrotate/models/losses/sym_kld_loss.py',
               'mmrotate/models/losses/sym_nfl_loss.py')
    for relative in s_files:
        assert sha(read(relative)) == protocol['sources'][relative], relative

    differences = [
        dict(item='main feature path', E='FPN -> shared 4-layer cls/reg towers -> predictors',
             S='FPN -> prediction convolutions; zero towers despite stacked_convs=4'),
        dict(item='auxiliary feature path', E='same tower outputs as main predictor',
             S='direct FPN, duplicated as cls/reg feature tuple'),
        dict(item='main classification', E=e_main['loss_cls'], S=s_main['loss_cls'],
             note='C off: SymNFL is ordinary hard-label focal, not quality targets'),
        dict(item='effective main regression', E=e_main['loss_iou'], S=s_main['loss_bbox'],
             note='EOOD computes L1 but excludes it from returned training loss dict'),
        dict(item='auxiliary classification', E=e_aux['loss_cls'], S=s_aux['loss_cls']),
        dict(item='auxiliary regression', E=e_aux['loss_bbox'], S=s_aux['loss_bbox']),
        dict(item='main assignment', E=e_main['train_cfg'], S=s['model']['train_cfg'],
             note='E: MaxIoU for epochs 0..3 then POLA; S: prediction-based 9->1 schedule and KLD/tau'),
        dict(item='inference', E=e_main['test_cfg'], S=s['model']['test_cfg'],
             note='E: per-level top-k and rotated NMS; S: ignores nms_pre/NMS and directly selects Top1'),
    ]
    # Verify the historical ablation switches: A0 does not restore EOOD's schedule.
    for name in ('000', '100', 'v1_base'):
        read('crane_project/configs/crane_symeood_k1_ablation_' + name +
             ('.py' if name == 'v1_base' else '_v1.py'))

    scope = load(DIAG + '/scope.json')
    completion = load(DIAG + '/completion.json')
    assert completion['status'] == 'COMPLETE' and completion['training_updates'] == 0
    assert not completion['TEST'] and not completion['VAL']
    assert scope['pairs'] == completion['pairs'] and len(scope['pairs']) == 32
    assert sha(read('crane_project/tools/diagnose_symeood_110_candidates_v2.py')) == scope['script_sha256']
    train_rows, gradient_rows, saturation, counts = [], [], {}, {}
    for arm in ('E', 'S', 'T4'):
        d = load(DIAG + '/' + arm + '.json')
        assert d == completion['arms'][arm]
        assert d['state_unchanged'] and d['state_before'] == d['state_after']
        assert len(d['ranking']) == 128 and len(d['gradients']) == 64
        assert all(r['bag_top1_replay_match'] for r in d['ranking'])
        assert d['config_sha256'] == sha(read(d['config']))
        counts[arm] = d['counts']
        for scale in (1.0, 0.5):
            for domain in ('all', 'real', 'sim'):
                rows = [r for r in d['ranking'] if r['scale'] == scale
                        and (domain == 'all' or r['image'].startswith(domain + '_'))]
                categories = dict(native_pass75=0, no_good_candidate=0,
                                  good_candidate_filtered=0, good_candidate_not_selected=0)
                for r in rows:
                    if r['native_top1_iou'] is not None and r['native_top1_iou'] >= .75:
                        key = 'native_pass75'
                    elif r['oracle_all_iou'] < .75:
                        key = 'no_good_candidate'
                    elif r['oracle_deployed_bag_iou'] is None or r['oracle_deployed_bag_iou'] < .75:
                        key = 'good_candidate_filtered'
                    else:
                        key = 'good_candidate_not_selected'
                    categories[key] += 1
                assert sum(categories.values()) == len(rows)
                assert len(rows) == (64 if domain == 'all' else 32)
                train_rows.append(dict(arm=arm, scale=scale, domain=domain, frames=len(rows),
                    categories=categories, mean_native_iou=statistics.mean(r['native_top1_iou'] or 0 for r in rows),
                    mean_bag_best_iou=statistics.mean(r['oracle_deployed_bag_iou'] or 0 for r in rows),
                    mean_all_best_iou=statistics.mean(r['oracle_all_iou'] for r in rows)))
            gradients = [r['fpn_gradient_l2'] for r in d['gradients'] if r['scale'] == scale]
            gradient_rows.append(dict(arm=arm, scale=scale, batches=len(gradients),
                median_l2={key: statistics.median(r[key] for r in gradients)
                           for key in ('main_cls', 'main_reg', 'auxiliary')}))
        raw = [v for r in d['assigned_regression'] for v in r['raw_kld']]
        if raw:
            saturation[arm] = dict(assigned_positives=len(raw), above120=sum(v > 120 for v in raw))

    manifest = load(AUDIT + '/source_audit_manifest.json')
    for relative, digest in manifest.items():
        assert sha(read(AUDIT + '/' + relative)) == digest, relative
    saved = load(AUDIT + '/source_audit_v2/summary.json')
    assert saved['passed'] and saved['model_forwards'] == saved['optimizer_updates'] == 0
    val_metrics = [r for r in saved['metrics'] if r['split'] == 'val']
    frame_groups = {}
    # The input archive contains previously exposed TEST; only VAL enters this audit.
    for line in read(AUDIT + '/source_audit_v2/per_frame.jsonl').splitlines():
        row = json.loads(line)
        if row['split'] == 'val':
            frame_groups.setdefault(row['side'], []).append(row)
    val_scores = []
    for arm, rows in frame_groups.items():
        assert len(rows) == 887 and len({r['image'] for r in rows}) == 887
        overall = next(r for r in val_metrics if r['side'] == arm and r['domain'] == 'all')
        assert sum(r['riou'] >= .75 for r in rows) == overall['TP75']
        assert sum(r['output'] for r in rows) == overall['outputs']
        real = [r for r in rows if r['domain'] == 'real' and r['output']]
        ranked = sorted(real, key=lambda r: (-r['score'], r['image']))
        boundary = ranked[99]['score']
        val_scores.append(dict(arm=arm, real_top100_low75=sum(r['riou'] < .75 for r in ranked[:100]),
            boundary_score=boundary, boundary_score_tie_count=sum(r['score'] == boundary for r in ranked),
            note='Diagnostic count, deterministic image tie-break, not an AP recomputation'))

    report = dict(status='VERIFIED', model_forwards=0, training_updates=0,
        GPU_used=False, new_TEST=False, TEST_analysis=False, selection_changed=False,
        config_matches=matches, config_differences=differences,
        source_identity=dict(EOOD_git_commit=e_revision, EOOD_core_files=historical,
            Sym_core_files_verified=list(s_files),
            limitation='Historical EOOD full uncommitted runtime was not captured; verified Git/core files and sealed config only.'),
        ablation_boundary='A toggles assignment geometry inside the Sym skeleton; A0 retains the Sym schedule/tau and is not original EOOD. 000 collapsed; no clean A gain versus EOOD is established.',
        TRAIN_candidate_readback=train_rows, TRAIN_gradient_readback=gradient_rows,
        TRAIN_saturation_readback=saturation, parameter_counts=counts,
        VAL_metrics=val_metrics, VAL_score_diagnostics=val_scores,
        limitations=['64 distinct TRAIN identities, two fixed scales, selected late checkpoints only; not unbiased generalization evidence.',
            'The all-candidate oracle includes arbitrarily low scores; it is a diagnosis bound, not an output rule.',
            'Filtered failures combine threshold and EOOD pre-top-k effects; cached records cannot separate them.',
            'Gradient magnitudes across architectures/losses are observations, not causal proof or proposed loss weights.',
            'No cached full candidate bags for A-only/R4 complete VAL; do not infer their candidate failure mechanism.',
            'T4 restores both towers within the Sym skeleton and does not restore the EOOD auxiliary path/loss balance.'],
        decision='Keep B2 untrained; first design minimal A/B changes on the effective EOOD skeleton. Do not assert either A effectiveness or A uselessness.',
        sources=sources)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], config_matches=len(matches),
        source_files=len(sources), TRAIN_partitions=len(train_rows), VAL_metric_rows=len(val_metrics),
        GPU_used=False, model_forwards=0, training_updates=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
