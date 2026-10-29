#!/usr/bin/env python3
"""CPU-only audit of frozen source identities, logs and existing VAL rows."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize_probe(path):
    r = json.loads(path.read_text())
    assert r['status'] == 'COMPLETE' and r['training_updates'] == 0
    assert r['TEST'] is False and r['VAL'] is False
    assert r['script_sha256'] == sha(ROOT / 'crane_project/tools/diagnose_symeood_110_frozen_train_v1.py')
    keys = []
    arms = {}
    for arm, d in r['arms'].items():
        assert d['state_before'] == d['state_after'] and d['state_unchanged']
        assert len(d['ranking']) == 16 and len(d['gradients']) == 8
        assert all(x['bag_top1_replay_match'] for x in d['ranking'])
        keys.append([(x['image'], x['scale']) for x in d['ranking']])
        rankings = []
        for scale in (1., .5):
            rr = [x for x in d['ranking'] if x['scale'] == scale]
            rankings.append(dict(scale=scale, views=len(rr),
                mean_native_iou=statistics.mean(x['native_top1_iou'] or 0 for x in rr),
                mean_deployed_bag_oracle_iou=statistics.mean(x['oracle_deployed_bag_iou'] or 0 for x in rr),
                mean_all_oracle_iou=statistics.mean(x['oracle_all_iou'] for x in rr),
                native_below75=sum((x['native_top1_iou'] or 0) < .75 for x in rr)))
        dd = [v for x in d['assigned_regression'] for v in x['raw_kld']]
        arms[arm] = dict(counts=d['counts'], ranking=rankings,
            median_fpn_gradient_l2={k: statistics.median(x['fpn_gradient_l2'][k] for x in d['gradients'])
                                    for k in ('main_cls', 'main_reg', 'auxiliary')},
            median_auxiliary_to_regression_fpn_gradient_ratio=statistics.median(
                x['fpn_gradient_l2']['auxiliary']/x['fpn_gradient_l2']['main_reg'] for x in d['gradients']),
            assigned_kld_count=len(dd), assigned_kld_max=max(dd) if dd else None,
            assigned_kld_above120=sum(v > 120 for v in dd))
    assert keys[0] == keys[1]
    assert arms['E']['counts']['total']-arms['S']['counts']['total'] == 4720640
    assert arms['S']['assigned_kld_count'] == 16
    return dict(status='VERIFIED', receipt_sha256=sha(path), paired_views=16,
        native_top1_replay_checks=32, state_unchanged_both=True, arms=arms,
        training_updates=0, TEST=False, limitations=[
            'Eight TRAIN identities at two deterministic scales; every native output IoU>=.75.',
            'Late selected checkpoints only; no claim about early training or all difficult frames.',
            'Gradient norms at each model own parameters do not prove causal deficiency or gradient conflict.'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--probe', type=Path, help='Summarize only this completed frozen TRAIN probe.')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    if a.probe is not None:
        result = summarize_probe(a.probe)
        (a.out / 'summary.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    sources = {}
    def read(p):
        p = ROOT / p
        sources[str(p.relative_to(ROOT))] = sha(p)
        return json.loads(p.read_text())
    protocol = read('work_dirs/symeood_110_retrain_seed0_20261026_v1/protocol.json')
    checks = []
    for name in ('mmrotate/models/dense_heads/sym_eood_head.py',
                 'mmrotate/models/dense_heads/rotated_retina_head.py',
                 'mmrotate/core/bbox/assigners/sym_pola.py',
                 'mmrotate/models/losses/sym_kld_loss.py',
                 'mmrotate/models/losses/sym_kld_calculator.py'):
        got = sha(ROOT / name)
        checks.append(dict(path=name, sha256=got, matches_frozen=got == protocol['sources'][name]))
    assert all(x['matches_frozen'] for x in checks)
    lognames = dict(E='work_dirs/crane_eood_k1_port_day2night_aug_b_v1/20260930_083040.log.json',
                   S='work_dirs/symeood_110_retrain_seed0_20261026_v1/110/20261026_162930.log.json')
    logs = {}
    for arm, name in lognames.items():
        p = ROOT / name
        sources[name] = sha(p)
        records = [json.loads(x) for x in p.read_text().splitlines()]
        train = [x for x in records if x.get('mode') == 'train']
        assert all(math.isfinite(v) for x in train for k, v in x.items()
                   if isinstance(v, (float, int)))
        epochs = []
        for ep in sorted({x['epoch'] for x in train}):
            rr = [x for x in train if x['epoch'] == ep]
            keys = [k for k in rr[0] if 'loss' in k or k == 'grad_norm']
            epochs.append(dict(epoch=ep, logged_windows=len(rr),
                means={k: statistics.mean(x[k] for x in rr) for k in keys},
                logged_grad_norm_above10=sum(x['grad_norm'] > 10 for x in rr)))
        logs[arm] = dict(logged_windows=len(train), per_epoch=epochs,
            note='Logger-window values, not a per-step clipping or KLD-saturation count.')
    names = dict(E='work_dirs/detector_module_ablation_v1_20261028/eood_l/epoch_17_val.rows.jsonl.gz',
        S='work_dirs/symeood_110_retrain_seed0_20261026_v1/downstream/lvsr/fixed_b/epoch_01_val.rows.jsonl.gz')
    rows = {}
    for arm, name in names.items():
        p = ROOT / name
        sources[name] = sha(p)
        with gzip.open(p, 'rt') as f:
            rr = [json.loads(x) for x in f]
        assert len(rr) == 887 and all(x['split'] == 'val' for x in rr)
        rows[arm] = {x['image']: x for x in rr}
        assert len(rows[arm]) == 887
    assert rows['E'].keys() == rows['S'].keys()
    comparison = []
    for domain in ('overall', 'real', 'sim'):
        keys = [k for k, x in rows['E'].items() if domain == 'overall' or x['domain'] == domain]
        for k in keys:
            assert max(abs(x - y) for x, y in zip(rows['E'][k]['gt'], rows['S'][k]['gt'])) < 1e-7
        common = [k for k in keys if rows['E'][k]['b'] and rows['S'][k]['b']]
        result = dict(group=domain, frames=len(keys), common_outputs=len(common), arms={})
        for arm in ('E', 'S'):
            rr = [rows[arm][k] for k in keys]
            valid = [x for x in rr if x['b']]
            mm = [x['metrics']['b'] for x in valid]
            result['arms'][arm] = dict(outputs=len(valid),
                center_correct=sum(x['center_hit'] for x in mm),
                localization_hits=sum(x['riou'] >= .5 for x in mm),
                RIoU_full=sum(x['riou'] for x in mm) / len(keys),
                center_MAE_px=statistics.mean(x['center_error_px'] for x in mm),
                long_MAE_pct=100*statistics.mean(x['long_relative_error'] for x in mm),
                short_MAE_pct=100*statistics.mean(x['short_relative_error'] for x in mm),
                joint10=sum(x['joint_size10'] for x in mm),
                common_center_MAE_px=statistics.mean(rows[arm][k]['metrics']['b']['center_error_px'] for k in common),
                common_RIoU=statistics.mean(rows[arm][k]['metrics']['b']['riou'] for k in common))
        result['S_minus_E_common_RIoU'] = result['arms']['S']['common_RIoU']-result['arms']['E']['common_RIoU']
        comparison.append(result)
    result = dict(status='COMPLETE', source_checks=checks, logs=logs, VAL=comparison,
        source_SHA256=sources, training_updates=0, model_forwards=0, TEST_read=False,
        oracle_ranking_and_actual_assigned_loss='See separate bounded GPU TRAIN diagnostic; cannot infer from Top1 rows.',
        active_tower_additional_E_parameters=8*(256*256*3*3+256))
    (a.out / 'audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(status=result['status'], source_checks=len(checks), VAL=comparison), indent=2))


if __name__ == '__main__':
    main()
