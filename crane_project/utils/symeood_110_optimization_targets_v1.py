"""Predefined VAL protection and AP75 improvement goals, no model code."""
import math

TOLERANCE = 1e-5
MIN_AP75_GAIN_PP = 1.0


def assess(candidate, sim, reference, reference_sim):
    if candidate['frames'] != 887 or sim['frames'] != 512:
        raise ValueError('Requires the complete frozen VAL and Sim VAL')
    checks = {}
    for k in ('AP75', 'mean_RIoU', 'center_MAE_px', 'long_MAE_pct', 'short_MAE_pct'):
        value, baseline = candidate[k], reference[k]
        if not math.isfinite(value) or not math.isfinite(baseline):
            raise ValueError('Nonfinite metric: ' + k)
        checks[k] = (value >= baseline - TOLERANCE if k in ('AP75', 'mean_RIoU')
                     else value <= baseline + TOLERANCE)
    tp50 = round(candidate['Recall_pct'] * 887 / 100)
    assert abs(candidate['Recall_pct']-100*tp50/887) <= TOLERANCE
    checks.update(Recall_count=tp50 >= 883,
                  center_correct=candidate['center_correct'] >= 880,
                  MCML=candidate['MCML_max_native'] <= 3,
                  joint10=candidate['joint10'] >= 632,
                  Sim_angle=(math.isfinite(sim['pure_angle_RMSE_deg']) and
                      sim['pure_angle_RMSE_deg'] <= reference_sim['pure_angle_RMSE_deg'] + TOLERANCE))
    protected = all(checks.values())
    return dict(checks=checks, protected=protected,
                AP75_gain_pp=candidate['AP75']-reference['AP75'],
                meaningful_gain=candidate['AP75'] >= reference['AP75']+MIN_AP75_GAIN_PP-TOLERANCE,
                eligible=protected and candidate['AP75'] >= reference['AP75']+MIN_AP75_GAIN_PP-TOLERANCE)


def select(metrics):
    reference = next(r for r in metrics if r['side']=='S' and r['selected'] and r['domain']=='all')
    sim_reference = next(r for r in metrics if r['side']=='S' and r['selected'] and r['domain']=='sim')
    rows = [r for r in metrics if r['side']=='R4' and r['domain']=='all']
    assert {r['epoch'] for r in rows} == {'epoch_16','epoch_18','epoch_20','epoch_22','epoch_24'}
    assessments = []
    for row in rows:
        sim = next(r for r in metrics if r['side']=='R4' and r['domain']=='sim' and r['epoch']==row['epoch'])
        assessments.append(dict(epoch=row['epoch'], **assess(row,sim,reference,sim_reference)))
    eligible = {r['epoch'] for r in assessments if r['eligible']}
    ordered = sorted((r for r in rows if r['epoch'] in eligible),
                     key=lambda r:(-r['AP75'],-r['mean_RIoU'],int(r['epoch'].split('_')[1])))
    return dict(selected=ordered[0]['epoch'] if ordered else None,
                eligible_epochs=sorted(eligible), assessments=assessments,
                reference='frozen original S epoch20, complete VAL, native outputs',
                minimum_AP75_gain_pp=MIN_AP75_GAIN_PP, numeric_tolerance=TOLERANCE,
                tolerance_is_not_performance_tradeoff=True, TEST=False, automatic_adoption=False)
