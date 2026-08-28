"""Metric-only policy for the relaxed Dual-Tower V2.1 source gate."""


HIGHER_IS_BETTER = (
    'real/R_center(%)',
    'real/mean_RIoU',
    'real/ACI',
    'sim/R_center(%)',
    'sim/mean_RIoU',
    'sim/ACI',
)

LOWER_IS_BETTER = (
    'real/DFR(%/frame)',
    'sim/A-RMSE(deg)',
    'sim/DFR(%/frame)',
)


def relaxed_composite_gate(candidate, reference, min_composite_gain=0.005):
    """Allow average improvement while bounding every individual regression.

    The reference is the audited recomposed Dual-Tower V2 source-val stream.
    Continuous metrics contribute equal relative utility.  Absolute guardrails
    remain interpretable in each paper metric and prevent one large gain from
    hiding a severe regression elsewhere.
    """
    required = set(HIGHER_IS_BETTER + LOWER_IS_BETTER)
    required.update({
        'real/TDR_w10(%)', 'sim/TDR_w10(%)',
        'real/MCML_max(frames)', 'sim/MCML_max(frames)'})
    missing = sorted(required - set(candidate))
    missing += sorted(required - set(reference))
    if missing:
        raise RuntimeError('Source-gate metric is missing: ' + ', '.join(
            sorted(set(missing))))

    relative_utility = {}
    for key in HIGHER_IS_BETTER:
        denominator = max(abs(float(reference[key])), 1e-9)
        relative_utility[key] = (
            float(candidate[key]) - float(reference[key])) / denominator
    for key in LOWER_IS_BETTER:
        denominator = max(abs(float(reference[key])), 1e-9)
        relative_utility[key] = (
            float(reference[key]) - float(candidate[key])) / denominator
    composite = sum(relative_utility.values()) / len(relative_utility)

    checks = dict(
        composite_mean_relative_gain=(composite >= min_composite_gain),
        real_center_drop_le_1pp=(
            float(candidate['real/R_center(%)']) >=
            float(reference['real/R_center(%)']) - 1.0),
        sim_center_drop_le_1pp=(
            float(candidate['sim/R_center(%)']) >=
            float(reference['sim/R_center(%)']) - 1.0),
        real_riou_drop_le_0p03=(
            float(candidate['real/mean_RIoU']) >=
            float(reference['real/mean_RIoU']) - 0.03),
        sim_riou_drop_le_0p03=(
            float(candidate['sim/mean_RIoU']) >=
            float(reference['sim/mean_RIoU']) - 0.03),
        real_dfr_increase_le_0p75pp=(
            float(candidate['real/DFR(%/frame)']) <=
            float(reference['real/DFR(%/frame)']) + 0.75),
        sim_dfr_increase_le_0p75pp=(
            float(candidate['sim/DFR(%/frame)']) <=
            float(reference['sim/DFR(%/frame)']) + 0.75),
        real_aci_drop_le_0p02=(
            float(candidate['real/ACI']) >=
            float(reference['real/ACI']) - 0.02),
        sim_aci_drop_le_0p02=(
            float(candidate['sim/ACI']) >=
            float(reference['sim/ACI']) - 0.02),
        sim_a_rmse_increase_le_2deg=(
            float(candidate['sim/A-RMSE(deg)']) <=
            float(reference['sim/A-RMSE(deg)']) + 2.0),
        real_tdr_ge_99=(float(candidate['real/TDR_w10(%)']) >= 99.0),
        sim_tdr_ge_99=(float(candidate['sim/TDR_w10(%)']) >= 99.0),
        real_mcml_le_5=(
            int(candidate['real/MCML_max(frames)']) <= 5),
        sim_mcml_le_5=(
            int(candidate['sim/MCML_max(frames)']) <= 5),
    )
    return dict(
        reference_policy='audited_dual_tower_v2_source_val',
        metric_directions=dict(
            higher_is_better=list(HIGHER_IS_BETTER),
            lower_is_better=list(LOWER_IS_BETTER)),
        relative_utility=relative_utility,
        composite_mean_relative_gain=composite,
        min_composite_gain=float(min_composite_gain),
        guardrails=dict(
            center_drop_pp=1.0,
            mean_riou_drop=0.03,
            dfr_increase_pp_per_frame=0.75,
            aci_drop=0.02,
            sim_a_rmse_increase_deg=2.0,
            tdr_floor_percent=99.0,
            mcml_max_frames=5),
        checks=checks,
        passed=all(checks.values()))
