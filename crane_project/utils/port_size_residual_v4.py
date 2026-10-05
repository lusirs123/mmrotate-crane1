"""Frozen-box SIZE residual experiment; NumPy oracle and GT-free readout.

This is a project adaptation of post-hoc dispersion learning, not a PQA
reproduction or a calibrated probability claim. No box is reconstructed.
"""
from copy import deepcopy
import math

import numpy as np
from crane_project.utils import port_simple_component_reliability_v1 as simple

VERSION = 'port_size_residual_v4'
ARMS = ('a0', 'a1')
SETTINGS = dict(seed=1701, epochs=4, fit_frames=384, holdout_frames=432,
    guard_frames=32, slots_per_arm=1536, initial_views=14, smoke_updates=4,
    lr=.001, weight_decay=0., clip_norm=10., hidden_channels=8,
    sigma_floor=.001, initial_sigma=.05, output_weight_std=.001,
    moment_epsilon=1e-6, descriptor_scale_floor=1e-6,
    size_relative_limit=.1, center_limit_px=15., coverage_fractions=[.9, .95])
CONTINUATION = dict(real_FA_gain=5, real_AUROC_gain=.02,
    primary_accept_counts={'real':333, 'sim':510}, tolerance=1e-10,
    per_video_no_FA_FR_regression=True, final_workpoint_created=False)


def protocol_document():
    return dict(protocol=VERSION, settings=deepcopy(SETTINGS),
        arms={'a0':'box_descriptor_only_image_code_zero',
              'a1':'same_head_plus_8_channel_image_mean_std'},
        initialization='identical_state_seed1701_random_output_weights_std0.001',
        sampling='same384_fit_identities_same_epoch_shuffle_no_error_resampling_no_augmentation',
        missing_fit_output='skip_for_both_arms_no_GT_fill_report_slots_and_actual_updates',
        objective='equal_axes_zero_mean_diagonal_Gaussian_NLL_coefficient1_no_auxiliary_loss',
        target='signed_log(pred_canonical_long_short/GT_canonical_long_short)',
        distribution='zero_mean_independent_axes_assumption_not_proven_error_independence',
        predicted_mean='frozen_log_box_edges_not_trainable_no_box_correction',
        sigma='softplus(raw)+0.001_no_upper_clamp',
        reader='1-product(P(log0.9<=residual<=log1.1))_model_assumption_score',
        feature='frozen_P3_final_midpoint_aligned9x9_context1.5_then_256_to8_1x1_weighted_mean_std',
        descriptor='existing3_descriptor_standardized_on_fit_outputs_only',
        checkpoint='fixed_epoch04_fresh_after_discarded_smoke_no_resume_no_selection',
        calibration='none_this_version_ranking_and_distribution_diagnostics_separate',
        evaluation_roles={'residual_holdout_train':432, 'val':887},
        continuation=deepcopy(CONTINUATION),
        literature={'url':'https://arxiv.org/html/2607.26921v1',
                    'status':'2026_preprint',
                    'borrowed':'frozen_predictions_residual_NLL_and_separate_ranking_calibration',
                    'adapted':'OBB_log_edge_diagonal_dispersion_compact_image_side_head'},
        GT_online=False, domain_sequence_history_online=False,
        detector_updates=0, midpoint_updates=0, baseline_policy_updates=0,
        test_read=False, test_repeatedly_exposed=True,
        comparison_only=True, final_deployment_policy_created=False,
        residual_holdout_is_not_detector_or_simple_holdout=True)


def residual_target(pred_original, gt_original):
    """Offline supervision only. Sorting edges preserves swap/pi equivalence."""
    p = simple.prediction(pred_original)
    if p is None:
        return None
    b, g = simple.canonical(p[:5]), simple.canonical(gt_original)
    return np.log(b[2:4])-np.log(g[2:4])


def target_is_good(residual):
    e = np.asarray(residual, dtype=float)
    if e.shape != (2,) or not np.isfinite(e).all():
        raise ValueError('Expected two finite signed log residuals')
    # Same definition as max(abs(pred/GT - 1)) <= .1; asymmetric log bounds.
    return bool(np.all(e >= math.log(.9)-1e-12) and np.all(e <= math.log(1.1)+1e-12))


def fit_standardizer(rows):
    if any(r['split'] not in ('train', 'train_sim') for r in rows):
        raise ValueError('Only fitting TRAIN may standardize descriptors')
    values = [simple.descriptor(r['pred'], r['image_size']) for r in rows if r['pred'] is not None]
    if not values:
        raise ValueError('No fitting outputs')
    x = np.asarray(values, dtype=float)
    return dict(mean=x.mean(0).tolist(),
        scale=np.maximum(x.std(0), SETTINGS['descriptor_scale_floor']).tolist(),
        fit_outputs=len(values), role='fit_TRAIN_only_no_residual_labels_used')


def sigma_from_raw(raw):
    z = np.asarray(raw, dtype=float)
    if z.shape != (2,) or not np.isfinite(z).all():
        raise ValueError('Expected two finite dispersion outputs')
    return np.logaddexp(0., z)+SETTINGS['sigma_floor']


def nll_numpy(raw, residual):
    e = np.asarray(residual, dtype=float)
    if e.shape != (2,) or not np.isfinite(e).all():
        raise ValueError('Invalid residual supervision')
    sigma = sigma_from_raw(raw)
    quadratic = float(.5*np.mean((e/sigma)**2))
    normalization = float(np.mean(np.log(sigma))+.5*math.log(2*math.pi))
    derivative = .5*(1/sigma-e**2/sigma**3)*simple.sigmoid(np.asarray(raw, dtype=float))
    return dict(total=quadratic+normalization, quadratic=quadratic,
                normalization=normalization, sigma=sigma.tolist(), derivative=derivative.tolist(),
                quadratic_derivative=(-.5*e**2/sigma**3*simple.sigmoid(raw)).tolist(),
                normalization_derivative=(.5/sigma*simple.sigmoid(raw)).tolist())


def distribution_reading(raw):
    """Online raw dispersion only; never takes GT/domain/sequence/threshold."""
    if raw is None:
        return dict(defined=False, risk=None, sigma=None, reason='no_usable_image_evidence')
    sigma = sigma_from_raw(raw)
    probabilities = [.5*(math.erf(math.log(1.1)/(s*math.sqrt(2)))
                          -math.erf(math.log(.9)/(s*math.sqrt(2)))) for s in sigma]
    good = float(np.clip(np.prod(probabilities), 0., 1.))
    return dict(defined=True, risk=1-good, sigma=sigma.tolist(),
        model_good_mass=good, model_axis_good_mass=probabilities,
        reason='defined', calibrated_error_probability=False,
        assumptions='zero_mean_diagonal_Gaussian_log_residual')


def online_evidence(raw_by_arm, final_box, image_size, runtime):
    """Attach SIZE scores while retaining all three original flags and box."""
    before = deepcopy(final_box)
    frozen = runtime.decide(final_box, image_size)
    score_only = runtime.decide(final_box, image_size, 'score_only')
    for value in (frozen, score_only):
        if value['final_box_original'] != before or value['center_accepted'] != (before is not None):
            raise ValueError('Frozen output/center changed')
    if set(raw_by_arm) != set(ARMS):
        raise ValueError('Both fixed arms required')
    arms = {a:distribution_reading(raw_by_arm[a] if before is not None else None) for a in ARMS}
    if final_box != before:
        raise ValueError('Evidence computation mutated the box')
    return dict(arms=arms, frozen=frozen, frozen_score_only=score_only,
                candidate_flags_created=False, GT_online=False)


def support_summary(rows):
    """Offline supervision support, not a performance claim."""
    values = [residual_target(r['pred'], r['gt']) for r in rows if r['pred'] is not None]
    if not values:
        return dict(frames=len(rows), outputs=0, size_bad=0, signed=None)
    e = np.asarray(values)
    return dict(frames=len(rows), outputs=len(e),
        size_bad=sum(simple.geometry_errors(r['gt'],r['pred'])['size_max_relative']>.1 for r in rows if r['pred'] is not None),
        signed=dict(mean=e.mean(0).tolist(), p10=np.percentile(e,10,axis=0).tolist(),
                    median=np.median(e,axis=0).tolist(), p90=np.percentile(e,90,axis=0).tolist(),
                    under_10pct=(e<math.log(.9)).sum(0).tolist(),
                    over_10pct=(e>math.log(1.1)).sum(0).tolist(),
                    second_moment=np.mean(e**2,axis=0).tolist()),
        many_video_frames_are_not_independent_error_types=True)


def numerical_probe(residual):
    raw = np.array([math.log(math.expm1(.05-.001))]*2)
    cases = []
    for e in (np.zeros(2), np.asarray(residual), np.array([-.25,.4])):
        exact = nll_numpy(raw, e); derivatives=[]
        for axis in (0,1):
            positive=raw.copy(); negative=raw.copy(); positive[axis]+=1e-5; negative[axis]-=1e-5
            derivatives.append((nll_numpy(positive,e)['total']-nll_numpy(negative,e)['total'])/2e-5)
        passed = bool(np.allclose(derivatives,exact['derivative'],rtol=1e-7,atol=1e-8))
        cases.append(dict(residual=e.tolist(),analytic=exact['derivative'],finite_difference=derivatives,passed=passed))
    return dict(cases=cases, passed=all(v['passed'] for v in cases))
