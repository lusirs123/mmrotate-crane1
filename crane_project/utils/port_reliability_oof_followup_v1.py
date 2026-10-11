"""Separate TRAIN forensics and VAL-selected auxiliary-source contracts."""
from copy import deepcopy
import math
import numpy as np
from crane_project.utils import port_reliability_oof_training_v2 as repair
from crane_project.utils import port_reliability_complementarity_v1 as metrics

VERSION = 'port_reliability_oof_followup_v1'
TRACE = dict(max_updates=7000, component_interval=50, dense_start=6000,
             gradient_limit=1000., main_cls_limit=5., main_reg_limit=10.)
SELECTION = dict(detector_epochs=list(range(2,25,2)), midpoint_epochs=[1,2,3],
    selector='ckpt_sweep.select_best_checkpoint',
    selection_config=dict(r_center_delta=.005,mcml_limit=5,weight_sim=.7,weight_real=.3,
        soft_w_tdr=.35,soft_w_r_center=.25,soft_w_aci=.20,soft_w_angle=.20),
    fallback='unchanged_main_detector_two_level_fallback',
    extra_size_or_95_90_selection_gates=False,
    validation_role='shared_development_VAL_not_independent_validation',
    TEST_used=False, weights_averaged=False)
HOOK = 'mmrotate.core.hooks.port_oof_trace_hook'


def json_safe(value):
    """Preserve invalid numeric evidence explicitly, never replace it with zero."""
    if isinstance(value,np.ndarray): return json_safe(value.tolist())
    if isinstance(value,np.generic): return json_safe(value.item())
    if isinstance(value,float) and not math.isfinite(value): return repr(value)
    if isinstance(value,dict): return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    return value


def compare_replay_trace(actual_loss, actual_norm, original):
    """Compare saved scalar evidence, including json_safe NaN/Inf strings.

    A matching nonfinite value reproduces an anomaly; it does not make that
    training update valid. Its arithmetic difference is undefined, not zero.
    """
    def compare(actual, expected, atol):
        actual, expected = float(actual), float(expected)
        nonfinite = not (math.isfinite(actual) and math.isfinite(expected))
        if nonfinite:
            difference = None
            matched = ((math.isnan(actual) and math.isnan(expected)) or
                       actual == expected)
        else:
            difference = abs(actual - expected)
            matched = difference <= atol + abs(expected) * 1e-5
        return dict(actual=actual, expected=expected, difference=difference,
                    matched=matched, nonfinite=nonfinite)

    loss = compare(actual_loss, original['total_loss'], 1e-5)
    gradient = compare(actual_norm, original['gradient_before_clip'], 1e-4)
    return dict(loss_difference=loss['difference'],
                gradient_difference=gradient['difference'],
                loss_comparison=loss, gradient_comparison=gradient,
                passed=loss['matched'] and gradient['matched'])


def configure(cfg, purpose, trace_directory=None):
    """Keep all optimizer/augmentation settings; only IO/inspection differs."""
    repair.validate_config(cfg)
    if purpose == 'diagnose':
        if trace_directory is None: raise ValueError('Trace output directory required')
        cfg.checkpoint_config = None
        cfg.custom_imports['imports'] = list(cfg.custom_imports['imports']) + [HOOK]
        cfg.custom_hooks = [dict(type='PortOOFTraceHook', priority='VERY_HIGH',
            directory=str(trace_directory), settings=deepcopy(TRACE)),
            dict(type='PortOOFTraceCommitHook', priority='LOW')]
    elif purpose == 'selected':
        cfg.checkpoint_config = dict(interval=2, max_keep_ckpts=12)
    else: raise ValueError('Unknown auxiliary purpose')
    if cfg.data.val or cfg.data.test or cfg.evaluation is not None:
        raise ValueError('Trainer must not construct VAL/TEST')
    repair.validate_config(cfg)
    return cfg


def anomaly(losses, norm, settings=TRACE):
    values = dict(main_cls=float(losses['main_cls']), main_reg=float(losses['main_reg']),
                  gradient_before_clip=float(norm))
    if any(not math.isfinite(v) for v in values.values()): return ['nonfinite']
    return [key for key, limit in (('main_cls', settings['main_cls_limit']),
        ('main_reg', settings['main_reg_limit']),
        ('gradient_before_clip', settings['gradient_limit'])) if values[key] >= limit]


def validate_val(rows, identities):
    lookup = {r['image']:r for r in identities if r['split']=='val'}
    if (len(rows)!=len(lookup) or len({r['image'] for r in rows})!=len(rows) or
            {r['image'] for r in rows}!=set(lookup)):
        raise ValueError('Checkpoint selection requires every original VAL identity once')
    for row in rows:
        original = lookup[row['image']]
        if any(row[k]!=original[k] for k in ('split','domain','sequence','gt','image_size')):
            raise ValueError('VAL role/GT identity changed')
        if row['pred'] is not None:
            p = row['pred']
            if len(p)!=6 or not all(math.isfinite(v) for v in p) or min(p[2:4])<=0:
                raise ValueError('Invalid delivered VAL box')


def selection_metrics(rows, identities):
    """Offline only. Missing boxes penalize quality; never invent judgement labels."""
    validate_val(rows, identities)
    groups = {}
    for key, part in metrics.grouped(rows).items():
        present = [r for r in part if r['pred'] is not None]
        center = lambda r: math.hypot(r['pred'][0]-r['gt'][0],r['pred'][1]-r['gt'][1]) < 15.
        errors = [np.abs(np.sort(r['pred'][2:4])/np.sort(r['gt'][2:4])-1.) for r in present]
        groups[key] = dict(frames=len(part), outputs=len(present),
            output_coverage=len(present)/len(part),
            center_hit_rate_on_outputs=sum(center(r) for r in present)/len(present) if present else 0.,
            center_correct_full_frame_coverage=sum(center(r) for r in present)/len(part),
            size_correct_full_frame_coverage=sum(not metrics.bad(r) for r in present)/len(part),
            joint_correct_full_frame_coverage=sum(center(r) and not metrics.bad(r) for r in present)/len(part),
            edge_MAE_on_outputs=np.mean(errors,axis=0).tolist() if errors else None,
            penalized_edge_MAE=(sum(float(np.minimum(e,1.).mean()) for e in errors)+len(part)-len(present))/len(part))
    if set(r['domain'] for r in rows)!={'real','sim'}:
        raise ValueError('Both complete VAL domains required')
    return dict(groups=groups,diagnostic_only=True,used_for_checkpoint_choice=False)


def offline_metrics(rows,identities):
    """Exactly the main detector/formal midpoint metric protocol, on VAL only."""
    validate_val(rows,identities)
    from crane_project.tools.eval_crane_offline import CraneOfflineEvaluator
    from crane_project.utils.port_geometry_size_support_v1 import canonical
    records=[dict(domain=r['domain'],seq_id=r['sequence'].split('_',1)[1],frame_id=r['frame_id'],
        gt_box=canonical(r['gt']),pred_box=canonical(r['pred'][:5]) if r['pred'] is not None else None,
        score=r['pred'][5] if r['pred'] is not None else 0.,plc_rope=None,image=r['image']) for r in rows]
    # 'test' here requests full temporal metrics; the input identities are all VAL.
    return CraneOfflineEvaluator(mode='test').evaluate_records(records)


def main_selector():
    from crane_project.tools.ckpt_sweep import SELECTION_CONFIG,select_best_checkpoint
    if SELECTION_CONFIG!=SELECTION['selection_config']:
        raise ValueError('Main detector selection contract changed')
    return select_best_checkpoint,SELECTION_CONFIG


def select(candidates, kind):
    expected = SELECTION[kind+'_epochs']
    if (sorted(c['epoch'] for c in candidates)!=expected or
            len({c['epoch'] for c in candidates})!=len(candidates)):
        raise ValueError('Incomplete fixed checkpoint scan; no survivor-only selection')
    selector,settings=main_selector()
    ordered=sorted(candidates,key=lambda c:c['epoch'])
    results={str(c['epoch']):dict(metrics=c['offline_metrics']) for c in ordered}
    name,_,info=selector(results,settings)
    if name is None: return None
    return dict(next(c for c in ordered if str(c['epoch'])==name),selection_info=info)
