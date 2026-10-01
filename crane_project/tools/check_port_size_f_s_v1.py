#!/usr/bin/env python3
"""CPU saved-box formula checks; no detector, optimizer, coefficient fitting."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.diagnose_port_center_size_d_v1 import sha
from crane_project.tools.diagnose_port_shape_e_h_train_gradients_v1 import (
    SOURCES as OLD_SOURCES, signed_response)

EXPECTED = dict(type='LogSizeLoss', beta=.1, loss_weight=.1, eps=1e-6, reduction='mean')
EXPERIMENT = ROOT/'crane_project/configs/crane_symeood_k1_port_day2night_size_f_s_v1.py'
FIXTURE = ROOT/'crane_project/tools/port_size_f_s_v1_saved_boxes.json'
MANIFEST = ROOT/'crane_project/tools/port_size_f_s_v1_sources.json'
SOURCES = tuple(dict.fromkeys(OLD_SOURCES + (
    'mmrotate/models/losses/log_size_loss.py',
    'mmrotate/models/losses/center_size_compensation.py',
    'crane_project/configs/crane_symeood_k1_port_day2night_size_f_s_v1.py',
    'crane_project/tools/check_port_size_f_s_v1.py',
    'crane_project/tools/preflight_port_size_f_s_v1.py',
    'crane_project/tools/port_size_f_s_v1_saved_boxes.json')))


def checked_sources():
    sources = {s: sha(ROOT/s) for s in SOURCES}
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get('sources') != sources:
        raise ValueError('Reviewed F-S source manifest differs; synchronize the bundle')
    return sources


def candidate_loss():
    from mmrotate.models.losses.log_size_loss import LogSizeLoss
    return LogSizeLoss(**{k: v for k, v in EXPECTED.items() if k != 'type'})


def reduction_multiplier(normalizer, dtype=None):
    """Measure the inherited reducer, including version-specific denominator eps."""
    import torch
    from mmdet.models.losses.utils import weight_reduce_loss
    return float(weight_reduce_loss(torch.ones(1, dtype=dtype or torch.float64),
                                    reduction='mean', avg_factor=normalizer))


def analyze_box(record, role):
    """Leaf replay is mathematical evidence, never a captured detector graph."""
    import torch
    from mmrotate.models.losses.sym_kld_loss import SymKLDLoss
    loss = candidate_loss()
    p = torch.tensor([record['pred']], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([record['gt']], dtype=torch.float64)
    w = torch.tensor([record['weight']], dtype=torch.float64)
    n = float(record['normalizer'])
    extra = loss(p, t, weight=w, avg_factor=n)
    main = SymKLDLoss(loss_weight=2.)(p, t, weight=w, avg_factor=n)
    gs = {k: torch.autograd.grad(value, p, retain_graph=True)[0][0]
          for k, value in (('main', main), ('size', extra), ('main_plus_size', main+extra))}
    response = {k: signed_response(p[0].detach(), t[0], g) for k, g in gs.items()}
    if not all(torch.isfinite(g).all() for g in gs.values()):
        raise RuntimeError('Nonfinite saved-box gradient')
    size = response['size']
    factor = reduction_multiplier(n, p.dtype)
    analytic = [loss.loss_weight*float(w[0])*factor/2*max(-1., min(1., size[k]/loss.beta))
                for k in ('log_long_error', 'log_short_error')]
    observed = [size[k] for k in ('log_long_gradient', 'log_short_gradient')]
    if not all(abs(a-b) < 1e-10 for a, b in zip(analytic, observed)):
        raise RuntimeError('Size derivative differs from analytical formula')
    if torch.count_nonzero(gs['size'][[0, 1, 4]]):
        raise RuntimeError('Independent size has xy/angle dependence')
    if any(size[e]*size[g] < 0 for e, g in
           (('log_long_error', 'log_long_gradient'), ('log_short_error', 'log_short_gradient'))):
        raise RuntimeError('Independent size points away from its target')
    fd_errors = []
    # Sorted ties can be nonsmooth when target edges differ; do not assert a
    # two-sided derivative at that branch boundary.
    tied = abs(record['pred'][2]-record['pred'][3]) <= 1e-12
    if not tied:
        for name in ('size', 'main_plus_size'):
            for j in (2, 3):
                plus, minus = p.detach().clone(), p.detach().clone()
                h = 1e-5
                plus[0, j] *= math.exp(h); minus[0, j] *= math.exp(-h)
                def value(box):
                    v = loss(box, t, weight=w, avg_factor=n)
                    return v if name == 'size' else v+SymKLDLoss(loss_weight=2.)(box, t, weight=w, avg_factor=n)
                fd = float((value(plus)-value(minus))/(2*h))
                error = abs(fd-float(p[0, j].detach()*gs[name][j]))
                fd_errors.append(error)
        if max(fd_errors) > 2e-6:
            raise RuntimeError('Saved-box log-edge finite difference failed')
    return dict(**record, role=role, graph_scope='float64_saved_box_leaf_replay',
                loss_main=float(main.detach()), loss_size=float(extra.detach()), direct=response,
                analytic_size_log_edge_gradient=analytic,
                inherited_reduction_multiplier=factor,
                maximum_finite_difference_error=max(fd_errors) if fd_errors else None,
                sorted_edge_tie=tied,
                size_eps_guard_edges=int((p[:, 2:4] <= loss.eps).sum()))


def mechanism_checks():
    import torch
    loss = candidate_loss()
    gt = torch.tensor([[10., 20., 80., 40., .2]], dtype=torch.float64)
    rows = []
    for long_ratio, short_ratio in ((.95, .95), (1.05, 1.05), (.9, .9), (1.1, 1.1),
                                   (.95, 1.05), (1.05, .95), (1., .9), (.9, 1.)):
        for angle_deg in (0., 2., 5., 10.):
            p = gt[0].tolist()
            p[2] *= long_ratio; p[3] *= short_ratio
            p[4] += math.radians(angle_deg)
            rows.append(analyze_box(dict(pred=p, gt=gt[0].tolist(), weight=1., normalizer=2.),
                                    'synthetic_ratio_angle_perturbation'))
    p = gt.clone(); p[:, 2:4] *= .95
    base = loss(p, gt)
    changes = []
    for kind in ('center_angle', 'swap', 'period', 'isotropic'):
        q, target = p.clone(), gt.clone()
        if kind == 'center_angle':
            q[:, :2] += 100.; q[:, 4] += .3
        elif kind == 'swap':
            q[:, 2:4] = q[:, [3, 2]]; q[:, 4] += math.pi/2
        elif kind == 'period':
            q[:, 4] += math.pi
        else:
            q[:, :4] *= .5; target[:, :4] *= .5
        changes.append(abs(float(loss(q, target)-base)))
    for near in (0., 1e-4):
        target = [10., 20., 80.+near, 80., .2]
        q = [10., 20., (80.+near)*.95, 80.*.95, .5]
        rows.append(analyze_box(dict(pred=q, gt=target, weight=1., normalizer=2.),
                                'square_or_near_square'))
    if max(changes) > 1e-12 or float(base) <= 0:
        raise RuntimeError('Scale/representation/independence or common-size check failed')
    return dict(rows=rows, maximum_invariance_error=max(changes),
                common_scale_error_detected=True, square_tie_policy='torch.sort branch; no two-sided FD at exact ties')


def review_fixture():
    data = json.loads(FIXTURE.read_text())
    if (data.get('protocol') != 'port_size_f_s_v1_saved_boxes'
            or len(data['train']) != 16 or len(data['val_descriptive']) != 8):
        raise ValueError('Saved-box fixture contract differs')
    train = [analyze_box(r, 'previous_train_snapshot_replay') for r in data['train']]
    val = [analyze_box(r, 'previous_val_descriptive_only_not_coefficient_fitting')
           for r in data['val_descriptive']]
    checks = mechanism_checks()
    return dict(source_report_sha256=data['sources'], fixture_sha256=sha(FIXTURE),
                selection=data['selection'], train_replay=train, val_descriptive=val,
                mechanisms=checks, all_mathematical_checks_passed=True,
                train_unique_images=sorted({r['image'] for r in train}),
                optimizer_steps=0, detector_forwards=0,
                inherited_normalizer_2_multiplier_float64=reduction_multiplier(2.),
                note='Composed main+size need not correct every edge; no new training or accuracy evidence.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    out = Path(args.out_json).resolve()
    if out.exists():
        raise FileExistsError('Keep old results; use a new output name')
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='CHECK_STARTED', evidence_role='saved_box_mathematical_check',
                  candidate_settings=EXPECTED, formal_training_authorized=False,
                  test_repeatedly_exposed=True, optimizer_steps=0)
    try:
        report['sources'] = checked_sources()
        report['source_manifest_sha256'] = sha(MANIFEST)
        report['mathematics'] = review_fixture()
        if checked_sources() != report['sources']:
            raise RuntimeError('Sources changed during check')
        report['status'] = 'SAVED_BOX_FORMULA_CHECK_COMPLETE_REVIEW_REQUIRED'
    except Exception as error:
        report.update(status='CHECK_FAILED', error=type(error).__name__+': '+str(error))
        raise
    finally:
        with out.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print('Saved', out, 'status', report['status'], flush=True)


if __name__ == '__main__':
    main()
