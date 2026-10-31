"""Synthetic OBB loss/gradient response audit. No detector, data or optimizer."""
import argparse
from collections import defaultdict
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils.eood_B_localization_probe_v1 import (
    band, cases, decode, derivative_intervals, losses as reference_losses,
    matches_intervals, riou, sym_distance)

SOURCES = (
    'crane_project/utils/eood_B_localization_probe_v1.py',
    'crane_project/tools/audit_eood_B_localization_response_v1.py',
    'tests/test_eood_B_localization_response_v1.py',
    'mmrotate/models/losses/rotated_iou_loss.py',
    'mmrotate/models/losses/sym_kld_calculator.py',
    'mmrotate/models/losses/sym_kld_loss.py',
    'mmrotate/models/losses/sym_iou_hybrid_loss_v1.py',
    'mmrotate/core/bbox/transforms.py',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def telemetry(gpu):
    text = subprocess.check_output([
        'nvidia-smi', '-i', str(gpu),
        '--query-gpu=index,memory.free,memory.used,temperature.gpu',
        '--format=csv,noheader,nounits'], text=True)
    row = [int(v.strip()) for v in text.strip().split(',')]
    if len(row) != 4 or row[0] != gpu:
        raise ValueError('Invalid selected-GPU response')
    apps = subprocess.check_output([
        'nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    own = 0
    for line in apps.splitlines():
        parts = [v.strip() for v in line.split(',')]
        if len(parts) == 2 and all(v.isdigit() for v in parts) and int(parts[0]) == os.getpid():
            own = int(parts[1])
    return dict(gpu=gpu, free_mib=row[1], total_used_mib=row[2], temperature_C=row[3], own_mib=own)


def guard(gpu, before=False):
    row = telemetry(gpu)
    if row['free_mib'] < (3072 if before else 512) or row['own_mib'] > 2048:
        raise RuntimeError('Synthetic loss audit memory guard: '+str(row))
    return row


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['band'], row['family'], row['aspect'])].append(row)
    result = []
    for (bucket, family, aspect), group in sorted(groups.items()):
        by_loss = {}
        for name in ('DIoU', 'SymKLD', 'hybrid025'):
            values = [r['response'][name] for r in group]
            by_loss[name] = dict(
                gradient_norm_median=statistics.median(v['q_gradient_norm'] for v in values),
                directional_derivative_median=statistics.median(v['radial_derivative'] for v in values),
                independent_fd_mismatch_count=sum(not v['fd_matches'] for v in values),
                independent_partial_interval_mismatch_count=sum(not v['partial_intervals_match'] for v in values),
                independent_value_mismatch_count=sum(not v['value_matches'] for v in values),
                box_step_riou_improved=sum(v['virtual_box_step_riou_delta'] > 1e-7 for v in values),
                box_step_riou_decreased=sum(v['virtual_box_step_riou_delta'] < -1e-7 for v in values))
        ratios = [r['response']['SymKLD']['q_gradient_norm']/r['response']['DIoU']['q_gradient_norm']
                  for r in group if r['response']['DIoU']['q_gradient_norm'] > 1e-10]
        result.append(dict(band=bucket, family=family, aspect=aspect, n=len(group),
            KLD_over_DIoU_box_gradient_median=statistics.median(ratios) if ratios else None,
            losses=by_loss))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if ROOT not in out.parents:
        raise ValueError('Output must be a new directory under this checkout')
    out.mkdir(parents=True, exist_ok=False)
    sources = {name: sha(ROOT/name) for name in SOURCES}
    protocol = dict(
        synthetic_only=True, checkpoint_loaded=False, detector_loaded=False,
        dataset_loaded=False, training_runs=0, optimizer_updates=0, VAL=False, TEST=False,
        physical_gpu=args.gpu, allocator_cap_mib=1024, own_driver_stop_mib=2048,
        startup_free_mib=3072, minimum_free_mib=512, temperature_stop_C=None,
        default_driver_thermal_policy=True, batch=126,
        q_coordinates='object-frame dx/w,dy/h,log(wp/w),log(hp/h),angle error radians',
        losses='actual production DIoU and SymKLD, weight 2; existing hybrid alpha=.25, no coefficient search',
        independent_reference='stdlib float64 polygon clipping and analytic Gaussian divergence',
        finite_difference='central difference along signed ray plus all-five-coordinate one-sided slope intervals at polygon cusps',
        numerical_tolerances=dict(value_abs=2e-3, value_rel=2e-4, fd_abs=5e-3, fd_rel=.03),
        virtual_box_step='q-step norm=min(5e-4,error*.1), equal across losses; skip norm<=1e-5 to avoid amplifying FP32 noise; counterfactual box geometry only',
        scope='Box-space derivatives are coordinate-dependent and are not FPN gradients, model updates or AP evidence.',
        literature=[
            'https://ming71.github.io/assets/pdf/2024/ming2024gradient.pdf',
            'https://arxiv.org/pdf/2206.15109'], sources=sources)
    save(out/'protocol.json', protocol)
    started = time.monotonic()
    resource_rows = [guard(args.gpu, before=True)]
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['OMP_NUM_THREADS'] = '1'
    try:
        import torch
        import mmcv
        from mmcv.ops import diff_iou_rotated_2d
        from mmrotate.models.losses.rotated_iou_loss import RotatedIoULoss
        from mmrotate.models.losses.sym_kld_loss import SymKLDLoss
        from mmrotate.models.losses.sym_iou_hybrid_loss_v1 import SymIoUHybridLossV1
        native_source = Path(importlib.import_module('mmcv.ops.diff_iou_rotated').__file__)
        native_sha = sha(native_source)
        save(out/'environment.json', dict(torch=torch.__version__, CUDA=torch.version.cuda,
            mmcv=mmcv.__version__, native_diff_iou_source=str(native_source), native_diff_iou_sha256=native_sha))
        (out/'native_diff_iou_rotated.py').write_bytes(native_source.read_bytes())
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(1024*2**20/torch.cuda.get_device_properties(0).total_memory, 0)
        def forbidden_optimizer(*a, **kw):
            raise AssertionError('Optimizer construction forbidden in synthetic audit')
        torch.optim.Optimizer.__init__ = forbidden_optimizer
        modules = dict(DIoU=RotatedIoULoss(mode='linear', diou=True, loss_weight=2.),
                       SymKLD=SymKLDLoss(loss_weight=2., strict_finite=True),
                       hybrid025=SymIoUHybridLossV1(geometry_weight=.25))
        assert sum(p.numel() for loss in modules.values() for p in loss.parameters()) == 0

        def decode_tensor(target, q):
            dx, dy = q[:, 0]*target[:, 2], q[:, 1]*target[:, 3]
            c, s = target[:, 4].cos(), target[:, 4].sin()
            return torch.stack((target[:, 0]+c*dx-s*dy, target[:, 1]+s*dx+c*dy,
                target[:, 2]*q[:, 2].exp(), target[:, 3]*q[:, 3].exp(), target[:, 4]+q[:, 4]), dim=1)

        rows, inventory = [], cases()
        max_value_error, max_iou_error = 0., 0.
        for start in range(0, len(inventory), 126):
            if time.monotonic()-started > 180:
                raise RuntimeError('Synthetic audit time bound; no retry')
            resource_rows.append(guard(args.gpu))
            batch = inventory[start:start+126]
            target = torch.tensor([r['target'] for r in batch], device='cuda')
            q = torch.tensor([r['q'] for r in batch], device='cuda', requires_grad=True)
            pred = decode_tensor(target, q)
            native_iou = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0)).squeeze(0).detach().cpu().tolist()
            values, gradients = {}, {}
            for name, loss in modules.items():
                value = loss(pred, target, reduction_override='none')
                gradient = torch.autograd.grad(value.sum(), q, retain_graph=True)[0]
                assert value.shape == (len(batch),) and torch.isfinite(value).all() and torch.isfinite(gradient).all()
                values[name], gradients[name] = value.detach(), gradient.detach()
            assert torch.allclose(values['hybrid025'], values['DIoU']+.25*values['SymKLD'], atol=1e-5, rtol=1e-5)
            mixed_expected = gradients['DIoU']+.25*gradients['SymKLD']
            mixed_error = (gradients['hybrid025']-mixed_expected).abs()
            mixed_matches = (mixed_error <= 1e-4+1e-4*mixed_expected.abs()).all(dim=1).cpu().tolist()
            mixed_max_error = mixed_error.max(dim=1)[0].cpu().tolist()
            values = {name: value.cpu().tolist() for name, value in values.items()}
            gradients = {name: value.cpu().tolist() for name, value in gradients.items()}
            # Use the actually decoded float32 coordinates for the independent CUDA-IoU value check.
            native_pred = pred.detach().cpu().tolist()
            native_target = target.cpu().tolist()
            for i, case in enumerate(batch):
                ref = reference_losses(case['target'], case['q'])
                overlap = riou(decode(case['target'], case['q']), case['target'])
                iou_error = abs(native_iou[i]-riou(native_pred[i], native_target[i]))
                max_iou_error = max(max_iou_error, iou_error)
                row = dict(case, riou=overlap, band=band(overlap),
                           cuda_riou=native_iou[i], raw_sym_kld=sym_distance(decode(case['target'], case['q']), case['target']),
                           independent_iou_matches=iou_error < 2e-4,
                           parallel_edges=abs(case['q'][4]) < 1e-12,
                           hybrid_gradient_identity_matches=mixed_matches[i],
                           hybrid_gradient_identity_max_abs_error=mixed_max_error[i], response={})
                h = min(5e-4, case['error']*.1)
                intervals = derivative_intervals(case['target'], case['q'], min(5e-5, case['error']*.05))
                direction = [case['sign']*v for v in case['direction']]
                plus = reference_losses(case['target'], [x+h*v for x, v in zip(case['q'], direction)])
                minus = reference_losses(case['target'], [x-h*v for x, v in zip(case['q'], direction)])
                for name in modules:
                    value, gradient = values[name][i], gradients[name][i]
                    error = abs(value-ref[name])
                    max_value_error = max(max_value_error, error)
                    norm = math.sqrt(sum(v*v for v in gradient))
                    radial = sum(v*d for v, d in zip(gradient, direction))
                    finite_difference = (plus[name]-minus[name])/(2*h)
                    step_applied = norm > 1e-5
                    moved = [x-h*g/norm for x, g in zip(case['q'], gradient)] if step_applied else list(case['q'])
                    new_overlap = riou(decode(case['target'], moved), case['target'])
                    row['response'][name] = dict(
                        value=value, independent_value=ref[name], value_matches=error <= 2e-3+2e-4*abs(ref[name]),
                        q_gradient=gradient, q_gradient_norm=norm,
                        radial_derivative=radial, independent_fd=finite_difference,
                        fd_matches=abs(radial-finite_difference) <= 5e-3+.03*abs(finite_difference),
                        partial_intervals=intervals[name],
                        partial_intervals_match=matches_intervals(gradient, intervals[name]),
                        virtual_box_step_applied=step_applied,
                        virtual_box_step_riou_delta=new_overlap-overlap,
                        virtual_box_step_loss_delta=reference_losses(case['target'], moved)[name]-ref[name])
                rows.append(row)
            with (out/'rows.jsonl').open('a') as f:
                for row in rows[-len(batch):]:
                    f.write(json.dumps(row, allow_nan=False)+'\n')
            print('Synthetic cases checked:', len(rows), '/', len(inventory), flush=True)
            del q, pred, target, value, gradient
        assert len(rows) == 3780
        resource_rows.append(guard(args.gpu))
        for name, digest in sources.items():
            assert sha(ROOT/name) == digest, name
        assert sha(native_source) == native_sha
        mismatches = {name: sum(not r['response'][name]['fd_matches'] for r in rows) for name in modules}
        partial_mismatches = {name: sum(not r['response'][name]['partial_intervals_match'] for r in rows) for name in modules}
        identity_mismatches = sum(not r['hybrid_gradient_identity_matches'] for r in rows)
        value_mismatches = {name: sum(not r['response'][name]['value_matches'] for r in rows) for name in modules}
        iou_mismatches = sum(not r['independent_iou_matches'] for r in rows)
        numerical_checks_passed = not (any(mismatches.values()) or any(partial_mismatches.values()) or identity_mismatches or any(value_mismatches.values()) or iou_mismatches)
        # Keep every mismatch, including parallel-edge cases; never relabel these as passed.
        summary = dict(passed=numerical_checks_passed, execution_complete=True, optimizer_updates=0, training_runs=0,
            detector_loaded=False, checkpoint_loaded=False, dataset_loaded=False, VAL=False, TEST=False,
            synthetic_cases=len(rows), source_unchanged=True, max_independent_iou_abs_error=max_iou_error,
            max_independent_loss_abs_error=max_value_error, independent_fd_mismatch_counts=mismatches,
            finite_difference_agreement=not any(mismatches.values()),
            independent_partial_interval_mismatch_counts=partial_mismatches,
            hybrid_gradient_identity_mismatch_count=identity_mismatches,
            independent_value_mismatch_counts=value_mismatches, independent_iou_mismatch_count=iou_mismatches,
            zero_kld_gradient_cases=sum(r['response']['SymKLD']['q_gradient_norm'] <= 1e-10 for r in rows),
            groups=summarize(rows), resources=dict(
                own_driver_peak_mib=max(r['own_mib'] for r in resource_rows),
                minimum_free_mib=min(r['free_mib'] for r in resource_rows),
                maximum_observed_temperature_C=max(r['temperature_C'] for r in resource_rows),
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20),
            limitations=[
                'Synthetic boxes, no TRAIN distribution or detector performance evidence.',
                'Fixed alpha .25 is a historical reference, not a selected coefficient.',
                'Directional and coordinate slope intervals are numerical checks, not proof of convergence.',
                'Geometry steps do not update a network; equal q-step norms depend on the declared coordinates.',
                'Execution may complete with passed=false; all numerical discrepancies remain in raw rows.'])
        save(out/'summary.json', summary)
        save(out/'resources.json', resource_rows)
        save(out/'completion.json', dict(status='COMPLETE' if numerical_checks_passed else 'COMPLETE_WITH_FINDINGS', optimizer_updates=0, training_runs=0,
            source_unchanged=True, finite_difference_agreement=summary['finite_difference_agreement']))
    except BaseException as error:
        save(out/'failure.json', dict(status='STOPPED', error=repr(error), retry=False,
            optimizer_updates=0, training_runs=0, VAL=False, TEST=False))
        raise


if __name__ == '__main__':
    main()
