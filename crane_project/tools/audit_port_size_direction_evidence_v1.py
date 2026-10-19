#!/usr/bin/env python3
"""S037 review/check by default; explicit --run-cache is frozen TRAIN head only.

No optimizer, model changes, detector inference, VAL, TEST or depth inputs.
--check uses synthetic tensors + real V20 solely for engineering, never for
claims about real visual direction evidence. Original ROI shards are needed
for --run-cache; metadata cannot substitute for them.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.analyze_port_size_joint_v1 import sha, read, write, csv_write, mean

VERSION = 'port_size_direction_evidence_v1'
HEAD_SHA = '0a44b4254e5e1b1c929b672421dce40b0acd5ed75671f2767a08134c8aa8ca24'


def protocol():
    return dict(protocol=VERSION, stage='design_review_and_engineering_first',
        question='Does frozen V use image-bound input to reverse useful single-edge responses, with coordinated other-edge behavior? Not statistical identifiability of all ROI information.',
        identity='Fixed V20, unchanged2800parameters, no fitting/epoch/weight/loss search; original B-ROI/support/coordinates frozen.',
        proposals='GT-free naturalM and long/short times exp(+/-0.0625), five slots. Reuses old perturbation amplitude; not a new training mechanism. No GT-exact slots.',
        changed='Q changes8reference-vector channels, additive reference and normalization; does not move9x9image samples, B frame or support.',
        visuals=['normal', 'zero_visual_ablation', 'within_video_halfcycle'],
        controls='Same frozen head: zeroF retains geometry/support/xy; donorF is half-cycle in image-sorted same video/scale with target support fixed. Neither is a trained geometry-only baseline; both can be OOD. Donor support mismatch reported, no GT selection.',
        masks='Source naturalM validated. Q finite/positive/samecenter-angle-score. Axis crossings reported unavailable, never relabeled by GT. No GT filtering.',
        delivery='Primary response BEFORE guard; official naturalM delivery/fallback separately. A fallback toM is not evidence of learned Q correction. NaturalQ=M must exactly match official V on same CPU.',
        data='For later run: all5116TRAINviews/2558images,5112outputs; standard/half separately. No VAL/TEST/depth. No perturbation/visual condition counted as independent sample.',
        evaluation='FixedM3%near groups retained. Q near also3% for protection; nonnear direction sign, actual absolute-relative error change, crossGT paired opposite-Q directions and joint benefits. Near never requires zero update.',
        count_epsilon=1e-6, bidirectional='OppositeQ actually straddlesGT; also separately both endpoints nonnear. +/- relativeM need not straddleGT. Direction correctness alone cannot replace useful error reduction.',
        cross_edge='Compare unperturbed-edge prediction to same-visual naturalQ response and GT; report fixedMnear/nonnear. Not force the other edge to stay wrong.',
        visual_comparison='Same views/Q, paired direction and error gain-harm differences vs both controls; report domain/scale/video, counts and unavailable cases. No IID significance claim.',
        interpretation='Positive normal-vs-controls with useful bidirectional responses supports THIS frozen head using image-linked evidence under constructed states only. Negative controls do NOT establish visual-information insufficiency. No natural/VAL generalization conclusion.',
        stop='No new training contract from engineering or synthetic success. If real evidence unclear, keep mechanism unresolved; no extra loss, scale grid, dynamic reader, multiscale or TEST selection.',
        limits=dict(optimizer_updates=0, detector_runs=0, val_reads=0, test_reads=0, depth_reads=0, gpu_allocations=0))


def fixture(torch, swap=False, anisotropic=False):
    b = torch.tensor([[128., 128., 112., 56., .2, .9]])
    if swap:
        b[:, 2:4] = b[:, 2:4].flip(-1); b[:, 4] -= math.pi/2
    xy = torch.tensor([[.998, 1.]]) if anisotropic else torch.ones(1, 2)
    bm = b[:, :5].clone(); bm[:, :4] *= xy[:, [0, 1, 0, 1]]
    m = b.clone(); m[:, :2] += 1.; m[:, 2:4] *= 1.01; m[:, 4] += .01
    return torch.randn(1, 256, 9, 9), torch.ones(1, 1, 9, 9), b, bm, xy, m


def load_head(args, torch):
    from crane_project.utils.port_size_local_vectors_v1 import LocalVectorSizeHead
    path = args.parent/'size_only/head_epoch_20.pth'
    if sha(path) != HEAD_SHA:
        raise ValueError('Fixed V20 checkpoint identity')
    h = LocalVectorSizeHead().cpu().eval()
    h.load_state_dict(torch.load(str(path), map_location='cpu')['head_state'])
    for p in h.parameters():
        p.requires_grad_(False)
    return h


def engineering(args, torch, h):
    from crane_project.utils import port_size_direction_evidence_v1 as d
    before = {k: v.clone() for k, v in h.state_dict().items()}
    cases = []
    torch.manual_seed(1703)
    with torch.no_grad():
        for swap, aniso in ((False, False), (True, True)):
            x = fixture(torch, swap, aniso)
            q, safe = d.proposals(x[-1]); original = h(*x)
            probe = d.readout(h, *x, q[:, 0])
            for k in ('boxes_original', 'candidate_original', 'vector_valid'):
                if not torch.equal(original[k], probe[k]):
                    raise ValueError('Natural same-CPU replay differs '+k)
            response = []
            for slot in range(5):
                o = d.readout(h, *x, q[:, slot])
                z = d.readout(h, torch.zeros_like(x[0]), *x[1:], q[:, slot])
                response.append(dict(slot=d.SLOTS[slot], axis_safe=bool(safe[:, slot].all()),
                    q_input_changes=float((q[:, slot]-x[-1]).abs().max()),
                    reference_vector_changes=float((o['reference_vectors']-probe['reference_vectors']).abs().max()),
                    normal_minus_zero_readout_max=float((o['preguard_candidate']-z['preguard_candidate']).abs().max())))
            cases.append(dict(raw_axis_swapped=swap, anisotropic_resize=aniso,
                              natural_exact=True, synthetic_feature_responses=response))
    if any(not torch.equal(before[k], v) for k, v in h.state_dict().items()):
        raise ValueError('Frozen state mutated')
    return dict(status='ENGINEERING_ONLY_PASS', cases=cases, frozen_state_unchanged=True,
                feature_source='seed1703 synthetic Gaussian tensors, not TRAIN images',
                scope='Nonzero differences show executable conditional/visual paths only; not direction correctness, real visual contribution or effect.')


def score_rows(records, h, torch):
    """Explicit real-cache mode, not executed by --check. CPU, fixed V20."""
    from crane_project.utils import port_size_direction_evidence_v1 as d
    present = [r for r in records if len(r['midpoint_original'])]
    donors = d.donor_indices(present)
    rows, skips = [], []
    with torch.no_grad():
        for i, r in enumerate(present):
            x = [r[k] for k in ('roi', 'support', 'boxes_original', 'boxes_model', 'scale_xy', 'midpoint_original')]
            q, safe = d.proposals(x[-1]); raw = (x[-1][0, 2] < x[-1][0, 3]).long().item()
            gs = r['gt_original'][0, 2:4].sort(descending=True)[0].tolist()  # OFFLINE only
            donor = present[donors[i]] if donors[i] is not None else None
            visuals = dict(normal=x[0], zero_visual_ablation=torch.zeros_like(x[0]),
                           within_video_halfcycle=donor['roi'] if donor else None)
            for visual, roi in visuals.items():
                if roi is None:
                    skips.append(dict(image=r['image'], scale=r['scale'], visual=visual, reason='no_nonself_donor'))
                    continue
                natural = d.readout(h, roi, *x[1:], q[:, 0])
                if visual == 'normal' and not torch.equal(natural['boxes_original'], h(*x)['boxes_original']):
                    raise ValueError('Real natural exact replay')
                for slot in range(5):
                    if not bool(safe[0, slot]):
                        skips.append(dict(image=r['image'], scale=r['scale'], visual=visual, slot=d.SLOTS[slot], reason='axis_crossing'))
                        continue
                    o = d.readout(h, roi, *x[1:], q[:, slot])
                    stem = dict(image=r['image'], sequence=r['sequence'], domain=r['domain'], scale=r['scale'],
                                visual=visual, slot=d.SLOTS[slot], vector_valid=bool(o['vector_valid'][0]),
                                delivery_accepted=bool(o['accepted'][0]), neutral=bool(o['neutral_bypass'][0]),
                                fallback=not bool(o['accepted'][0] or o['neutral_bypass'][0] or o['square_bypass'][0]),
                                donor_image=donor['image'] if donor and visual=='within_video_halfcycle' else None,
                                donor_support_absmean=float((donor['support']-x[1]).abs().mean()) if donor and visual=='within_video_halfcycle' else None)
                    for stage, box in (('preguard', o['preguard_candidate']), ('delivered', o['boxes_original'])):
                        for j, edge in enumerate(('long', 'short')):
                            p, qb, m = float(box[0, 2+(raw if j == 0 else 1-raw)]), float(q[0, slot, 2+(raw if j == 0 else 1-raw)]), float(x[-1][0, 2+(raw if j == 0 else 1-raw)])
                            if not math.isfinite(p) or p <= 0 or (stage=='preguard' and not stem['vector_valid']):
                                skips.append(dict(stem, stage=stage, edge=edge, reason='invalid_preguard'))
                                continue
                            e = d.edge_evaluation(qb, p, gs[j])
                            natural_edge = float(natural['preguard_candidate' if stage=='preguard' else 'boxes_original'][0, 2+(raw if j == 0 else 1-raw)])
                            fixed = d.edge_evaluation(m, m, gs[j])['required_group']
                            rows.append(dict(stem, stage=stage, edge=edge, fixed_M_group=fixed,
                                current_edge=qb, gt_edge=gs[j], predicted_edge=p, **e,
                                other_edge_response=bool(slot and (('long' in d.SLOTS[slot]) != (edge=='long'))),
                                response_vs_same_visual_natural_log=math.log(p/natural_edge) if natural_edge>0 else None,
                                error_vs_same_visual_natural=abs(p/gs[j]-1)-abs(natural_edge/gs[j]-1)))
    return rows, skips


def paired_tables(rows):
    """Same-view visual contrasts and actual cross-GT response pairs."""
    index = {(r['image'], r['scale'], r['visual'], r['slot'], r['stage'], r['edge']): r for r in rows}
    if len(index) != len(rows):
        raise ValueError('Duplicate response identity')
    visual_pairs, opposite_pairs = [], []
    for r in rows:
        if r['visual'] != 'normal':
            continue
        for control in ('zero_visual_ablation', 'within_video_halfcycle'):
            x = index.get((r['image'],r['scale'],control,r['slot'],r['stage'],r['edge']))
            if x is None:
                continue
            ds = r['direction_correct'] is not None and x['direction_correct'] is not None
            visual_pairs.append(dict(image=r['image'],sequence=r['sequence'],domain=r['domain'],scale=r['scale'],
                control=control,slot=r['slot'],stage=r['stage'],edge=r['edge'],Q_group=r['required_group'],
                direction_evaluable=ds,
                only_normal_direction_correct=bool(ds and r['direction_correct'] and not x['direction_correct']),
                only_control_direction_correct=bool(ds and x['direction_correct'] and not r['direction_correct']),
                normal_abs_error_delta=r['abs_error_delta'],control_abs_error_delta=x['abs_error_delta'],
                normal_minus_control_abs_error=abs(r['after_error'])-abs(x['after_error']),
                normal_improved=r['improved'],control_improved=x['improved'],
                normal_fallback=r['fallback'],control_fallback=x['fallback']))
    for r in rows:
        if r['slot'] != r['edge']+'_minus':
            continue
        p = index.get((r['image'],r['scale'],r['visual'],r['edge']+'_plus',r['stage'],r['edge']))
        if p is None:
            continue
        crossed = r['before_error'] < -1e-6 and p['before_error'] > 1e-6
        opposite_pairs.append(dict(image=r['image'],sequence=r['sequence'],domain=r['domain'],scale=r['scale'],
            visual=r['visual'],stage=r['stage'],edge=r['edge'],crossed_GT=crossed,
            both_nonnear=crossed and r['required_group']!='near' and p['required_group']!='near',
            both_directions_correct=bool(crossed and r['update_gt_normalized']>1e-6 and p['update_gt_normalized'] < -1e-6),
            both_errors_improved=bool(crossed and r['improved'] and p['improved']),
            final_size_secant=math.log(p['predicted_edge']/r['predicted_edge'])/(2*.0625),
            correction_secant=math.log(p['predicted_edge']/r['predicted_edge'])/(2*.0625)-1.,
            minus_fallback=r['fallback'],plus_fallback=p['fallback']))
    return visual_pairs, opposite_pairs


def summaries(rows):
    groups = defaultdict(list)
    for r in rows:
        for group in ('overall', r['domain'], r['sequence']):
            for label in ('all', r['required_group']):
                groups[r['scale'],group,r['visual'],r['slot'],r['stage'],r['edge'],label].append(r)
    out = []
    for key, rs in groups.items():
        ds = [r for r in rs if r['direction_correct'] is not None]
        changes = [r['abs_error_delta'] for r in rs]
        out.append(dict(zip(('scale','group','visual','slot','stage','edge','Q_group'),key),
            count=len(rs), direction_evaluable=len(ds), direction_correct=sum(r['direction_correct'] for r in ds),
            improved=sum(r['improved'] for r in rs), harmed=sum(r['harmed'] for r in rs), holds=sum(r['hold'] for r in rs),
            before_MAE_pct=100*mean([abs(r['before_error']) for r in rs]),
            after_MAE_pct=100*mean([abs(r['after_error']) for r in rs]),
            gain_pp_edges=100*sum(-x for x in changes if x<0), harm_pp_edges=100*sum(x for x in changes if x>0),
            fallback_count=sum(r['fallback'] for r in rs)))
    return out


def main(args):
    if args.out.exists():
        raise ValueError('Fresh result directory required')
    import torch
    torch.set_num_threads(1)
    h = load_head(args, torch)
    result = engineering(args, torch, h)
    availability = {n: dict(present=(args.cache/n).is_file()) for n in ('cache_manifest.json','train_s1.pt','train_s05.pt')}
    if args.run_cache and not all(v['present'] for v in availability.values()):
        raise ValueError('Actual TRAIN cache absent; do not replace with metadata/synthetic features')
    rows = skips = None
    if args.run_cache:
        from crane_project.tools.run_port_size_local_vectors_v1 import checked_sources
        from crane_project.tools.audit_port_size_auxiliary_conflict_v1 import load_train
        source_proof = checked_sources()
        records, cache_identity = load_train(args, torch)
        replay_path=args.parent/'size_only/epoch_20_train.rows.jsonl.gz'
        if sha(replay_path)!=read(args.parent/'artifacts.json')['files']['size_only/epoch_20_train.rows.jsonl.gz']:
            raise ValueError('Saved V20 TRAIN replay identity')
        with gzip.open(replay_path,'rt') as f:
            replay={(r['image'],r['scale']):r for r in map(json.loads,f)}
        with torch.no_grad():
            for r in records:
                if not len(r['midpoint_original']):
                    continue
                x=[r[k] for k in ('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original')]
                ref=torch.tensor([replay[r['image'],r['scale']]['edge_residual']],dtype=x[0].dtype)
                if not torch.allclose(h(*x)['boxes_original'],ref,atol=2e-4,rtol=2e-6):
                    raise ValueError('Saved CPU/server V20 natural replay differs')
        rows, skips = score_rows(records, h, torch)
        result.update(status='FROZEN_TRAIN_RESPONSE_COMPLETE', cache=cache_identity,parent_source_proof=source_proof,
                      missing_views=[dict(image=r['image'], scale=r['scale']) for r in records if not len(r['midpoint_original'])])
    args.out.mkdir(parents=True)
    write(args.out/'protocol.json', protocol());write(args.out/'engineering.json', result)
    write(args.out/'cache_availability.json', availability)
    source_names = ('crane_project/tools/audit_port_size_direction_evidence_v1.py',
                    'crane_project/utils/port_size_direction_evidence_v1.py',
                    'crane_project/utils/port_size_local_vectors_v1.py',
                    'crane_project/utils/port_geometry_midpoint_edge_residual_v1.py',
                    'crane_project/utils/port_geometry_midpoint_v1.py')
    write(args.out/'source_identity.json', dict(head_sha256=HEAD_SHA, sources={n:sha(ROOT/n) for n in source_names},
        mode='frozen_real_TRAIN' if args.run_cache else 'synthetic_engineering_only', completed_utc=datetime.now(timezone.utc).isoformat()))
    if rows is not None:
        csv_write(args.out/'responses.csv', rows);csv_write(args.out/'summary.csv', summaries(rows));write(args.out/'unavailable.json', skips)
        visual_pairs,opposite_pairs=paired_tables(rows)
        csv_write(args.out/'visual_pairs.csv',visual_pairs);csv_write(args.out/'opposite_pairs.csv',opposite_pairs)
    write(args.out/'artifacts.json', dict(files={p.name:sha(p) for p in args.out.iterdir() if p.is_file()}))
    print(dict(status=result['status'], actual_TRAIN_cache_ready=all(v['present'] for v in availability.values()),
               real_response_rows=len(rows) if rows is not None else 0))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True);modes.add_argument('--check', action='store_true');modes.add_argument('--run-cache', action='store_true')
    p.add_argument('--parent', type=Path, default=ROOT/'work_dirs/port_size_local_vectors_v1/train_20261018_v1')
    p.add_argument('--cache', type=Path, default=ROOT/'work_dirs/port_geometry_midpoint_sigma15_v1_results_20261005_095254/port_geometry_midpoint_formal_v1_roi_cache')
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
