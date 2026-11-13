"""One fresh24-epoch decoder-conditioned edge-proxy LVSR control on physical GPU3. No TEST."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT/'crane_project/configs/crane_lvsr_r4_decoded_edge_proxy_v1.json'
SOURCES = ('crane_project/configs/crane_lvsr_r4_decoded_edge_proxy_v1.json',
    'crane_project/utils/port_size_pair_decoded_edge_proxy_v1.py', 'crane_project/utils/lvsr_prime_review_v1.py',
    'crane_project/utils/port_size_pair_prime_v1.py',
    'crane_project/utils/lvsr_decoded_edge_proxy_review_v1.py',
    'crane_project/tools/train_lvsr_decoded_edge_proxy_v1.py', 'tests/test_lvsr_decoded_edge_proxy_train_v1.py',
    'crane_project/tools/train_lvsr_shared_spatial_v1.py',
    'work_dirs/lvsr_diagnostic_20261112_v1/diagnose.py',
    'crane_project/configs/crane_lvsr_r4_prime_v1.json')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def source_hashes():
    from crane_project.utils.symeood_r4_lvsr_protocol_v1 import sha
    return {name: sha(ROOT/name) for name in SOURCES}


def check_config():
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    from crane_project.utils.port_size_pair_decoded_edge_proxy_v1 import SETTINGS
    cfg = read(CONFIG)
    for key in ('epochs', 'batch_size', 'seed', 'lr', 'weight_decay', 'betas', 'eps', 'clip_norm', 'parameters', 'alignment'):
        if cfg[key] != p.SETTINGS[key]: raise ValueError('Original setting differs: '+key)
    if cfg['physical_gpu'] != 3 or cfg['TEST'] or cfg['continuation'] or cfg['automatic_adoption']:
        raise ValueError('Fixed single-GPU3 TRAIN/VAL boundary differs')
    if cfg['output_root'] != 'work_dirs/lvsr_diagnostic_20261112_v1/edge_proxy_001' or cfg['training_wall_seconds_limit'] != 3600:
        raise ValueError('Independent output/budget differs')
    if cfg['selection'] != read(ROOT/'crane_project/configs/crane_lvsr_r4_prime_v1.json')['selection']:
        raise ValueError('Original representative ranking differs')
    if cfg['prime'] != SETTINGS or cfg['head'] != 'PairDecodedEdgeProxyHead':
        raise ValueError('PRIME settings differ from reviewed configuration')
    if cfg['training_parameters'] != 15440 or cfg['proxy_parameters'] != 12960:
        raise ValueError('Training/inference parameter accounting differs')
    return cfg


def joint_optimizer(torch, head, bank, cfg):
    return torch.optim.Adam([{'params': head.parameters()}, {'params': bank.parameters()}],
        lr=cfg['lr'], betas=tuple(cfg['betas']), eps=cfg['eps'], weight_decay=cfg['weight_decay'])


def verify_joint(engine, head_class, bank_class, centers, train, cfg, root):
    from crane_project.utils.port_size_pair_decoded_edge_proxy_v1 import update
    from crane_project.tools.run_symeood_110_lvsr_v1 import batch
    from crane_project.tools.run_port_size_native_train_v1 import seed, rng_get, rng_set, equal
    torch = engine.torch
    rr = [r for domain in ('real', 'sim') for scale in (.5, 1.)
          for r in [q for q in train if q['domain'] == domain and q['scale'] == scale and len(q['midpoint_original'])][:4]]
    assert len(rr) == 16
    seed(torch); head = head_class().cuda(); bank = bank_class(centers).cuda()
    initial_head = engine.g.state_digest(head); initial_bank = engine.g.state_digest(bank)
    optimizer = joint_optimizer(torch, head, bank, cfg)
    logs = [update(head, bank, optimizer, batch(rr[start:start+8], torch)) for start in (0, 8)]
    if any(q.grad is None or float(q.grad.norm()) <= 0 for q in bank.parameters()):
        raise ValueError('Missing proxy gradient')
    state = dict(head=deepcopy(head.state_dict()), proxy=deepcopy(bank.state_dict()),
        optimizer=deepcopy(optimizer.state_dict()), rng=rng_get(torch))
    path = root/'verification/edge_proxy_discarded_trial.pth'
    engine.atomic_save(path, state); loaded = torch.load(str(path), map_location='cpu')
    if not equal(state, loaded, torch): raise ValueError('Joint trial serialization differs')
    other = head_class().cuda(); other_bank = bank_class(centers).cuda()
    other.load_state_dict(loaded['head'], strict=True); other_bank.load_state_dict(loaded['proxy'], strict=True)
    other_optimizer = joint_optimizer(torch, other, other_bank, cfg)
    other_optimizer.load_state_dict(loaded['optimizer'])
    rng_set(state['rng'], torch); a = update(head, bank, optimizer, batch(rr[:8], torch))
    rng_set(loaded['rng'], torch); b = update(other, other_bank, other_optimizer, batch(rr[:8], torch))
    if not (equal(a, b, torch) and equal(head.state_dict(), other.state_dict(), torch)
            and equal(bank.state_dict(), other_bank.state_dict(), torch)
            and equal(optimizer.state_dict(), other_optimizer.state_dict(), torch)):
        raise ValueError('Joint exact next-update continuation differs')
    return dict(passed=True, discarded_updates=4, initial_head=initial_head, initial_proxy=initial_bank,
                exact_state_optimizer_rng_next_step=True, logs=logs)


def run(args):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    cfg = check_config(); hashes = source_hashes()
    if args.stage == 'check':
        return dict(passed=True, settings=cfg, sources=hashes, training_started=False)
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    from crane_project.utils import lvsr_prime_review_v1 as reviewed
    from crane_project.utils.port_size_pair_decoded_edge_proxy_v1 import PairDecodedEdgeProxyHead, ProxyBank, update
    from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
    from crane_project.tools import train_symeood_r4_lvsr_pair_v1 as original
    from crane_project.tools import train_lvsr_shared_spatial_v1 as rules
    from crane_project.tools import run_symeood_110_lvsr_v1 as base
    from crane_project.tools.run_port_size_native_train_v1 import seed, rng_get, equal, near_joint
    uuid = subprocess.check_output(['nvidia-smi', '-i', '3', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if uuid != cfg['physical_gpu_uuid']: raise ValueError('Physical GPU3 identity differs')
    parent = p.checked_sources(); root = ROOT/cfg['output_root']; root.mkdir(exist_ok=False)
    source = p.BASE/'lvsr/attempt_001'; started = time.monotonic()
    identity = dict(protocol=cfg['protocol'], sources=hashes, settings=cfg, parent_identity=parent)
    save(root/'protocol.json', identity)

    def status(stage, **extra):
        value = dict(stage=stage, pid=os.getpid(), time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(), **extra)
        temporary = root/'status.tmp'; temporary.write_text(json.dumps(value)); temporary.replace(root/'status.json')
        print(json.dumps(value), flush=True)

    updates = 0; events = []; frozen_sha = {}; memory_before = None; records = None
    try:
        status('engineering')
        engine, _, _, vector, runtime = original.runtime(3); torch = engine.torch
        resources = []
        def resource():
            value = engine.resource_check(0)
            resources.append(value)
            return value
        train, val, inputs = original.load_inputs(p.paths(), parent, engine)
        if inputs['counts']['train_present'] != 5112 or len(val) != 887: raise ValueError('Wrong inputs')
        import runpy
        common = runpy.run_path(str(ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/diagnose.py'))
        records = train+val; memory_before = common['memory_digest'](records)
        review = ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/shared_spatial_001'
        review_complete = read(review/'prime_review_v2_completion.json')
        if not review_complete['passed'] or review_complete['training_updates'] != 0: raise ValueError('Review receipt differs')
        old_selection = read(source/'train/selection.json'); l5_event = read(source/'train/epoch_05_VAL.json')
        if old_selection['selected_epoch'] is not None: raise ValueError('Preserve original L5 null selection')
        pins = [ROOT/p.parent.R4_CHECKPOINT, p.midpoint_directory()/'train/head_epoch_20.pth',
            source/'train/head_epoch_05.pth', source/'train/head_epoch_24.pth',
            source/'inputs/train.pt', source/'inputs/val.pt', source/'train/selection.json',
            review/'prime_review_v2_summary.json', review/'prime_review_v2_completion.json',
            ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/prime_001/core_summary.json',
            ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/prime_001/selection.json',
            ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/prime_001/head_epoch_05.pth',
            ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/prime_001/proxy_targets.json']
        pins.extend(ROOT/name for name in cfg['review_receipts'])
        mapping = read(ROOT/cfg['review_receipts'][0])
        if not mapping['passed'] or mapping['updates'] != 0 or mapping['TEST']:
            raise ValueError('Successful frozen mapping review required')
        if any(p.sha(ROOT/name) != value for name,value in mapping['sources_SHA'].items()):
            raise ValueError('Reviewed mapping source changed')
        frozen_sha = {str(q.relative_to(ROOT)): p.sha(q) for q in pins}
        if p.sha(pins[0]) != p.parent.R4_SHA or p.sha(pins[1]) != p.MIDPOINT_SHA: raise ValueError('Parents differ')
        if p.sha(pins[2]) != 'b272c50dfe6329414803d5f22c6e1e84d9c3db37b6bb2e31f1c7ed140f662c3e': raise ValueError('L5 differs')
        present = [r for r in train if len(r['midpoint_original'])]
        targets = torch.cat([reviewed.canonical_pair(vector.residual.size_targets(r['gt_original'],
            r['midpoint_original'], r['boxes_model'])['log_residual'], r['boxes_model'], r['midpoint_original']) for r in present])
        target_record = read(ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/prime_001/proxy_targets.json')
        if not target_record['TRAIN_only'] or target_record['source_input_SHA256'] != p.sha(source/'inputs/train.pt'):
            raise ValueError('Original TRAIN-only proxy center provenance differs')
        centers = torch.tensor(target_record['centers'], dtype=targets.dtype)
        center_receipt = target_record['receipt']
        center_sha = hashlib.sha256(centers.numpy().tobytes()).hexdigest()
        if center_sha != target_record['target_SHA256']: raise ValueError('Original proxy center serialization differs')
        if center_sha != cfg['proxy_targets_SHA256']: raise ValueError('Audited TRAIN-only proxy targets changed')
        save(root/'proxy_targets.json', dict(TRAIN_only=True, target_SHA256=center_sha,
            centers=centers.tolist(), receipt=center_receipt, source_input_SHA256=p.sha(source/'inputs/train.pt')))
        seed(torch); old = PairSpecificSizeHead().cuda(); old_state = deepcopy(old.state_dict()); rng = torch.get_rng_state()
        seed(torch); candidate = PairDecodedEdgeProxyHead().cuda(); initial_bank = ProxyBank(centers).cuda()
        if not equal(old_state, candidate.state_dict(), torch) or not torch.equal(rng, torch.get_rng_state()):
            raise ValueError('Head initialization/RNG differs')
        initial_proxy_digest = engine.g.state_digest(initial_bank)
        candidate.load_state_dict(torch.load(str(pins[2]), map_location='cpu')['head_state'], strict=True)
        old.load_state_dict(candidate.state_dict(), strict=True)
        trial = [r for domain in ('real', 'sim') for scale in (.5, 1.)
                 for r in [q for q in present if q['domain'] == domain and q['scale'] == scale][:4]]
        if len(trial) != 16: raise ValueError('Need16 domain/scale inference-control views')
        if base.evaluate(candidate, trial, torch) != base.evaluate(old, trial, torch):
            raise ValueError('Nonneutral original inference differs')
        del old, candidate, initial_bank, old_state
        paths = dict(attempt=root, verify=root/'verification', inputs=source/'inputs')
        # Original geometric/gradient/Adam proof, without auxiliary state:4 discarded updates.
        original.verify(paths, parent, engine, base, PairDecodedEdgeProxyHead, train, val, inputs)
        original_proof = read(paths['verify']/'completion.json')
        # The new joint objective and optimizer need a separate4-update proof.
        joint = verify_joint(engine, PairDecodedEdgeProxyHead, ProxyBank, centers, train, cfg, root)
        if source_hashes() != hashes: raise ValueError('Sources changed during engineering')
        save(root/'engineering.json', dict(passed=True, initial_parameters_equal=True, initial_rng_equal=True,
            inference_control_frames=16, inference_control_max_abs=0., neutral_views=6003,
            discarded_original_updates=4, discarded_joint_updates=4, total_discarded_updates=8, joint=joint))
        seed(torch); head = PairDecodedEdgeProxyHead().cuda(); bank = ProxyBank(centers).cuda()
        if engine.g.state_digest(head) != original_proof['initial_state'] or engine.g.state_digest(bank) != initial_proxy_digest:
            raise ValueError('Fresh training reused trial state')
        optimizer = joint_optimizer(torch, head, bank, cfg)
        if optimizer.state: raise ValueError('Fresh Adam required')
        steps = math.ceil(len(present)/8)
        if steps != 639: raise ValueError('Wrong update budget')
        sampling = dict(view_identity_order_sha256=hashlib.sha256(json.dumps([(r['image'], r['scale']) for r in present], separators=(',', ':')).encode()).hexdigest(),
            views=5112, video_scale_counts=dict(Counter(r['sequence']+'|'+str(r['scale']) for r in present)), each_view_once_per_epoch=True)
        baseline = read(paths['verify']/'baseline_VAL.json'); events = []; updates = 0
        for epoch in range(1, 25):
            tick = time.monotonic(); order = base.schedule_indices(5112, epoch)
            legacy_order = list(range(5112)); random.Random(1703+epoch).shuffle(legacy_order)
            if order != legacy_order or sorted(order) != list(range(5112)): raise ValueError('Sampling changed')
            totals = dict(loss=0., size_loss=0., alignment_loss=0., proxy_loss=0., zero_embeddings=0, anchors=0)
            max_gradient = 0.; max_proxy_gradient = 0.; minimum_sample_norm = math.inf; minimum_proxy_norm = math.inf; mixed_views = 0
            status('train', epoch=epoch, step=0, updates=updates)
            with (root/'progress.jsonl').open('a') as stream:
                for start in range(0, 5112, 8):
                    rr = [present[i] for i in order[start:start+8]]
                    if time.monotonic()-started > cfg['training_wall_seconds_limit']: raise RuntimeError('Fixed3600-second trial budget exceeded')
                    stat = update(head, bank, optimizer, base.batch(rr, torch)); updates += 1
                    if any(int(optimizer.state[q]['step']) != updates for q in list(head.parameters())+list(bank.parameters())):
                        raise ValueError('Joint Adam step counter differs')
                    if any(g['lr'] != .001 or g['weight_decay'] != 0. for g in optimizer.param_groups): raise ValueError('Optimizer drift')
                    for key in totals: totals[key] += stat[key]
                    minimum_sample_norm = min(minimum_sample_norm, stat['minimum_sample_norm'])
                    minimum_proxy_norm = min(minimum_proxy_norm, stat['minimum_proxy_norm'])
                    mixed_views += stat['mixed_role_views']
                    max_gradient = max(max_gradient, stat['grad_norm']); max_proxy_gradient = max(max_proxy_gradient, stat['proxy_grad_norm'])
                    stream.write(json.dumps(dict(epoch=epoch, step=start//8+1, updates=updates, **stat), allow_nan=False)+'\n')
                    if start//8 % 100 == 0:
                        stream.flush(); resource(); status('train', epoch=epoch, step=start//8+1, total_steps=639, updates=updates)
            if updates != epoch*639: raise ValueError('Update budget differs')
            if not torch.equal(bank.targets.cpu(), centers): raise ValueError('TRAIN proxy targets changed')
            vr = base.evaluate(head, val, torch); metrics = original.summarize(vr)
            for domain, cell in metrics.items():
                if not cell['frozen_identity']: raise ValueError('Non-size M identity changed')
                for arm in ('b', 'midpoint'):
                    if cell[arm] != baseline[domain][arm]: raise ValueError('Upstream metrics changed')
            checkpoint = root/('head_epoch_%02d.pth'%epoch)
            payload = dict(protocol=cfg['protocol'], head_type=cfg['head'], identity=identity, epoch=epoch, updates=updates,
                head_state={k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                proxy_state={k: v.detach().cpu().clone() for k, v in bank.state_dict().items()},
                optimizer_state=deepcopy(optimizer.state_dict()), rng=rng_get(torch), parameters=2480,
                training_parameters=15440, proxy_parameters=12960, detector_sha256=p.parent.R4_SHA, midpoint_sha256=p.MIDPOINT_SHA,
                parent_checkpoint=None, input_completion_sha256=p.sha(source/'inputs/completion.json'), proxy_target_SHA256=center_sha)
            engine.atomic_save(checkpoint, payload)
            if not equal(payload, torch.load(str(checkpoint), map_location='cpu'), torch): raise ValueError('Joint checkpoint reload differs')
            event = dict(epoch=epoch, updates=updates, metrics=metrics,
                eligibility_vs_L5=rules.assess(metrics, l5_event['metrics']), increment_vs_M=rules.increment_gate(metrics),
                original_chain_gate=p.assess(metrics), selection_key=p.selection_key(metrics, epoch),
                checkpoint=dict(path=checkpoint.name, sha256=p.sha(checkpoint), epoch=epoch),
                near_edges=near_joint(vr), cross_edge=vector.cross_edge(vr),
                sample_schedule=dict(views=5112, unique_views=5112, each_once=True,
                    sha256=hashlib.sha256(json.dumps(order, separators=(',', ':')).encode()).hexdigest(),
                    frame_order_sha256=hashlib.sha256(json.dumps([(present[i]['image'], present[i]['scale']) for i in order], separators=(',', ':')).encode()).hexdigest()),
                auxiliary=dict(size_loss_mean=totals['size_loss']/steps, alignment_loss_mean=totals['alignment_loss']/steps,
                    proxy_loss_mean=totals['proxy_loss']/steps, anchors=totals['anchors'], zero_embeddings=totals['zero_embeddings'],
                    proxy_parameters=12960, proxy_grad_norm_max=max_proxy_gradient,
                    minimum_sample_norm=minimum_sample_norm,minimum_proxy_norm=minimum_proxy_norm,mixed_role_views=mixed_views), loss_mean=totals['loss']/steps,
                max_grad_norm=max_gradient, seconds=time.monotonic()-tick, resource=resource())
            save(root/('epoch_%02d_VAL.json'%epoch), event); original.write_rows(root/('epoch_%02d_VAL.rows.jsonl.gz'%epoch), vr)
            events.append(event)
            if source_hashes() != hashes: raise ValueError('Trial source drift')
            m = metrics['overall']['edge_residual']
            status('epoch_complete', epoch=epoch, updates=updates, loss=event['loss_mean'], long_MAE=m['long_MAE'],
                short_MAE=m['short_MAE'], joint10=m['joint10'], eligible_vs_L5=event['eligibility_vs_L5']['pass_all'], seconds=event['seconds'])
        selection = rules.choose(events); save(root/'selection.json', dict(**selection, protocol=cfg['protocol'], rules=cfg['selection']))
        chosen = next(e for e in events if e['epoch'] == selection['representative_epoch'])
        status('final_analysis', representative_epoch=chosen['epoch'])
        replay = PairDecodedEdgeProxyHead().cuda()
        replay.load_state_dict(torch.load(str(root/chosen['checkpoint']['path']), map_location='cpu')['head_state'], strict=True)
        if original.summarize(base.evaluate(replay, val, torch)) != chosen['metrics']: raise ValueError('Representative VAL reload differs')
        rows = base.evaluate(replay, records, torch)
        detail = rules.report_rows(rows, common)
        original.write_rows(root/'representative_TRAIN_VAL.rows.jsonl.gz', rows)
        save(root/'core_detail.json', dict(epoch=chosen['epoch'], metrics=detail))
        if common['memory_digest'](records) != memory_before or any(p.sha(ROOT/n) != h for n, h in frozen_sha.items()):
            raise ValueError('Frozen parents, caches or legacy selection changed')
        if p.checked_sources() != parent or source_hashes() != hashes: raise ValueError('Final source drift')
        save(root/'core_summary.json', dict(protocol=cfg['protocol'], settings=cfg, selection=selection,
            representative=chosen, baseline_L5=l5_event, epoch24=events[-1], epochs=events, source_SHA=hashes,
            upstream_SHA=frozen_sha, sampling=sampling, original_L5_selected_epoch=None, weight_download=False, TEST=False))
        columns = ['epoch', 'loss', 'long_MAE_pct', 'short_MAE_pct', 'joint10', 'RIoU', 'AP75', 'real_joint', 'sim_joint', 'eligible_L5', 'increment_M', 'original_chain']
        with (root/'core_epochs.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
            for event in events:
                m = event['metrics']['overall']['edge_residual']
                writer.writerow(dict(epoch=event['epoch'], loss=event['loss_mean'], long_MAE_pct=100*m['long_MAE'],
                    short_MAE_pct=100*m['short_MAE'], joint10=m['joint10'], RIoU=m['mean_RIoU'], AP75=m['AP75'],
                    real_joint=event['metrics']['real']['edge_residual']['joint10'], sim_joint=event['metrics']['sim']['edge_residual']['joint10'],
                    eligible_L5=event['eligibility_vs_L5']['pass_all'], increment_M=event['increment_vs_M']['pass_all'], original_chain=event['original_chain_gate']['pass_all']))
        save(root/'core_completion.json', dict(passed=True, epochs_completed=24, formal_optimizer_updates=updates,
            discarded_engineering_updates=8, selected_epoch=selection['selected_epoch'], representative_epoch=selection['representative_epoch'],
            checkpoint=chosen['checkpoint'], head_type=cfg['head'], parents_unchanged=True, inputs_unchanged=True,
            proxy_targets_unchanged=True, source_SHA=hashes, detector_updates=0, midpoint_updates=0,
            sample_order_and_exposure_unchanged=True, TEST=False, automatic_adoption=False, original_L5_selection_unchanged=True,
            inference_parameters=2480, proxy_training_parameters=12960, training_parameters=15440, physical_gpu=3,
            runtime=runtime, resources=resources, elapsed_seconds=time.monotonic()-started))
        files = ['core_summary.json', 'core_detail.json', 'core_epochs.csv', 'core_completion.json']
        save(root/'core_transfer_manifest.json', {n: p.sha(root/n) for n in files})
        status('COMPLETE', selected_epoch=selection['selected_epoch'], representative_epoch=selection['representative_epoch'], updates=updates)
        return dict(completed=True, selected_epoch=selection['selected_epoch'], TEST=False)
    except BaseException as error:
        failure = dict(error=repr(error), TEST=False, automatic_retry=False, formal_optimizer_updates=updates,
            completed_epochs=len(events), last_completed_epoch=events[-1]['epoch'] if events else None,
            last_completed_metrics=events[-1]['metrics'] if events else None,
            completed_epoch_records=events, source_SHA=hashes, output_root=cfg['output_root'],
            parents_unchanged=all(p.sha(ROOT/name)==value for name,value in frozen_sha.items()) if frozen_sha else None,
            inputs_unchanged=common['memory_digest'](records)==memory_before if records is not None else None,
            auxiliary_failure=getattr(error,'details',None), automatic_adoption=False)
        save(root/'failure.json', failure)
        save(root/'failure_transfer_manifest.json', {'failure.json':p.sha(root/'failure.json'), 'protocol.json':p.sha(root/'protocol.json')})
        status('FAILED', error=repr(error), updates=updates); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'run'), default='check')
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False))
