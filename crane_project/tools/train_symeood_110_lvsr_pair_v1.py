#!/usr/bin/env python3
"""One fresh pair-specific LVSR24 run; sealed preparation remains unchanged."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_symeood_110_lvsr_pair_v1 as prep
from crane_project.tools.run_port_size_native_train_v1 import sha, read, write, rows_write, seed, rng_get, rng_set, equal, near_joint

COMMON, OUT = prep.COMMON, prep.OUT
CONTRACT = ROOT/'crane_project/data_contracts/symeood_110_lvsr_pair_train_v1.json'


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def measures(rr):
    from crane_project.tools.diagnose_symeood_110_lvsr_edges import groups, edge_diagnostic
    from crane_project.utils.port_size_a20_protection_review_v1 import relative_errors
    result = {}
    for name, group in groups(rr).items():
        cell = edge_diagnostic(group, 'edge_residual')[0]
        present = [r for r in group if r['midpoint'] is not None]
        for j, edge in enumerate(('long', 'short')):
            for subset, threshold in (('other', .03), ('large10', .10)):
                values = [relative_errors(r['edge_residual'], r['gt'])['absolute'][j]
                    for r in present if relative_errors(r['midpoint'], r['gt'])['absolute'][j] > threshold]
                cell[edge+'_'+subset+'_count'] = len(values)
                cell[edge+'_'+subset+'_mae'] = sum(values)/len(values) if values else None
        result[name] = cell
    return result


def eligibility(summary, near, damage, control, control_damage):
    """New frozen size protocol; old DFR-inclusive gate retained separately."""
    original = prep.base.eligibility(summary, near)
    checks = {'M/'+k: v for k, v in original['checks'].items() if not k.endswith('/dfr')}
    checks['original/overall_joint_strict'] = summary['overall']['edge_residual']['sizes']['joint10'] > control['overall']['edge_residual']['sizes']['joint10']
    for domain in ('real', 'sim'):
        candidate = summary[domain]['edge_residual']['sizes']
        baseline = control[domain]['edge_residual']['sizes']
        checks[domain+'/joint_vs_original'] = candidate['joint10'] >= baseline['joint10']
        for edge in ('long', 'short'):
            value, old = candidate['edges'][edge]['mae'], baseline['edges'][edge]['mae']
            checks[domain+'/'+edge+'_vs_original'] = value < old-1e-6 if (domain, edge)==('sim', 'long') else value <= old+.001
            harm = damage[domain][edge+'_near_positive_harm_mean_pct']
            ref = control_damage[domain][edge+'_near_positive_harm_mean_pct']
            checks[domain+'/'+edge+'_near_harm_vs_original'] = harm is not None and ref is not None and harm/100 <= ref/100+.001
            value, old = damage[domain][edge+'_large10_mae'], control_damage[domain][edge+'_large10_mae']
            checks[domain+'/'+edge+'_large10_vs_original'] = value is not None and old is not None and value <= old+.001
    d, c = damage['sim'], control_damage['sim']
    a, b = d['long_near_positive_harm_mean_pct'], c['long_near_positive_harm_mean_pct']
    checks['sim/near_long_harm_strict_original'] = a is not None and b is not None and a/100 < b/100-1e-6
    checks['sim/near_long_material_original'] = d['long_near_material_harmed_0p1pp'] <= c['long_near_material_harmed_0p1pp']
    return dict(pass_all=all(checks.values()), checks=checks, failed=[k for k,v in checks.items() if not v],
        old_protocol_reference_only=original, DFR_role='reported_auxiliary_not_new_gate')


def selection_key(summary, epoch):
    return [-summary['overall']['edge_residual']['sizes']['joint10']]+[
        summary[d]['edge_residual']['sizes']['edges'][e]['mae'] for d in ('sim', 'real') for e in ('long', 'short')]+[epoch]


def choose(events):
    qualified = [e for e in events if e['eligibility']['pass_all']]
    best = min(qualified or events, key=lambda e:e['selection_key'])
    return dict(selected=best['epoch'] if qualified else None, representative=best['epoch'],
        eligible_epochs=[e['epoch'] for e in qualified], fixed_primary=24)


def val_observer(torch):
    """Read-only pair-role counts; no extra model forward or random draws."""
    from crane_project.utils.port_size_local_vectors_v1 import residual
    count = dict(frames=0, active_locations=0, swapped_locations=0, frames_with_swaps=0)
    def observe(module, args, output):
        with torch.no_grad():
            n = len(args[2])
            if not n:
                return
            v = output['vectors'].detach()
            q = residual.original.roi_delta_to_original(v.reshape(n,324,2), *args[2:5]).reshape(n,81,4,2)
            lengths = torch.stack(((q[:,:,0]-q[:,:,2]).norm(dim=-1), (q[:,:,1]-q[:,:,3]).norm(dim=-1)), -1)
            active = output['common_support'].reshape(n,81)>0
            swapped = (lengths[:,:,0]<lengths[:,:,1]) & active
            count['frames'] += n
            count['active_locations'] += int(active.sum())
            count['swapped_locations'] += int(swapped.sum())
            count['frames_with_swaps'] += int(swapped.any(-1).sum())
    return count, observe


def check():
    proof = read(CONTRACT)
    prep.checked_sources()
    if (proof['epochs'] != 24 or proof['updates_per_epoch'] != 639 or proof['physical_gpu'] != 0 or
            not proof['formal_training_authorized'] or proof['TEST'] or proof['depth']):
        raise ValueError('Frozen authorization differs')
    if sha(prep.CONTRACT) != proof['smoke_contract_sha256']:
        raise ValueError('Sealed smoke contract changed')
    for name, digest in {**proof['sources'], **proof['control_files']}.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Formal source/control differs: '+name)
    receipt = read(OUT/'smoke.json')
    if (sha(OUT/'smoke.json') != proof['smoke_receipt_sha256'] or not receipt['passed'] or
            receipt['actual_optimizer_updates'] != 4 or not receipt['state_discarded']):
        raise ValueError('Original accepted smoke required')
    return proof


def status(state, **kw):
    value = dict(state=state, time=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        physical_gpu=0, TEST=False, depth=False, **kw)
    temporary = OUT/'formal_status.tmp'
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, OUT/'formal_status.json')


def backup(epochs, torch):
    folders = [OUT/'backups', Path('/home/omnisky/symeood_model_backups/symeood_110_retrain_seed0_20261026_v1/lvsr/pair_specific')]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=False)
    if len({p.stat().st_dev for p in folders}) != 2:
        raise ValueError('Backup filesystems not independent')
    result = []
    for epoch in sorted(set(epochs)):
        source = OUT/'train'/('head_epoch_%02d.pth'%epoch)
        payload = torch.load(str(source), map_location='cpu')
        h = prep.new_head(torch, 'cpu')
        keys = h.load_state_dict(payload['head_state'], strict=True)
        copies = []
        for folder in folders:
            target = folder/source.name
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            if sha(target) != sha(source):
                raise ValueError('Backup SHA differs')
            copies.append(dict(path=str(target), sha256=sha(target), device=target.stat().st_dev))
        result.append(dict(epoch=epoch, source=str(source), sha256=sha(source),
            missing_keys=keys.missing_keys, unexpected_keys=keys.unexpected_keys, copies=copies))
    return result


def fit(proof, torch):
    from crane_project.tools import run_symeood_110_lvsr_c_ep7 as sealed
    from crane_project.tools import train_symeood_110_midpoint_v1 as f
    from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
    from crane_project.utils import port_size_qalign_v1 as vector
    from crane_project.tools.run_port_geometry_size_condition_v1 import components
    f.runtime_imports()
    runtime = f.runtime(0)
    if runtime != read(OUT/'smoke.json')['runtime']:
        raise ValueError('Accepted runtime differs')
    torch.cuda.set_per_process_memory_fraction(2048*2**20/torch.cuda.get_device_properties(0).total_memory, 0)
    # Existing loader verifies cache and M21. Its ep7 payload is provenance
    # only and is discarded; NO learned size parameters/optimizer are copied.
    train, val, _, unused_parent = sealed.load_inputs(torch)
    del unused_parent
    present = [r for r in train if len(r['midpoint_original'])]
    if len(train)!=5116 or len(present)!=5112 or len(val)!=887 or sum(bool(len(r['midpoint_original'])) for r in val)!=885:
        raise ValueError('Frozen view/output counts differ')
    controls = {}
    for epoch in range(1,25):
        old = read(COMMON/'fixed_b'/('epoch_%02d_val.json'%epoch))
        order = prep.base.schedule_indices(len(present), epoch)
        digest = hashlib.sha256(json.dumps(order, separators=(',',':')).encode()).hexdigest()
        if digest != old['schedule_sha256'] or old['updates'] != 639*epoch:
            raise ValueError('Same-budget original schedule differs')
        controls[epoch] = old
    seed(torch)
    h = prep.new_head(torch, 'cuda:0')
    initial = f.g.state_digest(h)
    if initial != read(OUT/'smoke.json')['initial_digest']:
        raise ValueError('Fresh neutral initialization differs from accepted head')
    opt = torch.optim.Adam(h.parameters(), lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=0.)
    if opt.state:
        raise ValueError('Adam not fresh')
    dest = OUT/'train'
    dest.mkdir()
    write(OUT/'formal_identity.json', dict(contract_sha256=sha(CONTRACT), runtime=runtime,
        initial_digest=initial, optimizer='fresh Adam', parent_checkpoint=None, reused_smoke_updates=False,
        train_views=5116, train_present=5112, VAL_frames=887, parameters=2480,
        input_identity=read(COMMON/'identity.json')['inputs'], feature_extractions=0, midpoint_forwards=0))
    updates, events, riou_cache = 0, [], {}
    for epoch in range(1,25):
        tick = time.monotonic()
        ids = prep.base.schedule_indices(len(present), epoch)
        totals, maxgrad = {}, 0.
        status('training', epoch=epoch, updates=updates)
        with (dest/'progress.jsonl').open('a') as log:
            for start in range(0,len(ids),8):
                rr = [present[i] for i in ids[start:start+8]]
                stats = prep.base.update(h, opt, prep.base.batch(rr, torch), 'fixed_b', torch)
                updates += 1
                if any(int(opt.state[p]['step'])!=updates for p in h.parameters()):
                    raise ValueError('Adam update counter differs')
                for k,v in stats.items():
                    totals[k] = totals.get(k,0.)+v
                maxgrad = max(maxgrad, stats['grad_norm'])
                log.write(json.dumps(dict(epoch=epoch, step=start//8+1, updates=updates, batch_size=len(rr), **stats), allow_nan=False)+'\n')
                if updates == 1:
                    print(json.dumps(dict(state='first_update_complete', updates=updates, loss=stats['loss'], parameters=2480)), flush=True)
        if updates!=epoch*639:
            raise ValueError('Epoch budget differs')
        role_counts, observer = val_observer(torch)
        handle = h.register_forward_hook(observer)
        try:
            vr = prep.base.evaluate(h, val, torch)
        finally:
            handle.remove()
        summary, near = metrics.summary(vr, riou_cache), near_joint(vr)
        damage = measures(vr)
        oldrows = rows(COMMON/'fixed_b'/('epoch_%02d_val.rows.jsonl.gz'%epoch))
        for a,b in zip(vr, oldrows):
            if any(a[k]!=b[k] for k in ('image','scale','gt','b','midpoint','domain','split','sequence','frame_id')):
                raise ValueError('Same-epoch VAL inputs differ')
        gate = eligibility(summary, near, damage, controls[epoch]['metrics'], measures(oldrows))
        tr = prep.base.evaluate(h, train, torch)
        td = dict(size_groups=metrics.train_report(tr), joint=near_joint(tr),
            cross_edge_by_scale={str(sc):vector.cross_edge([r for r in tr if r['scale']==sc]) for sc in (.5,1.)})
        saved_rng = rng_get(torch)
        payload = dict(protocol=proof['protocol'], contract_sha256=sha(CONTRACT), epoch=epoch, updates=updates,
            updates_per_epoch=639, head_state={k:v.detach().cpu().clone() for k,v in h.state_dict().items()},
            optimizer_state=deepcopy(opt.state_dict()), rng=saved_rng, parent_checkpoint=None,
            input_identity=read(COMMON/'identity.json')['inputs'])
        path = dest/('head_epoch_%02d.pth'%epoch)
        with path.open('xb') as stream:
            torch.save(payload, stream)
        if not equal(payload, torch.load(str(path),map_location='cpu'), torch):
            raise ValueError('Checkpoint serialization differs')
        rng_set(saved_rng, torch)
        ep = dict(epoch=epoch, updates=updates, updates_per_epoch=639, eligibility=gate,
            selection_key=selection_key(summary,epoch), metrics=summary, components=components(vr),
            diagnostics=near, damages=damage, cross_edge=vector.cross_edge(vr),
            pair_role_counts=role_counts,
            schedule_sha256=controls[epoch]['schedule_sha256'], checkpoint_sha256=sha(path),
            training_losses={k:v/639 for k,v in totals.items()}, max_grad_norm=maxgrad,
            resource=f.resource_check(0), epoch_seconds=time.monotonic()-tick)
        write(dest/('epoch_%02d_val.json'%epoch), ep)
        rows_write(dest/('epoch_%02d_val.rows.jsonl.gz'%epoch), vr)
        write(dest/('epoch_%02d_train.json'%epoch), td)
        rows_write(dest/('epoch_%02d_train.rows.jsonl.gz'%epoch), tr)
        events.append(ep)
        status('epoch_complete', epoch=epoch, updates=updates, qualified=gate['pass_all'])
        print(json.dumps(dict(epoch=epoch, updates=updates, qualified=gate['pass_all'],
            joint=summary['overall']['edge_residual']['sizes']['joint10'], seconds=ep['epoch_seconds'])), flush=True)
    result = dict(**choose(events), epochs=events, formal_updates=updates, initial_digest=initial)
    write(dest/'selection.json', result)
    copies = backup([result['representative'],24], torch)
    write(OUT/'formal_completion.json', dict(state='complete', formal_updates=updates,
        selected=result['selected'], representative=result['representative'], fixed_primary=24,
        backups=copies, contract_sha256=sha(CONTRACT), TEST=False, depth=False, extra_epochs=0))
    status('complete', epoch=24, updates=updates)


def main(args):
    proof = check()
    if not args.run:
        print(json.dumps(dict(passed=True, updates=0, formal_training_started=False)))
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('WORLD_SIZE','1')!='1':
        raise ValueError('One authorized physical GPU0 required')
    if (OUT/'formal_start.json').exists() or (OUT/'train').exists():
        raise ValueError('Existing formal attempt preserved; no automatic retry')
    write(OUT/'formal_start.json', dict(time=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),contract_sha256=sha(CONTRACT),physical_gpu=0))
    try:
        import torch
        torch.set_num_threads(1)
        fit(proof, torch)
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    main(parser.parse_args())
