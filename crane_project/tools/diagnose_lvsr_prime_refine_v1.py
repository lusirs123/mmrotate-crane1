"""One bounded paired refinement: ep5->ep6, original auxiliaries vs size only.

Only TRAIN/VAL; exact original ep6 replay is required before the new arm.
Diagnostic continuation is not a new24-epoch ablation or an adopted model.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/prime_001'
EXPECTED_SHA = '9590597e713d562c3fa953cd96866657d1d17de14d00c822155592e0a2d77301'
SPEC = dict(version='prime_ep5_paired_aux_off_20261113_v1', physical_gpu=3, GPU_budget_seconds=240,
    arms=['keep_aux', 'size_only'], start_epoch=5, endpoint_epoch=6, steps_per_arm=639,
    total_diagnostic_optimizer_updates=1278, lr=.001, original_Adam_state_preserved=True,
    sampling='original epoch6 order, all5112 TRAIN views once per arm, no reweighting',
    change='size_only disables both newly added auxiliary terms after existingep5; proxy state stays frozen',
    control='keep_aux must exactly reproduce original epoch6 head/proxy/optimizer/VAL before new arm',
    formal_training=False, adoption=False, TEST=False,
    stop='identity/resource/nonfinite/budget/control replay failure; one endpoint, no coefficient/epoch search')


def read(path): return json.loads(Path(path).read_text())
def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''): h.update(block)
    return h.hexdigest()
def rows(path):
    with gzip.open(path, 'rt') as stream: return [json.loads(line) for line in stream]


def audit():
    from crane_project.utils.lvsr_prime_refine_metrics_v1 import boundary_pairs
    current = rows(OUT/'representative_TRAIN_VAL.rows.jsonl.gz')
    old = rows(OUT.parent/'attempt_001/L5_all.rows.jsonl.gz')
    result = {}
    for split, scale in (('train_s1', 1.), ('train_s05', .5), ('val', 1.)):
        choose = lambda r: r['split'] == 'val' if split == 'val' else r['split'] != 'val' and r['scale'] == scale
        a, b = [r for r in old if choose(r)], [r for r in current if choose(r)]
        m = [dict(r, edge_residual=r['midpoint']) for r in b]
        result[split] = dict(vs_L5=boundary_pairs(a, b), vs_M=boundary_pairs(m, b))
    report = dict(passed=True, CPU_only=True, model_forwards=0, updates=0, TEST=False,
                  boundary_metrics=result, source_SHA={str(p.relative_to(ROOT)):sha(p) for p in
                  (OUT/'representative_TRAIN_VAL.rows.jsonl.gz', OUT.parent/'attempt_001/L5_all.rows.jsonl.gz')})
    save(OUT/'refine_boundary_audit.json', report)
    print(json.dumps(dict(passed=True, VAL=result['val'])), flush=True)


def run():
    from crane_project.utils.lvsr_prime_refine_metrics_v1 import boundary_pairs
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'; os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    from crane_project.tools import train_lvsr_prime_v1 as training
    from crane_project.tools import train_lvsr_shared_spatial_v1 as rules
    from crane_project.tools import train_symeood_r4_lvsr_pair_v1 as original
    from crane_project.tools import run_symeood_110_lvsr_v1 as base
    from crane_project.tools.run_port_size_native_train_v1 import equal, rng_get, rng_set
    from crane_project.utils import port_size_pair_prime_v1 as model
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    import runpy
    summary = read(OUT/'core_summary.json'); cfg = training.check_config()
    if summary['source_SHA'] != training.source_hashes(): raise ValueError('Original training sources differ')
    assert read(OUT/'refine_boundary_audit.json')['passed']
    assert sha(OUT/'head_epoch_05.pth') == EXPECTED_SHA
    assert summary['selection']['representative_epoch'] == 5
    uuid = subprocess.check_output(['nvidia-smi', '-i', '3', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    assert uuid == cfg['physical_gpu_uuid']
    sources = [Path(__file__).resolve(), ROOT/'crane_project/utils/lvsr_prime_refine_metrics_v1.py', ROOT/'tests/test_lvsr_prime_refine_metrics_v1.py']
    source_sha = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    save(OUT/'refine_protocol.json', dict(spec=SPEC, sources=source_sha, starting_checkpoint_sha=EXPECTED_SHA))
    started = time.monotonic()
    old_handler = signal.getsignal(signal.SIGALRM)
    def timeout(*unused): raise RuntimeError('Paired refinement240-second budget exceeded')
    signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, 240)
    try:
        engine, _, _, vector, runtime = original.runtime(3); torch = engine.torch
        parent = p.checked_sources()
        train, val, inputs = original.load_inputs(p.paths(), parent, engine)
        common = runpy.run_path(str(ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/diagnose.py'))
        memory_before = common['memory_digest'](train+val)
        present = [r for r in train if len(r['midpoint_original'])]
        order = base.schedule_indices(len(present), 6)
        assert len(present) == len(set(order)) == 5112
        schedule_SHA = hashlib.sha256(json.dumps(order, separators=(',', ':')).encode()).hexdigest()
        expected_event = read(OUT/'epoch_06_VAL.json')
        assert schedule_SHA == expected_event['sample_schedule']['sha256']
        pins = dict(summary['upstream_SHA'])
        for name in ('head_epoch_05.pth', 'head_epoch_06.pth', 'epoch_06_VAL.json', 'selection.json'):
            pins[str((OUT/name).relative_to(ROOT))] = sha(OUT/name)
        assert sha(OUT/'head_epoch_06.pth') == expected_event['checkpoint']['sha256']
        start = torch.load(str(OUT/'head_epoch_05.pth'), map_location='cpu')
        expected = torch.load(str(OUT/'head_epoch_06.pth'), map_location='cpu')
        old_VAL = rows(OUT/'epoch_05_VAL.rows.jsonl.gz')
        baseline_L5 = [r for r in rows(OUT.parent/'attempt_001/L5_all.rows.jsonl.gz') if r['split'] == 'val']
        results = {}; resources = []; update_count = 0
        for arm in SPEC['arms']:
            head = model.PairPrimeSizeHead().cuda()
            bank = model.ProxyBank(start['proxy_state']['targets']).cuda()
            head.load_state_dict(start['head_state'], strict=True); bank.load_state_dict(start['proxy_state'], strict=True)
            optimizer = training.joint_optimizer(torch, head, bank, cfg)
            optimizer.load_state_dict(deepcopy(start['optimizer_state']))
            assert equal(head.state_dict(), start['head_state'], torch) and equal(bank.state_dict(), start['proxy_state'], torch)
            if arm == 'keep_aux':
                assert original.summarize(base.evaluate(head, val, torch)) == summary['representative']['metrics']
            rng_set(start['rng'], torch)
            total_loss = 0.; tick = time.monotonic()
            for offset in range(0, 5112, 8):
                batch = base.batch([present[i] for i in order[offset:offset+8]], torch)
                stat = model.update(head, bank, optimizer, batch) if arm == 'keep_aux' else model.update(head, bank, optimizer, batch, 0., 0.)
                update_count += 1; total_loss += stat['loss']; step = offset//8+1
                assert all(int(optimizer.state[q]['step']) == 3195+step for q in head.parameters())
                if arm == 'size_only':
                    assert all(q.grad is None for q in bank.parameters())
                    assert all(int(optimizer.state[q]['step']) == 3195 for q in bank.parameters())
                if step % 200 == 1:
                    resources.append(engine.resource_check(0))
                    print(json.dumps(dict(stage='refine', arm=arm, step=step, total_steps=639, elapsed=time.monotonic()-started)), flush=True)
            vr = base.evaluate(head, val, torch); metrics = original.summarize(vr)
            if arm == 'keep_aux':
                assert equal(head.state_dict(), expected['head_state'], torch), 'Original epoch6 head replay failed'
                assert equal(bank.state_dict(), expected['proxy_state'], torch), 'Original epoch6 proxy replay failed'
                assert equal(optimizer.state_dict(), expected['optimizer_state'], torch), 'Original epoch6 Adam replay failed'
                assert metrics == expected_event['metrics'], 'Original epoch6 VAL replay failed'
            else:
                assert equal(bank.state_dict(), start['proxy_state'], torch), 'Disabled proxy state changed'
            assert all(cell['frozen_identity'] for cell in metrics.values())
            file = OUT/('refine_'+arm+'_ep06_server_only.pth')
            payload = dict(head_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
                proxy_state={k:v.detach().cpu().clone() for k,v in bank.state_dict().items()},
                optimizer_state=deepcopy(optimizer.state_dict()), rng=rng_get(torch),
                source_start_sha=EXPECTED_SHA, completed_epoch=6, total_head_updates=3834,
                diagnostic_arm=arm, formal_candidate=False)
            engine.atomic_save(file, payload)
            assert equal(payload, torch.load(str(file), map_location='cpu'), torch)
            original.write_rows(OUT/('refine_'+arm+'_VAL.rows.jsonl.gz'), vr)
            detail = None
            if arm == 'size_only':
                full_rows = base.evaluate(head, train+val, torch)
                original.write_rows(OUT/'refine_size_only_TRAIN_VAL.rows.jsonl.gz', full_rows)
                detail = rules.report_rows(full_rows, common)
            results[arm] = dict(metrics=metrics, grouped_TRAIN_VAL=detail,
                boundary_vs_start=boundary_pairs(old_VAL, vr), boundary_vs_L5=boundary_pairs(baseline_L5, vr),
                checkpoint=dict(path=file.name, sha256=sha(file)), steps=639,
                head_total_updates=3834, proxy_total_updates=3834 if arm == 'keep_aux' else 3195,
                original_ep6_exact_replayed=arm == 'keep_aux', proxy_frozen=arm == 'size_only',
                L5_gate=rules.assess(metrics, summary['baseline_L5']['metrics']), M_increment=rules.increment_gate(metrics),
                size_or_total_loss_mean=total_loss/639, seconds=time.monotonic()-tick)
            print(json.dumps(dict(stage='arm_complete', arm=arm, VAL=metrics['overall']['edge_residual'])), flush=True)
            del head, bank, optimizer, payload
        assert update_count == 1278
        assert all(sha(ROOT/name) == value for name,value in pins.items())
        assert common['memory_digest'](train+val) == memory_before
        assert p.checked_sources() == parent and summary['source_SHA'] == training.source_hashes()
        assert all(sha(ROOT/name) == value for name,value in source_sha.items())
        save(OUT/'refine_summary.json', dict(spec=SPEC, results=results, original_start_VAL=summary['representative']['metrics'],
            original_L5_VAL=summary['baseline_L5']['metrics'], source_SHA=source_sha))
        save(OUT/'refine_completion.json', dict(passed=True, physical_gpu=3, diagnostic_optimizer_updates=1278,
            original_ep6_exact_replayed=True, control_steps=639, size_only_steps=639, source_start_sha=EXPECTED_SHA,
            parents_inputs_selection_unchanged=True, pinned_SHA=pins, resources=resources, runtime=runtime,
            elapsed_seconds=time.monotonic()-started, formal_training=False, TEST=False, automatic_adoption=False))
        names = ['refine_protocol.json','refine_summary.json','refine_completion.json','refine_boundary_audit.json']
        save(OUT/'refine_manifest.json', {name:sha(OUT/name) for name in names})
        print(json.dumps(dict(passed=True, diagnostic_updates=1278, elapsed=time.monotonic()-started)), flush=True)
    except BaseException as error:
        save(OUT/'refine_failure.json', dict(error=repr(error), TEST=False, automatic_retry=False)); raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('audit', 'refine'), default='audit')
    args = parser.parse_args()
    audit() if args.stage == 'audit' else run()
