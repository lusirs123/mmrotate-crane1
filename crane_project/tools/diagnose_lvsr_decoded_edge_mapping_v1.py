"""Bounded CPU audit of original-slot decoder-conditioned auxiliary mapping."""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'work_dirs/lvsr_diagnostic_20261112_v1/prime_001'
SOURCES = ['crane_project/utils/lvsr_decoded_edge_proxy_review_v1.py',
           'crane_project/tools/check_lvsr_decoded_edge_proxy_review_v1.py',
           'crane_project/tools/diagnose_lvsr_decoded_edge_mapping_v1.py']
SPEC = dict(version='decoded_edge_mapping_CPU_20261113_v1', CPU_threads=1, seconds=600,
    snapshots=['initial', 'L5', 'PRIME5'], present_views_per_head=5999,
    missing_neutral_initial_views=4, total_forward_views=18001,
    routing='detached original decoder sort indices; never GT; retain all original648 channel positions',
    masks='training auxiliary only; same sqrt(W) role mask on sample and proxies; inference W unchanged',
    bank='same20x648; initial random seed1703 for initial, frozen trained PRIME bank for L5/PRIME',
    targets='original TRAIN20 two-dimensional centers projected onto each edge; no new clustering',
    losses='mean over edge-specific alignment and mask-conditional proxy arrangements; .001 each, no optimization',
    parameters='2480 inference;12960 proxy training parameters; no new learned module',
    near_ties='relative pair-length gap <=0.001 and <=0.01, diagnostic only',
    parameter_batches='first16 natural epoch6 batches per snapshot, no optimizer',
    training_conditions='mapping exact, gradients finite, no zero selected descriptors at initialization; later-head zeros/near ties remain risks',
    stop='600s/nonfinite/identity/mapping failures stop; zero descriptors block training recommendation; no automatic repair or search',
    GPU=False, updates=0, TEST=False, download='grouped core only')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def save(name, value):
    with (OUT / name).open('x') as f: json.dump(value, f, indent=2, allow_nan=False)


def describe(values):
    v = sorted(values); n = len(v)
    return dict(n=n, mean=sum(v)/n, min=v[0], max=v[-1], median=(v[(n-1)//2]+v[n//2])/2)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from torch.nn import functional as F
    from types import SimpleNamespace
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    from crane_project.utils import lvsr_prime_review_v1 as prime
    from crane_project.utils import lvsr_decoded_edge_proxy_review_v1 as candidate
    from crane_project.utils import port_size_qalign_v1 as geometry
    from crane_project.utils.port_size_pair_prime_v1 import PairPrimeSizeHead
    from crane_project.tools import train_symeood_r4_lvsr_pair_v1 as original
    from crane_project.tools import diagnose_lvsr_direction_representation_v1 as common
    from crane_project.tools.run_symeood_110_lvsr_v1 import schedule_indices
    from crane_project.tools.check_lvsr_decoded_edge_proxy_review_v1 import main as checks
    torch.set_num_threads(1)
    started = time.monotonic(); sources = {name: sha(ROOT/name) for name in SOURCES}
    save('edge_mapping_protocol.json', dict(spec=SPEC, source_SHA=sources))
    def timeout(*unused): raise RuntimeError('Fixed600-second CPU budget exceeded')
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, 600)
    old_opt = torch.optim.Optimizer.__init__
    def forbidden(*unused, **kw): raise RuntimeError('Frozen mapping audit forbids optimizer')
    torch.optim.Optimizer.__init__ = forbidden
    try:
        checks()
        identity = p.checked_sources()
        train, val, _ = original.load_inputs(p.paths(), identity, SimpleNamespace(torch=torch))
        inputs_digest = common.records_digest(train+val)
        present = [v for v in train if len(v['midpoint_original'])]
        assert len(present) == 5112 and len(val) == 887 and len(train)-len(present) == 4
        records = [present[i] for i in schedule_indices(5112, 6)] + val
        previous = read(OUT/'edge_review_completion.json'); pins = previous['pinned_SHA']
        assert all(sha(ROOT/name) == value for name, value in pins.items())
        assert all(sha(ROOT/name) == value for name, value in read(OUT/'core_summary.json')['source_SHA'].items())
        checkpoint = OUT/'head_epoch_05.pth'
        payload = torch.load(str(checkpoint), map_location='cpu')
        centers = payload['proxy_state']['targets'].detach()
        groups = defaultdict(list); parameter_rows = []; replay = {}; states = {}
        forwards = 0; max_decode = 0.; max_jac_outside = 0.; max_aux_outside = 0.
        for label, path, reference in [
            ('initial', None, None),
            ('L5', p.BASE/'lvsr/attempt_001/train/head_epoch_05.pth', OUT.parent/'attempt_001/L5_all.rows.jsonl.gz'),
            ('PRIME5', checkpoint, OUT/'representative_TRAIN_VAL.rows.jsonl.gz')]:
            torch.manual_seed(1703); head = PairPrimeSizeHead().cpu().train()
            if path: head.load_state_dict(torch.load(str(path), map_location='cpu')['head_state'], strict=True)
            before = common.tensor_digest(head.state_dict())
            if label == 'initial':
                bank = prime.new_proxy_features(20, 648)
                for row in train:
                    if not len(row['midpoint_original']):
                        with torch.no_grad():
                            x = common.batch([row]); o = head(*x[:-1])
                            assert torch.equal(o['boxes_original'], x[-2])
                        forwards += 1
            else: bank = payload['proxy_state']['features'].detach().clone().requires_grad_()
            bank_before = common.tensor_digest({'features': bank})
            if reference:
                with gzip.open(reference, 'rt') as f:
                    saved = {(v['image'], v['scale']): v for v in map(json.loads, f)}
            maximum = 0.
            for start in range(0, len(records), 8):
                rr = records[start:start+8]; x = common.batch(rr); n = len(rr)
                o = head(*x[:-1]); f = o['training_features']; w = o['common_support']; forwards += n
                restored = geometry.residual.original.roi_delta_to_original(o['vectors'].reshape(n,324,2), x[2], x[3], x[4]).reshape(n,81,4,2)
                lengths = torch.stack(((restored[:,:,0]-restored[:,:,2]).norm(dim=-1),
                                       (restored[:,:,1]-restored[:,:,3]).norm(dim=-1)), -1)
                masks, role, order = candidate.masks(lengths, w)
                assert not masks.requires_grad and bool(role.sum(1).eq(1).all())
                assert torch.equal(masks.sum(1), w.sqrt().expand(n,8,9,9).flatten(1))
                weights = w.flatten(1); mass = weights.sum(1)
                rebuilt = (lengths.sort(-1, descending=True)[0]*weights[:,:,None]).sum(1)/mass[:,None]
                max_decode = max(max_decode, float((rebuilt-o['decoded_sizes']).abs().max()))
                assert torch.equal(rebuilt, o['decoded_sizes'])
                target = prime.canonical_pair(geometry.residual.size_targets(x[-1],x[-2],x[3])['log_residual'], x[3], x[-2])
                pred = prime.canonical_pair(o['delta_roi'], x[3], x[-2])
                size = geometry.size_loss(o,x[-1],x[-2],x[3])
                a = candidate.objectives(f,masks,target,bank,centers)
                z, _ = prime.embedding(f,w)
                original_align = prime.alignment_per_sample(z,target,bank,centers)[0]
                gj = torch.autograd.grad(original_align.sum(),f,retain_graph=True)[0]
                gs = torch.autograd.grad(size*n,f,retain_graph=True)[0]
                grads = []
                for edge in range(2):
                    g = torch.autograd.grad(.5*a['alignment'][:,edge].sum(),f,retain_graph=True)[0]
                    jac = torch.autograd.grad(pred[:,edge].sum(),f,retain_graph=True)[0]
                    outside = masks[:,edge].reshape_as(f) == 0
                    aux_out = float(g[outside].abs().max()) if bool(outside.any()) else 0.
                    jac_out = float(jac[outside].abs().max()) if bool(outside.any()) else 0.
                    max_aux_outside = max(max_aux_outside, aux_out)
                    max_jac_outside = max(max_jac_outside, jac_out)
                    assert aux_out == 0 and jac_out == 0
                    grads.append(g)
                assert bool(torch.isfinite(a['alignment']).all()) and bool(torch.isfinite(a['proxy']).all())
                assert all(bool(torch.isfinite(g).all()) for g in grads)
                assert all(v is None for v in torch.autograd.grad(a['alignment'].mean(), tuple(head.terminal.parameters()), retain_graph=True, allow_unused=True))
                total = grads[0]+grads[1]
                if start < 128:
                    flatten = lambda g: torch.cat([v.flatten() for v in g]).double()
                    params = tuple(head.stem.parameters())
                    s = flatten(torch.autograd.grad(size,params,retain_graph=True))
                    j = flatten(torch.autograd.grad(.001*original_align.mean(),params,retain_graph=True))
                    b = flatten(torch.autograd.grad(.001*a['alignment'].mean(),params,retain_graph=True))
                    pg = torch.autograd.grad(.001*(a['alignment'].mean()+a['proxy'].mean()),bank,retain_graph=True)[0]
                    assert bool(torch.isfinite(pg).all()) and bool(torch.isfinite(b).all())
                    cosine = lambda v: float(s.dot(v)/(s.norm()*v.norm())) if s.norm()*v.norm()>0 else None
                    parameter_rows.append(dict(head=label,batch=start//8+1,size_norm=float(s.norm()),joint_aux_norm=float(j.norm()),
                        mapped_aux_norm=float(b.norm()),joint_cosine=cosine(j),mapped_cosine=cosine(b),proxy_gradient_norm=float(pg.norm())))
                relative_gap = (lengths[:,:,0]-lengths[:,:,1]).abs()/lengths.amax(-1).clamp_min(1e-12)
                active = weights > 0; long0 = order[:,:,0] == 0
                for j, row in enumerate(rr):
                    if reference:
                        expected = torch.tensor(saved[row['image'],row['scale']]['edge_residual'])
                        maximum = max(maximum,float((o['boxes_original'][j]-expected).abs().max()))
                        assert torch.allclose(o['boxes_original'][j],expected,atol=1e-4,rtol=1e-6)
                    else: assert torch.equal(o['boxes_original'][j], x[-2][j])
                    assert torch.equal(o['boxes_original'][j,[0,1,4,5]],x[-2][j,[0,1,4,5]])
                    error = (x[-2][j,2:4].sort(descending=True)[0]/x[-1][j,2:4].sort(descending=True)[0]-1).abs()
                    split = 'val' if row['split']=='val' else 'train_s1' if row['scale']==1. else 'train_s05'
                    for edge in range(2):
                        norm = grads[edge][j].norm(); joint_norm = gj[j].norm()
                        entry = dict(mixed=int(bool((long0[j]&active[j]).any() and ((~long0[j])&active[j]).any())),
                            near_001_mass=float((weights[j]*(relative_gap[j]<=.001)).sum()/mass[j]),
                            near_01_mass=float((weights[j]*(relative_gap[j]<=.01)).sum()/mass[j]),
                            exact_tie_mass=float((weights[j]*(relative_gap[j]==0)).sum()/mass[j]),
                            sample_norm=float(a['sample_norm'][j,edge]),zero_sample=int(a['sample_norm'][j,edge]<=prime.EPS),
                            proxy_min_norm=float(a['proxy_norm'][j,edge].min()),
                            zero_proxy=int(bool((a['proxy_norm'][j,edge]<=prime.EPS).any())),
                            aux_grad_norm=float(norm),zero_aux_grad=int(norm==0),
                            total_to_joint_grad_norm=float(total[j].norm()/joint_norm) if joint_norm>0 else 0.,
                            size_aux_cosine=float(F.cosine_similarity(gs[j].flatten()[None],grads[edge][j].flatten()[None],dim=1)[0]),
                            alignment=float(a['alignment'][j,edge]),proxy_loss=float(a['proxy'][j,edge]))
                        need = 'shrink' if target[j,edge]<math.log(.99) else 'enlarge' if target[j,edge]>math.log(1.01) else 'near'
                        subsets = ['all',need]
                        if error[edge]<=.03: subsets.append('accurate')
                        if error[edge]>.1: subsets.append('difficult')
                        for group in ('overall',row['domain'],row['sequence']):
                            for subset in subsets: groups[label,split,group,['long','short'][edge],subset].append(entry)
                assert not any(v.grad is not None for v in head.parameters()) and bank.grad is None
                if start%1024==0: print(json.dumps(dict(head=label,done=start+n,seconds=time.monotonic()-started)),flush=True)
            assert common.tensor_digest(head.state_dict())==before and common.tensor_digest({'features':bank})==bank_before
            replay[label] = maximum; states[label] = dict(head=before,proxy=bank_before,unchanged=True)
        report = [dict(zip(('head','split','group','edge','subset'), key),n=len(values),
                       metrics={field:describe([v[field] for v in values]) for field in values[0]})
                  for key,values in sorted(groups.items())]
        assert common.records_digest(train+val)==inputs_digest
        assert p.checked_sources()==identity and all(sha(ROOT/name)==value for name,value in pins.items())
        assert all(sha(ROOT/name)==value for name,value in sources.items())
        assert not torch.cuda.is_initialized() and forwards == SPEC['total_forward_views']
        save('edge_mapping_summary.json',dict(spec=SPEC,source_SHA=sources,groups=report,parameter_gradients=parameter_rows,
            replay_max_abs=replay,max_decode_abs=max_decode,max_jacobian_outside_mask=max_jac_outside,
            max_aux_gradient_outside_mask=max_aux_outside,state_SHA=states,performance_evidence=False))
        save('edge_mapping_completion.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,optimizer_created=False,
            updates=0,TEST=False,elapsed_seconds=time.monotonic()-started,forward_views=forwards,
            sources_SHA=sources,pinned_SHA=pins,source_checks_passed=True,parameters_inputs_unchanged=True,
            initial_neutral_present=5999,initial_neutral_missing=4,mathematical_checks=7))
        names=['edge_mapping_protocol.json','edge_mapping_summary.json','edge_mapping_completion.json']
        save('edge_mapping_manifest.json',{name:sha(OUT/name) for name in names})
        print(json.dumps(dict(passed=True,seconds=time.monotonic()-started,forwards=forwards)),flush=True)
    except BaseException as e:
        save('edge_mapping_failure.json',dict(error=repr(e),updates=0,retry=False)); raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        torch.optim.Optimizer.__init__=old_opt


if __name__=='__main__': main()
