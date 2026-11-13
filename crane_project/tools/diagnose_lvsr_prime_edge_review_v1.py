"""Frozen CPU review of branch/edge correspondence and joint normalization.

No optimizer, parameter update, new candidate forward, TEST or fitted probes.
Equal-block normalization is a loss/gradient counterfactual only.
"""
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

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/prime_001'
SPEC=dict(version='prime_edge_review_CPU_20261113_v1',CPU_threads=1,budget_seconds=600,
    heads=['L5','PRIME5'],views_per_head=6003,updates=0,optimizer=False,TEST=False,
    branch='first4/last4 channels; never assume permanently long/short',
    mapping='actual per-location original-image pair lengths and inherited sort; no GT routing',
    near_tie='abs(length0-length1)/max(length0,length1)<=.01',
    intervention='loss-only equal-block normalization of sample and same trained proxy; unchanged2D targets',
    actual_parameter_gradients='first16 natural epoch6 batches for PRIME only; original alignment and size',
    branch_descriptor='324 dimensions each; auxiliary reindex geometry only, no decoder edit',
    adoption=False,download='grouped core only; no weights/features/rows/gradients')

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(2**20),b''):h.update(v)
    return h.hexdigest()
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def mean(v):return sum(v)/len(v) if v else None
def describe(v):
    v=sorted(v);n=len(v)
    return dict(n=n,mean=mean(v),min=v[0] if n else None,max=v[-1] if n else None,
        median=(v[(n-1)//2]+v[n//2])/2 if n else None)

def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1)
    from types import SimpleNamespace
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    from crane_project.utils import lvsr_prime_review_v1 as r
    from crane_project.utils import port_size_qalign_v1 as geometry
    from crane_project.utils.port_size_pair_prime_v1 import PairPrimeSizeHead
    from crane_project.tools import train_symeood_r4_lvsr_pair_v1 as original
    from crane_project.tools import diagnose_lvsr_direction_representation_v1 as d
    from crane_project.tools.run_symeood_110_lvsr_v1 import schedule_indices
    started=time.monotonic();sources={str(Path(__file__).resolve().relative_to(ROOT)):sha(__file__)}
    save(OUT/'edge_review_protocol.json',dict(spec=SPEC,source_SHA=sources))
    def timeout(*unused):raise RuntimeError('Fixed600-second CPU budget exceeded')
    handler=signal.getsignal(signal.SIGALRM);signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,600)
    old_opt=torch.optim.Optimizer.__init__
    def forbidden(*unused,**kw):raise RuntimeError('Frozen edge review forbids optimizer')
    torch.optim.Optimizer.__init__=forbidden
    try:
        identity=p.checked_sources();train,val,_=original.load_inputs(p.paths(),identity,SimpleNamespace(torch=torch))
        digest=d.records_digest(train+val);present=[v for v in train if len(v['midpoint_original'])]
        order=schedule_indices(5112,6);records=[present[i] for i in order]+val
        cp=OUT/'head_epoch_05.pth';assert sha(cp)=='9590597e713d562c3fa953cd96866657d1d17de14d00c822155592e0a2d77301'
        payload=torch.load(str(cp),map_location='cpu');proxy=payload['proxy_state']['features'].detach();centers=payload['proxy_state']['targets'].detach()
        pins=dict(read(OUT/'core_summary.json')['upstream_SHA']);pins[str(cp.relative_to(ROOT))]=sha(cp)
        pins[str((OUT/'selection.json').relative_to(ROOT))]=sha(OUT/'selection.json')
        def blocknorm(x):return F.normalize(x.reshape(len(x),2,324),dim=2,eps=1e-6).flatten(1)/math.sqrt(2.)
        # Exact counterfactual properties, not tests mirroring a train implementation.
        example=torch.randn(3,648);changed=example.clone();changed[:,:324]*=7
        assert torch.allclose(blocknorm(example),blocknorm(changed),atol=1e-6,rtol=1e-6)
        assert torch.allclose(blocknorm(example).norm(dim=1),torch.ones(3),atol=1e-6)
        groups=defaultdict(list);param=[];replay={};forwards=0;max_decode=0.
        for label,path,reference in [('L5',p.BASE/'lvsr/attempt_001/train/head_epoch_05.pth',OUT.parent/'attempt_001/L5_all.rows.jsonl.gz'),
                                     ('PRIME5',cp,OUT/'representative_TRAIN_VAL.rows.jsonl.gz')]:
            with gzip.open(reference,'rt') as f:saved={(v['image'],v['scale']):v for v in map(json.loads,f)}
            pins[str(reference.relative_to(ROOT))]=sha(reference)
            head=PairPrimeSizeHead().cpu().train();head.load_state_dict(torch.load(str(path),map_location='cpu')['head_state'],strict=True)
            state=d.tensor_digest(head.state_dict());maximum=0.
            for start in range(0,len(records),8):
                rr=records[start:start+8];x=d.batch(rr);o=head(*x[:-1]);f=o['training_features'];w=o['common_support'].detach().flatten(1)
                forwards+=len(rr);weighted=(f*o['common_support'].sqrt()).reshape(len(rr),2,324)
                norms=weighted.norm(dim=2);energy=norms.square()/norms.square().sum(1,keepdim=True).clamp_min(1e-12)
                vectors=o['vectors'];n=len(rr)
                restored=geometry.residual.original.roi_delta_to_original(vectors.reshape(n,324,2),x[2],x[3],x[4]).reshape(n,81,4,2)
                lengths=torch.stack(((restored[:,:,0]-restored[:,:,2]).norm(dim=-1),(restored[:,:,1]-restored[:,:,3]).norm(dim=-1)),-1)
                role=(lengths[:,:,0]>=lengths[:,:,1]);active=w>0;mass=w.sum(1)
                fraction=(w*role).sum(1)/mass
                mixed=((role&active).any(1)&((~role)&active).any(1))
                tie=((lengths[:,:,0]-lengths[:,:,1]).abs()/lengths.amax(-1).clamp_min(1e-12)<=.01)
                rebuilt=(lengths.sort(-1,descending=True)[0]*w[:,:,None]).sum(1)/mass[:,None]
                max_decode=max(max_decode,float((rebuilt-o['decoded_sizes']).abs().max()))
                assert torch.equal(rebuilt,o['decoded_sizes'])
                target=r.canonical_pair(geometry.residual.size_targets(x[-1],x[-2],x[3])['log_residual'],x[3],x[-2])
                pred=r.canonical_pair(o['delta_roi'],x[3],x[-2]);size=geometry.size_loss(o,x[-1],x[-2],x[3])
                jac=[]
                for edge in range(2):
                    g=torch.autograd.grad(pred[:,edge].sum(),f,retain_graph=True)[0].reshape(n,2,324).square().sum(2)
                    jac.append(g/g.sum(1,keepdim=True).clamp_min(1e-30))
                ga=gb=gs=None
                if label=='PRIME5':
                    z,_=r.embedding(f,o['common_support']);align,t,_=r.alignment_per_sample(z,target,proxy,centers)
                    balanced=-(t*(-5*torch.cdist(blocknorm(weighted.flatten(1)),blocknorm(proxy))).log_softmax(1)).sum(1)
                    ga=torch.autograd.grad(align.sum(),f,retain_graph=True)[0].reshape(n,2,324)
                    gb=torch.autograd.grad(balanced.sum(),f,retain_graph=True)[0].reshape(n,2,324)
                    gs=torch.autograd.grad(size*n,f,retain_graph=True)[0].reshape(n,2,324)
                    if start<128:
                        pars=list(head.stem.parameters())
                        a=torch.cat([v.flatten() for v in torch.autograd.grad(size,pars,retain_graph=True)]).double()
                        b=torch.cat([v.flatten() for v in torch.autograd.grad(.001*align.mean(),pars,retain_graph=True)]).double()
                        param.append(dict(batch=start//8+1,size_norm=float(a.norm()),aux_norm=float(b.norm()),
                            cosine=float(a.dot(b)/(a.norm()*b.norm())) if a.norm()*b.norm()>0 else None))
                for j,record in enumerate(rr):
                    expected=torch.tensor(saved[record['image'],record['scale']]['edge_residual'])
                    maximum=max(maximum,float((o['boxes_original'][j]-expected).abs().max()))
                    assert torch.allclose(o['boxes_original'][j],expected,atol=1e-4,rtol=1e-6)
                    assert torch.equal(o['boxes_original'][j,[0,1,4,5]],x[-2][j,[0,1,4,5]])
                    error=(x[-2][j,2:4].sort(descending=True)[0]/x[-1][j,2:4].sort(descending=True)[0]-1).abs()
                    entry=dict(energy0=float(energy[j,0]),dominant_energy=float(energy[j].max()),
                        zero0=int(norms[j,0]<=1e-6),zero1=int(norms[j,1]<=1e-6),
                        mixed=int(mixed[j]),role0_long_mass=float(fraction[j]),tie_mass=float((w[j]*tie[j]).sum()/mass[j]),
                        long_jac_branch1=float(jac[0][j,1]),short_jac_branch0=float(jac[1][j,0]))
                    if ga is not None:
                        norm=ga[j].norm(dim=1);balanced_norm=gb[j].norm(dim=1)
                        entry.update(aux_energy0=float(norm[0].square()/norm.square().sum().clamp_min(1e-30)),
                            aux_balanced_energy0=float(balanced_norm[0].square()/balanced_norm.square().sum().clamp_min(1e-30)),
                            norm_change_cosine=float(F.cosine_similarity(ga[j].flatten()[None],gb[j].flatten()[None],dim=1)[0]))
                        for branch in range(2):
                            entry['size_aux_cosine'+str(branch)]=float(F.cosine_similarity(gs[j,branch][None],ga[j,branch][None],dim=1)[0])
                    split='val' if record['split']=='val' else 'train_s1' if record['scale']==1. else 'train_s05'
                    for group in ('overall',record['domain'],record['sequence']):
                        for edge in range(2):
                            need='shrink' if target[j,edge]<math.log(.99) else 'enlarge' if target[j,edge]>math.log(1.01) else 'near'
                            subsets=['all',need]
                            if error[edge]<=.03:subsets.append('accurate')
                            if error[edge]>.1:subsets.append('difficult')
                            for subset in subsets:groups[label,split,group,['long','short'][edge],subset].append(entry)
                assert not any(v.grad is not None for v in head.parameters())
                if start%1024==0:print(json.dumps(dict(head=label,done=start+len(rr),elapsed=time.monotonic()-started)),flush=True)
            replay[label]=maximum
            assert d.tensor_digest(head.state_dict())==state
        report=[]
        for key,values in sorted(groups.items()):
            report.append(dict(zip(('head','split','group','edge','subset'),key),n=len(values),
                metrics={field:describe([v[field] for v in values]) for field in values[0]}))
        assert d.records_digest(train+val)==digest and all(sha(ROOT/name)==value for name,value in pins.items())
        assert p.checked_sources()==identity and not torch.cuda.is_initialized()
        save(OUT/'edge_review_summary.json',dict(spec=SPEC,groups=report,parameter_gradients=param,
            replay_max_abs=replay,decode_reconstruction_max_abs=max_decode,
            trained_proxy_block_norms=proxy.reshape(20,2,324).norm(dim=2).tolist(),source_SHA=sources,
            inference_modified=False,performance_evidence=False))
        save(OUT/'edge_review_completion.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,
            optimizer_created=False,updates=0,TEST=False,forward_views=forwards,elapsed_seconds=time.monotonic()-started,
            parents_inputs_selection_unchanged=True,pinned_SHA=pins,source_SHA=sources,
            synthetic_block_scaling_invariance=True,decode_reconstruction_exact=True))
        names=['edge_review_protocol.json','edge_review_summary.json','edge_review_completion.json']
        save(OUT/'edge_review_manifest.json',{n:sha(OUT/n) for n in names})
        print(json.dumps(dict(passed=True,seconds=time.monotonic()-started,forward_views=forwards)),flush=True)
    except BaseException as e:
        save(OUT/'edge_review_failure.json',dict(error=repr(e),updates=0,retry=False));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler);torch.optim.Optimizer.__init__=old_opt

if __name__=='__main__':main()
