"""CPU chain-rule check of the13 L5 zero descriptors; no training/TEST."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/prime_001'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,value):
    with (OUT/name).open('x') as f: json.dump(value,f,indent=2,allow_nan=False)


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from types import SimpleNamespace
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    from crane_project.utils import lvsr_prime_review_v1 as prime
    from crane_project.utils import lvsr_decoded_edge_proxy_review_v1 as candidate
    from crane_project.utils import port_size_qalign_v1 as geometry
    from crane_project.utils.port_size_pair_prime_v1 import PairPrimeSizeHead
    from crane_project.tools import train_symeood_r4_lvsr_pair_v1 as original
    from crane_project.tools import diagnose_lvsr_direction_representation_v1 as common
    torch.set_num_threads(1);started=time.monotonic()
    receipt=json.loads((OUT/'edge_mapping_completion.json').read_text());pins=receipt['pinned_SHA']
    for name,value in receipt['sources_SHA'].items(): assert sha(ROOT/name)==value
    save('edge_mapping_zero_chain_protocol.json',dict(CPU_threads=1,seconds=120,updates=0,TEST=False,
        scope='Only13 zero edge-view cases already identified; scan L5 TRAIN/VAL, no new candidate or tuning',
        coefficient='.5 per-view edge alignment, before .001 weight and batch averaging',
        source_SHA=sha(__file__),GPU=False))
    def timeout(*unused): raise RuntimeError('Fixed120-second CPU supplement exceeded')
    previous=signal.getsignal(signal.SIGALRM);signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,120)
    old_opt=torch.optim.Optimizer.__init__
    def forbidden(*unused,**kw): raise RuntimeError('No optimizer permitted')
    torch.optim.Optimizer.__init__=forbidden
    try:
        identity=p.checked_sources();train,val,_=original.load_inputs(p.paths(),identity,SimpleNamespace(torch=torch))
        digest=common.records_digest(train+val);records=[v for v in train if len(v['midpoint_original'])]+val
        head=PairPrimeSizeHead().cpu().train()
        head.load_state_dict(torch.load(str(p.BASE/'lvsr/attempt_001/train/head_epoch_05.pth'),map_location='cpu')['head_state'])
        state=common.tensor_digest(head.state_dict())
        payload=torch.load(str(OUT/'head_epoch_05.pth'),map_location='cpu')
        bank=payload['proxy_state']['features'].detach().clone().requires_grad_();centers=payload['proxy_state']['targets']
        bank_state=common.tensor_digest({'features':bank});rows=defaultdict(list)
        for start in range(0,len(records),8):
            rr=records[start:start+8];x=common.batch(rr);n=len(rr);o=head(*x[:-1]);f=o['training_features']
            v=geometry.residual.original.roi_delta_to_original(o['vectors'].reshape(n,324,2),x[2],x[3],x[4]).reshape(n,81,4,2)
            lengths=torch.stack(((v[:,:,0]-v[:,:,2]).norm(dim=-1),(v[:,:,1]-v[:,:,3]).norm(dim=-1)),-1)
            masks,_,_=candidate.masks(lengths,o['common_support'])
            norms=(f.flatten(1)[:,None]*masks).norm(dim=-1)
            indices=(norms<=prime.EPS).nonzero()
            if not len(indices): continue
            target=prime.canonical_pair(geometry.residual.size_targets(x[-1],x[-2],x[3])['log_residual'],x[3],x[-2])
            a=candidate.objectives(f,masks,target,bank,centers)
            params=tuple(head.stem.parameters())
            for j,edge in indices.tolist():
                loss=.5*a['alignment'][j,edge]
                fg=torch.autograd.grad(loss,f,retain_graph=True)[0]
                gradients=torch.autograd.grad(loss,params+(bank,),retain_graph=True)
                stem=math_norm(gradients[:-1]);proxy=float(gradients[-1].double().norm())
                row=rr[j];split='val' if row['split']=='val' else 'train_s1' if row['scale']==1. else 'train_s05'
                rows[split,['long','short'][edge]].append(dict(sample_norm=float(norms[j,edge]),
                    feature_grad_norm=float(fg[j].double().norm()),stem_grad_norm=stem,proxy_grad_norm=proxy))
        result=[]
        for key,values in sorted(rows.items()):
            result.append(dict(split=key[0],edge=key[1],n=len(values),
                sample_norm_max=max(v['sample_norm'] for v in values),
                feature_grad_norm_max=max(v['feature_grad_norm'] for v in values),
                stem_grad_norm_max=max(v['stem_grad_norm'] for v in values),
                proxy_grad_norm_max=max(v['proxy_grad_norm'] for v in values)))
        assert sum(v['n'] for v in result)==13
        assert common.tensor_digest(head.state_dict())==state and common.tensor_digest({'features':bank})==bank_state
        assert common.records_digest(train+val)==digest and p.checked_sources()==identity
        assert all(sha(ROOT/name)==value for name,value in pins.items())
        assert all(sha(ROOT/name)==value for name,value in receipt['sources_SHA'].items())
        assert not torch.cuda.is_initialized() and bank.grad is None and all(v.grad is None for v in head.parameters())
        save('edge_mapping_zero_chain_summary.json',dict(passed=True,groups=result,elapsed_seconds=time.monotonic()-started,
            forward_views=len(records),zero_edge_views=13,parameters_inputs_unchanged=True,CPU_only=True,
            CUDA_initialized=False,optimizer_created=False,updates=0,TEST=False,source_SHA=sha(__file__)))
        save('edge_mapping_zero_chain_manifest.json',{name:sha(OUT/name) for name in
            ('edge_mapping_zero_chain_protocol.json','edge_mapping_zero_chain_summary.json')})
        print(json.dumps(dict(passed=True,groups=result,seconds=time.monotonic()-started)),flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous);torch.optim.Optimizer.__init__=old_opt


def math_norm(values): return sum(float(v.double().square().sum()) for v in values)**.5


if __name__=='__main__': main()
