"""Focused CPU check of actual historical/current loss_single bodies.

Only tests the discovered boolean-mask empty-positive guard difference.
Uses actual SymNFL/SymKLD class ASTs, identity decoder fixture, no registry
mutation or full detector equivalence claim. No optimizer or CUDA execution.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def source(path,old):
    return subprocess.check_output(['git','show','af8307c:'+path],cwd=ROOT).decode() if old else (ROOT/path).read_text()


def actual_class(text,name,namespace):
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name==name)
    node.decorator_list=[]
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual_class_AST>','exec'),namespace)
    return namespace[name]


def loss_body(text,namespace):
    cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='SymEOODHead')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='loss_single');node.decorator_list=[]
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual_loss_single_AST>','exec'),namespace)
    return namespace['loss_single']


def main(out):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from mmdet.models.losses.utils import weight_reduce_loss
    from mmrotate.models.losses.sym_kld_calculator import sym_kld
    paths=['mmrotate/models/dense_heads/sym_eood_head.py','mmrotate/models/losses/sym_nfl_loss.py','mmrotate/models/losses/sym_kld_loss.py']
    versions=[];identities=[]
    for old in (True,False):
        texts={p:source(p,old) for p in paths};ns=dict(torch=torch,nn=nn,F=F,weight_reduce_loss=weight_reduce_loss,sym_kld=sym_kld)
        cls=actual_class(texts[paths[1]],'SymNFLLoss',ns);reg=actual_class(texts[paths[2]],'SymKLDLoss',ns);fn=loss_body(texts[paths[0]],ns)
        versions.append((fn,cls,reg));identities.append(dict(role='historical' if old else 'current',sha256={p:hashlib.sha256(s.encode()).hexdigest() for p,s in texts.items()}))
    rows=[]
    for mask in ([1,1,1],[0,1,1],[1,1,0],[0,0,1],[0,0,0]):
        for iteration in (0,4000):
            for maxsize in (None,1024):
                results=[]
                for fn,cls,reg in versions:
                    loss_cls=cls(loss_weight=.25);loss_cls._local_iter.fill_(5*iteration)
                    head=SimpleNamespace(num_classes=1,cls_out_channels=1,bbox_coder=SimpleNamespace(decode=lambda anchors,preds:anchors+preds),
                        loss_cls=loss_cls,loss_bbox=reg(loss_weight=2.),strict_training_checks=False,
                        use_equi_loss=False,use_invar_loss=False,use_degraded_cls_loss=False,use_degraded_aux2_loss=False,
                        center_size_compensation=None,shape_compensation=None)
                    logits=torch.tensor([[[[.1,-.2,.3]]]],requires_grad=True)
                    pred=torch.tensor([[[[1.,-2.,.5]],[[.3,.1,-.4]],[[3.,-2.,1.]],[[1.,2.,-.5]],[[.01,.03,-.02]]]],requires_grad=True)
                    anchors=torch.tensor([[100.,100.,40.,20.,.2],[110.,120.,42.,21.,.1],[130.,110.,38.,18.,-.1]])
                    labels=torch.tensor(mask);weights=torch.tensor([1.,.5,1.]);target=torch.zeros((3,5));bbox_weights=(labels==0)[:,None].expand(3,5).float()
                    losses=fn(head,logits,pred,anchors,labels,weights,target,bbox_weights,max(int((labels==0).sum()),1),decode_max_size=maxsize)
                    gradients=torch.autograd.grad(sum(losses),[logits,pred],allow_unused=True)
                    assert all(bool(torch.isfinite(x).all()) for x in losses)
                    results.append((losses,gradients))
                old,new=results;equal_losses=all(torch.equal(a,b) for a,b in zip(old[0],new[0]));equal_grads=all(torch.equal(a,b) for a,b in zip(old[1],new[1]))
                rows.append(dict(labels=mask,positive_count=sum(v==0 for v in mask),iteration=iteration,decode_max_size=maxsize,
                    historical_losses=[float(x) for x in old[0]],current_losses=[float(x) for x in new[0]],
                    losses_exact=equal_losses,gradients_exact=equal_grads,
                    gradient_max_abs=max(float((a-b).abs().max()) for a,b in zip(old[1],new[1]))))
    result=dict(cases=len(rows),passed=all(r['losses_exact'] and r['gradients_exact'] for r in rows),results=rows,source=identities,
        interpretation='pos_inds is boolean mask, not index vector. numel>0 vs any differs only on no-positive masks; actual SymKLD empty-pred guard returnszero. Finite fixture losses/gradients equal, not full detector/history equivalence.')
    assert result['passed'];out.write_text(json.dumps(result,indent=2));print('FOCUSED_CPU',len(rows),result['passed'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
