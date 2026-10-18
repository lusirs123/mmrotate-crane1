import io
from copy import deepcopy
import torch
import numpy as np
import pytest
from crane_project.tools import run_port_size_native_train_v1 as t
from crane_project.utils import port_size_native_p3_v1 as n
from test_port_size_native_p3_v1 import fixture


def test_actual_loss_adam_resume_and_axis_targets():
    x=fixture(swap=True,anisotropic=True);r=n.read_points(*x[:6],'cached_p3')
    gt=x[5][:,:5].clone();gt[:,2:4]*=torch.tensor([.98,1.03])
    batch=(r,*x[2:6],gt)
    torch.manual_seed(1703);h=n.CommonPointSizeHead();opt=torch.optim.Adam(h.parameters(),lr=.001)
    first=t.update(h,opt,batch,torch)
    assert first['stem_grad_norm']==0 and first['terminal_grad_norm']>0
    saved=dict(head=deepcopy(h.state_dict()),opt=deepcopy(opt.state_dict()),rng=t.rng_get(torch))
    s=io.BytesIO();torch.save(saved,s);s.seek(0);got=torch.load(s)
    clone=n.CommonPointSizeHead();clone.load_state_dict(got['head']);op=torch.optim.Adam(clone.parameters(),lr=.001);op.load_state_dict(got['opt'])
    t.rng_set(saved['rng'],torch);a=t.update(h,opt,batch,torch);end=t.rng_get(torch)
    t.rng_set(got['rng'],torch);b=t.update(clone,op,batch,torch)
    assert a['stem_grad_norm']>0 and a==b
    assert t.equal(h.state_dict(),clone.state_dict(),torch) and t.equal(opt.state_dict(),op.state_dict(),torch)
    assert t.equal(end,t.rng_get(torch),torch)


def test_fixed_near_and_cross_edge_damage():
    gt=[0,0,100,50,0];m=[0,0,100,46,0,.9];c=[0,0,102,49,0,.9]
    r=dict(gt=gt,b=m,midpoint=m,edge_residual=c,domain='sim',sequence='sim1',split='val',scale=1.)
    d=t.near_joint([r]);v=d['sim']
    assert v['near']['long']['count']==1 and v['near']['long']['harmed']==1
    assert v['near']['short']['count']==0
    assert v['patterns']['wrong_edge_fixed_near_other_harmed']==1
    assert v['patterns']['short_only']==1


def test_frozen_selection_key_does_not_use_train():
    s={d:{'edge_residual':{'sizes':{'edges':{e:{'mae':v} for e,v in zip(('long','short'),mae)}}}} for d,mae in [('sim',[.04,.03]),('real',[.08,.07])]}
    assert t.ranking(s,7)==[.04,.03,.08,.07,7]
    p=t.protocol()
    assert p['training']['epochs']==24 and p['limits']['epochs_per_arm']==24
    assert p['parameter_count']==2328 and '0.1pp' in p['qualifications']


def test_mmap_batch_identity_and_missing_rejection(tmp_path):
    x=fixture();r=n.read_points(*x[:6],'cached_p3')
    path=tmp_path/'points.npy';np.save(str(path),r['samples'].numpy())
    store=object.__new__(t.PointStore);store.torch=torch
    store.maps={('train_s1','cached_p3'):np.load(str(path),mmap_mode='r')}
    store.shared={'train_s1':{k:r[k] for k in ('masks','local','geometry')}}
    row=dict(shard='train_s1',offset=0,present=True,b=x[2][0].tolist(),bm=x[3][0].tolist(),scale_xy=x[4][0].tolist(),midpoint=x[5][0].tolist(),gt=x[5][0,:5].tolist())
    batch=store.batch([row],'cached_p3','cpu')
    assert torch.equal(batch[0]['samples'],r['samples']) and torch.equal(batch[-2],x[5])
    assert not store.maps['train_s1','cached_p3'].flags.writeable
    with pytest.raises(ValueError):store.batch([dict(row,present=False)],'cached_p3','cpu')
