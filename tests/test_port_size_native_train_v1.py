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


def test_original_train_sim_partition_and_reject_wrong_split():
    rows=[dict(shard=shard,domain=domain,split=split,image=domain,scale=sc)
          for shard,sc in [('train_s1',1.),('train_s05',.5),('val_s1',1.)]
          for domain,split in [('real','val' if shard=='val_s1' else 'train'),('sim','val' if shard=='val_s1' else 'train_sim')]]
    tr,va=t.partition_records(rows)
    assert len(tr)==4 and len(va)==2 and {r['domain'] for r in tr}=={'real','sim'}
    with pytest.raises(ValueError):t.partition_records([dict(rows[1],split='train')])
    with pytest.raises(ValueError):t.partition_records([dict(rows[0],shard='unknown')])


def test_only_exact_loader_repair_compatible_with_immutable_cache():
    old={'crane_project/tools/audit_port_size_native_p3_v1.py':'39531b247ecd32e246a991f0c78c9270aa010e6661ecd6d6e2ff3192999ded0a',
         'crane_project/tools/port_size_native_train_v1_protocol.json':'9d2e3d7c017eb3d557c6f884555a34bf7a802de1c773379604e34be9f9b2128b',
         'crane_project/tools/run_port_size_native_train_v1.py':'531ec9156fafdb8012bf01ab0b4203a6d8ec9840cef23c88e5d49b577930ca4a',
         'tests/test_port_size_native_train_v1.py':'6d0a5fb987c9bdb39df8c1a1081bd9ff1750b2292dcabffd13d7916551c6515f'}
    cached=dict(current=dict(protocol='same',sources=dict(old,head='unchanged')),parent={'x':1},native={'y':2})
    current=deepcopy(cached)
    for k in old:current['current']['sources'][k]='repaired'
    assert t.cache_source_compatibility(cached,current)['status']=='EXACT_KNOWN_LOADER_REPAIR'
    assert t.cache_source_compatibility(current,current)['status']=='EXACT'
    current['current']['sources']['head']='changed'
    with pytest.raises(ValueError):t.cache_source_compatibility(cached,current)
    current=deepcopy(cached);current['parent']['x']=3
    with pytest.raises(ValueError):t.cache_source_compatibility(cached,current)


def test_explicit_revised_resource_cap_keeps_single_gpu(monkeypatch,tmp_path):
    from crane_project.tools import audit_port_size_native_p3_v1 as audit
    class Thread:
        def __init__(self,**kwargs):pass
        def start(self):pass
    monkeypatch.setattr(audit.threading,'Thread',Thread)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3')
    monkeypatch.setattr(audit,'smi',lambda *args:[['2','gpu2','8000'],['3','gpu3','8000']])
    state,stop,thread=audit.setup_gpu(tmp_path,user_limit_mib=7000,process_stop_mib=6800,allocator_cap_mib=6000)
    assert state['physical_gpu']==3 and state['user_limit_mib']==7000 and state['process_stop_mib']==6800
    assert audit.os.environ['CUDA_VISIBLE_DEVICES']=='3'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2,3')
    with pytest.raises(ValueError):audit.setup_gpu(tmp_path)
