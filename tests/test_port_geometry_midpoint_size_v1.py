"""Decoded size gradients, unchanged online behavior and finite TRAIN isolation."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from crane_project.utils import port_geometry_midpoint_v1 as m
from crane_project.utils import port_geometry_midpoint_size_v1 as s
from crane_project.tools import preflight_port_geometry_midpoint_size_v1 as t


def inputs(swap=False, square=False, angle=.2, scale=.5):
    b = torch.tensor([[64.,48.,32.,16.,angle,.8]])
    if swap:
        b[:,[2,3]] = b[:,[3,2]]
    if square:
        b[:,2:4] = 24.
    xy = b.new_tensor([[scale,scale+.0001]])
    bm = b[:,:5].clone(); bm[:,:4] *= xy[:,[0,1,0,1]]
    gt = b[:,:5].clone(); gt[:,:2] += b.new_tensor([.5,-.3])
    gt[:,2:4] *= b.new_tensor([1.06,.94]); gt[:,4] += .03
    return b,bm,xy,gt


@pytest.mark.parametrize('swap',[False,True])
@pytest.mark.parametrize('angle',[.2,math.pi/2-1e-5,-math.pi/2+1e-5])
def test_continuous_projection_matches_original_decoder_with_raw_association(swap,angle):
    b,bm,xy,gt = inputs(swap,angle=angle)
    target = m.target_points(gt,b,bm,xy)
    p = target['original'].clone(); p[:,0] += p.new_tensor([.15,-.10])
    local = m.original_to_roi(p,b,bm,xy)
    decoded = m.decode_points(p,b,local,torch.tensor([False]))
    expected = s.projected_edges(p)
    raw = expected[:,[1,0]] if swap else expected
    assert torch.allclose(decoded['candidate_original'][:,2:4],raw,atol=1e-5,rtol=0.)
    assert torch.equal(decoded['boxes_original'][:,5],b[:,5])


def test_zero_online_identity_but_size_gradient_effective_then_stem_and_inputs_detached():
    torch.manual_seed(1703)
    b,bm,xy,gt = inputs(); gt.requires_grad_(); b.requires_grad_(); bm.requires_grad_(); xy.requires_grad_()
    roi = torch.randn(1,256,9,9,requires_grad=True); support = torch.ones(1,1,9,9,requires_grad=True)
    head = m.SpatialMidpointHead(); optimizer = torch.optim.Adam(head.parameters(),lr=.001)
    for step in range(2):
        optimizer.zero_grad(); out = head(roi,support,b,bm,xy)
        target = m.target_points(gt,b,bm,xy)
        edge,_ = s.size_loss(out['points_original'],target)
        if step==0:
            assert torch.equal(out['boxes_original'],b)
            # Loss on the old identity-replaced candidate cannot train the head.
            wrong = (out['candidate_original'][:,2:4].log()-gt.detach()[:,2:4].log()).square().mean()
            assert torch.autograd.grad(wrong,head.output.weight,retain_graph=True)[0].norm()==0
        edge.backward()
        assert head.output.weight.grad.norm()>0
        stem = sum(float(p.grad.square().sum()) for p in head.stem.parameters())
        assert stem==0 if step==0 else stem>0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        optimizer.step()
    assert all(v.grad is None for v in (b,bm,xy,gt,roi,support))


@pytest.mark.parametrize('square',[False,True])
def test_matched_gt_equivalence_scale_translation_and_no_per_pred_edge_sort(square):
    b,bm,xy,gt = inputs(square=square)
    target = m.target_points(gt,b,bm,xy)
    predicted = m.box_midpoints(b[:,:5]).detach().requires_grad_()
    loss,detail = s.size_loss(predicted,target)
    alternate = gt.clone(); alternate[:,[2,3]] = gt[:,[3,2]]; alternate[:,4] += math.pi/2
    other = m.target_points(alternate,b,bm,xy)
    equivalent,_ = s.size_loss(predicted,other)
    assert torch.allclose(loss,equivalent,atol=1e-6,rtol=0.)
    shifted,_ = s.size_loss(predicted+20.,dict(original=target['original']+20.))
    scaled,_ = s.size_loss(predicted*.5,dict(original=target['original']*.5))
    assert torch.allclose(loss,shifted,atol=2e-6,rtol=0.)
    assert torch.allclose(loss,scaled,atol=2e-6,rtol=0.)
    loss.backward()
    assert torch.allclose(predicted.grad.sum(dim=1),torch.zeros(1,2),atol=1e-7,rtol=0.)
    # Axis crossing retains matched pair identity, rather than sorting sides.
    pts = torch.tensor([[[16.,0.],[0.,20.],[-16.,0.],[0.,-20.]]])
    assert torch.equal(s.projected_edges(pts),torch.tensor([[32.,40.]]))
    assert detail['log_residual'].shape==(1,2)


@pytest.mark.parametrize('bad',['nan','collapsed','negative','empty'])
def test_invalid_geometry_aborts_without_clamping_or_drop(bad):
    b,_,_,_ = inputs(); points = m.box_midpoints(b[:,:5])
    if bad=='nan': points[:,0,0]=float('nan')
    elif bad=='collapsed': points[:]=0.
    elif bad=='negative': points[:,[1,3]]=points[:,[3,1]]
    else: points=points[:0]
    with pytest.raises(ValueError): s.projected_edges(points)


def metadata_records():
    return [dict(image=seq+'_'+str(i).zfill(5),sequence=seq,domain=seq.split('_')[0],
        frame_id=i,role='train',scale=scale,eligible=True)
        for scale in (1.,.5) for seq in ('real_seq01','real_seq05','real_seq06','real_seq12','real_seq13','sim_seq08')
        for i in range(220)]


def test_metadata_only_blocks_two_views_roles_gaps_and_fit_only_balanced_schedule():
    source = metadata_records(); before=deepcopy(source)
    records,blocks = t.select_blocks(source)
    assert source==before and len(records)==256 and len(blocks)==16
    identities = {(r['image'],r['role']) for r in records}
    assert len(identities)==128
    assert sum(r['role']=='fit' for r in records)==192
    for sequence in {r['sequence'] for r in blocks}:
        chunks = sorted((r['frame_ids'][0],r['frame_ids'][-1]) for r in blocks if r['sequence']==sequence)
        assert all(b[0]-a[1]-1>=16 for a,b in zip(chunks,chunks[1:]))
    # Eligibility cannot alter selected identities/roles.
    modified=deepcopy(source)
    for r in modified: r['eligible']=r['frame_id']%3!=0
    selected,newblocks=t.select_blocks(modified); assert newblocks==blocks
    batches=t.schedule(selected); assert batches==t.schedule(selected) and len(batches)==200
    assert all(len(b)==8 and sum(selected[i]['domain']=='real' for i in b)==4 and
               all(selected[i]['role']=='fit' and selected[i]['eligible'] for i in b) for b in batches)
    assert {i for b in batches for i in b}=={i for i,r in enumerate(selected) if r['role']=='fit' and r['eligible']}


@pytest.mark.parametrize('bad',['val','test','scale','duplicate'])
def test_block_selection_rejects_wrong_splits_pairing_or_duplicates(bad):
    records=metadata_records()
    if bad=='val': records[0]['role']='val'
    elif bad=='test':
        for r in records:
            if r['sequence']=='real_seq01':
                r['sequence']='real_seq03'; r['image']='real_seq03_'+str(r['frame_id']).zfill(5)
    elif bad=='scale':records.pop()
    else:records.append(deepcopy(records[0]))
    with pytest.raises(ValueError):t.select_blocks(records)


def test_review_requires_geometry_and_motion_not_only_loss_or_dfr():
    records,_=t.select_blocks(metadata_records())
    from crane_project.tools import analyze_port_geometry_midpoint_size_temporal_v1 as d
    rows=[]
    for r in records:
        gt=[64.,48.,32.,16.,.2]; box=gt+[.8]
        rows.append(dict(r,gt=gt,b=box,midpoint=box,candidate=box,accepted=True,
            metrics={'b':{'output':True,'center_hit':True,'center_error_px':0.,'angle_error_deg':0.,'riou':1.},
                     'midpoint':{'output':True,'center_hit':True,'center_error_px':0.,'angle_error_deg':0.,'riou':1.}}))
    groups={}
    for role in ('fit','probe'):
        for domain in ('real','sim'):
            for scale in (1.,.5):
                fs=[d.static_frame(r) for r in rows if (r['role'],r['domain'],r['scale'])==(role,domain,scale)]
                ps,support=d.temporal_pairs(fs)
                groups[role+'/'+domain+'/'+str(scale)]=dict(d.group_summary(fs,ps),temporal_support=support)
    base=dict(rows=rows,groups=groups)
    assert not t.review(base,base)['diagnostic_conditions_met']
    assert not t.review(base,base)['formal_training_approved']
    changed=deepcopy(base)
    changed['groups']['probe/sim/1.0']['midpoint']['dfr_pct_per_frame']=-1.
    changed['groups']['probe/sim/1.0']['midpoint']['temporal']['short']['log_increment_error']['rmse']=.1
    result=t.review(base,changed)
    assert not result['checks']['probe/sim/1.0']['short/gt_relative_motion_rmse']


def fake_contract(tmp_path):
    parent=t.read_json(t.PARENT)
    identity=dict(sources=parent['sources'],sources_sha256=t.sha(t.PARENT),
        protocol_sha256=t.sha(t.ROOT/'crane_project/tools/port_geometry_midpoint_formal_v1_protocol.json'),
        frozen_b=t.read_json(t.ROOT/'crane_project/tools/port_geometry_midpoint_formal_v1_protocol.json')['frozen_b'])
    train=tmp_path/'training';cache=tmp_path/'cache';train.mkdir();cache.mkdir()
    document=dict(status='COMPLETE_FROZEN_B_TRAIN_VAL_CACHE',identity=identity,
        record_counts=dict(train_s1=2558,train_s05=2558,val_s1=887),
        files={'train_s1.pt':'a','train_s05.pt':'b','val_s1.pt':'c'},
        feature_extractions=6003,native_head_calls=18009,detector_updates=0,test_access=False,
        runtime={'synthetic':True},detector_state={'synthetic':True})
    t.write_json(cache/'cache_manifest.json',document)
    complete=dict(identity=identity,status='FORMAL_MIDPOINT_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED',
        epochs_completed=24,detector_updates=0,test_access=False,cache_manifest_sha256=t.sha(cache/'cache_manifest.json'))
    t.write_json(train/'completion.json',complete)
    t.write_json(train/'artifacts.json',dict(files={'completion.json':t.sha(train/'completion.json')}))
    return train,cache,document


def test_source_contract_static_does_not_import_runtime_load_tensors_or_overwrite(tmp_path,monkeypatch):
    train,cache,_=fake_contract(tmp_path)
    def forbidden(*a,**k): raise AssertionError('Runtime/tensor load in static mode')
    monkeypatch.setattr(t,'runtime_modules',forbidden)
    args=t.parser().parse_args(['--check-only','--training-dir',str(train),'--cache-dir',str(cache),'--out-dir',str(tmp_path/'out')])
    report=t.run(args)
    assert report['status']=='STATIC_SIZE_CONTRACT_PASS_NO_TENSOR_LOAD_NO_GPU_NO_UPDATES'
    assert report['head_updates_total']==report['cache_tensor_loads']==0
    assert not report['test_access'] and not report['val_access'] and not report['gpu_devices_used']
    with pytest.raises(FileExistsError):t.run(args)
    altered=deepcopy(t.read_json(t.SOURCES)); altered['sources'][str(Path(t.__file__).relative_to(t.ROOT))]='bad'
    path=tmp_path/'bad_sources.json';t.write_json(path,altered);monkeypatch.setattr(t,'SOURCES',path)
    with pytest.raises(ValueError,match='Source SHA'):t.checked_contract(train,cache)


@pytest.mark.parametrize('relative',[True,False])
def test_python38_script_entry_resolves_file_and_preserves_failed_or_existing_outputs(tmp_path,relative):
    train,cache,_=fake_contract(tmp_path)
    script=Path(t.__file__).resolve()
    target=str(script.relative_to(t.ROOT)) if relative else str(script)
    out=tmp_path/'static'
    command=[sys.executable,'-B',target,'--check-only','--training-dir',str(train),
             '--cache-dir',str(cache),'--out-dir',str(out)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run(command,cwd=str(t.ROOT),env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=t.read_json(out/'completion.json')
    assert report['status']=='STATIC_SIZE_CONTRACT_PASS_NO_TENSOR_LOAD_NO_GPU_NO_UPDATES'
    assert report['head_updates_total']==report['cache_tensor_loads']==0
    assert report['gpu_devices_used']==[]
    # The same protection applies to directories left by failed attempts.
    sentinel=out/'failure_sentinel.json';sentinel.write_text('{"preserve":true}\n')
    original={p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}
    again=subprocess.run(command,cwd=str(t.ROOT),env=env,capture_output=True,text=True)
    assert again.returncode!=0 and 'Existing results are preserved' in again.stderr
    assert {p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}==original


def test_cache_loader_reads_train_shards_only_and_rejects_bad_eligibility(tmp_path,monkeypatch):
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as f
    b,bm,xy,gt=inputs(scale=1.)
    rows=[]
    for seq in ('real_seq01','sim_seq08'):
        r=dict(image=seq+'_00000',sequence=seq,domain=seq.split('_')[0],frame_id=0,
            role='train',split='train_sim' if seq.startswith('sim') else 'train',scale=1.,eligible=True,
            roi=torch.randn(1,256,9,9),support=torch.ones(1,1,9,9),boxes_original=b,
            boxes_model=bm,scale_xy=xy,gt_original=gt)
        rows.append(r)
    # Production counts are fixed; this test intercepts loader payload count
    # rather than weakening run-time checks. A bad flag must fail before merge.
    broken=deepcopy(rows[0]);broken['eligible']=False
    class PayloadRows(list):
        def __len__(self):return 2558
    monkeypatch.setattr(f.g.ready,'TRAIN_COUNTS',{'real_seq01':1,'sim_seq08':1})
    monkeypatch.setattr(t,'sha',lambda path:'same')
    seen=[]
    def load(path,**kwargs):
        seen.append(Path(path).name)
        return dict(identity={},detector_state={},records=PayloadRows([broken,rows[1]]))
    monkeypatch.setattr(torch,'load',load)
    with pytest.raises(ValueError,match='eligibility'):
        t.load_train_cache(tmp_path,dict(identity={},detector_state={},files={'train_s1.pt':'same','train_s05.pt':'same'}),f,torch)
    assert seen==['train_s1.pt'] and 'val_s1.pt' not in seen


def test_complete_train_loader_never_deserializes_val_and_accepts_native_support_roundoff(tmp_path,monkeypatch):
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as f
    monkeypatch.setattr(f.g.ready,'TRAIN_COUNTS',{'real_seq01':1810,'sim_seq08':748})
    monkeypatch.setattr(t,'sha',lambda p:'same')
    seen=[]
    def load(path,**kwargs):
        name=Path(path).name;seen.append(name)
        assert name in ('train_s1.pt','train_s05.pt')
        scale=1. if name=='train_s1.pt' else .5
        b,bm,xy,gt=inputs(scale=scale)
        common=dict(roi=torch.ones(1,256,9,9),support=torch.ones(1,1,9,9)+1e-7,
            boxes_original=b,boxes_model=bm,scale_xy=xy,gt_original=gt)
        rows=[dict(common,image=seq+'_'+str(i).zfill(5),sequence=seq,domain=seq.split('_')[0],
            frame_id=i,role='train',split='train_sim' if seq.startswith('sim') else 'train',
            scale=scale,eligible=True) for seq,n in (('real_seq01',1810),('sim_seq08',748)) for i in range(n)]
        return dict(identity={},detector_state={},records=rows)
    monkeypatch.setattr(torch,'load',load)
    calls=[]
    rows=t.load_train_cache(tmp_path,dict(identity={},detector_state={},files={'train_s1.pt':'same','train_s05.pt':'same'}),f,torch,calls.append)
    assert len(rows)==5116 and seen==calls==['train_s1.pt','train_s05.pt']


def test_probe_missing_output_keeps_three_denominators_and_breaks_temporal_pairs():
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as f
    torch.set_num_threads(1)
    views,_=t.select_blocks(synthetic_views())
    probe=[r for r in views if r['role']=='probe' and r['domain']=='real' and r['scale']==1.]
    missing=probe[3]
    for field in ('roi','support','boxes_original','boxes_model','scale_xy'):
        missing[field]=missing[field][:0]
    missing['eligible']=False
    result=t.evaluate(m.SpatialMidpointHead(),views,torch.device('cpu'),f)
    group=result['groups']['probe/real/1.0']
    assert group['output_coverage']['numerator']==15 and group['output_coverage']['denominator']==16
    assert group['b']['conditional_center_correct']['denominator']==15
    assert group['b']['all_frame_center_correct']['denominator']==16
    assert group['temporal_pairs']==12 and group['temporal_support']['missing_output_pairs']==2
    assert result['groups']['probe/real/0.5']['temporal_pairs']==14


def synthetic_views():
    metadata,_=t.select_blocks(metadata_records())
    generator=torch.Generator().manual_seed(92)
    result=[]
    for r in metadata:
        b,bm,xy,gt=inputs(scale=r['scale'])
        result.append(dict(r,role='train',boxes_original=b,boxes_model=bm,scale_xy=xy,gt_original=gt,
            roi=torch.randn(1,256,9,9,generator=generator)*.1,support=torch.ones(1,1,9,9)))
    return result


def test_two_step_paired_entry_outputs_json_and_failure_budget_preserved(tmp_path,monkeypatch):
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as f
    torch.set_num_threads(1)
    train,cache,document=fake_contract(tmp_path)
    views=synthetic_views();monkeypatch.setitem(t.SETTINGS,'steps_per_arm',2)
    monkeypatch.setattr(t,'checked_contract',lambda *a:(t.protocol_document(),document,{'synthetic':True}))
    monkeypatch.setattr(t,'execution_device',lambda *a:(torch.device('cpu'),{'synthetic':True}))
    monkeypatch.setattr(t,'load_train_cache',lambda *a:views)
    args=t.parser().parse_args(['--training-dir',str(train),'--cache-dir',str(cache),'--out-dir',str(tmp_path/'run')])
    report=t.run(args)
    assert report['head_updates_total']==4 and not report['formal_training_approved']
    assert not list((tmp_path/'run').glob('*.pth'))
    arms=[t.read_json(tmp_path/'run'/(arm+'.json'))['result'] for arm in t.ARMS]
    assert arms[0]['initial']==arms[1]['initial'] and arms[0]['initial_state']==arms[1]['initial_state']
    assert arms[0]['final_state']!=arms[1]['final_state']
    assert arms[0]['logs'][0]['size_weighted_loss']>0 and not arms[0]['logs'][0]['size_applied']
    assert arms[1]['logs'][0]['size_applied'] and arms[1]['logs'][0]['component_gradients']['size_weighted_norm']>0
    original=t.fit_arm
    def fail_second(*args):
        if args[3]=='point_size':raise ValueError('synthetic second arm failure')
        return original(*args)
    monkeypatch.setattr(t,'fit_arm',fail_second)
    args.out_dir=str(tmp_path/'failed')
    with pytest.raises(ValueError,match='second arm'):t.run(args)
    failure=t.read_json(tmp_path/'failed/completion.json')
    assert failure['head_updates_total']==2 and failure['status']=='FAILED_SIZE_PREFLIGHT_REVIEW_REQUIRED'
    assert (tmp_path/'failed/point_only.json').exists() and (tmp_path/'failed/artifacts.json').exists()
