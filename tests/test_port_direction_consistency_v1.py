"""CPU numerical and routing fixtures; no real reliability performance claim."""
from argparse import Namespace
from copy import deepcopy
import json
import math
import numpy as np
import pytest
import torch
from crane_project.tools import diagnose_port_direction_consistency_v1 as tool
from crane_project.utils import port_direction_consistency_v1 as d
from crane_project.utils.port_structure_reliability_v1 import (
    component_probes_original,quality_targets_original,response_targets,map_points,map_boxes)


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def meta(size=256):
    return dict(scale_factor=[1.]*4,ori_shape=(size,size,3),img_shape=(size,size,3),pad_shape=(size,size,3),flip=False)


def fixture():
    gt=torch.tensor([128.,128.,128.,32.,.25]);probes,names=component_probes_original(gt)
    boxes=torch.cat((gt[None],probes));target,mask,_=quality_targets_original(boxes,gt)
    u=torch.tensor([math.cos(.25),math.sin(.25)])
    axis=gt[:2]+torch.tensor([[-1.],[1.]])*64*u
    response,_=response_targets(axis,(32,32),meta())
    features,details=d.direction_features(response.numpy()[0,1],boxes,meta())
    return dict(image='fixture',domain='real',descriptor=torch.randn(14,8),target=target,mask=mask,
        genuine_count=1,probe_names=names),features,details,boxes,response


def test_ideal_features_discriminate_both_angle_signs_without_GT_argument():
    row,features,details,boxes,response=fixture()
    assert features.shape==(14,7) and details[1]['direction_defined']
    scores=d.analytic_direction_scores(features)
    assert scores[1]>scores[12] and scores[1]>scores[13]
    assert features[1,0]>features[12,0] and features[1,0]>features[13,0]
    assert features[12,1]<0<features[13,1]
    assert abs(details[1]['reference_angle_rad']-.25)<.01


def test_pi_and_width_swap_equivalence_and_no_mutation():
    _,_,_,boxes,response=fixture();original=boxes.clone()
    changed=boxes.clone();changed[:,4]+=math.pi
    a,_=d.direction_features(response.numpy()[0,1],boxes,meta())
    b,_=d.direction_features(response.numpy()[0,1],changed,meta())
    swap=boxes.clone();swap[:,2]=boxes[:,3];swap[:,3]=boxes[:,2];swap[:,4]+=math.pi/2
    c,_=d.direction_features(response.numpy()[0,1],swap,meta())
    assert torch.allclose(a,b,atol=2e-6) and torch.allclose(a,c,atol=2e-6)
    assert torch.equal(boxes,original)


def test_flat_isotropic_and_missing_evidence_never_trusted_angle():
    boxes=torch.tensor([[128.,128.,128.,32.,0.]])
    flat,details=d.direction_features(np.ones((32,32))*.7,boxes,meta())
    assert flat[0,6]==0 and details[0]['reference_angle_rad'] is None
    assert d.analytic_direction_scores(flat)==[None]
    yy,xx=np.mgrid[:32,:32];round_map=np.exp(-((xx-16)**2+(yy-16)**2)/8.)
    iso,details=d.direction_features(round_map,boxes,meta())
    assert iso[0,6]==0 and details[0]['reference_angle_rad'] is None
    far=torch.tensor([[-1000.,-1000.,32.,16.,0.]])
    missing,_=d.direction_features(round_map,far,meta())
    assert torch.isfinite(missing).all() and missing[0,5]==0 and missing[0,6]==0


def test_padding_values_excluded_and_support_records_cropping():
    m=meta();m['img_shape']=(128,128,3);m['ori_shape']=(128,128,3)
    a=np.zeros((32,32));a[8,6:12]=1
    b=a.copy();b[16:,:]=1;b[:,16:]=1
    boxes=torch.tensor([[96.,64.,128.,32.,0.]])
    fa,_=d.direction_features(a,boxes,m);fb,_=d.direction_features(b,boxes,m)
    assert torch.equal(fa,fb) and 0<fa[0,5]<1


def test_reflection_and_isotropic_coordinate_contract():
    _,_,_,boxes,response=fixture();axis=torch.tensor([[64.,128.],[192.,128.]])
    m=meta();m.update(scale_factor=[.5]*4,img_shape=(128,128,3),pad_shape=(128,128,3),flip=True,flip_direction='horizontal')
    transformed=map_boxes(boxes,m,size_mode='annotation')
    restored=map_boxes(transformed,m,inverse=True,size_mode='annotation')
    assert torch.allclose(boxes[:,:4],restored[:,:4],atol=1e-5)
    r,_=response_targets(map_points(axis,m),(16,16),m)
    f,_=d.direction_features(r.numpy()[0,1],transformed,m)
    assert torch.isfinite(f).all()
    bad=meta();bad['scale_factor']=[.5,.7,.5,.7]
    with pytest.raises(ValueError,match='Anisotropic'):
        d.direction_features(response.numpy()[0,1],boxes,bad)


def test_same_initial_scores_equal_capacity_and_frozen_outputs():
    row,features,_,_,_=fixture()
    quality=torch.nn.Sequential(torch.nn.Linear(8,64),torch.nn.ReLU(),torch.nn.Linear(64,3))
    model=d.AngleOnlyQuality(quality)
    a=torch.cat((row['descriptor'],torch.zeros_like(features)),1)
    b=torch.cat((row['descriptor'],features),1)
    assert torch.equal(model(a),model(b))
    assert torch.allclose(model(a),quality(row['descriptor']),atol=1e-6,rtol=0)
    assert torch.equal(model(a)[:,:2],quality(row['descriptor'])[:,:2])
    row['descriptor']=b
    before=deepcopy(quality.state_dict())
    result,fitted=d.fit(model,[row],steps=20,lr=.01,milestones=(10,20))
    assert result['center_size_quality_exact'] and result['optimizer_steps']==20
    assert all(torch.equal(v,quality.state_dict()[k]) for k,v in before.items())
    assert all(not p.requires_grad and p.grad is None for p in fitted.base.parameters())
    assert all(np.array_equal(np.array(a['qualities'])[:,:2],np.array(b['qualities'])[:,:2])
               for a,b in zip(result['before']['views'],result['final']['views']))


def test_direction_loss_gradient_keeps_original_component_denominators():
    row,features,_,_,_=fixture()
    quality=torch.nn.Sequential(torch.nn.Linear(8,64),torch.nn.ReLU(),torch.nn.Linear(64,3))
    model=d.AngleOnlyQuality(quality);row['descriptor']=torch.cat((row['descriptor'],features),1)
    report=d.additional_metrics(model,[row],tool.parent.fitutil.assess(model,[row]))
    q=model(row['descriptor']).sigmoid();t=row['target'];mask=row['mask']
    terms=torch.nn.functional.smooth_l1_loss(q,t,reduction='none')*mask
    expected=.5*terms[0,2]/mask[0].sum()+.5*terms[1:,2].sum()/mask[1:].sum()
    assert abs(report['direction_loss_contribution']-float(expected))<1e-7
    full=tool.parent.fitutil.objective(model,[row]);g1=torch.autograd.grad(full,model.angle[0].weight,retain_graph=True)[0]
    g2=torch.autograd.grad(expected,model.angle[0].weight)[0]
    assert torch.allclose(g1,g2,atol=1e-7)


def test_atomic_serialization_no_partial_file_no_overwrite(tmp_path):
    path=tmp_path/'report.json'
    with pytest.raises(ValueError):d.atomic_json(path,{'bad':float('nan')})
    assert not path.exists()
    d.atomic_json(path,{'complete':True})
    with pytest.raises(FileExistsError):d.atomic_json(path,{'overwritten':True})
    assert json.loads(path.read_text())=={'complete':True}
    assert list(tmp_path.iterdir())==[path]


def test_cache_roundtrip_and_tamper_rejection(tmp_path):
    path=tmp_path/'direction_cache.pt';sources={'fixed':'fixture'};protocol={'candidate_report_sha256':'fixture'}
    cache=dict(sources=sources,protocol=protocol,rows={n:[{}]*8 for n in tool.ARMS},
        candidate_report_sha256='fixture',detector_updates=0,original_head_updates=0)
    tool.save_cache(path,cache)
    assert tool.load_cache(path,sources,protocol)['sources']==sources
    with path.open('ab') as stream:stream.write(b'tampered')
    with pytest.raises(ValueError,match='identity'):tool.load_cache(path,sources,protocol)


def test_cpu_three_arm_orchestration_keeps_centers_and_no_export(tmp_path,monkeypatch):
    row,features,_,_,_=fixture();quality=tool.branch.make_arms()['structure'].quality
    row['descriptor']=torch.randn(14,quality[0].in_features)
    rows={}
    for name,extra in zip(tool.ARMS,(torch.zeros_like(features),features,features)):
        saved=deepcopy(row);saved['descriptor']=torch.cat((row['descriptor'],extra),1);rows[name]=[saved]
    sources={'fixed':'fixture'};protocol={'steps':3,'lr':.001,'milestones':[1,3]}
    cache=dict(sources=sources,protocol=protocol,initial=quality.state_dict(),rows=rows)
    monkeypatch.setattr(tool,'checked_sources',lambda:(sources,protocol,{},{}))
    tool.fit_cache(cache,tmp_path)
    r=json.loads((tmp_path/'direction_report.json').read_text())
    assert r['all_controls_start_scores_exact'] and r['center_size_quality_exact']
    assert not r['deployable_weights_saved'] and not r['val_or_test_read']
    assert r['results']['ideal_template_reference']['uses_GT_in_features']
    assert len({v['parameters'] for v in r['results'].values()})==1
    assert not list(tmp_path.glob('*.pth'))


def test_sources_and_fixed_protocol():
    _,p,old,_=tool.checked_sources()
    assert p['steps']==old['steps']==2000 and p['lr']==old['lr']==.001
    assert p['feature_names']==list(d.FEATURE_NAMES) and p['val_or_test_read'] is False


@pytest.mark.parametrize('change_B',[False,True])
def test_collection_cpu_fixture_checks_B_freeze_and_separates_ideal(tmp_path,monkeypatch,change_B):
    row,_,_,boxes,response=fixture();p3=torch.randn(1,256,32,32)
    gt=boxes[0];u=torch.tensor([math.cos(float(gt[4])),math.sin(float(gt[4]))])
    axis=gt[:2]+torch.tensor([[-1.],[1.]])*64*u
    _,valid=response_targets(axis,(32,32),meta())
    raw=np.array([gt.tolist()+[.7]],dtype=np.float32)
    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor([1.]),requires_grad=False);self.eval()
        def simple_test_from_features(self,feats,metas,rescale=False):
            result=raw.copy()
            if change_B:result[0,2]+=1
            return [[result]]
    detector=Detector();frozen=tool.prior.state_digest(detector)
    make=tool.branch.make_arms;original=make()
    for arm in original.values():arm.eval()
    descriptor,q,_,_=tool.parent.fitutil.capture(original['structure'],p3,boxes,torch.full((14,),.7),meta())
    states={name:tool.prior.state_digest(a) for name,a in original.items()}
    proof=dict(b_cache_state=frozen);inputs=[];oldrows=[];rows=[];replays=[]
    for i in range(8):
        domain='real' if i<4 else 'sim';name='fixture_'+str(i)
        inputs.append(dict(image=name,domain=domain))
        replay=dict(image=name,input_sha256='fixture',view_seed=i,view_image_sha256='fixture_image',
            scale_factor=[1.]*4,img_shape=[256,256,3],pad_shape=[256,256,3],flip=False,flip_direction=None,
            genuine_b_original=raw.tolist())
        oldrows.append(dict(case=dict(image=name,domain=domain,dataset_index=i,view_seed=i),replay=replay))
        saved=deepcopy(row);saved.update(image=name,domain=domain,descriptor=descriptor.clone(),initial_qualities=q.clone())
        rows.append(saved);replays.append(dict(p3_sha256=tool.prior.tensor_sha(p3)))
    oldcache=dict(proof=proof,frozen_b_state=frozen,heads_before=states,
        arms={'structure':dict(initial=original['structure'].quality.state_dict(),rows=rows)},replay=replays)
    args=Namespace(input_snapshot=tmp_path/'snapshot',train_cache=tmp_path/'train',structure_report=tmp_path/'structure',
        branch_checkpoint=tmp_path/'epoch_08.pth',candidate_cache=tmp_path/'descriptors.pt',candidate_report=tmp_path/'fit_report.json',
        b_checkpoint=tmp_path/'B.pth',gpu=0)
    for path in (args.branch_checkpoint,args.candidate_cache,args.candidate_report):path.write_bytes(b'CPU fixture')
    sources=dict(parent_candidate=dict(parent=dict(training_sources={})));protocol={};oldprotocol=dict(checkpoint_sha256=tool.branch.sha(args.branch_checkpoint))
    payload=dict(frozen_b_state=frozen,contract=dict(architecture=tool.branch.architecture(original)),
        arms={name:a.state_dict() for name,a in original.items()})
    monkeypatch.setattr(tool.parent.parent.previous.base,'fixed_bundle',lambda *unused:payload)
    monkeypatch.setattr(tool.prior,'check_cfg',lambda:Namespace(data=Namespace(train=[])))
    monkeypatch.setattr(tool.train,'checked_inputs',lambda *unused:(inputs,proof))
    monkeypatch.setattr(tool.train,'build_runtime',lambda *unused:(detector,dict(synthetic_cpu=True)))
    monkeypatch.setattr(tool.branch,'make_arms',lambda *unused:make('cpu'))
    monkeypatch.setattr(tool.train,'datasets_for',lambda *unused:list(range(8)))
    def view(detector,index,source,gpu,seed):
        values=(p3,boxes,torch.full((14,),.7),meta(),row['target'],row['mask'],1,response,valid)
        return values,deepcopy(oldrows[index]['replay']),([p3],raw.copy(),[meta()])
    monkeypatch.setattr(tool.train,'training_view',view)
    monkeypatch.setattr(tool.train,'axis_for',lambda *unused:axis.numpy())
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda *unused:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *unused:0)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda *unused:0)
    if change_B:
        with pytest.raises(ValueError,match='changed raw B'):
            tool.collect(args,sources,protocol,oldprotocol,{},None,None,oldrows,oldcache)
        return
    cache=tool.collect(args,sources,protocol,oldprotocol,{},None,None,oldrows,oldcache)
    assert cache['heads_before']==cache['heads_after'] and cache['detector_updates']==0
    assert all(r['b_raw_exact_before_after'] for r in cache['replay'])
    assert len(cache['features'])==8 and all(len(v)==8 for v in cache['rows'].values())
    assert cache['features'][4]['annotation_role']=='Webots_OBB_axis_proxy'
    assert not torch.equal(cache['rows']['predicted_relation'][0]['descriptor'],cache['rows']['ideal_template_reference'][0]['descriptor'])
