"""Bounded CPU support/boundary audit of a review-only shared operator."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from collections import Counter
import json,math,random,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/partial_visual_norm_001'
SPEC=dict(version='shared_spatial_review_20261113_v1',CPU_threads=1,GPU_used=False,TEST_access=False,
    budget_seconds=600,optimizer_updates=0,formal_training=False,parameters=2480,
    operator='ReLU(0.5 conv3x3_d1(h,W)+0.5 conv3x3_d2(h,W)+b); same weight, fixed coefficient, bias once',
    support='unchanged common W; no new hidden-feature mask or renormalization',
    scopes=['6003 fresh zero-correction identity','L5 disabled replay','L5 frozen mechanism intervention','six synthetic CPU tests'],
    natural_gradient_probe='first16 original epoch6 batches8; compare frozen original vs enabled operator gradient, no optimizer',
    reporting='tap geometry and hidden unsupported-source contribution grouped by TRAIN scale/VAL/domain/video/fixedM error; no candidate metrics or ranking',
    caveat='frozen L5 response is out-of-architecture intervention, not a trained candidate; source influence fractions can exceed1 due to cancellation',
    train_admission='engineering findings only; no invented metric threshold or automatic training',
    download='only grouped core JSON/SHA; weights/cache/hidden maps/gradients/per-frame rows stay server')


def stats(v):
    from crane_project.tools.diagnose_lvsr_conr_failure_v1 import mean,quantile
    return dict(mean=mean(v),p50=quantile(v,.5),p95=quantile(v,.95),max=max(v) if v else None)


def grouped(rows):
    report={}
    for split in ('train_s1','train_s05','val'):
        rr=[r for r in rows if r['split']==split];groups={'overall':rr}
        groups.update({d:[r for r in rr if r['domain']==d] for d in ('real','sim')})
        groups.update({s:[r for r in rr if r['sequence']==s] for s in sorted({r['sequence'] for r in rr})})
        groups.update(M_joint_hard=[r for r in rr if r['M_hard']],M_both_accurate=[r for r in rr if r['M_accurate']])
        report[split]={}
        for name,cell in groups.items():
            if not cell:continue
            keys=list(cell[0]['values'])
            report[split][name]=dict(n=len(cell),scalar={k:stats([r['values'][k] for r in cell]) for k in keys},
                original_accepted=sum(r['old_accepted'] for r in cell),intervention_accepted=sum(r['new_accepted'] for r in cell),
                vector_valid=sum(r['vector_valid'] for r in cell),
                all_supported_outputs_have_new_valid_neighbor=sum(r['values']['supported_outputs_with_any_new_valid_neighbor_fraction']>=1-1e-6 for r in cell))
    return report


def flat(g):
    import torch
    return torch.cat([v.reshape(-1) for v in g])


def run():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1)
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    from crane_project.utils import lvsr_shared_spatial_review_v1 as s
    from crane_project.utils import port_size_local_vectors_v1 as vectors
    from crane_project.utils import port_size_qalign_v1 as q
    from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
    from crane_project.tools import train_symeood_r4_lvsr_pair_v1 as trainer
    from crane_project.tools import diagnose_lvsr_direction_representation_v1 as d
    from crane_project.tools import diagnose_lvsr_conr_failure_v1 as io
    from crane_project.tools import diagnose_lvsr_small_fit_v1 as small
    started=time.monotonic();identity=p.checked_sources()
    sources={str(Path(z).resolve().relative_to(ROOT)):io.sha(z) for z in (__file__,s.__file__,ROOT/'tests/test_lvsr_shared_spatial_review_v1.py')}
    io.save(OUT/'shared_spatial_protocol.json',dict(settings=SPEC,source_SHA=sources))
    try:
        train,val,_=trainer.load_inputs(p.paths(),identity,SimpleNamespace(torch=torch));records=train+val
        before=d.records_digest(records)
        cp=p.BASE/'lvsr/attempt_001/train/head_epoch_05.pth';cp_sha=io.sha(cp);assert cp_sha=='b272c50dfe6329414803d5f22c6e1e84d9c3db37b6bb2e31f1c7ed140f662c3e'
        pins=[p.paths()['inputs']/n for n in ('train.pt','val.pt')]+[cp,p.BASE/'lvsr/attempt_001/train/selection.json',ROOT/p.parent.R4_CHECKPOINT,p.midpoint_directory()/'train/head_epoch_20.pth']
        pinsha={str(z.relative_to(ROOT)):io.sha(z) for z in pins}
        torch.manual_seed(1703);old=PairSpecificSizeHead();rng=torch.get_rng_state()
        torch.manual_seed(1703);head=s.SharedSpatialReviewHead();assert torch.equal(rng,torch.get_rng_state())
        assert d.tensor_digest(old.state_dict())==d.tensor_digest(head.state_dict());initial_digest=d.tensor_digest(head.state_dict())
        neutral=small.evaluate(head,records)
        assert len(neutral)==6003 and all(r['midpoint']==r['edge_residual'] for r in neutral)
        assert d.tensor_digest(head.state_dict())==initial_digest
        print(json.dumps(dict(stage='fresh_neutral',views=6003,parameters=2480,elapsed=time.monotonic()-started)),flush=True)
        payload=torch.load(str(cp),map_location='cpu');head.load_state_dict(payload['head_state'],strict=True);head.eval()
        state_before=d.tensor_digest(head.state_dict());saved=d.rows(OUT.parent/'attempt_001/L5_all.rows.jsonl.gz')
        present=[r for r in records if len(r['midpoint_original'])];outrows=[];replay_max=0.;delta_decode_max=0.;capture={};handles=[]
        for branch,b in enumerate(head.stem.branches):
            handles.append(b[1].register_forward_hook(lambda module,args,output,k=branch:capture.__setitem__(k,output.detach())))
        try:
            with torch.no_grad():
                for start in range(0,len(present),8):
                    rr=present[start:start+8];x=d.batch(rr)
                    head.set_spatial_review(False);original=head(*x[:-1]);hidden={k:v.clone() for k,v in capture.items()}
                    expected=torch.tensor([saved[r['image'],r['scale']]['edge_residual'] for r in rr],dtype=x[0].dtype)
                    difference=float((original['boxes_original']-expected).abs().max());replay_max=max(replay_max,difference)
                    assert torch.allclose(original['boxes_original'],expected,atol=1e-4,rtol=1e-6)
                    head.set_spatial_review(True);changed=head(*x[:-1])
                    assert torch.equal(changed['common_support'],original['common_support'])
                    for o in (original,changed):
                        assert torch.equal(o['boxes_original'][:,[0,1,4,5]],x[-2][:,[0,1,4,5]])
                        assert torch.equal(o['boxes_original'][:,2]<o['boxes_original'][:,3],x[-2][:,2]<x[-2][:,3])
                    # Verify the inherited decode computes the same delivered-size driver.
                    rebuilt=vectors.decode(changed['vectors'],changed['common_support'],x[2],x[3],x[4])
                    delta_decode_max=max(delta_decode_max,float((rebuilt['sizes']-changed['decoded_sizes']).abs().max()))
                    assert torch.equal(rebuilt['sizes'],changed['decoded_sizes'])
                    w=changed['common_support'];mass=w.sum((1,2,3)).clamp_min(1e-12)
                    geom=s.tap_geometry(w);values={}
                    for key,part in geom.items():
                        if isinstance(part,dict):
                            for name,v in part.items():values[key+'_'+name]=v
                        else:values[key]=part
                    for branch,b in enumerate(head.stem.branches):
                        h=hidden[branch];layer=b[2]
                        def rms(v):return ((v.square()*w).sum((1,2,3))/(4*mass)).sqrt()
                        a=F.conv2d(h,layer.weight,None,padding=1);c=F.conv2d(h,layer.weight,None,padding=2,dilation=2)
                        hu=h*(w<=0).to(h)
                        ua=F.conv2d(hu,layer.weight,None,padding=1);uc=F.conv2d(hu,layer.weight,None,padding=2,dilation=2)
                        bias=layer.bias[None,:,None,None]
                        values['b%d_d2_minus_d1_RMS_over_d1'%branch]=rms(c-a)/rms(a).clamp_min(1e-12)
                        values['b%d_d1_unsupported_source_RMS_fraction'%branch]=rms(ua)/rms(a).clamp_min(1e-12)
                        values['b%d_d2_unsupported_source_RMS_fraction'%branch]=rms(uc)/rms(c).clamp_min(1e-12)
                        values['b%d_blend_unsupported_source_RMS_fraction'%branch]=rms((ua+uc)*.5)/rms((a+c)*.5).clamp_min(1e-12)
                        values['b%d_second_relu_change_RMS'%branch]=rms(F.relu((a+c)*.5+bias)-F.relu(a+bias))
                    for j,r in enumerate(rr):
                        gt=sorted(r['gt_original'][0,2:4].tolist(),reverse=True);m=sorted(r['midpoint_original'][0,2:4].tolist(),reverse=True)
                        e=[abs(v/t-1) for v,t in zip(m,gt)]
                        outrows.append(dict(split='val' if r['split']=='val' else 'train_s1' if r['scale']==1. else 'train_s05',
                            sequence=r['sequence'],domain=r['domain'],M_hard=max(e)>.1,M_accurate=max(e)<=.03,
                            old_accepted=bool(original['accepted'][j]),new_accepted=bool(changed['accepted'][j]),vector_valid=bool(changed['vector_valid'][j]),
                            values={k:float(v[j]) for k,v in values.items()}))
                    if start%1600==0:print(json.dumps(dict(stage='frozen_support_audit',views=start+len(rr),elapsed=time.monotonic()-started)),flush=True)
                    if time.monotonic()-started>600:raise RuntimeError('CPU review budget exceeded')
        finally:
            for h in handles:h.remove()
        # Gradient change is descriptive; no optimizer and no training-effect inference.
        train_present=[r for r in train if len(r['midpoint_original'])];order=list(range(len(train_present)));random.Random(1709).shuffle(order);batches=[]
        for start in range(0,128,8):
            rr=[train_present[i] for i in order[start:start+8]];x=d.batch(rr);grads=[];losses=[]
            for enabled in (False,True):
                head.set_spatial_review(enabled);o=head(*x[:-1]);loss=q.size_loss(o,x[-1],x[-2],x[3])
                g=flat(torch.autograd.grad(loss,list(head.parameters())));assert torch.isfinite(g).all();grads.append(g);losses.append(float(loss.detach()))
            na,nb=[float(g.norm()) for g in grads]
            batches.append(dict(batch=start//8,n=8,original_loss=losses[0],intervention_loss=losses[1],original_grad_norm=na,intervention_grad_norm=nb,
                cosine=float(grads[0].dot(grads[1]))/(na*nb) if na and nb else None))
        head.set_spatial_review(False);restored=small.evaluate(head,records[:16])
        for r in restored:
            if r['edge_residual'] is not None:
                assert torch.allclose(torch.tensor(r['edge_residual']),torch.tensor(saved[r['image'],r['scale']]['edge_residual']),atol=1e-4,rtol=1e-6)
        assert d.tensor_digest(head.state_dict())==state_before and d.records_digest(records)==before
        assert p.checked_sources()==identity and not torch.cuda.is_initialized()
        assert all(io.sha(ROOT/z)==h for z,h in {**sources,**pinsha}.items())
        # Conv-only MAC accounting excludes geometry, grids, decode, activation and memory.
        original_MAC=2*81*(267*4+4*4*9+4*4);new_MAC=original_MAC+2*81*4*4*9
        summary=dict(passed=True,settings=SPEC,neutral_views=6003,audited_outputs=len(outrows),missing_outputs=4,
            initial_digest=initial_digest,L5_SHA=cp_sha,original_replay_max=replay_max,decode_reconstruction_max=delta_decode_max,
            groups=grouped(outrows),natural_batches=batches,original_conv_MAC=original_MAC,review_conv_MAC=new_MAC,
            conv_MAC_increase_fraction=new_MAC/original_MAC-1,model_state_before=state_before,model_state_after=state_before,
            support_mapping_exact=True,non_size_inheritance_exact=True,parameter_and_RNG_identity=True,production_trainer_changed=False,
            measured_inference_latency=False,adoption=False,formal_training_started=False)
        io.save(OUT/'shared_spatial_summary.json',summary)
        io.save(OUT/'shared_spatial_completion.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,optimizer_updates=0,TEST_access=False,
            elapsed_seconds=time.monotonic()-started,source_SHA=sources,pinned_SHA=pinsha,cache_digest_before=before,cache_digest_after=before,
            original_weights_and_selection_unchanged=True,review_weights_saved=False,source_files_unchanged=True))
        files=['shared_spatial_'+v+'.json' for v in ('protocol','summary','completion')]
        io.save(OUT/'shared_spatial_manifest.json',{n:io.sha(OUT/n) for n in files})
        print(json.dumps(dict(completed=True,seconds=time.monotonic()-started,views=len(outrows),updates=0,original_replay_max=replay_max)),flush=True)
    except BaseException as e:
        io.save(OUT/'shared_spatial_failure.json',dict(error=repr(e),TEST=False,retry=False));raise


if __name__=='__main__':run()
