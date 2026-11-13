"""Fixed-budget CPU hard-residual learning check; no candidate or TEST."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib,json,math
from pathlib import Path
import random,runpy,sys,time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/partial_visual_norm_001'
SPEC=dict(version='lvsr_hard_fit_review_20261113_v1',CPU_threads=1,GPU=False,TEST=False,budget_seconds=600,
    heldout='real_seq13 entire video; native304 evaluated, both scales excluded from fitting',
    selection='all other TRAIN views with M joint error>10% and verified original-delivery witness; matched same-video/scale controls with both M edges<=3%, evenly spaced frame identities',
    formal_dataset_and_sampling_changed=False,diagnostic_copies_only=True,
    arms=['original_full2480','terminal_bias_only8'],steps_per_arm=500,batch_size=8,seed=1703,
    lr=.001,weight_decay=0.,betas=[.9,.999],eps=1e-8,clip_norm=10.,loss='original SmoothL1 beta.1; no BSAM or reweighting',
    checkpoints=[0,500],representative='fixed500, no VAL selection or candidate adoption',
    clear_fit='hard pool both MAEs <=0.9 trained-bias control and <M; joint10 strictly greater than bias control',
    migration='report separately whether fixed final head preserves M both-edge MAE and joint10 on heldout video and VAL; no theorem or adoption from a small fit',
    download='core grouped JSON only; diagnostic weights/optimizer/samples/prediction rows stay server')


def errors(r):
    g=sorted(r['gt_original'][0,2:4].tolist(),reverse=True);m=sorted(r['midpoint_original'][0,2:4].tolist(),reverse=True)
    return [abs(a/b-1) for a,b in zip(m,g)]


def select_fit(records,feasible,held='real_seq13'):
    present=[r for r in records if r['split']!='val' and len(r['midpoint_original'])]
    key=lambda r:(r['image'],r['scale'])
    hard=sorted([r for r in present if r['sequence']!=held and max(errors(r))>.1 and key(r) in feasible],key=key)
    cells=Counter((r['sequence'],r['scale']) for r in hard);controls=[]
    for cell,n in sorted(cells.items()):
        pool=sorted([r for r in present if (r['sequence'],r['scale'])==cell and max(errors(r))<=.03],key=lambda r:(r['frame_id'],r['image']))
        if len(pool)<n:raise ValueError('Insufficient matched accurate controls')
        controls.extend([pool[round(i*(len(pool)-1)/(n-1))] for i in range(n)] if n>1 else pool[:1])
    controls.sort(key=key)
    if not hard or len(hard)!=len(controls):raise ValueError('Invalid hard/control diagnostic pool')
    fit=sorted(hard+controls,key=key)
    if len(set(map(key,fit)))!=len(fit):raise ValueError('Duplicate diagnostic view')
    if any(r['sequence']==held or r['split']=='val' for r in fit):raise ValueError('Holdout leakage')
    return hard,controls,fit


def reachable(r,torch):
    from crane_project.utils import port_geometry_midpoint_edge_feasibility_v1 as g
    from crane_project.utils import port_geometry_midpoint_edge_residual_v1 as residual
    from crane_project.tools.diagnose_port_geometry_midpoint_edge_feasibility_v1 import native_source_guards
    from diagnose_cross_trial_support import witness
    b,bm,xy,m=[r[k] for k in ('boxes_original','boxes_model','scale_xy','midpoint_original')]
    _,_,_,sides,rot=residual.original.frame(b,bm,xy)
    q=g.diagnose_frame(b[0].tolist(),m[0].tolist(),r['gt_original'][0].tolist(),bm[0].tolist(),xy[0].tolist(),
        runtime_frame=dict(context_sides_model=sides[0].tolist(),rotation_model=rot[0].tolist()),
        source_guard_checks=native_source_guards(torch,residual.original,b,m),
        verify_witness=lambda pair,goal:witness(torch,residual,b,bm,xy,m,pair,goal))
    return q['delivered_size_reachable']


def run():
    import torch
    torch.set_num_threads(1);sys.path.insert(0,str(OUT.parent))
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
    from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
    from crane_project.tools import diagnose_lvsr_small_fit_v1 as old
    from crane_project.tools import diagnose_lvsr_direction_representation_v1 as d
    from crane_project.tools import diagnose_lvsr_conr_failure_v1 as io
    from crane_project.tools import train_symeood_r4_lvsr_pair_v1 as trainer
    started=time.monotonic();identity=p.checked_sources()
    source_names=[__file__,ROOT/'tests/test_lvsr_hard_fit_v1.py',old.__file__,d.__file__,ROOT/'work_dirs/lvsr_diagnostic_20261112_v1/diagnose_cross_trial_support.py']
    sources={str(Path(q).resolve().relative_to(ROOT)):io.sha(q) for q in source_names}
    io.save(OUT/'hard_fit_protocol.json',dict(settings=SPEC,source_SHA=sources))
    try:
        train,val,_=trainer.load_inputs(p.paths(),identity,SimpleNamespace(torch=torch));records=train+val
        before=d.records_digest(records);hard_all=[r for r in train if len(r['midpoint_original']) and max(errors(r))>.1]
        available={};feasible=set()
        for r in hard_all:
            v=reachable(r,torch);available[v]=available.get(v,0)+1
            if v is True:feasible.add((r['image'],r['scale']))
        assert len(hard_all)==131 and len(feasible)==127
        hard,control,fit=select_fit(train,feasible)
        holdout=[r for r in train if r['sequence']=='real_seq13' and r['scale']==1.]
        assert len(holdout)==304 and len(val)==887
        pools={'fit_hard':hard,'fit_accurate_controls':control,'heldout_TRAIN_real_seq13':holdout,'VAL_unfitted':val}
        for label,rr in list(pools.items())[:2]:
            for scale in (1.,.5):pools[label+('_s1' if scale==1. else '_s05')]=[r for r in rr if r['scale']==scale]
        pins=[p.paths()['inputs']/n for n in ('train.pt','val.pt')]+[ROOT/p.parent.R4_CHECKPOINT,p.midpoint_directory()/'train/head_epoch_20.pth',p.BASE/'lvsr/attempt_001/train/head_epoch_05.pth',p.BASE/'lvsr/attempt_001/train/selection.json']
        hashes={str(q.relative_to(ROOT)):io.sha(q) for q in pins}
        sample_id=[dict(image=r['image'],scale=r['scale'],sequence=r['sequence'],domain=r['domain'],hard=max(errors(r))>.1) for r in fit]
        io.save(OUT/'hard_fit_server_only_samples.json',dict(samples=sample_id,heldout='real_seq13'))
        schedule=[];ep=1
        while len(schedule)<4000:
            order=list(range(len(fit)));random.Random(1703+ep).shuffle(order);schedule.extend(order);ep+=1
        schedule=schedule[:4000];schedule_sha=hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
        common=runpy.run_path(str(OUT.parent/'diagnose.py'));histories={};checkpoints={};frozen_checks={};initial_digest=None
        baseline=None;updates=0
        print(json.dumps(dict(stage='hard_fit_prepared',hard=len(hard),controls=len(control),available_hard=len(hard_all),reachable=127,heldout=304)),flush=True)
        for arm in SPEC['arms']:
            random.seed(1703);torch.manual_seed(1703);head=PairSpecificSizeHead();params=old.configure(head,arm)
            digest=d.tensor_digest(head.state_dict())
            if initial_digest is None:initial_digest=digest
            assert digest==initial_digest
            if baseline is None:
                baseline=old.evaluate(head,records)
                assert len(baseline)==6003 and all(r['midpoint']==r['edge_residual'] for r in baseline)
            base={(r['image'],r['scale']):r for r in baseline}
            zero={name:old.report([base[r['image'],r['scale']] for r in rr],common) for name,rr in pools.items() if rr}
            frozen={k:v.clone() for k,v in head.state_dict().items() if arm=='terminal_bias_only8' and not(k.startswith('terminal.') and k.endswith('.bias'))}
            opt=torch.optim.Adam(params,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
            for i in range(500):
                rr=[fit[j] for j in schedule[8*i:8*(i+1)]];old.step(head,opt,d.batch(rr));updates+=1
                if i in (249,499):print(json.dumps(dict(stage='hard_fit_updates',arm=arm,step=i+1,elapsed=time.monotonic()-started)),flush=True)
                if time.monotonic()-started>600:raise RuntimeError('Hard fit CPU budget exceeded')
            assert all(int(opt.state[q]['step'])==500 for q in params)
            for k,v in frozen.items():assert torch.equal(v,head.state_dict()[k])
            final={name:old.report(old.evaluate(head,rr),common) for name,rr in pools.items() if rr}
            histories[arm]=[dict(step=0,metrics=zero),dict(step=500,metrics=final)]
            checkpoint=OUT/('hard_fit_'+arm+'_server_only_step500.pth')
            payload=dict(head_state=head.state_dict(),optimizer_state=opt.state_dict(),step=500,arm=arm,diagnostic_only=True)
            torch.save(payload,str(checkpoint));saved=torch.load(str(checkpoint),map_location='cpu')
            clone=PairSpecificSizeHead();clone.load_state_dict(saved['head_state'],strict=True)
            assert d.tensor_digest(clone.state_dict())==d.tensor_digest(head.state_dict())
            assert old.evaluate(clone,fit[:8])==old.evaluate(head,fit[:8])
            checkpoints[arm]=dict(path=checkpoint.name,sha256=io.sha(checkpoint),step=500,diagnostic_only=True)
            frozen_checks[arm]=dict(unchanged_tensors=len(frozen),trainable_parameters=sum(q.numel() for q in params))
        full=histories['original_full2480'][-1]['metrics']['fit_hard']['overall']['summary']['methods'];bias=histories['terminal_bias_only8'][-1]['metrics']['fit_hard']['overall']['summary']['methods']
        clear=all(full['edge_residual'][e+'_MAE']<=.9*bias['edge_residual'][e+'_MAE'] and full['edge_residual'][e+'_MAE']<full['midpoint'][e+'_MAE'] for e in ('long','short')) and full['edge_residual']['joint10']>bias['edge_residual']['joint10']
        assert d.records_digest(records)==before and p.checked_sources()==identity and not torch.cuda.is_initialized()
        assert all(io.sha(ROOT/q)==h for q,h in {**sources,**hashes}.items())
        io.save(OUT/'hard_fit_summary.json',dict(settings=SPEC,hard_all=len(hard_all),hard_reachable=127,fit_hard=len(hard),fit_controls=len(control),heldout=304,VAL=887,
            fit_video_scale_counts=dict(Counter(r['sequence']+'|'+str(r['scale']) for r in fit)),histories=histories,
            initial_digest=initial_digest,schedule_SHA=schedule_sha,checkpoints=checkpoints,frozen_checks=frozen_checks,clear_fit_signal=clear))
        io.save(OUT/'hard_fit_completion.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,formal_training=False,TEST_access=False,
            optimizer_updates=updates,updates_per_arm=500,elapsed_seconds=time.monotonic()-started,
            source_SHA=sources,pinned_SHA=hashes,cache_digest_before=before,cache_digest_after=before,
            parent_weights_and_selection_unchanged=True,samples_SHA=io.sha(OUT/'hard_fit_server_only_samples.json'),clear_fit_signal=clear))
        files=['hard_fit_'+s+'.json' for s in ('protocol','summary','completion')]
        io.save(OUT/'hard_fit_manifest.json',{n:io.sha(OUT/n) for n in files})
        print(json.dumps(dict(completed=True,clear_fit_signal=clear,seconds=time.monotonic()-started,updates=updates)),flush=True)
    except BaseException as e:
        io.save(OUT/'hard_fit_failure.json',dict(error=repr(e),retry=False,TEST=False));raise


if __name__=='__main__':run()
