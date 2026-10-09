#!/usr/bin/env python3
"""One bounded spatial/axis comparison, complete TRAIN -> complete VAL."""
import argparse
from copy import deepcopy
import gzip,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.utils import port_reliability_spatial_axis_v1 as core
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_complementarity_v1 as metrics
from crane_project.utils import port_reliability_state_continuity_v1 as states
from crane_project.tools import run_port_reliability_box_contrast_v1 as historical
prior=historical.prior
sha=historical.sha
PROTOCOL=ROOT/'crane_project/tools/port_reliability_spatial_axis_v1_protocol.json'
SOURCES=ROOT/'crane_project/tools/port_reliability_spatial_axis_v1_sources.json'


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def jsonl(path):
    with gzip.open(path,'rt') as f:return [json.loads(l) for l in f]
def checked(allow_missing_weights=False):
    contract=json.loads(PROTOCOL.read_text());manifest=json.loads(SOURCES.read_text())
    if contract['settings']!=core.SETTINGS or contract['protocol']!=core.VERSION:raise ValueError('Contract changed')
    for p,v in manifest['sources'].items():
        if sha(ROOT/p)!=v:raise ValueError('Source changed: '+p)
    for p,v in contract['input_pins'].items():
        if allow_missing_weights and not (ROOT/p).exists() and p in (prior.HEAD,historical.historical.B_PATH,prior.CACHE+'/cache_manifest.json'):continue
        if sha(ROOT/p)!=v:raise ValueError('Frozen input changed: '+p)
    return contract,manifest

def parts():
    rows,policy=historical.load_rows()
    result={k:sorted([r for r in rows if r['reliability_role']==v],key=lambda r:r['image']) for k,v in [('TRAIN','train'),('VAL','val')]}
    if any(r['split'] not in ('train','train_sim') for r in result['TRAIN']) or any(r['split']!='val' for r in result['VAL']):raise ValueError('Unexpected role/TEST')
    if [len(result[k]) for k in ('TRAIN','VAL')]!=[2558,887] or [sum(metrics.bad(r) is True for r in result[k]) for k in ('TRAIN','VAL')]!=[71,223]:raise ValueError('Label support changed')
    return result,policy


def collect(rows, role, gpu, out):
    """One frozen forward/image. GT is excluded from sampling/online APIs."""
    import torch
    from torch.nn import functional as F
    from mmcv.parallel import collate, scatter
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as frozen
    from crane_project.utils.port_geometry_midpoint_sigma_v1 import SigmaMidpointHead
    from crane_project.utils import port_geometry_refine_g_v1 as geometry
    if role not in ('TRAIN', 'VAL'): raise ValueError('TRAIN/VAL only; no TEST access')
    cfg = frozen.g.check_cfg(); sources, data_identity = frozen.checked_data()
    source = {r['image']:r for r in sources}; lookup = {r['image']:r for r in rows}
    detector = frozen.build_detector(cfg, gpu); before = frozen.g.state_digest(detector)
    manifest = json.loads((ROOT/prior.CACHE/'cache_manifest.json').read_text())
    if before != manifest['detector_state']: raise ValueError('Frozen B state differs')
    head = SigmaMidpointHead(1.5).cuda(gpu)
    payload = prior.load_trusted(ROOT/prior.HEAD)
    if (payload['sigma_cells'] != 1.5 or payload['epoch'] != 3 or payload['updates'] != 2706
            or payload['frozen_b']['checkpoint_sha256'] != historical.PINS[historical.historical.B_PATH]):
        raise ValueError('Wrong frozen midpoint identity')
    head.load_state_dict(payload['head_state'], strict=True); head.eval(); head.requires_grad_(False)
    head_before = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    dataset = frozen.dataset_for(cfg, role.lower(), 1.)
    names = [Path(info['filename']).stem for info in frozen.infos(dataset)]
    if len(names) != len(lookup) or set(names) != set(lookup): raise ValueError('Complete split membership differs')
    captured = {}; traces = []; maximum_b = np.zeros(6); maximum_m = np.zeros(6)
    started = time.monotonic()
    for index, name in enumerate(names):
        row = lookup[name]; value = scatter(collate([dataset[index]], samples_per_gpu=1), [gpu])[0]
        image, metas = value['img'][0], value['img_metas'][0]; meta = metas[0]
        if (len(value['img']) != 1 or image.shape != (1,3,1024,1024) or len(metas) != 1
                or Path(meta['filename']).stem != name or meta.get('flip', False)
                or list(meta['ori_shape'][:2][::-1]) != row['image_size']):
            raise ValueError('Deterministic original view differs: '+name)
        frozen.g.checked_meta(meta); frozen.g.assert_detector_frozen(detector)
        with torch.no_grad():
            fpn = detector.extract_feat(image)
            if fpn[0].shape != (1,256,128,128): raise ValueError('P3 shape differs')
            raw = frozen.g.flatten_prediction(detector.simple_test_from_features(fpn, metas, rescale=False))
            b = frozen.g.flatten_prediction(detector.simple_test_from_features(fpn, metas, rescale=True))
            if len(b) != int(row['pred'] is not None): raise ValueError('Output/missing identity differs')
            trace = dict(image=name, role=role, image_sha256=source[name]['image_sha256'],
                         present=bool(len(b)), scale_factor=np.asarray(meta['scale_factor']).tolist(), GT_online=False)
            if len(b):
                stored_b = np.asarray(row['b_original']); delta = np.abs(b[0]-stored_b)
                delta[4] = abs((b[0,4]-stored_b[4]+np.pi/2)%np.pi-np.pi/2)
                maximum_b = np.maximum(maximum_b, delta)
                if (not np.allclose(b[0,:4], stored_b[:4], atol=5e-4, rtol=2e-5)
                        or delta[4] > 2e-6 or b[0,5] != stored_b[5]): raise ValueError('Original B replay differs: '+name)
                bt, rt = image.new_tensor(b), image.new_tensor(raw)
                if (not torch.equal(bt[:,5], rt[:,5]) or not torch.allclose(
                        frozen.g.map_boxes(rt[:,:5], meta, inverse=True), bt[:,:5], atol=1e-4, rtol=1e-6)):
                    raise ValueError('Original/model coordinate restoration differs')
                roi, support, _ = frozen.g.sample_local(fpn[0], rt[:,:5], meta, 'aligned')
                xy = image.new_tensor(np.asarray(meta['scale_factor'])[:2]).reshape(1,2)
                decoded = head(roi, support, bt, rt[:,:5], xy)['boxes_original'][0].cpu().numpy()
                stored_m = np.asarray(row['pred']); delta = np.abs(decoded-stored_m)
                delta[4] = abs((decoded[4]-stored_m[4]+np.pi/2)%np.pi-np.pi/2)
                maximum_m = np.maximum(maximum_m, delta)
                if (not np.allclose(decoded[:4], stored_m[:4], atol=5e-4, rtol=2e-5)
                        or delta[4] > 2e-6 or decoded[5] != stored_m[5]): raise ValueError('Original M replay differs: '+name)
                if metrics.bad(dict(row, pred=decoded.tolist())) != metrics.bad(row):
                    raise ValueError('Replay changes size label')
                # Always sample STORED M; frozen replays above never replace it.
                # Sample STORED final M only, never a GT-conditioned or synthetic box.
                boxes=stored_m[None].copy()
                model_boxes=geometry.map_boxes(image.new_tensor(boxes[:,:5]),meta)
                local, local_support, _=geometry.sample_local(fpn[0],model_boxes,meta,'aligned')
                raw_roi=local[0].cpu().numpy(); raw_support=local_support[0].cpu().numpy()
                if not np.isfinite(raw_roi).all() or not np.isfinite(raw_support).all():raise ValueError('Nonfinite spatial cache')
                after=frozen.g.flatten_prediction(detector.simple_test_from_features(fpn,metas,rescale=False))
                if not np.array_equal(raw,after):raise ValueError('Sampling changed B output')
                ds=simple.descriptor(stored_m,row['image_size'])[1:]
                trace.update(feature_sha256=hashlib.sha256(raw_roi.astype('<f4').tobytes()).hexdigest(),
                  support_sha256=hashlib.sha256(raw_support.astype('<f4').tobytes()).hexdigest(),
                  support_mean=float(raw_support.mean()),box_original=stored_m.tolist(),
                  model_box=model_boxes[0].cpu().tolist())
                captured[name]=(raw_roi,raw_support,ds)
            traces.append(trace)
        if index%100 == 0 or index==len(names)-1: print('Box-conditioned', role, index+1, '/', len(names), flush=True)
        del value, image, metas, fpn
    if (frozen.g.state_digest(detector) != before or any(p.grad is not None for p in head.parameters())
            or any(not torch.equal(v,head.state_dict()[k].cpu()) for k,v in head_before.items())):
        raise ValueError('Frozen detector/midpoint changed')
    ids=sorted(captured)
    raws=np.stack([captured[i][0] for i in ids]); supports=np.stack([captured[i][1] for i in ids])
    ds=np.stack([captured[i][2] for i in ids])
    core.validate_inputs(raws,supports,ds)
    np.savez_compressed(out/(role.lower()+'_spatial.npz'),features=raws,support=supports,descriptors=ds,images=np.asarray(ids))
    with gzip.open(out/(role.lower()+'_trace.jsonl.gz'),'xt') as stream:
        for trace in sorted(traces,key=lambda r:r['image']):stream.write(json.dumps(trace,allow_nan=False)+'\n')
    proof=dict(frames=len(rows),outputs=len(ids),data_identity=data_identity,detector_state=before,
      maximum_B_replay_difference=maximum_b.tolist(),maximum_M_replay_difference=maximum_m.tolist(),
      original_M_never_replaced=True,one_feature_extraction_per_frame=True,
      elapsed_seconds=time.monotonic()-started,detector_updates=0,midpoint_updates=0,GT_online=False)
    del detector,head,dataset,captured;torch.cuda.empty_cache()
    return raws,supports,ds,ids,proof


def evaluate(rows,roi,support,ds,ids,models,normalizer,device):
    import torch
    outputs=[r for r in rows if r['pred'] is not None]
    if ids!=[r['image'] for r in outputs]:raise ValueError('Feature order differs')
    x,geom=core.normalize(roi,support,ds,normalizer)
    base=core.checked_anchor([r['size_risks']['full_simple'] for r in outputs])
    scores=deepcopy(rows);lookup={r['image']:r for r in scores}
    for r in scores:r['risks']=dict(full_simple=r['size_risks']['full_simple'],score_only=1-r['pred'][5] if r['pred'] is not None else None,**dict.fromkeys(core.ARMS))
    axis_stats={};diagnostics={};replay={}
    for arm,model in models.items():
        risks=[];deltas=[];axes=[];maxdiff=0.
        with torch.no_grad():
            for start in range(0,len(x),128):
                ix=slice(start,start+128)
                v=model(torch.tensor(x[ix],device=device),torch.tensor(support[ix],device=device),torch.tensor(geom[ix],device=device),torch.tensor(base[ix],device=device),arm)
                risk,delta,axis=[v[k].cpu().numpy() for k in ('risk','delta','axis')]
                nr,nd,na=core.numpy_forward(__import__('crane_project.utils.port_reliability_spatial_axis_v1_torch',fromlist=['exported']).exported(model),x[ix],support[ix],geom[ix],base[ix],arm)
                maxdiff=max(maxdiff,float(np.max(np.abs(nr-risk))))
                if maxdiff>2e-5 or np.max(np.abs(na-axis))>2e-3:raise ValueError('Torch/NumPy spatial replay differs')
                risks.extend(risk);deltas.extend(delta);axes.extend(axis)
        for image,risk,axis in zip(ids,risks,axes):
            lookup[image]['risks'][arm]=float(risk)
            lookup[image].setdefault('axis_predictions',{})[arm]=np.asarray(axis).tolist()
        axis_stats[arm]=core.axis_metrics(outputs,np.asarray(axes))
        d=np.asarray(deltas);diagnostics[arm]=dict(min=float(d.min()),max=float(d.max()),saturation_fraction=float((np.abs(d)>.49).mean()))
        replay[arm]=maxdiff
    return scores,axis_stats,diagnostics,replay


def train(args):
    import torch
    from crane_project.utils import port_reliability_spatial_axis_v1_torch as head
    contract,manifest=checked();rows,policy=parts()
    for p,v in manifest['sources'].items():
        committed=subprocess.check_output(['git','show','HEAD:'+p],cwd=str(ROOT))
        if hashlib.sha256(committed).hexdigest()!=v:raise ValueError('Commit source before training: '+p)
    if args.out.resolve().parent.parent!=(ROOT/'work_dirs'/core.VERSION).resolve():raise ValueError('Use work_dirs/version/RUN_ID/result')
    if args.out.exists():raise FileExistsError('Refuse overwrite')
    args.out.mkdir(parents=True);started=time.monotonic()
    torch.set_num_threads(1);torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    device='cuda:'+str(args.gpu)
    roi,support,ds,ids,train_proof=collect(rows['TRAIN'],'TRAIN',args.gpu,args.out)
    outputs=[r for r in rows['TRAIN'] if r['pred'] is not None]
    if len(ids)!=2558:raise ValueError('Incomplete TRAIN')
    norm=core.fit_normalizer(roi,support,ds,'TRAIN');write(args.out/'normalizer.json',norm)
    x,geom=core.normalize(roi,support,ds,norm)
    tensors=[torch.tensor(v,device=device) for v in (x,support,geom)]
    anchor=torch.tensor(core.checked_anchor([r['size_risks']['full_simple'] for r in outputs]),device=device)
    labels=np.asarray([metrics.bad(r) for r in outputs],dtype=np.float64)
    y=torch.tensor(labels,device=device);w=torch.tensor(core.class_weights(labels),device=device)
    target=core.targets(outputs);t=torch.tensor(target,device=device)
    if not np.array_equal(np.max(np.abs(target),axis=1)>1,labels.astype(bool)):raise ValueError('Continuous/binary label mismatch')
    ix=torch.tensor(np.r_[np.flatnonzero(labels==0)[:4],np.flatnonzero(labels==1)[:4]],device=device)
    smoke_models=head.make_models(device);smoke={}
    for arm,model in smoke_models.items():
        with torch.no_grad():
            if not torch.equal(model(*tensors,anchor,arm)['risk'],anchor):raise ValueError('Not exactly neutral')
        opt=head.optimizer(model)
        batch=[v[ix] for v in tensors]+[anchor[ix],y[ix],w[ix],t[ix]]
        changes=[head.update(model,opt,*batch,arm,diagnose=True) for _ in range(2)]
        if not changes[0]['component_gradients']['weighted_aux_norm']>0:raise ValueError('Auxiliary disconnected')
        step=changes[0] if arm=='spatial_axis' else changes[1]
        if not step['gradient_by_parameter'].get('stem.0.weight',0)>0:raise ValueError('Spatial feature gradient disconnected')
        if arm!='spatial_axis' and any(k.startswith('axis_head') for k in changes[0]['gradient_by_parameter']):raise ValueError('Auxiliary gradient in BCE control')
        smoke[arm]=dict(exact_neutral=True,discarded_updates=changes)
    write(args.out/'smoke_report.json',smoke);del smoke_models
    models=head.make_models(device);initial={a:head.exported(m) for a,m in models.items()}
    if any(v!=initial[core.ARMS[0]] for v in initial.values()):raise ValueError('Initial states differ')
    count={a:sum(p.numel() for p in m.parameters()) for a,m in models.items()}
    if set(count.values())!={13115}:raise ValueError('Capacity differs')
    write(args.out/'initial_heads.json',initial);opts={a:head.optimizer(m) for a,m in models.items()}
    updates=dict.fromkeys(core.ARMS,0);rng=np.random.RandomState(1701);order_hash=hashlib.sha256()
    training_start=time.monotonic()
    with (args.out/'train_log.jsonl').open('x') as stream:
        for epoch in range(1,101):
            order=rng.permutation(len(ids));order_hash.update(order.astype('<i8').tobytes())
            for slot,start in enumerate(range(0,len(ids),256)):
                ix=torch.tensor(order[start:start+256],device=device)
                batch=[v[ix] for v in tensors]+[anchor[ix],y[ix],w[ix],t[ix]]
                for arm,model in models.items():
                    record=head.update(model,opts[arm],*batch,arm,diagnose=slot==0);updates[arm]+=1
                    stream.write(json.dumps(dict(epoch=epoch,slot=slot,arm=arm,batch_size=len(ix),**record),allow_nan=False)+'\n')
            stream.flush()
            if epoch==1 or epoch%10==0:print('TRAIN epoch',epoch,'updates',updates,flush=True)
    training_seconds=time.monotonic()-training_start
    if updates!=dict.fromkeys(core.ARMS,1000):raise ValueError('Budget differs')
    final={a:head.exported(m) for a,m in models.items()}
    torch.save({a:m.state_dict() for a,m in models.items()},args.out/'final_heads.pth')
    saved=torch.load(args.out/'final_heads.pth',map_location=device);reload=head.make_models(device)
    for a in core.ARMS:
        reload[a].load_state_dict(saved[a])
        if head.exported(reload[a])!=final[a]:raise ValueError('Save/reload differs')
    write(args.out/'models.json',dict(protocol=core.VERSION,models=final,normalizer=norm,epoch=100,update_counts=updates))
    scored={};axis_stats={};diagnostics={};replay={}
    scored['TRAIN'],axis_stats['TRAIN'],diagnostics['TRAIN'],replay['TRAIN']=evaluate(rows['TRAIN'],roi,support,ds,ids,models,norm,device)
    del roi,support,ds,x,geom,tensors,anchor,y,w,t,batch;torch.cuda.empty_cache()
    # Complete VAL extraction begins only after final weights have been saved.
    roi,support,ds,ids,val_proof=collect(rows['VAL'],'VAL',args.gpu,args.out)
    scored['VAL'],axis_stats['VAL'],diagnostics['VAL'],replay['VAL']=evaluate(rows['VAL'],roi,support,ds,ids,models,norm,device)
    cutoffs={a:core.calibrate(scored['VAL'],a) for a in core.METHODS}
    statistics={role:core.statistics(r,cutoffs) for role,r in scored.items()}
    gate=core.gate(statistics['VAL']);status='VAL_PASS_FROZEN_TEST_PENDING' if gate['passed'] else 'VAL_FAILED_STOP'
    for role,rs in scored.items():
        with gzip.open(args.out/('scored_'+role+'.jsonl.gz'),'xt') as stream:
            for r in rs:
                r['candidate_decisions']={a:core.decide(r['original_simple_decision'],r['risks'][a],cutoffs[a]['risk_le']) for a in core.ARMS}
                for a in core.ARMS:
                    dec=r['candidate_decisions'][a];old=r['original_simple_decision']
                    if any(dec[k]!=old[k] for k in ('center_accepted','angle_accepted','final_box_original')) or dec['risks']['angle']!=old['risks']['angle']:raise ValueError('Protected output changed')
                stream.write(json.dumps(r,allow_nan=False)+'\n')
    write(args.out/'cutoffs.json',cutoffs);checked()
    report=dict(protocol=core.VERSION,status=status,contract=contract,sources=manifest,git_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
      input_pins=contract['input_pins'],statistics=statistics,gate=gate,cutoffs=cutoffs,axis_metrics=axis_stats,diagnostics=diagnostics,
      ranking={role:{m:core.ranking_summary(rs,m) for m in core.METHODS} for role,rs in scored.items()},
      formal_policy_summary={role:{g:states.summarize(v,{r['image'] for r in v if r['original_simple_decision']['size_accepted']}) for g,v in metrics.grouped(rs).items()} for role,rs in scored.items()},
      center={role:{g:metrics.center(v) for g,v in metrics.grouped(rs).items()} for role,rs in scored.items()},
      collection=dict(TRAIN=train_proof,VAL=val_proof),engineering=dict(save_reload_exact=True,neutral_exact=True,smoke_discarded=True,torch_numpy_max_error=replay),
      parameter_counts=count,update_counts=updates,batch_order_sha256=order_hash.hexdigest(),fixed_final_epoch=100,
      TEST_read=False,GT_online=False,TEST_repeatedly_exposed=True,new_data_roles=False,original_policy_changed=False,
      boxes_scores_output_center_angle_unchanged=True,training_seconds=training_seconds,elapsed_seconds=time.monotonic()-started)
    write(args.out/'report.json',report);(args.out/'analysis.md').write_text(core.markdown(report))
    write(args.out/'completion.json',dict(protocol=core.VERSION,status=status,TEST_read=False,artifacts={p.name:sha(p) for p in sorted(args.out.iterdir()) if p.is_file()}))
    print(json.dumps(dict(status=status,gate=gate,training_seconds=training_seconds)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('prepare','train'),default='train');parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--out',type=Path);parser.add_argument('--allow-missing-server-inputs',action='store_true',help='Local prepare only; missing server weight/cache inputs explicitly disclosed, training requires all pins');args=parser.parse_args()
    if Path.cwd().resolve()!=ROOT:raise SystemExit('Run at project root')
    if args.mode=='prepare':
        checked(args.allow_missing_server_inputs);rows,policy=parts()
        for r in rows['TRAIN']+rows['VAL']:
            if r['pred'] is not None:core.targets([r])
            if core.decide(r['original_simple_decision'],r['size_risks']['full_simple'],policy['simple_policy']['cutoffs']['simple']['size']['risk_le'])!=r['original_simple_decision']:raise ValueError('Neutral flags differ')
        print(json.dumps(dict(passed=True,TRAIN=2558,VAL=887,TEST_read=False,no_training=True,all_original_inputs_checked=not args.allow_missing_server_inputs)));return
    if args.allow_missing_server_inputs:parser.error('--allow-missing-server-inputs is prepare-only')
    if args.out is None:parser.error('--out required')
    existed=args.out.exists()
    try:train(args)
    except Exception as e:
        if not existed and args.out.exists():write(args.out/'failure.json',dict(error=type(e).__name__+': '+str(e)))
        raise

if __name__=='__main__':main()
