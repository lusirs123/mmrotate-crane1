"""Frozen VAL candidate replay and CPU aggregate for U/S/G; no TEST path."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'crane_project/tools'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open('x') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader();writer.writerows(rows)


def candidates(work, config):
    import torch
    import numpy as np
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmdet.models import build_detector
    from diagnose_symeood_c_frozen_v1 import CandidateReader,dataset_infer,dataset_items,tensor_sha
    cfg=Config.fromfile(str(config));sweep=json.loads((work/'ckpt_sweep/sweep_results.json').read_text())
    for epoch in dict.fromkeys([sweep['selected_checkpoint'],'epoch_24']):
        cp=sweep['all_checkpoints'][epoch]
        assert sha(cp['checkpoint'])==cp['checkpoint_sha256']
        assert sha(cp['results_pkl'])==cp['results_pkl_sha256']
        dest=work/'candidate_diagnostic'/epoch;dest.mkdir(parents=True,exist_ok=False)
        model=build_detector(cfg.model).cuda().eval()
        load_checkpoint(model,cp['checkpoint'],map_location='cpu',strict=True)
        ds=dataset_infer(cfg,'val');items=dataset_items(ds)
        assert len(items)==887
        with open(cp['results_pkl'],'rb') as f:gold=pickle.load(f)
        observer=CandidateReader(model,dest/'candidates_val');h=hashlib.sha256()
        with torch.no_grad():
            for index,(child,i,frame,image,ann) in enumerate(items):
                data=scatter(collate([ds[index]],samples_per_gpu=1),[torch.cuda.current_device()])[0]
                h.update(frame.encode());h.update(tensor_sha(data['img'][0]).encode())
                observer.context=(frame,child.get_ann_info(i)['bboxes'])
                actual=model(return_loss=False,rescale=True,**data)[0][0]
                assert np.array_equal(actual,gold[index][0]), ('VAL replay mismatch',epoch,frame)
        observer.close()
        (dest/'receipt.json').write_text(json.dumps(dict(passed=True,VAL_exact=887,checkpoint_sha256=cp['checkpoint_sha256'],input_tensor_sequence_sha256=h.hexdigest()),indent=2))
        del model;torch.cuda.empty_cache()


def frame_rows(cp):
    import numpy as np
    from analyze_symeood_ablation_first4_v1 import txtbox,iou,angle
    gtroot=ROOT/'crane_project/data/crane_grab_port_day2night_v1/val/annfiles'
    p=Path(cp['results_pkl'])
    assert sha(p)==cp['results_pkl_sha256']
    with p.open('rb') as f:predictions=pickle.load(f)
    gtfiles=sorted(f for f in gtroot.glob('*.txt') if not f.name.startswith('._'))
    assert len(predictions)==len(gtfiles)==887
    rows=[]
    for gtfile,prediction in zip(gtfiles,predictions):
        boxes=prediction[0];assert len(boxes)<=1
        gt=txtbox(gtfile);assert gt is not None
        domain,seq,fid=re.match(r'(real|sim)_(.+)_(\d+)',gtfile.stem).groups()
        pred=None if len(boxes)==0 else np.asarray(boxes[0][:5],float)
        export=txtbox(p.parent/'Task1_grab'/gtfile.name)
        assert (export is None)==(pred is None)
        center=float(np.linalg.norm(export[:2]-gt[:2])) if export is not None else float('inf')
        r=dict(frame=gtfile.stem,domain=domain,seq=seq,fid=int(fid),output=pred is not None,
               score=float(boxes[0][-1]) if len(boxes) else 0.,center_export=center,
               riou_export=iou(export,gt),riou_native=iou(pred,gt),
               gated_angle_sq=(math.degrees(angle(export[4],gt[4])) if center<10 else 90.)**2)
        if pred is not None:
            assert np.isfinite(pred).all() and min(pred[2:4])>0
            lp,sp=sorted(pred[2:4],reverse=True);lg,sg=gt[2:4]
            el,es=math.log(lp/lg),math.log(sp/sg)
            r.update(diag_export=float(np.hypot(export[2],export[3])),angle_export=float(export[4]),
                     center_error=float(np.linalg.norm(pred[:2]-gt[:2])),
                     angle_eligible=lg/sg>=1.2,
                     angle_error_deg=math.degrees(angle(pred[4]+(math.pi/2 if pred[2]<pred[3] else 0),gt[4])),
                     long_error=abs(lp-lg)/lg,short_error=abs(sp-sg)/sg,
                     long_bias=(lp-lg)/lg,short_bias=(sp-sg)/sg,a=(el+es)/2,r=el-es)
        rows.append(r)
    return rows


def aggregate(base):
    import numpy as np
    from analyze_symeood_ablation_first4_v1 import aggregate as sizes
    from audit_symeood_c_existing_results_v1 import aggregate as temporal
    contract=json.loads((base/'protocol.json').read_text());assert contract['state']=='FROZEN'
    out=base/'analysis_v1';out.mkdir(exist_ok=False)
    summaries=[];video_rows=[];all_frames=[];paired=[];candidate_rows=[];probes=[];cache={};checks=0
    for seed in contract['seeds']:
        initial=set();input_maps={}
        for reg in ('iou','kld'):
            for obj in ('U','S','G'):
                runid='s%d_%s_%s'%(seed,reg,obj.lower());work=base/runid
                identities=[json.loads(x) for x in (work/'identity_rank0.jsonl').read_text().splitlines()]
                initial.add(identities[0]['initial_parameters_sha256'])
                for rank in (0,1):
                    records=[json.loads(x) for x in (work/('inputs_rank%d.jsonl'%rank)).read_text().splitlines()]
                    latest={r['epoch']:r for r in records};assert set(latest)==set(range(1,25))
                    for epoch,r in latest.items():
                        key=(rank,epoch)
                        if key in input_maps:assert input_maps[key]==r['sha256']
                        else:input_maps[key]=r['sha256']
                        checks+=1
                    pr=[json.loads(x) for x in (work/('probes_rank%d.jsonl'%rank)).read_text().splitlines()]
                    assert len(pr)==12
                    probes.extend(dict(run=runid,rank=rank,**r) for r in pr)
                sweep=json.loads((work/'ckpt_sweep/sweep_results.json').read_text())
                assert set(sweep['all_checkpoints'])=={'epoch_'+str(e) for e in (16,18,20,22,24)}
                logs=[json.loads(x) for f in work.glob('*.log.json') for x in f.read_text().splitlines()]
                assert {r['epoch'] for r in logs if r.get('mode')=='val'}==set(range(2,25,2))
                for ep,cp in sweep['all_checkpoints'].items():
                    assert sha(cp['checkpoint'])==cp['checkpoint_sha256']
                    rows=frame_rows(cp);cache[seed,reg,obj,ep]=rows
                    for domain in ('real','sim','all'):
                        rs=[r for r in rows if domain=='all' or r['domain']==domain]
                        extra=temporal(rs);m=sizes(rs)
                        row=dict(seed=seed,regression=reg,objective=obj,epoch=ep,domain=domain,
                                 selected=ep==sweep['selected_checkpoint'],**m,
                                 R_center_pct=100*extra['center']/len(rs),localization_hits=extra['hits'],
                                 failure_events=len(extra['events']))
                        # Rounded official metrics are checked independently.
                        if domain!='all':
                            for k,v,tol in [('R_center(%)',row['R_center_pct'],.0051),('mean_RIoU',m['mean_RIoU'],.0002),
                                            ('DFR(%/frame)',m['DFR'],.0002),('ACI',m['ACI'],.0002),
                                            ('MCML_max(frames)',m['MCML_max'],0)]:
                                key=domain+'/'+k
                                if key in cp['metrics']:
                                    assert v is not None and abs(v-cp['metrics'][key])<=tol,(runid,ep,key)
                                    checks+=1
                        summaries.append(row)
                    for domain,seq in sorted({(r['domain'],r['seq']) for r in rows}):
                        rs=[r for r in rows if r['domain']==domain and r['seq']==seq]
                        extra=temporal(rs)
                        video_rows.append(dict(seed=seed,regression=reg,objective=obj,epoch=ep,domain=domain,video=seq,
                            **sizes(rs),R_center_pct=100*extra['center']/len(rs),localization_hits=extra['hits'],failure_events=len(extra['events'])))
                    all_frames.extend(dict(run=runid,epoch=ep,**r) for r in rows)
                for ep in dict.fromkeys([sweep['selected_checkpoint'],'epoch_24']):
                    dest=work/'candidate_diagnostic'/ep
                    receipt=json.loads((dest/'receipt.json').read_text());assert receipt['passed'] and receipt['VAL_exact']==887
                    assert receipt['checkpoint_sha256']==sweep['all_checkpoints'][ep]['checkpoint_sha256']
                    counts=collections.defaultdict(collections.Counter)
                    seen=set()
                    for line in (dest/'candidates_val_summary.jsonl').read_text().splitlines():
                        r=json.loads(line);f=dest/'candidates_val'/(r['frame']+'.npz')
                        assert r['frame'] not in seen;seen.add(r['frame'])
                        assert sha(f)==r['candidate_file_sha256']
                        with np.load(f) as z:
                            for field in ('box','score','iou','center','angle'):assert np.isfinite(z[field]).all()
                            assert (z['iou']>=0).all() and (z['iou']<=1.0001).all()
                            q=z['iou']>=.5;passed=z['score']>.05;s=int(z['selected'])
                            assert np.array_equal(passed,z['passed'])
                            assert s<0 or z['score'][s]==z['score'][passed].max()
                            cat='no_qualified' if not q.any() else 'score_filter_lost' if not (q&passed).any() else 'success' if s>=0 and q[s] else 'ranking_lost'
                            assert cat==r['category']
                        domain=r['frame'].split('_')[0]
                        for d in (domain,'all'):counts[d][cat]+=1
                        checks+=1
                    assert len(seen)==887 and seen=={r['frame'] for r in cache[seed,reg,obj,ep]}
                    for d,c in counts.items():candidate_rows.append(dict(run=runid,epoch=ep,domain=d,frames=sum(c.values()),
                        **{k:c[k] for k in ('no_qualified','score_filter_lost','ranking_lost','success')}))
        assert len(initial)==1,('common initialization mismatch',seed)
    # Selected workflow and fixed ep24 are kept separate; all five same-epoch
    # comparisons are also retained. No new score, TEST, or equivalence test.
    for seed in contract['seeds']:
        for reg in ('iou','kld'):
            for off,on in [('U','S'),('S','G'),('U','G')]:
                for context in ['selected']+['epoch_'+str(e) for e in (16,18,20,22,24)]:
                    epochs={}
                    for obj in (off,on):
                        sw=json.loads((base/('s%d_%s_%s'%(seed,reg,obj.lower()))/'ckpt_sweep/sweep_results.json').read_text())
                        epochs[obj]=sw['selected_checkpoint'] if context=='selected' else context
                    for domain in ('real','sim','all'):
                        a=[r for r in cache[seed,reg,off,epochs[off]] if domain=='all' or r['domain']==domain]
                        b=[r for r in cache[seed,reg,on,epochs[on]] if domain=='all' or r['domain']==domain]
                        aa,bb=temporal(a),temporal(b);common=aa['pairs'].keys()&bb['pairs'].keys()
                        row=dict(seed=seed,regression=reg,off=off,on=on,context=context,domain=domain,
                                 off_epoch=epochs[off],on_epoch=epochs[on],common_pairs=len(common))
                        for k in ('center','hits','outputs','riou','MCML_max','MCML_mean','DFR','ACI','gated_RMSE'):
                            row['delta_'+k]=bb[k]-aa[k] if aa[k] is not None and bb[k] is not None else None
                        row['delta_failure_events']=len(bb['events'])-len(aa['events'])
                        for i,k in enumerate(('DFR','ACI')):
                            row['common_delta_'+k]=statistics.mean(bb['pairs'][p][i]-aa['pairs'][p][i] for p in common) if common else None
                        ga=sum(r['center_export']>=10 for r in a);gb=sum(r['center_export']>=10 for r in b)
                        row['angle_penalty_delta_MSE']=8100*(gb-ga)/len(a)
                        row['angle_pass_delta_MSE']=(sum(r['gated_angle_sq'] for r in b if r['center_export']<10)-sum(r['gated_angle_sq'] for r in a if r['center_export']<10))/len(a)
                        paired.append(row)
    seed_summary=[]
    buckets=collections.defaultdict(list)
    for r in paired:buckets[r['regression'],r['off'],r['on'],r['context'],r['domain']].append(r)
    lower=('MCML_max','MCML_mean','DFR','gated_RMSE','failure_events','common_delta_DFR')
    for key,rs in buckets.items():
        assert len(rs)==len(contract['seeds'])
        for metric in sorted(k for k in rs[0] if k.startswith(('delta_','common_delta_'))):
            values=[r[metric] for r in rs if r[metric] is not None]
            low=(metric[6:] if metric.startswith('delta_') else metric) in lower
            seed_summary.append(dict(regression=key[0],off=key[1],on=key[2],context=key[3],domain=key[4],metric=metric,
                valid_seeds=len(values),mean=statistics.mean(values) if values else None,
                minimum=min(values) if values else None,maximum=max(values) if values else None,
                improved_seeds=sum(v<0 if low else v>0 for v in values),
                interpretation='single seed exploratory' if len(rs)==1 else 'numerical sign only; no equivalence or significance claim'))
    write_csv(out/'metrics_all_epochs.csv',summaries);write_csv(out/'paired_metrics.csv',paired)
    write_csv(out/'paired_seed_summary.csv',seed_summary)
    write_csv(out/'metrics_by_video.csv',video_rows)
    write_csv(out/'candidate_failures.csv',candidate_rows);write_csv(out/'per_frame.csv',all_frames)
    with (out/'probes.json').open('x') as f:json.dump(probes,f,allow_nan=False)
    n=6*len(contract['seeds'])
    (out/'complete.json').write_text(json.dumps(dict(passed=True,runs=n,online_VAL=n*12,candidate_VAL=n*5,
        verification_checks=checks,TEST=0,depth=0,scope='single seed exploratory differences; no automatic adoption or equivalence claim'),indent=2))
    (out/'provenance.json').write_text(json.dumps(dict(protocol_sha256=sha(base/'protocol.json'),
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['candidates','aggregate']);p.add_argument('--base',type=Path);p.add_argument('--work-dir',type=Path);p.add_argument('--config',type=Path)
    a=p.parse_args()
    if a.stage=='candidates':candidates(a.work_dir.resolve(),a.config.resolve())
    else:aggregate(a.base.resolve())
