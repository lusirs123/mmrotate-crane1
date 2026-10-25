"""CPU-only audit of two frozen existing TEST prediction streams.

No training/inference/selection. Uses the same native offline evaluator for
both streams and independently attributes DFR to common/exclusive pairs.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'crane_project/tools'))
sys.path.insert(0,str(ROOT))
import numpy as np
from eval_crane_offline import CraneOfflineEvaluator,parse_dota_txt,parse_seq_frame,angle_diff
from ckpt_sweep import annotation_set_sha256,pkl_to_dota,dota_export_sha256


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_csv(p,rr):
    with p.open('x') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rr))));w.writeheader();w.writerows(rr)


def pair_map(records):
    result={};by_video=collections.defaultdict(list)
    for r in records:by_video[r['domain'],r['seq_id']].append(r)
    for (domain,seq),rr in by_video.items():
        rr.sort(key=lambda r:r['frame_id'])
        for a,b in zip(rr,rr[1:]):
            if b['frame_id']!=a['frame_id']+1 or any(r['pred_box'] is None or r['gt_box'] is None for r in (a,b)):continue
            da,db=(float(np.hypot(*r['pred_box'][2:4])) for r in (a,b))
            ga,gb=(float(np.hypot(*r['gt_box'][2:4])) for r in (a,b))
            result[domain,seq,a['frame_id'],b['frame_id']]=dict(dfr=100*abs(db-da)/da,
                signed_change_pct=100*(db-da)/da,previous_diag=da,current_diag=db,absolute_change=abs(db-da),
                GT_signed_change_pct=100*(gb-ga)/ga,
                ACI=max(0.,1-abs(float(angle_diff(np.array([b['pred_box'][4]]),np.array([a['pred_box'][4]]))[0]))/(math.radians(35)+1e-9)),
                previous_size=a['pred_box'][2:4].tolist(),current_size=b['pred_box'][2:4].tolist())
    return result


def mean(values):return float(np.mean(list(values))) if len(values) else None


def decompose(a,b):
    c=sorted(set(a)&set(b));ao=sorted(set(a)-set(b));bo=sorted(set(b)-set(a));assert c
    old,new=mean([v['dfr'] for v in a.values()]),mean([v['dfr'] for v in b.values()])
    ca,cb=mean([a[k]['dfr'] for k in c]),mean([b[k]['dfr'] for k in c])
    direct=cb-ca;scope=(new-cb)-(old-ca)
    assert abs((new-old)-direct-scope)<1e-12
    return dict(B_pairs=len(a),A111_pairs=len(b),common_pairs=len(c),B_only_pairs=len(ao),A111_only_pairs=len(bo),
        B_DFR=old,A111_DFR=new,delta=new-old,B_common_DFR=ca,A111_common_DFR=cb,
        common_prediction_effect=direct,pair_scope_effect=scope,B_only_DFR=mean([a[k]['dfr'] for k in ao]),A111_only_DFR=mean([b[k]['dfr'] for k in bo]),
        B_common_ACI=mean([a[k]['ACI'] for k in c]),A111_common_ACI=mean([b[k]['ACI'] for k in c]))


def performance(out):
    gt=ROOT/'crane_project/data/crane_grab_port_day2night_v1/test/annfiles';images=gt.parent/'images'
    files=sorted(p for p in gt.glob('*.txt') if not p.name.startswith('._'));ids=[p.stem for p in files]
    assert len(files)==1440
    identity=[];all_records={};all_native={};metrics={};verification=[];frame_rows=[]
    models=[('B',ROOT/'work_dirs/crane_symeood_k1_port_day2night_aug_b_v1','val_sweep_port_v1',24),
            ('111',ROOT/'work_dirs/symeood_ablation_v1/111','ckpt_sweep',18)]
    for label,root,sweep,epoch in models:
        final=root/sweep/'final_test'/('epoch_'+str(epoch));pred=final/'preds/results.pkl'
        provenance=json.loads(Path(str(pred)+'.provenance.json').read_text());official=json.loads((final/'final_test_metrics_v2.json').read_text())
        checkpoint=Path(provenance['checkpoint']);config=Path(provenance['config'])
        assert sha(checkpoint)==provenance['checkpoint_sha256']==official['checkpoint_sha256']
        assert sha(pred)==provenance['results_pkl_sha256']==official['results_pkl_sha256']
        assert sha(config)==provenance['config_sha256']==official['config_sha256']
        assert annotation_set_sha256(str(gt))==provenance['annotations_sha256']==official['gt_annotations_sha256']
        task=final/'preds/Task1_grab';export=json.loads(Path(str(task)+'.provenance.json').read_text())
        assert export['frame_ids']==ids and export['results_pkl_sha256']==sha(pred) and export['dota_sha256']==dota_export_sha256(str(task),ids)
        fresh=Path(pkl_to_dota(str(pred),ids,str(out/('reexport_'+label))))
        assert all((task/p.name).read_bytes()==(fresh/p.name).read_bytes() for p in files)
        predictions=pickle.loads(pred.read_bytes());assert len(predictions)==len(files)
        records=[];native=[]
        for p,box in zip(files,predictions):
            assert len(box)==1 and len(box[0])<=1
            g=parse_dota_txt(str(p));e=parse_dota_txt(str(task/p.name));assert len(g)==1 and len(e)<=1
            domain,seq,fid=parse_seq_frame(p.name);raw=None if not len(box[0]) else np.array(box[0][0,:5],float)
            assert (raw is None)==(not e)
            if raw is not None:assert np.isfinite(raw).all() and (raw[2:4]>0).all()
            row=dict(domain=domain,seq_id=seq,frame_id=fid,gt_box=g[0],pred_box=e[0] if e else None)
            records.append(row);native.append(dict(row,pred_box=raw))
            if raw is not None:
                # PKL is already in original-image coordinates. Compare to
                # archived exported polygon reconstruction, never rescale twice.
                ds=sorted(raw[2:4],reverse=True);es=sorted(e[0][2:4],reverse=True)
                center=float(np.linalg.norm(raw[:2]-e[0][:2]));size=max(abs(x-y) for x,y in zip(ds,es))
            else:center=size=0.
            frame_rows.append(dict(version=label,frame=p.stem,domain=domain,seq=seq,fid=fid,output=raw is not None,
                pkl_export_center_diff_px=center,pkl_export_size_diff_px=size))
        ev=CraneOfflineEvaluator(mode='test',center_thresh_px=15);m=ev.evaluate_records(records)
        mn=CraneOfflineEvaluator(mode='test',center_thresh_px=15).evaluate_records(native)
        pervideo={}
        for video in sorted(set((r['domain'],r['seq_id']) for r in records)):
            pervideo['_'.join(video)]=CraneOfflineEvaluator(mode='test',center_thresh_px=15).evaluate_records([r for r in records if (r['domain'],r['seq_id'])==video])
        for k,value in official['metrics'].items():
            delta=abs(m[k]-value);verification.append(dict(version=label,metric=k,reported=value,recomputed=m[k],absdiff=delta,passed=delta==0))
        assert all(r['passed'] for r in verification)
        all_records[label]=records;all_native[label]=native;metrics[label]=dict(export=m,native_PKLOBB=mn,per_video=pervideo)
        identity.append(dict(version=label,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),prediction=str(pred),prediction_sha256=sha(pred),
            config=str(config),config_sha256=sha(config),annotations_sha256=provenance['annotations_sha256'],export_sha256=export['dota_sha256'],
            frame_ids=ids,reexport1440_files_exact=True,original_metrics_sha256=sha(final/'final_test_metrics_v2.json')))
    pairs={label:pair_map(rr) for label,rr in all_records.items()};native_pairs={label:pair_map(rr) for label,rr in all_native.items()}
    decomp=[];native_decomp=[];pair_rows=[];windows=collections.defaultdict(list)
    videos=sorted(set(k[:2] for k in pairs['B'])|set(k[:2] for k in pairs['111']))
    for domain,seq in [('all','all'),('real','all'),('sim','all')]+videos:
        def filtered(m):return {k:v for k,v in m.items() if (domain=='all' or k[0]==domain) and (seq=='all' or k[1]==seq)}
        a,b=filtered(pairs['B']),filtered(pairs['111']);decomp.append(dict(domain=domain,seq=seq,**decompose(a,b)))
        na,nb=filtered(native_pairs['B']),filtered(native_pairs['111']);native_decomp.append(dict(domain=domain,seq=seq,**decompose(na,nb)))
    for k in sorted(set(pairs['B'])|set(pairs['111'])):
        a,b=pairs['B'].get(k),pairs['111'].get(k)
        row=dict(domain=k[0],seq=k[1],previous_fid=k[2],current_fid=k[3],B_valid=a is not None,A111_valid=b is not None,
                 B_DFR=a['dfr'] if a else None,A111_DFR=b['dfr'] if b else None)
        if a and b:
            delta=b['dfr']-a['dfr'];row['delta_pp']=delta
            row.update(B_diag_previous=a['previous_diag'],B_diag_current=a['current_diag'],A111_diag_previous=b['previous_diag'],A111_diag_current=b['current_diag'],
                B_previous_size=a['previous_size'],B_current_size=a['current_size'],A111_previous_size=b['previous_size'],A111_current_size=b['current_size'],
                GT_signed_diag_change_pct=a['GT_signed_change_pct'],
                change_magnitude_component=100*(b['absolute_change']-a['absolute_change'])/a['previous_diag'],
                previous_diag_denominator_component=100*b['absolute_change']*(1/b['previous_diag']-1/a['previous_diag']))
            assert abs(delta-row['change_magnitude_component']-row['previous_diag_denominator_component'])<1e-10
            windows[k[0],k[1],k[3]//20].append(row)
        pair_rows.append(row)
    ww=[]
    for (domain,seq,slot),rr in windows.items():
        n=next(x['common_pairs'] for x in decomp if x['domain']==domain and x['seq']=='all')
        ww.append(dict(domain=domain,seq=seq,block_start_fid=slot*20,block_end_fid=slot*20+19,actual_start_previous=min(x['previous_fid'] for x in rr),
            actual_end_current=max(x['current_fid'] for x in rr),common_pairs=len(rr),B_DFR=mean([x['B_DFR'] for x in rr]),A111_DFR=mean([x['A111_DFR'] for x in rr]),
            sum_delta_pp=sum(x['delta_pp'] for x in rr),contribution_to_domain_common_DFR_pp=sum(x['delta_pp'] for x in rr)/n))
    image_rows=[]
    for fid in ids:
        matches=list(images.glob(fid+'.*'));assert len(matches)==1
        image_rows.append(dict(frame=fid,image=str(matches[0]),sha256=sha(matches[0]),bytes=matches[0].stat().st_size))
    missing={name:[p.stem for p,r in zip(files,rr) if r['pred_box'] is None] for name,rr in all_records.items()}
    (out/'prediction_identity.json').write_text(json.dumps(identity,indent=2));(out/'unified_metrics.json').write_text(json.dumps(metrics,indent=2))
    (out/'missing_frame_ids.json').write_text(json.dumps(missing,indent=2))
    write_csv(out/'test_image_inventory.csv',image_rows);write_csv(out/'per_frame_identity.csv',frame_rows)
    write_csv(out/'dfr_decomposition.csv',decomp);write_csv(out/'dfr_native_pkl_decomposition.csv',native_decomp)
    write_csv(out/'dfr_pair_details.csv',pair_rows);write_csv(out/'dfr_20frame_blocks.csv',sorted(ww,key=lambda r:-r['contribution_to_domain_common_DFR_pp']))
    (out/'metric_verification.json').write_text(json.dumps(dict(passed=True,comparisons=verification,
        evaluator_sha256=sha(ROOT/'crane_project/tools/eval_crane_offline.py'),image_identity_limit='Current1440 image hashes recorded. Sidecars bind historical annotation/ordered frame IDs and PKL/weights, not historical image byte hashes.'),indent=2))
    print('DECOMPOSITION',decomp,flush=True)


def diff(a,b,path=''):
    result=[]
    if isinstance(a,dict) and isinstance(b,dict):
        for key in sorted(set(a)|set(b)):
            if key not in a or key not in b:result.append(dict(path=path+'.'+key,B=a.get(key,'MISSING'),A111=b.get(key,'MISSING')))
            else:result.extend(diff(a[key],b[key],path+'.'+key))
    elif a!=b:result.append(dict(path=path,B=a,A111=b))
    return result


def training(out):
    import torch
    from mmcv import Config
    roots=[ROOT/'work_dirs/crane_symeood_k1_port_day2night_aug_b_v1',ROOT/'work_dirs/symeood_ablation_v1/111'];records=[];states=[]
    for root,selected in zip(roots,(24,18)):
        paths=list(root.glob('*.log.json'));assert len(paths)==1
        rr=[json.loads(s) for s in paths[0].read_text().splitlines()];meta=rr[0];train=[r for r in rr if r.get('mode')=='train']
        selected_cp=torch.load(root/('epoch_'+str(selected)+'.pth'),map_location='cpu');final_cp=torch.load(root/'epoch_24.pth',map_location='cpu')
        cfg=Config.fromstring(meta['config'],file_format='.py').to_dict()
        dumped=Config.fromfile(str(next(root.glob('*.py')))).to_dict()
        records.append(dict(root=str(root),log_sha256=sha(paths[0]),log_meta={k:v for k,v in meta.items() if k!='config'},actual_logged_config=cfg,
            log_vs_saved_config_diff=diff(cfg,dumped),selected_epoch=selected,selected_iteration=selected_cp['meta']['iter'],
            selected_nfl_calls=int(selected_cp['state_dict']['bbox_head.loss_cls._local_iter']),final_iteration=final_cp['meta']['iter'],
            final_nfl_calls=int(final_cp['state_dict']['bbox_head.loss_cls._local_iter']),
            lr_trace=[(r['epoch'],r['iter'],r['lr']) for r in train],train_rows=len(train),loss_keys=sorted(k for k in train[0] if 'loss' in k),
            final_parameter_groups=final_cp['optimizer']['param_groups']))
        states.append(final_cp['state_dict']);del selected_cp,final_cp
    a,b=records;sa,sb=states
    frozen=[k for k in sa if k.startswith(('backbone.conv1.','backbone.bn1.','backbone.layer1.'))]
    source_paths=['mmrotate/models/dense_heads/sym_eood_head.py','mmrotate/models/detectors/sym_eood_detector.py','mmrotate/models/losses/sym_nfl_loss.py','mmrotate/models/losses/sym_kld_loss.py','mmrotate/core/bbox/assigners/sym_pola.py','mmrotate/models/losses/sym_kld_calculator.py','mmrotate/models/dense_heads/rotated_atss_head.py','mmrotate/models/detectors/single_stage.py','mmrotate/datasets/crane_custom_dota.py','mmrotate/datasets/pipelines/port_train_augment.py','tools/train.py','tools/dist_train.sh']
    source=[]
    for p in source_paths:
        old=subprocess.check_output(['git','show','af8307c:'+p],cwd=ROOT)
        (out/'historical_sources'/p).parent.mkdir(exist_ok=True,parents=True);(out/'historical_sources'/p).write_bytes(old)
        source.append(dict(path=p,historical_sha256=hashlib.sha256(old).hexdigest(),current_sha256=sha(ROOT/p),bytes_equal=old==(ROOT/p).read_bytes()))
    result=dict(records=records,logged_config_diff=diff(a['actual_logged_config'],b['actual_logged_config']),
        lr_trace_equal=a['lr_trace']==b['lr_trace'],state_key_shapes_equal=set(sa)==set(sb) and all(sa[k].shape==sb[k].shape for k in sa),
        frozen_stem_layer1_keys=len(frozen),frozen_stem_layer1_equal=all(torch.equal(sa[k],sb[k]) for k in frozen),sources=source,
        evidence_limits=['Oldcomplete initialization, pretrained whole-file SHA and augmented input/RNG stream not saved.','Historical commit represents logged version; past uncommitted imported byte state was not separately frozen.','Different original selected epochs retained; completed24epoch budget audited separately.'])
    (out/'training_identity.json').write_text(json.dumps(result,indent=2,default=str));print('TRAINING_DIFF',result['logged_config_diff'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    performance(out);training(out)
    (out/'complete.json').write_text(json.dumps(dict(training=0,inference=0,TEST='existing predictions audited only',depth=False,source_sha256=sha(Path(__file__))),indent=2))
