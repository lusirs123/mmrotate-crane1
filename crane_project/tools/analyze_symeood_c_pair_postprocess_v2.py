"""CPU aggregation of the two fixed runs. Never selects using TEST.

Original VAL selected weights are main; fixedep24/sameepoch curves auxiliary.
No model inference, USG counterfactuals, proposals or automatic adoption.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'crane_project/tools'))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_csv(path,rr):
    if not rr:return
    with path.open('x') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rr))));w.writeheader();w.writerows(rr)


def frames(cp,split,group,epoch):
    import numpy as np
    from analyze_symeood_ablation_first4_v1 import txtbox,iou,angle
    pred=Path(cp['results_pkl']);assert sha(pred)==cp['results_pkl_sha256']
    raw=pickle.loads(pred.read_bytes());gt=ROOT/'crane_project/data/crane_grab_port_day2night_v1'/split/'annfiles'
    files=sorted(p for p in gt.glob('*.txt') if not p.name.startswith('._'));assert len(raw)==len(files)==(887 if split=='val' else 1440)
    result=[]
    for p,output in zip(files,raw):
        box=output[0];assert len(box)<=1
        actual=None if not len(box) else np.asarray(box[0,:5],float)
        g=txtbox(p);export=txtbox(pred.parent/'Task1_grab'/p.name);assert g is not None and (actual is None)==(export is None)
        domain,seq,fid=re.fullmatch(r'(real|sim)_(.+)_(\d+)',p.stem).groups()
        center=float(np.linalg.norm(export[:2]-g[:2])) if export is not None else float('inf')
        r=dict(group=group,split=split,epoch=epoch,frame=p.stem,domain=domain,seq=seq,fid=int(fid),output=actual is not None,
            score=float(box[0,-1]) if len(box) else 0.,center_export=center,riou_export=iou(export,g),riou_native=iou(actual,g),
            gated_angle_sq=(math.degrees(angle(export[4],g[4])) if center<10 else 90.)**2)
        if actual is not None:
            lp,sp=sorted(actual[2:4],reverse=True);lg,sg=g[2:4];el,es=math.log(lp/lg),math.log(sp/sg)
            r.update(diag_export=float(np.hypot(*export[2:4])),angle_export=float(export[4]),
                long_error=abs(lp-lg)/lg,short_error=abs(sp-sg)/sg,long_bias=(lp-lg)/lg,short_bias=(sp-sg)/sg,
                center_error=float(np.linalg.norm(actual[:2]-g[:2])),angle_error_deg=math.degrees(angle(actual[4]+(math.pi/2 if actual[2]<actual[3] else 0),g[4])),
                angle_eligible=lg/sg>=1.2,a=(el+es)/2,r=el-es)
        result.append(r)
    return result


def analyze(base,out=None,historical_path=None):
    from analyze_symeood_ablation_first4_v1 import aggregate as sizes
    from audit_symeood_c_existing_results_v1 import aggregate as temporal,compare
    from audit_symeood_cross_entry_performance_v1 import selector
    out=out or base/'analysis_v1'
    historical_path=historical_path or ROOT/'work_dirs/symeood_ablation_v1/analysis_all8_20261024/analysis_all8.json'
    # Validate the external reference before creating a new output directory.
    # A postprocessing retry must preserve the failed directory and all runs.
    historical=json.loads(historical_path.read_text())['groups']
    assert all(g in historical for g in ('110','111'))
    out.mkdir(exist_ok=False)
    contract=json.loads((base/'protocol.json').read_text());assert contract['state']=='FROZEN'
    allframes=[];metrics=[];byvideo=[];cache={};selections={};verification=[]
    for group in ('110','111'):
        work=base/group;sweep=work/'ckpt_sweep/sweep_results.json';selections[group]=selector(sweep)['selected']
        data=json.loads(sweep.read_text());assert data['selection_config']==contract['selection']
        reports=list((work/'ckpt_sweep/final_test').glob('*/final_test_metrics_v2.json'));assert len(reports)==1
        test=json.loads(reports[0].read_text());assert Path(test['checkpoint']).stem==selections[group]
        cases=[('val',ep,cp) for ep,cp in data['all_checkpoints'].items()]+[('test',selections[group],test)]
        for split,ep,cp in cases:
            assert sha(cp['checkpoint'])==cp['checkpoint_sha256']
            rr=frames(cp,split,group,ep);cache[group,split,ep]=rr;allframes.extend(rr)
            for domain in ('real','sim','all'):
                ss=[r for r in rr if domain=='all' or r['domain']==domain];m=sizes(ss);t=temporal(ss)
                for k,other in [('DFR','DFR'),('ACI','ACI'),('mean_RIoU','riou'),('MCML_max','MCML_max'),('MCML_mean','MCML_mean')]:
                    assert math.isclose(m[k],t[other],abs_tol=1e-10),(group,split,ep,domain,k)
                row=dict(group=group,split=split,epoch=ep,selected=ep==selections[group],domain=domain,
                    center=t['center'],localization_hits=t['hits'],failure_events=len(t['events']),**m)
                metrics.append(row)
                if domain!='all':
                    formal={'R_center(%)':100*t['center']/len(ss),'mean_RIoU':m['mean_RIoU'],'DFR(%/frame)':m['DFR'],'ACI':m['ACI'],
                        'MCML_max(frames)':m['MCML_max'],'MCML_mean(frames)':m['MCML_mean'],'TDR_w10(%)':m['TDR_pct']}
                    if domain=='sim':formal['A-RMSE(deg)']=m['gated_angle_RMSE_deg']
                    for key,value in formal.items():
                        delta=abs(value-cp['metrics'][domain+'/'+key]);tol=.0051 if key in ('R_center(%)','MCML_mean(frames)','TDR_w10(%)') else .0002
                        verification.append(dict(group=group,split=split,epoch=ep,key=domain+'/'+key,absdiff=delta,tolerance=tol,passed=delta<=tol))
            for domain,seq in sorted({(r['domain'],r['seq']) for r in rr}):
                ss=[r for r in rr if (r['domain'],r['seq'])==(domain,seq)];t=temporal(ss)
                byvideo.append(dict(group=group,split=split,epoch=ep,domain=domain,seq=seq,center=t['center'],hits=t['hits'],failure_events=len(t['events']),**sizes(ss)))
    contexts=[('selected_VAL','val',None),('selected_TEST','test',None)]+[('VAL_ep'+str(e),'val','epoch_'+str(e)) for e in (16,18,20,22,24)]
    comparisons=[]
    for context,split,ep in contexts:
        a,b=(cache[g,split,ep or selections[g]] for g in ('110','111'))
        for domain in ('real','sim','all'):
            aa=[r for r in a if domain=='all' or r['domain']==domain];bb=[r for r in b if domain=='all' or r['domain']==domain]
            comparisons.append(dict(context=context,domain=domain,**compare(aa,bb)))
        for domain,seq in sorted({(r['domain'],r['seq']) for r in a}):
            comparisons.append(dict(context=context,domain=domain,seq=seq,**compare([r for r in a if (r['domain'],r['seq'])==(domain,seq)],[r for r in b if (r['domain'],r['seq'])==(domain,seq)])))
    assert all(r['passed'] for r in verification)
    # Preserve historical selected results as a separate provenance block,
    # never substitute them for new paired controls.
    old={g:dict(selected=historical[g]['selected'],selected_VAL=historical[g][historical[g]['selected']],TEST=historical[g]['test']) for g in ('110','111')}
    (out/'historical_reference.json').write_text(json.dumps(old,indent=2))
    write_csv(out/'all_candidate_VAL_and_selected_TEST.csv',metrics);write_csv(out/'per_video.csv',byvideo);write_csv(out/'per_frame.csv',allframes)
    (out/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2))
    (out/'verification.json').write_text(json.dumps(dict(passed=True,checks=verification,selection_reproduced=selections,
        source_sha256=sha(Path(__file__)),historical_reference=dict(path=str(historical_path),sha256=sha(historical_path)),
        adoption='No automated replacement or scoring; single seed rerun must be interpreted with prior batches',
        outputs_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2))
    (out/'complete.json').write_text(json.dumps(dict(passed=True,groups=['110','111'],selected=selections,frames=len(allframes),formal_runs=2,automatic_adoption=False,depth=False),indent=2))
    print('PAIR_ANALYSIS_COMPLETE',selections,len(allframes),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--out',type=Path);p.add_argument('--historical-reference',type=Path)
    a=p.parse_args();analyze(a.base.resolve(),a.out.resolve() if a.out else None,
                           a.historical_reference.resolve() if a.historical_reference else None)
