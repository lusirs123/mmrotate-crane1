"""CPU verification of the completed MAL arm's predeclared five VAL epochs.

The S_MAL study is incomplete. Historical110 is cross-batch reference only.
No TEST, inference, training, checkpoint loading or new selector is performed.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'crane_project/tools'))
from analyze_symeood_c_pair_v1 import frames,write_csv,sha
from analyze_symeood_ablation_first4_v1 import aggregate as geometry
from audit_symeood_c_existing_results_v1 import aggregate as temporal,compare
from audit_symeood_cross_entry_performance_v1 import selector


def read_historical(path):
    text={'frame','domain','seq','epoch','group','split'}
    flags={'output','angle_eligible'}
    result=[]
    for row in csv.DictReader(path.open()):
        if row['group']!='110' or row['split']!='val':continue
        r={}
        for k,v in row.items():
            if k in text:r[k]=v
            elif k in flags:r[k]=v=='True'
            elif v:r[k]=int(v) if k=='fid' else float(v)
        result.append(r)
    return result


def analyze(base,out):
    out.mkdir(parents=True,exist_ok=False)
    c=json.loads((base/'protocol.json').read_text())
    sweep=base/'MAL/ckpt_sweep/sweep_results.json'
    d=json.loads(sweep.read_text());selection=selector(sweep)
    assert d['selection_config']==c['selection']
    assert len(d['all_checkpoints'])==5
    assert set(d['all_checkpoints'])=={'epoch_%d'%i for i in (16,18,20,22,24)}
    # Only the original MAL selection is reproduced; no selection for S_MAL.
    val_gt=[r for r in c['data_inventory'] if '/val/' in r['annotation']]
    assert len(val_gt)==887
    for r in val_gt:assert sha(ROOT/r['annotation'])==r['annotation_sha256']
    metrics=[];videos=[];allframes=[];cache={};checks=[]
    server_base=Path(c['configurations']['MAL']).parents[1]
    for ep,cp in d['all_checkpoints'].items():
        cp=dict(cp);cp['results_pkl']=str(base/Path(cp['results_pkl']).relative_to(server_base))
        rr=frames(cp,'val','MAL',ep);cache[ep]=rr;allframes+=rr
        for domain in ('real','sim','all'):
            ss=[r for r in rr if domain=='all' or r['domain']==domain]
            t=temporal(ss);m=geometry(ss)
            metrics.append(dict(group='MAL',epoch=ep,domain=domain,
                selected=ep==selection['selected'],center=t['center'],localization_hits=t['hits'],
                failure_events=len(t['events']),**m))
            for k,k2 in [('DFR','DFR'),('ACI','ACI'),('mean_RIoU','riou'),('MCML_max','MCML_max')]:
                assert math.isclose(m[k],t[k2],abs_tol=1e-10)
            if domain!='all':
                expected={'R_center(%)':100*t['center']/len(ss),'mean_RIoU':m['mean_RIoU'],
                    'DFR(%/frame)':m['DFR'],'ACI':m['ACI'],'MCML_max(frames)':m['MCML_max'],
                    'MCML_mean(frames)':m['MCML_mean'],'TDR_w10(%)':m['TDR_pct']}
                if domain=='sim':expected['A-RMSE(deg)']=m['gated_angle_RMSE_deg']
                for k,v in expected.items():
                    delta=abs(v-cp['metrics'][domain+'/'+k])
                    tol=.0051 if k in ('R_center(%)','MCML_mean(frames)','TDR_w10(%)') else .0002
                    assert delta<=tol,(ep,domain,k,delta)
                    checks.append(dict(epoch=ep,domain=domain,metric=k,absdiff=delta,tolerance=tol))
        for domain,seq in sorted({(r['domain'],r['seq']) for r in rr}):
            ss=[r for r in rr if (r['domain'],r['seq'])==(domain,seq)];t=temporal(ss)
            videos.append(dict(epoch=ep,domain=domain,seq=seq,center=t['center'],
                localization_hits=t['hits'],failure_events=len(t['events']),**geometry(ss)))
    old=read_historical(ROOT/'work_dirs/symeood_c_pair_seed0_20261025_v2/analysis_v2/per_frame.csv')
    old_by_epoch={ep:[r for r in old if r['epoch']==ep] for ep in cache}
    differences=[]
    for context,old_ep,new_ep in [('selected','epoch_20',selection['selected'])]+[
            ('VAL_ep%d'%i,'epoch_%d'%i,'epoch_%d'%i) for i in (16,18,20,22,24)]:
        for domain in ('real','sim','all'):
            a=[r for r in old_by_epoch[old_ep] if domain=='all' or r['domain']==domain]
            b=[r for r in cache[new_ep] if domain=='all' or r['domain']==domain]
            differences.append(dict(context=context,domain=domain,old_epoch=old_ep,new_epoch=new_ep,
                inference_scope='cross-batch historical comparison only',**compare(a,b)))
    write_csv(out/'all_candidate_VAL.csv',metrics)
    write_csv(out/'per_video_VAL.csv',videos);write_csv(out/'per_frame_VAL.csv',allframes)
    (out/'historical110_comparisons.json').write_text(json.dumps(differences,indent=2))
    (out/'selection_verification.json').write_text(json.dumps(selection,indent=2))
    (out/'verification.json').write_text(json.dumps(dict(passed=True,checks=checks,
        VAL_gt_sha_verified=887,frames=len(allframes),
        geometry_limit='existing polygon/evaluator implementation reused; independent local aggregation, not independent polygon kernel',
        checkpoint_bytes='server manifest only; weights retained on server'),indent=2))
    (out/'complete.json').write_text(json.dumps(dict(audit_passed=True,MAL_VAL_complete=True,
        experiment_pair_complete=False,S_MAL_complete=False,TEST=False,
        selected=selection['selected'],frames=len(allframes),automatic_adoption=False),indent=2))
    print('MAL_VAL_ONLY_VERIFIED',selection['selected'],len(allframes),len(checks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.base.resolve(),a.out.resolve())
