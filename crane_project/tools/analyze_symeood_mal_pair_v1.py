"""Read-only CPU analysis; original VAL selectors, exposed TEST descriptive."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'crane_project/tools'))
from analyze_symeood_c_pair_v1 import frames,write_csv,sha
from analyze_symeood_ablation_first4_v1 import aggregate
from audit_symeood_c_existing_results_v1 import aggregate as temporal,compare
from audit_symeood_cross_entry_performance_v1 import selector


def analyze(base,require_weights=False):
    import numpy as np
    c=json.loads((base/'protocol.json').read_text())
    out=base/'analysis_v1';out.mkdir(exist_ok=False)
    cache={};allrows=[];metrics=[];videos=[];selected={};verification=[]
    for group in c['groups']:
        work=base/group;sweep=work/'ckpt_sweep/sweep_results.json'
        selected[group]=selector(sweep)['selected']
        d=json.loads(sweep.read_text());assert d['selection_config']==c['selection']
        reports=list((work/'ckpt_sweep/final_test').glob('*/final_test_metrics_v2.json'));assert len(reports)==1
        test=json.loads(reports[0].read_text());assert Path(test['checkpoint']).stem==selected[group]
        cases=[('val',ep,cp) for ep,cp in d['all_checkpoints'].items()]+[('test',selected[group],test)]
        for split,ep,cp in cases:
            cp=dict(cp)
            # Relocate server artifacts after verified download, without editing
            # the archived original result JSON or checkpoint identities.
            old=Path(cp['results_pkl'])
            server_base=Path(c['configurations'][group]).parents[1]
            cp['results_pkl']=str(base/old.relative_to(server_base))
            if require_weights:assert sha(cp['checkpoint'])==cp['checkpoint_sha256']
            rr=frames(cp,split,group,ep);cache[group,split,ep]=rr;allrows+=rr
            for domain in ('real','sim','all'):
                ss=[r for r in rr if domain=='all' or r['domain']==domain]
                m=aggregate(ss);t=temporal(ss)
                row=dict(group=group,split=split,epoch=ep,selected=ep==selected[group],domain=domain,
                    center=t['center'],localization_hits=t['hits'],failure_events=len(t['events']),**m)
                metrics.append(row)
                for k,other in [('DFR','DFR'),('ACI','ACI'),('mean_RIoU','riou'),('MCML_max','MCML_max')]:
                    assert math.isclose(m[k],t[other],abs_tol=1e-10)
                if domain!='all':
                    expected={'R_center(%)':100*t['center']/len(ss),'mean_RIoU':m['mean_RIoU'],
                        'DFR(%/frame)':m['DFR'],'ACI':m['ACI'],'MCML_max(frames)':m['MCML_max']}
                    for key,value in expected.items():
                        delta=abs(value-cp['metrics'][domain+'/'+key])
                        assert delta<=.0051,(group,split,ep,domain,key,delta)
                        verification.append(dict(group=group,split=split,epoch=ep,domain=domain,key=key,absdiff=delta))
            for domain,seq in sorted({(r['domain'],r['seq']) for r in rr}):
                ss=[r for r in rr if (r['domain'],r['seq'])==(domain,seq)];t=temporal(ss)
                videos.append(dict(group=group,split=split,epoch=ep,domain=domain,seq=seq,
                    center=t['center'],localization_hits=t['hits'],failure_events=len(t['events']),**aggregate(ss)))
    comparisons=[]
    for context,split,ep in [('selected_VAL','val',None),('selected_TEST','test',None)]+[
            ('VAL_ep%d'%e,'val','epoch_%d'%e) for e in c['candidate_VAL_epochs']]:
        a,b=(cache[g,split,ep or selected[g]] for g in c['groups'])
        for domain in ('real','sim','all'):
            comparisons.append(dict(context=context,domain=domain,
                **compare([r for r in a if domain=='all' or r['domain']==domain],
                          [r for r in b if domain=='all' or r['domain']==domain])))
    strengths={}
    for group in c['groups']:
        strengths[group]={}
        for rank in (0,1):
            rr=[json.loads(s) for s in (base/group/('alpha_rank%d.jsonl'%rank)).read_text().splitlines()]
            assert len(rr)==15360 and len({r['iteration'] for r in rr})==15360
            vals=np.array([r['alpha'] for r in rr])
            strengths[group][str(rank)]=dict(count=len(rr),minimum=float(vals.min()),maximum=float(vals.max()),
                quantiles=np.quantile(vals,[0,.01,.25,.5,.75,.99,1]).tolist(),
                degenerate=sum(r['degenerate'] for r in rr),
                positive_M=sum(r['Mpos'] for r in rr),negative_M=sum(r['Mneg'] for r in rr))
    # Latest fixed pair110, selectedep20; never mix it into contemporary arms.
    hist=ROOT/'work_dirs/symeood_c_pair_seed0_20261025_v2/analysis_v2/all_candidate_VAL_and_selected_TEST.csv'
    historical=[r for r in csv.DictReader(hist.open()) if r['group']=='110']
    write_csv(out/'historical_110_cross_batch.csv',historical)
    write_csv(out/'all_candidate_VAL_and_selected_TEST.csv',metrics)
    write_csv(out/'per_video.csv',videos);write_csv(out/'per_frame.csv',allrows)
    (out/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2))
    (out/'strength_summary.json').write_text(json.dumps(strengths,indent=2))
    (out/'verification.json').write_text(json.dumps(dict(passed=True,checks=verification,
        selected=selected,local_weights_verified=require_weights,polygon_source='existing project geometry implementation'),indent=2))
    (out/'complete.json').write_text(json.dumps(dict(passed=True,groups=c['groups'],selected=selected,
        frames=len(allrows),model_epochs=48,historical_control='cross batch only',automatic_adoption=False),indent=2))
    print('MAL_PAIR_ANALYSIS_COMPLETE',selected,len(allrows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--require-weights',action='store_true')
    a=p.parse_args();analyze(a.base.resolve(),a.require_weights)
