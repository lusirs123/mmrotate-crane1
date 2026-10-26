"""Frozen old/new110 VAL comparison, then optional selected TEST comparison.

Missing old detector weights are not loaded. Uses retained prediction PKL/TXT
identities and the existing independent metric/selector implementations.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'crane_project/tools'))
from run_symeood_110_retrain_v1 import OLD,sha,read,save


def analyze(base,include_test=False):
    from analyze_symeood_c_pair_v1 import frames,write_csv
    from analyze_symeood_ablation_first4_v1 import aggregate as geometry
    from audit_symeood_c_existing_results_v1 import aggregate,compare
    from audit_symeood_cross_entry_performance_v1 import selector
    from audit_symeood_c_pair_completion_v1 import rng_identity
    c=read(base/'protocol.json')
    for path,h in c['old_prediction_artifacts'].items():assert sha(path)==h,path
    out=base/('analysis_VAL_TEST_v1' if include_test else 'analysis_VAL_v1');out.mkdir(exist_ok=False)
    caches={};metrics=[];video=[];perframe=[];checks=[];selections={}
    for side,parent in [('old',OLD),('new',base)]:
        sweep=parent/'110/ckpt_sweep/sweep_results.json';s=read(sweep)
        assert s['selection_config']==c['selection']
        selections[side]=selector(sweep)['selected']
        cases=[('val',ep,cp) for ep,cp in s['all_checkpoints'].items()]
        if include_test:
            reports=list((parent/'110/ckpt_sweep/final_test').glob('*/final_test_metrics_v2.json'))
            assert len(reports)==1
            cp=read(reports[0]);assert Path(cp['checkpoint']).stem==selections[side]
            cases.append(('test',selections[side],cp))
        for split,ep,cp in cases:
            if side=='new':assert sha(cp['checkpoint'])==cp['checkpoint_sha256']
            rr=frames(cp,split,'110_'+side,ep);caches[side,split,ep]=rr;perframe.extend(rr)
            for domain in ('real','sim','all'):
                subset=[r for r in rr if domain=='all' or r['domain']==domain]
                g,t=geometry(subset),aggregate(subset)
                for k,j in [('DFR','DFR'),('ACI','ACI'),('mean_RIoU','riou'),('MCML_max','MCML_max'),('MCML_mean','MCML_mean')]:
                    assert math.isclose(g[k],t[j],abs_tol=1e-10)
                metrics.append(dict(side=side,split=split,epoch=ep,domain=domain,selected=ep==selections[side],
                    center=t['center'],localization_hits=t['hits'],failure_events=len(t['events']),**g))
                if domain!='all':
                    formal={'R_center(%)':100*t['center']/len(subset),'mean_RIoU':g['mean_RIoU'],
                        'DFR(%/frame)':g['DFR'],'ACI':g['ACI'],'MCML_max(frames)':g['MCML_max'],
                        'MCML_mean(frames)':g['MCML_mean'],'TDR_w10(%)':g['TDR_pct']}
                    if domain=='sim':formal['A-RMSE(deg)']=g['gated_angle_RMSE_deg']
                    for key,value in formal.items():
                        delta=abs(value-cp['metrics'][domain+'/'+key])
                        tol=.0051 if key in ('R_center(%)','MCML_mean(frames)','TDR_w10(%)') else .0002
                        checks.append(dict(side=side,split=split,epoch=ep,key=domain+'/'+key,absdiff=delta,tolerance=tol,passed=delta<=tol))
            for domain,seq in sorted({(r['domain'],r['seq']) for r in rr}):
                ss=[r for r in rr if (r['domain'],r['seq'])==(domain,seq)]
                t=aggregate(ss)
                video.append(dict(side=side,split=split,epoch=ep,domain=domain,seq=seq,
                    center=t['center'],localization_hits=t['hits'],**geometry(ss)))
    assert all(x['passed'] for x in checks)
    contexts=[('selected_VAL','val',None)]+[('same_VAL_ep%d'%e,'val','epoch_%d'%e) for e in (16,18,20,22,24)]
    if include_test:contexts.append(('selected_TEST','test',None))
    pairs=[];differences=[]
    keys=['center','hits','outputs','riou','MCML_max','MCML_mean','DFR','ACI','gated_RMSE']
    for context,split,ep in contexts:
        a=caches['old',split,ep or selections['old']];b=caches['new',split,ep or selections['new']]
        groups=[(d,None) for d in ('real','sim','all')]+sorted({(r['domain'],r['seq']) for r in a})
        for domain,seq in groups:
            aa=[r for r in a if (domain=='all' or r['domain']==domain) and (seq is None or r['seq']==seq)]
            bb=[r for r in b if (domain=='all' or r['domain']==domain) and (seq is None or r['seq']==seq)]
            pair=compare(aa,bb)
            pairs.append(dict(context=context,domain=domain,seq=seq,off_means='old110',on_means='new110',**pair))
            differences.append(dict(context=context,domain=domain,seq=seq,old_epoch=ep or selections['old'],new_epoch=ep or selections['new'],
                **{'delta_'+k:pair['on'][k]-pair['off'][k] for k in keys}))
    input_comparison=[]
    for rank in (0,1):
        oldfile=OLD/'110'/('repro_inputs_rank%d.jsonl'%rank);newfile=base/'110'/oldfile.name
        matched=0;mismatches=[];count=0
        with oldfile.open() as a,newfile.open() as b:
            for x,y in zip(a,b):
                x,y=json.loads(x),json.loads(y);count+=1
                equal=(x['epoch'],x['iteration'],x['combined_sha256'])==(y['epoch'],y['iteration'],y['combined_sha256'])
                matched+=int(equal)
                if not equal and len(mismatches)<20:mismatches.append(dict(iteration=y['iteration'],epoch=y['epoch']))
        assert count==15360
        oi=json.loads((OLD/'110'/('repro_initial_rank%d.jsonl'%rank)).read_text().splitlines()[0])
        ni=json.loads((base/'110'/('repro_initial_rank%d.jsonl'%rank)).read_text().splitlines()[0])
        assert oi['initial_state_sha256']==ni['initial_state_sha256'] and rng_identity(oi['rng'])==rng_identity(ni['rng'])
        input_comparison.append(dict(rank=rank,total=count,matched=matched,mismatch_count=count-matched,first_mismatches=mismatches,
            initial_state_and_rng_match=True))
    write_csv(out/'all_candidate_VAL_and_selected_TEST.csv',metrics)
    write_csv(out/'performance_differences.csv',differences)
    write_csv(out/'per_sequence.csv',video);write_csv(out/'per_frame.csv',perframe)
    save(out/'paired_comparisons.json',pairs)
    save(out/'verification.json',dict(passed=True,checks=checks,selected=selections,inputs=input_comparison,
        note='Numeric tolerances above verify metric recomputation only; no performance reproduction pass threshold'))
    save(out/'complete.json',dict(passed=True,selected=selections,frames=len(perframe),TEST_included=include_test,
        automatic_adoption=False,performance_reproduction_pass=None,single_seed_only=True))
    print('110_RETRAIN_ANALYSIS_COMPLETE',selections,include_test,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--include-test',action='store_true')
    a=p.parse_args();analyze(a.base.resolve(),a.include_test)
