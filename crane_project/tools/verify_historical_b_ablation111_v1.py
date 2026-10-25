"""Independent local CPU reaggregation: temporal pairs, native PKL and scope.

Requires NumPy to unpickle native arrays; no OpenCV/MMCV/model or inference.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_csv(p):return list(csv.DictReader(p.open()))
def avg(v):return sum(v)/len(v)


def valid_pairs(rr):
    by=collections.defaultdict(dict)
    for r in rr:by[r['domain'],r['seq']][int(r['fid'])]=r['output']=='True'
    result=set()
    for (domain,seq),frames in by.items():
        for fid,present in frames.items():
            if present and frames.get(fid+1,False):result.add((domain,seq,fid,fid+1))
    return result


def main(base,a111_pkl):
    r=base/'receipts';manifest=json.loads((r/'manifest.json').read_text());assert all(sha(r/p)==h for p,h in manifest.items())
    identities=json.loads((r/'prediction_identity.json').read_text());frames=load_csv(r/'per_frame_identity.csv');pairs=load_csv(r/'dfr_pair_details.csv')
    independent={};checks=0;decomp=load_csv(r/'dfr_decomposition.csv')
    for name,p in [('B',base/'B_results.pkl'),('111',a111_pkl)]:
        identity=next(x for x in identities if x['version']==name);assert sha(p)==identity['prediction_sha256']
        rr=[x for x in frames if x['version']==name];assert len(rr)==1440 and len(set(x['frame'] for x in rr))==1440
        assert [x['frame'] for x in rr]==identity['frame_ids']
        arr=pickle.loads(p.read_bytes());assert len(arr)==1440
        diags={}
        for x,y in zip(rr,arr):
            assert (len(y[0])>0)==(x['output']=='True')
            if len(y[0]):diags[x['domain'],x['seq'],int(x['fid'])]=math.hypot(float(y[0][0,2]),float(y[0][0,3]))
        pp={k:100*abs(diags[k[:2]+(k[3],)]-diags[k[:2]+(k[2],)])/diags[k[:2]+(k[2],)] for k in valid_pairs(rr)}
        independent[name]=pp
        archived={(x['domain'],x['seq'],int(x['previous_fid']),int(x['current_fid'])) for x in pairs if x[('B_valid' if name=='B' else 'A111_valid')]=='True'}
        assert archived==set(pp);checks+=len(archived)
    for row in pairs:
        if row['B_valid']==row['A111_valid']=='True':
            for name,col in [('B','B'),('111','A111')]:
                value=100*abs(float(row[col+'_diag_current'])-float(row[col+'_diag_previous']))/float(row[col+'_diag_previous'])
                assert abs(value-float(row[col+'_DFR']))<1e-12;checks+=1
    native=load_csv(r/'dfr_native_pkl_decomposition.csv');comparisons=[]
    for source in (native,decomp):
        for row in source:
            domain,seq=row['domain'],row['seq']
            if source is native:
                a={k:v for k,v in independent['B'].items() if (domain=='all' or k[0]==domain) and (seq=='all' or k[1]==seq)}
                b={k:v for k,v in independent['111'].items() if (domain=='all' or k[0]==domain) and (seq=='all' or k[1]==seq)}
            else:
                rr=[x for x in pairs if (domain=='all' or x['domain']==domain) and (seq=='all' or x['seq']==seq)]
                a={(x['domain'],x['seq'],int(x['previous_fid']),int(x['current_fid'])):float(x['B_DFR']) for x in rr if x['B_valid']=='True'}
                b={(x['domain'],x['seq'],int(x['previous_fid']),int(x['current_fid'])):float(x['A111_DFR']) for x in rr if x['A111_valid']=='True'}
            common=set(a)&set(b);old,new=avg(list(a.values())),avg(list(b.values()))
            oldc,newc=avg([a[k] for k in common]),avg([b[k] for k in common])
            vals=dict(B_pairs=len(a),A111_pairs=len(b),common_pairs=len(common),B_DFR=old,A111_DFR=new,
                B_common_DFR=oldc,A111_common_DFR=newc,delta=new-old,common_prediction_effect=newc-oldc,
                pair_scope_effect=(new-newc)-(old-oldc))
            for key,value in vals.items():
                delta=abs(value-float(row[key]));assert delta<1e-11,(domain,seq,key,delta)
                comparisons.append(dict(source='native_PKLOBB' if source is native else 'archived_export',domain=domain,seq=seq,metric=key,absdiff=delta))
    mv=json.loads((r/'metric_verification.json').read_text());assert mv['passed'] and all(x['absdiff']==0 for x in mv['comparisons'])
    focused=json.loads((r/'focused_boundary.json').read_text());assert focused['cases']==20 and all(x['losses_exact'] and x['gradients_exact'] for x in focused['results'])
    x=json.loads((r/'training_identity.json').read_text());assert x['lr_trace_equal'] and x['state_key_shapes_equal'] and x['frozen_stem_layer1_equal']
    assert [z['path'] for z in x['logged_config_diff']]==['.work_dir']
    assert all(not z['log_vs_saved_config_diff'] and z['final_iteration']==15360 and z['final_nfl_calls']==76800 for z in x['records'])
    env=json.loads((base/'additional_identity.json').read_text());assert env['current_cudnn']==8500 and all(z['matches_111_frozen_manifest'] for z in env['source_matches_ablation111'])
    result=dict(passed=True,receipt_files=len(manifest),pair_identity_and_common_formula_checks=checks,
        independent_aggregate_checks=len(comparisons),max_abs_delta=max(x['absdiff'] for x in comparisons),
        reported_metrics_exact_checks=len(mv['comparisons']),focused_boundary_cases=20,
        comparisons=comparisons,source_sha256=sha(Path(__file__)),
        historical_image_bytes_proven=False,historical_complete_init_proven=False,
        no_training=True,no_inference=True,no_GPU=True)
    (base/'local_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--a111-pkl',type=Path,required=True)
    a=p.parse_args();main(a.base.resolve(),a.a111_pkl.resolve())
