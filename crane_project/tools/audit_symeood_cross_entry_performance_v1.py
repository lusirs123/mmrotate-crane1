"""Four historical/native versus USG pairs; shared VAL only, no models."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
PAIRS=[('100','iou','U'),('101','iou','G'),('110','kld','U'),('111','kld','G')]


def selector(path):
    d=json.loads(path.read_text());c=d['selection_config'];values=[]
    assert d['candidate_epochs']==[16,18,20,22,24] and d['center_thresh_px']==15
    for name,cp in d['all_checkpoints'].items():
        m=cp['metrics'];rc=(c['weight_real']*m['real/R_center(%)']+c['weight_sim']*m['sim/R_center(%)'])/100
        td=(c['weight_real']*m['real/TDR_w10(%)']+c['weight_sim']*m['sim/TDR_w10(%)'])/100
        mc=max(m['real/MCML_max(frames)'],m['sim/MCML_max(frames)']);ar=m['sim/A-RMSE(deg)']
        score=c['soft_w_tdr']*td+c['soft_w_r_center']*rc+c['soft_w_aci']*m['sim/ACI']+c['soft_w_angle']*(1-min(ar,90)/90)
        values.append(dict(name=name,rc=rc,mc=mc,ar=ar,score=score))
    maximum=max(v['rc'] for v in values)
    feasible=[v for v in values if v['mc']<=c['mcml_limit'] and v['rc']>=maximum-c['r_center_delta']]
    if not feasible:feasible=[v for v in values if v['mc']<=c['mcml_limit']]
    chosen=(max(feasible,key=lambda v:v['score']) if feasible else min(values,key=lambda v:(v['mc'],-v['rc'],v['ar'])))['name']
    assert chosen==d['selected_checkpoint'],path
    return dict(selected=chosen,rules=c,candidate_epochs=d['candidate_epochs'],center_thresh=d['center_thresh_px'],
                ties=sum(v['score']==next(x['score'] for x in values if x['name']==chosen) for v in feasible),
                order=[v['name'] for v in values],reproduced=True)


def write_csv(path,rows):
    with path.open('x') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)


def main(out):
    out.mkdir(exist_ok=False,parents=True)
    oldbase=ROOT/'work_dirs/symeood_ablation_v1';newbase=ROOT/'work_dirs/symeood_usg_v1'
    counts=collections.defaultdict(lambda:collections.Counter())
    # The archived combined CSV contains both splits. Ignore non-VAL records
    # immediately; no TEST score, frame or metric is used in this audit.
    for r in csv.DictReader((oldbase/'analysis_all8_20261024/per_frame_all8.csv').open()):
        if r['split']!='val' or r['group'] not in ('100','101','110','111'):continue
        for domain in (r['domain'],'all'):
            c=counts[r['group'],r['epoch'],domain]
            c['frames']+=1;c['hits']+=float(r['riou_export'])>=.5
            c['center']+=float(r['center_export'])<15;c['outputs']+=r['output']=='True'
    olds={}
    for r in csv.DictReader((oldbase/'analysis_all8_20261024/candidates_val_all8.csv').open()):
        if r['group'] not in ('100','101','110','111'):continue
        c=counts[r['group'],r['epoch'],r['domain']]
        assert c['frames']==int(r['frames']) and c['outputs']==int(r['outputs'])
        olds[r['group'],r['epoch'],r['domain']]=dict(frames=c['frames'],center=c['center'],hits=c['hits'],outputs=c['outputs'],
            riou=float(r['mean_RIoU']),DFR=float(r['DFR']),ACI=float(r['ACI']),MCML_max=float(r['MCML_max']),
            MCML_mean=float(r['MCML_mean']),A_RMSE=float(r['gated_angle_RMSE_deg']))
    news={}
    for r in csv.DictReader((newbase/'analysis_v1/metrics_all_epochs.csv').open()):
        news[r['regression'],r['objective'],r['epoch'],r['domain']]=dict(frames=int(r['frames']),
            center=round(float(r['R_center_pct'])*int(r['frames'])/100),hits=int(r['localization_hits']),outputs=int(r['outputs']),
            riou=float(r['mean_RIoU']),DFR=float(r['DFR']),ACI=float(r['ACI']),MCML_max=float(r['MCML_max']),
            MCML_mean=float(r['MCML_mean']),A_RMSE=float(r['gated_angle_RMSE_deg']))
    selections=[];tables=[];deltas=[]
    for g,reg,obj in PAIRS:
        parent=(oldbase/'analysis_first4_20261023/snapshot' if g=='100' else oldbase/'analysis_all8_20261024/last4_snapshot')
        op=parent/'work_dirs/symeood_ablation_v1'/g/'ckpt_sweep/sweep_results.json'
        np=newbase/('s0_'+reg+'_'+obj.lower())/'ckpt_sweep/sweep_results.json'
        a,b=selector(op),selector(np)
        assert a['rules']==b['rules'] and a['order']==b['order']
        selections.extend([dict(version=g,side='old',**a),dict(version=reg+'-'+obj,side='new',**b)])
        for context in ['selected']+['epoch_'+str(e) for e in (16,18,20,22,24)]:
            oe,ne=(a['selected'],b['selected']) if context=='selected' else (context,context)
            for domain in ('real','sim','all'):
                x,y=olds[g,oe,domain],news[reg,obj,ne,domain]
                tables.extend([dict(pair=g,regression=reg,objective=obj,context=context,domain=domain,side='old',epoch=oe,**x),
                               dict(pair=g,regression=reg,objective=obj,context=context,domain=domain,side='new',epoch=ne,**y)])
                deltas.append(dict(pair=g,regression=reg,objective=obj,context=context,domain=domain,old_epoch=oe,new_epoch=ne,
                                   **{'delta_'+k:y[k]-x[k] for k in x if k!='frames'}))
    write_csv(out/'performance_values.csv',tables);write_csv(out/'performance_differences.csv',deltas)
    (out/'selection_verification.json').write_text(json.dumps(dict(passed=True,versions=selections,
        tie_rule='original insertion order: ascending epoch; max returns first exact-score tie'),indent=2))
    (out/'provenance.json').write_text(json.dumps(dict(scope='VAL only; formal selected plus five corresponding same epochs',
        TEST=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}),indent=2))
    for r in deltas:
        if r['context']=='selected' and r['domain']=='all':print(r)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out.resolve())
