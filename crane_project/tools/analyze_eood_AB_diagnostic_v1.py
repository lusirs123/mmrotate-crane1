"""CPU summary of a completed frozen TRAIN diagnosis; no model or new policy."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def describe(values):
    values=sorted(v for v in values if v is not None)
    if not values:return dict(n=0,mean=None,median=None,minimum=None,maximum=None)
    assert all(math.isfinite(v) for v in values)
    return dict(n=len(values),mean=statistics.mean(values),median=statistics.median(values),
                minimum=values[0],maximum=values[-1])


def norm(value):return math.sqrt(sum(x*x for x in value))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    args=parser.parse_args();base=args.base
    completion=json.loads((base/'result/completion.json').read_text())
    assert completion['status']=='COMPLETE' and completion['training_updates']==0
    rows=[json.loads(x) for x in (base/'result/rows.jsonl').read_text().splitlines()]
    assert len(rows)==384
    keys={}
    for arm in ('E','EA','EB'):
        group=[r for r in rows if r['arm']==arm]
        assert len(group)==128
        keys[arm]={(r['index'],r['scale']):r['input_sha256'] for r in group}
        assert completion['arms'][arm]['state_unchanged']
        assert completion['arms'][arm]['before_state']==completion['arms'][arm]['after_state']
    assert keys['E']==keys['EA']==keys['EB'] and len(keys['E'])==128
    assert all(r['ranking']['native_top1_replayed'] for r in rows)
    groups=[]
    for arm in ('E','EA','EB'):
        for domain in ('all','real','sim'):
            for scale in (1.,.5):
                rr=[r for r in rows if r['arm']==arm and r['scale']==scale and
                    (domain=='all' or r['domain']==domain)]
                ranking=[r['ranking'] for r in rr]
                assert sum(r['best_all_iou']<.75 or
                           (r['best_all_iou']>=.75 and r['best_prefilter_iou']<.75) or
                           (r['best_prefilter_iou']>=.75 and r['native_iou']<.75) for r in ranking)==sum(r['native_iou']<.75 for r in ranking)
                actual='KLD' if arm=='EA' else 'POLA'
                gradient=[]
                for r in rr:
                    p=r['gradient_probe'];ix=p['unique_indices'].index(p['role_indices'][actual])
                    g0=p['losses']['DIoU']['normalized'][ix]
                    g1=p['losses']['SymKLD']['normalized'][ix]
                    n0,n1=norm(g0),norm(g1)
                    gradient.append(dict(raw_kld=r['selected'][actual]['raw_kld'],
                        DIoU_value=p['losses']['DIoU']['values'][ix],SymKLD_value=p['losses']['SymKLD']['values'][ix],
                        normalized_gradient_ratio=n1/n0 if n0>1e-12 else None,
                        normalized_cosine=p['normalized_cosine'][ix],
                        DIoU_angle_gradient=abs(g0[4]),SymKLD_angle_gradient=abs(g1[4]),
                        DIoU_center_norm=norm(g0[:2]),SymKLD_center_norm=norm(g1[:2]),
                        DIoU_size_norm=norm(g0[2:4]),SymKLD_size_norm=norm(g1[2:4])))
                groups.append(dict(arm=arm,domain=domain,scale=scale,n=len(rr),
                    native_TP75=sum(r['native_iou']>=.75 for r in ranking),
                    all_candidate_TP75=sum(r['best_all_iou']>=.75 for r in ranking),
                    prefilter_TP75=sum(r['best_prefilter_iou']>=.75 for r in ranking),
                    failures_no_good_candidate=sum(r['best_all_iou']<.75 for r in ranking),
                    failures_prefilter=sum(r['best_all_iou']>=.75 and r['best_prefilter_iou']<.75 for r in ranking),
                    failures_score_or_NMS=sum(r['best_prefilter_iou']>=.75 and r['native_iou']<.75 for r in ranking),
                    native_iou=describe([r['native_iou'] for r in ranking]),
                    oracle_iou=describe([r['best_all_iou'] for r in ranking]),
                    assignment_agree=sum(r['assignment_agree'] for r in rr),
                    POLA_selected_iou=describe([r['selected']['POLA']['riou'] for r in rr]),
                    KLD_selected_iou=describe([r['selected']['KLD']['riou'] for r in rr]),
                    KLD_minus_POLA_iou=describe([r['selected']['KLD']['riou']-r['selected']['POLA']['riou'] for r in rr]),
                    KLD_better_count=sum(r['selected']['KLD']['riou']>r['selected']['POLA']['riou']+1e-6 for r in rr),
                    KLD_worse_count=sum(r['selected']['KLD']['riou']<r['selected']['POLA']['riou']-1e-6 for r in rr),
                    POLA_good_KLD_bad=sum(r['selected']['POLA']['riou']>=.75 and r['selected']['KLD']['riou']<.75 for r in rr),
                    KLD_good_POLA_bad=sum(r['selected']['KLD']['riou']>=.75 and r['selected']['POLA']['riou']<.75 for r in rr),
                    actual_positive_saturated=sum(r['selected'][actual]['B_saturated'] for r in rr),
                    candidate_pool_class_median=describe([r['pool_class_cost']['median'] for r in rr]),
                    candidate_pool_original_geometry_median=describe([r['pool_original_geometry']['median'] for r in rr]),
                    candidate_pool_raw_kld_median=describe([r['pool_raw_kld']['median'] for r in rr]),
                    actual_positive_gradient={k:describe([g[k] for g in gradient]) for k in gradient[0]},
                    negative_gradient_cosine=sum(g['normalized_cosine'] is not None and g['normalized_cosine']<0 for g in gradient)))
    resources=[json.loads(x) for x in (base/'guard.jsonl').read_text().splitlines()]
    network_path=base/'result/network_gradients.jsonl'
    network=[];network_groups=[]
    if network_path.exists():
        network=[json.loads(x) for x in network_path.read_text().splitlines()]
        assert len(network)==192
        assert all(r['whole_model_backward'] and r['all_network_gradients_finite'] and
                   r['optimizer_updates']==0 and r['batch']==2 for r in network)
        for arm in ('E','EA','EB'):
            for scale in (1.,.5):
                rr=[r for r in network if r['arm']==arm and r['scale']==scale]
                assert len(rr)==32
                network_groups.append(dict(arm=arm,scale=scale,n=len(rr),
                    fpn_counterfactual_cosine=describe([r['counterfactual_fpn_cosine'] for r in rr]),
                    negative_fpn_cosine=sum(r['counterfactual_fpn_cosine']<0 for r in rr),
                    SymKLD_over_DIoU_fpn_gradient=describe([r['fpn_gradient_l2']['counterfactual_SymKLD']/
                        r['fpn_gradient_l2']['counterfactual_DIoU'] for r in rr]),
                    auxiliary_over_main_reg_fpn_gradient=describe([r['fpn_gradient_l2']['auxiliary']/
                        r['fpn_gradient_l2']['main_reg'] for r in rr]),
                    actual_positive_saturated=sum(d>120 for r in rr for d in r['positive_raw_kld'])))
    result=dict(passed=True,views=len(rows),groups=groups,parameter_states_unchanged=True,
        native_replays=384,assignment_replays=768,shared_input_checks=256,
        training_updates=0,optimizer_constructed=False,model_parameter_gradients=bool(network),
        whole_model_backward_batches=len(network),network_groups=network_groups,TEST=False,VAL=False,
        peak_own_driver_mib=max(v for r in resources for v in r['own_process_mib'].values()),
        lowest_free_mib=min(r['GPU']['free_mib'] for r in resources),
        highest_temperature_C=max(r['GPU']['temperature_C'] for r in resources),
        completion=completion,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        notes=['Fixed late checkpoints, small TRAIN subset, deterministic scales; not generalization or training-history causality.',
               'Box-space gradient normalization is GT-axis normalized center, predicted log sides, angle radians; native batch2 whole-model gradients are separately recorded when present; no optimizer steps.',
               'Candidate oracles and GT-aware assignments are diagnostic only; inference policy unchanged.'])
    paired=[]
    for other in ('EA','EB'):
        for domain in ('all','real','sim'):
            for scale in (1.,.5):
                aa={r['index']:r for r in rows if r['arm']=='E' and r['scale']==scale and (domain=='all' or r['domain']==domain)}
                bb={r['index']:r for r in rows if r['arm']==other and r['scale']==scale and (domain=='all' or r['domain']==domain)}
                assert set(aa)==set(bb)
                paired.append(dict(arm=other,domain=domain,scale=scale,n=len(aa),
                    TP75_recovered=sum(aa[i]['ranking']['native_iou']<.75<=bb[i]['ranking']['native_iou'] for i in aa),
                    TP75_lost=sum(bb[i]['ranking']['native_iou']<.75<=aa[i]['ranking']['native_iou'] for i in aa),
                    mean_native_iou_delta=statistics.mean(bb[i]['ranking']['native_iou']-aa[i]['ranking']['native_iou'] for i in aa)))
    result['paired_checkpoint_comparisons']=paired
    with (base/'summary.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    for row in groups:
        if row['domain']=='all':
            print(json.dumps({k:row[k] for k in ('arm','scale','n','native_TP75','all_candidate_TP75',
                'failures_no_good_candidate','failures_prefilter','failures_score_or_NMS',
                'KLD_worse_count','KLD_better_count','POLA_good_KLD_bad','KLD_good_POLA_bad',
                'actual_positive_saturated','negative_gradient_cosine')},ensure_ascii=False))
    print('FROZEN_DIAGNOSIS_SUMMARY_COMPLETE')


if __name__=='__main__':main()
