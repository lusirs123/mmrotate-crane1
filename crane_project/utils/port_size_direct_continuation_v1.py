"""S048 D ep02 C/R continuation; no model, sampler or inference changes."""
import math
from crane_project.utils import port_size_structure_v1 as model
from crane_project.utils import port_size_a20_protection_train_v1 as historical

VERSION='port_size_direct_continuation_v1'
ARMS=('C','R')
D_SHA='615646e20f07b9cfec16c9fc18b56505dc737688ab5d0aceadb7b1eca8e8e299'
# These float64 fixed memberships are ONLY for offline diagnostics, no loss input.
key=historical.key
reference_table=historical.reference_table
def protection_report(rows,table):
    result=historical.protection_report(rows,table)
    for label,entry in result.items():
        rr=[r for r in rows if r['midpoint'] is not None and (label=='overall' or r['domain']==label or r['sequence']==label or r['domain']+'/'+str(r['scale'])==label)]
        for j,edge in enumerate(('long','short')):
            for mode,field in [('delivered','edge_residual'),('pre_guard','pre_guard')]:
                errors=[]
                for row in rr:
                    ref=table[key(row)]
                    if not ref['gain'][j]:continue
                    sizes=row[field][2:4]
                    sides=sizes[::-1] if mode=='pre_guard' and row['midpoint'][2]<row['midpoint'][3] else sizes if mode=='pre_guard' else sorted(sizes,reverse=True)
                    e=abs(sides[j]/sorted(row['gt'][2:4],reverse=True)[j]-1)
                    errors.append((ref['m_errors'][j],ref['a_errors'][j],e))
                cell=entry['edges'][edge][mode]
                cell.update(D_gain_preserved_or_better=sum(n<=d+1e-6 for m,d,n in errors),
                    D_gain_weakened_still_beneficial=sum(n>d+1e-6 and n<m-1e-6 for m,d,n in errors),
                    D_gain_extra_improvement_sum=sum(max(0.,d-n) for m,d,n in errors),
                    D_gain_erosion_sum=sum(max(0.,n-d) for m,d,n in errors))
    return result


def protocol():
    from pathlib import Path
    import json
    p=json.loads((Path(__file__).resolve().parents[1]/'tools/port_size_a20_protection_train_v1_protocol.json').read_text())
    p.update(protocol=VERSION,start_sha256=D_SHA,arms=list(ARMS),parameter_count=2746,formal_updates=7668,val_evaluations=12,
        optimizer='fresh Adam per arm lr1e-4 betas(.9,.999) eps1e-8 weight_decay0 amsgradFalse constant lr; no old optimizer loaded',
        initialization='Fixed S046 D_direct ep02 model only,1278 prior updates. Adam/RNG reset identically; all engineering updates discarded. Original head/forward unchanged.',
        inputs='Same original B-P3/M/commonW/267 channels. No GT in forward. Complete TRAIN5116views/5112outputs; complete VAL887/886 each epoch.',
        reference_precision='Fixed saved D/M float64 errors and memberships computed once; diagnostics ONLY, never loss/sampling/forward inputs. Historical helper a_errors means D start errors in this experiment.',
        engineering='Full saved D TRAIN/VAL replay, exact head-only load and empty Adam, original D loss/update equivalence2steps, C/R finite updates and exact saved optimizer/RNG next step,discard10temporaryupdates. No benefit gate.',
        stop='Source/replay/mapping/numerical/resource failures stop. No TRAIN effect gate.6epochs/arm only; no TEST/depth,lambda search,automatic M replacement.',
        exposure='Protocol follows exposed S047 TEST/Unknown description. No TEST/Unknown reads in this task, no compensation or validation on them. Independent generalization remains unproved.',
        scope='Two-stage D ep02 continuation vs historical vector ep20 continuation: same continuation budget/order, different initial training stages; NOT pure structure ablation.',
        mechanism='All6 same-epoch R-C metrics plus ratio-improved/short-worsened/accurate-other-edge-harmed intersections. No newly invented mechanism passing score or threshold. Epoch6 is fixed-budget comparison only.')
    p.pop('a20_sha256');p['losses']={k:v for k,v in p['losses'].items() if k in ('L0','Lratio','ratio_weight','C','R')}
    p['losses']['L0']='EXACT original S046 D mean2N SmoothL1_beta0.1(delta_roi-target_roi)'
    p['losses']['Lratio']='EXACT historical meanN SmoothL1_beta0.1(canonical_log_error_long-canonical_log_error_short), no weights/masks'
    p.pop('fixed_masks');p['diagnostic_sets']=dict(M_near='E_M<=.03',D_gain='E_D<E_M-1e-6',boundary='Offline only; saved float64 sets, no training penalty')
    p['selection']='Same historical eligibility and lexicographic Sim long,Sim short,Real long,Real short,earlier continuation epoch. Source-relative A20 clauses bind D ep02, same thresholds. No eligible=>descriptive representative only.'
    p['conditions']={k.replace('A20','D02'):v.replace('A20','D02') for k,v in p['conditions'].items()}
    p['resource']['single_physical_gpu_choices']=[0,1,2,3]
    p['resource']['choice']='Most free memory at launch across four cards; one physical GPU, existing7GB cap, no other task touched'
    return p


def losses(output,gt,m,bm,arm,torch):
    if arm not in ARMS:raise ValueError('Unknown C/R arm')
    size=model.size_loss(output,gt,m,bm,'D_direct')
    target=model.residual.size_targets(gt,m,bm)['log_residual']
    error=output['delta_roi']-target
    swap=(m[:,2]<m[:,3]) ^ (bm[:,2]<bm[:,3])
    canonical=torch.where(swap[:,None],error.flip(-1),error)
    F=torch.nn.functional
    point_size=F.smooth_l1_loss(canonical,torch.zeros_like(canonical),beta=.1,reduction='none')
    point_ratio=F.smooth_l1_loss(canonical[:,0]-canonical[:,1],torch.zeros_like(canonical[:,0]),beta=.1,reduction='none')
    ratio=point_ratio.mean()
    return dict(total=size if arm=='C' else size+.25*ratio,size=size,ratio=ratio,point_size=point_size,point_ratio=point_ratio)


def eligibility(summary,diag,start):
    result=historical.eligibility(summary,diag,start)
    result['checks']={k.replace('A20','D02'):v for k,v in result['checks'].items()}
    result['failed']=[k.replace('A20','D02') for k in result['failed']]
    return result


def ratio_intersections(crows,rrows):
    """Paired final delivery changes, fixed M near sets; no new subset search."""
    from crane_project.utils.port_size_a20_protection_review_v1 import relative_errors
    c={key(r):r for r in crows};r={key(x):x for x in rrows}
    if c.keys()!=r.keys() or len(c)!=len(crows) or len(r)!=len(rrows):raise ValueError('C/R identities differ')
    groups={'overall':list(c.values())}
    groups.update({d:[x for x in c.values() if x['domain']==d] for d in ('real','sim')})
    groups.update({v:[x for x in c.values() if x['sequence']==v] for v in sorted({x['sequence'] for x in c.values()})})
    if crows and all(x['split'] in ('train','train_sim') for x in crows):
        groups.update({d+'/'+str(sc):[x for x in c.values() if x['domain']==d and x['scale']==sc] for d in ('real','sim') for sc in (.5,1.)})
    out={}
    for label,rr in groups.items():
        count=dict(outputs=0,ratio_improved=0,ratio_improved_short_worse=0,ratio_improved_long_worse=0,ratio_improved_scale_worse=0,ratio_improved_any_Mnear_harmed=0)
        ids={k:[] for k in count if k!='outputs'};amounts={k:0. for k in ('ratio_gain','short_harm','long_harm','scale_harm','near_positive_harm')}
        for a in rr:
            b=r[key(a)]
            if any(a[k]!=b[k] for k in ('gt','b','midpoint','domain','sequence','frame_id','scale')):raise ValueError('C/R metadata differs')
            if a['edge_residual'] is None:
                if b['edge_residual'] is not None:raise ValueError('C/R presence differs')
                continue
            count['outputs']+=1
            ce,re,me=[relative_errors(x,a['gt']) for x in (a['edge_residual'],b['edge_residual'],a['midpoint'])]
            gain=abs(ce['r'])-abs(re['r'])
            if gain<=1e-12:continue
            flags=dict(ratio_improved=True,ratio_improved_short_worse=re['absolute'][1]>ce['absolute'][1]+1e-12,
                ratio_improved_long_worse=re['absolute'][0]>ce['absolute'][0]+1e-12,
                ratio_improved_scale_worse=abs(re['a'])>abs(ce['a'])+1e-12,
                ratio_improved_any_Mnear_harmed=any(me['absolute'][j]<=.03 and re['absolute'][j]>ce['absolute'][j]+1e-12 for j in (0,1)))
            for k,v in flags.items():
                if v:count[k]+=1;ids[k].append(dict(image=a['image'],scale=a['scale']))
            amounts['ratio_gain']+=gain
            for j,e in enumerate(('long','short')):
                amounts[e+'_harm']+=max(0.,re['absolute'][j]-ce['absolute'][j])
                if me['absolute'][j]<=.03:amounts['near_positive_harm']+=max(0.,re['absolute'][j]-ce['absolute'][j])
            amounts['scale_harm']+=max(0.,abs(re['a'])-abs(ce['a']))
        out[label]=dict(counts=count,amounts_on_ratio_improved=amounts,frame_ids=ids)
    return out
