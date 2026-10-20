"""S043 loss/diagnostics. Frozen S042 memberships NEVER recomputed on GPU.

No inference module is added; the old AlignedSizeHead remains unchanged.
"""
import math
from crane_project.utils import port_size_a20_protection_review_v1 as review

VERSION='port_size_a20_protection_train_v1'
ARMS=('C','R','P')
A20_SHA='545e18459d3ead4a54c441d64de27ca44a8394bfdc20cc2caf4c2f7534326479'


def key(row):return row['image'],row['scale']


def reference_table(rows):
    table={}
    for row in rows:
        if key(row) in table:raise ValueError('Duplicate frozen reference')
        if (row['midpoint'] is None)!=(row['edge_residual'] is None):raise ValueError('Frozen output identity')
        if row['midpoint'] is None:continue
        m,a=[review.relative_errors(row[name],row['gt'])['absolute'] for name in ('midpoint','edge_residual')]
        table[key(row)]=dict(review.fixed_references(m,a),m_errors=m,a_errors=a,
            image=row['image'],scale=row['scale'],split=row['split'],domain=row['domain'],sequence=row['sequence'])
    return table


def pack_references(rows,table,device,torch,training=False):
    if training and any(r['split'] not in ('train','train_sim') for r in rows):
        raise ValueError('VAL must never enter a training reference batch')
    cells=[table[key(r)] for r in rows]
    if any(c['split']!=r['split'] for c,r in zip(cells,rows)):raise ValueError('Reference split mismatch')
    # Python float64 labels determine membership ONCE; casts never reclassify.
    return dict(active=torch.tensor([c['active'] for c in cells],dtype=torch.bool,device=device),
                near=torch.tensor([c['near'] for c in cells],dtype=torch.bool,device=device),
                gain=torch.tensor([c['gain'] for c in cells],dtype=torch.bool,device=device),
                reference=torch.tensor([c['reference'] for c in cells],dtype=torch.float64,device=device))


def losses(output,gt,m,bm,refs,arm,torch):
    from crane_project.utils import port_size_qalign_v1 as model
    from torch.nn import functional as F
    if arm not in ARMS:raise ValueError('Unknown continuation arm')
    if not bool(output['vector_valid'].all()) or not bool(output['candidate_valid'].all()):
        raise ValueError('Invalid pre-guard prediction: stop, do not filter')
    size=model.size_loss(output,gt,m,bm)  # EXACT original control graph.
    target=model.residual.size_targets(gt,m,bm)['log_residual']
    error=output['delta_roi']-target
    swap=(m[:,2]<m[:,3]) ^ (bm[:,2]<bm[:,3])
    canonical=torch.where(swap[:,None],error.flip(-1),error)
    point_size=F.smooth_l1_loss(canonical,torch.zeros_like(canonical),beta=.1,reduction='none')
    point_ratio=F.smooth_l1_loss(canonical[:,0]-canonical[:,1],torch.zeros_like(canonical[:,0]),beta=.1,reduction='none')
    # Actual float32 decoded candidate, error computed against GT in float64.
    # Keep M raw-axis labels even for an invalid raw-order candidate; no GT sort.
    raw=output['candidate_original'][:,2:4].double()
    sides=torch.where((m[:,2]<m[:,3])[:,None],raw.flip(-1),raw)
    goals=gt[:,2:4].detach().double().sort(dim=-1,descending=True)[0]
    relative=(sides/goals-1).abs()
    if not bool(torch.isfinite(relative).all()):raise ValueError('Nonfinite candidate error')
    if refs['reference'].dtype!=torch.float64 or refs['active'].dtype!=torch.bool or refs['reference'].shape!=relative.shape:
        raise ValueError('Frozen reference dtype/shape differs')
    if refs['reference'].requires_grad:raise ValueError('Reference must be detached')
    hinge=torch.where(refs['active'],F.relu(relative-refs['reference']),torch.zeros_like(relative))
    ratio=point_ratio.mean();protect=hinge.mean()
    total=size if arm=='C' else size+.25*ratio
    if arm=='P':total=total+.1*protect
    return dict(total=total,size=size,ratio=ratio,protect=protect,hinge=hinge,relative=relative,
                point_size=point_size,point_ratio=point_ratio,active=(hinge>0)&refs['active'])


def protection_report(rows,table):
    """Fixed M/A20 sets. Report pre-guard AND actual delivery independently."""
    def reversed_direction(row,field,j):
        m=sorted(row['midpoint'][2:4],reverse=True)[j]
        g=sorted(row['gt'][2:4],reverse=True)[j]
        p=row[field][2:4][::-1] if row['midpoint'][2]<row['midpoint'][3] else row[field][2:4]
        return math.log(p[j]/m)*math.log(g/m)<-1e-12
    groups=dict(overall=rows)
    groups.update({d:[r for r in rows if r['domain']==d] for d in ('real','sim')})
    groups.update({s:[r for r in rows if r['sequence']==s] for s in sorted({r['sequence'] for r in rows})})
    if rows and all(r['split'] in ('train','train_sim') for r in rows):
        groups.update({d+'/'+str(s):[r for r in rows if r['domain']==d and r['scale']==s] for d in ('real','sim') for s in (.5,1.)})
    out={}
    for group,rr in groups.items():
        pp=[r for r in rr if r['midpoint'] is not None]
        entry=dict(outputs=len(pp),missing=len(rr)-len(pp),fallback=sum(r['size_delivery']=='fallback' for r in pp),edges={})
        for j,edge in enumerate(('long','short')):
            cell={}
            for mode,field in [('delivered','edge_residual'),('pre_guard','pre_guard')]:
                vals=[]
                for r in pp:
                    ref=table[key(r)];candidate=r[field]
                    if candidate is None:raise ValueError('Missing pre/final candidate for output')
                    # Pre-guard errors keep fixed M labels if the raw order flips.
                    if mode=='pre_guard':
                        sizes=candidate[2:4][::-1] if r['midpoint'][2]<r['midpoint'][3] else candidate[2:4]
                        error=abs(sizes[j]/sorted(r['gt'][2:4],reverse=True)[j]-1)
                    else:error=review.relative_errors(candidate,r['gt'])['absolute'][j]
                    vals.append((ref,error))
                near=[(ref,e) for ref,e in vals if ref['near'][j]]
                useful=[(ref,e) for ref,e in vals if ref['gain'][j]]
                retention=review.capped_retention([ref['m_errors'][j] for ref,e in useful],
                    [ref['a_errors'][j] for ref,e in useful],[e for ref,e in useful])
                gain_losses=[max(0.,e-ref['a_errors'][j]) for ref,e in useful]
                active=[(ref,e) for ref,e in vals if ref['active'][j]]
                excess=[max(0.,e-ref['reference'][j]) for ref,e in active]
                cell[mode]=dict(near_count=len(near),near_mae=sum(e for ref,e in near)/len(near) if near else None,
                    near_harmed=sum(e>ref['m_errors'][j]+1e-6 for ref,e in near),
                    near_positive_harm_sum=sum(max(0.,e-ref['m_errors'][j]) for ref,e in near),
                    useful=retention,useful_still_beneficial=sum(e<ref['m_errors'][j]-1e-6 for ref,e in useful),
                    useful_lost=sum(e>=ref['m_errors'][j]-1e-6 for ref,e in useful),
                    useful_error_worse_than_M=sum(e>ref['m_errors'][j]+1e-6 for ref,e in useful),
                    useful_gain_eroded_sum=sum(gain_losses),
                    useful_direction_reversed=sum(reversed_direction(r,field,j) for r in pp if table[key(r)]['gain'][j]),
                    protected_count=len(active),activated=sum(x>0 for x in excess),
                    excess_sum=sum(excess),excess_mean_per_all_edges=sum(excess)/len(vals) if vals else None)
            entry['edges'][edge]=cell
        out[group]=entry
    return out


def eligibility(summary,diag,a20_summary):
    """Every S038 gate expanded here; SAME full set for C/R/P."""
    checks={}
    for d in ('real','sim'):
        m,c=[summary[d][k] for k in ('midpoint','edge_residual')];a=a20_summary[d]['edge_residual']
        for e in ('long','short'):
            x,y=c['sizes']['edges'][e]['mae'],m['sizes']['edges'][e]['mae']
            checks[d+'/'+e+'_vs_M']=x<y-1e-6 if d=='sim' else x<=y+.001
            near=diag[d]['near'][e];checks[d+'/near_'+e]=near['after_mae']<=near['before_mae']+.001
            checks[d+'/'+e+'_vs_A20']=x<=a['sizes']['edges'][e]['mae']+(1e-6 if d=='sim' else .001)
        checks[d+'/riou']=c['mean_riou_full_frame']>=m['mean_riou_full_frame']-.001
        pair=summary[d]['paired'];checks[d+'/identity']=pair['frozen_identity'] and pair['new_output']==pair['lost_output']==0
        tm,tc=m['temporal'][d],c['temporal'][d]
        checks[d+'/dfr']=tc['dfr_percent_per_frame']<=tm['dfr_percent_per_frame']
        checks[d+'/mcml_max']=tc['mcml_max']<=tm['mcml_max']
        checks[d+'/mcml_mean']=tc['mcml_mean']<=tm['mcml_mean']
        checks[d+'/tdr']=tc['tdr_w10']['denominator']==tm['tdr_w10']['denominator'] and tc['tdr_w10']['numerator']>=tm['tdr_w10']['numerator']
        checks[d+'/aci']=tc['aci']==tm['aci'] and tc['aci_adjacent_pairs']==tm['aci_adjacent_pairs']
    for k in ('a','r'):
        x=summary['sim']['edge_residual']['sizes'][k]['abs_mae']
        checks['sim/'+k+'_vs_M']=x<=summary['sim']['midpoint']['sizes'][k]['abs_mae']+1e-6
        y=a20_summary['sim']['edge_residual']['sizes'][k]['abs_mae']
        checks['sim/'+k+'_vs_A20']=x<y-1e-6 if k=='r' else x<=y+1e-6
    checks['sim/one_edge_strictly_better_A20']=any(summary['sim']['edge_residual']['sizes']['edges'][e]['mae']<a20_summary['sim']['edge_residual']['sizes']['edges'][e]['mae']-1e-6 for e in ('long','short'))
    x=summary['overall']['edge_residual']['mean_riou_full_frame']
    checks['overall/riou_M']=x>=summary['overall']['midpoint']['mean_riou_full_frame']-1e-5
    checks['overall/riou_A20']=x>=a20_summary['overall']['edge_residual']['mean_riou_full_frame']-1e-5
    return dict(pass_all=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v])


def mechanism(representatives):
    c,r,p=[representatives[a] for a in ARMS];checks={}
    for d in ('real','sim'):
        for e in ('long','short'):
            x=p['metrics'][d]['edge_residual']['sizes']['edges'][e]['mae']
            if d=='sim':
                for name,ref in [('C',c),('R',r)]:checks[d+'/'+e+'_vs_'+name]=x<=ref['metrics'][d]['edge_residual']['sizes']['edges'][e]['mae']+1e-6
            pr,rr=[ep['protection'][d]['edges'][e]['delivered']['useful']['fraction'] for ep in (p,r)]
            checks[d+'/'+e+'/retention_vs_R']=pr is not None and rr is not None and pr>=rr-1e-6
        excess=lambda ep:sum(ep['protection'][d]['edges'][e]['delivered']['excess_mean_per_all_edges'] for e in ('long','short'))/2
        checks[d+'/protected_excess_vs_R']=excess(p)<excess(r)-1e-6
    return dict(pass_all=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],
                note='Representative comparison only; same-epoch curves also retained. Not automatic adoption.')
