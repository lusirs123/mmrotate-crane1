"""Independent CP objective and frozen eligibility; original C/R/P stay sealed."""
import math
from crane_project.utils import port_size_a20_protection_train_v1 as original
from crane_project.tools.diagnose_symeood_110_lvsr_edges import edge_diagnostic,groups


def losses(output,gt,m,bm,refs,torch):
    terms=original.losses(output,gt,m,bm,refs,'C',torch)
    return dict(terms,total=terms['size']+.1*terms['protect'])


def update(h,opt,x,refs,torch):
    h.train();opt.zero_grad();out=h(*x[:-1]);terms=losses(out,x[-1],x[-2],x[3],refs,torch)
    loss=terms['total']
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite CP loss')
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in h.parameters()):raise ValueError('Invalid CP gradients')
    norm=float(torch.nn.utils.clip_grad_norm_(h.parameters(),10.));opt.step()
    if any(not bool(torch.isfinite(p).all()) for p in h.parameters()):raise ValueError('Invalid CP parameters')
    stats={k:float(terms[k].detach()) for k in ('total','size','ratio','protect')}
    stats.update(grad_norm=norm,activated=int(terms['active'].sum()),protected=int(refs['active'].sum()),edges=int(terms['active'].numel()))
    return stats


def damage(rows):
    return {name:edge_diagnostic(rr,'edge_residual')[0] for name,rr in groups(rows).items()}


def hits(value):return value['frames']-sum(x['length'] for x in value['failure_intervals']['riou'])


def eligibility(summary,near,damages,control,control_damage,parent_summary):
    inherited=original.eligibility(summary,near,parent_summary);checks={'original/'+k:v for k,v in inherited['checks'].items()}
    for d in ('real','sim'):
        s=summary[d];checks[d+'/riou_hits_vs_M']=hits(s['edge_residual'])>=hits(s['midpoint'])
        checks[d+'/joint_vs_M']=s['edge_residual']['sizes']['joint10']>=s['midpoint']['sizes']['joint10']
        for edge in ('long','short'):
            a=damages[d][edge+'_near_positive_harm_mean_pct']/100
            b=control_damage[d][edge+'_near_positive_harm_mean_pct']/100
            checks[d+'/'+edge+'_positive_harm_vs_C']=a<=b+1e-6
        checks[d+'/long_material_harm_vs_C']=damages[d]['long_near_material_harmed_0p1pp']<=control_damage[d]['long_near_material_harmed_0p1pp']
    s=summary['overall'];a=s['edge_residual']['sizes'];m=s['midpoint']['sizes'];c=control['overall']['edge_residual']['sizes']
    checks['overall/joint_strict_M']=a['joint10']>m['joint10']
    checks['overall/joint_vs_C']=a['joint10']>=c['joint10']
    for k in ('a','r'):checks['overall/'+k+'_vs_M']=a[k]['abs_mae']<=m[k]['abs_mae']+1e-6
    checks['sim/long_strict_C']=summary['sim']['edge_residual']['sizes']['edges']['long']['mae']<control['sim']['edge_residual']['sizes']['edges']['long']['mae']-1e-6
    checks['overall/positive_harm_strict_C']=damages['overall']['long_near_positive_harm_mean_pct']/100<control_damage['overall']['long_near_positive_harm_mean_pct']/100-1e-6
    checks['overall/material_harm_fewer_C']=damages['overall']['long_near_material_harmed_0p1pp']<control_damage['overall']['long_near_material_harmed_0p1pp']
    chain={}
    for d in ('overall','real','sim'):
        c,b=[summary[d][k]['sizes'] for k in ('edge_residual','b')]
        chain[d+'/joint_vs_110']=c['joint10']>=b['joint10']
        if d!='overall':
            for edge in ('long','short'):chain[d+'/'+edge+'_vs_110']=c['edges'][edge]['mae']<=b['edges'][edge]['mae']+.001
        else:
            for k in ('a','r'):chain[d+'/'+k+'_vs_110']=c[k]['abs_mae']<=b[k]['abs_mae']+1e-6
    return dict(pass_all=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],
        original=inherited,full_chain_size_claim=dict(pass_all=all(chain.values()),checks=chain,failed=[k for k,v in chain.items() if not v]))


def selection_key(summary,damages,epoch):
    s=summary['overall']['edge_residual']['sizes']
    return [-s['joint10'],damages['overall']['long_near_positive_harm_mean_pct']/100,s['r']['abs_mae'],s['a']['abs_mae'],epoch]


def select(events):
    eligible=[e for e in events if e['eligibility']['pass_all']]
    return dict(selected=min(eligible,key=lambda e:e['selection_key'])['epoch'] if eligible else None,
        representative=min(events,key=lambda e:e['selection_key'])['epoch'],fixed_final=6,
        eligible_epochs=[e['epoch'] for e in eligible])
