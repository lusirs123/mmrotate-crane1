"""Offline size support only. No fitting, thresholds, source-role changes."""
from collections import Counter
import math
from crane_project.utils.port_reliability_paired_ranking_audit_v1 import summarize

def train_stems(split):
    train=split['train_stems'];val=split['aux_val_stems']
    if (len(train)!=203 or len(val)!=48 or len(set(train+val))!=251 or set(train)&set(val)
            or split['sequence']!='real_seq11' or split['k0']!=1.9 or split['target_geometry']!='top_beam_only'):
        raise ValueError('Require historical203 TRAIN/48 aux VAL; never merge or random split')
    import re
    if not all(re.fullmatch(r'real_seq11_[0-9]{6}',i) for i in train+val):raise ValueError('Wrong source identity')
    return sorted(train)

def detail(pred,gt):
    if pred is None:return None
    p=sorted(pred[2:4],reverse=True);g=sorted(gt[2:4],reverse=True)
    if not all(math.isfinite(x) and x>0 for x in p+g):raise ValueError('Nonpositive size')
    e=[(x-y)/y for x,y in zip(p,g)];over=[abs(v)>.1 for v in e];wrong=any(over)
    axis='both' if all(over) else 'L_only' if over[0] else 'S_only' if over[1] else 'correct'
    exceeded=[v for v,o in zip(e,over) if o]
    direction=('small' if all(v<0 for v in exceeded) else 'large' if all(v>0 for v in exceeded) else 'mixed') if wrong else 'correct'
    maximum=max(map(abs,e));severity='10_15' if maximum<=.15 else '15_25' if maximum<=.25 else 'over25'
    pa=pred[4]+(math.pi/2 if pred[2]<pred[3] else 0);ga=gt[4]+(math.pi/2 if gt[2]<gt[3] else 0)
    angle=abs(math.atan2(math.sin(2*(pa-ga)),math.cos(2*(pa-ga)))/2)*180/math.pi
    return dict(wrong=wrong,signed_long=e[0],signed_short=e[1],max_relative=maximum,axis=axis,direction=direction,severity=severity if wrong else 'correct',center_error_px=math.hypot(pred[0]-gt[0],pred[1]-gt[1]),angle_error_deg=angle)

def intervals(rows,predicate):
    result=[];run=[];prev=None
    def flush():
        if run:result.append(dict(start=run[0]['frame_id'],end=run[-1]['frame_id'],length=len(run),images=[r['image'] for r in run]));run.clear()
    for r in sorted(rows,key=lambda x:x['frame_id']):
        if prev is None or r['frame_id']!=prev+1:flush()
        if predicate(r):run.append(r)
        else:flush()
        prev=r['frame_id']
    flush();return result

def report(rows):
    raw={r['image'] for r in rows if r['pred'] is not None}
    simple={r['image'] for r in rows if r['decision']['size_accepted']}
    scores={r['image'] for r in rows if r['score_decision']['size_accepted']}
    wrong=[r for r in rows if r['detail'] is not None and r['detail']['wrong']]
    cells={}
    for state in ('FA','FR','ED','CR','MISSING'):
        part=[r for r in rows if r['size_state']==state]
        cells[state]=dict(count=len(part),images=[r['image'] for r in part])
    blocks=intervals(rows,lambda r:True);blockcounts=[]
    for b in blocks:
        ids=set(b['images']);part=[r for r in rows if r['image'] in ids]
        blockcounts.append(dict(start=b['start'],end=b['end'],frames=b['length'],outputs=sum(r['pred'] is not None for r in part),wrong=sum(r['detail'] is not None and r['detail']['wrong'] for r in part),states=dict(Counter(r['size_state'] for r in part))))
    error_type={k:dict(Counter(r['detail'][k] for r in wrong)) for k in ('axis','direction','severity')}
    return dict(frames=len(rows),methods=dict(raw=summarize(rows,raw),simple=summarize(rows,simple),score=summarize(rows,scores)),cells=cells,error_types=error_type,
                bad_center_ge15=sum(r['detail']['center_error_px']>=15 for r in wrong),bad_angle_gt10=sum(r['detail']['angle_error_deg']>10 for r in wrong),
                original_sampling_origin={origin:dict(frames=len(part),wrong=sum(r['detail'] is not None and r['detail']['wrong'] for r in part),states=dict(Counter(r['size_state'] for r in part))) for origin in sorted({r['sampling_origin'] for r in rows}) for part in [[r for r in rows if r['sampling_origin']==origin]]},
                source_contiguous_blocks=blockcounts,wrong_runs=intervals(rows,lambda r:r['detail'] is not None and r['detail']['wrong']),FA_runs=intervals(rows,lambda r:r['size_state']=='FA'),
                limitation='Source-frame adjacency and sampled windows are descriptive, not independent acquisition events. One historical exposed sequence; support counts do not prove training benefit.')
