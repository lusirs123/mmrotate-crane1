"""New TRAIN support protocol; no original dataset split or label is changed.

Planning and schedules use stdlib. Torch loss is imported only when training.
The original size-only online head/delivery guards remain unchanged.
"""
from collections import Counter, defaultdict
import random

from crane_project.utils import port_geometry_size_support_v1 as diagnostic

VERSION = 'port_geometry_size_contrast_v1'
SETTINGS = dict(seed=1703, steps=200, smoke_steps=2, batch_size=8,
                lr=.001, weight_decay=0., clip_norm=10., scales=[1., .5],
                holdout_divisor=5, purge_frames=10, error_repeat_cap=4,
                loss_beta=.1, ratio_weight=.25)
ARMS = ('dual_log', 'dual_log_ratio')


def build_plan(rows, legacy, sampling):
    """Caller supplies TRAIN only. Roles depend on identity/order, never errors.

    Last floor(n/5) frames in each TRAIN video are diagnostic holdout; ten
    preceding indices are purged. Legacy64 all stay out of new gradients.
    This does not create independent scenes or undo detector training exposure.
    """
    if (len(rows) != 2558 or len({r['image'] for r in rows}) != 2558 or
            Counter(r['sequence'] for r in rows) != Counter(diagnostic.COUNTS['train']) or
            any(r['reliability_role'] != 'train' for r in rows) or len(legacy) != 64):
        raise ValueError('Requires full fixed TRAIN only and original64 identities')
    if not set(legacy) <= {r['image'] for r in rows}:
        raise ValueError('Legacy samples outside TRAIN')
    result, summaries = [], {}
    for sequence in sorted(diagnostic.COUNTS['train']):
        group = sorted((r for r in rows if r['sequence'] == sequence), key=lambda r:r['frame_id'])
        boundary = len(group)-len(group)//SETTINGS['holdout_divisor']
        for position, row in enumerate(group):
            role = ('legacy' if row['image'] in legacy else 'probe' if position >= boundary
                    else 'purged' if position >= boundary-SETTINGS['purge_frames'] else 'fit')
            error = diagnostic.size_errors(row['gt'], row['pred'])
            stratum = ('missing' if error is None else 'both_small' if error['both_small']
                       else 'short_small' if error['short_too_small'] else 'ratio' if
                       error['ratio_relative_over_10pct'] else 'other_wrong' if error['size_wrong']
                       else 'correct')
            result.append(dict(image=row['image'], sequence=sequence, domain=row['domain'],
                split=row['split'], frame_id=row['frame_id'], sample_role=role,
                legacy_role=legacy.get(row['image']), sampling=sampling.get(row['image']),
                standard_gt=row['gt'], standard_b=row['b_original'], standard_m=row['pred'],
                size_wrong=None if error is None else error['size_wrong'], stratum=stratum,
                flags=None if error is None else {k:error[k] for k in diagnostic.FLAGS}))
        seq_rows = [r for r in result if r['sequence'] == sequence]
        summaries[sequence] = {role:dict(frames=sum(r['sample_role']==role for r in seq_rows),
            wrong=sum(r['sample_role']==role and r['size_wrong'] is True for r in seq_rows),
            strata=dict(Counter(r['stratum'] for r in seq_rows if r['sample_role']==role)),
            flags={flag:sum(r['sample_role']==role and r['flags'] is not None and r['flags'][flag]
                           for r in seq_rows) for flag in diagnostic.FLAGS})
            for role in ('fit','probe','legacy','purged')}
    return dict(protocol=VERSION, settings=SETTINGS, records=result, support=summaries,
        role_counts=dict(Counter(r['sample_role'] for r in result)),
        physical_reference_match='UNCONFIRMED_REQUIRES_OBJECT_AND_CAMERA_GEOMETRY_EVIDENCE',
        independent_holdout=False, support_sufficient='NOT_AUTOMATICALLY_DETERMINED',
        limitations=['New roles are TRAIN diagnostic roles, not dataset split reassignment.',
                     'All TRAIN was exposed to detector/midpoint training; probe is not independent generalization.',
                     'Ten-index gap is not a verified time gap for unmapped videos.',
                     'Legacy64 is reported separately and never enters new gradients.',
                     'Support summaries do not automatically certify sufficient independent scenes.'])


def schedule(records):
    """Equal domain/scale slots, round-robin videos, capped error prioritization.

    Every even step first slot attempts one error in the selected video; the
    other slots prefer correct frames. Missing error pools fall back to correct
    frames in that same video. Never search another video for extra errors.
    Exposure cap applies to ALL draws of size-error/ratio strata per view.
    """
    rng = random.Random(SETTINGS['seed'])
    keys = [(d,s) for d in ('real','sim') for s in SETTINGS['scales']]
    pools = {key:defaultdict(list) for key in keys}
    for i,r in enumerate(records):
        if r['sample_role']=='fit' and r['eligible']:
            pools[(r['domain'],r['scale'])][r['sequence']].append(i)
    required = {d:sorted(s for s in diagnostic.COUNTS['train'] if s.startswith(d+'_'))
                for d in ('real','sim')}
    if any(sorted(pools[k])!=required[k[0]] for k in keys):
        raise ValueError('Each TRAIN video needs eligible fit support in both scales')
    queues, offsets, video_offsets, strata_offsets = {}, {}, Counter(), Counter()
    exposures = Counter()
    def draw(key, seq, kind, used):
        qkey = (key,seq,kind)
        values = [i for i in pools[key][seq] if kind=='all' or records[i]['stratum']==kind]
        if not values:
            return None
        if qkey not in queues:
            queues[qkey] = rng.sample(values,len(values)); offsets[qkey]=0
        for _ in range(len(values)):
            if offsets[qkey]==len(queues[qkey]):
                queues[qkey]=rng.sample(values,len(values)); offsets[qkey]=0
            i=queues[qkey][offsets[qkey]]; offsets[qkey]+=1
            error = records[i]['stratum'] not in ('correct','missing')
            if i not in used and (not error or exposures[i]<SETTINGS['error_repeat_cap']):
                return i
        return None
    batches=[]
    error_types=('both_small','short_small','ratio','other_wrong')
    for step in range(SETTINGS['steps']):
        chosen=[]
        for key in keys:
            for slot in range(2):
                videos=required[key[0]]
                seq=videos[video_offsets[key]%len(videos)]; video_offsets[key]+=1
                i=None
                if step%2==0 and slot==0:
                    start=strata_offsets[(key,seq)]%4; strata_offsets[(key,seq)]+=1
                    for t in range(4):
                        i=draw(key,seq,error_types[(start+t)%4],chosen)
                        if i is not None: break
                if i is None: i=draw(key,seq,'correct',chosen)
                if i is None: i=draw(key,seq,'all',chosen)
                if i is None:
                    raise ValueError('No admissible video-local sample; do not silently change schedule')
                exposures[i]+=1; chosen.append(i)
        batches.append(chosen)
    return batches


def schedule_report(records,batches):
    exposures=Counter(i for b in batches for i in b)
    return dict(slots=sum(map(len,batches)), unique_views=len(exposures),
        exposures=[dict(image=records[i]['image'],scale=records[i]['scale'],
                        sequence=records[i]['sequence'],stratum=records[i]['stratum'],
                        times=n) for i,n in sorted(exposures.items())],
        by_sequence_scale={seq+'/'+str(scale):sum(n for i,n in exposures.items() if
            records[i]['sequence']==seq and records[i]['scale']==scale)
            for seq in sorted(diagnostic.COUNTS['train']) for scale in SETTINGS['scales']},
        eligible_fit_views=sum(r['sample_role']=='fit' and r['eligible'] for r in records),
        error_repeat_cap=SETTINGS['error_repeat_cap'])


def losses(delta,gt,m,bm,arm):
    """Unclipped OFFLINE log-edge and GT-projected-ratio supervision.

    ROI swap may reverse ratio sign, but SmoothL1 is even. Compare predicted
    residual difference to target residual difference; never force physical L/S.
    """
    if arm not in ARMS: raise ValueError('Unknown frozen comparison arm')
    from torch.nn import functional as F
    from crane_project.utils import port_geometry_midpoint_edge_residual_v1 as model
    target=model.size_targets(gt,m,bm)['log_residual']
    dual=model.size_loss(delta,gt,m,bm)
    ratio=F.smooth_l1_loss(delta[:,0]-delta[:,1],target[:,0]-target[:,1],
                         beta=SETTINGS['loss_beta'],reduction='mean')
    weight=SETTINGS['ratio_weight'] if arm=='dual_log_ratio' else 0.
    return dict(total=dual+weight*ratio,dual=dual,ratio=ratio,ratio_weight=weight)
