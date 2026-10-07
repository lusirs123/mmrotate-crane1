"""Offline aggregation of sealed size evidence. No inference, fitting or selection.

B=B24, M=B24+sigma1.5/epoch03, C=the rejected fixed-200 continuous head.
Input SHA pins identify observations, not new independent collection events.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_midpoint_depth_v1 as depth
from crane_project.utils import port_simple_component_reliability_v1 as reliability
from crane_project.tools.eval_crane_offline import compute_riou

VERSION = 'port_size_all_evidence_v1'
STAGES = ('B', 'M', 'C')


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def checked_source(spec, root=ROOT):
    path = Path(spec['path'])
    if not path.is_absolute():
        path = root/path
    if depth.sha(path) != spec['sha256']:
        raise ValueError('Input SHA mismatch: '+str(path))
    if 'member' in spec:
        with tarfile.open(path, 'r:gz') as archive:
            member = archive.getmember(spec['member'])
            if not member.isfile():
                raise ValueError('Expected regular archive member')
            data = archive.extractfile(member).read()
        if hashlib.sha256(data).hexdigest() != spec['member_sha256']:
            raise ValueError('Archive member SHA mismatch')
    else:
        data = path.read_bytes()
    return data


def load(spec):
    data = checked_source(spec)
    return [json.loads(line) for line in data.splitlines() if line.strip()] if spec.get('jsonl') else json.loads(data)


def index(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate frame identity: '+key)
    return result


def equal_box(a, b, tolerance=0):
    if a is None or b is None:
        if a is not b:
            raise ValueError('Missing-output identity differs')
    elif len(a) != len(b) or max(abs(x-y) for x,y in zip(a,b)) > tolerance:
        raise ValueError('Sealed box replay differs')


def sizes(box, gt):
    if box is None:
        return None
    b, g = depth.canonical(box), depth.canonical(gt)
    el, es = math.log(b[2]/g[2]), math.log(b[3]/g[3])
    return dict(long_log=el, short_log=es, scale_log=(el+es)/2,
        ratio_log=el-es, long_relative=math.expm1(el), short_relative=math.expm1(es),
        joint_correct=max(abs(b[2]/g[2]-1), abs(b[3]/g[3]-1)) <= .1)


def describe(values):
    a = np.array(values, dtype=float)
    if not len(a):
        return dict(count=0, mean=None, mean_abs=None, abs_p95=None, min=None, max=None)
    return dict(count=len(a), mean=float(a.mean()), mean_abs=float(np.abs(a).mean()),
        abs_p95=float(np.percentile(np.abs(a),95)), min=float(a.min()), max=float(a.max()))


def geometry_group(rows, stage):
    values = [r['size'][stage] for r in rows if stage in r['size'] and r['size'][stage] is not None]
    out = dict(frames=len(rows), outputs=len(values), output_coverage=len(values)/len(rows),
        both_edges_correct=sum(v['joint_correct'] for v in values))
    out['all_frame_joint_correct_coverage'] = out['both_edges_correct']/len(rows)
    out['errors'] = {key:describe([v[key] for v in values]) for key in
        ('long_relative','short_relative','long_log','short_log','scale_log','ratio_log')}
    out['signs'] = dict(both_small=sum(v['long_log']<0 and v['short_log']<0 for v in values),
        both_large=sum(v['long_log']>0 and v['short_log']>0 for v in values),
        mixed=sum(v['long_log']*v['short_log']<0 for v in values))
    out['descriptive_common_error_bins'] = {}
    for threshold in (.03,.05,.1):
        out['descriptive_common_error_bins'][str(threshold)] = dict(
            both_below=sum(max(v['long_relative'],v['short_relative']) <= -threshold for v in values),
            both_above=sum(min(v['long_relative'],v['short_relative']) >= threshold for v in values))
    return out


def substitute_size_source(b, midpoint):
    """Offline counterfactual only: M center/raw angle/score, B canonical sizes.

    No GT, per-frame method selection, delivered output or model changes.
    Respect M's raw width-height/angle association instead of pasting raw B w/h.
    """
    if (b is None)!=(midpoint is None):
        raise ValueError('Size-source substitution requires paired output presence')
    if b is None:return None
    long_edge,short_edge=depth.canonical(b)[2:4]
    result=list(midpoint)
    result[2:4]=[long_edge,short_edge] if midpoint[2]>=midpoint[3] else [short_edge,long_edge]
    return result


def outcome(good, accepted):
    return 'CR' if good and accepted else 'FR' if good else 'FA' if accepted else 'ED'


def ordered_accounting(states):
    """Exact geometry-first bookkeeping, not causal model attribution.

    states: (old_correct,new_correct,old_accept,new_accept) on shared outputs.
    """
    result = {}
    for name in ('FA','FR','ED','CR'):
        old = sum(outcome(a,x)==name for a,b,x,y in states)
        intermediate = sum(outcome(b,x)==name for a,b,x,y in states)
        new = sum(outcome(b,y)==name for a,b,x,y in states)
        result[name] = dict(old=old, new=new, delta=new-old,
            correctness_change_old_flags=intermediate-old,
            flags_change_new_correctness=new-intermediate)
    return result


def stage_pair(rows, old, new):
    for r in rows:
        if old in r['boxes'] and new in r['boxes']:
            if (r['boxes'][old] is None)!=(r['boxes'][new] is None):
                raise ValueError('Paired output presence differs')
    pairs = [r for r in rows if old in r['size'] and new in r['size']
        and r['size'][old] is not None and r['size'][new] is not None]
    transitions = Counter()
    changes = defaultdict(list)
    patterns = Counter()
    protected = Counter()
    for r in pairs:
        a,b = r['size'][old],r['size'][new]
        ab,bb=r['boxes'][old],r['boxes'][new]
        ac,bc=depth.canonical(ab),depth.canonical(bb)
        protected['center_changed'] += ab[:2]!=bb[:2]
        protected['canonical_angle_changed'] += abs((ac[4]-bc[4]+math.pi/2)%math.pi-math.pi/2)>1e-7
        protected['score_changed'] += ab[5]!=bb[5]
        transitions[str(int(a['joint_correct']))+'->'+str(int(b['joint_correct']))] += 1
        for edge in ('long','short'):
            changes[edge+'_log'].append(b[edge+'_log']-a[edge+'_log'])
        changes['scale_log'].append(b['scale_log']-a['scale_log'])
        changes['ratio_log'].append(b['ratio_log']-a['ratio_log'])
        patterns['both_shrunk'] += b['long_log']<a['long_log'] and b['short_log']<a['short_log']
        patterns['both_enlarged'] += b['long_log']>a['long_log'] and b['short_log']>a['short_log']
        patterns['ratio_improved_scale_worse'] += abs(b['ratio_log'])<abs(a['ratio_log']) and abs(b['scale_log'])>abs(a['scale_log'])
        patterns['size_abs_sum_improved'] += abs(b['long_relative'])+abs(b['short_relative']) < abs(a['long_relative'])+abs(a['short_relative'])
        patterns['size_abs_sum_worse'] += abs(b['long_relative'])+abs(b['short_relative']) > abs(a['long_relative'])+abs(a['short_relative'])
    out = dict(paired_outputs=len(pairs),joint_transitions=dict(transitions),
        changes={k:describe(v) for k,v in changes.items()},patterns=dict(patterns),
        protected_changes=dict(protected),correction_by_original_error={})
    if old=='M' and new=='C' and any(protected.values()):
        raise ValueError('Candidate changed protected center, canonical direction or score')
    predicates = dict(joint_correct=lambda s:s['joint_correct'],
        both_small_3pct=lambda s:max(s['long_relative'],s['short_relative'])<=-.03,
        both_large_3pct=lambda s:min(s['long_relative'],s['short_relative'])>=.03,
        mixed_sign=lambda s:s['long_log']*s['short_log']<0)
    for name,predicate in predicates.items():
        selected=[r for r in pairs if predicate(r['size'][old])]
        out['correction_by_original_error'][name] = dict(count=len(selected),
            mean_scale_change=describe([r['size'][new]['scale_log']-r['size'][old]['scale_log'] for r in selected]),
            correct_after=sum(r['size'][new]['joint_correct'] for r in selected),
            both_shrunk=sum(r['size'][new]['long_log']<r['size'][old]['long_log'] and r['size'][new]['short_log']<r['size'][old]['short_log'] for r in selected))
    if pairs and 'reliability' in pairs[0]:
        out['fixed_M_policy_ordered_bookkeeping'] = {}
        for component in ('center','size','angle'):
            states=[(r['reliability'][old]['good'][component],r['reliability'][new]['good'][component],
                r['reliability'][old]['accepted'][component],r['reliability'][new]['accepted'][component])
                for r in pairs if component!='angle' or r['angle_eligible']]
            out['fixed_M_policy_ordered_bookkeeping'][component]=ordered_accounting(states)
    if pairs and 'depth' in pairs[0]:
        usable=[r for r in pairs if r['depth'][old]['z_m'] is not None and r['depth'][new]['z_m'] is not None]
        delta=[abs(r['depth'][new]['z_m']-r['truth_z_m'])-abs(r['depth'][old]['z_m']-r['truth_z_m']) for r in usable]
        out['depth_abs_error_change_m']=describe(delta)
        out['depth_improved']=sum(v<0 for v in delta)
        out['depth_worse']=sum(v>0 for v in delta)
        # Equal-focal camera: center absent, angle cancels in Raw-opt.
        short_terms=[];ratio_terms=[];residual=[]
        for r in usable:
            a,b=r['depth'][old],r['depth'][new]
            short=- (r['size'][new]['short_log']-r['size'][old]['short_log'])
            ratio=r['beta']*(b['q_signed']**2-a['q_signed']**2)
            short_terms.append(short);ratio_terms.append(ratio)
            residual.append(math.log(b['z_m']/a['z_m'])-short-ratio)
        out['exact_log_depth_change']=dict(short_inverse=describe(short_terms),ratio_q=describe(ratio_terms),
            max_abs_identity_residual=max(map(abs,residual),default=0),note='Additive log depth change, not additive MAE attribution.')
    return out


def summarize(rows):
    stages=[s for s in STAGES if s in rows[0]['boxes']]
    out=dict(frames=len(rows),stages={s:geometry_group(rows,s) for s in stages},
        paired={a+'->'+b:stage_pair(rows,a,b) for a,b in (('B','M'),('M','C')) if a in stages and b in stages})
    if 'B' in stages and 'M' in stages:
        active=[r for r in rows if r['boxes']['M'] is not None]
        riou={s:describe([compute_riou(r['boxes'][s],r['gt']) for r in active]) for s in stages}
        riou['M_center_angle_B_size_offline']=describe([compute_riou(
            substitute_size_source(r['boxes']['B'],r['boxes']['M']),r['gt']) for r in active])
        out['size_source_counterfactual']=dict(role='Offline fixed source substitution, not a new delivered model or selection.',
            riou_frame_weighted=riou,note='Frame-weighted RIoU differs from the official evaluator average across videos. Canonical B sizes imply B size correctness; equal-focal Webots implies B depth exactly. No promotion or runtime claim.')
    if 'reliability' in rows[0]:
        for stage in stages:
            values=out['stages'][stage]
            values['fixed_M_policy']={}
            for component in ('center','size','angle'):
                eligible=[r for r in rows if component!='angle' or r['angle_eligible']]
                outputs=[r for r in eligible if r['size'][stage] is not None]
                counts=Counter(outcome(r['reliability'][stage]['good'][component],r['reliability'][stage]['accepted'][component]) for r in outputs)
                counts={k:counts[k] for k in ('FA','FR','ED','CR')}
                values['fixed_M_policy'][component]=dict(counts,eligible_frames=len(eligible),outputs=len(outputs),
                    accepted=counts['FA']+counts['CR'],
                    all_frame_correct_accepted_coverage=counts['CR']/len(eligible) if eligible else None,
                    raw_correct=sum(r['reliability'][stage]['good'][component] for r in outputs),
                    raw_correct_on_outputs=sum(r['reliability'][stage]['good'][component] for r in outputs)/len(outputs) if outputs else None)
    if 'depth' in rows[0]:
        for stage in stages:
            finite=[r for r in rows if r['depth'][stage]['z_m'] is not None]
            errors=[r['depth'][stage]['z_m']-r['truth_z_m'] for r in finite]
            absolute=[abs(v) for v in errors]
            out['stages'][stage]['depth']=dict(numeric=len(errors),numeric_coverage=len(errors)/len(rows),
                mae_m=float(np.mean(absolute)),rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
                bias_m=float(np.mean(errors)),p95_m=float(np.percentile(absolute,95)),max_m=max(absolute),
                within_m={str(t):sum(v<=t for v in absolute) for t in (.5,1.,2.)},
                q_in_fit_support=sum(r['q_support'][stage] for r in finite))
            terms=defaultdict(list)
            for r in finite:
                z=r['depth'][stage]['z_m'];zg=r['gt_depth']['z_m']
                zs=zg/(1+r['size'][stage]['short_relative'])
                zq=zg*math.exp(r['beta']*(r['depth'][stage]['q_signed']**2-r['gt_depth']['q_signed']**2))
                for key,value in zip(('GT_formula','short_scale','ratio_q','interaction'),
                        (zg-r['truth_z_m'],zs-zg,zq-zg,z-zs-zq+zg)):
                    terms[key].append(value)
            out['stages'][stage]['signed_bias_decomposition_m']={k:float(np.mean(v)) for k,v in terms.items()}
        gt_errors=[r['gt_depth']['z_m']-r['truth_z_m'] for r in rows]
        out['GT_formula_oracle']=dict(mae_m=float(np.mean(np.abs(gt_errors))),bias_m=float(np.mean(gt_errors)),
            note='Calibration/development/exposed diagnostic, not independent depth guarantee.')
        out['historical_four_arm_references']={}
        for key in ('eood','symeood'):
            errors=[r['references'][key]['z_m']-r['truth_z_m'] for r in rows if r['references'][key]['z_m'] is not None]
            out['historical_four_arm_references'][key]=dict(numeric=len(errors),mae_m=float(np.mean(np.abs(errors))),
                rmse_m=float(np.sqrt(np.mean(np.square(errors)))),bias_m=float(np.mean(errors)),
                within_1m=sum(abs(e)<=1 for e in errors),role='Original selected baseline, descriptive reference; not substituted for B/M/C.')
    return out


def port_row(meta, boxes, dataset, sample_role=None):
    return dict(id=meta['image'],sequence=meta['sequence'],domain=meta['domain'],dataset=dataset,
        frame_id=meta['frame_id'],gt=meta['gt'],image_size=meta['image_size'],
        angle_eligible=meta.get('angle_axis_well_defined',meta.get('train_angle_eligible',True)),
        sample_role=sample_role,boxes=boxes,size={s:sizes(b,meta['gt']) for s,b in boxes.items()})


def enrich_port(rows, policy):
    runtime=reliability.SimpleComponentReliability(policy)
    for r in rows:
        r['reliability']={}
        for s,b in r['boxes'].items():
            flags=runtime.decide(b,r['image_size'])
            errors=reliability.geometry_errors(r['gt'],b) if b is not None else None
            good=dict(center=errors['center_px']<15,size=errors['size_max_relative']<=.1,angle=errors['angle_deg']<=3) if errors else dict(center=False,size=False,angle=False)
            r['reliability'][s]=dict(good=good,accepted={c:flags[c+'_accepted'] for c in ('center','size','angle')})


def build(manifest):
    source={name:load(spec) for name,spec in manifest['inputs'].items()}
    collection=index(source['collection'],'image')
    train=[];half=[]
    for r in source['train_candidate']:
        meta=dict(collection[r['image']]);equal_box(meta['gt'],r['gt'],1e-4);meta['gt']=r['gt']
        if r['scale']==1:
            equal_box(r['b'],meta['b_original']);equal_box(r['midpoint'],meta['pred'])
        elif r['scale']!=.5:
            raise ValueError('Unexpected TRAIN view')
        target=train if r['scale']==1 else half
        target.append(port_row(meta,dict(B=r['b'],M=r['midpoint'],C=r['edge_residual']),
            'port_TRAIN' if r['scale']==1 else 'port_TRAIN_half_view',r['sample_role']))
    val=[port_row(r,dict(B=r['b_original'],M=r['pred']),'port_VAL') for r in collection.values() if r['reliability_role']=='val']
    expected_train={r['image'] for r in collection.values() if r['reliability_role']=='train'}
    if {r['id'] for r in train}!=expected_train or {r['id'] for r in half}!=expected_train:
        raise ValueError('TRAIN role/half-view identity differs')
    sealed=index(source['test_gt'],'image');metadata=index(source['test_metadata'],'image')
    test=[]
    for r in source['test_candidate']:
        g=sealed[r['image']];m=dict(metadata[r['image']]);m['gt']=g['gt']
        equal_box(r['b'],g['b']);equal_box(r['midpoint'],g['midpoint'])
        if list(reversed(r['original_size_hw'])) != m['image_size']:
            raise ValueError('Original image size differs')
        test.append(port_row(m,dict(B=r['b'],M=r['midpoint'],C=r['size_candidate']),'port_TEST'))
    if {r['id'] for r in test}!=set(sealed) or set(sealed)!=set(metadata):
        raise ValueError('TEST frame set differs')
    datasets=dict(port_TRAIN=train,port_TRAIN_half_view=half,port_VAL=val,port_TEST=test)
    policy=source['policy']['simple_policy']
    for rows in datasets.values():
        index(rows,'id');enrich_port(rows,policy)
    calibration=source['calibration'];beta=calibration['parameters']['beta']
    for name,spec in manifest['webots'].items():
        truth=index(source[name+'_truth'],'frame_index');m=source[name+'_manifest']
        intrinsics=m['camera']['intrinsics'];geometry=m['obb_reference_geometry']
        if intrinsics['fx']!=intrinsics['fy']:
            raise ValueError('Log short/q attribution requires equal focal lengths')
        rows=[]
        previous=source[name+'_four_arm'];candidate=index(source[name+'_candidate'],'frame_id') if name+'_candidate' in source else None
        for r in previous:
            i=int(r['frame_id'].split('_')[-1]);t=truth[i];gt=depth.truth_box(t)
            if abs(r['truth_z_m']-t['camera_geometry']['z_cg_opt_m'])>1e-9:
                raise ValueError('Metric truth identity differs')
            methods=r.get('methods',r)
            boxes=dict(B=methods['symeood_b']['box'],M=methods['symeood_b_midpoint']['box'])
            if candidate is not None:
                c=candidate[r['frame_id']];equal_box(gt,c['gt_box'],1e-4);equal_box(boxes['M'],c['midpoint']['box'])
                boxes['C']=c['size_candidate']['box']
            ds={s:depth.depth(b,intrinsics,geometry,calibration['parameters']) if b else dict(z_m=None,q_signed=None,status='missing') for s,b in boxes.items()}
            for s,key in (('B','symeood_b'),('M','symeood_b_midpoint')):
                if abs(ds[s]['z_m']-methods[key]['depth']['z_m'])>1e-8:
                    raise ValueError('Frozen Raw-opt replay differs')
            if candidate is not None and abs(ds['C']['z_m']-c['size_candidate']['depth']['z_m'])>1e-8:
                raise ValueError('Candidate Raw-opt replay differs')
            # Support identity is already sealed in source results; not a filter.
            qs={s:methods[key]['q_in_fit_support'] for s,key in (('B','symeood_b'),('M','symeood_b_midpoint'))}
            if candidate is not None:qs['C']=c['size_candidate']['q_in_fit_support']
            for s,d in ds.items():
                computed=d['q_signed'] is not None and calibration['parameters']['q_signed_min']<=d['q_signed']<=calibration['parameters']['q_signed_max']
                if computed!=qs[s]:raise ValueError('q support replay differs')
            rows.append(dict(id=r['frame_id'],dataset=name,sequence=name,domain='webots',frame_id=i,
                gt=gt,boxes=boxes,size={s:sizes(b,gt) for s,b in boxes.items()},depth=ds,q_support=qs,
                references={key:methods[key]['depth'] for key in ('eood','symeood')},
                truth_z_m=r['truth_z_m'],beta=beta,gt_depth=depth.depth(gt,intrinsics,geometry,calibration['parameters'])))
        index(rows,'id');datasets[name]=rows
    for name,count in manifest['expected_counts'].items():
        if len(datasets[name])!=count:raise ValueError('Frame count differs: '+name)
    return datasets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=ROOT/'tools/analysis/port_size_all_evidence_v1_inputs.json')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();manifest=json.loads(args.inputs.read_text());datasets=build(manifest)
    report=dict(protocol=VERSION,scope=manifest['scope'],identities=manifest['identities'],
        input_manifest_sha256=depth.sha(args.inputs),inputs=manifest['inputs'],datasets={})
    for name,rows in datasets.items():
        groups=dict(all=rows)
        for field in ('domain','sequence','sample_role'):
            for value in sorted({r.get(field) for r in rows if r.get(field) is not None}):
                groups[field+':'+value]=[r for r in rows if r.get(field)==value]
        report['datasets'][name]=dict(evidence_role=manifest['roles'][name],groups={k:summarize(v) for k,v in groups.items()})
    report['coverage']=dict(existing_frame_identities=sum(len(r) for n,r in datasets.items() if n!='port_TRAIN_half_view'),
        extra_transformed_train_views=len(datasets['port_TRAIN_half_view']),
        missing_candidate_evaluations=['port_VAL','calibration_train_03_tilt_obb'],
        independence='Frames are not independent events. Half views are not new frames; event independence was not established.')
    args.out.mkdir(parents=True,exist_ok=False)
    dump(args.out/'summary.json',report)
    with (args.out/'paired_rows.jsonl').open('x',encoding='utf-8') as f:
        for rows in datasets.values():
            for r in rows:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(report['coverage'],ensure_ascii=False))
    print('SIZE_ALL_EVIDENCE_OFFLINE_COMPLETE '+str(args.out))


if __name__=='__main__':main()
