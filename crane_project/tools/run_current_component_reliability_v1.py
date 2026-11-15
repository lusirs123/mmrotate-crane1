"""Fit the existing simple design to frozen R4/M20/L5 and evaluate three flags.

Fit uses native-scale TRAIN only. Size cutoffs use pooled VAL (887 frames);
angle fitting, calibration and quality evaluation use Sim only. TEST is read
only by the separate evaluate phase, after the policy has been saved.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from crane_project.utils import port_simple_component_reliability_v1 as simple

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'crane_project/data/crane_grab_port_day2night_v1'
INPUT = ROOT / 'work_dirs/component_reliability_20261115_v1/inputs/L5_all.rows.jsonl.gz'
INPUT_SHA = '0e17afb98958a7305a6b869aab13b4b125f64b46776b08b5ae5fd1aaac394ec4'
TEST = ROOT / ('work_dirs/symeood_r4_adaptation_20261031/readbacks/'
               'lvsr_fixed_test/lvsr/test/attempt_002/test_rows.jsonl')
TEST_SHA = '9a11d0b625f0a05e45d7f7bacd7e41ff60def9bcd1f31561d30bd412b64ad240'
COMPONENTS = {'center': 'center_px', 'size': 'size_max_relative', 'angle': 'angle_deg'}
LIMITS = {'center_px': 15., 'size_max_relative': .1, 'angle_deg': 3.}
COVERAGES = [1., .95, .9, .8, .5]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def load_rows(test=False):
    path, digest = (TEST, TEST_SHA) if test else (INPUT, INPUT_SHA)
    if sha(path) != digest:
        raise ValueError('Frozen row hash differs')
    manifest = json.loads((DATA/'manifest.json').read_text())
    roles = {'test'} if test else {'train', 'train_sim', 'val'}
    records = {r['id']: r for r in manifest['records'] if r['split'] in roles}
    if test:
        original = [json.loads(line) for line in path.read_text().splitlines()]
    else:
        with gzip.open(path, 'rt') as f:
            original = [r for r in map(json.loads, f) if r['scale'] == 1.]
    result = []
    seen = set()
    for r in original:
        name = r['image']
        if name in seen:
            raise ValueError('Duplicate native frame')
        seen.add(name)
        metadata = records[name]
        split = metadata['split']
        if not test and r['split'] != split:
            raise ValueError('Split role mismatch')
        image_path = DATA / metadata['images']['path']
        with Image.open(image_path) as image:
            image_size = list(image.size)
        pred = r['edge_residual']
        simple.canonical(r['gt'])
        simple.prediction(pred)
        if pred is not None:
            simple.descriptor(pred, image_size)
        if r['sequence'] != metadata['sequence']:
            raise ValueError('Sequence metadata differs')
        result.append(dict(image=name, sequence=r['sequence'], frame_id=r['frame_id'],
                           domain=r['domain'], split=split, gt=r['gt'], pred=pred,
                           image_size=image_size))
    if seen != set(records):
        raise ValueError('Missing/extra split frames')
    return sorted(result, key=lambda r: (r['sequence'], r['frame_id']))


def eligible(rows, component):
    return [r for r in rows if component != 'angle' or r['domain'] == 'sim']


def correct(row, component):
    if row['pred'] is None:
        return False
    error = simple.geometry_errors(row['gt'], row['pred'])[COMPONENTS[component]]
    limit = LIMITS[COMPONENTS[component]]
    return error < limit if component == 'center' else error <= limit


def max_gap(rows, accepted):
    best = run = 0
    previous = None
    for row in rows:
        if previous != (row['sequence'], row['frame_id'] - 1):
            run = 0
        run = 0 if row['image'] in accepted else run + 1
        best = max(best, run)
        previous = (row['sequence'], row['frame_id'])
    return best


def stats(rows, component, accepted):
    outputs = [r for r in rows if r['pred'] is not None]
    kept = [r for r in outputs if r['image'] in accepted]
    good = {r['image'] for r in outputs if correct(r, component)}
    errors = [simple.geometry_errors(r['gt'], r['pred'])[COMPONENTS[component]] for r in kept]
    cr = sum(r['image'] in good for r in kept)
    fa = len(kept) - cr
    return dict(
        frames=len(rows), outputs=len(outputs), accepted=len(kept),
        correct_accepted=cr, incorrect_accepted=fa,
        correct_rejected=len(good)-cr, incorrect_rejected=len(outputs)-len(good)-fa,
        missing=len(rows)-len(outputs),
        acceptance_coverage=len(kept)/len(rows),
        accepted_error_rate=fa/len(kept) if kept else None,
        correct_retention=cr/len(good) if good else None,
        all_frame_correct_coverage=cr/len(rows),
        error_mean=float(np.mean(errors)) if errors else None,
        error_p90=float(np.percentile(errors,90)) if errors else None,
        longest_unaccepted_frames=max_gap(rows,accepted),
        longest_no_accepted_correct_frames=max_gap(rows,accepted & good))


def risks(rows, component, method, policy):
    if method == 'raw' or component == 'center':
        return {r['image']: 0. for r in rows if r['pred'] is not None}
    values = {}
    for row in rows:
        if row['pred'] is None:
            continue
        value = 1-row['pred'][5]
        if method == 'simple':
            value = float(simple.linear_risk(
                policy['models'][component], simple.descriptor(row['pred'],row['image_size'])))
        values[row['image']] = value
    return values


def groups(rows):
    result = {'overall': rows}
    for field in ('domain','sequence'):
        result.update({field+'/'+v:[r for r in rows if r[field]==v]
                       for v in sorted({r[field] for r in rows})})
    return result


def fit(out):
    out.mkdir(exist_ok=False)
    spec = dict(
        model='R4 ep16 + M20 + dual LVSR ep5', input_sha256=INPUT_SHA,
        train='Native scale only: 2558 frames; no half-scale duplicates',
        center='retain_valid_B_output_no_extra_rejection',
        angle='Sim only for fit, cutoff calibration and reported quality',
        size='All TRAIN fit; one pooled 887-frame VAL cutoff',
        features=list(simple.FEATURES), fitting=dict(l2=.1,max_iterations=100,
        gradient_tolerance=1e-8), working_coverage=.95, error_limits=LIMITS,
        curves='all distinct risk boundaries; entire ties; no TEST threshold selection',
        display_coverages=COVERAGES, test_repeatedly_exposed=True,
        code_sha256=sha(__file__), utility_sha256=sha(Path(simple.__file__)),
        manifest_sha256=sha(DATA/'manifest.json'))
    write(out/'protocol.json',spec)
    rows=load_rows()
    train=[r for r in rows if r['split']!='val']
    val=[r for r in rows if r['split']=='val']
    assert len(train)==2558 and len(val)==887
    audit={}
    for split, data in [('train',train),('val',val)]:
        for group, rr in groups(data).items():
            for component in COMPONENTS:
                allowed=eligible(rr,component)
                if allowed:
                    audit[f'{split}/{group}/{component}']=stats(
                        allowed,component,{r['image'] for r in allowed if r['pred'] is not None})
    write(out/'support_audit.json',audit)
    policy=dict(models={}, cutoffs={})
    for component in ('size','angle'):
        rr=[r for r in eligible(train,component) if r['pred'] is not None]
        features=[simple.descriptor(r['pred'],r['image_size']) for r in rr]
        policy['models'][component]=simple.fit_linear_risk(
            features,[not correct(r,component) for r in rr],spec['fitting'])
        vr=eligible(val,component)
        policy['cutoffs'][component]={}
        for method in ('score_only','simple'):
            values=risks(vr,component,method,policy)
            policy['cutoffs'][component][method]=simple.coverage_cutoff(
                list(values.values()),len(vr),spec['working_coverage'])
    write(out/'policy.json',policy)
    if json.loads((out/'policy.json').read_text())!=policy:
        raise ValueError('Policy save/reload differs')
    write(out/'fit_lock.json',dict(
        policy_sha256=sha(out/'policy.json'),protocol_sha256=sha(out/'protocol.json'),
        native_train_frames=len(train),val_frames=len(val),model_forwards=0,
        detector_updates=0))
    print('FIT_LOCKED',json.dumps(policy['cutoffs']))


def evaluate(out):
    lock=json.loads((out/'fit_lock.json').read_text())
    spec=json.loads((out/'protocol.json').read_text())
    if (sha(out/'policy.json')!=lock['policy_sha256']
        or sha(out/'protocol.json')!=lock['protocol_sha256']
        or sha(__file__)!=spec['code_sha256']
        or sha(Path(simple.__file__))!=spec['utility_sha256']
        or sha(DATA/'manifest.json')!=spec['manifest_sha256']):
        raise ValueError('Frozen policy/protocol/source changed')
    if (out/'evaluation.json').exists():
        raise FileExistsError('Evaluation already exists')
    policy=json.loads((out/'policy.json').read_text())
    rows=load_rows(test=True)
    assert len(rows)==1440
    working={}
    curves={}
    display=[]
    # Global cutoffs are always applied unchanged within subgroup tables.
    for group,rr in groups(rows).items():
        for component in COMPONENTS:
            allowed=eligible(rr,component)
            if not allowed:
                continue
            key=group+'/'+component
            working[key]={}
            curves[key]={}
            for method in ('raw','score_only','simple'):
                values=risks(allowed,component,method,policy)
                threshold=0. if component=='center' or method=='raw' else (
                    policy['cutoffs'][component][method]['risk_le'])
                accepted={name for name,risk in values.items() if risk<=threshold}
                working[key][method]=stats(allowed,component,accepted)
                # Every curve uses only risk ranking. Subgroups are ranked
                # internally, separately from the fixed global working point.
                ordered=sorted(values.items(),key=lambda item:item[1])
                good={r['image'] for r in allowed if correct(r,component)}
                points=[]
                n=cr=0
                for index,(name,risk) in enumerate(ordered):
                    n+=1
                    cr+=name in good
                    if index+1==len(ordered) or ordered[index+1][1]!=risk:
                        points.append(dict(risk_max=risk,accepted=n,correct_accepted=cr,
                            incorrect_accepted=n-cr,coverage=n/len(allowed),
                            error_rate=(n-cr)/n))
                curves[key][method]=points
                for q in COVERAGES:
                    if not points:
                        continue
                    point=next((p for p in points if p['coverage']>=q),points[-1])
                    display.append(dict(group=group,component=component,method=method,
                        requested_all_frame_coverage=q,**point))
    write(out/'evaluation.json',dict(working=working,curves=curves,
          test_sha256=TEST_SHA,policy_sha256=lock['policy_sha256'],
          curve_role='descriptive internal subgroup ranking; not deployed cutoffs',
          model_forwards=0,detector_updates=0))
    with (out/'curve_display.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(display[0]))
        writer.writeheader()
        writer.writerows(display)
    print('TEST_EVALUATED',json.dumps({k:v for k,v in working.items() if k.startswith('overall/')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('fit','evaluate'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.phase=='fit':
        fit(args.out)
    else:
        evaluate(args.out)


if __name__=='__main__':
    main()
