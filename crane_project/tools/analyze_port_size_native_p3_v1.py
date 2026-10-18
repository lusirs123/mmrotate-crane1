#!/usr/bin/env python3
"""Local reproducible engineering analysis; saved TRAIN audit JSON only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main(root):
    root=Path(root);out=root/'analysis_20261018_v1'
    out.mkdir(exist_ok=False)
    verified={}
    for path in sorted(root.glob('audit_*/artifacts.json')):
        for name,expected in load(path)['files'].items():
            p=path.parent/name;actual=sha(p)
            if actual!=expected:raise ValueError('Artifact SHA mismatch: '+str(p))
            verified[str(p.relative_to(root))]=actual
    r=load(root/'audit_20261018_v3/engineering.json')
    if r['status']!='ENGINEERING_PASS' or r['evaluated_views']!=12:raise ValueError('Incomplete engineering')
    post=load(root/'server_postflight_20261018.json')
    if not post['all_pass']:raise ValueError('Server preservation failed')
    rows=[]
    for v in r['views']:
        row={k:v[k] for k in ('image','domain','sequence','scale')}
        row['missing']=int(v.get('missing',False))
        if not row['missing']:
            a,b=[v['timings'][k] for k in ('cached_p3','native_p3')]
            row.update(old_source_cells=v['old_roi_source_cells'],native_source_cells=v['native_point_source_cells'],
                extra_source_cells=v['extra_source_cells'],old_sampling_rank=v['linear_basis']['old_rank'],
                new_information_witness=int(v['linear_basis']['witness']),
                old_null_response=v['linear_basis']['old_null_response'],native_null_response=v['linear_basis']['native_response'],
                feature_mean_abs_difference=v['feature_difference_mean_abs'],roi_cache_max_abs_difference=v['roi_cache_max_abs'],
                cached_mean_ms=a['mean_ms'],native_mean_ms=b['mean_ms'],delta_ms=b['mean_ms']-a['mean_ms'],
                cached_incremental_peak_mib=a['incremental_allocator_peak_mib'],native_incremental_peak_mib=b['incremental_allocator_peak_mib'])
        rows.append(row)
    fields=list(dict.fromkeys(k for r0 in rows for k in r0))
    with (out/'view_evidence.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    present=[v for v in rows if not v['missing']]
    group={}
    for scale in (1.,.5):
        rr=[v for v in rows if v['scale']==scale];pp=[v for v in rr if not v['missing']]
        group[str(scale)]=dict(views=len(rr),present=len(pp),missing=len(rr)-len(pp),
            new_information=sum(v['new_information_witness'] for v in pp),
            no_new_information_witness=sum(not v['new_information_witness'] for v in pp))
    costs={a:statistics.mean(v[a+'_mean_ms'] for v in present) for a in ('cached','native')}
    delta=[v['delta_ms'] for v in present]
    memory=max(v['native_incremental_peak_mib']-v['cached_incremental_peak_mib'] for v in present)
    checks=all(v['roi_cache_max_abs_difference']==0 for v in present)
    assert checks and len(present)==11 and sum(v['new_information_witness'] for v in present)==7
    expected_backbone=[[1,256,256,256],[1,512,128,128],[1,1024,64,64],[1,2048,32,32]]
    expected_fpn=[[1,256,z,z] for z in (128,64,32,16,8)]
    assert all(v['backbone_shapes']==expected_backbone and v['fpn_shapes']==expected_fpn for v in r['views'])
    phase=[load(root/(n+'/engineering.json')) for n in ('audit_20261018_v1','audit_20261018_v2','audit_20261018_v3')]
    analysis=dict(status='FUNCTIONAL_ENGINEERING_PASS_NO_EFFECT_CLAIM',views=12,present=11,missing=1,
        information_by_scale=group,source_shapes=dict(backbone=expected_backbone,fpn=expected_fpn),
        runtime=r['runtime'],weights={k:v for k,v in r['input_sha256'].items() if k.endswith('.pth')},
        resource=dict(physical_gpu=r['resource']['physical_gpu'],process_peak_polled_mib=max(x['resource']['process_peak_polled_mib'] for x in phase),
            allocator_peak_mib=max(x.get('whole_run_allocated_peak_mib',0) for x in phase),
            final_run_reserved_peak_mib=r['allocator_reserved_mib'],user_cap_mib=2048,
            phase_actual_extractions=[x['feature_extractions'] for x in phase],unique_view_extractions=12,training_updates=0),
        timing=dict(cached_mean_ms=costs['cached'],native_mean_ms=costs['native'],mean_delta_ms=statistics.mean(delta),
            delta_min_ms=min(delta),delta_max_ms=max(delta),
            incremental_allocator_delta_mib=memory,nominal_mean_latency_budget_pass=statistics.mean(delta)<=5,
            allocator_budget_pass=memory<=128,
            limitation='Shared GTX1080 with other user workload;2 warmups+5 repeats/reader/view, A then B. CUDA read+head only, excludes B/M, I/O, and full pipeline. Not a stable isolated latency estimate.'),
        storage=dict(raw_point_mib_per_output=256*132*9*4/2**20,
            raw_point_gib_per_arm_full_train_val=(5112+886)*256*132*9*4/2**30,
            raw_point_gib_two_arms_full_train_val=2*(5112+886)*256*132*9*4/2**30,
            full_p3_mib_per1024_view=16.,whole_p3_gib_6003_views=6003*16/1024,
            limitation='Float32 tensor-only estimates, not measured serialized dataset size; no whole dataset cache created this audit.'),
        decision='Retain one native-P3 vs cached-P3 fresh common-readout contract. Evidence supports a real input difference in a subset, not feature insufficiency or beneficial corrections. No multiscale/training/TEST/depth launch.',
        limits=dict(val_images=0,test_images=0,depth_images=0,accuracy_metrics_computed=False,
            future_training_runner_connected=False,common_depth_adoption_available=False),
        repair_history=['v1 uninitialized cleanup variable with no extra cells, fixed',
            'v2 expected original half-scale MISSING stopped too strictly, fixed;10 complete views reused after hashes,2 remaining checked in v3'],
        verified_files=verified,server_preservation=post['all_pass'],analyzer_sha256=sha(__file__))
    write(out/'analysis.json',analysis)
    write(out/'artifacts.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps({k:analysis[k] for k in ('status','information_by_scale','resource','timing','storage')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    main(p.parse_args().root)
