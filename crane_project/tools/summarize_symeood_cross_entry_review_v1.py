"""CPU-only verification and synthesis of frozen cross-entry audit receipts.

No model, training data, TEST, selection changes, or numerical tolerance tuning.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

GROUPS=('100','101','110','111')
PHASES=('early','mature')
FIELDS=('input','rng_before','predictions','assignment_int','assignment_float','labels','losses',
        'logits_gradient','parameter_gradient','clip','momentum','parameters','buffers','counters','rng_after')


def rows(path):
    with path.open() as f:
        for line in f:
            if line.strip():yield json.loads(line)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path,data):
    with path.open('x') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in data))));w.writeheader();w.writerows(data)


def read_complete(root,mode,group,phase,rank):
    # Ignore the interrupted multi-model case; reuse only complete receipts.
    for folder in ('short_serial','short'):
        p=root/folder/mode/group/phase
        s=p/('summary_rank%d.json'%rank)
        if s.exists():return p,json.loads(s.read_text())
    raise AssertionError(('missing complete',mode,group,phase,rank))


def verify_s(row):
    return all([row['level_count']==5,row['single_batch_alpha'],not row['alpha_requires_grad'],
                row['G_to_actual_gradient_none'],row['per_level_formula_equal'],
                row['all_parameter_gradients_equal'],all(row['other_losses_equal'].values()),
                row['counter_equal'],row['classification_G_S_relative']<=4*2**-23,row['passed']])


def main(base,out):
    source=base/'receipts';out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'receipt_manifest.json').read_text())
    assert all(sha(source/p)==h for p,h in manifest.items())
    queue=json.loads((source/'queue_status.json').read_text());assert queue['state']=='complete'
    step_summary=[];firsts=[];hook_rows=[];nd_rows=[];resources=[]
    for mode in ('det','nondet'):
        for group in GROUPS:
            for phase in PHASES:
                for rank in (0,1):
                    case,summary=read_complete(source,mode,group,phase,rank)
                    assert summary['steps']==20
                    comps=list(rows(case/('comparisons_rank%d.jsonl'%rank)))
                    paths=('bare','full') if mode=='det' else ('old_repeat','bare','full')
                    assert len(comps)==20*len(paths)
                    assert collections.Counter((r['path'],r['step']) for r in comps)==collections.Counter((p,i) for p in paths for i in range(20))
                    for path in paths:
                        rr=[r for r in comps if r['path']==path];rr.sort(key=lambda r:r['step'])
                        assert all(r['fields']['input']['equal'] and r['fields']['rng_before']['equal'] and r['fields']['rng_after']['equal'] and r['fields']['counters']['equal'] for r in rr)
                        actual=next((r for r in rr if not r['equal']),None)
                        if mode=='det':assert actual is None and summary['all_steps_equal']
                        else:
                            assert actual is not None and actual['step']==0 and actual['first_field']=='parameter_gradient'
                            assert all(actual['fields'][k]['equal'] for k in FIELDS[:8])
                            for step in (0,10):
                                r=rr[step]
                                nd_rows.append(dict(group=group,phase=phase,rank=rank,path=path,step=step,
                                    gradient_max_abs=r['fields']['parameter_gradient']['global_max_abs'],
                                    parameter_max_abs=r['fields']['parameters']['global_max_abs'],
                                    reference_gradient_norm=r['fields']['parameter_gradient']['reference_norm'],
                                    gradient_max_over_reference_norm=r['fields']['parameter_gradient']['global_max_abs']/r['fields']['parameter_gradient']['reference_norm']))
                            firsts.append(dict(group=group,phase=phase,rank=rank,path=path,step=actual['step'],first_field=actual['first_field'],
                                gradient_max_abs=actual['fields']['parameter_gradient']['global_max_abs'],
                                parameter_max_abs=actual['fields']['parameters']['global_max_abs'],
                                first_prediction_difference=next((r['step'] for r in rr if not r['fields']['predictions']['equal']),None),
                                first_assignment_difference=next((r['step'] for r in rr if not r['fields']['assignment_int']['equal']),None),
                                first_loss_difference=next((r['step'] for r in rr if not r['fields']['losses']['equal']),None)))
                        step_summary.append(dict(mode=mode,group=group,phase=phase,rank=rank,path=path,steps=20,
                            equal_steps=sum(r['equal'] for r in rr),first_step=actual['step'] if actual else None,
                            first_field=actual['first_field'] if actual else None,
                            fields_equal_counts={k:sum(r['fields'][k]['equal'] for r in rr) for k in FIELDS}))
                    hh=list(rows(case/('hook_conservation_rank%d.jsonl'%rank)))
                    assert len(hh)==2 and all(r['passed'] and all(r['checks'].values()) and r['probe_empty'] and r['records_empty'] for r in hh)
                    hook_rows.extend(dict(mode=mode,group=group,phase=phase,rank=rank,**r) for r in hh)
                    ss=list(rows(case/('steps_rank%d.jsonl'%rank)))
                    expected_paths=3 if mode=='det' else 4
                    assert len(ss)==20*expected_paths
                    if 'allocated_mib' in ss[0]:resources.extend(dict(mode=mode,group=group,phase=phase,rank=rank,**r) for r in ss)
    bb=[r for p in sorted((source/'boundaries').glob('comparisons_rank*.jsonl')) for r in rows(p)]
    assert len(bb)==160 and all(r['equal'] for r in bb)
    ss=[r for p in sorted((source/'s_contract').glob('checks_rank*.jsonl')) for r in rows(p)]
    assert len(ss)==8 and all(verify_s(r) for r in ss)
    iden=json.loads((base/'identity.json').read_text());hist=json.loads((base/'historical_records.json').read_text())
    assert len(iden['common_frozen_sources'])==316 and not iden['changed_frozen_sources']
    assert len({r['parameters_sha256'] for r in iden['records']})==1
    assert len({r['buffers_sha256'] for r in iden['records']})==1
    for g in GROUPS:
        old=next(r for r in iden['records'] if r['role']=='old' and r['name']==g)
        new=next(r for r in iden['records'] if r['role']=='new' and r['old_group']==g)
        for k in ('constructor_parameters_sha256','constructor_cpu_rng_sha256','parameters_sha256','buffers_sha256','cpu_rng_sha256','parameter_names','init_trace'):
            assert old[k]==new[k],(g,k)
    assert all(all(p[k] for k in ('env_info_equal','lr_trace_equal','frozen_pretrained_portion_equal','iterations_equal')) for p in hist['pairs'])
    for p in hist['pairs']:
        for r in p['records']:
            assert r['checkpoint_iteration']==15360 and r['nfl_calls']==76800 and r['frozen_stem_and_layer1_equal_current_initial']
    selection=json.loads((base/'performance/selection_verification.json').read_text());assert len(selection['versions'])==8 and all(v['reproduced'] for v in selection['versions'])
    curves=list(csv.DictReader((base/'performance/performance_differences.csv').open()));curve_summary=[]
    for group in GROUPS:
        for domain in ('real','sim','all'):
            rr=[r for r in curves if r['pair']==group and r['domain']==domain and r['context']!='selected'];assert len(rr)==5
            curve_summary.append(dict(group=group,domain=domain,new_RIoU_better=sum(float(r['delta_riou'])>0 for r in rr),
                new_RIoU_worse=sum(float(r['delta_riou'])<0 for r in rr),new_DFR_better=sum(float(r['delta_DFR'])<0 for r in rr),
                new_DFR_worse=sum(float(r['delta_DFR'])>0 for r in rr),riou_delta_min=min(float(r['delta_riou']) for r in rr),riou_delta_max=max(float(r['delta_riou']) for r in rr)))
    write_csv(out/'short_path_summary.csv',step_summary);write_csv(out/'first_divergences.csv',firsts)
    write_csv(out/'numeric_full_steps0_10.csv',nd_rows);write_csv(out/'same_epoch_curve_summary.csv',curve_summary)
    (out/'hook_conservation.json').write_text(json.dumps(hook_rows,indent=2))
    (out/'S_contract_checks.json').write_text(json.dumps(ss,indent=2))
    direct=[]
    for p in sorted((source/'short_serial/nondet').glob('*/*/digests_rank*.jsonl')):
        records={}
        for r in rows(p):
            if r['path'] in ('bare','full'):
                records[r['path'],r['step']]={k:(v['sha256'] if isinstance(v,dict) and v.get('tensor') else v) for k,v in r.items() if k in FIELDS}
        differences=[]
        for step in range(20):
            a,b=records['bare',step],records['full',step]
            field=next((k for k in FIELDS if a[k]!=b[k]),None)
            if field:differences.append((step,field))
        assert differences and differences[0]==(0,'parameter_gradient')
        direct.append(dict(case=str(p.relative_to(source)),first=differences[0],unequal_steps=len(differences)))
    assert len(direct)==16
    (out/'bare_full_direct.json').write_text(json.dumps(direct,indent=2))
    inventory=[
        ('shared frozen source316','confirmed_same','Both old batches have identical316 common frozen files; current bytes match.'),
        ('common resolved config','confirmed_same','Loss/assignment/auxiliary/inference/data/optimizer/LR/batch/DDP same; wrapper types/hooks/work_dir differ.'),
        ('fresh initialization','confirmed_same','All10 CPU constructions/init parameters/buffers/RNG; four init traces exact.'),
        ('actual logged environment/LR/counts','confirmed_same','Eight logs environment and288 LR records equal;15360 updates;NFL76800.'),
        ('frozen pretrained portion','confirmed_same','66 stem/layer1 tensors identical across old/new/current initial; not whole historical init.'),
        ('wrappers/statistics/probe/recovery execution','changed_expected_inert_in_tested_deterministic_scope','640 deterministic comparisons,160 boundary comparisons exact;64 Hook state conservation events pass.'),
        ('nondeterministic execution and timing','potential_training_effect','Native repeat and both USG paths diverge first at raw parameter gradients; does not explain whole24epoch differences.'),
        ('historical complete init/pretrained whole-file hash','historical_evidence_missing','Current reconstruction/cache/frozen partial weights cannot recreate missing old initialization evidence.'),
        ('historical augmented input/order/batch/RNG','historical_evidence_missing','Current frozen prefixes match all paths; old complete input traces were not saved.'),
        ('full evaluator/logger hook execution','historical_evidence_missing','Production custom USG Hook is tested; actual online VAL and full24epoch runner history not replayed.'),
        ('S objective','intentional_training_change','Own per-rank batchalpha across fivelevels; no gradient throughG oralpha; other losses identical; no old S equivalent.')]
    write_csv(out/'training_identity_inventory.csv',[dict(item=a,classification=b,evidence=c) for a,b,c in inventory])
    result=dict(receipt_files_verified=len(manifest),selection_versions_verified=8,
        deterministic_comparisons=sum(r['steps'] for r in step_summary if r['mode']=='det'),
        deterministic_equal=sum(r['equal_steps'] for r in step_summary if r['mode']=='det'),
        nondeterministic_comparisons=sum(r['steps'] for r in step_summary if r['mode']=='nondet'),
        nondeterministic_first_cases=len(firsts),all_firsts_parameter_grad_step0=all(r['first_field']=='parameter_gradient' and r['step']==0 for r in firsts),
        boundary_comparisons=len(bb),boundary_equal=sum(r['equal'] for r in bb),hook_state_events=len(hook_rows),S_checks=len(ss),
        serial_peak_allocated_mib=max(r['peak_allocated_mib'] for r in resources),serial_peak_reserved_mib=max(r['peak_reserved_mib'] for r in resources),
        common_sources=316,historical_updates_each=15360,passed=True,TEST=False,depth=False,
        limits=['Current-source bounded diagnostic is not historical complete replay.',
                'Mature uses shared oldep24 weights/optimizer plus injected counts; not recovered historical schedule state.',
                'Low-VRAM serial CPU transfers change timing/residency; no full historical noise estimate.',
                'Exact byte SHA each step; numerical Linf is full only at steps0/10; bounded samples elsewhere.',
                'DDP parameter gradients checked after reduction; retained logits gradients are local.',
                'No24epoch rerun, no original checkpoint reselection or score substitution.'])
    (out/'analysis.json').write_text(json.dumps(result,indent=2))
    (out/'verification.json').write_text(json.dumps(dict(passed=True,receipt_manifest_sha256=sha(source/'receipt_manifest.json'),
        source_sha256=sha(Path(__file__)),outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.base.resolve(),a.out.resolve())
