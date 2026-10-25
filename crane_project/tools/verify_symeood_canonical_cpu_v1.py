"""Local standard-library verification of downloaded CPU-only receipts."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(base):
    source=base/'receipts';manifest=json.loads((source/'receipt_manifest.json').read_text())
    assert all(sha(source/p)==h for p,h in manifest.items())
    final=json.loads((source/'CPU_PREPARATION_FINAL.json').read_text());lock=json.loads((source/'runtime_lock_CPU_PENDING.json').read_text())
    assert final['state']=='CPU_READY_NATIVE_GPU_PENDING' and lock['state']=='CPU_PREPARED_GPU_ACCEPTANCE_PENDING'
    assert not final['GPU_initialized'] and final['updates']==final['model_forward_calls']==final['model_backward_calls']==0
    for k in ('native_4step_GPU_checks_run','trace_hook_GPU_runtime_validated','performance_repeatability_established','formal_training_started','run_count_or_seed_schedule_selected'):assert not final[k]
    assert not lock['formal_schedule_frozen'] and lock['repetition_seeds'] is None
    assert sha(source/'runtime_lock_CPU_PENDING.json')==final['runtime_lock_sha256']
    all_records=[];matched_inputs=0
    keys=('constructor_state_sha256','constructor_rng_sha256','complete_state_sha256','parameter_sha256','buffer_sha256',
          'post_initialization_rng_sha256','init_calls','classes','parameter_names','module_types','all_state_devices')
    for rank in (0,1):
        identity=json.loads((source/('cpu_identity_rank%d.json'%rank)).read_text());assert identity['passed']
        records=identity['records'];assert [r['role'] for r in records]==['B','111','B_trace']
        for record in records:
            assert record['rank']==rank and record['all_state_devices']==['cpu']
            for k in keys:assert record[k]==records[0][k],(rank,record['role'],k)
        all_records.extend(records)
        inputs=json.loads((source/('cpu_inputs_rank%d.json'%rank)).read_text());assert len(inputs)==12
        for step in range(4):
            rr=[r for r in inputs if r['step']==step];assert len(rr)==3
            for record in rr:
                assert record['rank']==rank and record['combined_sha256']==rr[0]['combined_sha256'] and record['metadata']==rr[0]['metadata']
                assert len(record['metadata'])==2
            matched_inputs+=len(rr)
    assert len({r['complete_state_sha256'] for r in all_records})==1
    inv=json.loads((source/'TRAIN_VAL_inventory.json').read_text());counts=collections.Counter(x['split'] for x in inv)
    assert counts==dict(train=2558,val=887) and sha(source/'TRAIN_VAL_inventory.json')==lock['dataset_inventory_sha256']
    for alias in ('B','111'):
        plan=json.loads((source/'plans'/(alias+'_seed0')/'plan.json').read_text())
        assert plan['alias']==alias and not plan['launch_allowed'] and plan['base_config']==lock['canonical_config']
        assert sha(source/'plans'/(alias+'_seed0')/'canonical_execution.py')==plan['execution_config_sha256']
    deferred=json.loads((source/'deferred_native_commands.json').read_text());assert deferred['state']=='NOT_AUTHORIZED_THIS_TURN' and not deferred['CPU_saved_tensor_offload']
    assert all(not x['executed'] and x['disposable_updates']==4 for x in deferred['commands'])
    result=dict(passed=True,receipt_files=len(manifest),CPU_instances=6,input_tensor_batches=matched_inputs,
        distinct_rank_step_batches=8,all_state_cpu=True,complete_initial_state_sha256=all_records[0]['complete_state_sha256'],
        pretrained_sha256=lock['pretrained']['sha256'],TRAIN_VAL_inventory=counts,CPU_mapped_libraries=len(lock['runtime']['libraries']),
        candidate_libraries=len(lock['candidate_library_sha256']),GPU_acceptance=False,performance_repeatability=False,
        GPU_training_started=False,forward_calls=0,backward_calls=0,updates=0,source_sha256=sha(Path(__file__)))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();result=verify(a.base.resolve())
    (a.base/'local_CPU_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
