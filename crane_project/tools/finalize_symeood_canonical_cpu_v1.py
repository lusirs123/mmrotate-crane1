"""Finish inspectable CPU-only plans; native GPU checks remain unscheduled."""
import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'crane_project/tools'));sys.path.insert(0,str(ROOT))


def config_closure(path,seen=None):
    seen=set() if seen is None else seen;path=path.resolve()
    if path in seen:return seen
    path.relative_to(ROOT);seen.add(path)
    for n in ast.parse(path.read_text()).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_base_' for t in n.targets):
            values=ast.literal_eval(n.value);values=[values] if isinstance(values,str) else values
            for name in values:config_closure(path.parent/name,seen)
    return seen


def main(base):
    import torch
    from mmcv import Config
    from crane_project.utils.symeood_repro_trace_v1 import sha,tensor_digest
    from prepare_symeood_b111_canonical_v1 import diff,B_CONFIG,ALIAS_CONFIG
    from ckpt_sweep import SELECTION_CONFIG
    assert __import__('os').environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    lock=json.loads((base/'runtime_lock_CPU_PENDING.json').read_text());assert lock['state']=='CPU_PREPARED_GPU_ACCEPTANCE_PENDING' and not lock['formal_schedule_frozen']
    identity=[json.loads((base/('cpu_identity_rank%d.json'%r)).read_text()) for r in (0,1)]
    assert all(x['passed'] for x in identity)
    all_records=[r for x in identity for r in x['records']];assert len(all_records)==6
    assert len({r['complete_state_sha256'] for r in all_records})==1
    for p,h in lock['source_sha256'].items():assert sha(ROOT/p)==h,p
    for p,h in lock['candidate_library_sha256'].items():assert sha(p)==h,p
    assert sha(base/'complete_initial_B_seed0.pth')==lock['complete_initial_file_sha256']
    initial=torch.load(base/'complete_initial_B_seed0.pth',map_location='cpu')
    assert tensor_digest(initial['state_dict'].items())==all_records[0]['complete_state_sha256']
    assert initial['pretrained_sha256']==lock['pretrained']['sha256'];del initial
    for x in identity:
        for first in x['first_augmented_batches']:assert sha(first['path'])==first['sha256']
    actual_plans=[]
    for alias in ('B','111'):
        folder=base/'plans'/(alias+'_seed0');plan=json.loads((folder/'plan.json').read_text());assert not plan['launch_allowed']
        assert plan['base_config']==str(B_CONFIG) and plan['method']=='complete_SymEOOD_scaleB'
        actual_plans.append(Config.fromfile(plan['execution_config']).to_dict())
    assert diff(*actual_plans)==['.work_dir']
    extra_classes={}
    for name,module,klass in all_records[0]['module_types']:
        if name in ('backbone','neck'):
            cls=getattr(importlib.import_module(module),klass);p=Path(inspect.getfile(cls))
            extra_classes[name]=dict(module=module,name=klass,file=str(p),sha256=sha(p))
    base_cfg=Config.fromfile(str(B_CONFIG));trace_cfg=base/'deferred_native_trace.py';assert not trace_cfg.exists()
    imports=list(base_cfg.custom_imports.imports)+['crane_project.utils.symeood_repro_trace_v1']
    trace_cfg.write_text('_base_ = '+repr([str(B_CONFIG)])+'\ncustom_imports = '+repr(dict(imports=imports,allow_failed_imports=False))+
        '\ncustom_hooks = '+repr([dict(type='SymEOODReproTraceV1',priority='LOWEST',engineering=True)])+'\n')
    commands=[]
    for i,(label,cfg) in enumerate([('B',B_CONFIG),('111',ALIAS_CONFIG),('B_trace',trace_cfg)]):
        work=base/'deferred_native_runs'/label;assert not work.exists()
        commands.append(dict(label=label,environment=dict(CUDA_VISIBLE_DEVICES='2,3',PYTHONPATH=str(ROOT)),
            argv=[sys.executable,'-m','torch.distributed.launch','--nproc_per_node=2','--master_port='+str(29666+i),
                'crane_project/tools/native_symeood_b111_entry_check_v1.py',str(cfg),'--execute-native-check',
                '--launcher','pytorch','--seed','0','--deterministic','--no-validate','--work-dir',str(work),'--cfg-options','runner.max_epochs=2'],
            disposable_updates=4,executed=False))
    (base/'deferred_native_commands.json').write_text(json.dumps(dict(state='NOT_AUTHORIZED_THIS_TURN',commands=commands,
        original_native_path=True,CPU_saved_tensor_offload=False,limited_loader_engineering_only=True,
        requires_future_user_GPU_arrangement=True,formal_training=False),indent=2))
    validation=dict(online=base_cfg.evaluation,candidate_epochs=[16,18,20,22,24],center_thresh_px=15,
                    selection=SELECTION_CONFIG,implementation='crane_project/tools/ckpt_sweep.py',implementation_sha256=sha(ROOT/'crane_project/tools/ckpt_sweep.py'))
    tools=['crane_project/tools/prepare_symeood_b111_canonical_v1.py','crane_project/tools/prepare_symeood_canonical_plan_v1.py',
           'crane_project/tools/native_symeood_b111_entry_check_v1.py','crane_project/tools/finalize_symeood_canonical_cpu_v1.py','crane_project/utils/symeood_repro_trace_v1.py']
    config_sources={str(p.relative_to(ROOT)):sha(p) for start in (B_CONFIG,ALIAS_CONFIG) for p in config_closure(start)}
    result=dict(state='CPU_READY_NATIVE_GPU_PENDING',CPU_identity_cases=6,augmented_input_batches=24,first_batches_saved=2,
        canonical_plan_aliases_same_except_workdir=True,complete_initial_state_sha256=all_records[0]['complete_state_sha256'],
        parameters_sha256=all_records[0]['parameter_sha256'],additional_class_sources=extra_classes,
        configuration_closure=config_sources,validation=validation,
        source_sha256={p:sha(ROOT/p) for p in tools},runtime_lock_sha256=sha(base/'runtime_lock_CPU_PENDING.json'),
        package_versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')},
        server_git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        updates=0,model_forward_calls=0,model_backward_calls=0,GPU_initialized=torch.cuda.is_initialized(),
        native_4step_GPU_checks_run=False,trace_hook_GPU_runtime_validated=False,performance_repeatability_established=False,
        run_count_or_seed_schedule_selected=False,formal_training_started=False,
        prior_obsolete_gate='CPU saved-tensor offload is not accepted as the native-GPU primary gate',
        preparation_repairs=['auxiliary attribute corrected to actual aux_heads','relative __file__ metadata path resolved; passing CPU receipts reused without rerun'])
    assert not torch.cuda.is_initialized()
    (base/'CPU_PREPARATION_FINAL.json').write_text(json.dumps(result,indent=2,default=str))
    print('CPU_ONLY_READY','native_GPU_pending','identity6','input24','zero updates')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();main(a.base.resolve())
