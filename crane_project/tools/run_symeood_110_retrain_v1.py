"""One110 seed0 native24epoch run, original VAL selection, one TEST, backups.

Never resumes, retries, changes model code, deletes weights, runs111/downstream,
or interprets a performance difference as a reason to train again.
"""
import argparse
import collections
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'crane_project/tools'))
OLD = ROOT / 'work_dirs/symeood_c_pair_seed0_20261025_v2'
LOCK_SHA = '4642a5060613a759db629afc4797a834205d9ed50e32b696667369e0372682ab'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def environment(old):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    for k, v in old['launch_environment'].items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    env['CUDA_VISIBLE_DEVICES'] = '0,1'
    return env


def assert_config_identity(old, new):
    a, b = copy.deepcopy(old), copy.deepcopy(new)
    a.pop('work_dir'); b.pop('work_dir')
    a['custom_hooks'][0].pop('runtime_lock')
    b['custom_hooks'][0].pop('runtime_lock')
    assert a == b, 'Only work_dir and runtime_lock may change'


def prepare(base):
    from mmcv import Config
    from ckpt_sweep import SELECTION_CONFIG
    assert sha(OLD / 'protocol.json') == '6130b8886c9dfc679492c12106a65c8f6b5daa4daed6f81a94c17c6f85d615b0'
    old = read(OLD / 'protocol.json')
    assert sha(OLD / 'configs/Coff_seed0.py') == old['config_sha256']['110']
    for p, h in old['sources'].items():
        assert sha(ROOT / p) == h, p
    for r in old['data_inventory']:
        for k in ('image', 'annotation'):
            assert sha(ROOT / r[k]) == r[k + '_sha256'], r[k]
    assert sha(old['pretrained']['path']) == old['pretrained']['sha256']
    original_lock = Path(old['native_gate_path']) / 'native_runtime_lock.json'
    assert sha(original_lock) == LOCK_SHA
    base.mkdir(exist_ok=False)
    (base / 'configs').mkdir()
    shutil.copyfile(original_lock, base / 'original_runtime_lock.json')
    derived = read(original_lock)
    derived['runtime']['environment']['CUDA_VISIBLE_DEVICES'] = '0,1'
    derived['physical_gpus'] = [0, 1]
    derived['derived_identity'] = dict(original_path=str(original_lock), original_sha256=LOCK_SHA,
        changes=['physical_gpus', 'runtime.environment.CUDA_VISIBLE_DEVICES'],
        meaning='Historical accepted reference adapted only to explicit GPU0,1 mapping; new-process gate is separate')
    save(base / 'runtime_identity_gpus01.json', derived)
    cfg = Config.fromfile(str(OLD / 'configs/Coff_seed0.py'))
    previous = cfg.to_dict()
    cfg.work_dir = str(base / '110')
    cfg.custom_hooks[0].runtime_lock = str(base / 'runtime_identity_gpus01.json')
    assert_config_identity(previous, cfg.to_dict())
    assert cfg.runner.max_epochs == 24 and cfg.load_from is None and cfg.resume_from is None
    assert cfg.data.samples_per_gpu == cfg.data.workers_per_gpu == 2
    cfg.dump(str(base / 'configs/Coff_seed0.py'))
    extras = ['run_symeood_110_retrain_v1.py', 'check_symeood_110_no_update_v1.py',
              'analyze_symeood_110_retrain_v1.py', 'audit_symeood_cross_entry_performance_v1.py',
              'audit_symeood_c_existing_results_v1.py', 'analyze_symeood_ablation_first4_v1.py',
              'audit_symeood_c_pair_completion_v1.py']
    sources = dict(old['sources'])
    sources.update({'crane_project/tools/' + p: sha(ROOT / 'crane_project/tools' / p) for p in extras})
    # Old missing checkpoint bytes are never required or reconstructed.
    predictions = {}
    for f in [OLD / '110/ckpt_sweep/sweep_results.json'] + list((OLD / '110/ckpt_sweep/final_test').glob('*/final_test_metrics_v2.json')):
        predictions[str(f)] = sha(f)
        report = read(f)
        for cp in report.get('all_checkpoints', {'test': report}).values():
            assert sha(cp['results_pkl']) == cp['results_pkl_sha256']
            predictions[cp['results_pkl']] = cp['results_pkl_sha256']
    contract = dict(state='FROZEN',protocol='symeood_110_single_retrain_seed0_native_v1',
        groups=['110'],formal_runs=1,seed=0,epochs=24,world_size=2,physical_gpus=[0,1],per_rank_batch=2,
        old_protocol_sha256=sha(OLD/'protocol.json'),old_selected='epoch_20',
        config=str(base/'configs/Coff_seed0.py'),config_sha256=sha(base/'configs/Coff_seed0.py'),
        runtime_lock=str(base/'runtime_identity_gpus01.json'),runtime_lock_sha256=sha(base/'runtime_identity_gpus01.json'),
        original_runtime_lock_sha256=LOCK_SHA,sources=sources,pretrained=old['pretrained'],
        data_inventory=old['data_inventory'],data_verified_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        old_prediction_artifacts=predictions,selection=dict(SELECTION_CONFIG),candidate_epochs=[16,18,20,22,24],
        online_VAL_epochs=list(range(2,25,2)),launch_environment=old['launch_environment'],
        fixed_flags=derived['formal_flags'],no_deterministic_override=True,
        gate='one disposable real no-grad loss forward per rank,0 backward,0 optimizer steps; fresh formal process',
        training='unchanged tools/train.py native DDP FP32, original trace hook only; no offload or new loss',
        TEST='One new original-VAL-selected checkpoint only, frozen before TEST; no score retry',
        interpretation='Report old/new selected, all five same-epoch differences and fixedep24; no posthoc pass threshold or multi-seed claim',
        downstream=False,auto_resume=False,retry=False,delete_weights=False,
        backup='retain all candidate and final recovery weights; selected copied and SHA-checked on project disk and /home filesystem',
        gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free,driver_version','--format=csv,noheader'],text=True))
    save(base/'protocol.json',contract)
    print('PREPARED_ONLY_NO_FORMAL_START',base,flush=True)


def native_command(config, work, gate=False):
    return [sys.executable,'-m','torch.distributed.launch','--nproc_per_node=2','--master_port=29684',
            'crane_project/tools/check_symeood_110_no_update_v1.py' if gate else 'tools/train.py',
            str(config),'--launcher','pytorch','--seed','0','--work-dir',str(work)]


def integrity(base,c):
    for p,h in c['sources'].items():
        assert sha(ROOT/p)==h,('source drift',p)
    assert sha(c['config'])==c['config_sha256']
    assert sha(c['runtime_lock'])==c['runtime_lock_sha256']
    assert sha(base/'original_runtime_lock.json')==LOCK_SHA


def status(base,**values):
    tmp=base/'status.tmp';tmp.write_text(json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values),indent=2))
    tmp.replace(base/'status.json')


def checked(base,c,stage,cmd,monitor=False):
    integrity(base,c);status(base,state='running',stage=stage,command=cmd)
    env=environment(c)
    with (base/(stage+'.log')).open('x') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                if monitor:
                    validate_train_logs(base/'110',partial=True)
                time.sleep(15)
            assert process.returncode==0,(stage,process.returncode)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise


def validate_train_logs(work,partial=False):
    rr=[]
    for p in work.glob('*.log.json'):
        for line in p.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:
                if partial:continue
                raise
            if r.get('mode') in ('train','val'):
                assert all(math.isfinite(v) for v in r.values() if isinstance(v,(int,float))),('nonfinite',r)
            rr.append(r)
    old=OLD/'110'
    for rank in (0,1):
        f=work/('repro_initial_rank%d.jsonl'%rank)
        if f.exists() and f.stat().st_size:
            current=json.loads(f.read_text().splitlines()[0])
            previous=json.loads((old/f.name).read_text().splitlines()[0])
            assert current['initial_state_sha256']==previous['initial_state_sha256']
        if not partial:
            lines=[json.loads(s) for s in (work/('repro_inputs_rank%d.jsonl'%rank)).read_text().splitlines()]
            assert [r['iteration'] for r in lines]==list(range(15360))
            assert collections.Counter(r['epoch'] for r in lines)=={e:640 for e in range(1,25)}
            conserved=[json.loads(s) for s in (work/('repro_conservation_rank%d.jsonl'%rank)).read_text().splitlines()]
            assert len(conserved)==24 and all(r['passed'] for r in conserved)
    if not partial:
        assert {r['epoch'] for r in rr if r.get('mode')=='train'}==set(range(1,25))
        assert {r['epoch'] for r in rr if r.get('mode')=='val'}==set(range(2,25,2))


def gate(base):
    c=read(base/'protocol.json')
    checked(base,c,'gate_no_update',native_command(c['config'],base/'gate_no_update',True))
    rows=[]
    for rank in (0,1):
        r=read_line(base/'gate_no_update'/('no_update_rank%d.jsonl'%rank))
        assert r['passed'] and r['optimizer_updates']==r['backward_calls']==0
        assert r['runtime']['environment']['CUDA_VISIBLE_DEVICES']=='0,1'
        rows.append(r)
    save(base/'gate_acceptance.json',dict(passed=True,physical_gpus=[0,1],formal_training=False,
        optimizer_updates=0,backward_calls=0,rows=rows,limit='First real batch only, not full24epoch reproduction'))
    status(base,state='gate_passed',formal_training=False,optimizer_updates=0)


def read_line(path):
    lines=Path(path).read_text().splitlines();assert len(lines)==1
    return json.loads(lines[0])


def backup_selected(base,selected):
    work=base/'110';source=work/(selected+'.pth');h=sha(source)
    destinations=[base/'model_backups',Path('/home/omnisky/symeood_model_backups')/base.name]
    copies=[]
    for dest in destinations:
        dest.mkdir(parents=True,exist_ok=False)
        p=dest/'selected_110.pth'
        with source.open('rb') as src,p.open('xb') as dst:shutil.copyfileobj(src,dst)
        assert sha(p)==h
        copies.append(dict(path=str(p),sha256=h,device=p.stat().st_dev,inode=p.stat().st_ino))
    assert copies[0]['device']!=copies[1]['device'],'Backups must use independent filesystems'
    resume=work/'repro_state_epoch24.pth';assert resume.is_file()
    target=destinations[1]/'repro_state_epoch24.pth'
    with resume.open('rb') as a,target.open('xb') as b:shutil.copyfileobj(a,b)
    assert sha(target)==sha(resume)
    save(base/'backup_receipt.json',dict(selected=str(source),sha256=h,copies=copies,
        retained_candidates={str(e):sha(work/('epoch_%d.pth'%e)) for e in (16,18,20,22,24)},
        recovery_backup=dict(path=str(target),sha256=sha(target))))


def strict_load_selected(base,selected):
    import torch
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    cfg=Config.fromfile(read(base/'protocol.json')['config']);import_modules_from_strings(**cfg.custom_imports)
    cp=torch.load(str(base/'110'/(selected+'.pth')),map_location='cpu')
    model=build_detector(copy.deepcopy(cfg.model));keys=model.load_state_dict(cp['state_dict'],strict=True)
    assert not keys.missing_keys and not keys.unexpected_keys and cp.get('optimizer')
    save(base/'selected_strict_load.json',dict(passed=True,checkpoint=selected,missing_keys=[],unexpected_keys=[],
        epoch=cp['meta']['epoch'],optimizer_state_present=True,device='cpu'))
    del model,cp


def run(base):
    c=read(base/'protocol.json');assert c['groups']==['110'] and c['formal_runs']==1 and c['epochs']==24
    assert read(base/'gate_acceptance.json')['passed']
    save(base/'queue_claim.json',dict(pid=os.getpid(),formal_runs=1,automatic_retry=False))
    try:
        assert not (base/'110').exists()
        checked(base,c,'train',native_command(c['config'],base/'110'),monitor=True)
        validate_train_logs(base/'110')
        args=[sys.executable,'crane_project/tools/ckpt_sweep.py','--config',c['config'],'--work-dir',str(base/'110'),
              '--gpu','1','--center-thresh','15','--mcml-limit','5']
        checked(base,c,'VAL_select',args+['--epochs','16','18','20','22','24'])
        from audit_symeood_cross_entry_performance_v1 import selector
        sweep=base/'110/ckpt_sweep/sweep_results.json';selected=selector(sweep)['selected']
        save(base/'selected_before_TEST.json',dict(selected=selected,sweep_sha256=sha(sweep),
            checkpoint_sha256=sha(base/'110'/(selected+'.pth')),TEST_accessed_by_new_model=False))
        backup_selected(base,selected);strict_load_selected(base,selected)
        checked(base,c,'VAL_analysis',[sys.executable,'crane_project/tools/analyze_symeood_110_retrain_v1.py','--base',str(base)])
        assert read(base/'selected_before_TEST.json')['sweep_sha256']==sha(sweep)
        checked(base,c,'TEST',args+['--final-test-from',str(sweep)])
        checked(base,c,'final_analysis',[sys.executable,'crane_project/tools/analyze_symeood_110_retrain_v1.py','--base',str(base),'--include-test'])
        integrity(base,c)
        status(base,state='complete',stage='all_done',formal_runs=1,epochs=24,selected=selected,backups_verified=True,downstream=False)
        bundle(base)
    except BaseException as e:
        status(base,state='stopped_on_error',error=repr(e),retry=False);raise


def bundle(base):
    paths=[p for p in base.rglob('*') if p.is_file() and p.suffix not in ('.pth','.gz')]
    manifest={str(p.relative_to(base)):sha(p) for p in paths}
    save(base/'transfer_manifest.json',manifest)
    with tarfile.open(base/'analysis_receipts.tar.gz','x:gz') as tar:
        for p in paths+[base/'transfer_manifest.json']:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
    save(base/'archive_receipt.json',dict(path=str(base/'analysis_receipts.tar.gz'),sha256=sha(base/'analysis_receipts.tar.gz'),
        selected_weight='model_backups/selected_110.pth',delete_archive_only_after_verified_local_transfer=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','gate','run'],required=True);a=p.parse_args();base=a.base.resolve()
    if a.stage=='prepare':prepare(base)
    elif a.stage=='gate':gate(base)
    else:run(base)
