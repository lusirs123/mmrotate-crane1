"""Freeze and execute paired U/S/G runs, VAL only, stop on any error."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'work_dirs/symeood_usg_v1'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def freeze():
    import torch,mmcv,mmdet,mmrotate
    from mmcv import Config
    draft=json.loads((BASE/'protocol_draft.json').read_text())
    assert draft['seeds']==[0]
    expected=json.loads((BASE/'expected_final_sources.json').read_text())
    for name,digest in expected.items():assert sha(ROOT/name)==digest,('Local/server new source mismatch',name)
    engineer=json.loads((BASE/'engineering_v2/engineering.json').read_text())
    preflight=json.loads((BASE/'ddp_preflight.json').read_text())
    assert engineer['passed'] and len(engineer['cases'])==12
    assert preflight['passed'] and len(preflight['cases'])==6
    unit=(BASE/'unit_tests_v2.log').read_text();assert 'Ran 7 tests' in unit and unit.rstrip().endswith('OK')
    post=json.loads((BASE/'postprocessing_engineering.json').read_text());assert post['passed']
    fixture=json.loads((BASE/'postprocessing_fixture_receipt.json').read_text());assert fixture['passed']
    # Verify the historical shared code and data identity, without opening TEST.
    prior=json.loads((ROOT/'work_dirs/symeood_ablation_v1/c_frozen_diagnostic_20261024_v1/draft.json').read_text())
    sources={}
    for name,digest in prior['sources'].items():
        assert sha(ROOT/name)==digest,name
        sources[name]=digest
    extra=list((ROOT/'crane_project/configs').glob('crane_symeood_usg_*_v1.py'))
    extra += [ROOT/'crane_project/utils/symeood_usg_v1.py',ROOT/'tests/test_symeood_usg_v1.py']
    extra += list((ROOT/'crane_project/tools').glob('*symeood_usg*v1.py'))
    extra += [ROOT/'crane_project/tools/ckpt_sweep.py',ROOT/'crane_project/tools/eval_crane_offline.py',
              ROOT/'crane_project/tools/analyze_symeood_ablation_first4_v1.py',
              ROOT/'crane_project/tools/audit_symeood_c_existing_results_v1.py',
              ROOT/'crane_project/tools/diagnose_symeood_c_frozen_v1.py',ROOT/'tools/train.py',ROOT/'tools/test.py']
    for p in extra:
        assert p.is_file(),p
        sources[str(p.relative_to(ROOT))]=sha(p)
    for rec in prior['data_inventory']:
        assert sha(rec['image'])==rec['image_sha256']
        assert sha(rec['annotation'])==rec['annotation_sha256']
    probe=prior['training_views'][:4]
    for rec in probe:assert sha(rec['path'])==rec['sha256']
    for group in draft['groups']:
        cfg=Config.fromfile(str(ROOT/group['config']))
        assert cfg.runner.max_epochs==24 and cfg.load_from is None and cfg.resume_from is None
        assert cfg.model.train_cfg.assigner.get('cost_reg_mode','kld')=='kld'
        assert cfg.optimizer.lr==.0025 and cfg.optimizer.momentum==.9
        assert cfg.model.bbox_head.loss_cls.objective==group['objective']
    draft['sources']=sources
    draft['data_inventory']=prior['data_inventory'];draft['probe_inputs']=probe
    cache=Path(torch.hub.get_dir())/'checkpoints'
    draft['pretrained_backbone_cache']={str(p):sha(p) for p in cache.glob('resnet50*.pth')}
    assert draft['pretrained_backbone_cache'], 'Pretrained backbone must already exist locally'
    draft['engineering_receipts']={str(p.relative_to(BASE)):sha(p) for p in [BASE/'engineering_v2/engineering.json',BASE/'ddp_preflight.json',BASE/'unit_tests_v2.log',BASE/'postprocessing_engineering.json',BASE/'postprocessing_fixture_receipt.json']}
    draft['environment']=dict(python=sys.version,torch=torch.__version__,mmcv=mmcv.__version__,mmdet=mmdet.__version__,mmrotate=mmrotate.__file__)
    draft['state']='FROZEN';draft['freeze_time']=time.strftime('%Y-%m-%d %H:%M:%S')
    with (BASE/'protocol.json').open('x') as f:json.dump(draft,f,indent=2,ensure_ascii=False)
    print('FROZEN',sha(BASE/'protocol.json'))


def run():
    os.chdir(ROOT)
    protocol=json.loads((BASE/'protocol.json').read_text());assert protocol['state']=='FROZEN'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2,3'
    assert protocol['seeds']==[0], 'Current user scope is six runs with historical seed 0'
    (BASE/'launch').mkdir(exist_ok=False)
    status=BASE/'launch/status.json'
    source_hashes=protocol['sources']
    input_reference={};initial_reference={}
    def check_sources():
        for name,digest in source_hashes.items():assert sha(ROOT/name)==digest,name
    def command(runid,stage,cmd):
        check_sources()
        status.write_text(json.dumps(dict(state='running',run=runid,stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),command=cmd),indent=2))
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT)
        with (BASE/'launch'/(runid+'_'+stage+'.log')).open('x') as f:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    for si,seed in enumerate(protocol['seeds']):
        order=('U','S','G');order=order[si:]+order[:si]
        for obj in order:
            for reg in ('iou','kld'):
                runid='s%d_%s_%s'%(seed,reg,obj.lower());work=BASE/runid
                assert not work.exists(), 'Never overwrite a run'
                cfg='crane_project/configs/crane_symeood_usg_%s_%s_v1.py'%(reg,obj.lower())
                command(runid,'train',[sys.executable,'-m','torch.distributed.launch','--nproc_per_node=2','--master_port=29644',
                    'tools/train.py',cfg,'--launcher','pytorch','--seed',str(seed),'--work-dir',str(work)])
                assert (work/'epoch_24.pth').is_file() and (work/'recovery_epoch_24.pth').is_file()
                for rank in (0,1):
                    identity=json.loads((work/('identity_rank%d.jsonl'%rank)).read_text().splitlines()[0])
                    key=seed
                    if key in initial_reference:assert identity['initial_parameters_sha256']==initial_reference[key]
                    else:initial_reference[key]=identity['initial_parameters_sha256']
                    rows=[json.loads(s) for s in (work/('inputs_rank%d.jsonl'%rank)).read_text().splitlines()]
                    assert {r['epoch'] for r in rows}==set(range(1,25))
                    for r in rows:
                        key=(seed,rank,r['epoch'])
                        if key in input_reference:assert input_reference[key]==r['sha256']
                        else:input_reference[key]=r['sha256']
                logs=[json.loads(s) for f in work.glob('*.log.json') for s in f.read_text().splitlines()]
                assert {r['epoch'] for r in logs if r.get('mode')=='val'}==set(range(2,25,2))
                command(runid,'val_select',[sys.executable,'crane_project/tools/ckpt_sweep.py',
                    '--config',cfg,'--work-dir',str(work),'--gpu','3','--epochs','16','18','20','22','24',
                    '--center-thresh','15','--mcml-limit','5'])
                command(runid,'candidates',[sys.executable,'crane_project/tools/evaluate_symeood_usg_v1.py',
                    'candidates','--config',cfg,'--work-dir',str(work)])
    command('all','CPU_aggregate',[sys.executable,'crane_project/tools/evaluate_symeood_usg_v1.py','aggregate','--base',str(BASE)])
    receipt=json.loads((BASE/'analysis_v1/complete.json').read_text());assert receipt['passed']
    check_sources()
    status.write_text(json.dumps(dict(state='complete',runs=6*len(protocol['seeds']),TEST=0,depth=0,analysis='analysis_v1',time=time.strftime('%Y-%m-%d %H:%M:%S')),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);a=parser.parse_args()
    try:
        freeze() if a.stage=='freeze' else run()
    except Exception as exc:
        if (BASE/'launch').exists():(BASE/'launch/failure.json').write_text(json.dumps(dict(error=repr(exc),time=time.strftime('%Y-%m-%d %H:%M:%S')),indent=2))
        raise
