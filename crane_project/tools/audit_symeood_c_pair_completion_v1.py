"""CPU-only final identity audit and bounded receipt bundle, never a runner."""
import argparse
import base64
import collections
import hashlib
import json
import math
from pathlib import Path
import pickle
import tarfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open() as f:
        for line in f:yield json.loads(line)


def rng_identity(payload):
    # Pickled tensor storage identifiers may differ between processes even
    # when RNG values are identical. Compare the decoded state, not pickle.
    def plain(x):
        if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [plain(v) for v in x]
        if hasattr(x,'tolist'):return x.tolist()
        return x
    state=plain(pickle.loads(base64.b64decode(payload)))
    return hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()


def audit(root,base):
    contract=json.loads((base/'protocol.json').read_text())
    assert contract['groups']==['110','111'] and contract['epochs_each']==24
    assert contract['seed']==0 and contract['world_size']==2
    complete=json.loads((base/'analysis_v2/complete.json').read_text());assert complete['passed']
    verification=json.loads((base/'analysis_v2/verification.json').read_text());assert verification['passed']
    # The original frozen analyzer remains untouched on the training server.
    for p,h in contract['sources'].items():assert sha(root/p)==h,('frozen source changed',p)
    for g,p in contract['configurations'].items():assert sha(Path(p))==contract['config_sha256'][g]
    records={};initial={};inputchecks=[]
    for group in contract['groups']:
        work=base/group
        logs=[r for p in work.glob('*.log.json') for r in rows(p)]
        train=[r for r in logs if r.get('mode')=='train']
        assert {r['epoch'] for r in train}==set(range(1,25))
        assert {r['epoch'] for r in logs if r.get('mode')=='val'}==set(range(2,25,2))
        assert all(math.isfinite(v) for r in train for v in r.values() if isinstance(v,(float,int)))
        records[group]=dict(train_epochs=24,online_VAL=12,logged_train_records=len(train),ranks={})
        for rank in (0,1):
            first=list(rows(work/('repro_initial_rank%d.jsonl'%rank)));assert len(first)==1
            initial[group,rank]=first[0]
            conservation=list(rows(work/('repro_conservation_rank%d.jsonl'%rank)))
            assert len(conservation)==24 and all(r['passed'] and all(r['checks'].values()) for r in conservation)
            inp=work/('repro_inputs_rank%d.jsonl'%rank)
            samples=list(rows(inp));assert [r['iteration'] for r in samples]==list(range(15360))
            assert collections.Counter(r['epoch'] for r in samples)=={e:640 for e in range(1,25)}
            records[group]['ranks'][rank]=dict(input_records=len(samples),input_file_sha256=sha(inp),conservation_passed=24)
    assert len({r['initial_state_sha256'] for r in initial.values()})==1
    for rank in (0,1):
        rng_a,rng_b=(rng_identity(initial[g,rank]['rng']) for g in ('110','111'))
        assert rng_a==rng_b,('initial RNG values',rank)
        a,b=(base/g/('repro_inputs_rank%d.jsonl'%rank) for g in ('110','111'))
        n=0
        for x,y in zip(rows(a),rows(b)):
            assert (x['epoch'],x['iteration'],x['combined_sha256'])==(y['epoch'],y['iteration'],y['combined_sha256'])
            n+=1
        assert n==15360;inputchecks.append(dict(rank=rank,matched_batches=n,initial_rng_values_sha256=rng_a))
    checkpoints=[]
    for group in contract['groups']:
        work=base/group
        sweep=json.loads((work/'ckpt_sweep/sweep_results.json').read_text())
        assert sweep['selected_checkpoint']==complete['selected'][group]
        reports=list((work/'ckpt_sweep/final_test').glob('*/final_test_metrics_v2.json'));assert len(reports)==1
        test=json.loads(reports[0].read_text())
        for split,cp in [('val',r) for r in sweep['all_checkpoints'].values()]+[('test',test)]:
            assert sha(Path(cp['checkpoint']))==cp['checkpoint_sha256']
            assert sha(Path(cp['results_pkl']))==cp['results_pkl_sha256']
            checkpoints.append(dict(group=group,split=split,checkpoint=cp['checkpoint'],checkpoint_sha256=cp['checkpoint_sha256'],
                                    results_pkl=cp['results_pkl'],results_pkl_sha256=cp['results_pkl_sha256']))
    result=dict(passed=True,formal_runs=2,train_epochs=48,online_VAL=24,candidate_VAL=10,TEST=2,
        initial_state_sha256=next(iter(initial.values()))['initial_state_sha256'],
        all_paired_inputs_identical=True,paired_input_checks=inputchecks,records=records,
        frozen_source_count=len(contract['sources']),checkpoints=checkpoints,
        analysis=complete,analysis_metric_checks=len(verification['checks']),
        original_queue_status=json.loads((base/'status.json').read_text()),
        followup='Original CPU aggregation lacked historical JSON; fixed CPU-only output in analysis_v2. No training or inference repeated.',
        no_new_training=True,no_new_inference=True,no_GPU=True,source_sha256=sha(Path(__file__)))
    with (base/'completion_audit.json').open('x') as f:json.dump(result,f,indent=2)
    paths=set(base.glob('*.json'))|set(base.glob('*.log'))|set((base/'configs').glob('*.py'))|set((base/'analysis_v2').glob('*'))
    for group in contract['groups']:
        work=base/group
        paths.update(work.glob('*.log.json'))
        for pattern in ('repro_initial_rank*.jsonl','repro_conservation_rank*.jsonl','repro_runtime_rank*.jsonl'):paths.update(work.glob(pattern))
        paths.update(p for p in (work/'ckpt_sweep').rglob('*') if p.is_file())
    paths={p for p in paths if p.is_file()}
    manifest={str(p.relative_to(base)):sha(p) for p in sorted(paths)}
    with (base/'transfer_manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
    archive=base/'completion_receipts_20261026.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for p in sorted(paths|{base/'transfer_manifest.json'}):tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
    print(json.dumps(dict(passed=True,receipt_files=len(manifest),archive=str(archive),archive_sha256=sha(archive),
                         selected=complete['selected'],paired_batches=sum(x['matched_batches'] for x in inputchecks)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    a=p.parse_args();audit(a.root.resolve(),a.base.resolve())
