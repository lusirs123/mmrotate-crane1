#!/usr/bin/env python3
"""One predeclared bounded-anchor comparison, complete TRAIN then VAL."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.utils import port_reliability_simple_anchor_v1 as core
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_complementarity_v1 as metrics
from crane_project.utils import port_reliability_state_continuity_v1 as states

PROTOCOL = ROOT/'crane_project/tools/port_reliability_simple_anchor_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_reliability_simple_anchor_v1_sources.json'
FIELDS = ('image','sequence','domain','split','frame_id','image_size','gt','pred',
          'train_angle_eligible','angle_axis_well_defined','b_original','reliability_role',
          'size_risks','original_simple_decision')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def jsonl(path):
    with gzip.open(path,'rt') as f:
        return [json.loads(line) for line in f]


def checked():
    contract = json.loads(PROTOCOL.read_text()); manifest = json.loads(SOURCES.read_text())
    if contract['protocol'] != core.VERSION or contract['settings'] != core.SETTINGS:
        raise ValueError('Contract differs')
    if manifest['protocol'] != core.VERSION or any(sha(ROOT/p) != v for p,v in manifest['sources'].items()):
        raise ValueError('Executable sources differ')
    if any(sha(ROOT/p) != v for p,v in contract['input_pins'].items()):
        raise ValueError('Frozen inputs differ')
    return contract,manifest


def load_rows(contract):
    primary = jsonl(ROOT/contract['primary_rows'])
    lookup = {r['image']:r for r in primary}
    if len(lookup) != len(primary):
        raise ValueError('Duplicate frozen frames')
    policy = json.loads((ROOT/contract['policy']).read_text())['simple_policy']
    parts = {}
    directory = ROOT/contract['historical_features']
    old = json.loads((directory/'report.json').read_text())
    complete = json.loads((directory/'completion.json').read_text())
    if old['protocol'] != 'port_reliability_feature_source_v1' or complete['status'] != 'VAL_FAILED_STOP':
        raise ValueError('Historical failed-source identity differs')
    for file in ('models.json','pca.json','train_native_projected.npz','val_native_projected.npz','scored_TRAIN.jsonl.gz','scored_VAL.jsonl.gz','report.json'):
        if sha(directory/file) != complete['artifacts'][file]:
            raise ValueError('Historical feature provenance differs: '+file)
    for role, role_name in (('TRAIN','train'),('VAL','val')):
        values = sorted((r for r in primary if r['reliability_role']==role_name),key=lambda r:r['image'])
        historical = jsonl(directory/('scored_'+role+'.jsonl.gz'))
        if len(historical) != len(values):
            raise ValueError('Historical complete role count differs')
        for a,b in zip(values,sorted(historical,key=lambda r:r['image'])):
            if {k:a[k] for k in FIELDS} != {k:b[k] for k in FIELDS}:
                raise ValueError('Frozen frame identity differs: '+a['image'])
            if (a['split'] not in ('train','train_sim') if role=='TRAIN' else a['split']!='val'):
                raise ValueError('No TEST or new role permitted')
            if a['pred'] is None:
                if a['size_risks']['full_simple'] is not None:
                    raise ValueError('Missing cannot have a size risk')
                continue
            descriptor = simple.descriptor(a['pred'],a['image_size'])
            replay = float(simple.linear_risk(policy['models']['size'],descriptor))
            if abs(replay-a['size_risks']['full_simple']) > 1e-12:
                raise ValueError('Frozen simple replay differs')
            if a['original_simple_decision']['final_box_original'] != a['pred']:
                raise ValueError('Original delivery differs')
        states.checked_order(values)
        expected = contract['role_counts'][role]
        if (len(values)!=expected['frames'] or sum(metrics.bad(r) is True for r in values)!=expected['bad']
                or sum(r['pred'] is not None for r in values)!=expected['outputs']
                or dict(Counter(r['domain'] for r in values))!=expected['domains']):
            raise ValueError('Complete frozen data role differs')
        parts[role] = values
    if len(primary) != sum(map(len,parts.values())):
        raise ValueError('Unexpected data role')
    return parts,policy


def matrix(contract, rows, role, normalizer):
    directory = ROOT/contract['historical_features']
    with np.load(directory/(role.lower()+'_native_projected.npz'),allow_pickle=False) as f:
        x, ids = f['features'],f['images'].tolist()
    outputs = [r for r in rows if r['pred'] is not None]
    if ids != [r['image'] for r in outputs] or x.shape != (len(outputs),259):
        raise ValueError('Native feature/output pairing differs')
    descriptors = np.stack([simple.descriptor(r['pred'],r['image_size']) for r in outputs])
    if not np.allclose(x[:,:3],descriptors,rtol=0,atol=1e-12):
        raise ValueError('Feature descriptors differ from final M')
    core.normalize(x,normalizer)
    if role=='TRAIN':
        scale = x.std(0); scale = np.where(scale<1e-8,1.,scale)
        if not np.array_equal(x.mean(0),normalizer['mean']) or not np.array_equal(scale,normalizer['scale']):
            raise ValueError('Reused normalizer is not exact original full TRAIN fit')
    return x,outputs


def normalizer_for(contract):
    # Reuse only TRAIN moments, not historical neural weights or predictions.
    return json.loads((ROOT/contract['historical_features']/'models.json').read_text())['normalizers']['native']


def score(rows, features, outputs, models, normalizer):
    data = deepcopy(rows); lookup = {r['image']:r for r in data}; deltas = {}
    for r in data:
        r['risks'] = dict(full_simple=r['size_risks']['full_simple'],score_only=1-r['pred'][5] if r['pred'] is not None else None,
                          score_anchor=None,simple_anchor=None)
    for arm in core.ARMS:
        risk,delta = core.numpy_forward(models[arm],features,normalizer,core.anchors(outputs,arm))
        deltas[arm] = dict(min=float(delta.min()),max=float(delta.max()),
            saturation_fraction=float((np.abs(delta)>.49).mean()),risk_min=float(risk.min()),risk_max=float(risk.max()))
        for row,value in zip(outputs,risk):
            lookup[row['image']]['risks'][arm] = float(value)
    return data,deltas


def save_rows(path,rows,cutoffs):
    with gzip.open(path,'xt') as f:
        for r in rows:
            r['candidate_decisions'] = {a:core.decide(r['original_simple_decision'],r['risks'][a],cutoffs[a]['risk_le']) for a in core.ARMS}
            f.write(json.dumps(r,allow_nan=False)+'\n')


def train(args):
    import torch
    from crane_project.utils import port_reliability_simple_anchor_v1_torch as head
    contract,manifest = checked(); parts,policy = load_rows(contract)
    if args.out.resolve().parent.parent != (ROOT/'work_dirs'/core.VERSION).resolve():
        raise ValueError('Use work_dirs/version/RUN_ID/result')
    if args.out.exists():
        raise FileExistsError('Refuse overwrite')
    args.out.mkdir(parents=True); start = time.monotonic()
    torch.set_num_threads(1); torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = 'cuda:'+str(args.gpu); normalizer = normalizer_for(contract)
    features,outputs = matrix(contract,parts['TRAIN'],'TRAIN',normalizer)
    labels = np.asarray([metrics.bad(r) for r in outputs],dtype=np.float64)
    tensor = lambda a:torch.tensor(a,dtype=torch.float64,device=device)
    x = tensor(core.normalize(features,normalizer)); y = tensor(labels); w = tensor(core.class_weights(labels))
    anchors = {a:tensor(core.anchors(outputs,a)) for a in core.ARMS}
    neutral = head.make_models(device); smoke = {}
    index = torch.tensor(np.r_[np.flatnonzero(labels==0)[:4],np.flatnonzero(labels==1)[:4]],device=device)
    for arm,model in neutral.items():
        with torch.no_grad():
            if not torch.equal(model.risk(x,anchors[arm]),anchors[arm]):
                raise ValueError('Neutral TRAIN risk not exactly anchor')
        opt = head.optimizer(model)
        updates = [head.update(model,opt,x[index],anchors[arm][index],y[index],w[index]) for _ in range(2)]
        if not any(v>0 for k,v in updates[1]['gradient_by_parameter'].items() if k.startswith('network.0')):
            raise ValueError('Hidden gradient disconnected')
        smoke[arm] = dict(exact_neutral=True,discarded_updates=updates)
    write(args.out/'smoke_report.json',smoke); del neutral
    models = head.make_models(device); initial = {a:head.exported(m) for a,m in models.items()}
    if initial[core.ARMS[0]] != initial[core.ARMS[1]]:
        raise ValueError('Initial model states differ')
    parameter_counts = {a:sum(p.numel() for p in m.parameters()) for a,m in models.items()}
    if any(n != 4290 for n in parameter_counts.values()):
        raise ValueError('Head capacity differs')
    write(args.out/'initial_heads.json',initial)
    optimizers = {a:head.optimizer(m) for a,m in models.items()}; counts = dict.fromkeys(core.ARMS,0)
    order_hash = hashlib.sha256(); rng = np.random.RandomState(1701)
    with (args.out/'train_log.jsonl').open('x') as stream:
        for epoch in range(1,101):
            order = rng.permutation(len(outputs)); order_hash.update(order.astype('<i8').tobytes())
            for slot,offset in enumerate(range(0,len(outputs),256)):
                indices = torch.tensor(order[offset:offset+256],device=device)
                for arm,model in models.items():
                    update = head.update(model,optimizers[arm],x[indices],anchors[arm][indices],y[indices],w[indices])
                    counts[arm] += 1
                    stream.write(json.dumps(dict(epoch=epoch,slot=slot,arm=arm,batch_size=len(indices),**update),allow_nan=False)+'\n')
            stream.flush()
            if epoch==1 or epoch%10==0:
                print('TRAIN epoch',epoch,'updates',counts,flush=True)
    if counts != dict.fromkeys(core.ARMS,1000):
        raise ValueError('Fixed budget differs')
    final = {a:head.exported(m) for a,m in models.items()}
    torch.save({a:m.state_dict() for a,m in models.items()},args.out/'final_heads.pth')
    saved = torch.load(args.out/'final_heads.pth',map_location=device)
    reload = head.make_models(device)
    for a in core.ARMS:
        reload[a].load_state_dict(saved[a])
        if head.exported(reload[a]) != final[a]:
            raise ValueError('Final save/reload differs')
    write(args.out/'models.json',dict(protocol=core.VERSION,models=final,normalizer=normalizer,
        epoch=100,update_counts=counts,dtype='float64'))
    # Load VAL features only after both final heads are frozen; no weight selection.
    val_features,val_outputs = matrix(contract,parts['VAL'],'VAL',normalizer)
    matrices = dict(TRAIN=(features,outputs),VAL=(val_features,val_outputs))
    scored = {}; diagnostics = {}; replay = {}; neutral_flags = 0
    zero = head.make_models(device)
    formal_cutoff = policy['cutoffs']['simple']['size']['risk_le']
    for role,(f,out) in matrices.items():
        scored[role],diagnostics[role] = score(parts[role],f,out,final,normalizer)
        replay[role] = {}
        for a in core.ARMS:
            tx = tensor(core.normalize(f,normalizer)); tr = tensor(core.anchors(out,a))
            with torch.no_grad():
                neutral_risks = zero[a].risk(tx,tr).cpu().numpy()
                torch_risks = models[a].risk(tx,tr).cpu().numpy()
            if not np.array_equal(neutral_risks,core.anchors(out,a)):
                raise ValueError('Neutral exact identity differs')
            expected = np.asarray([r['risks'][a] for r in scored[role] if r['pred'] is not None])
            replay[role][a] = float(np.max(np.abs(torch_risks-expected)))
            if replay[role][a]>1e-10:
                raise ValueError('Torch/NumPy risk differs')
        for r in parts[role]:
            if core.decide(r['original_simple_decision'],r['size_risks']['full_simple'],formal_cutoff) != r['original_simple_decision']:
                raise ValueError('Zero simple correction did not preserve original flags')
            neutral_flags += 1
    cutoffs = {a:core.calibrate(scored['VAL'],a) for a in core.METHODS}
    write(args.out/'cutoffs.json',cutoffs)
    statistics = {role:core.statistics(rows,cutoffs) for role,rows in scored.items()}
    gate = core.gate(statistics['VAL']); status = 'VAL_PASS_FROZEN_TEST_PENDING' if gate['passed'] else 'VAL_FAILED_STOP'
    for role,rows in scored.items():
        save_rows(args.out/('scored_'+role+'.jsonl.gz'),rows,cutoffs)
    checked()
    report = dict(protocol=core.VERSION,status=status,contract=contract,sources=manifest,
        input_pins=contract['input_pins'],git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip(),
        historical_source_commit=json.loads((ROOT/contract['historical_features']/'report.json').read_text())['git_commit'],
        statistics=statistics,cutoffs=cutoffs,gate=gate,diagnostics=diagnostics,
        formal_policy_summary={role:{g:states.summarize(v,{r['image'] for r in v if r['original_simple_decision']['size_accepted']}) for g,v in metrics.grouped(rows).items()} for role,rows in scored.items()},
        original_coverage95_comparison={role:{a:{g:states.summarize(v,metrics.accepted(v,a,policy['cutoffs']['simple' if a=='full_simple' else 'score_only']['size']['risk_le'])) for g,v in metrics.grouped(rows).items()} for a in ('full_simple','score_only')} for role,rows in scored.items()},
        center={role:{g:metrics.center(v) for g,v in metrics.grouped(rows).items()} for role,rows in scored.items()},
        engineering=dict(exact_neutral_frames=neutral_flags,torch_numpy_max_error=replay,save_reload_exact=True,smoke_discarded=True),
        parameter_counts=parameter_counts,update_counts=counts,batch_order_sha256=order_hash.hexdigest(),
        fixed_final_epoch=100,GT_online=False,TEST_read=False,TEST_repeatedly_exposed=True,new_data_roles=False,
        original_policy_changed=False,boxes_scores_output_center_angle_unchanged=True,
        detector_inferences_added=0,feature_forwards_added=0,elapsed_seconds=time.monotonic()-start)
    write(args.out/'report.json',report); (args.out/'analysis.md').write_text(core.markdown(report))
    write(args.out/'completion.json',dict(protocol=core.VERSION,status=status,TEST_read=False,
        artifacts={p.name:sha(p) for p in sorted(args.out.iterdir()) if p.is_file()}))
    print(json.dumps(dict(status=status,gate=gate,elapsed_seconds=report['elapsed_seconds']),ensure_ascii=False),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--mode',choices=('prepare','train'),default='train')
    p.add_argument('--gpu',type=int,default=0); p.add_argument('--out',type=Path)
    args = p.parse_args()
    if Path.cwd().resolve()!=ROOT:
        raise SystemExit('Run in project root')
    if args.mode=='prepare':
        contract,_ = checked(); parts,_ = load_rows(contract); normalizer = normalizer_for(contract)
        for role,rows in parts.items():
            x,outputs = matrix(contract,rows,role,normalizer)
            for a in core.ARMS:
                core.anchors(outputs,a)
        print(json.dumps(dict(passed=True,frames={k:len(v) for k,v in parts.items()},TEST_read=False)))
        return
    if args.out is None:
        p.error('--out is required for train')
    existed = args.out.exists()
    try:
        train(args)
    except Exception as e:
        if not existed and args.out.exists():
            write(args.out/'failure.json',dict(error=type(e).__name__+': '+str(e)))
        raise


if __name__=='__main__':
    main()
