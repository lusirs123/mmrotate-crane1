#!/usr/bin/env python3
"""Read-only D diagnosis: frozen TRAIN gradients and existing VAL; no TEST/optimizer.

CUDA runtime uses D's architecture for both frozen checkpoints. For the B
snapshot the added D loss is a counterfactual probe and is excluded from the
actual B objective. No model source, predictions or checkpoint is rewritten.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ordered_data_infos(dataset):
    """Metadata in the SAME global index order as nested ConcatDataset.__getitem__."""
    if hasattr(dataset,'datasets'):
        infos=[info for child in dataset.datasets for info in ordered_data_infos(child)]
    elif hasattr(dataset,'data_infos'):
        infos=list(dataset.data_infos)
    else:
        raise TypeError('Unsupported TRAIN wrapper: '+type(dataset).__name__)
    if len(infos)!=len(dataset):
        raise ValueError('TRAIN metadata count does not match global indexing')
    return infos


def pick_train_indices(dataset,per_domain,seed):
    from crane_project.tools.audit_port_train_val_geometry_v1 import pick_indices
    # The legacy sampler accepts flat metadata; retain its seed/stratification.
    return pick_indices(SimpleNamespace(data_infos=ordered_data_infos(dataset)),per_domain,seed)


def reuse_artifacts(artifact_path,progress_path,current_sources,sweeps,configs):
    """Resume ONLY before the first TRAIN batch, with frozen identity checks."""
    from crane_project.tools.ckpt_sweep import annotation_set_sha256
    if not artifact_path.is_file() or not progress_path.is_file():
        raise FileNotFoundError('Resume requires both existing artifact and progress sidecars')
    if progress_path.read_text().strip():
        raise ValueError('Resume accepts only zero completed TRAIN batches; preserve nonempty progress')
    cached=json.loads(artifact_path.read_text())
    if (cached.get('protocol')!='port_center_size_d_diagnosis_v1'
            or cached.get('evidence_role')!='train_val_only_diagnosis'):
        raise ValueError('Unexpected artifact cache protocol')
    script_key='crane_project/tools/diagnose_port_center_size_d_v1.py'
    stable=lambda x:{k:v for k,v in x.items() if k!=script_key}
    if stable(cached['sources'])!=stable(current_sources):
        raise ValueError('Model/config sources changed; cached artifacts cannot be reused')
    for arm in ('b','d'):
        identity=cached['identities'][arm]
        cfg=configs[arm]
        ann=(Path(cfg.data.val.data_root)/cfg.data.val.ann_file).resolve()
        selection_path=Path(sweeps[arm])/'sweep_results.json'
        if sha(selection_path)!=identity['selection_sha256']:
            raise ValueError('Frozen selection changed: '+arm)
        selection=json.loads(selection_path.read_text())
        selected='epoch_24' if arm=='b' else 'epoch_22'
        if (selection.get('evidence_role')!='source_val_checkpoint_selection'
                or selection['selected_checkpoint']!=selected
                or identity['selected_epoch']!=selected
                or selection['config_sha256']!=identity['config_sha256']):
            raise ValueError('Frozen VAL contract changed: '+arm)
        record=selection['all_checkpoints'][selected]
        for key,record_key in [('checkpoint','checkpoint'),('pkl','results_pkl')]:
            path=Path(identity[key]).resolve()
            if (Path(record[record_key]).resolve()!=path
                    or sha(path)!=identity[key+'_sha256']
                    or record[record_key+'_sha256']!=identity[key+'_sha256']):
                raise ValueError('Frozen '+key+' changed: '+arm)
        if (Path(selection['selected_path']).resolve()!=Path(identity['checkpoint']).resolve()
                or annotation_set_sha256(str(ann))!=identity['annotation_sha256']
                or selection['source_val_annotations_sha256']!=identity['annotation_sha256']):
            raise ValueError('VAL annotations/selected path changed: '+arm)
    return cached


def stats(values):
    import numpy as np
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not values:
        return dict(n=0, mean=None, median=None, p90=None, maximum=None)
    return dict(n=len(values), mean=float(np.mean(values)),
                median=float(np.median(values)), p90=float(np.percentile(values, 90)),
                maximum=max(values))


def scalar(value):
    return float(value.detach())


def sum_loss(value):
    return sum(value) if isinstance(value, (list, tuple)) else value


def d_parts(pred, target, weight, avg_factor, beta=.1, coefficient=.25):
    """Reconstruct the EXACT D center/size objectives on captured actual positives."""
    import torch
    import torch.nn.functional as F
    from mmdet.models.losses.utils import weight_reduce_loss
    target = target.detach()
    if pred.shape[0] == 0:
        zero = pred[:, :4].sum() * 0.
        return zero, zero
    p = pred[:, 2:4].sort(dim=-1, descending=True).values.clamp_min(1e-6)
    t = target[:, 2:4].sort(dim=-1, descending=True).values.clamp_min(1e-6)
    center = (pred[:, :2] - target[:, :2]) / t[:, 1:2]
    size = p.log() - t.log()
    c = F.smooth_l1_loss(center, torch.zeros_like(center), reduction='none', beta=beta).mean(-1)
    s = F.smooth_l1_loss(size, torch.zeros_like(size), reduction='none', beta=beta).mean(-1)
    return tuple(coefficient * weight_reduce_loss(x, weight, 'mean', avg_factor) for x in (c, s))


def positive_geometry(pred, target):
    """Normal-region C/S plus actual numeric guards; not a causal attribution."""
    import torch
    from mmrotate.models.losses.sym_kld_calculator import _xywha_to_gaussian, _inv2x2_safe, sym_kld
    with torch.no_grad():
        p, q = pred.detach(), target.detach()
        if not len(p):
            return dict(n=0)
        mp, sp = _xywha_to_gaussian(p)
        mq, sq = _xywha_to_gaussian(q)
        ip, iq = _inv2x2_safe(sp), _inv2x2_safe(sq)
        delta = (mp - mq).unsqueeze(-1)
        c = .5 * (delta.transpose(-1, -2) @ (ip + iq) @ delta).flatten()
        s = .5 * (torch.einsum('nij,nji->n', iq, sp) + torch.einsum('nij,nji->n', ip, sq) - 4)
        raw = sym_kld(p, q)
        pe = p[:, 2:4].sort(-1, descending=True).values
        qe = q[:, 2:4].sort(-1, descending=True).values
        norm_center = (p[:, :2] - q[:, :2]) / qe[:, 1:2]
        logs = pe.log() - qe.log()
        def arr(x):
            return x.cpu().reshape(-1).tolist()
        determinant_guard = 0
        inverse_guard = 0
        for cov in (sp, sq):
            a, b, d = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
            det = a*d-b*b
            determinant_guard += int((det <= 1e-6).sum())
            unconstrained_inv = torch.stack([d, -b, -b, a], -1) / det.clamp_min(1e-6)[:, None]
            inverse_guard += int((unconstrained_inv.abs() >= 1e4).any(-1).sum())
        return dict(n=len(p), center_component=stats(arr(c)), shape_component=stats(arr(s)),
            symkld_raw=stats(arr(raw)), symkld_loss_cap_count=int((raw > 120).sum()),
            symkld_raw_cap_count=int((raw >= 1e4).sum()),
            determinant_guard_count=determinant_guard, inverse_guard_count=inverse_guard,
            center_smoothl1_linear_coordinate_count=int((norm_center.abs() >= .1).sum()),
            size_smoothl1_linear_coordinate_count=int((logs.abs() >= .1).sum()),
            center_residual_over_short=stats(arr(norm_center.abs())),
            long_log_ratio=stats(arr(logs[:, 0])), short_log_ratio=stats(arr(logs[:, 1])),
            gt_short_input_px=stats(arr(qe[:, 1])))


def gradient_vector(loss, tensors):
    import torch
    grads = torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
    if any(g is not None and not bool(torch.isfinite(g).all()) for g in grads):
        raise RuntimeError('Nonfinite diagnostic gradient')
    return tuple(g.detach() if g is not None else None for g in grads)


def grad_norm(grads):
    return sum(float(g.double().square().sum()) for g in grads if g is not None) ** .5


def parameter_digest(model):
    """Weights only: schedule counters/buffers are deliberately exercised."""
    h=hashlib.sha256()
    for name,parameter in model.named_parameters():
        h.update(name.encode('utf-8'))
        h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def grad_cos(a, b):
    na, nb = grad_norm(a), grad_norm(b)
    if na == 0 or nb == 0:
        return None
    dot = sum(float(x.double().mul(y.double()).sum())
              for x, y in zip(a, b) if x is not None and y is not None)
    return max(-1., min(1., dot / (na * nb)))


def add_grads(a, b):
    return tuple(y if x is None else x if y is None else x+y for x, y in zip(a, b))


def gradient_report(losses, groups):
    """Full regression conv; explicitly representative FPN/backbone parameters."""
    tensors = [p for _, ps in groups for p in ps]
    slices, start = {}, 0
    for name, ps in groups:
        slices[name] = slice(start, start+len(ps))
        start += len(ps)
    gradients = {k: gradient_vector(v, tensors) for k, v in losses.items()}
    gradients['d_sum'] = add_grads(gradients['d_center'], gradients['d_size'])
    gradients['base_total'] = add_grads(add_grads(gradients['symkld'], gradients['classification']),
                                         gradients['other_losses'])
    report = {}
    for name, index in slices.items():
        g = {k: v[index] for k, v in gradients.items()}
        norms = {k: grad_norm(v) for k, v in g.items()}
        ratios = {k+'_to_symkld': norms[k]/norms['symkld'] if norms['symkld'] else None
                  for k in ('d_center', 'd_size', 'd_sum')}
        cosines = {a+'_vs_'+b: grad_cos(g[a], g[b]) for a, b in
                   [('d_center', 'symkld'), ('d_size', 'symkld'), ('d_center', 'd_size'),
                    ('d_sum', 'symkld'), ('d_sum', 'classification'),
                    ('d_sum', 'other_losses'), ('d_sum', 'base_total')]}
        report[name] = dict(norms=norms, ratios=ratios, cosines=cosines)
    return report


def read_logs(workdir):
    """Describe every available log separately; no invented center/size histories."""
    logs = []
    for path in sorted(Path(workdir).glob('*.log.json')):
        epochs, bad, count = defaultdict(lambda: defaultdict(list)), [], 0
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    bad.append(number)
                    continue
                if row.get('mode') != 'train':
                    continue
                count += 1
                for key, value in row.items():
                    if key in ('lr', 'grad_norm', 'center_size_positive_count') or 'loss' in key:
                        if isinstance(value, (float, int)):
                            epochs[str(row.get('epoch', 'unknown'))][key].append(value)
        logs.append(dict(path=str(path), sha256=sha(path), train_records=count,
            malformed_lines=bad, epochs={e:{k:dict(stats(v), nonfinite_count=sum(
                not math.isfinite(float(x)) for x in v), above_clip10_count=(
                sum(x > 10 for x in v) if k == 'grad_norm' else None)) for k,v in fields.items()}
                for e,fields in epochs.items()}))
    return dict(logs=logs, missing=not logs, interpretation=(
        'Separate log files may represent interrupted runs. Logged grad_norm is interpreted '
        'as preclip only if the active hook confirms it. Aggregated D loss cannot recover '
        'historical center/size parts. Nonfinite losses silently zeroed in the head cannot '
        'be excluded from finite log values alone.'))


def config_differences(actual, expected, prefix=''):
    """Exact structural differences, retaining list positions and missing keys."""
    if isinstance(actual,dict) and isinstance(expected,dict):
        result=[]
        for key in sorted(set(actual)|set(expected)):
            path=prefix+'.'+str(key) if prefix else str(key)
            if key not in actual or key not in expected:
                result.append(dict(path=path,actual=actual.get(key),expected=expected.get(key)))
            else:
                result.extend(config_differences(actual[key],expected[key],path))
        return result
    if isinstance(actual,(list,tuple)) and isinstance(expected,(list,tuple)):
        if len(actual)!=len(expected):
            return [dict(path=prefix,actual=actual,expected=expected)]
        return [row for i,(a,b) in enumerate(zip(actual,expected))
                for row in config_differences(a,b,prefix+'['+str(i)+']')]
    return [] if actual==expected else [dict(path=prefix,actual=actual,expected=expected)]


def checkpoint_contract(meta, expected):
    from mmcv import Config
    if not isinstance(meta.get('config'),str):
        return dict(status='MISSING_CONFIG_METADATA_UNVERIFIED')
    parsed=Config.fromstring(meta['config'],'.py').to_dict()
    fields=['model','data','optimizer','optimizer_config','lr_config','runner','load_from','resume_from']
    differences=config_differences({k:parsed.get(k) for k in fields},
                                   {k:expected.to_dict().get(k) for k in fields})
    return dict(status='MATCH' if not differences else 'DIFFERENCES_REVIEW_REQUIRED',
                differences=differences)


def set_probe_phase(head,phase):
    from crane_project.tools.preflight_port_center_size_v1 import set_assignment_phase
    set_assignment_phase(head,phase)
    if phase=='transition_o2m_to_o2o':
        # Halfway through the actual top-k ramp; tau/classification already warm.
        head.assigner._local_call_count=3*head.assigner.o2m_warmup_iters


def summarize_train(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row['arm'],row['phase'],row['scale'])].append(row)
    result=[]
    for (arm,phase,scale),values in sorted(groups.items()):
        gradient={}
        for scope in values[0]['gradients']:
            gradient[scope]=dict(
                norms={k:stats([v['gradients'][scope]['norms'][k] for v in values])
                       for k in values[0]['gradients'][scope]['norms']},
                ratios={k:stats([v['gradients'][scope]['ratios'][k] for v in values])
                        for k in values[0]['gradients'][scope]['ratios']},cosines={})
            for key in values[0]['gradients'][scope]['cosines']:
                cosines=[v['gradients'][scope]['cosines'][key] for v in values
                         if v['gradients'][scope]['cosines'][key] is not None]
                gradient[scope]['cosines'][key]=dict(stats(cosines),negative_count=sum(x<0 for x in cosines),
                    negative_fraction=sum(x<0 for x in cosines)/len(cosines) if cosines else None)
        result.append(dict(arm=arm,phase=phase,scale=scale,batches=len(values),
            losses={k:stats([v['losses'][k] for v in values]) for k in values[0]['losses']},
            gradient=gradient,clip_active_batches=sum(v['gradient_before_clip']>10 for v in values),
            symkld_loss_cap_positives=sum(v['positives'].get('symkld_loss_cap_count',0) for v in values),
            positive_count=sum(v['positives']['n'] for v in values)))
    return result


def load_frozen_val(cfg, config_path, sweep, expected_epoch):
    """Verify frozen selection and read existing PKL/TXT WITHOUT manifest writes."""
    import pickle
    from mmrotate.datasets import build_dataset
    from crane_project.tools.ckpt_sweep import annotation_set_sha256
    from crane_project.tools.audit_port_train_val_geometry_v1 import make_row, raw_box, validate_export, parse_dota_txt
    selection_path = Path(sweep) / 'sweep_results.json'
    selection = json.loads(selection_path.read_text())
    selected = selection['selected_checkpoint']
    if selected != expected_epoch or selection.get('evidence_role') != 'source_val_checkpoint_selection':
        raise ValueError('Requires frozen VAL selection '+expected_epoch)
    record = selection['all_checkpoints'][selected]
    checkpoint, pkl = Path(record['checkpoint']).resolve(), Path(record['results_pkl']).resolve()
    ann = (ROOT / 'crane_project/data/crane_grab_port_day2night_v1/val/annfiles').resolve()
    actual_ann = (Path(cfg.data.val.data_root) / cfg.data.val.ann_file).resolve()
    if (actual_ann != ann or selection['config_sha256'] != sha(config_path)
            or Path(selection['selected_path']).resolve() != checkpoint
            or record['checkpoint_sha256'] != sha(checkpoint)
            or record['results_pkl_sha256'] != sha(pkl)
            or selection['source_val_annotations_sha256'] != annotation_set_sha256(str(ann))):
        raise ValueError('Frozen VAL artifact identity mismatch')
    spec = deepcopy(cfg.data.val)
    spec.test_mode = True
    dataset = build_dataset(spec)
    with pkl.open('rb') as stream:
        predictions = pickle.load(stream)
    if len(dataset) != 887 or len(predictions) != 887:
        raise ValueError('Requires 887 ordered existing VAL predictions')
    rows = []
    for i, pred in enumerate(predictions):
        row = make_row(dataset, i, raw_box(pred), 'val')
        validate_export(pred, parse_dota_txt(str(pkl.parent/'Task1_grab'/(row['image']+'.txt'))), 1)
        rows.append(row)
    if {r['sequence'] for r in rows} != {'real_seq07', 'real_seq14', 'sim_seq10'}:
        raise ValueError('Unexpected VAL sequences')
    return rows, dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        pkl=str(pkl), pkl_sha256=sha(pkl), selection_sha256=sha(selection_path),
        selected_epoch=selected, config_sha256=sha(config_path),
        annotation_sha256=selection['source_val_annotations_sha256'])


def runs(rows, condition):
    """No-output or RIoU failure runs; reset at sequence or frame-number gap."""
    result, active, previous = [], [], None
    for row in sorted(rows, key=lambda r:(r['sequence'], r['frame_id'])):
        key = (row['sequence'], row['frame_id'])
        contiguous = previous and key[0] == previous[0] and key[1] == previous[1]+1
        if not contiguous or not condition(row):
            if active:
                result.append(dict(sequence=active[0]['sequence'], first=active[0]['image'],
                                   last=active[-1]['image'], length=len(active)))
                active = []
        if condition(row):
            active.append(row)
        previous = key
    if active:
        result.append(dict(sequence=active[0]['sequence'], first=active[0]['image'],
                           last=active[-1]['image'], length=len(active)))
    return sorted(result, key=lambda r:(-r['length'], r['sequence'], r['first']))


def val_analysis(b, d):
    from crane_project.tools.audit_port_train_val_geometry_v1 import paired_report, summarize
    from crane_project.tools.compare_port_center_size_val_v1 import name_arms
    if [r['image'] for r in b] != [r['image'] for r in d]:
        raise ValueError('VAL order differs')
    shared = [(a,c) for a,c in zip(b,d) if a['metrics']['output'] and c['metrics']['output']]
    severe = {r['image'] for r in b if r['domain']=='real' and r['metrics']['output'] and r['metrics']['riou']==0}
    ordinary = [(a,c) for a,c in shared if a['domain']=='real' and a['image'] not in severe]
    changes = []
    fields = ['riou', 'center_error_px', 'long_edge_relative_error', 'short_edge_relative_error', 'angle_error_deg']
    for a,c in zip(b,d):
        row = dict(image=a['image'], domain=a['domain'], sequence=a['sequence'],
            b_output=a['metrics']['output'], d_output=c['metrics']['output'],
            b_center_hit=a['metrics']['center_hit'], d_center_hit=c['metrics']['center_hit'],
            b_historical_severe=a['image'] in severe)
        if a['metrics']['output'] and c['metrics']['output']:
            row['d_minus_b'] = {k:c['metrics'][k]-a['metrics'][k] for k in fields}
        changes.append(row)
    paired = {}
    for name, pairs in [('real_shared',[(a,c) for a,c in shared if a['domain']=='real']),
                        ('real_ordinary_shared',ordinary),
                        ('sim_shared',[(a,c) for a,c in shared if a['domain']=='sim'])]:
        paired[name] = dict(n=len(pairs), b=summarize([a for a,c in pairs]), d=summarize([c for a,c in pairs]),
            delta={k:dict(stats([c['metrics'][k]-a['metrics'][k] for a,c in pairs]),
                improved_count=sum((c['metrics'][k]>a['metrics'][k]) if k=='riou' else
                    (c['metrics'][k]<a['metrics'][k]) for a,c in pairs)) for k in fields})
    temporal = {arm:{seq:dict(no_output=runs([r for r in rows if r['sequence']==seq],
                            lambda r:not r['metrics']['output']),
                    riou_failure=runs([r for r in rows if r['sequence']==seq],lambda r:r['metrics']['riou']<.5))
                    for seq in sorted({r['sequence'] for r in rows})}
                for arm,rows in [('b',b),('d',d)]}
    return dict(summary=name_arms(paired_report(b,d,True)), paired=paired, changes=changes,
        temporal=temporal, historical_severe_b_frames=sorted(severe),
        rows=dict(b=b,d=d), note='Paired deltas condition on shared outputs. All-frame coverage '
        'remains separately reported. Failure runs use identical sequence/gap rules; not control-time reliability.')


def runtime_probe(cfg, identities, gpu, per_domain, progress_path=None, control_cfg=None):
    import numpy as np
    import torch
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    from crane_project.tools.preflight_port_center_size_v1 import fixed_train_specs
    from crane_project.tools.ckpt_sweep import annotation_set_sha256
    torch.cuda.set_device(gpu)
    random.seed(1701); np.random.seed(1701); torch.manual_seed(1701); torch.cuda.manual_seed_all(1701)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    datasets = {s:build_dataset(fixed_train_specs(cfg,s)) for s in (1.,.5)}
    if any([len(p) for p in ds.datasets] != [1810,748] for ds in datasets.values()):
        raise ValueError('TRAIN split differs')
    # Check metadata order for both views before accessing image indices.
    first_infos=ordered_data_infos(datasets[1.])
    half_infos=ordered_data_infos(datasets[.5])
    if [i['filename'] for i in first_infos]!=[i['filename'] for i in half_infos]:
        raise ValueError('clean/half TRAIN order differs')
    indices = pick_train_indices(datasets[1.],per_domain,1701)
    real = [i for i in indices if i<1810]; sim = [i for i in indices if i>=1810]
    if len(real)!=per_domain or len(sim)!=per_domain:
        raise ValueError('Unbalanced diagnostic sample')
    rows, problems, snapshots = [], [], {}
    for arm in ('b','d'):
        model = build_detector(deepcopy(cfg.model))
        loaded = load_checkpoint(model,identities[arm]['checkpoint'],map_location='cpu',strict=True)
        meta = loaded.get('meta',{})
        snapshots[arm] = dict(epoch=meta.get('epoch'), iter=meta.get('iter'),
            epoch_matches_frozen_selection=(meta.get('epoch')==int(identities[arm]['selected_epoch'].split('_')[-1])
                if meta.get('epoch') is not None and 'selected_epoch' in identities[arm] else None),
            seed=meta.get('seed'), config_text=meta.get('config'),
            saved_config_contract=checkpoint_contract(meta,control_cfg if arm=='b' else cfg),
            note='Recorded metadata is evidence, not proof of every runtime step. Snapshot '
                 'phases below are diagnostic interventions, not restored historical phases.')
        del loaded
        model.cuda(gpu).train()
        parameter_sha_before=parameter_digest(model)
        head = model.bbox_head
        groups = [('regression_conv',[head.retina_reg.weight,head.retina_reg.bias]),
                  ('classification_conv',[head.retina_cls.weight,head.retina_cls.bias]),
                  ('fpn_parameter_probe',[model.neck.fpn_convs[0].conv.weight]),
                  ('backbone_parameter_probe',[model.backbone.layer4[-1].conv3.weight])]
        captured, levels = [], []
        original_extra = head.center_size_compensation.forward
        original_single = head.loss_single
        def extra_forward(pred,target,weight=None,avg_factor=None,**kwargs):
            value = original_extra(pred,target,weight=weight,avg_factor=avg_factor,**kwargs)
            captured.append((pred,target,weight,avg_factor,value))
            return value
        def loss_single_observed(self,*args,**kwargs):
            # Decode ONLY for guard statistics with no extra autograd graph.
            cls,reg,anchors,labels = args[:4]
            with torch.no_grad():
                flat=reg.permute(0,2,3,1).reshape(-1,5)
                p=self.bbox_coder.decode(anchors.reshape(-1,5),flat)
                positive=(labels.reshape(-1)>=0)&(labels.reshape(-1)<self.num_classes)
                pp=p[positive]
                maximum=float(kwargs.get('decode_max_size',2048))
                positive_indices=positive.nonzero().flatten()
                levels.append(dict(positive_indices=positive_indices.cpu().tolist(),
                    positive_image_indices=(positive_indices//(flat.shape[0]//reg.shape[0])).cpu().tolist(),
                    normalization=float(kwargs['num_total_samples']), positive_count=int(positive.sum()),
                    center_clamp_count=int(((pp[:,:2]<0)|(pp[:,:2]>maximum-1)).any(-1).sum()),
                    edge_clamp_count=int(((pp[:,2:4]<1)|(pp[:,2:4]>maximum)).any(-1).sum()),
                    delta_wh_clip_coordinate_count=int((flat[positive,2:4].abs()>abs(math.log(16/1000))).sum())))
            return original_single(*args,**kwargs)
        head.center_size_compensation.forward=extra_forward
        head.loss_single=MethodType(loss_single_observed,head)
        for scale in (1.,.5):
            for phase in ('warmup_o2m','transition_o2m_to_o2o','late_o2o'):
                for batch_number, pair in enumerate(zip(real,sim)):
                    captured.clear(); levels.clear(); model.zero_grad(set_to_none=True)
                    set_probe_phase(head,phase)
                    phase_state=dict(assigner_calls=head.assigner._local_call_count,
                        classification_calls=int(head.loss_cls._local_iter))
                    batch=scatter(collate([datasets[scale][i] for i in pair],samples_per_gpu=2),[gpu])[0]
                    if any(m.get('flip',False) for m in batch['img_metas']):
                        raise ValueError('Unexpected flip')
                    losses=model(return_loss=True,**batch)
                    center,size = [],[]
                    for p,t,w,n,value in captured:
                        c,s=d_parts(p,t,w,n)
                        if not torch.allclose(c+s,value,rtol=2e-5,atol=1e-7):
                            raise RuntimeError('Decomposed D loss differs from actual emitted loss')
                        center.append(c); size.append(s)
                    bbox=sum_loss(losses['loss_bbox']); cls=sum_loss(losses['loss_cls'])
                    other_terms=[sum_loss(v) for k,v in losses.items() if 'loss' in k and
                                 k not in ('loss_bbox','loss_cls','loss_center_size_compensation')]
                    other=sum(other_terms) if other_terms else bbox*0.
                    terms=dict(symkld=bbox,classification=cls,other_losses=other,
                               d_center=sum(center),d_size=sum(size))
                    if not all(bool(torch.isfinite(v).all()) for v in terms.values()):
                        raise RuntimeError('Nonfinite diagnostic loss')
                    gradient=gradient_report(terms,groups)
                    pos=[p for p,t,w,n,v in captured]; targets=[t for p,t,w,n,v in captured]
                    direct={}
                    decoded_gradients={}
                    for name,loss in [('d_center',sum(center)),('d_size',sum(size)),('symkld',bbox)]:
                        gs=gradient_vector(loss,pos)
                        decoded_gradients[name]=gs
                        direct[name]={key:grad_norm([g[:,j:j+1] if g is not None else None for g in gs])
                                      for j,key in enumerate(('x','y','w','h','angle'))}
                    domain_direct={}
                    for image_index,domain in enumerate(('real','sim')):
                        domain_grads={}
                        for name,gradients in decoded_gradients.items():
                            transformed=[]
                            for g,(p,t,w,n,v),level in zip(gradients,captured,levels):
                                if g is None or not len(t): transformed.append(None); continue
                                # Dimensionless local coordinates: center/GTshort, log edges, radians.
                                short=t[:,2:4].min(-1).values
                                factors=torch.stack([short,short,p[:,2],p[:,3],torch.ones_like(short)],-1).detach()
                                ids=torch.tensor(level['positive_image_indices'],device=g.device,dtype=torch.long)
                                transformed.append((g*factors)[ids==image_index])
                            domain_grads[name]=tuple(transformed)
                        domain_direct[domain]=dict(
                            positive_count=sum(i==image_index for l in levels for i in l['positive_image_indices']),
                            dimensionless_norms={k:grad_norm(v) for k,v in domain_grads.items()},
                            center_vs_symkld=grad_cos(domain_grads['d_center'],domain_grads['symkld']),
                            size_vs_symkld=grad_cos(domain_grads['d_size'],domain_grads['symkld']),
                            note='Direct decoded-positive gradients; not domain-specific FPN/backbone gradients.')
                    if direct['d_center']['angle']!=0 or direct['d_size']['angle']!=0:
                        problems.append('Unexpected direct D angle gradient')
                    total=bbox+cls+other+(sum(center)+sum(size) if arm=='d' else bbox*0.)
                    total.backward()
                    parameters=[p for p in model.parameters() if p.requires_grad]
                    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
                        raise RuntimeError('Nonfinite complete objective gradient')
                    clip=cfg.optimizer_config.grad_clip
                    before=float(torch.nn.utils.clip_grad_norm_(parameters,clip.max_norm,norm_type=clip.norm_type))
                    after=grad_norm([p.grad for p in parameters])
                    row=dict(arm=arm,phase=phase,scale=scale,batch_number=batch_number,
                        phase_state_before_forward=phase_state,
                        images=[Path(m['filename']).stem for m in batch['img_metas']],
                        image_sha256=[sha(m['filename']) for m in batch['img_metas']],
                        input_shapes=[list(m['img_shape']) for m in batch['img_metas']],
                        losses={k:scalar(v) for k,v in terms.items()}, gradients=gradient,
                        direct_decoded_box_gradient=direct, levels=deepcopy(levels),
                        domain_direct_gradient=domain_direct,
                        positives=positive_geometry(torch.cat(pos),torch.cat(targets)),
                        gradient_before_clip=before,gradient_after_clip=after,
                        clip_multiplier=min(1.,float(clip.max_norm)/(before+1e-6)),
                        actual_objective_includes_d=arm=='d')
                    rows.append(row)
                    if progress_path is not None:
                        with Path(progress_path).open('a') as stream:
                            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    print('TRAIN',arm,phase,'scale',scale,'batch',batch_number+1,'/',per_domain,
                          'D center/size',row['losses']['d_center'],row['losses']['d_size'],flush=True)
                    model.zero_grad(set_to_none=True)
                    del batch,losses,terms,center,size,other_terms,bbox,cls,other,total,pos,targets,gs,c,s,value,p,t,w,n
                    del decoded_gradients,domain_grads,transformed,gradients,g,factors,ids,short
                    captured.clear()
        parameter_sha_after=parameter_digest(model)
        if parameter_sha_before!=parameter_sha_after:
            raise RuntimeError('Diagnostic unexpectedly changed model parameters')
        snapshots[arm]['parameter_sha256_before']=parameter_sha_before
        snapshots[arm]['parameter_sha256_after']=parameter_sha_after
        snapshots[arm]['parameters_unchanged']=True
        head.center_size_compensation.forward=original_extra; head.loss_single=original_single
        del model,head,groups,original_extra,original_single,extra_forward,loss_single_observed,parameters
        torch.cuda.empty_cache()
    assignment=[]
    indexed={(r['arm'],r['phase'],r['scale'],r['batch_number']):r for r in rows}
    for r in rows:
        if r['arm']!='b': continue
        d=indexed[('d',r['phase'],r['scale'],r['batch_number'])]
        a={(l,i) for l,v in enumerate(r['levels']) for i in v['positive_indices']}
        b={(l,i) for l,v in enumerate(d['levels']) for i in v['positive_indices']}
        assignment.append(dict(phase=r['phase'],scale=r['scale'],images=r['images'],
            b_positive_count=len(a),d_positive_count=len(b),intersection=len(a&b),
            union=len(a|b),jaccard=len(a&b)/len(a|b) if a|b else None))
    return dict(rows=rows,summary=summarize_train(rows),assignment_overlap=assignment,checkpoint_metadata=snapshots,problems=problems,
        sample_indices=indices,per_domain=per_domain,seed=1701,
        source_annotations_sha256={s:annotation_set_sha256(str(ROOT/cfg.data.train[0].data_root/s/'annfiles'))
                                   for s in ('train','train_sim')},
        gradient_scope='Full main regression/classification convolutions; one explicitly labeled '
        'FPN and one backbone parameter probe. These are NOT full FPN/backbone gradient norms. '
        'Complete actual objective uses all trainable parameters for clip statistics.',
        phase_scope='Both checkpoints replay identical TRAIN views under forced early/transition/late '
        'assignment schedules. Late snapshot under early assignment is not training history.',
        optimizer_steps=0, training_epochs=0,
        resources=dict(torch_version=torch.__version__,gpu_name=torch.cuda.get_device_name(gpu),
                       peak_allocated_mib=torch.cuda.max_memory_allocated(gpu)/2**20))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--b-sweep',default='work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/val_sweep_port_v1')
    ap.add_argument('--d-sweep',default='work_dirs/crane_symeood_k1_port_day2night_center_size_d_v1/val_sweep_port_v1')
    ap.add_argument('--gpu',type=int,default=3)
    ap.add_argument('--per-domain',type=int,default=8)
    ap.add_argument('--artifact-only',action='store_true',help='Existing VAL and logs only; no model execution')
    ap.add_argument('--config-only',action='store_true',help='Local resolved-config check only')
    ap.add_argument('--resume-empty-progress',action='store_true',
                    help='Reuse existing VAL sidecar only when no TRAIN batch completed')
    ap.add_argument('--out-json',required=True)
    args=ap.parse_args()
    os.chdir(ROOT)
    out=Path(args.out_json).resolve()
    if out.exists(): raise FileExistsError(out)
    if args.resume_empty_progress and (args.config_only or args.artifact_only):
        raise ValueError('Resume requires the complete TRAIN diagnostic mode')
    if args.per_domain<5 or args.per_domain>32: raise ValueError('per-domain must be 5..32')
    from crane_project.tools.preflight_port_center_size_v1 import check_configs,CONTROL,EXPERIMENT
    b,d=check_configs()
    sources=[CONTROL,EXPERIMENT,Path(__file__).resolve(),ROOT/'mmrotate/models/dense_heads/sym_eood_head.py',
             ROOT/'mmrotate/models/losses/center_size_compensation.py',
             ROOT/'mmrotate/models/losses/sym_kld_loss.py',ROOT/'mmrotate/models/losses/sym_kld_calculator.py',
             ROOT/'mmrotate/core/bbox/assigners/sym_pola.py']
    report=dict(protocol='port_center_size_d_diagnosis_v1',evidence_role='train_val_only_diagnosis',
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        resolved_config_check='PASS: only D compensation and work_dir differ',
        limitations=['No TEST loading, checkpoint reselection, hyperparameter search or optimizer.',
            'Frozen endpoint gradients do not establish the cause of historical training outcomes.',
            'B/D VAL-selected epochs differ under the fixed protocol; single-seed uncertainty remains.',
            'Batch gradients mix one real and one sim image; do not interpret as separate domain gradients.',
            'The normalizer sums positives across levels on the current rank; no all-rank reduce is '
            'performed. Per-image max(npos,1) applies. B and D share this convention.',
            'Finite logs cannot exclude silently zeroed NaN/Inf original main-head losses.'])
    if args.config_only:
        report['status']='CONFIG_ONLY_PASS_RUNTIME_UNVERIFIED'
    else:
        artifact_path=out.with_suffix('.artifacts.json')
        progress_path=out.with_suffix('.progress.jsonl')
        if args.resume_empty_progress:
            cached=reuse_artifacts(artifact_path,progress_path,report['sources'],
                dict(b=args.b_sweep,d=args.d_sweep),dict(b=b,d=d))
            ids=cached['identities']
            report['identities']=ids
            report['val']=cached['val']
            report['resumed_artifacts']=dict(path=str(artifact_path),sha256=sha(artifact_path),
                original_sources=cached['sources'],completed_train_batches_before_resume=0)
        else:
            if not args.artifact_only and (artifact_path.exists() or progress_path.exists()):
                raise FileExistsError('Sidecars already exist; use --resume-empty-progress only '
                                     'if no TRAIN batch completed, or choose a new output name')
            rows,ids={},{}
            for arm,cfg,path,sweep,epoch in [('b',b,CONTROL,args.b_sweep,'epoch_24'),
                                           ('d',d,EXPERIMENT,args.d_sweep,'epoch_22')]:
                rows[arm],ids[arm]=load_frozen_val(cfg,path,Path(sweep),epoch)
            report['identities']=ids
            report['val']=val_analysis(rows['b'],rows['d'])
        report['logs']={a:read_logs(Path(p).parent) for a,p in [('b',args.b_sweep),('d',args.d_sweep)]}
        if args.artifact_only:
            report['status']='ARTIFACTS_READ_RUNTIME_UNVERIFIED'
        else:
            # Preserve artifact findings and completed batches even if GPU execution fails.
            artifact_path.parent.mkdir(parents=True,exist_ok=True)
            if not args.resume_empty_progress:
                with artifact_path.open('x') as stream:
                    json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
                progress_path.touch(exist_ok=False)
            report['train']=runtime_probe(d,ids,args.gpu,args.per_domain,progress_path,control_cfg=b)
            report['status']='DIAGNOSIS_COMPLETE_REVIEW_REQUIRED'
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print('Saved:',out,'status:',report['status'],flush=True)


if __name__=='__main__':
    main()
