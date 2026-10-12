#!/usr/bin/env python3
"""Trace two saved TRAIN batches through loss, decode, encoding and features.

No dataloader, no parameter updates, no VAL/TEST, no pretrained download.
"""
import argparse
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_reliability_oof_followup_v1 as parent
from crane_project.utils import port_reliability_oof_gradient_chain_v1 as contract


def run_one(source, out, boundary, gpu):
    import torch
    from mmcv import Config
    from mmcv.runner import build_optimizer
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from mmrotate.models.losses.sym_kld_calculator import sym_kld, _xywha_to_gaussian
    from mmrotate.core.hooks.port_oof_trace_hook import (
        PortOOFTraceHook, cpu_tree, restore_rng, restore_runtime, gradient_summary, sum_loss)
    from types import SimpleNamespace
    parent.checked(source); parent.old.check_fold(source, 'A')
    completion = json.loads((source/'A/trace/completion.json').read_text())
    name = 'before_previous_update.pt' if boundary == 'previous' else 'before_anomaly.pt'
    path = source/'A/trace'/name
    pin = parent.old.sha(path)
    if completion['snapshot_sha256'].get(name) != pin:
        raise ValueError('Snapshot SHA changed')
    snapshot = torch.load(str(path), map_location='cpu')
    contract.validate_boundary(snapshot, completion, boundary)
    cfg = Config.fromfile(str(source/'A/detector_config.py'))
    parent.old.training_repair.validate_config(cfg)
    import_modules_from_strings(**cfg.custom_imports)
    spec = deepcopy(cfg.model)
    spec.backbone.pop('init_cfg', None); spec.backbone.pretrained = None
    torch.cuda.set_device(gpu)
    model = build_detector(spec, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')).cuda(gpu)
    model.load_state_dict(snapshot['model'], strict=True)
    optimizer = build_optimizer(model, cfg.optimizer)
    optimizer.load_state_dict(snapshot['optimizer'])
    def device(value):
        if isinstance(value, torch.Tensor): return value.cuda(gpu)
        if isinstance(value, dict): return {k:device(v) for k,v in value.items()}
        if isinstance(value, list): return [device(v) for v in value]
        if isinstance(value, tuple): return tuple(device(v) for v in value)
        return value
    batch = {k:(cpu_tree(v) if k == 'img_metas' else device(v)) for k,v in snapshot['batch'].items()}
    trace = PortOOFTraceHook(out/'unused')
    trace.model = model; trace.before = snapshot
    trace.runner = SimpleNamespace(model=model, optimizer=optimizer, iter=snapshot['iteration'])
    trace.install(model, capture_failure=False)
    records = []; feature_graph = []; output_inputs = []; context = [None]
    original_features = model.extract_feat
    def features(_model, *args, **kwargs):
        value = original_features(*args, **kwargs)
        if feature_graph: raise ValueError('Unexpected additional feature forward')
        feature_graph.extend(value)
        return value
    model.extract_feat = types.MethodType(features, model)
    head = model.bbox_head; original_single = head.loss_single
    # The trace wrapper has a variadic signature. Bind against the class definition.
    signature = inspect.signature(type(head).loss_single)
    def single(_head, *args, **kwargs):
        bound = signature.bind(head, *args, **kwargs); bound.apply_defaults()
        context[0] = dict(bound.arguments)
        try: return original_single(*args, **kwargs)
        finally: context[0] = None
    head.loss_single = types.MethodType(single, head)
    loss_object = head.loss_bbox; original_loss = loss_object.forward
    def capture(_loss, pred, target, weight=None, avg_factor=None, reduction_override=None):
        value = original_loss(pred, target, weight, avg_factor, reduction_override)
        if context[0] is not None:
            c = context[0]; labels = c['labels'].reshape(-1)
            indices = ((labels >= 0) & (labels < head.num_classes)).nonzero().flatten()
            dense = c['bbox_pred']; anchors = c['anchors'].reshape(-1, 5)
            per_image = labels.numel()//dense.shape[0]
            records.append(dict(pred=pred, target=target, weight=weight,
                avg_factor=avg_factor, loss=value, dense=dense,
                anchors=anchors[indices], indices=indices,
                image_indices=(indices//per_image).cpu().tolist(),
                limit=float(c['decode_max_size']), level_shape=list(dense.shape[-2:])))
        return value
    loss_object.forward = types.MethodType(capture, loss_object)
    handle = head.retina_reg.register_forward_pre_hook(lambda m, args: output_inputs.append(args[0]))
    restore_runtime(model, snapshot['runtime']); restore_rng(snapshot['rng'])
    value = model.train_step(batch, optimizer)
    handle.remove()
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    params = [p for _,p in named]
    def check_memory():
        torch.cuda.synchronize(gpu)
        if torch.cuda.max_memory_reserved(gpu) > contract.LIMIT_BYTES:
            raise RuntimeError('Snapshot attribution exceeds fixed 4GiB limit')
    def grads(term, inputs):
        value = torch.autograd.grad(term, inputs, allow_unused=True, retain_graph=True)
        check_memory(); return value
    total_grad = grads(value['loss'], params)
    total_summary = gradient_summary(total_grad, named); del total_grad
    original = snapshot['trace'] if boundary == 'previous' else json.loads((source/'A/trace/updates.jsonl').read_text().splitlines()[-1])
    comparison = parent.contract.compare_replay_trace(float(value['loss']), total_summary['all'], original)
    if not comparison['passed']: raise ValueError('Instrumented forward does not reproduce original snapshot')
    reg = sum_loss(trace.losses['loss_bbox'])
    if not records or not torch.allclose(sum(r['loss'] for r in records), reg):
        raise ValueError('Actual positive regression reduction not captured')
    reg_grads = grads(reg, params)
    reg_summary = gradient_summary(reg_grads, named)
    expected_grads = [None if g is None else g.detach().cpu() for g in reg_grads]
    del reg_grads
    feature_gradients = grads(reg, feature_graph)
    feature_stats = [dict(shape=list(f.shape), value_norm=float(f.detach().norm()),
        value_abs_max=float(f.detach().abs().max()), gradient_norm=float(g.detach().norm()) if g is not None else 0.)
        for f,g in zip(feature_graph, feature_gradients)]
    del feature_gradients
    input_grads = grads(reg, output_inputs)
    input_stats = [dict(shape=list(f.shape), value_norm=float(f.detach().norm()),
        value_abs_max=float(f.detach().abs().max()), gradient_norm=float(g.detach().norm()) if g is not None else 0.)
        for f,g in zip(output_inputs, input_grads)]
    del input_grads
    physical_grads = grads(reg, [r['pred'] for r in records])
    dense_grads = grads(reg, [r['dense'] for r in records])
    box_records = []
    for r,gp,gd in zip(records,physical_grads,dense_grads):
        dense = r['dense'].permute(0,2,3,1).reshape(-1,5)
        codes = dense[r['indices']].detach().clone().requires_grad_(True)
        expected_encoded = gd.permute(0,2,3,1).reshape(-1,5)[r['indices']]
        decoded = contract.bounded_decode(head.bbox_coder,r['anchors'],codes,r['limit'])
        if not torch.equal(decoded.detach(), r['pred'].detach()):
            raise ValueError('Independent decode differs from actual loss input')
        jacobian = torch.stack([torch.autograd.grad(decoded[:, k].sum(),codes,retain_graph=True)[0]
                                for k in range(5)],dim=1)
        vjp = torch.einsum('nij,ni->nj',jacobian,gp.detach())
        if not torch.allclose(vjp,expected_encoded,atol=1e-4,rtol=2e-4):
            raise ValueError('Decoder VJP does not reconstruct actual encoded gradient')
        leaf = r['pred'].detach().double().requires_grad_(True)
        target64 = r['target'].detach().double()
        weight64 = r['weight'].detach().double() if r['weight'] is not None else None
        ref_loss = contract.reference_loss(leaf,target64,weight64,r['avg_factor'],loss_object.loss_weight)
        reference_grad = torch.autograd.grad(ref_loss,leaf,retain_graph=True)[0]
        leaf32 = r['pred'].detach().clone().requires_grad_(True)
        ref32 = contract.reference_loss(leaf32,r['target'].detach(),r['weight'],r['avg_factor'],loss_object.loss_weight)
        ref32_grad = torch.autograd.grad(ref32,leaf32)[0]
        terms64 = contract.analytic_kld_terms(leaf,target64)
        raw_original = sym_kld(r['pred'].detach(),r['target'].detach())
        shape = terms64['shape']; raw_ref = terms64['raw']
        compression_slope = .5/(1+raw_ref.clamp(0,1e4)).sqrt()
        active = (raw_ref>0)&(raw_ref<120)
        weights = weight64 if weight64 is not None else torch.ones_like(raw_ref)
        multiplier = compression_slope*active.to(leaf.dtype)*weights/float(r['avg_factor'])*loss_object.loss_weight
        part_grads = {k:torch.autograd.grad((terms64[k]*multiplier.detach()).sum(),leaf,retain_graph=True)[0].detach()
                      for k in ('shape','center_p','center_q')}
        if not torch.allclose(sum(part_grads.values()),reference_grad,atol=1e-8,rtol=1e-7):
            raise ValueError('Principal-axis KLD terms do not reconstruct reference derivative')
        _,cov = _xywha_to_gaussian(r['pred'].detach())
        determinant32 = cov[:,0,0]*cov[:,1,1]-cov[:,0,1]*cov[:,1,0]
        determinant64 = (leaf[:,2].clamp(min=1)*leaf[:,3].clamp(min=1)/4).square()
        item = dict(level_shape=r['level_shape'],images=[Path(batch['img_metas'][i]['filename']).stem for i in r['image_indices']],
            positive_indices=r['indices'].cpu().tolist(),anchors=r['anchors'].detach().cpu().tolist(),
            encoded_predictions=codes.detach().cpu().tolist(),pred=r['pred'].detach().cpu().tolist(),
            GT=r['target'].detach().cpu().tolist(),**contract.canonical_error(r['pred'].detach(),r['target'].detach()),
            physical_gradient=gp.detach().cpu().tolist(),encoded_gradient=expected_encoded.detach().cpu().tolist(),
            decoder_jacobian=jacobian.detach().cpu().tolist(),decoder_vjp_max_error=float((vjp-expected_encoded).abs().max()),
            raw_KLD_original=raw_original.cpu().tolist(),raw_KLD_reference64=raw_ref.detach().cpu().tolist(),
            raw_KLD_parts={k:terms64[k].detach().cpu().tolist() for k in ('shape','center_p','center_q')},
            reference_part_gradients={k:g.cpu().tolist() for k,g in part_grads.items()},
            actual_loss=float(r['loss']),reference_loss64=float(ref_loss),reference_loss32=float(ref32),
            physical_gradient_relative_error64=float((gp.double()-reference_grad).norm()/(reference_grad.norm()+1e-12)),
            physical_gradient_relative_error32=float((gp-ref32_grad).norm()/(ref32_grad.norm()+1e-12)),
            reference_gradient64=reference_grad.detach().cpu().tolist(),
            covariance_determinant_relative_error=((determinant32.double()-determinant64).abs()/determinant64).detach().cpu().tolist())
        box_records.append(item)
    # Coordinate contributions are vectors, not additive norm percentages.
    summed = [None if g is None else torch.zeros_like(g) for g in expected_grads]
    contributions = {}
    for k,label in enumerate(('center_x','center_y','width','height','angle')):
        term = sum((r['pred'][:,k]*g[:,k].detach()).sum() for r,g in zip(records,physical_grads))
        values = grads(term,params)
        contributions[label] = gradient_summary(values,named)
        for i,g in enumerate(values):
            if g is not None:
                if summed[i] is None: summed[i] = g.detach().cpu().clone()
                else: summed[i].add_(g.detach().cpu())
        del values
    difference = sum(float((s.double()-g.double()).square().sum()) for s,g in zip(summed,expected_grads) if g is not None)**.5
    expected_norm = sum(float(g.double().square().sum()) for g in expected_grads if g is not None)**.5
    if difference > 1e-4 + expected_norm*2e-4:
        raise ValueError('Physical coordinate contributions differ from regression backward')
    model.load_state_dict(snapshot['model'],strict=True); restore_runtime(model,snapshot['runtime'])
    if parent.old.sha(path) != pin: raise ValueError('Original snapshot changed')
    report = dict(protocol=contract.VERSION,boundary=boundary,step=snapshot['iteration']+1,snapshot_sha256=pin,
        comparison=comparison,total_gradient=total_summary,regression_gradient=reg_summary,
        positive_boxes=box_records,physical_coordinate_parameter_gradients=contributions,
        coordinate_sum_relative_error=difference/(expected_norm+1e-12),FPN=feature_stats,regression_output_input=input_stats,
        raw_logs=value['log_vars'],parameter_updates=0,VAL_read=False,TEST_read=False,
        original_snapshot_unchanged=True,peak_reserved_MiB=torch.cuda.max_memory_reserved(gpu)/2**20,
        note='Norms at different coordinates have different units; channel contributions can cancel.')
    parent.old.write(out/(boundary+'_report.json'),parent.contract.json_safe(report))
    print(json.dumps(dict(boundary=boundary,step=report['step'],regression_norm=reg_summary['all'],
        physical_contributions={k:v['all'] for k,v in contributions.items()},peak_reserved_MiB=report['peak_reserved_MiB'])),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--boundary',choices=('previous','anomaly'),required=True)
    args=parser.parse_args();source=args.source.resolve();out=args.out.resolve()
    if any(ROOT/'work_dirs' not in p.parents for p in (source,out)):
        raise ValueError('All inputs/outputs must remain in project work_dirs')
    out.mkdir(parents=True,exist_ok=True)
    if (out/(args.boundary+'_report.json')).exists(): raise ValueError('Refuse report overwrite')
    run_one(source,out,args.boundary,args.gpu)


if __name__ == '__main__': main()
