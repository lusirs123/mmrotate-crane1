#!/usr/bin/env python3
"""Read-only standard-scale inference on the snapshot's allowed TRAIN only."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def summarize(rows):
    import numpy as np
    from crane_project.utils.port_reliability_complementarity_v1 import center, grouped
    result = {}
    for name, part in grouped(rows).items():
        out = [r for r in part if r['pred'] is not None]
        entry = center(part)
        entry['missing'] = len(part)-len(out)
        entry['size_correct'] = sum(r['size_correct'] for r in out)
        entry['size_correct_coverage_all_frames'] = entry['size_correct']/len(part)
        entry['size_correct_rate_on_outputs'] = entry['size_correct']/len(out) if out else None
        if out:
            errors = np.asarray([r['signed_edge_error'] for r in out])
            entry.update(edge_MAE=np.abs(errors).mean(0).tolist(),
                edge_bias=errors.mean(0).tolist(), edge_P95=np.quantile(np.abs(errors), .95, axis=0).tolist(),
                score_quantiles=np.quantile([r['pred'][5] for r in out], [0,.05,.5,.95,1]).tolist())
        result[name] = entry
    return result


def run(source, out):
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.utils import import_modules_from_strings
    from mmdet.datasets.dataset_wrappers import ConcatDataset
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    from crane_project.tools import run_port_reliability_oof_v1 as original
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as frozen
    from crane_project.utils import port_reliability_oof_training_v2 as repair
    from crane_project.utils.port_oof_task_budget_v1 import check_task_config

    folder = source/'result/A'
    checkpoint = folder/'prevention/before_anomaly.pt'
    completion = json.loads((folder/'prevention/completion.json').read_text())
    pin = original.sha(checkpoint)
    if completion['snapshot_sha256']['before_anomaly.pt'] != pin:
        raise ValueError('Snapshot SHA differs from original completion')
    saved = torch.load(str(checkpoint), map_location='cpu')
    if (not saved['diagnostic_only'] or saved['iteration'] != 6066 or
            completion['completed_updates'] != 6066 or
            completion['status'] != 'ANOMALY_QUARANTINED_BEFORE_OPTIMIZER'):
        raise ValueError('Only the fixed 6066-update pre-forward diagnostic state supported')
    cfg_path = folder/'detector_config.py'
    cfg = Config.fromfile(str(cfg_path)); check_task_config(cfg)
    membership = json.loads((folder/'membership.json').read_text())
    allowed = membership['train_images']
    excluded = membership['predict_images']
    if len(allowed) != 1659 or len(set(allowed)) != 1659 or set(allowed)&set(excluded):
        raise ValueError('Auxiliary A allowed TRAIN membership differs')
    if cfg.model.test_cfg.score_thr != .05 or cfg.model.test_cfg.max_per_img != 1:
        raise ValueError('Original score/output-count rules changed')
    # The inherited inference helper validates the original repaired optimizer
    # type. Validate the real task-budget config above, then use a copy solely
    # for its unchanged pixel-transform checks; no optimizer is constructed.
    inference_cfg = deepcopy(cfg)
    inference_cfg.optimizer_config.type = 'PortStrictOptimizerHook'
    inference_cfg.optimizer_config.memory_guard = deepcopy(repair.MEMORY_GUARD)
    specs = repair.inference_specs(frozen.g.fixed_specs(Config.fromfile(str(original.CONFIG)), 1.), inference_cfg)
    for spec in specs:
        spec['data_root'] = str((folder/'data').resolve())+'/'
    # The copied deterministic pipeline is used exclusively on allowed TRAIN paths.
    data = ConcatDataset([build_dataset(spec) for spec in specs])
    infos = frozen.infos(data)
    names = [Path(info['filename']).stem for info in infos]
    if sorted(names) != sorted(allowed) or len(names) != len(set(names)):
        raise ValueError('Dataset is not precisely the allowed TRAIN')
    manifest_path = original.DATA/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    identities = {r['id']:r for r in manifest['records'] if r['id'] in set(allowed)}
    if set(identities) != set(allowed): raise ValueError('Missing original TRAIN identities')
    for name in allowed:
        record = identities[name]
        if record['split'] not in ('train','train_sim'):
            raise ValueError('Non-TRAIN role')
        for field in ('images','annfiles'):
            if original.sha(original.DATA/record[field]['path']) != record[field]['sha256']:
                raise ValueError('Input bytes differ: '+name)
    contract = dict(purpose='actual detector function on allowed TRAIN, not OOF qualification',
        snapshot_sha256=pin, completed_training_updates=6066, diagnostic_only=True,
        config_sha256=original.sha(cfg_path), script_sha256=original.sha(Path(__file__)),
        manifest_sha256=original.sha(manifest_path), frames=1659, scale=1., precision='FP32',
        score_threshold=.05, max_per_image=1, optimizer_updates=0,
        ImageNet_reinitialized=False, midpoint_loaded=False, VAL_TEST_datasets_built=False,
        excluded_fold_evaluated=False, allowed_images=sorted(allowed),
        inference_revision=repair.INFERENCE_REVISION)
    original.write(out/'contract.json',contract)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import_modules_from_strings(**cfg.custom_imports)
    spec = deepcopy(cfg.model); spec.backbone.pop('init_cfg',None); spec.backbone.pretrained = None
    model = build_detector(spec,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg')).cuda()
    model.load_state_dict(saved['model'],strict=True)
    del saved
    model.requires_grad_(False); model.eval()
    initial_state = frozen.g.state_digest(model)
    rows = []; start = time.monotonic()
    with (out/'predictions.jsonl').open('x') as stream, torch.inference_mode():
        for index in sorted(range(len(names)), key=lambda i:names[i]):
            name = names[index]; identity = identities[name]
            truth = np.asarray(infos[index]['ann']['bboxes'])
            if truth.shape != (1,5) or not np.isfinite(truth).all() or (truth[0,2:4]<=0).any():
                raise ValueError('Expected exactly one valid original-coordinate GT: '+name)
            batch = scatter(collate([data[index]],samples_per_gpu=1),[0])[0]
            # GT is evaluated offline only; simple_test receives pixels and transform metadata.
            prediction = frozen.g.flatten_prediction(model.simple_test(batch['img'][0],batch['img_metas'][0],rescale=True))
            row = dict(image=name,split=identity['split'],domain='sim' if identity['split']=='train_sim' else 'real',
                sequence=identity['sequence'],gt=truth[0].tolist(),pred=prediction[0].tolist() if len(prediction) else None)
            if len(prediction):
                errors = np.sort(prediction[0,2:4])[::-1]/np.sort(truth[0,2:4])[::-1]-1
                if not np.isfinite(errors).all() or (prediction[0,2:4]<=0).any():
                    raise ValueError('Invalid output geometry: '+name)
                row.update(signed_edge_error=errors.tolist(),size_correct=bool((np.abs(errors)<=.1).all()),
                    center_error=float(np.linalg.norm(prediction[0,:2]-truth[0,:2])))
            stream.write(json.dumps(row,allow_nan=False)+'\n'); rows.append(row)
            del batch, prediction
            if len(rows)%100 == 0 or len(rows)==32 or len(rows)==len(names):
                torch.cuda.synchronize()
                elapsed = time.monotonic()-start
                original.emit(dict(stage='read_only_detection',done=len(rows),total=len(names),
                    elapsed_seconds=elapsed,remaining_estimate_seconds=(len(names)-len(rows))*elapsed/len(rows),
                    allocated_MiB=torch.cuda.memory_allocated()/2**20,reserved_MiB=torch.cuda.memory_reserved()/2**20))
            if torch.cuda.max_memory_reserved()>4*2**30:
                raise RuntimeError('Inference exceeds fixed4GiB resource ceiling')
    state_unchanged = frozen.g.state_digest(model)==initial_state
    snapshot_unchanged = original.sha(checkpoint)==pin
    if not state_unchanged or not snapshot_unchanged: raise ValueError('Inference mutated model or snapshot')
    report = dict(contract=contract,statistics=summarize(rows),elapsed_inference_seconds=time.monotonic()-start,
        model_parameters_and_buffers_unchanged=state_unchanged,snapshot_unchanged=snapshot_unchanged,
        optimizer_updates=0,prevention_passed=False,OOF_source_qualified=False,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
    original.write(out/'report.json',report)
    original.emit(dict(stage='read_only_detection_completed',statistics=report['statistics']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args();source,out=args.source.resolve(),args.out.resolve()
    if any(ROOT/'work_dirs' not in p.parents for p in (source,out)): raise ValueError('Project work_dirs only')
    out.mkdir(parents=True,exist_ok=False);run(source,out)
