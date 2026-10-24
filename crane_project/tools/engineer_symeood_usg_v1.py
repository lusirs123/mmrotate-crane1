"""Real TRAIN native-versus-extension validation, discarded updates only."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(out):
    import torch
    from mmcv import Config
    from mmdet.models import build_detector
    from mmdet.apis import set_random_seed
    from crane_project.utils.symeood_usg_v1 import tensor_digest
    from crane_project.tools.diagnose_symeood_c_frozen_v1 import batch_to_device
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    data = batch_to_device(torch.load(ROOT / 'work_dirs/symeood_ablation_v1/c_frozen_diagnostic_20261024_v1/inputs/batch_000.pt', map_location='cpu'))
    for reg, oldgroup in [('iou', '100'), ('kld', '110')]:
        initial = None
        init_hash = None
        for mode in ('U', 'S', 'G'):
            cfg = Config.fromfile(str(ROOT / ('crane_project/configs/crane_symeood_usg_%s_%s_v1.py' % (reg, mode.lower()))))
            set_random_seed(0, deterministic=False)
            model = build_detector(copy.deepcopy(cfg.model)).cuda()
            model.init_weights()
            now = tensor_digest(model.named_parameters())
            if initial is None:
                initial = {n:t.cpu().clone() for n,t in model.state_dict().items()}
                init_hash = now
            assert now == init_hash, 'Initialization differs across U/S/G'
            if mode == 'U' and reg == 'iou': common_init = now
            assert now == common_init, 'Public initialization differs across regressions'
            model.train()
            outputs=[]
            hook=model.bbox_head.register_forward_hook(lambda m,i,o:outputs.extend(o[0]))
            assignments=[]
            original_assign=model.bbox_head.assigner.assign
            def observed_assign(*args,**kwargs):
                result=original_assign(*args,**kwargs)
                assignments.append(tensor_digest([('gt_inds',result.gt_inds),('labels',result.labels)]))
                return result
            model.bbox_head.assigner.assign=observed_assign
            for mod in model.modules():
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm): mod.eval()
            # Same actual forward and targets: native U or G must be unchanged.
            for mature in (False, True):
                model.load_state_dict(initial)
                count = 8000 if mature else 0
                model.bbox_head.assigner._local_call_count = count
                model.bbox_head.loss_cls._local_iter.fill_(20000 if mature else 0)
                before_keys = set(model.state_dict())
                outputs.clear();assignments.clear()
                result = model.forward_train(**data)
                main = sum(result['loss_cls'])
                logit_grad=torch.autograd.grad(main,outputs,retain_graph=True)
                logit_grad_sha=tensor_digest([(str(i),g) for i,g in enumerate(logit_grad)])
                stats = model.bbox_head.usg_stats
                expected = stats[mode]
                assert abs(float(main.detach()) - expected) <= 1e-6 * max(abs(expected), 1e-8) + 1e-8
                other = {k:float(sum(v).detach()) if isinstance(v,list) else float(v.detach()) for k,v in result.items() if k != 'loss_cls'}
                total = sum(sum(v) if isinstance(v,list) else v for k,v in result.items() if 'loss' in k)
                assert bool(torch.isfinite(total))
                total.backward()
                nonzero = sum(p.grad is not None and bool(p.grad.count_nonzero()) for p in model.parameters())
                assert nonzero > 0
                assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                assert int(model.bbox_head.loss_cls._local_iter) == (20000 if mature else 0) + 5
                assert model.bbox_head.assigner._local_call_count == count + 2
                rows.append(dict(regression=reg, objective=mode, mature=mature, loss=float(total.detach()),
                                 stats=stats, other=other, nonzero_gradient_tensors=nonzero,
                                 assignment_sha256=list(assignments),logit_gradient_sha256=logit_grad_sha,
                                 initial_parameters_sha256=now))
                model.zero_grad(set_to_none=True)
                del result, main, total
            hook.remove();del model,outputs,logit_grad
            gc.collect(); torch.cuda.empty_cache()
        # Native model comparisons use identical initial tensors and inputs.
        for mode in ('U','G'):
            cfg = Config.fromfile(str(ROOT / ('crane_project/configs/crane_symeood_k1_ablation_%s_v1.py' % oldgroup)))
            cfg.model.bbox_head.loss_cls.use_geometry_weight = mode == 'G'
            native = build_detector(copy.deepcopy(cfg.model)).cuda()
            native.load_state_dict(initial, strict=True)
            native.train()
            outputs=[];assignments=[]
            hook=native.bbox_head.register_forward_hook(lambda m,i,o:outputs.extend(o[0]))
            original_assign=native.bbox_head.assigner.assign
            native.bbox_head.assigner.assign=observed_assign
            for mod in native.modules():
                if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm): mod.eval()
            for mature in (False,True):
                native.load_state_dict(initial)
                native.bbox_head.assigner._local_call_count = 8000 if mature else 0
                native.bbox_head.loss_cls._local_iter.fill_(20000 if mature else 0)
                outputs.clear();assignments.clear()
                result = native.forward_train(**data)
                expected = next(r for r in rows if r['regression']==reg and r['objective']==mode and r['mature']==mature)
                assert float(sum(result['loss_cls']).detach()) == expected['stats'][mode], 'Native loss differs'
                grads=torch.autograd.grad(sum(result['loss_cls']),outputs)
                assert tensor_digest([(str(i),g) for i,g in enumerate(grads)])==expected['logit_gradient_sha256']
                assert assignments==expected['assignment_sha256']
                for k,v in result.items():
                    if k != 'loss_cls':assert float(sum(v).detach()) == expected['other'][k], (reg,mode,k)
                del result
            hook.remove();del native,outputs,grads
            gc.collect(); torch.cuda.empty_cache()
    # Regression/auxiliary paths are identical within each regression background.
    for reg in ('iou','kld'):
        for mature in (False,True):
            group=[r for r in rows if r['regression']==reg and r['mature']==mature]
            assert all(r['other']==group[0]['other'] for r in group)
            assert all(r['assignment_sha256']==group[0]['assignment_sha256'] for r in group)
    (out/'engineering.json').write_text(json.dumps(dict(passed=True, cases=rows, physical_gpu=2,
        scope='real cached TRAIN batch2, native loss equality, common init, finite backward; updates discarded'),indent=2))
    print('ENGINEERING_PASS', len(rows), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
