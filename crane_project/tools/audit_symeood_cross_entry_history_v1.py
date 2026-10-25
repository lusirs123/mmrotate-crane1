"""Read existing TRAIN logs/checkpoint metadata; no model execution."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main(out):
    import torch
    from crane_project.utils.symeood_usg_v1 import tensor_digest
    rows=[]
    for old,reg,obj in [('100','iou','u'),('101','iou','g'),('110','kld','u'),('111','kld','g')]:
        roots=[ROOT/'work_dirs/symeood_ablation_v1'/old,ROOT/'work_dirs/symeood_usg_v1'/('s0_'+reg+'_'+obj)]
        records=[];initial=torch.load(out/'identity'/('initial_'+old+'.pth'),map_location='cpu')['state_dict']
        frozen=[k for k in initial if k.startswith(('backbone.conv1.','backbone.bn1.','backbone.layer1.'))]
        for path in roots:
            logs=list(path.glob('*.log.json'));assert len(logs)==1
            all_rows=[json.loads(r) for r in logs[0].read_text().splitlines()]
            train=[r for r in all_rows if r.get('mode')=='train']
            checkpoint=torch.load(path/'epoch_24.pth',map_location='cpu')
            state=checkpoint['state_dict'];meta=checkpoint['meta']
            text=logs[0].with_suffix('').read_text()
            flags=[line for line in text.splitlines() if any(t in line.lower() for t in ('seed to','deterministic:','resume from','load checkpoint from torchvision','distributed training:'))]
            records.append(dict(root=str(path),log_sha256=hashlib.sha256(logs[0].read_bytes()).hexdigest(),
                env_info=all_rows[0].get('env_info'),seed=all_rows[0].get('seed'),log_flags=flags,
                lr_trace=[(r['epoch'],r['iter'],r['lr']) for r in train],train_logged_rows=len(train),first=train[0],last=train[-1],
                checkpoint_epoch=meta.get('epoch'),checkpoint_iteration=meta.get('iter'),
                nfl_calls=int(state['bbox_head.loss_cls._local_iter']),
                frozen_stem_and_layer1_equal_current_initial=all(torch.equal(state[k],initial[k]) for k in frozen),
                frozen_keys=frozen,frozen_values_sha256=tensor_digest((k,state[k]) for k in frozen),
                assigner_counter_in_state=any('local_call_count' in k for k in state),optimizer=checkpoint['optimizer']['param_groups']))
            del checkpoint,state
        a,b=records
        rows.append(dict(old=old,new=reg+'-'+obj.upper(),records=records,
                         env_info_equal=a['env_info']==b['env_info'],lr_trace_equal=a['lr_trace']==b['lr_trace'],
                         frozen_pretrained_portion_equal=a['frozen_values_sha256']==b['frozen_values_sha256'],
                         iterations_equal=a['checkpoint_iteration']==b['checkpoint_iteration']))
    (out/'historical_records.json').write_text(json.dumps(dict(pairs=rows,
        limitation='Frozen backbone portion verifies those tensors only; historical complete initialization, per-batch augmented inputs and pretrained whole-file SHA were not saved.'),indent=2))
    for r in rows:print(r['old'],r['env_info_equal'],r['lr_trace_equal'],r['frozen_pretrained_portion_equal'],r['iterations_equal'])


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
