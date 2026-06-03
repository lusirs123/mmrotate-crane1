#!/usr/bin/env python
import argparse
from collections import OrderedDict

import torch

#读取多个 .pth checkpoint
#对 state_dict 里的浮点权重做算术平均
#保存成新的 averaged checkpoint
#指标为 CraneDataset.evaluate() 的在线指标
"""
确定使用avg_20_22_24，不需要再去
阶段 1：比较 avg 和单个 checkpoint
我们要比较：
epoch_20.pth
epoch_22.pth
epoch_24.pth
avg_20_22_24.pth
看平均权重有没有明显变差。
可以先用在线指标比较，也就是你刚才跑的那种：
sim_A_RMSE
sim_R_center
real_R_center
Weighted_R_center
这一步叫：
sanity check,合理性检查 / 健康性检查
中文可以叫：
目的不是最终论文报告，而是确认平均权重没有崩。

阶段 2：跑完整 test 离线指标
如果 avg 权重没问题，再对最终权重导出预测结果，然后跑：
python crane_project/tools/eval_crane_offline.py
这个才是最终论文里更完整的 test 指标。
"""



def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    return checkpoint, state_dict


def average_state_dicts(state_dicts):
    ref_keys = list(state_dicts[0].keys())
    for i, state_dict in enumerate(state_dicts[1:], start=1):
        if list(state_dict.keys()) != ref_keys:
            missing = set(ref_keys) - set(state_dict.keys())
            extra = set(state_dict.keys()) - set(ref_keys)
            raise RuntimeError(
                f'Checkpoint {i} has mismatched state_dict keys. '
                f'Missing: {sorted(missing)[:10]}, extra: {sorted(extra)[:10]}')

    averaged = OrderedDict()
    for key in ref_keys:
        values = [state_dict[key] for state_dict in state_dicts]
        first = values[0]
        if torch.is_tensor(first) and torch.is_floating_point(first):
            avg = first.clone().float()
            for value in values[1:]:
                avg.add_(value.float())
            avg.div_(len(values))
            averaged[key] = avg.to(dtype=first.dtype)
        else:
            averaged[key] = first.clone() if torch.is_tensor(first) else first
    return averaged


def main():
    parser = argparse.ArgumentParser(
        description='Average floating-point parameters in MMRotate/MMCV checkpoints.')
    parser.add_argument('checkpoints', nargs='+', help='Input checkpoint paths')
    parser.add_argument('-o', '--output', required=True, help='Output checkpoint path')
    args = parser.parse_args()

    if len(args.checkpoints) < 2:
        raise RuntimeError('At least two checkpoints are required for averaging.')

    checkpoints = []
    state_dicts = []
    for path in args.checkpoints:
        checkpoint, state_dict = load_checkpoint(path)
        checkpoints.append(checkpoint)
        state_dicts.append(state_dict)

    output = checkpoints[-1].copy() if isinstance(checkpoints[-1], dict) else {}
    output['state_dict'] = average_state_dicts(state_dicts)
    output.pop('optimizer', None)
    output.setdefault('meta', {})
    if isinstance(output['meta'], dict):
        output['meta'] = output['meta'].copy()
        output['meta']['averaged_from'] = args.checkpoints

    torch.save(output, args.output)
    print(f'Saved averaged checkpoint to {args.output}')


if __name__ == '__main__':
    main()
