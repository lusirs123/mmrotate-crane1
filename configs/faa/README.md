# FAA 抓斗外部对比配置

当前双卡正式入口是 [crane_faa_oriented_rcnn_r50_fpn_24e_port_day2night_2gpu_v1.py](crane_faa_oriented_rcnn_r50_fpn_24e_port_day2night_2gpu_v1.py)，继承先前已验证的[单卡适配配置](crane_faa_oriented_rcnn_r50_fpn_24e_port_day2night_v1.py)。所有配置继承都位于根目录 `configs/`，不从 `work_dirs` 读取配置。训练继续复用仓库原有 `tools/train.py`。

采用官方 **Oriented R-CNN R50＋完整 FAA**，同时包含 `FAAFusionFPN` 与 `FAAHead`，不是仅 FAA Head，也不是给 EOOD 接入 FAA。官方来源：[Fourier Angle Alignment](https://github.com/gcy0423/Fourier-Angle-Alignment/tree/d3965484310fbb20347bbdccd07da19c0d8e52b8)，固定提交 `d3965484310fbb20347bbdccd07da19c0d8e52b8`。

## 文件与继承

- `crane_faa_oriented_rcnn_r50_fpn_24e_port_day2night_v1.py`：本项目单类数据与训练预算适配。
- `crane_faa_oriented_rcnn_r50_fpn_24e_port_day2night_2gpu_v1.py`：双卡正式配置，只增加设备、日志与显存监控 Hook，模型、数据、AdamW、累积次数和日程继承已验证适配。
- `oriented_rcnn_r50_fpn_1x_dota_le90_faa.py`：官方 R50/FAA 配置；只调整本地继承路径，增加显式模块注册。DOTA 默认路径与类别规则由抓斗入口完整覆盖。
- `_base_/dotav1.py`、`_base_/schedule_1x.py`：官方原样副本；官方此提交的 `1x` 实际为 **16 轮**，不能按文件名写成 12 轮。
- `../_base_/default_runtime.py`：现有公共运行配置，字节与本次官方版本相同。
- `../../mmrotate/models/necks/faafusion.py`、`../../mmrotate/models/roi_heads/bbox_heads/faa_head.py`：两项核心实现原样复制，通过 `custom_imports` 注册，不改全局 `__init__.py`，也不改已有 EOOD/R4/M/L。
- `upstream_manifest.json`：官方提交、文件 SHA、改动说明、VAL 候选与选权入口。
- `LICENSE.faa`：保留官方 CC BY-NC 4.0 许可；引用原论文 Gu et al., *Fourier Angle Alignment for Oriented Object Detection in Remote Sensing*, CVPR 2026。

## 适配内容

| 项目 | 当前设置与依据 |
| --- | --- |
| 模型 | 保留官方 R50、FAA 融合与头、RPN/RoI、损失、分配器、采样、原生 NMS；仅输出类别 15→1 |
| 数据 | `crane_project/data/crane_grab_port_day2night_v1/`；Real TRAIN 1810＋Sim TRAIN 748、VAL 887、TEST 1440；单类 `grab`，`le90` |
| 输入与增强 | 与尺度增强 EOOD 对照相同的 1024、缩小增强、三向翻转、Normalize、固定 Pad；覆盖官方 DOTA 专用 `PolyRandomRotate/rect_classes=[9,11]` |
| 初始化 | `torchvision://resnet50` ImageNet 骨干；无 DOTA 检测器权重、无旧实验续训；实际初始化权重 SHA 在训练前核对 |
| 优化器 | 保留官方 AdamW，LR `1e-4`、betas `(0.9,0.999)`、weight decay `0.05`、梯度裁剪 35；没有套用 EOOD 的 SGD LR |
| 批量 | 当前正式运行使用双 GPU、每卡每次 1 张、累积 2 次，有效 batch=4；初版单卡有效 batch=2 的适配文件保留 |
| 日程 | 共同预算 24 轮、衰减 `[16,22]`；官方原设置为 16 轮、`[8,11]`。warmup 1000 个微步，对应 500 次参数更新，保留官方 warmup ratio `1/3` |
| 保存与选权 | 每 2 轮保存并保留全部 12 份；VAL 候选固定 `16/18/20/22/24`，复用 `ckpt_sweep.py` 的任务约束选权；不以训练日志 best-mAP 自动替代 |
| 推理输出 | 保留原生多框结果；共同评价时按最高得分取 Top1，与 EOOD/R4 相同。Top1 AP50/AP75 与完整输出 AP 分开标识，不连接 Midpoint/LVSR |

24 轮是预先声明的共同训练预算，不是已经证实 FAA 收敛。后续先在 TRAIN/VAL 上验证显存、损失和收敛；如需改批量或日程，在正式运行前固定并记录，不能根据 TEST 修改。

## 已做检查与下一步

本地无需 Torch 的检查入口：

```bash
python3 crane_project/tools/check_faa_adaptation_v1.py
```

有现有 MMRotate 环境时，可加 `--runtime` 做 CPU 检查：配置由 MMCV 实际展开、完整模型构建、两个 FAA 模块小张量前向与反向、TRAIN Real/Sim 与 VAL 各一张数据管线、累积优化器 Hook 构建。不会初始化/下载骨干权重，不创建优化器执行更新，不运行 TEST 或占用 GPU。`--out` 仅接受新收据路径。

检查记录见 [FAA 适配与验证](../../docs/detection/09_外部模型对比与效率核查_20261101.md#faa-adaptation-record)。随后已完成服务器 GPU 2/3 的 1024 输入 DDP 短试跑：每卡两次微批次、一次分布式临时 AdamW 更新及 TRAIN 图像原生推理通过，进程采样峰值分别为 5556/5650 MiB；已退出并释放显存，未启动正式训练或 TEST。

随后用户已授权双卡正式训练。2026-11-01 17:27（台北时间）使用 GPU 2/3 启动，共 24 轮、每卡 1 张、累积 2 次，有效 batch=4；AdamW `1e-4`、warmup 1000 微步、衰减 `[16,22]`。初期已连续运行至少 170 个微批次/85 次更新，两卡整卡采样峰值均为 5758 MiB，最低剩余 2350 MiB。

资源模块 `crane_project.utils.faa_training_memory_v1` 只调用原有 `tools/train.py` 并观察显存，不实现另一套训练循环。每卡 allocator 上限 6144 MiB，进程保护线 6656 MiB、最低空闲 1536 MiB，越线停止，不自动重启或运行 TEST。实际训练身份与官方代码的核验见统一记录第 5.6 节。

`upstream_manifest.json` 保留初次本地适配的来源快照；当前正式资源设置及完整服务器执行源码 SHA 以正式运行 `protocol.json` 为准。长期资源与收敛仍需后续日志确认。
