# YOLO26m-OBB 抓斗对比适配

2026-11-06最新状态：用户已重新授权启动，08:53:07（台北时间）在GPU0/1启动 `work_dirs/yolo26_formal_20261106_v1/attempt_005/`，supervisor PID2997753，从头执行24轮。使用恢复后的最初资源方案6144/6656/1536MiB和 `max_split_size_mb:128`；两rank已完成至少50次更新，损失有限，初期每卡进程采样稳定在5306MiB，未触发保护。第10—54批实测约2.025秒/双卡迭代，按初期速度线性外推纯训练17.27小时，含在线验证/保存先保守预留18—22小时；这是启动估算，原生预热累积变化后可能缩短，不是完成收据。训练之外的共同VAL选权、精度TEST和效率测试尚未执行。具体证据与边界见09号文件第13.5节。下文停机决定为历史状态，已被本次新授权替代。

2026-11-06当前状态：按用户最新决定，正式训练和资源短验证均已停止，不自动重启。已在本机与服务器恢复最初观察稳定的 `configs/yolo26/resource_policy_v1.json`：allocator预算6144MiB、进程保护6656MiB、最低剩余1536MiB，`max_split_size_mb:128`，不主动回收缓存；此前初期进程占用约5306MiB/卡。输入1024、FP32、双卡各1张、nbs4、官方MuSGD、初始化与数据不变。`work_dirs/yolo26_formal_20261106_v1/attempt_004/` 和资源诊断 `work_dirs/yolo26_memory_fix_20261106_v1/probe_001/` 均为已停止的历史尝试，临时参数不用于正式实验。后续重新授权训练后仍需共同VAL选权、一次TEST精度和统一效率测试，尚无YOLO最终成绩。当前决定以09号文件第13.4节为准。以下早期准备/排队记录保留沿革。

2026-11-04，本地代码与数据适配完成，**尚未训练、下载真实预训练权重或运行服务器/GPU任务**。

2026-11-05状态更新：已核验官方ImageNet分类权重和派生骨干初始化，完成服务器独立venv、数据转换、CPU检查及物理GPU0/1双卡有界试跑。每rank两个1024批次、两次官方MuSGD临时更新，损失/梯度有限，驱动进程采样峰值两卡均4496MiB，结束后显存释放，无checkpoint/VAL/TEST推理，未启动正式训练。实际权重SHA、完整收据和边界见 `docs/detection/09_外部模型对比与效率核查_20261101.md` 第11.4节。下文初始准备时尚未完成的事项保留沿革，以本段和第11.4节实际收据为准。

## 官方身份与配置

[官方仓库](https://github.com/ultralytics/ultralytics)固定提交 `d1ebfd72646d5a5b9050d1d0557d7b7fb0ab31f4`，版本8.4.202。官方运行包、安装元数据与AGPL许可证保存在 `third_party/ultralytics/`，370份官方文件/配置逐字节保留，SHA见 `upstream_manifest.json`。

| 文件 | 用途 |
| --- | --- |
| `yolo26m-obb.yaml` | 完整模型配置，来自官方26-OBB，仅单类nc=1与medium scale显式化 |
| `official/yolo26-obb.yaml` | 原样官方模型配置 |
| `official/default.yaml` | 本次提交的原样官方默认超参数 |
| `crane_grab_port_day2night_v1.yaml` | TRAIN/VAL数据入口，路径相对本YAML目录，可直接迁移同一仓库目录树 |
| `crane_yolo26m_obb_train_v1.yaml` | 可直接交给官方 `yolo obb train cfg=...` 的训练配置 |
| `crane_yolo26m_obb_port_day2night_v1.json` | 初始化、训练预算、VAL候选、推理及证据边界契约 |

保留官方OBB26双头结构、损失、分配、训练器和预测器。此版本通过 `nms=False` 选择原生one-to-one/NMS-free推理支路，不能仅看到模型YAML的 `end2end: True` 就假定实际推理已选中该支路。该选择已纳入训练监控/正式导出配置。

## 数据转换

```bash
python crane_project/tools/prepare_external_obb_v1.py --action convert-yolo
python crane_project/tools/prepare_external_obb_v1.py --action check-yolo
```

本机已执行转换与核对。Real TRAIN 1810＋Sim TRAIN 748合并为2558帧TRAIN，VAL为887帧，输出位于 `crane_project/data/crane_grab_port_day2night_yolo26_v1/`。图像为指向原数据的相对符号链接，像素不变。需要实体副本的环境可在首次转换时使用 `--copy-images`。转换不会覆盖既有输出。

每条DOTA标注的8个原始四角坐标分别除以实际宽、高，生成官方YOLO OBB的 `class x1 y1 ... x4 y4`。不裁图、不裁坐标、不另拟矩形。原CraneDataset包含difficulty=0和1目标，转换同样全部保留。每份图像/标注核对冻结清单SHA，记录转换标签SHA，并检查反向坐标恢复。

默认不转换TEST，训练YAML也没有TEST入口。后续固定TEST推理可直接读原始图像，再交给同一封存GT评价，不需要把TEST加入训练数据配置。

## ImageNet初始化适配

为了与EOOD/FAA的ImageNet监督来源保持可解释的比较，本准备方案不使用DOTA/COCO检测器预训练。官方 `yolo26m-cls.pt` 的ImageNet分类骨干与OBB骨干的前9块（0..8）一致。分类第9块是C2PSA，OBB第9块是SPPF；其中部分BN键虽同名同形，也不应直接迁移。

`prepare_yolo26_initialization_v1.py` 因此只保留0..8块，生成**派生初始化副本** `pretrained/yolo26m_imagenet_backbone_v1.pt`。这不是新的检测架构，不改变官方OBB模型或加载算法。原分类权重保持原样，训练仍由官方trainer的 `pretrained=本地初始化副本` 执行名称/形状匹配。

本机随机权重CPU检查已验证246项状态、8,704,320个参数可迁移，约占23,472,100个注册参数的37.08%，OBB特有层保持原初始化；原生保存/加载和trainer模型重建保持这些值。**这只是迁移机制验证，不是真实ImageNet权重验证或性能结果。** 实际分类权重的来源、SHA、派生副本SHA尚待后续准备，JSON中相关SHA为空，不能声称初始化身份已经冻结。

未来在官方来源下载并核对权重后，以下工具才生成派生文件。本次没有执行这一步。

```bash
python crane_project/tools/prepare_yolo26_initialization_v1.py \
  --source-sha256 <官方分类权重的已核对SHA256>
```

比较表应披露ImageNet迁移来源、骨干不同及迁移覆盖，不称为完全相同初始化。也不把该方案写成“原论文DOTA预训练配方原样复现”。

## 训练配方与保存规则

2026-11-05按用户要求统一为24轮，batch=2、nbs=4、双卡每卡1张、FP32、seed=0、关闭提前终止。此前100轮是官方默认准备预算，没有正式执行，保留在短试跑历史收据中。保留固定官方版本的optimizer=auto和原生增强，因此与FAA是**各方法合理训练配方下的系统比较**，不能写成同增强、同轮数或同更新预算的控制消融。实际优化器由官方auto规则决定，需要正式日志记录。warmup期间累积动态变化，不能写每次更新都严格batch=4。

服务器运行前先验证依赖与GTX1080支持、初始化、显存、损失及TRAIN/VAL收敛。准备配置不代表这些已通过。建议独立环境，不升级/覆盖现有MMRotate环境；所选Torch wheel必须实际支持GTX1080的Pascal架构，不能直接假设最新CUDA/Torch wheel可用。

每轮保存原生权重，离线候选为完成16/18/20/22/24轮。**此提交的周期权重文件按零起点命名**，对应 `epoch15.pt/epoch17.pt/epoch19.pt/epoch21.pt/epoch23.pt`，不能将文件名直接当完成轮数。候选集合可在正式训练前根据TRAIN/VAL方案冻结，但不得根据TEST改变。原生best.pt只作在线监控产物，不自动作为共同任务选权结果。

以后通过检查并获得训练授权后，使用安装了该固定官方源码的原生CLI即可，没有另写训练器或训练循环。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
  yolo obb train cfg=configs/yolo26/crane_yolo26m_obb_train_v1.yaml
```

## 已做检查与评价接入

CPU运行检查工具为 `crane_project/tools/check_yolo26_runtime_v1.py`，已验证官方OBB26构建、128×128合成输入前向（原生one-to-one输出1×300×7）、TRAIN/VAL各一帧的1024原生图像与标签读入、骨干迁移和初始化保存/加载，优化器更新0、backward 0、TEST图像读取0。CPU依赖只安装在临时目录，没有改动项目既有环境。

`export_external_obb_v1.py --model yolo26` 默认只打印计划。显式推理时调用官方 `YOLO.predict`，读取已经恢复到原图像素的 `obb.xywhr/conf/cls`，只规范等价长短边与角度表示，保留全部原生框，按score排序供共同Top1评价。不会再次缩放坐标，也不把归一化角点当OBB角度。无输出帧保留为空。

`evaluate_external_obb_saved_v1.py` 复用FAA共同评价。`select_external_obb_val_v1.py` 复用现有任务选权规则，锁定配置/权重/VAL选择收据后才解锁TEST。当前没有真实训练权重，实际带权重VAL/TEST、GPU显存和效率尚未验证。

2026-11-05正式启动更新：用户确认两模型共同24轮预算后，已在物理GPU0/1启动顺序队列，先GauCho后YOLO。此刻GauCho已实际更新，YOLO排队；没有启动离线选权或TEST。冻结协议及启动记录见09号文件第11.5节，下文历史准备/短试跑状态不代表当前训练状态。
