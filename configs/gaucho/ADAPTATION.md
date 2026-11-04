# GauCho-FCOS R50 抓斗对比适配

2026-11-04，仅本地代码准备，尚未训练或服务器/GPU试跑。

## 官方身份与接入目录

论文为CVPR 2025 GauCho，作者仓库 https://github.com/jhlmarques/GauCho ，固定提交 `7d7ddd02cc587205c1fab44cc8ca23501042ca2b`。

按FAA相同方式，所需官方模块直接接入现有 `mmrotate/`，复用根目录 `tools/train.py`。正式配置及完整继承链位于根目录 `configs/gaucho/`，重复GauCho框架副本已经删除。无需再安装另一套同名MMRotate。

| 文件 | 官方实现与适配 |
| --- | --- |
| `mmrotate/models/dense_heads/gaussian_fcos_head.py` | 官方GaussianFCOSHead，只调整父类导入路径 |
| `mmrotate/models/dense_heads/gaucho_rotated_fcos_head.py` | 官方RotatedFCOSHead父类，注册名为GauChoRotatedFCOSHead |
| `mmrotate/models/losses/gaucho_gaussian_dist_loss_v1.py` | 官方高斯预测/ProbIoU损失，注册名为GauChoGDLossV1 |

本项目已有FCOS父类与作者版本不同，已有GDLoss_v1也不含完整GauCho损失路径，所以三份依赖一起接入。现有父类和损失文件不覆盖，训练参数键不增加包装层。三个官方模块只做导入/注册名适配，不改计算公式、分配、解码或NMS。无需修改全局 `__init__.py`，通过配置 `custom_imports` 加载。

`upstream_manifest.json` 保存8份官方模块/配置/许可证的来源和SHA。检查器反向还原三个模块的导入与注册名后核对官方SHA；另外9份共享依赖已与作者版本核对Python AST并固定本地SHA。原有NMS文件与官方仅有注释差别，AST一致。许可证保存在 `LICENSE.gaucho`。

## 数据、预算与初始化

正式入口为 `crane_gaucho_fcos_r50_probiou_port_day2night_v1.py`。继承作者完整FCOS R50/FPN＋GaussianFCOSHead＋GauCho编码＋ProbIoU配置，只改类别15→1并使用隔离的损失注册别名。模型计算不变。

数据与增强直接复用现有CraneDataset/PortIsotropicShrink，和FAA的1024管线逐项相同。Real TRAIN 1810＋Sim TRAIN 748、VAL 887、TEST 1440，原数据difficulty=0/1目标都保留。

保留官方SGD lr=0.0025、momentum=0.9、weight_decay=0.0001及梯度裁剪。当前准备双卡每卡1张、全局batch=2，无额外累积。官方12轮初始扩展为24轮、step16/22、每2轮保存；这只是待验证预算，正式训练前仍需TRAIN/VAL检查并冻结。ImageNet ResNet50沿用torchvision://resnet50，真实初始化SHA待服务器核对。

VAL候选16/18/20/22/24及既有共同任务选权规则见 `experiment.json`。在线VAL只监控，不自动TEST，不接midpoint/LVSR。

## 检查与原生入口

默认检查源码、配置、模型差异、共享依赖和数据清单，不训练。

```bash
python crane_project/tools/prepare_external_obb_v1.py
```

完整CPU运行检查工具已经准备，但本机缺少MMCV编译运行环境，尚未执行通过。该检查不下载权重、不调用init_weights/优化器/backward、不读TEST图像。

```bash
python crane_project/tools/check_gaucho_runtime_v1.py --out /tmp/gaucho_cpu_check.json
```

以后授权训练并通过服务器环境/显存检查后，直接复用当前原生入口，下面只是调用形式，本次未执行。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
  python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/train.py \
  configs/gaucho/crane_gaucho_fcos_r50_probiou_port_day2night_v1.py \
  --launcher pytorch --seed 0 --deterministic
```

没有另写训练循环或服务器启动脚本。服务器应检查当前项目包路径、官方父类/损失注册、Torch/CUDA、DDP和显存稳定性，不能把本地静态检查当作这些已经通过。

## 共同评价

`export_external_obb_v1.py --model gaucho` 默认打印计划。显式推理时调用现有官方MMRotate init_detector/inference_detector，严格加载权重，并核查GauCho官方父类/损失路径。保留完整原图K×6单类OBB、帧顺序、SHA和冻结状态。

`evaluate_external_obb_saved_v1.py` 复用FAA共同指标；`select_external_obb_val_v1.py` 复用原离线选权函数。锁定VAL候选和所选权重后，才解锁TEST。真实训练权重、VAL/TEST结果和GPU运行尚待后续授权。
