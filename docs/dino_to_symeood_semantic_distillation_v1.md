# DINO 到 SymEOOD 的几何保护语义蒸馏 V1

## 目的

本入口验证一个有限问题：冻结 DINOv2 已表现出的语义候选能力，能否在不保留
DINO 推理链的前提下，改善普通 SymEOOD K1 的分类排序。它不是 Base V3
时序 refiner 的替代实验，也不使用固定 TEST 选择权重或超参数。

## 实现边界

- 教师输入来自既有 DINOv2 ViT-L/14 离线特征缓存；训练过程不运行 DINO。
- 只读取 source TRAIN（`train:train` 与 `train_sim:train`），缓存必须通过文件身份、
  模型名、通道数、有限值和完整性预检。
- 教师特征固定为 FP16 `64x64`，只蒸馏 FPN 第 0 层对应的分类塔特征，最多
  使用 4096 个空间 token。
- 蒸馏区域由训练 GT 的 OBB 外接矩形给出；教师张量始终 detach。
- 原版 head 没有多层分类塔，因此增加零初始化的 `1x1` 分类残差适配器；蒸馏
  支路在适配器输入前 detach。蒸馏梯度只能更新分类适配器与训练期投影器，不能
  更新 backbone、FPN 或回归分支。原有 SymKLD、角度定义和 OBB 目标不变。
- 推理前导出学生 checkpoint，移除 `semantic_distillation.*`，保留轻量分类
  适配器。导出后的权重由 `crane_symeood_k1_dino_semantic_student_v1.py`
  加载，推理不实例化 DINO 或蒸馏投影器。

## 公平比较

1. GT-only：普通 `crane_symeood_k1.py`，相同 source 数据、seed 和训练轮次。
2. 蒸馏 V1：只增加前景分类塔语义蒸馏。
3. 两者分别在 source VAL 选择 checkpoint；固定 TEST 只做一次冻结评估。

必须同时报告完整 OBB、中心、尺度、角度、小目标切片、推理时间和显存。语义
排序改善不能替代尺度与角度结果。若 source VAL 无改善或几何指标退化，停止该
路线，不根据 TEST 调整 `loss_weight`。

## 当前状态

代码入口和本地单元测试已完成；尚未在服务器完成缓存完整性检查、训练、学生导出
和评估，因此尚不能声称性能提升或形成论文贡献。
