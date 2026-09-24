# K1 初始化的轻量蒸馏配对实验 V2

## 已有证据与本轮目的

已暴露的固定 TEST 配对审计 `k1_dino_student_paired_fixed_test_audit_v1` 表明：从头训练的语义特征蒸馏学生在 real 域从 K1 的 277 帧输出降至 222 帧；`real_seq03` 失去 60 帧 K1 原有输出，`real_seq02[2,41]` 与 `[137,169]` 两段均未恢复。共同输出帧在 `seq02` 有一定 RIoU 收益，但不能抵消缺测代价。审计 JSON 的本地 SHA256 为 `5ef70baf2897f1a79fc62cfbaa4b13ede95b6afab12435d484174d6be785f6f3`。这些 TEST 结果仅解释失败，不参与 V2 的训练样本、阈值和 checkpoint 选择。

V2 只回答一个问题：**从 source VAL 已选中的普通 K1 初始化后，原有的 DINO 前景特征损失是否相对同预算普通微调产生增益？** V2 不是新的 DINO 检测头候选蒸馏，也没有解决教师候选的 source 支持不足问题。学生推理仍只运行 SymEOOD；训练时从磁盘读取已有冻结 DINO 缓存，不运行 DINO 大主干。

## 两组唯一的计划差异

| 项目 | 无 DINO 损失对照 | DINO 特征蒸馏 |
|---|---|---|
| 配置 | `crane_symeood_k1_dino_warmstart_control_v2.py` | `crane_symeood_k1_dino_warmstart_distill_v2.py` |
| 初始化 | source VAL 选中的 `crane_symeood_k1/epoch_20.pth` | 同一文件 |
| 学生结构 | K1 + 零初始化分类残差适配器 | 完全相同 |
| source 数据 | train + train_sim | 完全相同 |
| DINO 缓存数据流 | 读取，用于匹配数据和运行条件 | 读取，供蒸馏损失使用 |
| 训练 | 4 epoch；SGD，学习率 0.00025；每卡 batch 2；2 GPU 顺序运行 | 完全相同 |
| 唯一有意差异 | 无 DINO 损失 | 原 V1 前景特征蒸馏损失，权重 0.05 |

先运行 CPU preflight：要求 K1 checkpoint SHA256 与新版 source VAL 选权结果、已生成的 K1 TEST 身份报告一致；检查 train 与 train_sim 两份缓存收据；解析两组配置；构建两组 CPU 模型并加载同一 K1 权重，确认新增分类适配器为零、初始分类和回归输出完全相同。若任何检查失败，停止训练，不回退到旧版选权结果。

4 epoch、低学习率是预先固定的微调预算。两组 checkpoint 只在 source VAL 以同一 `ckpt_sweep.py` 规则选择；固定 TEST 每组只运行一次。对照的含义是区分 DINO 损失与普通继续训练的影响；与原 24 epoch 从头训练 K1 的比较则仍须单独标明训练预算不同。不得依据已暴露 TEST 的特定帧挑选样本或调节损失。

## 服务器顺序

1. 同步本文件、三个 V2 config、preflight 工具和专项测试。保留原 V1 权重与报告。
2. 运行 `tests/test_k1_dino_warmstart_v2.py` 和 `tests/test_semantic_feature_distill_v1.py`。
3. CPU preflight 成功并输出 `MATCHED_WARMSTART_READY` 后，先训练 control，再训练 distill。仅使用 GPU 0、1，不并发训练。
4. 用统一学生推理配置 `crane_symeood_k1_dino_semantic_student_v1.py` 对两组 epoch 1–4 做 source VAL 选权；核对两个 `sweep_results.json` 的协议、候选和选择。
5. 各自按选权结果运行一次固定 TEST，再生成配对审计。正式评价同时报告全帧中心命中、输出覆盖率、共同输出帧几何、序列连续缺测，以及推理时间/显存；不把 TEST 当作优化输入。

本地没有服务器 K1 epoch 20 权重或 MMCV/MMRotate 运行环境，因此 CPU 模型构建与真实权重加载必须在服务器 preflight 中完成。本地验证只覆盖契约逻辑、配置文本、语法和已有蒸馏模块测试。


本轮 warmstart V2 的归档结果、配置身份纠正与后续迁移方向见 [蒸馏试验交接记录](dino_distillation_handoff_20260923.md)。
