# OBB 观测可靠性与连续输出

本文是小论文「OBB 观测可靠性与连续输出」主线的唯一维护入口。原始材料（四份方向交接、首轮审计、开发计划与整体收尾记录）见文末[来源索引](#appendix-sources)。

**阅读约定**

- 当前状态以包含真实在线指标 V3 的收尾记录为依据；早期记录的日期和协议角色在对应位置说明一次，不再逐节重复免责声明。
- 历史材料中的「当前」「最终」「唯一下一步」只在原阶段有效，不能覆盖本文第 1 节的结论。
- 本文档只整理已有记录，未重新核验权重、预测 JSON、CSV 或 SHA256。

---

## 1. 研究问题与当前结论

<a id="sec-1"></a>

<a id="sec-1-1"></a>

### 1.1 研究链路与范围

```text
复杂港口图像序列
  -> SymEOOD / Base V3 冻结旋转检测
  -> OBB 中心、尺度、周期方向的分量可靠性判定
  -> 带 measurement / prediction / unavailable 状态的连续视觉观测
```

- 研究对象为抓斗**顶梁** OBB；纯 RGB 单目，不使用 PLC、编码器；二维图像方向不等于机械自旋角或物理摆角。
- 已选择「OBB 观测可靠性与连续输出」作为新增研究方向；中心、尺度、图像方向分别评价可靠性。该新增贡献尚待实验成立。
- 相机共模补偿不再是必做模块。仿真—真实混合训练已经存在，不因数据混合本身就宣称新贡献。
- Raw-opt 米制深度、相机—吊点外参、物理摆角、EKF 属于大论文后半部分，不阻塞小论文。
- 大论文建议层次：检测 → 可靠二维观测 → 单目深度与物理状态估计；小论文主要覆盖前两层。

<a id="sec-1-2"></a>

### 1.2 要证明什么

<a id="sec-1-2-1"></a>

#### 1.2.1 输入输出定义

原图输入 `B_t = (u_t, v_t, w_t, h_t, gamma_t, s_t)`。候选输出为中心、尺度、方向三个可靠性量 `r_center`、`r_scale`、`r_angle` 及对应有效标志。用 `r` 表示可靠性，避免与深度公式中的 `q` 冲突。

核心问题：**在相同保留覆盖率下，能否比单纯检测置信度筛选更有效地减少中心、长短边、周期方向误差？**

第一阶段保留原始预测框，不做候选接管、不改检测几何。可靠性只用于接受/拒绝观测；角度等价表示规范化属于必要实现，不单独包装为创新。近正方形方向退化、边界截断、尺度跳变、时序残差是候选线索，不是已经验证的质量规则。

<a id="sec-1-2-2"></a>

#### 1.2.2 数据与推理约束

- GT 只用于离线标签和评价。检测器在训练图像上可能过于准确，因此先审计错误样本支持，再决定采用独立校准集还是 OOF 预测；不能直接利用已暴露 target 逐帧设计规则。
- 时序推理只用当前及过去帧，按完整序列/时序块隔离，必要时清除边界历史。
- 连续输出区分 `measurement` / `prediction` / `unavailable`；先完成逐帧可靠性，再增加有界连续观测管理。
- 不要求第一版使用 Kalman/EKF；不以复制历史框伪造检出。未补偿相机运动时，图像位移包含相机运动，不能称为物理摆动。

<a id="sec-1-3"></a>

### 1.3 当前结论

- **前端身份**：Base V3 epoch9，正式 K1 有框时作为锚框、K1 缺失时使用冻结 DINO 候选；refiner 输出的常数分数 `1.0` **不是置信度**。早期审计中的 V4+seq11-v2 epoch10 只是当时待绑定的候选；普通 K1 source-val 统计只是独立历史基线。
- **固定比较**：Base V3 原始输出、简单置信度拒绝、V5.1 分量策略。V5.1 保留为参考，不能称总体赢家；V5.2 主候选停止；V5.2.1 禁止把尺度保持当作有效观测；V5.3 图像质量候选已经停止。
- **评价原则**：等覆盖率比较中心、尺度、周期方向误差；所有帧和漏检都计入覆盖率，同时报告错误接受、拒绝代价、不可用长度与恢复。长漏检必须转为 `unavailable`。
- **最新报告链**：统一观测运行时 → RGB 真实在线入口 → V2 后验评估及无 GT 观测 CSV → V3 论文制表指标。在线决策不读取 GT；事后评估与模型推理解耦。
- **数据职责**：738 帧 source-val 在后期可靠性工作中已作为开发数据，不能继续当作未暴露的独立验证（历史阶段的 source gate 原称谓保留）；992 帧 fixed TEST 已暴露，V3 仅用于固定结果报告，不再用于选择策略。
- **尚未证明**：未知新序列、物理状态和端到端部署 FPS 都不由这些记录证明。时间戳缺失时使用 `UNAVAILABLE_NOT_SYNTHESIZED`，不伪造采样时间；风险值不是已校准概率或 EKF 测量方差。

<a id="sec-1-4"></a>

### 1.4 已有基础：已完成 / 部分完成 / 真正缺失

早期首轮审计记录的「未发现该新方案的完整实现」等表述属于 2026-09-07 的盘点状态；此后接口、报告链与多轮策略实验已经落地。以下按当前证据重新分类，不再原样保留早期清单。

**已完成（有冻结结果与可复现入口）**

| 内容 | 结论与入口 |
| --- | --- |
| 检测前端绑定 | Base V3 epoch9：K1 epoch24 有框优先、缺失时 DINO fallback；refiner 常数 `1.0` 不作置信度（见 2.2） |
| 分量化观测接口 | 中心 / 尺度 / 角度分别输出数值、有效性、状态、来源、帧龄、风险与拒绝原因（见 4.1） |
| 图像→连续观测的真实在线推理 | `base_v3_obb_true_online_pipeline_v1`，`execution_mode=true_online_model_inference`（见 4.2） |
| 无 GT 的逐帧观测导出 | V2 导出的观测 CSV，保留方法/分量/数值/状态/来源/帧龄/风险/门控/原因（见 4.3） |
| 固定 TEST 论文制表指标 | V3 输出 9 + 27 + 6 行三张表及补充诊断，并绑定三份输入的 SHA256（见 4.4） |
| 观测层运行开销 | 单独打印到终端，不写入需保持稳定哈希的正式 JSON（见 4.1） |
| 分量策略的冻结与停止裁决 | V5.1 冻结；V5.2 / V5.2.1 / V5.3 全部停止并给出 SHA256（见 3.3–3.5） |

**部分完成**

| 内容 | 已做到 | 尚缺 |
| --- | --- | --- |
| 分量可靠性方法 | 已实现并评估「拒绝 + 短时保持」两类机制，含 matched-coverage 基线比较 | 跨视频泛化；尚未出现稳定优于简单置信度筛选的分量策略 |
| 等覆盖率评价 | 已产出 scale 同覆盖率排序、角度保持收益与代价、拒绝恢复 | `R_center` / `TDR_w10` 的统一口径只存在于系统级 evaluator 侧，分量表尚未全部对齐 |
| 数据与证据层级 | source-val / fixed TEST 的职责已分表说明 | 未知新序列与部署证据缺失 |
| 深度衔接 | 观测 CSV 可作为 Raw-opt 与状态估计接口 | 风险值未校准为概率或测量方差，不能直接作 EKF 协方差 |

**真正缺失**

| 内容 | 状态 |
| --- | --- |
| 未知新序列验证 | 无（fixed TEST 已暴露，不能再充当未知域） |
| 物理状态与三维闭环 | 无（见 5.2 与 Webots 文档） |
| 端到端部署 FPS / 资源证据 | 无（只有观测层开销） |
| 阈值、特征与策略选择所需的独立校准支持 | 不足：738 帧 source-val 后期已作为开发数据（见 1.3） |
| 文献新颖性核查与稿件定稿 | 未做（见 5.4） |

**不再作为当前方向（历史线索）**

| 内容 | 处置 |
| --- | --- |
| 图像质量特征作为尺度错误排序的增量 | V5.3 已裁决 `STOP_IMAGE_QUALITY_CANDIDATE`；该分支只作为消融与失败边界报告 |
| 一帧尺度保持 | V5.2.1 已裁决 `DISABLE_V52_SCALE_HOLD_AS_VALID_OBSERVATION` |
| 长/短边拆分尺度策略 | V5.2 已裁决 `STOP_V52_PRIMARY_AND_ABLATIONS_ONLY` |
| 相机共模补偿 | 不再是必做模块 |

**补充的数据角色边界**

- **seq11-v2（251 帧，历史 203/48 划分）**：aux-val 48 帧不含任何困难支持样本，因此只能提供「同一视频内的机制验证」，不等于独立泛化；该数据属于检测融合支线，不作为本主线的校准或验证集。
- **seq02 / seq03 困难片段**：已用于诊断并已暴露，不是全新未知序列；不重开旧 S7 / router 实验。
- **fixed TEST（992 帧）**：已暴露，只用于固定模型的结果报告（见第 4 节），不能充当未知域。

检测侧的身份、数据角色与证据边界见 [冻结 DINOv2 与 SymEOOD 检测融合](冻结DINOv2与SymEOOD检测融合.md) 第 1.3 节与第 6.2 节；那里把 V4+seq11-v2 epoch10 明确列为 source-val 暂定候选，缺少困难 OOF 支持，不能直接升级为最终模型。用户说模型改进基本停止，意味着不自动重开训练，不意味着该候选通过全部验收。

用户背景材料位于 `/Users/mac/Desktop/大论文研究内容/背景.md`，按需读取。

另一条正式 native-S14 入口为 `crane_project/configs/crane_symeood_formal_dino_native_s14_v1.py`，其纯 DINO 身份与 SymEOOD K1、融合候选不同。不能把深度诊断所用的 native-S14 自动当作小论文的「SymEOOD 模型」。

---

## 2. 观测契约、方法与必要协议

<a id="sec-2"></a>

<a id="sec-2-1"></a>

### 2.1 第一轮可靠性审计协议

<a id="sec-2-1-1"></a>

#### 2.1.1 冻结输入

对一个唯一、可哈希的检测器版本导出全量逐帧表；每行包含 `split_role, sequence, frame, timestamp, prediction_present, cx, cy, w, h, angle_le90, score`，以及不可变 `config_sha256, checkpoint_sha256, prediction_sha256, manifest_sha256`。每帧只保留该模型实际推理的输出，不接管或修正几何。

<a id="sec-2-1-2"></a>

#### 2.1.2 离线误差标签

将 GT 只用于生成中心、尺度、方向、存在性误差标签。方向以 `pi` 周期表示；近正方形的方向退化必须单独标记或预注册为「不评价方向」，不可伪造角度正确性。漏检计入所有 coverage/risk 曲线。

<a id="sec-2-1-3"></a>

#### 2.1.3 先审计支持，再拟合

在 source-train 的 OOF 预测或职责清晰的独立 calibration split 上，统计 score、归一化尺度、长宽比/近方形标记、边界截断、当前-过去 causal 残差与三类误差的关系。时间特征只能使用当前和过去，时序块边界清除历史；不得按 target 逐帧调规则。

<a id="sec-2-1-4"></a>

#### 2.1.4 预注册三个输出

`r_center, r_scale, r_angle` 及各自 `valid`。先比较 score-only、score + current-frame geometry、score + causal residual 三层；整体 valid 必须由已冻结的组合规则导出，不允许评测时挑最好分量。

<a id="sec-2-1-5"></a>

#### 2.1.5 等 coverage 主比较

在每个分量上将 proposed 与 score-only 匹配相同的总体保留率（包括漏检），报告 risk-coverage curve、固定 coverage 的平均/分位误差、错误接受率、正确拒绝率和拒绝代价。不能只报告被保留帧的平均误差。

<a id="sec-2-1-6"></a>

#### 2.1.6 连续接口后置

首先验证逐帧可靠性；随后才比较 `measurement / bounded prediction / unavailable`，报告各状态覆盖、连续 unavailable 长度、延迟。长漏检必须转为 unavailable，历史框不能伪装成新测量。

<a id="sec-2-1-7"></a>

#### 2.1.7 证据层级

source OOF/校准用于选规则；official source gate 用于冻结；fixed target-dev 只诊断；未知完整序列验证泛化；资源/时延才支持工程接口结论。五者分表，不混图、不共享阈值选择。

<a id="sec-2-2"></a>

### 2.2 模型与数据身份清单

| 项目 | 可用状态 | 身份与约束 |
| --- | --- | --- |
| V4+seq11-v2 replay `epoch_10` | 缺失于本机 | 期望路径 `work_dirs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v4_seq11_v2_replay_seed3407/epoch_10.pth`；训练配置 `crane_project/configs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v4_seq11_v2_replay.py`。2984 source-train frames = original 2781 + seq11 train 203；official source-val 738；seq11 aux-val 48 仅同视频机制验证。不得把该候选升格为 source-gated 或最终模型 |
| V4 replay config | 主工作区可读 | 冻结 Base-V3 teacher、cached DINO proposals、K1 current anchor、4-frame causal history；不读 target/fixed TEST，禁止 domain/sequence/frame routing。config 写明新 checkpoint `source_gate_passed=False` |
| Base-V3 teacher | 本轮未本地绑定工件 | replay config 要求一个已 source-promoted epoch9 teacher；其 checkpoint SHA256 从 promotion report 动态读取。此报告不猜测其哈希 |
| 普通 SymEOOD K1 epoch24 | 本机可读 | config `crane_project/configs/crane_symeood_k1_source_val_eval.py`；checkpoint `work_dirs/crane_symeood_k1/epoch_24.pth`；SHA256 `57233e5423de4a9d0a67fd51058cd7d92adaac5d6621f275cc0ec5b0fc7f9ee2`。这是普通 K1，而非 V4+seq11-v2 |
| K1 source-val predictions | 本机可读 | `work_dirs/crane_symeood_k1/source_val_epoch24_results.pkl`；SHA256 `361ccd8848c162e92477e04e38f87cd7f28209f07129823e84045e8ec74031c9`；738 条、每帧 top prediction 为 `(cx, cy, w, h, angle, score)`，原图像素、`le90` |
| source-val GT | 本机可读 | `crane_project/data/crane_grab/val/annfiles/` 有 738 个 DOTA polygon txt；对应 738 images；序列为 `real_seq07` 和 `sim_seq10`。标签离线使用，不参与规则设计或推理 |
| fixed TEST / target-dev / unknown | 本轮未读取 | 不得用作该可靠性规则选择、阈值扫描或统计替代；未知序列仍缺失 |

**2026-09-07 的身份保留状态（历史）**：增量贡献仍未成立，必须先证明分量可靠性筛选在相同保留覆盖率下比检测置信度筛选更能减少中心、尺度和周期方向误差；当时唯一应优先绑定的检测候选是 V4+seq11-v2 replay `epoch_10`，按 2026-09-05 交接它只是 738-frame official source-val 的暂定 source-retention candidate（`source_gate_passed=False`，没有困难块 OOF/CV 支持），不能称为最终模型。本机没有该候选的 checkpoint、冻结合同、source-val 逐帧预测或交接所列复核包，因此不能在本机对它训练或评价可靠性规则，也不能用旧 K1 输出替代其身份。本机的旧 K1 source-val 输出角色明确、已用于验证审计数据管线可运行，仅为基线/通路证据，**不可与 V4+seq11-v2 指标或图表合并**。

**当前前端身份（2026-09-10 收尾记录起）**：可靠性主线绑定的是 **Base V3 epoch9**——正式 K1 epoch24 有框时作为锚框、K1 缺失时使用冻结 DINO 候选，refiner 输出常数分数 `1.0` 不作为置信度。V4+seq11-v2 `epoch_10` 保留为检测融合 seq11 支线的历史暂定候选，不再是本主线的待绑定项；上表的「缺失于本机」条目因此只描述 2026-09-07 的盘点结果。四个模型身份的区别见 [冻结 DINOv2 与 SymEOOD 检测融合](冻结DINOv2与SymEOOD检测融合.md) 第 1.3 节。

<a id="sec-2-3"></a>

### 2.3 服务器只读导出需求（历史执行记录）

当时要求在 `/media/omnisky/personal_files/ljj/symEOOD` 对**已存在**的 V4+seq11-v2 `epoch_10` 做只读打包/导出；不训练、不执行 fixed TEST、不读取未知数据。至少导出：

```text
work_dirs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v4_seq11_v2_replay_seed3407/
  epoch_10.pth
  checkpoint_eval_summary.json
  geometry_refiner_frozen_contract.json (或实际同等冻结合同)
  source-val epoch_10 results.pkl
  source-val per-frame all-lane prediction JSON
  训练日志中 config 的 resolved copy
  所有上述文件的 SHA256 清单

crane_project/configs/crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v4_seq11_v2_replay.py
work_dirs/crane_symeood_dino_conservative_takeover_v2/source_calibration_collect/source_val_fusion_source_audit.json
work_dirs/crane_symeood_dino_distill_support_v1/source_collect/source_train_all_lane_audit.json
work_dirs/crane_symeood_dino_source_inventory_v2/real_seq11_k1p9_v2/
  full_source_contract.json
  blocksplit_v2/audited_split_materialization.json
  blocksplit_v2/train_all_lane_audit.json
  blocksplit_v2/aux_val_all_lane_audit.json
crane_project/data/crane_grab/extra_source_real_seq11_pilot_k1p9_v2/split_manifest.json
```

逐帧 JSON 不能是 `records=0` 的 summary-only `failure_audit_real*.json`。它必须覆盖 official source-val 的全部 738 帧，且包含 filename、split、sequence、frame、timestamp（无则显式 null）、prediction_present、最终 OBB+score、K1/DINO/refiner provenance、原图坐标与 `le90` 角度约定。若需要在服务器生成该 JSON，推理命令必须只针对 official source-val，且将 config/checkpoint SHA256、阈值、NMS/max-per-image 和输入 manifest 写入旁路 metadata；这属于导出既定候选预测，不能用结果修改模型或规则。

当时的授权顺序是：收到完整 source-side 导出后，先验证 manifest/哈希/逐帧对齐与 OOF 或独立 calibration 的职责，再做只读错误支持和 score-only risk-coverage 基线。只有在相同 coverage 下出现可复现的分量风险降低，才进入可靠性方法实现；否则如实保留「检测输出可审计、可靠性新增贡献尚未成立」的结论。第一轮不训练检测器、不重跑旧实验、不读取未知集，也不把 source-val 或 fixed-dev 改名成训练集。

**环境说明**：服务器仓库为 `/media/omnisky/personal_files/ljj/symEOOD`；本机通常无权重，优先复用服务器预测，后续推理在服务器执行。

---

## 3. 实验结果与演进裁决

<a id="sec-3"></a>

<a id="sec-3-1"></a>

### 3.1 只读通路统计：K1 source-val

这一组数字来自普通 K1 `epoch_24` 的 source-val 输出，**不是候选模型结论**，只用于验证审计数据管线可运行。以文件名排序对齐 K1 `results.pkl` 与 738 个 source-val 标注。GT 四点框按代码的 `le90` 约定转换；对有预测的帧，中心误差为像素欧氏距离，尺度误差为 `mean(abs(log(w_hat/w)), abs(log(h_hat/h)))`，方向误差为 `pi` 周期的最小绝对差。所有缺失预测必须在正式 coverage 中保留，不能静默丢弃。

| 项目 | 结果 |
| --- | --- |
| 对齐 / 检出 / 漏检 | 738 / 736 / 2 |
| 中心误差 px (p50 / p90 / p95 / max) | 1.859 / 5.183 / 6.435 / 13.218 |
| 尺度误差 (p50 / p90 / p95 / max) | 0.0516 / 0.1359 / 0.1584 / 0.2903 |
| 周期方向误差 deg (p50 / p90 / p95 / max) | 1.284 / 4.769 / 6.186 / 10.172 |
| score (p50 / p90 / p95) | 0.9108 / 0.9515 / 0.9547 |
| Pearson(score, 中心/尺度/方向误差) | -0.3247 / -0.2985 / -0.2760 |

这些相关性仅说明在该 K1 source-val 上 score 与三种误差存在有限负关联，它们**不支持**声称「置信度基线已被超越」。中心和方向在该工件中错误尾部很少，正式训练/验证需要按每个分量的错误支持量和 OOF 角色重审，而非从这些数字预设阈值。

<a id="sec-3-2"></a>

### 3.2 V5.1：分量可靠性策略

source-val 上通过当时的冻结门，保留为后续比较参考：

- 中心使用锚框置信度拒绝，拒绝后输出不可用；
- 尺度使用 V5 单尺度风险，拒绝后输出不可用；
- 角度使用锚框置信度，并允许最多一帧保持；
- 后续 fixed TEST 已暴露跨序列泛化问题，因此 V5.1 **不是最终方法**，也不声明为总体最优方法。

<a id="sec-3-3"></a>

### 3.3 V5.2 与 V5.2.1：长/短边拆分尺度策略

**V5.2** 正式报告：`source_val_obb_split_scale_policy_development_v52.json`

```text
SHA256：ac85dc4c72499645dcc26d30c55f7e3f00bdf5b4fc9081e426c32723fed1e9b0
source split：294 calibration / 444 evaluation
主候选 split_existing_features 未通过
决策：STOP_V52_PRIMARY_AND_REPORT_ABLATIONS_ONLY
```

长边、短边拆分本身没有形成稳定优于 V5.1 或简单置信度的证据。

**V5.2.1** 正式报告：`source_val_obb_split_scale_policy_diagnostic_v521.json`

```text
SHA256：800f909cc72facdaecd4089cdb0205a11908b4935b2c4c605873ce78f46d9ce9
```

- V5.2 主候选的一帧尺度保持补充 19 帧，其中 2 帧正确、17 帧错误；
- 保持正确率：10.53%；
- 相对当前帧原始尺度的平均误差变化：`+0.002296`；
- 输出覆盖率增加 4.28 个百分点，正确输出覆盖率仅增加 0.45 个百分点；
- 最长不可用段从 5 帧缩短到 4 帧；
- 决策：`DISABLE_V52_SCALE_HOLD_AS_VALID_OBSERVATION`。

三个尺度保持分支均未通过严格门。图像质量分支在 pooled source-val 上相对 V5.1 的长边、短边和联合尺度各有 3/5 个覆盖率点胜出；real 联合尺度相对置信度为 2/5，sim 为 0/5。因此它只形成后续开发假设。

<a id="sec-3-4"></a>

### 3.4 V5.3：当前帧图像质量特征

唯一开发问题：当前帧 Base V3 OBB 区域的亮度、对比度、暗像素比例、亮像素比例和截断比例，能否在已有在线特征之外提高尺度错误排序能力。

固定比较：

1. `anchor_score`：简单置信度；
2. `existing_scale_risk`：已有 11 个在线几何、候选分歧和因果残差特征；
3. `existing_plus_image_quality`：已有特征加 5 个当前帧 OBB ROI 图像质量特征。

V5.3 只输出尺度 `measurement` 或 `unavailable`，不增加尺度保持，不修改融合模型，不使用作业阶段、域或序列身份进行路由；中心和角度继续沿用 V5.1 接口，不参与这次候选选择。

开发与最终验证边界：

- 当前 738 帧 source-val 全部属于开发数据；
- V5.3 使用 `real_seq07` 和 `sim_seq10` 各自的三段连续块 OOF，验证核心前后两帧从该折训练标签中隔离；
- OOF 用于决定图像质量分支是否值得冻结，不用于未知序列或部署声明；
- 已暴露 fixed TEST 不再用于 V5.3 调参或最终证明；
- 最终方法定稿后，使用预先冻结的其他视频或从未参与开发的本地视频进行一次最终验证。

结果：V5.3 正式报告 SHA256 `8f3d210ee1e657b7aadd5bb9691985e1e6240fab8ab2c200770ff0fd89466618`；real 域相对已有尺度风险仅 1/5 个覆盖率点胜出，三折只有一折胜多于负，决策 `STOP_IMAGE_QUALITY_CANDIDATE`。

<a id="sec-3-5"></a>

### 3.5 停止裁决与可继续使用的内容

| 候选 | 关键结果 | 决策 |
| --- | --- | --- |
| V5.2 长/短边拆分 | 主候选 `split_existing_features` 未通过；source split 294/444 | `STOP_V52_PRIMARY_AND_REPORT_ABLATIONS_ONLY` |
| V5.2.1 一帧尺度保持 | 补充 19 帧中 17 帧错误；覆盖 +4.28pp，正确覆盖仅 +0.45pp | `DISABLE_V52_SCALE_HOLD_AS_VALID_OBSERVATION` |
| V5.3 图像质量特征 | real 相对已有尺度风险仅 1/5 胜出，三折一折胜多于负 | `STOP_IMAGE_QUALITY_CANDIDATE` |

V5.2、V5.2.1 和 V5.3 作为消融与失败边界报告，不再列为待执行计划。

**当前可继续使用的内容**

- Base V3 作为冻结检测参考前端；
- K1 优先、K1 缺失时 DINO fallback 的确定性来源规则；
- 中心、尺度、角度分别输出数值和有效性状态的接口；
- V5.1 作为比较参考；
- 尺度拒绝后输出 `unavailable`，不把一帧尺度保持标为有效观测；
- 简单置信度必须作为 matched-coverage 基线。

---

## 4. 当前接口与复现入口

<a id="sec-4"></a>

**报告版本、用途与优先级**

本节涉及多个报告文件，它们不是互相替代的「最终指标」，而是分工不同。用途与优先级如下：

| 报告 | 输入 | 用途 | 能否用于选择 | 优先级 |
| --- | --- | --- | --- | --- |
| V1 在线推理报告（缓存回放参照） | 历史缓存 DINO/Base V3 输入 | 与 V2 逐项比较，定位候选输入变化与四帧历史传播的差异 | 不能 | 参照文档，保留不删 |
| V2 `base_v3_obb_true_online_finalization_v2` | 992 帧完整真实在线推理输出 | 计算中心/尺度/角度、联合有效性与完整 OBB RIoU；导出**无 GT** 逐帧观测 CSV | 不能（TEST 已暴露） | 系统实现的评估结果 |
| `fixed_test_obb_paper_final_report_v1` | 冻结的 Base V3 几何与三分量状态 | 论文固定指标来源；`--paper-final-report` 只做逐帧核对，不参与在线决策 | 不能 | 论文固定指标来源 |
| V3 `base_v3_obb_true_online_paper_metrics_v3` | 已冻结的 V2 真实在线评估 + V1 在线推理报告 | 论文制表字段：完整 OBB 9 行、分量 27 行、锚来源 6 行 + 补充诊断 | 不能（不重跑检测、不拟合阈值） | 论文表格字段来源 |
| 整体报告 `base_v3_obb_paper_final_report_v1` | 正式 V5.1 TEST 报告、V5.1 诊断、V5.3 停止结果 | 汇总分量误差与覆盖率、拒绝恢复、联合有效性、角度保持、尺度同覆盖率比较与完整 OBB RIoU | 不能 | 汇总报告 |

口径与优先级规则：

- 论文正文以**真实在线结果**（V2/V3）作为系统实现结果；若 V1 与 V2 的候选输入存在差异，V1 只标注为缓存回放参照。
- `fixed_test_obb_paper_final_report_v1` 与 V3 不是同一层：前者是冻结的几何/三分量状态报告，后者是从 V2/V1 推导出的论文表格字段。二者都不授权再次调参。
- 三个版本的 `R_center` 口径必须分别标注：`R_center^det@15px`（有输出帧条件中心精度，来自 `CraneOfflineEvaluator`）与 `R_center^all@25px`（含 silence 的全帧中心召回，real 25px / sim 10px）。系统级 evaluator 另有 `R_center` / `TDR_w10` / MCML 口径，见 [冻结 DINOv2 与 SymEOOD 检测融合](冻结DINOv2与SymEOOD检测融合.md) 第 2.1 节；分量表与系统表不得混用同一数字。

<a id="sec-4-1"></a>

### 4.1 统一观测运行时与整体报告

在线接口为 `base_v3_obb_paper_runtime_v1`。每帧保留检测来源、分数、原始 OBB，以及三个方法的中心、尺度、角度数值、有效性、状态、来源、历史年龄、风险和拒绝原因。**GT 不进入在线接口。**

小论文固定报告三个方法：Base V3 原始输出、简单置信度拒绝、V5.1 分量策略；V5.2、V5.2.1 和 V5.3 作为消融与失败边界。

整体报告为 `base_v3_obb_paper_final_report_v1`，绑定正式 V5.1 TEST 报告、V5.1 诊断和 V5.3 停止结果，统一输出分量误差与覆盖率、拒绝恢复、联合有效性、角度保持、尺度同覆盖率比较和完整 OBB RIoU。生成报告前必须由统一在线运行时逐帧复现正式 V5.1 输出，且完整 OBB 重组保留 Base V3 原始宽高与角度轴对应关系。不含检测器和图像 I/O 的观测层运行开销单独打印到终端，不写入需要保持稳定哈希的正式 JSON，也不能表述为端到端 FPS。

fixed TEST 已经暴露，因此整体报告可作为固定划分的可复现实验结果，不作为全新独立序列证据，也不再用于选择阈值、特征、策略或 checkpoint。

<a id="sec-4-2"></a>

### 4.2 从图像到连续观测的真实在线入口

`base_v3_obb_true_online_pipeline_v1` 把此前分开的两段正式串联为一个按序推理入口：

```text
RGB 图像
  -> 冻结 native-S14 DINO
  -> 当前 DINO Top-1 + 最近四帧严格因果 DINO 历史
  -> Base V3 epoch9
       Base V3 内部从共享 FPN 特征运行正式 SymEOOD K1 epoch24
       以 K1 有框优先、否则 DINO 回退的规则形成锚框并输出 Top-1 OBB
  -> 冻结 V5.1
  -> 输出中心、尺度、角度的 measurement / prediction / unavailable
```

该入口的 `execution_mode` 为 `true_online_model_inference`，不读取缓存 DINO 预测、缓存 Base V3 预测、标注或 GT。序列和帧号只用于排序与间断复位，不作为检测或可靠性路由。Base V3 新增的 `last_inference_records()` 只导出已在同一次 forward 中计算的 K1/DINO/锚来源证据，不重复运行 K1，也不改变模型输出。

可选的 `--paper-final-report` 只在全部在线输出完成后逐帧核对冻结报告中的 Base V3 几何和三分量状态；它不参与任何在线决策，也不授权使用已暴露 TEST 调参。`fixed_test_obb_paper_final_report_v1` 仍是论文固定指标来源；新入口负责证明这些检测与观测模块能够按设计真实串联。

source-val 的锚框来源分布为 K1 736 帧、DINO fallback 2 帧；Base V3 最终 refiner 分数 `1.0` 不作为置信度。

<a id="sec-4-3"></a>

### 4.3 V2：真实在线输出的后验收尾

完整 992 帧在线推理生成后，使用 `base_v3_obb_true_online_finalization_v2` 在推理结束后附加标注。该入口直接计算真实在线输出的中心、尺度、角度、联合有效性和完整 OBB RIoU，并把结果与历史缓存输入生成的 V1 报告逐项比较；它还分别审计 DINO、K1 和 Base V3 几何，因此候选输入变化与四帧历史传播不会再被合并成一个最大数值差。

V2 同时导出不含 GT 的逐帧观测 CSV，保留方法、分量、数值、状态、来源、帧龄、风险、门控和原因，可作为后续 Raw-opt 与状态估计接口。固定 TEST 已经暴露，V2 只用于最终复现和差异归因，不允许据此修改阈值、特征、策略、epoch 或 checkpoint。历史 V1 报告继续保留；若 V2 与 V1 存在输入差异，论文正文应以真实在线结果作为系统实现结果，并把 V1 标为缓存回放参照。

<a id="sec-4-4"></a>

### 4.4 V3：真实在线论文指标

`base_v3_obb_true_online_paper_metrics_v3` 只读取已经冻结的 V2 真实在线评估与 V1 在线推理报告，不重新执行检测、拟合阈值或选择策略。最终 JSON 固定输出三张可制表数据：

| 数据表 | 内容 |
| --- | --- |
| 完整 OBB 指标 | 3 方法 × all/real/sim，共 9 行 |
| 分量指标 | 3 方法 × 3 分量 × all/real/sim，共 27 行 |
| 锚来源指标 | 3 方法 × K1/DINO fallback，共 6 行 |

补充诊断包括尺度同覆盖率排序、角度保持收益与代价、拒绝恢复，以及 V5.1 相对简单置信度拒绝的差值。

V3 同时绑定真实在线报告、V2 后验指标和无 GT 观测 CSV 的 SHA256，并显式检查 992 帧集合、域/序列计数、冻结评估阈值、前端顺序和 Base V3 输出分数语义。`base_v3_output_score=1.0` 被记录为 `constant_refiner_output_not_confidence`，不能进入置信度比较。时间戳在现有输入中为空，报告为 `UNAVAILABLE_NOT_SYNTHESIZED`，不伪造时间信息。

该报告的「完整」仅指固定 TEST 上从 RGB、融合检测到连续 OBB 观测的指标链和论文表格字段已经齐全。它不表示 V5.1 是总体赢家，也不补足未知新序列、物理状态或部署 FPS 证据。fixed TEST 已暴露，V3 之后禁止继续根据该报告改阈值或策略。

<a id="sec-4-5"></a>

### 4.5 后续大论文接口

大论文可以消费各分量的 `value`、`state`、`source` 和 `age_since_measurement_frames`，并使用时间戳与图像尺寸字段连接 Raw-opt 深度和状态估计。当前风险值尚未校准为概率或测量方差，**不能直接作为 EKF 协方差**。

---

## 5. 未解决问题与适用边界

<a id="sec-5"></a>

<a id="sec-5-1"></a>

### 5.1 验收材料的完成状态

首轮审计列出的八项按当前证据重新归档。**不再是「全部缺失」**：接口、报告链与分量策略评估已完成，未完成的集中在未知序列、物理状态、校准支持与稿件。

**已完成**

| 原清单项 | 当前状态 |
| --- | --- |
| 1. 检测器唯一身份（config / checkpoint / SHA256 / 坐标与角度约定 / 各 split 产物） | 主线前端已绑定为 Base V3 epoch9（K1 epoch24 有框优先、DINO fallback），见 2.2；SHA256 可在 2.2 表逐项查到 |
| 4. 可靠性规则、分量有效标准与其超参数来源 | 已实现并冻结：V5.1 冻结，V5.2 / V5.2.1 / V5.3 的 SHA256 与停止裁决见 3.3–3.5 |
| 5. 等覆盖率误差、拒绝代价、连续不可用长度 | 已产出：尺度同覆盖率排序、拒绝恢复、角度保持收益与代价（见 4.1 整体报告） |
| 6. 置信度筛选基线与分量消融 | 已产出：三重固定比较为 Base V3 原始输出、简单置信度拒绝、V5.1（见 4.1） |

**部分完成**

| 原清单项 | 已做到 | 尚缺 |
| --- | --- | --- |
| 2. 可复用原图—GT—预测逐帧对照表 | source-val 侧已有 `source_val_epoch24_results.pkl` 与 738 标注的对齐产物；在线侧已导出无 GT 观测 CSV | 含序列与时间戳、覆盖三份正式 split 的统一对照表仍不完整；时间戳在现有在线输入中为空（报告为 `UNAVAILABLE_NOT_SYNTHESIZED`） |
| 3. 分量误差的监督侧支持审计；置信度与各分量误差关系 | 已完成 K1 source-val 的 Pearson 相关（−0.3247 / −0.2985 / −0.2760，见 3.1） | 该统计属普通 K1 工件，不是主线前端；分量级 OOF 支持量与独立校准集仍未建立 |
| 5. risk-coverage 曲线的完整口径 | 已有分量与同覆盖率比较 | `R_center` / `TDR_w10` / MCML 的分量表与系统表口径未完全统一（见第 4 节开头口径规则） |
| 6. 时序特征增量与延迟 | 已有分量策略的拒绝/保持代价与观测层开销 | 观测层开销不能表述为端到端 FPS |
| 7. 冻结后的独立序列验证与真实视频时延/资源 | covered：fixed TEST 报告已冻结 | 未知新序列与真实视频资源证据缺失 |

**真正缺失**

| 原清单项 | 原因 |
| --- | --- |
| 7. 独立序列泛化验证 | fixed TEST 已暴露，不能充当未知域；无第三序列 |
| 物理状态与三维闭环 | 属大论文后半部分，且 Webots 侧只在有条件下成立（见 5.2） |
| 8. 文献新颖性核查与稿件整理 | 未做；见 5.4。不能因为新增模块有名字或指标改善就保证创新性/发表 |

<a id="sec-5-2"></a>

### 5.2 深度分支的边界

Raw-opt 标量深度配对修复**不代表完整三维链闭合**。三维反投影尚未闭环、动态重力坐标转换缺失、`depth_valid` 尚未冻结；固定旋转仅在相机相对重力固定等条件下适用，其姿态变化具体数值本轮未重新计算。

Webots 文档中的 Unknown-02 只属旧 Plumb/Opt 契约的历史证据，不能作为 Raw-opt 的新未知序列验证。Train-03 是深度标定序列，不自动成为可靠性方法的校准数据；若使用必须明确新协议职责和既有暴露。不要为小论文继续实现三维外参/EKF。

详细契约与当前结论见 [Webots 单目深度估计](Webots单目深度估计.md)；其中早期「完整修复」措辞应以上述限制解读。新加入的 OBB 连续接口也不自动证明 Base V3 已通过 Webots 深度验证。

<a id="sec-5-3"></a>

### 5.3 允许与不允许的论文表述

**可以表述**（逐项标明实验、数据范围、比较基线与覆盖率胜出数）

- 已建立 OBB 分量化观测接口，并系统评估了拒绝与短时保持的收益和代价。
- **一帧尺度保持会补入大量错误观测**：实验 = V5.2.1 诊断（`source_val_obb_split_scale_policy_diagnostic_v521.json`，SHA256 `800f909cc72facdaecd4089cdb0205a11908b4935b2c4c605873ce78f46d9ce9`）；比较对象 = 当前帧原始尺度；结果 = 补充 19 帧中 2 帧正确、17 帧错误，平均误差变化 `+0.002296`，覆盖 +4.28pp 而正确覆盖仅 +0.45pp。
- **图像质量特征的早期探索信号（pooled / 分域比较，属开发假设）**：实验 = 图像质量分支（V5.2.1 阶段记录）；数据范围 = **pooled source-val**；比较基线 = 相对 **V5.1** 时，长边、短边、联合尺度各有 **3/5** 个覆盖率点胜出；相对 **简单置信度** 时，real 联合尺度 **2/5**、sim **0/5**。结论 = 只形成后续开发假设。
- **V5.3 的正式结论（OOF，独立于上一行，两套数字不可合并）**：实验 = V5.3 正式报告（SHA256 `8f3d210ee1e657b7aadd5bb9691985e1e6240fab8ab2c200770ff0fd89466618`，与 4.4 的 `base_v3_obb_true_online_paper_metrics_v3` 不是同一文件）；数据范围 = `real_seq07` / `sim_seq10` 各自三段连续块 OOF；比较基线 = **已有 11 个在线尺度风险特征**（不是 V5.1，也不是置信度）；结果 = real 域仅 **1/5** 个覆盖率点胜出，三折只有一折胜多于负；决策 = `STOP_IMAGE_QUALITY_CANDIDATE`。该分支只作为失败边界与消融报告，**不再是当前开发方向**。

**不能表述**：学习式可靠性已经跨视频泛化；V5.2 优于置信度；尺度保持提高了有效观测准确率；该接口已经输出物理摆角和三维状态；V5.1 是总体最优方法；真实在线入口证明检测与观测链已经在未知序列或部署条件下通过验收；图像质量特征仍是当前开发方向；观测层运行开销等于端到端 FPS；fixed TEST 的 V2/V3 报告等于未知序列证据。

**写作自检**

- 贡献：检测有历史证据；可靠性是待验证贡献。
- 表述：二维图像方向不是机械自旋/物理摆角；连续输出不是连续有效测量；Base V3 的 refiner 分数 `1.0` 不是置信度。
- 实验：模型、split、阈值来源、覆盖率必须可追溯；三份报告的用途与优先级见第 4 节开头。
- 完整性：同视频 OOF 不能替代未知序列；本文未审查稿件完成度。
- 方法：可靠性筛选不能修复错误几何，必须计入拒绝代价。

<a id="sec-5-4"></a>

### 5.4 稿件盘点

`/Users/mac/Desktop/lunwenxiezuo/EAAI_Paper/main.tex` 是完整的 Elsevier 单文件脚手架，不是空目录：已有背景/相关工作、OBB 输出边界、数据与结果表结构、draft pipeline 图，以及明确的「非闭环」限制。它没有可提交的结果：贡献条目、数据协议、对比/消融表、定量结果、图例、结论和声明仍为 placeholder/draft；当前稿件也尚未实现本报告的「分量可靠性—连续输出」实验叙事。`YSU_Thesis/document.tex` 是大论文资产，应保持深度、外参、物理摆角等后半部分与小论文分开。

---

<a id="appendix-sources"></a>

## 附录 A 来源索引

原文件已逐字节保存在 [整合前归档](archive/20260915_pre_consolidation/)，只用于追溯，不继续维护。

| 锚点 | 归档文件 | 原文件开头（历史快照） | 正文去向 |
| --- | --- | --- | --- |
| <a id="source-h"></a>`source-h` | `archive/20260915_pre_consolidation/journal_obb_reliability_handoff_20260907.md` | *小论文进度与新对话交接：OBB观测可靠性与连续输出*，更新 2026-09-07；「不是模型性能验收报告。模型融合另一对话若有更新，应读取其实际产物后更新模型身份」 | 1.1、1.4、5.1、5.2、5.3、5.4 |
| <a id="source-a"></a>`source-a` | `archive/20260915_pre_consolidation/journal_obb_reliability_first_audit_20260907.md` | *小论文 OBB 观测可靠性首轮审计*，2026-09-07；「范围限于 SymEOOD 抓斗旋转检测 → OBB 分量可靠性判定 → 连续视觉观测输出。本文是只读盘点和审计设计，不是模型验收」 | 1.2、2.1、2.2、2.3、3.1、5.4 |
| <a id="source-p"></a>`source-p` | `archive/20260915_pre_consolidation/20260910_obb_reliability_results_ledger_and_v53_plan.md` | *OBB 分量可靠性结果台账与 V5.3 计划*，2026-09-10 | 3.2、3.3、3.4、3.5 |
| <a id="source-r"></a>`source-r` | `archive/20260915_pre_consolidation/20260910_obb_reliability_results_ledger_and_finalization.md` | *OBB 分量可靠性结果台账与整体收尾*，2026-09-10 | 1.3、3.5、4.1–4.5 |

原文件在仓库中的旧路径为 `docs/<原文件名>`。旧引用（含记忆文件与其它文档）按归档文件名在本表定位即可；本节保留的 `source-*` 锚点即原来的跨文档引用目标。

整合清单与哈希：[manifest.json](archive/20260915_pre_consolidation/manifest.json)。该 manifest 是 **2026-09-15 首次机械整合的快照**：其中的 `anchor`、`integrated_body_sha256` 字段描述的是当时的章节编号与结构，本文件的语义重写已改变编号与组织方式，因此这些字段不再对应当前正文，只作为归档原文的身份与完整性凭据。本轮（第二轮）的「原锚点 → 新正文位置」映射与状态裁定见 [融合核对表](融合核对表.md)。本次为语义级重写：研究背景、观测契约、符号与证据边界只保留一处完整说明；V5.1—V5.3 的重复台账合并为单一记录；重复的交接说明、自检清单与「下一授权步骤」已删除，其仍然有效的内容并入 5.1 与 5.3；已被后续裁决取代的计划退出行动清单。

相关文档：[冻结 DINOv2 与 SymEOOD 检测融合](冻结DINOv2与SymEOOD检测融合.md)、[Webots 单目深度估计](Webots单目深度估计.md)、[研究文档入口](README.md)。
