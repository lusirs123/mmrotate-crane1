# SymEOOD与midpoint：文件保留与清理核查

日期：2026-10-22。范围：当前本机symEOOD工作区，优先核对第一项SymEOOD与第二项midpoint。操作仅盘点和形成清单，未删除、迁移或下载文件，未连接服务器、加载模型或新增实验。原有工作区在本轮开始时Git状态干净；本轮仅新增本核查MD和不含文件正文副本的JSON索引。

## 1. 核查结论

不要把“旧方案名称”“未被首页引用”“文件字节相同”单独作为删除依据。当前正式midpoint入口依赖多份旧几何工具；历史结果的协议、来源、逐帧输出、选权和核验收据共同构成证据链。先保留整条证据链，再考虑副本合并。

名称匹配及四组主实验范围共528个文件，309.17 MiB；这是局部盘点，不是整个仓库大小，也不是全部可清理空间。自动扫描发现20组内容一致文本，理论重复上限1.73 MiB，尚未认定可删除。

第一批低风险候选为84份有源码的Python字节码及4份.DS_Store，共2.32 MiB，均未被Git跟踪。精确路径和字节数见[JSON清单](SymEOOD与midpoint_文件保留与清理核查_20261022.json)的`regenerable_candidates`；本轮未删除。

## 2. 应保留的文档入口

| 用途 | 文件 |
|---|---|
| SymEOOD方法与结果 | [SymEOOD几何一致性训练_方法与结果核心记录_20261021.md](../../docs/detection/SymEOOD几何一致性训练_方法与结果核心记录_20261021.md) |
| 历史训练身份索引 | [SymEOOD历史训练文件索引_20261021.json](../../docs/detection/SymEOOD历史训练文件索引_20261021.json) |
| 数据与四格整体对照 | [港口新数据集与EOOD_SymEOOD尺度增强实验总记录.md](../../docs/detection/港口新数据集与EOOD_SymEOOD尺度增强实验总记录.md) |
| midpoint完整方法与版本历史 | [旋转ROI四边中点几何修正_midpoint_v1_方法与结果_20261003.md](../../docs/detection/旋转ROI四边中点几何修正_midpoint_v1_方法与结果_20261003.md) |
| midpoint同容量机制对照 | [midpoint_readout_v1_机制对照结果_20261012.md](../../docs/detection/midpoint_readout_v1_机制对照结果_20261012.md) |
| 原生AP协议与结果 | [AP评价口径与运行说明_20261010.md](../../docs/detection/AP评价口径与运行说明_20261010.md) |
| 几何过程与历史根因证据 | [geometry_precision_handoff_20261001.md](../../docs/geometry_precision_handoff_20261001.md) |
| 几何执行与时序记录 | [geometry_precision_activity_20261005.md](../../docs/geometry_precision_activity_20261005.md) |

前六项是方法/结果主入口；后两项保留执行过程、失败路线及历史合同。它们不因篇幅长或结论较旧而自动删除。第三项LVSR、可靠性、深度及原始数据材料属于共享/其他研究范围，本轮不列为无用文件。

## 3. 应保留的模型、配置和运行依赖

- 第一项：`mmrotate/models/detectors/{eood,sym_eood_detector}.py`、`mmrotate/models/dense_heads/{eood_head,rotated_eood_head,sym_eood_head}.py`、对称KLD/SymNFL及POLA/SymPOLA、ATSS辅助训练和注册文件。历史版本从统一索引的Git提交读取。
- 配置：四组`crane_project/configs/crane_{eood,symeood}_k1_port_day2night*_v1.py`及其继承基础配置；不能只保留最末层配置。`tools/train.py`、分布式入口、`ckpt_sweep.py`、`eval_crane_offline.py`及数据集/增强/框解码等基础运行代码保留。
- midpoint读出：`crane_project/utils/port_geometry_midpoint_v1.py`、`port_geometry_midpoint_sigma_v1.py`、`port_geometry_midpoint_readout_v1.py`。
- 正式训练：`train_port_geometry_midpoint_formal_v1.py`、`train_port_geometry_midpoint_sigma_v1.py`、`train_port_geometry_midpoint_readout_v1.py`。
- 正式评价：`eval_port_geometry_midpoint_formal_v1.py`、`eval_port_geometry_midpoint_sigma15_v1.py`、`eval_port_geometry_midpoint_readout_test_v1.py`。旧`eval_port_geometry_midpoint_v1_test.py`仍被正式工具导入，不能因名称含旧v1就删除。
- 每个入口对应`*_protocol.json`、`*_sources.json`及测试夹具保留。全部`mmrotate/`、框架依赖和原始数据不按本次局部导入图裁剪。

从五个正式入口静态跟踪得到28份本地crane_project Python依赖，历史midpoint来源清单涉及115个不同文件路径。该结果是保留的最低集合，不是完整运行环境依赖证明。

特别不能误删的历史工具包括：`preflight_port_center_size_v1.py`、`preflight_port_shape_e_v1.py`、`preflight_port_shape_e_h_v1.py`、`diagnose_port_center_size_d_v1.py`、`diagnose_port_shape_e_h_train_gradients_v1.py`和`preflight_port_structure_reliability_v1.py`。虽然名称属于旧路线，当前midpoint工具仍通过导入复用它们。LVSR尺寸路线还通过edge_residual工具使用原midpoint实现。

## 4. 应保留的训练与评价产物

| 材料 | 保留原因 |
|---|---|
| 四组`work_dirs/crane_{eood,symeood}_k1_port_day2night*_v1/` | 20份日志、展开配置、VAL选权和固定TEST摘要刚归位，是第一项整体对照的原始依据 |
| `work_dirs/port_geometry_midpoint_sigma15_v1_results_20261005_095254/` | 含正式σ1、σ0.5/1.5的24轮结果、选择、逐帧输出、在线VAL/TEST及缓存身份，约241.16 MiB；不能整体判为重复 |
| `docs/detection/midpoint_readout_fixed_test_evidence_20261012/{train,val,test}/` | 均匀/高斯同容量对照，含24轮选择、TEST前VAL门槛和完整TEST；也是测试夹具使用的证据路径 |
| `docs/detection/midpoint_readout_evidence_20261011/` | 已精简的零更新工程预检及核验来源，不冒充正式性能结果；不是第二份正式消融 |
| `docs/detection/ap_evidence_20261010/` | 标准AP、封存Top-1预测、原生报告和CPU核验依据 |
| `work_dirs/port_detection_runtime_v1/` | 四组同硬件耗时、原始timings与收据，支持运行开销的论文描述 |
| 权重、ROI分片、缓存清单与来源清单 | 选中版本及公平对照的复现依据；不因本地仅有清单就删除该清单 |

应保留checkpoint身份：B ep24、当前M σ1.5/ep03、旧σ1/ep23、均匀先验ep04，以及相关训练选权记录。当前指定B/M/σ1权重路径本机不存在，均匀头本地证据README也明确未归档权重；这些是本地复现材料缺口，不是删除理由，服务器是否仍完整需另行只读核验。

原ROI目录目前只有`cache_manifest.json`，所列`train_s1.pt`、`train_s05.pt`、`val_s1.pt`在该本地目录均不存在；本轮未搜索或下载服务器缓存，不能宣布本地可重训。

## 5. 已发现重复及处理边界

| 重复类型 | 例子 | 当前决定 |
|---|---|---|
| 当前协议与运行快照一致 | tools中的protocol与VAL/TEST目录中的protocol | 保留独立运行身份；直接删快照会影响artifact/门槛关联 |
| 选中轮摘要与selected_val_compare一致 | σ1.5的val_epoch_03.json与selected_val_compare.json | 可考虑以后通过明确映射合并，但先核对消费者与manifest |
| 旧σ1.5摘要与新消融基线重放一致 | 原val_compare与高斯gaussian_summary | 内容相同也是重放一致性证据，不将两次运行合并成同一身份 |
| local/server重复诊断表 | midpoint simple disagreement下四组/误差表 | 属可靠性下游范围；本轮只标记，不删除整目录 |

本次20组重复均是JSON/CSV/MD内容比较；没有通过JSON字段重排判断语义重复，也没有将相似文件名视为相同。源码和结果还可能受其他研究脚本引用，未完成消费者核对前不直接删除。

## 6. 来源身份检查与不确定项

对第一项统一索引的28个当前文件逐份检查：存在且SHA匹配28个。历史Git版本关系沿用已核验的索引，本轮未改写历史证据。

历史midpoint来源清单共1005条文件比对，950条与当前字节匹配、55条不同，未发现该范围路径缺失。55条是5份当前检测源文件在11份历史清单中的重复记录；固定TEST合同已经披露历史5文件差异，不是55个待删除文件或本轮新增故障。清理时不得为消除差异而重写历史SHA。

本轮只完成名称范围盘点、静态导入、来源清单检查及内容重复比较；未穷尽动态注册、全仓库引用、数据真值或外部环境。没有引用只能作为下一步人工检查线索，不能作为自动删除依据。

## 7. 建议执行顺序

1. 先以JSON列出的84份字节码和4份.DS_Store为第一批清理候选；执行前再次确认源码存在、文件未变化及无新增Git跟踪。
2. 再逐组核对20组重复的引用和运行索引。只有明确了保留位置及迁移映射，才合并冗余副本；不先删后补。
3. 旧方案代码和文档留待单独依赖审查。当前没有足够依据删除任何完整实验目录、24轮结果或训练入口。

本清单不授权后台自动删除。第一步的“先查看”已经完成，后续按具体候选范围执行，不把“清理无用文件”扩展为删除第三研究点、可靠性、深度或原始数据。
