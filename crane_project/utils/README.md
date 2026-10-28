# crane_project/utils：用途导航与清理记录

整理日期：2026-10-28。原Git跟踪文件155份，本次删除7份（{'.py': 7}），其余文件内容和文件名保持。额外删除可再生成的`__pycache__`／`.pyc`及`.DS_Store`，不另存一套旧代码。

## 当前核心依赖

| 用途 | 文件 |
| --- | --- |
| midpoint几何与空间先验 | `port_geometry_midpoint_v1.py`、`port_geometry_midpoint_sigma_v1.py`及原解码依赖 |
| 原共享尺寸头／向量解码 | `port_size_qalign_v1.py`、`port_size_local_vectors_v1.py` |
| 小论文双分支尺寸头 | `port_size_pair_specific_v1.py` |
| 深度与相机几何 | `port_midpoint_depth_v1.py`及其几何工具；原标定不改 |
| 训练复现记录 | `symeood_repro_trace_v1.py` |
| 独立功能与回归 | 既有可靠性、旧DINO／几何模块及MAL/USG底层数学代码，在仍被调用／测试／冻结清单使用时保留 |

不能按文件名中的“旧版本”判断可删：当前头仍复用历史几何、支持区域和尺寸交付工具。保留原路径和原字节，防止破坏来源SHA；本次没有合并或改写数学实现。

## 清理边界

保留当前110→midpoint→共享／双分支LVSR的冻结来源，保留仍被项目代码、现有测试、完整Shell入口和相关协议调用的依赖。专项入口退出当前流程且不在这些闭包中时才移除。未仅凭文件日期或名称批量删除。

历史执行收据、旧协议和文档中的路径继续表示当时版本，不通过修改旧SHA伪装成仍可在当前工作区直接执行。需要重放某个旧研究时，从Git恢复该版本的脚本及其协议／配置依赖，或在另一个checkout运行该提交。当前保留列表不代表所有模块重新成为论文方法。

## 本次删除清单与恢复

删除前逐项核对原始字节，全部与提交`030395be620d743b64db3bb94bb9ae8d8eb05828`一致；Git历史没有删除或重写。清单中的SHA是该提交对应文件的完整SHA256。

| 删除文件（相对本目录） | 原因／范围 | 原始SHA256 |
| --- | --- | --- |
| `dark_degradation.py` | 旧暗光／光度诊断，未在当前代码或回归调用闭包中使用 | `d3086ada82c7cea8618a8bc82ca93ca5040446ff352394db9246f48521695cfa` |
| `geometry_refiner_source_gate.py` | 旧宽松Dual-Tower V2.1门控，当前调用闭包未使用 | `e6aac0898db143f2ae6113eca94b179298c12fde4438fe147d5aa482aca61538` |
| `photo_diag.py` | 旧暗光／光度诊断，未在当前代码或回归调用闭包中使用 | `f0d6088c22a1c416a9bc9c14820a971ff98531423d592f36ca6d9ad3654e9fb2` |
| `photo_diag_trainval.py` | 旧暗光／光度诊断，未在当前代码或回归调用闭包中使用 | `61a64df70b334aab91bdfd15e0b0273fc0fc02090ab7a770433b95a08aac4e25` |
| `platform_context_sidecar.py` | 旧独立平台侧路候选，当前模型与回归调用闭包未使用 | `a94f64d0c55cc8a3f8b8fb2eb26751eeb6505b4c40f1616a4e0aaabacab9609c` |
| `port_size_direction_evidence_v1.py` | 旧S037方向诊断适配器，不是当前尺寸头 | `287f0ce6d5d9b6630c0ee8eafa315ed544cc43205a6c5fe1776522720c2863ad` |
| `structured_dark_proxy.py` | 旧暗光／光度诊断，未在当前代码或回归调用闭包中使用 | `2122467927ce31a60abadc16f22666003a6445e4f84659b882ab2277e90ef961` |

在项目根目录查看旧内容：

```bash
git show 030395be620d743b64db3bb94bb9ae8d8eb05828:crane_project/utils/dark_degradation.py
```

仅在确需恢复且目标路径没有新改动时执行：

```bash
git restore --source=030395be620d743b64db3bb94bb9ae8d8eb05828 -- crane_project/utils/dark_degradation.py
```

本次未运行模型、训练、推理、TEST或测距，未同步服务器，未自动提交／推送。实际验证是静态依赖／配置闭包、Git字节及当前冻结来源校验；完整数值测试未在此说明中冒称执行。

## 清理后验证

已完成保留Python语法、JSON解析和静态调用闭包核对；714份保留的原文件与整理前Git字节一致，137项当前链路目标保留，midpoint的127项冻结来源SHA校验通过。使用现有 bundled Python（含NumPy）运行当前midpoint、LVSR共同阶段及固定下游的3组现有unittest，共26项全部通过；这些是接口／身份／方法复用测试，没有模型训练或数据推理。系统Python首次运行因缺NumPy有1项未执行，换用现有环境后已补齐，未安装新依赖。
