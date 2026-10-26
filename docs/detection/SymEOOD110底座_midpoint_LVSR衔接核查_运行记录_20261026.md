# 110—midpoint—LVSR核查运行记录

日期：2026-10-26。任务范围：核查与记录，不训练。本文件与[核心核查及合同草案](SymEOOD110底座_midpoint_LVSR衔接核查与最小重训合同_20261026.md)配套。

## R001：工作区与历史证据恢复

本机仓库`/Users/mac/Documents/paper/symEOOD`，HEAD为`30b25eb`。开始时已有三份SymEOOD核心／分类改进文档修改、检测底座合同与最终方法规范未跟踪；后续还出现20份DINO／USG配置删除。本轮未重置、回滚、删除、覆盖、暂存或提交这些改动；来源不替用户认定。

复核最近110／111配对结果和110采用合同，及midpoint formal／sigma源码、缓存清单、LVSR共同／延续源码与第三研究点记录。最近110固定ep20，SHA前缀`a4a53af6`，不改用旧八组110。

## R002：服务器只读底座核查

服务器`omnisky-codex`，项目`/media/omnisky/personal_files/ljj/symEOOD`，解释器为既有`mmrotljj/bin/python`。未执行Git拉取／重置，未改变服务器源码、配置、权重或旧缓存。

新增独立[CPU审查辅助脚本](../../crane_project/tools/audit_symeood_110_downstream_identity_v1.py)，通过SSH标准输入运行，设置空CUDA可见设备。只在新audit目录保存收据。该脚本不替换训练器或方法代码。

- 第一次在目标权重读取前发现文件不存在，停止；保留新建的v1空目录，不覆盖历史目录。
- 第二次完善缺失文件的诊断分支，写入新v2目录；审查自身完成，但状态是`BLOCKED_FIXED_110_CHECKPOINT_MISSING`，不是模型加载验收通过。
- 配置SHA与协议SHA匹配；328项源码与配对训练锁一致。解析实际配置确认A开启、SymKLD回归和分类几何加权关闭。
- 原ep20缺失，latest为断链；严格加载未尝试，缺失／多余键未知。未构建模型、未前向、未创建优化器、未读取图像或缓存张量；CUDA未初始化。
- 历史B、midpoint、A20和LVSR-C6权重均存在并完成文件SHA核对。
- 最近110的VAL与TEST预测PKL仍存在且SHA匹配；仅核对文件身份，未重放推理或重新评价、选版。
- 旧B缓存清单SHA、三个张量文件存在性与大小核查完成；没有读取张量或重新计算三个张量SHA，不能宣称完整原始数组核验。

完整收据：[CPU身份与阻断项](../../work_dirs/symeood_110_midpoint_lvsr_linkage_audit_20261026_v2/cpu_identity_load.json)。

## R003：原权重与废纸篓查找

用户表示可能曾要求删除固定110。项目原目录及项目内候选文件未发现原权重；保留旧八组110，不作替代。两份相关归档只检查成员元数据，没有解压：

| 归档 | 成员数 | 权重成员数 |
| --- | --- | --- |
| 配对实验`completion_receipts_20261026.tar.gz` | 11,830 | 0 |
| `_geometry_archive/closed_outputs_20261005_103946_70187e2eb29c.tar.gz` | 23,682 | 0 |

按用户追问继续只读检查Linux废纸篓：

- `/home/omnisky/.local/share/Trash`存在，info记录0，无目标。
- `/media/omnisky/personal_files/.Trash-1000`存在，扫描6,632条trashinfo，未找到该项目或目标权重。另检查了项目／父目录整体被移入废纸篓的路径记录，无相关候选。
- `/media/omnisky/personal_files/.Trash/1000`不存在。

Linux命令行`rm`等通常绕过废纸篓；当前未查到当时具体删除命令，不能据此断言删除方式。没有运行磁盘恢复、其他用户目录扫描或备份服务操作，也没有恢复／删除任何文件。

收据：[归档检索](../../work_dirs/symeood_110_midpoint_lvsr_linkage_audit_20261026_v2/archive_search.json)、[废纸篓检索摘要](../../work_dirs/symeood_110_midpoint_lvsr_linkage_audit_20261026_v2/trash_search.json)。本地只下载JSON收据，未新建或下载压缩包。

## R004：下游源码与预算核查

本机逐项计算历史来源清单与当前文件SHA：midpoint formal 69项／5项不同，sigma 74项／5项不同；LVSR共同65项／0不同，延续74项／0不同。五项差异为检测器、检测头、分配器与两种损失源码；这不是功能等价检查。完整差异见[来源比较](../../work_dirs/symeood_110_midpoint_lvsr_linkage_audit_20261026_v2/source_dependency_comparison.json)。

追踪发现旧入口绑定B／M权重、缓存、历史样本计数及基线数字；不能只改权重参数或去掉检查。新方案应复用计算函数并新增110身份入口，在工程验收后依次生成新缓存、训练新midpoint、固定版本、准备新LVSR共同与延续阶段。

本轮仅提出有条件预算：midpoint24轮、LVSR共同24轮、原损失延续6轮，共54头轮，检测器0轮；精确更新计数依赖新缓存。该预算不涵盖丢失底座的重新训练，也不表示后续训练已获启动授权。

## R005：交付与未完成事项

汇总：[summary.json](../../work_dirs/symeood_110_midpoint_lvsr_linkage_audit_20261026_v2/summary.json)；工作区快照见同目录`workspace_snapshot.json`。本轮新增核查脚本、两份记录及收据；未修改检测器／midpoint／LVSR训练与方法实现。

未完成项明确保留：原110权重恢复及strict load、真实新缓存坐标／支持／计数验收、缓存与在线一致性、有限GPU更新及恢复、资源测量、所有新链路训练与VAL／TEST。不能把源码／配置核对替代这些验收。

下一步先恢复原文件并核SHA；无备份时，需要用户另行决定新的底座身份。当前不启动检测器重训，也不启动两个小头。本地材料未提交、未推送。
