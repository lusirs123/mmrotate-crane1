# 可同步的AP结果与固定VAL／TEST入口

本目录不受项目`.gitignore`的`work_dirs`规则影响，是论文写作／报告核对和TEST AP的可同步证据入口。2026-10-10按用户要求从本地原结果目录逐字节复制9份必要JSON，共约1.6MB；不重新计算、不修改数值、不删除历史原件。复制来源和SHA见`sync_manifest.json`。

- `val_cpu/`：完整VAL CPU参考AP、PR详情、当时输出／源码artifact。
- `val_native/`：已核验服务器原生VAL报告，以及本机收件检查／清单；服务器PR／源码清单尚未回传，这两份本机文件不冒充服务器artifact。
- `test_cpu/`：固定已暴露TEST CPU参考AP、PR详情、当时输出／源码artifact；服务器原生TEST核对尚未完成。

TEST脚本默认从`val_native/ap_report.json`读取固定原生VAL门槛，报告SHA保持`6e682f64824fe0b68bbe6b49276ebf7d147fe9891a70b0fe3d7f7757b4bc7d34`。代码在`crane_project/tools/`、测试在`tests/`、运行说明在`docs/detection/`，均可同步。

保存报告里的旧输出／源码SHA是当时执行快照，保留原值；本次只修改TEST脚本默认VAL报告路径，未改AP算法。不要用改写历史artifact的方式伪装成重新运行后的报告。服务器预测输入和运行输出仍可以位于`work_dirs/`，它们由服务器已有数据和命令产生，不需要将本机`work_dirs/`整体同步。

## A061：补齐可同步的TEST原始输入

`test_source/`新增AP必需的五份原封存文件，逐字节复制共4,624,450字节，来源与SHA见其`bundle_manifest.json`。保存1440帧标准尺度的GT／B／midpoint／原分数和空输出，并保留原completion、protocol、VAL选权及artifacts；不包含图片、权重或新推理。原artifacts还列有非AP必需文件，本次未复制，不能据此称原结果目录全部归档。前述sync_manifest仅记录A060的9份报告副本，保持历史内容。

TEST默认eval-dir现为本目录下的test_source，服务器命令见[运行说明](../固定TEST_AP补充评价与运行说明_20261010.md)。这解决服务器旧work_dirs输入目录缺失，并保留显式指定完整归档目录的能力；不自动搜索或选择其它候选。须同步test_source中的test_rows.jsonl，只有汇总AP不能重新进行原生核对。原标注仍读取服务器既有test/annfiles。
