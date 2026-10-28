# 固定 TEST 机制对照必要证据

完整数值、反面结果、运行命令与论文边界见[统一结果文档](../历史归档/第二个创新点_完整研究与运行记录.md#original-02-02)末尾“固定 TEST 对照”。

2026-10-12清理时曾保留22份文件；随后合并训练证据，当前共31份：本说明、运行元数据、服务器最终核对／本机收件记录，双头在线VAL门槛和固定TEST的摘要、原始预测、协议／来源与哈希清单，以及train/内9份必要训练与选权材料。重复CSV、各运行日志、两个退出码和静态报告副本共8份已移入废纸篓，恢复路径／SHA见收件记录local_retention_history；21项测试通过及退出0保留在已核验收件记录中，不冒称本机仍存测试日志。没有权重、ROI缓存、源码副本或全部中间进度；原artifacts仍索引服务器完整输出，不表示当前本地子集齐备。服务器完整原件在`work_dirs/port_geometry_midpoint_readout_test_v1/20261012_gpu2_v1/`。

完成状态：VAL `READOUT_FIXED_ONLINE_VAL_PASS_BEFORE_TEST`、TEST `READOUT_FIXED_TEST_COMPLETE_REVIEW_REQUIRED`、最终退出0。均匀epoch04／高斯σ1.5-epoch03由原VAL固定，TEST无更新／无选权；旧B与高斯TEST输出及摘要复现。中心命中仅用输出帧分母，覆盖与全帧正确覆盖另报。

精简前本机逐项核对18份artifact文件SHA、当前91份来源与权重身份，独立复算15,348项摘要最大差0，逐框几何分解最大差8.6927e−6。没有本机重新推理或权重重载；收件检查见`local_review_receipt.json`，不混同服务器报告。

传输包SHA256：`88b00bf5f44fd050e70c3e34002fc35a8814ed004162616edf41e316fa061b79`。本机与服务器压缩包已删除并核验不存在，原报告／权重保留。单种子、TEST已多次暴露，不能称新的独立泛化确认；不据结果重选参数或模型。

## 单一主目录的读取顺序

1. `test/compare.json`及两头summary：论文机制对照主表。
2. `train/selection.json`、`train/val_readout_compare.json`、`train/completion.json`：证明全部24轮训练与完整VAL选权；5份训练报告／逐帧输入也供本机读取夹具使用。
3. `val/completion.json`及其关联输入：TEST前的在线VAL一致性门槛；不能当作TEST数值。

原formal目录已移除；9份材料迁移时逐份SHA相同，随后仅本机收件记录追加迁移说明；其余8份原报告／metadata字节不变。重复README移入废纸篓，迁移映射见本机receipt的local_directory_merge。历史运行metadata、报告和source快照保留原路径／身份，不改成迁移后重新执行。仅本机夹具读取路径和当前评价来源清单对应SHA调整；已完成报告不重写，旧门槛不能冒充与新来源匹配。服务器原件不动。
