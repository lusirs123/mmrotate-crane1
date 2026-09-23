# DINO 语义蒸馏学生与普通 K1 的同口径比较

## 固定设计

- 两组训练均使用原有的 source train / train_sim 数据和 24 epoch 日程；蒸馏学生只在训练时读取冻结 DINO 特征，评估使用学生配置 `crane_symeood_k1_dino_semantic_student_v1.py`。
- 两组均只从 `epoch_16/18/20/22/24.pth` 选择权重。source VAL 共 738 帧；固定 TEST 不用于选权。
- `ckpt_sweep.py` 沿用历史两阶段规则：跨域加权中心召回率距离候选最优值不超过 0.005，最长连续漏检不超过 5 帧；合格候选按 `0.35*TDR + 0.25*R_center + 0.20*sim/ACI + 0.20*(1-min(sim/A-RMSE,90)/90)` 排序。跨域权重为 sim 0.7、real 0.3。没有合格候选时沿用旧脚本的两级 fallback，并在结果中显式记录。
- `eval_crane_offline.py` 的 `mode='test'` 用于输出完整时序指标；在 source VAL 阶段，数据来源仍然是 `val/annfiles`，不是固定 TEST。
- 两组均通过相同的 `tools/test.py`、DOTA 转换和当前 `eval_crane_offline.py` 指标版本计算。选择完成后才分别对各自选中权重运行一次固定 TEST。

## 历史结果与修订结果的关系

旧 `work_dirs/crane_symeood_k1/ckpt_sweep/sweep_results.json` 保持原样。它来自指标修复前的离线评估器；不能与当前评估器生成的蒸馏结果直接比较。使用该文件保存的普通 K1 五组 VAL 预测在本地以当前评估器重算时，`epoch_24` 的 `sim/A-RMSE(deg)` 从旧报告的 1.6529 变为 9.0415，`real/mean_RIoU` 从 0.9026 变为 0.769。相同五组预测在当前指标版本和原选权公式下，初步重选为 `epoch_20`。这说明旧 `epoch_24` 是历史选择，不应被悄悄改名为当前指标版本下的选择。

服务器必须在独立 `ckpt_sweep_metric_v2` 目录生成两组新的选权和 TEST 报告；保留旧结果。普通 K1 的 VAL 预测可从历史 `ckpt_sweep` 缓存重算指标，蒸馏学生则需运行五个候选权重的 VAL 推理。两组报告都要核对 checkpoint、VAL 标注、预测和指标协议身份。若普通 K1 缺少候选 checkpoint 或缓存预测，应停止比较并补齐来源，不用旧指标填补。

## 解释边界

同口径 TEST 对照必须比较当前重新选择的普通 K1 与蒸馏学生。历史冻结 `epoch_24` 的既有结果仍可作为历史参考，但它使用旧指标版本选权，不能与新选权协议混成同一组公平对照。VAL 和 TEST 的数据角色应分别标注。只有 TEST 计算完成后，才能判断蒸馏是否改善检测表现；训练损失下降和训练钩子的 `save_best` 不能代替该结论。
