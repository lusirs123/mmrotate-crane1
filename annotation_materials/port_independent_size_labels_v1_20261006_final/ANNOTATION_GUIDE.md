# 2026-10-07合同更正与已保存标注状态（A055当前／A054历史）

**A055完成确认：两帧补勾已保存，当前20/20均independent_reviewed=true、unmeasurable=false。** candidate08_f00386.json／candidate08_f00452.json仅复核标志更新，所有20帧点坐标和原PNG保持；最新逐帧SHA见[annotation_completion.json](review_20261007/annotation_completion.json)，另一对话先读[活动A055](../../docs/geometry_precision_activity_20261005.md#activity-a055)。对象／测量依据仍为空，四点→OBB读出与旧实体角点合同适配仍待完成；人工标注复核标志完成不等于可训练。下面A054的18份已勾／两份未勾及叠加图是补勾前快照，不再作为当前待办。

用户已完成20帧四点标注，明确目标为顶梁主体及长边梯形凸起的包围范围；不标近似正方形抓斗平面，不改为多点轮廓。矩形包围框四角允许在背景上，不能要求都是可见实体角点。原20份JSON／原PNG／packet及artifacts保持，以下旧指南是A050创建时实体四角点合同，与本次包围框声明不等价。

本轮检查：20份均有一个四点polygon，图像身份及有限／界内／循环凸形结构通过；18份已勾independent_reviewed，candidate08_f00386.json／candidate08_f00452.json未勾；20份object_definition和measurement_basis为空。四点并非严格旋转矩形，旧check返回2；不得填写manual_visible_four_corners冒充符合旧合同，也不自动改点或勾标志。需先统一四点→OBB读取与包围方向／范围，再适配测量声明及检查器。完整逐帧报告及20帧叠加图在[review_20261007/annotation_review.json](review_20261007/annotation_review.json)，接续规则见[活动A054](../../docs/geometry_precision_activity_20261005.md#activity-a054)。

固定k偏差需同帧原轴及适用序列k配对、按原轴方向独立测短边；四边形对边均长不是已认可OBB或物理尺寸。20候选仍未分配split，无GT导出／训练；物理参考、事件／近重复及两个下游收益未确认。本指南新说明不重写artifacts的创建时SHA。

---

## A050旧实体四角点指南（历史存档，当前不直接沿用）

# 独立四角点标注材料：尚未进入TRAIN

images/是20张原分辨率PNG；annotations/是空白JSON，可在标注器中指定该输出目录。同一物理对象、同一参考面须先统一定义；不要直接默认外侧抓斗轮廓或固定比例框就是深度参考面。

仅在四角点都能独立识别时，以polygon类型、central_structure_reference标签，按顺／逆时针依次标4点。坐标为原图像素。禁止用预测框、旧固定比例、隐蔽角点推算或自动传播结果代替独立人工复核。

逐张填写object_definition（具体部件及四个角点身份）、measurement_basis=manual_visible_four_corners；确认后flags.independent_reviewed=true。遮挡或模糊不能确定时保留shapes=[]，置unmeasurable=true、independent_reviewed=true，并在review_note写原因。未处理的JSON保持PENDING。

标注器可能丢弃自定义字段，保存后检查这些字段。保留packet.json、artifacts.json及原PNG不变；仅annotations/JSON允许人工修改。check只检查结构、身份及声明，不验证人工标注真值。

这些是无split的候选观测，不导出DOTA、不覆盖旧GT、不启动训练。角点观测的对边均长也不等同于原生OBB长短边或米制尺寸；后续仍需对象、参考面、标注复核、近重复／事件级角色及模型错误支持检查。容器帧编号不保证物理时间间隔。
