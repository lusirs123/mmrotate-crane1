# 独立四角点标注材料：尚未进入TRAIN

images/是20张原分辨率PNG；annotations/是空白JSON，可在标注器中指定该输出目录。同一物理对象、同一参考面须先统一定义；不要直接默认外侧抓斗轮廓或固定比例框就是深度参考面。

仅在四角点都能独立识别时，以polygon类型、central_structure_reference标签，按顺／逆时针依次标4点。坐标为原图像素。禁止用预测框、旧固定比例、隐蔽角点推算或自动传播结果代替独立人工复核。

逐张填写object_definition（具体部件及四个角点身份）、measurement_basis=manual_visible_four_corners；确认后flags.independent_reviewed=true。遮挡或模糊不能确定时保留shapes=[]，置unmeasurable=true、independent_reviewed=true，并在review_note写原因。未处理的JSON保持PENDING。

标注器可能丢弃自定义字段，保存后检查这些字段。保留packet.json、artifacts.json及原PNG不变；仅annotations/JSON允许人工修改。check只检查结构、身份及声明，不验证人工标注真值。

这些是无split的候选观测，不导出DOTA、不覆盖旧GT、不启动训练。角点观测的对边均长也不等同于原生OBB长短边或米制尺寸；后续仍需对象、参考面、标注复核、近重复／事件级角色及模型错误支持检查。容器帧编号不保证物理时间间隔。
