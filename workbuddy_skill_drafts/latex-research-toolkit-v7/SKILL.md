---
name: latex-research-toolkit
description: 科研 LaTeX 绘图、排版与论文项目助手。用于 TikZ 配图、编译诊断、PDF 版面检查、参考文献与公式核对、多文件项目处理、期刊模板适配和文档转换。
display_name: "科研 LaTeX 绘图与排版助手"
display_name_en: "Research LaTeX Figure and Layout Assistant"
description_zh: "支持 TikZ 科研示意图绘制与修改、LaTeX 编译诊断、多文件项目处理、PDF 版面检查、参考文献与公式核对、期刊模板适配、Word 表格转 LaTeX，以及 LaTeX 转表单文本或可编辑 Word。"
description_en: "Create and edit TikZ research diagrams, diagnose LaTeX builds, handle multi-file projects, review PDF layouts, check references and equations, adapt journal templates, convert Word tables to LaTeX, and assist with LaTeX-to-form text or editable Word conversion."
version: "0.3.2"
author: "待填写"
examples_zh:
  - "根据论文方法绘制可编辑的 TikZ 框架图。"
  - "检查 LaTeX 编译日志，定位并修复报错。"
  - "检查论文 PDF 的图表、分页和字体。"
examples_en:
  - "Create an editable TikZ framework diagram from a research method."
  - "Diagnose LaTeX build errors from the project log."
  - "Review figure placement, pagination, and fonts in a paper PDF."
---
# 科研 LaTeX 工作台

根据用户指定任务读取对应参考：绘图 `references/figure.md`；编译 `references/build.md`；版面 `references/layout.md`；Word 表格到 LaTeX `references/table.md`；LaTeX 到表单 `references/form.md`；LaTeX 到 Word `references/word.md`；参考文献 `references/bibliography.md`；公式 `references/equations.md`；多文件项目 `references/project.md`；期刊模板 `references/journal.md`。跨任务按需组合。

读取实际文件、项目主文件、配置、字体和依赖，优先复用已有模板。只读审查不自动修改；按用户授权编辑，避免覆盖不相关内容。选择本机可用工具，依赖缺失时报告并给出可选方案，不静默安装。图中数据和几何关系以用户材料为准，示意值不冒充实测。编译成功须以真实结果为准，必要时渲染整页与局部检查。报告交付文件、验证和限制。
