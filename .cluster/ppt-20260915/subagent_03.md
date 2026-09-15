# subagent_03 复核报告：模板继承与规范检查

- 复核对象：`deliverables/group_meeting_20260915/组会汇报_20260915.pptx`（sha256 `1d6b0e3c…ddddbd`）、同名 `.pdf`（sha256 `fbbce060…1be0f6`，13 页 / 960×540pt）
- 模板原件（只读基准）：`build/template_source.pptx`（sha256 `2a5fb4d1…f0c8b24`）
- 规格：`build/SPEC.md`、`build/SPEC_补充_第二轮修正20260915.md`
- 方法：`exec`（shasum / unzip / diff / grep）+ python3 3.9 / python-pptx 1.0.2 解析；本报告未修改任何被检查文件。
- 唯一写入文件：本文件。

## 结论总表

| # | 检查项 | 结论 |
|---|---|---|
| 1 | 模板未被修改（sha256 + mtime） | 合规 |
| 2 | 主题/母版/页面尺寸/配色未被改动 | 合规 |
| 3 | 字体与字号（新增/变化项） | 合规 |
| 4 | 占位符残留 | 合规 |
| 5 | 备注覆盖（13 页 + 来源行） | 合规 |
| 6 | 隐藏文字（AlternateContent / show=0） | 合规 |
| 7 | 打包完整性（Content_Types / slides / sldId 三数一致） | 合规 |
| 8 | 仓库只读性 + 脚本写入落点 | 合规 |

**违规项：0 项。**（非违规观察项见文末。）

---

## 1. 模板未被修改  —— 合规

命令与输出：
```
$ shasum -a 256 build/template_source.pptx
2a5fb4d19e2343820221bb4855b287096bc2baa2e16238dc903bb87c3f0c8b24  build/template_source.pptx
$ shasum -a 256 "/Users/mac/.../.openclaw-attachments/20260915-110227-d97b49d0-60f-路佳金-2.pptx"
2a5fb4d19e2343820221bb4855b287096bc2baa2e16238dc903bb87c3f0c8b24  …路佳金-2.pptx
```
- 两边 sha256 **逐字节一致** → 上传原件未被改动。
- `stat -f`：`template_source.pptx  mtime=2026-09-15 11:08:30 size=27208814`；上传原件 `mtime=2026-09-15 11:02:27`。
- 说明：模板 `mtime=11:08:30`（= 首次拷入 build/ 的时刻，与 build_log 起始同刻），此后**无再写**（成品 `build_pptx.py` 12:17 才运行）。按"构建期间模板未被改动"判定为合规；若要求严格早于 `11:08:00`，此为边界情形，但 `sha256` 一致已独立证明其内容未被修改。

## 2. 主题/母版/页面尺寸/配色未被改动  —— 合规

解包后比对（python-pptx 1.0.2 写出时的唯一系统性差异是 XML 声明引号风格 `"`→`'`）。

| 部件 | 结果 |
|---|---|
| `ppt/theme/theme1.xml` | **sha256 完全一致** `d13fabecc876…c9df641a`（逐字节相同） |
| `ppt/theme/theme2.xml` | sha256 完全一致（同上） |
| `ppt/tableStyles.xml` | sha256 完全一致 `bd44f4704e41…8567d40` |
| `ppt/slideMasters/slideMaster1.xml` | 内容一致，**仅第 1 行 XML 声明** `<?xml version="1.0"…?>` → `<?xml version='1.0'…?>` |
| `ppt/slideLayouts/slideLayout1..13.xml` | 内容一致，**仅第 1 行 XML 声明**不同 |
| `ppt/notesMasters/notesMaster1.xml` | 内容一致，**仅第 1 行 XML 声明**不同 |
| `ppt/presentation.xml` `sldSz` | 两边均为 `<p:sldSz cx="12192000" cy="6858000"/>`（13.333×7.5 in，16:9） |

命令（要点）：
```
$ shasum -a 256  TPL/ppt/theme/theme1.xml  OUT/ppt/theme/theme1.xml
$ diff TPL/ppt/slideMasters/slideMaster1.xml OUT/ppt/slideMasters/slideMaster1.xml
1c1
< <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
---
> <?xml version='1.0' encoding='UTF-8' standalone='yes'?>
$ grep -o '<p:sldSz[^/]*/>' presentation.xml     # 两边输出相同
```
（对 master / 13 个 layout / notesMaster 逐一 `tail -n +2` 去首行后比较 → **CONTENT-SAME**。）
主题与配色定义在 `theme1.xml`，sha 完全一致 ⇒ 配色未被改动；母版/版式几何亦未改动。

## 3. 字体与字号  —— 合规

统计成品全部 slide XML 的 `typeface` 与 `sz` 取值并与模板比对。

- **成品 slide 中出现的 `typeface`**：`+mj-lt, +mn-cs, +mn-ea, +mn-lt, SimSun, Times New Roman, 宋体, 微软雅黑, 等线`
- **模板独有**（均属被删页）：`Arial, Cambria Math, DengXian, ElsevierSans, Google Sans Text` → 无新增字体
- **`sz` 取值**：成品 `1200,1275,1350,1400,1500,1575,1600,1725,1800,1875,1950,2000,2250,2325,2400,2800,3750,4000,4400,6600`；模板独有 `1125,1700`（属被删页）。
- **相对模板"新增/变化"的取值（逐 slide 部件比对）**：

| 成品页 | 部件 | 变化 | 与预期一致？ |
|---|---|---|---|
| P11 | slide10.xml | `sz` **-2000 / +1600**（20pt→16pt）；`typeface` **+等线**、-Cambria Math；`sz` +1800 | ✅ P11 三标签 20→16pt；新增文本框用「等线 18pt」 |
| P12 | slide13.xml | `typeface` -Times New Roman（替换长文所致，无新增） | ✅ |
| P10 | slide4.xml | `typeface` -Cambria Math（替换长文所致） | ✅ 无新增 |
| P8 | slide8.xml | `sz` -1125 / -1800（删除"滑窗示意"整组形状所致） | ✅ 属删除，非改字号 |
| P5 | slide9.xml | 仅文本/几何（`矩形 31` = `DINO 语义分支`），无字体字号变化 | ✅ P5 标签文本变化 |

- 成品未出现任何**模板中不存在**的新 `typeface` 或新 `sz` 取值（`typefaces only in OUT = []`、`sz only in OUT = []`）。新文本框所用「宋体 / 等线」在模板中本已存在，故不计为"新增字体"。

命令要点：
```
typefaces only in OUT slides: []
sz only in OUT slides: []
slide10.xml: typeface +['等线'] -['Cambria Math']; sz +['1600','1800'] -['2000']
slide13.xml: typeface +[] -['Times New Roman']
```
实测文本核对：P5 `矩形 31` = `DINO 语义分支`；P11 三标签 = `第一步：冻结与评价 / 第二步：小目标与数据补齐 / 第三步：泛化与收尾`；P12 依次为 问题一/问题二/问题三（逐字符合 SPEC 补充）。

## 4. 占位符残留  —— 合规

对成品 zip 内**全部 XML**扫描 `点击此处 / Click to add / xxx / lorem / TODO / [insert / 待补充`，全部 NONE；PDF 文本层复核同样为 0。

```
=== OUT ===
  '点击此处': NONE   'Click to add': NONE   'xxx': NONE
  'lorem': NONE      'TODO': NONE           '[insert': NONE   '待补充': NONE
```
（模板原件本身亦无这些串，说明不是"模板自带"。）

## 5. 备注覆盖  —— 合规

用 python-pptx 重新打开，逐页读取 `notes_slide`：**13/13 页均有 speaker notes，且每页备注都含 `来源：` 行**，无缺项。

```
P01 notes_len=157 来源=YES   P02 notes_len= 94 来源=YES   P03 notes_len=215 来源=YES
P04 notes_len= 80 来源=YES   P05 notes_len=318 来源=YES   P06 notes_len=229 来源=YES
P07 notes_len=251 来源=YES   P08 notes_len=300 来源=YES   P09 notes_len=546 来源=YES
P10 notes_len=352 来源=YES   P11 notes_len=279 来源=YES   P12 notes_len=265 来源=YES
P13 notes_len= 71 来源=YES
```
（`[Content_Types].xml` 中 notesSlide Override = 13，件数 = 13，无悬挂备注页。）

## 6. 隐藏文字（AlternateContent / show=0）  —— 合规

- 成品全部 slide XML 中 `mc:AlternateContent` 块数 = **0**（模板 P11/P12 原有、被 build 显式删除，见 build_log「P11 删除 mc:AlternateContent 内陈旧文本框…」）⇒ 不存在 python-pptx 看不见但 PowerPoint 会渲染的文字。
- 全包扫描 `show="0"`：**0 处** ⇒ 没有被隐藏的幻灯片。

```
total AlternateContent blocks: 0
hidden markers: 0
```

## 7. 打包完整性  —— 合规

三数一致，均为 **13**：
- `[Content_Types].xml` 的 slide Override = 13（`slide1..slide10, slide13, slide18, slide19`）
- `ppt/slides/slideN.xml` 件数 = 13（同名）
- `ppt/presentation.xml` 的 `<p:sldId ` 数 = 13

并用 python-pptx 重新打开，遍历 13 页、共 180 个形状（含表格逐格）**无异常**：
```
Reopened OK. slides = 13
P01:6 P02:3 P03:19 P04:3 P05:34 P06:40 P07:9 P08:31 P09:5 P10:5 P11:12 P12:7 P13:6  → total 180
```
**页序核对（成品新页 ← 模板原页）**：`1,2,6,5,9,7,18,8,3,4,10,13,19` —— 与 SPEC 第 1 节表格**完全一致**。
（原始关系 `rId3→slide1, rId5→slide2, rId9→slide6, rId8→slide5, rId12→slide9, rId10→slide7, rId21→slide18, rId11→slide8, rId6→slide3, rId7→slide4, rId13→slide10, rId16→slide13, rId22→slide19`；`slide11/12/14/15/16/17` 已删。）
PDF 为 13 页、960×540pt，与 16:9 版面一致。

## 8. 仓库只读性 + 脚本写入落点  —— 合规

```
$ git status --short        # 仅出现本次新增文件，全部位于 deliverables/group_meeting_20260915/ 下
?? deliverables/group_meeting_20260915/build/...
?? deliverables/group_meeting_20260915/组会汇报_20260915.pptx
?? deliverables/group_meeting_20260915/组会汇报_20260915.pdf
```
- **无任何被跟踪文件被修改**（`git status --short | grep '^[ MARD]'` → NONE）；未出现 `crane_project/ mmrotate/ docs/ plan/ tests/` 等目录的改动。
- `template_source.pptx` 为已跟踪文件且状态未变（无 ` M`）。
- 脚本路径常量审计：所有写操作都落在 `deliverables/group_meeting_20260915/`：
```
build_pptx.py:29   OUT_DIR = os.path.dirname(HERE)      # = deliverables/group_meeting_20260915
build_pptx.py:30   OUT     = os.path.join(OUT_DIR,'组会汇报_20260915.pptx')
export_notes_md.py / verify_pptx.py / write_evidence.py  # 均以 OUT_DIR 为基准写 .md/.txt/pptx
export_pdf.sh:15   DIR="/Users/mac/.../deliverables/group_meeting_20260915"   (PDF 落点)
make_qa_png.sh:5   DIR=同上                    OUT="$DIR/build/qa_png"          (PNG 落点)
```
唯一出现的仓库外/系统路径仅为只读工具 `/opt/homebrew/bin/pdftoppm`；**未发现任何把写入落到 deliverables 之外的常量或调用**（无 `os.chdir`、无指向仓库其他目录的写路径）。

---

## 违规项清单

**无。** 8 项检查全部合规，未发现模板被改动、母版/主题/尺寸/配色被改动、非法字体字号、占位符残留、备注缺项、隐藏文字/隐藏页、打包不一致或越界写入。

## 非违规观察项（供主 agent 参考，不构成违规）

1. **元数据陈旧**：`docProps/app.xml` 仍记录 `<Slides>19</Slides>` 与 51 个 lpstr（沿用模板元数据，未随删页更新）。属标题栏/属性层面的非渲染元数据，不影响版面与内容，也非本规格的约束项。
2. **XML 声明引号风格**：母版/版式/备注母版相对模板仅首行 `"`→`'`（python-pptx 规范化写法），语义等价、无内容差异。
3. **画布外模板遗留形状被保留**：如 P6/P8 的 `矩形 43..46`（`DEP 深度估算误差传播率` 等指标定义表）位于 `y≈-2.2..-0.4 in`、P5 `矩形 59`（y=8.34）、P11 `肘形连接符 18`（y<0）。这些是模板原有的画布外形状，SPEC 允许保留，**PDF 亦未渲染**（PDF 文本层无 `DEP`），与"模板继承"一致。
4. **安全护栏记录**：本次审计期间，一条用于清理临时解包目录的 `rm -rf …` 命令被 AutoClaw Safety Guard 判定为高危删除并拒绝（`APPROVAL_TIMEOUT_DENIED`）。随即放弃一切删除操作，改用新目录继续，仅做只读检查；该事件与被复核成品无关，不涉及任何写入。
5. **未逐字节比对 slide XML**：因成品对保留页做了文字替换，slide XML 必然不同于模板，故按"字体/字号取值的新增与消失 + 逐页文本"进行核对，而非字节比对；theme 则做了 sha256 逐字节比对。

## 无法核实项

- 无重大无法核实项。唯一受限之处：#3 只做"取值集合"层面的字体/字号核对（未对每个 run 逐一断言其来源于哪个形状），如需 run 级别的字体归属可再做一次逐 run 导出。
