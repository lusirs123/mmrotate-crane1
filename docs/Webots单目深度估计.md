# Webots 单目深度估计

本文是「Webots 单目深度与几何接口」主线的唯一维护入口。几何契约、数据证据清单和早期标定记录统一于此；原始材料见文末[来源索引](#appendix-sources)。

**阅读约定**

- 原文件中的公式、数值、表格、字段与复现入口完整保留；重复记录的同一实验已合并为单一记录。
- 历史材料中的「当前」「最终」「唯一下一步」只在原阶段有效，不能覆盖本文第 1 节的结论。
- 本文档只整合已有文字，不重新计算深度、读取未知序列或验证外部数据。

---

## 1. 研究问题与当前结论

<a id="sec-1"></a>

<a id="sec-1-1"></a>

### 1.1 研究问题与路线

正式路线固定为：

```text
原始图像 OBB
  -> Raw-opt 尺度模型得到相机光轴深度
  -> 原相机坐标系反投影
  -> 固定相机—吊点外参变换
  -> 吊点到抓斗的三维相对向量
  -> 两轴摆角
```

后续深度模块的首要预测量是抓斗顶梁参考中心 `G` 相对相机光心 `C` 的光轴深度：

\[
Z_{CG}^{\mathrm{opt}}={}^{C}G_z .
\]

该定义与小孔相机反投影直接对应；它既不同于相机到抓斗中心的欧氏距离，也不同于钢丝绳悬挂点 `P` 到抓斗参考中心的垂直高度差。深度实验按两层组织：

1. 第一层评价 `Z_{CG}^{\mathrm{opt}}`；
2. 第二层通过固定相机—吊点外参得到 `H_{PG}`、`X_{PG}`、`Y_{PG}` 和摆角。

这样的划分能够分别归因尺度公式误差与安装外参误差。

<a id="sec-1-2"></a>

### 1.2 当前结论

- **当前尺度路线是 Raw-opt**：原图 OBB 与实体相机光轴深度 `Z_c := Z_{CG,opt}` 配对，使用 `q = ln(Z_L/Z_S)` 和 `Z_hat = c Z_S exp(beta q²)`。光轴深度、吊点铅垂距离和欧氏距离不能混用。
- **已记录的组件结果**：Train-03 拟合；Fixed-dev-01 的 GT OBB 下 Raw-opt MAE `0.03417 m`、RMSE `0.04045 m`。**这不是检测 OBB 端到端深度结果。** 正式 native-S14 在 Train-03 检出 `981/981` 帧，但仅 `52/981` 帧的 `q` 在 GT 拟合支持域内，未继续进入 fixed-dev。
- **三维链是有条件的公式接口**：原图中心反投影、相机—吊点变换和摆角推导保留；已修复的标量深度配对不等于完整三维验证。若吊点坐标轴定义为重力方向，相机与吊点**刚性连接本身不足以证明**旋转到重力坐标的矩阵恒定：旋转矩阵仅在「姿态不变」假设成立时可取为常量，姿态变化时必须改用 \(R(t)\) 动态变换（平移向量亦须核对坐标系与时间依赖性）。三种情形与完整条件见 [2.2](#sec-2-2)。检测有效性门、完整三维闭环和部署均未由本组记录证明。
- **Unknown-02 已暴露且只属旧 Plumb/Opt 历史证据**。下文原表中「冻结后的单次未知序列评估」是旧协议职责，不是 Raw-opt 的新未知验证；2971 帧索引包含该历史序列，不表示三条序列均是当前 Raw-opt 正式验证集。
- **早期 Train-01/02** 的幂律参数和自旋审计保留为历史基线，不与 Raw-opt 的参数、坐标和误差拼接。
- **下一研究条件**：若继续深度工作，先在 Train-03 比较并冻结框几何来源与 `depth_valid` 规则，再进入固定开发验证；这是一项未执行条件，不是本文的整理授权。新的 OBB 连续接口也不自动证明 Base V3 已通过 Webots 深度验证。

<a id="sec-1-3"></a>

### 1.3 证据边界

- 全流程不使用 PLC、编码器或现场传感器融合；Webots GPS 与 Supervisor 仅用于仿真真值交叉审计，不是模型输入。
- `calibration_train` 只用于拟合或训练侧机制审计；`fixed_dev`、未知序列泛化与真实部署必须单独报告，不与标定结果混用。
- 真实视频没有独立米制真值，不报告真实域绝对深度或摆角精度。
- 全文若出现同名字母，应按大小写和上下文区分：\(C\) 是相机光心，\(c\) 是 Raw-opt 标定系数，\((c_x,c_y)\) 是相机主点。

---

## 2. 符号、公式与几何链

<a id="sec-2"></a>

<a id="sec-2-1"></a>

### 2.1 统一符号定义

<a id="sec-2-1-1"></a>

#### 2.1.1 点、坐标系与上下标

| 符号 | 定义 |
| --- | --- |
| \(C\) | 实体相机光心（camera center） |
| \(P\) | 钢丝绳悬挂点/摆动铰点（pivot） |
| \(G\) | 抓斗顶梁参考平面中心（grab reference center） |
| \(\{C\}\) | 实体相机坐标系；第三轴为相机光轴正方向 |
| \(\{P\}\) | 吊点坐标系；**其性质必须显式声明**：要么是「以吊点为原点的重力对齐坐标系」（第三轴沿重力向下），要么是「随机构运动的吊点固连坐标系」。两者的分量含义不同，见 2.2 的适用条件 |
| \(\{W\}\) | Webots 世界坐标系，仅用于生成和审计仿真真值 |
| \({}^{A}\mathbf x\) | 向量 \(\mathbf x\) 用坐标系 \(\{A\}\) 表示 |
| 下标 \(CG\)、\(PG\) | 分别表示从 \(C\) 到 \(G\)、从 \(P\) 到 \(G\) 的几何关系 |
| 上标 \(\mathrm{opt}\)、\(\mathrm{plumb}\) | 分别表示实体相机光轴量、虚拟铅垂相机轴向量 |
| \(\widehat{(\cdot)}\) | 模型估计量；无帽符号通常表示定义量或真值 |
| \((\cdot)_{\mathrm{gt}}\) | Webots 独立真值，不是部署输入 |

<a id="sec-2-1-2"></a>

#### 2.1.2 图像 OBB、相机与物理尺寸

| 符号 | 单位 | 定义 |
| --- | ---: | --- |
| \(u_G,v_G\) | px | 原图中抓斗顶梁 OBB 中心；不是相机主点 |
| \(w,h\) | px | 规范化后的 OBB 长边、短边像素长度，满足 \(w\ge h\) |
| \(\gamma\) | rad 或 ° | OBB 长边在图像平面内的方向角；不等于机械自旋角或摆角 |
| \(s\) | — | 检测置信度；不能单独证明框尺寸可靠 |
| \(f_x,f_y\) | px | 相机横、纵焦距 |
| \(c_x,c_y\) | px | 相机主点坐标；与 OBB 中心 \(u_G,v_G\) 不同 |
| \(K\) | — | 由 \(f_x,f_y,c_x,c_y\) 构成的相机内参矩阵 |
| \(L_G,S_G\) | m | 被 OBB 标注的顶梁参考平面真实长边、短边 |
| \(l_{\mathrm{diag}}\) | px | 原图 OBB 对角线，\(\sqrt{w^2+h^2}\) |
| \(l'_{\mathrm{diag}}\) | px | 虚拟铅垂图像 OBB 对角线；撇号表示 plumb 图像，不属于 Raw-opt 主线输入 |

<a id="sec-2-1-3"></a>

#### 2.1.3 深度公式与三维状态

| 符号 | 单位 | 定义 |
| --- | ---: | --- |
| \(s_L,s_S\) | — | 由原图 OBB、方向角和内参得到的长轴、短轴归一化投影尺度 |
| \(Z_L,Z_S\) | m | 由 \(L_G/s_L\)、\(S_G/s_S\) 得到的两项针孔深度候选 |
| \(q\) | — | 长短轴深度不一致量，\(q=\ln(Z_L/Z_S)\)；对检测框长短边比例误差敏感 |
| \(c,\beta\) | — | Raw-opt 的离线冻结标定参数；小写 \(c\) 不是相机光心 \(C\) |
| \(Z_c:=Z_{CG}^{\mathrm{opt}}\) | m | \(G\) 在实体相机坐标系光轴上的深度；不是 \(C\!\text{-}\!G\) 欧氏距离，也不是吊点垂距 |
| \(Z_{CG}^{\mathrm{plumb}}\) | m | \(G\) 在共光心虚拟铅垂相机第三轴上的深度 |
| \(X_{PG},Y_{PG}\) | m | 吊点坐标系中从 \(P\) 到 \(G\) 的两个水平分量 |
| \(H_{PG}\) | m | 从 \(P\) 到 \(G\) 沿重力向下的铅垂分量 |
| \(L_{PG}\) | m | \(P\) 到 \(G\) 的三维欧氏距离，\(\sqrt{X_{PG}^2+Y_{PG}^2+H_{PG}^2}\) |
| \(R_{P\leftarrow C}\) | — | 将相机坐标向量旋转到吊点坐标系的旋转矩阵；**仅在 2.2 情形 1（姿态不变）下可视为常量（固定外参）**，情形 2 须写作 \(R_{P\leftarrow C}(t)\) |
| \({}^{C}\mathbf d_{CP}\) | m | 在相机坐标系表达的有符号向量 \(P-C\)，不是无符号距离 \(\lVert C-P\rVert\) |
| \(\theta_x,\theta_y\) | rad 或 ° | 两个方向的有符号摆角；正负表示方向 |
| \(\theta\) | rad 或 ° | 非负合摆角 |
| \(\psi\) | rad 或 ° | 抓斗相对机械正位绕自身竖直轴的自旋角；一般不等于 \(\gamma\) |

<a id="sec-2-1-4"></a>

#### 2.1.4 评价与协议缩写

| 符号/术语 | 定义 |
| --- | --- |
| GT OBB / oracle | 由 Webots 三维角点投影得到的真值框，用于隔离尺度公式误差 |
| detector OBB | 检测器预测框，用于评价检测误差向深度的传播 |
| RIoU | 预测 OBB 与 GT OBB 的旋转交并比 |
| MAE / RMSE / AbsRel / P95 | 平均绝对误差、均方根误差、平均绝对相对误差、绝对误差第 95 百分位 |
| source-only / source gate | 只用源域训练或选择形成的证据；不等同于目标域验证 |
| calibration-train | 允许拟合公式并设计检测输出有效性规则的序列 |
| fixed target-dev | 冻结后用于选择/诊断的目标开发序列；不得回填训练 |
| unknown sequence | 全部模型、公式和门限冻结后的新序列泛化评价 |
| deployment | 真实运行的有效率、稳定性、时延和资源证据；真实视频无真值时不报告米制精度 |

<a id="sec-2-2"></a>

### 2.2 三个参考点、\(Z_c\) 定义与坐标系适用条件

三个参考点 \(C\)（实体相机光心）、\(P\)（钢丝绳悬挂/摆动铰点）、\(G\)（抓斗顶梁参考平面中心）的完整符号定义见 [2.1.1](#sec-2-1-1)，此处不再重复。

本文唯一保留的 \(Z_c\) 定义为

\[
\boxed{Z_c:=Z_{CG}^{\mathrm{opt}}}
\]

即抓斗参考中心 \(G\) 在**实体相机坐标系光轴**上的深度。它不是吊点到抓斗的铅垂距离，也不是 \(C\) 到 \(G\) 的欧氏距离。另定义

\[
H_{PG}=\text{吊点到抓斗中心的向下铅垂分量},\qquad
L_{PG}=\lVert G-P\rVert_2 ,
\]

\[
H_{PG}=(G-P)^\mathsf T\mathbf e_{\mathrm{down}} .
\]

**坐标系性质与适用条件（关键）**

\(H_{PG}\)、\(X_{PG}\)、\(Y_{PG}\) 与摆角的物理含义完全取决于 \(\{P\}\) 的选择。**刚性安装只能保证相机与吊点之间的相对位姿固定，不能自动保证相机相对重力方向的姿态固定。** 按证据条件分三种情形：

1. **\(\{P\}\) 为重力对齐坐标系，且姿态不变假设成立**：此时 \(R_{P\leftarrow C}\) 才可视为**常量**（可称固定外参），\(H_{PG}\) 为沿重力向下的铅垂分量，\(X_{PG},Y_{PG}\) 为水平分量，\(\theta_x,\theta_y\) 可解释为物理摆角分量。使用前须说明姿态不变的依据（塔身/小车/吊具在采集期间姿态不变，或已验证 \(R_{W\leftarrow C}\) 的波动可忽略）。
2. **\(\{P\}\) 为重力对齐坐标系，但姿态发生变化且有逐帧旋转证据**：必须使用**动态变换**
   \[
   {}^P\widehat{\mathbf d}_{PG}(t)=R_{P\leftarrow C}(t)\left({}^C\widehat{\mathbf g}(t)-{}^C\mathbf d_{CP}(t)\right),\qquad
   R_{P\leftarrow C}(t)=R_{P\leftarrow W}(t)\,R_{W\leftarrow C}(t),
   \]
   此时**不能把 \(R_{P\leftarrow C}\) 称为固定外参**；2.5 中所有把 \(R_{P\leftarrow C}\) 当常量的式子只在这种情形下退化为近似式。
3. **\(\{P\}\) 为随机构运动的固连坐标系**：分量只描述「吊点到抓斗」在该固连系中的相对几何，**不是**重力方向下的铅垂/水平分量；摆角定义必须按该固连系与重力系之间的实际变换重写，不能沿用 2.6 的 `atan2` 公式解释为物理摆角。

**平移向量同样要核对其坐标系与时间依赖性**：2.5 中的 \({}^C\mathbf d_{CP}\) 被约定为「在**相机坐标系**表达的有符号向量 \(P-C\)」。它的常量性依赖两个条件——表达坐标系固定，且 \(C\) 与 \(P\) 的相对位置在同一刚体内不随时间变化。若吊点相对相机可动（如吊具与小车的相对运动），或该向量被改到重力系/世界系表达，则须写成 \({}^C\mathbf d_{CP}(t)\) 或换算到相应坐标系，不能沿用固定值。欧氏距离 \(\lVert C-P\rVert\) 即使在相对位置固定时也不等于任一轴向分量，不能直接从 \(Z_c\) 中扣除。

摆角分母使用 \(H_{PG}\)，不能直接用 \(Z_c\)，除非已经证明 \(C=P\) 且相机坐标轴与吊点铅垂坐标轴完全一致。

对本项目的历史记录而言：配方约定 \(\{P\}\) 第三轴朝下，属于情形 1 或 2；「相机—吊点刚性外参已标定并在运行中保持固定」是该约定的**必要但不充分**条件，姿态不变假设或逐帧动态旋转证据尚未补齐。因此该坐标转换目前只能作为**条件接口**使用，**三维位置与摆角尚未验证**（见 [4.1](#sec-4-1)、[5.2](#sec-5-2)）。

<a id="sec-2-3"></a>

### 2.3 原图 OBB 与相机内参

检测器在原始图像上输出

\[
B=(u_G,v_G,w,h,\gamma,s),
\]

其中 \((u_G,v_G)\) 是 OBB 中心像素，\(w,h\) 是规范化后的长、短边像素长度，\(\gamma\) 是图像平面方向角，\(s\) 是检测置信度。\(\gamma\) 不是抓斗绕竖直轴的机械自旋角，也不是三维摆角。相机内参为

\[
K=\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix}.
\]

这里 \((c_x,c_y)\) 是主点像素坐标，不是 OBB 中心，更不是相机到吊点的米制偏移。要求检测输出已恢复到原图坐标；若经过 resize、padding 或 crop，必须先做逆变换，再与该原图内参配对。

<a id="sec-2-4"></a>

### 2.4 Raw-opt 深度公式

已知抓斗参考结构的物理长边 \(L_G\) 和短边 \(S_G\)。由原图 OBB 构造两个归一化投影尺度：

\[
s_L=w\sqrt{\left(\frac{\cos\gamma}{f_x}\right)^2+
                 \left(\frac{\sin\gamma}{f_y}\right)^2},
\]

\[
s_S=h\sqrt{\left(\frac{\sin\gamma}{f_x}\right)^2+
                 \left(\frac{\cos\gamma}{f_y}\right)^2}.
\]

相应的针孔深度候选为

\[
Z_L=\frac{L_G}{s_L},\qquad Z_S=\frac{S_G}{s_S}.
\]

定义长短轴深度不一致量

\[
q=\ln\left(\frac{Z_L}{Z_S}\right).
\]

当前选定的轻量双尺度修正式为

\[
\boxed{
\widehat Z_c
=cZ_S\exp(\beta q^2)
}
\]

其作用是以短边针孔深度为主体，用长短轴的各向异性缩短线索补偿倾斜影响；不新增关键点网络或深度主干。该形式是针对固定相机、固定抓斗几何构造并标定的任务特定解析模型，不能宣称为通用闭式真理。

Raw-opt v1 参数为

\[
c=1.0044577341,\qquad \beta=20.6822895437,
\]

标定集 GT OBB 的拟合支持范围为

\[
q\in[-0.0207032,\ 0.00670345].
\]

该范围只能作公式外推诊断。检测输出的有效性门必须由 `calibration_train` 上的检测 OBB 单独冻结，不能根据 fixed-dev 的失败结果事后调整。

**早期基线（历史）**：更早的纯对角线幂律模型为

\[
\widehat Z_{CG}^{\mathrm{opt}}=k(l'_{\mathrm{diag}})^\alpha ,
\]

其参数与误差见 4.5 节，只作历史基线，不与 Raw-opt 的参数、坐标和误差拼接。

<a id="sec-2-5"></a>

### 2.5 原相机坐标系反投影与相机—吊点外参

Raw-opt 的关键闭合关系是：

\[
\boxed{
\text{原图 }(u_G,v_G,w,h,\gamma)
\quad\longleftrightarrow\quad
Z_c=Z_{CG}^{\mathrm{opt}}
}
\]

因此直接用原图中心反投影：

\[
{}^C\widehat{\mathbf g}=
\begin{bmatrix}
(u_G-c_x)\widehat Z_c/f_x\\
(v_G-c_y)\widehat Z_c/f_y\\
\widehat Z_c
\end{bmatrix}.
\]

正式 Raw-opt 路线**不计算** \((u',v')\)，也不读取 `homography_raw_to_plumb`。相机无需数学上严格垂直向下。几何上需要区分三件事：

- **可以保证的**：只要内参正确、相机与吊点之间的**刚性安装**关系已标定，则「相机坐标系 → 吊点固连坐标系」的变换在运行期间固定，原相机坐标中的三维点可由该刚性变换转到吊点固连坐标；
- **不能自动保证的**：该刚性关系**不蕴含**相机相对重力方向的姿态固定。若 \(\{P\}\) 被定义为重力对齐坐标系，则 \(R_{P\leftarrow C}(t)=R_{P\leftarrow W}(t)R_{W\leftarrow C}(t)\)；
- **两种情况要分别处理**：姿态不变假设成立时 \(R_{P\leftarrow C}\) 才可取为常量（可称固定外参）；若姿态发生变化但已有逐帧旋转证据，则必须按 \(R_{P\leftarrow C}(t)\) 做动态变换，**不得称为固定外参**。平移向量 \({}^C\mathbf d_{CP}\) 也要核对其表达坐标系与是否随时间变化。三情形与平移条件的完整说明见 [2.2](#sec-2-2)。

令 \(R_{P\leftarrow C}\) 表示从相机坐标系到吊点坐标系的旋转，并约定吊点坐标系第三轴朝下；**下文所有以 \(R_{P\leftarrow C}\) 为固定量的式子都继承 2.2 的适用条件（情形 1），在情形 2 下应替换为动态形式**。定义

\[
{}^C\mathbf d_{CP}:={}^C(\mathbf p-\mathbf c)
\]

为「从相机光心 \(C\) 指向吊点 \(P\)」的固定米制向量（其常量性依赖表达坐标系与 \(C\)、\(P\) 相对位置两个条件，见 [2.2](#sec-2-2)）。则

\[
\boxed{
{}^P\widehat{\mathbf d}_{PG}
=R_{P\leftarrow C}
\left({}^C\widehat{\mathbf g}-{}^C\mathbf d_{CP}\right)
}
\]

写成

\[
{}^P\widehat{\mathbf d}_{PG}
=\begin{bmatrix}\widehat X_{PG}&\widehat Y_{PG}&\widehat H_{PG}\end{bmatrix}^{\mathsf T}.
\]

这一步同时处理相机的固定俯仰/横滚安装误差以及 \(C\) 与 \(P\) 的三维位置偏移。相机安装在钢丝绳出口附近只意味着外参可能较小，不意味着可以省略。

若离线标定证实 \(R_{P\leftarrow C}\approx I\)，且 \({}^C\mathbf d_{CP}=(d_x,d_y,d_z)^{\mathsf T}\)，才可化简为

\[
\widehat X_{PG}=\frac{(u_G-c_x)\widehat Z_c}{f_x}-d_x,\qquad
\widehat Y_{PG}=\frac{(v_G-c_y)\widehat Z_c}{f_y}-d_y,\qquad
\widehat H_{PG}=\widehat Z_c-d_z .
\]

这里的正负号来自 \(\mathbf d_{CP}=P-C\) 的明确定义，不能把 \(D_x,D_y\) 当作无符号常数随意相加。所谓「直接减去 \(CP\)」只有在轴对齐且 \(CP\) 指相应的有符号轴向分量时成立；欧氏距离 \(\lVert C-P\rVert\) 不能直接从 \(Z_c\) 中扣除。

<a id="sec-2-6"></a>

### 2.6 摆角计算

获得吊点到抓斗中心的三维向量后：

\[
\boxed{
\widehat\theta_x=\operatorname{atan2}(\widehat X_{PG},\widehat H_{PG}),
\qquad
\widehat\theta_y=\operatorname{atan2}(\widehat Y_{PG},\widehat H_{PG})
}
\]

合摆角可定义为

\[
\widehat\theta=\operatorname{atan2}
\left(\sqrt{\widehat X_{PG}^2+\widehat Y_{PG}^2},
\widehat H_{PG}\right).
\]

\(\theta_x,\theta_y\) 为有符号角，正负表示摆动方向；合摆角 \(\theta\ge0\)。

**物理解释的前提**：上式要给出**物理摆角**，必须先把三维向量表达在重力对齐坐标系中，因此分三种情形：

- 情形 1（重力对齐坐标系 + 姿态不变）：用常量 \(R_{P\leftarrow C}\) 代入即可，上式给出物理摆角；
- 情形 2（重力对齐坐标系 + 姿态变化但有逐帧旋转证据）：必须用 \(R_{P\leftarrow C}(t)\) 的动态变换代入后再计算，结果仍可解释为物理摆角，但**不得把该旋转称为固定外参**；
- 情形 3（\(\{P\}\) 为吊点固连坐标系）：上式只给出该固连系内的相对倾角，**不能称为物理摆角**，也不能与重力方向的铅垂偏移互换使用。

坐标系的适用条件见 [2.2](#sec-2-2)。

<a id="sec-2-7"></a>

### 2.7 虚拟铅垂路线的保留边界

虚拟铅垂图像若使用

\[
\mathbf x'\sim KR^{-1}K^{-1}\mathbf x,
\]

则 \((u',v')\)、\((w',h',\gamma')\) 必须与同一虚拟相机第三轴上的深度 \(Z_{CG}^{\mathrm{plumb}}\) 配对。**不能用虚拟铅垂像素反投影实体相机光轴深度 \(Z_{CG}^{\mathrm{opt}}\)。**

本项目保留 Plumb/Plumb 作为 Webots oracle 或姿态补偿消融，但不作为当前部署主线。历史上的 Plumb/Opt 拟合结果只用于复现实验，不再作为正式几何证据。

<a id="sec-2-8"></a>

### 2.8 Webots 真值与 GPS

- `pivot_gt_gps` 应位于真实悬挂铰点 \(P\)，不应为了简化公式移动到相机光心。
- `grab_reference_gt_gps` 应位于四个顶梁参考角点的三维均值 \(G\)，并随抓斗刚体运动。
- GPS 返回设备原点的世界坐标；场景编辑器显示的局部彩色轴不改变世界坐标读数。
- GPS 与 Supervisor 仅用于交叉验证 \(C,P,G\) 和真值，不作为训练/推理输入。
- 单个中心 GPS 不能证明参考平面姿态；平面法向、四角点共面误差和中心误差应由四角点计算。

仿真真值至少保留：相机内外参、\(C/P/G\) 世界坐标、四角点世界坐标、原图 GT OBB、\(Z_{CG}^{\mathrm{opt}}\)、\(H_{PG}\)、\(L_{PG}\)、摆角、时间戳和必要审计字段。

<a id="sec-2-9"></a>

### 2.9 七项几何问题的处理结论

| 问题 | 当前处理 |
| --- | --- |
| 虚拟铅垂 OBB 与实体光轴深度混配 | 已取消；主线改为 Raw-opt |
| \((u',v')\) 与 \(Z_{CG}^{\mathrm{opt}}\) 反投影 | 已取消；改用原图 \((u_G,v_G)\) |
| \(Z_c\) 同时表示相机深度和悬挂垂距 | 已拆分为 \(Z_c:=Z_{CG}^{\mathrm{opt}}\) 与 \(H_{PG}\) |
| 只补偿水平偏移、遗漏垂向偏移 | 外参使用完整三维 \((d_x,d_y,d_z)\) |
| \(D_x,D_y\) 加减号不明确 | 固定定义 \(\mathbf d_{CP}=P-C\)，统一先相减再旋转 |
| 部署依赖 Webots Supervisor 姿态/单应矩阵 | Raw-opt 推理不读取单应矩阵；仅需离线外参，且该外参的常量性受 [2.2](#sec-2-2) 条件约束（姿态不变或逐帧动态旋转证据） |
| 将旧 Plumb/Opt 指标当作新协议证据 | 明确降级为历史结果；Raw-opt 重新分层验证 |
| 刚性安装被当作「相对重力姿态固定」 | **仅澄清、未闭环**：刚性外参只保证相机与吊点的相对位姿固定；旋转矩阵**仅在姿态不变假设成立时才可取为常量**，姿态变化时必须用 \(R(t)\) 动态变换且不能称固定外参；平移向量同样须核对表达坐标系与时间依赖性。条件已写入 [2.2](#sec-2-2)、[2.5](#sec-2-5)、[2.6](#sec-2-6)；**三维位置与摆角仍未验证** |

这些问题在**公式、配置和评价入口的定义层面**已解决，并已由 calibration-train 的 GT OBB 闭合结果验证代码路径；但检测 OBB 有效性门、fixed-dev 检测 OBB、未知序列和真实部署仍待实验。其中**最后一行不属于「已解决」**：坐标系适用条件目前只是被显式化，姿态不变假设或动态旋转证据尚未补齐。

论文可用表述（含坐标系的必要条件）见 [5.2](#sec-5-2)。

---

## 3. 数据、标定与真值

<a id="sec-3"></a>

<a id="sec-3-1"></a>

### 3.1 数据文件与可复现入口

Webots 正式数据位于：

```text
/Users/mac/Documents/fangzhen/grab_副本/datasets/yolo_obb_depth_dataset
```

当前稳定世界文件位于：

```text
/Users/mac/Documents/fangzhen/grab_副本2/worlds/crane.wbt
```

两个工程目录中的 `worlds/crane.wbt` 内容一致，SHA256 均为：

```text
96ca6422cfa9625051b9895130b97fb074e68d88d76fd1b892f60c41b69f8750
```

两份控制器的几何导出主体相同，但当前激活的采集协议不同。`grab_副本2` 中控制器停留在 Train-03 配置；`grab_副本` 中控制器停留在 Unknown-02 单次覆盖补采配置。因此，**序列自身的 `metadata/manifest.json` 和 `metadata/frames.jsonl` 是判断其协议身份的正式依据**，不能只根据当前控制器顶部变量推断旧序列的采集配置。

| 控制器快照 | 当前协议配置 | SHA256 |
| --- | --- | --- |
| `grab_副本2/controllers/crane_driver_obb/crane_driver_obb.py` | Train-03 | `a8885c1cde1bba9dc1d05ebeaeb3290ec3932c72ce88a5f4607403d8db4368c0` |
| `grab_副本/controllers/crane_driver_obb/crane_driver_obb.py` | Unknown-02 | `b6f5769b6b8fa81fb126509188549d93f5dd53761937f7d59911e22659ad75bc` |

现有目录没有为每条正式序列单独冻结一份采集时控制器源码。序列 manifest 能够锁定随机种子、帧率、协议角色、场景名称和 run ID，但若论文要求逐字节重建每次控制逻辑，还应将对应控制器与世界文件按序列归档。该问题不改变已保存逐帧真值的完整性，但属于采集程序级复现限制。

当前工程没有单独的抓斗或起重机 `.proto` 文件；场景结构、相机、悬挂点、抓斗参考点和四个角点均直接定义在 `crane.wbt` 中。采集与真值导出程序为：

```text
/Users/mac/Documents/fangzhen/grab_副本/controllers/crane_driver_obb/crane_driver_obb.py
```

仓库内交付物：

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `docs/data/webots_depth_formal_frame_index_v1.csv` | 2,971 帧正式数据的逐帧索引与核心真值 | 拟合、评价、绘图及帧级误差追踪 |
| `docs/data/webots_depth_formal_frame_index_v1.manifest.json` | 数据源哈希、序列统计和完整性结果 | 复现实验与论文数据审计 |
| `tools/depth/export_webots_depth_frame_index.py` | 从原始 manifest/JSONL 重新导出索引 | 防止人工复制字段造成错位 |

CSV 使用相对路径，例如 `fixed_dev_01_unseen_depth_swing/images/frame_00000.jpg`；与上方数据根目录拼接后即可定位原图。检测输出不写入该 CSV，从而保证 Webots 真值与模型预测保持两个独立证据层。

<a id="sec-3-2"></a>

### 3.2 相机标定数据

三条正式序列使用一致的相机模型：

| 参数 | 数值 |
| --- | ---: |
| 图像宽度 | 1024 px |
| 图像高度 | 1024 px |
| 水平视场角 | 1.0 rad，约 57.296° |
| \(f_x\) | 937.2097135167754 px |
| \(f_y\) | 937.2097135167754 px |
| \(c_x\) | 512.0 px |
| \(c_y\) | 512.0 px |
| 投影模型 | 平面小孔相机 |
| 镜头畸变 | Webots 数据中不模拟镜头畸变 |

每帧 `camera_geometry.position_world_xyz_m` 保存相机光心的世界坐标，`camera_geometry.rotation_world_from_camera_webots` 保存从相机坐标系到 Webots 世界坐标系的旋转矩阵，`homography_raw_to_plumb` 保存原始图像到共光心虚拟铅垂图像的逐帧单应矩阵。

上述内参只能用于当前 1024×1024 原始 Webots 图像。若检测前发生 resize、padding 或裁剪，必须将预测 OBB 恢复到原图像素坐标后再使用这些内参。仓库中的 `tools/depth/infer_webots_obb_sequence.py` 通过 MMDetection 推理入口取得 `rescale=True` 的检测结果，并将 OBB 记录为原图像素坐标；后续若更换推理脚本，需要重新核对这一行为。

<a id="sec-3-3"></a>

### 3.3 顶梁真实尺寸与参考平面

OBB 标注对象是抓斗顶梁参考平面，不是整个抓斗平台。三条正式序列的物理尺寸合同一致：

| 几何量 | 数值 |
| --- | ---: |
| 顶梁参考平面平均长边 \(L_G\) | 1.7750000000 m |
| 顶梁参考平面平均短边 \(S_G\) | 0.6100409809 m |
| 长宽比 | 2.9096405909 |
| 初始共面误差 | \(9.386\times10^{-8}\) m |
| 参考中心误差 | \(2.346\times10^{-8}\) m |

场景树中的 `GRAB_PT_TL`、`GRAB_PT_TR`、`GRAB_PT_BL` 和 `GRAB_PT_BR` 定义顶梁参考平面的四个三维角点，`GRAB_REFERENCE` 位于四角点均值位置。抓斗发生刚体倾斜时，参考点仍随同一刚体保持在该平面中心。

四角点平均值给出平面中心，而不是对角线长度。图像对角线应由四个投影角点拟合 OBB 后按

\[
l_{\mathrm{diag}}=\sqrt{w^2+h^2}
\]

计算。`l_diag_raw_px` 属于原始相机图像，`l_diag_plumb_px` 属于虚拟铅垂图像，后者才对应公式中的 \(l'_{\mathrm{diag}}\)。

<a id="sec-3-4"></a>

### 3.4 正式序列及协议职责

当前主实验仅包含 Train-03、Fixed-dev-01 和 Unknown-02，共 2,971 帧。每条序列均满足图像数、标签数和 JSONL 行数相等，帧号从 0 连续递增，OBB、深度真值和运行时真值审计通过率均为 100%。

| 序列 | 协议职责 | FPS | 帧数 | \(Z_{CG}^{\mathrm{opt}}\) 范围 | \(H_{PG}\) 范围 | 合摆角范围 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `calibration_train_03_tilt_obb` | 拟合公式参数 | 10.417 | 981 | 11.888–24.575 m | 11.780–24.487 m | 0.037–13.968° |
| `fixed_dev_01_unseen_depth_swing` | 选择公式并冻结 | 12.5 | 893 | 12.518–24.774 m | 12.435–24.696 m | 0.029–10.789° |
| `unknown_test_02_coverage_retry` | 冻结后的单次未知序列评估 | 10.417 | 1,097 | 12.647–23.070 m | 12.538–22.993 m | 0.114–10.232° |

摆角覆盖：

| 序列 | 0–3° | 3–5° | 5–8° | ≥8° |
| --- | ---: | ---: | ---: | ---: |
| Train-03 | 228 | 166 | 294 | 293 |
| Fixed-dev-01 | 510 | 248 | 90 | 45 |
| Unknown-02 | 351 | 268 | 374 | 104 |

各序列的用途**不可交换**：Train-03 可用于拟合；Fixed-dev-01 可用于选择公式形式，但不能回填训练；Unknown-02 只能在方案冻结后评价。Unknown-02 是 Unknown-01 未暴露模型误差、仅因 ≥8° 帧数不足而进行的预注册单次覆盖补采，不能再用于调参或触发新的未知序列重采。

以下数据保留历史价值，但不进入当前主实验：

| 序列 | 帧数 | 状态与作用 |
| --- | ---: | --- |
| `calibration_train_01` | 550 | 早期纯幂函数 \(k(l'_{\mathrm{diag}})^\alpha\) 标定 |
| `calibration_train_02_yaw_small_step` | 1,143 | ±8° 自旋训练侧机制审计，不重新加权拟合 |
| `unknown_test_01_mixed_depth_swing` | 868 | ≥8° 仅 6 帧，覆盖门槛失败；不得用于公式或阈值设计 |
| `depth_pilot_04`、`depth_pilot_05_compact` | — | 真值与存储流程调试，不属于正式证据 |

`depth_pilot_04` 和 `depth_pilot_05_compact` 仅用于完整真值审计与 compact JSONL 存储流程调试，不进入标定拟合、`fixed_dev`、未知序列泛化或部署证据。正式 v6/v7 序列已通过相同运行时真值审计，因此 pilot 原始图像和标签可删除；其阶段性结论保留在本文档中。

<a id="sec-3-5"></a>

### 3.5 图像、标签与逐帧真值的对应关系

每条序列均采用以下目录结构：

```text
<sequence_id>/
├── images/frame_00000.jpg
├── labels/frame_00000.txt
└── metadata/
    ├── manifest.json
    └── frames.jsonl
```

`frames.jsonl` 的每一行对应一帧，核心对齐字段为：

| 字段 | 含义 |
| --- | --- |
| `sequence_id` | 完整序列身份 |
| `run_instance_id` | 一次连续采集实例身份 |
| `frame_index` | 从 0 开始的连续帧号 |
| `simulation_time_s` | Webots 仿真时间戳，用于动态指标 |
| `image_file` | 相对序列目录的原图路径 |
| `label_file` | 相对序列目录的 YOLO-OBB 标签路径 |
| `obb_valid` | 四角点投影及 OBB 是否有效 |
| `truth_valid` | 本帧深度真值是否有效 |
| `frame_truth_audit.passed` | GPS、平面与投影闭合审计总结果 |

YOLO-OBB 标签每行包含 6 项：类别、归一化中心 \((c_x,c_y)\)、归一化宽高 \((w,h)\) 和方向角 \(\gamma\)（rad）。JSONL 同时保存原图像素单位的中心、宽高、方向角、四角点和虚拟铅垂 OBB，后续几何评价应优先读取 JSONL，避免重复反归一化造成精度损失。

`simulation_time_s` 是仿真时钟，不是现实世界 UTC 时间。动态评价应按该时间戳计算，不能默认所有序列具有相同帧率。

<a id="sec-3-6"></a>

### 3.6 GT 标注、独立深度真值与姿态字段

逐帧 CSV 已导出下列数据组。

**GT OBB**

- 原始图像：`gt_cx_px`、`gt_cy_px`、`gt_raw_w_px`、`gt_raw_h_px`、`gt_raw_gamma_deg`、`gt_l_diag_raw_px`；
- 虚拟铅垂图像：`gt_plumb_w_px`、`gt_plumb_h_px`、`gt_plumb_gamma_deg`、`gt_l_diag_plumb_px`；
- 四角点：`gt_pixel_corners_json`；
- 几何状态：`obb_valid`、`projected_ar_warning`。

这里的 GT OBB 由 Webots 三维角点投影获得，不是人工标注框。当前日志没有逐角点遮挡率或显式可见性类别，`obb_valid=1` 只能说明几何投影有效，不能等价解释为无遮挡。如果后续需要遮挡分层，应重新增加可见性/深度缓冲审计，不能从现有字段推断。

**独立距离真值**

- `z_cg_opt_gt_m`：相机光心到抓斗参考中心的光轴深度，是当前深度模型的首要监督与评价目标；
- `z_cg_plumb_gt_m`：相机到抓斗中心沿世界重力方向的投影深度；
- `h_pg_vertical_gt_m`：悬挂点到抓斗参考中心的垂直高度差；
- `l_pg_euclidean_gt_m`：悬挂点到抓斗参考中心的三维直线距离。

这些值来自 Webots 仿真几何，与 GT OBB 尺寸是不同类型的真值。GT OBB 不能替代米制深度真值，卷扬机传感器读数也不能替代上述距离定义。

**相机、目标和摆角状态**

- `grab_x_camera_cv_m`、`grab_y_camera_cv_m`、`z_cg_opt_gt_m`：抓斗参考中心在相机 CV 坐标系中的三维位置；
- `camera_world_x_m`、`camera_world_y_m`、`camera_world_z_m` 及 `camera_r_wc_00`～`camera_r_wc_22`：相机世界位姿；
- `x_pg_gt_m`、`y_pg_gt_m`、`h_pg_vertical_gt_m`：悬挂点至抓斗参考中心的三维分量；
- `theta_x_gt_deg`、`theta_y_gt_deg`、`theta_total_gt_deg`：两个有符号摆角分量及非负合摆角；
- `psi_yaw_relative_deg`：抓斗相对机械正位的自旋角，与图像 OBB 方向角不同。

有符号摆角分量允许为负，其符号表示相对坐标轴的摆动方向；合摆角用于表示摆幅，始终非负。抓斗自旋角 \(\psi\) 是三维关节绕自身轴的旋转，OBB 方向角 \(\gamma\) 是顶梁投影在图像中的二维方向，两者受相机姿态和抓斗倾斜共同影响，定义上不要求相等。

现有紧凑日志保存了抓斗中心位置、相对自旋角和摆角，但没有直接保存顶梁完整三维旋转矩阵、逐帧平面法向量或四角点世界坐标。运行时审计证明参考平面共面与中心闭合，但若论文需要按真实平面法向倾角分层，应补充新的导出字段，或明确使用 \(q\) 与合摆角作为现有姿态代理。该缺口不妨碍当前 \(Z_{CG}^{\mathrm{opt}}\) 尺度公式验证。

<a id="sec-3-7"></a>

### 3.7 数据完整性结论

围绕当前深度可行性验证，已有数据已经满足下列条件：原始 RGB、帧号与仿真时间戳连续对应；相机内参固定且明确；顶梁标注范围与物理尺寸一致；每帧具有 GT OBB、独立光轴深度、吊点垂距、直线距离和摆角真值；相机世界位姿可用于真值审计。Train-03 与 Fixed-dev-01 已用于 Raw-opt 的拟合和组件诊断；Unknown-02 仅保留为旧 Plumb/Opt 历史结果。

仍缺少两类材料：其一是 Train-03 上 K1/SymEOOD 原始框与统一 SymEOOD+DINO 各候选框的同帧几何输出，以及由此冻结的唯一框来源和 `depth_valid` 规则；其二是逐帧顶梁完整三维姿态和显式可见性信息，后者只在开展更细的姿态/遮挡误差分解时需要。

---

## 4. 实验结果与演进裁决

<a id="sec-4"></a>

<a id="sec-4-1"></a>

### 4.1 Raw-opt fixed-dev 组件结果

数据职责：`calibration_train_03_tilt_obb`（981 帧）只拟合参数；`fixed_dev_01_unseen_depth_swing`（893 帧）只选择/诊断公式；未读取任何未知序列来生成本次 Raw-opt 参数。

GT OBB 条件下结果：

| 模型 | MAE (m) | RMSE (m) |
| --- | ---: | ---: |
| M0 对角线幂律 | 0.10764 | 0.11361 |
| M1S 短边针孔 | 0.04490 | 0.05053 |
| **M2S Raw-opt** | **0.03417** | **0.04045** |

M2S 的 AbsRel 为 `0.001957`（即 0.1957%），P95 绝对误差 `0.07679 m`，最大绝对误差 `0.10716 m`。相对 M0，MAE 降低约 68.25%；相对 M1S，MAE 降低约 23.89%。参数只由 Train-03 的 981 帧拟合，公式形式在 Fixed-dev-01 上诊断。

这些数值只证明「Webots 真值 OBB + Raw-opt 尺度公式」的组件性能，**不证明**：

- 当前检测模型输出 OBB 后仍有相同精度；
- 未知序列泛化已经通过；
- 真实视频具有米制深度或绝对摆角精度；
- 已完成部署。

**历史对照（不可迁移）**：旧 Plumb/Opt 契约曾在 Unknown-02 的 1,097 帧上得到 MAE `0.0307 m`、RMSE `0.0370 m`、AbsRel `0.1756%` 和 P95 绝对误差 `0.0681 m`。由于该路线把虚拟铅垂 OBB 与实体相机光轴深度配对，且 Unknown-02 已经暴露结果，这组数值现在仅作为历史可复现记录，不能证明 Raw-opt 的未知序列泛化。Raw-opt 尚未读取新的未知序列。

<a id="sec-4-2"></a>

### 4.2 正式 native-S14 的 calibration-train 检测 OBB 审计

审计使用 `crane_symeood_formal_dino_native_s14_v1.py` 和 `source_safe_interpolated_head.pth`，在 `calibration_train_03_tilt_obb` 的 981 帧上运行。该模型属于冻结 DINOv2 native-S14、ROI classifier `alpha=0.5`、S7 disabled 的正式 source-gated 组件；本次运行仅是 **calibration-train 接口诊断**，不是 fixed target-dev、未知序列或部署结果。

协议检查结果：处理 981/981 帧、检出 981/981 帧，预测坐标已恢复到原图，`coordinate_contract=raw_opt_v1`，未使用 `homography_raw_to_plumb`，未重拟合 Raw-opt 参数，未调阈值，也未读取未知序列。同一评价入口的 GT OBB Raw-opt 结果为 MAE `0.04090 m`、RMSE `0.05343 m`，证明此前原图 OBB 与实体相机光轴深度错配的问题已经在代码和坐标契约层面修复。

但是，该模型的预测框几何不适合直接驱动当前双尺度公式：

| calibration-train 指标 | 结果 |
| --- | ---: |
| 检出率 | 100%（981/981） |
| mean RIoU | 0.6198 |
| 中心误差均值 / P95 | 6.38 / 12.71 px |
| 长边相对误差均值 | -7.50% |
| 短边相对误差均值 / P95 | +3.98% / +42.78% |
| 方向角绝对误差均值 / P95 | 5.29° / 20.82° |
| 检测 \(q\) 均值 / P95 | 0.10385 / 0.54673 |
| \(q\) 落入 GT 拟合支持域 | 52/981（5.30%） |

GT 拟合支持域为 \(q\in[-0.0207032,0.00670345]\)，而 929/981 帧的检测 \(q\) 越界。由于公式包含 \(\exp(\beta q^2)\)，长短边比例误差会被指数项放大，无约束代入会产生无物理意义的极端深度。因此当前结论只能表述为：

> **native-S14 在该 Webots calibration-train 上能够稳定检出抓斗，但其 OBB 长短边几何精度不足以直接支撑 Raw-opt 深度；这不是 Raw-opt 坐标修复失败，也不能据此笼统断言模型的所有检测能力都失效。**

该模型不进入 fixed-dev 深度评价。置信度也不能单独充当有效性门；越过冻结支持域的帧应输出 `depth_valid=False`，而不是截断 \(q\) 后继续声称有效深度。

<a id="sec-4-3"></a>

### 4.3 检测 OBB 深度接口的当前数据状态

**仍缺少的同帧候选比较**：K1/SymEOOD 原始框、确定性 fallback 和 Base V3 尚未在同一 Train-03 原图上形成可与 2,971 帧索引连接的完整多来源框几何证据。下一步应先在 calibration-train 内比较各来源的长边、短边、长宽比、\(q\) 和 Raw-opt 误差，不读取 fixed-dev 选择几何来源。

先前已读取的 K1 固定开发集结果覆盖 893/893 帧，但预测 OBB 的长短边比例与深度标定域严重失配，曾记录 858/893 帧的 \(q\) 超出标定范围，并导致四项预设深度门槛失败。这一现象说明 GT OBB 条件下可行的尺度公式会受到检测框几何误差放大。原始 `predictions_full.jsonl`、`predictions_full.jsonl.manifest.json`、`inference_full.log` 和 `evaluation_vs_k1.json` 当前已不在本机可检索路径中，因此上述数字只能作为阶段性诊断记录，不能仅凭本文件作为论文结果引用；正式写作前需恢复原始结果及哈希，或按冻结配置重新运行。

确定性 K1-present-else-DINO fallback 和 Base V3 的现有结果主要属于 source-val/seq11 检测机制审计，并非本批 Webots fixed-dev 图像上的同步预测。V3 source gate 也不能替代 Webots 深度接口实验。最新模型融合交接仍将候选限定在 source-val 层，尚未形成同时通过 fixed TEST、未知序列和部署验证的「全面最终模型」。

| 所需检测输入 | 当前状态 | 能否用于端到端深度论文结果 |
| --- | --- | --- |
| GT OBB | 三条正式序列完整 | 可以，作为深度公式 oracle/上界 |
| K1 fixed-dev OBB | 曾运行，当前原始文件缺失 | 暂不能正式引用 |
| 确定性 fallback fixed-dev OBB | 未发现同帧 Webots 结果 | 不能 |
| Base V3 fixed-dev OBB | 未发现同帧 Webots 结果 | 不能 |
| 正式 native-S14 calibration-train OBB | 981/981 已恢复；框几何不满足 Raw-opt | 不能作为端到端结果；仅作失败诊断 |
| 正式 native-S14 fixed-dev OBB | 按协议未继续运行 | 不能 |

模型融合改进停止并不自动把某个候选升级为部署模型。进入检测 OBB 深度实验前，仍需明确唯一的 config、checkpoint、checkpoint SHA256、推理阈值和坐标恢复流程。不同模型、checkpoint 或证据层级的结果不得拼接成同一个「最终模型」结论。

<a id="sec-4-4"></a>

### 4.4 检测器责任与证据顺序

检测器误差会同时改变 \((u_G,v_G,w,h,\gamma)\)。因此必须依次报告：

1. GT OBB + Raw-opt：隔离尺度公式误差；
2. 检测 OBB + Raw-opt：测量检测误差传播；
3. 未知序列：验证冻结系统的序列泛化；
4. 真实视频：只报告连续性、有效输出率、时延和资源占用。

旧 K1 fixed-dev 上曾出现大量检测输出 \(q\) 越过旧标定范围；native-S14 calibration-train 审计进一步确认，长短边比例失配会被指数项放大。两者属于不同模型和不同协议层，**不得合并计数或指标**。检测输出有效性规则只能在 calibration-train 上设计并版本化，不能利用 fixed-dev 失败结果事后调门。

<a id="sec-4-5"></a>

### 4.5 早期幂律标定与自旋机制审计

**calibration_train_01：连续垂向标定**（550 帧）：作用为拟合 `Z_CG,opt = k (l'_diag)^alpha`，拟合参数为

```text
k = 1764.5919116475932
alpha = -1.0005117394222605
```

训练内误差：MAE 5.847 mm、RMSE 7.056 mm、AbsRel 0.0354%、最大绝对误差 14.690 mm。

**calibration_train_02_yaw_small_step：小幅自旋机制审计**：schema `webots_crane_depth_gt_v7`；帧数 1143，图像、标签和 JSONL 全部对齐，帧序号连续；采样 12.5 FPS，相邻帧固定间隔 0.08 s，单一 run ID；有效性 1143/1143 OBB 有效、深度真值有效、运行时真值审计通过。三个完整自旋循环分别位于约 8.3 m、16.4 m 和 25.5 m；相对机械正位自旋角完整覆盖 `[-8 deg, +8 deg]`；总摆角范围 0.005–2.514 deg，仅 38 帧大于 2 deg，没有帧大于 3 deg。直接使用 `calibration_train_01` 参数时：MAE 7.331 mm、RMSE 8.562 mm、AbsRel 0.0542%、最大绝对误差 19.503 mm。绝对残差与绝对自旋角的相关系数为 `0.0366`，与总摆角的相关系数为 `0.4617`。

裁决：小幅自旋不是当前尺度误差的主要来源；摆角是更值得在 `fixed_dev` 中独立验证的因素。本序列是训练侧机制审计，不因为帧数更多而重新加权拟合 `k, alpha`。

**已冻结的暂定标定文件**：`crane_project/configs/depth_scale_calibration_provisional_v1.json` 只使用 `calibration_train_01` 拟合参数；`calibration_train_02_yaw_small_step` 只提供训练侧自旋应力审计；在固定 `fixed_dev` 运行前不再改参数。

**历史阶段的下一步（已执行并取代）**：当时的唯一后续动作是采集一条预先固定的 `fixed_dev_01_unseen_depth_swing`，使用未在标定中重复的连续深度轨迹与自然摆动，不开启主动自旋，只验证已冻结的 `k, alpha` 而不回填拟合。该序列已完成采集，并已用于 Raw-opt 的组件诊断；此后 Raw-opt 取代了该幂律参数路线。

**历史 Plumb/Opt 与 Unknown-02 的降级规则**（对应 4.1 中的历史对照数值）：

- 旧 Plumb/Opt 结果只能作为历史可复现记录，不能与 Raw-opt fixed-dev 数值并列声称未知序列泛化；
- Unknown-02 已被 Raw-opt 复用为「新未知序列」是禁止的；Raw-opt 的未知序列必须重新注册；
- 相关文档中旧「完整修复」措辞应以上述限制解读。

---

## 5. 当前接口与复现入口

<a id="sec-5"></a>

<a id="sec-5-1"></a>

### 5.1 脚本与数据入口

```text
数据根目录：/Users/mac/Documents/fangzhen/grab_副本/datasets/yolo_obb_depth_dataset
采集/真值导出：/Users/mac/Documents/fangzhen/grab_副本/controllers/crane_driver_obb/crane_driver_obb.py
帧索引重建：tools/depth/export_webots_depth_frame_index.py
检测推理（rescale=True，输出原图坐标）：tools/depth/infer_webots_obb_sequence.py
检测 OBB 深度评价：evaluate_detector_obb_depth.py
正式 native-S14 config：crane_project/configs/crane_symeood_formal_dino_native_s14_v1.py
暂定标定文件：crane_project/configs/depth_scale_calibration_provisional_v1.json
帧索引 CSV：docs/data/webots_depth_formal_frame_index_v1.csv
帧索引 manifest：docs/data/webots_depth_formal_frame_index_v1.manifest.json
```

<a id="sec-5-2"></a>

### 5.2 论文写作可用表述

**数据与目标定义**：仿真数据采用相机光心、钢丝绳悬挂点与抓斗顶梁参考中心相互独立的三点几何定义。单目尺度模型以抓斗参考中心在相机坐标系中的光轴深度 \(Z_{CG}^{\mathrm{opt}}\) 为预测目标，并利用已知相机内参、顶梁物理长短边和原始图像 OBB 构造尺度观测。完成相机深度估计后，再通过离线标定且运行期间固定的相机—悬挂点三维外参将抓斗位置转换至悬挂点坐标系，得到垂直高度差 \(H_{PG}\) 及两个水平方向偏移，进而计算抓斗摆角。该处理避免了将相机深度、悬挂垂距和两点直线距离混用。

**该摆角表述的必要条件（必须与上式一起写）**：吊点坐标系须定义为**重力对齐坐标系**，且坐标变换的形态取决于姿态证据：

- 若**相机相对重力方向的姿态在运行期间不变**，则 \(R_{P\leftarrow C}\) 与 \({}^C\mathbf d_{CP}\) 可视为常量，可称固定外参；
- 若**姿态发生变化但已有逐帧旋转证据**，则必须使用 \(R_{P\leftarrow C}(t)\) 与相应的 \({}^C\mathbf d_{CP}(t)\) 动态变换，**不能称为固定外参**；
- 若上述两者都不成立，或 \(\{P\}\) 取吊点固连坐标系，则只能报告该固连坐标系中的相对几何量，或先运行姿态补偿消融，不得称为重力方向下的物理偏移与摆角。

相机与吊点的刚性安装只保证二者相对位姿固定，**不蕴含**相对重力的姿态条件。条件的形式化与平移向量的坐标系/时间依赖性见 [2.2](#sec-2-2)。

Webots 数据按完整连续序列划分为参数拟合、固定开发和未知序列评估三部分，相邻帧不得随机拆分。Train-03 与 Fixed-dev-01 分别包含 981 和 893 帧，已承担 Raw-opt 的参数拟合和固定开发职责；包含 1,097 帧的 Unknown-02 属于旧契约历史证据，新的 Raw-opt 未知序列尚未注册。Webots GPS 与 Supervisor 坐标只用于生成和交叉审计仿真真值，不作为模型输入。真实视频不具备独立距离真值，因此只承担跨域可运行性与输出稳定性验证，不用于支持真实场景米制深度精度结论。

**当前结果**：冻结的 Raw-opt 双尺度修正公式在 Fixed-dev-01 的真值 OBB 条件下取得 0.03417 m 的 MAE 和 0.04045 m 的 RMSE，反映尺度模型在 Webots 投影真值框条件下的组件上界。Unknown-02 的 0.0307 m MAE 来自旧 Plumb/Opt 坐标契约，只能作为历史结果，不能与 Raw-opt fixed-dev 数值并列声称未知序列泛化。正式 native-S14 在 Train-03 上虽实现 981/981 帧检出，但预测框长短边比例失配，仅 52/981 帧的 \(q\) 位于 GT 拟合支持域，因此当前不具备端到端深度结果。

---

## 6. 未解决问题与后续实验条件

<a id="sec-6"></a>

<a id="sec-6-1"></a>

### 6.1 唯一允许的实验顺序

1. 不让当前 native-S14 进入 fixed-dev；它保留为 calibration-train 失败诊断；
2. 在同一 `calibration_train_03_tilt_obb` 上比较 K1/SymEOOD 原始框与统一 SymEOOD+DINO 各候选框的长边、短边、长宽比和 \(q\)，不得读取 fixed-dev 选框源；
3. 选择并冻结唯一的框几何来源，同时冻结越界时 `depth_valid=False` 的规则；
4. 不调参数地运行 `fixed_dev_01_unseen_depth_swing`，分别报告 GT OBB 与冻结检测 OBB；
5. 若最终检测模型或框来源发生变化，重新执行 calibration-train 与 fixed-dev，旧结果只保留为对应模型版本的诊断；
6. 只有检测模型、Raw-opt 参数、有效性门和外参均冻结后，才注册并运行全新的未知序列；
7. 真实视频只做迁移稳定性和部署性能评价。

不得重复使用已经暴露结果的旧未知序列来调整新 Raw-opt 系统，也不得把 source gate、fixed target-dev、未知序列泛化和部署结论混写。

**最小后续动作**：现阶段无需重新采集标定或 fixed-dev 尺度数据，也无需先采集新的真实视频。最小动作就是在上面第 2 步比较各候选框来源、冻结唯一几何来源与 `depth_valid` 规则，随后按第 4 步在不调参数的前提下运行 Fixed-dev-01 的 893 帧评价（`evaluate_detector_obb_depth.py`）。若最终检测前端或框来源改变，这两步必须按新版本重做。

**尚缺且不在本流程内可补的材料**：逐帧顶梁完整三维姿态（含平面法向量、四角点世界坐标）与显式可见性信息；坐标系适用条件所需的「相机相对重力姿态不变」证据，或逐帧旋转 \(R_{W\leftarrow C}(t)\) 与相应 \({}^C\mathbf d_{CP}(t)\) 的动态变换证据。前者只在开展更细的姿态/遮挡误差分解时需要，后者是摆角物理解释的前提（见 [2.2](#sec-2-2)、[2.9](#sec-2-9)）。在这些材料补齐前，三维位置与摆角仍属**未验证**。

真实视频仍只报告连续性、稳定性、有效输出率、延迟和显存，不报告米制深度或绝对摆角精度。

---

<a id="appendix-sources"></a>

## 附录 A 来源索引

原文件已逐字节保存在 [整合前归档](archive/20260915_pre_consolidation/)，只用于追溯，不继续维护。

| 锚点 | 归档文件 | 原文件开头（历史快照） | 正文去向 |
| --- | --- | --- | --- |
| <a id="source-g"></a>`source-g` | `archive/20260915_pre_consolidation/Webots 单目深度几何契约.md` | *Webots 单目深度几何契约：Raw-opt 路线*；「Raw-opt 公式已在 calibration_train_03_tilt_obb 上拟合，并在 fixed_dev_01_unseen_depth_swing 的 GT OBB 条件下完成组件级诊断……fixed-dev 检测 OBB、未知序列和部署证据尚未完成」 | 2.2–2.9、4.1、4.2、4.4、6.1、5.2 |
| <a id="source-e"></a>`source-e` | `archive/20260915_pre_consolidation/Webots深度估计现有数据与论文证据清单_20260906.md` | *Webots 深度估计现有数据与论文证据清单*，2026-09-07；「2026-09-06 契约修订：正式部署路线已改为 Raw-opt。原图 OBB 只能与实体相机光轴深度配对」 | 2.1、3.1–3.7、4.1、4.2、4.3、5.2、6.2 |
| <a id="source-c"></a>`source-c` | `archive/20260915_pre_consolidation/webots_depth_calibration_status_20260823.md` | *Webots 单目尺度标定状态（2026-08-23）* | 3.4、4.5 |

整合清单与哈希：[manifest.json](archive/20260915_pre_consolidation/manifest.json)。该 manifest 是 **2026-09-15 首次机械整合的快照**：其中的 `anchor`、`integrated_body_sha256` 字段描述的是当时的章节编号与结构，本轮语义重写已改变编号与组织方式，因此这些字段不再对应当前正文，只作为归档原文的身份与完整性凭据。本轮映射见 [融合核对表](融合核对表.md)。

本轮对该文件的语义处理：符号、公式、数据协议与评价边界只保留一处完整说明；Raw-opt 参数与 fixed-dev 数值合并为 2.4 与 4.1；native-S14 calibration-train 审计合并为 4.2，其数据状态一览表并入 4.3；两段论文可用表述合并为 5.2 一处（2.9 只留指针）；Unknown-02 的降级规则不再单列小节，改并入 4.1 之后的历史对照（数值只在 4.1 出现一次）；后续顺序与最小后续动作合并为一节；坐标系适用条件（刚性安装 ≠ 相对重力姿态固定）写入 2.1.1、2.2、2.5、2.6、2.9 与 5.2。早期「完整修复」措辞以 1.2 与 4.1 的限制解读。

相关文档：[冻结 DINOv2 与 SymEOOD 检测融合](冻结DINOv2与SymEOOD检测融合.md)、[OBB 观测可靠性与连续输出](OBB观测可靠性与连续输出.md)、[研究文档入口](README.md)。
