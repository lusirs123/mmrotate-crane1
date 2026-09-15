# Webots 单目深度几何契约：Raw-opt 路线

> 当前状态：Raw-opt 公式已在 `calibration_train_03_tilt_obb` 上拟合，并在 `fixed_dev_01_unseen_depth_swing` 的 **GT OBB** 条件下完成组件级诊断。Raw-opt 坐标契约与评价入口已经修复并通过 GT OBB 闭合；正式 native-S14 已完成 calibration-train 检测 OBB 审计，但其框几何当前不满足该深度接口。fixed-dev 检测 OBB、未知序列和部署证据尚未完成。
>
> 系统边界：纯 RGB 单目视觉，不使用 PLC、编码器或其他现场传感器。Webots GPS、Supervisor 位姿和虚拟铅垂单应矩阵只用于生成/审计仿真真值，不是部署输入。

## 1. 最终路线与符号约定

正式路线固定为：

```text
原始图像 OBB
  -> Raw-opt 尺度模型得到相机光轴深度
  -> 原相机坐标系反投影
  -> 固定相机—吊点外参变换
  -> 吊点到抓斗的三维相对向量
  -> 两轴摆角
```

三个参考点必须分开：

| 符号 | 含义 |
| --- | --- |
| \(C\) | 实体相机光心 |
| \(P\) | 钢丝绳悬挂/摆动铰点 |
| \(G\) | 抓斗顶梁 OBB 参考平面中心 |

本文唯一保留的 \(Z_c\) 定义为

\[
\boxed{Z_c:=Z_{CG}^{\mathrm{opt}}}
\]

即抓斗参考中心 \(G\) 在**实体相机坐标系光轴**上的深度。它不是吊点到抓斗的铅垂距离，也不是 \(C\) 到 \(G\) 的欧氏距离。

另定义：

\[
H_{PG}=\text{吊点到抓斗中心的向下铅垂分量},\qquad
L_{PG}=\lVert G-P\rVert_2.
\]

摆角分母使用 \(H_{PG}\)，不能直接用 \(Z_c\)，除非已经证明 \(C=P\) 且相机坐标轴与吊点铅垂坐标轴完全一致。

## 2. 原图 OBB 与相机内参

检测器在原始图像上输出

\[
B=(u_G,v_G,w,h,\gamma,s),
\]

其中 \((u_G,v_G)\) 是 OBB 中心像素，\(w,h\) 是规范化后的长、短边像素长度，\(\gamma\) 是图像平面方向角，\(s\) 是检测置信度。\(\gamma\) 不是抓斗绕竖直轴的机械自旋角，也不是三维摆角。

相机内参为

\[
K=\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix}.
\]

这里 \((c_x,c_y)\) 是主点像素坐标，不是 OBB 中心，更不是相机到吊点的米制偏移。

要求检测输出已恢复到原图坐标；若经过 resize、padding 或 crop，必须先做逆变换，再与该原图内参配对。

## 3. Raw-opt 深度公式

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

Raw-opt v1 参数为：

\[
c=1.0044577341,\qquad \beta=20.6822895437.
\]

标定集 GT OBB 的拟合支持范围为

\[
q\in[-0.0207032,\ 0.00670345].
\]

该范围只能作公式外推诊断。检测输出的有效性门必须由 `calibration_train` 上的检测 OBB 单独冻结，不能根据 fixed-dev 的失败结果事后调整。

## 4. 原相机坐标系反投影

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

正式 Raw-opt 路线**不计算** \((u',v')\)，也不读取 `homography_raw_to_plumb`。相机无需数学上严格垂直向下；只要内参正确、实体相机与吊点之间的刚性外参已标定并在运行中保持固定，原相机坐标中的三维点可通过外参转换到吊点坐标。

## 5. 相机—吊点外参转换

令 \(R_{P\leftarrow C}\) 表示从相机坐标系到吊点坐标系的旋转。定义

\[
{}^C\mathbf d_{CP}:={}^C(\mathbf p-\mathbf c)
\]

为“从相机光心 \(C\) 指向吊点 \(P\)”的固定米制向量，并约定吊点坐标系第三轴朝下。则

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

若离线标定证实 \(R_{P\leftarrow C}\approx I\)，且
\({}^C\mathbf d_{CP}=(d_x,d_y,d_z)^{\mathsf T}\)，才可化简为

\[
\widehat X_{PG}=\frac{(u_G-c_x)\widehat Z_c}{f_x}-d_x,
\]

\[
\widehat Y_{PG}=\frac{(v_G-c_y)\widehat Z_c}{f_y}-d_y,
\]

\[
\widehat H_{PG}=\widehat Z_c-d_z.
\]

这里的正负号来自 \(\mathbf d_{CP}=P-C\) 的明确定义，不能把 \(D_x,D_y\) 当作无符号常数随意相加。所谓“直接减去 \(CP\)”只有在轴对齐且 \(CP\) 指相应的有符号轴向分量时成立；欧氏距离 \(\lVert C-P\rVert\) 不能直接从 \(Z_c\) 中扣除。

## 6. 摆角计算

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

## 7. 虚拟铅垂路线的保留边界

虚拟铅垂图像若使用

\[
\mathbf x'\sim KR^{-1}K^{-1}\mathbf x,
\]

则 \((u',v')\)、\((w',h',\gamma')\) 必须与同一虚拟相机第三轴上的深度 \(Z_{CG}^{\mathrm{plumb}}\) 配对。不能用虚拟铅垂像素反投影实体相机光轴深度 \(Z_{CG}^{\mathrm{opt}}\)。

本项目保留 Plumb/Plumb 作为 Webots oracle 或姿态补偿消融，但不作为当前部署主线。历史上的 Plumb/Opt 拟合结果只用于复现实验，不再作为正式几何证据。

## 8. Webots 真值与 GPS

- `pivot_gt_gps` 应位于真实悬挂铰点 \(P\)，不应为了简化公式移动到相机光心。
- `grab_reference_gt_gps` 应位于四个顶梁参考角点的三维均值 \(G\)，并随抓斗刚体运动。
- GPS 返回设备原点的世界坐标；场景编辑器显示的局部彩色轴不改变世界坐标读数。
- GPS 与 Supervisor 仅用于交叉验证 \(C,P,G\) 和真值，不作为训练/推理输入。
- 单个中心 GPS 不能证明参考平面姿态；平面法向、四角点共面误差和中心误差应由四角点计算。

仿真真值至少保留：相机内外参、\(C/P/G\) 世界坐标、四角点世界坐标、原图 GT OBB、\(Z_{CG}^{\mathrm{opt}}\)、\(H_{PG}\)、\(L_{PG}\)、摆角、时间戳和必要审计字段。

## 9. 当前实验结果与严格证据边界

### 9.1 Raw-opt fixed-dev 组件结果

数据职责：

- `calibration_train_03_tilt_obb`：981 帧，只拟合参数；
- `fixed_dev_01_unseen_depth_swing`：893 帧，只选择/诊断公式；
- 未读取任何未知序列来生成本次 Raw-opt 参数。

GT OBB 条件下结果：

| 模型 | MAE (m) | RMSE (m) |
| --- | ---: | ---: |
| M0 对角线幂律 | 0.10764 | 0.11361 |
| M1S 短边针孔 | 0.04490 | 0.05053 |
| **M2S Raw-opt** | **0.03417** | **0.04045** |

M2S 的 AbsRel 为 0.001957，P95 绝对误差为 0.07679 m，最大绝对误差为 0.10716 m。相对 M0，MAE 降低约 68.25%；相对 M1S，MAE 降低约 23.89%。

这些数值只证明“Webots 真值 OBB + Raw-opt 尺度公式”的组件性能，不证明：

- 当前检测模型输出 OBB 后仍有相同精度；
- 未知序列泛化已经通过；
- 真实视频具有米制深度或绝对摆角精度；
- 已完成部署。

此前 Unknown-02 与 Plumb/Opt 路线的结果属于旧坐标契约，不能直接迁移为 Raw-opt 的未知序列结果。新的未知序列只能在公式、检测模型、有效性门和评价规则全部冻结后执行一次。

### 9.2 正式 native-S14 的 calibration-train 检测 OBB 审计

本次审计使用 `crane_symeood_formal_dino_native_s14_v1.py` 和
`source_safe_interpolated_head.pth`，在 `calibration_train_03_tilt_obb` 的 981 帧上运行。该模型属于冻结 DINOv2 native-S14、ROI classifier `alpha=0.5`、S7 disabled 的正式 source-gated 组件；本次运行仅是 **calibration-train 接口诊断**，不是 fixed target-dev、未知序列或部署结果。

协议检查结果为：处理 981/981 帧、检出 981/981 帧，预测坐标已恢复到原图，`coordinate_contract=raw_opt_v1`，未使用 `homography_raw_to_plumb`，未重拟合 Raw-opt 参数，未调阈值，也未读取未知序列。在同一评价入口中，GT OBB 的 Raw-opt MAE 为 0.04090 m、RMSE 为 0.05343 m，说明修复后的坐标配对和代码路径能够正常闭合。

但是，native-S14 的预测框几何不适合直接驱动当前双尺度公式：

| calibration-train 指标 | 结果 |
| --- | ---: |
| 检出率 | 100%（981/981） |
| mean RIoU | 0.6198 |
| 中心误差均值 / P95 | 6.38 / 12.71 px |
| 长边相对误差均值 | -7.50% |
| 短边相对误差均值 / P95 | +3.98% / +42.78% |
| 方向角绝对误差均值 / P95 | 5.29° / 20.82° |
| \(q\) 落入 GT 拟合支持域 | 52/981（5.30%） |

929/981 帧的检测 \(q\) 超出 GT OBB 拟合支持域。由于公式包含

\[
\exp(\beta q^2),
\]

长短边比例误差会被指数项放大，直接代入会产生无物理意义的极端深度。因此，当前结论应表述为：**native-S14 在该 Webots calibration-train 上能够稳定检出抓斗，但其 OBB 长短边几何精度不足以直接支撑 Raw-opt 深度；这不是 Raw-opt 坐标修复失败，也不能据此笼统断言模型的所有检测能力都失效。**

该模型不进入 fixed-dev 深度评价。置信度也不能单独充当有效性门；越过冻结支持域的帧应输出 `depth_valid=False`，而不是截断 \(q\) 后继续声称有效深度。

### 9.3 检测器责任与证据顺序

检测器误差会同时改变 \((u_G,v_G,w,h,\gamma)\)。因此必须依次报告：

1. GT OBB + Raw-opt：隔离尺度公式误差；
2. 检测 OBB + Raw-opt：测量检测误差传播；
3. 未知序列：验证冻结系统的序列泛化；
4. 真实视频：只报告连续性、有效输出率、时延和资源占用。

旧 K1 fixed-dev 上曾出现大量检测输出 \(q\) 越过旧标定范围；本次 native-S14 calibration-train 审计进一步确认，长短边比例失配会被指数项放大。两者属于不同模型和不同协议层，不得合并计数或指标。检测输出有效性规则只能在 calibration-train 上设计并版本化，不能利用 fixed-dev 失败结果事后调门。

## 10. 七项问题的处理结论

| 问题 | 当前处理 |
| --- | --- |
| 虚拟铅垂 OBB 与实体光轴深度混配 | 已取消；主线改为 Raw-opt |
| \((u',v')\) 与 \(Z_{CG}^{\mathrm{opt}}\) 反投影 | 已取消；改用原图 \((u_G,v_G)\) |
| \(Z_c\) 同时表示相机深度和悬挂垂距 | 已拆分为 \(Z_c:=Z_{CG}^{\mathrm{opt}}\) 与 \(H_{PG}\) |
| 只补偿水平偏移、遗漏垂向偏移 | 外参使用完整三维 \((d_x,d_y,d_z)\) |
| \(D_x,D_y\) 加减号不明确 | 固定定义 \(\mathbf d_{CP}=P-C\)，统一先相减再旋转 |
| 部署依赖 Webots Supervisor 姿态/单应矩阵 | Raw-opt 推理不读取单应矩阵；仅需离线固定外参 |
| 将旧 Plumb/Opt 指标当作新协议证据 | 明确降级为历史结果；Raw-opt 重新分层验证 |

这些问题在**公式、配置和评价入口的定义层面**已解决，并已由 calibration-train 的 GT OBB 闭合结果验证代码路径；但检测 OBB 有效性门、fixed-dev 检测 OBB、未知序列和真实部署仍待实验，不能写成整个端到端系统已经验证完成。

## 11. 下一步唯一允许的实验顺序

1. 不让当前 native-S14 进入 fixed-dev；它保留为 calibration-train 失败诊断；
2. 在同一 `calibration_train_03_tilt_obb` 上比较 K1/SymEOOD 原始框与统一 SymEOOD+DINO 各候选框的长边、短边、长宽比和 \(q\)，不得读取 fixed-dev 选框源；
3. 选择并冻结唯一的框几何来源，同时冻结越界时 `depth_valid=False` 的规则；
4. 不调参数地运行 `fixed_dev_01_unseen_depth_swing`，分别报告 GT OBB 与冻结检测 OBB；
5. 若最终检测模型或框来源发生变化，重新执行 calibration-train 与 fixed-dev，旧结果只保留为对应模型版本的诊断；
6. 只有检测模型、Raw-opt 参数、有效性门和外参均冻结后，才注册并运行全新的未知序列；
7. 真实视频只做迁移稳定性和部署性能评价。

不得重复使用已经暴露结果的旧未知序列来调整新 Raw-opt 系统，也不得把 source gate、fixed target-dev、未知序列泛化和部署结论混写。

## 12. 论文可用的简要表述

本系统直接复用旋转检测器在原始图像中输出的 OBB 中心、长短边与方向角，结合相机内参和抓斗已知刚体尺寸构造长、短轴针孔深度候选。以二者的对数比作为各向异性透视缩短指标，通过两参数解析校正式估计抓斗参考中心相对实体相机的光轴深度。随后，利用离线标定且运行期间固定的相机—悬挂点三维外参，将抓斗中心从相机坐标系转换至吊点坐标系，得到水平偏移和铅垂距离，并通过 `atan2` 计算两个方向的有符号摆角。该流程不依赖 PLC 或 Webots 真值输入；Webots GPS 和 Supervisor 仅承担仿真真值生成及坐标审计职责。
