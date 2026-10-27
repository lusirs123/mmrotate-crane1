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

## 新底座身份更新：只封存，未开展衔接

2026-10-26：用户授权封存本次重训110 ep20，原选版不变。新合同替代旧合同作为未来链路入口，旧合同及结果保留。训练及VAL／TEST完成不代表旧性能复现通过或下游有效；本轮未训练、推理、加载模型或访问服务器。详情见[封存运行记录](SymEOOD110新底座封存_运行记录_20261026.md)与[核心结果](SymEOOD110新底座封存_核心结果_20261026.md)。


## R006：按用户要求复制旧入口，仅本机修改及控制核查

本轮先读取实际工作区与v2底座封存文件。开始时已有8份DINO配置删除、五份相关文档修改及v2底座合同／封存两份文档未跟踪；这些内容原样保留，不重置、不暂存、不推送。衔接合同及本运行记录本身已有“新底座封存”修改，本轮仅接续更新。

用户再次明确：直接复制旧文件，只修改不同内容。因此复制`train_port_geometry_midpoint_formal_v1.py`为`train_symeood_110_midpoint_v1.py`，没有修改旧入口或重写计算链。修改新检测器身份与strict-load元数据检查、sigma1.5构造、来源／缓存／检查点身份，增加支持与有限验收检查；原采样、update、evaluate等20函数AST相同（标注SHA引用变化单列）。旧模型、损失、几何、sigma、LVSR文件未改。

新增本轮JSON来源合同与一份unittest测试；来源闭包来自原sigma／LVSR共同／延续清单的并集，绑定当前127项文件。历史midpoint五项既有差异保留记录，未替换历史SHA。新入口check在标准库即可运行；数值运行依赖按需导入，CLI及run均不开放正式训练／TEST。

本地使用内置Python及已有numpy执行10项unittest，全部通过：新底座身份、普通Focal配置、来源篡改拒绝、旧版本拒绝、20函数复用、stage隔离、实际计数与空域拒绝、只读check、原平衡采样覆盖。未安装新依赖；torch／MMCV不存在，未做模型数值检查。来源SHA及静态编译通过，结果写入既有110目录下`downstream/preparation_check.json`，没有新增计划MD或按轮次建结果目录。

本轮未执行SSH／SCP、服务器Git拉取、GPU检查、缓存提取、有限更新、训练、VAL／TEST或权重复制；本机优化器更新为0。未提交或推送Git。未启用自动化或训练队列。

LVSR本轮只核查旧源码与绑定点并冻结改动范围：共同fixed_b、原损失C延续，实际父epoch／SHA与有效计数待新midpoint，不复制旧A20、5112／886或639断言。LVSR执行副本尚未适配，不宣称其已可启动。详细边界见核心合同§6。


## R007：服务器有限验收启动；用户要求等待时退出

用户授权仅check→cache→smoke，不进入正式训练。精确部署新增入口、来源锁、v2底座合同和测试四文件，未同步本机配置删除；服务器127项源码、展开配置、ep20权重SHA通过，原Python3.8.20／torch1.13.1+cu117／CUDA11.7／CuDNN8500／MMCV1.7.0／MMDetection2.25.1保持。初始GPU2空闲4756MiB最多，采用物理GPU2、逻辑0，单进程。

首次cache在模型构建前因Python3.8相对__file__与绝对ROOT不匹配停止，GPU模型与更新均为0。原失败收据保留cache_audit/failure.json，原新增入口三文件封存为downstream/source_before_path_fix.tar.gz。仅将Path(__file__).relative_to(ROOT)改为resolve().relative_to(ROOT)，增加对应测试；本机及服务器11项控制测试通过，来源锁随实际修改更新。未修改计算函数。

再次静态check通过，缓存输出沿用同一downstream/roi_cache，提取收据位于cache_audit/attempt_02，日志为cache_attempt02.log。服务器已完成数据身份核对及新110严格加载，正在提取TRAIN标准尺度视图；缓存进程PID396208，SSH执行会话仍运行。已记录实际GPU与16个已加载库路径，未重算这些库SHA，不能声称新增完整库文件核验。

用户随后要求“如果只剩下等待结果的话就退出对话”。尝试安排一次性后台衔接cache通过→smoke，被自动审批拒绝，理由为先前“不设置自动化”偏好及未来自动启动计算／写入；该命令没有执行，没有创建worker或finite_acceptance_status.json，也未绕过拒绝。保持已有cache继续，smoke尚未执行；正式训练、TEST、LVSR均未启动。

当前只是运行中收据，不是完整缓存通过或smoke通过。下次先读cache_attempt02.log及cache_audit/attempt_02/completion.json或failure.json，再决定是否继续已约定的有限smoke；不得把已开始提取写成6003视图已经完成。


## R008：用户明确授权本次缓存结束后自动smoke

用户最新指令明确要求当前缓存结束后自动进行smoke，解决前次自动审批所指的授权歧义。重新提交后审批通过，安排一次性脱离SSH的后台进程，仅等待现有cache PID396208正常结束；核验完整6003视图、来源身份与cache通过收据，GPU2资源足够后执行一次原smoke。四次小头更新全部丢弃，失败即停，不重试、不启动正式训练／TEST／LVSR，不创建Codex定时自动化。

服务器downstream/finite_acceptance_launch.json保存授权、启动PID、完整worker源码及SHA；finite_acceptance_status.json记录waiting_for_cache／running_smoke／complete或stopped_on_error，smoke.log保存日志。此安排本身不代表smoke已经开始或通过。前次被拒绝的命令未执行这一历史事实保留，最新授权后的新命令与其分开记录。


## R009：最新授权改为独立少量样本；有限试运行通过并停止

21:36后读取实际进程／命令，确认worker472288只等待本任务cache396208。先SIGTERM停止worker，再SIGINT停止cache；两者均退出，自动smoke未启动。未向其他任务发信号，旧缓存train_s1.pt及记录保留，未读取作为trial输入；原状态文件更新stopped_by_user。停止收据回传至同一downstream目录。

保留修订前新增入口／来源锁／测试三文件于downstream/source_before_bounded_trial.tar.gz，只在现有入口增加trial，不修改旧计算文件。trial拒绝旧缓存路径，调用原capture_cache提取20视图、保存独立trial_cache.pt并逐张记录身份；复用原smoke的4次丢弃更新及精确恢复检查。配置、模型、原损失、采样、优化器和解码未改；没有新增独立审查框架或文档。未重新启动原全量缓存；原自动smoke及正式阶段未启动。

实际样本：TRAIN real_seq01／sim_seq08各00000—00003、尺度1.0和0.5，共16视图；VAL real_seq07的00001／00002及sim_seq10的00000／00001、尺度1.0，共4视图。日志的1810／748／887仅为原数据集索引，不代表进行全量推理。新110 ep20权重SHA为7f35dacc...52ec054；strict missing／unexpected均0，检测器状态摘要守恒、参数冻结。

本轮trial直接通过，无新增错误或数值修复。实际4次小头更新由optimizer.step完成调用计数，全部丢弃；首两步loss0.00769217／0.00259103，梯度有限、clip通过，第二步stem梯度1.3103e-5。中性输出20视图通过，保存恢复输出一致，继续一步后head／Adam状态一致。显存采样728MiB、allocator峰值343.38MiB。试运行进程595770已正常退出；没有正式训练、TEST、LVSR或后续自动任务。

仅下载completion.json、停止／状态收据和trial日志，未下载试运行张量或权重；回传SHA一致。通过收据SHA acdef79078f22086ec2cda3eada393c7f0a3878c0bfa07b9b5e00074ae83ff28。结果在核心合同§6.3；本轮有限运行无剩余阻断。未来正式运行须再次明确授权，从头生成新的完整缓存，所有此前缓存及trial缓存不进入正式训练；trial更新不得作初始化。未推送Git，保留其他工作区改动。


## R010：用户正式授权midpoint，GPU2新完整缓存→24轮

用户在有限trial通过后明确“可以运行了，使用2号卡”。本轮恢复一次正式midpoint，不训练检测器、不进入TEST／LVSR。读取实际工作区保留8份配置删除及原文档／合同新增，不回滚、不同步无关删除。

GPU2空闲8096MiB，旧cache／trial／自动worker均已停止。只在现有适配入口增加formal阶段，复用已有create_cache、load_cache和原train_formal，调用同一SigmaMidpointHead(1.5)。新增固定目录拒绝旧缓存、检查已通过trial收据及数值依赖未变、启动约束物理GPU2、原预算验证和选版权重双文件系统备份。原20函数复用检查及归一化后的原24轮函数AST比较通过；16项本机／服务器控制测试通过，没有重复执行数值smoke。

服务器保存有限trial版本三文件于downstream/source_before_formal.tar.gz后，精确部署入口／来源合同／测试三文件。127项锁核验通过，试运行通过收据SHA acdef790...83ff28不变。当前来源锁SHA `7e46cb748c3a6bb5312e741d5072cf6bf2396da32c15688951662ca628088628`。不修改运行中的源码或锁。

2026-10-26 23:31:31＋08:00，通过Popen脱离SSH启动PID1089321，固定CUDA_VISIBLE_DEVICES=2、OMP／MKL线程1，逻辑gpu0，正式目录此前不存在。formal_launch.json保存命令、GPU UUID、来源锁SHA与范围，formal.log为日志。正式root为downstream/midpoint/formal，先新建roi_cache并重新提取6,003视图，再全量校验和从seed1703零输出头初始化；试运行的4次更新完全丢弃，旧缓存和trial缓存都不复用。

预计执行一次24轮、每轮完整VAL、原选版及选中头双备份后停止。各阶段异常立即stopped_on_error，无重试或续训；不设置定时自动化。本记录只确认已启动、当前生成新缓存，尚不填报训练完成或效果。原历史输出、上次已中止缓存及trial均保留。本轮未推送Git。


## R011：24轮完成、ep21选版、结果回传与独立分析（2026-10-27）

07:34读取服务器，formal/status=complete，完成时间00:19:14＋08:00，PID1089321已退出。重新生成6003视图缓存、24轮／900步共21600正式更新、全部24候选VAL及选中ep21缓存VAL重放完成。原规则选ep21，不根据平均RIoU、尺寸或TEST另选；TEST／LVSR均未运行。

读取真实计数及缓存manifest：标准TRAIN2558全eligible，半尺度2538 eligible／4缺输出／16中心不匹配；两域eligible3600／1496，VAL2缺输出。原初始头与通过试运行摘要一致，但试运行缓存与权重未读取作正式输入。原日志记录loss0.005394→0.001424，只作优化过程，不以loss替代最终评价。

打包60项分析所需JSON／JSONL与日志，排除pth／pt。临时analysis_receipts.tar.gz约12.16MB，SHA671c27e2975b8f9bf74753e64dcf47eecb339ecad7b37307142e4bdcb2f88c51。回传同一formal目录，逐份manifest SHA一致，原本机启动status另存status_at_launch.json，不覆盖历史状态。未下载张量缓存和候选权重，未新建重复文档。临时本机压缩包核验后删除，服务器临时包按一致SHA再清理；原日志、缓存和权重保留。

本机标准库独立核对21288条VAL帧：真实框中心／尺寸／角度算术、分域分序列汇总、相邻DFR／ACI、MCML／TDR及原选版；173904检查通过。全部21600训练记录梯度／loss有限，裁剪不超原限，24个epoch预算900各轮吻合。未重算多边形RIoU交集，未在本机二次全量核ROI原始张量，边界保留。生成analysis.json、selected_VAL_comparison.csv和all_epochs_VAL.csv。

选版ep21 SHA7aaaacc6c666116e466c9a99ac59d71acf7f77312dc86de224fcf965569fd9ac，18900次更新。独立SSH回读原选中与项目磁盘／home两备份三份SHA一致；strict missing／unexpected=0收据保留，两文件系统2082／66306。没有额外模型运行或GPU检查。

主结论：总体原生RIoU0.848678→0.862338，中心MAE3.6162→3.2016px，Sim纯角度RMSE1.6593→1.0881；中心880、定位883、输出885均不变。总体DFR2.776733→2.924277，Real长边及Sim短边MAE恶化，联合双边≤10%632→622；中心恢复seq07_00149同时新失败00181。原审阅五旗标未通过，不自动采用／晋升正式M。详表和轨迹见核心§7，结果只来自新110链路的VAL，不继承历史B／M成绩。

本轮只分析和记录，不运行TEST／LVSR、不调整epoch或参数、不自动重训、不修改论文正文，未提交或推送Git。停止等待下一步明确指令。


## R012：固定ep21少量VAL衔接→TEST；GPU0完成并停止（2026-10-27）

用户明确授权固定新110 ep20 SHA7f35dacc...52ec054及midpoint ep21 SHA7aaaacc6...69fd9ac，只执行少量代表VAL核对与1440帧TEST，不训练、不重选、不全887在线VAL、不重建缓存或启动LVSR。新增eval_symeood_110_midpoint_ep21_test.py、一个身份合同和4项控制测试；不修改已封存训练入口或计算模块。新入口直接调用旧sigma15评价模块、原capture／prediction_row／verify_val_rows／fixed_test_inputs／test_summaries。来源130项及训练选版收据绑定，固定模型不提供CLI更换参数。首次本机测试曾将dict.update误判为优化更新，修正测试限定对象后4项通过；不涉及模型公式修复。

精确部署三个新增文件，服务器4项测试通过。起初GPU2已通过8帧VAL并完成300帧TEST时，用户明确要求改用GPU0。确认进程3343058命令归属后SIGINT停止，保存test_ep21_stopped_gpu2目录、日志、入口源码包及test_gpu_change.json；300帧预测不参与最终统计。只修改物理GPU0绑定与相应来源锁，GPU0空余8071MiB，重新执行同一固定模型的8帧VAL及完整TEST，没有按中途分数重试或改版。

GPU0进程3356155：8帧（两域正常、两帧已有缺输出及变化代表）框／分数／输出／candidate与回退决策全部对齐原ep21行，通过后才读取TEST。随后完成Real868＋Sim572共1440帧，一次特征提取／帧，原三个native head调用／帧，B和midpoint同前向取得。eval／no_grad、参数／缓冲摘要前后一致，优化器更新0，源配置／阈值／解码／回退不改。完成耗时373.37s，显存采样710MiB，allocator峰值343.45MiB。未再运行模型。

单次完整性收据核对：1440唯一帧、3序列数、输出／分数守恒、1440特征提取／4320native调用、来源／结果SHA一致。打包11项JSON／JSONL／必要日志，临时包SHA a6b2b56c62c22f50e94a78269a56f836cf9825831aa0b0c37aeada3e36e542e3，回传同一test_ep21目录并核验；本机临时包删除。原生预测保存test_rows.jsonl。沿用既有纯标准库compare函数算同域／序列的时序与变化帧，补充输出分母MAE／P95／signed relative bias与联合≤10%数，生成metrics_comparison.csv、changed_failure_frames.csv与analysis.json。没有新全量审查框架或重复文档；RIoU沿用原生交集结果，未本机重算交集。

TEST：中心1427→1428、定位1419→1426、输出1431不变、全帧RIoU0.836162→0.855867、中心MAE2.302615→1.864374px、Sim角度2.020227→1.359704deg；DFR2.400009→2.494920、ACI0.944263→0.952772，MCML5／2.3333不变，TDR100%。总体长边MAE4.3944%→4.0949%，短边5.0442%→5.8232%，联合尺寸1205→1162；Real联合少9、Sim少34。恢复定位7帧／中心1帧，均无新失败；尺寸恢复65、新失败108。详表、序列代价及帧号见核心§8和CSV，不用旧TXT取整值混配。

结论为固定版本定位／方向收益与DFR／尺寸代价并存。TEST已多次暴露，只是冻结同条件比较。无训练、epoch重选、参数调整、LVSR／深度或自动后续任务；不改正式M身份或正文。GPU0进程退出后停止等待指令，未提交／推送Git。临时分析包完成本地核验后清理，原生预测、日志和全部重要历史材料保留。
