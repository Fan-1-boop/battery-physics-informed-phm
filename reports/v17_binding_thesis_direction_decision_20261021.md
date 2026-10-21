# v17 最终科研取舍报告：继续统一校准主线，停止代表性选样创新（2026-10-21）

## 最终决定（不得改成模糊措辞）

**研究问题层面：GO。继续研究RW源模型保持的少目标标签统一连续校准，并将其作为一个收缩后的硕士论文主方向。**

**具体创新点取舍：**
1. **停止把“源模型预测容量25%/75%四分位目标电芯选择”作为论文方法创新**：在本轮未见88块电芯上，三cohort全部**随机选样比四分位选样的Ridge更好**，平均四分位相对随机劣化0.286pp，bootstrap描述性95%区间[-0.386,-0.190]。虽然v16四分位往往更好，跨数据集结论相反，不能声称可靠的新主动标注机制。选择策略仍可在消融表中保留；默认简单随机。
2. **保留统一连续残差校准方法路线**：本轮唯一固定新核`K=1+RBF-X`在三个新队列均优于Ridge，主N30四分位cohort宏nMAE从7.171降至3.203%，新方法负迁移1/88 vs Ridge20/88，全部事前门槛通过。
3. **暂不宣称“全局+局部核组合”本身完成了新算法创新**：和固定v14 RBF-X相比新方法仅3.281→3.203%，改善0.079pp；事先bootstrap描述性95%区间[-0.064,0.217]，包含0，且MIT_2018A的RBF-X显著更好(1.445% vs 2.825%)。高质量前人论文中KRR、常数核+RBF加和、残差适配都是已知；因此`K=1+RBF`不是被证明超越前人和稳定提高的原创算法。**科研GO是继续以“局部+全局统一校准与负迁移权衡”作为集中问题开展论文写作和有约束的强基线复核，而不是允许继续无止境微调kernel/lambda**。
4. 用户要求本轮做取舍，**已经做出取舍，不以“继续尝试下一种公式”作为结论**。若学校硕士论文要求独立算法创新，当前版本还需要对照近期同行工作并明确新约束的理论动机，不能用v17 GO取代原创性判断。旧v8-v16保留作开发/泛化证据，v17不再用于后验算法选择。

## 1. 执行前冻结的Go/No-Go判定

最先提交仅一页科研决策的固定实验协议：`docs/V17_FINAL_GO_NO_GO_PROTOCOL_20261021.md`，commit `f0dd3c93694229ed36e62790df64b67fe2a3e0b7`。随后冻结全88份未用原作者数据CSV的路径/尺寸/Git SHA，commit `e1e3515d91f5b63d4a24d376162036c0e76d1ab9`。以上均在首次v17全数据测试前提交。**没有因测试结果调整方法、核权重、折分或门槛**。

预先冻结`RW8→目标`五折、两块目标训练池电芯各前10/20/30容量标签(20/40/60)、Random5次与预测四分位1次，六维充电统计`(0,4,5,8,12,13)`、统一RW尺度、源Ridge alpha=1、目标残差Ridge alpha10、RBF-X rho=.05。唯一新核`K_GL=1+K_RBF-X`，同rho=.05，各标注电芯总权重1/2。单一连续校准`Qhat=f_RW(x)+(1+k_RBF-X(x,X_L))@alpha`，`alpha=(1+K_RBF-X+rho diag(1/w))^{-1}(y_L-f_RW(X_L))`。修正可为零/弱/强，不另设“是否校准”分类器。

**预定校准GO条件全部通过**：
- 比Ridge至少降低0.5百分数点nominal nMAE且配对描述bootstrap区间下限>0：本轮是+3.969pp、区间[3.824,4.106]；
- 至少2/3新cohort宏MAE不差、没有队列劣化>1pp：三组均优于Ridge；
- 总负迁移电芯数不超过Ridge：1 vs 20；
- N20四分位和N30随机不能同时严重差于Ridge：均明显优于Ridge。

**预定选样单独GO条件全部失败**：
- N30四分位相对随机至少提升0.2pp：实际**-0.286pp**；
- 至少2/3新cohort改善：实际**0/3**；
- bootstrap区间下界>0：实际[-0.386,-0.190]，方向反转。

机械脚本输出`GO_UNIFIED_CALIBRATION`、`representative_acquisition_only_GO=false`。注意**这个GO是相对预先约定的旧Ridge强基线，并非从外部对照已证明该新核超过普通RBF基线；后者不成立。**

## 2. 新目标电芯与严格独立性

数据来源为Nature Communications 2024 PINN4SOH公开repo `wang-fujin/PINN4SOH`、固定上游commit`cc3cc5053caf38f04e0665f7f88cb109144d035e`，所有CSV均预处理为16充电统计特征+容量Ah；我们仅用六维现成数据，不宣称运行其PINN网络或电极物理真实状态。

新目标**88份全部有效，排除0**：
- TJU Dataset_3_NCM_NCA：混合NCM/NCA，9块，官方标称2.5Ah；
- MIT 2017-06-30批：43块，官方标称1.1Ah；
- MIT 2018-04-12批：36块，官方标称1.1Ah；
- 源XJTU RW8官方标称2.0Ah。统一通过元数据`Qtilde=Q_Ah*(2.0/Qnom_Ah)`换尺度。该变换没有使用外层测试容量真值拟合伸缩比例。

数据路径、原始字节Git blob SHA、额定容量预先固定并逐个执行校验。未见的新队列来自先前不曾在v8-v16评分或优化的原作者其它批次，但三队列之间2个MIT批次相关、来自同一预处理作者repo，**不能把88块视为88个完全独立实验室或将3个队列当作独立化学体系**。5折测试电芯ID互斥；同cohort各折的训练候选池存在交叠。Random5次仅对同一测试电芯平均，不伪造5倍样本数。

## 3. 正式主终点：N30、四分位选样

评估全寿命实际放电容量MAE，以官方标称容量为分母的`nominal nMAE (% nominal)`为跨不同容量队列主终点，**不是SOH真值误差的百分点**。每cohort内部按实际测试电芯等权，最终3-cohort等权。容量训练真值每折只来自两块已标注目标电芯各前30次循环（共60条）。

| 全新目标队列 | Source-only | Mean offset | 残差Ridge | v14 RBF-X | v17 全局+局部 GL-KRR |
|---|---:|---:|---:|---:|---:|
| TJU混合正极(9),2.5Ah | 7.592% | 12.675% | 12.239% | 6.061% | **5.112%** |
| MIT 2017B(43),1.1Ah | 9.671% | 4.893% | 4.664% | 2.338% | **1.671%** |
| MIT 2018A(36),1.1Ah | 6.263% | 4.749% | 4.611% | **1.445%** | 2.825% |
| **三个队列等权平均** | 7.842% | 7.439% | 7.171% | 3.281% | **3.203%** |

额定容量下对应真实容量MAE：TJU MIX Source189.80、Ridge305.98、RBF-X151.53、GL-KRR127.81mAh；MIT 2017B 106.38/51.30/25.72/18.38mAh；MIT 2018A 68.90/50.72/15.89/31.07mAh。

**警惕误读**：v17 GL-KRR vs Ridge增益3.969pp，RBF-X vs Ridge本身也很强(+3.890pp)。两者的新法差只有0.079pp， bootstrap CI跨零；这**不支持**“新增全局偏移核显著解决了v16跨数据集失效”。MIT2018A还给出了反例。

此处负迁移定义为**同一测试电芯全寿命MAE比未校准RW源模型高**：Ridge 20/88、RBF-X **0/88**、GL-KRR **1/88**。新法负迁移比RBF略多，不能把少于Ridge的事实包装成超过所有既有方法。

## 4. 所有六组实验（不可只呈现主终点）

三队列等权 nominal nMAE%，越低越好：

| 标注电芯前N次循环 | 选样 | Source | Offset | Ridge | RBF-X | GL-KRR |
|---:|---|---:|---:|---:|---:|---:|
| 10 | Random | 7.842 | 7.562 | 7.396 | 3.327 | **3.166** |
| 10 | Quartiles | 7.842 | 7.674 | 7.504 | 3.322 | **3.245** |
| 20 | Random | 7.842 | 7.408 | 7.164 | 3.304 | **3.127** |
| 20 | Quartiles | 7.842 | 7.545 | 7.342 | 3.310 | **3.224** |
| 30 | Random | 7.842 | 7.249 | 6.885 | 3.261 | **3.097** |
| 30 | Quartiles | 7.842 | 7.439 | 7.171 | 3.281 | **3.203** |

**GL-KRR六组都比RBF-X数值小**，但差异仅0.08–0.18pp量级，不可在小样本不同条件组合上做未计划的多重优效性推断。和Ridge相比六组都明显改善。

**预测容量四分位选样本轮全部3队列N30+Ridge均不如Random：**
- TJU MIX Random11.950% vs Quartiles12.239%；
- MIT2017B Random4.253% vs Quartiles4.664%；
- MIT2018A Random4.453% vs Quartiles4.611%。
三个队列平均Random6.885% vs Quartiles7.171%，四分位**差0.286pp**；描述性配对bootstrap 95%区间[-0.386,-0.190]（效应Random minus Quartiles）。**v16的四分位微小平均收益在未见电芯上没有复现。** 已决定停止将选样作为方法核心，今后研究若需标注两芯，以简单随机作为默认公平对照，而不是继续搜新的四分位阈值。

## 5. 文献对照——科学价值与原创性的不同层面

必须纳入近五年强相关方法讨论：
- Mondal et al., *State-of-Health Estimation of Li-Ion Batteries Using Semiparametric Adaptive Transfer Learning*, IEEE Transactions on Transportation Electrification (2024), DOI [10.1109/TTE.2023.3266499](https://doi.org/10.1109/TTE.2023.3266499)：基于任务相似性的AT-GPR跨数据集迁移；
- Xiong et al., *Transfer learning with composite kernel sparse Gaussian process-aided model for probabilistic state of health estimation ...*, Applied Energy (2025), DOI [10.1016/j.apenergy.2025.126762](https://doi.org/10.1016/j.apenergy.2025.126762)：复合核GP和稀疏目标数据情境；
- *Data sufficiency for transferable lithium-ion battery periodical SOH estimation under resource constraints*, Cell Reports Physical Science (2025), DOI [10.1016/j.xcrp.2025.102901](https://doi.org/10.1016/j.xcrp.2025.102901)：低标签传递数据充足性研究，与本研究成本预算定位密切相关。

**所有前人方法的训练输入、目标SOH定义、数据源、电化学/网络结构和真实标签成本可能不同，不得直接用文献的SOH%与本研究capacity nominal nMAE%排序；若无同条件复现就不要写SOTA。**

当前实验说明小标注跨工况/跨数据集**确有研究价值**，可以作为硕士论文一个窄而明确的方向：“已冻结RW源模型 + 每折两芯固定早期容量标签 + 一个连续校准函数”；但对KRR数学与复合核的算法原创性尚未提供足够证据。应停止开发第二创新点“代表性电芯分位选样”，把余下论文资源集中到单一方法及可靠性/边界报告、强基线公平对照和正式论文章节，不再无限更换核、lambda、gate来制造表面增益。

## 6. 计算诚信、可复现性与限制

实际 [GitHub Actions首次88文件正式执行#39603971268](https://github.com/Fan-1-boop/battery-physics-informed-phm/actions/runs/39603971268) **SUCCESS**：原历史v8/v9/v10/v12/v14/v15/v16单测7+8+5+7+10+10+10，v17新增10，共**67/67 PASS**；88/88原官方CSV SHA和RW8源权重/标准化核验，零排除，全部5折完整评价。

[只读结果压缩二次CI #39604299620](https://github.com/Fan-1-boop/battery-physics-informed-phm/actions/runs/39604299620) SUCCESS，未重新拟合模型，仅从已提交v17完整原始指标输出小型可读JSON`results/v17/compact_final_decision.json`。

Bootstrap 10000，seed20261707，观察效应及探索区间：
- Ridge − GL-KRR **+3.969pp**，CI **[3.824,4.106]**；
- RBF-X − GL-KRR **+0.079pp**，CI **[-0.064,0.217]**；
- Random-Ridge − Quartile-Ridge **-0.286pp**，CI **[-0.386,-0.190]**。
这些是按各cohort测试电芯配对重采样的**描述区间**，共享RW源、交叉折训练候选池重叠与MIT同系列批次的关联均可能使区间偏窄。绝不以其等同全部未来新实验室电芯的正式统计证据。

N10四分位选样先观察到12个无标签充电循环，早期10循环容量标签若没有既有记录，无法在循环12之后前瞻逆向采集；现实部署成本限制未解决。

历史结果v8–v16未改；v17也不能继续被用于后验选择新常数核权重/新任务引导特征后再称它是首次独立验证。

## 7. 读物与复算入口

- 固定去留协议：`docs/V17_FINAL_GO_NO_GO_PROTOCOL_20261021.md`；
- 新88原始作者CSV SHA的冻结清单：`docs/v17_final_unseen_manifest.json`；
- 新候选GL-KRR、完整五折/六组实验及自动决策代码：`experiments/rw_multitarget/final_go_no_go_v17.py`；
- 完整目标电芯折/ID/原始file SHA/误差归档：`results/v17/target_*.json`；
- 冻结大体量正式完整分析：`results/v17/summary_external_cohorts.json`；
- 可直接读取的浓缩结论：`results/v17/compact_final_decision.json`；
- 单元测试：`tests/test_final_go_no_go_v17.py`；
- CI正式执行：[#39603971268](https://github.com/Fan-1-boop/battery-physics-informed-phm/actions/runs/39603971268)。
