# v13 容量校准可识别性与 Oracle 门控性能空间诊断（开发集，2026-10-12）

> **时间顺序**：本文件先于本轮 v13 诊断脚本的执行而归档，但原 v8-v12 的方法、同47块目标电芯的性能、以及3C负迁移已公开、已被研究者反复探索。因此本文件**不是独立数据的预注册，也不是对新算法的确认验证**。

## 核心问题

v12 中只利用两块已标注电芯交叉验证拟合全局lambda的性能不如完全残差 Ridge，尤其当已标注电芯内部LOCO一致但未标注电芯本身无需校准时失灵。此轮不再添加可部署算法，不通过测试误差调超参，而只回答：

1. 在既有 `source-only` 与 `source+Ridge` **两名固定专家**之间，如果理想地知道测试误差，选择性应用校准究竟存在多少理论性能空间？
2. 此空间集中在个别电芯、折还是整个工况？只允许“对全工况、对同折、对同电芯”选择其一时有何差异？
3. 同一个标注电芯组合、同一个训练所得校准器与lambda，能否在不同测试折上对应明显不同的转移损益？

## 唯一数据与固定指标

- 输入：`results/v12/rw_to_{2C,3C,R2_5,R3,Sim_satellite}_loco.json` 中已冻结的每个测试电芯 `source_predictions`、`records[].ridge`、`records[].shrink`、`selected`、`heldout`、`lambda`；不训练新模型、不新增测试折、不下载新数据。
- 仅用既有目标训练池各选2块标注电芯，N10/20/30固定预算，选样 `pred_quartiles` / `random`（随机每折5次，且5次不作为额外电芯）。
- 以**N30 Pred-quartiles + Ridge**作为主要诊断对象，N10/N20及随机组只作预定敏感性分析。
- 正式误差：先单个目标电芯全寿命容量MAE (mAh)，再目标电芯等权，最后五工况等权。

## 三类使用真实目标测试容量误差的事后 Oracle（不可部署）

令 `S_i` 是被测电芯i的零样本源模型全寿命MAE，`R_{fi}` 是其所属fold的已冻结Ridge校准全寿命MAE，都是**外层测试标签事后算出的指标**：

1. **Domain oracle**: 对整个目标工况，一律选源模型或者该工况全部Ridge的平均较优者：`min(mean_i S_i, mean_i R_i)`。随机组先对每块测试电芯的5次Ridge分数平均，再算宏值。
2. **Fold oracle**: 对每个 `fold × selection repeat` 中全部被测电芯共同选择源或Ridge：`min(mean_{i in fold} S_i, mean_{i in fold} R_fi)`；折内如有2-3电芯须按电芯数加权，不能粗略按折等权。随机5次先在每个被测电芯上平均结果，再计算工况均值。
3. **Cell oracle**: 针对各个 `fold × repeat × test cell` 使用 `min(S_i,R_fi)`，再按电芯等权、五工况等权。它只有两个专家（不是从更多alpha或非线性模型中调优），基于**测试标签作弊**的理想性能，仅用来衡量理论上限，绝不可当作部署基线、合法迁移方法或论文性能成绩。
4. 同时计算LOCO旧收缩作为已有对照，及Ridge相对source的负迁移数量、分布与正负收益两侧幅值，保留原始逐电芯得分和折决策列表。
5. 单一目标域（尤其3C）以 `cell_oracle` 与 `fold_oracle` 的差别衡量折内电芯异质性的性能代价；`fold_oracle` 与 `domain_oracle` 的差别衡量同域跨折异质性。不可把数值差称为因果结论。

核验 `cell_oracle <= fold_oracle <= domain_oracle <= source_only` 以及 `domain_oracle <= original_Ridge` 的平均值不等式（数值容差1e-8 mAh）；random组在每个repeat上各自计算，整体依照重复等权平均，domain oracle定义为选源或所有repeat的Ridge均值较优者，不等式同样成立。

## 事先固定的报告与边界

- 优先报告N30 Pred-quartiles的五工况表：`source, ridge, LOCO, domain oracle, fold oracle, cell oracle`、严格负迁移电芯数量、折内 oracle 选择情况；N10/N20和Random给简表。
- 事后 Oracle 是**含有外层测试标签的信息泄漏分析器**，绝不能反向给我们的实际方法选择 `lambda` 或挑一个新方案。它只能回答“存在多少选择性校准潜力”，**不能证明特征或两块标注电芯提供了可识别的 gate 信号**。
- 与v12原始逐电芯输入的校验：按每折每次原始记录的测试ID、选样ID及N预算核对；和 `results/v12/summary_five_domains.json` 的源、Ridge、LOCO对应MAE对齐，阈值1e-5 mAh。
- 如果oracle空间大，仅可提出在**独立外部电芯数据上预先检验的研究假设**，不准在已知47电芯上搜新特征距离gate并将其性能称作独立验证。
- 同时明确“Che 2025的随机局部曲线片段不等于随机选择目标电芯”；SOH百分点与mAh分离；N10必须讨论选样需12次循环无标签特征导致的时间序列回溯成本。
