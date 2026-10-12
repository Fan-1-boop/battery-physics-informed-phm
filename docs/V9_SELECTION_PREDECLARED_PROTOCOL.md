# v9 选样方案追溯审计（非预注册文件）

**审计日期：2026-10-12。重要说明：v9 数值结果已经在此文件创建前存在。** 本文件是事后协议归档，不构成预注册；不能据此声称候选方法在测试结果揭晓前冻结。

## 固定实验条件

- 8 个 RW 电芯训练同一 Ridge 源模型，47 个目标电芯，5 种目标工况：2C、3C、R2.5、R3、Sim_satellite。
- v8 电芯级五折，RW 训练均值和标准差归一化，输入仅使用公开 XJTU 的 6 项逐循环充电统计特征。
- 目标训练候选池中每次只选 2 个电芯；每个被选电芯使用前 25% 全寿命容量标签，Ridge 残差 alpha=10。
- 未标注选样仅使用**候选池**前 12 循环特征及 RW 源预测，测试电芯特征/标签不得参与。
- 5 次随机选样；确定性方法每折只有 1 次选择；先电芯等权，再五目标工况等权汇总，不能把重复选样视为独立样本。
- 输入不是 Che 2025 的随机局部放电电压 V(Q)，不是真实电极 latent；目标是放电容量 mAh 而不是 SOH 百分点。

## 已有真实 v9 结果中执行的选样方法

真实方法包括：random、centroid+farthest (v8)、predicted capacity extremes、predicted capacity quartiles (25%/75%)、1D predicted capacity k-medoids、6D feature k-medoids、RW leave-one-battery-out ensemble disagreement top2、uncertainty-weighted feature medoids。此外有早期源域 novelty extremes 与 prediction median+extreme 的 screening。

数值见 `results/v9/summary_five_domains_and_paired_bootstrap.json` 和 `results/v9/summary_five_domain_selection.csv`。预测容量 quartiles 的五目标宏平均为 41.417 mAh，random 为 41.733 mAh，提升 0.317 mAh，47-cell 配对探索性 bootstrap CI 包含 0。**不能宣称统计显著优于随机或相对于外部文献 SOTA。**

`feature_kcenter` 和 `kernel_herding` 曾在 2026-10-12 的方法设计说明里被列为候选，但**没有在这里找到五目标实测或逐值验证**，不属于 v9 已完成成果，也不混进结果表。若未来尝试，必须另立 v10 协议，在测试前冻结，而不是追加方法追求同一数据集上最优。

## 文献关系

- Lu et al., *Nature Communications* 2023: https://doi.org/10.1038/s41467-023-38458-w （源域分布修剪和集成，非目标电芯选样直接复现）。
- Sener & Savarese, ICLR 2018: https://openreview.net/forum?id=H1aIuk-RW （k-center core-set 思想，非电池论文，且 v9 尚未实测 k-center）。
- *Nature* 2026 Discovery Learning: https://www.nature.com/articles/s41586-025-09951-7 （active learning + physics-guided，非本实验模型的直接复现）。
- 2024 *npj Materials Sustainability* battery probabilistic review: https://www.nature.com/articles/s44296-024-00011-1 （随机/分层采样和电池概率学习背景）。

## 复现与独立性

首轮探索之后，必须固定一个策略并在独立数据或预先隔离测试集上确认；进一步在原 47 电芯同一五折反复筛选算法会导致 benchmark 过拟合。GitHub 脚本 `experiments/rw_selection/benchmark.py` 提供实验定义，原作者数据需按固定 commit 下载；历史计算产生的 per-cell JSON 需做单独复算审计。
