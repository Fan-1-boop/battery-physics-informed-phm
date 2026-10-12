# v9 审计结论：预测容量选样与文献启发基线（2026-10-12）

**性质：既有 v9 真实数据实验归档的独立数值/协议复核；不是 2026-10-12 新执行的55电芯训练。** 历史 v9 结果此前已在本仓库保存。正式样本：RW8 源，五目标47个目标电芯，目标电芯级5折；仅标注被选中两块目标电芯各自前25%容量标签，残差 Ridge alpha=10，其余不变。

## 1. 选样基线数值（容量 MAE，mAh）

| Policy | 2C | 3C | R2.5 | R3 | Satellite | 5域等权 |
|---|---:|---:|---:|---:|---:|---:|
| Random | 27.79 | 44.03 | 42.50 | 46.63 | 47.71 | **41.73** |
| Centroid + farthest | 28.23 | 43.16 | 42.33 | 54.19 | 61.20 | 45.82 |
| Pred min/max | 26.97 | 50.94 | 42.38 | 47.52 | 54.15 | 44.39 |
| Pred 25%/75% quantiles | **26.76** | **43.46** | 42.22 | 47.97 | **46.67** | **41.42** |
| Pred 2 medoids | **26.13** | 45.87 | 42.97 | **43.85** | 56.29 | 43.02 |
| Feature 2 medoids | 28.11 | 48.70 | **42.04** | 43.94 | 56.40 | 43.84 |
| RW8 LOBO disagreement top-2 | 28.95 | 65.29 | **41.14** | 56.16 | 55.57 | 49.42 |
| Disagreement-weighted feature medoids | 26.64 | 48.37 | 41.75 | 50.72 | 57.52 | 45.00 |

早期有 source novelty 选择法，五域等权 41.82 mAh，未超过 random 41.73 mAh。

**差异大小与显著性界限：** 五工况 Random−Pred quartiles = **+0.317 mAh**（约0.76%），4/5 个工况 MAE 略有改善；47电芯分工况重新抽样的10,000次描述性 bootstrap CI为约 **[-1.98, +2.68] mAh**，包含0；此前单独归档 bootstrap 区间约 [-2.00,+2.63] mAh。由于所有样本共用RW模型/训练池且在此数据集上已经尝试多种选样法，这都不是确认性证据。**不要写出“预测容量分层选择显著优于随机标注”。**

## 2. 协议与数值完整性核验

读取所有五目标 `results/v9/rw_to_*_selection.json` 并逐条核对 `results/v8/`：

- 47个目标电芯均覆盖且每目标电芯每方法有预定次数的测试预测；
- 每次标注恰为2个互异的目标训练池电芯；未发现同折已标注与测试电芯ID重合；
- Random的每个测试电芯对应5个选择重复，确定性方法只有1个，汇总没有误当独立电芯；
- Random / Centroid+Farthest 五工况各与v8历史结果的汇总误差在1e-7 mAh内一致。
- 审计源JSON GitHub blob、每工况的测试电芯选样成本以及逐电芯配对统计见 `results/v9/audit_20261012_cellwise_costs.json`。

注意：这是**归档数据一致性/ID层隔离核验**，不是在新容器独立联网下载55个原始CSV并逐值重新训练模型。

## 3. 一个实际部署瓶颈：25%寿命标签成本

预测四分位选样时，五目标每折2个被标注电芯的前25%容量标签平均需求约：

| Target | Capacity-labeled cycles per fold, total for 2 selected batteries |
|---|---:|
| 2C | 186 |
| 3C | 116 |
| R2.5 | 262 |
| R3 | 293 |
| Sim_satellite | 378 |

标签数在跨工况间不一致；而且“未来全寿命前25%”无法在实验开始时实时获知。**两块标注电芯≠只有两个容量真值**，更不能将此结果当成零标签在线算法。

## 4. 方法与相关文献的关系

- Lu et al., *Nature Communications* 2023（https://doi.org/10.1038/s41467-023-38458-w）：研究源域样本分布的影响；本实验是选目标候选电芯，不等同论文方法。
- Sener & Savarese, ICLR 2018（https://openreview.net/forum?id=H1aIuk-RW）：core-set/k-center 的多样性思想；本轮没有执行 k-center原实现。
- Wang et al., PINN4SOH *Nature Communications* 2024（https://doi.org/10.1038/s41467-024-48779-z）：提供XJTU数据与物理约束网络基线，但目前实现的是轻量Ridge和距离式选样，**不是**原PINN4SOH的精确复现。
- *Nature* 2026 Discovery Learning（https://www.nature.com/articles/s41586-025-09951-7）：物理引导主动学习应用于电池设计/寿命试验，是下一阶段实验成本设计参考，不是本轮算法复现。
- *Batteries* 2024 sample transfer（https://doi.org/10.3390/batteries10050156）：TrAdaBoost.R2对源域样本重加权，不是本轮目标电芯选样。

## 5. 硕士论文最节制的下一轮计划

**不要继续在同样五折上大量追加“选样启发式”并择优。** 先冻结一个低成本选择策略（predicted 25/75% quartiles）作为候选，在固定前10/20/30个循环容量标签的现实预算下，直接比较五个策略组合：

1. RW source-only；
2. Random two cells + mean residual offset；
3. Random two cells + residual Ridge；
4. Pred-quartiles two cells + mean residual offset；
5. Pred-quartiles two cells + residual Ridge。

留出独立划分做验证，并清楚标记前述所有同数据集探索是“开发集”，不能再拿它当完全独立检验集。若四分位策略仍没有可重复的优势，应停止主动选样支线，专注更有实证依据的低标签容量校准方法。
