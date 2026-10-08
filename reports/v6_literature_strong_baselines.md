# v6 — 近五年强基线文献审计与统一低标签对照实验

日期：2026-10-08  
状态：**REAL-DATA EXPLORATORY BENCHMARK — not external validation or exact original-paper implementations**。

## 1. 研究动机与重要纠偏

用户建议引入近五年高质量论文的强基线并借鉴实验设计。文献审计发现：

- **2024 Nature Communications** Wang et al. *Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis*，DOI https://doi.org/10.1038/s41467-024-48779-z，公开代码 https://github.com/wang-fujin/PINN4SOH；它处理时间演化的退化状态和短期充电特征，不能当成当前单次 20 min 放电输入的同任务误差。
- **2024 IEEE TNNLS** *A Transfer Learning-Based Method for Personalized State of Health Estimation of Lithium-Ion Batteries*，DOI https://doi.org/10.1109/TNNLS.2022.3176925：CNN+针对回归的 MMD 跨域对齐，需要真实源域原始充电数据。
- **2025 Energy** Dou et al. *Cross-domain state of health estimation for lithium-ion battery based on latent space consistency using few-unlabeled data*，DOI https://doi.org/10.1016/j.energy.2025.135257：无标签目标域的 AAE / hidden-feature-similarity，比纯 Ridge 更接近本研究的目标，**无法因为任务定义不一致而直接复制其报告 RMSE 进行排名**。
- **2025 Energy** *A cross-material lithium-ion battery state of health estimation method based on three-stage domain adaptation*，DOI https://doi.org/10.1016/j.energy.2025.139376：MMD–SMDA–FT，比较跨化学体系和随机片段，仅有本地 Dataset2 没法作相同外部实验。
- **2024 IEEE TTE** *State-of-Health Estimation ... Semiparametric Adaptive Transfer Learning*，DOI https://doi.org/10.1109/TTE.2023.3266499：AT-GPR 含自适应跨任务核；本轮只运行**普通 RBF GPR**作为方法族强基线，绝不叫 AT-GPR 原文复现。
- 其余候选见 [`docs/LITERATURE_BASELINE_REGISTER_v6.md`](../docs/LITERATURE_BASELINE_REGISTER_v6.md)，含 SKDAN、few-shot meta-learning、Applied Energy physics-transfer。

因此先完成同数据同监督预算的统一方法族对照，而非谎称已逐篇精确复现。

## 2. 数据与对照协议

- **唯一当前实测数据**：Che 2025 Dataset2 ResVal，94 份文件经作者规则得到 91 个有效电芯；1C 放电前 20 分钟归一化为 (200,3)；诊断曲线目标为 C/80, C/40, 0.05 A, C/10, C/7, C/5 六倍率。
- 所有方法同一固定 Che Dataset1 预训练 checkpoint；**Dataset1 与 Dataset2 电芯人口有重叠**。因此当前检验的是同源物理电芯的诊断协议迁移，**不是**严格新电芯泛化，更不是跨化学体系。
- Outer CV = `KFold(n_splits=5,shuffle=True,random_state=0)`（18–19 测试电芯/折），每折 3 次标签子抽样，共 4 或 14 个有标签训练电芯；选样可看到 ~72 个外层训练池**无标签**电芯，但禁止外层测试电芯参加超参选择、归一化或选样。
- 四种选样：随机、源 SOH 聚类、物理状态+自洽残差聚类、原始片段摘要聚类。所有 kmeans 只在 outer training pool 做。
- 所有回归器都预测 **残差 SOH = 目标 SOH − 原 pretrained SOH**，对同一 pretrained SOH 进行校准。方法：source-only / physics+SOH / input-summary Ridge（alpha=0.1,1,10,100）、RBF KRR、RBF SVR、标准 RBF GP、CatBoost、XGBoost、ExtraTrees。前四类用 k 个标注电芯内部 LOOCV 选择有限超参；树模型用公开配置的固定超参，不声称全面优化。
- 评价单元：逐目标电芯 SOH MAE/RMSE，单位**百分点**，再对 6 rate、3 repeat、5 fold 均值。**90 个 rate×repeat×fold 样本不是 90 个独立 folds**；置信区间按 5 个 fold 的配对均值计算。
- **6480 条条件×fold×repeat×rate 预测记录**；主 protocol 实验设计及 split IDs 记录在本地完整 `per_run.csv`，GitHub 保存压缩/精简记录。source cache SHA256 `60cbb747ac878351f1cf0f27a7860dbc92ab86fc2d7ed71d006334dfc510dcd7`；预训练模型 SHA256 `1941b70c2d6e157049b3cf037f910c93f56f41f90a39d6bdcffbc65e01aebe21`。

## 3. 实验结果：固定标注=4

### 3.1 统一固定物理聚类选样

| 回归方法 | 随机选样 SOH MAE (pp) | 物理特征代表性选样 SOH MAE (pp) |
|---|---:|---:|
| **源 SOH + Ridge** | 2.866 | **1.857** |
| SOH + 四物理 latent + discrepancy + Ridge | 3.204 | 1.982 |
| RBF Kernel Ridge | 5.226 | 3.237 |
| 标准 RBF Gaussian Process | 5.961 | 3.807 |
| RBF SVR | 7.214 | 3.914 |
| CatBoost | 6.873 | 5.637 |
| Extra Trees | 6.417 | 6.374 |
| XGBoost | 8.323 | 7.350 |

**反直觉发现**：4 标签、物理特征聚类选样后，**仅用源模型 SOH 做线性 Ridge 校准 1.857 pp，比加入所有四个物理 latent 的 1.982 pp 略好**。这提示四标签条件下高维校准存在方差或冗余问题；不支持“物理状态越多越好”的简单叙述。

配对统计：选用 `physical_cluster`，source-only Ridge 1.857 对物理 Ridge 1.982，差 -0.125 pp，5-fold t(4) 95% CI [-0.365, 0.115]，**置信区间跨零**，不构成决定性差异。用 source-only Ridge，物理聚类 1.857 vs 随机选样 2.866，差 -1.009 pp，CI [-2.328, 0.310]；5/5 折方向有利但 CI 跨零。

### 3.2 两种“物理作用”需拆分

- (i) **无标签代表性选样**：在物理隐藏空间给 kmeans 选各类代表电芯。它可以影响选择到哪些昂贵 SOH 标签。
- (ii) **有标签 SOH 校准**：回归输入中包含四个隐状态和残差。标签只有 4 个时，(ii) 不必优于只用 SOH 的一维线性校准。

这两个贡献不能捆绑起来宣传。v5 给出物理 latent 的增量信息证据（固定 alpha），v6 改用内层 LOOCV，揭示结果依赖调参方式。

## 4. 实验结果：标注=14

| 选样策略 | Source SOH + Ridge | Physics6 + Ridge | RBF KRR | 标准 RBF GP |
|---|---:|---:|---:|---:|
| 随机 | 1.908 | 1.946 | 2.871 | 3.142 |
| 输入曲线摘要聚类 | 1.851 | **1.618** | 2.176 | 2.299 |
| 源 SOH 聚类 | 1.857 | 1.725 | 2.324 | 2.501 |
| 物理特征聚类 | 1.862 | 1.772 | 2.351 | 2.561 |

使用**原始输入统计量**选出 14 个代表电芯时，物理状态辅助 Ridge 1.618 pp，相同标签的源 SOH-only Ridge 1.851 pp；差 -0.233 pp，5-fold t(4) 95% CI [-0.443, -0.023]。但是**本研究在 Dataset2 上多轮探索并选择策略，属于测试集模型选择偏倚；该区间只是描述性指标，不能按冻结实验的确证性统计进行解释**。

## 5. 为什么这些结果并不意味着已超越 Che2025 论文

- 本研究是**预训练模型输出上的后校准**，它的 source 模型已接受过 Dataset1 训练，而 Dataset2 是 Dataset1 的物理电芯子集；4 或 14 个标签只统计新增目标域监督，不统计原有 source-domain 监督及成本。
- 本实验的 LOOCV 校准比原论文的 30-run full fine-tuning 使用不同模型选择方式与原先参数训练方法；不可拿 1.618 或 1.857 与 Che2025 1.86 的数字直接作为方法优越性结论。
- 此处的 **SOH 经过数值校准，但 latent Cp/Cn/x0/y0 与完整曲线没有同步更新**；无法据此声称实现了物理可信且内部自洽的健康估计。
- 公布 9 个方法族的统一基线不是重实现这些论文的完整网络。不同输入和文献任务需要各自独立复现及适配。

## 6. 发现与下一步假设

**更精确的创新问题不应是“物理特征+Ridge 比黑箱回归器准”。** 新的可证伪假设：

> 在无标签短片段中，电极容量/化学计量状态所定义的物理健康几何可能用于挑选信息量大的诊断标注点；随着可标注电芯数增加，物理特征可降低 SOH 校准的偏差，但若不投影回 DVA 物理可行状态空间，所得到的 SOH 校准只是一种标量修正而不是物理状态迁移。

下一阶段应**先封存** v6 方法和选样超参数，在完全独立的新电芯数据上验证。建议公开且带作者代码的数据与 PINN4SOH 源仓库 https://github.com/wang-fujin/PINN4SOH 、XJTU https://doi.org/10.5281/zenodo.10963339 作为一条独立 native-protocol 复现线；另行建立目标域无泄漏的 phys-latent consistency projection，分别测 SOH、曲线电压、自洽、物理可行率及预测不确定性。之后再用有原始 source 数据时的 MMD / CORAL / AAE / DANN / SMDA / PINN4SOH 原任务强基线对比。

## 7. 审计成果

- 代码 `experiments/literature_bench_v6/benchmark.py`, `analyze.py`，注册清单 `docs/LITERATURE_BASELINE_REGISTER_v6.md`。
- 测试 `tests/test_literature_benchmark_v6.py` (4 passed)；连同 v5 协议测试共 **8 passed**。
- 主要 CSV：`results/v6/formal/{summary.csv,foldmeans.csv,paired_fold_ci.csv,per_run_compact.csv,protocol.json}`。
- **v5 数值回归测试**：重算 `ridge_physics` 在 4×2 选样/标签方案中的均值，与 v5 六倍率结果差 < 1e−4 pp。
- 图 `method_family_comparison.png`、`label_budget_methods.png`，不包含原始真实电池数据或原作者权重。