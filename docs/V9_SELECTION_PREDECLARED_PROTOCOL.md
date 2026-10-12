# v9 选样方法冻结协议 — 2026-10-12

**研究问题：** RW 源域 8 电芯固定，五个目标域不变，2 个目标域电芯各自前 25% 容量标签，Ridge 残差校准固定。只比较从未标注训练电芯候选池选哪两块。严禁使用外层测试电芯信息选样、训练或调参。

**数据：** PINN4SOH 作者 XJTU 逐循环 16 充电统计特征 + 放电容量；只用列(0,4,5,8,12,13)，前 12 循环无标签特征；RW 标准化和预测器与 v8 `rw_source8_model.json` 完全相同；5-fold 测试电芯 ID、随机5次种子不变。

**同协议方法（先冻结，再运行所有工况，不按目标测试效果择方法）：**
1. `source`：未迁移。
2. `random`：v8 固定种子5次随机标注。
3. `centroid_farthest`：v8 中心最近电芯 + 最远点。
4. `pred_extremes`：每块未标注候选电芯前12循环的**源模型预测容量均值**最小、最大的两块。
5. `pred_medoids`：在同一维的源预测容量均值上选择两块，使整个候选池至最近所选样本的**平方距离之和**最小；全部配对枚举。
6. `feature_kmedoids`：六维前12循环标准化充电统计特征的**两个 medoids**，平方距离之和最小；全部配对枚举。
7. `feature_kcenter`：两个样本使所有未标注候选电芯至最近所选样本的**最大平方距离**最小；全部配对枚举。灵感：Sener & Savarese, ICLR 2018 core-set active learning；**不是原代码复现**。
8. `kernel_herding`：采用 Gaussian RBF kernel 及在当前候选池上确定的 median pairwise squared distance；枚举大小为2的子集，使其与候选池的 `MMD^2` 最小（有放回形式 empirical kernel mean embedding）；灵感：核均值匹配/核 herd 思想，**不是原论文代码复现**。
9. 同时对每种标注策略在**相同的选择结果**下计算 `residual Ridge(alpha=10)` 和 `mean offset`，用以分离“选样优势”与“校准优势”。

**注意：** 所有策略只可以读取目标训练候选电芯前12循环**可测充电特征**和源预测容量；不能读取候选电芯的真实放电容量后再决定选样。真实容量只用于被选中电芯的前25%循环校准。外层测试电芯特征和标签不参与选样。

**主结果：** 每个测试电芯先算全寿命平均绝对容量误差（mAh），再逐目标工况平均；报告5目标等权均值、每个工况排序、与随机的配对差值。随机的5次重复共享目标电芯，不能当作5倍独立测试。首轮为探索性方法筛选，后续需在**冻结一个方法**后做独立复验。

**文献界限：**
- Lu et al., *Nature Communications* 2023: https://www.nature.com/articles/s41467-023-38458-w （源域样本分布调整，不是目标电芯主动选样的直接方法）
- Sener & Savarese, ICLR 2018: https://openreview.net/forum?id=H1aIuk-RW （coreset k-center，有公开代码，但任务为图像分类）
- Severson / active learning and physics-guided Discovery Learning, *Nature* 2026: https://www.nature.com/articles/s41586-025-09951-7 （极少试验/主动学习主题，不代表当前选样的相同数据与任务）
- 2024 *npj Materials Sustainability* battery probabilistic ML review: https://www.nature.com/articles/s44296-024-00011-1 （随机与分层 bagging、训练样本分布讨论）
