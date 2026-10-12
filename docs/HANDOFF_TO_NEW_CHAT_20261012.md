# RW→多工况少样本容量估计：新对话正式交接（2026-10-12）

## 必要入口与权威顺序
研究代码仓库：https://github.com/Fan-1-boop/battery-physics-informed-phm

新对话先按以下顺序阅读（全部在 main 分支）：
1. 本文件 `docs/HANDOFF_TO_NEW_CHAT_20261012.md`
2. `reports/v9_verified_summary_20261012.md`
3. `docs/V9_SELECTION_PREDECLARED_PROTOCOL.md`（**标题保留历史文件名，正文明确说明是事后归档，不是预注册！**）
4. `reports/v9_rw_source_active_selection.md`
5. `results/v9/summary_five_domains_and_paired_bootstrap.json`
6. `results/v9/audit_20261012_cellwise_costs.json`
7. `experiments/rw_selection/benchmark.py`, `tests/test_rw_selection_v9.py`
8. `docs/V8_RW_SOURCE_MULTI_TARGET_FROZEN_PROTOCOL.md`, `reports/v8_rw_source_55cell_5target.md`, `experiments/rw_multitarget/benchmark.py`, `results/v8/rw_source8_model.json`。

注意：新对话不要只依据老聊天的“v9 还没运行”描述。仓库已经存在完成的 v9 结果和 2026-10-12 审计。以代码和归档报告为准，若文件之间存在冲突需明确报告。

## 研究目标（收缩主线）
硕士论文务实研究线：
- **仅 RW 工况**作为源域训练；目标为 **2C、3C、R2.5、R3、Sim_satellite**。不再做 3C→RW 反向迁移，不扩展为跨化学体系物理 latent 迁移。
- XJTU 公开 55 个 NCM 电芯，RW 8 个源电芯，47 个目标电芯；5 折外层**电芯级**测试。
- 原公开数据：`wang-fujin/PINN4SOH` 仓库 `data/XJTU data/`，固定原仓库 commit `cc3cc5053caf38f04e0665f7f88cb109144d035e`。16 个逐循环充电统计特征 + 1 个放电容量标签。本项目用 6 个 CC/CV 特征，任务为逐循环放电容量预测（MAE **mAh**）。
- 这**不是** Che et al. Joule 2025 的 20 min V(Q) 输入，不包含 Che 四个电极物理状态 `Cp,Cn,x0,y0` 的真值；不得混称复现 Che 跨化学体系迁移，也不得把 mAh 写成 SOH 百分点。
- 源模型 RW 8 电芯 Ridge alpha=1；特征缩放只用 RW 源数据；目标残差校准 Ridge alpha=10；目标候选池仅可在前 12 循环观察未标注的充电统计特征。历史 k=2 的标签限制为选中两块电芯各自**最初 25% 全寿命**的容量标签（该预算是回顾性，真实在线部署不可先知未来寿命）。
- 各策略同样的测试折与标签预算；随机标注每折 5 次重复只用于抽样方差，不算独立电芯。

## 历史实证状态
### v8：真实五目标实验完整
五个目标域容量 MAE（mAh，越低越好）：

| 方法 | 2C | 3C | R2.5 | R3 | Sim_satellite | 五域等权 |
|---|---:|---:|---:|---:|---:|---:|
| Source-only RW8 | 60.35 | 35.11 | 77.74 | 94.77 | 100.86 | 73.76 |
| Random + residual Ridge | 27.79 | 44.03 | 42.50 | 46.63 | 47.71 | **41.73** |
| Centroid+Farthest + Ridge | 28.23 | 43.16 | 42.33 | 54.19 | 61.20 | 45.82 |

3C 全部 15 电芯时校准不如 source-only；此前只用前 6 个目标电芯的 pilot 正向结论不能代替完整结果。

### v9：真实五目标选样探索已完成，且完成审计
容量 MAE 的五域等权均值：
- Random：**41.733 mAh**
- 旧 Centroid+Farthest：45.821 mAh
- 源预测容量极值 Pred-extremes：44.39 mAh
- **源预测容量 25%/75% 四分位 Pred-quartiles：41.417 mAh**
- 源预测容量 2-medoids：43.02 mAh
- 6D 特征 2-medoids：43.84 mAh
- RW8 LOBO 模型分歧最高两电芯：49.42 mAh
- 不确定度加权 6D medoids：45.00 mAh

四分位策略相对随机宏均值仅改善 **0.317 mAh（约0.76%）**；47 电芯按工况分层 bootstrap 的探索性 95% 区间约 **[-1.98, 2.68] mAh**（不同已归档计算约 [-2.00, 2.63]，同样包含零）。不能声称显著优于随机或达到文献 SOTA，多个启发式已在相同五目标数据上探索选优，存在 benchmark 过拟合风险。

v9 报告审计：47 电芯覆盖、target label/test ID 分离、Random 与 v8 结果一致；但**不是**新容器重新联网独立逐值运行所有 55 电芯的验收。

## 标签成本硬问题
“标注两电芯”不等于“两个容量标签”：若使用各电芯前 25% 寿命，Pred-quartiles 每目标折两个被选中电芯平均需标注约 2C 186、3C 116、R2.5 262、R3 293、Satellite 378 个循环。未来总寿命也无法在线事先知道。必须将下一阶段协议改为**固定前 N 个循环**（例如 N=10/20/30，须先固定并检查最短轨迹覆盖），显式记录真实的容量测量成本与可行性。

## 推荐下一步（不要重复 v9 已做的探索）
1. 首先执行数据/代码可移植性验收：在可联网 Python 环境按作者固定 commit 下载官方 55 个 CSV，复核源模型、外层折、Pred-quartiles 每折选择 ID 和每电芯预测数值，遇差异先修复，保留 hash/commit/日志。用户已授权 GitHub；新代码及时提交。
2. **先冻结方法，不继续在相同 47 电芯/相同五折上搜新启发式并宣称优越。** 以 Pred-quartiles 作为待确认的单一候选，与 Random 和 source-only 作公平比较。
3. 设计现实固定标签预算（前10/20/30循环，必要时先核对所有目标电芯最短样本数）。在不同的预先指定电芯划分或**额外独立数据**做确认；若只能使用原55电芯的重新划分，必须说这不是严格独立外部验证，因为该数据集已用于方法选择。
4. 最小 5 方法矩阵：Source-only；Random+mean offset；Random+Ridge；Pred-quartiles+mean offset；Pred-quartiles+Ridge。报告每工况、每电芯 MAE、随机重复的聚类依赖、标注循环数和不确定区间。不得窥视测试标签用于模型/阈值选择。
5. 若 Pred-quartiles 稳定超不过 Random，及时收缩结论，改聚焦“RW 源域+极少标签的轻量残差容量校准”，而不是制造选样优势。
6. 高质量文献仅在输入/协议公平时做精确复现；当前 v9 为文献**思想启发**的适配实现，非 PINN4SOH/Che/MMD/TrAdaBoost 官方方法逐项复现。可参考：
   - Lu et al., *Nature Communications*, 2023 DOI `10.1038/s41467-023-38458-w`
   - Wang et al., PINN4SOH, *Nature Communications*, 2024 DOI `10.1038/s41467-024-48779-z`
   - *Batteries*, 2024 sample transfer `10.3390/batteries10050156`
   - Che et al., *Joule*, 2025 `10.1016/j.joule.2025.102010`

## 交付与工作要求
- 所有新实验代码、折划分/选样 ID、per-cell 原始指标、图表、协议与科学报告进入 `Fan-1-boop/battery-physics-informed-phm` GitHub，原始电池 CSV、大型权重、未经许可第三方代码不上传。
- 先实际运行后报告数值；报告必须区分：已做实验、归档结果的复算审计、尚未执行的计划。没有独立真值证据，不能声称内部物理状态可信。
- 用户希望简洁、务实的硕士论文主线：唯一 RW 源域→五目标工况→有限容量标签。**一个清晰的改进胜过大量附加研究支路。**
