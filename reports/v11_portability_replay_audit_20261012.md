# v11 — 独立 Python 全量复算验收与逐电芯负迁移诊断（2026-10-12）

**验收结果：PASS。** 这是从 PINN4SOH 上游固定版本下载原始 CSV、在 GitHub Actions Ubuntu/Linux / Python 3.11 独立重跑现有 Python 实现的**数值可复现性验收**，不是一个新的外部目标数据集确认实验，也不是 v9 选样方法的预注册验证。

## 1. 复算对象与数据溯源

- 上游作者：`wang-fujin/PINN4SOH`，XJTU processed `data/XJTU data/`；冻结上游 commit：`cc3cc5053caf38f04e0665f7f88cb109144d035e`。
- 原始电芯：RW 8，目标 2C 8、3C 15、R2.5 8、R3 8、Sim_satellite 8，共55电芯。
- 输出任务：六个逐循环充电统计特征预测**放电容量**，MAE单位 mAh。非 Che 2025 的 V(Q) 片段/电极物理 latent，也没有直接验证 SOH 百分点。
- v8 源 RW8 标准化、Ridge alpha=1，从原始 CSV **重训**，不是仅调用保存的源模型权重；逐项与 `results/v8/rw_source8_model.json` 对比。
- 目标只对候选池前12次循环可测充电统计特征进行无标签选样；目标外层测试电芯不进入选样或监督校准。
- v8/v9 复现：原固定五折、两目标训练电芯、各前25%寿命标签（回顾性），固定随机种子，v9仅复核 Random、Representative、Pred-quartiles 核心组。
- v10 复现：目标两电芯各前10/20/30循环真实容量标签，Random重复5次、Pred-quartiles一次，固定 alpha=10 残差Ridge及两个电芯均值 residual offset。源模型不变。

## 2. GitHub Actions 独立执行的验收证据

- **Run**：[RW 55-cell Python replay audit #38198412254](https://github.com/Fan-1-boop/battery-physics-informed-phm/actions/runs/38198412254)，commit `1aaa1a10e6a063789c23d764d176c66913fc9547`，执行时间2026-10-12。
- GitHub-hosted Python 3.11.17，NumPy 2.4.6。
- 原有 v8 7项 + v9 8项 + v10 5项离线单元测试：**20/20 PASS**。
- 从上游固定commit获取55个原始CSV，逐源/目标电芯对比Git blob SHA与有效循环数量；复算源 scaler / Ridge 系数（容差1e-7）。
- 重建五域五折、选择ID与次数、v8/v9/v10的原归档 per-cell MAE，包含 v10 N=10/20/30、Ridge与offset全部组合。
- **7186 checks, 0 failures; 最大逐电芯MAE绝对差 6.630410211982962e-7 mAh，小于1e-6 mAh容差。**
- 完整程序产出的 `v11_python_replay_audit.json` 作为该 Actions run 的artifact `rw-python-replay-audit`（artifact ID `11724611867`）提供下载，包含 `per_cell_replayed_v10`、源数据SHA、选择ID、标签成本、各工况误差和检查结果。因上游CSV涉及第三方数据，本仓库**不重新分发原始CSV**。
- 验收脚本：`experiments/rw_multitarget/verify_portability_v11.py`；CI 配置：`.github/workflows/rw-python-replay-audit.yml`；机器可读验收摘要：`results/v11/python_replay_validation_summary.json`。

### 本轮找到的两个工程问题

1. v9 `experiments/rw_selection/benchmark.py` 引用v8兄弟目录时使用 `parents[2]`，路径错误，导致全新容器 v9测试导入失败；本分支将其精确修复为 `parents[1]`，**不改变模型方程或冻结结果**。
2. v10 3C 的JSON记录随机编号键为 `repeat`，其他四域为 `rep`；审计器兼容两种历史键，不修改历史归档。两项修复后复算成功。

## 3. 标注周期预算的真实性

55个原始CSV均通过检验，五个目标域的目标电芯有效循环长度范围为：

| Target | 电芯数 | 最短循环 | 最长循环 |
|---|---:|---:|---:|
| 2C | 8 | 362 | 388 |
| 3C | 15 | 121 | 313 |
| R2.5 | 8 | 503 | 651 |
| R3 | 8 | 473 | 646 |
| Sim_satellite | 8 | 669 | 942 |

每个目标电芯均有至少30次循环；v10的预算没有触发脚本中的 `min(N,n_cycles)` 截断，因此每折固定2个选中电芯，容量真值成本恰为20/40/60条，另须能获取其前12循环的**无标签充电特征**。如果部署系统获取容量真值本身需要真实完整放电，成本不能仅用标签条数描述。

## 4. 从已有 v10 结果执行的附加负迁移诊断（并非新数据拟合）

数据：`results/v11/v10_posthoc_cellwise_diagnostics.json`。随机策略五次抽样先在**同一被测电芯内平均**，然后按目标电芯比较。以N=30的Pred-quartiles+Ridge为例，相对 RW source-only，2C 0/8、3C **9/15**、R2.5 0/8、R3 1/8、Satellite 0/8 的目标测试电芯出现校准后MAE上升，合计 **10/47**。这明确了3C负迁移局部集中的现象，却不保证其他数据集也如此。

v10五目标等权宏MAE：N10 Random/Ridge 49.723，Quartile/Ridge 48.605；N20 44.498 vs 43.969；N30 41.758 vs 41.081 mAh。四分位对随机的提升仅1.118/0.529/0.677mAh，历史分层bootstrap区间均含零；**不得称为统计显著**。

## 5. 论文可写与不可写的边界

**已经成立**：本项目RW源+简单少标签残差校准的输入/代码/原始数据可复算；固定前N循环实验具有可核查的容量标签条数；3C负迁移需完整呈现。

**仍然不成立**：候选四分位选样在未接触数据上显著优于随机、跨全新电芯/全新化学体系广泛泛化、Che电极物理状态的跨域复现、SOH百分点精度。v8-v10在同一47目标电芯上的后验选择和消融属于开发集实验。下一阶段若要验证选样优势，需要在未用于v9策略筛选的外部电芯中**预先冻结**模型与指标，不再于这47电芯上搜新的选样规则。
