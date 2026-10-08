# v8 — RW 唯一源域 → 五目标工况，55 电芯完整一轮结果

**日期：2026-10-08。状态：真实数据实验完成，五折探索性 benchmark；并非 Che 2025 的四电极隐状态跨域复现。**

## 1. 实验问题（范围收缩）

遵照 `docs/V8_RW_SOURCE_MULTI_TARGET_FROZEN_PROTOCOL.md`：

- 源域仅 XJTU **RW 8 电芯**；目标域分别 2C(8)、3C(15)、R2.5(8)、R3(8)、Sim_satellite(8)，合计 **55 电芯**。
- 固定作者公开 GitHub `wang-fujin/PINN4SOH` commit `cc3cc5053caf38f04e0665f7f88cb109144d035e`，原始 CSV 不在本仓库分发。
- 所有目标工况使用相同 6 个可观测充电统计特征（列 0、4、5、8、12、13），预测每个循环的放电容量；指标是 **cell-equal-weighted MAE，mAh**，不是 SOH 百分点。
- 源域完整 RW 电芯容量标签训练 Ridge (alpha=1)，并且仅用 RW 拟合特征标准化。
- 目标域 **5 折电芯级交叉验证**；各折完全分离。目标训练候选池中，标注 **2 个电芯各自最初 25% 寿命循环**的容量标签，用 Ridge(alpha=10) 拟合源预测残差。
- 标注电芯从目标候选池按两种策略选出：①随机（5 次种子重复）；②仅查看未标注候选电芯前 12 循环的充电特征，先取距特征中心最近电芯，再取距已选电芯最远者。
- 不使用目标外层测试电芯特征做选样，也不使用测试电芯标签做训练或调参。目标测量片段的物理统计特征不是电极容量/化学计量 latent。

## 2. 全部 5 个目标工况的结果

| RW→目标 | 目标电芯数 | Source only | Random labels + residual Ridge | Representative labels + residual Ridge |
|:--|--:|--:|--:|--:|
| 2C | 8 | 60.35 | **27.79** | 28.23 |
| 3C | 15 | **35.11** | 44.03 | 43.16 |
| R2.5 | 8 | 77.74 | 42.50 | **42.33** |
| R3 | 8 | 94.77 | **46.63** | 54.19 |
| Sim_satellite | 8 | 100.86 | **47.71** | 61.20 |
| **五目标等权均值** | **47** | **73.76** | **41.73** | **45.82** |

**单位：mAh；越低越好。** 本表为五个预先指定目标工况，并非仅展示成功样例。随机策略的五次重复只改变标注电芯选择，**不是额外 5 倍独立目标电芯**。正式统计的独立电芯数为 47；因为具有共同 RW 源模型及重叠训练池，不应夸大独立性。

## 3. 对硕士研究路线的判断

- **可支持**：随机工况 RW 预训练 + 很少的目标域早期容量标签 + 简单残差校准，在 2C、R2.5、R3、Sim_satellite 上有实质收益；可以保留“单向 RW 源域，多个目标工况”的明确主线。
- **不能支持**：“代表性选样必然优于随机”。它只在 3C 和 R2.5 两域边际更好，而在 R3/Sim_satellite 明显不如随机。
- **不能支持**：“RW→3C 已确认效果很好”。上一步 12 电芯 pilot 是 RW 6 + 3C 6；本次 RW 8 + **3C 15** 给出相反结论，代表性选样 43.16 mAh **劣于**不迁移 35.11 mAh。先导样本及源域数量变化影响结论。
- **不能支持**：这些特征证明 Che OCP 物理 latent 可迁移或比 PINN4SOH 原论文模型性能更好。PINN4SOH 官方此处为 16 个逐循环充电统计特征；本次只取其中 6 个，既不是作者原论文 PINN4SOH 模型，也不是 Che 的 V(Q) 短片段回归。

## 4. 附加细胞级结果

本次保存了五目标的**完整 per-fold、per-cell、标注选择 ID、原文件 blob SHA**：
- `results/v8/rw_source8_model.json`
- `results/v8/rw_to_2C_fivefold.json`
- `results/v8/rw_to_3C_fivefold.json`
- `results/v8/rw_to_R2_5_fivefold.json`
- `results/v8/rw_to_R3_fivefold.json`
- `results/v8/rw_to_Sim_satellite_fivefold.json`
- `results/v8/summary_five_domains.json`

例如 3C 目标的 15 电芯中，只有 6 个电芯在随机 Ridge 下好于源模型；该事实说明均值隐藏了部分负迁移。R3/Sim_satellite 的代表性选样并没有稳定收益。

## 5. 统计与应用限制

- 此处是**五折的一次固定划分**，不是独立大样本多数据集确认；目标只有 8–15 电芯/域，置信区间可能很宽。
- 固定的 25% **寿命比例**是回顾性标签预算，现实部署并不知道未来总寿命；实施时应进一步转换为固定前 N 次循环的预算。
- 目前记录电芯级平均 MAE；没有把每个循环错误都当作独立测试样本。
- `Random + Ridge` 同样获得大量收益，所以这批结果尚不足以把“物理相关代表性选样”当成新方法贡献。
- 本轮和 v7 pilot 的自适应方案不同；在不同实验上比较绝对误差须明确源域 8 vs 6 电芯、目标 15 vs 6 电芯、测试划分不同。

## 6. 下一步只做小范围改进（不引入新大网络）

1. 保持 RW→五目标主线不变，加入源模型、**仅均值偏差校准**、随机 Ridge 与代表性 Ridge，作为最小强基线矩阵。
2. 单独比较一个可解释、无需电极 OCP 的**源预测容量分层选样**（依据目标未标注早期片段的源预测值，不读取容量真值）；这与特征空间代表性选样区分开。
3. 若新选样策略不能稳定胜过随机，就把论文贡献改为**真实工况下极少标签校准的规律与实证**，不声称创新算法具有稳健优势。
4. 如将来需要 SOH，须先核对额定容量定义；不可用测试电芯全寿命数据或真实首循环容量做归一化。

## 复现

```bash
pip install numpy
python experiments/rw_multitarget/benchmark.py \
  --targets 3C 2C R2.5 R3 Sim_satellite \
  --k 2 --label-frac 0.25 --random-repeats 5 \
  --output outputs/rw_multitarget_v8.json
python -m unittest discover -s tests -p 'test_rw_multitarget_v8.py' -v
```

脚本固定上游 commit，第一次运行需联网从作者仓库取得逐循环 CSV；后续从本地 `data/external/xjtu_processed` 缓存读取。本研究 GitHub 不二次分发作者原始数据。实验数值最初通过 GitHub 授权连接的逐文件公开 CSV 读取、独立 JS 计算归档；可移植 Python 实现提供相同公式、折号和随机种子，代码级回归 / 隔离测试通过，但由于沙盒不能访问外网，本次**没有在同一容器中重新以 Python 跑完全部 55 电芯**；该跨实现逐值重跑仍是后续必要的再现性验收。

原作者：https://github.com/wang-fujin/PINN4SOH ；相关论文 DOI：https://doi.org/10.1038/s41467-024-48779-z


## 7. 事后诊断：固定 RW8，只改变 3C 目标样本量

为解释先导实验与完整 3C 结果差异，固定上述 RW 8 电芯训练的源模型、alpha、标签比例和五折随机种子规则，单独取 3C 最初 6 个电芯再次实验：

| 3C target set | Source-only (mAh) | Random+Ridge (mAh) | Representative+Ridge (mAh) |
|---|---:|---:|---:|
| First 6 target batteries | 38.04 | **31.88** | 33.16 |
| All 15 target batteries | **35.11** | 44.03 | 43.16 |

因此即使 RW 源域都固定为 8 电芯，3C 前 6 电芯仍显示出迁移收益，而全 15 电芯结论相反。**样本组成及测试折定义影响结论**；该对照是看到完整结果之后做的解释性实验，不作为预注册的正式性能指标，更不是筛选前六电芯宣称成功。

详见 `results/v8/posthoc_rw8_3C_first6_diagnostic.json`。
