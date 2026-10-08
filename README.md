# Physics-Informed Battery PHM

## v9 — RW→五工况低标签电芯选样（2026-10-08）

**真实 55 电芯探索实验已完成；8 种选样策略共享 v8 的 RW8 源模型、相同电芯级五折及 2 个标注电芯前25%循环标签预算。**

**预测容量四分位分层**获得五工况等权容量 MAE **41.42 mAh**，随机标注 **41.73 mAh**，原六维中心+最远点 **45.82 mAh**。随机−四分位 MAE 差约 **+0.317 mAh**，探索性电芯 bootstrap 95% 区间 **[-2.00,+2.63] mAh**（包含零）。因此不能宣称预测分层已经稳定优于随机。根据近年论文思想尝试的六维/输出分布 medoids、源模型分歧及不确定性加覆盖策略亦无整体优势。

[正式研究报告](reports/v9_rw_source_active_selection.md) · [8 策略数值表](results/v9/summary_five_domain_selection.csv) · [可运行实验代码](experiments/rw_selection/benchmark.py) · [配对统计](results/v9/summary_five_domains_and_paired_bootstrap.json)

**下一个小范围实验**：固定前 N 次循环标签，比较随机/预测四分位选样下的均值偏差校准、残差 Ridge、实例加权式迁移，不增设新大网络。文献思想启发与完整原论文复现需区别报告。


## v8 55-cell 实际实验结果（2026-10-08）

固定 RW 全部 8 电芯训练源 Ridge；五个目标工况合计 47 电芯；每次只使用 2 个目标候选训练电芯的前 25% 寿命循环标签。全部目标域电芯级 5-fold 测试已经完成；可移植 Python 代码和 7 项协议测试见仓库。

| RW → target | Source MAE (mAh) | Random+Ridge | Representative+Ridge |
|---|---:|---:|---:|
| 2C (8) | 60.35 | **27.79** | 28.23 |
| 3C (15) | **35.11** | 44.03 | 43.16 |
| R2.5 (8) | 77.74 | 42.50 | **42.33** |
| R3 (8) | 94.77 | **46.63** | 54.19 |
| Sim_satellite (8) | 100.86 | **47.71** | 61.20 |
| 五目标等权 | 73.76 | **41.73** | 45.82 |

**解释：** 少标签校准在 4/5 个目标域降低容量 MAE，但在完整 3C 上不成立；当前代表性选样并未整体优于随机。不要把先前 6+6 电芯 pilot 当作 55-cell 结论，也不要将容量 MAE 直接写为 SOH MAE。

- [v8 完整研究报告](reports/v8_rw_source_55cell_5target.md)
- [可移植 benchmark](experiments/rw_multitarget/benchmark.py)
- [55-cell 逐电芯聚合](results/v8/summary_five_domains.json)
- [v8 冻结协议](docs/V8_RW_SOURCE_MULTI_TARGET_FROZEN_PROTOCOL.md)

## 当前硕士论文研究主线（v8，2026-10-08）

**唯一源域：XJTU RW 随机工况；目标域：3C、2C、R2.5、R3、Sim_satellite；只研究 RW → 目标工况。**

只保留一个核心方法问题：**目标域极少量容量标签下，利用无标签充电特征选择代表性电芯，并用简单残差校准改善容量/SOH 估计。**

前期 RW→3C 的 12 电芯 pilot：零样本 42.90 mAh，2 个目标电芯前 25% 标签下代表性选样 + Ridge 32.93 mAh。仅为探索性先导数据，不等于完整 55 电芯结论。

正式实验协议见 [v8 单向 RW→多工况冻结方案](docs/V8_RW_SOURCE_MULTI_TARGET_FROZEN_PROTOCOL.md)。历史 v3–v7 结果作为研究档案保留；v8 **已执行首轮五目标验证**；详见上方实测结果。


## v7 — External XJTU transfer stress test (2026-10-08)

**External pilot, NOT cross-chemistry Che physical-latent reproduction.** The first independent XJTU evaluation uses 12 separate NCM batteries from 3C/RW charge conditions, 2,942 cycle-level records, six observable charge statistics, and discharge-capacity targets. It is not the same input task as Che2025's partial discharge V(Q) to SOH.

- Fixed holdout sets first: 15 paired two-cell test splits per target condition; two labeled acquisition batteries; early-life labels restricted to 10%, 25%, or entire lifetime. All target test batteries are unseen as labeled data.
- RW → 3C: representative selection + residual Ridge, 25% early-life labels gives 32.928 mAh, vs source-only 42.896 mAh.
- 3C → RW: the same strategy gives **60.506 mAh**, WORSE than source-only 52.411 mAh (**negative transfer**).
- A cell-level leave-one-labeled-battery-out transfer gate also failed to prevent RW negative transfer (62.909 mAh).
- Results are exploratory: 12 of the public 55 cells, split overlap and deterministic acquisition repeats preclude treating every row as independent.

See [full research report](reports/v7_external_xjtu_transfer_and_negative_transfer.md), [portable benchmark runner](experiments/external_xjtu_v7/benchmark.py), and [fixed protocol](results/v7/PROTOCOL.md). Raw third-party CSV files are not committed. The full 55-cell test is implemented, but not yet run here.


Che et al. (Joule 2025) Dataset 2 reproduction, ablations and subsequent short-fragment SOH transfer-learning research.

**Current status:** Dataset 2: 6 rates × 5 folds, 30 runs, archived SOH MAE 1.880 percentage points vs paper 1.860. Dataset 1 and 3 have **not** been fully reproduced. The v4 ablations are **exploratory and use shorter training budgets**.

## Repository layout
- `scripts/legacy`: independently written experimental scripts copied from earlier research checkpoints **without silently changing their original absolute paths**. To execute, adjust `/mnt/data/che2025_official` paths or use the paths documented in the original reports.
- `src/che2025`: synthetic smoke test, **not paper replication**.
- `results/v3` and `results/v4`: compact measurements and charts exported from existing checkpoints.
- `reports`: detailed reproduction and ablation methods / boundaries.
- `docs`: dataset manifest and research protocol.

## Data and code provenance
Original dataset archives, original author's source code, model weights, experimental caches and patient/industrial data are not redistributed in this repository. Obtain the authors' datasets and checkpoints from https://data.matr.io/10/ and verify against `scripts/verify_official_assets.py`.

This is an archival transfer of previous research snapshots: **no new 30-run training is claimed to have happened as part of this repository import**.

## Upcoming work
Develop reproducible path-based CLI and tests; statistically robust few-shot protocols; encoder-only/full fine-tuning comparisons; physical-consistency and constraint ablations.

## v5 — Few-shot transfer / active label acquisition (2026-10-08)

**Status: exploratory, not cross-chemistry verified.** The 91 valid ResVal cells overlap the Dataset-1 source corpus physically, so these are adaptation-under-diagnostic-shift experiments rather than unseen-cell generalization.

See [v5 full report](reports/v5_fewshot_physical_transfer.md).

With four labeled target-training cells selected by unsupervised target-training-pool k-means (roughly 72 unlabeled candidates), physics-feature ridge calibration using labeled-only LOOCV achieved **1.982 SOH percentage-point MAE averaged over six diagnostic rates** versus 3.204 with random label acquisition. This is a post-hoc exploratory design requiring independent validation; its five-fold paired confidence interval versus random includes zero. Physics-specific superiority over SOH-only clustering is **not established**. Corrected SOH is not yet projected back onto physically consistent latent states.

Run after downloading the original author's assets, building `cache_dataset2.npz` with the legacy Maccor preprocessing script, and putting `best_model.pth` plus the two half-cell OCP CSVs under a local `data/external/che2025/` directory:

```bash
python experiments/fewshot_physical_transfer/run.py --cache ./data/processed/cache_dataset2.npz --assets ./data/external/che2025 --out ./outputs/fewshot --epochs 400
python experiments/fewshot_physical_transfer/cross_rate_acquisition.py --cache ./data/processed/cache_dataset2.npz --assets ./data/external/che2025 --out ./outputs/cross_rate.csv --alpha-selection loo
python experiments/fewshot_physical_transfer/plot_v5.py  # works on tracked compact CSV files
python -m pytest -q tests/test_fewshot_protocol.py
```

No source data, binary checkpoints, or original third-party source package are redistributed.

## v6 — Strong literature-informed baselines under matched low-label protocol

**Status: exploratory, not paper-by-paper exact reproduction.** An updated [2021–2026 literature baseline register](docs/LITERATURE_BASELINE_REGISTER_v6.md) covers PINN4SOH (Nature Communications 2024; official code confirmed), CNN+MMD (IEEE TNNLS 2024), unlabeled latent consistency (Energy 2025), MMD–SMDA–FT (Energy 2025), few-shot meta-learning and AT-GPR. The [v6 research report](reports/v6_literature_strong_baselines.md) documents method comparability and evidence limitations.

Independent method-family implementations (nine regression configurations): Ridge (source-only, physical latents, input summary), RBF kernel Ridge, RBF SVR, standard RBF Gaussian process, CatBoost, XGBoost and Extra Trees. Fixed Dataset2 5-fold splits, 3 labeled-cell selections, k=4/14, six diagnostic rates, and four acquisition policies produced **6480 measured rows**. All tuning occurs within labeled training cells; test data are not used for selection. Historical v5 physics-Ridge metrics reproduced to <0.0001 pp.

For *four labeled cells selected by physical clustering*, source-SOH-only Ridge achieves **1.857 pp** MAE vs 1.982 pp with all physical latents, and random-acquisition source-SOH Ridge 2.866 pp. For 14 labels selected via input-summary clustering, physics-Ridge reaches **1.618 pp** vs source-only Ridge 1.851 pp. Results are exploratory on an overlapping source/target physical-cell population, so **not evidence of previously unseen-cell or cross-chemistry transfer**. Calibrated scalar SOH is not projected back into physically self-consistent latent states.

Run `python -m pytest -q tests/test_literature_benchmark_v6.py` then see `experiments/literature_bench_v6/README.md` for the source assets required. Committed `results/v6/formal/` includes rate/fold summaries, paired blocked CIs, and 10 compact per-run CSV shards (`per_run_fold{0..4}_part{0,1}.csv`), without original datasets or model weights.