# Physics-Informed Battery PHM

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