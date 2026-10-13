# v16: FIRST external-corpus, untouched-target confirmation protocol (2026-10-13)

**Frozen before querying any external target capacity error or running a predictive model on the external sets.** GitHub official repository file names and the first two raw CSV rows have been checked solely to confirm **17-column schema**, capacity positivity and known nominal capacities. **No method has been trained or selected on these target cohort outcomes before the protocol freeze**. Historical v8–v15 development target XJTU 47 cells will NOT be reused for v16 method selection, and all earlier archives are immutable. This is independent cell/corpus validation of a historically selected method, not an independent validation of Che 2025 physical network.

## Research question (user's framework)

A *single* continuous source-preserving capacity calibration, including nearly-zero/weak/strong correction as natural model outcomes, vs fixed source/mean offset/Ridge. **No binary should-calibrate classifier, no per-condition special rule.** Does the pre-existing XJTU-RW8 source and same **pre-frozen v14 RBF-X / v15 disagreement** calibrators produce beneficial cross-battery calibration when target cells come from **unseen whole datasets**?

## Official author data corpus and genuinely new target cells

Upstream repository: https://github.com/wang-fujin/PINN4SOH , pinned commit `cc3cc5053caf38f04e0665f7f88cb109144d035e`. All 17-column processed CSVs have the same field names/order: 16 charging-cycle statistics and **capacity in Ah**; use only exact original v8 feature indices `(0,4,5,8,12,13)`.

Frozen four target cohorts:
1. `TJU_NCA`: `data/TJU data/Dataset_1_NCA_battery`, **66** separate CSV cells, official nominal 3.5 Ah;
2. `TJU_NCM`: `data/TJU data/Dataset_2_NCM_battery`, **55** cells, official nominal 3.5 Ah;
3. `HUST`: `data/HUST data`, **77** cells, official nominal 1.1 Ah;
4. `MIT_2017A`: `data/MIT data/2017-05-12`, **46** cells, official nominal 1.1 Ah.

Total **244 wholly unseen target cells**, spanning three independent *public dataset families*. All 244 precise file paths, Git blob SHA-1, sizes and associated cohort nominal capacities will be frozen in `docs/v16_external_manifest.json` as a separate versioned source manifest *before any external result computation*. These groups are the only v16 targets; do not run on alternative target cohorts and then conceal losers. Within a cohort, any cell with <30 valid positive, fully finite, 17-column data cycles is *excluded for feasibility* before score computation, with ID and reason fully reported; if a cohort has fewer than 7 feasible cells or any upstream file/blob mismatch, fail closed rather than silently alter protocol. Must report zero/nonzero exclusions.

**Nominal capacity metadata is source-published, not target test labels**. Author's pinned `dataloader/dataloader.py` has XJTU=2.0, TJU [3.5,3.5,2.5] and HUST, MIT=1.1 Ah. A common fraction-of-nominal bridge is mathematically required since raw target cells do not share XJTU's 2Ah nominal:

`y_tilde_target = Q_target_Ah * (2.0 / Qnom_target_Ah)` (2Ah-source reference frame).
`yhat_target_Ah = yhat_tilde * (Qnom_target_Ah / 2.0)`.

This uses **only cohort public metadata**, no target true capacity to learn the capacity scale. RW-only scaler, RW8 source weights and six feature indices **remain frozen**, so covariate support mismatch/chemistry shift is fully exposed (no cheating by target-label fit of affine capacity slope). The same bridge applies to Source-only, offset, global residual Ridge, v14 and v15; raw mAh error is then real target capacity. The bridge changes absolute-capacity regression units to a shared nominal-relative target; record the different target setting clearly, not an exact repeat of v10 XJTU condition transfer.

## Pre-declared target-cell experiment

- Deterministic **cell-disjoint 5-fold** per new cohort. Construct fold membership by stable lexicographic filename sort; seeds fixed by mapping `TJU_NCA:20261301, TJU_NCM:20261302, HUST:20261303, MIT_2017A:20261304`, using historical v8 `js_shuffle(range(n), seed)` followed by `idx[i::5]`.
- In each fold, source model never fits target. Heldout cells are used only for full-lifetime evaluation.
- Candidate pool = all other cohort cells; selection as **same frozen** `v10.acquire` for `random` (5 deterministic repeats with seeds based on fold/repeat, no independent new cells) and `pred_quartiles` (1 selection) using first 12 cycles' **unlabeled charging statistics** and the frozen RW model. Exactly **2 selected labeled target cells**, with early labels from cycles 1–N where N=10,20,30; actual labels total **20/40/60**, no other target capacity labels may fit the model, scale or selector.
- N10 warning: acquisition uses first 12 charging cycles, so retrospective archive labels cycles 1–10 would not be recoverable in strict prospective workflow without recording earlier capacity. Do not overclaim online first-ten-cycle implementation.

Methods, all **pre-existing v10/v14/v15 exact implementations and hyperparameters** (no external tuning):
  1. Source-only nominal-bridged RW8;
  2. `offset` (mean source prediction residual of 2 labeled cells);
  3. `Ridge` 6-feature target residual alpha10, RW source-standardized;
  4. `RBF-X` v14 vanilla source-preserving RBF residual (rho 0.05, RW source scale);
  5. `RBF-XQ` v14 ablation (same rho);
  6. `v15-disagreement` v15 uncalibrated two-labeled-cell local disagreement shrinkage of `RBF-X` (no binary switch).

Primary analysis: N30 `pred_quartiles`, all four cohorts and **cohort-equal macro** by **normalized capacity MAE as percent nominal**, rather than absolute mAh macro across unequal 1.1/3.5Ah cells. Within each cohort also report physically interpretable full-lifetime **capacity MAE in actual mAh**. All N10/N20 x both policies and N30 random must also be shown (all six design cells x all methods). Cell-level and cohort-equal metrics; each random repeat is averaged *within test battery first*, never pseudoreplicate 5 extra independent cells. Do not use an unseen cohort to select kernel type, shrink coefficient, alpha or new method.

What counts as success: prioritize both generalizability and negative transfer vs Source-only. A method improving only one cohort or suffering significant losses on other cohorts is not a robust unified cross-domain solution. Non-parametric descriptive paired bootstrap at **10,000** iterations (`seed=20261305`), resampling target cells separately in the 4 cohorts, will report baseline-Ridge minus RBF-X (positive=RBF-X gains) and baseline-RBF-X minus v15; intervals descriptive and not exact inference because one shared source model and overlapping training cohorts remain. No post-outcome hyperparameter tweaks allowed.

## Data science and provenance contract

Raw author CSV files are never committed; fetch into GitHub Actions local cache from `raw.githubusercontent.com/wang-fujin/PINN4SOH/<fixed commit>/...` and verify **Git-blob SHA1** of every byte against prior GitHub API manifest. Use original v8 `load_csv` as schema and preprocessing; reject missing/unknown header or duplicated cell IDs. Compare original v8 source SHA and RW8 refit weights/scaler at 1e-7.

Full outputs:
- results/v16/summary_external_cohorts.json
- results/v16/target_{TJU_NCA,TJU_NCM,HUST,MIT_2017A}.json — every fold/acquisition/method/heldout/cell metric, selected IDs, labels, original SHA and excluded cells.
- reports/v16_external_confirmation_20261013.md — scientific interpretation and limitations.
- Execute original pinned raw data on GitHub Actions; unit tests check no test-label leakage, exact nominal bridge for both capacity units, selection/heldout disjoint, frozen source and no automatic hyperparameter optimization.
- Preserve v8–v15 frozen historical results; no 'v16 success' claim without actual Actions complete and results merged.

## Interpretation boundary

Three external data families and 244 previously unused physical/processing series provide a meaningful new dataset confirmation of existing methods, **but** the samples are all supplied by the same upstream PINN4SOH preprocessor; target chemistries/protocols/nominal capacities may differ and source input distribution drift may make failure scientifically expected. Null or poor results must be published. Labeling cost includes acquiring full author-cycle charging features and measuring the early discharge capacities: not all costs are in label count.
