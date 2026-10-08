# v6: 2021–2026 research baseline register (curated 2026-10-08)

**Scope**: partial-charge / short-discharge battery SOH, few-shot/unsupervised transfer, physics-informed health states.  *Published papers* versus *our experimental implementations* must remain distinguishable. Published performance is NOT compared numerically without matching datasets, SOH definitions, train/test splitting and label budgets.

| Priority | Paper / published venue | DOI / code | Main method from published source | Fit to our current single-20-min-discharge input? | Status |
|---|---|---|---|---|---|
| P0 | Che et al., *Diagnostic-free onboard battery health assessment*, **Joule 2025** | https://doi.org/10.1016/j.joule.2025.102010; https://data.matr.io/10/ | DVA constrained physical health latents and short fragments | **Exact match** | Dataset2 6×5 baseline reproduced in previous v3, 1.880 pp SOH MAE, not re-run for v6 |
| P0 | Wang et al., *Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis*, **Nature Communications 2024** | https://doi.org/10.1038/s41467-024-48779-z; **official code** https://github.com/wang-fujin/PINN4SOH | Physics-informed degradation state-space model with short charging feature and time trajectories; paper reports MAPE 0.87% on its experiment setting | **Different temporal input, training targets, and datasets** | Official code presence verified; separate native-protocol reproduction on XJTU/TJU planned |
| P0 | *A Transfer Learning-Based Method for Personalized State of Health Estimation of Lithium-Ion Batteries*, **IEEE TNNLS 2024** (online 2022) | https://doi.org/10.1109/TNNLS.2022.3176925 | CNN + regression-oriented MMD alignment | **Needs actual source charging data for end-to-end transfer** | Method family for future matched-data benchmark; no exact reimplementation claimed |
| P0 | Dou et al., *Cross-domain state of health estimation for lithium-ion battery based on latent space consistency using few-unlabeled data*, **Energy 2025** | https://doi.org/10.1016/j.energy.2025.135257 | AE/AAE with adversarial latent distribution matching; latent similarity SOH prediction, 1 source cell + 20% unlabeled target in authors' setting | Input preprocessing and source data differ | Important closest competing concept; not implemented exactly, code not verified |
| P0 | *A cross-material lithium-ion battery state of health estimation method based on three-stage domain adaptation*, **Energy 2025** | https://doi.org/10.1016/j.energy.2025.139376 | MMD / SMDA / fine-tuning; partial-charging segments and NCA/NMC/LCO/hybrid cross-chemistry | Current target uses 1C discharge and no separate cross-chemistry data | Closely matched scientific task, but not exact-code baseline yet |
| P1 | *A self-attention knowledge domain adaptation network for commercial lithium-ion batteries state-of-health estimation under shallow cycles*, **Journal of Energy Storage 2024** | https://doi.org/10.1016/j.est.2024.111197 | Attention knowledge distillation + multi-kernel MMD; full to shallow charging SOC windows | Cannot claim exact equivalence to our single 1C discharge | Native-protocol extension needed |
| P1 | *A Meta-Learning Method for Few-Shot Multidomain State-of-Health Estimation of Lithium-Ion Batteries*, **IEEE TTE 2025** | https://doi.org/10.1109/TTE.2024.3470551 | CNN-attention + task meta-learning with relaxation-voltage samples | Relaxation voltage and labels per cycle differ from labels per cell | Source tasks and matched RV data needed; no exact reimplementation claimed |
| P1 | *State-of-Health Estimation of Li-Ion Batteries Using Semiparametric Adaptive Transfer Learning*, **IEEE TTE 2024** | https://doi.org/10.1109/TTE.2023.3266499 | Adaptive semiparametric transfer GPR kernel | 4–14 labeled cell regression is comparable as a method family, not original transfer kernel | **Plain RBF GPR** is included in v6, NOT the published AT-GPR algorithm |
| P2 | Le et al., *Physics-informed transfer learning by embedding physics into activation functions*, **Applied Energy 2026** | https://doi.org/10.1016/j.apenergy.2025.127161 | Arrhenius/SEI informed activation + transfer, source Stanford/MIT/Toyota -> XJTU | Their outcome includes RUL/MAPE, not directly comparable to SOH MAE | Mechanism reference / separate long-term experiment |

## Current v6 apples-to-apples benchmark

Uses original Che2025 Dataset1 pretrained model and Dataset2 same 91 usable physical cells. **This is transfer across diagnostic protocols for already-seen physical cell population**. Every method sees the same target training pool, 4 or 14 labeled cells, fixed outer 5-fold cell splits, three reproducible label selections, and six C/80–C/5 diagnostic-rate targets. Test cells are not used for target-domain normalization, representation selection, model selection, or training. Repeated selections share test cells and **must not** count as independent test folds.

**Compared method families, all independently implemented in our code, not paper reproductions:**

1. Source-only Ridge / source+physical-latent Ridge / input-summary Ridge: affine residual calibration, labeled-only leave-one-out alpha selection.
2. RBF Kernel Ridge; RBF SVR; RBF Gaussian Process: nonlinear low-shot response correction, labeled-only leave-one-out hyperparameter selection; **not** any one cited paper's exact transfer architecture.
3. CatBoost / XGBoost / Extra Trees on physically augmented features: strong nonlinear tabular learners with fixed conservative hyperparameters (not exhaustive optimizations).
4. Label acquisition: random, source-SOH kmeans medoids, physical-feature kmeans medoids, raw-fragment summary kmeans medoids. K-means uses exclusively the unlabeled **outer training pool**.

All regressors fit residual target SOH minus pretrained Che source SOH. This is calibrated SOH only: DVA latents / full predicted curves are NOT projected back to the calibrated scalar. Computational budget and model selection MUST be reported for fair comparison.

## To become a publishable stronger comparison

1. Freeze protocol **before opening an independent external dataset** (XJTU/TJU/CALCE, etc.), enforce **source–target physical cell disjointness** and train source model without any target cell history.
2. Reproduce PINN4SOH from the official repository **on its native task and native data**; then implement matched-input baselines on chosen external experiments and label results as **adaptations**.
3. For a true cross-domain method competition, acquire source trajectory data; compare feature-MMD, CORAL, DANN / AAE, target FT and proposed phys-latent method with consistent source sets and label budgets.
4. Fix: definitions of SOH and temperature / chemistry, sampling window, source training cell list, all label counts per cell, data preprocessing, split hashes, hyperparameter budget, five independent seeds, confidence intervals at independent cell/domain grouping level, physical feasibility and calibrated prediction-interval coverage.
5. Never compare paper MAPE vs our MAE or paper RMSE from different test sets as evidence of superiority; avoid paper-equivalent claims without implementation-level correspondence.

## Notes on innovation boundary

Physics-guided transfer and adversarial latent alignment already exist. A possible innovation must show **independent cross-domain benefit** beyond strong label acquisition + source-only ridge + nonphysical features, with improved DVA self-consistency and physical feasibility. The current Che latent vector is a physically *structured prediction*, not experimentally validated uniquely identifiable degradation mechanisms.