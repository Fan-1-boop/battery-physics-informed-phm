# Import provenance and reproducibility status

This repository was initialized from the previously prepared `battery-physics-informed-phm_import.zip` archive in the research conversation on 2026-10-08.

## Included
- Independent research code: ResVal preprocessing, 30-run Dataset 2 evaluation, training-resume and ablation scripts
- Compact v2–v4 research reports and corresponding CSV/JSON metric summaries
- Asset checksums and source-data acquisition notes
- `scripts/plot_results.py` to generate *new visualizations* from archived CSVs

## Intentionally excluded
- Original `ResValData.zip`, the authors' `diagnostic_free_code.zip`, released neural-network weights, and cached data
- Previous run-level prediction arrays and checkpoints
- Binary plots from the original local ZIP handoff
- Obsolete `docs/GITHUB_WRITE_ACCESS.md` (the required authorization is now active)

## Important boundaries
- **Dataset 2 full 6-rate × 5-fold:** archived result SOH MAE 1.880 percentage points versus paper 1.860; see `reports/v3_dataset2_30runs.md`.
- **Dataset 1 and Dataset 3:** not fully reproduced.
- **v4 loss/scope/label ablations:** *exploratory 2-fold and 1,000-epoch screens*, not full-benchmark estimates.
- **Legacy scripts:** preserve absolute historical working paths such as `/mnt/data/che2025_official`. They are audit artifacts, not a clean cross-machine CLI. Refactor and regression-test before production.
- **Synthetic smoke code:** not experimental proof on source data.

The authors' data and code remain available from https://data.matr.io/10/ under the authors' own distribution conditions. Do not commit private datasets, credentials, binary weights, or raw data into this public repository.
