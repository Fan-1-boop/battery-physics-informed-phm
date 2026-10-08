# Physics-Informed Battery PHM

Che et al. (Joule 2025) Dataset 2 reproduction, ablations and subsequent short-fragment SOH transfer-learning research.

**Current status:** Dataset 2: 6 rates × 5 folds, 30 runs, archived SOH MAE 1.880 percentage points vs paper 1.860. Dataset 1 and 3 have **not** been fully reproduced. The v4 ablations are **exploratory and use shorter training budgets**.

## Repository layout
- `scripts/legacy`: independently written experimental scripts copied from earlier research checkpoints **without silently changing their original absolute paths**. To execute, adjust `/mnt/data/che2025_official` paths or use the paths documented in the original reports.
- `src/che2025`: synthetic smoke test, **not paper replication**.
- `results/v3` and `results/v4`: compact CSV/JSON measurements; regenerate example charts with `python scripts/plot_results.py`.
- `reports`: detailed reproduction and ablation methods / boundaries.
- `docs`: dataset manifest and research protocol.

## Data and code provenance
Original dataset archives, original author's source code, model weights, experimental caches and patient/industrial data are not redistributed in this repository. Obtain the authors' datasets and checkpoints from https://data.matr.io/10/ and verify against `scripts/verify_official_assets.py`.

This is an archival transfer of previous research snapshots: **no new 30-run training is claimed to have happened as part of this repository import**.

See [import provenance and limitations](docs/IMPORT_NOTES.md) for precisely what was pushed and which experiment artifacts remain local.

```bash
python -m pip install -r requirements.txt
python scripts/plot_results.py  # regenerate archived-summary figures, no training
python scripts/verify_official_assets.py ./data  # check official archives if available
```

## Upcoming work
Develop reproducible path-based CLI and tests; statistically robust few-shot protocols; encoder-only/full fine-tuning comparisons; physical-consistency and constraint ablations.
