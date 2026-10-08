# External XJTU transfer benchmark (v7)

Use the original processed XJTU NCM dataset released alongside Wang et al., *Nature Communications* (2024), [PINN4SOH](https://github.com/wang-fujin/PINN4SOH). This is **not** a faithful reproduction of the PINN4SOH neural network and **not** a cross-chemistry validation of Che2025's four electrode latent states.

## Requirements

Python 3.10+ and NumPy. Internet access is needed for the **first run**, unless the 55 original CSV files have been placed into the local cache.

```bash
pip install numpy
# Pilot subset: 12 cells (6 from 3C, 6 from RW), full trajectories labeled
python experiments/external_xjtu_v7/benchmark.py \
  --mode pilot --cache data/external/xjtu_processed \
  --label-fraction 1.0 --output outputs/xjtu_v7_pilot_full_labels.json

# Pilot with early-life labels and cell-wise LOO transfer gate
python experiments/external_xjtu_v7/benchmark.py \
  --mode pilot --label-fraction 0.25 --risk-gate \
  --output outputs/xjtu_v7_pilot_prefix25.json

# Extension to all 55 original processed cells, 6 target regimes
python experiments/external_xjtu_v7/benchmark.py \
  --mode full --label-fraction 0.25 --risk-gate \
  --max-holdouts 30 --output outputs/xjtu_v7_full55.json

python -m unittest discover -s tests -p 'test_external_xjtu_v7.py' -v
```

The CSV files will be fetched from the official GitHub repo pinned at commit `cc3cc5053caf38f04e0665f7f88cb109144d035e`. The script stores their Git blob hashes and metadata locally; neither raw CSV nor original PINN4SOH code is redistributed by this repository. The script uses six of the published 16 features and labels (capacity); the remaining ten features are deliberately excluded.

## Caveats

- Capacity MAE in **mAh**, NOT SOH percentage points.
- Target source split is **battery-disjoint and condition-disjoint**, but the model is trained only on these processed observable charge statistics.
- `--mode pilot` labels 1 or 2 target cells; `--label-fraction 1.0` means **all lifetime capacity labels** are available for each chosen battery. Set `--label-fraction 0.25` or `0.1` for more realistic prefix supervision.
- Evaluation comparisons use the **same battery IDs** for every selection method; compare within paired holdout splits, not as unrelated means.
- Repeated trials of deterministic representative selection are exact repeats, not independent replications.
- The risk gate uses only target training pool labels and can fail, as shown by RW target results. A negative-transfer safety guarantee has **not** been established.
- Original PINN4SOH model runs in its own aging-prognosis protocol; do not label the Ridge baselines here as original PINN4SOH method reproduction.

See [`reports/v7_external_xjtu_transfer_and_negative_transfer.md`](../../reports/v7_external_xjtu_transfer_and_negative_transfer.md) for results and limitations.