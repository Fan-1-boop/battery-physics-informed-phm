# Frozen pilot metadata

- Data: Official PINN4SOH `data/XJTU data/` processed CSV; GitHub commit `cc3cc5053caf38f04e0665f7f88cb109144d035e`.
- True battery IDs: `3C_battery-[1-6]`, `RW_battery-[1-6]`, 2942 cycles total.
- XJTU feature indices (0-based): `[0,4,5,8,12,13]`, capacity label column 16.
- Condition-disjoint source/target; exactly 15 battery-test pairs per target group; exactly 2 battery test cells per split; 4 candidate unlabelled target batteries.
- Fixed target test set across random / representative label selection and across all calibration methods, as of v7.1.
- Target label budgets: 1/2 batteries; availability of each selected battery's capacity labels limited to the first 10%, 25%, or 100% of cycles. All cycles of *test* batteries are evaluation only.
- Per-battery-balanced source linear Ridge alpha=1, target residual Ridge alpha=10, fixed (no target test tuning); source-only, target mean-offset, residual Ridge, 2-label cell-wise leave-one-out gate.
- 5 random repeats. Deterministic representative selection repeats are **not** statistically independent.
- Pilot v7.0 **invalid/confounded**: the test cells were different under the two selection methods. v7.1+ fixed.
- Output metric: cell-macro discharge-capacity absolute error in milliamp-hours, *not* Che SOH percentage points.
- Source metadata/individual Git blob SHAs: tracked by `results/v7/xjtu_cross_condition_12cell_fixedtest_v7_1.json` on GitHub; raw data not redistributed.
- Full 55-cell leave-one-condition-out is **script-ready but NOT executed in this checkpoint**.