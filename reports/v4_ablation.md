# Che et al. 2025 Dataset 2 — Ablation Study v4

## Status

This checkpoint extends the previously reproduced Dataset-2 C/5 baseline. The original full-paper Dataset-2 result remains **PASS_REPRODUCED**; the exact C/5 5-fold reproduction gave SOH MAE = **1.598%** under the original long-training protocol.

The ablations below use a **controlled 1000-epoch compute budget** (patience 300) on C/5. This is deliberately shorter than the paper reproduction and is used for *relative mechanistic screening*, not as a replacement for the exact Table-1 result. Key conclusions were checked on two independent outer folds (folds 0 and 1); some secondary scope screens are fold-0 only and are labelled accordingly.

## A. Loss-component ablation

| Variant | SOH MAE mean (fold0–1) | Pred-V MAE | DVA-derived V MAE | Pred↔Derived MAE | Main observation |
|---|---:|---:|---:|---:|---|
| Full: reg + physics + bound | 2.240% | 17.02 mV | 32.80 mV | 16.36 mV | Reference |
| No physics | 1.906% | 5.04 mV | **89.53 mV** | **89.36 mV** | Decoder fits voltage, latent physics collapses |
| No boundary | 2.177% | 10.17 mV | 19.75 mV | 10.56 mV | Accuracy can improve, but Cp leaves physical range in ~78.4% of test cells |
| Regression only | 1.950% | **5.13 mV** | **199.61 mV** | **200.40 mV** | Best black-box curve fit, worst physical interpretability |

### Finding A1 — physics loss is not mainly an accuracy regularizer
Removing `L_phy` decreases direct decoder voltage MAE from roughly 17.0 to 5.0 mV, but DVA-derived-voltage MAE increases from 32.8 to 89.5 mV and predicted-vs-derived disagreement increases from 16.4 to 89.4 mV. Thus the physics term chiefly **anchors the latent state to an electrochemically meaningful representation**.

### Finding A2 — boundary loss serves a different role
Removing the boundary term does not necessarily hurt short-budget SOH accuracy; however, mean out-of-range `Cp` incidence rises to **78.4%** across folds 0–1. Therefore `L_bound` is primarily a **feasibility constraint**, not simply a prediction-accuracy term.

### Finding A3 — pure regression is a misleading baseline
The regression-only model reaches the smallest direct predicted-voltage error (5.13 mV) while its DVA-derived-voltage error reaches 199.6 mV. A paper that evaluates only terminal/curve reconstruction can therefore look better while the supposedly physical latent state is invalid.

## B. Fine-tuning scope ablation

| Trainable part | Folds | SOH MAE | Pred-V MAE | DVA-derived V MAE |
|---|---:|---:|---:|---:|
| All parameters | 2 | 2.240% | 17.02 | 32.80 |
| Encoder only | 2 | 2.335% | 13.50 | 30.14 |
| Decoder only | 2 | **6.125%** | 67.31 | 152.14 |
| Output layers only | 1 (screen) | 9.368% | 27.78 | 40.99 |
| Latent heads only | 1 (screen) | 8.784% | 13.77 | 21.25 |

**Interpretation:** encoder-only adaptation tracks full fine-tuning closely on the two confirmation folds, whereas decoder-only adaptation is dramatically worse. This suggests the dominant Dataset-1→Dataset-2 shift lies on the **observation/representation side**, not merely in the physical decoder. This is directly relevant to a future domain-adaptation method: adapt the fragment encoder first, while preserving the physics decoder as a shared structure.

## C. Target-label fraction screen

| Target labeled fraction | Approx. labeled cells/fold | SOH MAE fold 0 | SOH MAE fold 1 | Mean | Interpretation |
|---:|---:|---:|---:|---:|---|
| 100% | 72 | 1.384% | 3.095% | 2.240% | usable but degraded |
| 50% | 36 | 2.192% | 3.212% | 2.702% | usable but degraded |
| 20% | 14 | 2.883% | 11.692% | 7.287% | high variance / unstable |
| 10% | 7 | 7.409% | 12.270% | 9.839% | high variance / unstable |
| 5% | 4 | 8.410% | 12.061% | 10.236% | high variance / unstable |
| 0% | 0 | 10.651% | 14.280% | 12.465% | zero-shot |


The zero-shot model has mean-fold SOH MAE ≈ **12.47%** over folds 0–1 (and **12.28%** averaged over all five outer folds). Full fine-tuning is therefore essential in the released method.

The critical observation is not merely that fewer labels hurt. At 20% labels, fold 0 reaches 2.88% while fold 1 fails at 11.69%. At 10% and 5%, both folds frequently select a best epoch in the first few dozen updates. With only 4–15 labelled target cells and an 80/20 split, the target validation set itself becomes too small to support stable early stopping.

This exposes a **real research gap**: simply reducing the number of target labels in Che et al.'s full fine-tuning pipeline is not a robust few-shot solution. A new method must address both representation transfer **and** low-label model selection/regularization.

## D. Current research implications

1. **Physics constraint should be retained**, but its value should be evaluated using latent physical consistency, not just SOH MAE or voltage RMSE.
2. **Boundary constraints are necessary** to prevent apparently accurate but physically impossible latent states.
3. **Encoder is the natural transfer-learning target.** Freezing the decoder while adapting the encoder is far more viable than the reverse.
4. **Naive few-shot fine-tuning is unstable.** This creates a concrete thesis direction: physics-guided encoder adaptation with label-efficient or validation-free regularization.
5. The next formal experiment should replace naive tiny validation splits with a reproducible low-label protocol (e.g. fixed source-domain stopping criterion, nested repeated subsampling, or regularization calibrated on source data), then compare full FT, encoder-only FT, MMD/CORAL/DANN, and physics-guided alignment.

## Files

- `results/loss_ablation_2fold.csv`
- `results/scope_ablation.csv`
- `results/label_fraction_2fold.csv`
- `results/loss_ablation_soh.png`
- `results/loss_ablation_consistency.png`
- `results/scope_ablation_soh.png`
- `results/label_fraction_soh.png`
- `screen/` and `confirm/`: raw per-run JSON metrics

## Reproducibility note

These ablation conclusions are **not claimed as the paper's own results**. They are new experiments built on the successfully reproduced Dataset-2 pipeline. The original paper reproduction remains the long-training 5-fold result; v4 is a controlled-compute mechanistic audit designed to identify the next research hypothesis.