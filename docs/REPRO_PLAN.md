# Che et al. (Joule 2025) reproduction plan

Paper: *Diagnostic-free onboard battery health assessment*, Joule 9, 102010 (2025), DOI: 10.1016/j.joule.2025.102010.

## Scope and acceptance gates

### Gate A — Dataset 1 / health diagnosis (primary reproduction)
- Source: van Vlijmen et al. NCA/graphite+SiOx 21700 cells.
- 236 cells, 126 operating conditions.
- Input: first charge after each diagnostic cycle; randomly truncate start/end three times per diagnostic cycle.
- Input information: partial charging voltage, capacity, EFC.
- Target: full C/5 discharge curve from the directly preceding RPT, used as pseudo-OCV.
- Split: cells sharing the same charge/discharge protocol must appear exclusively in train or test; 20% of training samples form validation set; early stopping.
- Model: encoder -> 4 mechanistic states [Cn, Cp, x0, y0] -> decoder; DVA-derived OCV imposes physical consistency.
- Paper targets (Table 1):
  - Q sequence MAE 45.1 mAh; RMSE 67.2 mAh; R2 0.998
  - predicted V MAE 10.1 mV; RMSE 13.8 mV; R2 0.999
  - derived V MAE 18.4 mV; RMSE 25.2 mV; R2 0.996
  - SOH MAE 1.35%; RMSE 1.86%; R2 0.945

### Gate B — Physics / interpretability
- Reproduce DVA relation OCV = OCPp(y) - OCPn(x).
- Reproduce latent states [x0, y0, Cp, Cn].
- Total diagnosis loss: L = Lreg + Lphy + sum_i 0.1 Lbound,i (paper: regression and physical loss weights 1.0; 10 boundaries each 0.1).
- Verify ablation without physics/bounds preserves regression ability but loses latent/derived-OCV consistency.

### Gate C — Dataset 2 / rate transfer
- 94 degraded 21700 cells; RPT rates C/80 to 2C.
- Fine-tune model pretrained on Dataset 1.
- Input: 20 min of 1C discharge; target: low-rate curves/capacity.
- 5-fold CV.
- Paper target: SOH MAE 1.86%; capacity MAE 43.1 mAh; voltage MAE 22.4 mV.

### Gate D — Dataset 3 / dynamic cycling
- Geslin et al., 92 cells, CC/periodic/synthetic/real-driving aging.
- Target C/40 RPT pseudo-OCV.
- Charging baseline: pretrain on Dataset 1, fine-tune on about one-third of cells, test remaining.
- Dynamic-discharge case: random portions of dynamic discharge as input.
- Paper targets:
  - charging SOH MAE 0.86%
  - dynamic discharge SOH MAE 1.15%; predicted-V MAE 7.33 mV; normalized-Q MAE 0.53%.

### Gate E — Prognosis
- Current EFC + current predicted SOH + [Cp,Cn,x0,y0] -> prognosis decoder.
- Reproduce future capacity/EFC sequence and cycle life (80% SOH threshold).
- Paper target Dataset 1 cycle-life MAE about 76 EFC at 83.5–86.5% SOH; early prediction about 118 EFC.

## Reproduction levels
- **Exact reproduction**: author code + author data + author split/hyperparameters.
- **Independent reproduction**: reimplemented equations/model/data processing from paper/supplement, same public data and split.
- **Structural smoke test**: synthetic DVA-consistent data only; verifies code paths, physics loss, boundaries, training and metrics. It is NOT evidence that paper results are reproduced.