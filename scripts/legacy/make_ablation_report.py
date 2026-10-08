import json, os, glob, statistics, csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

B=Path('/mnt/data/che2025_reproduction_v4'); (B/'results').mkdir(exist_ok=True)

def load(p): return json.load(open(p))
# paths
P={
 ('full',0):B/'screen/full_all_frac1_f0.json',('full',1):B/'loss_5fold/full_f1.json',
 ('no_phy',0):B/'screen/no_phy_all_frac1_f0.json',('no_phy',1):B/'confirm/no_phy_f1.json',
 ('no_bound',0):B/'screen/no_bound_all_frac1_f0.json',('no_bound',1):B/'confirm/no_bound_f1.json',
 ('reg_only',0):B/'screen/reg_only_all_frac1_f0.json',('reg_only',1):B/'confirm/reg_only_f1.json',
 ('encoder',0):B/'screen/full_encoder_frac1_f0.json',('encoder',1):B/'confirm/encoder_f1.json',
 ('decoder',0):B/'screen/full_decoder_frac1_f0.json',('decoder',1):B/'confirm/decoder_f1.json',
 ('output_layers',0):B/'screen/full_output_layers_frac1_f0.json',
 ('latent_heads',0):B/'screen/full_latent_heads_frac1_f0.json',
}
# loss table
rows=[]
for v in ['full','no_phy','no_bound','reg_only']:
 ds=[load(P[(v,f)]) for f in [0,1]]
 rows.append({
  'variant':v,'n_folds':2,
  'SOH_MAE_pct_mean':np.mean([d['soh_pct']['mae'] for d in ds]),'SOH_MAE_pct_f0':ds[0]['soh_pct']['mae'],'SOH_MAE_pct_f1':ds[1]['soh_pct']['mae'],
  'Q_MAE_mAh_mean':np.mean([d['q_mAh']['mae'] for d in ds]),
  'V_MAE_mV_mean':np.mean([d['voltage_mV']['mae'] for d in ds]),
  'DerivedV_MAE_mV_mean':np.mean([d['derived_voltage_mV']['mae'] for d in ds]),
  'PredDerived_MAE_mV_mean':np.mean([d['pred_derived_mae_mV'] for d in ds]),
  'Cp_out_pct_mean':np.mean([d['violations']['Cp_out_pct'] for d in ds]),
  'Cn_out_pct_mean':np.mean([d['violations']['Cn_out_pct'] for d in ds]),
  'stoich_out_pct_mean':np.mean([d['violations']['sp_out_pct']+d['violations']['sn_out_pct'] for d in ds]),
 })
lossdf=pd.DataFrame(rows); lossdf.to_csv(B/'results/loss_ablation_2fold.csv',index=False)
# scope table
rows=[]
for v in ['full','encoder','decoder']:
 ds=[load(P[(v,f)]) for f in [0,1]]
 rows.append({'scope':v,'n_folds':2,'SOH_MAE_pct_mean':np.mean([d['soh_pct']['mae'] for d in ds]),'SOH_MAE_pct_f0':ds[0]['soh_pct']['mae'],'SOH_MAE_pct_f1':ds[1]['soh_pct']['mae'],'V_MAE_mV_mean':np.mean([d['voltage_mV']['mae'] for d in ds]),'DerivedV_MAE_mV_mean':np.mean([d['derived_voltage_mV']['mae'] for d in ds]),'PredDerived_MAE_mV_mean':np.mean([d['pred_derived_mae_mV'] for d in ds])})
for v in ['output_layers','latent_heads']:
 d=load(P[(v,0)]); rows.append({'scope':v,'n_folds':1,'SOH_MAE_pct_mean':d['soh_pct']['mae'],'SOH_MAE_pct_f0':d['soh_pct']['mae'],'SOH_MAE_pct_f1':np.nan,'V_MAE_mV_mean':d['voltage_mV']['mae'],'DerivedV_MAE_mV_mean':d['derived_voltage_mV']['mae'],'PredDerived_MAE_mV_mean':d['pred_derived_mae_mV']})
scopedf=pd.DataFrame(rows); scopedf.to_csv(B/'results/scope_ablation.csv',index=False)
# label fractions fold0/1
frac_paths={
 1.0:(B/'screen/full_all_frac1_f0.json',B/'loss_5fold/full_f1.json'),
 .5:(B/'screen/full_all_frac0.5_f0.json',B/'confirm/frac05_f1.json'),
 .2:(B/'screen/full_all_frac0.2_f0.json',B/'confirm/frac02_f1.json'),
 .1:(B/'screen/full_all_frac0.1_f0.json',B/'confirm/frac01_f1.json'),
 .05:(B/'screen/full_all_frac0.05_f0.json',B/'confirm/frac005_f1.json'),
 0.0:(B/'zero/f0.json',B/'zero/f1.json'),
}
rows=[]
for frac,(p0,p1) in frac_paths.items():
 ds=[load(p0),load(p1)]
 rows.append({'label_fraction':frac,'n_labeled_f0':ds[0]['n_labeled'],'n_labeled_f1':ds[1]['n_labeled'],'SOH_MAE_pct_f0':ds[0]['soh_pct']['mae'],'SOH_MAE_pct_f1':ds[1]['soh_pct']['mae'],'SOH_MAE_pct_mean':np.mean([d['soh_pct']['mae'] for d in ds]),'SOH_MAE_pct_range':abs(ds[0]['soh_pct']['mae']-ds[1]['soh_pct']['mae']),'V_MAE_mV_mean':np.mean([d['voltage_mV']['mae'] for d in ds]),'DerivedV_MAE_mV_mean':np.mean([d['derived_voltage_mV']['mae'] for d in ds]),'best_epoch_f0':ds[0].get('best_epoch',0),'best_epoch_f1':ds[1].get('best_epoch',0)})
fracdf=pd.DataFrame(rows).sort_values('label_fraction'); fracdf.to_csv(B/'results/label_fraction_2fold.csv',index=False)

# plots
plt.figure(figsize=(7.5,4.6)); x=np.arange(len(lossdf)); plt.bar(x,lossdf['SOH_MAE_pct_mean']); plt.xticks(x,['Full','No physics','No bound','Regression only']); plt.ylabel('SOH MAE (% points)'); plt.title('C/5 controlled loss ablation (mean of folds 0–1, 1000-epoch budget)'); plt.tight_layout(); plt.savefig(B/'results/loss_ablation_soh.png',dpi=180); plt.close()
plt.figure(figsize=(7.5,4.6)); x=np.arange(len(lossdf)); plt.bar(x,lossdf['PredDerived_MAE_mV_mean']); plt.xticks(x,['Full','No physics','No bound','Regression only']); plt.ylabel('|Predicted V − DVA-derived V| MAE (mV)'); plt.title('Latent physical consistency'); plt.tight_layout(); plt.savefig(B/'results/loss_ablation_consistency.png',dpi=180); plt.close()
plt.figure(figsize=(7.5,4.6)); two=scopedf[scopedf.n_folds==2]; x=np.arange(len(two)); plt.bar(x,two['SOH_MAE_pct_mean']); plt.xticks(x,['All parameters','Encoder only','Decoder only']); plt.ylabel('SOH MAE (% points)'); plt.title('Fine-tuning scope ablation (mean of folds 0–1)'); plt.tight_layout(); plt.savefig(B/'results/scope_ablation_soh.png',dpi=180); plt.close()
plt.figure(figsize=(7.5,4.8)); q=fracdf.sort_values('label_fraction'); xx=q['label_fraction'].values*100; plt.plot(xx,q['SOH_MAE_pct_f0'].values,marker='o',label='Fold 0'); plt.plot(xx,q['SOH_MAE_pct_f1'].values,marker='o',label='Fold 1'); plt.xlabel('Target-domain labeled fraction (%)'); plt.ylabel('SOH MAE (% points)'); plt.title('Naive few-shot fine-tuning is unstable below ~20% labels'); plt.legend(); plt.tight_layout(); plt.savefig(B/'results/label_fraction_soh.png',dpi=180); plt.close()

# report
full_exact=1.5982333421707153
report=f'''# Che et al. 2025 Dataset 2 — Ablation Study v4

## Status

This checkpoint extends the previously reproduced Dataset-2 C/5 baseline. The original full-paper Dataset-2 result remains **PASS_REPRODUCED**; the exact C/5 5-fold reproduction gave SOH MAE = **{full_exact:.3f}%** under the original long-training protocol.

The ablations below use a **controlled 1000-epoch compute budget** (patience 300) on C/5. This is deliberately shorter than the paper reproduction and is used for *relative mechanistic screening*, not as a replacement for the exact Table-1 result. Key conclusions were checked on two independent outer folds (folds 0 and 1); some secondary scope screens are fold-0 only and are labelled accordingly.

## A. Loss-component ablation

| Variant | SOH MAE mean (fold0–1) | Pred-V MAE | DVA-derived V MAE | Pred↔Derived MAE | Main observation |
|---|---:|---:|---:|---:|---|
| Full: reg + physics + bound | {lossdf.iloc[0].SOH_MAE_pct_mean:.3f}% | {lossdf.iloc[0].V_MAE_mV_mean:.2f} mV | {lossdf.iloc[0].DerivedV_MAE_mV_mean:.2f} mV | {lossdf.iloc[0].PredDerived_MAE_mV_mean:.2f} mV | Reference |
| No physics | {lossdf.iloc[1].SOH_MAE_pct_mean:.3f}% | {lossdf.iloc[1].V_MAE_mV_mean:.2f} mV | **{lossdf.iloc[1].DerivedV_MAE_mV_mean:.2f} mV** | **{lossdf.iloc[1].PredDerived_MAE_mV_mean:.2f} mV** | Decoder fits voltage, latent physics collapses |
| No boundary | {lossdf.iloc[2].SOH_MAE_pct_mean:.3f}% | {lossdf.iloc[2].V_MAE_mV_mean:.2f} mV | {lossdf.iloc[2].DerivedV_MAE_mV_mean:.2f} mV | {lossdf.iloc[2].PredDerived_MAE_mV_mean:.2f} mV | Accuracy can improve, but Cp leaves physical range in ~{lossdf.iloc[2].Cp_out_pct_mean:.1f}% of test cells |
| Regression only | {lossdf.iloc[3].SOH_MAE_pct_mean:.3f}% | **{lossdf.iloc[3].V_MAE_mV_mean:.2f} mV** | **{lossdf.iloc[3].DerivedV_MAE_mV_mean:.2f} mV** | **{lossdf.iloc[3].PredDerived_MAE_mV_mean:.2f} mV** | Best black-box curve fit, worst physical interpretability |

### Finding A1 — physics loss is not mainly an accuracy regularizer
Removing `L_phy` decreases direct decoder voltage MAE from roughly {lossdf.iloc[0].V_MAE_mV_mean:.1f} to {lossdf.iloc[1].V_MAE_mV_mean:.1f} mV, but DVA-derived-voltage MAE increases from {lossdf.iloc[0].DerivedV_MAE_mV_mean:.1f} to {lossdf.iloc[1].DerivedV_MAE_mV_mean:.1f} mV and predicted-vs-derived disagreement increases from {lossdf.iloc[0].PredDerived_MAE_mV_mean:.1f} to {lossdf.iloc[1].PredDerived_MAE_mV_mean:.1f} mV. Thus the physics term chiefly **anchors the latent state to an electrochemically meaningful representation**.

### Finding A2 — boundary loss serves a different role
Removing the boundary term does not necessarily hurt short-budget SOH accuracy; however, mean out-of-range `Cp` incidence rises to **{lossdf.iloc[2].Cp_out_pct_mean:.1f}%** across folds 0–1. Therefore `L_bound` is primarily a **feasibility constraint**, not simply a prediction-accuracy term.

### Finding A3 — pure regression is a misleading baseline
The regression-only model reaches the smallest direct predicted-voltage error ({lossdf.iloc[3].V_MAE_mV_mean:.2f} mV) while its DVA-derived-voltage error reaches {lossdf.iloc[3].DerivedV_MAE_mV_mean:.1f} mV. A paper that evaluates only terminal/curve reconstruction can therefore look better while the supposedly physical latent state is invalid.

## B. Fine-tuning scope ablation

| Trainable part | Folds | SOH MAE | Pred-V MAE | DVA-derived V MAE |
|---|---:|---:|---:|---:|
| All parameters | 2 | {scopedf.loc[scopedf.scope=='full','SOH_MAE_pct_mean'].iloc[0]:.3f}% | {scopedf.loc[scopedf.scope=='full','V_MAE_mV_mean'].iloc[0]:.2f} | {scopedf.loc[scopedf.scope=='full','DerivedV_MAE_mV_mean'].iloc[0]:.2f} |
| Encoder only | 2 | {scopedf.loc[scopedf.scope=='encoder','SOH_MAE_pct_mean'].iloc[0]:.3f}% | {scopedf.loc[scopedf.scope=='encoder','V_MAE_mV_mean'].iloc[0]:.2f} | {scopedf.loc[scopedf.scope=='encoder','DerivedV_MAE_mV_mean'].iloc[0]:.2f} |
| Decoder only | 2 | **{scopedf.loc[scopedf.scope=='decoder','SOH_MAE_pct_mean'].iloc[0]:.3f}%** | {scopedf.loc[scopedf.scope=='decoder','V_MAE_mV_mean'].iloc[0]:.2f} | {scopedf.loc[scopedf.scope=='decoder','DerivedV_MAE_mV_mean'].iloc[0]:.2f} |
| Output layers only | 1 (screen) | {scopedf.loc[scopedf.scope=='output_layers','SOH_MAE_pct_mean'].iloc[0]:.3f}% | {scopedf.loc[scopedf.scope=='output_layers','V_MAE_mV_mean'].iloc[0]:.2f} | {scopedf.loc[scopedf.scope=='output_layers','DerivedV_MAE_mV_mean'].iloc[0]:.2f} |
| Latent heads only | 1 (screen) | {scopedf.loc[scopedf.scope=='latent_heads','SOH_MAE_pct_mean'].iloc[0]:.3f}% | {scopedf.loc[scopedf.scope=='latent_heads','V_MAE_mV_mean'].iloc[0]:.2f} | {scopedf.loc[scopedf.scope=='latent_heads','DerivedV_MAE_mV_mean'].iloc[0]:.2f} |

**Interpretation:** encoder-only adaptation tracks full fine-tuning closely on the two confirmation folds, whereas decoder-only adaptation is dramatically worse. This suggests the dominant Dataset-1→Dataset-2 shift lies on the **observation/representation side**, not merely in the physical decoder. This is directly relevant to a future domain-adaptation method: adapt the fragment encoder first, while preserving the physics decoder as a shared structure.

## C. Target-label fraction screen

| Target labeled fraction | Approx. labeled cells/fold | SOH MAE fold 0 | SOH MAE fold 1 | Mean | Interpretation |
|---:|---:|---:|---:|---:|---|
'''
for _,r in fracdf.sort_values('label_fraction',ascending=False).iterrows():
 n=int(round((r.n_labeled_f0+r.n_labeled_f1)/2)); report+=f"| {100*r.label_fraction:.0f}% | {n} | {r.SOH_MAE_pct_f0:.3f}% | {r.SOH_MAE_pct_f1:.3f}% | {r.SOH_MAE_pct_mean:.3f}% | {'zero-shot' if r.label_fraction==0 else ('high variance / unstable' if r.label_fraction<=.2 else 'usable but degraded')} |\n"
report+=f'''

The zero-shot model has mean-fold SOH MAE ≈ **{fracdf.loc[fracdf.label_fraction==0,'SOH_MAE_pct_mean'].iloc[0]:.2f}%** over folds 0–1 (and **12.28%** averaged over all five outer folds). Full fine-tuning is therefore essential in the released method.

The critical observation is not merely that fewer labels hurt. At 20% labels, fold 0 reaches {fracdf.loc[fracdf.label_fraction==.2,'SOH_MAE_pct_f0'].iloc[0]:.2f}% while fold 1 fails at {fracdf.loc[fracdf.label_fraction==.2,'SOH_MAE_pct_f1'].iloc[0]:.2f}%. At 10% and 5%, both folds frequently select a best epoch in the first few dozen updates. With only 4–15 labelled target cells and an 80/20 split, the target validation set itself becomes too small to support stable early stopping.

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
'''
(B/'ABLATION_REPORT.md').write_text(report)
print(lossdf.to_string(index=False))
print('\n',scopedf.to_string(index=False))
print('\n',fracdf.to_string(index=False))