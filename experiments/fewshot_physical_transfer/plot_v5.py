"""Recreate v5 result figures from committed CSVs only."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'v5'/'figures'
OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'results'/'v5'

cross=pd.read_csv(DATA/'cross_rate_alpha_loo_compact.csv')
rank=['C/80','C/40','0.05A','C/10','C/7','C/5']
for k in (4,14):
 fig,ax=plt.subplots(figsize=(8.2,4.6))
 a=cross[cross.k==k].groupby(['rate','policy']).soh_mae_pp.mean().unstack('policy').reindex(rank)
 for c in ['random','input_clustering','source_soh_clustering','physics_clustering']:
  ax.plot(rank,a[c],marker='o',label=c.replace('_',' '))
 ax.set(xlabel='Target diagnostic discharge rate',ylabel='SOH MAE (percentage points)',
        title=f'{k} labeled target cells; alpha chosen by label-only LOOCV')
 ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(OUT/f'six_rate_{k}labels.png',dpi=170);plt.close(fig)

train=pd.read_csv(DATA/'formal_screen'/'summary.csv')
fig,ax=plt.subplots(figsize=(8,4.5))
for mode in ('full','encoder','encoder_prox','encoder_prox_unlab'):
 d=train[train['mode']==mode].sort_values('k')
 ax.plot(d.k,d.soh_mae_mean,marker='o',label=mode)
ax.set(xlabel='Labeled target cells',ylabel='SOH MAE (percentage points)',
       title='Neural fine-tuning: 400 epochs / C5 / mean of 5 folds x 3 samples')
ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'finetuning.png',dpi=170);plt.close(fig)
print('Figures generated in',OUT)