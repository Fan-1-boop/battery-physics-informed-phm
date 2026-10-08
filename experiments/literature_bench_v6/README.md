# v6 method-family baselines: apples-to-apples, not exact prior-paper replication

The source checkpoint belongs to Che et al. Joule 2025. All experiments use its exported Dataset1 model on Dataset2's 1C 20-minute observations. Previous v5 protocol remains frozen for direct comparison.

## Run

```bash
python -m pip install -r experiments/literature_bench_v6/requirements.txt
python experiments/literature_bench_v6/benchmark.py \
  --cache ./data/processed/cache_dataset2.npz \
  --assets ./data/external/che2025 \
  --out ./outputs/literature_bench_v6
python -m pytest -q tests/test_literature_benchmark_v6.py
```

Files required in `assets`: `best_model.pth` and the two electrode OCP `.csv` tables from original author's https://data.matr.io/10/ package. Cache generated using prior Dataset2 Maccor parsing. Private/raw assets stay out of git.

All nine regressors predict a *residual* to the same fixed pretrained model's scalar SOH, using the same fixed physical latent feature vector except expressly labeled source-only / input-only Ridge controls. Hyperparameters for Ridge/SVR/GP/KRR chosen by LOOCV on only labeled target cells; the three tree models use declared fixed configurations. Outer folds use 91 original cells and `KFold(5,shuffle=True,random_state=0)`; k=4/14 annotated cells; 3 label selections; six diagnostic targets. Four acquisition strategies use only the source features of the outer training pool and no test labels.

**Not included:** the original papers' full architectures such as PINN4SOH, MMD-CNN, AAE, SMDA, AT-GPR, or SKDAN. Those are separate faithful-reproduction tasks. **This experiment alone does not prove transferable electrochemical parameters, independent cross-cell generalization, or physically consistent post-calibration SOH.**