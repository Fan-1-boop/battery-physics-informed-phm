"""Render archived experiment summaries as figures; does not rerun training."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

rate = pd.read_csv(ROOT / "results/v3/ratewise_metrics.csv")
fig, ax = plt.subplots()
ax.plot(rate["rate"], rate["SOH_MAE_pct"], marker="o")
ax.set(xlabel="Diagnostic rate", ylabel="SOH MAE (percentage points)", title="Che2025 Dataset 2: SOH error by rate")
fig.tight_layout()
fig.savefig(OUT / "soh_by_rate.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots()
ax.plot(rate["rate"], rate["V_MAE_mV"], marker="o", label="Predicted voltage")
ax.plot(rate["rate"], rate["DV_MAE_mV"], marker="o", label="Derived voltage")
ax.set(xlabel="Diagnostic rate", ylabel="Voltage MAE (mV)", title="Dataset 2: voltage error by rate")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "voltage_by_rate.png", dpi=160)
plt.close(fig)

few = pd.read_csv(ROOT / "results/v4/label_fraction_2fold.csv")
few = few.sort_values("label_fraction")
fig, ax = plt.subplots()
ax.plot(few["label_fraction"] * 100, few["SOH_MAE_pct_mean"], marker="o")
ax.set(xlabel="Labeled target cells (%)", ylabel="SOH MAE (percentage points)", title="Exploratory few-shot screening (2 folds)")
fig.tight_layout()
fig.savefig(OUT / "fewshot_soh.png", dpi=160)
plt.close(fig)
print(f"Figures regenerated at: {OUT}")
