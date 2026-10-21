"""Non-training compact readout of frozen v17 decision experiment (stdlib only)."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
folder=ROOT/"results/v17"
original=json.loads((folder/"summary_external_cohorts.json").read_text())
out={
 "status":original["status"],
 "n_frozen_raw_external_files":original["n_frozen_raw_external_files"],
 "n_eligible_external_cells":original["n_eligible_external_cells"],
 "n_excluded_external_cells":original["n_excluded_external_cells"],
 "upstream_commit":original["upstream_commit"],
 "binding_go_no_go_decision":original["binding_go_no_go_decision"],
 "primary_bootstrap":original["bootstrap_primary"],
 "bootstrap_acquisition":original["bootstrap_acquisition"],
 "by_budget":original["macro"],
 "cohorts":{}
}
for name,group in original["cohorts"].items():
 out["cohorts"][name]={
   "n_eligible":group["n_eligible"],
   "nominal_Ah":group["nominal_Ah"],
   "excluded":group["excluded"],
   "primary_n30_quartile_metrics":group["summary"]["30"]["pred_quartiles"]["metrics"],
   "n20_quartile_metrics":group["summary"]["20"]["pred_quartiles"]["metrics"],
   "n30_random_metrics":group["summary"]["30"]["random"]["metrics"]
 }
path=folder/"compact_final_decision.json"
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
print("DECISION:",out["binding_go_no_go_decision"]["automatic_final_decision"])
print("PRIMARY:",json.dumps(out["by_budget"]["30"]["pred_quartiles"],ensure_ascii=False))
print("NEW METHOD INCREMENTAL BOOTSTRAP:",json.dumps(out["primary_bootstrap"]["RBF-X_minus_GL-KRR"],ensure_ascii=False))
print("TARGET-ACQUISITION BOOTSTRAP:",json.dumps(out["bootstrap_acquisition"],ensure_ascii=False))
