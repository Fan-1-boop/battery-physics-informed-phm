"""v11: Independent Python raw-data replay of archived v8/v9/v10 XJTU experiments.

Read original upstream CSV at the exact pinned commit, independently rebuild RW8
source Ridge using existing Python code, and compare per-cell/fold results with the
immutable archived JS-origin results.  No test labels enter acquisition/training.

This is a cross-implementation reproducibility audit, NOT external validation.
Requires outbound access to raw.githubusercontent.com for original 55 CSVs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
V8_FILE = ROOT / "experiments/rw_multitarget/benchmark.py"
V9_FILE = ROOT / "experiments/rw_selection/benchmark.py"
V10_FILE = ROOT / "experiments/rw_multitarget/fixed_budget_v10.py"
DOMAINS = ("2C", "3C", "R2.5", "R3", "Sim_satellite")
TOL_MAH = 0.000001
TOL_MODEL = 0.0000001


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def cell_map(rows):
    return {r["id"]: float(r["mae_mAh"]) for r in rows}


def run(cache: Path):
    v8 = load_script("replay_v8", V8_FILE)
    v9 = load_script("replay_v9", V9_FILE)
    v10 = load_script("replay_v10", V10_FILE)
    frozen = read_json(ROOT / "results/v8/rw_source8_model.json")
    issues = []
    n_checked = 0
    max_diff = 0.0
    by_domain = {}
    full_rows = []
    source_info = []

    def check(ok, context):
        nonlocal n_checked
        n_checked += 1
        if not ok:
            issues.append(context)

    def close(a, b, context, tol=TOL_MAH):
        nonlocal max_diff
        a, b = float(a), float(b)
        diff = abs(a-b)
        max_diff = max(max_diff, diff) if tol == TOL_MAH else max_diff
        check(np.isfinite(diff) and diff <= tol, f"{context}: {a} vs {b}, delta={diff}")

    def same(a, b, context):
        check(a == b, f"{context}: {a!r} != {b!r}")

    def compare_predictions(now, old, context):
        x, y = cell_map(now), cell_map(old)
        same(set(x), set(y), context + " test cell IDs")
        for cid in set(x) & set(y):
            close(x[cid], y[cid], context + " " + cid)

    rw = [v8.get_cell("RW", i, cache) for i in range(1, 9)]
    scaler = v8.fit_scaler(rw)
    model = v8.ridge(rw, scaler, v8.SOURCE_ALPHA)
    same(v8.COMMIT, frozen["source_commit"], "pinned upstream data commit")
    same(list(v8.FEATURES), frozen["feature_indices"], "six input features")
    for i, c in enumerate(rw):
        expected = frozen["source_cells"][i]
        same(c.name + ".csv", expected["name"], "RW filename")
        same(c.git_sha, expected["sha"], c.name + " Git blob")
        same(c.n_cycles, expected["cycles"], c.name + " valid cycles")
        source_info.append({"id": c.name, "sha": c.git_sha, "cycles": c.n_cycles})
    for name, actual, expected in (
        ("scaler.mean", scaler[0], frozen["mean"]),
        ("scaler.sd", scaler[1], frozen["sd"]),
        ("RW8 Ridge coefficients", model, frozen["coefficients"]),
    ):
        for i, (a, b) in enumerate(zip(actual, expected)):
            close(a, b, f"{name}[{i}]", TOL_MODEL)

    for domain in DOMAINS:
        cs = [v8.get_cell(domain, i, cache) for i in range(1, v8.COUNTS[domain] + 1)]
        same(len(cs), v8.COUNTS[domain], domain + " target cell count")
        check(min(c.n_cycles for c in cs) >= 30, domain + " minimum cycle count < 30")
        check(all(c.n_cycles >= 12 for c in cs), domain + " fewer than 12 unlabeled feature cycles")
        old8 = read_json(ROOT / "results/v8" / f"rw_to_{domain.replace('.', '_')}_fivefold.json")
        old9 = read_json(ROOT / "results/v9" / f"rw_to_{domain.replace('.', '_')}_selection.json")
        old10 = read_json(ROOT / "results/v10" / f"rw_to_{domain.replace('.', '_')}_fixed_N.json")
        folds = v8.make_folds(len(cs), domain)
        expected_folds = [[cs[i].name for i in f] for f in folds]
        same(old8["fold_indices"], expected_folds, domain + " v8 heldout fold IDs")
        same(old10["folds"], expected_folds, domain + " v10 heldout fold IDs")
        same(sorted(cid for f in expected_folds for cid in f), sorted(c.name for c in cs),
             domain + " each test cell exactly once")
        old_targets = {r["name"]:r for r in old8["target_cells"]}
        for c in cs:
            ref = old_targets.get(c.name)
            check(ref is not None, domain + " target missing " + c.name)
            if ref:
                same(c.git_sha, ref["sha"], c.name + " original CSV sha")
                same(c.n_cycles, ref["cycles"], c.name + " number of valid cycles")

        # v8 Python implementation: source-only, random, six-feature representative.
        fresh8 = v8.run_domain(cs, domain, model, scaler, k=2, label_frac=.25, random_repeats=5)
        old8_map = {(r["fold"],r["policy"],r["repeat"]):r for r in old8["results"]}
        same(len(old8_map), len(old8["results"]), domain + " v8 unique fold/policy/repeat")
        for r in fresh8["records"]:
            key = (r["fold"],r["policy"],r["repeat"])
            ref = old8_map.get(key)
            check(ref is not None, domain + " v8 missing archived record " + str(key))
            if ref is None:
                continue
            same(r["heldout"], ref["heldout"], domain + " v8 test ID " + str(key))
            same(r["labeled"], ref["labeled"], domain + " v8 selected ID/cost " + str(key))
            compare_predictions(r["per_cell"], ref["per_cell"], domain + " v8 " + str(key))
            if r["policy"] != "source":
                compare_predictions(r["per_cell_offset"], ref["per_cell_offset"],
                                    domain + " v8 offset " + str(key))
        for pol, stats in fresh8["summary"].items():
            for k in ("mae_mAh", "offset_mae_mAh"):
                if k in stats:
                    close(stats[k], old8["summary"][pol][k], domain + f" v8 summary {pol}/{k}")

        # v9 Python code (original 25% hindsight budget), only the frozen
        # comparison policies required by v11. Ignore unrelated v9 heuristics.
        fresh9 = v9.run_domain(cs, domain, model, scaler,
                               policies=("random", "representative", "pred_quartiles"),
                               label_fraction=.25, random_repeats=5)
        old9_map = {(r["fold"],r["policy"],r["rep"]):r for r in old9["records"]
                    if r["policy"] in ("random", "representative", "pred_quartiles")}
        for r in fresh9["records"]:
            if r["policy"] == "source":
                continue
            key = (r["fold"],r["policy"],r["repeat"])
            ref = old9_map.get(key)
            check(ref is not None, domain + " v9 missing archived record " + str(key))
            if ref is None:
                continue
            same([(s["id"],s["n_label_cycles"]) for s in ref["selected"]],
                 [(s["id"],s["n_label_cycles"]) for s in r["selected"]],
                 domain + " v9 selected ID/cost " + str(key))
            compare_predictions(r["test"], ref["test"], domain + " v9 " + str(key))
        for pol in ("random", "representative", "pred_quartiles"):
            close(fresh9["summary"][pol]["mae_mAh"], old9["summary"][pol]["mae_mAh"],
                  domain + " v9 summary " + pol)

        # v10 first 10/20/30 labels: both residual calibrators, exact IDs and
        # per-heldout-cell MAE. Fully refit every correction from raw labels.
        fresh10 = v10.evaluate_domain(cs, domain, scaler, model, budgets=(10,20,30),
                                       random_repeats=5)
        old10_map = {(r["fold"],r["policy"],r["rep"],r["budget"]):r
                     for r in old10["records"]}
        for r in fresh10["records"]:
            if r["policy"] == "source":
                continue
            key = (r["fold"],r["policy"],r["repeat"],r["budget"])
            ref = old10_map.get(key)
            check(ref is not None, domain + " v10 missing archived record " + str(key))
            if ref is None:
                continue
            same(r["heldout"], ref["heldout"], domain + " v10 heldout " + str(key))
            same([x["id"] for x in r["selected"]], ref["selected"],
                 domain + " v10 selected ID " + str(key))
            same([x["n_labeled_cycles"] for x in r["selected"]], ref["n_labeled_cycles"],
                 domain + " v10 cost " + str(key))
            same(sum(ref["n_labeled_cycles"]), 2*r["budget"],
                 domain + " v10 fixed true-label budget " + str(key))
            compare_predictions(r["ridge"], ref["ridge"], domain + " v10 ridge " + str(key))
            compare_predictions(r["bias"], ref["bias"], domain + " v10 offset " + str(key))
            for method in ("ridge","bias"):
                for p in r[method]:
                    full_rows.append({
                        "domain": domain, "fold": r["fold"], "test_cell": p["id"],
                        "policy": r["policy"], "random_repeat": r["repeat"],
                        "budget_per_labeled_cell": r["budget"], "calibrator": method,
                        "selected": [x["id"] for x in r["selected"]],
                        "n_capacity_labels": sum(x["n_labeled_cycles"] for x in r["selected"]),
                        "mae_mAh": p["mae_mAh"],
                    })
        by_domain[domain] = {"n_target_cells":len(cs),
                             "min_cycles":min(c.n_cycles for c in cs),
                             "max_cycles":max(c.n_cycles for c in cs),
                             "replayed_v10_summary":fresh10["summary"]}
        for budget in (10,20,30):
            for policy in ("random","pred_quartiles"):
                for method in ("ridge","bias"):
                    close(fresh10["summary"][budget][policy][method],
                          old10["summary"][str(budget)][policy][method],
                          domain + f" v10 summary {budget}/{policy}/{method}")

    return {
        "replay_status": "PASS" if not issues else "FAIL",
        "audit_kind": "independent raw-CSV Python replay, not independent external validation",
        "source_repo": v8.SOURCE_REPO,
        "source_commit": v8.COMMIT,
        "source_cells": source_info,
        "source_retrained_coefficients": model.tolist(),
        "n_target_cells": 47,
        "n_checks": n_checked,
        "max_per_cell_mae_abs_diff_mAh": max_diff,
        "prediction_tolerance_mAh": TOL_MAH,
        "issues": issues[:200],
        "n_issues": len(issues),
        "by_domain": by_domain,
        "per_cell_replayed_v10": full_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path,
                        default=ROOT / "data/external/xjtu_processed")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/v11_python_replay_audit.json")
    args = parser.parse_args()
    report = run(args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("replay_status","n_checks","n_issues",
                               "max_per_cell_mae_abs_diff_mAh","issues")},
                     ensure_ascii=False, indent=2), flush=True)
    if report["n_issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
