"""Independent-cell, leave-condition-out pilot and full XJTU benchmark.

Original processed features belong to wang-fujin/PINN4SOH. This script
fetches them when run on a networked machine. No original data is distributed.

Important: cycle-level charge statistics -> capacity. This is NOT Che2025
20-minute V(Q) -> SOH and NOT its 4 physical electrode latent states.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SOURCE_REPO = "wang-fujin/PINN4SOH"
SOURCE_COMMIT = "cc3cc5053caf38f04e0665f7f88cb109144d035e"
GROUP_COUNTS = {"2C": 8, "3C": 15, "R2.5": 8, "R3": 8, "RW": 8, "Sim_satellite": 8}
FEATURE_INDICES = (0, 4, 5, 8, 12, 13)
FEATURE_LABELS = ("voltage mean", "CC Q", "CC charge time", "current mean", "CV Q", "CV charge time")


@dataclass(frozen=True)
class Cell:
    cell_id: str
    group: str
    features: np.ndarray  # cycles x six observable charging features
    capacity_ah: np.ndarray  # discharge capacity per cycle
    git_blob_sha: str

    @property
    def n_cycles(self) -> int:
        return self.features.shape[0]


def git_blob_sha(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def load_cell_csv(raw: bytes, name: str) -> Cell:
    import io
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")))
    header = next(reader)
    if len(header) != 17 or header[-1].strip().lower() != "capacity":
        raise ValueError(f"Incompatible XJTU processed format: {name}, header={header}")
    x, y = [], []
    for line in reader:
        try:
            row = np.asarray([float(s) for s in line], dtype=np.float64)
        except (ValueError, TypeError):
            continue
        if len(row) != 17 or not np.all(np.isfinite(row)) or row[-1] <= 0:
            continue
        x.append(row[list(FEATURE_INDICES)])
        y.append(row[-1])
    if not x:
        raise ValueError(f"No valid cycles: {name}")
    return Cell(name.removesuffix(".csv"), name.split("_battery-")[0],
                np.asarray(x), np.asarray(y), git_blob_sha(raw))


def fetch_cell(name: str, cache: Path) -> Cell:
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / name
    if not dst.exists():
        escaped = urllib.parse.quote(name)
        url = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/data/XJTU%20data/{escaped}"
        with urllib.request.urlopen(url, timeout=50) as response:
            blob = response.read()
        if len(blob) > 3_000_000:
            raise ValueError("Unexpectedly large dataset object")
        dst.write_bytes(blob)
    return load_cell_csv(dst.read_bytes(), name)


def cell_names(full: bool) -> list[str]:
    if full:
        return [f"{group}_battery-{i}.csv" for group, n in GROUP_COUNTS.items() for i in range(1, n + 1)]
    return [f"{group}_battery-{i}.csv" for group in ("3C", "RW") for i in range(1, 7)]


def scaler_fit(cells: list[Cell]):
    means = np.vstack([cell.features.mean(axis=0) for cell in cells])
    squares = np.vstack([(cell.features ** 2).mean(axis=0) for cell in cells])
    mean = means.mean(axis=0)
    sd = np.sqrt(np.maximum(squares.mean(axis=0) - mean**2, 0))
    return mean, np.maximum(sd, 1e-5)


def design(x: np.ndarray, stats) -> np.ndarray:
    mean, sd = stats
    z = np.clip((x - mean) / sd, -6, 6)
    return np.column_stack([np.ones(len(z)), z])


def ridge_fit(cells: list[Cell], stats, alpha: float, source_model=None) -> np.ndarray:
    gram = np.zeros((7, 7), dtype=float)
    rhs = np.zeros(7, dtype=float)
    for cell in cells:
        z = design(cell.features, stats)
        y = cell.capacity_ah.copy()
        if source_model is not None:
            y -= z @ source_model
        w = 1.0 / len(y)  # every battery counts once
        gram += w * (z.T @ z)
        rhs += w * (z.T @ y)
    gram[np.arange(1, 7), np.arange(1, 7)] += alpha  # penalize slopes, not intercept
    return np.linalg.solve(gram, rhs)


def cell_mae_mAh(cell: Cell, stats, source_b: np.ndarray, offset: float = 0,
                  correction: np.ndarray | None = None) -> float:
    z = design(cell.features, stats)
    pred = z @ source_b + offset
    if correction is not None:
        pred += z @ correction
    return float(np.abs(pred - cell.capacity_ah).mean() * 1000)


def pick_representative(pool: list[Cell], k: int, stats) -> list[int]:
    vectors = np.asarray([design(c.features[:12], stats)[:, 1:].mean(axis=0) for c in pool])
    distances = ((vectors - vectors.mean(axis=0))**2).sum(axis=1)
    selected = [int(np.argmin(distances))]
    while len(selected) < k:
        d = ((vectors[:, None, :] - vectors[np.array(selected)][None, :, :])**2).sum(axis=2).min(axis=1)
        d[selected] = -1
        selected.append(int(np.argmax(d)))
    return selected


def pick_random(pool_size: int, k: int, seed: int) -> list[int]:
    a = list(range(pool_size))
    x = seed & 0xFFFFFFFF
    for i in range(pool_size-1, 0, -1):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        j = x % (i+1)
        a[i], a[j] = a[j], a[i]
    return a[:k]


def perform(cells: list[Cell], *, mode: str = "pilot", repeats: int = 5,
            label_budgets: tuple[int, ...] = (1, 2), label_fraction: float = 1.0,
            max_holdouts: int | None = None, risk_gate: bool = False):
    """Hold out fixed target battery IDs BEFORE any label acquisition.

    label_fraction applies only to the available cycles of the k chosen target
    training batteries. All test batteries stay unseen as labeled examples.
    """
    records = []
    groups = ["3C", "RW"] if mode == "pilot" else list(GROUP_COUNTS)
    for gi, group in enumerate(groups):
        source = [c for c in cells if c.group != group]
        target = [c for c in cells if c.group == group]
        stats = scaler_fit(source)
        source_b = ridge_fit(source, stats, 1.0)
        pairs = list(itertools.combinations(range(len(target)), 2))
        if max_holdouts is not None and len(pairs) > max_holdouts:
            rng = np.random.default_rng(20261008 + gi)
            subset = np.sort(rng.choice(len(pairs), size=max_holdouts, replace=False))
            pairs = [pairs[i] for i in subset]
        for ti, tj in pairs:
            test = [target[ti], target[tj]]
            pool = [c for i, c in enumerate(target) if i not in (ti, tj)]
            for k in label_budgets:
                if k > len(pool):
                    continue
                for rep in range(repeats):
                    for policy in ("random", "representative"):
                        inds = (pick_random(len(pool), k, 8789 + gi * 1013 + ti * 167 + tj * 29 + rep * 311)
                                if policy == "random" else pick_representative(pool, k, stats))
                        labeled_full = [pool[i] for i in inds]
                        labeled = []
                        for c in labeled_full:
                            n = max(1, int(np.ceil(c.n_cycles * label_fraction)))
                            labeled.append(Cell(c.cell_id, c.group, c.features[:n], c.capacity_ah[:n], c.git_blob_sha))
                        off = float(np.mean([np.mean(c.capacity_ah - design(c.features, stats) @ source_b)
                                             for c in labeled]))
                        delta = ridge_fit(labeled, stats, 10.0, source_model=source_b)
                        base = float(np.mean([cell_mae_mAh(c, stats, source_b) for c in test]))
                        vals = {
                            "source": base,
                            "offset": float(np.mean([cell_mae_mAh(c, stats, source_b, offset=off) for c in test])),
                            "residual_ridge": float(np.mean([cell_mae_mAh(c, stats, source_b, correction=delta) for c in test])),
                        }
                        gate_accept_offset = False
                        gate_accept_ridge = False
                        if risk_gate and k >= 2:
                            # No target TEST labels here. Entire validation group is
                            # held-out among the k *labeled acquisition* batteries.
                            gain_offset = 0.0
                            gain_ridge = 0.0
                            for vi, validation in enumerate(labeled):
                                train_other = [c for li, c in enumerate(labeled) if li != vi]
                                source_mae = cell_mae_mAh(validation, stats, source_b)
                                one_offset = float(np.mean([np.mean(
                                    c.capacity_ah - design(c.features, stats) @ source_b)
                                    for c in train_other]))
                                one_delta = ridge_fit(train_other, stats, 10.0, source_model=source_b)
                                gain_offset += source_mae - cell_mae_mAh(validation, stats, source_b, offset=one_offset)
                                gain_ridge += source_mae - cell_mae_mAh(validation, stats, source_b, correction=one_delta)
                            gate_accept_offset = gain_offset > 0
                            gate_accept_ridge = gain_ridge > 0
                        if risk_gate:
                            vals["gate_offset"] = vals["offset"] if gate_accept_offset else base
                            vals["gate_ridge"] = vals["residual_ridge"] if gate_accept_ridge else base
                        for method, mae in vals.items():
                            records.append({"target_domain": group, "heldout_test_ids": [c.cell_id for c in test],
                                "acquisition_pool_ids": [c.cell_id for c in pool], "k": k, "repeat": rep,
                                "policy": policy, "method": method, "labeled_ids": [c.cell_id for c in labeled_full],
                                "capacity_MAE_mAh": mae, "paired_gain_mAh": base - mae,
                                "n_test_cells": len(test), "label_cycle_fraction": label_fraction,
                                "gate_accept_offset": gate_accept_offset,
                                "gate_accept_ridge": gate_accept_ridge})
    return records


def summarize(records: list[dict]) -> list[dict]:
    keys = sorted({(r["target_domain"], r["k"], r["policy"], r["method"]) for r in records})
    return [{"domain": key[0], "k": key[1], "policy": key[2], "method": key[3],
             "n_split_repeats": len(v := [r for r in records if (r["target_domain"], r["k"], r["policy"], r["method"]) == key]),
             "capacity_MAE_mAh": float(np.mean([r["capacity_MAE_mAh"] for r in v])),
             "paired_gain_mAh": float(np.mean([r["paired_gain_mAh"] for r in v]))}
            for key in keys]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--cache", type=Path, default=Path("data/external/xjtu_processed"))
    ap.add_argument("--output", type=Path, default=Path("outputs/xjtu_v7.json"))
    ap.add_argument("--label-fraction", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-holdouts", type=int, default=None)
    ap.add_argument("--risk-gate", action="store_true", help="Cell-wise LOO gate using ONLY labeled target acquisition cells")
    args = ap.parse_args()
    if not 0 < args.label_fraction <= 1:
        ap.error("--label-fraction must belong to (0,1]")
    data = [fetch_cell(name, args.cache) for name in cell_names(args.mode == "full")]
    records = perform(data, mode=args.mode, repeats=args.repeats,
                      label_fraction=args.label_fraction, max_holdouts=args.max_holdouts,
                      risk_gate=args.risk_gate)
    out = {"mode": args.mode, "data_source_repo": SOURCE_REPO, "data_source_commit": SOURCE_COMMIT,
           "n_cells": len(data), "n_cycles": sum(c.n_cycles for c in data),
           "feature_indices": FEATURE_INDICES, "feature_labels": FEATURE_LABELS,
           "full_target_cell_trajectories_labeled": args.label_fraction == 1,
           "label_fraction": args.label_fraction, "risk_gate": args.risk_gate,
           "warning": "Independent NCM capacity benchmark, NOT Che NCA electrode latent transfer",
           "cells": [{"id": c.cell_id, "cycles": c.n_cycles, "git_blob_sha": c.git_blob_sha} for c in data],
           "summary": summarize(records), "per_run": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"n_cells": len(data), "n_records": len(records), "summary": out["summary"]}, indent=2))


if __name__ == "__main__":
    main()