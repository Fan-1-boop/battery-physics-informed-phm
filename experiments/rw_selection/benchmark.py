"""v9: RW-source acquisition policy benchmark on PINN4SOH processed XJTU data.

Purpose: select only target *training pool* cell IDs from first 12 unlabeled
charging-feature cycles. No target holdout feature or label is used for selection.

This is a battery-capacity regression benchmark in mAh, NOT a reproduction of
Che2025 physics-informed electrode latent states or of the BNN methods cited.
"""
from __future__ import annotations
import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np

V8_PATH = Path(__file__).resolve().parents[2] / 'rw_multitarget' / 'benchmark.py'
if not V8_PATH.is_file():
    V8_PATH = Path(__file__).resolve().parent / '_v8_reference.py'
_spec = importlib.util.spec_from_file_location('phm_v8_benchmark', V8_PATH)
v8 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = v8
_spec.loader.exec_module(v8)

POLICIES = ('random', 'representative', 'pred_extremes', 'pred_quartiles',
            'pred_medoids', 'feature_medoids',
            'uncertainty_top2', 'uncertainty_weighted_medoids')


def unlabeled_embeddings(pool, scaler, source_weights):
    if not pool:
        raise ValueError('Empty target acquisition pool')
    if any(c.n_cycles < 12 for c in pool):
        raise ValueError('Need 12 observable cycles per target training candidate')
    x = np.stack([v8.design(c.features[:12], scaler)[:, 1:].mean(0) for c in pool])
    # Using the same clipping/normalization as the frozen source model.
    yhat = np.c_[np.ones(len(pool)), x] @ source_weights
    return x, yhat


def medoid_pair(embeddings, weights=None):
    """Exact k=2 facility-location medoids, using Euclidean coverage distances."""
    z = np.asarray(embeddings, dtype=float)
    if z.ndim != 2 or len(z) < 2:
        raise ValueError('At least 2 feature vectors required')
    if weights is None:
        weights = np.ones(len(z))
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(z),) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Invalid medoid weights')
    dist = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    return list(min(itertools.combinations(range(len(z)), 2),
                    key=lambda pair: (float(weights @ np.minimum(dist[:, pair[0]],
                                                                  dist[:, pair[1]])), pair)))


def source_ensemble(rw_cells, scaler):
    """One-model-per-left-out-RW-battery disagreement proxy; not calibrated BNN uncertainty."""
    return np.stack([v8.ridge([c for j, c in enumerate(rw_cells) if j != i],
                              scaler, v8.SOURCE_ALPHA) for i in range(len(rw_cells))])


def acquire(pool, scaler, source_model, policy, *, fold, repeat=0, k=2, ensemble=None):
    if policy not in POLICIES:
        raise ValueError('Unknown policy: ' + policy)
    if k != 2:  # frozen study; extending to more labeled cells requires a new protocol
        raise ValueError('v9 methods are intentionally frozen for k=2')
    if len(pool) < k:
        raise ValueError('Too few acquisition candidates')
    if policy == 'random':
        return v8.js_shuffle(range(len(pool)), 20261108 + fold*101 + repeat*7)[:k]
    x, yhat = unlabeled_embeddings(pool, scaler, source_model)
    if policy == 'representative':
        return v8.representative(pool, scaler, k=k)
    if policy == 'pred_extremes':
        ranks = np.argsort(yhat, kind='stable')
        return [int(ranks[0]), int(ranks[-1])]
    if policy == 'pred_quartiles':
        ranks = np.argsort(yhat, kind='stable')
        # Match original JS Math.round(0.25*(n-1)) for positive indices.
        i = int(np.floor(0.25 * (len(ranks)-1) + 0.5))
        j = int(np.floor(0.75 * (len(ranks)-1) + 0.5))
        return [int(ranks[i]), int(ranks[j])]
    if policy == 'pred_medoids':
        return medoid_pair(yhat[:, None])
    if policy == 'feature_medoids':
        return medoid_pair(x)
    if ensemble is None:
        raise ValueError('Uncertainty policy requires RW source LOO ensemble')
    q = np.c_[np.ones(len(pool)), x] @ ensemble.T
    disagreement = q.std(axis=1, ddof=0)
    if policy == 'uncertainty_top2':
        return sorted(range(len(pool)), key=lambda i: (-disagreement[i], i))[:2]
    return medoid_pair(x, weights=disagreement + 1e-6)


def run_domain(cells, group, source_model, scaler, *, policies, ensemble=None, label_fraction=0.25,
               random_repeats=5):
    if not 0 < label_fraction <= 1:
        raise ValueError('label_fraction must be in (0,1]')
    cells = sorted(cells, key=lambda c: int(c.name.rsplit('-', 1)[1]))
    folds = v8.make_folds(len(cells), group)
    out = []
    for fold, held_idx in enumerate(folds):
        held = set(held_idx)
        test = [c for i, c in enumerate(cells) if i in held]
        pool = [c for i, c in enumerate(cells) if i not in held]
        assert set(c.name for c in pool).isdisjoint(c.name for c in test)
        out.append({'fold':fold, 'policy':'source', 'repeat':0, 'selected':[],
                    'test':v8.score(test,scaler,source_model)})
        for policy in policies:
            for repeat in range(random_repeats if policy == 'random' else 1):
                idx = acquire(pool, scaler, source_model, policy, fold=fold, repeat=repeat,
                              ensemble=ensemble)
                assert len(set(idx)) == 2
                labeled = []
                for ix in idx:
                    c = pool[ix]
                    n = max(1, int(np.ceil(c.n_cycles * label_fraction)))
                    labeled.append(v8.Cell(c.name, c.group, c.features[:n], c.capacity[:n], c.git_sha))
                assert set(c.name for c in labeled).isdisjoint(c.name for c in test)
                delta = v8.ridge(labeled, scaler, v8.RESIDUAL_ALPHA, base=source_model)
                out.append({'fold': fold, 'policy': policy, 'repeat': repeat,
                            'selected':[{'id':c.name,'n_label_cycles':c.n_cycles} for c in labeled],
                            'test':v8.score(test,scaler,source_model,correction=delta)})
    summary = {}
    for policy in ['source',*policies]:
        errs = [pred['mae_mAh'] for record in out if record['policy'] == policy
                for pred in record['test']]
        summary[policy] = {'mae_mAh':float(np.mean(errs)), 'n_cell_predictions':len(errs)}
    return {'domain':group, 'fold_indices':[[cells[i].name for i in fold] for fold in folds],
            'target_cells':[{'id':c.name,'sha':c.git_sha,'cycles':c.n_cycles} for c in cells],
            'records':out, 'summary':summary}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, default=Path('data/external/xjtu_processed'))
    ap.add_argument('--output', type=Path, default=Path('outputs/rw_selection_v9.json'))
    ap.add_argument('--targets', nargs='+', choices=v8.TARGET_GROUPS, default=list(v8.TARGET_GROUPS))
    ap.add_argument('--label-frac', type=float, default=.25)
    ap.add_argument('--random-repeats', type=int, default=5)
    ap.add_argument('--policies', nargs='+', choices=POLICIES, default=list(POLICIES))
    args = ap.parse_args()
    rw = [v8.get_cell('RW',i,args.cache) for i in range(1,9)]
    scaler = v8.fit_scaler(rw)
    model = v8.ridge(rw, scaler, v8.SOURCE_ALPHA)
    use_ensemble = any('uncertainty' in policy for policy in args.policies)
    ensemble = source_ensemble(rw,scaler) if use_ensemble else None
    result = {'source':'RW8', 'dataset':'PINN4SOH XJTU cycle-aggregated features',
              'source_repo':v8.SOURCE_REPO, 'commit':v8.COMMIT,
              'feature_columns':v8.FEATURES,'label_budget_cells':2,
              'label_fraction':args.label_frac,'random_repeats':args.random_repeats,
              'policies':args.policies,
              'method_note':'Literature-inspired adaptations, NOT official methods or Che latent transfer',
              'source_weights':model.tolist(),'results':{}}
    for group in args.targets:
        cells = [v8.get_cell(group,i,args.cache) for i in range(1,v8.COUNTS[group]+1)]
        result['results'][group] = run_domain(cells,group,model,scaler,policies=args.policies,
                                               ensemble=ensemble,label_fraction=args.label_frac,
                                               random_repeats=args.random_repeats)
        print(group,result['results'][group]['summary'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__': main()