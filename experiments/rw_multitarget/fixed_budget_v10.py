"""RW8 source -> five XJTU targets: fixed early-cycle label budgets (v10).

Original PINN4SOH XJTU data must be fetched by user; uses v8 frozen source
model and v8 cell-level folds. Two target train-pool cells, first N labels.
Outputs CELL-AVERAGED discharge-capacity MAE in mAh (not SOH percentage points).
This is a retrospective development-set experiment, not independent validation.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

V8 = Path(__file__).resolve().parent/'benchmark.py'
spec = importlib.util.spec_from_file_location('phm_v8_fixed_budget', V8)
v8 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v8
spec.loader.exec_module(v8)
POLICIES = ('random', 'pred_quartiles')
BUDGETS = (10, 20, 30)


def acquire(pool, scaler, source, policy, fold, repeat):
    if len(pool) < 2: raise ValueError('Need two target acquisition candidates')
    if policy == 'random':
        return v8.js_shuffle(range(len(pool)), 20261108 + fold*101 + repeat*7)[:2]
    if policy != 'pred_quartiles': raise ValueError(policy)
    # Unlabeled first-12-cycle observable charging statistics ONLY.
    p = np.array([(v8.design(c.features[:12], scaler)@source).mean() for c in pool])
    ranks = np.argsort(p, kind='stable')
    n = len(ranks)
    a = int(np.floor(0.25*(n-1)+0.5))  # JS Math.round convention
    b = int(np.floor(0.75*(n-1)+0.5))
    if a == b: raise ValueError('Quartile selection collapsed to one cell')
    return [int(ranks[a]), int(ranks[b])]


def label_prefix(pool, indices, budget):
    if budget < 1: raise ValueError('Positive cycle budget required')
    out = []
    for i in indices:
        c = pool[i]
        n = min(budget, c.n_cycles)
        out.append(v8.Cell(c.name, c.group, c.features[:n], c.capacity[:n], c.git_sha))
    return out


def evaluate_domain(cells, domain, scaler, source, budgets=BUDGETS, random_repeats=5):
    cells = sorted(cells, key=lambda c: int(c.name.rsplit('-',1)[1]))
    folds = v8.make_folds(len(cells),domain)
    records = []
    for fold, held in enumerate(folds):
        held = set(held)
        test = [c for i,c in enumerate(cells) if i in held]
        pool = [c for i,c in enumerate(cells) if i not in held]
        source_err = v8.score(test,scaler,source)
        records.append(dict(fold=fold,policy='source',repeat=0,budget=0,
                            heldout=[c.name for c in test],selected=[],ridge=source_err))
        for policy in POLICIES:
            for rep in range(random_repeats if policy=='random' else 1):
                chosen = acquire(pool,scaler,source,policy,fold,rep)
                assert len(set(chosen)) == 2
                assert set(pool[i].name for i in chosen).isdisjoint(c.name for c in test)
                for n in budgets:
                    labeled = label_prefix(pool,chosen,n)
                    correction = v8.ridge(labeled,scaler,v8.RESIDUAL_ALPHA,base=source)
                    # Each selected battery has identical weight regardless of cycles.
                    offset = np.mean([(c.capacity - v8.design(c.features,scaler)@source).mean()
                                      for c in labeled])
                    records.append(dict(fold=fold,policy=policy,repeat=rep,budget=n,
                              heldout=[c.name for c in test],
                              selected=[dict(id=c.name,n_labeled_cycles=c.n_cycles) for c in labeled],
                              ridge=v8.score(test,scaler,source,correction=correction),
                              bias=v8.score(test,scaler,source,offset=float(offset))))
    summary = {}
    for n in budgets:
        summary[n] = {}
        for policy in POLICIES:
            subset = [r for r in records if r['budget']==n and r['policy']==policy]
            summary[n][policy] = {
                key:float(np.mean([p['mae_mAh'] for r in subset for p in r[key]]))
                for key in ('ridge','bias')}
    return dict(domain=domain,n_target_cells=len(cells),
                fold_indices=[[cells[i].name for i in f] for f in folds],
                summary=summary,records=records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache',type=Path,default=Path('data/external/xjtu_processed'))
    ap.add_argument('--v8-source',type=Path,default=Path('results/v8/rw_source8_model.json'))
    ap.add_argument('--output',type=Path,default=Path('outputs/fixed_budget_v10.json'))
    ap.add_argument('--targets',nargs='+',choices=v8.TARGET_GROUPS,default=list(v8.TARGET_GROUPS))
    args = ap.parse_args()
    source = json.loads(args.v8_source.read_text())
    if source['source_commit'] != v8.COMMIT or source['feature_indices'] != list(v8.FEATURES):
        raise ValueError('Frozen v8 source version mismatch')
    model = np.asarray(source['coefficients'])
    scaler = np.asarray(source['mean']), np.asarray(source['sd'])
    results = {}
    for group in args.targets:
        cs = [v8.get_cell(group,i,args.cache) for i in range(1,v8.COUNTS[group]+1)]
        results[group] = evaluate_domain(cs,group,scaler,model)
        print(group,results[group]['summary'],flush=True)
    out = dict(protocol='docs/V10_FIXED_CYCLE_BUDGET_PROTOCOL.md',status='development-screen',
               source_model='results/v8/rw_source8_model.json',
               budgets=BUDGETS,policies=POLICIES,results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,ensure_ascii=False))

if __name__ == '__main__': main()