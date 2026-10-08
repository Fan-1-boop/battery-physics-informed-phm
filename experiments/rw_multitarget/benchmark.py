"""RW-source cross-condition battery-capacity benchmark (v8).

Data: PINN4SOH's processed XJTU CSVs, by Wang et al. (Nature Communications,
2024). This does NOT reconstruct Che2025's electrode physical latent variables.

Author data is downloaded or supplied outside this repository.
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

COMMIT = 'cc3cc5053caf38f04e0665f7f88cb109144d035e'
SOURCE_REPO = 'wang-fujin/PINN4SOH'
TARGET_GROUPS = ('3C','2C','R2.5','R3','Sim_satellite')
COUNTS = {'RW':8,'3C':15,'2C':8,'R2.5':8,'R3':8,'Sim_satellite':8}
FEATURES = (0,4,5,8,12,13)
SOURCE_ALPHA = 1.0
RESIDUAL_ALPHA = 10.0

@dataclass
class Cell:
    name: str
    group: str
    features: np.ndarray
    capacity: np.ndarray
    git_sha: str = ''

    @property
    def n_cycles(self): return len(self.capacity)


def load_csv(raw: bytes, filename: str) -> Cell:
    import io
    csvread = csv.reader(io.StringIO(raw.decode('utf-8-sig')))
    header = next(csvread)
    if len(header)!=17 or header[-1].lower().strip() != 'capacity':
        raise ValueError('Expected 16 charging stats and capacity, got '+repr(header))
    rows=[]
    for line in csvread:
        try: v=np.array([float(x) for x in line])
        except ValueError: continue
        if len(v)==17 and np.isfinite(v).all() and v[-1]>0: rows.append(v)
    if not rows: raise ValueError('No valid cycles in '+filename)
    a=np.array(rows,dtype=float)
    return Cell(filename.removesuffix('.csv'),filename.split('_battery-')[0],a[:,FEATURES],a[:,-1],
                hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\x00'+raw).hexdigest())


def get_cell(group:str, idx:int, cache:Path):
    name=f'{group}_battery-{idx}.csv'
    file=cache/name
    if not file.exists():
        cache.mkdir(parents=True,exist_ok=True)
        url=(f'https://raw.githubusercontent.com/{SOURCE_REPO}/{COMMIT}'
             f'/data/XJTU%20data/{urllib.parse.quote(name)}')
        with urllib.request.urlopen(url,timeout=90) as r: raw=r.read()
        if len(raw)>3_000_000: raise ValueError('Unusually large cell file')
        file.write_bytes(raw)
    return load_csv(file.read_bytes(),name)


def fit_scaler(cells):
    if not cells: raise ValueError('No source cells')
    avg=np.mean([c.features.mean(axis=0) for c in cells],axis=0)
    avg2=np.mean([(c.features**2).mean(axis=0) for c in cells],axis=0)
    return (avg,np.maximum(np.sqrt(np.maximum(avg2-avg**2,0)),1e-5))


def design(x,scaler):
    return np.c_[np.ones(len(x)),np.clip((x-scaler[0])/scaler[1],-6,6)]


def ridge(cells,scaler,alpha,base=None):
    g=np.zeros((7,7));h=np.zeros(7)
    for c in cells:
        x=design(c.features,scaler)
        y=c.capacity-(x@base if base is not None else 0)
        w=1/len(y)
        g+=w*x.T@x
        h+=w*x.T@y
    g[1:,1:]+=alpha*np.eye(6)
    return np.linalg.solve(g,h)


def xorshift32(x):
    # Match the historical JS v8 fold/shuffle routine.
    x=(x^(x<<13))&0xffffffff
    x=(x^(x>>17))&0xffffffff
    x=(x^(x<<5))&0xffffffff
    return x


def js_shuffle(arr,seed):
    a=list(arr);x=seed&0xffffffff
    for i in range(len(a)-1,0,-1):
        x=xorshift32(x)
        j=x%(i+1)
        a[i],a[j]=a[j],a[i]
    return a


def make_folds(n,domain):
    # v8 ID order is numeric 1..count; domain-specific seeds are frozen.
    domain_index={'3C':1,'2C':2,'R2.5':3,'R3':4,'Sim_satellite':5}[domain]
    idx=js_shuffle(range(n),20261008+domain_index)
    return [idx[i::5] for i in range(5)]


def representative(pool,scaler,k=2):
    vectors=np.array([design(c.features[:12],scaler)[:,1:].mean(axis=0) for c in pool])
    d=np.sum((vectors-vectors.mean(axis=0))**2,axis=1)
    pick=[int(np.argmin(d))]
    while len(pick)<k:
        d=np.sum((vectors[:,None,:]-vectors[np.array(pick)][None,:,:])**2,axis=2).min(axis=1)
        d[pick]=-1
        pick.append(int(np.argmax(d)))
    return pick


def score(test,scaler,source_delta,correction=None,offset=0):
    out=[]
    for c in test:
        x=design(c.features,scaler)
        yhat=x@source_delta+offset
        if correction is not None: yhat=yhat+x@correction
        out.append({'id':c.name,'mae_mAh':float(np.abs(yhat-c.capacity).mean()*1000)})
    return out


def run_domain(cells,group,source_model,scaler,*,k=2,label_frac=.25,random_repeats=5):
    if not 0<label_frac<=1:raise ValueError('label_frac must be (0,1]')
    cells=sorted(cells,key=lambda c:int(c.name.rsplit('-',1)[1]))
    folds=make_folds(len(cells),group)
    records=[]
    for fi,held_idx in enumerate(folds):
        held=set(held_idx)
        test=[c for j,c in enumerate(cells) if j in held]
        pool=[c for j,c in enumerate(cells) if j not in held]
        if len(pool)<k:raise ValueError('Too few acquisition batteries')
        baseline=score(test,scaler,source_model)
        records.append({'fold':fi,'policy':'source','repeat':0,'heldout':[c.name for c in test],
                        'labeled':[],'per_cell':baseline})
        for pol in ('representative','random'):
            for rep in range(random_repeats if pol=='random' else 1):
                chosen=(representative(pool,scaler,k) if pol=='representative' else
                        js_shuffle(range(len(pool)),20261108+fi*101+rep*7)[:k])
                labeled=[Cell(pool[j].name,group,pool[j].features[:n],pool[j].capacity[:n],pool[j].git_sha)
                         for j in chosen for n in [max(1,int(np.ceil(pool[j].n_cycles*label_frac)))]]
                assert set(c.name for c in test).isdisjoint(c.name for c in labeled)
                delta=ridge(labeled,scaler,RESIDUAL_ALPHA,base=source_model)
                offsets=np.mean([np.mean(c.capacity-design(c.features,scaler)@source_model)
                                 for c in labeled])
                records.append({'fold':fi,'policy':pol,'repeat':rep,'heldout':[c.name for c in test],
                    'labeled':[{'id':c.name,'n_labeled_cycles':c.n_cycles} for c in labeled],
                    'per_cell':score(test,scaler,source_model,correction=delta),
                    'per_cell_offset':score(test,scaler,source_model,offset=offsets)})
    # Each target battery appears once as test per repeat. Repeat count is not a count of independent cells.
    summary={}
    for policy in ('source','random','representative'):
        rr=[r for r in records if r['policy']==policy]
        mae=[v['mae_mAh'] for r in rr for v in r['per_cell']]
        summary[policy]={'mae_mAh':float(np.mean(mae)), 'n_cell_evaluations':len(mae)}
        if policy!='source':
            offset=[v['mae_mAh'] for r in rr for v in r['per_cell_offset']]
            summary[policy]['offset_mae_mAh']=float(np.mean(offset))
    return {'group':group,'fold_indices':[[cells[i].name for i in f] for f in folds],
        'target_cells':[{'id':c.name,'sha':c.git_sha,'n_cycles':c.n_cycles} for c in cells],
        'records':records,'summary':summary}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache',type=Path,default=Path('data/external/xjtu_processed'))
    ap.add_argument('--output',type=Path,default=Path('outputs/rw_multitarget_v8.json'))
    ap.add_argument('--targets',nargs='+',choices=TARGET_GROUPS,default=list(TARGET_GROUPS))
    ap.add_argument('--k',type=int,default=2)
    ap.add_argument('--label-frac',type=float,default=.25)
    ap.add_argument('--random-repeats',type=int,default=5)
    args=ap.parse_args()
    source=[get_cell('RW',i,args.cache) for i in range(1,9)]
    scaler=fit_scaler(source)
    model=ridge(source,scaler,SOURCE_ALPHA)
    results={}
    for domain in args.targets:
        cells=[get_cell(domain,i,args.cache) for i in range(1,COUNTS[domain]+1)]
        results[domain]=run_domain(cells,domain,model,scaler,k=args.k,label_frac=args.label_frac,
                                   random_repeats=args.random_repeats)
        print(domain,results[domain]['summary'],flush=True)
    data={'status':'v8 RW-only source, target cell-disjoint fivefold',
          'source_repo':SOURCE_REPO,'source_commit':COMMIT,
          'source_cells':[{'id':c.name,'sha':c.git_sha,'n_cycles':c.n_cycles} for c in source],
          'feature_indices':FEATURES,'scaler':{'mean':scaler[0].tolist(),'std':scaler[1].tolist()},
          'source_model':model.tolist(),'source_alpha':SOURCE_ALPHA,'residual_alpha':RESIDUAL_ALPHA,
          'k':args.k,'label_frac':args.label_frac,'targets':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()