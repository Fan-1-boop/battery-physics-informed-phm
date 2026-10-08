"""Descriptive (exploratory) blocked statistics for Dataset2 v6 benchmark."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/v6/formal'
frame=pd.read_csv(OUT/'per_run.csv')
assert len(frame)==6480 and frame['fold'].nunique()==5 and frame['rate_idx'].nunique()==6
assert len(frame.groupby(['fold','repeat','k','rate','policy','method']))==len(frame)
# confirm v5 ridge physical results up to rounding/scikit-learn conventions
old=pd.read_csv(ROOT/'results/v5/cross_rate_summary_loo.csv')
new=frame.query("method=='ridge_physics'").groupby(['k','policy'],as_index=False).soh_mae_pp.mean()
vm={'input_cluster':'input_clustering','physical_cluster':'physics_clustering','source_cluster':'source_soh_clustering','random':'random'}
for _,r in new.iterrows():
    exact=old.loc[(old['k']==r['k']) & (old['policy']==vm[r['policy']]), 'mean'].iloc[0]
    assert abs(float(exact)-float(r.soh_mae_pp))<0.0001, (r,exact)

# Fold blocking: trials share test cells; CI uses only 5 independent partition IDs, not 90 runs.
foldmeans=frame.groupby(['fold','k','policy','method'],as_index=False).soh_mae_pp.mean()
foldmeans.to_csv(OUT/'foldmeans.csv',index=False)
comparisons=[
 (4,'ridge_physics','physical_cluster','ridge_physics','random'),
 (4,'ridge_source','physical_cluster','ridge_source','random'),
 (4,'ridge_source','physical_cluster','ridge_physics','physical_cluster'),
 (4,'ridge_source','source_cluster','ridge_physics','source_cluster'),
 (14,'ridge_physics','input_cluster','ridge_source','input_cluster'),
 (14,'ridge_physics','input_cluster','ridge_physics','physical_cluster'),
 (14,'ridge_physics','source_cluster','ridge_source','source_cluster'),
]
stat=[]
for k,m1,p1,m2,p2 in comparisons:
    a=foldmeans.query('k==@k and method==@m1 and policy==@p1').sort_values('fold').soh_mae_pp.to_numpy()
    b=foldmeans.query('k==@k and method==@m2 and policy==@p2').sort_values('fold').soh_mae_pp.to_numpy()
    assert len(a)==len(b)==5
    d=a-b
    avg=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(len(d)))
    low=avg-float(t_dist.ppf(.975,4))*se;high=avg+float(t_dist.ppf(.975,4))*se
    stat.append(dict(k=k,method_a=m1,policy_a=p1,mae_a=float(a.mean()),method_b=m2,policy_b=p2,mae_b=float(b.mean()),
     diff_pp=avg,ci95_low=low,ci95_high=high,n_fold=5,fold_benefit_count=int((d<0).sum())))
pd.DataFrame(stat).to_csv(OUT/'paired_fold_ci.csv',index=False)

# Separate tests: learning method and acquisition policy must be reported independently.
subset=frame.query("(policy=='physical_cluster' or policy=='random') and method in ['ridge_physics','ridge_source','krr_rbf','gp_rbf','svr_rbf','catboost','xgboost','extra_trees']")
group=subset.groupby(['k','policy','method'],as_index=False).soh_mae_pp.mean()
fig,ax=plt.subplots(figsize=(11,5))
keep=['ridge_source','ridge_physics','krr_rbf','gp_rbf','svr_rbf','extra_trees','catboost','xgboost']
xs=np.arange(len(keep))
width=.21
for i,(k,policy) in enumerate([(4,'random'),(4,'physical_cluster'),(14,'random'),(14,'physical_cluster')]):
    lookup={(q.method):q.soh_mae_pp for q in group.itertuples() if q.k==k and q.policy==policy}
    ax.bar(xs+(i-1.5)*width,[lookup[s] for s in keep],width,label=f'{k} labels / {policy}')
ax.set_xticks(xs,keep,rotation=35,ha='right')
ax.set_ylabel('SOH MAE (percentage points)')
ax.set_title('Independent method-family baselines, same Dataset 2 splits / labeled cells')
ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'method_family_comparison.png',dpi=160);plt.close(fig)

fig,ax=plt.subplots(figsize=(8,5))
for meth in ['ridge_physics','ridge_source','krr_rbf','gp_rbf']:
    y=frame.query('policy=="physical_cluster" and method==@meth').groupby('k').soh_mae_pp.mean().sort_index()
    ax.plot(y.index,y.values,'o-',label=meth)
ax.set_xticks([4,14]);ax.set_xlabel('Labeled target-training cells');ax.set_ylabel('SOH MAE (percentage points)')
ax.set_title('Representative cell acquisition; label-budget effect')
ax.legend();fig.tight_layout();fig.savefig(OUT/'label_budget_methods.png',dpi=160);plt.close(fig)

compact=frame.drop(columns=['prediction','train_ids'])
compact.to_csv(OUT/'per_run_compact.csv',index=False)
print('PASS v6-v5 numeric equivalence; rows',len(frame),'methods',frame.method.nunique())
print(pd.DataFrame(stat).round(4).to_string(index=False))
print(group.sort_values(['k','policy','soh_mae_pp']).round(3).to_string(index=False))