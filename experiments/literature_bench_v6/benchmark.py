"""Few-shot battery SOH benchmark inspired by recent literature METHOD FAMILIES.

IMPORTANT: This is NOT an exact implementation of any named paper. The pretrained
Che 2025 source model's training population overlaps Dataset 2 physically.

Both the acquisition and feature preprocessing use ONLY an outer training pool.
Hyperparameters are selected by LOOCV on the k *labeled* target cells only.
Evaluation reads outer test labels only AFTER prediction.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

V5 = Path(__file__).resolve().parents[1] / 'fewshot_physical_transfer'
sys.path.insert(0,str(V5))
from run import load_model, split_data
from label_acquisition import choose_kmeans

RATES = ['C/80','C/40','0.05A','C/10','C/7','C/5']
METHODS = ['ridge_physics','ridge_source','ridge_input','krr_rbf','svr_rbf','gp_rbf','catboost','xgboost','extra_trees']


def source_features(cache: Path, assets: Path):
    d = np.load(cache, allow_pickle=False)
    raw = d['X']
    mdl = load_model(assets)
    with torch.no_grad():
        pred, derived, latent, _, _ = mdl(torch.tensor(raw,dtype=torch.float32))
    p = pred.numpy()
    s = p[:,-1,1]*100
    z = np.concatenate([v.numpy() for v in latent],axis=1)
    consistency = np.mean(np.abs(p[:,1:,0]-derived[:,1:].numpy()),axis=1)*4200
    phys = np.column_stack([s,z,consistency])
    ins = np.column_stack([raw[:,0,0],raw[:,-1,0],raw[:,-1,1],raw[:,:,0].mean(axis=1),raw[:,:,0].std(axis=1),raw[:,:,1].mean(axis=1)])
    return dict(source=s, physical=phys, input_summary=ins), d


def normalize_features(x: np.ndarray, train_ids: np.ndarray):
    center=x[train_ids].mean(0)
    scale=np.maximum(x[train_ids].std(0),1e-4)
    return (x-center)/scale


def _fit(method: str, x, y, params: dict):
    if method.startswith('ridge_'):
        return Ridge(alpha=params['alpha']).fit(x,y)
    if method=='krr_rbf':
        return KernelRidge(kernel='rbf',alpha=params['alpha'],gamma=params['gamma']).fit(x,y)
    if method=='svr_rbf':
        return SVR(kernel='rbf',C=params['C'],epsilon=params['epsilon'],gamma='scale').fit(x,y)
    if method=='gp_rbf':
        # No marginal-likelihood optimization on 4 training labels; fixed prior.
        kernel=ConstantKernel(4.,constant_value_bounds='fixed')*RBF(params['length'],length_scale_bounds='fixed')
        return GaussianProcessRegressor(kernel=kernel,alpha=params['alpha'],optimizer=None,normalize_y=False).fit(x,y)
    if method=='catboost':
        return CatBoostRegressor(iterations=100,depth=2,learning_rate=0.045,l2_leaf_reg=8,
                                 loss_function='RMSE',verbose=False,thread_count=1,random_seed=123).fit(x,y)
    if method=='xgboost':
        return XGBRegressor(n_estimators=100,max_depth=2,learning_rate=.045,reg_lambda=15,
                             objective='reg:squarederror',subsample=1,colsample_bytree=1,
                             n_jobs=1,random_state=123).fit(x,y)
    if method=='extra_trees':
        return ExtraTreesRegressor(n_estimators=80,max_features=1.,min_samples_leaf=2,
                                   random_state=123,n_jobs=1).fit(x,y)
    raise ValueError(method)


def candidates(method: str):
    if method.startswith('ridge_'):
        return [{'alpha':a} for a in (0.1,1.,10.,100.)]
    if method=='krr_rbf':
        return [{'alpha':a,'gamma':g} for a in (.1,1.,10.) for g in (.05,.5)]
    if method=='svr_rbf':
        return [{'C':c,'epsilon':e} for c in (1.,10.,100.) for e in (.2,1.)]
    if method=='gp_rbf':
        return [{'alpha':a,'length':l} for a in (1.,10.) for l in (1.,3.)]
    return [{}]


def select_labeled_only(method: str,x:np.ndarray,y:np.ndarray):
    opts=candidates(method)
    if len(opts)==1:
        return opts[0],float('nan')
    loo=list(LeaveOneOut().split(x))
    errors=[]
    for choice in opts:
        abs_errors=[]
        for a,b in loo:
            m=_fit(method,x[a],y[a],choice)
            pred=m.predict(x[b]);abs_errors.append(abs(float(pred[0]-y[b[0]])))
        errors.append(float(np.mean(abs_errors)))
    best=int(np.argmin(errors))
    return opts[best], errors[best]


def get_results(features, truth, outer_train, chosen, test, method):
    source=features['source']
    feature_key=('source' if method=='ridge_source' else 'input_summary' if method=='ridge_input' else 'physical')
    x=normalize_features(np.atleast_2d(features[feature_key]).T if features[feature_key].ndim==1 else features[feature_key], outer_train)
    residual=truth[chosen]-source[chosen]
    pars,loo_err=select_labeled_only(method,x[chosen],residual)
    model=_fit(method,x[chosen],residual,pars)
    pred=source[test]+model.predict(x[test])
    return np.asarray(pred,dtype=float),pars,loo_err


def run(cache:Path,assets:Path,out:Path,methods,policies,k_values,repeats,folds):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    features,d=source_features(cache,assets)
    n=len(features['source'])
    assert n==91, n
    truth=np.stack([d[f'Y{i}'][:,-1,1]*100 for i in range(6)],axis=1)
    allrows=[]
    for fold in folds:
        for k in k_values:
            for rep in range(repeats):
                tr,te,ran,_=split_data(n,fold,k,rep)
                select={'random':ran}
                for pol in policies:
                    if pol!='random':
                        repr=features[{'physical_cluster':'physical','source_cluster':'source','input_cluster':'input_summary'}[pol]]
                        if repr.ndim == 1: repr=repr[:,None]
                        select[pol]=choose_kmeans(tr,repr,k,rep)
                assert all(set(v).issubset(set(tr)) for v in select.values())
                for pol,lab in select.items():
                    for rid,rate in enumerate(RATES):
                        y=truth[:,rid]
                        for meth in methods:
                            pred,params,loocv=get_results(features,y,tr,lab,te,meth)
                            errs=pred-y[te]
                            allrows.append(dict(fold=fold,repeat=rep,k=k,rate=rate,rate_idx=rid,policy=pol,method=meth,
                             soh_mae_pp=float(np.abs(errs).mean()),soh_rmse_pp=float(np.sqrt(np.mean(errs**2))),
                             selected_hyperparameters=json.dumps(params,sort_keys=True),labeled_loo_mae=loocv,
                             label_ids=';'.join(map(str,lab)),test_ids=';'.join(map(str,te)),train_ids=';'.join(map(str,tr)),
                             prediction=';'.join(f'{v:.7f}' for v in pred)))
        print(f'fold {fold} completed; rows={len(allrows)}',flush=True)
        pd.DataFrame(allrows).to_csv(out/'per_run.csv',index=False)
    frame=pd.DataFrame(allrows)
    summary=frame.groupby(['k','policy','method']).agg(n_runs=('soh_mae_pp','size'),mae_pp=('soh_mae_pp','mean'),sd_pp=('soh_mae_pp','std'),rmse_pp=('soh_rmse_pp','mean')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    config=dict(cache_sha256=None,source_model='Che2025 Dataset1 best_model.pth',
     warning='Source model training includes overlapping cells; not unseen-cell or cross-chemistry validation.',
     data='Dataset2 ResVal 91 usable cells; six diagnostic rates, one input fragment per cell',
     model_class='Independently implemented algorithm families, NOT exact published paper architectures',
     folds=folds,shots=k_values,repeats=repeats,methods=methods,policies=policies,
     selection='LOOCV among target labels only; no source/test target labels for selection',
     data_access='Unlabeled target training pool available for acquisition and standardization',
     task='SOH output residual calibration; not physics-consistent latent correction',
     metric='outer test cell MAE, SOH percentage points; 3 repeated labeled selections do NOT mean 15 independent folds')
    (out/'protocol.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf8')
    print(summary.sort_values(['k','policy','mae_pp']).to_string(index=False),flush=True)
    return frame,summary

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--methods',default=','.join(METHODS))
    parser.add_argument('--policies',default='random,source_cluster,physical_cluster,input_cluster')
    parser.add_argument('--shots',default='4,14')
    parser.add_argument('--folds',default='0,1,2,3,4')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    run(args.cache,args.assets,args.out,args.methods.split(','),args.policies.split(','),
        [int(v) for v in args.shots.split(',')],args.repeats,[int(v) for v in args.folds.split(',')])