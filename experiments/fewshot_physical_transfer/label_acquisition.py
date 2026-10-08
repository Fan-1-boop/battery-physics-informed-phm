"""Budgeted label selection using target training pool physical latent geometry.

All candidates are drawn from outer TRAINING fold. No test measurements,
labels, or validation scores are used for acquisition.
Exploratory comparisons: random labels vs unlabeled k-means medoids.
"""
from pathlib import Path
import argparse
import numpy as np,pandas as pd,torch
from sklearn.cluster import KMeans
from run import load_model,split_data
from physical_calibration import regression


def choose_kmeans(training, features, k, seed):
    feats=features[training]
    mu=feats.mean(axis=0);sd=np.maximum(feats.std(axis=0),1e-4)
    vals=(feats-mu)/sd
    km=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(vals)
    inds=[]
    # select nearest observed member of each cluster
    for c in range(k):
        eligible=np.flatnonzero(km.labels_==c)
        dist=np.sum((vals[eligible]-km.cluster_centers_[c])**2,axis=1)
        inds.append(int(training[eligible[int(np.argmin(dist))]]))
    assert len(set(inds)) == k
    return np.sort(inds)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    dat=np.load(args.cache,allow_pickle=False)
    X=torch.tensor(dat['X'],dtype=torch.float32)
    y=dat['Y5'][:,-1,1]*100
    source=load_model(args.assets)
    with torch.no_grad():pred,dv,lat,sp,sn=source(X)
    s=pred[:,-1,1].numpy()*100
    phys=np.concatenate([v.numpy() for v in lat],axis=1)
    disagree=np.mean(np.abs(pred[:,1:,0].numpy()-dv[:,1:].numpy()),axis=1)*4200
    f=np.column_stack([s,phys,disagree])
    arr=X.numpy()
    input_summary=np.column_stack([arr[:,0,0],arr[:,-1,0],arr[:,-1,1],arr[:,:,0].mean(axis=1),
                                   arr[:,:,0].std(axis=1),arr[:,:,1].mean(axis=1)])
    acquisition_features={
        'kmeans_source_soh':s[:,None],
        'kmeans_physical_z':phys,
        'kmeans_soh_plus_z':np.column_stack([s,phys]),
        'kmeans_physical_all':f,
        'kmeans_input_summary':input_summary,
    }
    rows=[]
    for fold in range(5):
      for k in [4,14]:
        for rep in range(3):
          train,test,lab_random,_=split_data(len(X),fold,k,rep)
          sets={'random':lab_random}
          for policy,representation in acquisition_features.items():
             medoids=choose_kmeans(train,representation,k,seed=rep)
             assert len(set(medoids)&set(test))==0
             sets[policy]=medoids
          for policy,lab in sets.items():
            estimate=regression(f,train,lab,test,s,y,alpha=10)
            error=estimate-y[test]
            rows.append(dict(fold=fold,repeat=rep,k=k,policy=policy,
               soh_mae_pp=float(np.abs(error).mean()),soh_rmse_pp=float(np.mean(error**2)**.5),
               label_ids=';'.join(map(str,lab)),test_ids=';'.join(map(str,test))))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows);frame.to_csv(args.out,index=False)
    print(frame.groupby(['k','policy']).agg(mean=('soh_mae_pp','mean'),std=('soh_mae_pp','std')).to_string())

if __name__=='__main__':main()