"""Protocol invariants for cross-literature method-family benchmark."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'experiments/literature_bench_v6/benchmark.py'
spec=importlib.util.spec_from_file_location('benchmark_v6',TARGET)
bm=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=bm
spec.loader.exec_module(bm)


def toy():
    rng=np.random.default_rng(72)
    n=38
    s=rng.uniform(80,97,n)
    z=rng.normal(size=(n,5))
    features={'source':s,'physical':np.column_stack([s,z]),'input_summary':rng.normal(size=(n,6))}
    y=s+3+.3*z[:,0]
    tr=np.arange(27);chosen=tr[[2,9,16,21]];test=np.arange(27,38)
    return features,y,tr,chosen,test


def test_test_labels_do_not_enter_training_or_selection():
    features,y,tr,lab,te=toy()
    for method in bm.METHODS:
        yp,params,_=bm.get_results(features,y,tr,lab,te,method)
        changed=y.copy();changed[te]+=10000
        yp2,params2,_=bm.get_results(features,changed,tr,lab,te,method)
        assert np.allclose(yp,yp2),method
        assert params==params2
        assert np.isfinite(yp).all()


def test_unlabeled_test_features_not_used_for_fit_or_scale():
    features,y,tr,lab,te=toy()
    for name in ('source','physical','input_summary'):
        x=features[name]
        x=x[:,None] if x.ndim==1 else x
        p=bm.normalize_features(x,tr)
        x2=x.copy();x2[te]+=30000
        p2=bm.normalize_features(x2,tr)
        assert np.allclose(p[tr],p2[tr])


def test_rate_agnostic_acquisition_and_fold_disjoint():
    from sklearn.model_selection import KFold
    for fold in range(5):
        tr,te,chosen,_=bm.split_data(91,fold,4,0)
        assert len(set(tr)&set(te))==0
        assert set(chosen).issubset(set(tr))
        assert len(chosen)==4


def test_hyperparams_only_read_training_labels():
    x=np.arange(16,dtype=float).reshape(-1,2)
    y=np.arange(len(x),dtype=float)
    for method in ('ridge_physics','krr_rbf','svr_rbf','gp_rbf'):
        pars,_=bm.select_labeled_only(method,x[:4],y[:4])
        assert isinstance(pars,dict)
        assert np.isfinite(bm._fit(method,x[:4],y[:4],pars).predict(x[4:])).all()