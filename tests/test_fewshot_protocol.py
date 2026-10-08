"""Static safeguards for the exploratory Che2025 few-shot protocol."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'/'fewshot_physical_transfer'))
from run import split_data
from label_acquisition import choose_kmeans
from physical_calibration import regression


def test_split_no_target_test_leak():
    for fold in range(5):
        for k in (4,14):
            for rep in range(3):
                tr,te,lab,unlab=split_data(91,fold,k,rep)
                assert len(set(tr) & set(te))==0
                assert len(set(te) & set(lab))==0
                assert len(set(te) & set(unlab))==0
                assert set(lab).isdisjoint(set(unlab))
                assert set(lab).union(set(unlab))==set(tr)
                assert len(lab)==k


def test_selection_never_enters_test_and_deterministic():
    rng=np.random.default_rng(5)
    data=rng.normal(size=(91,6))
    tr,te,_,_=split_data(91,0,4,1)
    first=choose_kmeans(tr,data,4,1)
    second=choose_kmeans(tr,data,4,1)
    assert np.array_equal(first,second)
    assert len(first)==len(set(first))==4
    assert set(first).isdisjoint(set(te))


def test_ridge_invariant_to_test_labels_and_test_features():
    rng=np.random.default_rng(2026)
    pred=rng.normal(90,5,91)
    y=pred+rng.normal(size=91)
    features=rng.normal(size=(91,6))
    tr,te,lab,_=split_data(91,2,14,2)
    a=regression(features,tr,lab,te,pred,y,10)
    new_truth=y.copy();new_truth[te]+=1000
    assert np.allclose(a,regression(features,tr,lab,te,pred,new_truth,10))
    feat_changed=features.copy();feat_changed[te]*=100
    # Test features are inputs during inference (as expected): predictions can
    # change, but train-only means/std stay invariant.
    assert np.allclose(features[tr].mean(0),feat_changed[tr].mean(0))


def test_small_split_kmeans_support():
    rng=np.random.default_rng(6)
    data=rng.normal(size=(91,3))
    for fold in range(5):
        tr,te,_,_=split_data(91,fold,14,0)
        picks=choose_kmeans(tr,data,14,0)
        assert set(picks) <= set(tr)
        assert not set(picks)&set(te)