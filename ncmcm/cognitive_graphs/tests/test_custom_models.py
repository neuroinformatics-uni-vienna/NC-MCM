import numpy as np
from sklearn.linear_model import LogisticRegression
from ncmcm.cognitive_graphs.custom_models import CustomEnsembleModel


def test_fit_returns_self():
    rng = np.random.default_rng(0)
    X = rng.random((50, 5))
    y = rng.integers(0, 3, size=50)
    
    model = CustomEnsembleModel(LogisticRegression(max_iter=200))
    result = model.fit(X, y)
    
    assert result is model
    
    
def test_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.random((50, 5))
    y = rng.integers(0, 3, size=50)
    
    model = CustomEnsembleModel(LogisticRegression(max_iter=200))
    model.fit(X, y)
    preds = model.predict(X)
    
    assert len(preds) == 50
    

def test_predict_proba_shape():
    rng = np.random.default_rng(0)
    X = rng.random((50, 5))
    y = rng.integers(0, 3, size=50)
    
    model = CustomEnsembleModel(LogisticRegression(max_iter=200))
    model.fit(X, y)
    proba = model.predict_proba(X)
    
    assert proba.shape[0] == 50
    
    
def test_get_params():
    base = LogisticRegression()
    model = CustomEnsembleModel(base)
    
    params = model.get_params()
    
    assert 'base_model' in params
    assert params['base_model'] is base
    
    
def test_combinatorics_count():
    rng = np.random.default_rng(0)
    X = rng.random((50, 5))
    y = rng.integers(0, 3, size=50)
    
    model = CustomEnsembleModel(LogisticRegression(max_iter=200))
    model.fit(X, y)
    
    assert len(model.ensemble_models) == 3