import numpy as np
from sklearn.linear_model import LogisticRegression
from ncmcm.cognitive_graphs.calculations import adj_matrix_ncmcm, fit_model


def test_adj_matrix_shape():
    C = np.array([0, 1, 0, 1])
    B = np.array([0, 1, 2, 0])
    
    T, C_B_states = adj_matrix_ncmcm(C, B)
    
    assert T.shape == (6, 6)
    assert len(C_B_states) == 6
    
    
def test_adj_matrix_known_transition():
    C = np.array([0, 1, 0])
    B = np.array([0, 0, 0])
    
    T, C_B_states = adj_matrix_ncmcm(C, B)
    
    state_from = np.where('1-0' == C_B_states)[0][0]
    state_to = np.where('2-0' == C_B_states)[0][0]
    
    assert T[state_from, state_to] > 0
    
    
def test_adj_matrix_normalized():
    C = np.array([0, 1, 0, 0, 1, 1, 0])
    B = np.array([0, 0, 1, 0, 1, 0, 1])

    T, _ = adj_matrix_ncmcm(C, B)

    assert np.all(T >= 0)
    assert np.all(T.sum(axis=0) <= 1.0 + 1e-10)
    

def test_fit_model_returns_tuple():
    rng = np.random.default_rng(0)
    neuron_traces = rng.random((10, 50))
    B = rng.integers(0, 3, size=50)
    
    result = fit_model(neuron_traces=neuron_traces, B=B, base_model=LogisticRegression(max_iter=200))
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    

def test_fit_model_yp_map_shape():
    rng = np.random.default_rng(0)
    neuron_traces = rng.random((10, 50))
    B = rng.integers(0, 3, size=50)
    
    yp_map, _ = fit_model(neuron_traces=neuron_traces, B=B, base_model=LogisticRegression(max_iter=200))
    
    assert yp_map.shape[0] == 50