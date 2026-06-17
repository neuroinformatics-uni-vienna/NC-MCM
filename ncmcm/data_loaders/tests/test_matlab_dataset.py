import numpy as np
from ncmcm.data_loaders.matlab_dataset import merge_discrete_variables


def test_merge_discrete_variables():
    B = np.array([0, 0, 1, 1])
    R = np.array([0, 1, 0, 1])
    behaviour_names = ['walk', 'run']
    response_names = ['low', 'high']
    
    br, br_names = merge_discrete_variables(B, R, behaviour_names, response_names)
    
    assert len(br) == 4
    assert len(np.unique(br)) == 4
    assert br_names[br[0]] == 'walk low'
    assert br_names[br[1]] == 'walk high'
    assert br_names[br[2]] == 'run low'
    assert br_names[br[3]] == 'run high'