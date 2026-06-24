import numpy as np
from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None


def test_plotting_neuronal_behavioural_no_raise():
    x = np.random.rand(100, 10)
    fig, axs = plotting_neuronal_behavioural(x, show_fig=False)
    assert fig is not None
    assert len(axs) == 1
    
    
def test_plotting_neuronal_behavioural_with_all_optional():
    x = np.random.rand(100, 10)
    b = np.random.randint(0, 3, size=100)
    s = np.random.randint(0, 2, size=100)
    r = np.random.randint(0, 2, size=100)
    b_names = {0: 'rest', 1: 'move', 2: 'turn'}
    s_names = {0: 'off', 1: 'on'}
    r_names = {0: 'no', 1: 'yes'}

    fig, axs = plotting_neuronal_behavioural(
        x, b=b, b_names=b_names, s=s, s_names=s_names, r=r, r_names=r_names, show_fig=False
    )
    assert fig is not None
    assert len(axs) == 4