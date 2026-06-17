import numpy as np
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.show = lambda *args, **kwargs: None


def test_latent_space_visualiser_no_raise():
    vis = LatentSpaceVisualiser(
        np.random.rand(100, 3),
        np.random.randint(0, 4, 100),
        ['a', 'b', 'c', 'd'],
    )
    vis.plot_latent_timeseries()
    vis.plot_phase_space()