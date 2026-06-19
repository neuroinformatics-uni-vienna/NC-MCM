"""
@authors:
Akshey Kumar
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import matplotlib as cm


def plotting_neuronal_behavioural(
    x, 
    b=None, 
    b_names={}, 
    s=None, 
    s_names={}, 
    r=None, 
    r_names={}, 
    show_fig=True, 
    **kwargs
):
    """
    Visualize simultaneously recorded neuronal activations and behavioral data.

    This function plots neuronal traces and optionally includes behavioral, 
    stimulus, and response variables if provided.

    Parameters:
    - x: 2D numpy array of neuronal activation data, 
        with shape (neurons, time).
    - b: 1D numpy array of behavioral data (optional).
    - b_names: Dictionary mapping behavior labels to their names (optional).
    - s: 1D numpy array of stimulus data (optional).
    - s_names: Dictionary mapping stimulus labels to their names (optional).
    - r: 1D numpy array of response data (optional).
    - r_names: Dictionary mapping response labels to their names (optional).
    - show_fig: Boolean indicating whether to display the plot
    - kwargs: Additional keyword arguments for customizing the neuronal 
        activation plot.

    Returns:
    - fig: The matplotlib figure object.
    - axs: A list of the matplotlib axes objects.

    Example usage:
    ```
    # Basic usage with neuronal data and behavior
    plotting_neuronal_behavioural(x, b=b, b_names={0: 'Rest', 1: 'Move'})

    # Including stimulus and response data
    plotting_neuronal_behavioural(
        x,
        b=b, b_names={0: 'Rest', 1: 'Move'},
        s=s, s_names={0: 'No Stimulus', 1: 'Stimulus'},
        r=r, r_names={0: 'No Response', 1: 'Response'},
        vmin=0, vmax=1)
    ```
    """
    num_plots = 1 + sum([1 if x is not None else 0 for x in [b, s, r]])
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, num_plots * 2), squeeze=False)
    axs = axs.flatten()
    im0 = axs[0].imshow(x.T, aspect='auto', interpolation='None', **kwargs)
    # tell the colorbar to tick at integers
    axs[0].set_xlabel("time $t$")
    axs[0].set_ylabel("Neuronal activation")
    cax0 = plt.colorbar(im0)

    if isinstance(b_names, (list, np.ndarray)):
        b_names = {i: str(name) for i, name in enumerate(b_names)}

    if isinstance(s_names, (list, np.ndarray)):
        s_names = {i: str(name) for i, name in enumerate(s_names)}

    if isinstance(r_names, (list, np.ndarray)):
        r_names = {i: str(name) for i, name in enumerate(r_names)}


    def discrete_plot(ax, b, b_names, y_label, cmap, alpha=1.0):
        colors = sns.color_palette(cmap, len(b_names))
        cmap = ListedColormap(colors)
        im1 = ax.imshow(
            [b], 
            cmap=cmap, 
            vmin=np.min(b) - 0.5, 
            vmax=np.max(b) + 0.5, 
            aspect='auto',
            alpha=alpha
        )
        cbar = plt.colorbar(im1, ticks=np.unique(b))
        if b_names:
            cbar.ax.invert_yaxis() 
            cbar.ax.set_yticklabels(list(b_names.values()))
        ax.set_xlabel("time $t$")
        ax.set_ylabel(y_label)
        ax.set_yticks([])

    if b is not None:
        discrete_plot(axs[1], b, b_names, y_label="Behaviour", cmap=sns.color_palette("deep", as_cmap=True), alpha=0.6)
    if s is not None:
        discrete_plot(axs[2], s, s_names, y_label="Stimulus", cmap='Set2')
    if r is not None:
        discrete_plot(axs[3], r, r_names, y_label="Response", cmap='Set3')

    if show_fig:
        plt.show()

    return fig, axs


