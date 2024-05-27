"""
@authors:
Akshey Kumar
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as cm

def plotting_neuronal_behavioural(X, B=None, B_names={}, S=None, S_names={}, R=None, R_names={}, show=True, **kwargs):
    """
    Visualize simultaneously recorded neuronal activations and behavioral data.

    This function plots neuronal traces and optionally includes behavioral, 
    stimulus, and response variables if provided.

    Parameters:
    - X: 2D numpy array of neuronal activation data, with shape (neurons, time).
    - B: 1D numpy array of behavioral data (optional).
    - B_names: Dictionary mapping behavior labels to their names (optional).
    - S: 1D numpy array of stimulus data (optional).
    - S_names: Dictionary mapping stimulus labels to their names (optional).
    - R: 1D numpy array of response data (optional).
    - R_names: Dictionary mapping response labels to their names (optional).
    - show: Boolean indicating whether to display the plot immediately (default: True).
    - kwargs: Additional keyword arguments for customizing the neuronal activation plot.

    Returns:
    - fig: The matplotlib figure object.
    - axs: A list of the matplotlib axes objects.

    Example usage:
    ```
    # Basic usage with neuronal data and behavior
    plotting_neuronal_behavioural(X, B=B, B_names={0: 'Rest', 1: 'Move'})

    # Including stimulus and response data
    plotting_neuronal_behavioural(
        X, 
        B=B, B_names={0: 'Rest', 1: 'Move'}, 
        S=S, S_names={0: 'No Stimulus', 1: 'Stimulus'}, 
        R=R, R_names={0: 'No Response', 1: 'Response'},
        vmin=0, vmax=1)
    ```
    """
    num_plots = 1 + sum([1 if x is not None else 0 for x in [B, S, R]])
    fig, axs = plt.subplots(num_plots,1,figsize=(12,num_plots*2))
    im0 = axs[0].imshow(X.T, aspect='auto', interpolation='None',**kwargs)
    # tell the colorbar to tick at integers
    cax0 = plt.colorbar(im0)
    axs[0].set_xlabel("time $t$")
    axs[0].set_ylabel("Neuronal activation")
    
    def discrete_plot(ax, B, B_names, y_label, cmap):
        colors = sns.color_palette(cmap, len(B_names))
        cmap = plt.get_cmap(cm.colors.ListedColormap(colors), np.max(B) - np.min(B) + 1)
        im1 = ax.imshow([B], cmap=cmap, vmin=np.min(B) - 0.5, vmax=np.max(B) + 0.5, aspect='auto')
        cax = plt.colorbar(im1, ticks=np.unique(B))
        if B_names:
            cax.ax.set_yticklabels(list(B_names.values()))
        ax.set_xlabel("time $t$")
        ax.set_ylabel(y_label)
        ax.set_yticks([])
    
    if B is not None:
        discrete_plot(axs[1], B, B_names, y_label="Behaviour", cmap = 'Pastel1')
    if S is not None:
        discrete_plot(axs[2], S, S_names, y_label="Stimulus", cmap = 'Set2')
    if R is not None:
        discrete_plot(axs[3], R, R_names, y_label="Response", cmap = 'Set3')
    
    if show:
        plt.show()
    else: 
        pass
    
    return fig, axs