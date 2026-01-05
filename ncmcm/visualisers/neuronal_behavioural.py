"""
@authors:
Akshey Kumar
Kerim Atak
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plotting_neuronal_behavioural(
    x, 
    b=None, 
    b_names={}, 
    b_colors=None,
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
    - b_colors: Dictionary mapping behavior state IDs to hex color codes (optional).
                 e.g., {0: '#ffffff', 1: '#e74c3c'}. If None, uses default palette.
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
    
    # Using custom colors from dataset
    color_map = dataset.get_color_map_for_plotting()
    plotting_neuronal_behavioural(x, b=b, b_names=b_names, b_colors=color_map)
    ```
    """
    num_plots = 1 + sum([1 if x is not None else 0 for x in [b, s, r]])
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, num_plots * 2), squeeze=False)
    axs = axs.flatten()  # Convert to 1D array for consistent indexing
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


    def discrete_plot(ax, b, b_names, y_label, cmap, alpha=1.0, custom_colors=None):
        if custom_colors is not None:
            # Use custom colors: convert dict {state_id: hex_color} to ordered list
            unique_states = sorted(np.unique(b))
            colors = [custom_colors.get(int(state), '#000000') for state in unique_states]
            cmap = ListedColormap(colors)
        else:
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
        discrete_plot(axs[1], b, b_names, y_label="Behaviour", cmap=sns.color_palette("deep", as_cmap=True), alpha=0.6, custom_colors=b_colors)
    if s is not None:
        discrete_plot(axs[2], s, s_names, y_label="Stimulus", cmap='Set2')
    if r is not None:
        discrete_plot(axs[3], r, r_names, y_label="Response", cmap='Set3')

    if show_fig:
        plt.show()

    return fig, axs


def plotting_neuronal_behavioural_plotly(
    x, 
    b=None, 
    b_names={}, 
    b_colors=None,
    s=None, 
    s_names={}, 
    r=None, 
    r_names={}, 
    show_fig=True, 
    **kwargs
):
    """
    Visualize simultaneously recorded neuronal activations and behavioral data using Plotly.

    This function plots neuronal traces and optionally includes behavioral, 
    stimulus, and response variables if provided.

    Parameters:
    - x: 2D numpy array of neuronal activation data, 
        with shape (neurons, time).
    - b: 1D numpy array of behavioral data (optional).
    - b_names: Dictionary mapping behavior labels to their names (optional).
    - b_colors: Dictionary mapping behavior state IDs to hex color codes (optional).
                 e.g., {0: '#ffffff', 1: '#e74c3c'}. If None, uses default palette.
    - s: 1D numpy array of stimulus data (optional).
    - s_names: Dictionary mapping stimulus labels to their names (optional).
    - r: 1D numpy array of response data (optional).
    - r_names: Dictionary mapping response labels to their names (optional).
    - show_fig: Boolean indicating whether to display the plot
    - kwargs: Additional keyword arguments for customizing the neuronal 
        activation plot (vmin, vmax, colorscale, etc.).

    Returns:
    - fig: The plotly figure object.

    Example usage:
    ```
    # Basic usage with neuronal data and behavior
    plotting_neuronal_behavioural_plotly(x, b=b, b_names={0: 'Rest', 1: 'Move'})

    # Including stimulus and response data
    plotting_neuronal_behavioural_plotly(
        x,
        b=b, b_names={0: 'Rest', 1: 'Move'},
        s=s, s_names={0: 'No Stimulus', 1: 'Stimulus'},
        r=r, r_names={0: 'No Response', 1: 'Response'},
        vmin=0, vmax=1)
    
    # Using custom colors from dataset
    color_map = dataset.get_color_map_for_plotting()
    plotting_neuronal_behavioural_plotly(x, b=b, b_names=b_names, b_colors=color_map)
    ```
    """
    num_plots = 1 + sum([1 if var is not None else 0 for var in [b, s, r]])
    
    fig = make_subplots(
        rows=num_plots, 
        cols=1,
        subplot_titles=None,
        vertical_spacing=0.1,
        row_heights=[2] * num_plots,
        shared_xaxes=True
    )
    
    # Extract vmin, vmax, colorscale from kwargs
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    colorscale = kwargs.pop('colorscale', 'Viridis')
    
    # Plot neuronal activation
    heatmap = go.Heatmap(
        z=x.T,
        colorscale=colorscale,
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(len=1/num_plots, y=1 - 0.5/num_plots)
    )
    fig.add_trace(heatmap, row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="Neuronal activation", row=1, col=1)

    if isinstance(b_names, (list, np.ndarray)):
        b_names = {i: str(name) for i, name in enumerate(b_names)}

    if isinstance(s_names, (list, np.ndarray)):
        s_names = {i: str(name) for i, name in enumerate(s_names)}

    if isinstance(r_names, (list, np.ndarray)):
        r_names = {i: str(name) for i, name in enumerate(r_names)}

    def discrete_plot(row, data, names, y_label, palette, custom_colors=None):
        unique_vals = np.unique(data)
        
        if custom_colors is not None:
            # Use custom colors: convert dict {state_id: hex_color} to RGB list
            colors_rgb = [custom_colors.get(int(val), '#000000') for val in unique_vals]
        else:
            colors = sns.color_palette(palette, len(names))
            colors_rgb = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors]
        
        # Create discrete colorscale with sharp boundaries between colors
        colorscale_list = []
        n_colors = len(unique_vals)
        for i in range(n_colors):
            lower_bound = i / n_colors
            upper_bound = (i + 1) / n_colors
            colorscale_list.append([lower_bound, colors_rgb[i]])
            colorscale_list.append([upper_bound, colors_rgb[i]])
        
        heatmap = go.Heatmap(
            z=[data],
            colorscale=colorscale_list,
            zmin=np.min(data) - 0.5,
            zmax=np.max(data) + 0.5,
            colorbar=dict(
                len=1/num_plots,
                y=1 - (row - 0.5)/num_plots,
                tickvals=unique_vals,
                ticktext=[names.get(int(v), str(v)) for v in unique_vals] if names else [str(v) for v in unique_vals]
            ),
            opacity=0.6 if row == 2 and b is not None else 1.0
        )
        fig.add_trace(heatmap, row=row, col=1)
        fig.update_xaxes(title_text="time <i>t</i>", row=row, col=1)
        fig.update_yaxes(title_text=y_label, row=row, col=1, showticklabels=False)

    current_row = 2
    if b is not None:
        discrete_plot(current_row, b, b_names, "Behaviour", "deep", custom_colors=b_colors)
        current_row += 1
    if s is not None:
        discrete_plot(current_row, s, s_names, "Stimulus", "Set2")
        current_row += 1
    if r is not None:
        discrete_plot(current_row, r, r_names, "Response", "Set3")

    fig.update_layout(
        height=num_plots * 200,
        showlegend=False
    )

    if show_fig:
        fig.show()

    return fig


