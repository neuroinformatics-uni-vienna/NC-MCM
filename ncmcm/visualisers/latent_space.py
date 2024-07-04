"""
@authors:
Akshey Kumar
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap


class LatentSpaceVisualiser:
    def __init__(self, y, b, b_names, show_points=False, legend=True):
        self.y = y
        self.b = b
        self.b_names = b_names
        self.show_points = show_points
        self.legend = legend

    def plot_latent_timeseries(self, show_fig=True, filename='figures/latent_time_series.png'):
        """
        Plot time series of dynamics in latent space.

        This function generates a plot showing the time series of neuronal dynamics, in 
        latent space, with discrete behavior states represented by colors and latent 
        variables plotted over time.

        Parameters:
        -----------
        show_fig : bool, optional
            If True, the plot will be displayed interactively. Default is True.
        
        filename : str, optional
            The path and filename where the plot will be saved. 
            Default is 'figures/latent_time_series.png'.

        Returns:
        --------
        None
        """
        plt.figure(figsize=(19, 5))
        cmap = plt.get_cmap('Pastel1', np.max(self.b) - np.min(self.b) + 1)
        im = plt.imshow([self.b], aspect=600, cmap=cmap, vmin=np.min(self.b) - 0.5, vmax=np.max(self.b) + 0.5)
        cbar = plt.colorbar(im, ticks=np.arange(np.min(self.b), np.max(self.b) + 1))
        cbar.ax.set_yticklabels(self.b_names)
        plt.plot(self.y / np.max(np.abs(self.y)) / 3)
        plt.xlabel("time $t$")
        plt.axis([0, self.y.shape[0], -0.5, 0.5])

        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        plt.savefig(filename)

        if show_fig:
            plt.show()

    def plot_phase_space(self, show_fig=True, filename='figures/phase_space_dynamics.png', axis_view=None, **kwargs):
        """
        Plot the neuronal dynamics in a 3D phase space.

        This function creates a 3D phase space plot of the neuronal activity, with 
        arrows representing the transitions between states over time.

        Parameters:
        -----------
        show_fig : bool, optional
            If True, the plot will be displayed interactively. 
            Default is True.
        
        filename : str, optional
            The path and filename where the plot will be saved. Default is 
            'figures/phase_space_dynamics.png'.
        
        axis_view : (float, float), optional
            A tuple specifying the elevation and azimuthal angles for the view of 
            the 3D plot. If None, the default view is used. Default is None.
        
        **kwargs : additional keyword arguments to customise plot
            Additional keyword arguments are passed to the ax.quiver() function.
            (e.g., color, alpha). 

        Returns:
        --------
        fig : matplotlib.figure.Figure
        
        ax : matplotlib.axes._subplots.Axes3DSubplot

        Notes:
        ------
        This method uses the internal `_plot_ps` method to handle the core plotting 
        logic.
        """
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection='3d')
        if axis_view is not None:
            ax.view_init(elev=axis_view[0], azim=axis_view[1])
        self._plot_ps(fig, ax, **kwargs)

        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        plt.savefig(filename)

        if show_fig:
            plt.show()

        return fig, ax

    def _plot_ps(self, fig, ax, colors=None, **kwargs):
        """
        Helper to plot neuronal dynamics in a 3D phase space.
        """

        if self.y.shape[0] != self.b.shape[0]:
            raise ValueError("Y and b must have the same number of time steps")

        if colors is None:
            colors = sns.color_palette('deep', len(self.b_names))
            color_dict = {name: color for name, color in zip(np.unique(self.b), colors)}

        for i in range(len(self.y) - 1):
            d = (self.y[i + 1] - self.y[i])
            ax.quiver(self.y[i, 0], self.y[i, 1], self.y[i, 2],
                      d[0], d[1], d[2],
                      color=color_dict[self.b[i]], arrow_length_ratio=0.1 / np.linalg.norm(d), linewidths=1, **kwargs)
        ax.set_axis_off()

        if self.legend:
            legend_elements = [Line2D([0], [0], color=color_dict[b], lw=4, label=self.b_names[b]) for b in color_dict]
            ax.legend(handles=legend_elements)

        if self.show_points:
            ax.scatter(self.y[:, 0], self.y[:, 1], self.y[:, 2], c='k', s=1, cmap=ListedColormap(colors))
        return fig, ax

    def rotating_plot(self, show_fig=True, filename='figures/rotation.gif', **kwargs):
        """
        Create a rotating 3D phase space plot of the neuronal dynamics.

        Parameters:
        -----------
        show_fig : bool, optional
            If True, the rotating plot will be displayed. 
            Default is True.

        filename : str, optional
            The path and filename where the rotating plot will be saved as a GIF.
            Default is 'figures/rotation.gif'.

        **kwargs : additional keyword arguments to customise plot
            Additional keyword arguments are passed to the ax.quiver() function.
            (e.g., color, alpha). 

        Returns:
        --------
        fig : matplotlib.figure.Figure

        ax : matplotlib.axes._subplots.Axes3DSubplot
        """
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection='3d')

        def rotate(angle):
            ax.view_init(azim=angle)

        self._plot_ps(fig, ax, **kwargs)
        rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 5), interval=150)

        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        rot_animation.save(filename, dpi=150, writer='imagemagick')

        if show_fig:
            plt.show()

        return fig, ax
