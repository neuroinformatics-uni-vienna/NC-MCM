"""
@authors:
Akshey Kumar
Michael Hofer
Jinook Oh
Kerim Atak
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.animation as anim  # FuncAnimation
from tqdm import tqdm

class LatentSpaceVisualiser:
    def __init__(self, y, b, b_names, show_points=False, legend=True, colors=None):
        """
        Initialize the LatentSpaceVisualiser.

        Parameters:
        -----------
        y : numpy.ndarray
            The latent space coordinates of the neuronal data.
        b : numpy.ndarray
            The behavioral labels corresponding to the data points.
        b_names : dict
            A dictionary mapping behavior labels to their names.
        show_points : bool, optional
            Whether to show individual data points in the plot. Default is False.
        legend : bool, optional
            Whether to display a legend in the plot. Default is True.
        colors: numpy.ndarray, optional
                If given, defines colors of behaviors.
        """
        self.y = y
        self.b = b
        self.b_names = b_names
        self.show_points = show_points
        self.legend = legend
        self.colors = colors

        if isinstance(self.b_names, (list, np.ndarray)):
            self.b_names = {i: str(name) for i, name in enumerate(self.b_names)}
        elif not isinstance(b_names, dict):
            raise ValueError("`b_names` must be either a dictionary or a list or numpy array.")

        if self.colors is None:
            self.colors = sns.color_palette('deep', len(self.b_names))

        self.color_dict = {
            name: color
            for name, color in zip(np.unique(self.b), self.colors)
        }
        self.cmap = ListedColormap(self.colors)


    def plot_latent_timeseries(
        self, 
        show_fig=True, 
        filename='figures/latent_time_series.png'
    ):
        """
        Plot time series of dynamics in latent space.

        This function generates a plot showing the time series of neuronal 
        dynamics, in latent space, with discrete behavior states represented 
        by colors and latent variables plotted over time.

        Parameters:
        -----------
        show_fig : bool, optional
            If True, the plot will be displayed interactively. 
            Default is True.
        
        filename : str, optional
            The path and filename where the plot will be saved. 
            Default is 'figures/latent_time_series.png'.

        Returns:
        --------
        None
        """
        plt.figure(figsize=(19, 5))
        im = plt.imshow(
            [self.b], 
            aspect=600, 
            cmap=self.cmap,
            vmin=np.min(self.b) - 0.5, 
            vmax=np.max(self.b) + 0.5,
            alpha=0.6
        )
        cbar = plt.colorbar(
            im, 
            ticks=np.arange(np.min(self.b), np.max(self.b) + 1)
        )
        
        cbar.ax.invert_yaxis() 
        cbar.ax.set_yticklabels(list(self.b_names.values()), fontsize=12)
        bbox = im.axes.get_position()  # bounding box of `imshow`
        cbar.ax.set_position([bbox.x1 + 0.03, bbox.y0, 0.02, bbox.height])

        plt.plot(self.y / np.max(np.abs(self.y)) / 3, linewidth=2)
        plt.xlabel("time $t$", fontsize=14)
        plt.axis([0, self.y.shape[0], -0.5, 0.5])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        plt.savefig(filename)

        if show_fig:
            plt.show()

    def plot_latent_timeseries_plotly(
        self, 
        show_fig=True, 
        filename='figures/latent_time_series.html'
    ):
        """
        Plot time series of dynamics in latent space using Plotly.

        This function generates an interactive plot showing the time series 
        of neuronal dynamics in latent space, with discrete behavior states 
        represented by a colored heatmap background and latent variables 
        plotted over time.

        Parameters:
        -----------
        show_fig : bool, optional
            If True, the plot will be displayed interactively. 
            Default is True.
        
        filename : str, optional
            The path and filename where the plot will be saved as HTML. 
            Default is 'figures/latent_time_series.html'.

        Returns:
        --------
        fig : plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go
        
        # Create figure
        fig = go.Figure()
        
        # Create custom colorscale for behavioral states with discrete colors
        unique_vals = np.unique(self.b)
        
        # Create discrete colorscale by defining sharp boundaries
        colorscale_list = []
        n_colors = len(unique_vals)
        for i, val in enumerate(unique_vals):
            # Get color from color_dict using the actual state ID
            color_tuple = self.color_dict[int(val)]
            color = f'rgb({int(color_tuple[0]*255)},{int(color_tuple[1]*255)},{int(color_tuple[2]*255)})'
            
            if n_colors == 1:
                colorscale_list = [[0, color], [1, color]]
            else:
                # Define lower and upper bounds for this discrete color
                lower_bound = i / n_colors
                upper_bound = (i + 1) / n_colors
                
                # Add both bounds with the same color for sharp transition
                if i == 0:
                    colorscale_list.append([0, color])
                else:
                    colorscale_list.append([lower_bound, color])
                
                if i == n_colors - 1:
                    colorscale_list.append([1, color])
                else:
                    colorscale_list.append([upper_bound, color])
        
        # Add behavioral state heatmap as background
        fig.add_trace(go.Heatmap(
            z=[self.b],
            x=np.arange(len(self.b)),
            y=[0],
            colorscale=colorscale_list,
            zmin=np.min(self.b) - 0.5,
            zmax=np.max(self.b) + 0.5,
            colorbar=dict(
                tickvals=unique_vals,
                ticktext=[self.b_names[int(v)] for v in unique_vals],
                len=0.5,
                y=0.5
            ),
            opacity=0.6,
            showscale=True,
            hoverinfo='skip'
        ))
        
        # Normalize latent variables
        y_normalized = self.y / np.max(np.abs(self.y)) / 3
        
        # Add latent variable traces
        for i in range(self.y.shape[1]):
            fig.add_trace(go.Scatter(
                x=np.arange(len(y_normalized)),
                y=y_normalized[:, i],
                mode='lines',
                name=f'Latent {i+1}',
                line=dict(width=2),
                yaxis='y2'
            ))
        
        # Update layout with dual y-axes
        fig.update_layout(
            width=1400,
            height=400,
            xaxis=dict(
                title='time <i>t</i>',
                range=[0, self.y.shape[0]]
            ),
            yaxis=dict(
                visible=False,
                range=[-0.5, 0.5]
            ),
            yaxis2=dict(
                overlaying='y',
                side='left',
                range=[-0.5, 0.5],
                showgrid=True
            ),
            showlegend=True,
            legend=dict(
                x=1.05,
                y=1,
                xanchor='left',
                yanchor='top'
            )
        )
        
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        fig.write_html(filename)
        
        if show_fig:
            fig.show()
        
        return fig

    def plot_phase_space(
        self, 
        show_fig=True, 
        filename='figures/phase_space_dynamics.png', 
        axis_view=None, 
        **kwargs
    ):
        """
        Plot the neuronal dynamics in a 3D phase space.

        This function creates a 3D phase space plot of the neuronal activity, 
        with arrows representing the transitions between states over time.

        Parameters:
        -----------
        show_fig : bool, optional
            If True, the plot will be displayed interactively. 
            Default is True.
        
        filename : str, optional
            The path and filename where the plot will be saved. Default is 
            'figures/phase_space_dynamics.png'.
        
        axis_view : (float, float), optional
            A tuple specifying the elevation and azimuthal angles for 
            the view of the 3D plot. If None, the default view is used. 
            Default is None.
        
        **kwargs : additional keyword arguments to customise plot
            Additional keyword arguments are passed to the ax.quiver() 
            function. (e.g., color, alpha). 

        Returns:
        --------
        fig : matplotlib.figure.Figure
        
        ax : matplotlib.axes._subplots.Axes3DSubplot

        Notes:
        ------
        This method uses the internal `_plot_ps` method to handle the core 
        plotting logic.
        """
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection='3d')
        if axis_view is not None:
            ax.view_init(elev=axis_view[0], azim=axis_view[1])
        self._plot_ps(fig, ax, **kwargs)

        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        plt.savefig(filename, dpi=300)

        if show_fig:
            plt.show()

        return fig, ax

    def plot_phase_space_continuous(
        self,
        c,
        cmap='viridis',
        label='',
        show_fig=False,
        filename='figures/phase_space_continuous.png',
        axis_view=None,
    ):
        """
        Plot the latent space trajectory coloured by a continuous variable.

        Uses scatter with a continuous colormap and colorbar instead of the
        discrete quiver arrows used in plot_phase_space.

        Parameters
        ----------
        c : array-like, shape (T,)
            Continuous values to map to colour (e.g. HGF belief trajectory).
        cmap : str, optional
            Matplotlib colormap name. Default is 'viridis'.
        label : str, optional
            Label for the colorbar axis. Default is ''.
        show_fig : bool, optional
            If True, display the figure interactively. Default is False.
        filename : str, optional
            Path to save the figure.
        axis_view : (float, float), optional
            (elevation, azimuth) angles for the 3D view.

        Returns
        -------
        fig, ax
        """
        c = np.asarray(c)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        if axis_view is not None:
            ax.view_init(elev=axis_view[0], azim=axis_view[1])

        sc = ax.scatter(
            self.y[:, 0],
            self.y[:, 1],
            self.y[:, 2],
            c=c,
            cmap=cmap,
            s=1,
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label(label)
        ax.set_axis_off()

        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        plt.savefig(filename, dpi=300, bbox_inches='tight')

        if show_fig:
            plt.show()

        return fig, ax

    def _plot_ps(self, fig, ax, **kwargs):
        """
        Helper to plot neuronal dynamics in a 3D phase space.
        """

        if self.y.shape[0] != self.b.shape[0]:
            raise ValueError("Y and b must have the same number of time steps")

        for i in range(len(self.y) - 1):
            d = (self.y[i + 1] - self.y[i])
            kwargs.setdefault('arrow_length_ratio', 0.01 / np.linalg.norm(d))
            kwargs.setdefault('linewidths', 1)
            ax.quiver(
                self.y[i, 0], 
                self.y[i, 1], 
                self.y[i, 2],
                d[0], 
                d[1], 
                d[2],
                color=self.color_dict[self.b[i]],
                **kwargs
            )
        ax.set_axis_off()

        if self.legend:
            legend_elements = [
                Line2D(
                    [0], 
                    [0], 
                    color=self.color_dict[b],
                    lw=4, 
                    label=self.b_names[b]
                ) for b in self.color_dict]
            ax.legend(handles=legend_elements)

        if self.show_points:
            ax.scatter(
                self.y[:, 0], 
                self.y[:, 1], 
                self.y[:, 2],
                s=1,
                c='k'
            )
        return fig, ax

    def rotating_plot(
        self, 
        show_fig=True, 
        filename='figures/rotation.gif', 
        **kwargs
    ):
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
        rot_animation = anim.FuncAnimation(
            fig, 
            rotate, 
            frames=np.arange(0, 362, 5), 
            interval=150
        )

        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        rot_animation.save(filename, dpi=150, writer='imagemagick')

        if show_fig:
            plt.show()

        return fig, ax

    def plot_interactive_3d(
        self, 
        show_fig=True, 
        filename='figures/interactive_phase_space.html'
    ):
        """
        Plot the neuronal dynamics in an interactive 3D phase space using Plotly.
        
        This function creates an interactive 3D phase space plot of the neuronal 
        activity, with line segments representing the transitions between states 
        over time. The plot can be rotated and zoomed interactively.
        
        Parameters:
        -----------
        show_fig : bool, optional
            If True, the plot will be displayed interactively. 
            Default is True.
        
        filename : str, optional
            The path and filename where the plot will be saved as HTML. 
            Default is 'figures/interactive_phase_space.html'.
        
        Returns:
        --------
        fig : plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go
        
        if self.y.shape[0] != self.b.shape[0]:
            raise ValueError("Y and b must have the same number of time steps")
        
        fig = go.Figure()
        
        # Create line segments for each transition
        for i in range(len(self.y) - 1):
            color = self.color_dict[self.b[i]]
            fig.add_trace(go.Scatter3d(
                x=[self.y[i, 0], self.y[i + 1, 0]],
                y=[self.y[i, 1], self.y[i + 1, 1]],
                z=[self.y[i, 2], self.y[i + 1, 2]],
                mode='lines',
                line=dict(
                    color=f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})',
                    width=2
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add legend traces
        if self.legend:
            for b in self.color_dict:
                color = self.color_dict[b]
                fig.add_trace(go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode='lines',
                    line=dict(
                        color=f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})',
                        width=4
                    ),
                    name=self.b_names[b]
                ))
        
        # Add individual points if requested (after all lines so they're on top)
        if self.show_points:
            fig.add_trace(go.Scatter3d(
                x=self.y[:, 0],
                y=self.y[:, 1],
                z=self.y[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color='black'
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Update layout to match matplotlib style
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            showlegend=self.legend,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        fig.write_html(filename)
        
        if show_fig:
            fig.show()
        
        return fig

    def make_movie(
            self,
            fps=None,
            filename='figures/movie.gif',
            show_fig=False,
            initial_alpha=None,
            fade_time=0,
            bitrate=1800,
            dpi=144,
            **kwargs
    ):
        """
        Creates a GIF from data and saves it in "filename".

        Parameters:
            -----------
            dpi: int, optional
                Gives resolution of GIF.

            bitrate: int, optional
                Gives bitrate of the GIF.

            fade_time: int, optional
                If given and greater than 0 it will define how many
                last frames are seen simultaneously.

            initial_alpha: float, optional
                If given and greater than 0, all frames in initial figure will
                be drawn at the start with this initial_alpha.
                'initial figure' is necessary to properly draw the animation
                on the space with all the given data.
                If initial_alpha is greater than 0, the entire data will be
                visible and the animation will be drawn on top of it.

            fps: int, optional
                Gives frames per second of the GIF.

            filename: str, optional
                Name and path of the GIF-file that will be saved.

            show_fig: bool, optional
                If True, the GIF will play after saving.

        Returns:
        --------
        """
        if self.y.shape[1] != 3:
            print('The mapping does not map into a 3D space.')
            return False

        if fps is not None:
            interval = 1000 / fps
        else:
            print('The movie will be played with 100 fps.')
            interval = 10

        movie_animation = self._create_animation(
            initial_alpha=initial_alpha,
            fade_time=fade_time,
            interval=interval,
            **kwargs
        )

        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        with tqdm(total=self.y.shape[0], desc="Creating movie") as pbar:
            movie_animation.save(
                filename,
                dpi=dpi,
                writer=anim.PillowWriter(
                    fps=int(1000 / interval),
                    metadata=dict(artist='Me'),
                    bitrate=bitrate
                ),
                progress_callback=lambda i, n: pbar.update(1)
            )

        if show_fig:
            plt.show()

        return True

    def _create_animation(
        self,
        initial_alpha=None,
        fade_time=0,
        interval=10,
        **kwargs
    ):
        """
        Creates the animation.

        Parameters:
            -----------
            colors: numpy.ndarray, optional
                Gives colors for quivers in GIF.

            fade_time: int, optional
                If given and greater than 0 it will define how many last 
                frames are seen simultaneously.

            initial_alpha: float, optional
                If given and greater than 0, all frames will be drawn at 
                the start with this initial_alpha.

            colors: numpy.ndarray, optional
                If given, defines colors of quivers.

            interval: int, optional
                Gives interval between frames in milliseconds.

        Returns:
        --------
            animation: matplotlib.animation.FuncAnimation
                The animation generated.

        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.scatter = None
        kwargs.setdefault('alpha', 0.4)
        kwargs_initial = kwargs.copy()
        if initial_alpha is None:
            kwargs_initial['alpha'] = 0
        else:
            kwargs_initial['alpha'] = initial_alpha

        ax = self._plot_ps_comp(ax, **kwargs_initial)
        
        self.quiver_artists = []

        if self.legend:
            legend_elements = [
                Line2D(
                    [0], 
                    [0], 
                    color=self.color_dict[b],
                    lw=4, 
                    label=self.b_names[b]
                ) for b in self.color_dict
            ]
            ax.legend(handles=legend_elements)

        animation = anim.FuncAnimation(
            fig, 
            self._update,
            fargs=(ax, legend_elements, fade_time, kwargs),
            frames=range(self.y.shape[0]),
            interval=interval
        )

        return animation

    def _plot_ps_comp(self, ax, **kwargs):
        """
        Helper to comparison plot in a 3D phase space.
        """
        for i in range(len(self.y) - 1):
            d = (self.y[i + 1] - self.y[i])
            kwargs.setdefault('arrow_length_ratio', 0.01 / np.linalg.norm(d))
            kwargs.setdefault('linewidths', 1)
            ax.quiver(self.y[i, 0], self.y[i, 1], self.y[i, 2],
                      d[0], d[1], d[2],
                      color=self.color_dict[self.b[i]], **kwargs)
        ax.set_axis_off()
        return ax

    def _update(self, 
        frame, 
        ax, 
        legend_elements,
        fade_time, 
        kwargs
    ):
        """
        Update function to create a frame in the movie.
        """
        if frame == 0:
            return ax

        if fade_time and len(self.quiver_artists) > fade_time:
            to_remove = self.quiver_artists.pop(0)
            to_remove.remove()

        d = (self.y[frame] - self.y[frame - 1])
        kwargs.setdefault('arrow_length_ratio', 0.01 / np.linalg.norm(d))
        kwargs.setdefault('linewidths', 1)
        quiver = ax.quiver(
            self.y[frame - 1, 0],
            self.y[frame - 1, 1],
            self.y[frame - 1, 2],
            d[0],
            d[1],
            d[2],
            color=self.color_dict[self.b[frame - 1]],
            **kwargs
        )
        self.quiver_artists.append(quiver)

        ax.set_axis_off()
        
        # Red Point at current frame
        if self.scatter is not None:
            self.scatter.remove()
        x, y, z = self.y[frame, :]
        self.scatter = ax.scatter(x, y, z, s=20, alpha=0.8, color='red')

        # Title and Legend of the frame
        title = f'Frame: {frame}\nBehavior: {self.b_names[self.b[frame - 1]]}'
        ax.set_title(title)

        if legend_elements:
            legend = ax.legend(
                handles=legend_elements,
                loc='lower center',
                bbox_to_anchor=[1, 0]
            )

        return ax