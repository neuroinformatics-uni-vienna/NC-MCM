"""
@authors:
Akshey Kumar
Michael Hofer
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import matplotlib.animation as anim  # FuncAnimation


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
            kwargs.setdefault('arrow_length_ratio', 0.1 / np.linalg.norm(d))
            kwargs.setdefault('linewidths', 1)
            ax.quiver(self.y[i, 0], self.y[i, 1], self.y[i, 2],
                      d[0], d[1], d[2],
                      color=color_dict[self.b[i]], **kwargs)
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

    def comparison_model(self,
                         model,
                         original_data=None,
                         show_titles=True,
                         filename='comparison.png',
                         show_fig=True,
                         fig_size=(10, 8),
                         **kwargs):
        """
        Creates a comparison plot between the True labels and the predicted labels.
        The model added will be trained on either the embedding or on the original values,
        to predict the labels. The legend that is created, gives information on the amount
        of correct and incorrectly predicted labels.

        Parameters:
        -----------
        model: model, required
            Specifies the prediction model used to predict from either the embedding or
            from the original values that are specified using the 'y_original' parameter.

        y_original: numpy.ndarray, required
            If given, these values will be used to train the model attached and
            to predict the labels in the third plot

        show_titles: bool, optional
            If additional titles showing model used and accuracy should be shown.

        show_fig : bool, optional
            If True, the plot will be displayed interactively. Default is True.

        fig_size : tuple, optional
            Sets the size of the figure.

        filename : str, optional
            The path and filename where the plot will be saved.
            Default is 'figures/latent_time_series.png'.

        Returns:
        --------
            Success indicator
        """

        if self.y.shape[1] != 3:
            print('The mapping does not map into a 3D space.')
            return False

        if original_data is None:
            origin = 'Latent Embedding'
            model.fit(self.y, self.b)
            B_pred = model.predict(self.y)
        else:
            if self.b.shape[0] != original_data.shape[0]:
                print('The original values attached must be the same size as the labels (\'self.b\')')
                print(f'\t\'y_original\' is {original_data.shape[0]} long, while \'self.b\' is {self.b.shape[0]}')
                return False
            origin = 'Original Values'
            model.fit(original_data, self.b)
            B_pred = model.predict(original_data)

        fig, axis = plt.subplots(figsize=fig_size, ncols=3, subplot_kw={'projection': '3d'}, )
        diff_mask = self.b != B_pred
        diff_predicts = np.where(diff_mask, self.b, -1)
        self._generate_diff_label_counts(diff_predicts, B_pred)
        colors = sns.color_palette('deep', len(self.b_names))
        color_dict = {name: color for name, color in zip(np.unique(self.b), colors)}
        color_dict[-1] = 'grey'

        self._plot_ps_comp(axis[0], self.b, color_dict, **kwargs)
        self._plot_ps_comp(axis[1], diff_predicts, color_dict, **kwargs)
        self._plot_ps_comp(axis[2], B_pred, color_dict, **kwargs)

        if self.legend:
            legend_1 = self._generate_legend(color_dict, labels=self.b)
            axis[0].legend(title='True Labels',
                           handles=legend_1,
                           loc='upper center',
                           bbox_to_anchor=(0.5, 0.),
                           fontsize='small')
            legend_2 = self._generate_legend(color_dict, labels=None, diff=True)
            axis[1].legend(title='Incorrect Predictions',
                           handles=legend_2,
                           loc='upper center',
                           bbox_to_anchor=(0.5, 0.),
                           fontsize='small')
            legend_3 = self._generate_legend(color_dict, labels=B_pred)
            axis[2].legend(title='Predictions',
                           handles=legend_3,
                           loc='upper center',
                           bbox_to_anchor=(0.5, 0.),
                           fontsize='small')

        if show_titles:
            axis[1].set_title(f'\nModel: {type(model)}\n'
                              f'Trained/Predicting from {origin}\n\n'
                              f'Accuracy at {round(accuracy_score(self.b, B_pred), 3)}\n')
            fig.suptitle(f'{self.y.shape[0]} Frames',
                         fontsize='x-large',
                         fontweight='bold')

        plt.savefig(filename)
        if show_fig:
            plt.show()
        return True

    def _generate_diff_label_counts(self,
                                    diff_predict,
                                    B_pred):
        """
        Generates the counts of wrong predictions by the model from a numpy array
        were correct predictions are marked as "-1" while wrong ones are correctly labeled.

        Parameters:
            -----------
            diff_predict: numpy.ndarray, required
                Array with correct predictions (as "-1") and incorrect predictions (as "0", "1", ...)

            B_pred: numpy.ndarray, required
                Array with predictions from the model

        Returns:
        --------
        None
        """
        # Create dictionary to count different predictions for each label
        self.diff_label_counts = {l: {state: 0 for state in self.b_names} for l in np.unique(self.b)}
        for idx, wrong_predict in enumerate(diff_predict):
            pred_label = B_pred[idx]
            true_label = self.b[idx]
            if wrong_predict > -1:
                self.diff_label_counts[true_label][self.b_names[pred_label]] += 1

    def _generate_legend(self,
                         color_dict,
                         labels=None,
                         diff=False):
        """
        Generates legend handles from earlier created "self.diff_label_counts" or
        from labels given as a parameter.

        Parameters:
            -----------
            color_dict: dict, required
                Dictionary with colors for each label.

            labels: numpy.ndarray, optional
                Array with labels for legend.

            diff: bool, optional
                If True, the legend will also say how many labels are different from the true ones.

        Returns:
        --------
            legend_elements: list
                The legend elements used for the legend of each axis.
        """
        # if the legend for the difference plot is requested
        if diff:
            y_labels_diff = {
                label: {wrong: count for wrong, count in self.diff_label_counts[c_idx].items() if count}
                for c_idx, label in enumerate(self.b_names)
            }

            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[idx],
                                          markersize=10,
                                          label=r'$\mathbf{' + keyval[0] + '}$' + ' to ' +
                                                "; ".join(
                                                    [r"$\mathbf{" + w_behav + "}$" + f"({amount})"
                                                     for w_behav, amount in keyval[1].items()]
                                                )
                                          if keyval[1] else r'$\mathbf{' + keyval[
                                              0] + '}$' + " predictions were always correct")
                               for idx, keyval in enumerate(y_labels_diff.items())]

            return legend_elements

        # Create custom legend handles
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[idx],
                                      markersize=10,
                                      label=r'$\mathbf{' + lab + '}$' + f' ({list(labels).count(idx)})')
                           for idx, lab in enumerate(self.b_names)]
        return legend_elements

    def _plot_ps_comp(self, ax, b, color_dict, **kwargs):
        """
        Helper to comparison plot in a 3D phase space.
        """

        for i in range(len(self.y) - 1):
            d = (self.y[i + 1] - self.y[i])
            kwargs.setdefault('arrow_length_ratio', 0.1 / np.linalg.norm(d))
            kwargs.setdefault('linewidths', 1)
            ax.quiver(self.y[i, 0], self.y[i, 1], self.y[i, 2],
                      d[0], d[1], d[2],
                      color=color_dict[b[i]], **kwargs)
        ax.set_axis_off()
        return ax

    def make_movie(self,
                   fps=None,
                   filename='figures/movie.gif',
                   show_fig=False,
                   colors=None,
                   initial_alpha=None,
                   fade=0,
                   bitrate=1800,
                   dpi=144,
                   **kwargs):
        """
        Creates a GIF from data and saves it in "filename".

        Parameters:
            -----------
            dpi: int, optional
                Gives resolution of GIF.

            bitrate: int, optional
                Gives bitrate of the GIF.

            fade: int, optional
                If given and greater than 0 it will define how many frames are seen simultaneously.

            initial_alpha: float, optional
                If given and greater than 0, all frames will be drawn at the start with this initial_alpha.

            colors: numpy.ndarray, optional
                If given, defines colors of quivers.

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
            print('The movie well be played with 100 fps.')
            interval = 10
        movie_animation = self._create_animation(colors=colors,
                                                 initial_alpha=initial_alpha,
                                                 fade=fade,
                                                 interval=interval,
                                                 **kwargs)
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        movie_animation.save(filename,
                             dpi=dpi,
                             writer=anim.PillowWriter(fps=int(1000 / interval),
                                                      metadata=dict(artist='Me'),
                                                      bitrate=bitrate))
        if show_fig:
            plt.show()
        return True

    def _create_animation(self,
                          colors=None,
                          initial_alpha=None,
                          fade=0,
                          interval=10,
                          **kwargs):
        """
        Creates the animation.

        Parameters:
            -----------
            colors: numpy.ndarray, optional
                Gives colors for quivers in GIF.

            fade: int, optional
                If given and greater than 0 it will define how many frames are seen simultaneously.

            initial_alpha: float, optional
                If given and greater than 0, all frames will be drawn at the start with this initial_alpha.

            colors: numpy.ndarray, optional
                If given, defines colors of quivers.

            interval: int, optional
                Gives interval between frames in milliseconds.

        Returns:
        --------
            animation: matplotlib.animation.FuncAnimation
                The animation generated.

        """
        if colors is None:
            colors = sns.color_palette('deep', len(self.b_names))
        color_dict = {name: color for name, color in zip(np.unique(self.b), colors)}

        fig = plt.figure()
        movie_ax = fig.add_subplot(111, projection='3d')
        self.scatter = None
        kwargs.setdefault('alpha', 0.4)
        kwargs_initial = kwargs.copy()
        if initial_alpha is None:
            kwargs_initial['alpha'] = 0
        else:
            kwargs_initial['alpha'] = initial_alpha

        movie_ax = self._plot_ps_comp(movie_ax, self.b, color_dict, **kwargs_initial)
        self.quiver_artists = []

        if self.legend:
            legend_elements = self._generate_legend(color_dict, labels=self.b)
        else:
            legend_elements = []


        animation = anim.FuncAnimation(fig, self._update,
                                       fargs=(movie_ax, legend_elements, color_dict, fade, kwargs),
                                       frames=self.y.shape[0],
                                       interval=interval)

        return animation

    def _update(self,
                frame,
                movie_ax,
                legend_elements,
                color_dict,
                fade,
                kwargs):

        """
        Update function to create a frame in the movie.
        """
        if frame == 0:
            return movie_ax

        if fade and len(self.quiver_artists) > fade:
            to_remove = self.quiver_artists.pop(0)
            to_remove.remove()

        d = (self.y[frame] - self.y[frame - 1])
        kwargs.setdefault('arrow_length_ratio', 0.1 / np.linalg.norm(d))
        kwargs.setdefault('linewidths', 1)
        quiver = movie_ax.quiver(self.y[frame - 1, 0], self.y[frame - 1, 1], self.y[frame - 1, 2],
                                 d[0], d[1], d[2],
                                 color=color_dict[self.b[frame - 1]], **kwargs)
        self.quiver_artists.append(quiver)
        movie_ax.set_axis_off()

        # Red Point at current frame
        if self.scatter is not None:
            self.scatter.remove()
        x, y, z = self.y[frame, :]
        self.scatter = movie_ax.scatter(x, y, z, s=20, alpha=0.8, color='red')

        # Title adn Legend of the frame
        movie_ax.set_title(f'Frame: {frame}\nBehavior: {self.b_names[self.b[frame - 1]]}')
        if legend_elements:
            movie_ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=[1, 0])

        return movie_ax

