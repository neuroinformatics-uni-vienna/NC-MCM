# Neuronal Visualizer classes

from .visualiser_base import *

class NativeNeuronalVisualiser(VisualiserBase):
    def __init__(self, 
            x : np.ndarray, x_names : list = None, x_cmap='viridis',
            b : list | np.ndarray = None, b_names : dict =None, b_cmap='deep', 
            s : list | np.ndarray = None, s_names : dict =None, s_cmap='Set2',
            r : list | np.ndarray = None, r_names : dict =None, r_cmap='Set3',
            figsize_x=12, figsize_y_per_plot=2, image_title: str = None,
            show_fig=True, save_fig=False, save_path=None, dpi=300,
            **kwargs
    ):
        assert x is not None, "Neuronal activity must be provided (x cannot be None)." 
        assert isinstance(x, np.ndarray), "Neuronal activity (x) must be a numpy array."

        if b is not None:
            assert isinstance(b, (list, np.ndarray)), "Behavioural data (b) must be a list or numpy array."
            assert len(b) == x.shape[0], "Behavioural data (b) must have the same length as the number of time points in neuronal activity (x)."
        if s is not None:
            assert isinstance(s, (list, np.ndarray)), "Stimulus data (s) must be a list or numpy array."
            assert len(s) == x.shape[0], "Stimulus data (s) must have the same length as the number of time points in neuronal activity (x)."
        if r is not None:            
            assert isinstance(r, (list, np.ndarray)), "Response data (r) must be a list or numpy array."
            assert len(r) == x.shape[0], "Response data (r) must have the same length as the number of time points in neuronal activity (x)."

        super().__init__(show_fig=show_fig, save_fig=save_fig, save_path=save_path, dpi=dpi, **kwargs)

        # Neuronal activity data
        self.x = x
        self.x_names = x_names
        self.x_cmap = x_cmap

        # Behavioural data
        self.b = b
        self.b_names = b_names
        self.b_cmap = b_cmap

        # Stimulus data
        self.s = s
        self.s_names = s_names
        self.s_cmap = s_cmap

        # Response data
        self.r = r
        self.r_names = r_names
        self.r_cmap = r_cmap


        self.figsize_x = figsize_x
        self.figsize_y_per_plot = figsize_y_per_plot
        self.image_title = image_title


    def _add_discrete_plot(self, ax, data, name, y_label, cmap, alpha=1.0):
        cols = sns.color_palette(cmap, len(name))
        cmap = ListedColormap(cols)

        im = ax.imshow(
            [data], 
            cmap=cmap, 
            vmin=np.min(data) - 0.5, 
            vmax=np.max(data) + 0.5, 
            aspect='auto',
            alpha=alpha
        )
        cbar = plt.colorbar(im, ax=ax, ticks=np.unique(data))
        if name:
            cbar.ax.invert_yaxis() 
            cbar.ax.set_yticklabels(list(name.values()))
        ax.set_xlabel("time $t$")
        ax.set_ylabel(y_label)
        ax.set_yticks([])

    def plot(self):
        """
        Plots the neuronal activity and (optionally) the behavioural, stimulus, and responses time series.
        If information is provided, plots labeled with the provided names.
        
        This functions uses the matplotlib API 
        """

        num_plots = 1 + sum([1 if data is not None else 0 for data in [self.b, self.s, self.r]])
        
        fig, axs = None, None
        if self.x_names is None:
            fig, axs = plt.subplots(
                num_plots,
                1, 
                figsize=(self.figsize_x, num_plots * 2),
            )

        else:
            fig, axs = plt.subplots(
                num_plots,
                1, 
                figsize=(self.figsize_x, (num_plots * 4 + (num_plots - 1) * 2)),
                gridspec_kw={'height_ratios': [0.8] + [0.2 / (num_plots - 1)] * (num_plots - 1)}
            )

        fig.suptitle(self.image_title) if self.image_title is not None else None

        if self.image_title is not None:
            fig.suptitle(self.image_title)
        
        # We first show the neuronal activity plot
        im_neuronal = axs[0].imshow(self.x.T, aspect='auto', cmap=self.x_cmap)
        axs[0].set_xlabel("time $t$")
        axs[0].set_ylabel("Neuronal Activation")

        if self.x_names is not None:
            axs[0].set_yticks(np.arange(len(self.x_names)))
            axs[0].set_yticklabels(self.x_names, fontsize=6)

        plt.colorbar(im_neuronal, ax=axs[0])

        if self.b is not None:
            self._add_discrete_plot(axs[1], self.b, self.b_names, y_label="Behaviour", cmap=self.b_cmap, alpha=0.6)
        if self.s is not None:
            self._add_discrete_plot(axs[2], self.s, self.s_names, y_label="Stimulus", cmap=self.s_cmap)
        if self.r is not None:
            self._add_discrete_plot(axs[3], self.r, self.r_names, y_label="Response", cmap=self.r_cmap)

        self._finalize_plot(fig, axs)