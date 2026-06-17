# Plotting of latent space dynamics

from io import BytesIO

from .visualiser_base import *
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

import joblib
from PIL import Image
from tqdm import tqdm


class LatentAnimationBackend:
    def __init__(self):
        self.dpi = None
        self.elevation = None
        self.azimuth = None
        pass

    def _init_view(self, elevation=30, azimuth=45):
        self.elevation = elevation
        self.azimuth = azimuth

    def _init_dpi(self, dpi):
        self.dpi = dpi

    def animate(self):
        raise NotImplementedError(f"Subclasses of {self.__class__.__name__} should implement the animate method.")

    def _render_frame(self, frame_idx):
        raise NotImplementedError(f"Subclasses of {self.__class__.__name__} should implement the _render_frame method.")

class PillowBackend(LatentAnimationBackend):
    def __init__(self):
        super().__init__()

class MatplotlibBackend(LatentAnimationBackend):
    def __init__(self):
        super().__init__()
        self.fig = None

        self.y = None
        self.b = None
        self.b_names = None
        self.b_color_dict = None
        self.show_points = None
        self.legend = None

        self.precomputed_segments = None
        self.precomputed_inverse_norms = None
        self.precomputed_colors = None

    def _plot_space(self, fig, ax):
        if self.y.shape[0] != self.b.shape[0]:
            raise ValueError("Length of y and b must be the same.")
        
        x = self.y[:-1, 0]
        y = self.y[:-1, 1]
        z = self.y[:-1, 2]

        x_fin = self.y[1:, 0] - self.y[:-1, 0]
        y_fin = self.y[1:, 1] - self.y[:-1, 1]
        z_fin = self.y[1:, 2] - self.y[:-1, 2]

        ax.quiver(
            x, y, z, x_fin, y_fin, z_fin,
            color=self.precomputed_colors,
            arrow_length_ratio=0.2,
            linewidth=1,
            alpha=0.8
        )

        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()


        if self.legend:
            elements = [
                plt.Line2D(
                    [0], 
                    [0], 
                    color=col, 
                    lw=4, 
                    label=self.b_names[key])
                for key, col in self.b_color_dict.items()
            ]

            ax.legend(handles=elements)

        if self.show_points:
            ax.scatter(
                self.y[:, 0], 
                self.y[:, 1], 
                self.y[:, 2], 
                edgecolor='k', 
                s=1)

        return fig, ax

    def _render_frame(self, angle):
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection='3d')
        self._plot_space(fig, ax)

        ax.view_init(elev=self.elevation, azim=angle)
        buffer = BytesIO()

        plt.savefig(buffer, format='png', dpi=self.dpi)
        plt.close(fig)
        buffer.seek(0)
        return buffer

    def _precompute(self):
        self.precomputed_segments = self.y[1:] - self.y[:-1]
        self.precomputed_inverse_norms = 0.01* np.array([1 / np.linalg.norm(segment) if np.linalg.norm(segment) != 0 else 0 for segment in tqdm(self.precomputed_segments, desc="Precomputing inverse norms", leave=False)])
        self.precomputed_colors = np.array([self.b_color_dict[int(self.b[i])] for i in tqdm(range(self.b.shape[0] - 1), desc="Precomputing colors", leave=False)])


    def render_frames(self, y, b, b_names, b_color_dict, show_points, legend, angles):
        assert self.elevation is not None and self.azimuth is not None, "View angles must be set before rendering frames."
        
        self.y = y
        self.b = b
        self.b_names = b_names
        self.b_color_dict = b_color_dict
        self.show_points = show_points
        self.legend = legend

        self._precompute()

        frames = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self._render_frame)(angle)
            for angle in tqdm(angles, desc="Rendering frames", leave=False)
        )
        return frames
        
class OptLatentSpaceVisualiser(LatentSpaceVisualiser, VisualiserBase):
    
    def __init__(self, 
            y: list | np.ndarray,
            b: np.ndarray, b_names: list | np.ndarray | dict, b_cmap='deep',
            show_points = False, legend = True, backend : LatentAnimationBackend = MatplotlibBackend(),
            show_fig = True, save_fig=False, save_path=None, dpi=300,
            **kwargs
    ):
        
        VisualiserBase.__init__(self,
            show_fig=show_fig,
            save_fig=save_fig,
            save_path=save_path,
            dpi=dpi,
            **kwargs
        )

        self.colors = sns.color_palette(b_cmap, n_colors=len(np.unique(b))) if b_cmap is not None else None

        LatentSpaceVisualiser.__init__(self, 
            y=y,
            b=b, 
            b_names=b_names, 
            show_points=show_points, 
            legend=legend, 
            colors=self.colors,
        )

        backend._init_dpi(dpi)

        self.y = y
        self.b = b
        self.b_names = b_names
        self.b_cmap = b_cmap
        self.show_points = show_points
        self.legend = legend
        self.backend = backend
        
        if isinstance(b_names, (list, np.ndarray)):
            self.b_names = {i: str(name) for i, name in enumerate(b_names)}
        elif not isinstance(b_names, dict):
            raise ValueError("b_names should be a list, numpy array, or dictionary.")
        
        self.b_color_dict = {
            int(id): color for id, color in zip(np.unique(self.b), self.colors)
        }

        self.cmap = ListedColormap(self.colors)
        

    def rotating_plot(self, view, filename='rotation.gif', fps=30, save_images=False, one_each=15, output_dir='rotation_frames'):
        if save_images:
            os.makedirs(output_dir, exist_ok=True)
        
        self.backend._init_view(elevation=view[0], azimuth=view[1])
        angles = np.linspace(0, 360, num=2*fps)

        frames = self.backend.render_frames(self.y, self.b, self.b_names, self.b_color_dict, self.show_points, self.legend, angles)

        images = [Image.open(frame) for frame in frames]
        if save_images:
            for i, frame in enumerate(frames):
                if i % one_each == 0:
                    with open(os.path.join(output_dir, f'frame_{i}.png'), 'wb') as f:
                        f.write(frame.getbuffer())
                        
        images[0].save(filename, save_all=True, append_images=images[1:], duration=5000/fps, loop=1)        
        print(f"Rendering complete. Animation saved to {filename}.")
