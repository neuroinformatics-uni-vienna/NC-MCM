# Base class for visualizers

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

import os

class VisualizerBase:
    def __init__(self, **kwargs):
        print("Initializing VisualizerBase with kwargs:", kwargs)
        self.config = {}
        for key, value in kwargs.items():
            self.config[key] = value

    def _finalize_plot(self, fig, axs):
        # Adjust layout and show/save the figure
        plt.tight_layout()

        if self.config['save_fig'] and self.config['save_path'] is not None:
            print(f"Saving figure to {self.config['save_path']}")
            os.makedirs(os.path.dirname(self.config['save_path']), exist_ok=True)
            fig.savefig(self.config['save_path'], dpi=self.config.get('dpi', 300), bbox_inches='tight')

        if self.config['show_fig']:
            plt.show()

        plt.close(fig)