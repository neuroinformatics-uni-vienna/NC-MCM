
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as cm

from ncmcm.bundlenet.utils import prep_data
from ncmcm.data_loaders.matlab_dataset import Database


def plotting_neuronal_behavioural(x, b=None, b_names={}, s=None, s_names={}, r=None, r_names={}, show_fig=True,
                                  **kwargs):
    num_plots = 1 + sum([1 if x is not None else 0 for x in [b, s, r]])
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, num_plots * 2))
    im0 = axs[0].imshow(x.T, aspect='auto', interpolation='None', **kwargs)
    # tell the colorbar to tick at integers
    cax0 = plt.colorbar(im0)
    axs[0].set_xlabel("time $t$")
    axs[0].set_ylabel("Neuronal activation")

    def discrete_plot(ax, b, b_names, y_label, cmap):
        colors = sns.color_palette(cmap, len(b_names))
        cmap = plt.get_cmap(cm.colors.ListedColormap(colors), np.max(b) - np.min(b) + 1)
        im1 = ax.imshow([b], cmap=cmap, vmin=np.min(b) - 0.5, vmax=np.max(b) + 0.5, aspect='auto')
        cax = plt.colorbar(im1, ticks=np.unique(b))
        if b_names:
            cax.ax.set_yticklabels(list(b_names.values()))
        ax.set_xlabel("time $t$")
        ax.set_ylabel(y_label)
        ax.set_yticks([])

    if b is not None:
        discrete_plot(axs[1], b, b_names, y_label="Behaviour", cmap='Pastel1')
    if s is not None:
        discrete_plot(axs[2], s, s_names, y_label="Stimulus", cmap='Set2')
    if r is not None:
        discrete_plot(axs[3], r, r_names, y_label="Response", cmap='Set3')

    if show_fig:
        plt.show()

    return fig, axs




# Load Data (excluding behavioural neurons) and plot
worm_num = 0
algorithm = 'BunDLeNet'
b_neurons = [
    'AVAR',
    'AVAL',
    'SMDVR',
    'SMDVL',
    'SMDDR',
    'SMDDL',
    'RIBR',
    'RIBL'
]
data_path = 'datasets/raw/c_elegans/NoStim_Data.mat'
data = Database(data_path=data_path, dataset_no=worm_num)
data.exclude_neurons(b_neurons)
mask = data.categorise_neurons('datasets/raw/c_elegans')
X = data.neuron_traces.T
B = data.behaviour
b_names = data.behaviour_names

Xs = X[:, mask == 1]
Xi = X[:, mask == 2]
Xm = X[:, mask == 3]

print(Xs.shape, Xi.shape, Xm.shape)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.cm as cm

# Assuming Xs, Xi, Xm, and B are defined
# Assuming b_names is defined

fig, axs = plt.subplots(2, 1, figsize=(12, 4), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

# Define colors for each group
colors = {
    "Sensory": sns.color_palette("dark", 3)[0],  # Dark blue-gray
    "Inter": sns.color_palette("dark", 3)[1],     # Dark grayish-brown
    "Motor": sns.color_palette("dark", 3)[2]     # Dark slate blue
}
# Initialize the starting position for the plot
i = 0

# Neuronal activation plot with group colors and labels
group_positions = []
group_labels = ["Sensory", "Inter", "Motor"]
populations = [Xs, Xi, Xm]
group_colors = [colors["Sensory"], colors["Inter"], colors["Motor"]]

for idx, (population, color) in enumerate(zip(populations, group_colors)):
    group_positions.append(i + population.shape[1] / 2)  # Store the position for labeling
    for x in population.T:
        i += 0.5
        axs[0].plot(x + i, c=color)

# Label the y-axis with the neuron groups
axs[0].set_yticks(group_positions)
axs[0].set_yticklabels(group_labels, rotation=90)
axs[0].set_ylabel("Neuronal activation")

# Discrete plot
def discrete_plot(ax, b, b_names, y_label, cmap):
    colors = sns.color_palette(cmap, len(b_names))
    cmap = cm.colors.ListedColormap(colors)
    im1 = ax.imshow([b], cmap=cmap, vmin=np.min(b) - 0.5, vmax=np.max(b) + 0.5, aspect='auto')

    # Remove borders, tick labels, and ticks
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel("Behaviour")
    return ax, im1

ax, im1 = discrete_plot(axs[1], B, b_names, y_label="", cmap='Pastel1')
plt.tight_layout()
plt.show()



plt.figure(figsize=(2.5, 6))  # Adjust figure size as needed
cbar_ax = plt.gca()
cbar = plt.colorbar(im1, cax=cbar_ax, orientation='vertical', ticks=np.arange(len(b_names)))
cbar.ax.set_yticklabels(list(b_names.values()), fontsize=10)  # Adjust font size if necessary

# Adjust the layout to ensure labels are not cut off
plt.subplots_adjust(left=0.2, right=0.4, top=0.9, bottom=0.1)

plt.show()

