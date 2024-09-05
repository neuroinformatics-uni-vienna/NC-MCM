
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as cm

from ncmcm.bundlenet.utils import prep_data
from ncmcm.data_loaders.matlab_dataset import Database


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

fig, axs = plt.subplots(2, 1, figsize=(12, 4), gridspec_kw={'height_ratios': [3, 0.5]}, sharex=True)

# Define colors for each group
# colors = {
#     "Sensory": sns.color_palette("Set2")[0], # sns.color_palette("dark", 3)[0],  # Dark blue-gray
#     "Inter": sns.color_palette("Set2")[1], # sns.color_palette("dark", 3)[1],     # Dark grayish-brown
#     "Motor": sns.color_palette("Set2")[2] # sns.color_palette("dark", 3)[2]     # Dark slate blue
# }
colors = {
    "Sensory": "#111D4A",  # Dark blue-gray
    "Inter": "#563F1B",    # Dark grayish-brown
    "Motor": "#38726C"     # Dark teal
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


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

tab10_colors = plt.cm.tab10.colors
# Function to lighten a color
def lighten_color(color, amount=0.5):
    # Use matplotlib's Lightness and Saturation modification
    c = mcolors.to_rgba(color)
    return mcolors.to_rgba((c[0] + (1.0 - c[0]) * amount,
                            c[1] + (1.0 - c[1]) * amount,
                            c[2] + (1.0 - c[2]) * amount,
                            c[3]))
lighter_tab10 = [lighten_color(c, 0.5) for c in tab10_colors]

ax, im1 = discrete_plot(axs[1], B, b_names, y_label="", cmap=lighter_tab10)
plt.tight_layout()
plt.show()



plt.figure(figsize=(2.5, 6))  # Adjust figure size as needed
cbar_ax = plt.gca()
cbar = plt.colorbar(im1, cax=cbar_ax, orientation='vertical', ticks=np.arange(len(b_names)))
cbar.ax.set_yticklabels(list(b_names.values()), fontsize=10)  # Adjust font size if necessary

# Adjust the layout to ensure labels are not cut off
plt.subplots_adjust(left=0.2, right=0.4, top=0.9, bottom=0.1)

plt.show()



'''


# visualisation - single dimension
import seaborn as sns

for dim, group_name in enumerate(['sensory', 'inter', 'motor']):
    plt.figure()
    sns.histplot([Y0_[B_ == i][:, dim] for i in range(8)])
    ax = plt.gca()
    ax.set_xlabel(group_name)
    plt.show()

# visualisation - pair of dimensions
axis_labels = ['sensory', 'inter', 'motor']
for pair in [[0, 1], [1, 2], [0, 2]]:
    plt.figure()
    [plt.scatter(Y0_[B_ == i][:, pair[0]], Y0_[B_ == i][:, pair[1]], alpha=0.3) for i in range(8)]

    plt.xlabel(axis_labels[pair[0]])
    plt.ylabel(axis_labels[pair[1]])
    plt.show()


'''