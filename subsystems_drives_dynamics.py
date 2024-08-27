import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.subsystem_fit.bundlenet_subsystem import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
import tensorflow as tf

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
X_, B_ = prep_data(X, B, win=15)
Y0_ = np.load(f"temp/selected_models/Y0_unreg_min_train_loss_worm_0.npy")


direction_vectors = []
for b in np.unique(B):
    direction_vectors.append(np.sum(np.abs( Y0_[B_ == b][1:] - Y0_[B_ == b][:-1]), axis=0))
direction_vectors = np.abs(direction_vectors)
direction_vectors = direction_vectors/np.sum(direction_vectors, axis=1)[:, np.newaxis]
print(direction_vectors)

fig, ax = plt.subplots(figsize=(8,4))
b_names = [data.behaviour_names[i] for i in range(8)]
ax.bar(b_names, direction_vectors[:, 0], label='Sensory')
ax.bar(b_names, direction_vectors[:, 1], bottom=direction_vectors[:, 0], label='Interneurons')
ax.bar(b_names, direction_vectors[:, 2], bottom=direction_vectors[:, 0] + direction_vectors[:, 1], label='Motor')

ax.set_xlabel('Behaviors')
ax.set_ylabel('Contributions')
ax.legend()

plt.xticks(rotation=40)
plt.tight_layout()



# Plotting latent space dynamics
vis = LatentSpaceVisualiser(Y0_, B_, data.behaviour_names)
# vis.plot_latent_timeseries()
fig, ax = vis.plot_phase_space(show_fig=False, arrow_length_ratio=0.1)
ax.set_axis_on()
ax.set_xlabel('sensory neurons axis ')
ax.set_ylabel('inter neuron axis')
ax.set_zlabel('motor neuron axis')
plt.show()


plt.show()


