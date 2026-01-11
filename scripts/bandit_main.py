import torch
import numpy as np
import matplotlib.pyplot as plt
from ncmcm.data_loaders.bandit_task import BanditTaskNeuroPixelsDataset
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space
from ncmcm.bundlenet.utils import prep_data
from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
from sklearn.preprocessing import LabelEncoder

# Load bandit task data
data_path = 'datasets/raw/twoArmBandit/JPAS_0023_20230922'
dataset = BanditTaskNeuroPixelsDataset(
    data_path=data_path,
    downsample_fs=20,
    downsample_method='count',
    good_neurons_only=False,
    normalize_method='minmax_global'
)

# Extract neuronal and behavioral data
X = dataset.x.toarray().T  # Convert sparse to dense and transpose to (time, neurons)
B = dataset.b.toarray().flatten()  # Convert sparse to dense and flatten to 1D

print(f"Neuronal data shape: {X.shape}")
print(f"Behavioral data shape: {B.shape}")
print(f"Behavioral labels: {dataset.b_labels_dict}")
print(f"Recording length: {dataset.get_recording_length_mins():.2f} minutes")

# Plot neuronal and behavioral data overview
plotting_neuronal_behavioural(
    X.T,  # Expects (neurons, time)
    b=B,
    b_names=dataset.b_labels_dict,
    b_colors=dataset.get_color_map_for_plotting(),
    show_fig=True
)

# Prepare data for BunDLeNet
label_encoder = LabelEncoder()
B_encoded = label_encoder.fit_transform(B)
X_, B_ = prep_data(X, B_encoded, win=50)

print(f"Prepared data shape: X_={X_.shape}, B_={B_.shape}")

# Deploy BunDLeNet
model = BunDLeNet(
    latent_dim=3,
    num_behaviour=len(dataset.b_labels_dict),
    input_shape=X_.shape
)

loss_array, _ = train_model(
    X_,
    B_,
    model,
    b_type='discrete',
    gamma=0.75,
    learning_rate=0.00005,
    n_epochs=500
)

# Plot training loss
plt.figure(figsize=(10, 6))
for i, label in enumerate([
    r"$\mathcal{L}_{\mathrm{Markov}}$",
    r"$\mathcal{L}_{\mathrm{Behavior}}$",
    r"Total loss $\mathcal{L}$"
]):
    plt.plot(loss_array[:, i], label=label)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Project into latent space
Y0_ = project_into_latent_space(X_, model)

# Save results (optional)
save_results = False
if save_results:
    output_dir = 'datasets/generated/bandit_task'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_weights(f'{output_dir}/bundlenet_model_weights.h5')
    np.savetxt(f'{output_dir}/latent_trajectories.txt', Y0_)
    np.savetxt(f'{output_dir}/behavior_labels.txt', B_)

# Visualize latent space dynamics
vis = LatentSpaceVisualiser(
    Y0_,
    B_,
    dataset.b_labels_dict,
    colors=dataset.get_rgb_colors_for_visualizer()
)

vis.plot_latent_timeseries()
vis.plot_phase_space()
vis.rotating_plot(filename='figures/rotation_bandit_task.gif', show_fig=True)