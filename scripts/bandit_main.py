import torch
import numpy as np
import matplotlib.pyplot as plt
from ncmcm.data_loaders.bandit_task import BanditTaskNeuroPixelsDataset
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space
from ncmcm.bundlenet.utils import (
    prep_data, make_hybrid_b,
    segment_trials, prep_data_trials, trial_train_test_split,
)
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

# ── Hybrid mode example ──────────────────────────────────────────────────────
# Uncomment the block below to train with joint discrete + continuous behaviour.
# Requires that dataset.hgf_beliefs has been computed (pass hgf_model / hgf_column
# to BanditTaskNeuroPixelsDataset to enable it).
#
# n_classes = len(dataset.b_labels_dict)          # number of discrete classes
# B_hybrid = make_hybrid_b(B_encoded, dataset.hgf_beliefs)  # (T, 1 + n_continuous)
# X_h, B_h = prep_data(X, B_hybrid, win=50)
# model_hybrid = BunDLeNet(
#     latent_dim=3,
#     num_behaviour=n_classes + 1,                # n_classes logits + 1 continuous output
#     input_shape=X_h.shape
# )
# loss_array_hybrid, _ = train_model(
#     X_h, B_h, model_hybrid,
#     b_type='hybrid',
#     n_classes=n_classes,
#     gamma=0.75,
#     learning_rate=0.00005,
#     n_epochs=500
# )
# ─────────────────────────────────────────────────────────────────────────────

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


# ── Trial-based training regime ───────────────────────────────────────────────
# An alternative to the continuous time-series approach above.
# Benefits:
#   - Windows never cross trial boundaries (no artificial reward→intertrial pairs)
#   - Train/test split is trial-level and randomised — no temporal ordering constraint
#   - Pairs can be freely shuffled within the training set

# 1. Segment the session into trials (each trial starts at 'intertrial')
trial_segments = segment_trials(X, B_encoded, dataset.b_labels_dict,
                                trial_start_state='intertrial')
print(f"Number of trials: {len(trial_segments)}")
print(f"Trial lengths (timesteps): min={min(len(b) for _, b in trial_segments)}, "
      f"max={max(len(b) for _, b in trial_segments)}")

# 2. Window each trial independently — no cross-trial pairs
X_trials, B_trials, trial_ids = prep_data_trials(trial_segments, win=50)
print(f"Trial-based prepared data: X={X_trials.shape}, B={B_trials.shape}, "
      f"trial_ids range=[{trial_ids.min()}, {trial_ids.max()}]")

# 3. Random trial-level train/test split (no temporal ordering required)
(X_train, B_train), (X_test, B_test) = trial_train_test_split(
    X_trials, B_trials, trial_ids, test_ratio=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]} pairs | Test: {X_test.shape[0]} pairs")

# 4. Train BundleNet — API is identical to the continuous-series case
model_trial = BunDLeNet(
    latent_dim=3,
    num_behaviour=len(dataset.b_labels_dict),
    input_shape=X_train.shape
)

loss_array_trial, loss_array_test = train_model(
    X_train,
    B_train,
    model_trial,
    b_type='discrete',
    n_classes=len(dataset.b_labels_dict),
    gamma=0.75,
    learning_rate=0.00005,
    n_epochs=500,
    validation_data=(X_test, B_test),
)
# ─────────────────────────────────────────────────────────────────────────────
