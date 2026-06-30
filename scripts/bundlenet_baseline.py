import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space
from ncmcm.bundlenet.utils import prep_data
from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
from sklearn.preprocessing import LabelEncoder

FIGURES_DIR = 'figures/bundlenet_baseline'
SAVE_DIR = 'data/generated/bundlenet_baseline'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
worm_num = 0
b_neurons = [
    'AVAR', 'AVAL', 'SMDVR', 'SMDVL',
    'SMDDR', 'SMDDL', 'RIBR', 'RIBL',
]
data_path = 'datasets/raw/c_elegans/NoStim_Data.mat'
data = Database(data_path=data_path, dataset_no=worm_num)

print("=" * 60)
print(f"Dataset:   {data_path}")
print(f"Worm:      {worm_num}")
print(f"Neurons:   {data.neuron_traces.shape[0]}")
print(f"Timesteps: {data.neuron_traces.shape[1]}")
print(f"FPS:       {data.fps}")
print(f"\nBehaviour distribution (raw):")
for idx, name in data.behaviour_names.items():
    count = int(np.sum(data.behaviour == idx))
    pct = 100 * count / len(data.behaviour)
    print(f"  {idx}  {name:22s}: {count:5d} ({pct:4.1f}%)")
print(f"\nExcluding {len(b_neurons)} behavioural neurons: {b_neurons}")

data.exclude_neurons(b_neurons)
X = data.neuron_traces.T   # (T, N)
B = data.behaviour

print(f"Neurons after exclusion: {X.shape[1]}")
print(f"X shape: {X.shape}  (timesteps x neurons)")
print(f"B shape: {B.shape}")
print("=" * 60)

fig, _ = plotting_neuronal_behavioural(X, B, b_names=data.behaviour_names, show_fig=False)
nb_path = os.path.join(FIGURES_DIR, 'neuronal_behavioural.png')
fig.savefig(nb_path, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Saved → {nb_path}")

# ── Data prep ─────────────────────────────────────────────────────────────────
WIN = 1
label_encoder = LabelEncoder()
B = label_encoder.fit_transform(B)
X_, B_ = prep_data(X, B, win=WIN)

print(f"\nData prep  (window = {WIN})")
print(f"  X_ shape: {X_.shape}  (samples × 2 × win × neurons)")
print(f"  B_ shape: {B_.shape}")
print(f"  Behaviour distribution (prepared):")
for enc_idx in np.unique(B_):
    orig_idx = label_encoder.classes_[enc_idx]
    name = data.behaviour_names[orig_idx]
    count = int(np.sum(B_ == enc_idx))
    pct = 100 * count / len(B_)
    print(f"    {name:22s}: {count:5d} ({pct:4.1f}%)")
print("=" * 60)

# ── Model ─────────────────────────────────────────────────────────────────────
LATENT_DIM = 3
GAMMA = 0.9
LR = 0.001
N_EPOCHS = 1000

model = BunDLeNet(
    latent_dim=LATENT_DIM,
    num_behaviour=len(data.behaviour_names),
    input_shape=X_.shape,
)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel: BunDLeNet")
print(model)
print(f"\nTrainable parameters: {n_params:,}")
print(f"\nHyperparameters:")
print(f"  latent_dim:    {LATENT_DIM}")
print(f"  num_behaviour: {len(data.behaviour_names)}")
print(f"  gamma:         {GAMMA}")
print(f"  learning_rate: {LR}")
print(f"  n_epochs:      {N_EPOCHS}")
print(f"  window:        {WIN}")
print("=" * 60)

# ── Training ──────────────────────────────────────────────────────────────────
print(f"\nTraining...")
t0 = time.time()
loss_array, _ = train_model(
    X_, B_, model,
    b_type='discrete',
    gamma=GAMMA,
    learning_rate=LR,
    n_epochs=N_EPOCHS,
)
train_time = time.time() - t0

print(f"\nTraining time: {train_time:.1f}s  ({train_time / N_EPOCHS * 1000:.1f} ms/epoch)")
print(f"\nFinal epoch losses:")
for i, name in enumerate(['Markov', 'Behaviour', 'Total']):
    print(f"  {name:12s}: {loss_array[-1, i]:.6f}")
print(f"Best total loss: {loss_array[:, 2].min():.6f}  (epoch {loss_array[:, 2].argmin()})")

N_STEPS_EVAL = 20
X_eval, B_eval = prep_data(X, B, win=WIN, n_steps=N_STEPS_EVAL)

model.eval()
with torch.no_grad():
    device = next(model.parameters()).device
    x0_ev = torch.tensor(X_eval[:, 0], dtype=torch.float, device=device)
    yt = model.tau(x0_ev)
    print(f"\nBehaviour accuracy (n_steps_eval={N_STEPS_EVAL}):")
    for step in range(N_STEPS_EVAL):
        x_j = torch.tensor(X_eval[:, step + 1], dtype=torch.float, device=device)
        acc_direct = (model.predictor(model.tau(x_j)).argmax(dim=1).cpu().numpy() == B_eval[:, step]).mean()
        yt = yt + model.T_Y(yt)
        acc_unrolled = (model.predictor(yt).argmax(dim=1).cpu().numpy() == B_eval[:, step]).mean()
        print(f"  step {step + 1}:  direct {acc_direct * 100:.1f}%   unrolled {acc_unrolled * 100:.1f}%")
print("=" * 60)

# ── Loss plots ────────────────────────────────────────────────────────────────
tex_labels = [
    r"$\mathcal{L}_{\mathrm{Markov}}$",
    r"$\mathcal{L}_{\mathrm{Behavior}}$",
    r"Total $\mathcal{L}$",
]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

for i, label in enumerate(tex_labels):
    ax1.plot(loss_array[:, i], label=label)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training loss — all components")
ax1.legend()

for i, label in enumerate(tex_labels):
    ax2.semilogy(loss_array[:, i], label=label)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss (log scale)")
ax2.set_title("Training loss — log scale")
ax2.legend()

plt.tight_layout()
loss_path = os.path.join(FIGURES_DIR, 'training_loss.png')
fig.savefig(loss_path, dpi=150)
plt.close(fig)
print(f"Saved → {loss_path}")

# save raw loss values for later comparison
np.savetxt(os.path.join(SAVE_DIR, f'loss_array_worm_{worm_num}.txt'), loss_array,
           header='markov_loss  behaviour_loss  total_loss')

# ── Latent space ──────────────────────────────────────────────────────────────
Y0_ = project_into_latent_space(X_, model)
print(f"\nLatent space: Y0_ shape {Y0_.shape}")
print(f"  mean: {Y0_.mean(axis=0).round(4)}")
print(f"  std:  {Y0_.std(axis=0).round(4)}")

# ── Save model + arrays ───────────────────────────────────────────────────────
model_path = os.path.join(SAVE_DIR, f'BunDLeNet_worm_{worm_num}.pt')
y_path = os.path.join(SAVE_DIR, f'Y0__worm_{worm_num}.txt')
b_path = os.path.join(SAVE_DIR, f'B__worm_{worm_num}.txt')

torch.save(model.state_dict(), model_path)
np.savetxt(y_path, Y0_)
np.savetxt(b_path, B_)
print(f"\nSaved model weights   → {model_path}")
print(f"Saved latent coords   → {y_path}")
print(f"Saved behaviour array → {b_path}")
print("=" * 60)

# ── Latent space plots ────────────────────────────────────────────────────────
vis = LatentSpaceVisualiser(Y0_, B_, data.behaviour_names)

ts_path = os.path.join(FIGURES_DIR, 'latent_time_series.png')
vis.plot_latent_timeseries(show_fig=False, filename=ts_path)
print(f"Saved → {ts_path}")

ps_path = os.path.join(FIGURES_DIR, 'phase_space_dynamics.png')
vis.plot_phase_space(show_fig=False, filename=ps_path)
print(f"Saved → {ps_path}")

rot_path = os.path.join(FIGURES_DIR, f'rotation_worm_{worm_num}.gif')
vis.rotating_plot(show_fig=False, filename=rot_path)
print(f"Saved → {rot_path}")
