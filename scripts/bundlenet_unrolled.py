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

UNROLL = 'both'
FIGURES_DIR = f'figures/bundlenet_unrolled_{UNROLL}'
SAVE_DIR = f'data/generated/bundlenet_unrolled_{UNROLL}'
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
N_STEPS = 10
DISCOUNT = 0.9

label_encoder = LabelEncoder()
B = label_encoder.fit_transform(B)
X_, B_ = prep_data(X, B, win=WIN, n_steps=N_STEPS)

B_step0 = B_[:, 0]  # step-0 labels for visualisation and saving

print(f"\nData prep  (window = {WIN}, n_steps = {N_STEPS})")
print(f"  X_ shape: {X_.shape}  (samples × {N_STEPS + 1} × win × neurons)")
print(f"  B_ shape: {B_.shape}  (samples × n_steps)")
print(f"  Behaviour distribution at step 0:")
for enc_idx in np.unique(B_step0):
    orig_idx = label_encoder.classes_[enc_idx]
    name = data.behaviour_names[orig_idx]
    count = int(np.sum(B_step0 == enc_idx))
    pct = 100 * count / len(B_step0)
    print(f"    {name:22s}: {count:5d} ({pct:4.1f}%)")
print("=" * 60)

# ── Model ─────────────────────────────────────────────────────────────────────
LATENT_DIM = 3
GAMMA = 0.9
LR = 0.001
N_EPOCHS = 1000

discount_weights = [DISCOUNT ** j for j in range(N_STEPS)]

print(f"\nUnrolling parameters:")
print(f"  n_steps:  {N_STEPS}")
print(f"  discount: {DISCOUNT}")
print(f"  weights per step: {[round(w, 4) for w in discount_weights]}")
print(f"  (sum of weights = {sum(discount_weights):.4f}  →  normalised)")

model = BunDLeNet(
    latent_dim=LATENT_DIM,
    num_behaviour=len(data.behaviour_names),
    input_shape=X_.shape,
    n_steps=N_STEPS,
)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel: BunDLeNet (unrolled, n_steps={N_STEPS})")
print(model)
print(f"\nTrainable parameters: {n_params:,}")
print(f"\nHyperparameters:")
print(f"  latent_dim:    {LATENT_DIM}")
print(f"  num_behaviour: {len(data.behaviour_names)}")
print(f"  gamma:         {GAMMA}")
print(f"  learning_rate: {LR}")
print(f"  n_epochs:      {N_EPOCHS}")
print(f"  window:        {WIN}")
print(f"  n_steps:       {N_STEPS}")
print(f"  discount:      {DISCOUNT}")
print("=" * 60)

# ── Training ──────────────────────────────────────────────────────────────────
print(f"\nTraining...")
loss_array, _ = train_model(
    X_, B_, model,
    b_type='discrete',
    gamma=GAMMA,
    learning_rate=LR,
    n_epochs=N_EPOCHS,
    n_steps=N_STEPS,
    discount=DISCOUNT,
    unroll=UNROLL,
)

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

baseline_loss_path = f'data/generated/bundlenet_baseline/loss_array_worm_{worm_num}.txt'
baseline_loss = None
if os.path.exists(baseline_loss_path):
    baseline_loss = np.loadtxt(baseline_loss_path)
    print(f"\nBaseline results found at {baseline_loss_path} — overlaying for comparison")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
for i, label in enumerate(tex_labels):
    ax1.plot(loss_array[:, i], label=f"unrolled {label}")
    ax2.semilogy(loss_array[:, i], label=f"unrolled {label}")
    if baseline_loss is not None:
        ax1.plot(baseline_loss[:, i], '--', alpha=0.6, label=f"baseline {label}")
        ax2.semilogy(baseline_loss[:, i], '--', alpha=0.6, label=f"baseline {label}")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title(f"Training loss — all components  (n_steps={N_STEPS}, discount={DISCOUNT})")
ax1.legend(fontsize=7)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss (log scale)")
ax2.set_title("Training loss — log scale")
ax2.legend(fontsize=7)
plt.tight_layout()
loss_path = os.path.join(FIGURES_DIR, 'training_loss.png')
fig.savefig(loss_path, dpi=150)
plt.close(fig)
print(f"Saved → {loss_path}")

if baseline_loss is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, label in enumerate(tex_labels):
        ax1.plot(loss_array[:, i], color=colors[i], label=f"unrolled {label}")
        ax1.plot(baseline_loss[:, i], '--', color=colors[i], alpha=0.6, label=f"baseline {label}")
        ax2.semilogy(loss_array[:, i], color=colors[i], label=f"unrolled {label}")
        ax2.semilogy(baseline_loss[:, i], '--', color=colors[i], alpha=0.6, label=f"baseline {label}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Unrolled (n_steps={N_STEPS}) vs baseline")
    ax1.legend(fontsize=7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (log scale)")
    ax2.set_title("Unrolled vs baseline — log scale")
    ax2.legend(fontsize=7)
    plt.tight_layout()
    cmp_path = os.path.join(FIGURES_DIR, 'loss_comparison.png')
    fig.savefig(cmp_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {cmp_path}")

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
np.savetxt(b_path, B_step0)
print(f"\nSaved model weights   → {model_path}")
print(f"Saved latent coords   → {y_path}")
print(f"Saved behaviour array → {b_path}")
print("=" * 60)

# ── Latent space plots ────────────────────────────────────────────────────────
vis = LatentSpaceVisualiser(Y0_, B_step0, data.behaviour_names)

ts_path = os.path.join(FIGURES_DIR, 'latent_time_series.png')
vis.plot_latent_timeseries(show_fig=False, filename=ts_path)
print(f"Saved → {ts_path}")

ps_path = os.path.join(FIGURES_DIR, 'phase_space_dynamics.png')
vis.plot_phase_space(show_fig=False, filename=ps_path)
print(f"Saved → {ps_path}")

rot_path = os.path.join(FIGURES_DIR, f'rotation_worm_{worm_num}.gif')
vis.rotating_plot(show_fig=False, filename=rot_path)
print(f"Saved → {rot_path}")
