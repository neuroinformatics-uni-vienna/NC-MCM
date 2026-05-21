<!-- last-updated: 2026-05-05 | agent: Repo Context Documenter -->

# Two-Arm Bandit Task — Domain Documentation

This file documents everything specific to the branch that applies NC-MCM to the two-arm bandit NeuroPixels dataset. It complements the general module docs with bandit-specific semantics, data schemas, pipeline details, and result conventions.

---

## 1. Experimental Paradigm

### Task structure
A head-fixed mouse (subject ID `JPAS_0023`) performs a probabilistic two-armed bandit task:
- The mouse must hold in a neutral zone (`hold` state), then reach left or right to choose an arm
- Each arm has a reward probability set by the experimenter (e.g. 0.78 vs 0.11)
- The rewarded arm (`better left` / `better right`) switches when the animal reaches a performance criterion
- The block structure therefore tracks the animal's learned policy across reversals

### Behavioural states (raw)
States recorded by the behavioural software, in order of appearance:

| State name | Meaning |
|-----------|---------|
| `delay` | Pre-recording hardware alignment period (trimmed before analysis) |
| `waiting` | Mouse is present but no trial is running; trimmed from start/end |
| `intertrial` | Between trials |
| `hold` | Mouse is holding at the neutral zone |
| `choosing left` | Mouse is in motion toward or at the left arm |
| `choosing right` | Mouse is in motion toward or at the right arm |
| `reward` | Outcome period — reward was delivered |
| `no reward` | Outcome period — no reward |

The `delay` and `waiting` states at the start and end are automatically trimmed by `_trim_waiting_periods`. After trimming, analysis data typically contains 6 active states.

### Block types
- `delay` — hardware sync period at recording start (not a task block)
- `standby` — mouse not actively running task
- `better left` — current block where the left arm has the high reward probability
- `better right` — current block where the right arm has the high reward probability

Each block entry has `probabilities l/r` (two floats, one high ~0.78, one low ~0.11) and `block_transition_perf_criteria` (the reward rate threshold that triggered the switch).

### Sessions
Three recording sessions from the same animal:

| Session | Date | Trials | Notes |
|---------|------|--------|-------|
| `JPAS_0023_20230922` | 2023-09-22 | 258 | Primary session; most cache variants present |
| `JPAS_0023_20230927` | 2023-09-27 | — | Second session |
| `JPAS_0023_20230928` | 2023-09-28 | — | Third session |

---

## 2. Raw Data Files (per session directory)

| File | Format | Contents |
|------|--------|---------|
| `spike_times.npy` | `(N_spikes,)` int64 | Spike times in neuronal sample units (Kilosort 4 output) |
| `spike_clusters.npy` | `(N_spikes,)` int | Cluster ID for each spike |
| `spike_times_milliseconds_sync_to_behav.npy` | `(N_spikes,)` float | Spike times in milliseconds, aligned to behavioural clock |
| `spike_templates.npy` | `(N_spikes,)` | Template assignments from Kilosort |
| `cluster_info.tsv` | TSV | Cluster metadata: `cluster_id`, `group` (`good`/`mua`/`noise`), `ch`, `depth`, etc. |
| `cluster_Amplitude.tsv` | TSV | Per-cluster mean amplitude |
| `cluster_group.tsv` | TSV | Manual curation labels |
| `channel_map.npy` | `(384,)` | Electrode channel layout |
| `channel_positions.npy` | `(384, 2)` | XY positions of each channel (micrometres) |
| `metrics.json` | JSON | Behavioural state, trial, and block time-series (see §3) |
| `params.py` | Python | `sample_rate` (~32050 Hz), `n_channels_dat=384`, `dtype='float32'` |

**Hardware sample rate:** `sample_rate ≈ 32050.76 Hz` (Neuropixels probe via SpikeGLX, read from `params.py`).

**Time alignment:** `spike_times_milliseconds_sync_to_behav.npy` provides a lookup from neuronal sample index to behavioural clock milliseconds. During loading, `np.interp` is called across all neuronal time indices to produce a dense translation array (`translation_indices_neuronal_to_behavioral`) that maps every downsampled neuronal bin to its corresponding behavioural millisecond.

---

## 3. `metrics.json` Schema

Top-level keys: `experiment data`, `performance`, `metrics`, `kayeton cam (t ms/#frame/vid time)`, `wheel`, `clock 100ms`, `clock 1s`, `clock 10s`, `clock 5min`.

The relevant subtree is `metrics`:

```
metrics.json
└── "metrics"
    ├── "trials"    list of trial dicts
    ├── "blocks"    list of block dicts
    └── "states"    list of [t_ms, state_name] pairs
```

### `trials` entry fields
| Field | Type | Meaning |
|-------|------|---------|
| `start` | int ms | Trial start time (intertrial onset) |
| `hold_start` | int ms | Time mouse entered hold zone |
| `t choosing` | int ms | Time choosing period began |
| `t chosen` | int ms | Time choice was registered |
| `choice` | `"l"` \| `"r"` | Which arm was chosen |
| `rewarded` | bool | Whether reward was delivered |
| `set reward probabs l/r` | [float, float] | Experimenter-set probabilities at time of trial |
| `Achievement reached` | bool | Whether this trial caused a block switch |
| `reward_rate_recent` | float | Rolling reward rate (window unspecified) |
| `rew rate in recent rights` | float | Rolling reward rate for right choices |
| `rew rate in recent lefts` | float | Rolling reward rate for left choices |

### `blocks` entry fields
| Field | Type | Meaning |
|-------|------|---------|
| `t` | int ms | Block start time in behavioural clock |
| `block` | str | Block name: `delay`, `standby`, `better left`, `better right` |
| `probabilities l/r` | [float, float] | Reward probabilities (only on non-standby/delay blocks) |
| `block_transition_perf_criteria` | float | Performance threshold that ended the previous block |

### `states` entries
Each entry is a 2-element list `[t_ms, state_name]` where `t_ms` is the onset time in behavioural milliseconds. The raw states are translated to neuronal time bins during loading.

---

## 4. HGF Models

The Hierarchical Gaussian Filter (HGF) is a Bayesian model of learning under uncertainty. It is fitted to the animal's trial-by-trial choices to infer latent belief trajectories.

### Location
`{session_dir}/hgf_models/*.pkl` — each file is a pandas DataFrame with one row per trial.

**Example files in `JPAS_0023_20230922/hgf_models/`:**
- `20230922_input1_binary2.pkl` — HGF with 2-level binary input, volatility included
- `20230922_input1_binary3.pkl` — 3-level variant
- `20230922_input2_nomasking_1volnode.pkl` — alternative parameterisation

### Loading in `BanditTaskNeuroPixelsDataset`
Controlled by:
- `hgf_model` — substring matched against the PKL filename (e.g. `'binary2'` matches `*binary2*.pkl`)
- `hgf_column` — column name in the loaded DataFrame

### Key HGF columns
| Column | Conceptual range | Meaning |
|--------|-----------------|---------|
| `x_1_expected_mean` | `[-2, 2]` | Precision-weighted log-odds belief at level 1 |
| `x_0_expected_mean` | `[0, 1]` | Prior reward probability at level 0 |

### Rescaling
Beliefs are rescaled to `[-1, 1]` via:
$$\text{rescaled} = 2 \cdot \frac{x - \text{lo}}{\text{hi} - \text{lo}} - 1$$

Ranges are looked up from `KNOWN_HGF_RANGES` or passed explicitly via `hgf_belief_range`. The rescaled values are aligned to neuronal timepoints by repeating each trial's belief value across all bins belonging to that trial.

---

## 5. `BanditTaskNeuroPixelsDataset` — Deep Dive

### Cache mechanism
Cache files are stored in `{data_path}/BanditTaskNeuroPixelsDataset/`. The filename encodes the key preprocessing parameters:
```
fs{downsample_fs}_{downsample_method}_{good/all}_norm{normalize_method}_{hash8}.pkl
```
Example: `fs20_count_good_normminmax_global_3c9634e2.pkl`

The 8-character hash is a truncated MD5 of all constructor arguments (including `state_transitions`, `choosing_state_mode`, `hgf_model`, `hgf_column`, etc.). This means any change in parameters creates a new cache file. To force regeneration: `recompute_cache=True`.

### Loading pipeline (step-by-step)

```
params.py          → sample_rate (~32050 Hz)
spike_times.npy    ─┐
spike_clusters.npy  ├─→ _create_sparse_neuronal_data_matrix  → x (n_neurons, T_raw)  [COO→CSR]
cluster_info.tsv   ─┘   (filtered to 'good' if good_neurons_only=True)
spike_times_ms.npy → translation_indices (neuronal sample → behavioural ms)

↓ if downsample_fs:
  _downsample_translation_indices  (center-of-bin sampling)
  _downsample_spike_data           → x (n_neurons, T_ds)
  fs ← downsample_fs (adjusted for rounding)

metrics.json ──────→ _create_behavioral_data_matrix      → b (1, T_ds) sparse, b_labels_dict
                   → _create_continuous_behavioral_data  → b_continuous (T_ds,) running avg last 10
                   → _create_trial_indices               → trial_indices (T_ds,)
                   → _create_block_indices               → block_indices (T_ds,)
                   → _create_block_labels                → block_labels (T_ds,) str array
                   → _create_behavioral_time_array       → behavioral_time (T_ds,) ms

hgf_models/*.pkl ──→ _align_hgf_beliefs                  → hgf_beliefs (T_ds,) rescaled

↓ _trim_waiting_periods     (removes leading/trailing 'waiting' state bins)
↓ _normalize_neuronal_data  (optional: minmax per-neuron or minmax_global)
↓ _apply_state_transitions  (optional: merge consecutive states)
↓ _relabel_behavioral_states (ensure 0-indexed contiguous state IDs)
↓ self.b_labels = sorted list of state names
```

### Downsampling methods detail

| Method | Output dtype | Description |
|--------|-------------|-------------|
| `binary` | `uint8` | OR: 1 if any spike in the bin |
| `count` | `uint16` | Sum of spikes per bin |
| `rate` | `float32` | `counts * original_fs / bin_size_samples` (Hz) |
| `mean` | `float32` | `counts / bin_size_samples` (spike proportion, 0-1) |
| `gaussian` | `float32` | Gaussian kernel convolution (σ=`gaussian_sigma_ms` ms), firing rate Hz; kernel truncated at ±3σ |

All methods operate on the sparse COO representation and return a `scipy.sparse.csr_matrix`.

### State transition merging
`state_transitions` is a dict of `{(state1_name, state2_name): combined_name}`. When consecutive timepoints transition from `state1` to `state2`, the entire contiguous segment spanning both is relabelled as `combined_name`. The entire original state segment and the following state segment are merged.

Pre-defined combinations (class constants):
- `HOLD_TO_CHOOSING_TRANSITIONS`: merges `hold → choosing left/right` into single combined states
- `CHOOSING_TO_OUTCOME_TRANSITIONS`: merges `choosing left/right → reward/no reward`
- `CHOOSING_TO_CORRECTNESS_TRANSITIONS`: relabels outcomes as `choosing reward` / `choosing no reward` (ignores side)

### `choosing_state_mode`
- `'side'` (default): the choosing state is split by arm → `choosing left` / `choosing right`
- `'correctness'`: the choosing state is split by outcome → `choosing correct` / `choosing wrong` (correct = chose the currently better arm)

### `b_mode`
Controls which timepoints are included in the `b` (behavioural label) array and how they map to trial choices:

| `b_mode` | Coverage | Window per trial N | Label | Notes |
|----------|----------|-------------------|-------|-------|
| `'full'` | All timepoints | entire recording | state machine label | includes all ITI, reward, etc. |
| `'decision'` | Trial windows only | `[trial_start, trial_end]` | choice (L/R) | raw trial boundaries from metrics.json |
| `'decision_strict'` | Trial windows only | `[t_chosen[N-1]+1, t_chosen[N]]` for N>0; `[trial_start, t_chosen[0]]` for N=0 | choice (L/R) | boundaries truncated to adjacent t_chosen |
| `'reward_to_choice'` | Trial windows only | `[t_chosen[N-1]+1, t_chosen[N]]` for N≥1 | choice (L/R) | first trial dropped (no prior t_chosen); cleanest upcoming-decision training |

`reward_to_choice` notes:
- Trial 0 is dropped; training starts from the second trial.
- Each segment spans exactly from the previous choice to the current choice — no trial-start boundary contamination.
- 237 detectable segments on `JPAS_0023_20230922` at 30 Hz (same as `decision_strict`; 17 trials extend beyond recording end).
- Cache key is distinct from other modes; safe to load cached datasets for other modes in parallel.

### Attributes summary
| Attribute | Shape | dtype | Description |
|-----------|-------|-------|-------------|
| `x` | `(n_neurons, T)` | sparse | Neuronal activity — **call `.toarray()` before use** |
| `b` | `(1, T)` | sparse | Behavioural state IDs — **call `.toarray().flatten()` before use** |
| `b_labels_dict` | dict `{int: str}` | — | Maps state ID to state name |
| `b_labels` | list[str] | — | Ordered by state ID |
| `b_continuous` | `(T,)` | float | Running average of last 10 decisions: left=-1, right=+1 |
| `trial_indices` | `(T,)` | int | 0-indexed trial number per timepoint; -1 = outside trials |
| `block_indices` | `(T,)` | int | 0-indexed block number per timepoint; -1 = pre-first block |
| `block_labels` | `(T,)` | str array | Block name (e.g. `'better left'`) per timepoint |
| `behavioral_time` | `(T,)` | float | Behavioural clock time in ms at each bin's centre |
| `fs` | float | — | Actual sampling frequency after downsampling |
| `hgf_beliefs` | `(T,)` or None | float32 | HGF belief rescaled to [-1, 1]; None if `hgf_model=None` |

### Public methods
- `load_data()` — called automatically by `__init__`; can be called manually with `recompute_cache=True`
- `get_recording_length_mins()` — returns `T / (fs * 60)` as a float
- `get_color_map_for_plotting()` — returns a subset of `DEFAULT_COLOR_MAP` restricted to states present in `b_labels`

---

## 6. Behaviour Representations for BunDLe-Net

Three modes, all constructed from `BanditTaskNeuroPixelsDataset` attributes:

### Discrete (`b_type='discrete'`)
```python
B_encoded = data.b.toarray().flatten()           # (T,)  int
x_paired, b_1 = prep_data(X, B_encoded, win=50)  # b_1 shape (m,)
```
CrossEntropy loss against the `num_behaviour = len(data.b_labels)` classes.

### Continuous (`b_type='continuous'`)
```python
B_cont = data.b_continuous.reshape(-1, 1)        # (T, 1) float
x_paired, b_1 = prep_data(X, B_cont, win=50)     # b_1 shape (m, 1)
```
MSE loss against the continuous running-average signal.

### Hybrid (`b_type='hybrid'`)
```python
B_hybrid = make_hybrid_b(B_encoded, data.hgf_beliefs)  # (T, 2): col0=class idx, col1=HGF
x_paired, b_1 = prep_data(X, B_hybrid, win=50)         # b_1 shape (m, 2)
BunDLeNet(latent_dim=3, num_behaviour=len(data.b_labels), n_classes=len(data.b_labels))
train_model(..., b_type='hybrid', alpha=0.1)
```
Loss = `gamma * DCC + (1-gamma) * [alpha * CE_normalised + (1-alpha) * MSE_HGF]`

The `alpha` parameter weights how much the network is driven toward discrete state separation vs. encoding the HGF belief trajectory.

---

## 7. Analysis Pipeline (Full)

```
Raw Kilosort output
    └─→ BanditTaskNeuroPixelsDataset(session_path, downsample_fs, ...)
            └─→ x (sparse), b (sparse), hgf_beliefs, trial_indices, block_indices

Preprocessing
    └─→ x.toarray().T         → X shape (T, n_neurons)
    └─→ b.toarray().flatten() → B shape (T,)
    └─→ make_hybrid_b(B, hgf_beliefs) → B_hybrid shape (T, 2)

Data preparation
    └─→ prep_data(X, B, win=50)  → x_paired (m, 2, win, n_neurons), b_1 (m,)
    └─→ timeseries_train_test_split / timeseries_train_test_split_cv
    └─→ torch_batch_prep → DataLoader

BunDLe-Net training
    └─→ BunDLeNet(latent_dim=3, num_behaviour=n_states, ...)
    └─→ train_model(x_train, b_train_1, model, b_type='hybrid', gamma=0.75, lr=5e-5, n_epochs=500)
    └─→ train_history, test_history  shape (n_epochs, 5)

Latent space extraction
    └─→ project_into_latent_space(x_paired, model) → Y0_ (T, 3)

Visualisation (per run, saved to results/)
    └─→ LatentSpaceVisualiser(Y0_, B, data.b_labels, ...)
         .plot_latent_timeseries_plotly()  → latent_time_series_{train,validation}.html
         .plot_phase_space(...)            → phase_space_dynamics_{train,validation}_{front,right,top}.png
    └─→ plotting_neuronal_behavioural_plotly(X, B, ...) → neural_behavioural_overview.html
    └─→ (hybrid only) phase_space_continuous_*_HGF_belief_{front,right,top}.png

Evaluation
    └─→ bandit_behaviour_decoding.py (latent space → behaviour decodability)
         ├─→ Discrete decoder: linear (10 runs, 200 epochs) unweighted + weighted CE
         ├─→ Permutation baseline (200 permutations)
         └─→ HGF belief regression (if hgf_beliefs available)

    └─→ bandit_microvariable_evaluation.py (raw neural → behaviour decodability, baseline)
         ├─→ Discrete: same decoder setup as above
         ├─→ Hybrid: discrete + continuous joint decoder
         └─→ Continuous: HGF belief regression
```

---

## 8. Scripts Reference (Bandit-Specific)

### `scripts/bandit_main.py`
Quickstart **continuous time-series** training script. No CLI args — edit constants at the top.
- `data_path = 'datasets/raw/twoArmBandit/JPAS_0023_20230922'`
- `b_type = 'discrete'`, `gamma=0.75`, `lr=5e-5`, `n_epochs=500`, `win=50`
- Trial-based block removed from this file in `ae1a929` — see `bandit_main_trial_based.py`

### `scripts/bandit_main_trial_based.py` *(new in `ae1a929`)*
Standalone quickstart for the **trial-based regime**. No CLI args.
- `segment_trials` → `prep_data_trials(win=50)` → `trial_train_test_split(test_ratio=0.2, random_state=42)`
- `b_type='discrete'`, `gamma=0.75`, `lr=5e-5`, `n_epochs=500`
- Optional save to `datasets/generated/bandit_task/` (controlled by `save_results=False`)

### `scripts/bandit_gridsearch.py`
Full CLI-driven training + visualisation with grid-search support. All list-valued args are Cartesian-producted.

**Key CLI patterns:**
```bash
# Single run, hybrid mode
python scripts/bandit_gridsearch.py \
  --data_path datasets/raw/twoArmBandit/JPAS_0023_20230922 \
  --downsample_fs 20 --downsample_method gaussian \
  --normalize_method minmax_global --good_neurons_only true \
  --b_type hybrid --alpha 0.1 --gamma 0.75 --learning_rate 5e-5 \
  --n_epochs 500 --window 50 --pca_init \
  --output_dir results/twoArmBandit/my_run

# Grid search over alpha values
python scripts/bandit_gridsearch.py \
  --alpha 0.1 0.3 0.5 0.7 0.9 \
  --output_dir results/twoArmBandit/hybrid_alpha_search
```

**Grid search directory naming:**
- Top-level: `grid_search_{YYYYMMDD_HHMMSS}/`
- Per-run: `run_{NNN}_{param1=val1}_{param2=val2}_...` (all grid parameters concatenated)

### `scripts/bandit_behaviour_decoding.py`
Post-hoc latent space decodability evaluation. Run after a gridsearch run.
```bash
python scripts/bandit_behaviour_decoding.py results/twoArmBandit/my_run/run_000_...
```
Outputs to `{run_dir}/data/decoding/`. Reads `latent_trajectories_{train,validation}.npy` and `behavior_labels_{train,validation}.npy` from the run's `data/` directory.

### `scripts/bandit_microvariable_evaluation.py`
Baseline evaluation on raw neural activity. Configure by editing constants at the top.
```bash
python scripts/bandit_microvariable_evaluation.py datasets/raw/twoArmBandit/JPAS_0023_20230922
# Optional: override split mode
python scripts/bandit_microvariable_evaluation.py datasets/raw/twoArmBandit/JPAS_0023_20230922 cv
```
Outputs to `{data_path}/microvariable_evaluation/` (inside the dataset folder, not `results/`).

### `scripts/overnight_run.sh`
Full pipeline for session `JPAS_0023_20230927`:
1. Train hybrid BunDLe-Net (Gaussian method, `fs=30`, `win=90`, `alpha=0.1`, `gamma=0.75`)
2. Run latent-space decoding
3. Run microvariable evaluation
Logs to `/tmp/bundlenet_overnight.log`.

---

## 9. Results Directory Structure

```
results/twoArmBandit/
├── hybrid_alpha_search/               ← alpha sweep runs
│   └── grid_search_YYYYMMDD_HHMMSS/
│       ├── grid_search_summary.json   ← all run configs + metrics in one file
│       └── run_NNN_param1=v1_.../
│           ├── config.json            ← this run's parameters
│           ├── run_summary.json       ← config + final losses + metadata
│           ├── data/                  ← latent trajectories + behaviour labels (npy)
│           ├── figures/               ← HTML + PNG visualisations (see below)
│           └── model/                 ← bundlenet_model.pt
│
├── smoke_test/                        ← quick sanity-check runs
├── trial_based/                       ← trial-based training runs
└── microvariable_evaluation/          ← top-level baseline results
```

### Per-run `figures/` contents
| File | Description |
|------|-------------|
| `training_loss.png` | Loss curves (DCC, BCC, total for train + val) |
| `neural_behavioural_overview.html` | Plotly interactive neuronal overview |
| `latent_time_series_train.html` | Interactive latent time series (train split) |
| `latent_time_series_validation.html` | Interactive latent time series (validation split) |
| `phase_space_dynamics_{train,val}_{front,right,top}.png` | 3 orthographic views of latent 3D phase space coloured by discrete behaviour |
| `phase_space_continuous_{train,val}_HGF_belief_{front,right,top}.png` | Same views coloured by HGF belief (hybrid runs only) |
| `rotating_latent_space.gif` | Optional animated rotation (requires `--generate_gif`) |
| `latent_space_3d.html` | Optional interactive 3D Plotly (requires `--generate_3d_html`) |

### Per-run `data/` contents
| File | Shape | Description |
|------|-------|-------------|
| `latent_trajectories_train.npy` | `(T_train, latent_dim)` | BunDLe-Net latent space (train split) |
| `latent_trajectories_validation.npy` | `(T_val, latent_dim)` | BunDLe-Net latent space (val split) |
| `behavior_labels_train.npy` | `(T_train,)` | Discrete behaviour labels |
| `behavior_labels_validation.npy` | `(T_val,)` | Discrete behaviour labels |
| `hgf_belief_train.npy` | `(T_train,)` | HGF belief (rescaled, hybrid runs only) |
| `hgf_belief_validation.npy` | `(T_val,)` | HGF belief (rescaled, hybrid runs only) |
| `decoding/` | — | Output of `bandit_behaviour_decoding.py` |

---

## 10. Notebooks (Bandit-Specific)

### `demos/bundlenet_on_two_arm_bandit_data.ipynb`
Entry-point tutorial for the bandit dataset. Covers discrete and hybrid mode, includes `make_hybrid_b` example.

### `demos/cognitive_graphs_bandit.ipynb`
Cognitive graph construction from bandit latent space. Shows how block structure and learned policy relate to cognitive states.

### `demos/bandit_latent_space_visualization.ipynb`
Extended Plotly-based exploration of latent trajectories with trial and block annotations.

### `demos/trial_based_bundlenet.ipynb`
Full tutorial on the trial-based regime. Covers motivation (boundary-crossing windows), API walkthrough (`segment_trials` → `prep_data_trials` → `trial_train_test_split`), pairs-per-trial distribution plot, and end-to-end BunDLe-Net training and visualisation.

### Raw data exploration notebooks (inside dataset directory)
| Notebook | Path | Purpose |
|----------|------|---------|
| `raw_data_exploration.ipynb` | `datasets/raw/twoArmBandit/` | Inspect spike times, cluster info, ISI distributions |
| `preprocessed_data_exploration.ipynb` | `datasets/raw/twoArmBandit/` | Inspect `BanditTaskNeuroPixelsDataset` output |
| `time_alignment/plotting.ipynb` | `datasets/raw/twoArmBandit/time_alignment/` | Visualise sync between neural and behavioural clocks |

---

## 11. Common Gotchas

| Issue | Cause | Fix |
|-------|-------|-----|
| `TypeError: sparse matrix` passed to PyTorch | `x` and `b` are sparse | Always call `.toarray()` (and `.flatten()` for `b`) |
| `ValueError: hgf_belief_range` | HGF column not in `KNOWN_HGF_RANGES` | Pass `hgf_belief_range=(lo, hi)` explicitly or add entry to `KNOWN_HGF_RANGES` |
| Stale cache after changing `state_transitions` | Different hash → different cache filename | Either use `recompute_cache=True` or the new hash will auto-create a new cache |
| `choosing_state_mode='correctness'` with `CHOOSING_TO_OUTCOME_TRANSITIONS` | Incompatible: correctness mode already merges by outcome | Use only one or the other |
| `b_type='hybrid'` requires `n_classes` | `BccDccLoss` needs this for normalising CE | Always pass `n_classes=len(data.b_labels)` to `BunDLeNet` when using hybrid |
| `x.T` shape `(T, n_neurons)` needed for `prep_data` | `x` from dataset is `(n_neurons, T)` | Use `data.x.toarray().T` |
| Block labels after trimming | `block_labels` may start with `'standby'` for some sessions | Filter by `block_labels == 'better left'` / `'better right'` for task-only analysis |
