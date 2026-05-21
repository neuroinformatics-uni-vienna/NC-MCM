<!-- last-updated: 2026-05-09 | agent: Repo Context Documenter -->

# Scripts & Entry Points

## Overview

All runnable scripts are in `scripts/`. They are not installed as package entry points — run them directly with `python scripts/<script>.py` from the repo root with `.venv` activated.

---

## `scripts/main.py`

**Purpose:** Train BunDLe-Net on C. elegans discrete-behaviour data, visualise the latent space, and optionally save embeddings.

**No CLI arguments** — configure by editing constants at the top of the file:
- `worm_num` (0–4)
- `b_neurons` (list of neuron names to exclude)
- `save_model` (bool)

**Data path:** `datasets/raw/c_elegans/NoStim_Data.mat`

**Pipeline:**
1. `Database(data_path, worm_num)` → `exclude_neurons(b_neurons)`
2. `plotting_neuronal_behavioural(X, B)` — raw data overview
3. `prep_data(X, B, win=15)` → train `BunDLeNet(latent_dim=3, num_behaviour=8)`
4. `train_model(..., b_type='discrete', gamma=0.9, lr=0.001, n_epochs=1000)`
5. `project_into_latent_space` → `LatentSpaceVisualiser.plot_latent_timeseries()`, `plot_phase_space()`, `rotating_plot()`
6. If `save_model=True`, saves to `data/generated/saved_Y/` (note: path uses old `data/` not `datasets/`)

**Runtime:** ~minutes for 1000 epochs on CPU; seconds on GPU.

---

## `scripts/continuous_main.py`

**Purpose:** Train BunDLe-Net on rat hippocampus continuous-behaviour data.

**No CLI arguments** — configure `rat_name` (one of `['achilles', 'gatsby', 'cicero', 'buddy']`).

**Data path:** `datasets/raw/rat_hippocampus/{rat_name}.npz`

**Pipeline:**
1. `np.load(...)` → `prep_data(x, b, win=20)`
2. `BunDLeNet(latent_dim=3, num_behaviour=b_.shape[1])` — `num_behaviour` inferred from continuous `b`
3. `train_model(..., b_type='continuous', gamma=0.9, lr=0.001, n_epochs=100)`
4. 3D scatter plot of latent space coloured by continuous behaviour using raw `matplotlib`

---

## `scripts/bandit_main.py`

**Purpose:** Quickstart **continuous time-series** training script for the two-arm bandit dataset. No CLI args — edit constants at the top. Includes commented-out hybrid mode example.

**No CLI arguments** — configure constants at the top.

**Data path:** `datasets/raw/twoArmBandit/JPAS_0023_20230922`

**Key parameters:**
- `downsample_fs=20`, `downsample_method='count'`, `good_neurons_only=False`, `normalize_method='minmax_global'`
- `b_type='discrete'`, `gamma=0.75`, `lr=0.00005`, `n_epochs=500`, `win=50`

**Hybrid mode** is included as a commented block demonstrating `make_hybrid_b` + `b_type='hybrid'`.

**Note (commit `ae1a929`):** The trial-based training block that was previously at the bottom of this script has been removed and moved to `bandit_main_trial_based.py`.

---

## `scripts/bandit_main_trial_based.py` *(new in commit `ae1a929`)*

**Purpose:** Standalone quickstart for the **trial-based BunDLe-Net training regime** on bandit data. No CLI args — edit constants at the top.

**Data path:** `datasets/raw/twoArmBandit/JPAS_0023_20230922`

**Pipeline:**
1. `BanditTaskNeuroPixelsDataset(downsample_fs=20, normalize_method='minmax_global')`
2. `LabelEncoder` → `B_encoded`
3. `segment_trials(X, B_encoded, b_labels_dict, trial_start_state='intertrial')` → `trial_segments`
4. `prep_data_trials(trial_segments, win=50)` → `X_trials, B_trials, trial_ids`
5. `trial_train_test_split(X_trials, B_trials, trial_ids, test_ratio=0.2, random_state=42)` — random trial-level split
6. Train `BunDLeNet(latent_dim=3, num_behaviour=n_states)` with `b_type='discrete'`, `gamma=0.75`, `lr=5e-5`, `n_epochs=500`
7. `project_into_latent_space(X_train, model)` → `Y0_`
8. `LatentSpaceVisualiser` → `.plot_latent_timeseries()`, `.plot_phase_space()`, `.rotating_plot()`

**Optional save:** Controlled by `save_results=False` flag; saves weights as `.h5` and latent trajectories as `.txt` to `datasets/generated/bandit_task/`.

---

## `scripts/bandit_gridsearch.py` ⚠️ Long-running

**Purpose:** Run exhaustive grid-search training of BunDLeNet on two-arm bandit NeuroPixels data. Any CLI argument that accepts `nargs='+'` may be given as a list; the script enumerates every combination and executes each run in its own timestamped directory.

**Key CLI arguments (from `parse_args()`):**

- `--data_path`: default `['/home/kerim/Projects/Neural Algorithms/NC-MCM/datasets/raw/twoArmBandit/JPAS_0023_20230922']` — one or more dataset paths to evaluate (grid-searchable).
- `--downsample_fs`: default `[15]` — downsample frequency(s) in Hz (grid-searchable).
- `--downsample_method`: default `['count']` — choices: `binary`, `count`, `rate`, `mean`, `gaussian` (grid-searchable).
- `--good_neurons_only`: default `['false']` — `'true'`/`'false'` strings accepted (converted to booleans for the grid).
- `--apply_hold_transitions`: default `['none']` — state transition mapping to apply; accepts descriptive keys or exact class constants.
- `--normalize_method`: default `['None']` — choices: `None`, `minmax`, `minmax_global` (string `'None'` is converted to no-normalisation).
- `--window`: default `[50]` — time-delay embedding window length (grid-searchable).
- `--latent_dim`: default `[3]` — latent dimension(s) to try (grid-searchable).
- `--batch_size`: default `[50]` — training batch size(s) (grid-searchable).
- `--n_epochs`: default `500` — training epochs (single value; not a list in the parser).
- `--learning_rate`: default `[0.0001]` — learning-rate(s) (grid-searchable).
- `--gamma`: default `[0.75]` — behaviour-loss weight(s) (grid-searchable).
- `--device`: default `cuda` if available else `cpu` — choices `cpu`/`cuda`.
- `--vis_samples`: default `None` — optional `START END` pair to restrict visualization ranges.
- `--recurrence_threshold`: default `None` — optional float to enable recurrence plots.
- `--generate_gif`: flag — enable rotating 3D GIF generation.
- `--generate_3d_html`: flag — enable interactive 3D HTML generation.
- `--lazy_loading`: flag — use lazy (memory-efficient) data preparation/projection.
- `--cv_folds`: default `None` — if set, runs time-series cross-validation with that many folds.
- `--kfold_n_splits`: default `7` — used for the single-split (non-CV) KFold splitting helper.
- `--kfold_test_fold`: default `4` — which fold to use as the test set in the single-split helper.
- `--b_type`: default `['discrete']` — behaviour type(s): `discrete`, `continuous`, `hybrid` (grid-searchable).
- `--hgf_model`: default `binary2` — HGF model variant used when `b_type=hybrid`.
- `--hgf_column`: default `x_1_expected_mean` — HGF output column used as continuous signal for hybrid b.
- `--alpha`: default `[0.5]` — hybrid CE weight(s) (grid-searchable): alpha*CE + (1-alpha)*MSE.
- `--pca_init`: flag — run PCA initialisation of `tau` before training.
- `--choosing_state_mode`: default `side` — `side` or `correctness` labelling for choosing states.
- `--gaussian_sigma_ms`: default `25.0` — Gaussian kernel sigma in ms (used when `downsample_method=gaussian`).
- `--recompute_cache`: flag — force recompute of dataset cache even if present.
- `--b_mode`: default `full` — behavioural representation: `full` (per-timepoint), `decision` (trial decision label per timepoint), `decision_strict` (like `decision` but windows bounded by t_chosen of adjacent trials), or `reward_to_choice` (segment from t_chosen[i-1]+1 to t_chosen[i] labeled with trial i's choice; first trial dropped; cleanest upcoming-decision training target).
- `--trial_based`: flag — use trial-based regime (windows per trial, trial-level train/test split).
- `--trial_test_ratio`: default `0.2` — fraction of trials to hold out in trial-based mode.
- `--trial_random_state`: default `None` — RNG seed for trial-level split.
- `--output_dir`: default `./results` — base directory for grid runs and single runs.

**Parameter combination behaviour:**

- The script builds a `param_grid` from the grid-searchable arguments and enumerates every combination via `itertools.product` (see `generate_param_combinations()`).
- For multi-run grid searches the script creates a timestamped directory under `--output_dir` named `grid_search_{YYYYmmdd_HHMMSS}` and writes `grid_search_summary.json` there. Individual runs are created with `create_run_directory()` as `run_{idx:03d}_{sanitised_param_str}` and contain `figures/`, `model/` and `data/` subdirectories.

**Outputs (per run):** saved under `{output_dir}/run_{...}/` (single-run uses `{output_dir}/run_{timestamp}`):

- `config.json` — saved at the start of a run (full parser namespace as JSON).
- `run_summary.json` — comprehensive run summary (saved by `save_comprehensive_config()`; includes metrics, execution time, and `full_configuration`).
- `model/bundlenet_model.pt` — PyTorch state dict of the trained model.
- `data/loss_array.npy` and `data/test_loss_array.npy` — training & validation loss histories (shape `n_epochs x n_components`).
- If hybrid mode (`loss_array.shape[1] == 5`): `data/disc_loss_array.npy`, `data/cont_loss_array.npy`, `data/disc_test_loss_array.npy`, `data/cont_test_loss_array.npy` are also saved.
- `data/latent_trajectories_{train,validation}.npy` — full latent trajectories for each split.
- `data/behaviour_labels_{train,validation}.npy` — behaviour label arrays saved alongside trajectories.
- `data/trial_ids_{train,validation}.npy` — saved when `segment_ids` (trial IDs) are provided to the visualiser.
- `data/{safe_name}_{train,validation}.npy` — continuous-variable arrays saved when provided (e.g. HGF beliefs; safe name sanitised to lowercase/underscores).
- `figures/` — many visual outputs: `training_loss.png`, `latent_time_series_{split}.html`, `phase_space_dynamics_{split}_{view}.png`, optional `rotation_3d_{split}.gif` (if `--generate_gif`), optional `interactive_3d_{split}.html` (if `--generate_3d_html`), and `phase_space_continuous_{split}_{var}_{view}.png`.
- In CV mode: `cv_summary.json` is written (aggregated fold metrics) and `grid_search_summary.json` (at the grid root) is updated with per-run metrics.

**Notes & gotchas:**

- The run directory naming sanitises parameter values (slashes, backslashes, colons replaced) and uses the last path component of `data_path` to keep names short (`create_run_directory()`).
- The script creates `figures/`, `model/`, and `data/` subfolders for every run (single-run mode creates them under the timestamped run directory).
- Grid-level summary is `grid_search_summary.json` at the grid directory root; individual runs write `run_summary.json` and `config.json` inside the run folder.
- Visualiser filenames and saved arrays are robust to mismatched lengths (the code checks and skips `segment_ids` if lengths don't align).

**Intended use:** Long, potentially cluster/overnight runs. Use `overnight_run.sh` for pre-configured runs or adapt the CLI for programmatic invocation.

---

## `scripts/load_bundlenet.py`

**Purpose:** Load a pre-saved BunDLe-Net model from a run directory and run inference.

**CLI arguments:**
- Positional: `run_dir` — path to a run directory created by `bandit_gridsearch.py`

**How it works:**
1. Reads `run_summary.json` or `config.json` to reconstruct model parameters
2. Infers `n_neurons` from saved `latent_trajectories_train.npy` shape or tau weight matrix
3. Rebuilds `BunDLeNet` and loads weights from `model/bundlenet_model.pt`

---

## `scripts/subsystems_script.py`

**Purpose:** Train the subsystem BunDLe-Net on C. elegans data split by neuron category (sensory / inter / motor).

**No CLI arguments.**

**Pipeline:**
1. `Database` + `exclude_neurons` + `categorise_neurons('datasets/raw/c_elegans')`
2. Splits `x_paired` into `Xs_`, `Xi_`, `Xm_` using the category mask
3. `subsystem_fit.BunDLeNet(latent_dim=3, num_behaviour=8, input_shapes=(...))`
4. Manually projects each stream's tau at inference; concatenates into `Y0_`

---

## `scripts/bandit_behaviour_decoding.py`

**Purpose:** Evaluate linear decodability of behaviour from BunDLe-Net latent space. Reads latent trajectories from an existing run directory.

**CLI arguments:**
- Positional: `run_dir`

**What it evaluates:**
- Discrete state classification: weighted + unweighted cross-entropy decoder (10 runs, 200 epochs, permutation baseline)
- HGF belief regression (if `hgf_belief_{train,validation}.npy` exist): R², permutation chance, predicted vs true scatter

**Outputs:** PDFs and JSON under `{run_dir}/data/decoding/`

---

## `scripts/bandit_microvariable_evaluation.py`

**Purpose:** Evaluate linear decodability of behaviour directly from raw neuronal activity (not the latent space). Produces a complementary figure to `bandit_behaviour_decoding.py`.

**Configuration:** Edit constants at the top of the file (not CLI args):
- `DOWNSAMPLE_FS`, `WINDOW_SIZE`, `NUM_OF_SPLITS`, `SPLIT_MODE`, etc.
- `RUN_DISCRETE`, `RUN_HYBRID`, `RUN_CONTINUOUS` — enable/disable evaluation modes
- `USE_HGF`, `HGF_MODEL`, `HGF_COLUMN` — HGF belief loading

**Intended use:** Run once to establish a baseline for a dataset before comparing to BunDLe-Net latent space decoding.

---

## `scripts/overnight_run.sh`

**Purpose:** Orchestrate a full analysis pipeline (train → decode → evaluate) in one shot. Uses `set -euo pipefail` — exits immediately on any error.

**Usage:** `bash scripts/overnight_run.sh` from the repo root.

**Steps:**
1. `bandit_gridsearch.py` — trial-based hybrid BunDLe-Net training on `JPAS_0023_20230927` with specific hyperparameters (b_type=hybrid, alpha=0.9, window=90, latent_dim=3, lr=5e-5, gamma=0.75, n_epochs=500, pca_init, trial_based, downsample_fs=30, normalize=minmax_global, good_neurons_only=true)
2. Parses `RUN_DIR` from the final "All results saved to:" line in the log
3. `bandit_behaviour_decoding.py {RUN_DIR}` — linear decoding evaluation
4. `bandit_microvariable_evaluation.py datasets/raw/twoArmBandit/JPAS_0023_20230922 test_split` — raw neural baseline on a *different* session (0922 vs 0927)

**Note:** Logs to `/tmp/bundlenet_overnight.log`. Step 4 uses a hardcoded different session from steps 1-3. The two CLI args to `bandit_microvariable_evaluation.py` are the data path and split mode — the script normally reads all config from internal constants but these two override.

---

## `todo.py`

Not a runnable script. Contains project-level TODO comments for the development team:
- Check overfitting (shuffle-behaviour control)
- Unit test coverage
- Bifurcation analysis
- Block-wise colour coding
- Continuous behaviour and confidence integration
