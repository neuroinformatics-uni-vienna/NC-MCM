# AGENTS.md — NC-MCM Repository Guide for AI Coding Agents

> **Last updated:** 2026-05-21  
> This file is the single source of truth for AI agents working in this repository. Read it before touching any code.

---

## What This Project Does

NC-MCM (**Neuro-Cognitive Multilevel Causal Models**) is a Python toolbox that learns interpretable, causal representations of brain activity from simultaneously recorded neuronal and behavioural data.

It implements two algorithms:

1. **BunDLe-Net** (`ncmcm/bundlenet/`) — a PyTorch neural network that learns a low-dimensional latent space preserving both dynamics (Markovian structure) and behaviour (predictive of behavioural labels). Handles discrete, continuous, and hybrid (discrete + continuous) behaviour.

2. **Cognitive Graphs** (`ncmcm/cognitive_graphs/`) — builds a directed graph from the latent space whose nodes are (cognitive-state, behaviour) pairs and edges are empirical transition probabilities. The graph is the "causal model".

---

## Who Built It & References

- **Authors:** Akshey Kumar, Michael Hofer, Vittorio Boarini, Kerim Atak  
- **Institution:** University of Vienna (Grosse-Wentrup lab)

**Key publications:**
- PLOS Comp Bio 2024 — [Neuro-Cognitive Multilevel Causal Modeling](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012674)
- Preprint 2023 — [BunDLe-Net: Neuronal Manifold Learning Meets Behaviour](https://www.biorxiv.org/content/early/2024/04/15/2023.08.08.551978)

---

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .            # installs ncmcm package in editable mode
```

**Python version required:** ≥ 3.10  
**Key dependencies:** `torch`, `numpy`, `scikit-learn`, `scipy`, `mat73`, `pyvis`, `plotly`, `ray[tune,train]`

---

## Module Map

| Module | Role |
|--------|------|
| `ncmcm/bundlenet/` | BunDLe-Net model, training, inference, data preparation |
| `ncmcm/bundlenet/subsystem_fit/` | 3-stream subsystem variant (sensory/inter/motor neurons) |
| `ncmcm/cognitive_graphs/` | Cognitive graph construction, clustering, visualisation |
| `ncmcm/data_loaders/` | Dataset classes for C. elegans, rat hippocampus, bandit task |
| `ncmcm/statistical_testing/` | Markov and stationarity tests for validating cognitive states |
| `ncmcm/visualisers/` | All plotting functions (matplotlib + plotly) |
| `ncmcm/closed_loop/` | **Stub only** — no code |
| `datasets/` | Raw data (MATLAB, npz, Kilosort) and generated embeddings |
| `scripts/` | Runnable training and evaluation scripts |
| `demos/` | Jupyter notebooks — best entry point for learning the API |

---

## Entry Points

| Goal | Use |
|------|-----|
| Learn the API | Start with `demos/bundlenet_on_discrete_behaviour.ipynb` |
| Train on C. elegans | `python scripts/main.py` |
| Train on rat data | `python scripts/continuous_main.py` |
| Train on bandit data | `python scripts/bandit_main.py` |
| Grid search + full pipeline | `bash scripts/overnight_run.sh` |
| Evaluate latent decodability | `python scripts/bandit_behaviour_decoding.py <run_dir>` |
| Load pre-saved model | `python scripts/load_bundlenet.py <run_dir>` |

---

## Critical Conventions

### Data shapes
- Neuronal traces: `(T, n_neurons)` when fed to `prep_data`; `(n_neurons, T)` when fed to visualisers
- After `prep_data`: `x_paired` is `(m, 2, win, n_neurons)`, `b_1` is `(m,)` for discrete
- Latent embeddings `Y0_`: `(T, latent_dim)` from `project_into_latent_space`
- `BanditTaskNeuroPixelsDataset.x` is sparse — always call `.toarray()` before use

### Behaviour types
- `b_type='discrete'`: integer labels, CrossEntropy loss, `b_train_1` shape `(m,)`
- `b_type='continuous'`: float arrays, MSE loss, `b_train_1` shape `(m, k)`
- `b_type='hybrid'`: combined — requires `n_classes` param; `b_train_1` shape `(m, 1+k_continuous)`; use `make_hybrid_b()` to construct it

### Model weights
- Saved as `.pt` (PyTorch state dict) or legacy `.h5`; never commit to version control
- PCA init weights go to `temp/<uuid>/`; grid-search run models go to `results/.../model/bundlenet_model.pt`

### Never do this
- Edit files in `build/` — they are generated artefacts
- Import `ncmcm.bundlenet.scale_invariant_mse` — it requires TensorFlow and is not part of the PyTorch pipeline
- Pass `x.T` to `prep_data` when `x` is already `(T, n_neurons)` — check shape before calling
- Use `b.toarray()` on a numpy array — check `scipy.sparse.issparse(b)` first
- Use `behavior_labels_train.npy` (US spelling) — gridsearch writes `behaviour_labels_train.npy` (British)
- Assert `train_history.shape == (n_epochs, 3)` — current `train_model` returns `(n_epochs, 5)` columns

---

## How to Run Tests

```bash
# From repo root with .venv active
pytest ncmcm/
```

Test locations:
- `ncmcm/bundlenet/tests/` — BunDLe-Net training, losses, utils
- `ncmcm/cognitive_graphs/tests/` — graph construction
- `ncmcm/statistical_testing/tests/` — Markov/stationarity tests
- `ncmcm/visualisers/tests/` — **empty** (no tests implemented yet)

---

## Dataset Locations

| Dataset | Path | Load with |
|---------|------|----------|
| C. elegans (5 worms) | `datasets/raw/c_elegans/NoStim_Data.mat` | `ncmcm.data_loaders.matlab_dataset.Database` |
| Rat hippocampus (4 rats) | `datasets/raw/rat_hippocampus/*.npz` | `numpy.load` |
| Bandit NeuroPixels (3 sessions) | `datasets/raw/twoArmBandit/JPAS_0023_*/` | `ncmcm.data_loaders.bandit_task.BanditTaskNeuroPixelsDataset` |
| Pre-computed embeddings | `datasets/generated/saved_embeddings/` | `numpy.loadtxt` |

---

## Documentation Index (`.ai/` folder)

| File | Contents |
|------|---------|
| [.ai/architecture.md](.ai/architecture.md) | Package layout, data flow diagrams, dependency graph, test locations |
| [.ai/data_shapes.md](.ai/data_shapes.md) | Tensor/array shapes at every pipeline step — input→output dtypes and dimensions |
| [.ai/hyperparameters.md](.ai/hyperparameters.md) | Every tunable parameter: default, range, location in code, gridsearch coverage |
| [.ai/outputs.md](.ai/outputs.md) | Every file written to disk: path pattern, format, producer script, consumer |
| [.ai/known_issues.md](.ai/known_issues.md) | All TODO/FIXME/HACK comments, dead code, runtime gotchas, and known bugs |
| [.ai/test_coverage.md](.ai/test_coverage.md) | Public API test coverage map — which symbols are tested, which are not |
| [.ai/datasets.md](.ai/datasets.md) | Dataset formats, shapes, loading patterns, embedding naming convention |
| [.ai/scripts.md](.ai/scripts.md) | Every script — purpose, CLI args, expected outputs |
| [.ai/demos.md](.ai/demos.md) | Every Jupyter notebook — dataset used, key steps |
| [.ai/conventions.md](.ai/conventions.md) | Naming, model saving, testing, logging, style patterns |
| [.ai/dependencies.md](.ai/dependencies.md) | Per-dependency explanation, version-sensitive notes |
| [.ai/modules/bundlenet.md](.ai/modules/bundlenet.md) | Full BunDLe-Net API: classes, functions, signatures, shapes |
| [.ai/modules/cognitive_graphs.md](.ai/modules/cognitive_graphs.md) | Cognitive Graphs API: graph construction, clustering, helpers |
| [.ai/modules/data_loaders.md](.ai/modules/data_loaders.md) | DataLoader classes: attributes, methods, data formats |
| [.ai/modules/statistical_testing.md](.ai/modules/statistical_testing.md) | Markov/stationarity test API |
| [.ai/modules/visualisers.md](.ai/modules/visualisers.md) | All visualiser functions/classes |
| [.ai/modules/closed_loop.md](.ai/modules/closed_loop.md) | Stub module documentation |
| [.ai/bandit.md](.ai/bandit.md) | **Bandit-branch deep-dive**: paradigm, `metrics.json` schema, HGF models, preprocessing pipeline, behaviour representations, results structure, gotchas |

---

## Changelog

| Date | Commit | What was updated |
|------|--------|-----------------|
| 2026-04-28 | `acacadc` | Initial full documentation scan — all `.ai/` files and `AGENTS.md` created from scratch |
| 2026-04-28 | `acacadc` | Added `.ai/bandit.md` — bandit-branch domain documentation covering paradigm, raw data schema, HGF, preprocessing pipeline, behaviour modes, scripts, results structure, and gotchas |
| 2026-05-05 | `ae1a929` | `prep_data_trials` semantics change (cross-trial context borrowing); new `bandit_main_trial_based.py`; `trial_based_bundlenet.ipynb` completed — updated `.ai/modules/bundlenet.md`, `.ai/scripts.md`, `.ai/demos.md`, `.ai/bandit.md` |
| 2026-05-05 | `ae1a929` | Full scan: created `.ai/data_shapes.md`, `.ai/hyperparameters.md`, `.ai/outputs.md`, `.ai/known_issues.md`, `.ai/test_coverage.md`; updated `.ai/modules/data_loaders.md` (corrected `b` shape, added 3 methods), `.ai/modules/visualisers.md` (added `plot_phase_space_continuous`, missing `behavioural_discrete` functions), `.ai/scripts.md` (overnight_run exact params); added 2 "Never do this" entries to AGENTS.md |
| 2026-05-21 | `2f054b8` | Prompt 028: added `b_mode='reward_to_choice'` to `BanditTaskNeuroPixelsDataset`; updated `--b_mode` CLI in `bandit_gridsearch.py`; added `scripts/verify_reward_to_choice.py` diagnostic (4/4 criteria pass, JPAS_0023_20230922, 30 Hz gaussian); updated `.ai/bandit.md` (`b_mode` table) and `.ai/scripts.md` |
| 2026-05-21 | `0c2c6f5` | Prompt 029: retrained clean BunDLe-Net with `b_mode=reward_to_choice`, `context_policy=same_partition`, `trial_random_state=42`, JPAS_0023_20230922, 30Hz gaussian. Hybrid (alpha=0.5) run: `grid_search_20260521_150124_same_partition_reward_to_choice_hybrid_alpha_050/run_20260521_150127` (58m). Discrete-only: `grid_search_20260521_150124_same_partition_reward_to_choice_discrete_only/run_20260521_155945` (56m). Choice acc (hybrid): 0.825/bacc=0.823; (discrete): 0.816/0.815. HGF R²=0.690. Stay=0.905 bacc, Switch=0.618 bacc. Hybrid PCA: 3 active dims (79.5/20.5/0.0%). Added `scripts/preflight_reward_to_choice.py`, `scripts/run_reward_to_choice_tmux.sh`, `scripts/analyze_reward_to_choice_results.py`, `scripts/start_reward_to_choice_decoding.sh`. Analysis at `results/analysis/rtc_analysis_20260521_172508/`. |
| 2026-05-21 | `0a59067` | Prompt 030: colocalized all post-training analysis outputs inside `<run_dir>/analysis/<name>_{ts}/`; fixed broken `hybrid_vs_discrete_summary.json` cell in `time_resolved_predictability_discrete_only.ipynb`; updated `analyze_reward_to_choice_results.py` and `event_aligned_predictability.py` defaults. |
| 2026-05-21 | `9439dcd` | Prompt 032: new `ncmcm/experiment_archive/` module (`folders.py`, `manifest.py`, `report.py`) + `scripts/run_experiment.py` orchestrator. Creates `results/experiments/<experiment_id>/` with 9 subfolders, `manifest.json`, `config.json`, `status.json`, `reports/experiment_report.md`. Added `--out` arg to `bandit_behaviour_decoding.py`. Smoke run: `JPAS_0023_20230922_reward_to_choice_hybrid_alpha_050_seed42_20260521_150127`. |
