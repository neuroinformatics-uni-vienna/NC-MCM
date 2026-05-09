#!/usr/bin/env bash
set -euo pipefail

# Move to repo root
cd "$(dirname "$0")/.."

# Activate virtualenv if present
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Determine device dynamically (match script default behaviour)
DEVICE=$(python - <<'PY'
import torch
print('cuda' if torch.cuda.is_available() else 'cpu')
PY
)

# --- Explicit configuration (requested + defaults) ---
DATA_PATH="/home/kerim/Projects/Neural Algorithms/NC-MCM/datasets/raw/twoArmBandit/JPAS_0023_20230922"
DOWNSAMPLE_FS=30
DOWNSAMPLE_METHOD="gaussian"
GOOD_NEURONS_ONLY="false"
APPLY_HOLD_TRANSITIONS="none"
NORMALIZE_METHOD="minmax_global"
WINDOW=30
LATENT_DIM=3
BATCH_SIZE=50
N_EPOCHS=500
LEARNING_RATE=0.0001
GAMMA=0.75
OUTPUT_DIR="./results"
KFOLD_N_SPLITS=7
KFOLD_TEST_FOLD=4
B_TYPE="discrete"
HGF_MODEL="binary2"
HGF_COLUMN="x_1_expected_mean"
ALPHA=0.5
PCA_INIT=1
CHOOSING_STATE_MODE="side"
GAUSSIAN_SIGMA_MS=25.0
RECOMPUTE_CACHE=0
B_MODE="decision"
TRIAL_BASED=1
TRIAL_TEST_RATIO=0.2
TRIAL_RANDOM_STATE=""

# Build command
ARGS=(python scripts/bandit_gridsearch.py
  --data_path "$DATA_PATH"
  --downsample_fs "$DOWNSAMPLE_FS"
  --downsample_method "$DOWNSAMPLE_METHOD"
  --good_neurons_only "$GOOD_NEURONS_ONLY"
  --apply_hold_transitions "$APPLY_HOLD_TRANSITIONS"
  --normalize_method "$NORMALIZE_METHOD"
  --window "$WINDOW"
  --latent_dim "$LATENT_DIM"
  --batch_size "$BATCH_SIZE"
  --n_epochs "$N_EPOCHS"
  --learning_rate "$LEARNING_RATE"
  --gamma "$GAMMA"
  --device "$DEVICE"
  --output_dir "$OUTPUT_DIR"
  --kfold_n_splits "$KFOLD_N_SPLITS"
  --kfold_test_fold "$KFOLD_TEST_FOLD"
  --b_type "$B_TYPE"
  --hgf_model "$HGF_MODEL"
  --hgf_column "$HGF_COLUMN"
  --alpha "$ALPHA"
  --choosing_state_mode "$CHOOSING_STATE_MODE"
  --gaussian_sigma_ms "$GAUSSIAN_SIGMA_MS"
  --b_mode "$B_MODE"
  --trial_test_ratio "$TRIAL_TEST_RATIO"
)

# Flags
if [ "$PCA_INIT" -eq 1 ]; then ARGS+=(--pca_init); fi
if [ "$TRIAL_BASED" -eq 1 ]; then ARGS+=(--trial_based); fi
if [ "$RECOMPUTE_CACHE" -eq 1 ]; then ARGS+=(--recompute_cache); fi

echo "Starting bandit gridsearch; logging to $OUTPUT_DIR/run_bandit_gridsearch.log"
mkdir -p "$OUTPUT_DIR"

# Run and tee output to log
"${ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/run_bandit_gridsearch.log"
