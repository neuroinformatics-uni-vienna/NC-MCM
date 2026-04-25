#!/usr/bin/env bash
# overnight_run.sh — BunDLeNet + behaviour decoding + microvariable eval (test_split)
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

LOG=/tmp/bundlenet_overnight.log
echo "=== Starting overnight run at $(date) ===" | tee "$LOG"

# ── 1. BunDLeNet training ────────────────────────────────────────────────────
echo ">>> [1/3] BunDLeNet training (alpha=0.1, HGF hybrid)" | tee -a "$LOG"

python scripts/bandit_gridsearch.py \
    --data_path "datasets/raw/twoArmBandit/JPAS_0023_20230927" \
    --output_dir "results/twoArmBandit/hybrid_alpha_search" \
    --b_type hybrid \
    --alpha 0.9 \
    --downsample_fs 30 \
    --downsample_method count \
    --good_neurons_only true \
    --apply_hold_transitions none \
    --normalize_method minmax_global \
    --window 90 \
    --latent_dim 3 \
    --batch_size 50 \
    --learning_rate 5e-05 \
    --gamma 0.75 \
    --n_epochs 500 \
    --lazy_loading \
    --pca_init \
    2>&1 | tee -a "$LOG"

# Extract the run directory from gridsearch output
RUN_DIR=$(grep "All results saved to:" "$LOG" | tail -1 | awk '{print $NF}')
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
    echo "ERROR: Could not determine run directory from gridsearch output" | tee -a "$LOG"
    exit 1
fi
echo "Run directory: $RUN_DIR" | tee -a "$LOG"

# ── 2. Behaviour decoding (latent space) ────────────────────────────────────
echo "" | tee -a "$LOG"
echo ">>> [2/3] Behaviour decoding on BunDLeNet latent space" | tee -a "$LOG"
python scripts/bandit_behaviour_decoding.py "$RUN_DIR" 2>&1 | tee -a "$LOG"

# ── 3. Microvariable evaluation (raw neural, test_split = KFold-7 fold-4) ───
echo "" | tee -a "$LOG"
echo ">>> [3/3] Microvariable evaluation (raw neural, test_split mode)" | tee -a "$LOG"
python scripts/bandit_microvariable_evaluation.py \
    "datasets/raw/twoArmBandit/JPAS_0023_20230922" \
    test_split \
    2>&1 | tee -a "$LOG"

# ── 4. BunDLeNet training (trial-based, same hyperparams) ───────────────────
echo "" | tee -a "$LOG"
echo ">>> [4/5] BunDLeNet training — trial-based regime" | tee -a "$LOG"

python scripts/bandit_gridsearch.py \
    --data_path "datasets/raw/twoArmBandit/JPAS_0023_20230927" \
    --output_dir "results/twoArmBandit/trial_based" \
    --b_type hybrid \
    --alpha 0.9 \
    --downsample_fs 30 \
    --downsample_method count \
    --good_neurons_only true \
    --apply_hold_transitions none \
    --normalize_method minmax_global \
    --window 90 \
    --latent_dim 3 \
    --batch_size 50 \
    --learning_rate 5e-05 \
    --gamma 0.75 \
    --n_epochs 500 \
    --pca_init \
    --trial_based \
    --trial_start_state intertrial \
    --trial_test_ratio 0.2 \
    --trial_random_state 42 \
    2>&1 | tee -a "$LOG"

# Extract the trial-based run directory
TRIAL_RUN_DIR=$(grep "All results saved to:" "$LOG" | tail -1 | awk '{print $NF}')
if [[ -z "$TRIAL_RUN_DIR" || ! -d "$TRIAL_RUN_DIR" ]]; then
    echo "ERROR: Could not determine trial-based run directory from gridsearch output" | tee -a "$LOG"
    exit 1
fi
echo "Trial-based run directory: $TRIAL_RUN_DIR" | tee -a "$LOG"

# ── 5. Behaviour decoding (trial-based latent space) ────────────────────────
echo "" | tee -a "$LOG"
echo ">>> [5/5] Behaviour decoding on trial-based BunDLeNet latent space" | tee -a "$LOG"
python scripts/bandit_behaviour_decoding.py "$TRIAL_RUN_DIR" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== ALL DONE at $(date) ===" | tee -a "$LOG"
echo "Full log: $LOG"
