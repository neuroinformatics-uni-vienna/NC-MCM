#!/usr/bin/env bash
# run_reward_to_choice_tmux.sh
# Launch two BunDLe-Net training runs (hybrid + discrete) for
# b_mode='reward_to_choice' / same_partition, seed=42.
#
# Usage:  bash scripts/run_reward_to_choice_tmux.sh
#
# Naming convention:
#   tmux session : bandit_reward_to_choice_clean_YYYYMMDD_HHMMSS
#   log file     : logs/bandit_reward_to_choice_clean_YYYYMMDD_HHMMSS.log
#   output dirs  : results/grid_search_YYYYMMDD_HHMMSS_same_partition_reward_to_choice_{hybrid,discrete_only}

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION="bandit_reward_to_choice_clean_${TIMESTAMP}"
LOG="logs/bandit_reward_to_choice_clean_${TIMESTAMP}.log"
mkdir -p logs results

# ── Shared hyperparameters ────────────────────────────────────────────────────
DATA_PATH="/home/kerim/Projects/Neural Algorithms/NC-MCM/datasets/raw/twoArmBandit/JPAS_0023_20230922"
FS=30
METHOD=gaussian
WINDOW=50
LATENT_DIM=3
BATCH_SIZE=50
N_EPOCHS=500
LR=0.0001
GAMMA=0.75
NORM=minmax_global
B_MODE=reward_to_choice
TRIAL_TEST_RATIO=0.2
TRIAL_RANDOM_STATE=42
CONTEXT=same_partition
HGF_MODEL=binary2
HGF_COLUMN=x_1_expected_mean
SIGMA_MS=25.0

# ── Output directory names ────────────────────────────────────────────────────
OUT_HYBRID="./results/grid_search_${TIMESTAMP}_same_partition_reward_to_choice_hybrid_alpha_050"
OUT_DISC="./results/grid_search_${TIMESTAMP}_same_partition_reward_to_choice_discrete_only"

# ── Build command strings ─────────────────────────────────────────────────────
BASE="python scripts/bandit_gridsearch.py \
  --data_path \"${DATA_PATH}\" \
  --downsample_fs ${FS} \
  --downsample_method ${METHOD} \
  --good_neurons_only false \
  --normalize_method ${NORM} \
  --window ${WINDOW} \
  --latent_dim ${LATENT_DIM} \
  --batch_size ${BATCH_SIZE} \
  --n_epochs ${N_EPOCHS} \
  --learning_rate ${LR} \
  --gamma ${GAMMA} \
  --device cuda \
  --lazy_loading \
  --pca_init \
  --choosing_state_mode side \
  --gaussian_sigma_ms ${SIGMA_MS} \
  --b_mode ${B_MODE} \
  --trial_based \
  --trial_test_ratio ${TRIAL_TEST_RATIO} \
  --trial_random_state ${TRIAL_RANDOM_STATE} \
  --context_policy ${CONTEXT} \
  --hgf_model ${HGF_MODEL} \
  --hgf_column ${HGF_COLUMN}"

CMD_HYBRID="${BASE} --b_type hybrid --alpha 0.5 --output_dir \"${OUT_HYBRID}\""
CMD_DISC="${BASE} --b_type discrete --output_dir \"${OUT_DISC}\""

# Run both sequentially in the same tmux window, logging everything
FULL_CMD="
set -euo pipefail
echo '=========================================================='
echo 'bandit_reward_to_choice_clean  ${TIMESTAMP}'
echo '=========================================================='
echo ''
echo '[1/2] Hybrid (alpha=0.5)...'
${CMD_HYBRID}
echo ''
echo '[1/2] Hybrid done.'
echo ''
echo '[2/2] Discrete-only control...'
${CMD_DISC}
echo ''
echo '[2/2] Discrete done.'
echo ''
echo 'ALL RUNS COMPLETE'
"

echo "Starting tmux session: ${SESSION}"
echo "Log file: ${LOG}"
echo ""
echo "Hybrid output dir:   ${OUT_HYBRID}"
echo "Discrete output dir: ${OUT_DISC}"
echo ""

tmux new-session -d -s "${SESSION}" \
  bash -lc "${FULL_CMD} 2>&1 | tee ${LOG}"

echo "Session started.  Monitor with:"
echo "  tmux attach -t ${SESSION}"
echo "  tail -f ${LOG}"
